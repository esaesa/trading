from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta

# Use new class-based architecture
from rule_chain import RuleChain
from rules.rule_factory import RuleFactory
try:
    from contracts import Ctx  # your ctx dataclass (preferred)
except Exception:
    # Minimal fallback type (won't be used if your contracts.Ctx is available)
    @dataclass
    class Ctx:  # type: ignore
        now: datetime
        price: float
        entry_price: Optional[float]
        position_size: float
        dca_level: int
        indicators: Dict[str, Any]
        equity_per_cycle: float
        config: Dict[str, Any]
        position_pl_pct: float
        base_order_price: Optional[float] = None
        base_order_quantity: Optional[float] = None
        last_filled_price: Optional[float] = None

from .execution.types import OrderIntent, IntentType, OrderSide, OrderKind
from .utils.precision import PrecisionHelper

# ---------- Math model for sizing (can be swapped with your sizers later) ----------
# Base quantity q0 and SO quantities qi follow a geometric progression
# q0 = (E * f) / P0
# qi = q0 * m^(i-1)   for i = 1..L, where m = so_size_multiplier
def _base_qty(equity_per_cycle: float, entry_fraction: float, price0: float) -> float:
    return (equity_per_cycle * entry_fraction) / max(price0, 1e-12)

def _safety_qty(q0: float, so_mult: float, level_index_zero_based: int) -> float:
    return q0 * (so_mult ** level_index_zero_based)

# ---------- Adapter “owner” that your rules receive via build_rule_chain ----------
class _RuleOwner:
    """
    Minimal facade to provide attributes/methods your rules might expect.
    Extend with helpers (e.g., ATR, RSI getters) without touching the engine.
    """
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.strategy_params = cfg.get("strategy_params", {})
        self.indicator_params = cfg.get("indicator_params", {})
        self.mode = cfg.get("mode", "futures")

    # Example helpers your rules may call
    def get_param(self, name: str, default: Any = None) -> Any:
        return self.strategy_params.get(name, default)

    def get_indicator(self, ctx: Ctx, key: str, default: Any = None) -> Any:
        return ctx.indicators.get(key, default)

    def now(self) -> datetime:
        return datetime.utcnow()

# ---------- Main decider ----------
class RuleChainDecider:
    """
    Uses your rule registries to decide ENTRY/SAFETY/EXIT, while this class handles:
      - order sizing, precision & min-notional
      - safety-order cooldown
      - reduce-only for exits (futures)
      - market/limit mapping from config
    """
    def __init__(self, cfg: Dict[str, Any], state=None):
        self.cfg = cfg
        self.sp = cfg["strategy_params"]
        self.owner = _RuleOwner(cfg)

        # Build chains once using new class-based architecture
        self._entry_chain = self._build_rule_chain("entry_rule_names", ["RSIOverbought", "EntryFundsAndNotional"])
        self._safety_chain = self._build_rule_chain("safety_rule_names", [
            "RSIUnderDynamicThreshold", "RSIReversalStaticThreshold", "RSIReversalDynamicThreshold",
            "RSIUnderStaticThreshold", "CooldownBetweenSOs", "MaxLevelsNotReached", "SufficientFundsAndNotional"
        ])
        self._exit_chain = self._build_rule_chain("exit_rule_names", [
            "ATRTakeProfitReached", "TakeProfitReached", "TPDecayReached",
            "StopLossReached", "TrailingStopReached"
        ])

        # cached knobs
        self.order_type = cfg.get("order_type", "market").lower()  # "market" | "limit"
        self.reduce_only_cfg = bool(cfg.get("reduce_only", True))
        self.so_cooldown_minutes = int(self.sp.get("so_cooldown_minutes", 0))
        self.max_dca_levels = int(self.sp.get("max_dca_levels", 0))
        self.entry_fraction = float(self.sp.get("entry_fraction", 0.2))
        self.so_size_multiplier = float(self.sp.get("so_size_multiplier", 1.0))

        # state reference (LiveState), used only for cooldown and level tracking if needed
        self.state = state  # engine will pass its LiveState; can be None in tests

    def _build_rule_chain(self, config_key: str, default_rules: List[str]) -> RuleChain:
        """Build a rule chain using the new class-based architecture."""
        # Get rule names from config or use defaults
        rule_names = self.sp.get(config_key, default_rules)

        # Create rule instances using RuleFactory
        rules = []
        for rule_name in rule_names:
            try:
                rule = RuleFactory.create_rule(rule_name, self.owner)
                rules.append(rule)
            except ValueError as e:
                print(f"Warning: Failed to create rule '{rule_name}': {e}")
                continue

        return RuleChain(rules, mode="any")

    # ---- internal helpers ----
    def _cooldown_ok(self) -> bool:
        if not self.so_cooldown_minutes or not self.state or not self.state.last_so_time:
            return True
        try:
            last = datetime.fromisoformat(self.state.last_so_time)
            return datetime.utcnow() >= last + timedelta(minutes=self.so_cooldown_minutes)
        except Exception:
            return True

    def _is_market(self) -> bool:
        return self.order_type == "market"

    def _order_kind(self) -> OrderKind:
        return OrderKind.MARKET if self._is_market() else OrderKind.LIMIT

    # ---- public: compute intents for this closed bar ----
    def next_intents(self, ctx: Ctx, prec: PrecisionHelper, symbol: str) -> List[OrderIntent]:
        intents: List[OrderIntent] = []

        # 1) EXIT has highest priority
        ok, exit_reason = self._exit_chain.ok(ctx)
        if ok:
            if ctx.position_size > 0:
                qty = abs(ctx.position_size)
                intents.append(OrderIntent(
                    intent=IntentType.EXIT,
                    side=OrderSide.SELL,
                    qty=prec.amount(symbol, qty),
                    kind=self._order_kind(),
                    price=None if self._is_market() else prec.price(symbol, ctx.price),
                    reduce_only=True  # always reduce-only for safety
                ))
                return intents  # done

        # 2) ENTRY when flat (no position and no base order committed)
        flat_like = (ctx.position_size == 0) and (ctx.dca_level == 0) and (ctx.base_order_price is None)
        entry_ok, entry_reason = self._entry_chain.ok(ctx)
        if flat_like and entry_ok:
            p0 = ctx.price
            q0_raw = _base_qty(ctx.equity_per_cycle, self.entry_fraction, p0)
            q0 = prec.amount(symbol, prec.clip_to_min_qty(symbol, q0_raw))
            if q0 > 0 and prec.ensure_min_notional(symbol, p0, q0):
                intents.append(OrderIntent(
                    intent=IntentType.ENTRY,
                    side=OrderSide.BUY,
                    qty=q0,
                    kind=self._order_kind(),
                    price=None if self._is_market() else prec.price(symbol, p0),
                    note="ENTRY via rule-chain"
                ))
                return intents  # only one intent per bar

        # 3) SAFETY order (respect cooldown, max levels, rule-chain)
        safety_ok, safety_reason = self._safety_chain.ok(ctx)
        if safety_ok:
            if ctx.dca_level < self.max_dca_levels and self._cooldown_ok():
                # size from geometric schedule based on the base-order computed q0
                p0 = ctx.base_order_price or ctx.entry_price or ctx.price
                q0_raw = _base_qty(ctx.equity_per_cycle, self.entry_fraction, p0)
                qi_raw = _safety_qty(q0_raw, self.so_size_multiplier, ctx.dca_level)
                qi = prec.amount(symbol, prec.clip_to_min_qty(symbol, qi_raw))
                if qi > 0 and prec.ensure_min_notional(symbol, ctx.price, qi):
                    intents.append(OrderIntent(
                        intent=IntentType.SAFETY,
                        side=OrderSide.BUY,
                        qty=qi,
                        kind=self._order_kind(),
                        price=None if self._is_market() else prec.price(symbol, ctx.price),
                        note=f"SAFETY L{ctx.dca_level+1} via rule-chain"
                    ))

        return intents

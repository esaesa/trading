# wiring.py
from __future__ import annotations
from typing import Any
from order_alloc import (
    FractionalCashEntrySizer,
    EmaEntryMultiplier,
    FixedEntryMultiplier,
    DefaultAffordabilityGuard,
)
from commissions import FixedRateCommission
from ports import PriceEngine, SizeEngine
from rules.pricing import StaticPriceEngine, DynamicATRPriceEngine
from rules.sizing import ValueModeSizeEngine, VolumeModeSizeEngine
from rules.exit import ExitRuleDecider
from rules.entry import EntryRuleDecider
from rules.safety import SafetyRuleDecider
from indicators_provider import StrategyIndicatorProvider
from position_view import BacktestingPositionView
from preflight import EntryPreflight


# === default rule selection lives in wiring, not strategy ===
def _apply_default_rules(owner, params: dict) -> dict:
    """
    Centralize defaults for rule wiring. Does NOT override user-provided lists/dicts.
    """
    cfg = dict(params or {})

    # Entry defaults
    default_entry = (["RSIOverboughtGate"] if owner.config.avoid_rsi_overbought else [])
    default_entry.append("EntryFundsAndNotional")

    # Safety defaults
    default_safety = ["RSIUnderDynamicThreshold", "MaxLevelsNotReached"]
    if owner.config.require_rsi_reset and not owner.config.rsi_dynamic_threshold:
        default_safety.append("StaticRSIReset")

    # Exit defaults
    default_exit = ["TPDecayReached"]

    # Only set if user hasn't provided them
    cfg.setdefault("entry_conditions", default_entry)
    cfg.setdefault("safety_conditions", default_safety)
    cfg.setdefault("exit_conditions", default_exit)

    return cfg


def _build_price_engine(strategy: Any) -> PriceEngine:
    mode = (strategy.config.safety_order_price_mode or "dynamic").lower()
    if mode == "static":
        return StaticPriceEngine(strategy)
    return DynamicATRPriceEngine(strategy)


def _build_size_engine(strategy: Any) -> SizeEngine:
    mode = (strategy.config.safety_order_mode or "value").lower()
    if mode == "volume":
        return VolumeModeSizeEngine(strategy)
    return ValueModeSizeEngine(strategy)


def wire_strategy(strategy: Any, strategy_params: dict) -> None:
    """
    Attach deciders/engines to the strategy instance.
    Entry/Safety/Exit deciders build their own chains from config (supports nested groups).
    """
    cfg = _apply_default_rules(strategy, strategy_params)

    entry_names  = cfg.get("entry_conditions", [])
    safety_names = cfg.get("safety_conditions", [])
    exit_names   = cfg.get("exit_conditions", [])

    strategy.entry_decider  = EntryRuleDecider(strategy, entry_names,  default_mode="any")
    strategy.safety_decider = SafetyRuleDecider(strategy, safety_names, default_mode="all")
    strategy.exit_decider   = ExitRuleDecider(strategy, exit_names,    default_mode="any")

    strategy.price_engine   = _build_price_engine(strategy)
    strategy.size_engine    = _build_size_engine(strategy)
    # Entry sizing policy
    if strategy.config.enable_ema_calculation:
        entry_mult = EmaEntryMultiplier(mult_above=2, default=1)
    else:
        entry_mult = FixedEntryMultiplier(1)

    strategy.entry_sizer = FractionalCashEntrySizer(
    entry_fraction=strategy.config.entry_fraction,
        multiplier=entry_mult,
    )
    
    # Commission policy (single source of truth)
    strategy.commission_calc = FixedRateCommission(rate=getattr(strategy, "commission_rate", 0.0))


    # Affordability policy (for DCA & optionally entry)
    strategy.affordability_guard = DefaultAffordabilityGuard()
  
    # Position view adapter
    strategy.position_view = BacktestingPositionView(
        last_price_fn=lambda: strategy._broker.last_price,
        position_fn=lambda: strategy.position,
    )
    # Indicator provider (so engines/rules don’t couple to Strategy fields)
    strategy.indicator_provider = StrategyIndicatorProvider(strategy)
    
    strategy.entry_preflight = EntryPreflight(strategy.entry_sizer, strategy.commission_calc)

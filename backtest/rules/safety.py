# rules/safety.py
from datetime import timedelta
import math
from typing import Tuple
import numpy as np
from contracts import Ctx
from logger_config import logger

def rsi_under_dynamic_threshold(self, ctx: Ctx) -> Tuple[bool, str]:
    """
    Allow DCA if RSI is below the chosen threshold.
      - If rsi_dynamic_threshold=True and ctx.dynamic_rsi_thr is available (not NaN),
        use that; otherwise fall back to self.rsi_threshold.
      - If RSI is NaN, we allow (existing behavior).
    """
    rsi_val = ctx.indicators.get("rsi", np.nan)

    dyn_thr = ctx.dynamic_rsi_thr
    if  np.isnan(dyn_thr):
        return False, "Dyn RSI NaN → deny"
    
    use_dyn = bool(getattr(self, "rsi_dynamic_threshold", False)) and dyn_thr is not None and not np.isnan(dyn_thr)
    threshold = float(dyn_thr) if use_dyn else float(getattr(self, "rsi_threshold", 50) or 50.0)

    if np.isnan(rsi_val):
        return False, "RSI NaN → deny"
    ok = rsi_val < threshold
    if not ok:
        pass

    if getattr(self, "debug_trade", False) and not ok:
        level = ctx.config.get("next_level", ctx.dca_level + 1)
        logger.debug(f"RSI={rsi_val:.2f} not below threshold={threshold:.2f}, skipping DCA-{level}")

    return ok, f"RSI {rsi_val:.2f} < thr {threshold:.2f}" if ok else f"RSI {rsi_val:.2f} ≥ thr {threshold:.2f}"

def rsi_under_static_threshold(self, ctx: Ctx) -> Tuple[bool, str]:
    rsi_val = ctx.indicators.get("rsi", np.nan)
    threshold = self.config.rsi_static_threshold_under
    if np.isnan(rsi_val):
        return True, "RSI NaN → allow"
    ok = rsi_val < threshold
    if ok:
        pass

    if getattr(self, "debug_trade", False) and not ok:
        level = ctx.config.get("next_level", ctx.dca_level + 1)
        logger.debug(f"RSI={rsi_val:.2f} not below threshold={threshold:.2f}, skipping DCA-{level}")

    return ok, f"RSI {rsi_val:.2f} < thr {threshold:.2f}" if ok else f"RSI {rsi_val:.2f} ≥ thr {threshold:.2f}"

def cooldown_between_sos(self, ctx: Ctx) -> Tuple[bool, str]:
    """
    Enforce a minimum elapsed time between BO/SO and the next SO.
    """
    mins = int(getattr(self, "so_cooldown_minutes", 0) or 0)
    if mins <= 0:
        return True, "No cooldown"

    now = ctx.now
    last = getattr(self, "last_safety_order_time", None) or getattr(self, "base_order_time", None)
    if last is None:
        return True, "No previous SO/BO time"

    if now - last >= timedelta(minutes=mins):
        return True, f"Cooldown {mins}m elapsed"

    if getattr(self, "debug_trade", False):
        level = ctx.config.get("next_level", ctx.dca_level + 1)
        logger.debug(f"Skip DCA-{level}: cooldown {(now - last)} < {mins}m")
    return False, "Cooldown not elapsed"

def rsi_wave_reset(self, ctx: Ctx) -> Tuple[bool, str]:
    """
    Enforce RSI wave cycle for safety orders in static threshold mode.
    Only allows one safety order per RSI wave cycle.
    Active only if:
      - self.require_rsi_reset is True
      - self.rsi_dynamic_threshold is False
    Behavior:
      - If RSI >= reset threshold → mark wave as available (rsi_wave_available = True)
      - If RSI < trading threshold AND wave is available → allow trade and mark wave as used
      - After trade, wave remains used until RSI goes above reset threshold again
    """

    rsi_val = ctx.indicators.get("rsi", np.nan)
    if np.isnan(rsi_val):
        return True, "RSI NaN → allow"

    # Calculate thresholds
    base_thr = float(getattr(self, "rsi_threshold", 31) or 31.0)
    reset_pct = float(getattr(self, "rsi_reset_percentage", 60) or 60)
    reset_thr = base_thr * (1.0 + reset_pct / 100.0)

    # Initialize wave state if not exists
    if not hasattr(self, "rsi_wave_available"):
        self.rsi_wave_available = True

    # If RSI goes above reset threshold, make wave available
    reset_thr = 56
    if rsi_val >= reset_thr:
        self.rsi_wave_available = True
        return False, f"RSI {rsi_val:.2f} ≥ reset threshold {reset_thr:.2f} → wave available"

    # If RSI is below trading threshold and wave is available, allow trade
    trading_thr = base_thr
    if rsi_val < trading_thr and self.rsi_wave_available:
        self.rsi_wave_available = False  # Mark wave as used
        return True, f"RSI {rsi_val:.2f} < trading threshold {trading_thr:.2f} → wave used"

    # Otherwise, don't allow trade
    if rsi_val < trading_thr:
        return False, f"RSI {rsi_val:.2f} < trading threshold {trading_thr:.2f} but wave not available"
    else:
        return False, f"RSI {rsi_val:.2f} ≥ trading threshold {trading_thr:.2f}"


def max_levels_not_reached(self, ctx: Ctx) -> Tuple[bool, str]:
    """
    Allow SO only if current DCA level < max_dca_levels.
    Mirrors the old imperative check in strategy.process_dca.
    """
    max_levels = self.config.max_dca_levels
    ok = ctx.dca_level < max_levels
    return ok, (
        f"Level {ctx.dca_level} < max {max_levels}"
        if ok else f"Reached max levels ({max_levels})"
    )
def sufficient_funds_and_notional(self, ctx: Ctx):
    """
    Allow SO only if there exists an integer quantity q we can afford (after commission)
    such that ORDER notional q * P_so >= minimum_notional, with q <= max_possible.
    If next_so_price is unknown, be lenient and allow.
    """
    P_so = ctx.next_so_price
    if P_so is None or P_so <= 0:
        return True, "SO price unknown → allow"

    c = float(ctx.config.get("commission_rate", 0.0) or 0.0)
    cost_per_share = P_so * (1.0 + c)
    if cost_per_share <= 0:
        return True, "Invalid cost/share → allow"

    max_possible = int(ctx.available_cash // cost_per_share)

    min_notional = float(getattr(self, "minimum_notional", 0.0) or 0.0)
    required_qty = max(1, math.ceil(min_notional / P_so))

    if max_possible < required_qty:
        return False, (
            f"Insufficient funds for min order: cash={ctx.available_cash:.2f}, "
            f"cost/share={cost_per_share:.6f}, max_possible={max_possible}, "
            f"required_qty={required_qty}, min_notional={min_notional:.6f}"
        )

    return True, f"Funds OK (max_possible≥{required_qty}) & order notional OK (≥{min_notional:.6f})"


SAFETY_RULES = {
    "RSIUnderDynamicThreshold": rsi_under_dynamic_threshold,
    "CooldownBetweenSOs": cooldown_between_sos,
    "RSIWaveReset": rsi_wave_reset,
    "MaxLevelsNotReached": max_levels_not_reached,
    "SufficientFundsAndNotional": sufficient_funds_and_notional,
    "RSIUnderStaticThreshold": rsi_under_static_threshold
}

# Keep the decider as-is
from ports import SafetyDecider
from rule_chain import build_rule_chain

class SafetyRuleDecider(SafetyDecider):
    """
    Builds its own RuleChain from config (strings or nested ANY/ALL dicts).
    """
    def __init__(self, strategy, names, default_mode: str = "any") -> None:
        self._chain = build_rule_chain(strategy, names, SAFETY_RULES, mode=default_mode)  # type: ignore[arg-type]

    def ok(self, ctx):
        return self._chain.ok(ctx)

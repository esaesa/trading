# rules/exit.py
from typing import Union
from typing import Dict, Any, Tuple, Callable
import numpy as np
from contracts import Ctx
from datetime import datetime, timedelta
from logger_config import logger
import warnings


def atr_take_profit_reached(self: Any, ctx: Ctx) -> Tuple[bool, str]:
    """
    Exit rule: TP dynamically set from ATR%.
    """
    if not self.position:
        return False, "No position"

    atr_pct = getattr(ctx, "current_atr", None)
    if atr_pct is None or (isinstance(atr_pct, float) and np.isnan(atr_pct)):
        prov = getattr(self, "indicator_provider", None)
        if prov is not None:
            try:
                atr_pct = prov.atr_pct(getattr(ctx, "now", None))
            except Exception:
                atr_pct = None

    if atr_pct is None or (isinstance(atr_pct, float) and np.isnan(atr_pct)):
        # 🔴 NEW: warning for debugging
        msg = f"[ATR TP] ATR% unavailable at {getattr(ctx, 'now', None)}"
        logger.warning(msg)
        warnings.warn(msg)   # optional, shows in stdout
        return False, msg

    # Normal ATR TP calc
    f = float(getattr(self, "atr_tp_fraction", 0.65))
    tp_min = float(getattr(self, "atr_tp_min_pct", 0.8))
    tp_max = float(getattr(self, "atr_tp_max_pct", 5.0))

    tp_target = max(tp_min, min(tp_max, float(atr_pct) * f))
    profit_pct = float(self.position.pl_pct)

    ok = profit_pct >= tp_target
    reason = (
        f"ATR TP: profit {profit_pct:.2f}% >= target {tp_target:.2f}% "
        f"(ATR% {float(atr_pct):.2f} × f {f:.2f}, clamp [{tp_min:.2f}–{tp_max:.2f}])"
    )
    return ok, reason

def take_profit_reached(self: Any, ctx: Ctx) -> Tuple[bool, str]:
    """Exit rule based on fixed take profit percentage"""
    if not self.position:
        return False, "No position"
    
    profit_pct = self.position.pl_pct
    take_profit_pct = self.config.take_profit_percentage  # From strategy config
    
    should_exit = profit_pct >= take_profit_pct
    reason = f"Take profit: current profit {profit_pct:.2f}% >= target {take_profit_pct:.2f}%"
    
    return should_exit, reason


def tp_decay_reached(self: Any, ctx: Ctx) -> Tuple[bool, str]:
    """Exit rule using shared logic"""
    if not ctx.entry_price or not ctx.last_entry_time:
        return False, "No position"
    
    profit_pct = self.position.pl_pct
    adjusted_tp = calculate_decaying_tp(
        ctx.last_entry_time,
        ctx.now,
        self.config.take_profit_percentage,
        self.config.take_profit_decay_grace_period_hours,
        self.config.take_profit_decay_duration_hours
    )
    
    should_exit = profit_pct >= adjusted_tp
    reason = f"TP decay: current profit {profit_pct:.2f}% >= adjusted TP {adjusted_tp:.2f}%"
    
    return should_exit, reason


def stop_loss_reached(self: Any,ctx: Ctx) -> Tuple[bool, str]:
    """Exit rule for stop loss"""
    stop_loss_threshold = -25  # -2% stop loss
    should_exit = self.position.pl_pct <= stop_loss_threshold
    if should_exit:
        pass
    reason = f"Stop loss triggered at { self.position.pl_pct:.2f}%"
    
    return should_exit, reason

def trailing_stop_reached(self: Any,ctx: Ctx) -> Tuple[bool, str]:
    """Exit rule for trailing stop"""
    trail_pct = 1.0  # 1% trailing stop
    should_exit = ctx.max_profit_pct - ctx.unrealized_profit_pct >= trail_pct
    reason = f"Trailing stop: drawdown from peak {ctx.max_profit_pct:.2f}% to {ctx.unrealized_profit_pct:.2f}%"
    
    return should_exit, reason

# Dictionary containing all exit rules
EXIT_RULES = {
    "TPDecayReached": tp_decay_reached,
    "StopLossReached": stop_loss_reached,
    "TakeProfitReached": take_profit_reached,
    "ATRTakeProfitReached": atr_take_profit_reached,
    
}

def calculate_decaying_tp(
    entry_time: datetime,
    current_time: datetime,
    initial_tp: float,
    grace_period_hours: Union[int, float],
    decay_duration_hours: Union[int, float],
    min_tp: float = 0.2
) -> float:
    """
    Calculates a decaying take-profit (TP) target based on time since entry.

    The TP remains at `initial_tp` during a `grace_period_hours`. After the grace period,
    the TP linearly decays from `initial_tp` down to `min_tp` over `decay_duration_hours`.

    Args:
        entry_time: The datetime when the position was entered.
        current_time: The current datetime for calculation.
        initial_tp: The initial take-profit percentage.
        grace_period_hours: The duration in hours during which TP does not decay.
        decay_duration_hours: The duration in hours over which TP decays after the grace period.
        min_tp: The minimum take-profit percentage the TP can decay to.

    Returns:
        The adjusted decaying take-profit percentage.

    Raises:
        ValueError: If grace_period_hours or decay_duration_hours are negative,
                    or if initial_tp is less than min_tp.
    """
    # 1. Error Handling and Edge Cases
    if not isinstance(entry_time, datetime) or not isinstance(current_time, datetime):
        # Log a warning or raise a more specific error if context allows
        raise ValueError("entry_time and current_time must be datetime objects.")
    if grace_period_hours < 0:
        raise ValueError("grace_period_hours cannot be negative.")
    if decay_duration_hours <= 0:
        # Decay duration must be positive for decay to occur
        # If decay duration is 0 or negative, TP immediately goes to min_tp (if initial_tp > min_tp)
        return min_tp if initial_tp > min_tp else initial_tp
    if initial_tp < min_tp:
        raise ValueError("initial_tp cannot be less than min_tp.")

    # If entry_time is in the future or current_time is before entry_time,
    # it implies an invalid state or a new position, so return initial_tp.
    if current_time < entry_time:
        return initial_tp

    # Calculate time differences
    time_since_entry: timedelta = current_time - entry_time
    grace_period_td: timedelta = timedelta(hours=grace_period_hours)
    decay_span_td: timedelta = timedelta(hours=decay_duration_hours)

    # 2. Readability and Maintainability
    # If the grace period has not passed, return the original TP
    if time_since_entry <= grace_period_td:
        return initial_tp

    # Calculate time elapsed since the grace period ended
    time_after_grace_period: timedelta = time_since_entry - grace_period_td

    # If the decay phase is complete, return the minimum TP
    if time_after_grace_period >= decay_span_td:
        return min_tp

    # 3. Performance Optimization & Best Practices
    # Calculate the fraction of the decay duration that has passed
    # This is equivalent to (time_after_grace_period.total_seconds() / decay_span_td.total_seconds())
    reduction_factor: float = time_after_grace_period / decay_span_td

    # Calculate the adjusted TP using linear interpolation
    adjusted_tp: float = initial_tp * (1 - reduction_factor) + min_tp * reduction_factor

    # Ensure the adjusted TP is within the valid range [min_tp, initial_tp]
    return max(min_tp, min(initial_tp, adjusted_tp))


from ports import ExitDecider
from rule_chain import build_rule_chain

class ExitRuleDecider(ExitDecider):
    def __init__(self, strategy, names, default_mode: str = "any") -> None:
        self._chain = build_rule_chain(strategy, names, EXIT_RULES, mode=default_mode)

    def ok(self, ctx):
        return self._chain.ok(ctx)

    # NEW: used to capture which rule actually triggered
    def ok_with_reason(self, ctx):
        return self._chain.ok_reason(ctx)

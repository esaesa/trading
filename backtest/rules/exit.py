# rules/exit.py
from typing import Dict, Any, Tuple, Callable

from contracts import Ctx
from datetime import datetime, timedelta

def take_profit_reached(self: Any, ctx: Ctx) -> Tuple[bool, str]:
    """Exit rule based on fixed take profit percentage"""
    if not self.position:
        return False, "No position"
    
    profit_pct = self.position.pl_pct
    take_profit_pct = self.take_profit_percentage  # From strategy config
    
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
        self.take_profit_percentage,
        self.take_profit_decay_grace_period_hours,
        self.take_profit_decay_duration_hours
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
    "TakeProfitReached": take_profit_reached
    
}

def calculate_decaying_tp(entry_time, current_time, initial_tp, grace_period_hours, decay_duration_hours, min_tp=0.2):
    """Shared function for calculating decaying take profit"""
    if entry_time is None:
        return initial_tp
    
    time_since_entry = current_time - entry_time
    grace_period_td = timedelta(hours=grace_period_hours)
    
    # If grace period has not passed, return original TP
    if time_since_entry <= grace_period_td:
        return initial_tp
    
    # Calculate time into decay phase
    time_into_decay = time_since_entry - grace_period_td
    decay_span_td = timedelta(hours=decay_duration_hours)
    
    # If decay phase is complete
    if time_into_decay >= decay_span_td:
        return min_tp
    
    # Linear reduction
    reduction_factor = time_into_decay / decay_span_td
    adjusted_tp = initial_tp - (initial_tp - min_tp) * reduction_factor
    
    return max(min_tp, min(initial_tp, adjusted_tp))


from ports import ExitDecider
from rule_chain import build_rule_chain

class ExitRuleDecider(ExitDecider):
    """
    Builds its own RuleChain from config (strings or nested ANY/ALL dicts).
    """
    def __init__(self, strategy, names, default_mode: str = "any") -> None:
        self._chain = build_rule_chain(strategy, names, EXIT_RULES, mode=default_mode)  # type: ignore[arg-type]
    def ok(self, ctx):
        return self._chain.ok(ctx)

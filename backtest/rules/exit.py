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
        self.take_profit_percentage,  # Could be passed through context
        self.take_profit_reduction_duration_hours
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

def calculate_decaying_tp(entry_time, current_time, initial_tp, reduction_duration_hours, min_tp=0.2):
    """Shared function for calculating decaying take profit"""
    if entry_time is None:
        return initial_tp
    
    time_since_entry = current_time - entry_time
    initial_delay_td = timedelta(hours=reduction_duration_hours)
    
    # If initial delay period has not passed, return original TP
    if time_since_entry <= initial_delay_td:
        return initial_tp
    
    # Calculate time into reduction phase
    time_into_reduction = time_since_entry - initial_delay_td
    reduction_span_td = initial_delay_td
    
    # If reduction phase is complete
    if time_into_reduction >= reduction_span_td:
        return min_tp
    
    # Linear reduction
    reduction_factor = time_into_reduction / reduction_span_td
    adjusted_tp = initial_tp - (initial_tp - min_tp) * reduction_factor
    
    return max(min_tp, min(initial_tp, adjusted_tp))

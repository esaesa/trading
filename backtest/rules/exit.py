# rules/exit.py
from typing import Dict, Any, Tuple, Callable

from contracts import Ctx
from datetime import datetime

def tp_decay_reached(self: Any,ctx: Ctx) -> Tuple[bool, str]:
    """Exit rule based on take profit decay"""
    current_time = ctx.now
    entry_time = ctx.last_entry_time
    profit_pct = self.position.pl_pct
    
    # Time-based decay logic
    time_diff = (current_time - entry_time).total_seconds() / 3600  # hours
    min_profit = max(0.2, 1.0 - (time_diff * 0.1))  # Decay rate of 0.1 per hour
    
    should_exit = profit_pct > min_profit
    reason = f"TP decay: current profit {profit_pct:.2f}% > minimum {min_profit:.2f}%"
    
    return should_exit, reason

def stop_loss_reached(self: Any,ctx: Ctx) -> Tuple[bool, str]:
    """Exit rule for stop loss"""
    stop_loss_threshold = -20  # -2% stop loss
    should_exit = self.position.pl_pct <= stop_loss_threshold
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
    
}
# overlays.py
from __future__ import annotations
import numpy as np
from rules.exit import calculate_decaying_tp

def init_overlays(strategy) -> None:
    """Allocate arrays and register overlay plots."""
    if  strategy.config.show_indicators.get('average_entry_price', False):
        strategy.breakeven_prices = np.full(len(strategy.data), np.nan)
        strategy.breakeven_indicator = strategy.I(
            lambda: strategy.breakeven_prices, name='Average Entry Price', overlay=True
        )
    if  strategy.config.show_indicators.get('take_profit', False):
        strategy.take_profit_prices = np.full(len(strategy.data), np.nan)
        strategy.tp_indicator = strategy.I(
            lambda: strategy.take_profit_prices, name='Take Profit', overlay=True, color='red'
        )

def update_overlays(strategy, current_time) -> None:
    """Write breakeven & TP values for the current bar."""
    idx = len(strategy.data) - 1

    # Breakeven/TP lines only if that overlay is on
    if  strategy.config.show_indicators.get('average_entry_price', False):
        if strategy.position:
            entry_price = strategy.get_entry_price()
            strategy.breakeven_prices[idx] = entry_price

            if strategy.config.show_indicators.get('take_profit', False):
                adjusted_tp_percentage = calculate_decaying_tp(
                    strategy.last_safety_order_time or strategy.base_order_time,
                    current_time,
                    strategy.config.take_profit_percentage,
                    strategy.config.take_profit_decay_grace_period_hours,
                    strategy.config.take_profit_decay_duration_hours
                )
                strategy.take_profit_prices[idx] = entry_price * (1 + adjusted_tp_percentage / 100)
        else:
            strategy.breakeven_prices[idx] = None

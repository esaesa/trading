from datetime import datetime
from typing import TYPE_CHECKING, Union, Dict, List, Any
import numpy as np
from contracts import Ctx
from reporting import render_cycle_plan_table, render_exit_summary_table
from instrumentation import log_trade_info
from simulation import apply_slippage
from logger_config import logger

if TYPE_CHECKING:
    from strategy import DCAStrategy

# Constants
MARGIN_BUFFER_FACTOR = 0.99


def _uses_dynamic_pricing(price_mode_config: Union[str, Dict[str, Any]]) -> bool:
    """Check if dynamic pricing is used anywhere in the configuration.

    Args:
        price_mode_config: Either a string ("static", "dynamic") or a dict composite spec

    Returns:
        True if dynamic pricing is used anywhere in the specification
    """
    if isinstance(price_mode_config, str):
        return price_mode_config.lower() == "dynamic"
    elif isinstance(price_mode_config, dict):
        # Recursively search through composite specifications
        for operation, engine_specs in price_mode_config.items():
            if isinstance(engine_specs, list):
                for spec in engine_specs:
                    if _uses_dynamic_pricing(spec):
                        return True
        return False
    else:
        return False


class TradeProcessor:
    def __init__(self, strategy: 'DCAStrategy'):
        self.strategy = strategy

    def process_entry(self, ctx: Ctx, price: float, current_time: datetime) -> bool:
        """Handle entry logic"""
        entry_ok, entry_reason = self.strategy.entry_decider.ok(ctx)
        if not self.strategy.position and entry_ok:
            rsi_now = self.strategy.indicator_service.get_indicator_value("rsi", current_time, np.nan)
            qty, investment = self.strategy.entry_sizer.qty_and_investment(ctx, price, self.strategy.commission_calc)
            if qty < 1:
                raise ValueError("Backtesting.py does not support fractional units. Adjust your entry fraction.")

            order = self.strategy.buy(size=qty, tag="BO")
            self.strategy.base_order_time = current_time
            self.strategy.last_safety_order_time = None
            self.strategy.dca_level = 0
            self.strategy.base_order_price = price
            self.strategy.initial_investment = investment
            self.strategy.last_filled_price = price
            self.strategy.base_order_quantity = qty
            self.strategy.base_order_value = qty * self.strategy.base_order_price
            self.strategy._cycle_cash_entry = float(self.strategy._broker._cash)
            self.strategy._cycle_invested = qty * price * (1.0 + self.strategy.commission_rate)
            self.strategy._cycle_peak_invested = self.strategy._cycle_invested

            # Start cycle buffer
            self.strategy.state_manager.start_cycle(current_time, price)

            if self.strategy.config.debug_trade:
                logger.debug(f"\nEntry: Price {price:.10f}, Qty {qty}, RSI {rsi_now:.2f} @ {current_time}.")
                table = render_cycle_plan_table(self.strategy, qty)
                logger.info(table)
                log_trade_info(self.strategy, "Entry", order, price, qty * price)
            return True
        return False

    def process_dca(self, ctx: Ctx, price: float, current_time: datetime) -> None:
        """Handle DCA logic"""
        safety_ok, safety_reason = self.strategy.safety_decider.ok(ctx)
        if not safety_ok:
            if self.strategy.config.debug_dca_skips:
                logger.debug(f"SO skipped due to {safety_reason}. @ {current_time}.")
            return
        level = self.strategy.dca_level + 1
        so_price = self.strategy.price_engine.so_price(ctx, level)
        if price > so_price:
            # Use the new specific flag for skip logging instead of general debug_trade
            if self.strategy.config.debug_dca_skips:
                logger.debug(f"Price {price:.10f} > SO-{level} trigger {so_price:.10f}. Skipping DCA. @ {current_time}.")
            return
        
        # Keep debug_trade for actual DCA executions (useful debugging)
        so_size = self.strategy.size_engine.so_size(ctx, price, level)
        size_to_buy = self.strategy.affordability_guard.clamp_qty(
            desired_qty=so_size,
            price=price,
            available_cash=self.strategy._broker.margin_available * self.strategy._broker._leverage * MARGIN_BUFFER_FACTOR,
            commission=self.strategy.commission_calc,
            min_notional=self.strategy.config.minimum_notional,
        )
        
        if self.strategy.config.debug_trade:
            # Get current RSI value for additional context
            rsi_now = self.strategy.indicator_service.get_indicator_value("rsi", current_time, np.nan)
            rsi_info = f", RSI {rsi_now:.2f}" if not np.isnan(rsi_now) else ""
            logger.debug(
                f"Price {price:.10f} <= SO-{level} trigger {so_price:.10f}. "
                f"Proceeding with DCA: Size {size_to_buy:.8f}{rsi_info}. @ {current_time}."
            )
        order = None
        if size_to_buy > 0:
            order = self.strategy.buy(size=size_to_buy, tag=f"S{level}")
            self.strategy.commit_trade_state()
            delta = size_to_buy * price * (1.0 + self.strategy.commission_rate)
            self.strategy._cycle_invested += delta
            self.strategy._cycle_peak_invested = max(self.strategy._cycle_peak_invested, self.strategy._cycle_invested)
            self.strategy.state_manager.log_so_fill(level, current_time)

        self.strategy.last_safety_order_time = current_time
        self.strategy.dca_level += 1
        if _uses_dynamic_pricing(self.strategy.config.safety_order_price_mode):
            self.strategy.last_filled_price = price
        if self.strategy.config.debug_trade and order is not None:
            log_trade_info(self.strategy, "DCA", order, price, size_to_buy * price)

    def process_exit(self, ctx: Ctx, price: float, current_time: datetime) -> bool:
        """Handle exit logic"""
        if not self.strategy.position:
            return False

        ok, exit_reason = self.strategy.exit_decider.ok(ctx)
        if not ok:
            return False

        # Capture ROI before closing
        ok, exit_reason = self.strategy.exit_decider.ok(ctx)
        roi_pct = float(self.strategy.position.pl_pct)
        if self.strategy.config.debug_process:
            table = render_exit_summary_table(self.strategy, price, current_time)
            logger.info(table)
            logger.debug(f"Trade: Exit Triggered at {current_time} price {price:.10f} ({exit_reason})")

        # Finalize current cycle before state reset
        self.strategy.state_manager.finalize_cycle(
            exit_time=current_time,
            exit_price=price,
            roi_pct=roi_pct,
            exit_reason=exit_reason
        )

        self.strategy.position.close()
        cash0 = self.strategy._cycle_cash_entry or 0.0
        peak = self.strategy._cycle_peak_invested or 0.0
        util_pct = (peak / cash0 * 100.0) if cash0 > 0 else 0.0
        self.strategy.cycle_cash_records.append({
            "cash_at_entry": cash0,
            "peak_invested": peak,
            "util_pct": util_pct,
        })
        self.strategy.state_manager.reset_process()
        return True

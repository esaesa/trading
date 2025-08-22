from datetime import datetime
from typing import TYPE_CHECKING
import numpy as np
from contracts import Ctx
from reporting import render_cycle_plan_table, render_exit_summary_table
from instrumentation import log_trade_info
from simulation import apply_slippage
from logger_config import logger

if TYPE_CHECKING:
    from strategy import DCAStrategy

class TradeProcessor:
    def __init__(self, strategy: 'DCAStrategy'):
        self.strategy = strategy

    def process_entry(self, ctx: Ctx, price: float, current_time: datetime) -> bool:
        """Handle entry logic"""
        if not self.strategy.position and self.strategy.entry_decider.ok(ctx):
            rsi_now = ctx.indicators.get("rsi", np.nan)
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
                table = render_cycle_plan_table(self.strategy, qty)
                logger.info(table)
                log_trade_info(self.strategy, "Entry", order, price, qty * price)
            return True
        return False

    def process_dca(self, ctx: Ctx, price: float, current_time: datetime) -> None:
        """Handle DCA logic"""
        if not self.strategy.safety_decider.ok(ctx):
            return
        level = self.strategy.dca_level + 1
        so_price = self.strategy.price_engine.so_price(ctx, level)
        if price > so_price:
            if self.strategy.config.debug_trade:
                logger.debug(f"Price {price:.10f} > SO-{level} trigger {so_price:.10f}")
            return
        so_size = self.strategy.size_engine.so_size(ctx, so_price, level)
        size_to_buy = self.strategy.affordability_guard.clamp_qty(
            desired_qty=so_size,
            price=so_price,
            available_cash=self.strategy._broker.margin_available * self.strategy._broker._leverage * 0.99,
            commission=self.strategy.commission_calc,
            min_notional=self.strategy.config.minimum_notional,
        )
        order = None
        if size_to_buy > 0:
            order = self.strategy.buy(size=size_to_buy, tag=f"S{level}")
            delta = size_to_buy * so_price * (1.0 + self.strategy.commission_rate)
            self.strategy._cycle_invested += delta
            self.strategy._cycle_peak_invested = max(self.strategy._cycle_peak_invested, self.strategy._cycle_invested)
            self.strategy.state_manager.log_so_fill(level, current_time)

        self.strategy.last_safety_order_time = current_time
        self.strategy.dca_level += 1
        if self.strategy.config.safety_order_price_mode.lower() == "dynamic":
            self.strategy.last_filled_price = price
        if self.strategy.config.debug_trade and order is not None:
            log_trade_info(self.strategy, "DCA", order, so_price, size_to_buy * so_price)

    def process_exit(self, ctx: Ctx, price: float, current_time: datetime) -> bool:
        """Handle exit logic"""
        if not self.strategy.position:
            return False

        # Try to get which exit rule passed
        exit_reason = ""
        ok_with_reason = getattr(self.strategy.exit_decider, "ok_with_reason", None)
        if callable(ok_with_reason):
            ok, reason = ok_with_reason(ctx)
            if not ok:
                return False
            exit_reason = reason or ""
        else:
            if not self.strategy.exit_decider.ok(ctx):
                return False

        # Capture ROI before closing
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

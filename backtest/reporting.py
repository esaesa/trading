# reporting.py
from __future__ import annotations
from typing import Any
from rich.table import Table
from datetime import datetime
from logger_config import logger

def render_cycle_plan_table(strategy: Any, base_order_quantity: float) -> Table:
    """
    Generates the BO + planned DCA table at entry time using the configured engines.
    Uses a live ctx snapshot (strategy._ctx()) for consistency.
    """
    price = strategy.base_order_price
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Stage")
    table.add_column("Price (BO% - PrevCh%)")
    table.add_column("Size")
    table.add_column("Value")
    table.add_column("Cumulative Value")
    table.add_column("Cumulative Size")

    cumulative_value = 0.0
    cumulative_size = 0.0

    # BO row
    bo_value = price * base_order_quantity
    cumulative_value += bo_value
    cumulative_size += base_order_quantity
    table.add_row(
        "BO",
        f"{price:.10f} (0.00% - 0.00%)",
        f"{base_order_quantity:.2f}",
        f"{bo_value:.2f}",
        f"{cumulative_value:.2f}",
        f"{cumulative_size:.2f}",
    )

    # Engines + ctx
    price_mode = (strategy.safety_order_price_mode or "dynamic").lower()
    so_mode    = (strategy.safety_order_mode or "value").lower()
    ctx        = strategy._ctx()  # live snapshot

    if price_mode == "static":
        # Compute full static plan using engines
        prev_so_price = None
        for i in range(1, strategy.max_dca_levels + 1):
            so_price = strategy.price_engine.so_price(ctx, i)
            so_size  = strategy.size_engine.so_size(ctx, so_price, i)
            so_value = so_price * so_size

            bo_change = ((so_price - strategy.base_order_price) / strategy.base_order_price) * 100.0
            if prev_so_price is None:
                prev_change = bo_change
            else:
                prev_change = ((so_price - prev_so_price) / prev_so_price) * 100.0

            cumulative_value += so_value
            cumulative_size  += so_size
            prev_so_price     = so_price

            table.add_row(
                f"DCA-{i}",
                f"{so_price:.10f} ({bo_change:.2f}% - {prev_change:.2f}%)",
                f"{so_size:.2f}",
                f"{so_value:.2f}",
                f"{cumulative_value:.2f}",
                f"{cumulative_size:.2f}",
            )

    else:
        # Dynamic price mode:
        # - Volume sizing: we can show sizes (price unknown)
        # - Value sizing: final sizes depend on fill prices → show N/A
        if so_mode == "volume":
            for i in range(1, strategy.max_dca_levels + 1):
                # price argument is ignored by VolumeModeSizeEngine
                so_size = strategy.size_engine.so_size(ctx, float("nan"), i)
                cumulative_size += so_size
                table.add_row(
                    f"DCA-{i} [DYN]",
                    "N/A - dynamic",
                    f"{so_size:.2f}",
                    "N/A",
                    "N/A",
                    f"{cumulative_size:.2f}",
                )
        else:
            for i in range(1, strategy.max_dca_levels + 1):
                table.add_row(
                    f"DCA-{i} [DYN]",
                    "N/A - dynamic",
                    "N/A",
                    "N/A",
                    "N/A",
                    "N/A",
                )

    return table


def render_exit_summary_table(strategy: Any, price: float, current_time: datetime) -> Table:
    """
    Summarizes final position at exit using executed trades + current price.
    """
    table = Table(show_header=True, header_style="bold green")
    table.add_column("Stage")
    table.add_column("Entry Price")
    table.add_column("Size")
    table.add_column("Value")
    table.add_column("Cumulative Value")
    table.add_column("Cumulative Size")
    table.add_column("PnL")
    table.add_column("ROI")

    cumulative_value = 0.0
    cumulative_size = 0.0

    # BO trade
    bo_trade = next((t for t in strategy.trades if t.is_long and t.tag == "BO"), None)
    if bo_trade:
        bo_price = bo_trade.entry_price
        bo_size  = bo_trade.size
        bo_value = bo_price * bo_size
        cumulative_value += bo_value
        cumulative_size  += bo_size
        # Commission display here is simplistic; your engine already accounts for it on orders
        bo_pnl = bo_trade.pl
        bo_roi = bo_trade.pl_pct * 100.0

        table.add_row(
            "BO",
            f"{bo_price:.10f}",
            f"{bo_size:.2f}",
            f"{bo_value:.2f}",
            f"{cumulative_value:.2f}",
            f"{cumulative_size:.2f}",
            f"{bo_pnl:.2f}",
            f"{bo_roi:.2f}%",
        )

    # SO trades
    so_trades = [t for t in strategy.trades if t.is_long and t.tag and t.tag.startswith("S")]
    for i, so_trade in enumerate(so_trades, start=1):
        so_price = so_trade.entry_price
        so_size  = so_trade.size
        so_value = so_price * so_size
        pnl      = (price - so_price) * so_size
        roi      = (pnl / so_value * 100.0) if so_value else 0.0

        cumulative_value += so_value
        cumulative_size  += so_size

        table.add_row(
            f"DCA-{i}",
            f"{so_price:.10f}",
            f"{so_size:.2f}",
            f"{so_value:.2f}",
            f"{cumulative_value:.2f}",
            f"{cumulative_size:.2f}",
            f"{pnl:.2f}",
            f"{roi:.2f}%",
        )

    # Final TP row (current price)
    if strategy.position:
        tp_value = strategy.position.size * price
        tp_pnl   = strategy.position.pl
        tp_roi   = strategy.position.pl_pct
        table.add_row(
            "TP",
            f"{price:.10f}",
            f"{strategy.position.size:.2f}",
            f"{tp_value:.2f}",
            "-",
            "-",
            f"{tp_pnl:.2f}",
            f"{tp_roi:.2f}%",
        )

    return table

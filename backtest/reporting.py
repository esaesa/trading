# reporting.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
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
# reporting.py (add this helper at bottom or a new section)

from collections import Counter
import statistics as stats

def _hours(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    try:
        return round(float(x) / 3600.0, 2)  # seconds → hours
    except Exception:
        return None

def aggregate_cycles(cycles: List[Dict[str, Any]],
                     cash_records: Optional[List[Dict[str, float]]] = None) -> Dict[str, Any]:
    """
    Aggregate per-cycle metrics (readable hours) and cash utilization.
    Expects each cycle dict to *optionally* include:
      - 'roi_percentage' or 'roi_pct'
      - 'dca_levels_executed'
      - 'max_level_reached'
      - 'cycle_duration_sec'
      - 'time_in_last_level_sec'
      - 'bo_to_first_so_sec'
      - 'max_so_wait_sec'
    Any missing fields are skipped gracefully.
    """
    n = len(cycles)
    if n == 0:
        return {"cycles": 0}

    # ---- numeric extractions (robust to missing keys) ----
    roi_list = []
    dca_exec = []
    max_level = []
    dur = []
    last_lvl = []
    bo_to_first = []
    max_so_wait = []

    for c in cycles:
        if c.get("roi_percentage") is not None:
            roi_list.append(float(c["roi_percentage"]))
        elif c.get("roi_pct") is not None:
            roi_list.append(float(c["roi_pct"]))

        dca_exec.append(int(c.get("dca_levels_executed", c.get("levels_executed", 0))))
        max_level.append(int(c.get("max_level_reached", c.get("max_level", c.get("dca_levels_executed", 0)))))

        v = c.get("cycle_duration_sec")
        if v is not None: dur.append(float(v))
        v = c.get("time_in_last_level_sec")
        if v is not None: last_lvl.append(float(v))
        v = c.get("bo_to_first_so_sec")
        if v is not None: bo_to_first.append(float(v))
        v = c.get("max_so_wait_sec")
        if v is not None: max_so_wait.append(float(v))

    # ---- safe stats ----
    def avg(xs): return round(statistics.mean(xs), 6) if xs else 0.0
    def med(xs): return round(statistics.median(xs), 6) if xs else 0.0

    levels_only = [lvl for lvl in dca_exec if lvl is not None]
    freq = dict(sorted(Counter([lvl for lvl in max_level if lvl is not None and lvl > 0]).items()))
    wins = [roi for roi in roi_list if roi is not None and roi >= 0]
    win_rate = round(100.0 * len(wins) / len(roi_list), 4) if roi_list else 0.0

    # ---- durations → hours ----
    out = {
        "cycles": n,
        "win_rate": win_rate,
        "avg_roi_pct": avg(roi_list),
        "median_roi_pct": med(roi_list),
        "max_level_reached_max": max(max_level) if max_level else 0,
        "max_level_reached_avg": avg(max_level) if max_level else 0.0,
        "dca_levels_executed_avg": avg(dca_exec) if dca_exec else 0.0,
        "avg_cycle_duration_hours": _hours(avg(dur)) if dur else 0.0,
        "max_cycle_duration_hours": _hours(max(dur)) if dur else 0.0,
        "avg_time_in_last_level_hours": _hours(avg(last_lvl)) if last_lvl else 0.0,
        "avg_bo_to_first_so_hours": _hours(avg(bo_to_first)) if bo_to_first else 0.0,
        "avg_max_so_wait_hours": _hours(avg(max_so_wait)) if max_so_wait else 0.0,
        "per_level_frequency": freq,
    }

    # ---- cash utilization from strategy tracker ----
    cash_records = cash_records or []
    util_pcts = [float(r.get("util_pct", 0.0)) for r in cash_records if r.get("util_pct") is not None]
    if util_pcts:
        util_pcts.sort()
        p90 = util_pcts[int(0.9 * (len(util_pcts) - 1))] if len(util_pcts) > 1 else util_pcts[0]
        out.update({
            "peak_cash_utilization_pct": round(max(util_pcts), 2),
            "avg_cash_utilization_pct": round(avg(util_pcts), 2),
            "p90_cash_utilization_pct": round(p90, 2),
            "cycles_over_80pct": sum(1 for u in util_pcts if u >= 80.0),
            "cycles_over_90pct": sum(1 for u in util_pcts if u >= 90.0),
        })
    else:
        out.update({
            "peak_cash_utilization_pct": 0.0,
            "avg_cash_utilization_pct": 0.0,
            "p90_cash_utilization_pct": 0.0,
            "cycles_over_80pct": 0,
            "cycles_over_90pct": 0,
        })

    return out
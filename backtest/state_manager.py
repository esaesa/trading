from datetime import datetime
from typing import List, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from strategy import DCAStrategy

class StateManager:
    def __init__(self, strategy: 'DCAStrategy'):
        self.strategy = strategy
        self._cycle = None

    def start_cycle(self, current_time: datetime, entry_price: float) -> None:
        """Start a new trading cycle"""
        self._cycle = {
            "entry_time": current_time,
            "entry_price": float(entry_price),
            "so_fills": [],  # list of {"level": int, "time": datetime}
        }

    def log_so_fill(self, level: int, timestamp: datetime) -> None:
        """Log a safety order fill"""
        if self._cycle is not None:
            self._cycle["so_fills"].append({"level": int(level), "time": timestamp})

    def finalize_cycle(self, exit_time: datetime, exit_price: float, roi_pct: float, exit_reason: str = None) -> None:
        """Finalize the current cycle and record statistics"""
        if not self._cycle:
            return

        entry_time = self._cycle["entry_time"]
        entry_price = self._cycle["entry_price"]
        so_fills = sorted(self._cycle["so_fills"], key=lambda x: x["time"])

        # DCA stats
        levels = [f["level"] for f in so_fills]
        max_level = max(levels) if levels else 0
        dca_levels_executed = len(levels)

        # Times
        cycle_duration_sec = (exit_time - entry_time).total_seconds()
        if levels:
            # waits between SOs
            waits = [(so_fills[i]["time"] - so_fills[i-1]["time"]).total_seconds()
                     for i in range(1, len(so_fills))]
            max_so_wait_sec = max(waits) if waits else 0.0
            bo_to_first_so_sec = (so_fills[0]["time"] - entry_time).total_seconds()
            last_so_time = so_fills[-1]["time"]
            time_in_last_level_sec = (exit_time - last_so_time).total_seconds()
            avg_so_to_exit_sec = sum((exit_time - f["time"]).total_seconds() for f in so_fills) / len(so_fills)
        else:
            max_so_wait_sec = 0.0
            bo_to_first_so_sec = None
            time_in_last_level_sec = (exit_time - entry_time).total_seconds()
            avg_so_to_exit_sec = None

        cycle_price_change_pct = ((exit_price / entry_price) - 1.0) * 100.0 if entry_price else 0.0

        self.strategy.completed_processes.append({
            "entry_time": entry_time,
            "exit_time": exit_time,
            "entry_price": entry_price,
            "exit_price": float(exit_price),
            "roi_percentage": float(roi_pct),
            "cycle_duration_sec": cycle_duration_sec,
            "cycle_price_change_pct": cycle_price_change_pct,
            "dca_levels_executed": dca_levels_executed,
            "max_level_reached": max_level,
            "levels_reached": levels,
            "bo_to_first_so_sec": bo_to_first_so_sec,
            "max_so_wait_sec": max_so_wait_sec,
            "time_in_last_level_sec": time_in_last_level_sec,
            "avg_so_to_exit_sec": avg_so_to_exit_sec,
            "exit_reason": exit_reason or "",
        })
        self._cycle = None

    def reset_process(self) -> None:
        """Reset process state for new cycle"""
        self.strategy.base_order_price = None
        self.strategy.base_order_quantity = None
        self.strategy.base_order_value = None
        self.strategy.dca_level = 0
        self.strategy.last_filled_price = None
        self.strategy.rsi_reset = True
        self.strategy.base_order_time = None
        self.strategy.last_safety_order_time = None
        self.strategy._cycle_cash_entry = 0.0
        self.strategy._cycle_invested = 0.0
        self.strategy._cycle_peak_invested = 0.0

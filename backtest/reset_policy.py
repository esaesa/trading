# reset_policy.py
from __future__ import annotations
import numpy as np

def need_reset(strategy) -> bool:
    """Static-RSI reset is relevant only when: RSI is available, dynamic threshold is OFF, and the user asked for reset."""
    return (
        not getattr(strategy, "rsi_dynamic_threshold", False)
        and getattr(strategy, "require_rsi_reset", False)
    )

def update_reset(strategy, rsi_val: float) -> None:
    """
    If reset is relevant:
      - set rsi_reset = True when RSI is NaN, or RSI >= reset_thr
    where reset_thr = rsi_threshold * (1 + rsi_reset_percentage/100).
    """
    if not need_reset(strategy):
        return

    if np.isnan(rsi_val):
        strategy.rsi_reset = True
        return

    rsi_thr = getattr(strategy, "rsi_threshold", 50) or 50
    reset_pct = getattr(strategy, "rsi_reset_percentage", 50) or 50
    reset_thr = rsi_thr * (1 + reset_pct / 100.0)

    if rsi_val >= reset_thr:
        strategy.rsi_reset = True

def consume_reset(strategy) -> None:
    """
    Immediately after placing BO/SO in static mode, consume the reset:
      rsi_reset = False  (only if reset is relevant)
    """
    if need_reset(strategy):
        strategy.rsi_reset = False

# indicators_provider.py
from __future__ import annotations
from typing import Protocol, Optional
import numpy as np

class IndicatorProvider(Protocol):
    def ema(self, now=None) -> Optional[float]: ...
    def atr_pct(self, now=None) -> Optional[float]: ...
    def rsi(self, now=None) -> Optional[float]: ...
    def laguerre_rsi(self, now=None) -> Optional[float]: ...
    def bbw(self, now=None) -> Optional[float]: ...
    def cv(self, now=None) -> Optional[float]: ...

class StrategyIndicatorProvider:
    """Thin adapter that reads indicators off Strategy safely."""
    def __init__(self, strategy) -> None:
        self.s = strategy

    def _at_or_prev(self, series, now):
        """
        Return the series value at 'now' or the most recent value BEFORE 'now'
        (no lookahead). Uses searchsorted (fast, no exceptions).
        """
        if series is None or now is None:
            return None
        try:
            idx = series.index
            pos = idx.searchsorted(now, side="right") - 1
            if pos >= 0:
                return float(series.iloc[pos])
            return None  # now is before first index -> no value
        except Exception:
            return None

    def ema(self, now=None):
        return self._at_or_prev(getattr(self.s, "ema_dynamic", None), now)

    def atr_pct(self, now=None):
        return self._at_or_prev(getattr(self.s, "atr_pct_series", None), now)

    def rsi(self, now=None):
        return self._at_or_prev(getattr(self.s, "rsi_values", None), now)

    def laguerre_rsi(self, now=None):
        return self._at_or_prev(getattr(self.s, "laguerre_series", None), now)

    def bbw(self, now=None):
        return self._at_or_prev(getattr(self.s, "bbw_series", None), now)

    def cv(self, now=None):
        return self._at_or_prev(getattr(self.s, "cv_series", None), now)

# indicators_provider.py
from __future__ import annotations
from typing import Protocol, Optional
import numpy as np

class IndicatorProvider(Protocol):
    def ema(self, now=None) -> Optional[float]: ...
    def atr_pct(self, now=None) -> Optional[float]: ...
    def rsi(self, now=None) -> Optional[float]: ...

class StrategyIndicatorProvider:
    """Thin adapter that reads indicators off Strategy safely."""
    def __init__(self, strategy) -> None:
        self.s = strategy

    def _last_or_at(self, series, now):
        if series is None:
            return None
        try:
            if now is not None:
                return float(series.loc[now])
        except Exception:
            pass
        try:
            return float(series.iloc[-1])
        except Exception:
            return None

    def ema(self, now=None) -> Optional[float]:
        ser = getattr(self.s, "ema_dynamic", None)
        return self._last_or_at(ser, now)

    def atr_pct(self, now=None) -> Optional[float]:
        ser = getattr(self.s, "atr_pct_series", None)
        return self._last_or_at(ser, now)

    def rsi(self, now=None) -> Optional[float]:
        ser = getattr(self.s, "rsi_values", None)
        return self._last_or_at(ser, now)

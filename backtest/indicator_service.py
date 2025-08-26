# indicator_service.py
from __future__ import annotations
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd

class IndicatorService:
    """
    Single source of truth for indicator values.
    Clean architecture: Caching, validation, and direct access to strategy attributes.
    """
    def __init__(self, strategy):
        self.strategy = strategy
        self.series_mapping = {
            "rsi": "rsi_values",
            "ema": "ema_dynamic",
            "atr_pct": "atr_pct_series",
            "laguerre_rsi": "laguerre_series",
            "bbw": "bbw_series",
            "cv": "cv_series",
            "dynamic_rsi_threshold": "rsi_dynamic_threshold_series",
        }
        # Cache for performance
        self._cache = {}

    def _validate_timestamp(self, timestamp) -> bool:
        """Validate timestamp is within strategy data range"""
        if not hasattr(self.strategy, 'data') or self.strategy.data is None:
            return False
        try:
            # Check if timestamp exists in index or is within range
            index = self.strategy.data.index
            return index.min() <= timestamp <= index.max()
        except Exception:
            return False

    def _store_series(self, name: str, series):
        """Store series in IndicatorService (single point of truth)"""
        self._cache[name] = series

    def _get_series(self, name: str):
        """Get series from IndicatorService cache (single source of truth)"""
        return self._cache.get(name)

    def get_indicator_value(self, name: str, timestamp, default=np.nan) -> float:
        """Single point of truth for indicator values with caching and validation"""
        # Validate timestamp
        if not self._validate_timestamp(timestamp):
            return default

        # Get series (with caching)
        series = self._get_series(name)
        if series is None or len(series) == 0:
            return default

        try:
            # Try exact timestamp match
            return float(series.loc[timestamp])
        except KeyError:
            try:
                # Fallback to latest value if timestamp not found
                return float(series.iloc[-1])
            except (IndexError, TypeError, ValueError):
                return default


    def clear_cache(self):
        """Clear cache for memory management"""
        self._cache.clear()
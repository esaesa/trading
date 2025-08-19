# indicators_bootstrap.py
from __future__ import annotations
from indicators_manager import IndicatorManager
import numpy as np

class IndicatorsBootstrap:
    """
    SRP: decide which indicators to compute, compute them once, and (optionally)
    register plots on the strategy via strategy.I(...).
    Writes the same attributes that DCAStrategy previously populated:
      - rsi_values
      - rsi_dynamic_threshold_series
      - ema_dynamic
      - atr_pct_series
      - laguerre_series
      - bbw_series
      - cv_series
    """

    def __init__(self, params: dict):
        self.params = params or {}

    def _resolve_dynamic_rsi_window(self, strategy) -> int:
        # Keep original logic: use explicit param if present, else 1000 * rsi_window (fallback 14)
        if self.params.get("rsi_dynamic_window") is not None:
            return int(self.params["rsi_dynamic_window"])
        return int((getattr(strategy, "rsi_window", None) or 14) * 1000)

    def run(self, strategy) -> None:
        # attach manager (idempotent)
        if not hasattr(strategy, "indicators"):
            df = strategy.data.df
            strategy.indicators = IndicatorManager(df)

        # compute decisions
        show = getattr(strategy, "show_indicators", {}) or {}
        calc_rsi = show.get("rsi", False) or getattr(strategy, "enable_rsi_calculation", False)
        calc_atr = show.get("atr", False) or getattr(strategy, "enable_atr_calculation", False)
        calc_ema = show.get("ema", False) or getattr(strategy, "enable_ema_calculation", False)
        calc_lag = show.get("laguerre", False) or getattr(strategy, "enable_laguere_calculation", False)
        calc_bbw = show.get("bbw", False)
        calc_cv  = show.get("cv", False)
        calc_dyn = show.get("dynamic_rsi", False) or getattr(strategy, "rsi_dynamic_threshold", False)

        # If dynamic RSI, RSI reset, or overbought gate is used → must compute RSI
        if calc_dyn or getattr(strategy, "require_rsi_reset", False) or getattr(strategy, "avoid_rsi_overbought", False):
            calc_rsi = True

        # ---- RSI ----
        rsi_series = None
        if calc_rsi:
            rsi_series = strategy.indicators.compute_rsi(
                window=getattr(strategy, "rsi_window", 14),
                resample_interval=getattr(strategy, "rsi_resample_interval", None),
            )
            strategy.rsi_values = rsi_series
            if show.get("rsi", False):
                strategy.I(lambda _: strategy.rsi_values, strategy.data.Close, name="RSI")

        # ---- Dynamic RSI threshold ----
        if calc_dyn and (rsi_series is not None):
            strategy.rsi_dynamic_window = self._resolve_dynamic_rsi_window(strategy)
            strategy.rsi_dynamic_threshold_series = (
                rsi_series.rolling(window=strategy.rsi_dynamic_window)
                          .quantile(getattr(strategy, "rsi_percentile", 0.05))
            )
            if show.get("dynamic_rsi", True):
                strategy.I(lambda _: strategy.rsi_dynamic_threshold_series,
                           strategy.data.Close,
                           name="Dynamic RSI Threshold")

        # ---- EMA ----
        if calc_ema:
            ema_series = strategy.indicators.compute_ema(
                window=getattr(strategy, "ema_window", 200),
                resample_interval=getattr(strategy, "ema_resample_interval", "1h"),
            )
            strategy.ema_dynamic = ema_series
            if show.get("ema", False):
                strategy.I(lambda _: ema_series, strategy.data.Close, name="EMA")

        # ---- ATR% ----
        if calc_atr:
            atr_pct = strategy.indicators.compute_atr_percentage(
                window=getattr(strategy, "atr_window", 14),
                resample_interval=getattr(strategy, "atr_resample_interval", "1D"),
            )
            strategy.atr_pct_series = atr_pct
            if show.get("atr", False):
                strategy.I(lambda _: atr_pct, strategy.data.Close, name="ATR%")

        # ---- CV ----
        if calc_cv:
            cv_series = strategy.indicators.compute_coefficient_of_variation(window=40)
            strategy.cv_series = cv_series
            if show.get("cv", False):
                strategy.I(lambda _: cv_series, strategy.data.Close, name="CV")

        # ---- Laguerre RSI ----
        if calc_lag:
            lag = strategy.indicators.compute_laguerre_rsi(gamma=0.1, resample_interval="4h")
            strategy.laguerre_series = lag
            if show.get("laguerre", False):
                strategy.I(lambda _: lag, strategy.data.Close, name="Laguerre RSI")

        # ---- BBW ----
        if calc_bbw:
            bbw = strategy.indicators.compute_bbw(window=20, num_std=2)
            strategy.bbw_series = bbw
            if show.get("bbw", False):
                strategy.I(lambda _: bbw, strategy.data.Close, name="BBW")

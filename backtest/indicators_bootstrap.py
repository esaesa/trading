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
        return int((strategy.config.rsi_window or 14) * 1000)

    def run(self, strategy) -> None:
        # attach manager (idempotent)
        if not hasattr(strategy, "indicators"):
            df = strategy.data.df
            strategy.indicators = IndicatorManager(df)

        # compute decisions
        show = strategy.config.show_indicators or {}
        calc_rsi = show.get("rsi", False) or strategy.config.enable_rsi_calculation
        calc_atr = show.get("atr", False) or strategy.config.enable_atr_calculation
        calc_ema = show.get("ema", False) or strategy.config.enable_ema_calculation
        calc_lag = show.get("laguerre", False) or strategy.config.enable_laguere_calculation
        calc_bbw = show.get("bbw", False)
        calc_cv  = show.get("cv", False)
        calc_dyn = show.get("dynamic_rsi", False) or strategy.config.rsi_dynamic_threshold

        # If dynamic RSI, RSI reset, or overbought gate is used → must compute RSI
        if calc_dyn or strategy.config.require_rsi_reset or strategy.config.avoid_rsi_overbought:
            calc_rsi = True

        # ---- RSI ----
        rsi_series = None
        if calc_rsi:
            rsi_series = strategy.indicators.compute_rsi(
                window=strategy.config.rsi_window,
                resample_interval=strategy.config.rsi_resample_interval,
            )
            strategy.rsi_values = rsi_series
            if show.get("rsi", False):
                strategy.I(lambda _: strategy.rsi_values, strategy.data.Close, name="RSI")

        # ---- Dynamic RSI threshold ----
        if calc_dyn and (rsi_series is not None):
            strategy.rsi_dynamic_window = self._resolve_dynamic_rsi_window(strategy)
            strategy.rsi_dynamic_threshold_series = (
                rsi_series.rolling(window=strategy.rsi_dynamic_window)
                          .quantile(strategy.config.rsi_percentile)
            )
            if show.get("dynamic_rsi", True):
                strategy.I(lambda _: strategy.rsi_dynamic_threshold_series,
                           strategy.data.Close,
                           name="Dynamic RSI Threshold")

        # ---- EMA ----
        if calc_ema:
            ema_series = strategy.indicators.compute_ema(
                window=strategy.config.ema_window,
                resample_interval=strategy.config.ema_resample_interval,
            )
            strategy.ema_dynamic = ema_series
            if show.get("ema", False):
                strategy.I(lambda _: ema_series, strategy.data.Close, name="EMA")

        # ---- ATR% ----
        if calc_atr:
            atr_pct = strategy.indicators.compute_atr_percentage(
                window=strategy.config.atr_window,
                resample_interval=strategy.config.atr_resample_interval,
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

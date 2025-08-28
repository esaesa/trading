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

    def run(self, strategy) -> None:
        # attach manager (idempotent)
        if not hasattr(strategy, "indicators"):
            df = strategy.data.df
            strategy.indicators = IndicatorManager(df)

        # compute decisions - automatic detection based on rule dependencies + centralized display
        show_indicators = strategy.config.show_indicators or {}
        indicators_config = self.params.get("indicators", {})

        # Create flexible display logic - OR combination with other conditions
        def should_display(indicator_name, default=False):
            """Display if explicitly requested OR if indicator is being calculated for rules"""
            show_requested = show_indicators.get(indicator_name, False)
            is_calculated = indicator_name in required_indicators
            return show_requested or is_calculated or default

        # Collect required indicators from all active rules (SOLID: rules declare dependencies)
        required_indicators = set()

        # Check entry rules
        if hasattr(strategy, 'entry_decider') and strategy.entry_decider:
            required_indicators.update(strategy.entry_decider.get_required_indicators())

        # Check safety rules
        if hasattr(strategy, 'safety_decider') and strategy.safety_decider:
            required_indicators.update(strategy.safety_decider.get_required_indicators())

        # Check exit rules
        if hasattr(strategy, 'exit_decider') and strategy.exit_decider:
            required_indicators.update(strategy.exit_decider.get_required_indicators())

        # Check pricing engine
        if hasattr(strategy, 'price_engine') and strategy.price_engine:
            required_indicators.update(strategy.price_engine.get_required_indicators())

        # Check entry sizing policy
        if hasattr(strategy, 'entry_sizer') and strategy.entry_sizer:
            required_indicators.update(strategy.entry_sizer.get_required_indicators())

        # Set calculation flags based on rule dependencies OR explicit show request
        calc_rsi = ("rsi" in required_indicators) or show_indicators.get("rsi", False)
        calc_atr = ("atr_pct" in required_indicators) or show_indicators.get("atr", False)
        calc_ema = ("ema" in required_indicators) or show_indicators.get("ema", False)
        calc_lag = ("laguerre_rsi" in required_indicators) or show_indicators.get("laguerre", False)
        calc_cv = ("cv" in required_indicators) or show_indicators.get("cv", False)
        calc_bbw = ("bbw" in required_indicators) or show_indicators.get("bbw", False)
        calc_dynamic_rsi = ("dynamic_rsi_threshold" in required_indicators) or show_indicators.get("dynamic_rsi", False)

        # ---- RSI ----
        if calc_rsi:
            rsi_config = indicators_config.get("rsi", {})
            rsi_window = rsi_config.get("window", strategy.config.rsi_window or 14)
            rsi_resample = rsi_config.get("resample_interval", "30min")

            rsi_series = strategy.indicators.compute_rsi(
                window=rsi_window,
                resample_interval=rsi_resample,
            )
            # Store in IndicatorService for rule access (single point of truth)
            strategy.indicator_service._store_series("rsi", rsi_series)
            # Store indicator object for framework plotting (centralized display control)
            if should_display("rsi", False):
                strategy.rsi_indicator = strategy.I(lambda _: rsi_series, strategy.data.Close, name="RSI")

        # ---- Dynamic RSI threshold ----
        if calc_dynamic_rsi and (rsi_series is not None):
            # Get both percentile and dynamic_window from indicators config
            dynamic_rsi_config = indicators_config.get("dynamic_rsi_threshold", {})
            percentile = dynamic_rsi_config.get("percentile", 0.05)
            dynamic_window = dynamic_rsi_config.get("dynamic_window", 8500)

            dynamic_rsi_series = (
                rsi_series.rolling(window=dynamic_window)
                          .quantile(percentile)
            )
            strategy.indicator_service._store_series("dynamic_rsi_threshold", dynamic_rsi_series)
            # Store indicator object for framework plotting
            if should_display("dynamic_rsi", True):
                strategy.dynamic_rsi_indicator = strategy.I(lambda _: dynamic_rsi_series,
                           strategy.data.Close,
                           name="Dynamic RSI Threshold")

        # ---- EMA ----
        if calc_ema:
            ema_config = indicators_config.get("ema", {})
            ema_window = ema_config.get("window", strategy.config.ema_window or 200)
            ema_resample = ema_config.get("resample_interval", strategy.config.ema_resample_interval or "1h")

            ema_series = strategy.indicators.compute_ema(
                window=ema_window,
                resample_interval=ema_resample,
            )
            strategy.indicator_service._store_series("ema", ema_series)
            # Store indicator object for framework plotting (centralized display control)
            if should_display("ema", False):
                strategy.ema_indicator = strategy.I(lambda _: ema_series, strategy.data.Close, name="EMA")

        # ---- ATR% ----
        if calc_atr:
            atr_config = indicators_config.get("atr", {})
            atr_window = atr_config.get("window", strategy.config.atr_window or 14)
            atr_resample = atr_config.get("resample_interval", strategy.config.atr_resample_interval or "1h")

            atr_pct = strategy.indicators.compute_atr_percentage(
                window=atr_window,
                resample_interval=atr_resample,
            )
            strategy.indicator_service._store_series("atr_pct", atr_pct)
            # Store indicator object for framework plotting (centralized display control)
            if should_display("atr", False):
                strategy.atr_indicator = strategy.I(lambda _: atr_pct, strategy.data.Close, name="ATR%")

        # ---- CV ----
        if calc_cv:
            cv_config = indicators_config.get("cv", {})
            cv_window = cv_config.get("window", 40)

            cv_series = strategy.indicators.compute_coefficient_of_variation(window=cv_window)
            strategy.indicator_service._store_series("cv", cv_series)
            # Store indicator object for framework plotting (centralized display control)
            if should_display("cv", False):
                strategy.cv_indicator = strategy.I(lambda _: cv_series, strategy.data.Close, name="CV")

        # ---- Laguerre RSI ----
        if calc_lag:
            lag_config = indicators_config.get("laguerre_rsi", {})
            lag_gamma = lag_config.get("gamma", 0.1)
            lag_resample = lag_config.get("resample_interval", "4h")

            lag = strategy.indicators.compute_laguerre_rsi(gamma=lag_gamma, resample_interval=lag_resample)
            strategy.indicator_service._store_series("laguerre_rsi", lag)
            # Store indicator object for framework plotting (centralized display control)
            if should_display("laguerre", False):
                strategy.laguerre_indicator = strategy.I(lambda _: lag, strategy.data.Close, name="Laguerre RSI")

        # ---- BBW ----
        if calc_bbw:
            bbw_config = indicators_config.get("bbw", {})
            bbw_window = bbw_config.get("window", 20)
            bbw_num_std = bbw_config.get("num_std", 2)

            bbw = strategy.indicators.compute_bbw(window=bbw_window, num_std=bbw_num_std)
            strategy.indicator_service._store_series("bbw", bbw)
            # Store indicator object for framework plotting (centralized display control)
            if should_display("bbw", False):
                strategy.bbw_indicator = strategy.I(lambda _: bbw, strategy.data.Close, name="BBW")

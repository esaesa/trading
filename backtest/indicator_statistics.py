# indicator_statistics.py
"""
Generic indicator statistics analyzer for any technical indicator.

This module provides comprehensive statistical analysis capabilities for any
indicator series, including RSI, ATR, EMA, BBW, CV, and custom indicators.

Usage:
    from backtest.indicator_statistics import IndicatorStatistics

    # Analyze any indicator series
    rsi_stats = IndicatorStatistics(strategy.indicator_service._get_series("rsi"))
    stats_summary = rsi_stats.get_comprehensive_stats()

    # Cross-indicator analysis
    analyzer = IndicatorAnalyzer(strategy.indicator_service)
    correlation_matrix = analyzer.get_indicator_correlations()
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
from scipy.stats import gmean, hmean

@dataclass
class IndicatorStatsSummary:
    """Container for comprehensive indicator statistics"""
    name: str
    descriptive_stats: Dict[str, float]
    distribution_stats: Dict[str, Any]
    temporal_stats: Dict[str, Any]
    volatility_stats: Dict[str, float]
    range_analysis: Dict[str, Any]

class IndicatorStatistics:
    """
    Comprehensive statistical analyzer for technical indicator series.

    Works with any indicator that returns a pandas Series with timestamps.
    Handles NaN values gracefully and provides robust statistical analysis.
    """

    def __init__(self, series: pd.Series, name: str = "Indicator"):
        """
        Initialize with any indicator series.

        Args:
            series: pandas Series with datetime index
            name: Name/label for the indicator
        """
        self.series = series.copy()
        self.name = name

        # Clean data - remove NaNs for statistical calculations
        self.series_clean = self.series.dropna()

        if len(self.series_clean) == 0:
            warnings.warn(f"No valid data in {name} series", UserWarning)

    
    def get_future_returns_analysis(self, price_series: pd.Series, periods=(5, 10, 20, 50)):
        """
        Calculates the average future returns for different periods, binned by indicator quintiles.
        This version uses a robust, vectorized approach to avoid "jagged" data issues.
        """
        if self.series_clean.empty or len(self.series_clean) < max(periods):
            # Not enough data to calculate future returns, return empty dict
            return {}

        # 1. Create a single DataFrame aligning the indicator and price series.
        #    This is crucial to ensure data points match up correctly.
        df = pd.DataFrame({'indicator': self.series_clean, 'price': price_series}).dropna()

        if df.empty:
            return {}

        # 2. Bin the indicator values into quintiles.
        try:
            # 'duplicates=drop' handles cases where data is not distributed enough for 5 clean bins.
            df['quintile'] = pd.qcut(df['indicator'], 5, labels=[f'Q{i}' for i in range(1, 6)], duplicates='drop')
        except ValueError:
            # If it still fails, we can't perform the analysis.
            return {}

        # 3. Calculate all future return columns in a vectorized way.
        for n in periods:
            # Use the aligned 'price' column for the calculation.
            df[f'{n}p_ret'] = (df['price'].shift(-n) / df['price'] - 1) * 100

        # 4. Perform a single groupby().mean() operation.
        # This is the core of the fix. It produces a perfectly rectangular DataFrame,
        # filling any missing quintile/period combinations with NaN automatically.
        return_cols = [f'{n}p_ret' for n in periods]
        analysis_df = df.groupby('quintile')[return_cols].mean()

        # 5. Convert the clean DataFrame to a dictionary. This is now a safe operation.
        return analysis_df.to_dict(orient='index')
   
   
   
    
    def get_drawdown_behavior(self, equity_series: pd.Series) -> Dict[str, Any]:
        """
        Analyzes indicator behavior during the strategy's equity drawdowns.
        Args:
            equity_series: The portfolio equity curve from the backtest.
        Returns:
            A dictionary containing key stats about the indicator during drawdowns.
        """
        if self.series_clean.empty:
            return {}

        # 1. Calculate the drawdown series from the equity curve
        peak = equity_series.cummax()
        drawdown = (equity_series / peak - 1) * 100  # In percentage

        # 2. Align the indicator with the equity curve and drawdown series
        df = pd.DataFrame({
            'indicator': self.series_clean,
            'drawdown': drawdown
        }).dropna()

        # 3. Separate indicator values into two groups: during a drawdown and not.
        in_drawdown_indicator = df[df['drawdown'] < 0]['indicator']
        not_in_drawdown_indicator = df[df['drawdown'] == 0]['indicator']

        if in_drawdown_indicator.empty:
            return {
                'avg_indicator_in_dd': None,
                'avg_indicator_not_in_dd': not_in_drawdown_indicator.mean(),
                'in_drawdown_data': pd.DataFrame(), # Return empty DataFrame
            }

        return {
            'avg_indicator_in_dd': in_drawdown_indicator.mean(),
            'avg_indicator_not_in_dd': not_in_drawdown_indicator.mean(),
            # Return the raw data for plotting
            'in_drawdown_data': df[df['drawdown'] < -0.1], # Only where DD is meaningful
        }

    def get_volatility_regime_analysis(self, price_series: pd.Series, vol_window: int = 20):
        """
        Analyzes the relationship between indicator value and future price volatility.
        Args:
            price_series: The close price series of the asset.
            vol_window: The window for calculating future realized volatility.
        Returns:
            A DataFrame with indicator quintiles and their corresponding future volatility.
        """
        if self.series_clean.empty:
            return pd.DataFrame()

        # 1. Calculate future realized volatility of log returns (standard method)
        log_returns = np.log(price_series / price_series.shift(1))
        future_vol = log_returns.rolling(window=vol_window).std().shift(-vol_window) * np.sqrt(365*24*60) # Annualized for 1m data

        # 2. Align data
        df = pd.DataFrame({
            'indicator': self.series_clean,
            'future_vol': future_vol
        }).dropna()

        if df.empty:
            return pd.DataFrame()

        # 3. Bin indicator into quintiles
        try:
            df['quintile'] = pd.qcut(df['indicator'], 5, labels=[f'Q{i}' for i in range(1, 6)], duplicates='drop')
        except ValueError:
            return pd.DataFrame()
            
        # 4. Group by quintile and calculate the average future volatility
        vol_by_quintile = df.groupby('quintile')['future_vol'].mean()
        
        return vol_by_quintile.reset_index()

    def get_volatility_analysis(self, price_series: pd.Series, vol_window: int = 20) -> pd.DataFrame:
        """
        Analyzes the relationship between indicator value and future price volatility.

        Args:
            price_series: The close price series of the asset.
            vol_window: The window for calculating future realized volatility.

        Returns:
            A DataFrame with indicator values and corresponding future volatility.
        """
        if len(self.series_clean) == 0:
            return pd.DataFrame()

        log_returns = np.log(price_series / price_series.shift(1))
        
        # Calculate future realized volatility
        future_vol = log_returns.shift(-vol_window).rolling(window=vol_window).std() * np.sqrt(252) # Annualized
        
        combined = pd.DataFrame({
            'indicator': self.series_clean,
            'future_volatility': future_vol
        }).dropna()
        
        return combined

    def get_descriptive_stats(self) -> Dict[str, float]:
        """Calculate basic descriptive statistics"""
        if len(self.series_clean) == 0:
            return {k: np.nan for k in ['count', 'mean', 'std', 'min', 'max', 'median', 'q25', 'q75', 'iqr', 'range', 'skewness', 'kurtosis', 'p05', 'p95', 'gmean', 'hmean', 'mad']}

        desc = self.series_clean.describe()

        return {
            'count': len(self.series_clean),
            'mean': self.series_clean.mean(),
            'std': self.series_clean.std(),
            'min': self.series_clean.min(),
            'max': self.series_clean.max(),
            'median': self.series_clean.median(),
            'q25': self.series_clean.quantile(0.25),
            'q75': self.series_clean.quantile(0.75),
            'iqr': self.series_clean.quantile(0.75) - self.series_clean.quantile(0.25),
            'range': self.series_clean.max() - self.series_clean.min(),
            'skewness': self.series_clean.skew(),
            'kurtosis': self.series_clean.kurtosis(),
            'p05': self.series_clean.quantile(0.05), # 5th percentile
            'p95': self.series_clean.quantile(0.95), # 95th percentile
            'gmean': gmean(self.series_clean[self.series_clean > 0]), # Geometric Mean (must be positive values)
            'hmean': hmean(self.series_clean[self.series_clean > 0]), # Harmonic Mean (must be positive values)
            'mad': (self.series_clean - self.series_clean.mean()).abs().mean(), # Mean Absolute Deviation
        }

    def get_distribution_stats(self, bins: Optional[List[float]] = None) -> Dict[str, Any]:
        """Analyze distribution characteristics"""
        if len(self.series_clean) == 0:
            return {'value_counts': {}, 'histogram_bins': [], 'histogram_counts': [], 'distribution_type': 'empty'}

        # Default bins based on indicator type (assuming RSI-like ranges)
        if bins is None:
            min_val = self.series_clean.min()
            max_val = self.series_clean.max()
            bins = np.linspace(min_val, max_val, 21)  # 20 bins

        hist, bin_edges = np.histogram(self.series_clean, bins=bins)

        # Categorize distribution
        skew_coeff = self.series_clean.skew()
        if skew_coeff > 0.5:
            dist_type = "right-skewed"
        elif skew_coeff < -0.5:
            dist_type = "left-skewed"
        else:
            dist_type = "symmetric"

        return {
            'value_counts': self.series_clean.value_counts(bins=bins).to_dict(),
            'histogram_bins': bin_edges.tolist(),
            'histogram_counts': hist.tolist(),
            'distribution_type': dist_type,
            'mode': self.series_clean.mode().iloc[0] if len(self.series_clean.mode()) > 0 else np.nan,
            'bimodal': len(self.series_clean.mode()) > 1
        }

    def get_volatility_stats(self) -> Dict[str, float]:
        """Calculate volatility, variability, and signal-to-noise measures."""
        if len(self.series_clean) < 2: # Need at least 2 points for most calcs
            keys = ['rolling_std_mean', 'coefficient_variation', 'noise_ratio', 'signal_to_noise']
            return {k: np.nan for k in keys}
            
        s = self.series_clean
        mean_val = s.mean()
        std_val = s.std()

        # Rolling standard deviation (20-period default)
        rolling_std_mean = s.rolling(window=20, min_periods=5).std().mean()
        
        # Coefficient of variation
        cv = (std_val / abs(mean_val)) * 100 if mean_val != 0 else np.inf

        # Noise ratio (high frequency changes)
        changes = s.pct_change().abs()
        noise_ratio = changes.mean() * 100 if len(changes.dropna()) > 0 else np.nan

        # --- NEW STATISTIC ---
        # Signal-to-Noise Ratio
        signal_to_noise = abs(mean_val / std_val) if std_val != 0 else np.inf

        return {
            'rolling_std_mean': rolling_std_mean,
            'coefficient_variation': cv,
            'noise_ratio': noise_ratio,
            'signal_to_noise': signal_to_noise,
        }

    
    def get_temporal_stats(self) -> Dict[str, Any]:
        """Analyze temporal patterns and trends"""
        if len(self.series_clean) == 0:
            return {'hourly_patterns': {}, 'weekly_patterns': {}, 'trend_stats': {}, 'stability_periods': 0}

        # Hourly patterns
        hourly_avg = self.series_clean.groupby(self.series_clean.index.hour).mean()
        hourly_vol = self.series_clean.groupby(self.series_clean.index.hour).std()

        # Weekly patterns
        weekly_avg = self.series_clean.groupby(self.series_clean.index.dayofweek).mean()
        weekly_vol = self.series_clean.groupby(self.series_clean.index.dayofweek).std()

        # Trend analysis
        if len(self.series_clean) >= 30:
            short_trend = self.series_clean.tail(10).mean() - self.series_clean.tail(30).mean()
            long_trend = self.series_clean.tail(30).mean() - self.series_clean.head(len(self.series_clean.tail(30))).mean()
        else:
            short_trend = long_trend = np.nan

        # Periods of relative stability (consecutive values within 5% range)
        changes = self.series_clean.pct_change().abs()
        stable_periods = (changes < 0.05).sum() if len(changes.dropna()) > 0 else 0

        return {
            'hourly_patterns': {
                'means': hourly_avg.to_dict(),
                'volatility': hourly_vol.to_dict()
            },
            'weekly_patterns': {
                'means': weekly_avg.to_dict(),
                'volatility': weekly_vol.to_dict()
            },
            'trend_stats': {
                'short_term_trend': short_trend,
                'long_term_trend': long_trend,
                'autocorrelation_1': self.series_clean.autocorr(lag=1) if len(self.series_clean) > 1 else np.nan
            },
            'stability_periods': int(stable_periods)
        }

    def get_range_analysis(self, custom_ranges: Optional[Dict[str, Tuple[float, float]]] = None) -> Dict[str, Any]:
        """Analyze time spent in different ranges"""
        if len(self.series_clean) == 0:
            return {'default_ranges': {}, 'custom_ranges': {}}

        # Default ranges for common indicators
        default_ranges = {
            'extremes_low': (self.series_clean.quantile(0.05), self.series_clean.quantile(0.15)),
            'lows': (self.series_clean.quantile(0.15), self.series_clean.quantile(0.25)),
            'neutral': (self.series_clean.quantile(0.25), self.series_clean.quantile(0.75)),
            'highs': (self.series_clean.quantile(0.75), self.series_clean.quantile(0.85)),
            'extremes_high': (self.series_clean.quantile(0.85), self.series_clean.quantile(0.95))
        }

        # Calculate time spent in each range
        range_analysis = {}
        total_periods = len(self.series_clean)

        for range_name, (lower, upper) in default_ranges.items():
            count = ((self.series_clean >= lower) & (self.series_clean <= upper)).sum()
            percentage = (count / total_periods * 100) if total_periods > 0 else 0
            avg_value = self.series_clean[(self.series_clean >= lower) & (self.series_clean <= upper)].mean()
            range_analysis[range_name] = {
                'count': int(count),
                'percentage': round(percentage, 2),
                'average': avg_value if not np.isnan(avg_value) else np.nan
            }

        custom_range_analysis = {}
        if custom_ranges:
            for range_name, (lower, upper) in custom_ranges.items():
                count = ((self.series_clean >= lower) & (self.series_clean <= upper)).sum()
                percentage = (count / total_periods * 100) if total_periods > 0 else 0
                avg_value = self.series_clean[(self.series_clean >= lower) & (self.series_clean <= upper)].mean()
                custom_range_analysis[range_name] = {
                    'count': int(count),
                    'percentage': round(percentage, 2),
                    'average': avg_value if not np.isnan(avg_value) else np.nan
                }

        return {
            'default_ranges': range_analysis,
            'custom_ranges': custom_range_analysis
        }

    def get_comprehensive_stats(self, custom_ranges: Optional[Dict[str, Tuple[float, float]]] = None) -> IndicatorStatsSummary:
        """Get complete statistical analysis"""
        return IndicatorStatsSummary(
            name=self.name,
            descriptive_stats=self.get_descriptive_stats(),
            distribution_stats=self.get_distribution_stats(),
            temporal_stats=self.get_temporal_stats(),
            volatility_stats=self.get_volatility_stats(),
            range_analysis=self.get_range_analysis(custom_ranges)
        )

    def compare_series(self, other_series: pd.Series, other_name: str = "Comparison Series") -> Dict[str, Any]:
        """Compare this indicator with another series"""
        other_clean = other_series.dropna()

        if len(self.series_clean) == 0 or len(other_clean) == 0:
            return {'correlation': np.nan, 'comparison_msg': 'Insufficient data for comparison'}

        # Correlation analysis
        correlation = self.series_clean.corr(other_clean)

        # Difference statistics
        min_len = min(len(self.series_clean), len(other_clean))
        diff = self.series_clean.tail(min_len).values - other_clean.tail(min_len).values
        mean_diff = np.mean(diff)
        std_diff = np.std(diff)

        return {
            'correlation': correlation,
            'mean_difference': mean_diff,
            'std_difference': std_diff,
            'max_difference': np.max(diff),
            'min_difference': np.min(diff),
            'comparison_series_name': other_name
        }


class IndicatorAnalyzer:
    """
    Multi-indicator analyzer for correlation and comparative analysis.

    Analyzes relationships between multiple indicators and provides
    comprehensive cross-indicator statistics.
    """

    def __init__(self, indicator_service):
        """
        Initialize with an IndicatorService instance.

        Args:
            indicator_service: Strategy's indicator service for accessing indicator series
        """
        self.indicator_service = indicator_service
        self.available_indicators = [
            "rsi", "dynamic_rsi_threshold", "atr_pct", "ema",
            "laguerre_rsi", "cv", "bbw"
        ]

    def get_indicator_correlations(self) -> pd.DataFrame:
        """Calculate correlation matrix for all available indicators"""
        indicator_data = {}

        for indicator_name in self.available_indicators:
            series = self.indicator_service._get_series(indicator_name)
            if series is not None and not series.empty:
                # Ensure series are aligned by timestamp
                series_clean = series.dropna()
                if len(series_clean) > 10:  # Minimum data requirement
                    indicator_data[indicator_name] = series_clean

        if len(indicator_data) <= 1:
            return pd.DataFrame()  # Not enough indicators for correlation

        # Create DataFrame with aligned data
        combined_df = pd.DataFrame(indicator_data)

        # Calculate correlation matrix
        correlation_matrix = combined_df.corr()

        return correlation_matrix

    def analyze_indicator_relationships(self, reference_indicator: str = "rsi") -> Dict[str, Any]:
        """Analyze relationships of other indicators with a reference indicator"""
        reference_series = self.indicator_service._get_series(reference_indicator)
        if reference_series is None or reference_series.empty:
            return {'error': f'Reference indicator {reference_indicator} not available'}

        ref_stats = IndicatorStatistics(reference_series, reference_indicator)
        relationships = {}

        for indicator_name in self.available_indicators:
            if indicator_name == reference_indicator:
                continue

            other_series = self.indicator_service._get_series(indicator_name)
            if other_series is not None and not other_series.empty:
                comparison = ref_stats.compare_series(other_series, indicator_name)

                # Calculate lead/lag relationship
                lead_lag_corr = self._calculate_lead_lag_correlation(reference_series, other_series)

                relationships[indicator_name] = {
                    **comparison,
                    'lead_lag_correlations': lead_lag_corr
                }

        return {
            'reference_indicator': reference_indicator,
            'relationships': relationships
        }

    def _calculate_lead_lag_correlation(self, series1: pd.Series, series2: pd.Series,
                                      max_lag: int = 5) -> Dict[int, float]:
        """Calculate correlations at different time lags"""
        series1_clean = series1.dropna()
        series2_clean = series2.dropna()

        # Align series
        min_len = min(len(series1_clean), len(series2_clean))
        s1 = series1_clean.tail(min_len)
        s2 = series2_clean.tail(min_len)

        correlations = {}
        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                # series1 lags behind series2
                corr_series = s1.shift(-lag)
            else:
                # series2 lags behind series1
                corr_series = s2.shift(lag)

            corr = s1.corr(corr_series)
            correlations[lag] = corr if not np.isnan(corr) else 0.0

        return correlations


# Convenience functions
def analyze_indicator(strategy, indicator_name: str) -> IndicatorStatsSummary:
    """
    Quick helper function to analyze any indicator.

    Args:
        strategy: Trading strategy instance
        indicator_name: Name of indicator to analyze

    Returns:
        Comprehensive statistics summary
    """
    series = strategy.indicator_service._get_series(indicator_name)
    if series is None or series.empty:
        raise ValueError(f"Indicator '{indicator_name}' not available or empty")

    analyzer = IndicatorStatistics(series, indicator_name)
    return analyzer.get_comprehensive_stats()


def get_all_indicator_stats(strategy) -> Dict[str, IndicatorStatsSummary]:
    """
    Get statistics for all available indicators.

    Args:
        strategy: Trading strategy instance

    Returns:
        Dictionary mapping indicator names to their statistics
    """
    analyzer = IndicatorAnalyzer(strategy.indicator_service)
    stats = {}

    for indicator_name in analyzer.available_indicators:
        try:
            stats[indicator_name] = analyze_indicator(strategy, indicator_name)
        except (ValueError, AttributeError):
            continue  # Skip unavailable indicators

    return stats


# Export functions
__all__ = [
    'IndicatorStatistics',
    'IndicatorAnalyzer',
    'IndicatorStatsSummary',
    'analyze_indicator',
    'get_all_indicator_stats'
]
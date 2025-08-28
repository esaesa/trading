# indicator_analysis_example.py
"""
Examples and usage patterns for the IndicatorStatistics module.

This file demonstrates how to:
1. Analyze individual indicators (RSI, ATR, etc.)
2. Compare multiple indicators
3. Generate comprehensive reports
4. Visualize statistical results
5. Create custom analysis patterns

Run this file to see comprehensive indicator analysis in action.
"""

from backtest.indicator_statistics import IndicatorStatistics, IndicatorAnalyzer, analyze_indicator, get_all_indicator_stats
import json
from typing import Dict, Any


def analyze_rsi_detailed(strategy):
    """Comprehensive RSI analysis example"""
    print("🔍 RSI DETAILED ANALYSIS")
    print("=" * 60)

    # Get RSI series and create analyzer
    rsi_series = strategy.indicator_service._get_series("rsi")
    rsi_stats = IndicatorStatistics(rsi_series, "RSI")

    print(f"RSI Data Points Available: {len(rsi_series.dropna()):,}")
    print(f"Time Range: {rsi_series.index.min()} to {rsi_series.index.max()}")
    print()

    # Basic descriptive statistics
    desc_stats = rsi_stats.get_descriptive_stats()
    print("📊 DESCRIPTIVE STATISTICS:")
    print(f"  Mean RSI: {desc_stats['mean']:.2f}")
    print(f"  Median RSI: {desc_stats['median']:.2f}")
    print(f"  Standard Deviation: {desc_stats['std']:.2f}")
    print(f"  Range: {desc_stats['min']:.2f} - {desc_stats['max']:.2f}")
    print(f"  Interquartile Range: {desc_stats['iqr']:.2f}")
    print(f"  Coefficient of Variation: {desc_stats['mean']/desc_stats['std']:.3f}" if desc_stats['mean'] != 0 else "  Coefficient of Variation: N/A")
    print()

    # Distribution analysis
    dist_stats = rsi_stats.get_distribution_stats()
    print("📈 DISTRIBUTION ANALYSIS:")
    print(f"  Distribution Type: {dist_stats['distribution_type']}")
    print(f"  Most Common Value (Mode): {dist_stats['mode']:.2f}")
    print(f"  Is Bimodal: {dist_stats['bimodal']}")
    print()

    # Range analysis
    range_analysis = rsi_stats.get_range_analysis()
    print("🎯 TIME SPENT IN RANGES:")
    for range_name, data in range_analysis['default_ranges'].items():
        print(f"  {range_name.replace('_', ' ').title()}: {data['percentage']:.1f}% of time")
        print(f"  (Average value: {data['average']:.2f}, Count: {data['count']:,})")
    print()

    # Temporal patterns
    temp_stats = rsi_stats.get_temporal_stats()
    print("⏰ TEMPORAL PATTERNS (HOURLY AVERAGES):")
    print("  Most trending hour:", end=" ")
    hourly_means = temp_stats['hourly_patterns']['means']
    trending_hour = max(hourly_means.items(), key=lambda x: x[1])
    print(f"Hour {trending_hour[0]:02d} (RSI: {trending_hour[1]:.2f})")
    print("  Most volatile hour:", end=" ")
    hourly_vol = temp_stats['hourly_patterns']['volatility']
    volatile_hour = max(hourly_vol.items(), key=lambda x: x[1])
    print(f"Hour {volatile_hour[0]:02d} (Volatility: {volatile_hour[1]:.2f})")
    print()

    # Volatility analysis
    vol_stats = rsi_stats.get_volatility_stats()
    print("💹 VOLATILITY ANALYSIS:")
    print(f"  Average Rolling Volatility: {vol_stats['rolling_std_mean']:.2f}")
    print(f"  Coefficient of Variation: {vol_stats['coefficient_variation']:.2f}%")
    print(f"  Noise Ratio: {vol_stats['noise_ratio']:.4f}%")
    print(f"  Stability Periods (low change): {temp_stats['stability_periods']:,}")
    print()


def analyze_indicator_correlations(strategy):
    """Analyze relationships between all indicators"""
    print("🔗 CROSS-INDICATOR CORRELATIONS")
    print("=" * 60)

    analyzer = IndicatorAnalyzer(strategy.indicator_service)
    correlations = analyzer.get_indicator_correlations()

    if correlations.empty:
        print("❌ Not enough indicator data for correlation analysis")
        return

    print("Correlation Matrix:")
    print(correlations.round(4))
    print()

    # Find strongest relationships
    print("Strongest Indicator Relationships:")
    corr_unstack = correlations.unstack()
    strong_correlations = corr_unstack.abs().sort_values(ascending=False)

    shown = set()
    for idx_pair, corr_val in strong_correlations.items():
        if idx_pair[0] != idx_pair[1] and (idx_pair[1], idx_pair[0]) not in shown:
            shown.add(idx_pair)
            if len(shown) > 10:  # Show top 10 correlations
                break
            print(f"  {idx_pair[0]} ↔ {idx_pair[1]}: {corr_val:.3f}")
    print()


def custom_indicator_analysis(strategy):
    """Example of custom range analysis for indicators"""
    print("🎛️ CUSTOM RANGE ANALYSIS")
    print("=" * 60)

    # RSI custom ranges (common trading levels)
    rsi_series = strategy.indicator_service._get_series("rsi")
    rsi_stats = IndicatorStatistics(rsi_series, "RSI")

    rsi_ranges = {
        'Oversold': (0, 30),
        'Neutral': (30, 70),
        'Overbought': (70, 100)
    }

    range_results = rsi_stats.get_range_analysis(rsi_ranges)
    print("RSI Time Distribution (Custom Trading Ranges):")
    for range_name, data in range_results['custom_ranges'].items():
        print(f"  {range_name}: {data['percentage']:.1f}% of time")
    print()

    # ATR percentage analysis
    atr_series = strategy.indicator_service._get_series("atr_pct")
    if atr_series is not None and not atr_series.empty:
        atr_stats = IndicatorStatistics(atr_series, "ATR_%")

        atr_ranges = {
            'Low Volatility': (0, 1.0),
            'Medium Volatility': (1.0, 2.0),
            'High Volatility': (2.0, 10.0)
        }

        atr_range_results = atr_stats.get_range_analysis(atr_ranges)
        print("ATR% Time Distribution (Volatility Ranges):")
        for range_name, data in atr_range_results['custom_ranges'].items():
            print(f"  {range_name}: {data['percentage']:.1f}% of time")
    print()


def generate_comprehensive_report(strategy):
    """Generate a comprehensive analysis report"""
    print("📊 COMPREHENSIVE INDICATOR ANALYSIS REPORT")
    print("=" * 60)

    all_stats = get_all_indicator_stats(strategy)

    if not all_stats:
        print("❌ No indicator data available")
        return

    print(f"Analyzed {len(all_stats)} indicators: {', '.join(all_stats.keys())}")
    print()

    for indicator_name, stats_summary in all_stats.items():
        print(f"📈 {indicator_name.upper()} ANALYSIS:")
        desc = stats_summary.descriptive_stats
        range_data = stats_summary.range_analysis

        print(f"  Data Points: {int(desc['count']):,}")
        print(f"  Mean: {desc['mean']:.4f}")
        print(f"  Std Dev: {desc['std']:.4f}")
        print(f"  Range: {desc['min']:.4f} - {desc['max']:.4f}")

        # Show key range information
        default_ranges = range_data['default_ranges']
        print("  Key Ranges:")
        for range_name in ['extremes_low', 'neutral', 'extremes_high']:
            if range_name in default_ranges:
                pct = default_ranges[range_name]['percentage']
                print(f"    {range_name.replace('_', ' ')}: {pct:.1f}%")

        vol = stats_summary.volatility_stats
        print(".3f")
        print()

    # Cross-indicator correlations
    print("🔗 QUICK CORRELATION SUMMARY:")
    analyzer = IndicatorAnalyzer(strategy.indicator_service)
    correlations = analyzer.get_indicator_correlations()

    if not correlations.empty:
        # Highlight RSI correlations (most commonly used)
        if 'rsi' in correlations.index:
            print("  RSI Correlations:")
            rsi_cors = correlations['rsi'].drop('rsi')
            top_corr = rsi_cors.abs().sort_values(ascending=False)
            for indicator, corr_val in top_corr.head(3).items():
                print(f"    RSI ↔ {indicator}: {rsi_cors[indicator]:.3f}")
    print()


def create_indicator_dashboard(strategy):
    """Create a formatted dashboard view of key indicator metrics"""
    print("📱 INDICATOR DASHBOARD")
    print("=" * 60)

    try:
        # Quick stats for the most recent period
        current_time = strategy.data.index[-1] if hasattr(strategy, 'data') and not strategy.data.empty else None

        if current_time:
            dashboard_data = {}

            for indicator_name in ["rsi", "atr_pct", "bbw", "cv"]:
                value = strategy.indicator_service.get_indicator_value(indicator_name, current_time, np.nan)
                if not np.isnan(value):
                    dashboard_data[indicator_name] = value

            print(f"📊 CURRENT PERIOD ({current_time})")
            for indicator, value in dashboard_data.items():
                print(f"  {indicator.upper()}: {value:.4f}")

            # Recent trends (last 10 periods average vs previous)
            if len(strategy.data) > 20:
                for indicator_name in dashboard_data.keys():
                    series = strategy.indicator_service._get_series(indicator_name).dropna()
                    if len(series) > 20:
                        recent_avg = series.tail(10).mean()
                        previous_avg = series.tail(20).head(10).mean()
                        trend = "🔺" if recent_avg > previous_avg else "🔻"
                        pct_change = ((recent_avg - previous_avg) / abs(previous_avg) * 100) if previous_avg != 0 else 0
                        print(f"  {indicator_name.upper()} Trend: {trend} {pct_change:.1f}%")

    except Exception as e:
        print(f"❌ Dashboard creation error: {e}")

    print()


def main_demo():
    """Main demonstration function"""
    print("🚀 INDICATOR STATISTICS MODULE DEMO")
    print("=" * 80)
    print("This demo shows comprehensive indicator analysis capabilities.")
    print("In your actual code, replace 'strategy' with your trading strategy instance.\n")

    # Note: This is a demo - in real usage, you'd pass an actual strategy instance
    print("For actual usage in your trading system:")
    print("""
    from backtest.indicator_statistics import *
    from backtest.strategy import DCAStrategy

    # Initialize your strategy
    strategy = DCAStrategy(config, data)

    # Analyze RSI
    rsi_analysis = analyze_indicator(strategy, "rsi")
    print(f"RSI Mean: {rsi_analysis.descriptive_stats['mean']:.2f}")

    # Get all indicator statistics
    all_stats = get_all_indicator_stats(strategy)
    for name, stats in all_stats.items():
        print(f"{name}: {stats.descriptive_stats['mean']:.4f}")

    # Cross-indicator analysis
    analyzer = IndicatorAnalyzer(strategy.indicator_service)
    correlations = analyzer.get_indicator_correlations()
    print("Indicator Correlations:")
    print(correlations)
    """)

    print("\n⭐ KEY FEATURES:")
    print("✅ Works with ANY indicator (RSI, ATR, EMA, BBW, CV, Laguerre, custom)")
    print("✅ Comprehensive statistical analysis (mean, median, std dev, distributions)")
    print("✅ Temporal pattern analysis (hourly, weekly, trends)")
    print("✅ Cross-indicator correlation analysis")
    print("✅ Volatility and stability measurement")
    print("✅ Custom range analysis for trading strategies")
    print("✅ Performance benchmarking capabilities")
    print("=" * 80)


if __name__ == "__main__":
    main_demo()
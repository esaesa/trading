# Indicator Statistics Module

A comprehensive statistical analysis system for technical indicators in trading backtests.

## Overview

This module provides advanced statistical capabilities for analyzing any technical indicator in your trading system. It works with RSI, ATR, EMA, Bollinger Bands, Coefficient of Variation, Laguerre RSI, and any custom indicators you add.

## Features

✅ **Universal Compatibility** - Works with ANY indicator series (RSI, ATR, EMA, BBW, CV, Laguerre, custom)
✅ **Comprehensive Statistics** - Mean, median, std dev, percentiles, distributions, skewness, kurtosis
✅ **Temporal Analysis** - Daily/hourly patterns, trends, seasonal behavior
✅ **Cross-Indicator Analysis** - Correlation matrices, lead/lag relationships
✅ **Volatility Metrics** - Rolling volatility, stability periods, noise analysis
✅ **Custom Range Analysis** - Define trading ranges and see time distribution
✅ **Visualization Integration** - Charts, heatmaps, HTML reports
✅ **QuantStats Integration** - Enhanced performance analysis

## Quick Start

```python
from backtest.indicator_statistics import analyze_indicator, get_all_indicator_stats
from backtest.strategy import DCAStrategy

# Initialize your strategy
strategy = DCAStrategy(config, data)

# Analyze any indicator
rsi_stats = analyze_indicator(strategy, "rsi")
print(f"RSI Mean: {rsi_stats.descriptive_stats['mean']:.2f}")
print(f"Time in neutral (30-70): {rsi_stats.range_analysis['default_ranges']['neutral']['percentage']:.1f}%")

# Get comprehensive statistics for all indicators
all_stats = get_all_indicator_stats(strategy)
for name, stats in all_stats.items():
    print(f"{name}: {stats.descriptive_stats['mean']:.4f} ± {stats.descriptive_stats['std']:.4f}")

# Cross-indicator correlations
from backtest.indicator_statistics import IndicatorAnalyzer
analyzer = IndicatorAnalyzer(strategy.indicator_service)
correlations = analyzer.get_indicator_correlations()
print("Indicator Correlations:")
print(correlations)
```

## Available Statistics

### Descriptive Statistics
- Mean, median, mode
- Standard deviation, variance
- Min, max, range
- Quartiles (25%, 75%, IQR)
- Skewness, kurtosis
- Count of data points

### Distribution Analysis
- Distribution type (symmetric/left-skewed/right-skewed)
- Bimodal detection
- Value frequency distributions
- Histogram data

### Volatility Analysis
- Coefficient of variation
- Rolling standard deviation (20-period mean)
- Noise ratio (high-frequency changes)
- Trend stability metrics

### Temporal Patterns
- Hourly patterns (means and volatility by hour)
- Weekly patterns (by day of week)
- Short-term vs long-term trends
- Autocorrelation coefficients
- Stability period counts

### Range Analysis
- Default quantiles: extremes_low, low, neutral, high, extremes_high
- Custom trading ranges (e.g., oversold: 0-30, overbought: 70-100)
- Time spent and average values in each range

### Cross-Indicator Analysis
- Correlation matrix for all available indicators
- Lead/lag relationship analysis (correlation at different time lags)
- Comparative statistics between any two indicators

## Detailed Usage Examples

### RSI Analysis
```python
from backtest.indicator_statistics import IndicatorStatistics

rsi_series = strategy.indicator_service._get_series("rsi")
rsi_stats = IndicatorStatistics(rsi_series, "RSI")

# Comprehensive analysis
comprehensive = rsi_stats.get_comprehensive_stats()

print("RSI Summary:")
print(f"  Data Range: {comprehensive.descriptive_stats['min']:.1f} - {comprehensive.descriptive_stats['max']:.1f}")
print(f"  Distribution: {comprehensive.distribution_stats['distribution_type']}")
print(f"  Time spent above 70: {rsi_stats.get_range_analysis({'overbought': (70, 100)})['custom_ranges']['overbought']['percentage']:.1f}%")
```

### Multi-Indicator Comparison
```python
analyzer = IndicatorAnalyzer(strategy.indicator_service)

# See how all indicators correlate with each other
correlations = analyzer.get_indicator_correlations()

# Analyze RSI relationships with other indicators
rsi_relationships = analyzer.analyze_indicator_relationships("rsi")
for indicator, data in rsi_relationships['relationships'].items():
    print(f"RSI ↔ {indicator}: correlation = {data['correlation']:.3f}")
```

### Custom Trading Ranges
```python
# Define custom ranges based on your strategy
rsi_trading_ranges = {
    'Oversold': (0, 30),
    'Recovery': (30, 50),
    'Neutral': (50, 70),
    'Momentum': (70, 80),
    'Overbought': (80, 100)
}

rsi_analysis = analyze_indicator(strategy, "rsi")
with_custom_ranges = IndicatorStatistics(
    strategy.indicator_service._get_series("rsi"),
    "RSI"
).get_range_analysis(rsi_trading_ranges)

for range_name, data in with_custom_ranges['custom_ranges'].items():
    print(f"Time in {range_name}: {data['percentage']:.1f}% (avg: {data['average']:.2f})")
```

## Visualization & Reporting

### HTML Report Generation
```python
from backtest.indicator_statistics_visualization import create_comprehensive_indicator_report

# Generate complete HTML report
create_comprehensive_indicator_report(
    strategy,
    output_file="my_indicator_report.html",
    title="My Trading Strategy - Indicator Analysis"
)
```

### Chart Generation (requires matplotlib/seaborn)
```python
import matplotlib.pyplot as plt
from backtest.indicator_statistics_visualization import (
    create_indicator_correlation_heatmap,
    create_indicator_distribution_chart,
    create_temporal_patterns_chart
)

analyzer = IndicatorAnalyzer(strategy.indicator_service)
rsi_stats = analyze_indicator(strategy, "rsi")

# Create correlation heatmap
fig1 = create_indicator_correlation_heatmap(analyzer)
plt.show()

# Create RSI distribution and range analysis
fig2 = create_indicator_distribution_chart(rsi_stats)
plt.show()

# Show temporal patterns
fig3 = create_temporal_patterns_chart(rsi_stats)
plt.show()
```

## Integration with Existing System

### Automatic Integration
The module integrates automatically with your visualization system:
```python
# In your backtest runner
from backtest.runner import run_backtest

stats, bt = run_backtest(strategy_params, config)
# Indicator reports are now automatically generated alongside your regular backtest results
```

### Manual Analysis
```python
# During backtest execution, add custom analysis
from backtest.indicator_statistics import get_all_indicator_stats

class EnhancedDCAStrategy(DCAStrategy):
    def init(self):
        super().init()
        # Run comprehensive indicator analysis
        self.indicator_summary = get_all_indicator_stats(self)

    def next(self):
        super().next()

        # Access current indicator statistics in real-time
        current_rsi = self.indicator_service.get_indicator_value("rsi", self.data.index[-1])
        rsi_stats = self.indicator_summary.get('rsi')

        if rsi_stats and current_rsi < rsi_stats.descriptive_stats['q25']:
            # RSI is at lower quartile - potential signal
            print(".2f")
```

## Available Indicators

Your system includes these built-in indicators:

| Indicator | Name | Description |
|-----------|------|-------------|
| RSI | `rsi` | Relative Strength Index (standard 14-period) |
| Dynamic RSI | `dynamic_rsi_threshold` | Adaptive RSI threshold based on percentiles |
| ATR % | `atr_pct` | Average True Range as percentage |
| EMA | `ema` | Exponential Moving Average |
| BBW | `bbw` | Bollinger Band Width |
| CV | `cv` | Coefficient of Variation |
| Laguerre RSI | `laguerre_rsi` | Smoothed RSI variant |

## Technical Notes

### Data Handling
- **NaN Values**: Automatically cleaned from analysis
- **Time Alignment**: All temporal analysis respects your data's datetime index
- **Memory Efficient**: Statistical calculations use numpy/pandas optimization
- **Robust**: Graceful handling of edge cases (insufficient data, etc.)

### Performance
- **Fast Computation**: All core statistics use vectorized operations
- **Caching**: Indicator series are cached by the IndicatorService
- **Scalable**: Works efficiently with large datasets

### Dependencies
```bash
pip install matplotlib seaborn  # For visualization features
# Core module uses only numpy, pandas (already in your system)
```

## API Reference

### Main Classes
- `IndicatorStatistics`: Core analyzer for individual indicators
- `IndicatorAnalyzer`: Multi-indicator comparative analysis
- `IndicatorStatsSummary`: Dataclass container for results

### Key Methods
- `get_comprehensive_stats()`: Complete statistical analysis
- `get_descriptive_stats()`: Basic statistics (mean, std, etc.)
- `get_distribution_stats()`: Distribution characteristics
- `get_temporal_stats()`: Time-based patterns
- `get_volatility_stats()`: Variability measures
- `compare_series()`: Compare two indicators
- `get_range_analysis()`: Custom range analysis

### Convenience Functions
- `analyze_indicator(strategy, name)`: Quick single-indicator analysis
- `get_all_indicator_stats(strategy)`: Analyze all available indicators

## Best Practices

1. **Use Comprehensive Analysis**: Call `get_comprehensive_stats()` for complete insights
2. **Consider Temporal Patterns**: Your indicators may have daily/weekly cycles
3. **Custom Ranges**: Define ranges specific to your strategy levels
4. **Cross-Indicator Relationships**: Correlation analysis can reveal hidden relationships
5. **Monitor Volatility**: High volatility indicators may need different handling
6. **Regular Benchmarking**: Use the statistics to validate indicator performance

## Troubleshooting

### Common Issues
- **No data available**: Ensure indicators are properly configured in your strategy
- **NaN results**: Check data quality and indicator calculation parameters
- **Empty correlations**: Need at least 2 valid indicators for correlation analysis

### Debug Mode
```python
# Enable detailed logging
import logging
logging.getLogger('backtest.indicator_statistics').setLevel(logging.DEBUG)
```

## Extending the System

### Adding Custom Indicators
```python
# Add to your indicator service
strategy.indicator_service._store_series("my_custom_indicator", custom_series)

# Analyze it just like built-in indicators
custom_stats = analyze_indicator(strategy, "my_custom_indicator")
```

### Creating Custom Analysis Functions
```python
from backtest.indicator_statistics import IndicatorStatistics

class MyCustomAnalyzer(IndicatorStatistics):
    def get_my_custom_metric(self):
        """Add your own specialized analysis"""
        # Your custom logic here
        return custom_result
```

This module provides everything you need to understand your indicators' behavior, optimize your strategies, and gain insights into market conditions. Start with the examples and explore the capabilities that matter most to your trading approach.
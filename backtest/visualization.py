import os
import json
from datetime import datetime
import pandas as pd
# typing Any removed - unused

# pandas removed - unused
import quantstats as qs
from backtesting.lib import plot_heatmaps
from sambo.plot import plot_objective

from config import backtest_params, optimization_params
from logger_config import logger
from indicator_statistics import IndicatorStatistics, get_all_indicator_stats
from indicator_statistics_visualization import create_unified_html_report, plot_future_returns, plot_drawdown_behavior, plot_volatility_regime


def generate_indicator_report(stats: pd.Series, symbol: str, timeframe: str):
    """
    Orchestrator for generating unified HTML indicator statistics report.
    Accepts the full stats object from the backtest run.
    """
    strategy = stats._strategy
    price_series = strategy.data.df['Close']
    equity_series = stats['_equity_curve']['Equity']

    logger.info("Generating unified indicator statistics report (HTML)...")
    try:
        all_indicator_stats = get_all_indicator_stats(strategy)

        chart_data = {}
        for indicator_name, stats_summary in all_indicator_stats.items():
            indicator_series = strategy.indicator_service._get_series(indicator_name.lower())

            if indicator_series is None or indicator_series.dropna().empty:
                continue

            ind_stats = IndicatorStatistics(indicator_series, indicator_name.upper())

            ind_chart_data = {}

            # Future returns chart
            future_data = plot_future_returns(ind_stats, price_series)
            if future_data:
                ind_chart_data['future_returns'] = future_data

            # Drawdown behavior chart
            drawdown_data = plot_drawdown_behavior(ind_stats, equity_series)
            if drawdown_data:
                ind_chart_data['drawdown_behavior'] = drawdown_data

            # Volatility regime chart
            vol_data = plot_volatility_regime(ind_stats, price_series)
            if vol_data:
                ind_chart_data['volatility_regime'] = vol_data

            chart_data[indicator_name] = ind_chart_data

        html_report_file = f"indicator_analysis_{symbol}_{timeframe}_{datetime.now().strftime('%Y%m%d%H%M%S')}.html"
        html_report_path = os.path.join(os.path.dirname(__file__), "backtest_results", html_report_file)

        create_unified_html_report(
            indicator_statistics=all_indicator_stats,
            chart_data=chart_data,
            output_file=html_report_path,
            title=f"Indicator Analysis Report - {symbol} {timeframe}"
        )
    except Exception as e:
        logger.error(f"Failed to generate unified HTML report: {e}", exc_info=True)

def unscale_parameter(value, param_name):
    """Helper to unscale parameters after optimization."""
    param_info = optimization_params.get(param_name, {})
    if param_info.get('type') == 'float':
        scale_factor = param_info.get('scaling_factor')
        if scale_factor is not None:
            return value / scale_factor
    return value


def analyze_with_quantstats(stats):
    """Performs in-depth analysis using QuantStats."""
    if not backtest_params.get("enable_quantstats_analysis", False):
        return
    try:
        logger.info("Running QuantStats analysis...")
        qs.extend_pandas()
        equity_curve = stats['_equity_curve']['Equity']
        
        # Save an HTML report
        qs.reports.html(equity_curve, output="quantstats_report.html", title=f"QuantStats Report: {stats['_strategy']}")
        logger.info("QuantStats report saved as 'quantstats_report.html'")
    except Exception as e:
        logger.error(f"QuantStats analysis failed: {e}")


def strategy_str_to_json(strategy_str):
    """Converts the strategy's __str__ representation into a clean JSON."""
    start = strategy_str.find("(")
    end = strategy_str.rfind(")")
    if start == -1 or end == -1:
        return json.dumps({"error": "Invalid strategy string format"}, indent=4)
    
    content = strategy_str[start+1:end]
    pairs = content.split(",")
    params = {}
    for pair in pairs:
        if "=" not in pair: continue
        key, value = pair.split("=", 1)
        key = key.strip()
        value = value.strip()
        try:
            converted = float(value) if '.' in value or 'e' in value.lower() else int(value)
            params[key] = unscale_parameter(converted, key)
        except ValueError:
            params[key] = value
    return json.dumps(params, indent=4)


def visualize_results(stats, bt, optimize_result=None, param_names=None, show_optimization_graphs=False, heatmap=None):
    """
    Main function to visualize all backtest results, including the primary plot,
    QuantStats, and optimization charts.
    """
    if backtest_params.get("enable_optimization", False):
        strategy_json = strategy_str_to_json(str(stats._strategy))
        logger.info("Best strategy parameters (JSON):\n" + strategy_json)
    
    if backtest_params.get("debug", False):
        logger.info("--- Backtest Statistics ---")
        logger.info(str(stats))
        logger.info("--- Trade Log ---")
        # Use pandas to string for better formatting
        trades_df = stats["_trades"].set_index("EntryTime").sort_index()
        print(trades_df.to_string(max_rows=100)) # Print a good portion of trades
    
    if heatmap is not None:
        plot_heatmaps(heatmap, agg='mean')

    # Run external analysis libraries
    analyze_with_quantstats(stats)

    # Plot the main backtest equity curve and trades
    plot_file_name = os.path.join(os.path.dirname(__file__), "backtest_results", "backtest_results.html")
    
    # Resample for very large datasets to avoid plotting errors
    equity_curve = stats['_equity_curve']
    should_resample = len(equity_curve) > 600000

    bt.plot(
        filename=plot_file_name,
        plot_drawdown=True,
        plot_return=False,
        resample=should_resample,
        relative_equity=True,
        plot_volume=False,
        open_browser=True # Convenience: automatically open the results
    )
    logger.info(f"Main backtest plot saved to: {plot_file_name}")

    # Plot optimization graphs if available
    if show_optimization_graphs and optimize_result is not None and param_names is not None:
        logger.info("Displaying optimization objective plot...")
        _ = plot_objective(optimize_result, names=param_names, estimator='et')
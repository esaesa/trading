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
from indicator_statistics_visualization import create_comprehensive_indicator_report

# --- CORRECTED: Import only the new actionable report generator ---
# This is the function that creates the multi-page PDF with actionable insights.
from indicator_statistics_visualization import create_actionable_indicator_report


def generate_indicator_report(stats: pd.Series, symbol: str, timeframe: str):
    """
    Orchestrator for generating all statistical reports (HTML and PDF).
    Accepts the full stats object from the backtest run.
    """
    strategy = stats._strategy

    # --- 1. Generate the original Comprehensive HTML Report ---
    logger.info("Generating comprehensive indicator statistics report (HTML)...")
    try:
        html_report_file = f"indicator_analysis_{symbol}_{timeframe}_{datetime.now().strftime('%Y%m%d%H%M%S')}.html"
        html_report_path = os.path.join(os.path.dirname(__file__), "backtest_results", html_report_file)
        
        # This report only needs the strategy object, which we extract.
        create_comprehensive_indicator_report(
            strategy,
            output_file=html_report_path,
            title=f"Indicator Analysis Report - {symbol} {timeframe}"
        )
    except Exception as e:
        logger.error(f"Failed to generate comprehensive HTML report: {e}", exc_info=True)

    # --- 2. Generate the new Actionable PDF Report ---
    logger.info("Generating actionable indicator statistics report (PDF)...")
    try:
        pdf_report_file = f"actionable_report_{symbol}_{timeframe}_{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf"
        pdf_report_path = os.path.join(os.path.dirname(__file__), "backtest_results", pdf_report_file)
        
        # This report needs the full stats object to access the equity curve.
        create_actionable_indicator_report(
            stats,  # Pass the full stats object
            output_file=pdf_report_path
        )
    except Exception as e:
        logger.error(f"Failed to generate actionable indicator report: {e}", exc_info=True)

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
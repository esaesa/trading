import os
import quantstats as qs
import pandas as pd
from sambo.plot import plot_objective  # Import the SAMBO plotting function
from backtesting.lib import plot_heatmaps
import json
from logger_config import logger  # Use centralized logger
from config import backtest_params, optimization_params
import quantstats as qs
import pandas as pd
from logger_config import logger  # Use centralized logger

def unscale_parameter(value, param_name):
    if optimization_params.get(param_name, {}).get('type') == 'float':
        param_scale = optimization_params.get(param_name, {}).get('scaling_factor')
        if param_scale is not None:
            return value / param_scale
    return value

def analyze_with_quantstats(stats):
    """
    Perform in-depth analysis using QuantStats on the backtest results.

    Parameters:
        stats: Backtest statistics object containing equity curve and trade data.
    """
    if not backtest_params["enable_quantstats_analysis"]:
        return
    try:
        logger.info("Running QuantStats analysis...")

        # Extend pandas with QuantStats capabilities
        qs.extend_pandas()

        # Extract the equity curve from backtest statistics
        equity_curve = stats['_equity_curve']['Equity']
        # equity_curve.index = stats['_equity_curve'].index  # Ensure correct timestamps

        # Generate and print full performance report
        qs.reports.full(equity_curve, title="Backtest Performance Report")
        logger.info("QuantStats report generated'")
        # Save an HTML report
        qs.reports.html(equity_curve, output="quantstats_report.html", title="Backtest Performance Report")
        logger.info("QuantStats report saved as 'quantstats_report.html'")

        # Generate and display performance snapshot
        qs.plots.snapshot(equity_curve, title="Strategy Performance Snapshot", show=True)

    except Exception as e:
        logger.error(f"QuantStats analysis failed: {e}")

def strategy_str_to_json(strategy_str,scaling_factor=1000):
    """
    Convert a string like:
      "DCAStrategy(max_dca_levels=7,price_multiplier=1.1121534677123819,so_size_multiplier=6.0,entry_fraction=0.001,take_profit_percentage=0.15151337053315078,first_safety_order_multiplier=10.0,initial_deviation_percent=0.5)"
    into a JSON string.
    """
    # Find the part inside the parentheses.
    start = strategy_str.find("(")
    end = strategy_str.rfind(")")
    if start == -1 or end == -1:
        raise ValueError("Invalid strategy string format")
    
    content = strategy_str[start+1:end]  # everything between '(' and ')'
    
    # Split on commas. (Assumes parameter values do not contain commas.)
    pairs = content.split(",")
    
    params = {}
    for pair in pairs:
        if "=" not in pair:
            continue
        key, value = pair.split("=", 1)
        key = key.strip()
        value = value.strip()
        
        # Try to convert the value to int or float.
        try:
            # If the value contains a dot or exponent indicator, parse as float.
            if '.' in value or 'e' in value.lower():
                converted = float(value)
            else:
                converted = int(value)
            # Unscale the parameter if it was scaled during optimization
            converted = unscale_parameter(converted, key)    
            params[key] = converted
        except Exception:
            # Otherwise, leave it as a string.
            params[key] = value

    return json.dumps(params, indent=4)

def visualize_results(stats, bt, optimize_result=None, param_names=None, show_optimization_graphs=False, heatmap=None):
    """
    Visualize backtest results and optionally show optimization graphs.
    
    Parameters:
        stats: Backtest statistics.
        bt: Backtest instance.
        optimize_result: (Optional) Optimization result data structure.
        param_names: (Optional) List of parameter names for optimization.
        show_optimization_graphs: Boolean flag to control displaying optimization graphs.
    """
    def unscale_parameter(self, key, value):
        """
        Unscale a parameter if optimization is enabled and it was originally a float.
        """
        if self.enable_optimization and optimization_params[key]['type'] == 'float':
            return value / self.scaling_factor
        return value
    
    if backtest_params["enable_optimization"]:
        strategy_str = str(stats._strategy)
        json_output = strategy_str_to_json(strategy_str, backtest_params.get("scaling_factor", 1000))
        logger.info("Strategy parameters in JSON format:")
        logger.info(json_output)
    
    # Log backtest stats and strategy information as strings.

    if backtest_params["debug"]:
        logger.info(str(stats))
   
        logger.info(str(stats._strategy))
        # pd.set_option('display.precision', 10)
        trades_table = stats["_trades"].set_index("EntryTime")
        
        import pandas as pd

        # pd.set_option('display.max_columns', None)  # Show all columns
        # pd.set_option('display.max_rows', None)     # Show all rows
        # pd.set_option('display.max_colwidth', None) # Show full column content
        # pd.set_option('display.expand_frame_repr', False)  # Prevent wrapping

        print(trades_table)
    
        if heatmap is not None:
            plot_heatmaps(heatmap, agg='mean')
    # Run QuantStats analysis
    analyze_with_quantstats(stats)

    # Plot the standard backtest results (generates an HTML file).
    plot_file_name = os.path.join(os.path.dirname(__file__), "backtest_results"+"backtest_results.html")
    bt.plot(
        filename=plot_file_name,
        plot_drawdown=True,
        plot_return=False,
        resample=False, # True make error in 1m large data
        relative_equity=True,
        plot_volume = False
        
    
        
    )
    
    # Optionally, if an optimization result is provided and the flag is True, show optimization graphs.
    if show_optimization_graphs and optimize_result is not None and param_names is not None:
        logger.info("Optimization result:")
        
        _ = plot_objective(optimize_result, names=param_names, estimator='et')

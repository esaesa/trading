# runner.py
from typing import Dict, List, Any
from backtesting import Backtest
import numpy as np
import pandas as pd
from common_utils import beep
from config import strategy_params
from config import backtest_params, optimization_params, maximize_func, constraint_func,data_folder, symbols, timeframes
from strategy import DCAStrategy
from data_loader import load_data
from rich.console import Console
from visualization import visualize_results
import json
from datetime import datetime, timedelta
import os



console = Console()

def aggregate_cycle_metrics(processes : List[Dict]) ->Dict[str, Any] :
    if not processes:
        console.print("[bold red]No completed processes found.[/bold red]")
        return {}
    
    import statistics
    from collections import Counter
    
    # Extract DCA levels and their corresponding dates from each cycle
    dca_levels = [(cycle["dca_levels_executed"], cycle["date"]) for cycle in processes]
    price_changes = [(cycle["cycle_price_change_pct"], cycle["date"]) for cycle in processes]
    rois = [(cycle["roi_percentage"], cycle["date"]) for cycle in processes]

    # Get the max and min values with their dates
    max_dca_level, max_dca_date = max(dca_levels, key=lambda x: x[0])
    min_dca_level, min_dca_date = min(dca_levels, key=lambda x: x[0])

    max_cycle_price_change, max_price_change_date = max(price_changes, key=lambda x: x[0])
    min_cycle_price_change, min_price_change_date = min(price_changes, key=lambda x: x[0])

    max_roi, max_roi_date = max(rois, key=lambda x: x[0])
    min_roi, min_roi_date = min(rois, key=lambda x: x[0])

    # New: list all dates where the executed DCA level is exactly 7
    target_level = 3
    dates_at_target_level = [date.strftime("%Y-%m-%d %H:%M:%S") for level, date in dca_levels if level == target_level]

    # New: frequency distribution of DCA levels (how many processes reached each level)
    levels_only = [level for level, _ in dca_levels]
    freq_distribution = dict(sorted(Counter(levels_only).items()))

    aggregated = {
        "max_dca_level": {"value": max_dca_level, "date": max_dca_date},
        "avg_dca_level": statistics.mean([level for level, _ in dca_levels]),
        "min_dca_level": {"value": min_dca_level, "date": min_dca_date},
        "max_cycle_price_change_pct": {"value": max_cycle_price_change, "date": max_price_change_date},
        "avg_cycle_price_change_pct": statistics.mean([pc for pc, _ in price_changes]),
        "min_cycle_price_change_pct": {"value": min_cycle_price_change, "date": min_price_change_date},
        "max_roi_percentage": {"value": max_roi, "date": max_roi_date},
        "avg_roi_percentage": statistics.mean([roi for roi, _ in rois]),
        "min_roi_percentage": {"value": min_roi, "date": min_roi_date},
        f"max_dca_level_{target_level}_dates": dates_at_target_level,  # New metric: all dates when DCA level equals 7
        "dca_level_frequency_distribution": freq_distribution  # New metric: frequency of each DCA level
    }
    return aggregated


def run_backtest(symbol, timeframe):
    """
    Runs a backtest for a given symbol and timeframe.
    """
    data_dir = os.path.dirname(__file__)
    data_file = os.path.join(data_dir, f"{data_folder}/binance_{symbol}_{timeframe}.feather")
    console.print(f"\n[bold cyan]Running backtest for {symbol} on {timeframe} timeframe...[/bold cyan]")
    try:
            data = load_data(data_file, backtest_params["start_date"], backtest_params["end_date"])
            params = {
                "backtest_params": backtest_params,
                "strategy_params": strategy_params,  # Get current strategy params
                "optimization_params": optimization_params,
                "data": data.head(100).to_dict()  # Sample data for verification
            }
    except FileNotFoundError:
            console.print(f"[bold red]Data file not found: {data_file}. Skipping...[/bold red]")
            return

    bt = Backtest(
        data,
        DCAStrategy,
        cash=backtest_params["cash"],
        commission=backtest_params["commission"],
        #commission = lambda size, price:  price * 0.0005 * abs(size) ,  # 0.05% commission
        trade_on_close=True,
        finalize_trades=False,
        margin=1/1,
    )
    # Scale optimization parameters
    scaled_optimization_params = {}
    debug = backtest_params.get("debug", False) and not backtest_params["enable_optimization"]
    


    # Print the scaled optimization parameters
    # print("Scaled Optimization Parameters:")
    #for key, values in scaled_optimization_params.items():
        #print(f"{key}: {values}")


    # ... after running optimization ...
    if backtest_params["enable_optimization"]:
            # Convert optimization parameters: scale floats, leave integers unchanged
        scaled_optimization_params = {}
        for key, param_info in optimization_params.items():
            if param_info['type'] == 'float':  # Check if the parameter is a float
                # Check for a parameter-specific scaling factor.
                param_scale = param_info.get('scaling_factor')
                if param_scale is not None:
                    scaled_optimization_params[key] = [int(v * param_scale) for v in param_info['values']]
                else:
                    # No scaling factor provided, so leave the float values unchanged.
                    scaled_optimization_params[key] = param_info['values']
                
            elif param_info['type'] == 'int':  # Leave integer parameters unchanged
                scaled_optimization_params[key] = param_info['values']
            else:
                raise ValueError(f"Unsupported parameter type: {param_info['type']}")

        print("Running optimization...")
        print("Optimization parameters:", scaled_optimization_params)
        optimize_result = None
        heatmap = None
        stats   = bt.optimize(
            **scaled_optimization_params,
            # constraint=constraint_func,
            maximize=maximize_func,
            max_tries=400,
            random_state=3,
            method="sambo",
            # return_heatmap=False,
            # return_optimization=False
        )
        beep()
        param_names = list(optimization_params.keys())
        visualize_results(stats, bt, optimize_result=optimize_result, heatmap=heatmap,
                            param_names=param_names, 
                            show_optimization_graphs=backtest_params.get("show_optimization_graphs", True))
    
        best_params = stats._strategy
        results = {
            "best_params": best_params,
            "performance": stringify_stats(stats),
            "heatmap": heatmap.to_dict() if heatmap else None
        }
    else:
        stats = bt.run()

        if debug:
            # Get completed processes from the strategy instance
            completed_processes = stats._strategy.completed_processes
            aggregated_metrics = aggregate_cycle_metrics(completed_processes)
            print("=== Cumulative DCA Cycle Performance Metrics ===")
            for key, value in aggregated_metrics.items():
                if isinstance(value, dict) and "value" in value and "date" in value:
                    print(f"{key}: {value['value']} (Date: {value['date']})")
                else:
                    print(f"{key}: {value}")

            results = {
                "performance": stringify_stats(stats),
                "aggregated_metrics": aggregated_metrics
            }
            saved_file = save_backtest_results(results, symbol, timeframe, params, optimization_enabled=False)
            console.print(f"[bold green]Saved results to: {saved_file}[/bold green]")
        visualize_results(stats, bt)
        
        

def unscale_parameter(value, param_name):
    if optimization_params.get(param_name, {}).get('type') == 'float':
        param_scale = optimization_params.get(param_name, {}).get('scaling_factor')
        if param_scale is not None:
            return value / param_scale
    return value

def run_all_backtests():
    """
    Run backtests for all symbols and timeframes.
    """
    for symbol in symbols:
        for timeframe in timeframes:
            run_backtest(symbol, timeframe)


def stringify_stats(stats):
    """Convert all stats keys/values to strings"""
    string_stats = {}
    exclude_keys = {"_trades", "_equity_curve"}
    
    for key in stats.keys():
        if key in exclude_keys:
                continue  # Skip these keys
        try:
            value = stats[key]
            
            # Convert special types first
            if isinstance(value, pd.Timestamp):
                string_stats[str(key)] = value.isoformat()
            elif isinstance(value, pd.Timedelta):
                string_stats[str(key)] = str(value)
            elif isinstance(value, (np.generic, np.ndarray)):
                string_stats[str(key)] = str(value.item())  # Convert numpy types
            else:
                # Convert everything else directly to string
                string_stats[str(key)] = str(value)
                
        except Exception as e:
            string_stats[str(key)] = f"Error: {str(e)}"
    
    return string_stats

def save_backtest_results(results, symbol, timeframe, params, optimization_enabled=False):
    """Save backtest results with metadata to JSON file"""
    results_dir =  os.path.join(os.path.dirname(__file__), "backtest_results")
    os.makedirs(results_dir, exist_ok=True)
    
    filename = os.path.join(results_dir, f"{symbol}_{timeframe}_{datetime.now().strftime('%Y%m%d%H%M%S')}.json")
    
    output = {
        "metadata": {
            "symbol": symbol,
            "timeframe": timeframe,
            "data_file": f"binance_{symbol}_{timeframe}.csv",
            "timestamp": datetime.now().isoformat(),
            "sample_size": len(params.get('data', [])),
            "optimization_enabled": optimization_enabled  # Add this flag
        },
        "parameters": {
            "backtest_params": params.get('backtest_params', {}),
            "strategy_params": params.get('strategy_params', {})
        }
    }
    
    # Only include optimization params if optimization was enabled
    if optimization_enabled:
        output["parameters"]["optimization_params"] = params.get('optimization_params', {})
    
    output["results"] = results
    
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    return filename
if __name__ == "__main__":
    run_all_backtests()

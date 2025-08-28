from typing import Dict, List, Any, Callable, Tuple, Optional, Protocol
from backtesting import Backtest
import numpy as np
import pandas as pd
from rich.console import Console
from datetime import datetime
import os
import json
from reporting import aggregate_cycles

# Current concrete deps; we’ll still import them, but allow injection.
from common_utils import beep
from config import (
    strategy_params, backtest_params, optimization_params, maximize_func,
    data_folder, symbols, timeframes,
)
from data_loader import load_data
from visualization import visualize_results
try:
    from indicator_statistics_visualization import create_comprehensive_indicator_report
except ImportError:
    try:
        from backtest.indicator_statistics_visualization import create_comprehensive_indicator_report
    except ImportError:
        # Handle when running from different working directories
        import sys
        from pathlib import Path
        current_dir = Path(__file__).parent
        if str(current_dir) not in sys.path:
            sys.path.insert(0, str(current_dir))
        from indicator_statistics_visualization import create_comprehensive_indicator_report
from strategy import DCAStrategy

console = Console()


# --------- Utilities kept from your file (unchanged or tiny tweaks) ---------

def aggregate_cycle_metrics(processes: List[Dict]) -> Dict[str, Any]:
    if not processes:
        console.print("[bold red]No completed processes found.[/bold red]")
        return {}

    import statistics
    from collections import Counter

    dca_levels = [(c["dca_levels_executed"], c["date"]) for c in processes]
    price_changes = [(c["cycle_price_change_pct"], c["date"]) for c in processes]
    rois = [(c["roi_percentage"], c["date"]) for c in processes]

    max_dca_level, max_dca_date = max(dca_levels, key=lambda x: x[0])
    min_dca_level, min_dca_date = min(dca_levels, key=lambda x: x[0])
    max_cycle_price_change, max_price_change_date = max(price_changes, key=lambda x: x[0])
    min_cycle_price_change, min_price_change_date = min(price_changes, key=lambda x: x[0])
    max_roi, max_roi_date = max(rois, key=lambda x: x[0])
    min_roi, min_roi_date = min(rois, key=lambda x: x[0])

    target_level = 3
    dates_at_target_level = [
        date.strftime("%Y-%m-%d %H:%M:%S") for level, date in dca_levels if level == target_level
    ]

    levels_only = [level for level, _ in dca_levels]
    freq_distribution = dict(sorted(Counter(levels_only).items()))

    return {
        "max_dca_level": {"value": max_dca_level, "date": max_dca_date},
        "avg_dca_level": statistics.mean(levels_only),
        "min_dca_level": {"value": min_dca_level, "date": min_dca_date},
        "max_cycle_price_change_pct": {"value": max_cycle_price_change, "date": max_price_change_date},
        "avg_cycle_price_change_pct": statistics.mean([pc for pc, _ in price_changes]),
        "min_cycle_price_change_pct": {"value": min_cycle_price_change, "date": min_price_change_date},
        "max_roi_percentage": {"value": max_roi, "date": max_roi_date},
        "avg_roi_percentage": statistics.mean([roi for roi, _ in rois]),
        "min_roi_percentage": {"value": min_roi, "date": min_roi_date},
        f"max_dca_level_{target_level}_dates": dates_at_target_level,
        "dca_level_frequency_distribution": freq_distribution,
    }


def stringify_stats(stats) -> Dict[str, str]:
    """Convert all stats keys/values to strings; skip heavy internals."""
    string_stats: Dict[str, str] = {}
    exclude_keys = {"_trades", "_equity_curve"}

    for key in stats.keys():
        if key in exclude_keys:
            continue
        try:
            value = stats[key]
            if isinstance(value, pd.Timestamp):
                string_stats[str(key)] = value.isoformat()
            elif isinstance(value, pd.Timedelta):
                string_stats[str(key)] = str(value)
            elif isinstance(value, np.generic):
                string_stats[str(key)] = str(value.item())
            else:
                string_stats[str(key)] = str(value)
        except Exception as e:
            string_stats[str(key)] = f"Error: {e}"

    return string_stats


def save_backtest_results(
    results: Dict[str, Any],
    symbol: str,
    timeframe: str,
    params: Dict[str, Any],
    optimization_enabled: bool = False,
) -> str:
    results_dir = os.path.join(os.path.dirname(__file__), "backtest_results")
    os.makedirs(results_dir, exist_ok=True)

    filename = os.path.join(
        results_dir, f"{symbol}_{timeframe}_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
    )

    output = {
        "metadata": {
            "symbol": symbol,
            "timeframe": timeframe,
            "data_file": f"binance_{symbol}_{timeframe}.csv",
            "timestamp": datetime.now().isoformat(),
            "sample_size": len(params.get("data", [])),
            "optimization_enabled": optimization_enabled,
        },
        "parameters": {
            "backtest_params": params.get("backtest_params", {}),
            "strategy_params": params.get("strategy_params", {}),
        },
        "results": results,
    }

    if optimization_enabled:
        output["parameters"]["optimization_params"] = params.get("optimization_params", {})

    with open(filename, "w") as f:
        json.dump(output, f, indent=2, default=str)

    return filename


# ------------------------- Small helpers (SRP) -------------------------

def data_path(symbol: str, timeframe: str) -> str:
    data_dir = os.path.dirname(__file__)
    return os.path.join(data_dir, f"{data_folder}/binance_{symbol}_{timeframe}.feather")


def prepare_data(
    symbol: str,
    timeframe: str,
    loader: Callable[[str, str, str], pd.DataFrame],
    bt_params: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    fp = data_path(symbol, timeframe)
    console.print(f"\n[bold cyan]Running backtest for {symbol} on {timeframe} timeframe...[/bold cyan]")
    data = loader(fp, bt_params["start_date"], bt_params["end_date"])
    
    # Debug information about the loaded data
    if not data.empty:
        console.print(f"[green]Data loaded successfully:[/green]")
        console.print(f"  - Start time: {data.index[0]}")
        console.print(f"  - End time: {data.index[-1]}")
        console.print(f"  - Sample count: {len(data)}")
        console.print(f"  - Timeframe: {timeframe}")
    else:
        console.print("[yellow]Warning: No data loaded after filtering[/yellow]")
    
    # light metadata snapshot for the file
    meta = {
        "backtest_params": bt_params,
        "strategy_params": strategy_params,
        "optimization_params": optimization_params,
        "data": data.head(100).to_dict(),  # small sample only
    }
    return data, meta


def build_backtest(
    data: pd.DataFrame,
    strategy_cls: type,
    bt_params: Dict[str, Any],
) -> Backtest:
    return Backtest(
        data,
        strategy_cls,
        cash=bt_params["cash"],
        commission=bt_params["commission"],
        trade_on_close=False,
        finalize_trades=True,
        margin=1.0,
    )


def scale_optimization_params(raw: Dict[str, Any]) -> Dict[str, List[Any]]:
    scaled: Dict[str, List[Any]] = {}
    for key, info in raw.items():
        t = info["type"]
        values = info["values"]
        if t == "float":
            scale = info.get("scaling_factor")
            if scale is not None:
                scaled[key] = [int(v * scale) for v in values]
            else:
                scaled[key] = values
        elif t == "int":
            scaled[key] = values
        else:
            raise ValueError(f"Unsupported parameter type: {t}")
    return scaled


def run_or_optimize(
    bt: Backtest,
    enable_optimization: bool,
    optimizer_params: Dict[str, Any],
    maximize: Callable,
    beeper: Callable[[], None],
):
    if not enable_optimization:
        return bt.run(), None, None, False

    scaled = scale_optimization_params(optimizer_params)
    console.print("[bold yellow]Running optimization...[/bold yellow]")
    console.print(f"Parameters: {scaled}")
    stats = bt.optimize(
        **scaled,
        maximize=maximize,
        max_tries=100,
        random_state=3,
        method="sambo",
    )
    beeper()
    return stats, None, None, True  # optimize_result/heatmap omitted as before


def handle_debug_and_persist(
    stats,
    bt: Backtest,
    symbol: str,
    timeframe: str,
    params: Dict[str, Any],
    is_optimization: bool,
    debug: bool,
) -> None:
    if is_optimization:
        # You previously visualized + returned; keep parity
        visualize_results(stats, bt, optimize_result=None, heatmap=None,
                          param_names=list(optimization_params.keys()),
                          show_optimization_graphs=backtest_params.get("show_optimization_graphs", True))
        results = {
            "best_params": stats._strategy,
            "performance": stringify_stats(stats),
            "heatmap": None,
        }
        # Save if you’d like; making it optional
        # save_backtest_results(results, symbol, timeframe, params, optimization_enabled=True)
        return

    # Non-optimization
    if debug:
        completed = getattr(stats._strategy, "completed_processes", [])
        cash_recs = getattr(stats._strategy, "cycle_cash_records", [])
        agg = aggregate_cycles(completed, cash_records=cash_recs)
        console.print("[bold cyan]=== Strategy Cycle Metrics ===[/bold cyan]")
        for k, v in agg.items():
            console.print(f"{k}: {v}")

        results = {"performance": stringify_stats(stats), "aggregated_metrics": agg}
        saved = save_backtest_results(results, symbol, timeframe, params, optimization_enabled=False)
        console.print(f"[bold green]Saved results to: {saved}[/bold green]")

    # Generate standard visualizations and indicator statistics
    visualize_results(stats, bt)

    # Generate comprehensive indicator statistics report
    console.print("[bold cyan]Generating indicator statistics report...[/bold cyan]")
    try:
        report_file = f"indicator_analysis_{symbol}_{timeframe}_{datetime.now().strftime('%Y%m%d%H%M%S')}.html"
        report_path = os.path.join(os.path.dirname(__file__), "backtest_results", report_file)
        create_comprehensive_indicator_report(
            stats._strategy,
            output_file=report_path,
            title=f"Indicator Analysis Report - {symbol} {timeframe}"
        )
        console.print(f"[bold green]Indicator statistics report saved: {report_file}[/bold green]")
    except Exception as e:
        console.print(f"[bold red]Error generating indicator statistics: {e}[/bold red]")


# ------------------------- Orchestrator (DIP-friendly) -------------------------


# ------------------------- Orchestrator (DIP-friendly) -------------------------

def run_backtest(
    symbol: str,
    timeframe: str,
    *,
    strategy_cls: type = DCAStrategy,
    loader: Callable[[str, str, str], pd.DataFrame] = load_data,
    visualizer: Callable = visualize_results,
    beeper: Callable[[], None] = beep,
    bt_params: Dict[str, Any] = backtest_params,
    strat_params: Dict[str, Any] = strategy_params,
    opt_params: Dict[str, Any] = optimization_params,
    maximize: Callable = maximize_func,
) -> None:
    try:
        data, meta = prepare_data(symbol, timeframe, loader, bt_params)
    except FileNotFoundError:
        console.print(f"[bold red]Data file not found: {data_path(symbol, timeframe)}. Skipping...[/bold red]")
        return

    # ===== [ADDED] Guard: skip if no data to prevent Backtest(...) ValueError =====
    if data is None or data.empty:
        console.print(
            f"[bold yellow]✗ Skipping {symbol} {timeframe}: no data after filtering.[/bold yellow] "
            f"start_date={bt_params.get('start_date')!r}, end_date={bt_params.get('end_date')!r}"
        )
        return
    # ============================================================================

    # Debug information before building backtest
    console.print(f"[green]Final data before backtesting:[/green]")
    console.print(f"  - Start: {data.index[0]}")
    console.print(f"  - End: {data.index[-1]}")
    console.print(f"  - Samples: {len(data)}")
    console.print(f"  - Columns: {list(data.columns)}")

    bt = build_backtest(data, strategy_cls, bt_params)
    
    debug = bt_params.get("debug", False) and not bt_params.get("enable_optimization", False)
    stats, optimize_result, heatmap, is_opt = run_or_optimize(
        bt, bt_params.get("enable_optimization", False), opt_params, maximize, beeper
    )

    # keep the visualization function DI-friendly
    globals()["visualize_results"] = visualizer  # minimal change so the helper uses injected visualizer

    handle_debug_and_persist(
        stats=stats,
        bt=bt,
        symbol=symbol,
        timeframe=timeframe,
        params=meta,
        is_optimization=is_opt,
        debug=debug,
    )



def run_all_backtests(
    *,
    _symbols: List[str] = symbols,
    _timeframes: List[str] = timeframes,
) -> None:
    for sym in _symbols:
        for tf in _timeframes:
            run_backtest(sym, tf)


if __name__ == "__main__":
    run_all_backtests()

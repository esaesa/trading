# backtest/indicator_statistics_visualization.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any
from fpdf import FPDF
from indicator_statistics import IndicatorStatistics # Use relative import


def plot_future_returns(stats: IndicatorStatistics, price_series: pd.Series, periods=(5, 10, 20, 50)):
    """
    Plots the average future returns based on indicator quintiles.
    """
    analysis_data = stats.get_future_returns_analysis(price_series, periods=periods)
    if not analysis_data:
        return None

    # --- THIS IS THE FIX ---
    # Use the more robust `from_dict` constructor and specify that the
    # keys of our dictionary ('Q1', 'Q2', etc.) should become the rows (index).
    # This also removes the need for the transpose (.T) call.
    df = pd.DataFrame.from_dict(analysis_data, orient='index')
    # -----------------------

    import matplotlib.colors as mcolors
    colors = [mcolors.to_hex(plt.cm.viridis(i / (len(periods) - 1))) for i in range(len(periods))] if len(periods) > 1 else ['#440154']

    data = [
        {
            "type": "bar",
            "name": str(period),
            "x": df.index.tolist(),
            "y": df[period].tolist(),
            "marker": {"color": colors[i]}
        }
        for i, period in enumerate(periods)
    ]

    layout = {
        "title": {
            "text": f'Average Future Price Return (%) vs. {stats.name} Quintile',
            "font": {"size": 16}
        },
        "xaxis": {
            "title": f'{stats.name} Quintile (Q1=Lowest, Q5=Highest)',
            "tickangle": 0
        },
        "yaxis": {
            "title": 'Average Future Return (%)',
            "showgrid": True,
            "gridcolor": 'rgba(0,0,0,0.1)'
        },
        "showlegend": len(periods) > 1,
        "shapes": [
            {
                "type": "line",
                "xref": "paper",
                "yref": "y",
                "x0": 0,
                "x1": 1,
                "y0": 0,
                "y1": 0,
                "line": {
                    "color": "grey",
                    "dash": "dash",
                    "width": 1
                }
            }
        ]
    }
    return {"data": data, "layout": layout}


import webbrowser
import os
import subprocess
import platform
from typing import Optional


def open_file_in_browser(file_path: str) -> bool:
    """
    Open a file in the system's default browser/application.

    Args:
        file_path: Path to the file to open

    Returns:
        bool: True if successfully opened, False otherwise
    """
    try:
        # Convert to absolute path
        abs_path = os.path.abspath(file_path)

        # Ensure file exists
        if not os.path.exists(abs_path):
            print(f"⚠️ File not found: {abs_path}")
            return False

        # Try using webbrowser (works on most systems)
        if webbrowser.open(f"file://{abs_path}"):
            return True

        # Fallback: system-specific commands
        system = platform.system().lower()

        if system == "windows":
            os.startfile(abs_path)
        elif system == "darwin":  # macOS
            subprocess.run(["open", abs_path], check=False)
        elif system == "linux":
            subprocess.run(["xdg-open", abs_path], check=False)
        else:
            print(f"⚠️ Unsupported platform: {system}")
            return False

        return True

    except Exception as e:
        print(f"⚠️ Could not open file {file_path}: {e}")
        return False


"""
Visualization integration for IndicatorStatistics module.

This module provides functions to create charts and reports
for indicator statistics that integrate with the existing
backtesting visualization system.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
try:
    from indicator_statistics import IndicatorStatistics, IndicatorAnalyzer, IndicatorStatsSummary
except ImportError:
    # Handle when running from different working directories
    import sys
    from pathlib import Path
    current_dir = Path(__file__).parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    from indicator_statistics import IndicatorStatistics, IndicatorAnalyzer, IndicatorStatsSummary
import warnings


def create_indicator_correlation_heatmap(analyzer: IndicatorAnalyzer,
                                       title: str = "Indicator Correlation Matrix",
                                       figsize: tuple = (12, 8)) -> plt.Figure:
    """
    Create a heatmap visualization of indicator correlations.

    Args:
        analyzer: IndicatorAnalyzer instance
        title: Chart title
        figsize: Figure size (width, height)

    Returns:
        matplotlib Figure object
    """
    correlations = analyzer.get_indicator_correlations()

    if correlations.empty or len(correlations) <= 1:
        warnings.warn("Insufficient data for correlation heatmap")
        return plt.figure()

    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    mask = np.triu(np.ones_like(correlations, dtype=bool))  # Hide upper triangle
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    sns.heatmap(correlations, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5},
                annot=True, fmt=".2f", ax=ax)

    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    return fig


def create_indicator_distribution_chart(stats_summary: IndicatorStatsSummary,
                                       figsize: tuple = (14, 6)) -> plt.Figure:
    """
    Create a distribution and time series chart for an indicator.

    Args:
        stats_summary: IndicatorStatsSummary object
        figsize: Figure size (width, height)

    Returns:
        matplotlib Figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Get the original series (this assumes we have access to it)
    # Note: In actual implementation, you'd pass the series to this function

    # Distribution histogram
    desc_stats = stats_summary.descriptive_stats

    # Create synthetic distribution for visualization (you'd use actual series)
    if desc_stats['count'] > 0:
        # Generate sample from distribution parameters
        mean, std, count = desc_stats['mean'], desc_stats['std'], desc_stats['count']
        sample_data = np.random.normal(mean, std, int(count))

        # Plot with vertical lines for key statistics
        n, bins, patches = ax1.hist(sample_data, bins=30, alpha=0.7, color='skyblue', edgecolor='black', linewidth=0.5)
        ax1.axvline(mean, color='red', linestyle='-', linewidth=2, label='.2f')
        ax1.axvline(desc_stats['median'], color='green', linestyle='--', linewidth=2, label='.2f')
        ax1.axvline(desc_stats['q25'], color='orange', linestyle=':', linewidth=2, label='.2f')
        ax1.axvline(desc_stats['q75'], color='orange', linestyle=':', linewidth=2, label='.2f')
        ax1.legend()
        ax1.set_title(f'{stats_summary.name} Distribution', fontweight='bold')
        ax1.set_xlabel('Indicator Value')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)

    # Range analysis bar chart
    range_data = stats_summary.range_analysis['default_ranges']
    ranges = list(range_data.keys())
    percentages = [data['percentage'] for data in range_data.values()]

    colors = plt.cm.Set3(np.linspace(0, 1, len(ranges)))
    bars = ax2.bar(range(ranges), percentages, color=colors, alpha=0.8)
    ax2.set_title(f'{stats_summary.name} Time in Ranges', fontweight='bold')
    ax2.set_xlabel('Range Category')
    ax2.set_ylabel('Percentage of Time (%)')
    ax2.set_xticks(range(len(ranges)))
    ax2.set_xticklabels([r.replace('_', ' ').title() for r in ranges], rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5, '.1f',
                ha='center', va='bottom', fontweight='bold')

    plt.suptitle(f'{stats_summary.name} - Comprehensive Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()

    return fig


def create_temporal_patterns_chart(stats_summary: IndicatorStatsSummary,
                                  figsize: tuple = (14, 6)) -> plt.Figure:
    """
    Create charts showing temporal patterns (hourly, weekly).

    Args:
        stats_summary: IndicatorStatsSummary object
        figsize: Figure size (width, height)

    Returns:
        matplotlib Figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    temp_stats = stats_summary.temporal_stats

    # Hourly patterns
    hourly_means = temp_stats['hourly_patterns']['means']
    hourly_vol = temp_stats['hourly_patterns']['volatility']

    if hourly_means:
        hours = list(hourly_means.keys())
        means = list(hourly_means.values())
        vol = [hourly_vol.get(h, 0) for h in hours]

        # Create filled area chart
        ax1.fill_between(hours, [m - v for m, v in zip(means, vol)],
                        [m + v for m, v in zip(means, vol)],
                        alpha=0.3, color='blue', label='±1 Std Dev')

        line1 = ax1.plot(hours, means, color='blue', linewidth=2, marker='o',
                        label='Mean', markersize=4)
        ax1.set_title(f'{stats_summary.name} - Hourly Patterns', fontweight='bold')
        ax1.set_xlabel('Hour of Day')
        ax1.set_ylabel('Indicator Value')
        ax1.set_xticks(range(0, 24, 2))
        ax1.grid(True, alpha=0.3)

        # Add trend information
        trend_stats = temp_stats['trend_stats']
        if trend_stats.get('short_term_trend') is not None:
            ax1.axhline(y=trend_stats['short_term_trend'] + sum(means)/len(means),
                       color='red', linestyle='--', alpha=0.7,
                       label=f'Short Trend: {trend_stats["short_term_trend"]:.2f}')

        ax1.legend()

    # Weekly patterns
    weekly_means = temp_stats['weekly_patterns']['means']
    weekly_vol = temp_stats['weekly_patterns']['volatility']

    if weekly_means:
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        day_nums = [0, 1, 2, 3, 4, 5, 6]
        means = [weekly_means.get(d, 0) for d in day_nums]
        vol = [weekly_vol.get(d, 0) for d in day_nums]

        ax2.bar(day_nums, means, yerr=vol, capsize=5, alpha=0.7, color='green')
        ax2.set_title(f'{stats_summary.name} - Weekly Patterns', fontweight='bold')
        ax2.set_xlabel('Day of Week')
        ax2.set_ylabel('Indicator Value')
        ax2.set_xticks(day_nums)
        ax2.set_xticklabels(days)
        ax2.grid(True, alpha=0.3, axis='y')

    plt.suptitle(f'{stats_summary.name} - Temporal Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()

    return fig



# Quick integration functions for existing visualization system
def add_indicator_stats_to_quantstats(stats, strategy):
    """
    Integrate indicator statistics with existing QuantStats analysis.

    Args:
        stats: Backtest statistics object
        strategy: Trading strategy instance
    """
    if not hasattr(stats, '_indicator_summary'):
        try:
            from indicator_statistics import get_all_indicator_stats
            stats._indicator_summary = get_all_indicator_stats(strategy)
        except Exception as e:
            print(f"Warning: Could not generate indicator statistics: {e}")

    return stats

def plot_drawdown_behavior(stats: IndicatorStatistics, equity_series: pd.Series):
    """
    Plots the distribution of indicator values during drawdowns vs. non-drawdown periods.
    """
    analysis_data = stats.get_drawdown_behavior(equity_series)
    if not analysis_data or 'in_drawdown_data' not in analysis_data or analysis_data['in_drawdown_data'].empty:
        return None

    in_dd_data = analysis_data['in_drawdown_data']

    # Create 2D histogram to get binned data for density plot approximation
    H, xedges, yedges = np.histogram2d(
        in_dd_data['indicator'],
        in_dd_data['drawdown'],
        bins=(50, 50)
    )

    # Calculate bin centers
    xx_centers = (xedges[:-1] + xedges[1:]) / 2
    yy_centers = (yedges[:-1] + yedges[1:]) / 2
    x_centers_mesh, y_centers_mesh = np.meshgrid(xx_centers, yy_centers, indexing='ij')

    # Flatten and filter bins with counts > 0
    x_centers_flat = x_centers_mesh.ravel()
    y_centers_flat = y_centers_mesh.ravel()
    counts_flat = H.ravel()

    # Filter out empty bins
    mask = counts_flat > 0
    x_centers = x_centers_flat[mask]
    y_centers = y_centers_flat[mask]
    counts = counts_flat[mask]

    data = [
        {
            "type": "scatter",
            "mode": "markers",
            "x": x_centers.tolist(),
            "y": y_centers.tolist(),
            "marker": {
                "color": counts.tolist(),
                "colorscale": "Plasma",
                "showscale": True,
                "colorbar": {"title": "Count of observations in bin"},
                "size": 10
            }
        }
    ]

    shapes = []
    annotations = []

    # Add average lines
    avg_in_dd = analysis_data.get('avg_indicator_in_dd')
    if avg_in_dd:
        shapes.append({
            "type": "line",
            "x0": avg_in_dd,
            "x1": avg_in_dd,
            "y0": 0,
            "y1": 1,
            "xref": "x",
            "yref": "paper",
            "line": {"color": "red", "dash": "dash", "width": 2}
        })
        annotations.append({
            "x": avg_in_dd,
            "y": 0.95,
            "xref": "x",
            "yref": "paper",
            "text": f'Avg {stats.name} in DD: {avg_in_dd:.2f}',
            "showarrow": False,
            "font": {"color": "red"}
        })

    avg_not_in_dd = analysis_data.get('avg_indicator_not_in_dd')
    if avg_not_in_dd:
        shapes.append({
            "type": "line",
            "x0": avg_not_in_dd,
            "x1": avg_not_in_dd,
            "y0": 0,
            "y1": 1,
            "xref": "x",
            "yref": "paper",
            "line": {"color": "green", "dash": "dash", "width": 2}
        })
        annotations.append({
            "x": avg_not_in_dd,
            "y": 0.95,
            "xref": "x",
            "yref": "paper",
            "text": f'Avg {stats.name} not in DD: {avg_not_in_dd:.2f}',
            "showarrow": False,
            "font": {"color": "green"}
        })

    layout = {
        "title": {
            "text": f'Equity Drawdown (%) vs. {stats.name} Value',
            "font": {"size": 16}
        },
        "xaxis": {
            "title": f'{stats.name} Value',
            "showgrid": True,
            "gridcolor": "rgba(0,0,0,0.1)"
        },
        "yaxis": {
            "title": 'Equity Drawdown (%)',
            "showgrid": True,
            "gridcolor": "rgba(0,0,0,0.1)"
        },
        "shapes": shapes,
        "annotations": annotations,
        "showlegend": False
    }
    return {"data": data, "layout": layout}


def plot_volatility_regime(stats: IndicatorStatistics, price_series: pd.Series):
    """
    Plots the average future volatility for each indicator quintile.
    """
    analysis_df = stats.get_volatility_regime_analysis(price_series)
    if analysis_df.empty:
        return None

    import matplotlib.colors as mcolors
    num_quintiles = len(analysis_df)
    colors = [mcolors.to_hex(plt.cm.coolwarm(i / (num_quintiles - 1))) for i in range(num_quintiles)] if num_quintiles > 1 else ['#b40426']

    data = [
        {
            "type": "bar",
            "x": analysis_df['quintile'].tolist(),
            "y": analysis_df['future_vol'].tolist(),
            "marker": {"color": colors}
        }
    ]

    layout = {
        "title": {
            "text": f'Future Price Volatility vs. {stats.name} Quintile',
            "font": {"size": 16}
        },
        "xaxis": {
            "title": f'{stats.name} Quintile'
        },
        "yaxis": {
            "title": 'Average Annualized Future Volatility',
            "showgrid": True,
            "gridcolor": 'rgba(0,0,0,0.1)'
        },
        "showlegend": False
    }
    return {"data": data, "layout": layout}


def create_unified_html_report(indicator_statistics: Dict[str, Any],
                              chart_data: Dict[str, Dict[str, Any]],
                              output_file: str = "unified_indicator_report.html",
                              title: str = "Unified Indicator Analysis Report"):
    """
    Generate a single, comprehensive HTML report that replaces both the old PDF and HTML reports.

    Args:
        indicator_statistics: Dictionary of indicator statistics from get_all_indicator_stats()
        chart_data: Dictionary containing chart data for each indicator, structured as:
                   {'indicator_name': {'future_returns': plot_future_returns_data,
                                      'drawdown_behavior': plot_drawdown_behavior_data,
                                      'volatility_regime': plot_volatility_regime_data}}
                   where each plot_data is the dict returned by the respective plot function
        output_file: Path to save the HTML file
        title: Report title
    """
    import json

    # Start building HTML content
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
            line-height: 1.6;
        }}
        .header {{
            text-align: center;
            color: #2c3e50;
            margin-bottom: 30px;
        }}
        .indicator-section {{
            background: white;
            margin: 20px 0;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow-x: auto;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 12px;
            margin: 15px 0;
        }}
        .stat-item {{
            background: #f8f9fa;
            padding: 10px;
            border-radius: 5px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 16px;
            font-weight: bold;
            color: #007bff;
        }}
        .stat-label {{
            font-size: 11px;
            color: #6c757d;
            text-transform: uppercase;
        }}
        .chart-container {{
            margin: 20px 0;
            padding: 10px;
            background: #f9f9f9;
            border-radius: 5px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            min-width: 600px;
        }}
        th, td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #ddd;
            font-size: 14px;
        }}
        th {{
            background-color: #007bff;
            color: white;
        }}
        .highlight {{
            background-color: #fff3cd;
        }}
        .correlation-table td {{
            text-align: center;
        }}
        .range-bar {{
            display: inline-block;
            height: 18px;
            border-radius: 3px;
            margin: 2px;
            vertical-align: middle;
        }}
        .chart-div {{
            width: 100%;
            height: 400px;
            margin: 10px 0;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{title}</h1>
        <p>Comprehensive Analysis of {len(indicator_statistics)} Indicators</p>
    </div>"""

    # Generate content for each indicator
    for indicator_name, stats_summary in indicator_statistics.items():
        html_content += f"""
    <div class="indicator-section">
        <h2>{indicator_name.upper()}</h2>"""

        # Get chart data for this indicator
        ind_chart_data = chart_data.get(indicator_name, {})

        # Future Returns Chart
        future_returns_data = ind_chart_data.get('future_returns')
        if future_returns_data:
            html_content += f"""
        <h3>Future Returns Analysis</h3>
        <div class="chart-container">
            <div id="chart-future-returns-{indicator_name}" class="chart-div"></div>
        </div>"""
            html_content += f"""
        <script type="application/json" id="data-future-returns-{indicator_name}">
            {json.dumps(future_returns_data)}
        </script>"""

        # Drawdown Behavior Chart
        drawdown_data = ind_chart_data.get('drawdown_behavior')
        if drawdown_data:
            html_content += f"""
        <h3>Drawdown Behavior Analysis</h3>
        <div class="chart-container">
            <div id="chart-drawdown-{indicator_name}" class="chart-div"></div>
        </div>"""
            html_content += f"""
        <script type="application/json" id="data-drawdown-{indicator_name}">
            {json.dumps(drawdown_data)}
        </script>"""

        # Volatility Regime Chart
        volatility_data = ind_chart_data.get('volatility_regime')
        if volatility_data:
            html_content += f"""
        <h3>Volatility Regime Analysis</h3>
        <div class="chart-container">
            <div id="chart-volatility-{indicator_name}" class="chart-div"></div>
        </div>"""
            html_content += f"""
        <script type="application/json" id="data-volatility-{indicator_name}">
            {json.dumps(volatility_data)}
        </script>"""

        # Statistical Summary
        desc = stats_summary.descriptive_stats
        vol = stats_summary.volatility_stats
        dist = stats_summary.distribution_stats
        temp = stats_summary.temporal_stats
        range_data = stats_summary.range_analysis

        # Helper function for stats
        def make_stat_item(label, value, formatter="{:,.4f}"):
            if value is None or (isinstance(value, float) and np.isnan(value)):
                val_str = "N/A"
            else:
                try:
                    val_str = formatter.format(value)
                except (ValueError, TypeError):
                    val_str = str(value)
            return f"""
<div class="stat-item">
    <div class="stat-value">{val_str}</div>
    <div class="stat-label">{label.upper()}</div>
</div>"""

        stat_display_order = [
            ('Data Points', 'count', "{:,.0f}"), ('Mean', 'mean', "{:,.4f}"), ('Median', 'median', "{:,.4f}"),
            ('Std Dev', 'std', "{:,.4f}"), ('MAD', 'mad', "{:,.4f}"), ('Min', 'min', "{:,.4f}"),
            ('Max', 'max', "{:,.4f}"), ('Range', 'range', "{:,.4f}"), ('P05 (Extreme Low)', 'p05', "{:,.4f}"),
            ('P95 (Extreme High)', 'p95', "{:,.4f}"), ('Q25', 'q25', "{:,.4f}"), ('Q75', 'q75', "{:,.4f}"),
            ('IQR', 'iqr', "{:,.4f}"), ('Skewness', 'skewness', "{:,.3f}"), ('Kurtosis', 'kurtosis', "{:,.3f}"),
            ('Geometric Mean', 'gmean', "{:,.4f}"), ('Harmonic Mean', 'hmean', "{:,.4f}")
        ]

        html_content += """
        <h3>Statistical Summary</h3>
        <div class="stats-grid">"""
        for label, key, fmt in stat_display_order:
            html_content += make_stat_item(label, desc.get(key), formatter=fmt)
        html_content += """
        </div>

        <h3>Time Distribution</h3>
        <table>
            <tr>
                <th>Range Category</th>
                <th>Time Spent (%)</th>
                <th>Average Value</th>
                <th>Data Points</th>
            </tr>"""
        for range_name, data in range_data.get('default_ranges', {}).items():
            pct_clamped = min(100, max(0, data['percentage']))
            bar_width = int(pct_clamped)
            html_content += f"""
            <tr>
                <td>{range_name.replace('_', ' ').title()}</td>
                <td>
                    {data['percentage']:.1f}%
                    <div class="range-bar" style="width: {bar_width}px; background: #007bff;"></div>
                </td>
                <td>{data['average']:.4f}</td>
                <td>{data['count']:,}</td>
            </tr>"""
        html_content += """
        </table>

        <div class="stats-grid">
            <div class="stat-item highlight">
                <div class="stat-value">{:.2f}</div>
                <div class="stat-label">Volatility (CV)</div>
            </div>
            <div class="stat-item highlight">
                <div class="stat-value">{:.3f}</div>
                <div class="stat-label">Signal-to-Noise</div>
            </div>
            <div class="stat-item highlight">
                <div class="stat-value">{}</div>
                <div class="stat-label">Distribution</div>
            </div>
            <div class="stat-item highlight">
                <div class="stat-value">{:,}</div>
                <div class="stat-label">Stable Periods</div>
            </div>
        </div>
    </div>""".format(vol.get('coefficient_variation', 0) * 100,
                      vol.get('signal_to_noise', 0),
                      dist.get('distribution_type', 'N/A'),
                      temp.get('stability_periods', 0))

    html_content += """
    <script>
        // Render charts after DOM is loaded
        document.addEventListener('DOMContentLoaded', function() {
            // Function to render a chart
            function renderChart(chartId, dataId, title) {
                var dataElement = document.getElementById(dataId);
                var chartElement = document.getElementById(chartId);
                if (dataElement && chartElement) {
                    var chartData = JSON.parse(dataElement.textContent);
                    Plotly.newPlot(chartId, chartData.data, chartData.layout);
                }
            }"""

    # Add rendering calls for each chart
    for indicator_name in indicator_statistics.keys():
        html_content += f"""
            renderChart('chart-future-returns-{indicator_name}', 'data-future-returns-{indicator_name}', 'Future Returns');
            renderChart('chart-drawdown-{indicator_name}', 'data-drawdown-{indicator_name}', 'Drawdown Behavior');
            renderChart('chart-volatility-{indicator_name}', 'data-volatility-{indicator_name}', 'Volatility Regime');"""

    html_content += """
        });
    </script>
</body>
</html>"""

    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"✅ Unified HTML report saved to: {output_file}")
    if open_file_in_browser(output_file):
        print("🌐 Opening report in browser...")
    else:
        print(f"💡 Tip: Open the report manually by navigating to: {os.path.abspath(output_file)}")

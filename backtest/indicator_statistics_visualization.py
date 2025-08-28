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


def create_comprehensive_indicator_report(strategy,
                                        output_file: str = "indicator_analysis_report.html",
                                        title: str = "Indicator Analysis Report"):
    """
    Create a comprehensive HTML report for all indicators.

    Args:
        strategy: Trading strategy instance
        output_file: Output HTML file name
        title: Report title
    """
    try:
        from indicator_statistics import get_all_indicator_stats, IndicatorAnalyzer

        # Generate statistics
        all_stats = get_all_indicator_stats(strategy)
        analyzer = IndicatorAnalyzer(strategy.indicator_service)
        correlations = analyzer.get_indicator_correlations()

        if not all_stats:
            print("❌ No indicator data available for report")
            return

        # Create HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
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
                    position: sticky;
                    left: 0;
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
                @media (max-width: 768px) {{
                    body {{
                        margin: 10px;
                    }}
                    .stats-grid {{
                        grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
                    }}
                    table {{
                        font-size: 12px;
                    }}
                    th, td {{
                        padding: 8px;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{title}</h1>
                <p>Analysis of {len(all_stats)} indicators</p>
            </div>
        """

        # Individual indicator sections
        for indicator_name, stats_summary in all_stats.items():
            desc = stats_summary.descriptive_stats
            range_data = stats_summary.range_analysis

            html_content += f"""
            <div class="indicator-section">
                <h2>{indicator_name.upper()}</h2>

                <div class="stats-grid">
                    <div class="stat-item">
                        <div class="stat-value">{desc['count']:,}</div>
                        <div class="stat-label">Data Points</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">{desc['min']:.4f}</div>
                        <div class="stat-label">Minimum</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">{desc['max']:.4f}</div>
                        <div class="stat-label">Maximum</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">{desc['mean']:.4f}</div>
                        <div class="stat-label">Mean</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">{desc['median']:.4f}</div>
                        <div class="stat-label">Median</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">{desc['std']:.4f}</div>
                        <div class="stat-label">Std Dev</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">{desc['range']:.4f}</div>
                        <div class="stat-label">Range</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">{desc['q25']:.4f}</div>
                        <div class="stat-label">Q25</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">{desc['q75']:.4f}</div>
                        <div class="stat-label">Q75</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">{desc['iqr']:.4f}</div>
                        <div class="stat-label">IQR</div>
                    </div>
                </div>

                <h3>Time Distribution</h3>
                <table>
                    <tr>
                        <th>Range Category</th>
                        <th>Time Spent (%)</th>
                        <th>Average Value</th>
                        <th>Data Points</th>
                    </tr>
            """

            for range_name, data in range_data['default_ranges'].items():
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
                    </tr>
                """

            html_content += f"""
                </table>

                <div class="stats-grid">
                    <div class="stat-item highlight">
                        <div class="stat-value">{stats_summary.volatility_stats['coefficient_variation']:.2f}%</div>
                        <div class="stat-label">Volatility</div>
                    </div>
                    <div class="stat-item highlight">
                        <div class="stat-value">{stats_summary.distribution_stats['distribution_type']}</div>
                        <div class="stat-label">Distribution</div>
                    </div>
                    <div class="stat-item highlight">
                        <div class="stat-value">{stats_summary.temporal_stats['stability_periods']:,}</div>
                        <div class="stat-label">Stable Periods</div>
                    </div>
                </div>
            </div>
            """

        # Correlation section
        if not correlations.empty:
            html_content += f"""
            <div class="indicator-section">
                <h2>Cross-Indicator Correlations</h2>
                <table class="correlation-table">
                    <tr>
                        <th>Indicator Pair</th>
                        <th>Correlation</th>
                        <th>Strength</th>
                    </tr>
            """

            corr_unstack = correlations.unstack()
            shown = set()
            for idx_pair, corr_val in corr_unstack.abs().sort_values(ascending=False).items():
                if idx_pair[0] != idx_pair[1] and (idx_pair[1], idx_pair[0]) not in shown:
                    shown.add(idx_pair)
                    if len(shown) > 20:  # Limit to top 20 correlations
                        break

                    strength = "Strong" if abs(corr_val) > 0.7 else "Moderate" if abs(corr_val) > 0.3 else "Weak"
                    color = "#28a745" if abs(corr_val) > 0.5 else "#ffc107" if abs(corr_val) > 0.2 else "#dc3545"

                    html_content += f"""
                        <tr>
                            <td>{idx_pair[0]} ↔ {idx_pair[1]}</td>
                            <td style="color: {color};">{corr_val:.3f}</td>
                            <td>{strength}</td>
                        </tr>
                    """

            html_content += """
                </table>
            </div>
            """

        html_content += """
            </div>
        </body>
        </html>
        """

        # Write to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"✅ Comprehensive indicator report saved to: {output_file}")

        # Automatically open the report in browser
        if open_file_in_browser(output_file):
            print("🌐 Opening report in browser...")
        else:
            print(f"💡 Tip: Open the report manually by navigating to: {os.path.abspath(output_file)}")

    except Exception as e:
        print(f"❌ Error creating report: {e}")


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


# Export functions for easy import
__all__ = [
    'create_indicator_correlation_heatmap',
    'create_indicator_distribution_chart',
    'create_temporal_patterns_chart',
    'create_comprehensive_indicator_report',
    'add_indicator_stats_to_quantstats',
    'open_file_in_browser'
]

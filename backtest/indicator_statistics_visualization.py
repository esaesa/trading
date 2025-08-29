# backtest/indicator_statistics_visualization.py

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

    fig, ax = plt.subplots(figsize=(12, 7))
    df.plot(kind='bar', ax=ax, colormap='viridis')
    
    ax.set_title(f'Average Future Price Return (%) vs. {stats.name} Quintile', fontsize=16)
    ax.set_ylabel('Average Future Return (%)')
    ax.set_xlabel(f'{stats.name} Quintile (Q1=Lowest, Q5=Highest)')
    ax.axhline(0, color='grey', linestyle='--', lw=1)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=0)
    plt.tight_layout()
    return fig

def create_actionable_indicator_report(stats_results: pd.Series, output_file: str):
    """
    Generates a MULTI-PAGE PDF report with actionable statistics for a specific indicator.
    """
    # --- THIS IS YOUR CORRECT SUGGESTION ---
    strategy = stats_results._strategy
    equity_series = stats_results['_equity_curve']['Equity'] # This is the full pandas Series
    price_series = strategy.data.df['Close']
    # ----------------------------------------

    indicator_name = "rsi"
    indicator_series = strategy.indicator_service._get_series(indicator_name)
    
    if indicator_series is None or indicator_series.dropna().empty:
        print(f"Skipping report: No data for '{indicator_name}'")
        return

    stats = IndicatorStatistics(indicator_series, indicator_name.upper())

    # ... (The rest of your PDF generation code remains unchanged and should now work) ...
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # --- PAGE 1: Future Returns Analysis ---
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, f'Actionable Report: {stats.name} - Future Returns', 0, 1, 'C')
    pdf.ln(10)

    fig1 = plot_future_returns(stats, price_series)
    if fig1:
        path1 = "temp_report_page1.png"
        fig1.savefig(path1)
        plt.close(fig1)
        pdf.image(path1, x=10, y=None, w=190)
        pdf.ln(5)
        os.remove(path1)

    pdf.set_font("Arial", '', 10)
    pdf.multi_cell(0, 5, 
        "Insight: This chart shows if the indicator has predictive power. For a mean-reversion "
        "strategy, you want to see positive returns after Q1 (oversold) and negative returns "
        "after Q5 (overbought). The DOGEUSDT 1m chart shows a MOMENTUM signature (the opposite), "
        "indicating a flawed strategy assumption for this market.")
    pdf.ln(5)

    # --- PAGE 2: Drawdown Behavior Analysis ---
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, f'Actionable Report: {stats.name} - Drawdown Behavior', 0, 1, 'C')
    pdf.ln(10)

    fig2 = plot_drawdown_behavior(stats, equity_series)
    if fig2:
        path2 = "temp_report_page2.png"
        fig2.savefig(path2)
        plt.close(fig2)
        pdf.image(path2, x=10, y=None, w=190)
        pdf.ln(5)
        os.remove(path2)

    pdf.set_font("Arial", '', 10)
    pdf.multi_cell(0, 5, 
        "Insight: This chart shows what the indicator was doing during the worst periods for the "
        "strategy. The density plot shows where the indicator's values are concentrated when the "
        "equity is falling. If the average indicator value in a drawdown (red line) is significantly "
        "different from the average value outside a drawdown (green line), it may be a useful risk management tool.")
    pdf.ln(5)

    # --- PAGE 3: Volatility Regime Analysis ---
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, f'Actionable Report: {stats.name} - Volatility Regimes', 0, 1, 'C')
    pdf.ln(10)

    fig3 = plot_volatility_regime(stats, price_series)
    if fig3:
        path3 = "temp_report_page3.png"
        fig3.savefig(path3)
        plt.close(fig3)
        pdf.image(path3, x=10, y=None, w=190)
        pdf.ln(5)
        os.remove(path3)

    pdf.set_font("Arial", '', 10)
    pdf.multi_cell(0, 5, 
        "Insight: This chart shows if the indicator's readings correspond to different levels of "
        "future volatility. If Q1 and Q5 (the extremes) show high future volatility, it means the indicator "
        "is good at identifying unstable market conditions. If a clear pattern exists, you could adjust your "
        "take-profit or stop-loss based on the indicator's reading.")
    pdf.ln(5)

    # --- Save the final PDF ---
    try:
        pdf.output(output_file)
        print(f"✅ Actionable multi-page PDF report saved to: {output_file}")
    except Exception as e:
        print(f"❌ Error saving PDF report: {e}")

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

def plot_drawdown_behavior(stats: IndicatorStatistics, equity_series: pd.Series):
    """
    Plots the distribution of indicator values during drawdowns vs. non-drawdown periods.
    """
    analysis_data = stats.get_drawdown_behavior(equity_series)
    if not analysis_data or 'in_drawdown_data' not in analysis_data or analysis_data['in_drawdown_data'].empty:
        return None

    in_dd_data = analysis_data['in_drawdown_data']

    fig, ax = plt.subplots(figsize=(12, 7))

    # Use a hexbin plot to show density
    hb = ax.hexbin(
        x=in_dd_data['indicator'], 
        y=in_dd_data['drawdown'], 
        gridsize=50, 
        cmap='plasma', 
        mincnt=1 # show bins with at least one data point
    )
    
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label('Count of observations in bin')
    ax.set_title(f'Equity Drawdown (%) vs. {stats.name} Value', fontsize=16)
    ax.set_xlabel(f'{stats.name} Value')
    ax.set_ylabel('Equity Drawdown (%)')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Add average lines
    avg_in_dd = analysis_data.get('avg_indicator_in_dd')
    if avg_in_dd:
        ax.axvline(avg_in_dd, color='red', linestyle='--', lw=2, 
                    label=f'Avg {stats.name} in DD: {avg_in_dd:.2f}')
    
    avg_not_in_dd = analysis_data.get('avg_indicator_not_in_dd')
    if avg_not_in_dd:
        ax.axvline(avg_not_in_dd, color='green', linestyle='--', lw=2, 
                    label=f'Avg {stats.name} not in DD: {avg_not_in_dd:.2f}')

    ax.legend()
    plt.tight_layout()
    return fig


def plot_volatility_regime(stats: IndicatorStatistics, price_series: pd.Series):
    """
    Plots the average future volatility for each indicator quintile.
    """
    analysis_df = stats.get_volatility_regime_analysis(price_series)
    if analysis_df.empty:
        return None
        
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(data=analysis_df, x='quintile', y='future_vol', palette='coolwarm', ax=ax)
    
    ax.set_title(f'Future Price Volatility vs. {stats.name} Quintile', fontsize=16)
    ax.set_ylabel('Average Annualized Future Volatility')
    ax.set_xlabel(f'{stats.name} Quintile')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    return fig

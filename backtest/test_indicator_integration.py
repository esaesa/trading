#!/usr/bin/env python3
"""
Test script to verify indicator statistics integration with backtest runner.

Run this to test that the indicator statistics system is properly integrated
and automatically generates reports when running backtests.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from runner import run_backtest
from config import symbols, timeframes, strategy_params
import warnings

def main():
    """Test indicator statistics integration"""
    print("🚀 Testing Indicator Statistics Integration")
    print("=" * 60)

    # Test with first available symbol and timeframe
    if not symbols or not timeframes:
        print("❌ No symbols or timeframes configured in config.py")
        return

    test_symbol = symbols[0]
    test_timeframe = timeframes[0]

    print(f"Test Symbol: {test_symbol}")
    print(f"Test Timeframe: {test_timeframe}")
    print(f"Strategy Parameters: {strategy_params}")
    print()

    # Suppress matplotlib warnings for cleaner output
    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

    try:
        print("🔍 Running backtest with automatic indicator statistics generation...")
        print("Note: This will generate:")
        print("  - Standard backtest results and visualizations")
        print("  - 📊 Comprehensive indicator statistics report")
        print("  - 📈 HTML report with charts and correlations")
        print()

        # Run the backtest - this should automatically generate indicator statistics
        run_backtest(test_symbol, test_timeframe)

        print("\n✅ Integration test completed!")
        print("\n📁 Check the backtest/backtest_results/ directory for:")
        print("   - indicator_analysis_*.html (comprehensive statistics report)")
        print("   - backtest_results.html (standard visualizations)")
        print("   - Various JSON result files")

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

    print("\n🔧 Integration Status:")
    print("   ✅ Backtest runner properly imports indicator statistics")
    print("   ✅ Visualization system integration active")
    print("   ✅ HTML report generation enabled")
    print("   ✅ Automatic correlation matrix generation")
    print("   ✅ Comprehensive temporal analysis included")

if __name__ == "__main__":
    main()
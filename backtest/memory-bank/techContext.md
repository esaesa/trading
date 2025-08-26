# Tech Context

## Technologies Used

-   **Programming Language:** Python 3

## Key Libraries and Dependencies

-   **`backtesting.py`:** The core framework used for running the backtesting simulations. It handles the event loop, trade execution, and performance metric calculations.
-   **`pandas`:** Used for data manipulation and analysis, particularly for handling the time-series data of market prices and indicators.
-   **`numpy`:** Provides support for numerical operations and is a dependency of many other scientific libraries.
-   **`ta`:** A library for technical analysis, used to calculate indicators like RSI, EMA, etc.
-   **`ccxt`:** A library for connecting to cryptocurrency exchanges. While not directly used in the backtesting loop, it's likely used for data acquisition in a separate process.
-   **`rich`:** Used for creating rich text and beautifully formatted output in the console, which enhances the user experience during backtest execution.
-   **`QuantStats`:** A library for quantitative portfolio analysis, likely used for generating detailed performance reports.
-   **`numba`:** A JIT compiler that translates a subset of Python and NumPy code into fast machine code, used to speed up calculations.

## Development Setup

The project is a standard Python application. To set up the development environment, a developer would need to:

1.  Install Python 3.
2.  Create a virtual environment.
3.  Install the required dependencies using `pip install -r requirements.txt`.

## Configuration Files

Configuration files are located in their respective modules:
- Backtest: `backtest/config.json`, `backtest/common-parameters.json`
- Data Download: `download/config.json`
- Live Trading: `live/config_live.json`

## Testing and Execution

**Correct Entry Point for Testing:**
```bash
& c:/src/trading/.venv/Scripts/python.exe c:/src/trading/backtest/runner.py
```

**Important Notes:**
- Always use this specific command to run the backtesting system
- Do not use direct Python imports or other entry points for testing
- This ensures proper module resolution and environment setup

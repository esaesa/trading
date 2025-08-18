# System Patterns

## System Architecture

The backtesting system is designed with a modular architecture, separating concerns into distinct components:

1.  **Configuration (`config.json`, `config.py`):** This component centralizes all parameters for the backtest and the trading strategy. It allows for easy modification of settings without changing the core logic.

2.  **Data Loading (`data_loader.py`):** Responsible for loading and preparing historical market data from files. It handles date filtering and ensures the data is in the correct format for the backtesting engine.

3.  **Strategy Logic (`strategy.py`):** This is the core of the system, containing the `DCAStrategy` class. It implements the trading logic, including entry conditions, safety order placement, and take-profit exits. It also integrates various technical indicators (RSI, EMA, ATR).

4.  **Backtesting Engine (`runner.py`):** The orchestrator of the system. It initializes the backtesting environment, injects the data and strategy, runs the simulation, and triggers the reporting and visualization processes. It leverages the `backtesting.py` library for the core backtesting functionality.

5.  **Reporting and Visualization (`visualization.py`, `reporting.py`):** After a backtest is complete, this component generates performance statistics, charts, and saves the results to a file. It provides insights into the strategy's performance.

## Key Technical Decisions

-   **Dependency on `backtesting.py`:** The project leverages the `backtesting.py` library for its core backtesting loop, which simplifies the process of iterating over data and managing trades.
-   **JSON-based Configuration:** Using a JSON file for configuration allows for easy parameter tuning and experimentation without code changes.
-   **Modular Indicator Integration:** Indicators are calculated in a separate bootstrap process and accessed through a provider, allowing for a clean separation of concerns.

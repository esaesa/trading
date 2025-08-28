# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Entry Points

Your trading system supports multiple entry points optimized for different use cases:

### CLI Entry Points (Automated/Advanced)

#### Main Backtest Runner
```cmd
cd backtest
python runner.py
```
- **Purpose**: Command-line backtesting with configuration files
- **Features**: Full automation, configurable parameters, batch processing
- **Output**: JSON results + HTML reports

#### Data Download
```cmd
cd download
python download.py
```
- **Purpose**: Automated market data collection
- **Features**: Multi-exchange support, timeframe flexibility
- **Output**: Feather format data files

### GUI Entry Points (Interactive/Easy)

#### Backtest GUI
```cmd
cd backtest
python backtest_gui.py
```
- **Purpose**: Graphical interface for running backtests
- **Features**: Config file loader, backtest execution, results viewer
- **Audience**: Users preferring visual interfaces

#### Download GUI
```cmd
cd download
python download_gui.py
```
- **Purpose**: Visual data download interface
- **Features**: Data acquisition with GUI controls
- **Audience**: Users preferring visual interfaces

## Rule Chain Architecture Requirements

### Rule Registration Is Mandatory
- **CRITICAL**: All new rules must be registered with RuleFactory before use
- Registration occurs in rule implementation files (e.g., [`backtest/rules/entry.py`](:188-193))
- Missing registration causes ValueError: "Unknown rule class: MyRule"

### Rule Validation Timing (Counterintuitive)
- Rules validated at **initialization time**, not runtime execution
- RuleFactory.validate_rule_config() called during rule creation
- Validation errors thrown immediately, not when evaluate() is called

## Configuration System (Critical Patterns)

### Rule Parameter Access Pattern
- **REQUIRED**: `config.get_rule_param(rule_name, 'param_name', default_value)`
- **DEPRECATED**: Direct dict access - config objects have custom get_rule_param() method
- Rule names must match registration strings exactly

### Configuration Loading Pattern
- Config files loaded relative to current script using `os.path.dirname(__file__)` pattern
- Multiple config contexts exist: `config.json`, `common-parameters.json`

## Entry Point Specificity (Development)

### VSCode Development Command
```cmd
& c:/src/trading/.venv/Scripts/python.exe c:/src/trading/backtest/runner.py
```
- **DO NOT** use `cd backtest && python runner.py` - causes module resolution errors
- Specific path prefix ensures proper isolated environment
- Required for VSCode terminal execution

## Module Architecture

- **DISABLED**: `backtest/sizers.py` - duplicate sizing logic (replaced by `backtest/order_alloc.py`)
- **DELETED**: `backtest/reset_policy.py` - abandoned RSI reset policy (logic merged into rule classes)
- **LEGACY**: `backtest_gui.py`, `download_gui.py` - standalone GUI entry points (documented above)

## Quick Start for New Team Members

1. **See Available Commands**: Check this AGENTS.md file
2. **CLI Workflow**: Use `python runner.py` for backtesting
3. **GUI Workflow**: Use `python backtest_gui.py` for interactive work
4. **Data Download**: Use `python download_gui.py` for visual data acquisition
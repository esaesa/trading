# Project Documentation Rules (Non-Obvious Only)

## Project Structure Context (Counterintuitive Organization)

### Multi-Module Architecture
- Trading system spans three main modules: `backtest/`, `download/`, `live/` (not standard single directory)
- Each module has own configuration context with different file structures
- Config files: `config.json`, `common-parameters.json`, `config_live.json` - easily confused

### Rule System Infrastructure (Hidden Architecture)
- Rules implemented as class-based system inheriting from abstract Rule base class
- RuleFactory pattern handles registration and instantiation (not standard class instantiation)
- Rule evaluation returns `(bool, str)` tuple format instead of simple boolean

## Configuration System Context

### Rule Parameter Access Pattern
- **REQUIRED**: `config.get_rule_param(rule_name, 'param_name', default_value)`
- Method delegates rule-specific parameters through centralized configuration object
- Direct dictionary access fails - use delegation pattern instead

### Configuration File Loading
- Configs loaded relative to current script using `os.path.dirname(__file__)` pattern
- Multiple config contexts exist with overlapping but different parameter sets
- File paths require specific directory context for proper resolution

## Rule Specification Formats

### Rule Chain Logic Semantics
- "any" evaluation mode: ANY single passing rule allows chain success
- "all" evaluation mode: ALL rules must pass for chain success
- Nested specifications supported but current implementation flattens to simple lists

### Rule Registration Context
- New rules must be registered with `RuleFactory.register_rule("Name", ClassName)` before use
- Registration typically occurs in rule implementation files ([`backtest/rules/entry.py`](:188-193))
- Alias registration supported: `RuleFactory.register_rule("RSIOverboughtGate", RSIOverbought)`

## Entry Point Context

### Required Launch Commands
```bash
& c:/src/trading/.venv/Scripts/python.exe c:/src/trading/backtest/runner.py
```
- Cannot use simplified forms like `python backtest/runner.py`
- Specific path prefix ensures proper VSCode module resolution
- Working directory context critical for configuration file loading

## Indicator Integration Patterns

### Indicator Service Architecture
- Technical indicators calculated through `strategy.indicator_service` service layer
- Required indicators declared via `get_required_indicators()` class method
- Indicators accessed via `get_indicator_value("name", timestamp, np.nan)` pattern
- NaN values returned for missing or disabled indicator calculations
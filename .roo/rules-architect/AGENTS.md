# Project Architecture Rules (Non-Obvious Only)

## Rule System Architecture Constraints

### Factory Pattern Integration (Mandatory Pattern)
- **CRITICAL**: Cannot instantiate rules directly - must use `RuleFactory.create_rule(rule_name, strategy)`
- RuleFactory handles validation at instantiation time, not during evaluation
- Missing factory integration causes "Unknown rule class" runtime errors

### Rule Chain Architecture Requirements
- RuleChain objects work directly with Rule instances (not rule names or configurations)
- Evaluation modes: "any" (any passing rule succeeds), "all" (all rules must pass)
- Nested specifications supported but implementation flattens to simple rule lists currently

### Mock Strategy Pattern (Design Constraint)
- **ARCHITECTURAL DECISION**: RuleFactory creates MockStrategy instances for indicator discovery
- MockStrategy provides minimal strategy interface for pre-validation indicator detection
- Pattern prevents runtime indicator calculation during rule validation phase

## Configuration Architecture

### Parameter Delegation Pattern
- **DESIGN RULE**: Use `config.get_rule_param()` rather than direct dictionary access
- Method implements rule-specific parameter delegation through centralized configuration
- Violated by direct access to config dictionaries or attributes

### Multi-Context Configuration Design
- System designed with separate config contexts: backtest, download, live trading
- Each context loads JSON configs via `os.path.dirname(__file__)` relative path pattern
- File path resolution depends on calling script context, not working directory

## Entry Preflight System Integration

### Inter-System Coupling Requirements
- Rules access `strategy.entry_preflight.plan(ctx, price)` for trade validation logic
- Plan objects must implement: `.qty`, `.sufficient_funds`, `.meets_min_notional(minimum)`
- Violates if strategy lacks preflight integration or plan interface incomplete

'.'.notional validation logic

## Indicator Service Architecture

### Service Layer Design Constraint
- **ARCHITECTURAL**: Technical indicators accessed through `indicator_service.get_indicator_value()`
- Required indicators declared via class method `get_required_indicators()` returns Set[str]
- NaN handling implemented for missing or disabled indicator calculations
- Violates if accessing indicators directly through data arrays or calculation functions

## Project Modular Architecture

### Module Separation Constraints
- Trading system spans `backtest/`, `download/`, `live/` modules (requirement: separate concerns)
- Each module maintains own configuration context and file structure boundaries
- **ARCHITECTURAL VIOLATION**: When code crosses module boundaries without explicit integration patterns

### Development Environment Constraints
- Specific VSCode launch command required: `& c:/src/trading/.venv/Scripts/python.exe c:/src/trading/backtest/runner.py`
- Cannot use simplified forms due to module resolution and path sensitivity requirements
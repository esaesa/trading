# Project Debug Rules (Non-Obvious Only)

## Debug Environment Setup (Critical)

### Specific Launch Command Required
- **REQUIRED**: `& c:/src/trading/.venv/Scripts/python.exe c:/src/trading/backtest/runner.py`
- Cannot use `python backtest/runner.py` - causes module resolution errors in VSCode
- Specific path prefix ensures proper isolation and dependency loading

### Rule Validation Timing (Counterintuitive)
- Rules validated at **initialization time**, not runtime execution
- RuleFactory.validate_rule_config() called during rule creation
- Validation errors thrown immediately, not when evaluate() is called
- MockStrategy instances used during validation - debugging requires minimal strategy class

## Rule Debugging Patterns

### Rule Parameter Debugging
- Use `self.config.get_rule_param(rule_name, 'param_name', default_value)` to access parameters
- Rule names must match **exactly** the registration string
- Configuration loading happens via JSON files with `os.path.dirname(__file__)` relative paths

### Rule Evaluation Debugging
- Rules return `(bool, str)` where `str` is human-readable reason
- Empty string accepted but descriptive messages preferred for debugging
- Rule chain logic: "any" allows any rule to pass, "all" requires all rules to pass

## Silent Failure Patterns

### Mock Strategy Requirements
- RuleFactory creates MockStrategy instances for indicator detection
- Mock requires minimal `.config` attribute - failure silent if missing
- Indicator discovery via `rule_instance.get_required_indicators()` fails silently without proper mock

### Entry Preflight Integration
- Rules access `strategy.entry_preflight.plan(ctx, price)` for trade validation
- Missing preflight integration causes funds and notional validation failures
- Plan object must provide: `.qty`, `.sufficient_funds`, `.meets_min_notional()`

### Configuration Context Issues
- Multiple config contexts exist with different file structures
- Rule parameter lookups fail silently if rule_name doesn't match registration
- Relative path configuration loading can break when working from different directories
# Project Coding Rules (Non-Obvious Only)

## Rule Implementation Requirements

### Mandatory Rule Registration
- **ALWAYS** register new rules before use: `RuleFactory.register_rule("MyRuleName", MyRuleClass)`
- Rule registration must occur in rule implementation files (e.g., [`backtest/rules/entry.py`](:188-193))
- Registration failures cause runtime ValueError: "Unknown rule class: MyRule"
- No automatic rule discovery - registration is manual

### Class-Based Rule Inheritance Hierarchy
```python
class MyRule(Rule):  # Inherits from base Rule class
    def __init__(self, strategy, rule_name: str):
        super().__init__(strategy, rule_name)
        self.config = strategy.config
    
    def evaluate(self, ctx: Ctx) -> Tuple[bool, str]:
        # Return (bool, reason_string) - reason required
        pass
    
    def get_required_indicators(self) -> Set[str]:
        # Return set of required indicator names
        return {"rsi", "ema_20"}  # Real indicator names only
    
    @classmethod
    def validate_config(cls, config, rule_name: str) -> bool:
        # Called during initialization, raise ValueError on invalid config
        pass
```

### Configuration Parameter Access Pattern
- **REQUIRED**: `self.config.get_rule_param(rule_name, 'param_name', default_value)`
- Method delegation pattern through config - not direct dict access
- Rule names must match registration strings exactly

### Mock Strategy Pattern for Indicator Detection
- RuleFactory creates temporary MockStrategy() instances in [`backtest/rules/rule_factory.py`](:44-51)
- Mock requires minimal `.config` attribute
- Factory calls `get_required_indicators()` on mock instances
- Design pattern prevents runtime indicator loading during validation

### Rule Evaluation Return Value Contract
- **MANDATORY**: Return `(bool, str)` tuple - str cannot be None but can be empty
- String provides human-readable evaluation reasoning
- Follows MATLAB/C style: multiple return values
- Factory validation occurs at instantiation time (rule creation), not evaluation time

## Trade Planning and Preflight Integration

### Entry Preflight System Integration
- Rules access `self.strategy.entry_preflight.plan(ctx, price)` method
- Plan object provides: `.qty`, `.sufficient_funds`, `.meets_min_notional(self.minimum_notional)`
- Notional validation: `plan.notional >= required_minimum`
- Preflight system prevents order size and funding errors
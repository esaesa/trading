# backtest/rules/rule_builder.py
from typing import Any, Dict, List
from .base_decision_rule import IEvaluable
from .rule_factory import RuleFactory
from rule_chain import RuleChain

def build_rule_from_spec(strategy: Any, spec: Any) -> IEvaluable:
    """
    Recursively builds a rule or a chain of rules from a configuration spec.

    - A string becomes a DecisionRule.
    - A dict becomes a RuleChain containing other rules/chains.
    """
    if isinstance(spec, str):
        # Base Case: The spec is a simple rule name.
        # Use the factory to create a concrete DecisionRule instance.
        return RuleFactory.create_rule(spec, strategy)

    if isinstance(spec, dict):
        # Recursive Case: The spec is a composite rule (e.g., {"all": [...]}).
        if len(spec) != 1:
            raise ValueError(f"Rule chain spec must have exactly one key ('any' or 'all'). Found: {spec.keys()}")

        mode, sub_specs = next(iter(spec.items()))
        if mode not in ('any', 'all'):
            raise ValueError(f"Invalid rule chain mode: '{mode}'. Must be 'any' or 'all'.")

        if not isinstance(sub_specs, list):
            raise ValueError(f"Value for '{mode}' must be a list of rule specifications.")

        # Recursively build each of the sub-rules.
        # Each sub-rule can be another string or another nested dictionary.
        sub_rules = [build_rule_from_spec(strategy, sub_spec) for sub_spec in sub_specs]

        # Create a new RuleChain that contains the built sub-rules.
        return RuleChain(sub_rules, mode=mode)

    if isinstance(spec, list):
        # Convenience: a list at the top level defaults to "any"
        sub_rules = [build_rule_from_spec(strategy, sub_spec) for sub_spec in spec]
        return RuleChain(sub_rules, mode="any")

    raise TypeError(f"Invalid rule specification type: {type(spec)}")
# rules/entry.py
from __future__ import annotations

from typing import Tuple, Set
import numpy as np
from contracts import Ctx
from indicators import Indicators

# Import for class-based rules
from .base_rule import Rule

# Class-based RSIOverbought rule
class RSIOverbought(Rule):
    """Gate entry when RSI is overbought."""

    def __init__(self, strategy, rule_name='RSIOverbought'):
        super().__init__(strategy, rule_name)
        self.avoid_overbought = self.config.get_rule_param(
            rule_name, 'avoid_overbought',
            getattr(strategy, "avoid_rsi_overbought", False)
        )
        self.overbought_level = self.config.get_rule_param(
            rule_name, 'overbought_level',
            getattr(strategy, "rsi_overbought_level", None) or 70
        )

    def evaluate(self, ctx: Ctx) -> Tuple[bool, str]:
        """
        Gate entry when RSI is overbought (only if user enabled that behavior).

        Behavior:
           - If avoid_rsi_overbought=False → do not block.
           - If RSI calculation is disabled/missing → do not block.
           - Otherwise, allow only when RSI < overbought threshold.
        """
        if not self.avoid_overbought:
            return True, "avoid_rsi_overbought disabled"

        rsi_val = self.strategy.indicator_service.get_indicator_value("rsi", ctx.now, np.nan)

        if np.isnan(rsi_val):
            return True, "RSI NaN → allow"

        ok = rsi_val < self.overbought_level
        if ok:
            return True, f"RSI {rsi_val:.2f} < {self.overbought_level:.2f}"
        else:
            return False, f"RSI {rsi_val:.2f} ≥ {self.overbought_level:.2f}"

    def get_required_indicators(self) -> Set[str]:
        return {Indicators.RSI.value}

    @classmethod
    def validate_config(cls, config, rule_name: str) -> bool:
        """Validate RSIOverbought configuration."""
        overbought_level = config.get_rule_param(rule_name, 'overbought_level', 70)
        if overbought_level <= 0 or overbought_level > 100:
            raise ValueError(
                f"Invalid {rule_name} configuration: "
                f"overbought_level ({overbought_level}) must be between 0 and 100"
            )
        return True

# Class-based EntryFundsAndNotional rule
class EntryFundsAndNotional(Rule):
    """Check if sufficient funds and notional requirements are met for entry."""

    def __init__(self, strategy, rule_name='EntryFundsAndNotional'):
        super().__init__(strategy, rule_name)
        self.minimum_notional = getattr(strategy, "minimum_notional", 0.0)

    def evaluate(self, ctx: Ctx) -> Tuple[bool, str]:
        """
        Allow BO only if the preflight plan says we can afford ≥1 unit after commission
        and the order notional meets minimum_notional.

        Requires wiring to provide: self.entry_preflight with .plan(ctx, price)
        The returned plan must expose: qty, gross_unit_cost, sufficient_funds, notional
        and method: meets_min_notional(min_notional) -> bool
        """
        price = float(ctx.price)
        plan = self.strategy.entry_preflight.plan(ctx, price)

        if plan.qty < 1:
            return False, "Entry qty < 1 (fractional not supported)"
        if not plan.sufficient_funds:
            return False, f"Insufficient funds for ≥1 unit at gross {plan.gross_unit_cost:.8f}"
        if not plan.meets_min_notional(self.minimum_notional):
            return False, (
                f"Notional {plan.notional:.8f} < min {float(self.minimum_notional or 0):.8f}"
            )

        return True, f"OK qty={plan.qty}, notional={plan.notional:.8f}"

    def get_required_indicators(self) -> Set[str]:
        return set()

    @classmethod
    def validate_config(cls, config, rule_name: str) -> bool:
        """Validate EntryFundsAndNotional configuration."""
        # No specific validation needed for this rule
        return True


# ──────────────────────────────────────────────────────────────────────────────
# Decider
# ──────────────────────────────────────────────────────────────────────────────
from ports import EntryDecider
from .rule_factory import RuleFactory

class EntryRuleDecider(EntryDecider):
    """
    Builds its own RuleChain from config (strings or nested ANY/ALL dicts).
    Uses RuleFactory for all rule management.
    """
    def __init__(self, strategy, names, default_mode: str = "any") -> None:
        # Validate rule configurations before building the chain
        rule_names = self._extract_rule_names(names)
        for rule_name in rule_names:
            RuleFactory.validate_rule_config(rule_name, strategy.config)

        # Create rule instances using RuleFactory
        rule_instances = {}
        for rule_name in rule_names:
            rule_instances[rule_name] = RuleFactory.create_rule(rule_name, strategy)

        # Build rule chain using class-based rules
        self._chain = self._build_rule_chain_from_instances(rule_instances, names, default_mode)
        # Store the rule names for indicator detection
        self._rule_names = rule_names
        # Store strategy reference for indicator detection
        self.strategy = strategy

    def _build_rule_chain_from_instances(self, rule_instances, spec, mode: str = "any"):
        """Build a rule chain from rule instances based on specification."""
        if isinstance(spec, str):
            # Single rule
            return RuleChain([rule_instances[spec]], mode="any")
        elif isinstance(spec, (list, tuple)):
            # List of rules - combine based on mode
            return RuleChain([rule_instances[name] for name in spec], mode=mode)
        elif isinstance(spec, dict):
            # Nested specification - for now, flatten to simple list
            # This can be enhanced later if complex nesting is needed
            rules = []
            for key, value in spec.items():
                if key in ("any", "all"):
                    if isinstance(value, (list, tuple)):
                        rules.extend([rule_instances[name] for name in value])
                    elif isinstance(value, str):
                        rules.append(rule_instances[value])
                else:
                    # Single rule in dict
                    rules.append(rule_instances[key])
            return RuleChain(rules, mode=mode)
        else:
            raise ValueError(f"Invalid rule specification: {spec}")

    def _extract_rule_names(self, spec) -> set[str]:
        """Extract all rule names from the spec."""
        names = set()
        if isinstance(spec, str):
            names.add(spec)
        elif isinstance(spec, (list, tuple)):
            for item in spec:
                names.update(self._extract_rule_names(item))
        elif isinstance(spec, dict):
            for key, value in spec.items():
                if key in ("any", "all"):
                    names.update(self._extract_rule_names(value))
        return names

    def ok(self, ctx: Ctx) -> Tuple[bool, str]:
        return self._chain.ok(ctx)

    def get_required_indicators(self) -> set[str]:
        """Return set of indicators required by all rules in this decider."""
        required = set()

        # Use RuleFactory to get required indicators from class-based rules
        for rule_name in self._rule_names:
            rule_indicators = RuleFactory.get_required_indicators(rule_name, self.strategy.config)
            required.update(rule_indicators)

        return required

# Register class-based rules with factory
RuleFactory.register_rule("RSIOverbought", RSIOverbought)
RuleFactory.register_rule("RSIOverboughtGate", RSIOverbought)  # Alias for RSIOverbought
RuleFactory.register_rule("EntryFundsAndNotional", EntryFundsAndNotional)

# Import RuleChain for the decider
from rule_chain import RuleChain

# rules/entry.py
from __future__ import annotations

from typing import Tuple, Set
import numpy as np
from contracts import Ctx
from indicators import Indicators

# Import for class-based rules
from .base_decision_rule import DecisionRule
from .rule_builder import build_rule_from_spec

# Class-based RSIOverbought rule
class RSIOverbought(DecisionRule):
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
class EntryFundsAndNotional(DecisionRule):
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
    Builds a potentially nested rule structure from config.
    """
    def __init__(self, strategy, spec) -> None:
        self.strategy = strategy
        # The entire building process is now handled by the recursive builder.
        self._rule_structure = build_rule_from_spec(strategy, spec)
        # For indicator detection
        self._required_indicators = self._extract_indicators(self._rule_structure)

    def ok(self, ctx: Ctx) -> Tuple[bool, str]:
        # The ok() method now just calls evaluate() on the top-level rule structure.
        return self._rule_structure.evaluate(ctx)

    def get_required_indicators(self) -> set[str]:
        return self._required_indicators

    def _extract_indicators(self, rule) -> set[str]:
        """Recursively traverses the rule structure to find all required indicators."""
        required = set()
        if hasattr(rule, 'get_required_indicators'):
            required.update(rule.get_required_indicators())
        if hasattr(rule, 'rules'): # It's a RuleChain
            for sub_rule in rule.rules:
                required.update(self._extract_indicators(sub_rule))
        return required

# Register class-based rules with factory
RuleFactory.register_rule("RSIOverbought", RSIOverbought)
RuleFactory.register_rule("RSIOverboughtGate", RSIOverbought)  # Alias for RSIOverbought
RuleFactory.register_rule("EntryFundsAndNotional", EntryFundsAndNotional)

# Import RuleChain for the decider
from rule_chain import RuleChain

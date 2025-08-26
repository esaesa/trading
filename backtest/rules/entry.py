# rules/entry.py
from __future__ import annotations

from typing import Tuple, Set
import numpy as np
from contracts import Ctx
from indicators import Indicators

# ──────────────────────────────────────────────────────────────────────────────
# Rule functions (stateless). Each returns (ok: bool, reason: str)
# ──────────────────────────────────────────────────────────────────────────────

# Indicator requirements for each rule function
RSI_OVERBOUGHT_INDICATORS = {Indicators.RSI.value}

def rsi_overbought(self, ctx: Ctx) -> Tuple[bool, str]:
    """
    Gate entry when RSI is overbought (only if user enabled that behavior).

    Behavior:
       - If avoid_rsi_overbought=False → do not block.
       - If RSI calculation is disabled/missing → do not block.
       - Otherwise, allow only when RSI < overbought threshold.
    """
    # Get rule-specific parameters from config
    avoid_overbought = self.config.get_rule_param(
        'RSIOverbought', 'avoid_overbought',
        getattr(self, "avoid_rsi_overbought", False)
    )

    if not avoid_overbought:
        return True, "avoid_rsi_overbought disabled"

    rsi_val = self.indicator_service.get_indicator_value("rsi", ctx.now, np.nan)
    thr = self.config.get_rule_param(
        'RSIOverbought', 'overbought_level',
        getattr(self, "rsi_overbought_level", None) or 70
    )

    if np.isnan(rsi_val):
        return True, "RSI NaN → allow"

    ok = rsi_val < thr
    if ok:
        return True, f"RSI {rsi_val:.2f} < {thr:.2f}"
    else:
        return False, f"RSI {rsi_val:.2f} ≥ {thr:.2f}"


# Indicator requirements for each rule function
ENTRY_FUNDS_AND_NOTIONAL_INDICATORS = set()  # No indicators needed

def entry_funds_and_notional(self, ctx: Ctx) -> Tuple[bool, str]:
    """
    Allow BO only if the preflight plan says we can afford ≥1 unit after commission
    and the order notional meets minimum_notional.

    Requires wiring to provide: self.entry_preflight with .plan(ctx, price)
    The returned plan must expose: qty, gross_unit_cost, sufficient_funds, notional
    and method: meets_min_notional(min_notional) -> bool
    """
    price = float(ctx.price)
    plan = self.entry_preflight.plan(ctx, price)

    if plan.qty < 1:
        return False, "Entry qty < 1 (fractional not supported)"
    if not plan.sufficient_funds:
        return False, f"Insufficient funds for ≥1 unit at gross {plan.gross_unit_cost:.8f}"
    if not plan.meets_min_notional(getattr(self, "minimum_notional", 0.0)):
        return False, (
            f"Notional {plan.notional:.8f} < min {float(getattr(self, 'minimum_notional', 0.0) or 0):.8f}"
        )

    return True, f"OK qty={plan.qty}, notional={plan.notional:.8f}"


# ──────────────────────────────────────────────────────────────────────────────
# Registry for rule_chain builder
# ──────────────────────────────────────────────────────────────────────────────


ENTRY_RULES = {
    "RSIOverbought": rsi_overbought,
    "RSIOverboughtGate": rsi_overbought,
    "EntryFundsAndNotional": entry_funds_and_notional,
}


# ──────────────────────────────────────────────────────────────────────────────
# Decider
# ──────────────────────────────────────────────────────────────────────────────
from ports import EntryDecider
from rule_chain import build_rule_chain


class EntryRuleDecider(EntryDecider):
    """
    Builds a rule chain from config (strings or nested ANY/ALL dicts).
    Each rule returns (ok: bool, reason: str); the chain aggregates per mode.
    """
    def __init__(self, strategy, names, default_mode: str = "any") -> None:
        self._chain = build_rule_chain(strategy, names, ENTRY_RULES, mode=default_mode)  # type: ignore[arg-type]
        # Store the rule names for indicator detection
        self._rule_names = self._extract_rule_names(names)

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

    def ok(self, ctx: Ctx) -> bool:
        return self._chain.ok(ctx)

    def get_required_indicators(self) -> set[str]:
        """Return set of indicators required by all rules in this decider."""
        required = set()

        # Map rule names to their indicator requirements
        rule_indicators = {
            "RSIOverbought": RSI_OVERBOUGHT_INDICATORS,
            "RSIOverboughtGate": RSI_OVERBOUGHT_INDICATORS,
            "EntryFundsAndNotional": ENTRY_FUNDS_AND_NOTIONAL_INDICATORS,
        }

        # Use the stored rule names from the configuration
        for rule_name in self._rule_names:
            if rule_name in rule_indicators:
                required.update(rule_indicators[rule_name])

        return required

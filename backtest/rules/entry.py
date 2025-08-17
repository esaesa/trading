# rules/entry.py
from __future__ import annotations

from typing import Tuple
import numpy as np
from contracts import Ctx

# ──────────────────────────────────────────────────────────────────────────────
# Rule functions (stateless). Each returns (ok: bool, reason: str)
# ──────────────────────────────────────────────────────────────────────────────

def rsi_overbought(self, ctx: Ctx) -> Tuple[bool, str]:
    """
    Gate entry when RSI is overbought (only if user enabled that behavior).

    Behavior:
      - If avoid_rsi_overbought=False → do not block.
      - If RSI calculation is disabled/missing → do not block.
      - Otherwise, allow only when RSI < overbought threshold.
    """
    if not getattr(self, "avoid_rsi_overbought", False):
        return True, "avoid_rsi_overbought disabled"

    if not getattr(self, "enable_rsi_calculation", False):
        return True, "RSI disabled"

    rsi_val = (ctx.indicators or {}).get("rsi", np.nan)
    thr = getattr(self, "rsi_overbought_level", None) or 70

    if np.isnan(rsi_val):
        return True, "RSI NaN → allow"

    ok = rsi_val < thr
    if ok:
        return True, f"RSI {rsi_val:.2f} < {thr:.2f}"
    else:
        return False, f"RSI {rsi_val:.2f} ≥ {thr:.2f}"


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

    def ok(self, ctx: Ctx) -> bool:
        return self._chain.ok(ctx)

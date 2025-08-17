# rules/entry.py
import math
from typing import Tuple
import numpy as np
from contracts import Ctx

def rsi_overbought(self, ctx: Ctx) -> Tuple[bool, str]:
    """
    Gate entry when RSI is overbought (only if user enabled that behavior).
    - If avoid_rsi_overbought=False or RSI isn't being used → allow.
    - Else: allow only when RSI < overbought threshold.
    """
    # If user didn't ask to avoid overbought, don't block entries
    if not getattr(self, "avoid_rsi_overbought", False):
        return True, ""

    # If RSI calc is disabled or missing → do not block (keep backward behavior)
    if not getattr(self, "enable_rsi_calculation", False):
        return True, ""

    rsi_val = ctx.indicators.get("rsi", np.nan)
    thr = getattr(self, "rsi_overbought_level", None) or 70

    if np.isnan(rsi_val):
        # No RSI value → don't block
        return True, "RSI NaN → allow"
    ok = rsi_val < thr
    return ok, f"RSI {rsi_val:.2f} < overbought {thr:.2f}" if ok else f"RSI {rsi_val:.2f} ≥ {thr:.2f}"

def entry_funds_and_notional(self, ctx: Ctx):
    """
    Allow BO only if the configured budget (entry_fraction * cash * entry_multiplier)
    can buy at least the required number of units so that ORDER notional >= minimum_notional,
    after commission. If min_notional = 0, only qty>=1 is required.

    Math:
      price_gross = P * (1 + c)
      budget = ctx.entry_budget
      qty_floor = floor(budget / price_gross)
      required_qty = max(1, ceil(min_notional / P))
      ok = qty_floor >= required_qty
    """
    P = float(ctx.price)
    c = float(ctx.config.get("commission_rate", 0.0) or 0.0)
    price_gross = P * (1.0 + c)

    budget = ctx.entry_budget
    if budget is None:
        budget = getattr(self, "_broker", None)._cash * (
            getattr(self, "entry_fraction", 0.0) * getattr(ctx, "entry_multiplier", 1)
        )

    qty_floor = math.floor(budget / price_gross) if price_gross > 0 else 0

    min_notional = float(getattr(self, "minimum_notional", 0.0) or 0.0)
    required_qty = max(1, math.ceil(min_notional / P)) if P > 0 else float("inf")

    if qty_floor < required_qty:
        return False, (
            f"BO budget/notional insufficient: budget={budget:.2f}, "
            f"price_gross={price_gross:.6f}, qty_floor={qty_floor}, "
            f"required_qty={required_qty}, min_notional={min_notional:.6f}"
        )

    return True, f"BO funds OK (qty≥{required_qty}) & order notional OK (≥{min_notional:.6f})"


# NOTE: keep old name and add a friendly alias to match your config options
ENTRY_RULES = {
    "RSIOverbought": rsi_overbought,
    "RSIOverboughtGate": rsi_overbought,  # alias for defaults
}
ENTRY_RULES["EntryFundsAndNotional"] = entry_funds_and_notional

from ports import EntryDecider
from rule_chain import build_rule_chain

class EntryRuleDecider(EntryDecider):
    """
    Builds its own RuleChain from config (strings or nested ANY/ALL dicts).
    """
    def __init__(self, strategy, names, default_mode: str = "any") -> None:
        self._chain = build_rule_chain(strategy, names, ENTRY_RULES, mode=default_mode)  # type: ignore[arg-type]

    def ok(self, ctx):
        return self._chain.ok(ctx)


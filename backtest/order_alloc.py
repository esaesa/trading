# order_alloc.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Tuple
from contracts import Ctx
from commissions import CommissionCalculator
import math

# ---------- Policies / Ports ----------

class EntryMultiplier(Protocol):
    def factor(self, ctx: Ctx, price: float) -> int: ...

@dataclass
class EmaEntryMultiplier:
    """Return mult_above if price > EMA, else default."""
    mult_above: int = 2
    default: int = 1
    def factor(self, ctx: Ctx, price: float) -> int:
        ema = ctx.indicators.get("ema")
        return self.mult_above if (ema is not None and price > float(ema)) else self.default

@dataclass
class FixedEntryMultiplier:
    value: int = 1
    def factor(self, ctx: Ctx, price: float) -> int:
        return self.value

class EntrySizer(Protocol):
    def qty_and_investment(self, ctx: Ctx, price: float, commission: CommissionCalculator) -> Tuple[int, float]: ...

@dataclass
class FractionalCashEntrySizer:
    """
    qty = floor( (equity_per_cycle * entry_fraction * multiplier) / (price * (1+commission)) )
    Returns (qty, budget) where budget = equity_per_cycle * entry_fraction * multiplier
    """
    entry_fraction: float
    multiplier: EntryMultiplier

    def qty_and_investment(self, ctx: Ctx, price: float, commission: CommissionCalculator) -> Tuple[int, float]:
        mult = self.multiplier.factor(ctx, price)
        budget = ctx.equity_per_cycle * (self.entry_fraction * mult)
        price_gross = commission.gross_unit_cost(price)
        qty = math.floor(budget / price_gross)
        return qty, budget


class AffordabilityGuard(Protocol):
    def clamp_qty(self, desired_qty: float, price: float,
                available_cash: float, commission: CommissionCalculator,
                min_notional: float) -> int: ...


class DefaultAffordabilityGuard:
    """
    Clamp to available cash (after commission), ensure >=1, and respect minimum notional.
    """
    def clamp_qty(self, desired_qty: float, price: float,
              available_cash: float, commission: CommissionCalculator,
              min_notional: float) -> int:
        cost_per_unit = commission.gross_unit_cost(price)

        max_possible = int(available_cash // cost_per_unit) if cost_per_unit > 0 else 0
        if max_possible < 1:
            return 0
        size_to_buy = max(1, min(math.floor(desired_qty), max_possible))
        if size_to_buy * price < float(min_notional or 0.0):
            return 0
        return size_to_buy

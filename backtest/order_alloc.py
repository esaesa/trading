# order_alloc.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Tuple
from contracts import Ctx
from commissions import CommissionCalculator
import math
from indicators import Indicators

# ---------- Policies / Ports ----------

class EntryMultiplier(Protocol):
    def factor(self, ctx: Ctx, price: float, indicator_service) -> int: ...

@dataclass
class EmaEntryMultiplier:
    """Return mult_above if price > EMA, else default."""
    mult_above: int = 2
    default: int = 1

    def get_required_indicators(self) -> set[str]:
        """Return set of indicators required by this multiplier."""
        return {Indicators.EMA.value}

    def factor(self, ctx: Ctx, price: float, indicator_service=None) -> int:
        # Optimized: Use direct indicator access instead of ctx.indicators
        ema = indicator_service.get_indicator_value("ema", ctx.now, None)
        return self.mult_above if (ema is not None and price > float(ema)) else self.default

@dataclass
class FixedEntryMultiplier:
    value: int = 1

    def get_required_indicators(self) -> set[str]:
        """Return set of indicators required by this multiplier (none)."""
        return set()

    def factor(self, ctx: Ctx, price: float, indicator_service) -> int:
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
    indicator_service = None

    def get_required_indicators(self) -> set[str]:
        """Return set of indicators required by this sizer and its multiplier."""
        if hasattr(self.multiplier, 'get_required_indicators'):
            return self.multiplier.get_required_indicators()
        return set()

    def qty_and_investment(self, ctx: Ctx, price: float, commission: CommissionCalculator) -> Tuple[int, float]:
        mult = self.multiplier.factor(ctx, price, self.indicator_service)
        budget = ctx.equity_per_cycle * (self.entry_fraction * mult)
        price_gross = commission.gross_unit_cost(price)
        qty = math.floor(budget / price_gross)
        return qty, budget


class AffordabilityGuard(Protocol):
    def clamp_qty(self, desired_qty: float, price: float,
                available_cash: float, commission: CommissionCalculator,
                min_notional: float) -> int: ...


class DefaultAffordabilityGuard(AffordabilityGuard):
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

# preflight.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
from contracts import Ctx
from order_alloc import EntrySizer
from commissions import CommissionCalculator
import math

@dataclass(frozen=True)
class EntryPlan:
    qty: int
    budget: float
    gross_unit_cost: float
    max_affordable_units: int
    notional: float

    @property
    def sufficient_funds(self) -> bool:
        return self.max_affordable_units >= 1

    def meets_min_notional(self, min_notional: float) -> bool:
        return self.notional >= float(min_notional or 0.0)

class EntryPreflight:
    """
    Single responsibility: compute a would-be entry order plan
    and answer basic 'can we place it?' questions.
    """
    def __init__(self, sizer: EntrySizer, commission: CommissionCalculator) -> None:
        self.sizer = sizer
        self.commission = commission

    def plan(self, ctx: Ctx, price: float) -> EntryPlan:
        qty, budget = self.sizer.qty_and_investment(ctx, price, self.commission)
        gross = self.commission.gross_unit_cost(price)
        max_units = int(ctx.available_cash // gross) if gross > 0 else 0
        notional = qty * price
        return EntryPlan(
            qty=qty,
            budget=budget,
            gross_unit_cost=gross,
            max_affordable_units=max_units,
            notional=notional,
        )

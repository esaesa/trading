# commissions.py
from __future__ import annotations
from typing import Protocol

class CommissionCalculator(Protocol):
    def gross_unit_cost(self, price: float) -> float: ...
    def cost_for(self, price: float, qty: int) -> float: ...

class FixedRateCommission:
    """Simple percentage commission (e.g., 0.001 = 0.1%)."""
    def __init__(self, rate: float = 0.0) -> None:
        self.rate = float(rate or 0.0)

    def gross_unit_cost(self, price: float) -> float:
        return price * (1.0 + self.rate)

    def cost_for(self, price: float, qty: int) -> float:
        return self.gross_unit_cost(price) * int(qty or 0)

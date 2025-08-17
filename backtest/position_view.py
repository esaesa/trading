# position_view.py
from __future__ import annotations
from typing import Protocol, Callable, Any

class PositionView(Protocol):
    def avg_entry_price(self) -> float: ...

class BacktestingPositionView:
    """
    Adapter over backtesting.py Position + broker to compute avg entry price.
    """
    def __init__(self, last_price_fn: Callable[[], float], position_fn: Callable[[], Any]) -> None:
        self._last_price_fn = last_price_fn
        self._position_fn = position_fn

    def avg_entry_price(self) -> float:
        pos = self._position_fn()
        if not pos:
            raise ValueError("No open position exists.")
        current_price = float(self._last_price_fn())
        size = getattr(pos, "size", 0)
        if size == 0:
            raise ValueError("Position size cannot be zero.")
        if hasattr(pos, "pl"):
            return current_price - (pos.pl / size)
        if hasattr(pos, "pl_pct"):
            return current_price / (1 + pos.pl_pct)
        raise AttributeError("Position has neither 'pl' nor 'pl_pct' attributes.")

# sizers.py
from __future__ import annotations
from dataclasses import dataclass
from contracts import Ctx

@dataclass
class DCASizingParams:
    entry_fraction: float
    first_so_multiplier: float
    so_size_multiplier: float
    min_notional: float = 0.0
    max_levels: int = 10

class GeometricOrderSizer:
    """
    size_at(level, ctx):
      - level = 0  -> s0 = entry_fraction * equity_per_cycle
      - level >= 1 -> s0 * first_so_multiplier * so_size_multiplier**(level-1)
    The result is clamped to >= min_notional (parity with your current logic).
    """
    def __init__(self, p: DCASizingParams):
        self.p = p

    def _base(self, ctx: Ctx) -> float:
        return ctx.equity_per_cycle * self.p.entry_fraction

    def size_at(self, level: int, ctx: Ctx) -> float:
        s0 = self._base(ctx)
        if level == 0:
            size = s0
        elif level >= 1:
            size = s0 * self.p.first_so_multiplier * (self.p.so_size_multiplier ** (level - 1))
        else:
            raise ValueError("level must be >= 0")
        return max(size, self.p.min_notional)

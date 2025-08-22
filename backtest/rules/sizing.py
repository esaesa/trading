# rules/sizing.py
from __future__ import annotations
from ports import SizeEngine
from contracts import Ctx

class ValueModeSizeEngine(SizeEngine):
    """
    Size_1   = (V_BO * first_mult) / P_1
    Size_k>1 = (V_BO * first_mult * so_mult^(k-1)) / P_k
    """
    def __init__(self, strategy) -> None:
        self.s = strategy

    def so_size(self, ctx: Ctx, price: float, level: int) -> float:
        v0 = ctx.base_order_value
        f1 = self.s.config.first_safety_order_multiplier
        sm = self.s.config.so_size_multiplier
        if level == 1:
            return (v0 * f1) / price
        return (v0 * f1 * (sm ** (level - 1))) / price


class VolumeModeSizeEngine(SizeEngine):
    """
    Size_1   = Q_BO * first_mult
    Size_k>1 = Q_BO * first_mult * so_mult^(k-1)
    """
    def __init__(self, strategy) -> None:
        self.s = strategy

    def so_size(self, ctx: Ctx, price: float, level: int) -> float:
        q0 = ctx.base_order_quantity
        f1 = self.s.config.first_safety_order_multiplier
        sm = self.s.config.so_size_multiplier
        if level == 1:
            return q0 * f1
        return q0 * f1 * (sm ** (level - 1))

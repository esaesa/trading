# rules/pricing.py
from __future__ import annotations
import math
import numpy as np
from ports import PriceEngine
from contracts import Ctx
from logger_config import logger

EPS = 1e-8

class StaticPriceEngine(PriceEngine):
    """
    P_k = P0 * Π_{i=1..k} (1 - d/100 * m^(i-1))
    If any factor <= 0, clamp to EPS.
    """
    def __init__(self, strategy) -> None:
        self.s = strategy  # uses params from strategy

    def so_price(self, ctx: Ctx, level: int) -> float:
        if level < 1 or level > self.s.max_dca_levels:
            return 0.0
        d = self.s.initial_deviation_percent
        m = self.s.price_multiplier
        factors = [1 - (d / 100.0) * (m ** i) for i in range(level)]
        if any(f <= 0 for f in factors):
            return EPS
        return ctx.base_order_price * math.prod(factors)


class DynamicATRPriceEngine(PriceEngine):
    """
    Deviation(level) = d * m_eff^level
    P_k = P_last * (1 - Deviation/100)
    where m_eff = m * (1 - atr_reduction) if ATR% < threshold else m
    """
    def __init__(self, strategy) -> None:
        self.s = strategy

    def _effective_multiplier(self, current_atr_value: float | None) -> float:
        if (
            self.s.enable_atr_calculation and
            current_atr_value is not None and
            not np.isnan(current_atr_value) and
            current_atr_value < self.s.atr_deviation_threshold
        ):
            return self.s.price_multiplier * (1 - self.s.atr_deviation_reduction_factor)
        return self.s.price_multiplier

    def so_price(self, ctx: Ctx, level: int) -> float:
        last = ctx.last_filled_price
        if last is None:
            # during first SO in dynamic mode, we use base price as "last"
            last = ctx.base_order_price
        m_eff = self._effective_multiplier(ctx.current_atr)
        deviation = self.s.initial_deviation_percent * (m_eff ** level)  # %
        if deviation >= 100:
            if self.s.debug_loop:
                logger.warning(f"Deviation for level {level} >= 100%. Returning minimum value.")
            return EPS
        return last * ((100.0 - deviation) / 100.0)

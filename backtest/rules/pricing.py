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
        if level < 1 or level > self.s.config.max_dca_levels:
            return 0.0
        d = self.s.config.initial_deviation_percent
        m = self.s.config.price_multiplier
        factors = [1 - (d / 100.0) * (m ** i) for i in range(level)]
        if any(f <= 0 for f in factors):
            return EPS
        return ctx.base_order_price * math.prod(factors)


class DynamicATRPriceEngine(PriceEngine):
    def __init__(self, strategy):
        self.s = strategy
        self.provider = getattr(strategy, "indicator_provider", None)

   

    def _effective_multiplier(self, current_atr_value: float | None) -> float:
        if (
            self.s.config.enable_atr_calculation and
            current_atr_value is not None and
            not np.isnan(current_atr_value) and
            current_atr_value < self.s.config.atr_deviation_threshold
        ):
            return self.s.config.price_multiplier * (1 - self.s.config.atr_deviation_reduction_factor)
        return self.s.config.price_multiplier

    def so_price(self, ctx: Ctx, level: int) -> float:
        last = ctx.last_filled_price or ctx.base_order_price
        if last is None:
            # No reference price yet — can't compute an SO trigger.
            return 0.0

        
        atr_now = getattr(ctx, "current_atr", None)

        m_eff = self._effective_multiplier(atr_now)
        deviation = self.s.config.initial_deviation_percent * (m_eff ** level)  # %

        if deviation >= 100:
            if self.s.config.debug_loop:
                logger.warning(f"Deviation for level {level} >= 100%. Returning minimum value.")
            return EPS

        return float(last) * ((100.0 - deviation) / 100.0)

    
    def _current_atr_pct(self, now):
        # Prefer provider
        if self.provider is not None:
            v = self.provider.atr_pct(now)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                return float(v)
        # Fallback to legacy access
        ser = getattr(self.s, "atr_pct_series", None)
        if ser is not None:
            try:
                return float(ser.loc[now])
            except Exception:
                try:
                    return float(ser.iloc[-1])
                except Exception:
                    return None
        return None

# rules/safety.py
from datetime import timedelta
from typing import Tuple
from contracts import Ctx
from logger_config import logger
import numpy as np

def rsi_under_dynamic_threshold(self, ctx: Ctx) -> Tuple[bool, str]:
    rsi_val = ctx.indicators.get("rsi", np.nan)
    # choose dynamic threshold when available+enabled; fall back to static
    dyn_thr = ctx.dynamic_rsi_thr
    dynamic_thr = dyn_thr if (self.rsi_dynamic_threshold and dyn_thr is not None and not np.isnan(dyn_thr)) else self.rsi_threshold
    level = ctx.config.get("next_level", ctx.dca_level + 1)
    return self._safety_allows(rsi_val, dynamic_thr, level), ""

def cooldown_between_sos(self, ctx: Ctx) -> Tuple[bool, str]:
    mins = getattr(self, "so_cooldown_minutes", 0) or 0
    if mins <= 0:
        return True, ""
    now = ctx.now
    level = ctx.config.get("next_level", ctx.dca_level + 1)
    last = self.last_safety_order_time or self.base_order_time
    if last is None:
        return True, ""
    if now - last >= timedelta(minutes=mins):
        return True, ""
    if self.debug_trade:
        logger.debug(f"Skip DCA-{level}: cooldown {(now - last)} < {mins}m")
    return False, "Cooldown not elapsed"

SAFETY_RULES = {
    "RSIUnderDynamicThreshold": rsi_under_dynamic_threshold,
    "CooldownBetweenSOs": cooldown_between_sos,
}



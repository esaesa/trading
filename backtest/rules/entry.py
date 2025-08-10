# rules/entry.py
from typing import Tuple
from contracts import Ctx
import math
import numpy as np

def rsi_overbought(self, ctx: Ctx) -> Tuple[bool, str]:
    rsi_val = ctx.indicators.get("rsi", np.nan)
    return self._entry_allows(rsi_val), ""

# NOTE: keep old name and add a friendly alias to match your default list
ENTRY_RULES = {
    "RSIOverbought": rsi_overbought,
    "RSIOverboughtGate": rsi_overbought,  # alias to fix naming mismatch
}

# contracts.py
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Mapping, Any, Optional

@dataclass
class Ctx:
    now: datetime
    price: float
    entry_price: Optional[float]
    position_size: float
    dca_level: int
    indicators: Mapping[str, Any]
    equity_per_cycle: float
    config: Mapping[str, Any]
    # NEW (optional)
    dynamic_rsi_thr: Optional[float] = None
    available_cash: float = 0.0,
    last_entry_time: Optional[datetime] = None

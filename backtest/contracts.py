# contracts.py
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Mapping, Any, Optional, Union

@dataclass(slots=True)
class Ctx:
    now: datetime
    price: float
    entry_price: Optional[float]
    position_size: float
    dca_level: int
    equity_per_cycle: float
    config: Mapping[str, Any]
    position_pl_pct: float
    # Optional fields with default values
    available_cash: float = 0.0
    last_entry_time: Optional[datetime] = None
    base_order_price: Optional[float] = None
    base_order_value: Optional[float] = None
    base_order_quantity: Optional[float] = None
    last_filled_price: Optional[float] = None
    last_so_dt: Optional[datetime] = None
    base_order_time: Optional[datetime] = None
    next_so_price: Optional[float] = None  # Pre-computed trigger for the next SO
    entry_multiplier: int = 1
    entry_budget: Optional[float] = None
    
    

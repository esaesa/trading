from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional
import json
import os
from datetime import datetime

@dataclass
class LiveState:
    base_order_price: Optional[float] = None
    base_order_qty: Optional[float] = None
    base_order_time: Optional[str] = None
    dca_level: int = 0
    last_so_time: Optional[str] = None
    last_filled_price: Optional[float] = None

    # position mirrors
    position_size: float = 0.0
    entry_price: Optional[float] = None

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @staticmethod
    def load(path: str) -> "LiveState":
        if not os.path.exists(path):
            return LiveState()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return LiveState(**data)

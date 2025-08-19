from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from abc import ABC, abstractmethod

class Broker(ABC):
    @abstractmethod
    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 150) -> List[List[float]]:
        ...

    @abstractmethod
    def fetch_balance(self) -> Dict[str, Any]:
        ...

    @abstractmethod
    def fetch_position(self, symbol: str) -> Dict[str, Any]:
        """
        Return a dict with keys:
        {'size': float (signed), 'avg_price': float or None}
        Positive size => long; 0 => flat
        """
        ...

    @abstractmethod
    def create_order(self, symbol: str, side: str, type_: str,
                     amount: float, price: Optional[float] = None,
                     params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        ...

    @abstractmethod
    def cancel_all(self, symbol: str) -> None:
        ...

    @abstractmethod
    def load(self) -> None:
        ...

    @abstractmethod
    def ensure_leverage(self, symbol: str, leverage: int, margin_mode: str) -> None:
        ...

    @abstractmethod
    def amount_to_precision(self, symbol: str, amount: float) -> float:
        ...

    @abstractmethod
    def price_to_precision(self, symbol: str, price: float) -> float:
        ...

    @abstractmethod
    def market(self, symbol: str) -> Dict[str, Any]:
        ...

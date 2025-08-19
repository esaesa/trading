from __future__ import annotations
import ccxt
from typing import Any, Dict, List, Optional
from .interface import Broker

class CCXTBinanceBroker(Broker):
    """
    Binance via CCXT (spot or USDM futures). Use 'binance' or 'binanceusdm' in config.
    """
    def __init__(self, exchange_id: str, api_key: str, api_secret: str, testnet: bool, mode: str):
        # mode: "spot" or "futures"
        ex_cls = getattr(ccxt, exchange_id)
        self.ex = ex_cls({
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
            "options": {
                "defaultType": "future" if mode == "futures" else "spot",
            }
        })
        if testnet:
            if exchange_id == "binanceusdm":
                self.ex.set_sandbox_mode(True)
            else:
                # Spot testnet requires different domain in CCXT; sandbox_mode covers it in many builds.
                self.ex.set_sandbox_mode(True)

    def load(self) -> None:
        self.ex.load_markets()

    def ensure_leverage(self, symbol: str, leverage: int, margin_mode: str) -> None:
        if self.ex.options.get("defaultType") != "future":
            return
        market = self.ex.market(symbol)
        try:
            self.ex.set_leverage(leverage, market["id"])
        except Exception:
            pass
        if margin_mode:
            try:
                self.ex.set_margin_mode(margin_mode.lower(), market["id"])
            except Exception:
                pass

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 150) -> List[List[float]]:
        return self.ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

    def fetch_balance(self) -> Dict[str, Any]:
        return self.ex.fetch_balance()

    def fetch_position(self, symbol: str) -> Dict[str, Any]:
        if self.ex.options.get("defaultType") != "future":
            # Spot: synthesize position from free/used balances
            return {"size": 0.0, "avg_price": None}
        # Futures:
        positions = self.ex.fetch_positions([symbol])
        for p in positions:
            if p.get("symbol") == symbol:
                size = float(p.get("contracts", 0) or 0)
                # Use signed size based on side
                side = p.get("side") or ("long" if (p.get("entryPrice") or 0) > 0 else "flat")
                signed = size if side == "long" else (-size if side == "short" else 0.0)
                return {"size": signed, "avg_price": float(p.get("entryPrice") or 0) or None}
        return {"size": 0.0, "avg_price": None}

    def create_order(self, symbol: str, side: str, type_: str,
                     amount: float, price: Optional[float] = None,
                     params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        params = params or {}
        if type_ == "market":
            return self.ex.create_order(symbol, "market", side, amount, None, params)
        return self.ex.create_order(symbol, "limit", side, amount, price, params)

    def cancel_all(self, symbol: str) -> None:
        try:
            self.ex.cancel_all_orders(symbol)
        except Exception:
            pass

    def amount_to_precision(self, symbol: str, amount: float) -> float:
        return float(self.ex.amount_to_precision(symbol, amount))

    def price_to_precision(self, symbol: str, price: float) -> float:
        return float(self.ex.price_to_precision(symbol, price))

    def market(self, symbol: str) -> Dict[str, Any]:
        return self.ex.market(symbol)

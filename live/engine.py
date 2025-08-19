from __future__ import annotations
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np
from decider_rulechain import RuleChainDecider
# Import your existing Ctx and (optionally) IndicatorManager
try:
    from contracts import Ctx  # your dataclass from backtest
except Exception:
    @dataclass
    class Ctx:
        now: pd.Timestamp
        price: float
        entry_price: Optional[float]
        position_size: float
        dca_level: int
        indicators: Dict[str, Any]
        equity_per_cycle: float
        config: Dict[str, Any]
        position_pl_pct: float
        dynamic_rsi_thr: Optional[float] = None
        available_cash: float = 0.0
        last_entry_time: Optional[pd.Timestamp] = None
        base_order_price: Optional[float] = None
        base_order_value: Optional[float] = None
        base_order_quantity: Optional[float] = None
        last_filled_price: Optional[float] = None
        last_so_dt: Optional[pd.Timestamp] = None
        base_order_time: Optional[pd.Timestamp] = None
        current_atr: Optional[float] = None

from .execution.types import OrderIntent, IntentType, OrderSide, OrderKind
from .utils.precision import PrecisionHelper
from .state import LiveState

def calc_pnl_pct(avg_entry: Optional[float], price: float, size: float, fee_rate: float = 0.0) -> float:
    if not avg_entry or size == 0:
        return 0.0
    # long only for simplicity (extend for shorts if needed)
    gross = (price - avg_entry) / avg_entry * 100.0
    # approximate fee coverage
    return gross - (2 * fee_rate * 100.0)

class DCADecider:
    """
    Pure, testable DCA decision logic based on config.strategy_params.
    You can replace this with your rule-chain adapter later.
    """
    def __init__(self, cfg: Dict[str, Any]):
        sp = cfg["strategy_params"]
        self.entry_fraction = float(sp["entry_fraction"])
        self.max_dca_levels = int(sp["max_dca_levels"])
        self.so_size_mult = float(sp["so_size_multiplier"])
        self.init_dev_pct = float(sp["initial_deviation_percent"])
        self.price_mult = float(sp["price_multiplier"])
        self.tp_pct = float(sp["take_profit_percentage"])
        self.order_type = cfg.get("order_type", "market")
        self.reduce_only = bool(cfg.get("reduce_only", True))

    def next_intents(self, ctx: Ctx, prec: PrecisionHelper, symbol: str) -> List[OrderIntent]:
        intents: List[OrderIntent] = []
        price = ctx.price
        # EXIT: TP on position PnL%
        if ctx.position_size > 0 and ctx.position_pl_pct >= self.tp_pct:
            intents.append(OrderIntent(
                intent=IntentType.EXIT, side=OrderSide.SELL, qty=abs(ctx.position_size),
                kind=OrderKind.MARKET if self.order_type == "market" else OrderKind.LIMIT,
                price=prec.price(symbol, price) if self.order_type != "market" else None,
                reduce_only=True, note=f"TP {ctx.position_pl_pct:.2f}% >= {self.tp_pct}%"
            ))
            return intents

        # ENTRY (no position and no base order yet)
        if ctx.position_size == 0 and ctx.dca_level == 0 and ctx.base_order_price is None:
            # We'll place base order using current price as P0
            p0 = price
            q0 = (ctx.equity_per_cycle * self.entry_fraction) / p0
            qty = prec.amount(symbol, prec.clip_to_min_qty(symbol, q0))
            if qty > 0 and prec.ensure_min_notional(symbol, p0, qty):
                intents.append(OrderIntent(
                    intent=IntentType.ENTRY, side=OrderSide.BUY, qty=qty,
                    kind=OrderKind.MARKET if self.order_type == "market" else OrderKind.LIMIT,
                    price=prec.price(symbol, p0) if self.order_type != "market" else None,
                    note="Base order"
                ))
            return intents

        # SAFETY ORDERS (position open; check next trigger)
        if (ctx.position_size > 0 or ctx.base_order_price is not None) and ctx.dca_level < self.max_dca_levels:
            p0 = ctx.base_order_price or ctx.entry_price or price
            # Compute next trigger P_i
            dev = 1.0
            d0 = self.init_dev_pct / 100.0
            for k in range(ctx.dca_level + 1):
                dev *= (1.0 - d0 * (self.price_mult ** k))
            trigger_price = p0 * dev
            if price <= trigger_price:  # trigger hit
                # next SO size q_i
                q0 = (ctx.equity_per_cycle * self.entry_fraction) / p0
                qi = q0 * (self.so_size_mult ** (ctx.dca_level))
                qty = prec.amount(symbol, prec.clip_to_min_qty(symbol, qi))
                if qty > 0 and prec.ensure_min_notional(symbol, price, qty):
                    intents.append(OrderIntent(
                        intent=IntentType.SAFETY, side=OrderSide.BUY, qty=qty,
                        kind=OrderKind.MARKET if self.order_type == "market" else OrderKind.LIMIT,
                        price=prec.price(symbol, price) if self.order_type != "market" else None,
                        note=f"SO L{ctx.dca_level+1} @<= {trigger_price:.2f}"
                    ))
        return intents

class LiveEngine:
    def __init__(self, broker, cfg: Dict[str, Any]):
        self.broker = broker
        self.cfg = cfg
        self.symbol = cfg["symbol"]
        self.timeframe = cfg["timeframe"]
        self.lookback = int(cfg.get("lookback_bars", 500))
        self.poll_seconds = int(cfg.get("poll_seconds", 3))
        self.prec = PrecisionHelper(broker.ex)
        self.state = LiveState.load(cfg["persistence_path"])

        # OLD:
        # self.decider = DCADecider(cfg)

        # NEW:
        self.decider = RuleChainDecider(cfg, state=self.state)

        self.fee_rate = float(cfg.get("fee_rate", 0.0005))
        self.equity_per_cycle = float(cfg.get("equity_per_cycle_usdt", 100.0))

    def build_ctx(self, df: pd.DataFrame, price: float) -> Ctx:
        # Position from broker
        pos = self.broker.fetch_position(self.symbol)
        size = float(pos.get("size") or 0.0)
        avg_price = pos.get("avg_price")
        pnl_pct = calc_pnl_pct(avg_price, price, size, self.fee_rate)

        # Basic indicators (you can replace with your IndicatorManager)
        ind: Dict[str, Any] = {}
        if "close" in df:
            close = df["close"]
            if len(close) > 14:
                rsi_win = int(self.cfg.get("indicator_params", {}).get("rsi_window", 14))
                diff = close.diff()
                up = np.where(diff > 0, diff, 0.0)
                down = np.where(diff < 0, -diff, 0.0)
                roll_up = pd.Series(up).rolling(rsi_win).mean()
                roll_down = pd.Series(down).rolling(rsi_win).mean()
                rs = roll_up / (roll_down + 1e-12)
                rsi = 100.0 - (100.0 / (1.0 + rs))
                ind["rsi"] = float(rsi.iloc[-1])

        return Ctx(
            now=pd.Timestamp.utcnow(),
            price=price,
            entry_price=avg_price,
            position_size=size,
            dca_level=self.state.dca_level,
            indicators=ind,
            equity_per_cycle=self.equity_per_cycle,
            config=self.cfg,
            position_pl_pct=pnl_pct,
            available_cash=float(self.broker.fetch_balance().get("USDT", {}).get("free", 0.0)),
            base_order_price=self.state.base_order_price,
            base_order_value=self.state.base_order_qty * (self.state.base_order_price or price)
                              if self.state.base_order_qty and self.state.base_order_price else None,
            base_order_quantity=self.state.base_order_qty,
            last_filled_price=self.state.last_filled_price
        )

    def apply_fills_to_state(self, intent: OrderIntent, fill_price: float, filled_qty: float) -> None:
        if intent.intent == IntentType.ENTRY and self.state.base_order_price is None:
            self.state.base_order_price = fill_price
            self.state.base_order_qty = filled_qty
            self.state.base_order_time = pd.Timestamp.utcnow().isoformat()
            self.state.last_filled_price = fill_price
        elif intent.intent == IntentType.SAFETY:
            self.state.dca_level += 1
            self.state.last_so_time = pd.Timestamp.utcnow().isoformat()
            self.state.last_filled_price = fill_price
        elif intent.intent == IntentType.EXIT:
            # reset cycle
            self.state = LiveState()
        self.state.save(self.cfg["persistence_path"])

    def execute_intents(self, intents: List[OrderIntent], price_now: float) -> None:
        for it in intents:
            qty = self.prec.amount(self.symbol, it.qty)
            params = {}
            if self.cfg.get("mode") == "futures" and it.reduce_only:
                params["reduceOnly"] = True
            order = self.broker.create_order(
                self.symbol, it.side.value, it.kind.value, qty,
                it.price if it.kind == "limit" else None, params
            )
            # Best-effort fill price extraction (market assumed immediate)
            fill_price = it.price or price_now
            self.apply_fills_to_state(it, fill_price, qty)

    def run(self) -> None:
        self.broker.load()
        # leverage/margin for futures
        if self.cfg.get("mode") == "futures":
            self.broker.ensure_leverage(self.symbol, int(self.cfg.get("leverage", 1)),
                                        self.cfg.get("margin_mode", "isolated"))

        last_closed_ts = None

        while True:
            try:
                ohlcv = self.broker.fetch_ohlcv(self.symbol, self.timeframe, limit=self.lookback)
                if not ohlcv:
                    time.sleep(self.poll_seconds)
                    continue
                df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
                df["ts"] = pd.to_datetime(df["ts"], unit="ms")
                df.set_index("ts", inplace=True)

                # proceed only when a new candle closed
                closed_ts = df.index[-1]
                price_now = float(df["close"].iloc[-1])
                if last_closed_ts is not None and closed_ts == last_closed_ts:
                    time.sleep(self.poll_seconds)
                    continue

                last_closed_ts = closed_ts

                ctx = self.build_ctx(df, price_now)
                intents = self.decider.next_intents(ctx, self.prec, self.symbol)

                if intents:
                    # basic risk checks and execution
                    self.execute_intents(intents, price_now)
            except Exception as e:
                print(f"[engine] error: {e}")
            time.sleep(self.poll_seconds)

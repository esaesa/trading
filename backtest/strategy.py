# strategy.py
from indicators_bootstrap import IndicatorsBootstrap
from simulation import apply_slippage
from wiring import wire_strategy
from instrumentation import log_loop_info, log_trade_info
from rich.console import Console
from datetime import datetime
from backtesting import Strategy
from config import strategy_params, backtest_params, optimization_params
from logger_config import logger
import numpy as np
import pandas as pd
from contracts import Ctx
from overlays import init_overlays, update_overlays
from types import SimpleNamespace

# New imports for refactored components
from strategy_config import StrategyConfig
from trade_processor import TradeProcessor
from state_manager import StateManager
from strategy_logger import StrategyLogger
from indicator_service import IndicatorService


class DCAStrategy(Strategy):
    # Initialize configuration
    config = StrategyConfig(
        so_cooldown_minutes=strategy_params.get("so_cooldown_minutes", 0),
        entry_fraction=strategy_params['entry_fraction'],
        so_size_multiplier=strategy_params['so_size_multiplier'],
        first_safety_order_multiplier=strategy_params['first_safety_order_multiplier'],
        safety_order_mode=strategy_params.get("safety_order_mode"),
        max_dca_levels=strategy_params['max_dca_levels'],
        take_profit_percentage=strategy_params['take_profit_percentage'],
        initial_deviation_percent=strategy_params['initial_deviation_percent'],
        price_multiplier=strategy_params['price_multiplier'],
        minimum_notional=strategy_params.get('minimum_notional', 0.0),
        rsi_threshold=strategy_params.get("rsi_threshold"),
        rsi_window=strategy_params.get("rsi_window"),
        require_rsi_reset=strategy_params.get("require_rsi_reset"),
        rsi_reset_percentage=strategy_params.get("rsi_reset_percentage"),
        rsi_dynamic_threshold=strategy_params.get("rsi_dynamic_threshold"),
        rsi_static_threshold_under=strategy_params.get("rsi_static_threshold_under"),
        rsi_percentile=strategy_params.get("rsi_percentile", 0.05),
        rsi_overbought_level=strategy_params.get("rsi_overbought_level"),
        rsi_oversold_level=strategy_params.get("rsi_oversold_level"),
        safety_order_price_mode=strategy_params.get("safety_order_price_mode"),
        start_trading_time=datetime.strptime(
            strategy_params.get("start_trading_time", "2025-02-01 00:00:00"),
            "%Y-%m-%d %H:%M:%S"
        ),
        take_profit_decay_grace_period_hours=strategy_params.get("take_profit_decay_grace_period_hours", 48),
        take_profit_decay_duration_hours=strategy_params.get("take_profit_decay_duration_hours", 24),
        atr_window=strategy_params.get("atr_window", 14),
        atr_resample_interval=strategy_params.get("atr_resample_interval", "h"),
        atr_deviation_threshold=strategy_params.get("atr_deviation_threshold", 5.0),
        atr_deviation_reduction_factor=strategy_params.get("atr_deviation_reduction_factor", 0.25),
        ema_window=strategy_params.get('ema_window', 200),
        ema_resample_interval=strategy_params.get('ema_resample_interval', "h"),
        rsi_resample_interval=strategy_params.get('rsi_resample_interval'),
        show_rsi=strategy_params.get("show_rsi", False),
        avoid_rsi_overbought=strategy_params.get("avoid_rsi_overbought"),
        show_indicators=strategy_params.get('show_indicators', {}),
        debug_backtest=backtest_params.get("debug", False),
        debug_loop=strategy_params["debug"]["loop"] and not backtest_params['enable_optimization'],
        debug_trade=strategy_params["debug"].get("trade", False) and not backtest_params['enable_optimization'],
        debug_process=strategy_params["debug"].get("process", False) and not backtest_params['enable_optimization'],
        slippage_probability=backtest_params.get("slippage_probability")
    )

    def __init__(self, broker, data, params):
        super().__init__(broker, data, params)
        self.enable_optimization = backtest_params.get("enable_optimization", False)

        # Initialize refactored components
        self.config = self.__class__.config.init_from_params(strategy_params)
        self.trade_processor = TradeProcessor(self)
        self.state_manager = StateManager(self)
        self.strategy_logger = StrategyLogger(self)
        self.indicator_service = IndicatorService(self)


    def _ctx(self) -> Ctx:
        now = self.data.index[-1]
        price = float(self.data.Close[-1])

        # Dynamic RSI threshold and ATR are now fetched on-demand by rules via indicator_service

        entry_price = getattr(self, "base_order_price", None)
        dca_level = getattr(self, "dca_level", 0) if self.position else 0

        cfg = {
            "last_so_dt": getattr(self, "last_safety_order_time", None),
            "base_order_time": getattr(self, "base_order_time", None),
            "safety_order_mode": self.config.safety_order_mode,
            "take_profit": self.config.take_profit_percentage,
            "next_level": dca_level + 1,
            "commission_rate": self.commission_rate,
        }

        equity_per_cycle = float(self._broker._cash)

        ctx = Ctx(
            entry_price=entry_price,
            equity_per_cycle=equity_per_cycle,
            config=cfg,
            last_entry_time=(self.last_safety_order_time or self.base_order_time),
            now=now,
            price=price,
            dca_level=getattr(self, 'dca_level', 0),
            base_order_price=getattr(self, 'base_order_price', None),
            base_order_value=getattr(self, 'base_order_value', None),
            base_order_quantity=getattr(self, 'base_order_quantity', None),
            last_filled_price=getattr(self, 'last_filled_price', None),
            last_so_dt=getattr(self, 'last_safety_order_time', None),
            base_order_time=getattr(self, 'base_order_time', None),
            available_cash=self._broker.margin_available * self._broker._leverage *.99,
            position_size=self.position.size if self.position else 0,
            position_pl_pct=self.position.pl_pct if self.position else 0,
        )

        # Pre-compute next SO trigger (guarded)
        try:
            next_level = cfg.get("next_level", ctx.dca_level + 1)
            if self.position and hasattr(self, "price_engine") and next_level <= self.config.max_dca_levels:
                ctx.next_so_price = self.price_engine.so_price(ctx, next_level)
        except Exception:
            ctx.next_so_price = None

        return ctx

 
    @property
    def commission_rate(self) -> float:
        """
        Read-only view of the commission rate coming from the wired calculator.
        Falls back to config before wiring occurs.
        """
        calc = getattr(self, "commission_calc", None)
        if calc is not None and hasattr(calc, "rate"):
            try:
                return float(calc.rate)
            except Exception:
                pass
        # Fallback before wiring happens
        return float(backtest_params.get("commission", 0.0))

    def unscale_parameter(self, key, value):
        if self.enable_optimization and optimization_params[key]['type'] == 'float':
            param_scale = optimization_params[key].get('scaling_factor')
            if param_scale is not None:
                return value / param_scale
        return value

    
    def init(self):
        debug_any = (self.config.debug_backtest or self.config.debug_loop or self.config.debug_trade or self.config.debug_process)
        self.console = Console(record=debug_any, no_color=True)

        if not isinstance(self.data.index, pd.DatetimeIndex):
            self.data.index = pd.to_datetime(self.data.index)

        # Apply scaling if optimization is enabled
        if self.enable_optimization:
            for key in self.__dict__:
                if key in optimization_params:
                    setattr(self, key, self.unscale_parameter(key, getattr(self, key)))

        # Internal state
        self.completed_processes = []
        self.process_metrics = {}
        self.dca_level = 0
        self.base_order_quantity = None
        self.base_order_price = None
        self.initial_investment = 0
        self.last_filled_price = None
        self.base_order_value = None
        self.on_trade_success_callbacks = []

        # Time tracking
        self.base_order_time = None
        self.last_safety_order_time = None

        init_overlays(self)
        # Wire rules/engines/providers first (creates deciders needed for indicator detection)
        wire_strategy(self, strategy_params)
        # Initialize indicators via bootstrap (now deciders exist for dynamic detection)
        IndicatorsBootstrap(strategy_params).run(self)
        self.completed_processes = []
        # NEW: minimal runtime buffer
        self._cycle = None
        # in DCAStrategy.init() after other state inits
        self._cycle_cash_entry = 0.0
        self._cycle_invested = 0.0
        self._cycle_peak_invested = 0.0
        self.cycle_cash_records = []  

        
        debug_any = (self.config.debug_backtest or self.config.debug_loop or self.config.debug_trade or self.config.debug_process)
        self.console = Console(record=debug_any, no_color=True)

        if not isinstance(self.data.index, pd.DatetimeIndex):
            self.data.index = pd.to_datetime(self.data.index)

        # Apply scaling if optimization is enabled
        if self.enable_optimization:
            for key in self.__dict__:
                if key in optimization_params:
                    setattr(self, key, self.unscale_parameter(key, getattr(self, key)))

        # Internal state
        self.completed_processes = []
        self.process_metrics = {}
        self.dca_level = 0
        self.base_order_quantity = None
        self.base_order_price = None
        self.initial_investment = 0
        self.last_filled_price = None
        self.base_order_value = None
        self.on_trade_success_callbacks = []

        # Time tracking
        self.base_order_time = None
        self.last_safety_order_time = None

        init_overlays(self)
        # Wire rules/engines/providers first (creates deciders needed for indicator detection)
        wire_strategy(self, strategy_params)
        # Initialize indicators via bootstrap (now deciders exist for dynamic detection)
        IndicatorsBootstrap(strategy_params).run(self)
        self.completed_processes = []
        # NEW: minimal runtime buffer
        self._cycle = None
        # in DCAStrategy.init() after other state inits
        self._cycle_cash_entry = 0.0
        self._cycle_invested = 0.0
        self._cycle_peak_invested = 0.0
        self.cycle_cash_records = []  

    def add_on_trade_success_callback(self, callback):
        self.on_trade_success_callbacks.append(callback)

    def commit_trade_state(self):
        for callback in self.on_trade_success_callbacks:
            callback()
        self.on_trade_success_callbacks = []



    def add_on_trade_success_callback(self, callback):
        self.on_trade_success_callbacks.append(callback)

    def commit_trade_state(self):
        for callback in self.on_trade_success_callbacks:
            callback()
        self.on_trade_success_callbacks = []


    def get_entry_price(self) -> float:
        return self.position_view.avg_entry_price()

    def next(self):
        current_time = self.data.index[-1]
        if current_time < self.config.start_trading_time:
            return

        update_overlays(self, current_time)

        price = self.data.Close[-1]
        low   = self.data.Low[-1]
        high  = self.data.High[-1]
        if self.config.slippage_probability:
            price, low, high = apply_slippage(price, low, high, self.config.slippage_probability)

        # Build once
        ctx = self._ctx()

        self.strategy_logger.log_loop_info(ctx)

        if self.position:
            if self.trade_processor.process_exit(ctx, high, current_time):
                return
            self.trade_processor.process_dca(ctx, low, current_time)
        else:
            entry_ok, entry_reason = self.entry_decider.ok(ctx)
            if not entry_ok:
                return
            self.trade_processor.process_entry(ctx, price, current_time)
            return True

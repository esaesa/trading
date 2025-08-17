# strategy.py
from indicators_bootstrap import IndicatorsBootstrap
from reset_policy import need_reset, update_reset, consume_reset
from simulation import apply_slippage
from reporting import render_cycle_plan_table, render_exit_summary_table
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


class DCAStrategy(Strategy):
    # Strategy Parameters
    so_cooldown_minutes = strategy_params.get("so_cooldown_minutes", 0)
    entry_fraction = strategy_params['entry_fraction']
    so_size_multiplier = strategy_params['so_size_multiplier']
    first_safety_order_multiplier = strategy_params['first_safety_order_multiplier']
    safety_order_mode = strategy_params.get("safety_order_mode")
    max_dca_levels = strategy_params['max_dca_levels']
    take_profit_percentage = strategy_params['take_profit_percentage']
    initial_deviation_percent = strategy_params['initial_deviation_percent']
    price_multiplier = strategy_params['price_multiplier']
    minimum_notional = strategy_params.get('minimum_notional', 0.0)
    rsi_threshold = strategy_params.get("rsi_threshold")
    rsi_window = strategy_params.get("rsi_window")
    require_rsi_reset = strategy_params.get("require_rsi_reset")
    rsi_reset_percentage = strategy_params.get("rsi_reset_percentage")
    rsi_dynamic_threshold = strategy_params.get("rsi_dynamic_threshold")

    rsi_percentile = strategy_params.get("rsi_percentile", 0.05)
    rsi_overbought_level = strategy_params.get("rsi_overbought_level")
    rsi_oversold_level = strategy_params.get("rsi_oversold_level")
    safety_order_price_mode = strategy_params.get("safety_order_price_mode")
    start_trading_time = datetime.strptime(
        strategy_params.get("start_trading_time", "2025-02-01 00:00:00"),
        "%Y-%m-%d %H:%M:%S"
    )

    # Duration over which TP reduces to zero (in hours)
    take_profit_reduction_duration_hours = strategy_params.get("take_profit_reduction_duration_hours", 24 * 1)

    # ATR Parameters
    enable_atr_calculation = strategy_params.get("enable_atr_calculation", False)
    atr_window = strategy_params.get("atr_window", 14)
    atr_resample_interval = strategy_params.get("atr_resample_interval", "h")
    atr_deviation_threshold = strategy_params.get("atr_deviation_threshold", 5.0)
    atr_deviation_reduction_factor = strategy_params.get("atr_deviation_reduction_factor", 0.25)

    # EMA Parameters
    enable_ema_calculation = strategy_params.get("enable_ema_calculation", False)
    ema_window = strategy_params.get('ema_window', 200)
    ema_resample_interval = strategy_params.get('ema_resample_interval', "h")

    # RSI / indicators
    enable_rsi_calculation = strategy_params.get("enable_rsi_calculation", False)
    rsi_resample_interval = strategy_params.get('rsi_resample_interval')
    show_rsi = strategy_params.get("show_rsi", False)
    avoid_rsi_overbought = strategy_params.get("avoid_rsi_overbought")
    enable_cv_calculation = strategy_params.get("enable_cv_calculation", False)
    enable_laguere_calculation = strategy_params.get("enable_laguere_calculation", False)
    show_indicators = strategy_params.get('show_indicators', {})

    # Backtest Parameters
    debug_backtest = backtest_params.get("debug", False)
    debug_loop = strategy_params["debug"]["loop"] and not backtest_params['enable_optimization']
    debug_trade = strategy_params["debug"].get("trade", False) and not backtest_params['enable_optimization']
    debug_process = strategy_params["debug"].get("process", False) and not backtest_params['enable_optimization']
    slippage_probability = backtest_params.get("slippage_probability")

    def __init__(self, broker, data, params):
        super().__init__(broker, data, params)
        self.enable_optimization = backtest_params.get("enable_optimization", False)

    def _ind_value(self, name: str, now, default=np.nan):
        """
        Read indicator 'name' at time 'now' via indicator_provider if available,
        otherwise fall back to legacy series on the Strategy.
        Supported names: rsi, ema, atr_pct, laguerre_rsi, bbw, cv
        """
        # 1) Try provider (preferred)
        provider = getattr(self, "indicator_provider", None)
        if provider is not None:
            # Expect methods named exactly like the indicator
            fn = getattr(provider, name, None)
            if callable(fn):
                try:
                    v = fn(now)
                    if v is not None:
                        return float(v)
                except Exception:
                    pass  # fall back

        # 2) Fallback to legacy series on the Strategy
        series_attr = {
            "rsi": "rsi_values",
            "ema": "ema_dynamic",
            "atr_pct": "atr_pct_series",
            "laguerre_rsi": "laguerre_series",
            "bbw": "bbw_series",
            "cv": "cv_series",
        }.get(name)

        if series_attr and hasattr(self, series_attr):
            ser = getattr(self, series_attr)
            if ser is not None and len(ser):
                try:
                    return float(ser.loc[now])
                except Exception:
                    try:
                        return float(ser.iloc[-1])
                    except Exception:
                        pass

        return default

    
    def _ctx(self) -> Ctx:
        now = self.data.index[-1]
        price = float(self.data.Close[-1])

        indicators = {}
        for n in ("rsi", "ema", "atr_pct", "laguerre_rsi", "bbw", "cv"):
            v = self._ind_value(n, now, default=np.nan)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                indicators[n] = v

        dyn_thr = None
        if getattr(self, "rsi_dynamic_threshold", False) and hasattr(self, "rsi_dynamic_threshold_series"):
            try:
                dyn_thr = float(self.rsi_dynamic_threshold_series.loc[now])
            except Exception:
                dyn_thr = float(self.rsi_dynamic_threshold_series.iloc[-1])

        current_atr = None
        if self.enable_atr_calculation and hasattr(self, 'atr_pct_series') and not self.atr_pct_series.empty:
            current_atr = self.get_current_atr(now)

        entry_price = getattr(self, "base_order_price", None)
        dca_level = getattr(self, "dca_level", 0) if self.position else 0

        cfg = {
            "last_so_dt": getattr(self, "last_safety_order_time", None),
            "base_order_time": getattr(self, "base_order_time", None),
            "safety_order_mode": self.safety_order_mode,
            "take_profit": self.take_profit_percentage,
            "next_level": dca_level + 1,
            "commission_rate" : self.commission_rate
        }

        equity_per_cycle = float(self._broker._cash)
        last_entry_time = self.last_safety_order_time if self.last_safety_order_time else self.base_order_time

        ctx = Ctx(
            entry_price=entry_price,
            equity_per_cycle=equity_per_cycle,
            config=cfg,
            last_entry_time=last_entry_time,
            now=now,
            price=price,
            indicators=indicators,
            # Position tracking
            dca_level=getattr(self, 'dca_level', 0),
            base_order_price=getattr(self, 'base_order_price', None),
            base_order_value=getattr(self, 'base_order_value', None),
            base_order_quantity=getattr(self, 'base_order_quantity', None),
            last_filled_price=getattr(self, 'last_filled_price', None),
            # Time tracking
            last_so_dt=getattr(self, 'last_safety_order_time', None),
            base_order_time=getattr(self, 'base_order_time', None),
            # Dynamic values
            dynamic_rsi_thr=(self.rsi_dynamic_threshold_series.loc[now] 
                           if hasattr(self, 'rsi_dynamic_threshold_series') 
                           else self.rsi_threshold),
            # Broker data
            available_cash=self._broker.margin_available * self._broker._leverage,
            position_size=self.position.size if self.position else 0,
            position_pl_pct=self.position.pl_pct if self.position else 0,
            current_atr=current_atr,   
        )

        # Pre-compute next SO trigger so rules can pre-flight funds/notional
        try:
            next_level = cfg.get("next_level", ctx.dca_level + 1)
            if self.position and hasattr(self, "price_engine") and next_level <= self.max_dca_levels:
                ctx.next_so_price = self.price_engine.so_price(ctx, next_level)
        except Exception:
            # be lenient if not computable
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

    def get_current_atr(self, current_time):
        """Fetch the current ATR% via provider first, then legacy series."""
        v = self._ind_value("atr_pct", current_time, default=np.nan)
        return v
    
    def init(self):
        self.console = Console(record=True)

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
        self.rsi_reset = True
        self.last_filled_price = None
        self.base_order_value = None
        self.rsi_values = None

        # Time tracking
        self.base_order_time = None
        self.last_safety_order_time = None
        # Compute indicators (this sets self.rsi_values, ema_dynamic, atr_pct_series, etc.)
        IndicatorsBootstrap(strategy_params).run(self)
        init_overlays(self)
        # Wire rules/engines/providers (provider will read the series computed above)
        wire_strategy(self, strategy_params)  
        assert (hasattr(self, "entry_sizer")
                and hasattr(self, "affordability_guard")
                and hasattr(self, "commission_calc")
                and hasattr(self, "indicator_provider")
                and hasattr(self, "entry_preflight")), \
            "Wiring failed: missing one of entry_sizer, affordability_guard, commission_calc, indicator_provider, entry_preflight"


    def process_entry(self, price: float, current_time):
        ctx = self._ctx()
        if not self.position and self.entry_decider.ok(ctx):
            rsi_now = self._ind_value("rsi", current_time, default=np.nan)
            if self.enable_rsi_calculation and (rsi_now is None or np.isnan(rsi_now)):
                return False
            # use injected entry_sizer (EMA-aware via EntryMultiplier)
            qty, investment = self.entry_sizer.qty_and_investment(ctx, price, self.commission_calc)


            # keep fractional guard for safety if someone removes EntryFundsAndNotional rule
            if qty < 1:
                raise ValueError("Backtesting.py does not support fractional units. Adjust your entry fraction.")
            order = self.buy(size=qty, tag="BO")
            self.base_order_time = current_time
            self.last_safety_order_time = None
            self.dca_level = 0
            self.base_order_price = price
            self.initial_investment = investment
            self.last_filled_price = price
            self.base_order_quantity = qty
            self.base_order_value = qty * self.base_order_price

            consume_reset(self)

            if self.debug_trade:
                table = render_cycle_plan_table(self, qty)
                logger.info(table)
                log_trade_info(self, "Entry", order, price, qty * price)

            return True
        return False

    def process_dca(self, price: float, current_time: datetime) -> None:
        ctx = self._ctx()
        if not self.safety_decider.ok(ctx):
            return
        level = self.dca_level + 1
        so_price = self.price_engine.so_price(ctx, level)
        if price > so_price:
            if self.debug_trade:
                logger.debug(f"Price {price:.10f} > SO-{level} trigger {so_price:.10f}")
            return
        so_size = self.size_engine.so_size(ctx, so_price, level)
        size_to_buy = self.affordability_guard.clamp_qty(
            desired_qty=so_size,
            price=so_price,
            available_cash=self._broker.margin_available * self._broker._leverage,
            commission=self.commission_calc,   # <-- pass the calculator, not a float
            min_notional=self.minimum_notional,
        )
        order = None
        if size_to_buy > 0:
            order = self.buy(size=size_to_buy, tag=f"S{level}")

        self.last_safety_order_time = current_time
        self.dca_level += 1
        consume_reset(self)
        if self.safety_order_price_mode.lower() == "dynamic":
            self.last_filled_price = price
        if self.debug_trade and order is not None:
            log_trade_info(self, "DCA", order, so_price, size_to_buy * so_price)

    def process_exit(self, price, current_time):
        if not self.position:
            return False
        ctx = self._ctx()
        if self.exit_decider.ok(ctx):
            if self.debug_process:
                table = render_exit_summary_table(self, price, current_time)
                logger.info(table)
                logger.debug(f"Trade: Exit Triggered at {current_time} with Price: {price:.10f}")
            self.position.close()
            self.reset_process()
            return True
        else:
            return False

    def reset_process(self):
        self.base_order_price = None
        self.base_order_quantity = None
        self.base_order_value = None
        self.dca_level = 0
        self.last_filled_price = None
        self.rsi_reset = True
        self.base_order_time = None
        self.last_safety_order_time = None

    def get_entry_price(self) -> float:
        return self.position_view.avg_entry_price()

    def next(self):
        current_time = self.data.index[-1]
        if current_time < self.start_trading_time:
            return
        update_overlays(self, current_time)
        price = self.data.Close[-1]
        current_low = self.data.Low[-1]
        current_high = self.data.High[-1]
        price, current_low, current_high = apply_slippage(price, current_low, current_high, self.slippage_probability)
        rsi_val = self._ind_value("rsi", current_time, default=np.nan)
        update_reset(self, rsi_val)
        log_loop_info(self, price, rsi_val, current_time)

        if self.position:
            if self.process_exit(current_high, current_time):
                return
            else:
                if not self.enable_rsi_calculation:
                    rsi_val = np.nan
                self.process_dca(current_low, current_time)
        else:
            ctx = self._ctx()
            if not self.entry_decider.ok(ctx):
                return
            self.process_entry(price, current_time)
            return True

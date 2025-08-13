# strategy.py

from sizers import DCASizingParams, GeometricOrderSizer 
from rich.console import Console
from datetime import datetime
import math
from backtesting import Strategy
from config import strategy_params, backtest_params, optimization_params
from rich.table import Table
from logger_config import logger
import numpy as np
import pandas as pd
from indicator_manager import IndicatorManager
from rule_chain import  build_rule_chain
from rules.entry import ENTRY_RULES
from rules.safety import SAFETY_RULES
from rules.exit import EXIT_RULES, calculate_decaying_tp
from contracts import Ctx




class DCAStrategy(Strategy):
    # Strategy Parameters
    # Optional: minimum minutes between safety orders (off when 0)
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
    take_profit_reduction_duration_hours = strategy_params.get("take_profit_reduction_duration_hours", 24*1) 
   
    
    # ATR Parameters
    enable_atr_calculation = strategy_params.get("enable_atr_calculation", False)
    atr_window = strategy_params.get("atr_window", 14)
    atr_resample_interval = strategy_params.get("atr_resample_interval", "h")
    
    atr_deviation_threshold = strategy_params.get("atr_deviation_threshold", 5.0) # Assuming ATR as percentage (e.g., 6%)
    atr_deviation_reduction_factor = strategy_params.get("atr_deviation_reduction_factor", 0.25) # Reduce deviation by 50%
    
    
    # EMA Parameters
    enable_ema_calculation = strategy_params.get("enable_ema_calculation", False)
    ema_window = strategy_params.get('ema_window', 200)
    ema_resample_interval = strategy_params.get('ema_resample_interval', "h")
    
    # RSI Parameters
    enable_rsi_calculation = strategy_params.get("enable_rsi_calculation", False)
    rsi_resample_interval = strategy_params.get('rsi_resample_interval')
    show_rsi = strategy_params.get("show_rsi", False)
    avoid_rsi_overbought = strategy_params.get("avoid_rsi_overbought")
    
    # CV Parameters
    enable_cv_calculation = strategy_params.get("enable_cv_calculation", False)
    
    # Laguerre RSI Parameters
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
        

    def _ctx(self) -> Ctx:
        now = self.data.index[-1]
        price = float(self.data.Close[-1])

        indicators = {}
        if getattr(self, "rsi_values", None) is not None and len(self.rsi_values):
            indicators["rsi"] = float(self.rsi_values.iloc[-1])
        if getattr(self, "ema_dynamic", None) is not None and len(self.ema_dynamic):
            indicators["ema"] = float(self.ema_dynamic.iloc[-1])
        if getattr(self, "atr_pct_series", None) is not None and len(self.atr_pct_series):
            indicators["atr_pct"] = float(self.atr_pct_series.iloc[-1])
        if getattr(self, "laguerre_series", None) is not None and len(self.laguerre_series):
            indicators["laguerre_rsi"] = float(self.laguerre_series.iloc[-1])
        if getattr(self, "bbw_series", None) is not None and len(self.bbw_series):
            indicators["bbw"] = float(self.bbw_series.iloc[-1])
        if getattr(self, "cv_series", None) is not None and len(self.cv_series):
            indicators["cv"] = float(self.cv_series.iloc[-1])

        # latest dynamic RSI threshold if enabled
        dyn_thr = None
        if getattr(self, "rsi_dynamic_threshold", False) and hasattr(self, "rsi_dynamic_threshold_series"):
            try:
                dyn_thr = float(self.rsi_dynamic_threshold_series.loc[now])
            except Exception:
                dyn_thr = float(self.rsi_dynamic_threshold_series.iloc[-1])
                
        # Get current ATR value
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
            # helpful for safety rules: which SO is pending if we fire one now
            "next_level": dca_level + 1,
            "commission_rate" : backtest_params.get("commission", 0.0)
        }

        # cash baseline for this cycle and available cash with leverage
        equity_per_cycle = float(self._broker._cash)
        last_entry_time = self.last_safety_order_time if self.last_safety_order_time else self.base_order_time
        return Ctx(
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
             current_atr=current_atr
        )


    def unscale_parameter(self, key, value):
        if self.enable_optimization and optimization_params[key]['type'] == 'float':
            param_scale = optimization_params[key].get('scaling_factor')
            if param_scale is not None:
                return value / param_scale
        return value
    def get_current_atr(self, current_time):
        """Fetches the current ATR value, or NaN if not available."""
        if self.atr_pct_series is not None and not self.atr_pct_series.empty:
            try:
                return self.atr_pct_series.loc[current_time]
            except KeyError:
                return self.atr_pct_series.iloc[-1] # Fallback to positional if time not exact
        return np.nan
    
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
        self.so_prices = []
        self.so_sizes = []
        self.rsi_reset = True
        self.last_filled_price = None
        self.base_order_value = None
        self.rsi_values = None
        
        # Initialize time tracking variables
        self.base_order_time = None
        self.last_safety_order_time = None
        
        # ----------------------
        # Initialize Indicators
        # ----------------------        
        
        # Resolve dynamic rsi window safely (class-level default could have None)
        # rsi_dynamic_window at run time as it may be based on rsi_window
        if strategy_params.get("rsi_dynamic_window") is not None:
            self.rsi_dynamic_window = strategy_params["rsi_dynamic_window"]
        else:
            # safe fallback: if rsi_window missing, use 1000
            self.rsi_dynamic_window = (self.rsi_window or 14) * 1000
            
        # Determine calculation needs (show OR strategy use)
        calculate_rsi = self.show_indicators.get('rsi', False) or self.enable_rsi_calculation
        calculate_atr = self.show_indicators.get('atr', False) or self.enable_atr_calculation
        calculate_ema = self.show_indicators.get('ema', False) or self.enable_ema_calculation
        calculate_laguerre = self.show_indicators.get('laguerre', False) or self.enable_laguere_calculation
        calculate_bbw = self.show_indicators.get('bbw', False)   
        calculate_cv = self.show_indicators.get('cv', False)
        calculate_dynamic_rsi = self.show_indicators.get('dynamic_rsi', False) or self.rsi_dynamic_threshold
        
        # --- Dependencies ---
        # If we need dynamic RSI, RSI reset, or overbought-rule, we must have RSI computed.
        if calculate_dynamic_rsi or self.require_rsi_reset or self.avoid_rsi_overbought:
            calculate_rsi = True
         
        df = self.data.df

       
        self.indicators = IndicatorManager(df)
        
        p = strategy_params
        self.order_sizer = GeometricOrderSizer(DCASizingParams(
            entry_fraction=p['entry_fraction'],
            first_so_multiplier=p['first_safety_order_multiplier'],
            so_size_multiplier=p['so_size_multiplier'],
            min_notional=p.get('minimum_notional', 0.0),
            max_levels=p['max_dca_levels'],
        ))

        if calculate_rsi:
                self.rsi_values = self.indicators.compute_rsi(
                    window=self.rsi_window,
            resample_interval=self.rsi_resample_interval
                )
                if self.show_indicators.get('rsi', False):
                    self.I(lambda _: self.rsi_values, self.data.Close, name="RSI")
                    
        # Compute dynamic RSI threshold if enabled and RSI exists       
        if calculate_dynamic_rsi and (self.rsi_values is not None):
            self.rsi_dynamic_threshold_series = (
                self.rsi_values.rolling(window=self.rsi_dynamic_window)
                .quantile(self.rsi_percentile)
            )
            if self.show_indicators.get('dynamic_rsi', True):
                self.I(
                    lambda _: self.rsi_dynamic_threshold_series,
                    self.data.Close,
                    name="Dynamic RSI Threshold"
                )
                    
        if calculate_ema:
            ema_series = self.indicators.compute_ema(window=self.ema_window, resample_interval='1h')
            self.ema_dynamic = ema_series
            if self.show_indicators.get('ema', False):
                self.I(lambda _: ema_series, self.data.Close, name="EMA")

        if calculate_atr:
            self.atr_pct_series = self.indicators.compute_atr_percentage(window=14, resample_interval='1D')
            if self.show_indicators.get('atr', False):
                self.I(lambda _: self.atr_pct_series, self.data.Close, name="ATR%")
    

        if calculate_cv:
            cv_series = self.indicators.compute_coefficient_of_variation(window=40)
            if self.show_indicators.get('cv', False):
                self.I(lambda _: cv_series, self.data.Close, name="CV")
        
        if calculate_laguerre:
            laguerre_series = self.indicators.compute_laguerre_rsi(gamma=0.1, resample_interval='4h')
            if self.show_indicators.get('laguerre', False):
                self.I(lambda _: laguerre_series, self.data.Close, name="Laguerre RSI")
        
        if calculate_bbw:
            bbw_series = self.indicators.compute_bbw(window=20, num_std=2)
            if self.show_indicators.get('bbw', False):
                self.I(lambda _: bbw_series, self.data.Close, name="BBW")
            
        if self.debug_backtest or self.show_indicators.get('average_entry_price', True):    
            self.breakeven_prices = np.full(len(self.data), np.nan)
            self.breakeven_indicator = self.I(lambda: self.breakeven_prices, name='Average Entry Price', overlay=True)

        self.take_profit_prices = np.full(len(self.data), np.nan)
        self.tp_indicator = self.I(lambda: self.take_profit_prices, name='Take Profit', overlay=True, color='red')

        # --- NEW: data-driven pipelines from strategy_params ---
        default_entry = ["RSIOverboughtGate"] if self.avoid_rsi_overbought else []
        default_safety = ["RSIUnderDynamicThreshold"]  # same as before
        default_exit = ["TPDecayReached"]              # same as before

        entry_names = strategy_params.get("entry_conditions", default_entry)
        safety_names = strategy_params.get("safety_conditions", default_safety)
        exit_names = strategy_params.get("exit_conditions", default_exit)

        self._entry_rules = build_rule_chain(self, entry_names, ENTRY_RULES,mode="any")
        self._safety_rules = build_rule_chain(self, safety_names, SAFETY_RULES,mode="any")
        self._exit_rules = build_rule_chain(self, exit_names, EXIT_RULES, mode="any")
        


    # --- Debug Helper Functions ---
    def debug_loop_info(self, price, rsi_val, current_time):
        """Log key loop details."""
        if self.debug_loop:
            logger.debug(f"[Loop Debug] Time: {current_time}")
            logger.debug(f"[Loop Debug] Current Price: {price}")
            logger.debug(f"[Loop Debug] Position Size: {self.position.size}")
            logger.debug(f"[Loop Debug] Position PnL%: {self.position.pl_pct}")
            logger.debug(f"[Loop Debug] RSI Value: {rsi_val}")
            logger.debug(f"[Loop Debug] RSI Reset: {self.rsi_reset}")
            # Log time tracking info
            if self.position:
                time_since_base = current_time - self.base_order_time if self.base_order_time else "N/A"
                time_since_so = current_time - self.last_safety_order_time if self.last_safety_order_time else "N/A"
                logger.debug(f"[Loop Debug] Time Since BO: {time_since_base} | Time Since Last SO: {time_since_so}")


    def debug_trade_info(self, trade_type, order, computed_trigger, order_value):
        executed_price = self._broker.last_price
        current_time = self.data.index[-1]     # current time from the data index
        logger.debug(
            f"{trade_type} at {current_time} | "
            f"Tag: {order.tag} | Order Size: {order.size} | "
            f"Computed Trigger Price: {computed_trigger:.10f} | "
            f"Executed Price: {executed_price:.10f} | "
            f"Order Value: {order_value:.2f}$"
        )

    def dynamic_safety_price(self, last_filled_price, order_num, current_atr_value=None) -> float:
        
        if self.enable_atr_calculation and not np.isnan(current_atr_value) and current_atr_value < self.atr_deviation_threshold:
            effective_price_multiplier = self.price_multiplier * (1 - self.atr_deviation_reduction_factor)
        else:
            effective_price_multiplier = self.price_multiplier
                   
        deviation = self.initial_deviation_percent * math.pow(effective_price_multiplier, order_num)
        if deviation >= 100:
            if self.debug_loop:
                logger.warning(f"Deviation for level {order_num} is greater than 100%. Returning minimum value.")
            return 1e-8
        computed_price = last_filled_price * ((100 - deviation) / 100)
        return computed_price
    
    def calculate_safety_order_qty(self, so_price, multiplier, level):
        if self.safety_order_mode.lower() == "value":
            if level == 1:
                safety_order_value = self.base_order_value * self.first_safety_order_multiplier
            elif level > 1:
                safety_order_value = self.base_order_value * self.first_safety_order_multiplier * (self.so_size_multiplier ** (level - 1))
            return safety_order_value / so_price
        elif self.safety_order_mode.lower() == "volume":
            if level == 1:
                return self.base_order_quantity * self.first_safety_order_multiplier
            elif level > 1:
                return self.base_order_quantity  * self.first_safety_order_multiplier * (self.so_size_multiplier ** (level - 1))
            

    def static_safety_price(self, base_price: float, target_level: int) -> float:
        if target_level < 1 or target_level > self.max_dca_levels:
            return 0.0

        # CHANGED: Compute factors as a list and use math.prod for efficiency.
        factors = [1 - (self.initial_deviation_percent / 100) * (self.price_multiplier ** i)
                for i in range(target_level)]
        if any(f <= 0 for f in factors):
            return 1e-8
        compounded_factor = math.prod(factors)
        # END CHANGED

        return base_price * compounded_factor
    
    def _safety_allows(self, rsi_val: float, dynamic_thr: float, level: int) -> bool:
        """
        Returns True if safety order is allowed by safety rules.
        (Currently only the RSI-under-dynamic-threshold rule; identical logic to old inline code.)
        """
        if (np.isnan(rsi_val) or rsi_val < dynamic_thr):
            return True
        # identical debug message as before
        if self.debug_trade:
            logger.debug(f"RSI={rsi_val:.2f} not below dynamic threshold={dynamic_thr:.2f}, skipping DCA-{level}")
        return False
    
    def _entry_allows(self, rsi_val: float) -> bool:
        """
        Returns True if base order is allowed by all entry rules.
        (Currently only the RSI-overbought rule; identical logic to the old inline code.)
        """
        if (
            self.avoid_rsi_overbought
            and self.enable_rsi_calculation
            and not np.isnan(rsi_val)
            and rsi_val >= (self.rsi_overbought_level or 70)
        ):
            if self.debug_loop:
                logger.debug(
                    f"Skip BO – RSI={rsi_val:.2f} ≥ thr={(self.rsi_overbought_level or 70):.2f}"
                )
            return False
        return True

    def _safety_affordable_size(self, so_price: float, desired_size: float) -> int:
        """
        Returns actual executable size (>=1) after cash & min-notional checks.
        0 means 'do nothing'. Logs & warnings match the old behavior.
        """
        commission_rate = backtest_params.get("commission", 0.0)
        available_cash = self._broker.margin_available * self._broker._leverage
        cost_per_share = so_price * (1 + commission_rate)

        max_possible = int(available_cash // cost_per_share) if cost_per_share > 0 else 0
        if max_possible < 1:
            if self.debug_trade:
                logger.warning(
                    f"Insufficient funds for DCA-{self.dca_level + 1}. "
                    f"Available cash: {available_cash:.2f}, cost per share: {cost_per_share:.2f}"
                )
            return 0

        desired_floor = math.floor(desired_size)
        if desired_floor > max_possible:
            print('WARNING: order will be reduced due to insufficient funds')

        size_to_buy = max(1, min(desired_floor, max_possible))

        # Min-notional check uses net (price*qty) same as before
        if size_to_buy * so_price < self.minimum_notional:
            if self.debug_trade:
                logger.warning(
                    f"DCA-{self.dca_level} order below minimum notional, skipping DCA level."
                )
            return 0

        return size_to_buy

    def _entry_multiplier(self, price: float) -> int:
        """EMA-aware entry multiplier (identical to your inline logic)."""
        return 2 if (
            self.enable_ema_calculation and
            getattr(self, "ema_dynamic", None) is not None and
            price > self.ema_dynamic.iloc[-1]
        ) else 1

    def _entry_size_and_investment(self, price: float, multiplier: int):
        """Compute base-order quantity (int) and initial_investment (float)."""
        commission_rate = backtest_params.get("commission", 0.0)
        initial_investment = self._broker._cash * (self.entry_fraction * multiplier)
        price_gross = price * (1 + commission_rate)
        base_order_quantity = math.floor(initial_investment / price_gross)
        return base_order_quantity, initial_investment

    def _entry_affordable_ok(self, qty: int, price: float) -> bool:
        """
        Keep exact original order:
        1) min-notional check (warn & skip if fail)
        2) qty >= 1 (raise ValueError if fail)
        """
        if qty * price < self.minimum_notional:
            if self.debug_trade:
                logger.warning("Initial order below minimum notional, skipping entry.")
            return False
        if qty < 1:
            raise ValueError("Backtesting.py does not support fractional units. Adjust your entry fraction.")
        return True

 

 

    def process_entry(self, price: float, current_time):
        ctx = self._ctx()
        if not self.position and  self._should_enter_position(ctx):
            if self.enable_rsi_calculation and self.rsi_values is None:
                return False
            multiplier = 2 if (self.enable_ema_calculation and price > ctx.indicators.get('ema', 0)) else 1
            qty, investment = self._entry_size_and_investment(price, multiplier)
            if not self._entry_affordable_ok(qty, price):
                return False
            order = self.buy(size=qty, tag="BO")
            self.base_order_time = current_time
            self.last_safety_order_time = None
            self.dca_level = 0
            self.base_order_price = price
            self.initial_investment = investment
            self.last_filled_price = price
            self.base_order_quantity = qty
            self.base_order_value = qty * self.base_order_price

            if self.debug_trade:
                self.log_entry_details(current_time, price, qty)
                self.debug_trade_info("Entry", order, price, qty * price)

            self._consume_rsi_reset()
            return True
        return False

    def log_entry_details(self, current_time, price, base_order_quantity):
        """
        Logs information at the moment of initial position opening.

        Changes:
        1) Checks if 'self.safety_order_price_mode' is 'static' or 'dynamic'.
        2) If 'dynamic' + 'volume', shows planned sizes from multipliers.
        3) If 'dynamic' + 'value', can't precompute final size => show "N/A".
        """


        
        console = Console()

        # Print initial BO info
        logger.info(
            f"{current_time} - Initial position opened with {base_order_quantity} units "
            f"at price {price:.10f}, total value: {self.initial_investment:.2f}$"
        )

        # Build the table
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Stage")
        table.add_column("Price (BO% - PrevCh%)")
        table.add_column("Size")
        table.add_column("Value")
        table.add_column("Cumulative Value")
        table.add_column("Cumulative Size")

        cumulative_value = 0.0
        cumulative_size = 0.0

        # Base Order
        bo_value = self.base_order_price * base_order_quantity
        cumulative_value += bo_value
        cumulative_size += base_order_quantity

        table.add_row(
            "BO",
            f"{self.base_order_price:.10f} (0.00% - 0.00%)",
            f"{base_order_quantity:.2f}",
            f"{bo_value:.2f}",
            f"{cumulative_value:.2f}",
            f"{cumulative_size:.2f}",
        )

        # (1) Check if price mode = "static" or "dynamic"
        price_mode = self.safety_order_price_mode.lower()
        so_mode = self.safety_order_mode.lower()  # "value" or "volume"

        if price_mode == "static":
            # Show planned so_price and so_size for each DCA level
            for i in range(1, self.max_dca_levels + 1):
                # (2) Use static_safety_price()
                so_price = self.static_safety_price(self.base_order_price, i)
                so_size = self.calculate_safety_order_qty(
                    so_price, self.first_safety_order_multiplier, i
                )
                so_value = so_price * so_size

                # Price changes vs base and vs previous
                bo_change = ((so_price - self.base_order_price) / self.base_order_price) * 100
                if i > 1:
                    prev_so_price = self.static_safety_price(self.base_order_price, i - 1)
                    prev_change = ((so_price - prev_so_price) / prev_so_price) * 100
                else:
                    prev_change = bo_change

                cumulative_value += so_value
                cumulative_size += so_size

                table.add_row(
                    f"DCA-{i}",
                    f"{so_price:.10f} ({bo_change:.2f}% - {prev_change:.2f}%)",
                    f"{so_size:.2f}",
                    f"{so_value:.2f}",
                    f"{cumulative_value:.2f}",
                    f"{cumulative_size:.2f}",
                )
        else:
            # price_mode == "dynamic"
            # (3) If volume-based, we can show planned sizes from multipliers.
            #     If value-based, final sizes depend on unknown fill prices => "N/A"
            if so_mode == "volume":
                # (a) We'll display the multiplier-based planned sizes
                so_size_prev = base_order_quantity * self.first_safety_order_multiplier
                for i in range(1, self.max_dca_levels + 1):
                    if i == 1:
                        so_size = so_size_prev
                    else:
                        so_size = so_size_prev * self.so_size_multiplier
                        so_size_prev = so_size

                    # We cannot know the fill price => "N/A"
                    # No direct 'cumulative_value' for dynamic unknown price
                    cumulative_size += so_size

                    table.add_row(
                        f"DCA-{i} [DYN]",
                        "N/A - dynamic",
                        f"{so_size:.2f}",
                        "N/A",
                        "N/A",
                        f"{cumulative_size:.2f}",
                    )
            

        console.print(table)

    
    def process_dca(self, price: float, rsi_val: float, dynamic_thr: float, current_time:datetime) -> None:
        ctx = self._ctx()
        if self.dca_level >= self.max_dca_levels:
            return
        if not self._should_add_safety_order(ctx):
            return
        level = self.dca_level + 1
        so_price = self._calculate_so_price(ctx, level)
        if price > so_price:
            if self.debug_trade:
                logger.debug(f"Price {price:.10f} > SO-{level} trigger {so_price:.10f}")
            return
        so_size = self._calculate_position_size(ctx, so_price, level)
        size_to_buy = self._safety_affordable_size(so_price, so_size)
        
        if size_to_buy > 0:
            order = self.buy(size=size_to_buy, tag=f"S{level}")
            
        self.last_safety_order_time = current_time
        self.dca_level += 1
        self._consume_rsi_reset()
        if self.safety_order_price_mode.lower() == "dynamic":
            self.last_filled_price = price
        if self.debug_trade:
            self.debug_trade_info("DCA", order, so_price, size_to_buy * so_price)

    def process_exit(self, price, current_time):
        if not self.position:
            return False

        # Exit rules (decoupled – currently only TP-decay reached)
        ctx = self._ctx()
        if self._exit_rules.ok(ctx):
            if self.debug_process:
                self.log_exit_details(price, current_time)
                logger.debug(f"Trade: Exit Triggered at {current_time} with Price: {price:.10f}")
            self.position.close()
            self.reset_process()
            return True
        else:
            return False
    
        # --- RSI-reset utility ---------------------------------------------------- #
    def _need_reset(self) -> bool:                            
        """
        We require an RSI reset **only** when:
        1) RSI is actually calculated,
        2) dynamic threshold is OFF (static mode),
        3) the user asked for a reset.
        """
        return (
            self.enable_rsi_calculation and
            not self.rsi_dynamic_threshold and
            self.require_rsi_reset
        )
    
    def log_exit_details(self, price, current_time):
        """
        Summarize the final position at exit using:
        1) Actual base-order (BO) trade from self.trades
        2) Actual safety-order (DCA) trades from self.trades
        3) Final exit (TP) stats from self.position
        
        Removes any direct usage of:
        - self.base_order_price
        - self.initial_investment
        - self.so_prices, self.so_sizes
        - self.dca_level
        in favor of real executed-trade data.
        """

        from rich.table import Table
        from rich.console import Console
        console = Console()

        # Create the results table
        summary_table = Table(show_header=True, header_style="bold green")
        summary_table.add_column("Stage")
        summary_table.add_column("Entry Price")
        summary_table.add_column("Size")
        summary_table.add_column("Value")
        summary_table.add_column("Cumulative Value")
        summary_table.add_column("Cumulative Size")
        summary_table.add_column("PnL")
        summary_table.add_column("ROI")

        # Track cumulative totals
        cumulative_value = 0.0
        cumulative_size = 0.0

        # -------------------------------
        # 1) BASE ORDER (tag == "BO")
        # -------------------------------
        # - Find the trade with tag == "BO"
        bo_trade = next((t for t in self.trades if t.is_long and t.tag == "BO"), None)
        if bo_trade:
            bo_price = bo_trade.entry_price
            bo_size = bo_trade.size
            bo_value = bo_price * bo_size
            cumulative_value += bo_value
            cumulative_size += bo_size
            bo_pnl = bo_trade.pl * (1-backtest_params.get("commission", 0.0))
            bo_roi = bo_trade.pl_pct*100

            # Base order row (no PnL yet)
            summary_table.add_row(
                "BO",
                f"{bo_price:.10f}",
                f"{bo_size:.2f}",
                f"{bo_value:.2f}",
                f"{cumulative_value:.2f}",
                f"{cumulative_size:.2f}",
                f"{bo_pnl:.2f}",  
                f"{bo_roi:.2f}%"
            ) 

        # -------------------------------
        # 2) SAFETY ORDERS (tag starts with "S")
        # -------------------------------
        # - Gather safety-order trades: S1, S2, ...
        so_trades = [t for t in self.trades if t.is_long and t.tag and t.tag.startswith("S")]
        for i, so_trade in enumerate(so_trades, start=1):
            so_price = so_trade.entry_price
            so_size = so_trade.size
            so_value = so_price * so_size

            # If position closed at 'price', hypothetical PnL:
            pnl = (price - so_price) * so_size
            roi = (pnl / so_value * 100) if so_value else 0.0

            cumulative_value += so_value
            cumulative_size += so_size

            # Safety-order row
            summary_table.add_row(
                f"DCA-{i}",
                f"{so_price:.10f}",
                f"{so_size:.2f}",
                f"{so_value:.2f}",
                f"{cumulative_value:.2f}",
                f"{cumulative_size:.2f}",
                f"{pnl:.2f}",
                f"{roi:.2f}%"
            )

        # -------------------------------
        # 3) FINAL TAKE-PROFIT (TP) ROW
        # -------------------------------
        # - If position is still open at exit, capture final stats
        if self.position:
            tp_value = self.position.size * price
            tp_pnl = self.position.pl
            tp_roi = self.position.pl_pct

            summary_table.add_row(
                "TP",
                f"{price:.10f}",
                f"{self.position.size:.2f}",
                f"{tp_value:.2f}",
                "-",    # Not accumulating further; it's a close
                "-",
                f"{tp_pnl:.2f}",
                f"{tp_roi:.2f}%"
            )

        # Print or log the final table
        console.log(summary_table)
        # console.log(self.trades)

    
    def _update_rsi_reset(self, rsi_val: float) -> None:
        """
        Static-RSI reset logic (no behavior change):
        - Only when _need_reset() is True (static mode with reset enabled)
        - RSI NaN ⇒ reset True
        - RSI >= reset_thr ⇒ reset True
        """
        if not self._need_reset():
            return
        if np.isnan(rsi_val):
            self.rsi_reset = True
            return
        reset_thr = self.rsi_threshold * (1 + self.rsi_reset_percentage / 100)
        if rsi_val >= reset_thr:
            self.rsi_reset = True

    def _consume_rsi_reset(self) -> None:
        """
        Consume the reset immediately after we place BO/SO (same as old inline 'set False').
        """
        if self._need_reset():
            self.rsi_reset = False

    
    
    # CHANGED: Renamed reset_cycle to reset_process
    def reset_process(self):
        self.base_order_price = None
        self.base_order_quantity = None
        self.base_order_value = None
        self.dca_level = 0
        self.last_filled_price = None
        self.rsi_reset = True
        # MODIFIED: Reset time tracking variables
        self.base_order_time = None
        self.last_safety_order_time = None
        
    def get_entry_price(self) -> float:
        """
        Calculate the average entry price of the current position using:
        - Position's profit/loss (pl) or percentage profit/loss (pl_pct)
        - Current price from the broker
        - Position size
        
        Returns:
            float: Average entry price
        """
        if not self.position:
            raise ValueError("No open position exists.")
        
        current_price = self._broker.last_price
        position = self.position
        size = position.size
        
        if size == 0:
            raise ValueError("Position size cannot be zero.")
        
        if hasattr(position, 'pl'):
            # Calculate using profit/loss in cash units
            return current_price - (position.pl / size)
        
        elif hasattr(position, 'pl_pct'):
            # Calculate using percentage profit/loss
            return current_price / (1 + position.pl_pct)
        
        else:
            raise AttributeError("Position has neither 'pl' nor 'pl_pct' attributes.")

    def next(self):
        current_time = self.data.index[-1]
        if current_time < self.start_trading_time:
            return
        
        dynamic_thr = (
            self.rsi_dynamic_threshold_series.loc[current_time]
            if self.rsi_dynamic_threshold and not  np.isnan (self.rsi_dynamic_threshold_series.loc[current_time])
            else self.rsi_threshold)
        
        current_idx = len(self.data) - 1
        if self.debug_backtest or self.show_indicators.get('average_entry_price', True):
            if self.position:
                # Calculate breakeven price
                entry_price = self.get_entry_price()
                self.breakeven_prices[current_idx] = entry_price
                # Calculate take profit price
                
                #adjusted_tp_percentage = self.calculate_adjusted_take_profit(current_time)
                        # Use shared function for plotting
                adjusted_tp_percentage = calculate_decaying_tp(
                    self.last_safety_order_time or self.base_order_time,
                    current_time,
                    self.take_profit_percentage,
                    self.take_profit_reduction_duration_hours
                )
                # take_profit_percent is in percent, so divide by 100
                tp_price = entry_price * (1 + adjusted_tp_percentage / 100)
                self.take_profit_prices[current_idx] = tp_price
            else:
                self.breakeven_prices[current_idx] = None
                

        price = self.data.Close[-1]
        current_low = self.data.Low[-1]
        current_high = self.data.High[-1]
        if self.slippage_probability > 0 and np.random.rand() < self.slippage_probability:
            price *= np.random.uniform(0.995, 1.005)
            current_low *= np.random.uniform(0.995, 1.005)
            current_high *= np.random.uniform(0.995, 1.005)
        
    

        if self.rsi_values is not None and not self.rsi_values.empty:
            try:
                rsi_val = self.rsi_values.loc[current_time]  # Use index-based lookup
            except KeyError:
                rsi_val = self.rsi_values.iloc[-1]  # Fallback to positional
        else:
            rsi_val = np.nan
        self._update_rsi_reset(rsi_val)
                    
        self.debug_loop_info(price, rsi_val, current_time)
        if self.position:
            if self.process_exit(current_high, current_time):  # Exit condition met
                return  # Skip further processing in this process
            else:
                if not self.enable_rsi_calculation: # Value may be a number in other cases i.e visualization only
                    rsi_val = np.nan
                self.process_dca(current_low, rsi_val, dynamic_thr, current_time)
        else:
 
            # Entry rules (decoupled – currently only RSI rule)
            ctx = self._ctx()
            if not self._entry_rules.ok(ctx):
                return
 
            self.process_entry(price, current_time)
            return True

    #▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
    # DECISION POINT HELPERS (Future migration targets)
    #▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
    
    def _should_enter_position(self, ctx: Ctx) -> bool:
        """Entry conditions - move to rules/entry.py later"""
        rsi_val = ctx.indicators.get('rsi', np.nan)
        return not (self.avoid_rsi_overbought 
                  and not np.isnan(rsi_val) 
                  and rsi_val >= (self.rsi_overbought_level or 70))

    def _should_add_safety_order(self, ctx: Ctx) -> bool:
        """Safety order conditions - move to rules/safety.py later"""
        rsi_val = ctx.indicators.get('rsi', np.nan)
        return (self.rsi_reset
               and (np.isnan(rsi_val) or rsi_val < ctx.dynamic_rsi_thr)
               and (self.so_cooldown_minutes == 0 or 
                   ctx.last_so_dt is None or
                   (ctx.now - ctx.last_so_dt).total_seconds() >= self.so_cooldown_minutes * 60))

    def _calculate_so_price(self, ctx: Ctx, level: int) -> float:
        """Compute SO trigger price for a given level"""
        """Pricing logic - move to rules/pricing.py later"""
        if self.safety_order_price_mode.lower() == "dynamic":
            return self.dynamic_safety_price(ctx.last_filled_price, level,  ctx.current_atr if ctx.current_atr is not None else np.nan)
        return self.static_safety_price(ctx.base_order_price, level)

    def _calculate_position_size(self, ctx: Ctx, price: float, level: int) -> float:
        """Sizing logic - move to rules/sizing.py later"""
        if self.safety_order_mode.lower() == "value":
            if level == 1:
                return ctx.base_order_value * self.first_safety_order_multiplier / price
            return ctx.base_order_value * self.first_safety_order_multiplier * (self.so_size_multiplier ** (level - 1)) / price
        else:  # volume mode
            if level == 1:
                return ctx.base_order_quantity * self.first_safety_order_multiplier
            return ctx.base_order_quantity * self.first_safety_order_multiplier * (self.so_size_multiplier ** (level - 1))

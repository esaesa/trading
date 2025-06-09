# strategy.py

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
class DCAStrategy(Strategy):
    # Strategy Parameters
    entry_fraction = strategy_params['entry_fraction']
    so_size_multiplier = strategy_params['so_size_multiplier']
    first_safety_order_multiplier = strategy_params['first_safety_order_multiplier']
    safety_order_mode = strategy_params.get("safety_order_mode")
    max_dca_levels = strategy_params['max_dca_levels']
    take_profit_percentage = strategy_params['take_profit_percentage']
    initial_deviation_percent = strategy_params['initial_deviation_percent']
    price_multiplier = strategy_params['price_multiplier']
    minimum_notional = strategy_params.get('minimum_notional', 5.0)
    rsi_threshold = strategy_params.get("rsi_threshold")
    rsi_window = strategy_params.get("rsi_window")
    require_rsi_reset = strategy_params.get("require_rsi_reset")
    rsi_reset_percentage = strategy_params.get("rsi_reset_percentage")
    rsi_dynamic_threshold = strategy_params.get("rsi_dynamic_threshold")
    rsi_dynamic_window = strategy_params.get("rsi_dynamic_window", rsi_window *10000)
    rsi_percentile = strategy_params.get("rsi_percentile", 0.05)
    safety_order_price_mode = strategy_params.get("safety_order_price_mode")
    start_trading_time = datetime.strptime(
        strategy_params.get("start_trading_time", "2025-02-01 00:00:00"),
        "%Y-%m-%d %H:%M:%S"
    )
    
    # ATR Parameters
    enable_atr_calculation = strategy_params.get("enable_atr_calculation", False)
    atr_window = strategy_params.get("atr_window", 14)
    atr_resample_interval = strategy_params.get("atr_resample_interval", "h")
    
    # EMA Parameters
    enable_ema_calculation = strategy_params.get("enable_ema_calculation", False)
    ema_window = strategy_params.get('ema_window', 200)
    ema_resample_interval = strategy_params.get('ema_resample_interval', "h")
    
    # RSI Parameters
    enable_rsi_calculation = strategy_params.get("enable_rsi_calculation", False)
    rsi_resample_interval = strategy_params.get('rsi_resample_interval')
    show_rsi = strategy_params.get("show_rsi", False)
    
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
        

    def unscale_parameter(self, key, value):
        if self.enable_optimization and optimization_params[key]['type'] == 'float':
            param_scale = optimization_params[key].get('scaling_factor')
            if param_scale is not None:
                return value / param_scale
        return value

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
        
        # ----------------------
        # Initialize Indicators
        # ----------------------        
        # Determine calculation needs (show OR strategy use)
        calculate_rsi = self.show_indicators.get('rsi', False) or self.enable_rsi_calculation
        calculate_atr = self.show_indicators.get('atr', False) or self.enable_atr_calculation
        calculate_ema = self.show_indicators.get('ema', False) or self.enable_ema_calculation
        calculate_laguerre = self.show_indicators.get('laguerre', False) or self.enable_laguere_calculation
        calculate_bbw = self.show_indicators.get('bbw', False)   
        calculate_cv = self.show_indicators.get('cv', False)
        calculate_dynamic_rsi = self.show_indicators.get('dynamic_rsi', False) or self.rsi_dynamic_threshold
         
        df = self.data.df

       
        self.indicators = IndicatorManager(df)

        if calculate_rsi:
                self.rsi_values = self.indicators.compute_rsi(
                    window=self.rsi_window,
            resample_interval=self.rsi_resample_interval
                )
                if self.show_indicators.get('rsi', False):
                    self.I(lambda _: self.rsi_values, self.data.Close, name="RSI")
                    
        # Compute dynamic RSI threshold if enabled            
        if calculate_dynamic_rsi:
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
            if self.show_indicators.get('ema', False):
                self.I(lambda _: ema_series, self.data.Close, name="EMA")

        if calculate_atr:
            atr_pct_series = self.indicators.compute_atr_percentage(window=14, resample_interval='1D')
            if self.show_indicators.get('atr', False):
                self.I(lambda _: atr_pct_series, self.data.Close, name="ATR%")
    

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

    # --- Modularized Functions ---

    def dynamic_safety_price(self, last_filled_price, order_num):
        deviation = self.initial_deviation_percent * math.pow(self.price_multiplier, order_num)
        if deviation >= 100:
            if self.debug_loop:
                logger.warning(f"Deviation for level {order_num} is greater than 100%. Returning minimum value.")
            return 1e-8
        computed_price = last_filled_price * ((100 - deviation) / 100)
        return computed_price

    def compute_order_quantity(self, price: float, fraction: float) -> int:
        """
        Compute maximum integer quantity based on available cash and fraction.
        """
        investment = self._broker._cash * fraction
        return math.floor(investment / price)
    
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


    def process_entry(self, price, current_time):
        if not self.position:
            if self.enable_rsi_calculation and self.rsi_values is None:
                return False
            multiplier = 2 if (self.enable_ema_calculation and self.ema_dynamic is not None and price > self.ema_dynamic[-1]) else 1
            initial_investment = self._broker._cash * (self.entry_fraction * multiplier)
            base_order_quantity = math.floor(initial_investment / price)
            effective_fraction = self.entry_fraction * multiplier
            if base_order_quantity * price >= self.minimum_notional:
                if base_order_quantity < 1:
                    raise ValueError("Backtesting.py does not support fractional units. Adjust your entry fraction.")
                order = self.buy(size=base_order_quantity, tag="BO")
                self.dca_level = 0
                self.base_order_price = price
                self.initial_investment = initial_investment
                self.last_filled_price = price
                self.base_order_quantity = base_order_quantity
                self.base_order_value = base_order_quantity * self.base_order_price
                # self.setup_safety_orders(base_order_quantity, self.base_order_price)
                if self.debug_trade:
                    self.log_entry_details(current_time, price, base_order_quantity)
                    self.debug_trade_info("Entry", order, price, base_order_quantity * price)
                return True
            else:
                if self.debug_trade:
                    logger.warning("Initial order below minimum notional, skipping entry.")
                return False
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

    
    def process_dca(self, price, rsi_val, dynamic_thr, current_time):
        if self.dca_level < self.max_dca_levels and self.rsi_reset:
            if self.safety_order_price_mode.lower() == "dynamic":
                so_price = self.dynamic_safety_price(self.last_filled_price, self.dca_level + 1)
            else:
                so_price = self.static_safety_price(self.base_order_price, self.dca_level + 1)
            if price <= so_price and (np.isnan(rsi_val) or rsi_val < dynamic_thr):
                
                so_size = self.calculate_safety_order_qty(so_price, self.first_safety_order_multiplier, self.dca_level + 1)
                # Calculate maximum affordable size considering commission and available cash
                commission_rate = backtest_params.get("commission", 0.0)
                available_cash = self._broker.margin_available * self._broker._leverage
                cost_per_share = so_price * (1 + commission_rate)
                
                if cost_per_share <= 0:
                    max_possible = 0
                else:
                    max_possible = int(available_cash // cost_per_share)
                
                if max_possible < 1:
                    if self.debug_trade:
                        logger.warning(f"Insufficient funds for DCA-{self.dca_level + 1}. Available cash: {available_cash:.2f}, cost per share: {cost_per_share:.2f}")
                    return
                
                size_to_buy = min(math.floor(so_size), max_possible)
                size_to_buy = max(size_to_buy, 1)  # Ensure at least 1 share
                
                if size_to_buy > max_possible:
                    print('WARNING: order will be reduced due to insufficient funds')
                
                
                if size_to_buy * so_price >= self.minimum_notional:
                    order = self.buy(size=size_to_buy, tag=f"S{self.dca_level+1}")
                    self.dca_level += 1
                    if self.require_rsi_reset:
                        self.rsi_reset = False
                    if self.safety_order_price_mode.lower() == "dynamic":
                        self.last_filled_price = price
                    if self.debug_trade:
                        self.debug_trade_info("DCA", order, so_price, size_to_buy * so_price)
                else:
                    if self.debug_trade:
                        logger.warning(f"DCA-{self.dca_level} order below minimum notional, skipping DCA level.")

    # def corrected_pl_pct(self) -> float:
    #     x = self.position.pl_pct
    #     """Returns the actual percentage P/L relative to total position size."""
    #     last_price = self.data.Close[-1]  # Current price
    #     trades = self._broker.trades
    #     if not trades:
    #         return 0.0
    #     total_pl = sum((last_price - trade.entry_price) * trade.size for trade in trades)
    #     total_size = sum(abs(trade.size) for trade in trades)
    #     if total_size == 0:
    #         return 0.0  # Prevent division by zero
    #     return (total_pl / total_size) * 100  # True PL percentage

    
    def process_exit(self, price, current_time):
        if self.position.pl_pct  >= self.take_profit_percentage:
            if self.debug_process:
                self.log_exit_details(price, current_time)
                logger.debug(f"Trade: Exit Triggered at {current_time} with Price: {price:.10f}")
            self.position.close()
            self.reset_process()
            return True
        else:
            return False

    
    
    
    
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

    
    
    
    
    # CHANGED: Renamed reset_cycle to reset_process
    def reset_process(self):
        self.base_order_price = None
        self.base_order_quantity = None
        self.base_order_value = None
        self.dca_level = 0
        self.last_filled_price = None
        self.rsi_reset = True
        
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
                tp_price = entry_price * (1 + self.take_profit_percentage / 100)
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
        #if np.isnan(rsi_val) or rsi_val >= self.rsi_threshold * (1 + self.rsi_reset_percentage/100):
        if np.isnan(rsi_val) or rsi_val >= dynamic_thr * (1 + self.rsi_reset_percentage/100):    
            self.rsi_reset = True
        self.debug_loop_info(price, rsi_val, current_time)
        if self.position:
            if self.process_exit(current_high, current_time):  # Exit condition met
                return  # Skip further processing in this process
            else:
                if not self.enable_rsi_calculation: # Value may be a number in other cases i.e visualization only
                    rsi_val = np.nan
                self.process_dca(current_low, rsi_val, dynamic_thr, current_time)
        else:
            self.process_entry(price, current_time)
            return True


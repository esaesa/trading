from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any

@dataclass
class StrategyConfig:
    # Safety Order Parameters
    so_cooldown_minutes: int = 0
    entry_fraction: float = 0.1
    so_size_multiplier: float = 2.0
    first_safety_order_multiplier: float = 1.5
    safety_order_mode: str = "fixed"
    max_dca_levels: int = 5
    take_profit_percentage: float = 1.0
    initial_deviation_percent: float = 1.0
    price_multiplier: float = 1.01
    minimum_notional: float = 0.0

    # RSI Parameters
    rsi_threshold: float = 50.0
    rsi_window: int = 14
    require_rsi_reset: bool = False
    rsi_reset_percentage: float = 50.0
    rsi_dynamic_threshold: bool = False
    rsi_static_threshold_under: float = 21.0
    rsi_percentile: float = 0.05
    rsi_overbought_level: float = 70.0
    rsi_oversold_level: float = 30.0

    # Safety Order Price Mode
    safety_order_price_mode: str = "fixed"

    # Trading Time
    start_trading_time: datetime = datetime(2025, 2, 1, 0, 0, 0)

    # Take Profit Decay
    take_profit_decay_grace_period_hours: float = 48.0
    take_profit_decay_duration_hours: float = 24.0

    # ATR Parameters
    enable_atr_calculation: bool = False
    atr_window: int = 14
    atr_resample_interval: str = "h"
    atr_deviation_threshold: float = 5.0
    atr_deviation_reduction_factor: float = 0.25

    # EMA Parameters
    enable_ema_calculation: bool = False
    ema_window: int = 200
    ema_resample_interval: str = "h"

    # RSI/Indicators
    enable_rsi_calculation: bool = False
    rsi_resample_interval: str = "h"
    show_rsi: bool = False
    avoid_rsi_overbought: bool = False
    enable_cv_calculation: bool = False
    enable_laguere_calculation: bool = False
    show_indicators: Dict[str, Any] = field(default_factory=dict)

    # Debug Parameters
    debug_backtest: bool = False
    debug_loop: bool = False
    debug_trade: bool = False
    debug_process: bool = False
    slippage_probability: float = 0.0

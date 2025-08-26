from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional

def _get_rule_param(params: Dict[str, Any], rule_name: str, param_name: str, default_value: Any) -> Any:
    """Get parameter from rule-based config structure."""
    rules_config = params.get('rules', {})
    return rules_config.get(rule_name, {}).get(param_name, default_value)

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
    atr_window: int = 14
    atr_resample_interval: str = "h"
    atr_deviation_threshold: float = 5.0
    atr_deviation_reduction_factor: float = 0.25

    # EMA Parameters
    ema_window: int = 200
    ema_resample_interval: str = "h"

    # RSI/Indicators
    rsi_resample_interval: str = "h"
    show_rsi: bool = False
    avoid_rsi_overbought: bool = False
    show_indicators: Dict[str, Any] = field(default_factory=dict)

    # Debug Parameters
    debug_backtest: bool = False
    debug_loop: bool = False
    debug_trade: bool = False
    debug_process: bool = False
    slippage_probability: float = 0.0

    # Rule-specific configuration storage
    _rules_config: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def init_from_params(self, params: Dict[str, Any]) -> 'StrategyConfig':
        """Initialize from parameters dict with rule-based config support."""
        # Store rule-specific configuration
        self._rules_config = params.get('rules', {})

        # Update rule-specific parameters using the helper function
        self.rsi_static_threshold_under = _get_rule_param(
            params, 'RSIUnderStaticThreshold', 'static_threshold_under', self.rsi_static_threshold_under
        )

        self.rsi_threshold = _get_rule_param(
            params, 'RSIUnderDynamicThreshold', 'threshold', self.rsi_threshold
        )
        self.rsi_window = _get_rule_param(
            params, 'RSIUnderDynamicThreshold', 'window', self.rsi_window
        )
        self.rsi_dynamic_threshold = _get_rule_param(
            params, 'RSIUnderDynamicThreshold', 'dynamic_threshold', self.rsi_dynamic_threshold
        )
        self.require_rsi_reset = _get_rule_param(
            params, 'RSIUnderDynamicThreshold', 'require_reset', self.require_rsi_reset
        )
        self.rsi_reset_percentage = _get_rule_param(
            params, 'RSIUnderDynamicThreshold', 'reset_percentage', self.rsi_reset_percentage
        )

        self.rsi_overbought_level = _get_rule_param(
            params, 'RSIOverbought', 'overbought_level', self.rsi_overbought_level
        )

        self.take_profit_decay_grace_period_hours = _get_rule_param(
            params, 'TPDecayReached', 'grace_period_hours', self.take_profit_decay_grace_period_hours
        )
        self.take_profit_decay_duration_hours = _get_rule_param(
            params, 'TPDecayReached', 'decay_duration_hours', self.take_profit_decay_duration_hours
        )

        self.atr_deviation_threshold = _get_rule_param(
            params, 'ATRTakeProfitReached', 'deviation_threshold', self.atr_deviation_threshold
        )
        self.atr_deviation_reduction_factor = _get_rule_param(
            params, 'ATRTakeProfitReached', 'deviation_reduction_factor', self.atr_deviation_reduction_factor
        )

        return self

    def get_rule_param(self, rule_name: str, param_name: str, default_value: Any) -> Any:
        """Get a parameter from rule-specific configuration."""
        if rule_name in self._rules_config and param_name in self._rules_config[rule_name]:
            return self._rules_config[rule_name][param_name]
        return default_value

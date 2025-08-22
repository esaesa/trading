import numpy as np
from datetime import datetime
from typing import TYPE_CHECKING
from logger_config import logger
from instrumentation import log_trade_info

if TYPE_CHECKING:
    from strategy import DCAStrategy

class StrategyLogger:
    def __init__(self, strategy: 'DCAStrategy'):
        self.strategy = strategy

    def log_trade(self, action: str, order, price: float, quantity: float) -> None:
        """Log trade information"""
        log_trade_info(self.strategy, action, order, price, quantity)

    def log_loop_info(self, price: float, rsi_val: float, current_time: datetime) -> None:
        """Log loop information for debugging"""
        if self.strategy.config.debug_loop:
            level = self.strategy.dca_level + 1
            logger.debug(
                f"Loop: time={current_time}, "
                f"price={price:.10f}, "
                f"rsi={rsi_val:.2f}, "
                f"level={level}"
            )

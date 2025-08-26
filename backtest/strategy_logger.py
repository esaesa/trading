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

    def log_loop_info(self, ctx) -> None:
        """Log loop information for debugging using context object"""
        if self.strategy.config.debug_loop:
            level = self.strategy.dca_level + 1
            rsi_val = self.strategy.indicator_service.get_indicator_value("rsi", ctx.now, float('nan'))
            logger.debug(
                f"Loop: time={ctx.now}, "
                f"price={ctx.price:.10f}, "
                f"rsi={rsi_val:.2f}, "
                f"level={level}"
            )

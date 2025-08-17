# instrumentation.py
from logger_config import logger

def log_loop_info(strategy, price, rsi_val, current_time):
    """SRP: logging-only; Strategy owns state/decisions."""
    if not getattr(strategy, "debug_loop", False):
        return
    pos = strategy.position
    logger.debug(f"[Loop Debug] Time: {current_time}")
    logger.debug(f"[Loop Debug] Current Price: {price}")
    logger.debug(f"[Loop Debug] Position Size: {pos.size if pos else 0}")
    logger.debug(f"[Loop Debug] Position PnL%: {pos.pl_pct if pos else 0}")
    logger.debug(f"[Loop Debug] RSI Value: {rsi_val}")
    logger.debug(f"[Loop Debug] RSI Reset: {getattr(strategy, 'rsi_reset', None)}")
    if pos:
        base_t = getattr(strategy, "base_order_time", None)
        so_t   = getattr(strategy, "last_safety_order_time", None)
        tsb = current_time - base_t if base_t else "N/A"
        tss = current_time - so_t   if so_t   else "N/A"
        logger.debug(f"[Loop Debug] Time Since BO: {tsb} | Time Since Last SO: {tss}")

def log_trade_info(strategy, trade_type, order, computed_trigger, order_value):
    """SRP: logging-only; Strategy owns state/decisions."""
    if not getattr(strategy, "debug_trade", False):
        return
    executed_price = strategy._broker.last_price
    current_time   = strategy.data.index[-1]
    logger.debug(
        f"{trade_type} at {current_time} | "
        f"Tag: {getattr(order, 'tag', '')} | Order Size: {getattr(order, 'size', 0)} | "
        f"Computed Trigger Price: {computed_trigger:.10f} | "
        f"Executed Price: {executed_price:.10f} | "
        f"Order Value: {order_value:.2f}$"
    )

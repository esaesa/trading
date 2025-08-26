# indicators.py - Centralized indicator definitions
from enum import Enum

class Indicators(Enum):
    """Centralized indicator name definitions for the entire trading system."""
    RSI = "rsi"
    DYNAMIC_RSI_THRESHOLD = "dynamic_rsi_threshold"
    ATR_PCT = "atr_pct"
    EMA = "ema"
    LAGUERRE_RSI = "laguerre_rsi"
    CV = "cv"
    BBW = "bbw"
# rules/pricing.py
from __future__ import annotations
from typing import Dict, Type, Any, List
import math
import numpy as np
from ports import PriceEngine
from contracts import Ctx
from logger_config import logger
from indicators import Indicators
import statistics # <-- Add this import for 'avg'

# --- NEW: Price Engine Factory (Now the single source of truth) ---
class PriceEngineFactory:
    _engine_classes: Dict[str, Type[PriceEngine]] = {}

    @classmethod
    def register(cls, name: str, engine_class: Type[PriceEngine]):
        cls._engine_classes[name] = engine_class

    @classmethod
    def create(cls, name: str, strategy: Any) -> PriceEngine:
        if name not in cls._engine_classes:
            # --- CORRECTED: Use class names as keys for clarity ---
            available = ', '.join(cls._engine_classes.keys())
            raise ValueError(f"Unknown PriceEngine '{name}'. Available engines: {available}")
        return cls._engine_classes[name](strategy)

# --- Concrete Engine Implementations (No logic changes needed) ---
EPS = 1e-8

class StaticPriceEngine(PriceEngine):
    def __init__(self, strategy) -> None:
        self.s = strategy
    def get_required_indicators(self) -> set[str]:
        return set()
    def so_price(self, ctx: Ctx, level: int) -> float:
        if not (1 <= level <= self.s.config.max_dca_levels): return 0.0
        d = self.s.config.initial_deviation_percent
        m = self.s.config.price_multiplier
        factors = [1 - (d / 100.0) * (m ** i) for i in range(level)]
        return EPS if any(f <= 0 for f in factors) else ctx.base_order_price * math.prod(factors)

class DynamicATRPriceEngine(PriceEngine):
    def __init__(self, strategy):
        self.s = strategy
    def get_required_indicators(self) -> set[str]:
        return {Indicators.ATR_PCT.value}
    def _effective_multiplier(self, current_atr_value: float | None) -> float:
        if current_atr_value is not None and not np.isnan(current_atr_value) and current_atr_value < self.s.config.atr_deviation_threshold:
            return self.s.config.price_multiplier * (1 - self.s.config.atr_deviation_reduction_factor)
        return self.s.config.price_multiplier
    def so_price(self, ctx: Ctx, level: int) -> float:
        last = ctx.last_filled_price or ctx.base_order_price
        if last is None: return 0.0
        atr_now = self.s.indicator_service.get_indicator_value("atr_pct", ctx.now, None)
        m_eff = self._effective_multiplier(atr_now)
        deviation = self.s.config.initial_deviation_percent * (m_eff ** level)
        if deviation >= 100: return EPS
        return float(last) * ((100.0 - deviation) / 100.0)

# --- CORRECTED: Register engines using their class names ---
PriceEngineFactory.register("StaticPriceEngine", StaticPriceEngine)
PriceEngineFactory.register("DynamicATRPriceEngine", DynamicATRPriceEngine)

# --- Composite Engine (Your implementation was good, minor tweaks) ---
class CompositePriceEngine(PriceEngine):
    def __init__(self, strategy: Any, sub_engines: list[PriceEngine], mode: str):
        if mode not in ["min", "max", "avg"]: raise ValueError(f"Invalid composite mode: {mode}")
        if not sub_engines: raise ValueError("CompositePriceEngine requires at least one sub-engine.")
        self.s = strategy
        self.sub_engines = sub_engines
        self.mode = mode
        self.op = {"min": min, "max": max, "avg": statistics.mean}[mode]
    def get_required_indicators(self) -> set[str]:
        required = set()
        for engine in self.sub_engines:
            if hasattr(engine, 'get_required_indicators'):
                required.update(engine.get_required_indicators())
        return required
    def so_price(self, ctx: Ctx, level: int) -> float:
        prices = [engine.so_price(ctx, level) for engine in self.sub_engines]
        valid_prices = [p for p in prices if p > 0]
        return self.op(valid_prices) if valid_prices else 0.0

# --- CORRECTED: Simplified, recursive builder function ---
def create_price_engine(strategy: Any, config_spec: Any) -> PriceEngine:
    """
    Recursively builds a PriceEngine from a configuration specification.
    - A string becomes a single engine (e.g., "StaticPriceEngine").
    - A dict becomes a CompositePriceEngine (e.g., {"min": [...]}).
    """
    if isinstance(config_spec, str):
        # Base case: create a single, named engine from the factory.
        return PriceEngineFactory.create(config_spec, strategy)

    if isinstance(config_spec, dict):
        # Recursive case: create a composite engine.
        if len(config_spec) != 1:
            raise ValueError("Composite engine spec must have exactly one key (min, max, or avg).")
        
        mode, sub_specs = next(iter(config_spec.items()))
        
        if not isinstance(sub_specs, list):
            raise ValueError(f"Value for '{mode}' must be a list of engine specifications.")

        # Recursively build the sub-engines for the composite.
        sub_engines = [create_price_engine(strategy, spec) for spec in sub_specs]
        
        return CompositePriceEngine(strategy, sub_engines, mode)
        
    raise TypeError(f"Invalid price engine configuration type: {type(config_spec)}")
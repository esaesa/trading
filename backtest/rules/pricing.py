# rules/pricing.py
from __future__ import annotations
from typing import Dict, Type, Any, List
import math
import numpy as np
from ports import PriceEngine
from contracts import Ctx
from logger_config import logger
from indicators import Indicators

EPS = 1e-8

class StaticPriceEngine(PriceEngine):
    """
    P_k = P0 * Π_{i=1..k} (1 - d/100 * m^(i-1))
    If any factor <= 0, clamp to EPS.
    """
    def __init__(self, strategy) -> None:
        self.s = strategy  # uses params from strategy

    def get_required_indicators(self) -> set[str]:
        """Return set of indicators required by this pricing engine."""
        return set()  # No indicators needed

    def so_price(self, ctx: Ctx, level: int) -> float:
        if level < 1 or level > self.s.config.max_dca_levels:
            return 0.0
        d = self.s.config.initial_deviation_percent
        m = self.s.config.price_multiplier
        factors = [1 - (d / 100.0) * (m ** i) for i in range(level)]
        if any(f <= 0 for f in factors):
            return EPS
        return ctx.base_order_price * math.prod(factors)

    @classmethod
    def validate_config(cls, config, name: str) -> bool:
        """Validate StaticPriceEngine configuration."""
        # Static engine requires initial_deviation_percent, price_multiplier, max_dca_levels
        required_attrs = ['initial_deviation_percent', 'price_multiplier', 'max_dca_levels']
        for attr in required_attrs:
            if not hasattr(config, attr):
                raise ValueError(f"Invalid {name} configuration: missing {attr}")
        return True


class DynamicATRPriceEngine(PriceEngine):
    def __init__(self, strategy):
        self.s = strategy
        self.provider = getattr(strategy, "indicator_provider", None)

    def get_required_indicators(self) -> set[str]:
        """Return set of indicators required by this pricing engine."""
        return {Indicators.ATR_PCT.value}

    def _effective_multiplier(self, current_atr_value: float | None) -> float:
        if (
            current_atr_value is not None and
            not np.isnan(current_atr_value) and
            current_atr_value < self.s.config.atr_deviation_threshold
        ):
            return self.s.config.price_multiplier * (1 - self.s.config.atr_deviation_reduction_factor)
        return self.s.config.price_multiplier

    def so_price(self, ctx: Ctx, level: int) -> float:
        last = ctx.last_filled_price or ctx.base_order_price
        if last is None:
            # No reference price yet — can't compute an SO trigger.
            return 0.0

        # Get current ATR as a regular indicator
        atr_now = self.s.indicator_service.get_indicator_value("atr_pct", ctx.now, None)

        m_eff = self._effective_multiplier(atr_now)
        deviation = self.s.config.initial_deviation_percent * (m_eff ** level)  # %

        if deviation >= 100:
            if self.s.config.debug_loop:
                logger.warning(f"Deviation for level {level} >= 100%. Returning minimum value.")
            return EPS

        return float(last) * ((100.0 - deviation) / 100.0)

    
    def _current_atr_pct(self, now):
        # Prefer provider
        if self.provider is not None:
            v = self.provider.atr_pct(now)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                return float(v)
        # Fallback to legacy access
        ser = getattr(self.s, "atr_pct_series", None)
        if ser is not None:
            try:
                return float(ser.loc[now])
            except Exception:
                try:
                    return float(ser.iloc[-1])
                except Exception:
                    return None
        return None

    @classmethod
    def validate_config(cls, config, name: str) -> bool:
        """Validate DynamicATRPriceEngine configuration."""
        # Dynamic engine requires same base attributes plus ATR-specific ones
        required_attrs = [
            'initial_deviation_percent', 'price_multiplier', 'max_dca_levels',
            'atr_deviation_threshold', 'atr_deviation_reduction_factor'
        ]
        for attr in required_attrs:
            if not hasattr(config, attr):
                raise ValueError(f"Invalid {name} configuration: missing {attr}")
        return True


class PriceEngineFactory:
    """Factory for creating and validating price engines."""

    _price_engines: Dict[str, Type[PriceEngine]] = {}

    @classmethod
    def register_engine(cls, name: str, engine_class: Type[PriceEngine]):
        """Register a price engine class with the factory."""
        cls._price_engines[name] = engine_class

    @classmethod
    def create_engine(cls, name: str, strategy) -> PriceEngine:
        """Create a price engine instance after validating its configuration."""
        if name not in cls._price_engines:
            raise ValueError(f"Unknown price engine: {name}")

        # Validate configuration first
        cls.validate_config(name, strategy.config)

        # Create engine instance
        return cls._price_engines[name](strategy)

    @classmethod
    def validate_config(cls, name: str, config) -> bool:
        """Validate price engine configuration during initialization."""
        if name not in cls._price_engines:
            raise ValueError(f"Unknown price engine: {name}")

        return cls._price_engines[name].validate_config(config, name)

    @classmethod
    def get_available_engines(cls) -> List[str]:
        """Get list of available price engine names."""
        return list(cls._price_engines.keys())


class CompositePriceEngine(PriceEngine):
    """Composite price engine that combines results from multiple engines using min/max/avg operations."""

    def __init__(self, strategy, engines: List[PriceEngine], operation: str):
        self.strategy = strategy
        self.engines = engines
        self.operation = operation.lower()

        if self.operation not in ['min', 'max', 'avg']:
            raise ValueError(f"Invalid composite operation: {operation}. Must be min, max, or avg.")

        if not engines:
            raise ValueError("Composite engine requires at least one engine.")

    def get_required_indicators(self) -> set[str]:
        """Return union of required indicators from all engines."""
        indicators = set()
        for engine in self.engines:
            indicators.update(engine.get_required_indicators())
        return indicators

    def so_price(self, ctx: Ctx, level: int) -> float:
        """Calculate price using composite operation on all engines."""
        prices = []
        for engine in self.engines:
            price = engine.so_price(ctx, level)
            if price > 0:  # Only include valid prices
                prices.append(price)

        if not prices:
            return 0.0

        if self.operation == 'min':
            return min(prices)
        elif self.operation == 'max':
            return max(prices)
        elif self.operation == 'avg':
            return sum(prices) / len(prices)

        return 0.0

    @classmethod
    def validate_config(cls, config, name: str) -> bool:
        """Composite engines are always valid."""
        return True


# Engine registration functions for backwards compatibility
def _ensure_price_engines_registered():
    """Ensure price engines are registered with PriceEngineFactory (called lazily to avoid circular imports)."""
    if "static" not in PriceEngineFactory._price_engines:
        PriceEngineFactory.register_engine("static", StaticPriceEngine)
    if "dynamic" not in PriceEngineFactory._price_engines:
        PriceEngineFactory.register_engine("dynamic", DynamicATRPriceEngine)

# Backward compatibility registry maintained for existing code
# Price engine registry for extensible selection
_PRICE_ENGINE_REGISTRY = {
    "static": lambda s: PriceEngineFactory.create_engine("static", s),
    "dynamic": lambda s: PriceEngineFactory.create_engine("dynamic", s),
}

def get_available_price_modes():
    """Return list of available price engine modes."""
    return list(_PRICE_ENGINE_REGISTRY.keys())

def create_price_engine(strategy, mode=None):
    """Create price engine with support for composite specifications.

    Supports both backward-compatible string format and new composite format:
    - String format: "static" or "dynamic"
    - Composite format: {"min": ["static", "dynamic"]}
    - Nested format: {"avg": ["static", {"min": ["dynamic", "custom"]}]}

    Args:
        strategy: The strategy instance
        mode: Optional mode override, defaults to strategy.config.safety_order_price_mode.
              Can be string or dict specification.

    Returns:
        PriceEngine instance

    Raises:
        ValueError: If an invalid mode is specified
    """
    # Ensure engines are registered (lazy registration to avoid circular imports)
    _ensure_price_engines_registered()

    # Get mode specification, allowing for dict-based composite specs
    mode_spec = mode or strategy.config.safety_order_price_mode or "dynamic"

    # Recursively create engine from specification
    return _create_engine_from_spec(strategy, mode_spec)


def _create_engine_from_spec(strategy, spec):
    """Recursively create engine from specification (string or dict)."""
    if isinstance(spec, str):
        # Simple string specification - create individual engine
        spec_lower = spec.lower()
        if spec_lower not in _PRICE_ENGINE_REGISTRY:
            available_modes = get_available_price_modes()
            raise ValueError(
                f"Invalid price engine mode '{spec}'. "
                f"Available modes: {', '.join(available_modes)}"
            )
        return _PRICE_ENGINE_REGISTRY[spec_lower](strategy)

    elif isinstance(spec, dict):
        # Dictionary specification - create composite engine
        if len(spec) != 1:
            raise ValueError(f"Composite specification must have exactly one operation, got: {list(spec.keys())}")

        operation = next(iter(spec.keys()))
        engine_specs = spec[operation]

        if not isinstance(engine_specs, list) or len(engine_specs) < 1:
            raise ValueError(f"Operation '{operation}' requires a non-empty list of engine specifications")

        if operation.lower() not in ['min', 'max', 'avg']:
            raise ValueError(f"Unknown composite operation '{operation}'. Must be min, max, or avg")

        # Recursively create child engines
        engines = [_create_engine_from_spec(strategy, child_spec) for child_spec in engine_specs]

        # Create composite engine
        return CompositePriceEngine(strategy, engines, operation)

    else:
        raise ValueError(f"Engine specification must be string or dict, got: {type(spec)}")



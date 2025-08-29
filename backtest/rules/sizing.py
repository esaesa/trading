# rules/sizing.py
from __future__ import annotations
from typing import Dict, Type, Any, List
from abc import ABC, abstractmethod
from ports import SizeEngine
from contracts import Ctx


class SizeFactory:
    """Specialized factory for creating and validating size engines."""

    _size_engines: Dict[str, Type[SizeEngine]] = {}

    @classmethod
    def register_engine(cls, name: str, engine_class: Type[SizeEngine]):
        """Register a size engine class with the factory."""
        cls._size_engines[name] = engine_class

    @classmethod
    def create_engine(cls, name: str, strategy) -> SizeEngine:
        """Create a size engine instance after validating its configuration."""
        if name not in cls._size_engines:
            raise ValueError(f"Unknown size engine: {name}")

        # Validate configuration first
        cls.validate_config(name, strategy.config)

        # Create engine instance
        return cls._size_engines[name](strategy)

    @classmethod
    def validate_config(cls, name: str, config) -> bool:
        """Validate size engine configuration during initialization."""
        if name not in cls._size_engines:
            raise ValueError(f"Unknown size engine: {name}")

        return cls._size_engines[name].validate_config(config, name)


class BaseSizeEngine(SizeEngine, ABC):
    """Base class for size engines focused on calculations."""

    def __init__(self, strategy):
        self.strategy = strategy
        self.config = strategy.config

    @abstractmethod
    def calculate_size(self, ctx: Ctx, price: float, level: int) -> float:
        """Calculate size for given context."""
        pass

    def so_size(self, ctx: Ctx, price: float, level: int) -> float:
        """Implement SizeEngine protocol using calculate_size."""
        return self.calculate_size(ctx, price, level)

    @classmethod
    @abstractmethod
    def validate_config(cls, config, name: str) -> bool:
        """Validate configuration at initialization."""
        pass




class ValueModeSizeRule(BaseSizeEngine):
    """
    Size_1   = (V_BO * first_mult) / P_1
    Size_k>1 = (V_BO * first_mult * so_mult^(k-1)) / P_k
    """

    def __init__(self, strategy):
        super().__init__(strategy)
        self.first_mult = self.config.first_safety_order_multiplier
        self.so_mult = self.config.so_size_multiplier

    def calculate_size(self, ctx: Ctx, price: float, level: int) -> float:
        """Calculate size using the value-based sizing formula."""
        v0 = ctx.base_order_value
        if level == 1:
            return (v0 * self.first_mult) / price
        return (v0 * self.first_mult * (self.so_mult ** (level - 1))) / price

    @classmethod
    def validate_config(cls, config, name: str) -> bool:
        """Validate ValueModeSizeRule configuration."""
        first_mult = config.first_safety_order_multiplier
        so_mult = config.so_size_multiplier

        if first_mult <= 0:
            raise ValueError(f"Invalid {name} configuration: first_safety_order_multiplier ({first_mult}) must be positive")
        if so_mult <= 0:
            raise ValueError(f"Invalid {name} configuration: so_size_multiplier ({so_mult}) must be positive")

        return True


class VolumeModeSizeRule(BaseSizeEngine):
    """
    Size_1   = Q_BO * first_mult
    Size_k>1 = Q_BO * first_mult * so_mult^(k-1)
    """

    def __init__(self, strategy):
        super().__init__(strategy)
        self.first_mult = self.config.first_safety_order_multiplier
        self.so_mult = self.config.so_size_multiplier

    def calculate_size(self, ctx: Ctx, price: float, level: int) -> float:
        """Calculate size using the volume-based sizing formula."""
        q0 = ctx.base_order_quantity
        if level == 1:
            return q0 * self.first_mult
        return q0 * self.first_mult * (self.so_mult ** (level - 1))

    @classmethod
    def validate_config(cls, config, name: str) -> bool:
        """Validate VolumeModeSizeRule configuration."""
        first_mult = config.first_safety_order_multiplier
        so_mult = config.so_size_multiplier

        if first_mult <= 0:
            raise ValueError(f"Invalid {name} configuration: first_safety_order_multiplier ({first_mult}) must be positive")
        if so_mult <= 0:
            raise ValueError(f"Invalid {name} configuration: so_size_multiplier ({so_mult}) must be positive")

        return True


class CompositeSizeEngine(BaseSizeEngine):
    """Composite size engine that combines results from multiple engines using min/max/avg operations."""

    def __init__(self, strategy, engines: List[BaseSizeEngine], operation: str):
        super().__init__(strategy)
        self.engines = engines
        self.operation = operation.lower()

        if self.operation not in ['min', 'max', 'avg']:
            raise ValueError(f"Invalid composite operation: {operation}. Must be min, max, or avg.")

        if not engines:
            raise ValueError("Composite engine requires at least one engine.")

    def calculate_size(self, ctx: Ctx, price: float, level: int) -> float:
        """Calculate size using composite operation on all engines."""
        sizes = []
        for engine in self.engines:
            size = engine.calculate_size(ctx, price, level)
            if size > 0:  # Only include valid sizes
                sizes.append(size)

        if not sizes:
            return 0.0

        if self.operation == 'min':
            return min(sizes)
        elif self.operation == 'max':
            return max(sizes)
        elif self.operation == 'avg':
            return sum(sizes) / len(sizes)

        return 0.0

    @classmethod
    def validate_config(cls, config, name: str) -> bool:
        """Composite engines are always valid."""
        return True


# For circular import avoidance, register size engines lazily (only when needed)
def _ensure_size_engines_registered():
    """Ensure size engines are registered with SizeFactory (called lazily to avoid circular imports)."""
    if "value_mode" not in SizeFactory._size_engines:
        SizeFactory.register_engine("value_mode", ValueModeSizeRule)
    if "volume_mode" not in SizeFactory._size_engines:
        SizeFactory.register_engine("volume_mode", VolumeModeSizeRule)

# Size engine registry for extensible selection (maintained for backward compatibility)
_SIZE_ENGINE_REGISTRY = {
    "value": lambda s: SizeFactory.create_engine("value_mode", s),
    "volume": lambda s: SizeFactory.create_engine("volume_mode", s),
}

def get_available_size_modes():
    """Return list of available size engine modes."""
    return list(_SIZE_ENGINE_REGISTRY.keys())

def create_size_engine(strategy, mode=None):
    """Create size engine with support for composite specifications.

    Supports both backward-compatible string format and new composite format:
    - String format: "value" or "volume"
    - Composite format: {"min": ["value", "volume"]}
    - Nested format: {"avg": ["value", {"min": ["volume", "custom"]}]}

    Args:
        strategy: The strategy instance
        mode: Optional mode override, defaults to strategy.config.safety_order_mode.
              Can be string or dict specification.

    Returns:
        SizeEngine instance

    Raises:
        ValueError: If an invalid mode is specified
    """
    # Ensure size engines are registered (lazy registration to avoid circular imports)
    _ensure_size_engines_registered()

    # Get mode specification, allowing for dict-based composite specs
    mode_spec = mode or strategy.config.safety_order_mode or "value"

    # Recursively create engine from specification
    return _create_size_engine_from_spec(strategy, mode_spec)


def _create_size_engine_from_spec(strategy, spec):
    """Recursively create size engine from specification (string or dict)."""
    if isinstance(spec, str):
        # Simple string specification - create individual engine
        spec_lower = spec.lower()
        if spec_lower not in _SIZE_ENGINE_REGISTRY:
            available_modes = get_available_size_modes()
            raise ValueError(
                f"Invalid size engine mode '{spec}'. "
                f"Available modes: {', '.join(available_modes)}"
            )
        return _SIZE_ENGINE_REGISTRY[spec_lower](strategy)

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
        engines = [_create_size_engine_from_spec(strategy, child_spec) for child_spec in engine_specs]

        # Create composite engine
        return CompositeSizeEngine(strategy, engines, operation)

    else:
        raise ValueError(f"Size engine specification must be string or dict, got: {type(spec)}")

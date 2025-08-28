# rules/sizing.py
from __future__ import annotations
from typing import Dict, Type, Any
from abc import ABC, abstractmethod
from ports import SizeEngine
from contracts import Ctx


class BaseSizeRule(ABC):
    """Base class for size engines following simplified Rule pattern."""

    def __init__(self, strategy, rule_name: str):
        self.strategy = strategy
        self.rule_name = rule_name
        self.config = strategy.config

    @abstractmethod
    def calculate_size(self, ctx: Ctx, price: float, level: int) -> float:
        """Calculate size for given context following Rule.evaluate pattern."""
        pass

    @classmethod
    @abstractmethod
    def validate_config(cls, config, rule_name: str) -> bool:
        """Validate configuration at initialization like rules pattern."""
        pass


class SizeFactory:
    """Factory for creating and validating size rule instances following RuleFactory pattern."""

    _size_rule_classes: Dict[str, Type[BaseSizeRule]] = {}

    @classmethod
    def register_size_rule(cls, rule_name: str, rule_class: Type[BaseSizeRule]):
        """Register a size rule class with the factory."""
        cls._size_rule_classes[rule_name] = rule_class

    @classmethod
    def create_size_rule(cls, rule_name: str, strategy) -> BaseSizeRule:
        """Create a size rule instance after validating its configuration."""
        if rule_name not in cls._size_rule_classes:
            raise ValueError(f"Unknown size rule class: {rule_name}")

        # Validate configuration at initialization
        cls.validate_size_config(rule_name, strategy.config)

        # Create rule instance
        return cls._size_rule_classes[rule_name](strategy, rule_name)

    @classmethod
    def validate_size_config(cls, rule_name: str, config) -> bool:
        """Validate size rule configuration during initialization."""
        if rule_name not in cls._size_rule_classes:
            raise ValueError(f"Cannot validate unknown size rule: {rule_name}")

        return cls._size_rule_classes[rule_name].validate_config(config, rule_name)


class ValueModeSizeRule(BaseSizeRule, SizeEngine):
    """
    Size_1   = (V_BO * first_mult) / P_1
    Size_k>1 = (V_BO * first_mult * so_mult^(k-1)) / P_k
    """

    def __init__(self, strategy, rule_name='value_mode'):
        BaseSizeRule.__init__(self, strategy, rule_name)
        SizeEngine.__init__(self)
        self.first_mult = self.config.get_rule_param(
            rule_name, 'first_safety_order_multiplier',
            getattr(self.strategy, 'first_safety_order_multiplier', 1.0)
        )
        self.so_mult = self.config.get_rule_param(
            rule_name, 'so_size_multiplier',
            getattr(self.strategy, 'so_size_multiplier', 1.0)
        )

    def calculate_size(self, ctx: Ctx, price: float, level: int) -> float:
        """Calculate size using the value-based sizing formula."""
        v0 = ctx.base_order_value
        if level == 1:
            return (v0 * self.first_mult) / price
        return (v0 * self.first_mult * (self.so_mult ** (level - 1))) / price

    # Maintain backward compatibility with SizeEngine interface
    def so_size(self, ctx: Ctx, price: float, level: int) -> float:
        return self.calculate_size(ctx, price, level)

    @classmethod
    def validate_config(cls, config, rule_name: str) -> bool:
        """Validate ValueModeSizeRule configuration."""
        first_mult = config.get_rule_param(rule_name, 'first_safety_order_multiplier', 1.0)
        so_mult = config.get_rule_param(rule_name, 'so_size_multiplier', 1.0)

        if first_mult <= 0:
            raise ValueError(f"Invalid {rule_name} configuration: first_safety_order_multiplier ({first_mult}) must be positive")
        if so_mult <= 0:
            raise ValueError(f"Invalid {rule_name} configuration: so_size_multiplier ({so_mult}) must be positive")

        return True


class VolumeModeSizeRule(BaseSizeRule, SizeEngine):
    """
    Size_1   = Q_BO * first_mult
    Size_k>1 = Q_BO * first_mult * so_mult^(k-1)
    """

    def __init__(self, strategy, rule_name='volume_mode'):
        BaseSizeRule.__init__(self, strategy, rule_name)
        SizeEngine.__init__(self)
        self.first_mult = self.config.get_rule_param(
            rule_name, 'first_safety_order_multiplier',
            getattr(self.strategy, 'first_safety_order_multiplier', 1.0)
        )
        self.so_mult = self.config.get_rule_param(
            rule_name, 'so_size_multiplier',
            getattr(self.strategy, 'so_size_multiplier', 1.0)
        )

    def calculate_size(self, ctx: Ctx, price: float, level: int) -> float:
        """Calculate size using the volume-based sizing formula."""
        q0 = ctx.base_order_quantity
        if level == 1:
            return q0 * self.first_mult
        return q0 * self.first_mult * (self.so_mult ** (level - 1))

    # Maintain backward compatibility with SizeEngine interface
    def so_size(self, ctx: Ctx, price: float, level: int) -> float:
        return self.calculate_size(ctx, price, level)

    @classmethod
    def validate_config(cls, config, rule_name: str) -> bool:
        """Validate VolumeModeSizeRule configuration."""
        first_mult = config.get_rule_param(rule_name, 'first_safety_order_multiplier', 1.0)
        so_mult = config.get_rule_param(rule_name, 'so_size_multiplier', 1.0)

        if first_mult <= 0:
            raise ValueError(f"Invalid {rule_name} configuration: first_safety_order_multiplier ({first_mult}) must be positive")
        if so_mult <= 0:
            raise ValueError(f"Invalid {rule_name} configuration: so_size_multiplier ({so_mult}) must be positive")

        return True


# Register size rules with factory (following RuleFactory pattern)
SizeFactory.register_size_rule("value_mode", ValueModeSizeRule)
SizeFactory.register_size_rule("volume_mode", VolumeModeSizeRule)

# Size engine registry for extensible selection (maintained for backward compatibility)
_SIZE_ENGINE_REGISTRY = {
    "value": lambda s: SizeFactory.create_size_rule("value_mode", s),
    "volume": lambda s: SizeFactory.create_size_rule("volume_mode", s),
}

def get_available_size_modes():
    """Return list of available size engine modes."""
    return list(_SIZE_ENGINE_REGISTRY.keys())

def create_size_engine(strategy, mode=None):
    """Create size engine with strict validation using SizeFactory.

    Args:
        strategy: The strategy instance
        mode: Optional mode override, defaults to strategy.config.safety_order_mode

    Returns:
        SizeEngine instance created via SizeFactory with validation

    Raises:
        ValueError: If an invalid mode is specified
    """
    mode = (mode or strategy.config.safety_order_mode or "value").lower()

    if mode not in _SIZE_ENGINE_REGISTRY:
        available_modes = get_available_size_modes()
        raise ValueError(
            f"Invalid size engine mode '{mode}'. "
            f"Available modes: {', '.join(available_modes)}"
        )

    return _SIZE_ENGINE_REGISTRY[mode](strategy)

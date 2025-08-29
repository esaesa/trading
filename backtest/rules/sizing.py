# rules/sizing.py
from __future__ import annotations
from typing import Dict, Type, Any, List
from abc import ABC, abstractmethod
from ports import SizeEngine
from contracts import Ctx
import statistics # <-- Add this import

# --- NEW: Size Engine Factory ---
class SizeEngineFactory:
    _engine_classes: Dict[str, Type[SizeEngine]] = {}
    
    @classmethod
    def register(cls, name: str, engine_class: Type[SizeEngine]):
        cls._engine_classes[name] = engine_class

    @classmethod
    def create(cls, name: str, strategy: Any) -> SizeEngine:
        if name not in cls._engine_classes:
            available = ', '.join(cls._engine_classes.keys())
            raise ValueError(f"Unknown SizeEngine '{name}'. Available engines: {available}")
        return cls._engine_classes[name](strategy)

# --- Concrete Engine Implementations (No logic changes needed) ---
class BaseSizeEngine(SizeEngine, ABC):
    def __init__(self, strategy):
        self.strategy = strategy
        self.config = strategy.config
    @abstractmethod
    def so_size(self, ctx: Ctx, price: float, level: int) -> float: ...

class ValueModeSizeRule(BaseSizeEngine):
    def so_size(self, ctx: Ctx, price: float, level: int) -> float:
        v0 = ctx.base_order_value
        first_mult = self.config.first_safety_order_multiplier
        so_mult = self.config.so_size_multiplier
        if level == 1:
            return (v0 * first_mult) / price
        return (v0 * first_mult * (so_mult ** (level - 1))) / price

class VolumeModeSizeRule(BaseSizeEngine):
    def so_size(self, ctx: Ctx, price: float, level: int) -> float:
        q0 = ctx.base_order_quantity
        first_mult = self.config.first_safety_order_multiplier
        so_mult = self.config.so_size_multiplier
        if level == 1:
            return q0 * first_mult
        return q0 * first_mult * (so_mult ** (level - 1))

# --- CORRECTED: Register engines using their class names ---
SizeEngineFactory.register("ValueModeSizeRule", ValueModeSizeRule)
SizeEngineFactory.register("VolumeModeSizeRule", VolumeModeSizeRule)

# --- Composite Engine ---
class CompositeSizeEngine(SizeEngine):
    def __init__(self, strategy: Any, sub_engines: list[SizeEngine], mode: str):
        if mode not in ["min", "max", "avg"]: raise ValueError(f"Invalid composite mode: {mode}")
        if not sub_engines: raise ValueError("CompositeSizeEngine requires at least one sub-engine.")
        self.s = strategy
        self.sub_engines = sub_engines
        self.mode = mode
        self.op = {"min": min, "max": max, "avg": statistics.mean}[mode]
    def so_size(self, ctx: Ctx, price: float, level: int) -> float:
        sizes = [engine.so_size(ctx, price, level) for engine in self.sub_engines]
        valid_sizes = [s for s in sizes if s > 0]
        return self.op(valid_sizes) if valid_sizes else 0.0

# --- CORRECTED: Simplified, recursive builder function ---
def create_size_engine(strategy: Any, config_spec: Any) -> SizeEngine:
    """
    Recursively builds a SizeEngine from a configuration specification.
    """
    if isinstance(config_spec, str):
        return SizeEngineFactory.create(config_spec, strategy)

    if isinstance(config_spec, dict):
        if len(config_spec) != 1:
            raise ValueError("Composite engine spec must have exactly one key (min, max, or avg).")
        
        mode, sub_specs = next(iter(config_spec.items()))
        
        if not isinstance(sub_specs, list):
            raise ValueError(f"Value for '{mode}' must be a list of engine specifications.")

        sub_engines = [create_size_engine(strategy, spec) for spec in sub_specs]
        
        return CompositeSizeEngine(strategy, sub_engines, mode)
        
    raise TypeError(f"Invalid size engine configuration type: {type(config_spec)}")
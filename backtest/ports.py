# ports.py
from __future__ import annotations
from typing import Protocol, runtime_checkable
from contracts import Ctx

@runtime_checkable
class EntryDecider(Protocol):
    def ok(self, ctx: Ctx) -> bool: ...

@runtime_checkable
class SafetyDecider(Protocol):
    def ok(self, ctx: Ctx) -> bool: ...

@runtime_checkable
class ExitDecider(Protocol):
    def ok(self, ctx: Ctx) -> bool: ...

@runtime_checkable
class PriceEngine(Protocol):
    def so_price(self, ctx: Ctx, level: int) -> float: ...

@runtime_checkable
class SizeEngine(Protocol):
    def so_size(self, ctx: Ctx, price: float, level: int) -> float: ...

# wiring.py
from __future__ import annotations
from typing import Any
from order_alloc import (
    FractionalCashEntrySizer, EmaEntryMultiplier, FixedEntryMultiplier,
    DefaultAffordabilityGuard
)
from commissions import FixedRateCommission
from ports import PriceEngine, SizeEngine
from rules.entry import EntryRuleDecider
from rules.exit import ExitRuleDecider
from rules.safety import SafetyRuleDecider
from indicators_provider import StrategyIndicatorProvider
from position_view import BacktestingPositionView
from preflight import EntryPreflight
# --- CORRECTED: Import the builder functions directly ---
from rules.pricing import create_price_engine
from rules.sizing import create_size_engine

# --- NO CHANGES NEEDED HERE ---
def _apply_default_rules(owner, params: dict) -> dict:
    cfg = dict(params or {})
    default_entry = ["EntryFundsAndNotional"]
    if owner.config.avoid_rsi_overbought:
        default_entry.insert(0, "RSIOverboughtGate")
    cfg.setdefault("entry_conditions", default_entry)
    cfg.setdefault("safety_conditions", ["RSIUnderDynamicThreshold", "MaxLevelsNotReached"])
    cfg.setdefault("exit_conditions", ["TPDecayReached"])
    return cfg

def wire_strategy(strategy: Any, strategy_params: dict) -> None:
    """
    Attach deciders/engines to the strategy instance.
    """
    cfg = _apply_default_rules(strategy, strategy_params)
    
    # Decision Rule Wiring (This part was already good)
    strategy.entry_decider  = EntryRuleDecider(strategy, cfg.get("entry_conditions"))
    strategy.safety_decider = SafetyRuleDecider(strategy, cfg.get("safety_conditions"), default_mode="all")
    strategy.exit_decider   = ExitRuleDecider(strategy, cfg.get("exit_conditions"))

    # --- CORRECTED: Engine wiring now passes the full config spec ---
    price_spec = strategy_params.get("safety_order_price_mode", "StaticPriceEngine")
    strategy.price_engine = create_price_engine(strategy, price_spec)

    size_spec = strategy_params.get("safety_order_mode", "ValueModeSizeRule")
    strategy.size_engine = create_size_engine(strategy, size_spec)

    # --- NO CHANGES NEEDED BELOW THIS LINE ---
    # Entry sizing policy
    try:
        ema_test = strategy.indicator_service.get_indicator_value("ema", strategy.data.index[0], None)
        entry_mult = EmaEntryMultiplier(mult_above=2, default=1) if ema_test is not None else FixedEntryMultiplier(1)
    except Exception:
        entry_mult = FixedEntryMultiplier(1)
    
    strategy.entry_sizer = FractionalCashEntrySizer(
        entry_fraction=strategy.config.entry_fraction,
        multiplier=entry_mult,
    )
    strategy.entry_sizer.indicator_service = strategy.indicator_service
    
    # Other policies
    strategy.commission_calc = FixedRateCommission(rate=getattr(strategy, "commission_rate", 0.0))
    strategy.affordability_guard = DefaultAffordabilityGuard()
    strategy.position_view = BacktestingPositionView(
        last_price_fn=lambda: strategy._broker.last_price,
        position_fn=lambda: strategy.position,
    )
    strategy.indicator_provider = StrategyIndicatorProvider(strategy)
    strategy.entry_preflight = EntryPreflight(strategy.entry_sizer, strategy.commission_calc)
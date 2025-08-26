# rules/safety.py
from datetime import timedelta
import math
from typing import Tuple, Set
import numpy as np
from contracts import Ctx
from logger_config import logger
from indicators import Indicators

# Indicator requirements for each rule function
RSI_UNDER_STATIC_THRESHOLD_INDICATORS = {Indicators.RSI.value, Indicators.DYNAMIC_RSI_THRESHOLD.value}

def rsi_under_dynamic_threshold(self, ctx: Ctx) -> Tuple[bool, str]:
    """
    Allow DCA if RSI is below the chosen threshold.
       - If rsi_dynamic_threshold=True and dynamic_rsi_threshold_series is available,
         use that; otherwise fall back to self.rsi_threshold.
       - If RSI is NaN, we allow (existing behavior).
    """
    rsi_val = self.indicator_service.get_indicator_value("rsi", ctx.now, np.nan)

    # Get dynamic RSI threshold directly as a regular indicator
    dyn_thr = self.indicator_service.get_indicator_value("dynamic_rsi_threshold", ctx.now, np.nan)
    if np.isnan(dyn_thr):
        return False, "Dyn RSI NaN → deny"

    # Get rule-specific parameters from config
    use_dyn = self.config.get_rule_param(
        'RSIUnderDynamicThreshold', 'dynamic_threshold', self.config.rsi_dynamic_threshold
    ) and dyn_thr is not None and not np.isnan(dyn_thr)

    base_threshold = float(dyn_thr) if use_dyn else float(self.config.get_rule_param(
        'RSIUnderDynamicThreshold', 'threshold', self.config.rsi_threshold
    ) or 50.0)

    need_rsi_reset = self.config.get_rule_param(
        'RSIUnderDynamicThreshold', 'need_rsi_reset', False
    )
    reset_threshold = self.config.get_rule_param(
        'RSIUnderDynamicThreshold', 'reset_threshold', 50
    )

    threshold = base_threshold

    if np.isnan(rsi_val):
        return False, "RSI NaN → deny"

    # RSI Wave Reset Logic (if enabled)
    if need_rsi_reset:
        # Initialize wave state if not exists
        if not hasattr(self, 'rsi_wave_available_dynamic'):
            self.rsi_wave_available_dynamic = True

        # Use static reset threshold
        reset_thr = reset_threshold

        # If RSI goes above reset threshold, make wave available
        if rsi_val >= reset_thr:
            self.rsi_wave_available_dynamic = True
            return False, f"RSI {rsi_val:.2f} ≥ reset threshold {reset_thr:.2f} → wave available"

        # If RSI is below trading threshold and wave is available, allow trade
        if rsi_val < threshold and self.rsi_wave_available_dynamic:
            self.rsi_wave_available_dynamic = False  # Mark wave as used
            return True, f"RSI {rsi_val:.2f} < trading threshold {threshold:.2f} → wave used"

        # Otherwise, don't allow trade
        if rsi_val < threshold:
            return False, f"RSI {rsi_val:.2f} < trading threshold {threshold:.2f} but wave not available"
        else:
            return False, f"RSI {rsi_val:.2f} ≥ trading threshold {threshold:.2f}"

    # Standard logic (no wave reset)
    ok = rsi_val < threshold
    if getattr(self, "debug_trade", False) and not ok:
        level = ctx.config.get("next_level", ctx.dca_level + 1)
        logger.debug(f"RSI={rsi_val:.2f} not below threshold={threshold:.2f}, skipping DCA-{level}")

    return ok, f"RSI {rsi_val:.2f} < thr {threshold:.2f}" if ok else f"RSI {rsi_val:.2f} ≥ thr {threshold:.2f}"

# Indicator requirements for each rule function
RSI_UNDER_STATIC_THRESHOLD_INDICATORS = {Indicators.RSI.value}

def rsi_under_static_threshold(self, ctx: Ctx) -> Tuple[bool, str]:
    rsi_val = self.indicator_service.get_indicator_value("rsi", ctx.now, np.nan)

    # Get rule-specific parameters from config
    threshold = self.config.get_rule_param(
        'RSIUnderStaticThreshold', 'static_threshold_under', self.config.rsi_static_threshold_under
    )
    need_rsi_reset = self.config.get_rule_param(
        'RSIUnderStaticThreshold', 'need_rsi_reset', False
    )
    reset_threshold = self.config.get_rule_param(
        'RSIUnderStaticThreshold', 'reset_threshold', 50
    )

    if np.isnan(rsi_val):
        return True, "RSI NaN → allow"

    # RSI Wave Reset Logic (if enabled)
    if need_rsi_reset:
        # Initialize wave state if not exists
        if not hasattr(self, 'rsi_wave_available_static'):
            self.rsi_wave_available_static = True

        # Use static reset threshold
        reset_thr = reset_threshold

        # If RSI goes above reset threshold, make wave available
        if rsi_val >= reset_thr:
            self.rsi_wave_available_static = True
            return False, f"RSI {rsi_val:.2f} ≥ reset threshold {reset_thr:.2f} → wave available"

        # If RSI is below trading threshold and wave is available, allow trade
        if rsi_val < threshold and self.rsi_wave_available_static:
            self.rsi_wave_available_static = False  # Mark wave as used
            return True, f"RSI {rsi_val:.2f} < trading threshold {threshold:.2f} → wave used"

        # Otherwise, don't allow trade
        if rsi_val < threshold:
            return False, f"RSI {rsi_val:.2f} < trading threshold {threshold:.2f} but wave not available"
        else:
            return False, f"RSI {rsi_val:.2f} ≥ trading threshold {threshold:.2f}"

    # Standard logic (no wave reset)
    ok = rsi_val < threshold
    if getattr(self, "debug_trade", False) and not ok:
        level = ctx.config.get("next_level", ctx.dca_level + 1)
        logger.debug(f"RSI={rsi_val:.2f} not below threshold={threshold:.2f}, skipping DCA-{level}")

    return ok, f"RSI {rsi_val:.2f} < thr {threshold:.2f}" if ok else f"RSI {rsi_val:.2f} ≥ thr {threshold:.2f}"

# Indicator requirements for each rule function
COOLDOWN_BETWEEN_SOS_INDICATORS = set()  # No indicators needed

def cooldown_between_sos(self, ctx: Ctx) -> Tuple[bool, str]:
    """
    Enforce a minimum elapsed time between BO/SO and the next SO.
    """
    mins = int(getattr(self, "so_cooldown_minutes", 0) or 0)
    if mins <= 0:
        return True, "No cooldown"

    now = ctx.now
    last = getattr(self, "last_safety_order_time", None) or getattr(self, "base_order_time", None)
    if last is None:
        return True, "No previous SO/BO time"

    if now - last >= timedelta(minutes=mins):
        return True, f"Cooldown {mins}m elapsed"

    if getattr(self, "debug_trade", False):
        level = ctx.config.get("next_level", ctx.dca_level + 1)
        logger.debug(f"Skip DCA-{level}: cooldown {(now - last)} < {mins}m")
    return False, "Cooldown not elapsed"


# Indicator requirements for each rule function
MAX_LEVELS_NOT_REACHED_INDICATORS = set()  # No indicators needed

def max_levels_not_reached(self, ctx: Ctx) -> Tuple[bool, str]:
    """
    Allow SO only if current DCA level < max_dca_levels.
    Mirrors the old imperative check in strategy.process_dca.
    """
    max_levels = self.config.max_dca_levels
    ok = ctx.dca_level < max_levels
    return ok, (
        f"Level {ctx.dca_level} < max {max_levels}"
        if ok else f"Reached max levels ({max_levels})"
    )
# Indicator requirements for each rule function
SUFFICIENT_FUNDS_AND_NOTIONAL_INDICATORS = set()  # No indicators needed

def sufficient_funds_and_notional(self, ctx: Ctx):
    """
    Allow SO only if there exists an integer quantity q we can afford (after commission)
    such that ORDER notional q * P_so >= minimum_notional, with q <= max_possible.
    If next_so_price is unknown, be lenient and allow.
    """
    P_so = ctx.next_so_price
    if P_so is None or P_so <= 0:
        return True, "SO price unknown → allow"

    c = float(ctx.config.get("commission_rate", 0.0) or 0.0)
    cost_per_share = P_so * (1.0 + c)
    if cost_per_share <= 0:
        return True, "Invalid cost/share → allow"

    max_possible = int(ctx.available_cash // cost_per_share)

    min_notional = float(getattr(self, "minimum_notional", 0.0) or 0.0)
    required_qty = max(1, math.ceil(min_notional / P_so))

    if max_possible < required_qty:
        return False, (
            f"Insufficient funds for min order: cash={ctx.available_cash:.2f}, "
            f"cost/share={cost_per_share:.6f}, max_possible={max_possible}, "
            f"required_qty={required_qty}, min_notional={min_notional:.6f}"
        )

    return True, f"Funds OK (max_possible≥{required_qty}) & order notional OK (≥{min_notional:.6f})"



SAFETY_RULES = {
    "RSIUnderDynamicThreshold": rsi_under_dynamic_threshold,
    "CooldownBetweenSOs": cooldown_between_sos,
    "MaxLevelsNotReached": max_levels_not_reached,
    "SufficientFundsAndNotional": sufficient_funds_and_notional,
    "RSIUnderStaticThreshold": rsi_under_static_threshold
}

# Keep the decider as-is
from ports import SafetyDecider
from rule_chain import build_rule_chain

class SafetyRuleDecider(SafetyDecider):
    """
    Builds its own RuleChain from config (strings or nested ANY/ALL dicts).
    """
    def __init__(self, strategy, names, default_mode: str = "any") -> None:
        self._chain = build_rule_chain(strategy, names, SAFETY_RULES, mode=default_mode)  # type: ignore[arg-type]
        # Store the rule names for indicator detection
        self._rule_names = self._extract_rule_names(names)
        # Store the rules config for dynamic indicator detection
        self._rules_config = getattr(strategy.config, '_rules_config', {})

    def _extract_rule_names(self, spec) -> set[str]:
        """Extract all rule names from the spec."""
        names = set()
        if isinstance(spec, str):
            names.add(spec)
        elif isinstance(spec, (list, tuple)):
            for item in spec:
                names.update(self._extract_rule_names(item))
        elif isinstance(spec, dict):
            for key, value in spec.items():
                if key in ("any", "all"):
                    names.update(self._extract_rule_names(value))
        return names

    def ok(self, ctx):
        return self._chain.ok(ctx)

    def get_required_indicators(self) -> set[str]:
        """Return set of indicators required by all rules in this decider."""
        required = set()

        # Use the stored rule names from the configuration
        for rule_name in self._rule_names:
            if rule_name == "RSIUnderDynamicThreshold":
                # Dynamic indicator requirements based on configuration
                required.add(Indicators.RSI.value)
                # Only require dynamic threshold if actually used
                if self._get_rule_param(rule_name, 'dynamic_threshold', False):
                    required.add(Indicators.DYNAMIC_RSI_THRESHOLD.value)
            elif rule_name == "RSIUnderStaticThreshold":
                required.update(RSI_UNDER_STATIC_THRESHOLD_INDICATORS)
            elif rule_name == "CooldownBetweenSOs":
                required.update(COOLDOWN_BETWEEN_SOS_INDICATORS)
            elif rule_name == "MaxLevelsNotReached":
                required.update(MAX_LEVELS_NOT_REACHED_INDICATORS)
            elif rule_name == "SufficientFundsAndNotional":
                required.update(SUFFICIENT_FUNDS_AND_NOTIONAL_INDICATORS)

        return required

    def _get_rule_param(self, rule_name: str, param_name: str, default_value: any) -> any:
        """Get a parameter from rule-specific configuration."""
        if rule_name in self._rules_config and param_name in self._rules_config[rule_name]:
            return self._rules_config[rule_name][param_name]
        return default_value

# rules/safety.py
from datetime import timedelta
import math
from typing import Tuple, Set
import numpy as np
from contracts import Ctx
from logger_config import logger
from indicators import Indicators



# Import for class-based rules
from .base_rule import Rule

# Class-based RSIUnderDynamicThreshold rule
class RSIUnderDynamicThreshold(Rule):
    """RSI under dynamic threshold rule with wave reset capability."""

    def __init__(self, strategy, rule_name='RSIUnderDynamicThreshold'):
        super().__init__(strategy, rule_name)
        self.dynamic_threshold = self.config.get_rule_param(
            rule_name, 'dynamic_threshold', self.config.rsi_dynamic_threshold
        )
        self.threshold = self.config.get_rule_param(
            rule_name, 'threshold', self.config.rsi_threshold
        )
        self.need_rsi_reset = self.config.get_rule_param(
            rule_name, 'need_rsi_reset', False
        )
        self.reset_threshold = self.config.get_rule_param(
            rule_name, 'reset_threshold', 50
        )

    def evaluate(self, ctx: Ctx) -> Tuple[bool, str]:
        rsi_val = self.strategy.indicator_service.get_indicator_value("rsi", ctx.now, np.nan)

        # Get dynamic RSI threshold directly as a regular indicator
        dyn_thr = self.strategy.indicator_service.get_indicator_value("dynamic_rsi_threshold", ctx.now, np.nan)
        if np.isnan(dyn_thr):
            return False, "Dyn RSI NaN → deny"

        # Get rule-specific parameters from config
        use_dyn = self.dynamic_threshold and dyn_thr is not None and not np.isnan(dyn_thr)

        base_threshold = float(dyn_thr) if use_dyn else float(self.threshold or 50.0)

        threshold = base_threshold

        if np.isnan(rsi_val):
            return False, "RSI NaN → deny"

        # RSI Wave Reset Logic (if enabled)
        if self.need_rsi_reset:
            # Initialize wave state if not exists
            if not hasattr(self.strategy, 'rsi_wave_available_dynamic'):
                self.strategy.rsi_wave_available_dynamic = True

            # Use static reset threshold
            reset_thr = self.reset_threshold

            # If RSI goes above reset threshold, make wave available
            if rsi_val >= reset_thr:
                self.strategy.rsi_wave_available_dynamic = True
                return False, f"RSI {rsi_val:.2f} ≥ reset threshold {reset_thr:.2f} → wave available"

            # If RSI is below trading threshold and wave is available, allow trade
            if rsi_val < threshold and self.strategy.rsi_wave_available_dynamic:
                self.strategy.rsi_wave_available_dynamic = False  # Mark wave as used
                return True, f"RSI {rsi_val:.2f} < trading threshold {threshold:.2f} → wave used"

            # Otherwise, don't allow trade
            if rsi_val < threshold:
                return False, f"RSI {rsi_val:.2f} < trading threshold {threshold:.2f} but wave not available"
            else:
                return False, f"RSI {rsi_val:.2f} ≥ trading threshold {threshold:.2f}"

        # Standard logic (no wave reset)
        ok = rsi_val < threshold
        if getattr(self.strategy, "debug_trade", False) and not ok:
            level = ctx.config.get("next_level", ctx.dca_level + 1)
            logger.debug(f"RSI={rsi_val:.2f} not below threshold={threshold:.2f}, skipping DCA-{level}")

        return ok, f"RSI {rsi_val:.2f} < thr {threshold:.2f}" if ok else f"RSI {rsi_val:.2f} ≥ thr {threshold:.2f}"

    def get_required_indicators(self) -> Set[str]:
        return {Indicators.RSI.value, Indicators.DYNAMIC_RSI_THRESHOLD.value}

    @classmethod
    def validate_config(cls, config, rule_name: str) -> bool:
        """Validate RSIUnderDynamicThreshold configuration."""
        need_rsi_reset = config.get_rule_param(rule_name, 'need_rsi_reset', False)
        if not need_rsi_reset:
            return True

        threshold = config.get_rule_param(rule_name, 'threshold', config.rsi_threshold)
        reset_threshold = config.get_rule_param(rule_name, 'reset_threshold', 50)

        if reset_threshold < threshold:
            raise ValueError(
                f"Invalid {rule_name} configuration: "
                f"reset_threshold ({reset_threshold}) must be >= threshold ({threshold}) "
                f"when need_rsi_reset is enabled"
            )
        return True

# Class-based RSIReversalStaticThreshold rule
class RSIReversalStaticThreshold(Rule):
    """RSI reversal from static threshold rule."""

    def __init__(self, strategy, rule_name='RSIReversalStaticThreshold'):
        super().__init__(strategy, rule_name)
        self.threshold = self.config.get_rule_param(
            rule_name, 'threshold', 20
        )
        self.bounce_threshold = self.config.get_rule_param(
            rule_name, 'bounce_threshold', 5
        )

    def evaluate(self, ctx: Ctx) -> Tuple[bool, str]:
        rsi_val = self.strategy.indicator_service.get_indicator_value("rsi", ctx.now, np.nan)

        if np.isnan(rsi_val):
            return False, "RSI NaN → deny"

        # Initialize tracking state if not exists
        if not hasattr(self.strategy, 'rsi_reversal_lowest'):
            self.strategy.rsi_reversal_lowest = None
            self.strategy.rsi_reversal_active = False

        # If RSI is above threshold, reset tracking and return false
        if rsi_val >= self.threshold:
            self.strategy.rsi_reversal_lowest = None
            self.strategy.rsi_reversal_active = False
            return False, f"RSI {rsi_val:.2f} ≥ threshold {self.threshold:.2f} → reset tracking"

        # RSI is below threshold - start/restart tracking
        if not self.strategy.rsi_reversal_active:
            # Start tracking from current RSI value
            self.strategy.rsi_reversal_lowest = rsi_val
            self.strategy.rsi_reversal_active = True
            return False, f"RSI {rsi_val:.2f} < threshold {self.threshold:.2f} → start tracking"
        else:
            # Update lowest RSI if current is lower
            if rsi_val < self.strategy.rsi_reversal_lowest:
                self.strategy.rsi_reversal_lowest = rsi_val
                return False, f"RSI {rsi_val:.2f} < lowest {self.strategy.rsi_reversal_lowest:.2f} → update lowest"

            # Check if RSI has bounced enough from lowest
            bounce_target = self.strategy.rsi_reversal_lowest + self.bounce_threshold
            if rsi_val >= bounce_target:
                # Store values before resetting for the message
                lowest_val = self.strategy.rsi_reversal_lowest
                # Reset for next potential reversal
                self.strategy.rsi_reversal_lowest = None
                self.strategy.rsi_reversal_active = False
                return True, f"RSI {rsi_val:.2f} ≥ lowest {lowest_val:.2f} + {self.bounce_threshold:.2f} → bounce detected"

            return False, f"RSI {rsi_val:.2f} < lowest {self.strategy.rsi_reversal_lowest:.2f} + {self.bounce_threshold:.2f} → waiting for bounce"

    def get_required_indicators(self) -> Set[str]:
        return {Indicators.RSI.value}

    @classmethod
    def validate_config(cls, config, rule_name: str) -> bool:
        """Validate RSIReversalStaticThreshold configuration."""
        threshold = config.get_rule_param(rule_name, 'threshold', 20)
        bounce_threshold = config.get_rule_param(rule_name, 'bounce_threshold', 5)

        if bounce_threshold <= 0:
            raise ValueError(
                f"Invalid {rule_name} configuration: "
                f"bounce_threshold ({bounce_threshold}) must be > 0"
            )
        return True

# Class-based RSIUnderStaticThreshold rule
class RSIUnderStaticThreshold(Rule):
    """RSI under static threshold rule with wave reset capability."""

    def __init__(self, strategy, rule_name='RSIUnderStaticThreshold'):
        super().__init__(strategy, rule_name)
        self.threshold = self.config.get_rule_param(
            rule_name, 'static_threshold_under', self.config.rsi_static_threshold_under
        )
        self.need_rsi_reset = self.config.get_rule_param(
            rule_name, 'need_rsi_reset', False
        )
        self.reset_threshold = self.config.get_rule_param(
            rule_name, 'reset_threshold', 50
        )
    
    def evaluate(self, ctx: Ctx) -> Tuple[bool, str]:
        rsi_val = self.strategy.indicator_service.get_indicator_value("rsi", ctx.now, np.nan)

        if np.isnan(rsi_val):
            return True, "RSI NaN → allow"

        # RSI Wave Reset Logic (if enabled)
        if self.need_rsi_reset:
            # Initialize wave state if not exists
            if not hasattr(self.strategy, 'rsi_wave_available_static'):
                self.strategy.rsi_wave_available_static = True

            # Use static reset threshold
            reset_thr = self.reset_threshold

            # If RSI goes above reset threshold, make wave available
            if rsi_val >= reset_thr:
                self.strategy.rsi_wave_available_static = True
                return False, f"RSI {rsi_val:.2f} ≥ reset threshold {reset_thr:.2f} → wave available"

            # If RSI is below trading threshold and wave is available, allow trade
            if rsi_val < self.threshold and self.strategy.rsi_wave_available_static:
                self.strategy.rsi_wave_available_static = False  # Mark wave as used
                return True, f"RSI {rsi_val:.2f} < trading threshold {self.threshold:.2f} → wave used"

            # Otherwise, don't allow trade
            if rsi_val < self.threshold:
                return False, f"RSI {rsi_val:.2f} < trading threshold {self.threshold:.2f} but wave not available"
            else:
                return False, f"RSI {rsi_val:.2f} ≥ trading threshold {self.threshold:.2f}"

        # Standard logic (no wave reset)
        ok = rsi_val < self.threshold
        if getattr(self.strategy, "debug_trade", False) and not ok:
            level = ctx.config.get("next_level", ctx.dca_level + 1)
            logger.debug(f"RSI={rsi_val:.2f} not below threshold={self.threshold:.2f}, skipping DCA-{level}")

        return ok, f"RSI {rsi_val:.2f} < thr {self.threshold:.2f}" if ok else f"RSI {rsi_val:.2f} ≥ thr {self.threshold:.2f}"
    
    def get_required_indicators(self) -> Set[str]:
        return {Indicators.RSI.value}
    
    @classmethod
    def validate_config(cls, config, rule_name: str) -> bool:
        """Validate RSIUnderStaticThreshold configuration."""
        need_rsi_reset = config.get_rule_param(rule_name, 'need_rsi_reset', False)
        if not need_rsi_reset:
            return True
        
        threshold = config.get_rule_param(rule_name, 'static_threshold_under', 
                                        config.rsi_static_threshold_under)
        reset_threshold = config.get_rule_param(rule_name, 'reset_threshold', 50)
        
        if reset_threshold < threshold:
            raise ValueError(
                f"Invalid {rule_name} configuration: "
                f"reset_threshold ({reset_threshold}) must be >= static_threshold_under ({threshold}) "
                f"when need_rsi_reset is enabled"
            )
        return True

# Class-based RSIReversalDynamicThreshold rule
class RSIReversalDynamicThreshold(Rule):
    """RSI reversal from dynamic threshold rule."""

    def __init__(self, strategy, rule_name='RSIReversalDynamicThreshold'):
        super().__init__(strategy, rule_name)
        self.bounce_threshold = self.config.get_rule_param(
            rule_name, 'bounce_threshold', 5
        )

    def evaluate(self, ctx: Ctx) -> Tuple[bool, str]:
        rsi_val = self.strategy.indicator_service.get_indicator_value("rsi", ctx.now, np.nan)

        # Get dynamic RSI threshold
        dyn_thr = self.strategy.indicator_service.get_indicator_value("dynamic_rsi_threshold", ctx.now, np.nan) * 1.2
        if np.isnan(dyn_thr):
            return False, "Dynamic RSI threshold NaN → deny"

        if np.isnan(rsi_val):
            return False, "RSI NaN → deny"

        # Initialize tracking state if not exists
        if not hasattr(self.strategy, 'rsi_reversal_dynamic_lowest'):
            self.strategy.rsi_reversal_dynamic_lowest = None
            self.strategy.rsi_reversal_dynamic_active = False

        # If RSI is above dynamic threshold, reset tracking and return false
        if rsi_val >= dyn_thr:
            self.strategy.rsi_reversal_dynamic_lowest = None
            self.strategy.rsi_reversal_dynamic_active = False
            return False, f"RSI {rsi_val:.2f} ≥ dynamic threshold {dyn_thr:.2f} → reset tracking"

        # RSI is below dynamic threshold - start/restart tracking
        if not self.strategy.rsi_reversal_dynamic_active:
            # Start tracking from current RSI value
            self.strategy.rsi_reversal_dynamic_lowest = rsi_val
            self.strategy.rsi_reversal_dynamic_active = True
            return False, f"RSI {rsi_val:.2f} < dynamic threshold {dyn_thr:.2f} → start tracking"
        else:
            # Update lowest RSI if current is lower
            if rsi_val < self.strategy.rsi_reversal_dynamic_lowest:
                self.strategy.rsi_reversal_dynamic_lowest = rsi_val
                return False, f"RSI {rsi_val:.2f} < lowest {self.strategy.rsi_reversal_dynamic_lowest:.2f} → update lowest"

            # Check if RSI has bounced enough from lowest
            bounce_target = self.strategy.rsi_reversal_dynamic_lowest + self.bounce_threshold
            if rsi_val >= bounce_target:
                # Store values before resetting for the message
                lowest_val = self.strategy.rsi_reversal_dynamic_lowest
                # Reset for next potential reversal
                self.strategy.rsi_reversal_dynamic_lowest = None
                self.strategy.rsi_reversal_dynamic_active = False
                return True, f"RSI {rsi_val:.2f} ≥ lowest {lowest_val:.2f} + {self.bounce_threshold:.2f} → bounce detected"

            return False, f"RSI {rsi_val:.2f} < lowest {self.strategy.rsi_reversal_dynamic_lowest:.2f} + {self.bounce_threshold:.2f} → waiting for bounce"

    def get_required_indicators(self) -> Set[str]:
        return {Indicators.RSI.value, Indicators.DYNAMIC_RSI_THRESHOLD.value}

    @classmethod
    def validate_config(cls, config, rule_name: str) -> bool:
        """Validate RSIReversalDynamicThreshold configuration."""
        bounce_threshold = config.get_rule_param(rule_name, 'bounce_threshold', 5)

        if bounce_threshold <= 0:
            raise ValueError(
                f"Invalid {rule_name} configuration: "
                f"bounce_threshold ({bounce_threshold}) must be > 0"
            )
        return True

# Class-based CooldownBetweenSOs rule
class CooldownBetweenSOs(Rule):
    """Enforce cooldown between safety orders."""

    def __init__(self, strategy, rule_name='CooldownBetweenSOs'):
        super().__init__(strategy, rule_name)
        self.cooldown_minutes = getattr(strategy, "so_cooldown_minutes", 0)

    def evaluate(self, ctx: Ctx) -> Tuple[bool, str]:
        mins = int(self.cooldown_minutes or 0)
        if mins <= 0:
            return True, "No cooldown"

        now = ctx.now
        last = getattr(self.strategy, "last_safety_order_time", None) or getattr(self.strategy, "base_order_time", None)
        if last is None:
            return True, "No previous SO/BO time"

        if now - last >= timedelta(minutes=mins):
            return True, f"Cooldown {mins}m elapsed"

        if getattr(self.strategy, "debug_trade", False):
            level = ctx.config.get("next_level", ctx.dca_level + 1)
            logger.debug(f"Skip DCA-{level}: cooldown {(now - last)} < {mins}m")
        return False, "Cooldown not elapsed"

    def get_required_indicators(self) -> Set[str]:
        return set()

    @classmethod
    def validate_config(cls, config, rule_name: str) -> bool:
        """Validate CooldownBetweenSOs configuration."""
        # No specific validation needed for this rule
        return True

# Class-based MaxLevelsNotReached rule
class MaxLevelsNotReached(Rule):
    """Check if maximum DCA levels have not been reached."""

    def __init__(self, strategy, rule_name='MaxLevelsNotReached'):
        super().__init__(strategy, rule_name)

    def evaluate(self, ctx: Ctx) -> Tuple[bool, str]:
        max_levels = self.config.max_dca_levels
        ok = ctx.dca_level < max_levels
        return ok, (
            f"Level {ctx.dca_level} < max {max_levels}"
            if ok else f"Reached max levels ({max_levels})"
        )

    def get_required_indicators(self) -> Set[str]:
        return set()

    @classmethod
    def validate_config(cls, config, rule_name: str) -> bool:
        """Validate MaxLevelsNotReached configuration."""
        # No specific validation needed for this rule
        return True

# Class-based SufficientFundsAndNotional rule
class SufficientFundsAndNotional(Rule):
    """Check if sufficient funds and notional requirements are met."""

    def __init__(self, strategy, rule_name='SufficientFundsAndNotional'):
        super().__init__(strategy, rule_name)
        self.minimum_notional = getattr(strategy, "minimum_notional", 0.0)

    def evaluate(self, ctx: Ctx) -> Tuple[bool, str]:
        P_so = ctx.next_so_price
        if P_so is None or P_so <= 0:
            return True, "SO price unknown → allow"

        c = float(ctx.config.get("commission_rate", 0.0) or 0.0)
        cost_per_share = P_so * (1.0 + c)
        if cost_per_share <= 0:
            return True, "Invalid cost/share → allow"

        max_possible = int(ctx.available_cash // cost_per_share)

        min_notional = float(self.minimum_notional or 0.0)
        required_qty = max(1, math.ceil(min_notional / P_so))

        if max_possible < required_qty:
            return False, (
                f"Insufficient funds for min order: cash={ctx.available_cash:.2f}, "
                f"cost/share={cost_per_share:.6f}, max_possible={max_possible}, "
                f"required_qty={required_qty}, min_notional={min_notional:.6f}"
            )

        return True, f"Funds OK (max_possible≥{required_qty}) & order notional OK (≥{min_notional:.6f})"

    def get_required_indicators(self) -> Set[str]:
        return set()

    @classmethod
    def validate_config(cls, config, rule_name: str) -> bool:
        """Validate SufficientFundsAndNotional configuration."""
        # No specific validation needed for this rule
        return True









# Register class-based rules with factory
from .rule_factory import RuleFactory
RuleFactory.register_rule("RSIUnderDynamicThreshold", RSIUnderDynamicThreshold)
RuleFactory.register_rule("RSIReversalStaticThreshold", RSIReversalStaticThreshold)
RuleFactory.register_rule("RSIReversalDynamicThreshold", RSIReversalDynamicThreshold)
RuleFactory.register_rule("RSIUnderStaticThreshold", RSIUnderStaticThreshold)
RuleFactory.register_rule("CooldownBetweenSOs", CooldownBetweenSOs)
RuleFactory.register_rule("MaxLevelsNotReached", MaxLevelsNotReached)
RuleFactory.register_rule("SufficientFundsAndNotional", SufficientFundsAndNotional)

# Simplified SafetyRuleDecider
from ports import SafetyDecider
from rule_chain import RuleChain

class SafetyRuleDecider(SafetyDecider):
    """
    Builds its own RuleChain from config (strings or nested ANY/ALL dicts).
    Uses RuleFactory for all rule management.
    """
    def __init__(self, strategy, names, default_mode: str = "any") -> None:
        # Validate rule configurations before building the chain
        rule_names = self._extract_rule_names(names)
        for rule_name in rule_names:
            RuleFactory.validate_rule_config(rule_name, strategy.config)

        # Create rule instances using RuleFactory
        rule_instances = {}
        for rule_name in rule_names:
            rule_instances[rule_name] = RuleFactory.create_rule(rule_name, strategy)

        # Build rule chain using class-based rules
        self._chain = self._build_rule_chain_from_instances(rule_instances, names, default_mode)
        # Store the rule names for indicator detection
        self._rule_names = rule_names
        # Store strategy reference for indicator detection
        self.strategy = strategy

    def _build_rule_chain_from_instances(self, rule_instances, spec, mode: str = "any"):
        """Build a rule chain from rule instances based on specification."""
        if isinstance(spec, str):
            # Single rule
            return RuleChain([rule_instances[spec]], mode="any")
        elif isinstance(spec, (list, tuple)):
            # List of rules - combine based on mode
            return RuleChain([rule_instances[name] for name in spec], mode=mode)
        elif isinstance(spec, dict):
            # Nested specification - for now, flatten to simple list
            # This can be enhanced later if complex nesting is needed
            rules = []
            for key, value in spec.items():
                if key in ("any", "all"):
                    if isinstance(value, (list, tuple)):
                        rules.extend([rule_instances[name] for name in value])
                    elif isinstance(value, str):
                        rules.append(rule_instances[value])
                else:
                    # Single rule in dict
                    rules.append(rule_instances[key])
            return RuleChain(rules, mode=mode)
        else:
            raise ValueError(f"Invalid rule specification: {spec}")

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

        # Use RuleFactory to get required indicators from class-based rules
        for rule_name in self._rule_names:
            rule_indicators = RuleFactory.get_required_indicators(rule_name, self.strategy.config)
            required.update(rule_indicators)

        return required

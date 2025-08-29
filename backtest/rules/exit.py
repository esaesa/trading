# rules/exit.py
from typing import Union
from typing import Dict, Any, Tuple, Callable, Set
import numpy as np
from contracts import Ctx
from datetime import datetime, timedelta
from logger_config import logger
import warnings
from indicators import Indicators

# Import for class-based rules
from .base_decision_rule import DecisionRule

# Class-based ATRTakeProfitReached rule
class ATRTakeProfitReached(DecisionRule):
    """Exit rule: TP dynamically set from ATR%."""

    def __init__(self, strategy, rule_name='ATRTakeProfitReached'):
        super().__init__(strategy, rule_name)
        self.atr_tp_fraction = float(getattr(strategy, "atr_tp_fraction", 0.65))
        self.atr_tp_min_pct = float(getattr(strategy, "atr_tp_min_pct", 0.8))
        self.atr_tp_max_pct = float(getattr(strategy, "atr_tp_max_pct", 5.0))

    def evaluate(self, ctx: Ctx) -> Tuple[bool, str]:
        if not self.strategy.position:
            return False, "No position"

        # Get ATR as a regular indicator via the service
        atr_pct = self.strategy.indicator_service.get_indicator_value("atr_pct", ctx.now, None)

        if atr_pct is None or (isinstance(atr_pct, float) and np.isnan(atr_pct)):
            # 🔴 NEW: warning for debugging
            msg = f"[ATR TP] ATR% unavailable at {getattr(ctx, 'now', None)}"
            logger.warning(msg)
            warnings.warn(msg)   # optional, shows in stdout
            return False, msg

        # Get rule-specific parameters from config
        deviation_threshold = self.config.get_rule_param(
            self.rule_name, 'deviation_threshold', self.config.atr_deviation_threshold
        )
        deviation_reduction_factor = self.config.get_rule_param(
            self.rule_name, 'deviation_reduction_factor', self.config.atr_deviation_reduction_factor
        )

        tp_target = max(self.atr_tp_min_pct, min(self.atr_tp_max_pct, float(atr_pct) * self.atr_tp_fraction))
        profit_pct = float(self.strategy.position.pl_pct)

        ok = profit_pct >= tp_target
        reason = (
            f"ATR TP: profit {profit_pct:.2f}% >= target {tp_target:.2f}% "
            f"(ATR% {float(atr_pct):.2f} × f {self.atr_tp_fraction:.2f}, clamp [{self.atr_tp_min_pct:.2f}–{self.atr_tp_max_pct:.2f}])"
        )
        return ok, reason

    def get_required_indicators(self) -> Set[str]:
        return {Indicators.ATR_PCT.value}

    @classmethod
    def validate_config(cls, config, rule_name: str) -> bool:
        """Validate ATRTakeProfitReached configuration."""
        # No specific validation needed for this rule
        return True

# Class-based TakeProfitReached rule
class TakeProfitReached(DecisionRule):
    """Exit rule based on fixed take profit percentage."""

    def __init__(self, strategy, rule_name='TakeProfitReached'):
        super().__init__(strategy, rule_name)

    def evaluate(self, ctx: Ctx) -> Tuple[bool, str]:
        if not self.strategy.position:
            return False, "No position"

        profit_pct = self.strategy.position.pl_pct
        take_profit_pct = self.config.take_profit_percentage  # From strategy config

        should_exit = profit_pct >= take_profit_pct
        reason = f"Take profit: current profit {profit_pct:.2f}% >= target {take_profit_pct:.2f}%"

        return should_exit, reason

    def get_required_indicators(self) -> Set[str]:
        return set()

    @classmethod
    def validate_config(cls, config, rule_name: str) -> bool:
        """Validate TakeProfitReached configuration."""
        # No specific validation needed for this rule
        return True

# Class-based TPDecayReached rule
class TPDecayReached(DecisionRule):
    """Exit rule using decaying take profit logic."""

    def __init__(self, strategy, rule_name='TPDecayReached'):
        super().__init__(strategy, rule_name)

    def evaluate(self, ctx: Ctx) -> Tuple[bool, str]:
        if not ctx.entry_price or not ctx.last_entry_time:
            return False, "No position"

        profit_pct = self.strategy.position.pl_pct

        # Get rule-specific parameters from config
        grace_period_hours = self.config.get_rule_param(
            self.rule_name, 'grace_period_hours', self.config.take_profit_decay_grace_period_hours
        )
        decay_duration_hours = self.config.get_rule_param(
            self.rule_name, 'decay_duration_hours', self.config.take_profit_decay_duration_hours
        )

        adjusted_tp = calculate_decaying_tp(
            ctx.last_entry_time,
            ctx.now,
            self.config.take_profit_percentage,
            grace_period_hours,
            decay_duration_hours
        )

        should_exit = profit_pct >= adjusted_tp
        reason = f"TP decay: current profit {profit_pct:.2f}% >= adjusted TP {adjusted_tp:.2f}%"

        return should_exit, reason

    def get_required_indicators(self) -> Set[str]:
        return set()

    @classmethod
    def validate_config(cls, config, rule_name: str) -> bool:
        """Validate TPDecayReached configuration."""
        # No specific validation needed for this rule
        return True

# Class-based StopLossReached rule
class StopLossReached(DecisionRule):
    """Exit rule for stop loss."""

    def __init__(self, strategy, rule_name='StopLossReached'):
        super().__init__(strategy, rule_name)
        self.stop_loss_threshold = -25  # -25% stop loss

    def evaluate(self, ctx: Ctx) -> Tuple[bool, str]:
        should_exit = self.strategy.position.pl_pct <= self.stop_loss_threshold
        reason = f"Stop loss triggered at {self.strategy.position.pl_pct:.2f}%"

        return should_exit, reason

    def get_required_indicators(self) -> Set[str]:
        return set()

    @classmethod
    def validate_config(cls, config, rule_name: str) -> bool:
        """Validate StopLossReached configuration."""
        # No specific validation needed for this rule
        return True

# Class-based TrailingStopReached rule
class TrailingStopReached(DecisionRule):
    """Exit rule for trailing stop."""

    def __init__(self, strategy, rule_name='TrailingStopReached'):
        super().__init__(strategy, rule_name)
        self.trail_pct = 1.0  # 1% trailing stop

    def evaluate(self, ctx: Ctx) -> Tuple[bool, str]:
        should_exit = ctx.max_profit_pct - ctx.unrealized_profit_pct >= self.trail_pct
        reason = f"Trailing stop: drawdown from peak {ctx.max_profit_pct:.2f}% to {ctx.unrealized_profit_pct:.2f}%"

        return should_exit, reason

    def get_required_indicators(self) -> Set[str]:
        return set()

    @classmethod
    def validate_config(cls, config, rule_name: str) -> bool:
        """Validate TrailingStopReached configuration."""
        # No specific validation needed for this rule
        return True


def calculate_decaying_tp(
    entry_time: datetime,
    current_time: datetime,
    initial_tp: float,
    grace_period_hours: Union[int, float],
    decay_duration_hours: Union[int, float],
    min_tp: float = 0.2
) -> float:
    """
    Calculates a decaying take-profit (TP) target based on time since entry.

    The TP remains at `initial_tp` during a `grace_period_hours`. After the grace period,
    the TP linearly decays from `initial_tp` down to `min_tp` over `decay_duration_hours`.

    Args:
        entry_time: The datetime when the position was entered.
        current_time: The current datetime for calculation.
        initial_tp: The initial take-profit percentage.
        grace_period_hours: The duration in hours during which TP does not decay.
        decay_duration_hours: The duration in hours over which TP decays after the grace period.
        min_tp: The minimum take-profit percentage the TP can decay to.

    Returns:
        The adjusted decaying take-profit percentage.

    Raises:
        ValueError: If grace_period_hours or decay_duration_hours are negative,
                    or if initial_tp is less than min_tp.
    """
    # 1. Error Handling and Edge Cases
    if not isinstance(entry_time, datetime) or not isinstance(current_time, datetime):
        # Log a warning or raise a more specific error if context allows
        raise ValueError("entry_time and current_time must be datetime objects.")
    if grace_period_hours < 0:
        raise ValueError("grace_period_hours cannot be negative.")
    if decay_duration_hours <= 0:
        # Decay duration must be positive for decay to occur
        # If decay duration is 0 or negative, TP immediately goes to min_tp (if initial_tp > min_tp)
        return min_tp if initial_tp > min_tp else initial_tp
    if initial_tp < min_tp:
        raise ValueError("initial_tp cannot be less than min_tp.")

    # If entry_time is in the future or current_time is before entry_time,
    # it implies an invalid state or a new position, so return initial_tp.
    if current_time < entry_time:
        return initial_tp

    # Calculate time differences
    time_since_entry: timedelta = current_time - entry_time
    grace_period_td: timedelta = timedelta(hours=grace_period_hours)
    decay_span_td: timedelta = timedelta(hours=decay_duration_hours)

    # 2. Readability and Maintainability
    # If the grace period has not passed, return the original TP
    if time_since_entry <= grace_period_td:
        return initial_tp

    # Calculate time elapsed since the grace period ended
    time_after_grace_period: timedelta = time_since_entry - grace_period_td

    # If the decay phase is complete, return the minimum TP
    if time_after_grace_period >= decay_span_td:
        return min_tp

    # 3. Performance Optimization & Best Practices
    # Calculate the fraction of the decay duration that has passed
    # This is equivalent to (time_after_grace_period.total_seconds() / decay_span_td.total_seconds())
    reduction_factor: float = time_after_grace_period / decay_span_td

    # Calculate the adjusted TP using linear interpolation
    adjusted_tp: float = initial_tp * (1 - reduction_factor) + min_tp * reduction_factor

    # Ensure the adjusted TP is within the valid range [min_tp, initial_tp]
    return max(min_tp, min(initial_tp, adjusted_tp))


from ports import ExitDecider
from .rule_factory import RuleFactory

class ExitRuleDecider(ExitDecider):
    """
    Builds a potentially nested rule structure from config.
    """
    def __init__(self, strategy, spec) -> None:
        self.strategy = strategy
        # The entire building process is now handled by the recursive builder.
        from .rule_builder import build_rule_from_spec
        self._rule_structure = build_rule_from_spec(strategy, spec)
        # For indicator detection
        self._required_indicators = self._extract_indicators(self._rule_structure)

    def ok(self, ctx):
        # The ok() method now just calls evaluate() on the top-level rule structure.
        return self._rule_structure.evaluate(ctx)

    def get_required_indicators(self) -> set[str]:
        return self._required_indicators

    def _extract_indicators(self, rule) -> set[str]:
        """Recursively traverses the rule structure to find all required indicators."""
        required = set()
        if hasattr(rule, 'get_required_indicators'):
            required.update(rule.get_required_indicators())
        if hasattr(rule, 'rules'): # It's a RuleChain
            for sub_rule in rule.rules:
                required.update(self._extract_indicators(sub_rule))
        return required

    # ok() now returns (bool, str) tuple directly

# Register class-based rules with factory
RuleFactory.register_rule("ATRTakeProfitReached", ATRTakeProfitReached)
RuleFactory.register_rule("TakeProfitReached", TakeProfitReached)
RuleFactory.register_rule("TPDecayReached", TPDecayReached)
RuleFactory.register_rule("StopLossReached", StopLossReached)
RuleFactory.register_rule("TrailingStopReached", TrailingStopReached)

# Import RuleChain for the decider
from rule_chain import RuleChain

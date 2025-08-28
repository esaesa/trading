from abc import ABC, abstractmethod
from typing import Tuple, Set, Dict, Any, TYPE_CHECKING
from contracts import Ctx
from indicators import Indicators

if TYPE_CHECKING:
    from strategy import DCAStrategy

class Rule(ABC):
    """Base class for all trading rules with standardized interface."""

    def __init__(self, strategy: 'DCAStrategy', rule_name: str):
        self.strategy = strategy
        self.rule_name = rule_name
        self.config = strategy.config

    @abstractmethod
    def evaluate(self, ctx: Ctx) -> Tuple[bool, str]:
        """Evaluate the rule and return (result, reason)."""
        pass

    @abstractmethod
    def get_required_indicators(self) -> Set[str]:
        """Return indicators required by this rule."""
        pass

    @classmethod
    @abstractmethod
    def validate_config(cls, config, rule_name: str) -> bool:
        """Validate rule configuration during initialization."""
        pass

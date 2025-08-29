from typing import Dict, Type, Any
from .base_decision_rule import DecisionRule

class RuleFactory:
    """Factory for creating and validating rule instances."""

    _rule_classes: Dict[str, Type[DecisionRule]] = {}

    @classmethod
    def register_rule(cls, rule_name: str, rule_class: Type[DecisionRule]):
        """Register a rule class with the factory."""
        cls._rule_classes[rule_name] = rule_class

    @classmethod
    def create_rule(cls, rule_name: str, strategy) -> DecisionRule:
        """Create a rule instance after validating its configuration."""
        if rule_name not in cls._rule_classes:
            # For backward compatibility, we'll handle this gracefully
            # by allowing function-based rules to continue working
            raise ValueError(f"Unknown rule class: {rule_name}")
        
        # Validate configuration first
        cls.validate_rule_config(rule_name, strategy.config)
        
        # Create rule instance
        return cls._rule_classes[rule_name](strategy, rule_name)
    
    @classmethod
    def validate_rule_config(cls, rule_name: str, config) -> bool:
        """Validate rule configuration during initialization."""
        if rule_name not in cls._rule_classes:
            # Skip validation for unknown rules (function-based ones)
            return True
        
        return cls._rule_classes[rule_name].validate_config(config, rule_name)
    
    @classmethod
    def get_required_indicators(cls, rule_name: str, config) -> set:
        """Get required indicators for a rule."""
        if rule_name not in cls._rule_classes:
            # For function-based rules, we'll need to handle this differently
            return set()
        
        # Create a mock strategy instance for indicator detection
        class MockStrategy:
            def __init__(self, config):
                self.config = config
        
        mock_strategy = MockStrategy(config)
        rule_instance = cls._rule_classes[rule_name](mock_strategy, rule_name)
        return rule_instance.get_required_indicators()

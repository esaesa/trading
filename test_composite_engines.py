#!/usr/bin/env python3
"""
Comprehensive test suite for Composite Engine Architectural Enhancement.

Tests both pricing and sizing engines with:
- Backward compatibility (string specs)
- Simple composition (min/max/avg operations)
- Nested composition (recursive specs)
- Error handling
- System integration
"""

import sys
import os
from types import SimpleNamespace
from datetime import datetime

# Add backtest to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backtest'))

from backtest.rules.pricing import create_price_engine, StaticPriceEngine, DynamicATRPriceEngine
from backtest.rules.sizing import create_size_engine, ValueModeSizeRule, VolumeModeSizeRule
from backtest.contracts import Ctx


def create_mock_strategy():
    """Create a mock strategy with required configuration."""

    class MockConfig:
        def __init__(self):
            # Pricing config
            self.safety_order_price_mode = "dynamic"
            self.initial_deviation_percent = 5.0
            self.price_multiplier = 1.5
            self.max_dca_levels = 5
            self.atr_deviation_threshold = 2.0
            self.atr_deviation_reduction_factor = 0.5

            # Sizing config
            self.safety_order_mode = "value"
            self.first_safety_order_multiplier = 2.0
            self.so_size_multiplier = 1.8

            # ATR support
            self.atr_price_mode = "static"

    class MockIndicatorService:
        def get_indicator_value(self, name, timestamp, default=None):
            # Return mock ATR value for testing
            if name == "atr_pct":
                return 1.5  # Mock ATR percentage value
            return default

    class MockStrategy:
        def __init__(self):
            self.config = MockConfig()
            self.indicator_service = MockIndicatorService()

            # Mock attribute access for legacy ATR access
            self.atr_pct_series = None  # No series available, will use indicator_service

        @property
        def indicator_provider(self):
            return None  # No provider, will fall back to indicator_service

    return MockStrategy()


def create_mock_context():
    """Create a mock context for testing."""
    from types import SimpleNamespace
    config = SimpleNamespace()

    # Create Ctx with all required fields
    ctx = Ctx(
        now=datetime.now(),
        price=100.0,
        entry_price=95.0,
        position_size=10.0,
        dca_level=1,
        equity_per_cycle=2000.0,
        config=config,
        position_pl_pct=5.0,
        base_order_price=100.0,
        base_order_value=2000.0,  # $2000 base order value
        base_order_quantity=10.0,  # 10 units base quantity
        last_filled_price=105.0
    )
    return ctx


def test_backward_compatibility():
    """Test that existing string-based specifications still work."""
    print("\n=== Testing Backward Compatibility ===")

    strategy = create_mock_strategy()
    ctx = create_mock_context()

    # Test pricing engines
    static_engine = create_price_engine(strategy, "static")
    dynamic_engine = create_price_engine(strategy, "dynamic")

    print(f"✓ StaticPriceEngine created: {type(static_engine).__name__}")
    print(f"✓ DynamicATRPriceEngine created: {type(dynamic_engine).__name__}")

    # Test pricing calculations
    static_price = static_engine.so_price(ctx, 1)
    dynamic_price = dynamic_engine.so_price(ctx, 1)

    print(".2f")
    print(".2f")

    # Test sizing engines
    value_engine = create_size_engine(strategy, "value")
    volume_engine = create_size_engine(strategy, "volume")

    print(f"✓ ValueModeSizeRule created: {type(value_engine).__name__}")
    print(f"✓ VolumeModeSizeRule created: {type(volume_engine).__name__}")

    # Test sizing calculations
    value_size = value_engine.so_size(ctx, 50.0, 1)
    volume_size = volume_engine.so_size(ctx, 50.0, 1)

    print(".2f")
    print(".2f")

    return True


def test_simple_composition():
    """Test simple min/max/avg composite operations."""
    print("\n=== Testing Simple Composition ===")

    strategy = create_mock_strategy()
    ctx = create_mock_context()

    # Test pricing composition
    print("\n--- Pricing Engine Composition ---")

    # Min composite: should take minimum of static and dynamic
    min_price_spec = {"min": ["static", "dynamic"]}
    min_price_engine = create_price_engine(strategy, min_price_spec)
    print(f"✓ Min composite pricing engine created: {type(min_price_engine).__name__}")

    min_price = min_price_engine.so_price(ctx, 1)
    independent_static = create_price_engine(strategy, "static").so_price(ctx, 1)
    independent_dynamic = create_price_engine(strategy, "dynamic").so_price(ctx, 1)

    expected_min = min(independent_static, independent_dynamic)
    print(".2f")
    assert abs(min_price - expected_min) < 0.01, f"Min composition failed: got {min_price}, expected {expected_min}"

    # Max composite: should take maximum
    max_price_spec = {"max": ["static", "dynamic"]}
    max_price_engine = create_price_engine(strategy, max_price_spec)
    max_price = max_price_engine.so_price(ctx, 1)
    expected_max = max(independent_static, independent_dynamic)
    print(".2f")
    assert abs(max_price - expected_max) < 0.01, f"Max composition failed: got {max_price}, expected {expected_max}"

    # Avg composite: should average the values
    avg_price_spec = {"avg": ["static", "dynamic"]}
    avg_price_engine = create_price_engine(strategy, avg_price_spec)
    avg_price = avg_price_engine.so_price(ctx, 1)
    expected_avg = (independent_static + independent_dynamic) / 2
    print(".2f")
    assert abs(avg_price - expected_avg) < 0.01, f"Avg composition failed: got {avg_price}, expected {expected_avg}"

    # Test sizing composition
    print("\n--- Sizing Engine Composition ---")

    # Min composite sizing
    min_size_spec = {"min": ["value", "volume"]}
    min_size_engine = create_size_engine(strategy, min_size_spec)
    print(f"✓ Min composite sizing engine created: {type(min_size_engine).__name__}")

    min_size = min_size_engine.so_size(ctx, 50.0, 1)
    independent_value = create_size_engine(strategy, "value").so_size(ctx, 50.0, 1)
    independent_volume = create_size_engine(strategy, "volume").so_size(ctx, 50.0, 1)

    expected_min_size = min(independent_value, independent_volume)
    print(".2f")
    assert abs(min_size - expected_min_size) < 0.01, f"Min size composition failed: got {min_size}, expected {expected_min_size}"

    return True


def test_nested_composition():
    """Test nested composite operations."""
    print("\n=== Testing Nested Composition ===")

    strategy = create_mock_strategy()
    ctx = create_mock_context()

    # Nested pricing: average of static and min of [dynamic, static]
    nested_price_spec = {"avg": ["static", {"min": ["dynamic", "static"]}]}
    nested_price_engine = create_price_engine(strategy, nested_price_spec)
    print(f"✓ Nested pricing engine created: {type(nested_price_engine).__name__}")

    nested_price = nested_price_engine.so_price(ctx, 1)

    # Calculate expected manually
    static_price = create_price_engine(strategy, "static").so_price(ctx, 1)
    dynamic_price = create_price_engine(strategy, "dynamic").so_price(ctx, 1)
    inner_min = min(dynamic_price, static_price)
    expected_nested = (static_price + inner_min) / 2

    print(".2f")
    assert abs(nested_price - expected_nested) < 0.01, f"Nested composition failed: got {nested_price}, expected {expected_nested}"

    # Nested sizing
    nested_size_spec = {"max": ["value", {"avg": ["volume", "value"]}]}
    nested_size_engine = create_size_engine(strategy, nested_size_spec)
    print(f"✓ Nested sizing engine created: {type(nested_size_engine).__name__}")

    nested_size = nested_size_engine.so_size(ctx, 50.0, 1)

    # Calculate expected manually
    value_size = create_size_engine(strategy, "value").so_size(ctx, 50.0, 1)
    volume_size = create_size_engine(strategy, "volume").so_size(ctx, 50.0, 1)
    inner_avg = (volume_size + value_size) / 2
    expected_nested_size = max(value_size, inner_avg)

    print(".2f")
    assert abs(nested_size - expected_nested_size) < 0.01, f"Nested size composition failed: got {nested_size}, expected {expected_nested_size}"

    return True


def test_error_handling():
    """Test error handling for invalid specifications."""
    print("\n=== Testing Error Handling ===")

    strategy = create_mock_strategy()

    # Test invalid operation
    try:
        invalid_spec = {"invalid_op": ["static", "dynamic"]}
        create_price_engine(strategy, invalid_spec)
        assert False, "Should have raised ValueError for invalid operation"
    except ValueError as e:
        print(f"✓ Correctly caught invalid operation: {e}")

    # Test empty engine list
    try:
        empty_spec = {"min": []}
        create_price_engine(strategy, empty_spec)
        assert False, "Should have raised ValueError for empty engine list"
    except ValueError as e:
        print(f"✓ Correctly caught empty engine list: {e}")

    # Test invalid spec type
    try:
        create_price_engine(strategy, 123)
        assert False, "Should have raised ValueError for invalid spec type"
    except ValueError as e:
        print(f"✓ Correctly caught invalid spec type: {e}")

    return True


def test_system_integration():
    """Test integration with the broader system."""
    print("\n=== Testing System Integration ===")

    strategy = create_mock_strategy()
    ctx = create_mock_context()

    # Test configuration-based engine creation (without explicit mode)
    default_price_engine = create_price_engine(strategy)
    default_size_engine = create_size_engine(strategy)

    print(f"✓ Config-based pricing engine: {type(default_price_engine).__name__}")
    print(f"✓ Config-based sizing engine: {type(default_size_engine).__name__}")

    # Test that engines work correctly
    price = default_price_engine.so_price(ctx, 1)
    size = default_size_engine.so_size(ctx, 50.0, 1)

    print(".2f")
    print(".2f")

    # Test with custom config mode
    custom_config = SimpleNamespace()
    custom_config.safety_order_price_mode = {"min": ["static", "dynamic"]}
    custom_config.safety_order_mode = {"avg": ["value", "volume"]}
    custom_config.initial_deviation_percent = 5.0
    custom_config.price_multiplier = 1.5
    custom_config.max_dca_levels = 5
    custom_config.atr_deviation_threshold = 2.0
    custom_config.atr_deviation_reduction_factor = 0.5
    custom_config.first_safety_order_multiplier = 2.0
    custom_config.so_size_multiplier = 1.8

    custom_strategy = SimpleNamespace()
    custom_strategy.config = custom_config

    custom_price_engine = create_price_engine(custom_strategy)
    custom_size_engine = create_size_engine(custom_strategy)

    print(f"✓ Custom config pricing engine: {type(custom_price_engine).__name__}")
    print(f"✓ Custom config sizing engine: {type(custom_size_engine).__name__}")

    return True


def main():
    """Run all tests."""
    print("🚀 Starting Composite Engine Tests")
    print("=" * 50)

    tests = [
        test_backward_compatibility,
        test_simple_composition,
        test_nested_composition,
        test_error_handling,
        test_system_integration,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
                print(f"✅ {test.__name__} PASSED")
            else:
                failed += 1
                print(f"❌ {test.__name__} FAILED")
        except Exception as e:
            failed += 1
            print(f"❌ {test.__name__} FAILED with exception: {e}")

    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("🎉 All tests passed! Composite Engine enhancement is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
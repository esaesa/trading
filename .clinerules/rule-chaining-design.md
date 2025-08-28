## Brief overview
Guidelines for implementing class-based rule architecture with proper validation and chaining patterns in trading systems, focusing on modern design without backward compatibility.

## Rule Architecture
- Use class-based rules inheriting from base Rule abstract class
- Implement required methods: evaluate(), get_required_indicators(), validate_config()
- Remove backward compatibility with function-based rules
- Migrate all existing rules to class-based implementation

## Validation Strategy  
- Validate configurations during rule decider initialization (one-time)
- Prevent logical contradictions like reset_threshold < static_threshold_under
- Provide clear, actionable error messages with specific parameter values
- Catch configuration errors early before runtime execution

## Rule Chaining Patterns
- Use RuleChain class for complex rule combinations
- Support nested any/all logical operators in specifications
- Ensure rules are composable and work seamlessly in chains
- Maintain clean separation between rule logic and chaining infrastructure

## Factory Implementation
- Use RuleFactory for centralized rule registration and management
- Register all class-based rules with consistent naming conventions
- Enable scalable addition of new rule types through factory pattern
- Handle validation through factory for consistency across all rules

## Error Handling
- Raise ValueError with descriptive messages for validation failures
- Include specific parameter values in error messages for debugging
- Ensure errors guide users to fix configurations properly
- Catch errors during initialization phase, not at runtime

## Migration Approach
- Identify and convert all function-based rules to class-based
- Remove legacy compatibility layers and function wrappers
- Update rule registries to use class references only
- Document the new class-based architecture for future development

## Testing Strategy
- Unit test validation logic independently from rule evaluation
- Test rule chaining with various combinations and nesting levels
- Verify error messages are clear and actionable
- Ensure class-based rules work correctly in isolation and chains

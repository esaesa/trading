# rule_chain.py - Simplified for class-based rules
from typing import List, Literal, Tuple, Union
from contracts import Ctx
from rules.base_rule import Rule

EvaluationMode = Literal["any", "all"]  # ok if ANY passes, or ALL pass

class RuleChain:
    """Simplified RuleChain that works directly with Rule objects."""

    def __init__(self, rules: List[Rule], mode: EvaluationMode = "any"):
        """
        :param rules: List of Rule objects
        :param mode:  'any' or 'all'
        """
        self.rules = rules
        self.mode = mode

    def ok(self, ctx: Ctx) -> bool:
        """Evaluate rules and return True if condition is met."""
        if self.mode == "any":
            for rule in self.rules:
                ok, _ = rule.evaluate(ctx)
                if ok:
                    return True
            return False
        elif self.mode == "all":
            for rule in self.rules:
                ok, _ = rule.evaluate(ctx)
                if not ok:
                    return False
            return True
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def ok_reason(self, ctx: Ctx) -> Tuple[bool, str]:
        """Evaluate rules and return (result, reason)."""
        reasons = []
        if self.mode == "any":
            for rule in self.rules:
                ok, reason = rule.evaluate(ctx)
                reasons.append(reason)
                if ok:
                    return True, reason or "Passed"
            # none passed
            return False, "; ".join([r for r in reasons if r])
        elif self.mode == "all":
            last_reason = ""
            for rule in self.rules:
                ok, reason = rule.evaluate(ctx)
                if not ok:
                    return False, reason or "One condition failed"
                last_reason = reason
            return True, last_reason or "All passed"
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

# --------------------------

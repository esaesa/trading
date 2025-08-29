# rule_chain.py - Unified rule chains: implement IEvaluable, change ok to evaluate
from typing import List, Literal, Tuple, Union
from contracts import Ctx
from rules.base_decision_rule import IEvaluable

EvaluationMode = Literal["any", "all"]  # ok if ANY passes, or ALL pass

class RuleChain(IEvaluable):
    """RuleChain that implements IEvaluable interface and can contain other RuleChains."""

    def __init__(self, rules: List[IEvaluable], mode: EvaluationMode = "any"):
        """
        :param rules: List of IEvaluable objects (RuleChains or DecisionRules)
        :param mode:  'any' or 'all'
        """
        self.rules = rules
        self.mode = mode

    def evaluate(self, ctx: Ctx) -> Tuple[bool, str]:
        """Evaluate rules and return (result, reason)."""
        reasons = []
        if self.mode == "any":
            for rule in self.rules:
                ok, reason = rule.evaluate(ctx)
                reasons.append(reason)
                if ok:
                    return True, f"ANY passed: ({reason})"
            return False, f"ANY failed: ({'; '.join(reasons)})"

        elif self.mode == "all":
            for rule in self.rules:
                ok, reason = rule.evaluate(ctx)
                if not ok:
                    return False, f"ALL failed: ({reason})"
                reasons.append(reason)
            return True, f"ALL passed: ({'; '.join(reasons)})"

        raise ValueError(f"Invalid mode: {self.mode}")

# --------------------------

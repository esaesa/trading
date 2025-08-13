# rule_chain.py
from functools import partial
from typing import Any, Callable, Dict, Iterable, Literal, Tuple

CheckFn = Callable[[Any, Dict[str, Any]], bool | Tuple[bool, str]]
EvaluationMode = Literal["any", "all"]  #  if ANY rule triggers, or ALL trigger

class RuleChain:
    def __init__(self, rules: Iterable[CheckFn],mode: EvaluationMode = "any"):
        """
        :param rules: Iterable of check functions (self, ctx) -> bool or (bool, str)
        :param mode: 
            - "any": return True (ok) if *any* rule passes; exit when *any* fails in negative logic?
            - But wait: we need to clarify semantics.
        """
        self.rules = list(rules)  # each rule(self_like, ctx) -> bool or (bool, reason)
        self.mode = mode  # "any" -> ok if any rule says yes; "all" -> only if all say yes

    def ok(self, ctx: Dict[str, Any]) -> bool:
        results = []
        for rule in self.rules:
            res = rule(ctx)
            ok = res[0] if isinstance(res, tuple) else bool(res)
            results.append(bool(ok))
            
        if self.mode == "any":
            # Ok if *any* rule triggers (i.e., any returns True)
            ok = any(results)
        elif self.mode == "all":
            # Ok only if *all* rules trigger
            ok = all(results)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        return ok  
    
  
def build_rule_chain(owner: Any, names: Iterable[str], registry: Dict[str, CheckFn],  mode: EvaluationMode = "any") -> RuleChain:
    rules: list[CheckFn] = []
    for n in names:
        try:
            fn = registry[n]
        except KeyError:
            raise ValueError(f"Unknown rule name: {n}")
        rules.append(partial(fn, owner))  # bind self
    return RuleChain(rules)

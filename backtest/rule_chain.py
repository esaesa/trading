# rule_chain.py
from functools import partial
from typing import Any, Callable, Dict, Iterable, Tuple

CheckFn = Callable[[Any, Dict[str, Any]], bool | Tuple[bool, str]]

class RuleChain:
    def __init__(self, rules: Iterable[CheckFn]):
        self.rules = list(rules)  # each rule(self_like, ctx) -> bool or (bool, reason)

    def ok(self, ctx: Dict[str, Any]) -> bool:
        for rule in self.rules:
            res = rule(ctx)
            ok = res[0] if isinstance(res, tuple) else bool(res)
            if not ok:
                return False
        return True

def build_rule_chain(owner: Any, names: Iterable[str], registry: Dict[str, CheckFn]) -> RuleChain:
    rules: list[CheckFn] = []
    for n in names:
        try:
            fn = registry[n]
        except KeyError:
            raise ValueError(f"Unknown rule name: {n}")
        rules.append(partial(fn, owner))  # bind self
    return RuleChain(rules)

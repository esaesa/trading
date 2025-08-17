# rule_chain.py
from functools import partial
from typing import Any, Callable, Dict, Iterable, Literal, Tuple, Union, List

# A single rule, already bound to the owner via functools.partial, called as rule(ctx)
CheckFn = Callable[[Any], bool | Tuple[bool, str]]
EvaluationMode = Literal["any", "all"]  # ok if ANY passes, or ALL pass

class RuleChain:
    def __init__(self, rules: Iterable[CheckFn], mode: EvaluationMode = "any"):
        """
        :param rules: Iterable of callables (ctx) -> bool | (bool, reason)
        :param mode:  'any' or 'all'
        """
        self.rules = list(rules)
        self.mode = mode

    def ok(self, ctx: Dict[str, Any]) -> bool:
        results: List[bool] = []
        for rule in self.rules:
            res = rule(ctx)
            ok = res[0] if isinstance(res, tuple) else bool(res)
            results.append(bool(ok))

        if self.mode == "any":
            return any(results)
        elif self.mode == "all":
            return all(results)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

# --------------------------
# Nested-spec builder
# --------------------------

Spec = Union[str, List[Any], Tuple[Any, ...], Dict[str, Any]]

def _wrap_chain(chain: RuleChain) -> CheckFn:
    """Wrap a RuleChain so it can be used like a rule function in a parent chain."""
    return lambda ctx, _c=chain: _c.ok(ctx)

def _build_from_list(owner: Any, items: List[Any], registry: Dict[str, Callable], default_mode: EvaluationMode) -> RuleChain:
    rules: List[CheckFn] = []
    for item in items:
        if isinstance(item, str):
            try:
                fn = registry[item]
            except KeyError:
                raise ValueError(f"Unknown rule name: {item}")
            rules.append(partial(fn, owner))  # bind self
        elif isinstance(item, (list, tuple, dict)):
            sub = build_rule_chain(owner, item, registry, mode="any")  # nested block chooses its own mode if dict
            rules.append(_wrap_chain(sub))
        else:
            raise TypeError(f"Unsupported spec type inside list: {type(item).__name__}")
    return RuleChain(rules, mode=default_mode)

def _build_from_dict(owner: Any, spec: Dict[str, Any], registry: Dict[str, Callable]) -> RuleChain:
    keys = [k for k in spec.keys() if k in ("all", "any")]
    if len(keys) != 1:
        raise ValueError(f"Dict spec must contain exactly one of 'all' or 'any', got keys={list(spec.keys())}")
    key = keys[0]
    mode: EvaluationMode = "all" if key == "all" else "any"
    items = spec[key]
    if not isinstance(items, (list, tuple)):
        raise TypeError(f"'{key}' value must be a list/tuple, got {type(items).__name__}")
    # Each child item can be a string name, a list/tuple, or another dict
    child_rules: List[CheckFn] = []
    for child in items:
        if isinstance(child, str):
            try:
                fn = registry[child]
            except KeyError:
                raise ValueError(f"Unknown rule name: {child}")
            child_rules.append(partial(fn, owner))
        elif isinstance(child, (list, tuple, dict)):
            sub = build_rule_chain(owner, child, registry, mode="any")  # nested dict chooses its own mode
            child_rules.append(_wrap_chain(sub))
        else:
            raise TypeError(f"Unsupported nested spec type: {type(child).__name__}")
    return RuleChain(child_rules, mode=mode)

def build_rule_chain(owner: Any, spec: Spec, registry: Dict[str, Callable], mode: EvaluationMode = "any") -> RuleChain:
    """
    Build a RuleChain from a spec:
      - "RSIUnderDynamicThreshold"                     (str)
      - ["A", "B", {"any": ["C", {"all": ["D","E"]}]}] (list with nested dicts)
      - {"all": ["A", {"any": ["B","C"]}]}             (dict)
    :param owner:    the Strategy or object bound as 'self' to each rule function
    :param spec:     str | list/tuple (optionally mixing dicts) | dict with 'all'/'any'
    :param registry: mapping of rule_name -> rule_fn(self, ctx) -> bool | (bool, reason)
    :param mode:     default evaluation for top-level *lists*; ignored when top-level is a dict
    """
    if isinstance(spec, str):
        # Treat as one-item list under given default mode
        return _build_from_list(owner, [spec], registry, default_mode=mode)

    if isinstance(spec, (list, tuple)):
        return _build_from_list(owner, list(spec), registry, default_mode=mode)

    if isinstance(spec, dict):
        return _build_from_dict(owner, spec, registry)

    raise TypeError(f"Unsupported spec type: {type(spec).__name__}")

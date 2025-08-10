# rules/exit.py
from typing import Tuple
from contracts import Ctx

def tp_decay_reached(self, ctx: Ctx) -> Tuple[bool, str]:
    return self._exit_allows(ctx.now), ""

EXIT_RULES = {"TPDecayReached": tp_decay_reached}

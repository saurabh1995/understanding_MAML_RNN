"""Training strategies. Add a new one by subclassing ``Strategy`` and registering it here
(plus a ``strategies.<name>`` section in ``configs/default.yaml``)."""

from .base import Strategy
from .classical import Classical
from .maml_variants import (MAML_v1_Basic, MAML_v2_RLIF, MAML_v3_MetaBatch, MAML_v4_MultiInner,
                            MAML_v5_TaskQuery, MAML_v6_AdaptInnerOuter)

STRATEGIES = {cls.name: cls for cls in (
    Classical,
    MAML_v1_Basic,
    MAML_v2_RLIF,
    MAML_v3_MetaBatch,
    MAML_v4_MultiInner,
    MAML_v5_TaskQuery,
    MAML_v6_AdaptInnerOuter,
)}

ALIASES = {"maml": MAML_v6_AdaptInnerOuter.name}


def resolve_name(name):
    return ALIASES.get(name, name)


def get_strategy(name, params, build_fn, verbose=1):
    name = resolve_name(name)
    try:
        cls = STRATEGIES[name]
    except KeyError:
        raise ValueError(f"Unknown strategy '{name}'. Available: {sorted(STRATEGIES) + sorted(ALIASES)}") from None
    return cls(params, build_fn, verbose=verbose)

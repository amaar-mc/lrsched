from collections.abc import Callable
from math import isfinite

from ._validate import check_step

# A group schedule maps a step to a dict of {group_name: learning_rate}.
GroupSchedule = Callable[[int], dict[str, float]]


def scale_by_group(
    schedule: Callable[[int], float], *, multipliers: dict[str, float]
) -> GroupSchedule:
    """Apply per-group learning-rate multipliers to a base schedule.

    At each step the base schedule produces one learning rate, and each named
    group scales that rate by its fixed multiplier. Returns a function from step
    to a dict mapping each group name to its learning rate.

    multipliers: mapping of group name to a positive finite float. Must not be empty.

    Raises ValueError for an empty mapping or for any multiplier that is not
    positive and finite.
    """
    if len(multipliers) == 0:
        raise ValueError("multipliers must not be empty; pass at least one group")
    for name, m in multipliers.items():
        if not isfinite(m) or m <= 0.0:
            raise ValueError(
                f"multiplier for group {name!r} must be a positive finite number, received {m!r}"
            )
    # Snapshot: freeze the dict so later mutation of the caller's dict has no effect.
    _multipliers: dict[str, float] = dict(multipliers)

    def apply(step: int) -> dict[str, float]:
        check_step(step)
        base = schedule(step)
        return {group: base * m for group, m in _multipliers.items()}

    return apply

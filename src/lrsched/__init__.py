"""Framework-agnostic learning-rate schedules as pure functions, with zero dependencies."""

from ._types import Schedule
from .clr import exp_range, triangular, triangular2
from .compose import sample, sequential, with_warmup
from .groups import GroupSchedule, scale_by_group
from .schedules import (
    constant,
    cosine,
    cosine_restarts,
    exponential,
    exponential_decay,
    inverse_sqrt,
    linear,
    multi_step,
    one_cycle,
    polynomial,
    polynomial_decay,
    step_decay,
)

__all__ = [
    "GroupSchedule",
    "Schedule",
    "constant",
    "cosine",
    "cosine_restarts",
    "exp_range",
    "exponential",
    "exponential_decay",
    "inverse_sqrt",
    "linear",
    "multi_step",
    "one_cycle",
    "polynomial",
    "polynomial_decay",
    "sample",
    "scale_by_group",
    "sequential",
    "step_decay",
    "triangular",
    "triangular2",
    "with_warmup",
]
__version__ = "0.4.0"

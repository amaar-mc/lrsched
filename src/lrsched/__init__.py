"""Framework-agnostic learning-rate schedules as pure functions, with zero dependencies."""

from ._types import Schedule
from .clr import exp_range, triangular, triangular2
from .compose import sample, sequential, with_warmup
from .schedules import (
    constant,
    cosine,
    cosine_restarts,
    exponential,
    inverse_sqrt,
    linear,
    multi_step,
    one_cycle,
    polynomial,
    step_decay,
)

__all__ = [
    "Schedule",
    "constant",
    "cosine",
    "cosine_restarts",
    "exp_range",
    "exponential",
    "inverse_sqrt",
    "linear",
    "multi_step",
    "one_cycle",
    "polynomial",
    "sample",
    "sequential",
    "step_decay",
    "triangular",
    "triangular2",
    "with_warmup",
]
__version__ = "0.2.0"

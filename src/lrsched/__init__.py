"""Framework-agnostic learning-rate schedules as pure functions, with zero dependencies."""

from ._types import Schedule
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
    "exponential",
    "inverse_sqrt",
    "linear",
    "multi_step",
    "one_cycle",
    "polynomial",
    "sample",
    "sequential",
    "step_decay",
    "with_warmup",
]
__version__ = "0.1.0"

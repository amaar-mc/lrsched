from math import isfinite


def check_finite(name: str, value: float) -> float:
    if not isfinite(value):
        raise ValueError(f"{name} must be a finite number, received {value!r}")
    return value


def check_positive_int(name: str, value: int) -> int:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, received {value!r}")
    return value


def check_step(step: int) -> int:
    if not isinstance(step, int) or step < 0:
        raise ValueError(f"step must be a non-negative integer, received {step!r}")
    return step

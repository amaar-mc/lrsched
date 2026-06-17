from collections.abc import Sequence
from math import cos, floor, pi

from ._types import Schedule
from ._validate import check_finite, check_positive_int, check_step


def constant(*, lr: float) -> Schedule:
    """A flat learning rate."""
    check_finite("lr", lr)

    def schedule(step: int) -> float:
        check_step(step)
        return lr

    return schedule


def step_decay(*, base_lr: float, drop: float, step_size: int) -> Schedule:
    """Decay by drop every step_size steps: base_lr * drop ** floor(step / step_size).

    drop must satisfy 0 < drop <= 1 (a multiplicative decay or hold).
    base_lr must be finite and positive.
    """
    check_finite("base_lr", base_lr)
    if base_lr <= 0.0:
        raise ValueError(f"base_lr must be a positive number, received {base_lr!r}")
    check_finite("drop", drop)
    if not (0.0 < drop <= 1.0):
        raise ValueError(f"drop must satisfy 0 < drop <= 1, received {drop!r}")
    check_positive_int("step_size", step_size)

    def schedule(step: int) -> float:
        check_step(step)
        return base_lr * float(drop ** floor(step / step_size))

    return schedule


def multi_step(*, base_lr: float, milestones: Sequence[int], gamma: float) -> Schedule:
    """Multiply by gamma at each milestone step."""
    check_finite("base_lr", base_lr)
    check_finite("gamma", gamma)
    ordered = sorted(milestones)
    for m in ordered:
        check_positive_int("milestone", m)

    def schedule(step: int) -> float:
        check_step(step)
        drops = sum(1 for m in ordered if step >= m)
        return base_lr * gamma**drops

    return schedule


def exponential(*, base_lr: float, gamma: float) -> Schedule:
    """Decay by a constant factor gamma each step: base_lr * gamma ** step."""
    check_finite("base_lr", base_lr)
    check_finite("gamma", gamma)

    def schedule(step: int) -> float:
        check_step(step)
        return base_lr * gamma**step

    return schedule


def polynomial(*, base_lr: float, end_lr: float, total_steps: int, power: float) -> Schedule:
    """Polynomial decay from base_lr to end_lr over total_steps, then hold end_lr."""
    check_finite("base_lr", base_lr)
    check_finite("end_lr", end_lr)
    check_positive_int("total_steps", total_steps)
    check_finite("power", power)

    def schedule(step: int) -> float:
        check_step(step)
        t = min(step, total_steps)
        remaining = 1.0 - t / total_steps
        return float(end_lr + (base_lr - end_lr) * remaining**power)

    return schedule


def linear(*, base_lr: float, end_lr: float, total_steps: int) -> Schedule:
    """Linear interpolation from base_lr to end_lr over total_steps, then hold end_lr."""
    check_finite("base_lr", base_lr)
    check_finite("end_lr", end_lr)
    check_positive_int("total_steps", total_steps)

    def schedule(step: int) -> float:
        check_step(step)
        t = min(step, total_steps)
        return base_lr + (end_lr - base_lr) * (t / total_steps)

    return schedule


def cosine(*, base_lr: float, min_lr: float, total_steps: int) -> Schedule:
    """Cosine annealing from base_lr at step 0 to min_lr at total_steps, then hold."""
    check_finite("base_lr", base_lr)
    check_finite("min_lr", min_lr)
    check_positive_int("total_steps", total_steps)

    def schedule(step: int) -> float:
        check_step(step)
        t = min(step, total_steps)
        return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + cos(pi * t / total_steps))

    return schedule


def cosine_restarts(*, base_lr: float, min_lr: float, period: int, t_mult: int) -> Schedule:
    """Cosine annealing with warm restarts (SGDR). Each cycle resets to base_lr; cycle
    length is multiplied by t_mult after each restart (t_mult = 1 keeps it periodic)."""
    check_finite("base_lr", base_lr)
    check_finite("min_lr", min_lr)
    check_positive_int("period", period)
    check_positive_int("t_mult", t_mult)

    def schedule(step: int) -> float:
        check_step(step)
        start = 0
        current = period
        while step >= start + current:
            start += current
            current *= t_mult
        t = step - start
        return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + cos(pi * t / current))

    return schedule


def inverse_sqrt(*, base_lr: float, warmup_steps: int) -> Schedule:
    """Transformer schedule: linear warmup to base_lr at warmup_steps, then decay by the
    inverse square root of the step."""
    check_finite("base_lr", base_lr)
    check_positive_int("warmup_steps", warmup_steps)

    def schedule(step: int) -> float:
        check_step(step)
        if step < warmup_steps:
            return base_lr * step / warmup_steps
        return float(base_lr * (warmup_steps / step) ** 0.5)

    return schedule


def one_cycle(
    *, max_lr: float, total_steps: int, pct_start: float, initial_div: float, final_div: float
) -> Schedule:
    """One-cycle policy: cosine ramp up to max_lr over the first pct_start of training,
    then cosine anneal down to max_lr / final_div. Starts at max_lr / initial_div."""
    check_finite("max_lr", max_lr)
    check_positive_int("total_steps", total_steps)
    check_finite("pct_start", pct_start)
    check_finite("initial_div", initial_div)
    check_finite("final_div", final_div)
    if not 0.0 < pct_start < 1.0:
        raise ValueError(f"pct_start must be between 0 and 1, received {pct_start!r}")
    if initial_div <= 0.0 or final_div <= 0.0:
        raise ValueError("initial_div and final_div must be positive")
    if total_steps < 2:
        raise ValueError("total_steps must be at least 2 for one_cycle")

    warmup = min(max(1, round(pct_start * total_steps)), total_steps - 1)
    start_lr = max_lr / initial_div
    end_lr = max_lr / final_div

    def schedule(step: int) -> float:
        check_step(step)
        if step <= warmup:
            frac = step / warmup
            return start_lr + 0.5 * (max_lr - start_lr) * (1.0 - cos(pi * frac))
        t = min(step, total_steps)
        frac = (t - warmup) / (total_steps - warmup)
        return end_lr + 0.5 * (max_lr - end_lr) * (1.0 + cos(pi * frac))

    return schedule


def polynomial_decay(*, start_lr: float, end_lr: float, total_steps: int, power: float) -> Schedule:
    """Polynomial decay from start_lr to end_lr over total_steps, then hold end_lr.

    lr(t) = (start_lr - end_lr) * (1 - min(t, total_steps) / total_steps) ** power + end_lr.
    power must be positive. total_steps must be a positive integer.
    start_lr and end_lr must be finite.
    """
    check_finite("start_lr", start_lr)
    check_finite("end_lr", end_lr)
    check_positive_int("total_steps", total_steps)
    check_finite("power", power)
    if power <= 0.0:
        raise ValueError(f"power must be positive, received {power!r}")

    def schedule(step: int) -> float:
        check_step(step)
        t = min(step, total_steps)
        remaining = 1.0 - t / total_steps
        return float((start_lr - end_lr) * remaining**power + end_lr)

    return schedule


def exponential_decay(*, base_lr: float, decay_rate: float, decay_steps: int) -> Schedule:
    """Continuous exponential decay: base_lr * decay_rate ** (step / decay_steps).

    decay_rate must satisfy 0 < decay_rate < 1 (strict decay).
    decay_steps must be a positive integer.
    base_lr must be finite and positive.
    """
    check_finite("base_lr", base_lr)
    if base_lr <= 0.0:
        raise ValueError(f"base_lr must be a positive number, received {base_lr!r}")
    check_finite("decay_rate", decay_rate)
    if not (0.0 < decay_rate < 1.0):
        raise ValueError(f"decay_rate must satisfy 0 < decay_rate < 1, received {decay_rate!r}")
    check_positive_int("decay_steps", decay_steps)

    def schedule(step: int) -> float:
        check_step(step)
        return base_lr * float(decay_rate ** (step / decay_steps))

    return schedule

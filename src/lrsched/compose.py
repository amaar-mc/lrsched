from collections.abc import Sequence

from ._types import Schedule
from ._validate import check_finite, check_positive_int, check_step


def with_warmup(schedule: Schedule, *, warmup_steps: int, start_lr: float) -> Schedule:
    """Prepend a linear warmup from start_lr to the wrapped schedule's value at step 0.
    After warmup, the wrapped schedule runs with its step counted from the warmup end."""
    check_positive_int("warmup_steps", warmup_steps)
    check_finite("start_lr", start_lr)
    target = schedule(0)

    def wrapped(step: int) -> float:
        check_step(step)
        if step < warmup_steps:
            return start_lr + (target - start_lr) * (step / warmup_steps)
        return schedule(step - warmup_steps)

    return wrapped


def sequential(phases: Sequence[tuple[int, Schedule]]) -> Schedule:
    """Run schedules back to back. Each phase is (duration, schedule), and within a phase
    the schedule sees a step counted from that phase's start. Past the final phase, the
    last phase's final value is held."""
    if len(phases) == 0:
        raise ValueError("sequential requires at least one phase")
    for duration, _ in phases:
        check_positive_int("phase duration", duration)

    def wrapped(step: int) -> float:
        check_step(step)
        offset = 0
        for duration, schedule in phases:
            if step < offset + duration:
                return schedule(step - offset)
            offset += duration
        last_duration, last_schedule = phases[-1]
        return last_schedule(last_duration - 1)

    return wrapped


def sample(schedule: Schedule, *, num_steps: int) -> list[float]:
    """Evaluate a schedule at steps 0..num_steps-1, useful for plotting or testing."""
    check_positive_int("num_steps", num_steps)
    return [schedule(i) for i in range(num_steps)]

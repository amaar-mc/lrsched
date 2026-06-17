from ._types import Schedule
from ._validate import check_finite, check_positive_int, check_step

# Cyclical learning rates (Smith, 2017). A full cycle spans 2 * step_size steps: the rate
# climbs from min_lr to max_lr over the first step_size steps, then falls back.


def _cycle_position(step: int, step_size: int) -> tuple[int, float]:
    cycle = 1 + step // (2 * step_size)
    distance = abs(step / step_size - 2 * cycle + 1)
    return cycle, max(0.0, 1.0 - distance)


def triangular(*, min_lr: float, max_lr: float, step_size: int) -> Schedule:
    """Triangular cyclical learning rate: oscillates linearly between min_lr and max_lr
    with a fixed amplitude, peaking every 2 * step_size steps."""
    check_finite("min_lr", min_lr)
    check_finite("max_lr", max_lr)
    check_positive_int("step_size", step_size)

    def schedule(step: int) -> float:
        check_step(step)
        _, height = _cycle_position(step, step_size)
        return min_lr + (max_lr - min_lr) * height

    return schedule


def triangular2(*, min_lr: float, max_lr: float, step_size: int) -> Schedule:
    """Triangular cyclical learning rate whose amplitude halves after each cycle."""
    check_finite("min_lr", min_lr)
    check_finite("max_lr", max_lr)
    check_positive_int("step_size", step_size)

    def schedule(step: int) -> float:
        check_step(step)
        cycle, height = _cycle_position(step, step_size)
        return min_lr + (max_lr - min_lr) * height * (0.5 ** (cycle - 1))

    return schedule


def exp_range(*, min_lr: float, max_lr: float, step_size: int, gamma: float) -> Schedule:
    """Triangular cyclical learning rate whose amplitude decays by gamma each step."""
    check_finite("min_lr", min_lr)
    check_finite("max_lr", max_lr)
    check_positive_int("step_size", step_size)
    check_finite("gamma", gamma)

    def schedule(step: int) -> float:
        check_step(step)
        _, height = _cycle_position(step, step_size)
        return min_lr + (max_lr - min_lr) * height * (gamma**step)

    return schedule

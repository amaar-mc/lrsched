from collections.abc import Callable

# A schedule maps a non-negative integer step to a learning rate.
Schedule = Callable[[int], float]

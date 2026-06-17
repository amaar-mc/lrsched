# Architecture

`lrsched` is a few small pure modules. The core idea is that a schedule is a function
`Callable[[int], float]` from step to learning rate, built by a factory that captures its
parameters in a closure.

## Modules

- `_types.py` defines `Schedule`.
- `_validate.py` holds the shared checks (finite floats, positive integers, non-negative
  step) so every factory reports bad input the same way.
- `schedules.py` holds the factories. Each validates its parameters once, then returns a
  closure that validates the step and computes the rate.
- `compose.py` holds `with_warmup`, `sequential`, and `sample`, which operate on schedules
  rather than parameters.

## Formulas

With base learning rate b, minimum m, total steps T, and step s clamped to T where noted:

- step_decay: `b * gamma ** (s // step_size)`.
- multi_step: `b * gamma ** (number of milestones <= s)`.
- exponential: `b * gamma ** s`.
- linear: `b + (end - b) * (min(s, T) / T)`.
- polynomial: `end + (b - end) * (1 - min(s, T) / T) ** power`.
- cosine: `m + 0.5 * (b - m) * (1 + cos(pi * min(s, T) / T))`.
- cosine_restarts: the cosine formula within the current cycle, where cycle length starts
  at `period` and is multiplied by `t_mult` after each restart. The current cycle is found
  by stepping through cycle lengths until the step falls inside one.
- inverse_sqrt: `b * s / warmup` during warmup, then `b * sqrt(warmup / s)`.
- one_cycle: a cosine ramp from `max_lr / initial_div` up to `max_lr` over the first
  `pct_start` of training, then a cosine anneal down to `max_lr / final_div`.

## Composition

`with_warmup` reads the wrapped schedule's value at step 0 as the warmup target, ramps
linearly from `start_lr` to that target over the warmup, then runs the wrapped schedule
with the step shifted so it begins at its own step 0 when warmup ends. `sequential` maps a
global step into the active phase and passes a phase-local step to that phase's schedule.

## Edges

Finite schedules clamp the step to their total, so they hold their final value rather than
extrapolating past the end. A negative step raises, since it is almost always a bug.

## Why pure Python

The formulas are small and exact. Implementing them with no dependencies means the same
schedule runs under PyTorch, JAX, NumPy, or a plain loop, and tests are exact rather than
approximate.

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
- `groups.py` holds `scale_by_group`, which wraps any schedule with per-group multipliers
  and returns a `GroupSchedule` (`Callable[[int], dict[str, float]]`).

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

## Cyclical learning rates

`clr.py` implements the Smith (2017) cyclical schedules. A cycle spans `2 * step_size`
steps. With `cycle = 1 + step // (2 * step_size)` and `height = max(0, 1 - |step /
step_size - 2 * cycle + 1|)`, the rate is `min_lr + (max_lr - min_lr) * height` for
`triangular`. `triangular2` multiplies the amplitude by `0.5 ** (cycle - 1)`, and
`exp_range` multiplies it by `gamma ** step`.

## Command line

`cli.py` exposes a `lrsched` console script that builds a schedule from subcommand flags
and prints `sample(schedule, num_steps=steps)` as one value per step, or as a sparkline.
`run(argv)` is the testable entry point and `main()` is the console hook, so tests drive
the CLI without touching `sys.argv`.

## Per-group multipliers

`scale_by_group(schedule, multipliers=...)` validates the multipliers once (all must be
positive and finite, mapping must be non-empty), snapshots the dict, and returns a closure
that evaluates `schedule(step)` and multiplies each entry. The `GroupSchedule` type alias
(`Callable[[int], dict[str, float]]`) is exported from the package for type annotations.

## Why pure Python

The formulas are small and exact. Implementing them with no dependencies means the same
schedule runs under PyTorch, JAX, NumPy, or a plain loop, and tests are exact rather than
approximate.

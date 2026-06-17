# Charter

## Purpose

Provide correct, dependency-free learning-rate schedules as pure functions, so that any
training loop or framework, or none, can use the same schedule and so that schedules are
easy to test, plot, and reason about.

## Scope

- The common schedules: constant, step, multi-step, exponential, linear, polynomial,
  cosine, cosine with warm restarts, inverse square root, and one-cycle.
- Composition: warmup, phase chaining, and sampling.

## Non-goals

- An optimizer or training framework. Schedules return a learning rate; the caller
  applies it.
- Framework coupling. There is no dependency on PyTorch, JAX, or any array library.
- Stateful schedulers. Schedules are pure functions of the step.

## Principles

- Correctness first. Each schedule is tested against its exact formula at known steps.
- Small, stable public API. Pure functions, explicit parameters.
- Zero runtime dependencies.
- Predictable edges. Schedules hold their final value past the end; a negative step
  raises.

## Audience

People writing custom training loops, using non-PyTorch stacks, driving schedules from
config, or teaching how learning-rate schedules behave.

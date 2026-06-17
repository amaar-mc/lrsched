# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0]

### Added
- `scale_by_group(schedule, *, multipliers)` applies per-group learning-rate multipliers to
  any base schedule, returning a function from step to `dict[str, float]`. Designed for
  discriminative or layer-wise learning rates: each named group scales the base rate by its
  fixed positive multiplier. Also exports `GroupSchedule` as a type alias for the return type.

## [0.2.0]

### Added
- Cyclical learning rates (Smith, 2017): `triangular`, `triangular2` (amplitude halves each cycle), and `exp_range` (amplitude decays by gamma).
- A `lrsched` command-line tool to sample a schedule, printing one value per step or a sparkline. Supports cosine, linear, exponential, step, triangular, and one-cycle.

## [0.1.0]

### Added
- Schedule type (a callable from step to learning rate).
- Schedules: constant, step_decay, multi_step, exponential, linear, polynomial, cosine,
  cosine_restarts (SGDR), inverse_sqrt (Transformer), one_cycle.
- Composition: with_warmup, sequential, sample.
- Validation with clear errors. Test suite with exact reference values and Hypothesis
  invariants.

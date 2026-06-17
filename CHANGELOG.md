# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Cyclical triangular schedules.
- A small CLI to print or sample a schedule.

## [0.1.0]

### Added
- Schedule type (a callable from step to learning rate).
- Schedules: constant, step_decay, multi_step, exponential, linear, polynomial, cosine,
  cosine_restarts (SGDR), inverse_sqrt (Transformer), one_cycle.
- Composition: with_warmup, sequential, sample.
- Validation with clear errors. Test suite with exact reference values and Hypothesis
  invariants.

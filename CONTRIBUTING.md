# Contributing to lrsched

Thanks for your interest. This project values correctness, a small surface area,
and zero runtime dependencies.

## Development

```sh
uv venv
uv pip install -e ".[dev]"
uv run pytest -q
uv run ruff check .
uv run mypy src
```

A standard virtual environment with `pip install -e ".[dev]"` works the same way.

## Guidelines

- No runtime dependencies.
- Each schedule is a pure function from step to learning rate, with required
  keyword-only parameters and no default values.
- Every schedule needs an exact reference-value test at known steps, a shape check, and
  any invariant it should satisfy.
- A bug fix starts with a failing test.
- Run `uv run ruff format .` before committing.
- Commit messages follow `type(scope): description`.

## Adding a schedule

State the formula in the docstring and the README, give exact test values at a few steps,
and document the behavior past the end of training.

## Reporting issues

Open an issue with the schedule and parameters, the step, the expected learning rate with
a source, and the value you observed.

# lrsched

Pure-Python, framework-agnostic learning-rate schedules. Zero runtime dependencies.

## Commands

- Create env and install: `uv venv && uv pip install -e ".[dev]"`
- Test: `uv run pytest -q`
- Lint: `uv run ruff check .` (format with `uv run ruff format .`)
- Types: `uv run mypy src`
- Build: `uv build` (then `uv run --with twine twine check dist/*` before publishing)

## Architecture

`src/lrsched/`:
- `_types.py` the Schedule type (Callable[[int], float])
- `_validate.py` shared validation (finite floats, positive ints, step)
- `schedules.py` schedule factories (cosine, one_cycle, sgdr, inverse_sqrt, and so on)
- `compose.py` with_warmup, sequential, sample
- `__init__.py` public surface

See `docs/architecture.md` for the formulas.

## Conventions

- Each schedule factory returns a pure function from step to learning rate.
- Parameters are required keyword-only arguments; no default values.
- Schedules hold their final value past the end; a negative step raises.
- No runtime dependencies. Keep the public API small.

## Testing rules

- Exact value of each schedule at known steps.
- Shape checks (restart boundaries, one-cycle peak, warmup handoff).
- Hypothesis invariants (cosine within bounds, warmup monotone).
- Bug fixes start with a failing test.

## Release

- Semantic versioning; update `CHANGELOG.md` and `__version__`.
- Gates: `uv run pytest && uv run ruff check . && uv run mypy src && uv build && uv run twine check dist/*`.
- Publish to PyPI, tag `vX.Y.Z`, GitHub release.

## Style

- No em dash characters in docs, comments, or commit messages.
- Comments explain non-obvious reasoning only.

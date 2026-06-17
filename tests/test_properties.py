from itertools import pairwise

from hypothesis import given
from hypothesis import strategies as st

from lrsched import constant, cosine, sample, with_warmup

steps = st.integers(min_value=1, max_value=1000)
rates = st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False)


@given(rates, rates, steps, st.integers(min_value=0, max_value=2000))
def test_cosine_stays_within_bounds(base: float, gap: float, total: int, step: int) -> None:
    base_lr = base + gap
    min_lr = base
    value = cosine(base_lr=base_lr, min_lr=min_lr, total_steps=total)(step)
    assert min_lr - 1e-9 <= value <= base_lr + 1e-9


@given(steps, rates)
def test_warmup_is_monotone_nondecreasing(warmup: int, target: float) -> None:
    main = cosine(base_lr=target, min_lr=0.0, total_steps=100)
    s = with_warmup(main, warmup_steps=warmup, start_lr=0.0)
    values = [s(i) for i in range(warmup + 1)]
    for a, b in pairwise(values):
        assert b >= a - 1e-9


@given(st.integers(min_value=1, max_value=500))
def test_sample_length(num_steps: int) -> None:
    assert len(sample(constant(lr=0.1), num_steps=num_steps)) == num_steps

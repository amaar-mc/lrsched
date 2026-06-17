import pytest

from lrsched import constant, cosine, sample, sequential, with_warmup


def test_with_warmup() -> None:
    s = with_warmup(cosine(base_lr=1.0, min_lr=0.0, total_steps=10), warmup_steps=5, start_lr=0.0)
    assert s(0) == pytest.approx(0.0)
    assert s(2) == pytest.approx(0.4)  # linear ramp 0 -> 1 over 5 steps
    assert s(5) == pytest.approx(1.0)  # hands off to cosine at its step 0
    assert s(10) == pytest.approx(cosine(base_lr=1.0, min_lr=0.0, total_steps=10)(5))


def test_sequential() -> None:
    s = sequential([(5, constant(lr=0.1)), (5, constant(lr=0.2))])
    assert s(0) == pytest.approx(0.1)
    assert s(4) == pytest.approx(0.1)
    assert s(5) == pytest.approx(0.2)
    assert s(9) == pytest.approx(0.2)
    assert s(50) == pytest.approx(0.2)  # held at the last phase past the end


def test_sequential_requires_a_phase() -> None:
    with pytest.raises(ValueError):
        sequential([])


def test_sample() -> None:
    assert sample(constant(lr=0.5), num_steps=3) == [0.5, 0.5, 0.5]
    assert len(sample(cosine(base_lr=1.0, min_lr=0.0, total_steps=100), num_steps=100)) == 100

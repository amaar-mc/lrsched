import pytest

from lrsched import exp_range, triangular, triangular2


def test_triangular() -> None:
    s = triangular(min_lr=0.0, max_lr=1.0, step_size=10)
    assert s(0) == pytest.approx(0.0)
    assert s(5) == pytest.approx(0.5)
    assert s(10) == pytest.approx(1.0)  # peak at step_size
    assert s(15) == pytest.approx(0.5)
    assert s(20) == pytest.approx(0.0)  # back to min at 2 * step_size


def test_triangular2_amplitude_halves_each_cycle() -> None:
    s = triangular2(min_lr=0.0, max_lr=1.0, step_size=10)
    assert s(0) == pytest.approx(0.0)
    assert s(10) == pytest.approx(1.0)  # first cycle peak
    assert s(30) == pytest.approx(0.5)  # second cycle peak halved


def test_exp_range_amplitude_decays() -> None:
    s = exp_range(min_lr=0.0, max_lr=1.0, step_size=10, gamma=0.9)
    assert s(0) == pytest.approx(0.0)
    assert s(10) == pytest.approx(0.9**10)


def test_bounds_and_validation() -> None:
    s = triangular(min_lr=0.1, max_lr=0.5, step_size=4)
    for step in range(0, 40):
        assert 0.1 - 1e-9 <= s(step) <= 0.5 + 1e-9
    with pytest.raises(ValueError):
        triangular(min_lr=0.0, max_lr=1.0, step_size=0)
    with pytest.raises(ValueError):
        triangular(min_lr=0.0, max_lr=1.0, step_size=10)(-1)

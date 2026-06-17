import pytest

from lrsched import (
    constant,
    cosine,
    cosine_restarts,
    exponential,
    inverse_sqrt,
    linear,
    multi_step,
    one_cycle,
    polynomial,
    step_decay,
)


def test_constant() -> None:
    s = constant(lr=0.1)
    assert s(0) == 0.1
    assert s(1000) == 0.1


def test_exponential() -> None:
    s = exponential(base_lr=1.0, gamma=0.9)
    assert s(0) == pytest.approx(1.0)
    assert s(1) == pytest.approx(0.9)
    assert s(2) == pytest.approx(0.81)


def test_step_decay() -> None:
    s = step_decay(base_lr=1.0, gamma=0.5, step_size=10)
    assert s(0) == pytest.approx(1.0)
    assert s(9) == pytest.approx(1.0)
    assert s(10) == pytest.approx(0.5)
    assert s(20) == pytest.approx(0.25)


def test_multi_step() -> None:
    s = multi_step(base_lr=1.0, milestones=[20, 10], gamma=0.1)
    assert s(9) == pytest.approx(1.0)
    assert s(10) == pytest.approx(0.1)
    assert s(19) == pytest.approx(0.1)
    assert s(20) == pytest.approx(0.01)


def test_linear() -> None:
    s = linear(base_lr=1.0, end_lr=0.0, total_steps=10)
    assert s(0) == pytest.approx(1.0)
    assert s(5) == pytest.approx(0.5)
    assert s(10) == pytest.approx(0.0)
    assert s(50) == pytest.approx(0.0)  # held past the end


def test_polynomial() -> None:
    s = polynomial(base_lr=1.0, end_lr=0.0, total_steps=10, power=2.0)
    assert s(0) == pytest.approx(1.0)
    assert s(5) == pytest.approx(0.25)
    assert s(10) == pytest.approx(0.0)


def test_cosine() -> None:
    s = cosine(base_lr=1.0, min_lr=0.0, total_steps=10)
    assert s(0) == pytest.approx(1.0)
    assert s(5) == pytest.approx(0.5)
    assert s(10) == pytest.approx(0.0)
    assert s(100) == pytest.approx(0.0)  # held at min past the end


def test_cosine_restarts_periodic() -> None:
    s = cosine_restarts(base_lr=1.0, min_lr=0.0, period=10, t_mult=1)
    assert s(0) == pytest.approx(1.0)
    assert s(5) == pytest.approx(0.5)
    assert s(10) == pytest.approx(1.0)  # restart
    assert s(15) == pytest.approx(0.5)


def test_cosine_restarts_growing() -> None:
    s = cosine_restarts(base_lr=1.0, min_lr=0.0, period=10, t_mult=2)
    assert s(10) == pytest.approx(1.0)  # restart, second cycle has length 20
    assert s(20) == pytest.approx(0.5)  # halfway through the length-20 cycle
    assert s(30) == pytest.approx(1.0)  # restart, third cycle


def test_inverse_sqrt() -> None:
    s = inverse_sqrt(base_lr=1.0, warmup_steps=10)
    assert s(0) == pytest.approx(0.0)
    assert s(5) == pytest.approx(0.5)
    assert s(10) == pytest.approx(1.0)  # peak at end of warmup
    assert s(40) == pytest.approx(0.5)  # sqrt(10/40)


def test_one_cycle() -> None:
    s = one_cycle(max_lr=1.0, total_steps=100, pct_start=0.3, initial_div=25.0, final_div=10000.0)
    assert s(0) == pytest.approx(0.04)  # max_lr / initial_div
    assert s(30) == pytest.approx(1.0)  # peak at end of the ramp
    assert s(100) == pytest.approx(0.0001)  # max_lr / final_div


def test_validation() -> None:
    with pytest.raises(ValueError):
        constant(lr=float("inf"))
    with pytest.raises(ValueError):
        cosine(base_lr=1.0, min_lr=0.0, total_steps=0)
    with pytest.raises(ValueError):
        one_cycle(max_lr=1.0, total_steps=100, pct_start=1.5, initial_div=25.0, final_div=1e4)
    with pytest.raises(ValueError):
        constant(lr=0.1)(-1)

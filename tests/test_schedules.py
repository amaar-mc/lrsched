from itertools import pairwise

import pytest

from lrsched import (
    constant,
    cosine,
    cosine_restarts,
    exponential,
    exponential_decay,
    inverse_sqrt,
    linear,
    multi_step,
    one_cycle,
    polynomial,
    polynomial_decay,
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
    s = step_decay(base_lr=1.0, drop=0.5, step_size=10)
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


def test_polynomial_decay_exact_values() -> None:
    # lr(t) = (start_lr - end_lr) * (1 - t / total_steps) ** power + end_lr
    # start_lr=1.0, end_lr=0.0, total_steps=10, power=2
    # t=0: (1.0 - 0.0) * (1 - 0/10)**2 + 0.0 = 1.0
    # t=5: 1.0 * (0.5)**2 = 0.25
    # t=10: 1.0 * (0.0)**2 = 0.0
    # t=20: clamped to t=10, holds 0.0
    s = polynomial_decay(start_lr=1.0, end_lr=0.0, total_steps=10, power=2.0)
    assert s(0) == pytest.approx(1.0)
    assert s(5) == pytest.approx(0.25)
    assert s(10) == pytest.approx(0.0)
    assert s(20) == pytest.approx(0.0)


def test_polynomial_decay_hits_boundaries() -> None:
    s = polynomial_decay(start_lr=2.0, end_lr=0.5, total_steps=8, power=3.0)
    # t=0: (2.0 - 0.5) * 1.0 + 0.5 = 2.0
    assert s(0) == pytest.approx(2.0)
    # t=8: (2.0 - 0.5) * 0.0 + 0.5 = 0.5
    assert s(8) == pytest.approx(0.5)
    assert s(100) == pytest.approx(0.5)  # held past the end


def test_polynomial_decay_monotone_nonincreasing_when_decaying() -> None:
    s = polynomial_decay(start_lr=1.0, end_lr=0.0, total_steps=20, power=1.5)
    values = [s(i) for i in range(21)]
    for a, b in pairwise(values):
        assert b <= a + 1e-9


def test_exponential_decay_exact_values() -> None:
    # lr(t) = base_lr * decay_rate ** (t / decay_steps)
    # base_lr=1.0, decay_rate=0.5, decay_steps=10
    # t=0:  1.0 * 0.5 ** 0.0 = 1.0
    # t=10: 1.0 * 0.5 ** 1.0 = 0.5
    # t=20: 1.0 * 0.5 ** 2.0 = 0.25
    # t=5:  1.0 * 0.5 ** 0.5 = sqrt(0.5) ~ 0.7071067811865476
    s = exponential_decay(base_lr=1.0, decay_rate=0.5, decay_steps=10)
    assert s(0) == pytest.approx(1.0)
    assert s(10) == pytest.approx(0.5)
    assert s(20) == pytest.approx(0.25)
    assert s(5) == pytest.approx(0.5**0.5)


def test_exponential_decay_monotone_decreasing() -> None:
    s = exponential_decay(base_lr=0.1, decay_rate=0.9, decay_steps=5)
    values = [s(i) for i in range(50)]
    for a, b in pairwise(values):
        assert b <= a + 1e-12


def test_step_decay_piecewise_constant() -> None:
    # Within each interval [k*step_size, (k+1)*step_size) the value is constant.
    s = step_decay(base_lr=1.0, drop=0.5, step_size=5)
    for step in range(0, 5):
        assert s(step) == pytest.approx(1.0)
    for step in range(5, 10):
        assert s(step) == pytest.approx(0.5)
    for step in range(10, 15):
        assert s(step) == pytest.approx(0.25)


def test_validation() -> None:
    with pytest.raises(ValueError):
        constant(lr=float("inf"))
    with pytest.raises(ValueError):
        cosine(base_lr=1.0, min_lr=0.0, total_steps=0)
    with pytest.raises(ValueError):
        one_cycle(max_lr=1.0, total_steps=100, pct_start=1.5, initial_div=25.0, final_div=1e4)
    with pytest.raises(ValueError):
        constant(lr=0.1)(-1)
    # polynomial_decay validation
    with pytest.raises(ValueError):
        polynomial_decay(start_lr=1.0, end_lr=0.0, total_steps=10, power=0.0)
    with pytest.raises(ValueError):
        polynomial_decay(start_lr=1.0, end_lr=0.0, total_steps=10, power=-1.0)
    with pytest.raises(ValueError):
        polynomial_decay(start_lr=float("nan"), end_lr=0.0, total_steps=10, power=1.0)
    with pytest.raises(ValueError):
        polynomial_decay(start_lr=1.0, end_lr=0.0, total_steps=0, power=1.0)
    with pytest.raises(ValueError):
        polynomial_decay(start_lr=1.0, end_lr=0.0, total_steps=10, power=1.0)(-1)
    # exponential_decay validation
    with pytest.raises(ValueError):
        exponential_decay(base_lr=1.0, decay_rate=0.0, decay_steps=10)
    with pytest.raises(ValueError):
        exponential_decay(base_lr=1.0, decay_rate=1.0, decay_steps=10)
    with pytest.raises(ValueError):
        exponential_decay(base_lr=1.0, decay_rate=1.1, decay_steps=10)
    with pytest.raises(ValueError):
        exponential_decay(base_lr=0.0, decay_rate=0.9, decay_steps=10)
    with pytest.raises(ValueError):
        exponential_decay(base_lr=1.0, decay_rate=0.9, decay_steps=0)
    with pytest.raises(ValueError):
        exponential_decay(base_lr=1.0, decay_rate=0.9, decay_steps=10)(-1)
    # step_decay validation
    with pytest.raises(ValueError):
        step_decay(base_lr=1.0, drop=0.0, step_size=10)
    with pytest.raises(ValueError):
        step_decay(base_lr=1.0, drop=1.1, step_size=10)
    with pytest.raises(ValueError):
        step_decay(base_lr=0.0, drop=0.5, step_size=10)
    with pytest.raises(ValueError):
        step_decay(base_lr=1.0, drop=0.5, step_size=10)(-1)

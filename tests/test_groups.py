import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from lrsched import GroupSchedule, constant, cosine, scale_by_group

# ---------------------------------------------------------------------------
# Exact-value tests
# ---------------------------------------------------------------------------


def test_exact_values_against_known_base() -> None:
    # cosine(base_lr=1.0, min_lr=0.0, total_steps=10) at step 5 is 0.5.
    base = cosine(base_lr=1.0, min_lr=0.0, total_steps=10)
    gs = scale_by_group(base, multipliers={"backbone": 0.1, "head": 1.0})
    result = gs(5)
    assert result["backbone"] == pytest.approx(0.5 * 0.1)
    assert result["head"] == pytest.approx(0.5 * 1.0)


def test_exact_values_constant_base() -> None:
    base = constant(lr=0.01)
    gs = scale_by_group(base, multipliers={"encoder": 0.01, "decoder": 0.1, "classifier": 1.0})
    result = gs(0)
    assert result["encoder"] == pytest.approx(0.01 * 0.01)
    assert result["decoder"] == pytest.approx(0.01 * 0.1)
    assert result["classifier"] == pytest.approx(0.01 * 1.0)


def test_exact_values_at_multiple_steps() -> None:
    base = cosine(base_lr=1.0, min_lr=0.0, total_steps=10)
    gs = scale_by_group(base, multipliers={"a": 2.0, "b": 0.5})
    for step in [0, 3, 7, 10, 20]:
        base_lr = base(step)
        result = gs(step)
        assert result["a"] == pytest.approx(base_lr * 2.0)
        assert result["b"] == pytest.approx(base_lr * 0.5)


def test_returns_correct_group_names() -> None:
    gs = scale_by_group(constant(lr=1.0), multipliers={"x": 0.1, "y": 0.2, "z": 0.3})
    result = gs(0)
    assert set(result.keys()) == {"x", "y", "z"}


# ---------------------------------------------------------------------------
# Invariant: multiplier 1.0 is identity relative to the base schedule
# ---------------------------------------------------------------------------


def test_multiplier_one_is_identity() -> None:
    base = cosine(base_lr=1e-3, min_lr=1e-5, total_steps=100)
    gs = scale_by_group(base, multipliers={"group": 1.0})
    for step in [0, 10, 50, 100, 200]:
        assert gs(step)["group"] == pytest.approx(base(step))


def test_multiplier_one_is_identity_constant() -> None:
    base = constant(lr=0.05)
    gs = scale_by_group(base, multipliers={"only": 1.0})
    for step in range(0, 100, 7):
        assert gs(step)["only"] == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# Invariant: comparing groups relative to each other
# ---------------------------------------------------------------------------


def test_group_ratio_is_constant_across_steps() -> None:
    # The ratio between two groups must equal the ratio of their multipliers at every step.
    base = cosine(base_lr=1.0, min_lr=0.0, total_steps=50)
    gs = scale_by_group(base, multipliers={"slow": 0.1, "fast": 1.0})
    for step in range(0, 60):
        result = gs(step)
        if result["fast"] != 0.0:
            assert result["slow"] / result["fast"] == pytest.approx(0.1)


def test_larger_multiplier_gives_larger_lr() -> None:
    base = cosine(base_lr=1.0, min_lr=0.0, total_steps=10)
    gs = scale_by_group(base, multipliers={"small": 0.01, "large": 10.0})
    for step in [1, 3, 5, 7]:
        result = gs(step)
        assert result["large"] > result["small"]


# ---------------------------------------------------------------------------
# Validation: rejects non-positive and non-finite multipliers
# ---------------------------------------------------------------------------


def test_rejects_zero_multiplier() -> None:
    with pytest.raises(ValueError, match="multiplier for group"):
        scale_by_group(constant(lr=0.1), multipliers={"g": 0.0})


def test_rejects_negative_multiplier() -> None:
    with pytest.raises(ValueError, match="multiplier for group"):
        scale_by_group(constant(lr=0.1), multipliers={"g": -0.5})


def test_rejects_inf_multiplier() -> None:
    with pytest.raises(ValueError, match="multiplier for group"):
        scale_by_group(constant(lr=0.1), multipliers={"g": float("inf")})


def test_rejects_nan_multiplier() -> None:
    with pytest.raises(ValueError, match="multiplier for group"):
        scale_by_group(constant(lr=0.1), multipliers={"g": float("nan")})


def test_rejects_empty_multipliers() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        scale_by_group(constant(lr=0.1), multipliers={})


def test_rejects_negative_step() -> None:
    gs = scale_by_group(constant(lr=0.1), multipliers={"g": 1.0})
    with pytest.raises(ValueError):
        gs(-1)


def test_mixed_valid_invalid_multipliers() -> None:
    # The invalid entry should raise even when other entries are fine.
    with pytest.raises(ValueError, match="multiplier for group"):
        scale_by_group(constant(lr=0.1), multipliers={"ok": 1.0, "bad": -1.0})


# ---------------------------------------------------------------------------
# Snapshot: mutating the original dict after construction has no effect
# ---------------------------------------------------------------------------


def test_snapshot_independence() -> None:
    mults: dict[str, float] = {"a": 0.5}
    base = constant(lr=1.0)
    gs = scale_by_group(base, multipliers=mults)
    mults["a"] = 99.0  # mutate the original
    assert gs(0)["a"] == pytest.approx(0.5)  # should still use 0.5


# ---------------------------------------------------------------------------
# Type: return type is GroupSchedule
# ---------------------------------------------------------------------------


def test_return_type_is_group_schedule() -> None:
    gs: GroupSchedule = scale_by_group(constant(lr=0.1), multipliers={"g": 1.0})
    result = gs(0)
    assert isinstance(result, dict)
    assert isinstance(result["g"], float)


# ---------------------------------------------------------------------------
# Property test: multiplied lr is always non-negative when base is non-negative
# ---------------------------------------------------------------------------

pos_floats = st.floats(min_value=1e-10, max_value=100.0, allow_nan=False, allow_infinity=False)
steps_st = st.integers(min_value=0, max_value=500)
total_steps_st = st.integers(min_value=1, max_value=500)


@given(pos_floats, pos_floats, total_steps_st, steps_st, pos_floats)
def test_scaled_lr_is_positive_when_base_is_positive(
    base_lr: float,
    gap: float,
    total: int,
    step: int,
    multiplier: float,
) -> None:
    sched = cosine(base_lr=base_lr + gap, min_lr=base_lr, total_steps=total)
    gs = scale_by_group(sched, multipliers={"g": multiplier})
    result = gs(step)["g"]
    assert math.isfinite(result)
    assert result >= 0.0

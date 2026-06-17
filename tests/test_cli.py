import pytest

from lrsched.cli import run


def test_cosine_prints_one_line_per_step(capsys: pytest.CaptureFixture[str]) -> None:
    code = run(
        ["cosine", "--base-lr", "1.0", "--min-lr", "0.0", "--total-steps", "10", "--steps", "11"]
    )
    assert code == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 11
    assert lines[0].split("\t")[0] == "0"


def test_triangular_sparkline(capsys: pytest.CaptureFixture[str]) -> None:
    code = run(
        [
            "triangular",
            "--min-lr",
            "0",
            "--max-lr",
            "1",
            "--step-size",
            "5",
            "--steps",
            "20",
            "--sparkline",
        ]
    )
    assert code == 0
    # Strip only the trailing newline: a leading space is a valid sparkline bar (the min).
    out = capsys.readouterr().out.rstrip("\n")
    assert len(out) == 20  # one character per sampled step


def test_one_cycle_runs(capsys: pytest.CaptureFixture[str]) -> None:
    code = run(
        [
            "one-cycle",
            "--max-lr",
            "1.0",
            "--total-steps",
            "100",
            "--pct-start",
            "0.3",
            "--initial-div",
            "25",
            "--final-div",
            "1000",
            "--steps",
            "100",
        ]
    )
    assert code == 0
    assert len(capsys.readouterr().out.strip().splitlines()) == 100

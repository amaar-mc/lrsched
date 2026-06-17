import argparse

from ._types import Schedule
from .clr import triangular
from .compose import sample
from .schedules import cosine, exponential, linear, one_cycle, step_decay

_BARS = " .:-=+*#%@"


def _sparkline(values: list[float]) -> str:
    lo = min(values)
    hi = max(values)
    span = hi - lo or 1.0
    last = len(_BARS) - 1
    return "".join(_BARS[min(last, int((v - lo) / span * last))] for v in values)


def _build(args: argparse.Namespace) -> Schedule:
    if args.schedule == "cosine":
        return cosine(base_lr=args.base_lr, min_lr=args.min_lr, total_steps=args.total_steps)
    if args.schedule == "linear":
        return linear(base_lr=args.base_lr, end_lr=args.end_lr, total_steps=args.total_steps)
    if args.schedule == "exponential":
        return exponential(base_lr=args.base_lr, gamma=args.gamma)
    if args.schedule == "step":
        return step_decay(base_lr=args.base_lr, gamma=args.gamma, step_size=args.step_size)
    if args.schedule == "triangular":
        return triangular(min_lr=args.min_lr, max_lr=args.max_lr, step_size=args.step_size)
    return one_cycle(
        max_lr=args.max_lr,
        total_steps=args.total_steps,
        pct_start=args.pct_start,
        initial_div=args.initial_div,
        final_div=args.final_div,
    )


def _parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--steps", type=int, required=True, help="number of steps to sample")
    common.add_argument("--sparkline", action="store_true", help="print a sparkline, not values")

    parser = argparse.ArgumentParser(
        prog="lrsched", description="Print or sample a learning-rate schedule."
    )
    sub = parser.add_subparsers(dest="schedule", required=True)

    cosine_p = sub.add_parser("cosine", parents=[common])
    cosine_p.add_argument("--base-lr", type=float, required=True)
    cosine_p.add_argument("--min-lr", type=float, required=True)
    cosine_p.add_argument("--total-steps", type=int, required=True)

    linear_p = sub.add_parser("linear", parents=[common])
    linear_p.add_argument("--base-lr", type=float, required=True)
    linear_p.add_argument("--end-lr", type=float, required=True)
    linear_p.add_argument("--total-steps", type=int, required=True)

    exp_p = sub.add_parser("exponential", parents=[common])
    exp_p.add_argument("--base-lr", type=float, required=True)
    exp_p.add_argument("--gamma", type=float, required=True)

    step_p = sub.add_parser("step", parents=[common])
    step_p.add_argument("--base-lr", type=float, required=True)
    step_p.add_argument("--gamma", type=float, required=True)
    step_p.add_argument("--step-size", type=int, required=True)

    tri_p = sub.add_parser("triangular", parents=[common])
    tri_p.add_argument("--min-lr", type=float, required=True)
    tri_p.add_argument("--max-lr", type=float, required=True)
    tri_p.add_argument("--step-size", type=int, required=True)

    one_p = sub.add_parser("one-cycle", parents=[common])
    one_p.add_argument("--max-lr", type=float, required=True)
    one_p.add_argument("--total-steps", type=int, required=True)
    one_p.add_argument("--pct-start", type=float, required=True)
    one_p.add_argument("--initial-div", type=float, required=True)
    one_p.add_argument("--final-div", type=float, required=True)

    return parser


def run(argv: list[str]) -> int:
    args = _parser().parse_args(argv)
    values = sample(_build(args), num_steps=args.steps)
    if args.sparkline:
        print(_sparkline(values))
    else:
        for step, value in enumerate(values):
            print(f"{step}\t{value:.8g}")
    return 0


def main() -> int:
    import sys

    return run(sys.argv[1:])

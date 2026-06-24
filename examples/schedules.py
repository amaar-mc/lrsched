"""Sample a few schedules and print them as small ASCII sparklines.

Run with: python examples/schedules.py
"""

from lrsched import cosine, one_cycle, sample, warmup_stable_decay, with_warmup

BARS = " .:-=+*#%@"


def sparkline(values: list[float]) -> str:
    lo = min(values)
    hi = max(values)
    span = hi - lo or 1.0
    return "".join(BARS[min(len(BARS) - 1, int((v - lo) / span * (len(BARS) - 1)))] for v in values)


total = 60
named = {
    "cosine": cosine(base_lr=1.0, min_lr=0.0, total_steps=total),
    "warmup+cosine": with_warmup(
        cosine(base_lr=1.0, min_lr=0.0, total_steps=total - 10), warmup_steps=10, start_lr=0.0
    ),
    "one_cycle": one_cycle(
        max_lr=1.0, total_steps=total, pct_start=0.3, initial_div=25.0, final_div=1000.0
    ),
    "wsd": warmup_stable_decay(
        base_lr=1.0,
        warmup_steps=10,
        decay_steps=20,
        total_steps=total,
        final_lr=0.0,
        warmup_start=0.0,
        decay_shape="1-sqrt",
    ),
}

for name, schedule in named.items():
    print(f"{name:>14}  {sparkline(sample(schedule, num_steps=total))}")

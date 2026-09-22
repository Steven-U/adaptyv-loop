"""B1: controlled regime-shift benchmark for context-aware history.

This is a synthetic software benchmark, not biological evidence.
"""

from __future__ import annotations

import argparse

from adaptyv_loop.regime_benchmark import (
    HISTORY_PER_METHOD,
    N_ROUNDS,
    POST_RATES,
    PRE_RATES,
    ROUND_SLOTS,
    run_benchmark,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=400)
    args = parser.parse_args()
    summary = run_benchmark(args.trials)

    print("B1 CONTROLLED REGIME SHIFT")
    print(
        f"pre-shift rates: A={PRE_RATES['method_a']:.0%}, B={PRE_RATES['method_b']:.0%}; "
        f"post-shift rates: A={POST_RATES['method_a']:.0%}, B={POST_RATES['method_b']:.0%}"
    )
    print(
        f"{HISTORY_PER_METHOD} old observations/method, {N_ROUNDS} rounds x "
        f"{ROUND_SLOTS} tests, {args.trials} deterministic seeds\n"
    )
    print(f"{'planner':<20}{'binders':>10}{'hit rate':>12}{'stale tests':>14}{'recovery':>12}")
    print("-" * 68)
    for planner, row in summary.items():
        print(
            f"{planner:<20}{row['binders']:>10.1f}"
            f"{row['hit_rate']:>12.1%}"
            f"{row['stale_tests']:>14.1f}"
            f"{row['recovery_round']:>12.2f}"
        )

    baseline = summary["context-free"]
    contextual = summary["context+recency"]
    lift = contextual["binders"] / baseline["binders"] - 1.0
    avoided = baseline["stale_tests"] - contextual["stale_tests"]
    print(
        f"\ncontext+recency vs context-free: {lift:+.1%} binders, "
        f"{avoided:.1f} stale-regime tests avoided."
    )
    print(
        "Interpretation: this benchmark only verifies behavior under an explicit "
        "synthetic regime shift; it is not evidence of biological lift."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

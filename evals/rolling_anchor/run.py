"""Count how seeded rolling plans end as admitted and as translated, in each of three worlds.

    uv run --package harness python evals/rolling_anchor/run.py
    uv run --package harness python evals/rolling_anchor/run.py --count 3000 --seed 7

It needs no model and no network. Every plan goes through `Plan` and the half
rule (`generator`).
"""

from __future__ import annotations

import argparse
import time

from generator import Outcome, WorldKind, tally


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--count", type=int, default=15_211)
    args = parser.parse_args()

    began = time.perf_counter()
    result = tally(args.seed, args.count)
    print(f"{args.count} admitted plans, seed {args.seed}, {time.perf_counter() - began:.1f} s")
    print(f"admitted but not compiled: {result.uncompiled}")
    print(f"reading a dataset's whole history: {result.whole_history}")
    print(f"restated at translation: {result.restated}")
    print()
    print(f"| world | side | {' | '.join(outcome.value for outcome in Outcome)} |")
    print(f"|---|---|{'---:|' * len(Outcome)}")
    for kind in WorldKind:
        for side, counts in (
            ("admitted", result.admitted[kind]),
            ("translated", result.translated[kind]),
        ):
            print(f"| {kind} | {side} | {' | '.join(str(counts[o]) for o in Outcome)} |")
    print()
    print(f"same-world admitted empties, translated: {_counts(result.admitted_empties)}")
    print(f"lagging-world anchor drops, translated: {_counts(result.dropped)}")


def _counts(counts: dict[Outcome, int]) -> str:
    return ", ".join(f"{outcome} {counts[outcome]}" for outcome in Outcome if counts[outcome])


if __name__ == "__main__":
    main()

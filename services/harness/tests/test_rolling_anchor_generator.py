"""CI's share of the seeded rolling-plan generator (`evals/rolling_anchor`).

500 admitted plans from a fixed seed, run as admitted and as translation
restates them, in the world each was planned over, one with newer years and one
with a lagging or stopped series. About 8 s, most of it compiling each plan and
each restated one against the live schemas. `evals/rolling_anchor/run.py` runs
the full count.

`evals/` sits outside the workspace, so the generator is loaded by file path.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _load() -> ModuleType:
    path = Path(__file__).resolve().parents[3] / "evals" / "rolling_anchor" / "generator.py"
    spec = importlib.util.spec_from_file_location("rolling_anchor_generator", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


generator = _load()
Outcome = generator.Outcome
WorldKind = generator.WorldKind


@pytest.fixture(scope="module")
def tally():
    return generator.tally(seed=0, count=500)


def test_plans_run_empty_as_admitted_are_refused_at_translation(tally) -> None:
    # The admitted empties are the control: without them the draws no longer reach the fault.
    assert tally.admitted[WorldKind.SAME][Outcome.EMPTY] > 0
    assert tally.admitted_empties.keys() <= {Outcome.RAN, Outcome.RESTATEMENT_REFUSED}
    assert tally.restated > 0


@pytest.mark.parametrize("kind", list(WorldKind))
def test_no_translated_run_reads_years_no_series_holds(tally, kind) -> None:
    assert tally.translated[kind][Outcome.EMPTY] == 0


def test_every_plan_admitted_and_every_restatement_admitted_compiles(tally) -> None:
    # Restated reads over a dataset's whole history are among them.
    assert tally.whole_history > 0
    assert tally.uncompiled == 0
    assert tally.translated[WorldKind.SAME][Outcome.RESTATEMENT_UNCOMPILED] == 0


def test_newer_years_never_trip_the_run_check(tally) -> None:
    assert tally.translated[WorldKind.NEWER][Outcome.ANCHOR_BEHIND] == 0


def test_every_dropped_anchor_is_refused_at_run(tally) -> None:
    assert tally.dropped[Outcome.ANCHOR_BEHIND] > 0
    assert tally.dropped.keys() == {Outcome.ANCHOR_BEHIND}


def test_series_coverage_misses_alike_on_both_sides(tally) -> None:
    same = WorldKind.SAME
    assert tally.translated[same][Outcome.MISSED] == tally.admitted[same][Outcome.MISSED]

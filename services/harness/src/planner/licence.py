"""The rule that decides which partly reported and projected years the user's words license.

A source holds back two kinds of year (`common.held_back.HeldBack`): a partly
reported one, such as OECD CRS's newest year, and a projected one, such as the
World Bank IDS years after the newest actual one. A read serves them only when its
window opts in with `include` (`"partial"`, `"projected"`), and it may opt in only
where the user's words license it. The licence (`Licence`) is the years the words
name and whether they ask about the future, held for the conversation in
`planner.stated_scope.LICENCE_STATE_KEY` (`planner.stated_scope.turn_licence`).

Search and get_data state every run a series' sources hold, whatever the read's
window (`resolved.coverage.held_back`), so a read's `include` is one function of
its years, the licence and those runs (`include_for`). It is written at three
sites: the windows a plan's retrievals pin (`planner.plan.licensed_include`, read
by `submission.service._window_pins` and `translator.discovery`), the window of a
source the translator found for a requirement with no pin
(`translator.discovery._licensed_outcome`), and the orchestrator's `fetch`
(`harness.orchestrator_tools._licensed_window`). Admission
(`submission.service._held_back_years_read`) refuses a read of a run the licence
does not grant (`unlicensed_year`, `endpoint_licensed`).

- A partly reported year is licensed for a read when it lies between the first and
  last named year, or when its run lies directly before a projected run the read
  also reaches and the words license (`run_licensed`).
- A projected run is licensed for a read when the words ask about the future, or
  name a year in the read at or after the run's first year (`projected_licensed`).
- A growth compares at a partly reported year only where the words name that year,
  and at a projected one under the projected rule over the growth's window
  (`endpoint_licensed`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal, NamedTuple

from common.held_back import HeldBack, held_back_of

HeldKind = Literal["partial", "projected"]

# The classes in the order a window's `include` lists them.
_KINDS: tuple[HeldKind, ...] = ("partial", "projected")


class Licence(NamedTuple):
    """The calendar years the user's words name, ascending, and whether they ask
    about the future or about projections."""

    named: tuple[int, ...] = ()
    projection: bool = False


def pin_block_held_back(block: Mapping[str, Any] | None) -> tuple[HeldBack, ...]:
    """The runs of years the spec `block`'s coverage states its sources hold back
    or served (`resolved.coverage.held_back`), empty for no block or no runs."""
    coverage = ((block or {}).get("resolved") or {}).get("coverage")
    if not isinstance(coverage, Mapping):
        return ()
    spec_id = (block or {}).get("spec_id")
    return held_back_of(coverage, label=spec_id if isinstance(spec_id, str) else None)


def closed_range(window: Mapping[str, Any]) -> tuple[int, int] | None:
    """The first and last year of `window`, a `requested_window`, where it is a
    range with both ends; `None` for any other window, which serves no held-back
    year, so a projection never reads as the newest figure."""
    start, end = window.get("start_year"), window.get("end_year")
    if window.get("kind") != "range" or not isinstance(start, int) or not isinstance(end, int):
        return None
    return start, end


def projected_licensed(licence: Licence, start: int, end: int, *, first: int) -> bool:
    """Whether the words license projected years from `first` on in a read of
    `start` to `end`: they ask about the future, or name a year in the read at or
    after `first`."""
    return licence.projection or any(
        start <= year <= end and year >= first for year in licence.named
    )


def _named_between(year: int, licence: Licence) -> bool:
    """Whether `year` lies between the first and last named year."""
    return bool(licence.named) and min(licence.named) <= year <= max(licence.named)


def _reaches(run: HeldBack, start: int, end: int) -> bool:
    return run.start_year <= end and start <= run.end_year


def run_licensed(
    run: HeldBack, start: int, end: int, *, licence: Licence, runs: Sequence[HeldBack]
) -> bool:
    """Whether the words license every year of `run` that a read of `start` to
    `end` reaches, among `runs`, every run the read's source states.

    A projected run follows `projected_licensed` from its first year. A partly
    reported year is licensed between the first and last named year, and a whole
    partly reported run directly before a projected run of `runs` the read reaches
    and the words license, since a read into the projections passes through it.
    """
    if run.kind == "projected":
        return projected_licensed(licence, start, end, first=run.start_year)
    years = range(max(start, run.start_year), min(end, run.end_year) + 1)
    if all(_named_between(year, licence) for year in years):
        return True
    return any(
        other.kind == "projected"
        and other.start_year == run.end_year + 1
        and _reaches(other, start, end)
        and projected_licensed(licence, start, end, first=other.start_year)
        for other in runs
    )


def include_for(
    start: int, end: int, *, licence: Licence, runs: Sequence[HeldBack] | None
) -> tuple[HeldKind, ...]:
    """A read of `start` to `end`'s `include`: each class of which `runs`, the runs
    the read's source states, hold a run the read reaches, and every such run is
    licensed (`run_licensed`).

    `runs` is `None` where no spec block is on record, so no run is known. The
    words then license from themselves alone. A partly reported year is
    licensed where the read's last year lies between the first and last named
    year, and projected years where the words ask about the future, since no
    run says where the projections start.
    """
    if runs is None:
        unknown = (("partial", _named_between(end, licence)), ("projected", licence.projection))
        return tuple(kind for kind, licensed in unknown if licensed)
    reached = [run for run in runs if _reaches(run, start, end)]
    return tuple(
        kind
        for kind in _KINDS
        if (of_kind := [run for run in reached if run.kind == kind])
        and all(run_licensed(run, start, end, licence=licence, runs=runs) for run in of_kind)
    )


def unlicensed_year(
    start: int, end: int, *, licence: Licence, runs: Sequence[HeldBack]
) -> tuple[int, HeldBack] | None:
    """The first year of a read of `start` to `end` that `runs` hold back and the
    words do not license (`run_licensed`), with its run; `None` where every held
    year the read reaches is licensed. Of a refused partly reported run, the first
    year not between the named ones is given."""
    reached = sorted(
        (run for run in runs if _reaches(run, start, end)), key=lambda run: run.start_year
    )
    for run in reached:
        if run_licensed(run, start, end, licence=licence, runs=runs):
            continue
        years = range(max(start, run.start_year), min(end, run.end_year) + 1)
        year = next((one for one in years if not _named_between(one, licence)), years[0])
        return year, run
    return None


def endpoint_licensed(
    year: int, run: HeldBack, *, window: tuple[int, int], licence: Licence
) -> bool:
    """Whether the words license a growth over `window`, its first and last year,
    comparing at `year` of `run`. A partly reported year must be one the words
    name; a projected one follows `projected_licensed` over the window."""
    if run.kind == "projected":
        return projected_licensed(licence, *window, first=run.start_year)
    return year in licence.named

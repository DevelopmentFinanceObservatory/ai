"""The held-back and reported-share facts dfo-apis states on a read's coverage.

`HeldBack` is one run of years a read's source holds back unless the read opts
in: the partly reported years of a source, or the projected years of an estimate that
continues a reported series. It sits in `held_back` on Search's
`resolved.coverage` and on a get_data manifest's `coverage`; both are optional,
and an absent or empty list means nothing was held back. `ReportedShare` is
how much of the prior year a partial period holds so far.

A read states every run its source holds, whatever its window and `include`.
`served` says whether the read served its years or left them out, and
`reached_held_back` keeps the runs the read's window reaches.

Every model drops unseen fields, so apis can add fields without breaking a
read, and `valid_entries` reads a list one entry at a time, so an entry in a
shape outside the contract is dropped with a warning and the rest are kept."""

import logging
from collections.abc import Iterable, Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError

logger = logging.getLogger(__name__)


class ReportedShare(BaseModel):
    """What fraction (`share`) of year `against` a partial period holds,
    compared on one `value` or on the `series` count: 0 or more, and above 1
    when the partial period already exceeds `against`."""

    model_config = ConfigDict(extra="ignore")
    basis: Literal["value", "series"]
    share: float
    against: int


class HeldBack(BaseModel):
    """Years `start_year` to `end_year` of `kind` that a source (`source`, shown
    as `source_name`) marks or continues: served when the read's `include`
    opted into them (`served`), left out otherwise. `reported_share` is set on
    a partial run only."""

    model_config = ConfigDict(extra="ignore")
    kind: Literal["partial", "projected"]
    served: bool = False
    source: str
    source_name: str | None = None
    start_year: int
    end_year: int
    reported_share: ReportedShare | None = None


def valid_entries[M: BaseModel](
    model: type[M], entries: Iterable[object], *, field: str, label: str | None
) -> list[M]:
    """Each of `entries` that validates as `model`, in order.

    An entry that fails is dropped with a warning naming `field`, the list it
    came from, and `label`, the spec_id or provenance being read.
    """
    kept: list[M] = []
    for index, entry in enumerate(entries):
        try:
            kept.append(model.model_validate(entry))
        except ValidationError as exc:
            logger.warning(
                "dropped %s[%d] of %s: %s",
                field,
                index,
                label or "an unnamed read",
                exc.errors(include_url=False, include_input=False),
            )
    return kept


def held_back_of(coverage: Mapping[str, Any], *, label: str | None = None) -> tuple[HeldBack, ...]:
    """The held-back runs of a `coverage` block, served or not; empty when it
    states none. An entry outside the contract is dropped with a warning naming
    `label` (`valid_entries`)."""
    entries = coverage.get("held_back")
    if not isinstance(entries, list):
        return ()
    return tuple(valid_entries(HeldBack, entries, field="held_back", label=label))


def reaches_left_out(start_year: int, *, last_year: int | None) -> bool:
    """Whether a read whose years end in `last_year` reaches a run it left out
    that starts in `start_year`: where it ends at or after the year before the
    run starts, so its figures would run on into it. A read stating no last
    year reaches it."""
    return last_year is None or last_year >= start_year - 1


def reached_held_back(
    coverage: Mapping[str, Any], *, label: str | None = None
) -> tuple[HeldBack, ...]:
    """The held-back runs of a `coverage` block (`held_back_of`) its read's
    window (`coverage.window`) reaches: every run it served, and each run it
    left out where the window ends at or after the year before the run starts
    (`reaches_left_out`). dfo-apis states every run a source holds whatever the
    window, so a read of 2005 to 2010 states the partly reported 2025 it never
    reaches, which this drops."""
    window = coverage.get("window")
    end = window.get("end_year") if isinstance(window, Mapping) else None
    last_year = end if isinstance(end, int) else None
    return tuple(
        run
        for run in held_back_of(coverage, label=label)
        if run.served or reaches_left_out(run.start_year, last_year=last_year)
    )


def newest_complete_year(end_year: int, runs: Iterable[HeldBack]) -> int:
    """The last year at or before `end_year` that no run of `runs` left out
    holds: `end_year` itself unless a left-out run covers it, else the year
    before that run starts, checked again against the other runs. A CRS
    window ending in 2025 whose partly reported 2025 was left out ends in
    2024; a window ending in a served projected year keeps its end."""
    left_out = [run for run in runs if not run.served]
    year = end_year
    while covering := next(
        (run for run in left_out if run.start_year <= year <= run.end_year), None
    ):
        year = covering.start_year - 1
    return year


def served_window(window: Mapping[str, Any], runs: Iterable[HeldBack]) -> Mapping[str, Any]:
    """`window`, a `{start_year, end_year}` coverage window, with its end at
    the newest complete year the left-out `runs` leave it
    (`newest_complete_year`). Unchanged where it states no end year, or where
    every year of it falls in runs left out."""
    end = window.get("end_year")
    if not isinstance(end, int):
        return window
    start = window.get("start_year")
    year = newest_complete_year(end, runs)
    if year == end or (isinstance(start, int) and year < start):
        return window
    return {**window, "end_year": year}

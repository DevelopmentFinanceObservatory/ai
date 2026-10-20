"""The sentence that says what a one-year frame's year holds beside nearby years.

A mint manifest's `coverage` states the year served (`window`), the members
the frame holds of each role (`dimensions`, each entry with the `noun` apis
calls that role's members by, where it states one), and, when apis sends them,
the members other years hold: `recent_years`, three calendar years newest
first, and `fullest_year`, the fullest of the last ten, each an entry of
`year`, `series` and `role_counts`, with `elected_year`, the year its
election chooses among the three. data-mcp's `get_data` digest and the
harness's tool results both state the sentence built here, so a model reads
one wording from either.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NamedTuple

# How many calendar years `recent_years` spans. The year served is one of them.
_RECENT_SPAN = 3

# The singular and plural a dimension's members are counted by when its
# `dimensions` entry states no `noun`.
_MEMBER_NOUNS = ("member", "members")


class _YearCount(NamedTuple):
    year: int
    role_counts: Mapping[str, int]


class _Counted(NamedTuple):
    """The dimension the note counts: its `role_counts` key, the members of
    it the frame holds, and the singular and plural they are called by."""

    role: str
    held: int
    nouns: tuple[str, str]


def period_note(coverage: Mapping[str, Any]) -> str | None:
    """One sentence naming the year `coverage` serves with its member count,
    and each other year worth setting beside it with its own; `None` when
    there is nothing to say.

    A frame of one year is noted for each of three facts: a newer year in
    `recent_years` holds data (the newest such year is named), the year
    served is not `elected_year`, the year apis' election chooses over
    `recent_years` (that year is named), and `coverage` carries a
    `fullest_year` (it is named). A recent year that holds more members than
    the frame is not otherwise mentioned. "So far" is said only of the newer
    year, and only when it is neither the elected nor the fullest year.

    `recent_years` describes the window only when the year served is one of
    its three calendar years. Entries of a list that leaves the served year
    out, an entry three or more years from it, and an `elected_year` the
    list does not hold each give no clause.

    Every count is of one dimension. A `role_counts` key is the `label` of
    the `dimensions` entry it counts, so the frame's count comes from that
    entry and each other year's from that year's `role_counts`: members are
    compared with members and no count is derived from `series`. The
    dimension is the first in `dimensions` order that a `role_counts` key
    names and that holds more than one member in the frame or in a listed
    year. Its members are called by its `noun`, the singular and plural
    apis words the role with for the frame's kind of flow (providers and
    recipients of aid, creditors and debtors of debt), and are "members"
    where the entry states none. With no such dimension, there is no note.
    """
    window = coverage.get("window")
    served = window.get("start_year") if isinstance(window, Mapping) else None
    if not isinstance(window, Mapping) or not isinstance(served, int):
        return None
    if window.get("end_year") != served:
        return None
    recent = _year_counts(coverage.get("recent_years"))
    if all(entry.year != served for entry in recent):
        recent = []
    recent = [entry for entry in recent if abs(entry.year - served) < _RECENT_SPAN]
    fullest = _year_counts([coverage.get("fullest_year")])
    listed = [entry for entry in (*recent, *fullest) if entry.year != served]
    counted = _counted_dimension(coverage.get("dimensions"), listed)
    if counted is None:
        return None
    role = counted.role

    elected_year = coverage.get("elected_year")
    newer = max(
        (entry for entry in recent if entry.year > served and entry.role_counts.get(role, 0) > 0),
        key=lambda entry: entry.year,
        default=None,
    )
    elected = next((entry for entry in recent if entry.year == elected_year), None)
    others = {
        entry.year: entry
        for entry in (newer, elected, *fullest)
        if entry is not None and entry.year != served and role in entry.role_counts
    }
    if not others:
        return None
    settled = {entry.year for entry in (elected, *fullest) if entry is not None}
    so_far = newer.year if newer is not None and newer.year not in settled else None
    clauses = [
        f"{year} covers {_members(entry.role_counts[role], counted.nouns)}"
        + (" so far" if year == so_far else "")
        for year, entry in others.items()
    ]
    return (
        f"The data is for {served}, which covers {_members(counted.held, counted.nouns)},"
        f" while {_listed(clauses)}."
    )


def _listed(clauses: list[str]) -> str:
    *head, last = clauses
    return f"{', '.join(head)} and {last}" if head else last


def _year_counts(entries: object) -> list[_YearCount]:
    """The readable entries of a `recent_years` list, in its order."""
    counts: list[_YearCount] = []
    for entry in entries if isinstance(entries, list) else ():
        if not isinstance(entry, Mapping) or not isinstance(year := entry.get("year"), int):
            continue
        role_counts = entry.get("role_counts")
        counts.append(
            _YearCount(
                year=year,
                role_counts={
                    str(role): count
                    for role, count in (
                        role_counts.items() if isinstance(role_counts, Mapping) else ()
                    )
                    if isinstance(count, int)
                },
            )
        )
    return counts


def _counted_dimension(dimensions: object, listed: list[_YearCount]) -> _Counted | None:
    """The dimension the note counts, matched to a `role_counts` key by its
    label, with the nouns its entry states (`_nouns`)."""
    roles = {role for entry in listed for role in entry.role_counts}
    for dimension in dimensions if isinstance(dimensions, list) else ():
        if not isinstance(dimension, Mapping):
            continue
        label, held = dimension.get("label"), dimension.get("count")
        if not isinstance(label, str) or label not in roles or not isinstance(held, int):
            continue
        if max(held, *(entry.role_counts.get(label, 0) for entry in listed)) > 1:
            return _Counted(role=label, held=held, nouns=_nouns(dimension.get("noun")))
    return None


def _nouns(noun: object) -> tuple[str, str]:
    """The singular and plural a `dimensions` entry's `noun` states, or
    `_MEMBER_NOUNS` unless it holds both as non-empty strings."""
    if not isinstance(noun, Mapping):
        return _MEMBER_NOUNS
    singular, plural = noun.get("singular"), noun.get("plural")
    if isinstance(singular, str) and singular and isinstance(plural, str) and plural:
        return singular, plural
    return _MEMBER_NOUNS


def _members(count: int, nouns: tuple[str, str]) -> str:
    singular, plural = nouns
    return f"{count:,} {singular if count == 1 else plural}"

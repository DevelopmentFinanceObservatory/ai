"""Tests for `common.period_note`: what a one-year frame's year holds beside nearby years."""

from __future__ import annotations

import re
from typing import Any

import pytest
from common.period_note import period_note

# The dimensions of an aid frame as dfo-apis sends them, each with the noun
# it words the role by.
_DONORS = {
    "label": "financingProvider",
    "axis": "origin",
    "noun": {"singular": "provider", "plural": "providers"},
}
_RECIPIENTS = {
    "label": "financingRecipient",
    "axis": "destination",
    "noun": {"singular": "recipient", "plural": "recipients"},
}
# A debt frame labels its creditors as an aid frame labels its providers.
_CREDITORS = {**_DONORS, "noun": {"singular": "creditor", "plural": "creditors"}}


def _year(year: int, donors: int) -> dict[str, Any]:
    """A year's entry as dfo-apis sends it: `role_counts` keyed by dimension label."""
    return {
        "year": year,
        "series": donors,
        "role_counts": {_DONORS["label"]: donors, _RECIPIENTS["label"]: 1},
    }


def _coverage(
    *,
    served: int,
    held: int,
    recent: list[dict[str, Any]] | None = None,
    elected: int | None = None,
    fullest: dict[str, Any] | None = None,
    dimensions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """A manifest's `coverage` for a frame of `held` donors to one recipient in `served`."""
    coverage: dict[str, Any] = {
        "rows": held,
        "dimensions": dimensions
        if dimensions is not None
        else [
            {**_DONORS, "count": held},
            {**_RECIPIENTS, "count": 1},
            {"label": "years", "count": 1},
        ],
        "window": {"kind": "latest", "start_year": served, "end_year": served},
    }
    if recent is not None:
        coverage["recent_years"] = recent
    if elected is not None:
        coverage["elected_year"] = elected
    if fullest is not None:
        coverage["fullest_year"] = fullest
    return coverage


def _named(note: str | None) -> list[tuple[int, int]]:
    """Each year the note names with the count stated after it, in order."""
    assert note is not None
    return [
        (int(year), int(count.replace(",", "")))
        for year, count in re.findall(r"\b((?:19|20)\d\d)\b\D+?([\d,]+)", note)
    ]


def _so_far(note: str | None) -> list[int]:
    """Each year whose count the note qualifies with "so far"."""
    assert note is not None
    return [int(year) for year in re.findall(r"\b(\d{4}) covers [\d,]+ \w+ so far", note)]


@pytest.mark.parametrize(
    ("coverage", "named", "so_far"),
    [
        pytest.param(
            _coverage(
                served=2024, held=85, recent=[_year(2025, 9), _year(2024, 85), _year(2023, 80)]
            ),
            [(2024, 85), (2025, 9)],
            [2025],
            id="a_newer_year_holds_data",
        ),
        pytest.param(
            _coverage(
                served=2025,
                held=9,
                recent=[_year(2025, 9), _year(2024, 85), _year(2023, 91)],
                elected=2024,
            ),
            [(2025, 9), (2024, 85)],
            [],
            id="a_named_partial_year_names_the_elected_year",
        ),
        pytest.param(
            _coverage(
                served=2024,
                held=85,
                recent=[_year(2025, 9), _year(2024, 85), _year(2023, 91)],
                elected=2024,
            ),
            [(2024, 85), (2025, 9)],
            [2025],
            id="the_elected_year_names_the_newer_year_alone",
        ),
        pytest.param(
            _coverage(
                served=2023,
                held=91,
                recent=[_year(2025, 9), _year(2024, 85), _year(2023, 91)],
                elected=2024,
                fullest=_year(2019, 120),
            ),
            [(2023, 91), (2025, 9), (2024, 85), (2019, 120)],
            [2025],
            id="a_newer_year_the_elected_year_and_the_fullest_year",
        ),
        pytest.param(
            _coverage(
                served=2024,
                held=309,
                recent=[_year(2024, 309), _year(2023, 300), _year(2022, 305)],
                fullest=_year(2021, 3344),
            ),
            [(2024, 309), (2021, 3344)],
            [],
            id="a_fullest_year",
        ),
        pytest.param(
            _coverage(served=2024, held=309, fullest=_year(2021, 3344)),
            [(2024, 309), (2021, 3344)],
            [],
            id="a_fullest_year_without_recent_years",
        ),
        pytest.param(
            # The count stated for the served year is the frame's own.
            _coverage(
                served=2024, held=70, recent=[_year(2025, 9), _year(2024, 85), _year(2023, 60)]
            ),
            [(2024, 70), (2025, 9)],
            [2025],
            id="the_served_count_is_the_frames_own",
        ),
        pytest.param(
            # The one newer year with data is the elected year, a complete year.
            _coverage(
                served=2023,
                held=91,
                recent=[_year(2025, 0), _year(2024, 85), _year(2023, 91)],
                elected=2024,
            ),
            [(2023, 91), (2024, 85)],
            [],
            id="the_newer_year_is_the_elected_year",
        ),
        pytest.param(
            _coverage(
                served=2023,
                held=40,
                recent=[_year(2024, 85), _year(2023, 40), _year(2022, 80)],
                fullest=_year(2024, 85),
            ),
            [(2023, 40), (2024, 85)],
            [],
            id="the_newer_year_is_the_fullest_year",
        ),
        pytest.param(
            # 2030 is outside the three calendar years that hold 2024.
            _coverage(
                served=2024, held=85, recent=[_year(2030, 4), _year(2025, 9), _year(2024, 85)]
            ),
            [(2024, 85), (2025, 9)],
            [2025],
            id="a_recent_year_outside_the_three_is_not_named",
        ),
        pytest.param(
            _coverage(
                served=2024,
                held=85,
                recent=[_year(2025, 9), _year(2024, 85), _year(2023, 91)],
                elected=2019,
            ),
            [(2024, 85), (2025, 9)],
            [2025],
            id="an_elected_year_outside_recent_years_is_not_named",
        ),
        pytest.param(
            # Recent years that leave the served year out give no clause.
            _coverage(
                served=2015,
                held=85,
                recent=[_year(2025, 9), _year(2024, 85), _year(2023, 91)],
                elected=2024,
                fullest=_year(2012, 120),
            ),
            [(2015, 85), (2012, 120)],
            [],
            id="recent_years_of_another_window_leave_the_fullest_year",
        ),
        pytest.param(
            _coverage(
                served=2025,
                held=1,
                recent=[_year(2025, 1), _year(2024, 85), _year(2023, 80)],
                elected=2024,
            ),
            [(2025, 1), (2024, 85)],
            [],
            id="a_frame_of_one_member_of_a_role_other_years_hold_many_of",
        ),
    ],
)
def test_the_note_names_the_served_year_and_each_other_year_with_its_count(
    coverage: dict[str, Any], named: list[tuple[int, int]], so_far: list[int]
) -> None:
    note = period_note(coverage)

    assert _named(note) == named
    assert _so_far(note) == so_far
    assert "provider" in note


@pytest.mark.parametrize(
    "coverage",
    [
        pytest.param(
            _coverage(
                served=2024, held=85, recent=[_year(2024, 85), _year(2023, 85), _year(2022, 85)]
            ),
            id="equal_counts_and_no_newer_year",
        ),
        pytest.param(
            _coverage(
                served=2024, held=85, recent=[_year(2025, 0), _year(2024, 85), _year(2023, 80)]
            ),
            id="a_newer_year_without_data",
        ),
        pytest.param(
            _coverage(
                served=2024,
                held=85,
                recent=[_year(2024, 85), _year(2023, 91), _year(2022, 90)],
                elected=2024,
            ),
            id="the_elected_year_served_beside_a_fuller_earlier_year",
        ),
        pytest.param(
            _coverage(
                served=2025, held=9, recent=[_year(2025, 9), _year(2024, 85), _year(2023, 91)]
            ),
            id="a_fuller_earlier_year_and_no_elected_year",
        ),
        pytest.param(_coverage(served=2024, held=85), id="no_members_from_dfo_apis"),
        pytest.param(
            {
                **_coverage(
                    served=2015,
                    held=85,
                    recent=[_year(2025, 9), _year(2024, 85), _year(2023, 91)],
                    elected=2024,
                ),
                "window": {"kind": "range", "start_year": 2015, "end_year": 2015},
            },
            id="recent_years_that_leave_the_served_year_out",
        ),
        pytest.param(
            # A key that is no dimension's label counts nothing, whatever axis
            # it might be read as. Two dimensions sit on the origin axis here.
            _coverage(
                served=2024,
                held=30,
                recent=[
                    {"year": 2025, "series": 9, "role_counts": {"donor": 9}},
                    {"year": 2024, "series": 30, "role_counts": {"donor": 30}},
                ],
                dimensions=[
                    {"label": "financingAgency", "axis": "origin", "count": 212},
                    {**_DONORS, "count": 30},
                    {"label": "years", "count": 1},
                ],
            ),
            id="a_key_that_is_no_dimension_label",
        ),
        pytest.param(
            {
                **_coverage(served=2024, held=85, recent=[_year(2025, 9), _year(2024, 85)]),
                "window": {"kind": "range", "start_year": 2020, "end_year": 2024},
            },
            id="a_frame_of_several_years",
        ),
        pytest.param(
            {**_coverage(served=2024, held=85, recent=[_year(2025, 9)]), "window": None},
            id="no_window",
        ),
        pytest.param(
            _coverage(
                served=2024,
                held=85,
                recent=[_year(2025, 9), _year(2024, 85), _year(2023, 91)],
                dimensions=[
                    {"label": "observationAbout", "count": 85},
                    {"label": "years", "count": 1},
                ],
            ),
            id="no_dimension_for_a_counted_role",
        ),
        pytest.param(
            _coverage(
                served=2024,
                held=85,
                recent=[
                    {"year": 2025, "series": 9},
                    {"year": 2024, "series": 85},
                    {"year": 2023, "series": 91},
                ],
            ),
            id="series_counts_alone",
        ),
        pytest.param(
            _coverage(
                served=2024,
                held=1,
                recent=[_year(2025, 1), _year(2024, 1), _year(2023, 1)],
            ),
            id="every_role_bound_to_one_member",
        ),
    ],
)
def test_no_note_when_nothing_can_be_set_beside_the_served_year(coverage: dict[str, Any]) -> None:
    assert period_note(coverage) is None


@pytest.mark.parametrize(
    ("dimension", "held", "newer", "stated"),
    [
        (_DONORS, 30, 9, ["30 providers", "9 providers"]),
        (_RECIPIENTS, 30, 9, ["30 recipients", "9 recipients"]),
        (_CREDITORS, 85, 9, ["85 creditors", "9 creditors"]),
        (_CREDITORS, 85, 1, ["85 creditors", "1 creditor so far"]),
        # An entry that states no noun, as a dfo-apis without the field sends
        # it, whatever its axis.
        ({"label": "financingProvider", "axis": "origin"}, 30, 9, ["30 members", "9 members"]),
        ({"label": "sector"}, 30, 1, ["30 members", "1 member so far"]),
        # A noun that is not a singular and a plural, each a word.
        ({"label": "sector", "noun": "providers"}, 30, 9, ["30 members", "9 members"]),
        ({"label": "sector", "noun": {"singular": "provider"}}, 30, 9, ["30 members"]),
        (
            {"label": "sector", "noun": {"singular": "", "plural": "providers"}},
            30,
            9,
            ["30 members", "9 members"],
        ),
        (
            {"label": "sector", "noun": {"singular": "provider", "plural": 3}},
            30,
            9,
            ["30 members", "9 members"],
        ),
    ],
)
def test_the_note_names_its_members_by_the_noun_its_dimension_states(
    dimension: dict[str, Any], held: int, newer: int, stated: list[str]
) -> None:
    label = dimension["label"]
    note = period_note(
        _coverage(
            served=2024,
            held=held,
            recent=[
                {"year": 2025, "series": newer, "role_counts": {label: newer}},
                {"year": 2024, "series": held, "role_counts": {label: held}},
            ],
            # A second dimension on the origin axis, which no key labels.
            dimensions=[
                {"label": "financingAgency", "axis": "origin", "count": 212},
                {**dimension, "count": held},
                {"label": "years", "count": 1},
            ],
        )
    )

    assert _named(note) == [(2024, held), (2025, newer)]
    assert all(one in note for one in stated)
    assert ("donor" in note, "member" in note) == (False, "member" in stated[0])

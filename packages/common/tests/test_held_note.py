"""Tests for `common.held_note`."""

from __future__ import annotations

import pytest
from common.held_back import HeldBack, served_window
from common.held_note import (
    held_back_fact,
    held_back_note,
    held_fact_line,
    held_lines,
    held_period,
    held_period_notes,
    held_runs,
    holds_projections,
    include_hint,
    reaches,
    served_note,
)


def _partial(share: dict[str, object] | None = None) -> HeldBack:
    return HeldBack.model_validate(
        {
            "kind": "partial",
            "source": "dfo/crs",
            "source_name": "OECD CRS",
            "start_year": 2025,
            "end_year": 2025,
            "reported_share": share,
        }
    )


def _projected(source_name: str | None = "World Bank IDS") -> HeldBack:
    return HeldBack(
        kind="projected",
        source="dfo/ids",
        source_name=source_name,
        start_year=2026,
        end_year=2030,
    )


@pytest.mark.parametrize(
    ("run", "sentence"),
    [
        (
            _partial(),
            "2025 is partly reported and is left out of these figures, which end in 2024.",
        ),
        (
            _partial({"basis": "value", "share": 0.001, "against": 2024}),
            "2025 is partly reported (0.1% of 2024's value so far) and is left out of these"
            " figures, which end in 2024.",
        ),
        (
            _partial({"basis": "series", "share": 0.123, "against": 2024}),
            "2025 is partly reported (12% of 2024's series so far) and is left out of these"
            " figures, which end in 2024.",
        ),
        (
            _projected(),
            "World Bank IDS also publishes projections for 2026 to 2030, which these figures"
            " leave out.",
        ),
        (
            _projected(None),
            "The source also publishes projections for 2026 to 2030, which these figures"
            " leave out.",
        ),
    ],
)
def test_held_back_note_words_each_kind_of_run(run: HeldBack, sentence: str) -> None:
    assert held_back_note(run) == sentence


def _served(run: HeldBack, **update: object) -> HeldBack:
    return run.model_copy(update={"served": True, **update})


@pytest.mark.parametrize(
    ("run", "sentence"),
    [
        (
            _served(_partial({"basis": "value", "share": 0.001, "against": 2024})),
            "2025 is partly reported: 0.1% of 2024's value so far. The newest complete year"
            " is 2024.",
        ),
        (
            _served(_partial({"basis": "series", "share": 0.0004, "against": 2024})),
            "2025 is partly reported: less than 0.1% of 2024's series so far. The newest"
            " complete year is 2024.",
        ),
        (
            _served(_partial()),
            "2025 is partly reported. The newest complete year is 2024.",
        ),
        (
            _served(_partial(), start_year=2024),
            "2024 to 2025 are partly reported. The newest complete year is 2023.",
        ),
        (
            _served(_projected("World Bank")),
            "Figures for 2026 to 2030 are World Bank projections. The latest reported year"
            " is 2025.",
        ),
        (
            _served(_projected(None), end_year=2026),
            "Figures for 2026 are the source's projections. The latest reported year is 2025.",
        ),
    ],
)
def test_served_note_words_partial_and_projected_runs(run: HeldBack, sentence: str) -> None:
    assert served_note(run) == sentence
    assert held_back_fact(run).note == sentence


def test_held_lines_say_each_run_a_manifest_states_once() -> None:
    """Lines come from `coverage.held_back` alone. The `incomplete_after`
    entries dfo-apis states inside a served run add nothing."""
    manifest = {
        "coverage": {
            "window": {"start_year": 2020, "end_year": 2030},
            "held_back": [_partial().model_dump(), _served(_projected("World Bank")).model_dump()],
        },
        "incomplete_after": [
            {"source": "dfo/ids", "status": "projected", "partial_period": "2026"},
            {"source": "dfo/ids", "status": "projected", "partial_period": "2028"},
        ],
    }

    assert held_lines([manifest]) == [
        "held: 2026-2030 projected (World Bank), served",
        "held: 2025 partial, left out",
    ]
    assert holds_projections(manifest)
    assert not holds_projections({"coverage": {"held_back": [_partial().model_dump()]}})


@pytest.mark.parametrize(
    ("run", "end", "line"),
    [
        pytest.param(
            _served(_projected()),
            2027,
            "held: 2026-2027 projected (World Bank IDS), served",
            id="served-cut-to-the-read",
        ),
        pytest.param(
            _served(_projected()),
            2030,
            "held: 2026-2030 projected (World Bank IDS), served",
            id="served-whole",
        ),
        pytest.param(
            _served(_partial(), start_year=2024),
            2024,
            "held: 2024 partial, served",
            id="served-partial-cut-to-the-read",
        ),
        pytest.param(
            _projected(),
            2027,
            "held: 2026-2030 projected (World Bank IDS), left out",
            id="left-out-whole",
        ),
    ],
)
def test_a_served_run_ends_where_the_read_ends(run: HeldBack, end: int, line: str) -> None:
    """dfo-apis states a run whole whatever the window. A read of 2020 to
    2027 serving IDS's 2026 to 2030 projections holds 2026 to 2027 alone,
    while a run left out keeps its span."""
    manifest = {
        "coverage": {
            "window": {"start_year": 2020, "end_year": end},
            "held_back": [run.model_dump()],
        }
    }

    assert held_lines([manifest]) == [line]
    assert [fact.line for fact in held_runs([manifest])] == [line]


@pytest.mark.parametrize(
    ("period", "drawn", "reached"),
    [
        pytest.param(
            {"start": "2026", "end": "2030"}, {"start": "2020", "end": "2027"}, True, id="overlap"
        ),
        pytest.param(
            {"start": "2026", "end": "2030"}, {"start": "2015", "end": "2024"}, False, id="before"
        ),
        pytest.param(
            {"start": "2026", "end": "2030"},
            {"start": None, "end": "2024"},
            False,
            id="no-start-before",
        ),
        pytest.param(
            {"start": "2026", "end": "2030"},
            {"start": None, "end": "2027"},
            True,
            id="no-start-overlap",
        ),
        pytest.param(
            {"start": "2026", "end": "2030"}, {"start": "2031", "end": "2035"}, False, id="after"
        ),
    ],
)
def test_a_drawn_run_is_reached_where_it_overlaps_the_drawn_window(
    period: dict[str, str], drawn: dict[str, str | None], reached: bool
) -> None:
    """A window stating its end alone reaches a drawn run only where the run
    starts by that end."""
    assert reaches({**period, "served": True}, drawn=drawn) is reached


def test_include_hint_names_the_request_field_for_each_kind_held() -> None:
    assert include_hint([_projected(), _partial(), _projected()]) == (
        "When the user names these years or asks about projections, get_data serves"
        " them if its scope.window is a range naming those years and adds"
        ' "include": ["partial", "projected"].'
    )
    assert include_hint([]) is None
    assert include_hint([_projected().model_copy(update={"served": True})]) is None


def test_two_projected_sources_each_state_their_own_run() -> None:
    """A frame drawing IMF 2025 to 2029 and IDS 2026 to 2030 states each run
    under its own source and years, whatever `incomplete_after` marks."""
    imf = HeldBack(
        kind="projected",
        served=True,
        source="dfo/imf",
        source_name="IMF WEO",
        start_year=2025,
        end_year=2029,
    )
    ids = _served(_projected())
    manifest = {
        "coverage": {
            "window": {"start_year": 2015, "end_year": 2030},
            "held_back": [imf.model_dump(), ids.model_dump()],
        },
        "citation": {"text": "", "sources": [{"ref": "dfo/ids", "label": "World Bank IDS"}]},
        "incomplete_after": [
            {"source": "dfo/imf", "status": "projected", "partial_period": "2025"},
            {"source": "dfo/ids", "status": "projected", "partial_period": "2026"},
        ],
    }

    assert held_lines([manifest]) == [
        "held: 2025-2029 projected (IMF WEO), served",
        "held: 2026-2030 projected (World Bank IDS), served",
    ]


def test_a_served_partial_run_of_two_years_is_stated_whole_and_reached_from_its_last_year() -> None:
    """A served partial run of 2024 to 2025 is one fact of both years, so a card
    drawing 2025 alone reaches it."""
    manifest = {
        "coverage": {
            "window": {"start_year": 2015, "end_year": 2025},
            "held_back": [_served(_partial(), start_year=2024).model_dump()],
        },
        "incomplete_after": [{"source": "dfo/crs", "status": "partial", "partial_period": "2024"}],
    }

    [run] = held_runs([manifest])

    assert (run.start, run.end, run.served) == ("2024", "2025", True)
    assert reaches(held_period(run), drawn={"start": "2025", "end": "2025"})


def test_a_run_one_read_drew_and_another_left_out_is_stated_once_as_drawn() -> None:
    """Aid to Kenya in 2025 against its 2015 to 2024 average: the 2025 read
    serves the partial year and the 2015 to 2024 read reaches it left out."""
    share = {"basis": "value", "share": 0.001, "against": 2024}
    drew = {
        "coverage": {
            "window": {"start_year": 2025, "end_year": 2025},
            "held_back": [_served(_partial(share)).model_dump()],
        },
        "incomplete_after": [{"source": "dfo/crs", "status": "partial", "partial_period": "2025"}],
    }
    left_out = {
        "coverage": {
            "window": {"start_year": 2015, "end_year": 2024},
            "held_back": [_partial(share).model_dump()],
        }
    }

    assert held_lines([left_out, drew]) == [
        "held: 2025 partial, 0.1% of 2024 value reported, served"
    ]


def test_held_period_notes_say_each_card_run_once_and_a_drawn_run_silences_it_left_out() -> None:
    """Card runs give one sentence per run (kind and source) across cards: the
    drawn ones first, and a run one card left out is not said where another
    card drew it, whatever name each card gives the source."""
    partial_drawn = {
        "status": "partial",
        "start": "2025",
        "end": "2025",
        "served": True,
        "source": "dfo/crs",
        "source_name": "OECD CRS",
        "share": 0.001,
        "basis": "value",
        "against": "2024",
    }
    projected_left = {
        "status": "projected",
        "start": "2026",
        "end": "2030",
        "served": False,
        "source": "dfo/ids",
        "source_name": "World Bank IDS",
    }
    partial_left = {**partial_drawn, "served": False, "source_name": "Creditor Reporting System"}
    assert held_period_notes([projected_left, partial_drawn, partial_left, projected_left]) == [
        "2025 is partly reported: 0.1% of 2024's value so far. The newest complete year is 2024.",
        "World Bank IDS also publishes projections for 2026 to 2030, which these figures leave out.",
    ]


@pytest.mark.parametrize(
    ("run", "window", "stated"),
    [
        (_partial(), {"start_year": 2005, "end_year": 2010}, False),
        (_projected(), {"start_year": 2000, "end_year": 2010}, False),
        (_partial(), {"start_year": 2005, "end_year": 2024}, True),
        (_projected(), {"start_year": 2000, "end_year": 2025}, True),
        (_projected(), {"start_year": 2000}, True),
    ],
)
def test_a_run_left_out_is_stated_only_where_the_reads_window_reaches_it(
    run: HeldBack, window: dict[str, int], stated: bool
) -> None:
    """dfo-apis states every run its source holds whatever the window. A run
    left out is stated where the window ends at or after the year before it
    starts, or states no end."""
    manifest = {"coverage": {"window": window, "held_back": [run.model_dump()]}}

    assert bool(held_lines([manifest])) is stated


@pytest.mark.parametrize(
    ("run", "line"),
    [
        (
            _partial({"basis": "value", "share": 0.005, "against": 2024}),
            "held: 2025 partial, 0.5% of 2024 value reported, left out",
        ),
        (
            _served(_partial({"basis": "series", "share": 0.123, "against": 2024})),
            "held: 2025 partial, 12% of 2024 series reported, served",
        ),
        (_served(_partial(), start_year=2024), "held: 2024-2025 partial, served"),
        (_projected(), "held: 2026-2030 projected (World Bank IDS), left out"),
        (_served(_projected(None)), "held: 2026-2030 projected, served"),
    ],
)
def test_held_fact_line_states_each_run_as_data(run: HeldBack, line: str) -> None:
    assert held_fact_line(run) == line
    assert held_back_fact(run).line == line


@pytest.mark.parametrize(
    ("window", "runs", "end"),
    [
        ({"start_year": 2024, "end_year": 2025}, [_partial()], 2024),
        ({"start_year": 2024, "end_year": 2025}, [_served(_partial())], 2025),
        (
            {"start_year": 2020, "end_year": 2030},
            [_served(_partial(), start_year=2024), _projected()],
            2025,
        ),
        ({"start_year": 2020, "end_year": 2030}, [_projected(), _partial()], 2024),
        ({"start_year": 2020, "end_year": 2030}, [_partial(), _served(_projected())], 2030),
        ({"start_year": 2025, "end_year": 2025}, [_partial()], 2025),
        ({"start_year": 2000, "end_year": 2010}, [_partial()], 2010),
    ],
)
def test_served_window_ends_at_the_newest_year_no_run_left_out_holds(
    window: dict[str, int], runs: list[HeldBack], end: int
) -> None:
    """A window ending inside runs left out ends the year before the earliest
    of them it runs into. A window wholly inside one keeps its end."""
    assert served_window(window, runs)["end_year"] == end

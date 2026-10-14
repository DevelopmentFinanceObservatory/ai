"""`harness.reply_figures`: what a drawn card holds and the values a reply may
state of it, read from the drawn payload of a write reply."""

from __future__ import annotations

from typing import Any

import pytest
from harness.chart_binding import TopMembers
from harness.reply_figures import card_result, drawn_series, unplaced_names, with_coordinates

from . import drawn_cards
from .resolved_revision import resolved_revision

KENYA = {"country/KEN": {"2000": 52.0, "2010": 58.0, "2024": 63.8}}
# Uganda has no 2010 value and Tanzania starts in 2005 and ends lower.
THREE = {
    **KENYA,
    "country/UGA": {"2000": 47.0, "2010": None, "2024": 68.3},
    "country/TZA": {"2005": 70.0, "2024": 67.0},
}
SIX = {f"country/C{number}": {"2000": 1.0, "2024": float(number)} for number in range(6)}
RANKED = {
    "country/SEN": {"2023": 68.7},
    "country/KEN": {"2023": 63.6},
    "country/NGA": {"2023": 54.5},
}
# Two places share the largest value and two the smallest.
TIED = {
    "country/AAA": {"2024": 5.0},
    "country/BBB": {"2024": 5.0},
    "country/CCC": {"2024": 3.0},
    "country/DDD": {"2024": 1.0},
    "country/EEE": {"2024": 1.0},
}
# Kenya as a card draws it newest first.
NEWEST_FIRST = {"country/KEN": dict(reversed(KENYA["country/KEN"].items()))}
# Twenty-four places at one period, drawn highest first: P00 holds 100 and P23 holds 77.
WIDE = {f"country/P{number:02d}": {"2023": 100.0 - number} for number in range(24)}


def _result(envelope: dict[str, Any], codes: Any, **asked: Any) -> dict[str, Any]:
    reply = drawn_cards.write_reply(envelope, drawn_cards.stored(codes))
    series = drawn_series(reply["payload"]["component"], reply["fact_summary"])
    return card_result(series, **asked)


def _stated(result: dict[str, Any]) -> list[tuple[Any, ...]]:
    """Each figure as (series, value, period), and its mark where it has one."""
    return [
        (
            figure["series"],
            figure["value"],
            figure.get("period"),
            *filter(None, [figure.get("mark")]),
        )
        for figure in result["reply_figures"]
    ]


@pytest.mark.parametrize(
    ("member", "series", "asked", "figures", "reported"),
    [
        pytest.param("line_chart", KENYA, {}, [("KEN", 63.8, "2024")], {}, id="one_series"),
        pytest.param(
            "line_chart", KENYA, {"on_canvas": ["s_ken"]}, [], {}, id="one_series_on_the_canvas"
        ),
        pytest.param(
            "line_chart",
            KENYA,
            {"on_canvas": ["s_ken"], "state_values_for": ["country/KEN"]},
            [("KEN", 63.8, "2024")],
            {},
            id="one_series_on_the_canvas_named",
        ),
        pytest.param("line_chart", THREE, {}, [], {}, id="several_series_none_named"),
        pytest.param(
            "bar_chart",
            RANKED,
            {"on_canvas": ["s_sen", "s_ken", "s_nga", "s_other"]},
            [],
            {},
            id="several_series_all_on_the_canvas",
        ),
        pytest.param(
            "bar_chart",
            RANKED,
            {"on_canvas": ["s_sen", "s_ken", "s_nga"], "state_values_for": ["country/NGA"]},
            [("NGA", 54.5, "2023")],
            {},
            id="several_series_all_on_the_canvas_one_named",
        ),
        pytest.param(
            "bar_chart",
            RANKED,
            # Nigeria is new to the canvas, so the card is a new card.
            {"on_canvas": ["s_sen", "s_ken"]},
            [("SEN", 68.7, "2023", "largest"), ("NGA", 54.5, "2023", "smallest")],
            {},
            id="several_series_one_new_to_the_canvas",
        ),
        pytest.param(
            "line_chart",
            THREE,
            # One by the frame's code, one by its label, in another case.
            {"state_values_for": ["country/KEN", "uga"]},
            [("KEN", 63.8, "2024"), ("UGA", 68.3, "2024")],
            {},
            id="named_series",
        ),
        pytest.param(
            "line_chart",
            SIX,
            {"state_values_for": [f"country/C{number}" for number in range(5)]},
            [(f"C{number}", float(number), "2024") for number in range(5)],
            {},
            id="five_named_series",
        ),
        pytest.param(
            "line_chart",
            THREE,
            {"state_values_for": ["country/KEN", "country/NGA"]},
            [("KEN", 63.8, "2024")],
            {"not_in_card": ["country/NGA"]},
            id="a_name_matching_no_series",
        ),
        pytest.param(
            "bar_chart",
            RANKED,
            {},
            [("SEN", 68.7, "2023", "largest"), ("NGA", 54.5, "2023", "smallest")],
            {},
            id="largest_and_smallest_of_one_period",
        ),
        pytest.param(
            "bar_chart",
            RANKED,
            {"state_values_for": ["country/KEN", "country/NGA"]},
            [
                ("KEN", 63.6, "2023"),
                ("NGA", 54.5, "2023", "smallest"),
                ("SEN", 68.7, "2023", "largest"),
            ],
            {},
            id="a_named_series_that_is_an_extreme_is_given_once",
        ),
        pytest.param(
            "line_chart",
            THREE,
            # Each series is dated by its own first period, and Tanzania starts in 2005.
            {"state_values_for": ["country/KEN", "country/TZA"], "state_span": True},
            [
                ("KEN", 52.0, "2000"),
                ("KEN", 63.8, "2024"),
                ("TZA", 70.0, "2005"),
                ("TZA", 67.0, "2024"),
            ],
            {},
            id="state_span",
        ),
        pytest.param(
            "line_chart",
            THREE,
            {"state_values_for": ["country/KEN"], "state_period": "2010"},
            [("KEN", 58.0, "2010")],
            {},
            id="a_named_past_period",
        ),
        pytest.param(
            "line_chart",
            THREE,
            {"state_values_for": ["country/KEN"], "state_period": "2010", "state_span": True},
            [("KEN", 52.0, "2000"), ("KEN", 58.0, "2010"), ("KEN", 63.8, "2024")],
            {},
            id="a_named_past_period_with_the_span",
        ),
        pytest.param(
            "line_chart",
            THREE,
            {"state_values_for": ["country/UGA"], "state_period": "2010"},
            [],
            {"no_value": [{"series": "UGA", "period": "2010"}]},
            id="a_named_period_the_series_holds_no_value_at",
        ),
        pytest.param(
            "line_chart",
            THREE,
            {"state_values_for": ["country/UGA"], "state_period": "2010", "state_span": True},
            [("UGA", 47.0, "2000"), ("UGA", 68.3, "2024")],
            {"no_value": [{"series": "UGA", "period": "2010"}]},
            id="a_named_period_with_no_value_beside_the_span",
        ),
        pytest.param(
            "line_chart",
            KENYA,
            {"state_period": "2010"},
            [("KEN", 58.0, "2010"), ("KEN", 63.8, "2024")],
            {},
            id="one_series_at_a_past_period_gives_its_last_as_well",
        ),
        pytest.param(
            "line_chart",
            KENYA,
            {"state_values_for": ["country/KEN"], "state_period": "2024"},
            [("KEN", 63.8, "2024")],
            {},
            id="one_series_at_its_last_period_is_given_once",
        ),
        pytest.param(
            "line_chart",
            KENYA,
            {"state_period": "1990"},
            [("KEN", 63.8, "2024")],
            {"no_value": [{"series": "KEN", "period": "1990"}]},
            id="one_series_at_a_period_it_holds_no_value_at",
        ),
        pytest.param(
            "bar_chart",
            TIED,
            {"state_values_for": ["country/BBB"]},
            [
                ("BBB", 5.0, "2024", "largest"),
                ("AAA", 5.0, "2024", "largest"),
                ("DDD", 1.0, "2024", "smallest"),
                ("EEE", 1.0, "2024", "smallest"),
            ],
            {},
            id="every_series_tying_for_an_extreme_is_marked",
        ),
        pytest.param(
            "line_chart",
            NEWEST_FIRST,
            {"state_span": True},
            [("KEN", 52.0, "2000"), ("KEN", 63.8, "2024")],
            {},
            id="first_and_last_are_by_period_whatever_the_drawn_order",
        ),
        pytest.param(
            "line_chart",
            # A month ends before the year holding it does.
            {"country/KEN": {"2024": 61.0, "2024-06": 60.5, "2023": 60.0}},
            {"state_span": True},
            [("KEN", 60.0, "2023"), ("KEN", 61.0, "2024")],
            {},
            id="periods_of_two_grains_order_by_their_ends",
        ),
        pytest.param(
            "line_chart",
            {"country/KEN": {"2020/2024": 2.0, "2015/2019": 1.0}},
            {},
            [("KEN", 2.0, "2020/2024")],
            {},
            id="one_series_of_spans_gives_the_latest_span",
        ),
        pytest.param(
            "line_chart",
            {"country/KEN": {"2015/2024": 3.0, "2020/2024": 2.0, "2015/2019": 1.0}},
            {"state_span": True},
            [("KEN", 1.0, "2015/2019"), ("KEN", 2.0, "2020/2024")],
            {},
            id="spans_order_by_their_end_then_their_start",
        ),
        pytest.param(
            "line_chart",
            # The frame grammar writes no quarter, so these keep their drawn order.
            {"country/KEN": {"2023-Q4": 1.0, "2024-Q1": 2.0}},
            {},
            [("KEN", 2.0, "2024-Q1")],
            {},
            id="one_series_of_periods_with_no_key_gives_its_last_drawn",
        ),
        pytest.param(
            "line_chart",
            {"country/KEN": {"2024": 3.0, "2023": 2.0, "2023-Q4": 1.0}},
            {"state_span": True},
            [("KEN", 3.0, "2024"), ("KEN", 1.0, "2023-Q4")],
            {},
            id="a_period_with_no_key_leaves_the_series_in_drawn_order",
        ),
    ],
)
def test_reply_figures_follow_what_the_chart_asks_to_state(
    member: str,
    series: dict[str, dict[str, float | None]],
    asked: dict[str, Any],
    figures: list[tuple[Any, ...]],
    reported: dict[str, Any],
) -> None:
    result = _result(drawn_cards.chart(member, series), series, **asked)

    assert _stated(result) == figures
    assert {key: result[key] for key in result.keys() - {"holds", "reply_figures"}} == reported
    unit = resolved_revision(member)["component"]["series"][0]["unit"]
    assert all(figure["unit"] == unit for figure in result["reply_figures"])
    if len(series) == 1 and figures and "state_period" not in asked:
        # The card's last period is the period of its one series' last figure.
        assert result["holds"]["last_period"] == figures[-1][2]


def test_a_card_holds_its_cover_and_how_its_series_moved_with_no_value() -> None:
    lines = _result(drawn_cards.chart("line_chart", THREE), THREE)["holds"]
    bars = _result(drawn_cards.chart("bar_chart", RANKED), RANKED)["holds"]

    unit = resolved_revision("line_chart")["component"]["series"][0]["unit"]
    assert lines == {
        "series_count": 3,
        "first_period": "2000",
        "last_period": "2024",
        "unit": unit,
        "series": ["KEN", "UGA", "TZA"],
        "direction": {
            "higher_at_last": 2,
            "lower_at_last": 1,
            "unchanged": 0,
            "among": {"drawn": 3},
        },
    }
    assert bars == {
        "series_count": 3,
        "first_period": "2023",
        "last_period": "2023",
        "unit": unit,
        "series": ["SEN", "KEN", "NGA"],
        "series_order": "highest_first",
    }
    # Series of one value sit in either order within a ranking.
    assert _result(drawn_cards.chart("bar_chart", TIED), TIED)["holds"]["series_order"] == (
        "highest_first"
    )


def test_series_with_no_unit_do_not_rank_against_each_other() -> None:
    envelope = drawn_cards.chart("bar_chart", RANKED)
    for drawn in envelope["component"]["series"]:
        drawn["unit"] = None

    result = _result(envelope, RANKED)

    assert result["reply_figures"] == []
    assert "series_order" not in result["holds"]


def test_a_point_with_a_null_value_is_no_value() -> None:
    """Senegal's only point holds a null value, so it has none to give and
    the other two rank without it."""
    envelope = drawn_cards.chart("bar_chart", RANKED)
    envelope["component"]["series"][0]["points"] = [{"period": "2023", "value": None}]

    result = _result(envelope, RANKED, state_values_for=["country/SEN"])

    assert _stated(result) == [("KEN", 63.6, "2023", "largest"), ("NGA", 54.5, "2023", "smallest")]
    assert result["reply_figures"][0]["among"] == {"drawn": 2}
    assert result["no_value"] == [{"series": "SEN"}]


def test_figures_of_a_card_of_24_series_cover_every_series_it_draws() -> None:
    """The platform's summary lists twelve of the 24 series. A series past
    the twelfth still gives its figure, the smallest is the smallest of 24,
    and a cut card says what its extremes were chosen among."""
    reply = drawn_cards.write_reply(drawn_cards.chart("bar_chart", WIDE), drawn_cards.stored(WIDE))
    assert len(reply["fact_summary"]["series"]) == drawn_cards.FACT_SUMMARY_SERIES_CAP
    series = drawn_series(reply["payload"]["component"], reply["fact_summary"])
    cut = TopMembers(
        members=list(WIDE),
        period="2023",
        holding=30,
        unvalued=["P40", "P41"],
        other_kinds=5,
    )

    result = card_result(series, state_values_for=["P19"], cut=cut)

    assert result["holds"]["series_count"] == 24
    assert result["holds"]["series"] == [f"P{number:02d}" for number in range(24)]
    assert result["holds"]["cut"] == {
        "drawn": 24,
        "with_a_value": 30,
        "period": "2023",
        "other_kinds": 5,
        "unvalued": 2,
    }
    assert _stated(result) == [
        ("P19", 81.0, "2023"),
        ("P00", 100.0, "2023", "largest"),
        ("P23", 77.0, "2023", "smallest"),
    ]
    assert result["reply_figures"][-1]["among"] == {"drawn": 24, "with_a_value": 30}
    assert "not_in_card" not in result


def test_a_direction_count_covers_all_24_series_of_a_card() -> None:
    """Sixteen of 24 table series end higher than they began and eight lower,
    whichever period a series' rows begin with."""
    series = {
        f"country/P{number:02d}": dict(
            sorted(
                {"2020": 50.0, "2023": 50.0 + (1 if number % 3 else -1)}.items(),
                reverse=bool(number % 2),
            )
        )
        for number in range(24)
    }

    holds = _result(drawn_cards.table(series), series)["holds"]

    assert holds["series_count"] == 24
    assert holds["direction"] == {
        "higher_at_last": 16,
        "lower_at_last": 8,
        "unchanged": 0,
        "among": {"drawn": 24},
    }


def _frame_rows(series: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    return [
        {**drawn_cards.coordinates(code), "time_period": period, "value": value}
        for code, points in series.items()
        for period, value in points.items()
    ]


def test_a_series_past_the_summarys_twelfth_is_named_by_its_code_through_the_frames_rows() -> None:
    """The summary states the coordinates of twelve series, so a code naming
    one of the other twelve is left unmatched, not called absent, until
    the frame's rows place the series by the points it draws."""
    reply = drawn_cards.write_reply(drawn_cards.chart("bar_chart", WIDE), drawn_cards.stored(WIDE))
    series = drawn_series(reply["payload"]["component"], reply["fact_summary"])
    asked = ["country/P19", "country/ZZZ"]

    unplaced = card_result(series, state_values_for=asked)
    placed = card_result(with_coordinates(series, _frame_rows(WIDE)), state_values_for=asked)

    assert unplaced_names(series, asked) == asked
    assert unplaced["not_matched"] == asked and "not_in_card" not in unplaced
    assert _stated(placed)[0] == ("P19", 81.0, "2023")
    assert placed["not_in_card"] == ["country/ZZZ"] and "not_matched" not in placed


def test_a_drawn_series_two_frame_series_could_be_keeps_no_coordinates() -> None:
    """P12 and P13 hold one value at the card's period, so neither's points
    say which frame series it is, and a code naming one stays unmatched."""
    tied = WIDE | {"country/P12": {"2023": 5.0}, "country/P13": {"2023": 5.0}}
    reply = drawn_cards.write_reply(drawn_cards.chart("bar_chart", tied), drawn_cards.stored(tied))
    series = drawn_series(reply["payload"]["component"], reply["fact_summary"])

    placed = with_coordinates(series, _frame_rows(tied))
    result = card_result(placed, state_values_for=["country/P13", "country/P20"])

    assert [one.label for one in placed if one.coordinates is None] == ["P12", "P13"]
    assert _stated(result)[0] == ("P20", 80.0, "2023")
    assert result["not_matched"] == ["country/P13"]


@pytest.mark.parametrize(
    ("member", "figures"),
    [
        ("stat_card", [("Malaria grant disbursements to Ethiopia", 441.9, "2024")]),
        ("table", [("Ethiopia", 441.9, "2024"), ("Ghana", 225.3, "2024")]),
    ],
)
def test_a_stat_card_and_a_table_give_figures_with_their_periods(
    member: str, figures: list[tuple[Any, ...]]
) -> None:
    envelope = resolved_revision(member)
    codes = ["country/ETH"] if member == "stat_card" else []

    named = [] if member == "stat_card" else ["Ethiopia", "Ghana"]

    result = _result(envelope, codes, state_values_for=named)

    assert _stated(result) == figures
    assert result["holds"]["series_count"] == len(figures)
    assert (result["holds"]["first_period"], result["holds"]["last_period"]) == (
        ("2024", "2024") if member == "stat_card" else ("2023", "2024")
    )


def test_a_stat_card_of_a_series_on_the_canvas_gives_no_figure() -> None:
    """A stat card's payload names no series, so its id is the summary's."""
    result = _result(resolved_revision("stat_card"), ["country/ETH"], on_canvas=["s_eth"])

    assert result["reply_figures"] == []

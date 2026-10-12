"""`run_join`: the inner-join policy (dropped-key counts on both sides,
never fabricated nulls), the long-format output shape, determinism (order
perturbation), and every refusal condition.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from analysis_mcp.engine.common import UnitHeterogeneousError
from analysis_mcp.engine.join import (
    JoinAmbiguousKeyError,
    JoinNoCommonKeyColumnsError,
    JoinResultTooLargeError,
    JoinUnknownKeyError,
    run_join,
)
from analysis_mcp.fixtures import join_compatible_frame_pair, multi_unit_frame
from dfo_frame_grammar import Frame

_FIXTURES = Path(__file__).parent / "fixtures"


def _frame(rows: list[dict[str, object]], *, role_keys: list[str] | None = None) -> Frame:
    return Frame.build(role_keys=role_keys or ["reporter"], rows=rows)


def _row(
    *,
    reporter: str = "dcid:USA",
    year: str = "2020",
    value: float = 1.0,
    stat_var: str = "dcid:X",
    **extra: object,
) -> dict:
    return {
        "stat_var": stat_var,
        "reporter": reporter,
        "time_period": year,
        "value": value,
        **extra,
    }


# Golden file -----------------------------------------------------------------


def test_golden_worked_join_matches_the_committed_fixture() -> None:
    golden = json.loads((_FIXTURES / "join_golden.json").read_text())
    oda, gni = join_compatible_frame_pair()

    result = run_join(oda, gni, join_keys=tuple(golden["join_keys"]))

    assert list(result.join_keys) == golden["expected_resolved_join_keys"]
    assert result.left_dropped_key_count == golden["expected_left_dropped_key_count"]
    assert result.right_dropped_key_count == golden["expected_right_dropped_key_count"]
    actual = [
        {
            "key": [list(pair) for pair in mk.key],
            "time_period": mk.time_period,
            "left_value": mk.left_value,
            "right_value": mk.right_value,
        }
        for mk in result.matched_keys.rows()
    ]
    assert actual == golden["expected_matched_keys"]


# Declared policy: inner join, dropped-key counts on both sides -----------------


def test_policy_unmatched_keys_are_dropped_and_counted_on_both_sides() -> None:
    """The declared policy: inner join; report dropped-key counts on both
    sides -- never a fabricated null for an unmatched key. Pinned so a
    switch to a left/right/full outer join fails this test."""
    left = _frame(
        [
            _row(reporter="dcid:USA", year="2020", value=1.0, stat_var="dcid:X"),
            _row(reporter="dcid:GBR", year="2020", value=2.0, stat_var="dcid:X"),
        ]
    )
    right = _frame(
        [
            _row(reporter="dcid:USA", year="2020", value=10.0, stat_var="dcid:Y"),
            _row(reporter="dcid:FRA", year="2020", value=30.0, stat_var="dcid:Y"),
        ]
    )

    result = run_join(left, right)

    assert result.left_dropped_key_count == 1  # GBR has no match on the right
    assert result.right_dropped_key_count == 1  # FRA has no match on the left
    assert len(result.matched_keys) == 1
    (matched,) = result.matched_keys.rows()
    assert dict(matched.key) == {"reporter": "dcid:USA"}
    assert matched.left_value == pytest.approx(1.0)
    assert matched.right_value == pytest.approx(10.0)
    # Every emitted row carries a real, finite value.
    assert all(row[3] is not None for row in result.frame.rows)


def test_unmatched_keys_on_each_side_are_reported_as_unmatched_key_exclusions() -> None:
    """The dropped-key counts above are the summary; `result.exclusions`
    carries the structured fact -- one `unmatched_key` record per side,
    tagged `left`/`right`, naming which entity had no partner, and its own
    `rows` count equals that side's own `left_dropped_key_count`/
    `right_dropped_key_count`."""
    left = _frame(
        [
            _row(reporter="dcid:USA", year="2020", value=1.0, stat_var="dcid:X"),
            _row(reporter="dcid:GBR", year="2020", value=2.0, stat_var="dcid:X"),
        ]
    )
    right = _frame(
        [
            _row(reporter="dcid:USA", year="2020", value=10.0, stat_var="dcid:Y"),
            _row(reporter="dcid:FRA", year="2020", value=30.0, stat_var="dcid:Y"),
        ]
    )

    result = run_join(left, right)

    by_side = {record.side: record for record in result.exclusions.records}
    assert {record.reason for record in result.exclusions.records} == {"unmatched_key"}
    assert by_side["left"].entities["reporter"] == ("dcid:GBR",)
    assert by_side["right"].entities["reporter"] == ("dcid:FRA",)
    assert by_side["left"].row_count == result.left_dropped_key_count == 1
    assert by_side["right"].row_count == result.right_dropped_key_count == 1


def test_declared_policy_is_never_reported_silently() -> None:
    """A key mismatch is surfaced as a digest-facing warning, not silently
    absorbed into the matched-key count alone."""
    left = _frame([_row(reporter="dcid:USA", year="2020", stat_var="dcid:X")])
    right = _frame([_row(reporter="dcid:GBR", year="2020", stat_var="dcid:Y")])

    result = run_join(left, right)

    assert any("dropped" in warning for warning in result.warnings)


# Long-format output: stat_var distinguishes sides -------------------------------


def test_output_frame_is_long_format_with_stat_var_distinguishing_sides() -> None:
    oda, gni = join_compatible_frame_pair()
    result = run_join(oda, gni)

    stat_var_idx = result.frame.columns.index("stat_var")
    distinct_stat_vars = {row[stat_var_idx] for row in result.frame.rows}
    assert distinct_stat_vars == {
        "dcid:OfficialDevelopmentAssistanceNet_Disbursement",
        "dcid:GrossNationalIncome",
    }
    # Two rows (one per side) per matched key.
    assert len(result.frame.rows) == 2 * len(result.matched_keys)


# Explicit join_keys --------------------------------------------------------------


def test_explicit_join_keys_are_used_instead_of_auto_detection() -> None:
    oda, gni = join_compatible_frame_pair()
    result = run_join(oda, gni, join_keys=("reporter",))
    assert result.join_keys == ("reporter",)


def test_unknown_explicit_join_key_refuses() -> None:
    oda, gni = join_compatible_frame_pair()
    with pytest.raises(JoinUnknownKeyError) as exc_info:
        run_join(oda, gni, join_keys=("donor",))
    assert exc_info.value.columns == ("donor",)


def test_time_period_as_an_explicit_join_key_refuses() -> None:
    """`time_period` is already matched implicitly by every join -- naming
    it explicitly in `join_keys` is refused, not redundantly accepted."""
    oda, gni = join_compatible_frame_pair()
    with pytest.raises(JoinUnknownKeyError):
        run_join(oda, gni, join_keys=("time_period",))


def test_stat_var_as_an_explicit_join_key_refuses() -> None:
    """Naming `stat_var` as a join key would only ever match rows sharing
    one StatVar, defeating the point of joining two different StatVars."""
    oda, gni = join_compatible_frame_pair()
    with pytest.raises(JoinUnknownKeyError):
        run_join(oda, gni, join_keys=("stat_var",))


def test_auto_detect_with_disjoint_role_columns_refuses() -> None:
    """Auto-detection (`join_keys` left empty) intersects the two frames'
    own role columns *by name*: a left frame declaring `donor` and a right
    frame declaring `recipient` share no column name at all, so the
    auto-detected key list would be empty -- even though both sides do
    declare a role column. Falling through to matching on `time_period`
    alone would silently pair every donor's own row against every
    recipient's row that merely shares a year (a USA disbursement matched
    to an unrelated Kenya receipt that shares nothing but "2020"). Must
    refuse rather than proceed on time_period alone."""
    left = Frame.build(
        role_keys=["donor"],
        rows=[
            {
                "stat_var": "dcid:Disbursed",
                "donor": "dcid:USA",
                "time_period": "2020",
                "value": 1.0,
                "unit": None,
            }
        ],
    )
    right = Frame.build(
        role_keys=["recipient"],
        rows=[
            {
                "stat_var": "dcid:Received",
                "recipient": "dcid:KEN",
                "time_period": "2020",
                "value": 2.0,
                "unit": None,
            }
        ],
    )
    with pytest.raises(JoinNoCommonKeyColumnsError) as exc_info:
        run_join(left, right)
    assert exc_info.value.left_columns == ("donor",)
    assert exc_info.value.right_columns == ("recipient",)


def test_auto_detect_with_neither_side_declaring_a_role_column_still_proceeds() -> None:
    """When *neither* side declares any role column at all, an empty
    auto-detected key list is legitimate -- there is no role dimension
    either side ever had for time_period-only matching to silently narrow,
    distinct from the previous test. Must not raise
    `JoinNoCommonKeyColumnsError`."""
    left = Frame.build(
        role_keys=[],
        rows=[{"stat_var": "dcid:X", "time_period": "2020", "value": 1.0, "unit": None}],
    )
    right = Frame.build(
        role_keys=[],
        rows=[{"stat_var": "dcid:Y", "time_period": "2020", "value": 2.0, "unit": None}],
    )
    result = run_join(left, right)
    assert result.join_keys == ()
    assert len(result.matched_keys) == 1


def test_auto_detect_with_only_one_side_declaring_a_role_column_still_proceeds() -> None:
    """A third case, distinct from both above: exactly *one* side declares a
    role column, the other none at all -- a world total (one row per
    period, no role column) joined against a frame already narrowed to a
    single reporter (still declares `reporter`, but with only one distinct
    value, so still one row per period). This is unambiguous: there is only
    one row per period on the role-less side to pair against the single row
    per period the other side already has, the identical relationship a
    role-less frame already has with time_period-only matching when it's
    the only side in play. Reachable through the toolkit's own verbs --
    `run_aggregate(frame, group_by=(), aggregation="sum").frame` produces
    exactly the role-less shape. Must not raise `JoinNoCommonKeyColumnsError`
    (unlike `test_auto_detect_with_disjoint_role_columns_refuses` above,
    where *both* sides declare a role column and share no name)."""
    world_total = Frame.build(
        role_keys=[],
        rows=[
            {"stat_var": "dcid:WorldGNI", "time_period": "2021", "value": 100.0, "unit": None},
            {"stat_var": "dcid:WorldGNI", "time_period": "2022", "value": 110.0, "unit": None},
        ],
    )
    single_reporter = Frame.build(
        role_keys=["reporter"],
        rows=[
            {
                "stat_var": "dcid:ODA",
                "reporter": "dcid:USA",
                "time_period": "2021",
                "value": 10.0,
                "unit": None,
            },
            {
                "stat_var": "dcid:ODA",
                "reporter": "dcid:USA",
                "time_period": "2022",
                "value": 12.0,
                "unit": None,
            },
        ],
    )
    result = run_join(world_total, single_reporter)
    assert result.join_keys == ()
    assert len(result.matched_keys) == 2


# Determinism -------------------------------------------------------------------


def _row_mapping_with_reversed_key_order(
    frame: Frame, row: tuple[object, ...]
) -> dict[str, object]:
    """`row`'s own cells, keyed by `frame.columns` but inserted in the
    opposite order (`status` dropped -- reserved, never caller-supplied).
    `Frame.build` sorts the *list* of rows it's given, which absorbs any
    perturbation to row order; it says nothing about a perturbation to one
    row *mapping's* own key insertion order, which this builds instead."""
    pairs = [(key, value) for key, value in zip(frame.columns, row, strict=True) if key != "status"]
    return dict(reversed(pairs))


def _matched_key(matched) -> tuple[object, ...]:
    """`MatchedKey` equality with `key` order-insensitive -- `key`'s own
    tuple order mirrors whatever order `join_keys` was called with
    (`run_join`'s `role_values = dict(zip(key_columns, ...))`), so two
    calls differing only in `join_keys`'s column order produce genuinely
    different `key` tuples describing the identical matched pair."""
    return (frozenset(matched.key), matched.time_period, matched.left_value, matched.right_value)


def test_row_key_order_and_join_keys_order_perturbation_produce_an_equivalent_result() -> None:
    """The two perturbations that actually reach this engine (row order
    itself is absorbed by `Frame.build`'s own sort, so shuffling it proves
    nothing): each row mapping's own key insertion order, and `join_keys`'s
    own column order. `join_keys` needs two role columns --
    `join_compatible_frame_pair`'s single shared role (`reporter`) can't
    exercise a real order permutation -- so a fresh two-role frame pair is
    built here instead. `run_join` never canonicalises an explicit
    `join_keys` itself (`_resolve_join_keys` returns `tuple(join_keys)`
    verbatim when given one; the tool boundary canonicalises before this
    function ever runs), so `MatchedKey.key`'s own tuple order does move
    with the perturbation; `_matched_key` is what makes the comparison
    honest about that instead of hiding it."""
    left_rows = [
        _row(reporter="dcid:USA", recipient="dcid:KEN", year="2020", value=10.0),
        _row(reporter="dcid:GBR", recipient="dcid:TZA", year="2020", value=5.0),
    ]
    right_rows = [
        _row(
            reporter="dcid:USA", recipient="dcid:KEN", year="2020", value=100.0, stat_var="dcid:Y"
        ),
        _row(reporter="dcid:GBR", recipient="dcid:TZA", year="2020", value=50.0, stat_var="dcid:Y"),
    ]
    left = _frame(left_rows, role_keys=["reporter", "recipient"])
    right = _frame(right_rows, role_keys=["reporter", "recipient"])
    original = run_join(left, right, join_keys=("reporter", "recipient"))

    reordered_left_rows = [_row_mapping_with_reversed_key_order(left, row) for row in left.rows]
    reordered_left = Frame.build(role_keys=["reporter", "recipient"], rows=reordered_left_rows)
    perturbed = run_join(reordered_left, right, join_keys=("recipient", "reporter"))

    assert perturbed.frame == original.frame
    assert {_matched_key(m) for m in perturbed.matched_keys.rows()} == {
        _matched_key(m) for m in original.matched_keys.rows()
    }


# Refusals ------------------------------------------------------------------------


def test_output_over_max_output_rows_refuses_before_building_the_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each matched key emits one output row per side, so the ceiling counts
    twice the matched keys. The refusal fires once that count is known:
    building the output frame fails here, and the refusal still arrives."""
    oda, gni = join_compatible_frame_pair()
    output_rows = 2 * len(run_join(oda, gni).matched_keys)

    def build_output(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("the output frame was built")

    monkeypatch.setattr("analysis_mcp.engine.join.frame_keeping_role_types", build_output)
    with pytest.raises(JoinResultTooLargeError) as refused:
        run_join(oda, gni, max_output_rows=output_rows - 1)
    assert (refused.value.row_count, refused.value.ceiling) == (output_rows, output_rows - 1)


def test_output_at_max_output_rows_is_admitted() -> None:
    oda, gni = join_compatible_frame_pair()
    output_rows = 2 * len(run_join(oda, gni).matched_keys)
    assert run_join(oda, gni, max_output_rows=output_rows).frame.num_rows == output_rows


def test_ambiguous_key_on_one_side_refuses() -> None:
    left = _frame(
        [
            _row(reporter="dcid:USA", year="2020", value=1.0, stat_var="dcid:X"),
            _row(
                reporter="dcid:USA",
                year="2020",
                value=2.0,
                stat_var="dcid:X",
                measurementMethod="dcid:M2",
            ),
        ]
    )
    right = _frame([_row(reporter="dcid:USA", year="2020", stat_var="dcid:Y")])

    with pytest.raises(JoinAmbiguousKeyError) as exc_info:
        run_join(left, right)
    assert exc_info.value.side == "left"


def test_heterogeneous_unit_frame_refuses() -> None:
    frame = multi_unit_frame()
    with pytest.raises(UnitHeterogeneousError):
        run_join(frame, frame)

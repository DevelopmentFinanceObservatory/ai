"""`run_join` matches keys on Arrow tables: matched keys come out in the
code-point order of their string parts, each matched key emits one output
row per side with its value copied verbatim, and a converted unit is
carried onto its side's rows.
"""

from __future__ import annotations

import pytest
from analysis_mcp.engine.join import run_join
from dfo_frame_grammar import Frame
from dfo_frame_grammar.grammar import UnitRequest

# Case, accents, astral characters and prefixes, where code-point order
# differs from any locale's collation.
_REPORTERS = ["dcid:é", "dcid:a", "dcid:ab", "dcid:A", "dcid:\U0001d538", "dcid:Z"]


def _frame(
    stat_var: str,
    cells: list[tuple[str, str, float]],
    *,
    unit: object = "dcid:USDollar",
) -> Frame:
    return Frame.build(
        role_keys=["reporter"],
        rows=[
            {
                "stat_var": stat_var,
                "reporter": reporter,
                "time_period": period,
                "value": value,
                "unit": unit,
            }
            for reporter, period, value in cells
        ],
    )


def test_matched_keys_follow_the_code_point_order_of_their_parts() -> None:
    left_cells = [
        (reporter, period, float(index))
        for index, (reporter, period) in enumerate(
            (reporter, period) for reporter in _REPORTERS for period in ("2021", "2020")
        )
    ]
    right_cells = [(reporter, period, value + 0.1) for reporter, period, value in left_cells]

    result = run_join(_frame("dcid:L", left_cells), _frame("dcid:R", right_cells))

    keys = [(dict(key.key)["reporter"], key.time_period) for key in result.matched_keys.rows()]
    assert keys == sorted((reporter, period) for reporter, period, _ in left_cells)


def test_each_matched_key_emits_one_row_per_side_with_values_verbatim() -> None:
    left = _frame("dcid:L", [("dcid:A", "2020", 0.1 + 0.2), ("dcid:B", "2020", 5e-324)])
    right = _frame("dcid:R", [("dcid:A", "2020", -1.5), ("dcid:C", "2020", 7.0)])

    result = run_join(left, right)

    expected = Frame.build(
        role_keys=["reporter"],
        rows=[
            {
                "stat_var": "dcid:L",
                "reporter": "dcid:A",
                "time_period": "2020",
                "value": 0.1 + 0.2,
                "unit": "dcid:USDollar",
            },
            {
                "stat_var": "dcid:R",
                "reporter": "dcid:A",
                "time_period": "2020",
                "value": -1.5,
                "unit": "dcid:USDollar",
            },
        ],
    )
    assert result.frame == expected
    assert [(key.left_value, key.right_value) for key in result.matched_keys.rows()] == [
        (0.1 + 0.2, -1.5)
    ]
    assert (result.left_dropped_key_count, result.right_dropped_key_count) == (1, 1)


_CONVERTED = {"currency": "USD", "prices": "current"}


def test_a_converted_unit_is_carried_onto_its_sides_rows() -> None:
    left = _frame("dcid:L", [("dcid:A", "2020", 1.0)], unit=_CONVERTED)
    right = _frame("dcid:R", [("dcid:A", "2020", 2.0)])

    result = run_join(left, right)

    converted = UnitRequest(currency="USD", prices="current", base_year=None)
    assert result.left_resolved_unit == converted
    assert {(row[0], row[4]) for row in result.frame.rows} == {
        ("dcid:L", converted),
        ("dcid:R", "dcid:USDollar"),
    }


def _role_frame(stat_var: str, roles: tuple[str, ...], cells: list[tuple[str, float]]) -> Frame:
    return Frame.build(
        role_keys=list(roles),
        rows=[
            {
                "stat_var": stat_var,
                **dict.fromkeys(roles, entity),
                "time_period": "2020",
                "value": value,
                "unit": "dcid:USDollar",
            }
            for entity, value in cells
        ],
    )


# The column names the join and the frame grammar's Arrow helpers give their
# own working tables, and double-underscore row-position names.
@pytest.mark.parametrize(
    "roles",
    [
        ("__left_row",),
        ("__right_row",),
        ("left_row",),
        ("right_row",),
        ("key_0",),
        ("key_1", "key_0"),
        ("__collate_0_rank",),
        ("__collate_0_payload",),
        ("code",),
        ("row",),
    ],
)
def test_a_role_named_like_a_working_column_joins_like_any_role(roles: tuple[str, ...]) -> None:
    left_cells = [("dcid:B", 2.0), ("dcid:A", 1.0), ("dcid:C", 3.0)]
    right_cells = [("dcid:C", 30.0), ("dcid:A", 10.0), ("dcid:D", 40.0)]
    reference_roles = tuple(f"reporter{position}" for position in range(len(roles)))

    result = run_join(
        _role_frame("dcid:L", roles, left_cells), _role_frame("dcid:R", roles, right_cells)
    )
    reference = run_join(
        _role_frame("dcid:L", reference_roles, left_cells),
        _role_frame("dcid:R", reference_roles, right_cells),
    )

    renamed = dict(zip(roles, reference_roles, strict=True))
    assert result.frame.rows == reference.frame.rows
    assert [
        ({renamed[column]: cell for column, cell in key.key}, key.time_period)
        for key in result.matched_keys.rows()
    ] == [(dict(key.key), key.time_period) for key in reference.matched_keys.rows()]
    assert [(key.left_value, key.right_value) for key in result.matched_keys.rows()] == [
        (1.0, 10.0),
        (3.0, 30.0),
    ]
    assert (result.left_dropped_key_count, result.right_dropped_key_count) == (1, 1)

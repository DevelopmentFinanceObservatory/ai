"""Tests for `compiler.frame_columns`: binding a plan's role columns to a frame's,
and bounding a step's StatVar count by the frame it reads.

`tests/executor/test_wide_frame.py` runs every verb over a CRS family frame
end to end; these pin the binding rules, the argument rewrite and the StatVar
bounds on their own.
"""

from __future__ import annotations

import pytest
from compiler.capabilities import CAPABILITIES
from compiler.frame_columns import (
    FrameColumns,
    UnboundColumnError,
    bound_arguments,
    bound_role_keys,
    bound_stat_var_count,
    frame_columns,
)
from planner.workflow import ToolId
from pydantic import JsonValue

FAMILY_ROLE_KEYS = ("financingRecipient", "financingProvider", "stat_var_crsPurpose")
AXES = {"financingProvider": "origin", "financingRecipient": "destination", "years": None}


def _analysis(name: str) -> ToolId:
    return ToolId(server="analysis-mcp", name=name)


def _family() -> FrameColumns:
    return frame_columns(
        role_keys=FAMILY_ROLE_KEYS, axis_by_column=AXES, compiled=("origin", "destination")
    )


def test_a_compiled_axis_binds_to_the_column_on_that_axis() -> None:
    columns = _family()

    assert columns.role_keys == ("financingProvider", "financingRecipient", "stat_var_crsPurpose")
    assert columns.column("origin") == "financingProvider"
    assert columns.column("destination") == "financingRecipient"


def test_a_compiled_name_the_frame_carries_binds_to_itself() -> None:
    columns = frame_columns(
        role_keys=("observationAbout",),
        axis_by_column={"observationAbout": None, "years": None},
        compiled=("observationAbout",),
    )

    assert columns.column("observationAbout") == "observationAbout"


def test_a_name_the_compiler_did_not_write_as_a_role_passes_unchanged() -> None:
    assert _family().column("time_period") == "time_period"


def test_an_axis_no_dimension_states_does_not_bind() -> None:
    columns = frame_columns(
        role_keys=FAMILY_ROLE_KEYS,
        axis_by_column={"financingProvider": None, "financingRecipient": None},
        compiled=("origin",),
    )

    with pytest.raises(UnboundColumnError, match="'origin' role"):
        columns.column("origin")


def test_an_axis_two_columns_share_binds_neither() -> None:
    columns = frame_columns(
        role_keys=("financingProvider", "provider"),
        axis_by_column={"financingProvider": "origin", "provider": "origin"},
        compiled=("origin",),
    )

    with pytest.raises(UnboundColumnError):
        columns.column("origin")


def test_an_axis_on_a_dimension_the_frame_does_not_carry_does_not_bind() -> None:
    columns = frame_columns(
        role_keys=("financingRecipient",),
        axis_by_column=AXES,
        compiled=("origin",),
    )

    with pytest.raises(UnboundColumnError):
        columns.column("origin")


def test_every_column_naming_argument_is_rewritten() -> None:
    family = _family()
    share = bound_arguments(
        _analysis("share_of"),
        {
            "frame_ref": {"spec_id": "spec_x"},
            "numerator_entity": {"destination": "country/KEN"},
            "denominator_entity": {"destination": "country/GHA"},
            "numerator_by": ["origin"],
            "denominator_by": {"origin": "origin"},
        },
        frames={"frame_ref": family},
    )
    filtered = bound_arguments(
        _analysis("filter"),
        {
            "predicates": [
                {"kind": "category_match", "column": "destination", "values": ["country/KEN"]},
                {
                    "kind": "ordered_threshold",
                    "column": "time_period",
                    "op": "gte",
                    "value": "2005",
                },
            ]
        },
        frames={"frame_ref": family},
    )
    grouped = bound_arguments(
        _analysis("aggregate"),
        {"aggregation": "sum", "group_by": ["origin", "time_period"]},
        frames={"frame_ref": family},
    )

    assert share == {
        "frame_ref": {"spec_id": "spec_x"},
        "numerator_entity": {"financingRecipient": "country/KEN"},
        "denominator_entity": {"financingRecipient": "country/GHA"},
        "numerator_by": ["financingProvider"],
        "denominator_by": {"financingProvider": "financingProvider"},
    }
    assert filtered["predicates"] == [
        {"kind": "category_match", "column": "financingRecipient", "values": ["country/KEN"]},
        {"kind": "ordered_threshold", "column": "time_period", "op": "gte", "value": "2005"},
    ]
    assert grouped == {"aggregation": "sum", "group_by": ["financingProvider", "time_period"]}


def test_a_two_sided_argument_binds_each_side_against_its_own_frame() -> None:
    other = frame_columns(
        role_keys=("provider",), axis_by_column={"provider": "origin"}, compiled=("origin",)
    )

    compared = bound_arguments(
        _analysis("compare"),
        {"left_entity": {"origin": "country/DEU"}, "right_entity": {"origin": "country/DEU"}},
        frames={"frame_ref": _family(), "right_frame_ref": other},
    )

    assert compared == {
        "left_entity": {"financingProvider": "country/DEU"},
        "right_entity": {"provider": "country/DEU"},
    }


@pytest.mark.parametrize("verb", ["share_of", "quotient_of"])
def test_a_dividing_verb_binds_each_sides_columns_against_its_own_frame(verb: str) -> None:
    # `denominator_by` pairs a numerator column with the denominator's, so its
    # keys take the numerator frame's spelling and its values the other's.
    population = frame_columns(
        role_keys=("observationAbout",),
        axis_by_column={"observationAbout": None},
        compiled=("observationAbout",),
    )

    divided = bound_arguments(
        _analysis(verb),
        {
            "numerator_entity": {"origin": "country/DEU"},
            "denominator_entity": {"observationAbout": "country/KEN"},
            "numerator_by": ["destination"],
            "denominator_by": {"destination": "observationAbout"},
        },
        frames={"frame_ref": _family(), "denominator_frame_ref": population},
    )

    assert divided == {
        "numerator_entity": {"financingProvider": "country/DEU"},
        "denominator_entity": {"observationAbout": "country/KEN"},
        "numerator_by": ["financingRecipient"],
        "denominator_by": {"financingRecipient": "observationAbout"},
    }


def test_a_join_key_needs_one_spelling_on_both_frames() -> None:
    other = frame_columns(
        role_keys=("provider",), axis_by_column={"provider": "origin"}, compiled=("origin",)
    )

    with pytest.raises(UnboundColumnError, match="every side"):
        bound_arguments(
            _analysis("join"),
            {"join_keys": ["origin"]},
            frames={"frame_ref": _family(), "right_frame_ref": other},
        )


def test_a_frame_whose_columns_are_unknown_keeps_the_plans_spelling() -> None:
    arguments = {"aggregation": "sum", "group_by": ["origin", "time_period"]}

    assert (
        bound_arguments(_analysis("aggregate"), arguments, frames={"frame_ref": None}) == arguments
    )


def test_a_filter_must_declare_every_role_column_of_its_input() -> None:
    assert bound_role_keys(
        _analysis("filter"),
        ("origin",),
        frame=_family(),
        input_role_keys=("financingRecipient", "financingProvider"),
    ) == ("financingProvider", "financingRecipient")


def test_any_other_verb_must_declare_its_predicted_roles_as_the_frame_spells_them() -> None:
    assert bound_role_keys(
        _analysis("aggregate"),
        ("destination", "origin"),
        frame=_family(),
        input_role_keys=FAMILY_ROLE_KEYS,
    ) == ("financingProvider", "financingRecipient")


def test_the_prediction_stands_when_the_frame_is_unknown() -> None:
    assert bound_role_keys(
        _analysis("aggregate"), ("origin",), frame=None, input_role_keys=None
    ) == ("origin",)
    assert bound_role_keys(_analysis("filter"), ("origin",), frame=None, input_role_keys=None) == (
        "origin",
    )


@pytest.mark.parametrize("verb", ["growth", "rank", "share_of", "quotient_of", "sum_of"])
@pytest.mark.parametrize("input_count", [1, 461, None])
def test_a_verb_reading_one_stat_var_declares_one(verb: str, input_count: int | None) -> None:
    assert bound_stat_var_count(_analysis(verb), {}, input_stat_var_count=input_count) == 1


@pytest.mark.parametrize("verb", ["filter", "compare"])
@pytest.mark.parametrize("input_count", [1, 461, None])
def test_a_verb_keeping_rows_of_its_input_declares_at_most_its_inputs_stat_vars(
    verb: str, input_count: int | None
) -> None:
    assert bound_stat_var_count(_analysis(verb), {}, input_stat_var_count=input_count) == (
        input_count
    )


def test_an_aggregate_declares_one_stat_var_unless_it_groups_by_stat_var() -> None:
    aggregate = _analysis("aggregate")

    assert (
        bound_stat_var_count(
            aggregate, {"group_by": ["financingProvider", "time_period"]}, input_stat_var_count=461
        )
        == 1
    )
    assert bound_stat_var_count(aggregate, {}, input_stat_var_count=461) == 1
    assert (
        bound_stat_var_count(
            aggregate, {"group_by": ["stat_var", "time_period"]}, input_stat_var_count=461
        )
        == 461
    )
    assert (
        bound_stat_var_count(aggregate, {"group_by": ["stat_var"]}, input_stat_var_count=None)
        is None
    )


@pytest.mark.parametrize("input_count", [1, None])
def test_a_join_declares_at_most_one_stat_var_per_side(input_count: int | None) -> None:
    assert bound_stat_var_count(_analysis("join"), {}, input_stat_var_count=input_count) == 2


def test_every_analysis_verb_has_a_stat_var_bound() -> None:
    # `resolve_anchor` implements no operation and returns no frame to bound.
    verbs = [
        capability.tool
        for capability in CAPABILITIES
        if capability.tool.server == "analysis-mcp" and capability.implements
    ]

    assert len(verbs) == 9
    for verb in verbs:
        bound_stat_var_count(verb, {}, input_stat_var_count=1)


_GET_DATA = ToolId(server="data-mcp", name="get_data")


def test_a_retrieval_summing_its_family_declares_one_stat_var() -> None:
    assert bound_stat_var_count(_GET_DATA, {"stat_vars": "sum"}, input_stat_var_count=None) == 1


@pytest.mark.parametrize("arguments", [{}, {"stat_vars": "separate"}])
def test_a_retrieval_that_does_not_sum_has_no_stat_var_bound(
    arguments: dict[str, JsonValue],
) -> None:
    assert bound_stat_var_count(_GET_DATA, arguments, input_stat_var_count=None) is None


def test_a_tool_the_table_does_not_cover_has_no_stat_var_bound() -> None:
    with pytest.raises(KeyError):
        bound_stat_var_count(
            ToolId(server="data-mcp", name="search"), {}, input_stat_var_count=None
        )

"""`result.VerbResult`, `observed_shape`, and the version map."""

from __future__ import annotations

from analysis_mcp.op_spec import Verb
from analysis_mcp.result import IMPLEMENTATION_VERSIONS, VERB_OUTPUT_SCHEMA, observed_shape
from dfo_frame_grammar import Frame


def test_every_registered_verb_has_an_implementation_version() -> None:
    assert set(IMPLEMENTATION_VERSIONS) == set(Verb)


def test_the_output_schema_declares_every_verb_result_field() -> None:
    assert VERB_OUTPUT_SCHEMA["properties"].keys() == {
        "spec_id",
        "content_version",
        "frame_content_version",
        "rows",
        "role_keys",
        "stat_var_count",
        "exclusions",
        "time",
    }
    assert set(VERB_OUTPUT_SCHEMA["required"]) >= {
        "spec_id",
        "content_version",
        "frame_content_version",
        "rows",
        "stat_var_count",
    }


def _row(*, stat_var: str, reporter: str, time_period: str, value: float) -> dict[str, object]:
    return {
        "stat_var": stat_var,
        "reporter": reporter,
        "time_period": time_period,
        "value": value,
        "unit": "dcid:USDollar",
    }


def test_observed_shape_of_a_keyless_frame_reports_no_role_columns() -> None:
    frame = Frame.build(
        role_keys=[],
        rows=[
            {"stat_var": "dcid:x", "time_period": "2022", "value": 1.0},
            {"stat_var": "dcid:x", "time_period": "2023", "value": 2.0},
        ],
    )

    role_keys, stat_var_count = observed_shape(frame)

    assert role_keys == ()
    assert stat_var_count == 1


def test_observed_shape_reports_the_frames_own_sorted_role_columns() -> None:
    frame = Frame.build(
        role_keys=["reporter"],
        rows=[_row(stat_var="dcid:x", reporter="dcid:USA", time_period="2022", value=1.0)],
    )

    role_keys, _ = observed_shape(frame)

    assert role_keys == ("reporter",)


def test_observed_shape_counts_distinct_stat_vars_across_two_measures() -> None:
    frame = Frame.build(
        role_keys=["reporter"],
        rows=[
            _row(stat_var="dcid:oda", reporter="dcid:USA", time_period="2022", value=1.0),
            _row(stat_var="dcid:gni", reporter="dcid:USA", time_period="2022", value=2.0),
        ],
    )

    _, stat_var_count = observed_shape(frame)

    assert stat_var_count == 2


def test_observed_shape_of_an_empty_frame_reports_zero_measures() -> None:
    frame = Frame.build(role_keys=["reporter"], rows=[])

    role_keys, stat_var_count = observed_shape(frame)

    assert role_keys == ("reporter",)
    assert stat_var_count == 0

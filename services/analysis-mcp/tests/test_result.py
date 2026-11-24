"""`result.VerbResult` and the version map."""

from __future__ import annotations

from analysis_mcp.op_spec import Verb
from analysis_mcp.result import IMPLEMENTATION_VERSIONS, VERB_OUTPUT_SCHEMA


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
        "dropped_key_count",
        "right_dropped_key_count",
        "ranked_before_limit",
        "exclusions",
        "time",
        "stat_vars_summed",
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

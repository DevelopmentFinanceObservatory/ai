"""The refusal grammar: every code has exactly one retry mode, every factory
builds a well-shaped `ToolRefusal`, and `to_tool_result` renders it correctly.
"""

from __future__ import annotations

import json
from typing import Any

import jsonschema
import pytest
from analysis_mcp import server
from analysis_mcp.runner_client import RUNNER_CODES
from analysis_mcp.tool_errors import (
    RetryMode,
    ToolErrorCode,
    ToolRefusal,
    to_tool_result,
)
from common.refusal import internal_error_message


def test_every_code_has_exactly_one_retry_mode() -> None:
    for code in ToolErrorCode:
        refusal = ToolRefusal(code=code, message="x", fix={})
        assert isinstance(refusal.retry_mode, RetryMode)


def test_internal_error_is_do_not_retry() -> None:
    refusal = ToolRefusal.internal_error(tool="aggregate")
    assert refusal.retry_mode == RetryMode.DO_NOT_RETRY


def test_frame_integrity_failed_is_do_not_retry() -> None:
    """Names a corrupted or tampered stored object, not a bad argument --
    retrying the identical frame_ref reaches the identical corrupted bytes."""
    refusal = ToolRefusal.frame_integrity_failed(
        spec_id="spec_" + "a" * 32, content_version="cv_" + "b" * 32, reason="sha256 mismatch"
    )
    assert refusal.retry_mode == RetryMode.DO_NOT_RETRY


def test_a_refusal_without_a_wait_carries_no_retry_after_s() -> None:
    result = to_tool_result(ToolRefusal.op_spec_invalid(detail="bad"))

    assert result.structured_content is not None
    assert "retry_after_s" not in result.structured_content
    assert "retry_after_s" not in result.content[0].text


def test_every_other_code_is_correct_and_retry() -> None:
    _not_correct_and_retry = {
        ToolErrorCode.NON_FINITE_RESULT,
        ToolErrorCode.INTERNAL_ERROR,
        ToolErrorCode.DERIVED_INPUT_UNRECOVERABLE,
        ToolErrorCode.MIXED_TIME_GRAIN,
        ToolErrorCode.FRAME_INTEGRITY_FAILED,
        ToolErrorCode.ROLLING_GRAIN_UNSUPPORTED,
        ToolErrorCode.SERVICE_UNAVAILABLE,
        ToolErrorCode.UPSTREAM_ERROR,
    }
    for code in ToolErrorCode:
        if code in _not_correct_and_retry:
            continue
        refusal = ToolRefusal(code=code, message="x", fix={})
        assert refusal.retry_mode == RetryMode.CORRECT_AND_RETRY


def test_op_spec_invalid_carries_the_detail_in_fix() -> None:
    refusal = ToolRefusal.op_spec_invalid(detail="missing field 'aggregation'")
    assert refusal.code == ToolErrorCode.OP_SPEC_INVALID
    assert refusal.fix["detail"] == "missing field 'aggregation'"


@pytest.mark.parametrize("whole_class", ["official", None], ids=["class", "no-class"])
def test_place_levels_mixed_beside_a_whole_total_counts_the_members_and_names_examples(
    whole_class: str | None,
) -> None:
    refusal = ToolRefusal.place_levels_mixed(
        column="financingProvider_party_kind",
        types=["party", "total"],
        whole_total="DACOfficialDonors",
        whole_class=whole_class,
        member_count=30,
        members=["country/AUT", "country/BEL", "country/DEU"],
    )

    assert refusal.fix == {
        "column": "financingProvider_party_kind",
        "types": ["party", "total"],
        "whole_total": "DACOfficialDonors",
        "whole_class": whole_class,
        "member_count": 30,
        "example_members": ["country/AUT", "country/BEL", "country/DEU"],
        "filters": [
            {"column": "financingProvider_party_whole", "values": ["true"]},
            {"column": "financingProvider_party_kind", "values": ["party", "residual"]},
        ],
    }
    # The count, not the three examples, is how many members the model is told of.
    assert "30 members" in refusal.message


def test_internal_error_names_the_tool_alone() -> None:
    refusal = ToolRefusal.internal_error(tool="aggregate")
    assert refusal.code == ToolErrorCode.INTERNAL_ERROR
    assert refusal.fix == {"tool": "aggregate"}
    assert refusal.message == internal_error_message(subject="aggregate")


def test_to_tool_result_is_an_error_result_with_the_code_and_retry_mode() -> None:
    refusal = ToolRefusal.frame_not_held(
        spec_id="spec_" + "a" * 32, content_version="cv_" + "b" * 32
    )
    result = to_tool_result(refusal)
    assert result.is_error is True
    assert result.structured_content["code"] == "frame_not_held"
    assert result.structured_content["retry_mode"] == "correct_and_retry"
    assert result.structured_content["fix"] == {
        "spec_id": "spec_" + "a" * 32,
        "content_version": "cv_" + "b" * 32,
    }


def test_to_tool_result_preamble_prefixes_content_but_not_structured_content() -> None:
    refusal = ToolRefusal.frame_not_held(
        spec_id="spec_" + "a" * 32, content_version="cv_" + "b" * 32
    )
    result = to_tool_result(refusal, preamble="Partial success: 2 of 3 resolved.")
    assert result.content[0].text.startswith("Partial success: 2 of 3 resolved.\n\n")
    assert result.structured_content["fix"] == {
        "spec_id": "spec_" + "a" * 32,
        "content_version": "cv_" + "b" * 32,
    }


# The refusal envelope rendered into `content` itself, for a client that shows the model
# only `content` and never reads `structured_content`. -----------------------------------


def test_to_tool_result_appends_the_envelope_as_one_line_of_json_after_a_blank_line() -> None:
    refusal = ToolRefusal.frame_not_held(
        spec_id="spec_" + "a" * 32, content_version="cv_" + "b" * 32
    )

    result = to_tool_result(refusal)
    text = result.content[0].text  # type: ignore[union-attr]
    message, _, envelope_line = text.partition("\n\n")

    assert message == refusal.message
    assert "\n" not in envelope_line
    assert json.loads(envelope_line) == {
        "code": "frame_not_held",
        "retry_mode": "correct_and_retry",
        "fix": refusal.fix,
    }


def test_envelope_never_carries_a_retry_after_s_key() -> None:
    """This module carries no `retry_after_s` at all (see the module docstring), so the
    envelope never includes one. data-mcp's equivalent does."""
    refusal = ToolRefusal.internal_error(tool="aggregate")

    result = to_tool_result(refusal)
    _, _, envelope_line = result.content[0].text.partition("\n\n")  # type: ignore[union-attr]

    assert "retry_after_s" not in json.loads(envelope_line)


def test_preamble_still_precedes_the_message_with_the_envelope_appended() -> None:
    refusal = ToolRefusal.frame_not_held(
        spec_id="spec_" + "a" * 32, content_version="cv_" + "b" * 32
    )

    result = to_tool_result(refusal, preamble="Partial success: 2 of 3 resolved.")
    text = result.content[0].text  # type: ignore[union-attr]

    assert text.startswith("Partial success: 2 of 3 resolved.\n\n")
    _, message, envelope_line = text.split("\n\n")
    assert message == refusal.message
    assert json.loads(envelope_line)["code"] == "frame_not_held"


# The fix contract: a fix names only arguments, and values, its tool accepts -------

_SPEC = "spec_" + "1" * 32
_CV = "cv_" + "2" * 32
_PAIR = {"spec_id": _SPEC, "content_version": _CV}

# One refusal of each factory, and a registered tool that refuses with it.
_FIX_SAMPLES: dict[str, tuple[str, dict[str, Any]]] = {
    "op_spec_invalid": ("rank", {"detail": "x"}),
    "place_level_side_unnamed": (
        "aggregate",
        {"columns": ["observationAbout_type"], "available": ["financingRecipient_type"]},
    ),
    "frame_not_held": ("aggregate", {"spec_id": _SPEC, "content_version": _CV}),
    "frame_expired": ("aggregate", {"spec_id": _SPEC, "content_version": _CV}),
    "frames_lost": ("compare", {"frame_refs": [_PAIR, _PAIR], "expired": False}),
    "bare_frames_lost": ("compare", {"spec_ids": ["spec_" + "1" * 32], "expired": False}),
    "frame_ref_incomplete": (
        "compare",
        {"argument": "right_frame_ref", "spec_id": "ops_" + "1" * 32},
    ),
    "frame_integrity_failed": (
        "aggregate",
        {"spec_id": _SPEC, "content_version": _CV, "reason": "r"},
    ),
    "share_units_differ": (
        "share_of",
        {"numerator_unit": "dcid:USDollar", "denominator_unit": "dcid:Person"},
    ),
    "derived_input_root_lost": (
        "growth",
        {"frame_refs": [_PAIR], "root": _PAIR, "upstream_code": "snapshot_not_found"},
    ),
    "place_levels_mixed": (
        "rank",
        {"column": "observationAbout_type", "types": ["Country", "Place"]},
    ),
    "rolling_anchor_unresolved": (
        "growth",
        {
            "detail": "anchor_record_missing",
            "reason": "r",
            "context": {"anchor_key": "rak_" + "1" * 32},
        },
    ),
    "latest_year_unresolved": ("growth", {"fix": {"detail": "no_qualifying_period"}}),
    "rolling_reanchor_unavailable": ("compare", {"frame_refs": [_PAIR]}),
    "frame_store_unavailable": (
        "aggregate",
        {"spec_id": _SPEC, "content_version": _CV, "retry_after_s": 5.0},
    ),
    "analysis_pending": ("aggregate", {"retry_after_s": 2.0, "operations": ["op_1"]}),
    "rendering_not_recorded": (
        "aggregate",
        {"spec_id": _SPEC, "content_version": _CV, "options": ["currency"]},
    ),
    "upstream_error": ("aggregate", {"upstream_code": "x", "detail": "d", "retry_after_s": None}),
    "internal_error": ("aggregate", {"tool": "aggregate"}),
}

# The fix keys whose value names an argument of the tool.
_NAMING_KEYS = ("parameter", "argument")


def test_every_factory_has_a_fix_sample_and_every_code_a_source() -> None:
    """Every code is built by a factory or arrives from the analysis runner."""
    factories = {
        name for name, member in vars(ToolRefusal).items() if isinstance(member, classmethod)
    }
    assert set(_FIX_SAMPLES) == factories
    built = {
        getattr(ToolRefusal, name)(**kwargs).code for name, (_, kwargs) in _FIX_SAMPLES.items()
    }
    assert {code for code in ToolErrorCode if code not in built} <= RUNNER_CODES


@pytest.mark.parametrize("factory", _FIX_SAMPLES)
async def test_a_fix_names_only_arguments_and_values_its_tool_accepts(factory: str) -> None:
    """A fix key that is an argument of any registered tool is one of this tool's,
    with a value its schema accepts, and a key that names an argument names one of
    this tool's."""
    schemas = {tool.name: tool.parameters for tool in await server.mcp.list_tools()}
    arguments = set().union(*(schema["properties"] for schema in schemas.values()))
    tool, kwargs = _FIX_SAMPLES[factory]
    properties = schemas[tool]["properties"]
    refusal = getattr(ToolRefusal, factory)(**kwargs)

    for key, value in refusal.fix.items():
        if key in arguments:
            assert key in properties, (factory, tool, key)
            jsonschema.validate(value, properties[key])
        if key in _NAMING_KEYS:
            assert value in properties, (factory, tool, key, value)

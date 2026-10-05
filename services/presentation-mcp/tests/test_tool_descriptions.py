"""Tool-description tests confirm the model-facing prose states the facts a caller needs.

The three tool docstrings and the schema constants they compose with are the
model-facing surface (`server.py`'s module docstring explains why).
`server.py::_reply` mirrors a tool's payload into `content` and
`structured_content` alike. The harness reads `structured_content` in code,
and is this deployment's only consumer of it. These tests read the same
strings a caller sees in `tools/list`. They check the wording, not whether
the described behaviour holds.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from presentation_mcp.server import describe_component, describe_frame, write_component
from presentation_mcp.tool_schema import (
    _COMPONENT_ID_DESCRIPTION,
    _DESCRIBE_COMPONENT_ID_DESCRIPTION,
    _DESCRIBE_INCLUDE_PROJECTION_DESCRIPTION,
    _FRAME_REF_DESCRIPTION,
    _INCLUDE_PROJECTION_DESCRIPTION,
    _INTENT_DESCRIPTION,
    _WRITE_TOKEN_DESCRIPTION,
    write_component_schema,
)

RELEASE: dict[str, Any] = json.loads(
    (Path(__file__).parent / "fixtures" / "release.json").read_text(encoding="utf-8")
)


def test_write_component_docstring_names_the_six_types() -> None:
    doc = write_component.__doc__ or ""
    for member in (
        "line_chart",
        "stat_card",
        "bar_chart",
        "table",
        "note_narration",
        "source_card",
    ):
        assert member in doc


def test_write_component_docstring_states_the_change_rule() -> None:
    doc = write_component.__doc__ or ""
    assert "fact_summary.change" in doc
    assert "stat_card" in doc
    assert "basis" in doc
    assert "no change property" in doc


def test_write_component_docstring_states_the_id_is_the_only_way_back() -> None:
    doc = " ".join((write_component.__doc__ or "").split())
    assert "only way back to that component" in doc
    assert "lease" not in doc
    assert "different caller" not in doc


def test_write_component_docstring_states_the_write_token_rule() -> None:
    doc = " ".join((write_component.__doc__ or "").split())
    assert "Passing component_id and write_token" in doc
    assert "the component_id alone lets anyone read the component" in doc
    assert "Without the token, create a new component" in doc


def test_write_token_description_states_it_is_required_with_component_id() -> None:
    assert "Required with component_id" in _WRITE_TOKEN_DESCRIPTION
    assert "Without it, create a new component" in _WRITE_TOKEN_DESCRIPTION
    assert "write_token" in _COMPONENT_ID_DESCRIPTION


def test_component_tool_docstrings_state_an_unidentified_client_gets_no_table() -> None:
    for tool in (write_component, describe_component):
        doc = " ".join((tool.__doc__ or "").split())
        assert "a client whose capabilities the server cannot see" in doc


def test_write_component_docstring_states_analysis_toolkit_frame_refs_are_presentable() -> None:
    doc = write_component.__doc__ or ""
    assert "ops_" in doc
    assert "ocv_" in doc
    assert "presentable" in doc


def test_write_component_docstring_states_the_projection_rule() -> None:
    doc = " ".join((write_component.__doc__ or "").split())
    assert "Markdown table" in doc
    assert "only title, scale, or range" in doc
    assert "one line naming what changed" in doc
    assert "include_projection" in doc
    assert "full always sends the table, none leaves it out" in doc


def test_component_tool_docstrings_state_which_clients_get_the_table() -> None:
    for tool in (write_component, describe_component):
        doc = " ".join((tool.__doc__ or "").split())
        assert "draws the component as an interactive view gets that lead without the" in doc
        assert "full always sends the table, none leaves it out" in doc


def test_include_projection_description_states_the_three_choices() -> None:
    assert (
        "auto sends it on a create and on an edit, except an edit whose patch "
        "changes only title, scale, or range"
    ) in _INCLUDE_PROJECTION_DESCRIPTION
    assert "full always sends the table" in _INCLUDE_PROJECTION_DESCRIPTION
    assert "none leaves it out" in _INCLUDE_PROJECTION_DESCRIPTION


def test_include_projection_descriptions_state_which_clients_get_the_table() -> None:
    for description in (_INCLUDE_PROJECTION_DESCRIPTION, _DESCRIBE_INCLUDE_PROJECTION_DESCRIPTION):
        assert "the component as an interactive view" in description
        assert "without the table" in description
        assert "full always sends the table" in description
        assert "none leaves it out" in description


def test_describe_frame_docstring_defines_frame_ref_and_the_bearer_property() -> None:
    doc = describe_frame.__doc__ or ""
    assert "spec_id" in doc
    assert "content_version" in doc
    assert "Any caller holding a frame_ref can use it" in doc


def test_describe_component_docstring_defines_revision_and_names_no_lease() -> None:
    doc = describe_component.__doc__ or ""
    assert "revision" in doc
    assert "1 on create" in doc
    assert "different caller" not in doc


def test_intent_description_states_the_type_conditional_create_contract() -> None:
    assert "complete member" in _INTENT_DESCRIPTION
    assert "line_chart, stat_card, bar_chart, and table require type, frame_ref, and select" in (
        _INTENT_DESCRIPTION
    )
    assert "note_narration requires type and text" in _INTENT_DESCRIPTION
    assert "source_card requires type and frame_ref, not select" in _INTENT_DESCRIPTION


def test_intent_description_states_the_patch_and_change_rules() -> None:
    assert "partial patch" in _INTENT_DESCRIPTION
    assert "fact_summary.change" in _INTENT_DESCRIPTION
    assert "basis" in _INTENT_DESCRIPTION
    for member in ("line_chart", "bar_chart", "table", "note_narration", "source_card"):
        assert member in _INTENT_DESCRIPTION


def test_component_id_descriptions_state_the_id_is_the_only_way_back() -> None:
    assert "only way back to a component" in _COMPONENT_ID_DESCRIPTION
    assert "reuse it to revise or describe" in _COMPONENT_ID_DESCRIPTION
    assert "only way back to a component" in _DESCRIBE_COMPONENT_ID_DESCRIPTION
    for description in (_COMPONENT_ID_DESCRIPTION, _DESCRIBE_COMPONENT_ID_DESCRIPTION):
        assert "lease" not in description
        assert "x-dfo-caller" not in description


def test_frame_ref_description_defines_the_pair_and_the_bearer_property() -> None:
    assert "spec_id" in _FRAME_REF_DESCRIPTION
    assert "content_version" in _FRAME_REF_DESCRIPTION
    assert "Any caller holding a frame_ref can use it" in _FRAME_REF_DESCRIPTION


def test_frame_ref_description_states_analysis_toolkit_ids_are_presentable() -> None:
    assert "ops_" in _FRAME_REF_DESCRIPTION
    assert "ocv_" in _FRAME_REF_DESCRIPTION
    assert "presentable" in _FRAME_REF_DESCRIPTION


def test_write_component_schema_frame_ref_def_names_the_analysis_toolkit_result() -> None:
    schema = write_component_schema(release=RELEASE)
    assert "ops_" in schema["$defs"]["frame_ref"]["description"]


def test_artifact_resource_docstring_states_the_bearer_property() -> None:
    doc = (
        Path(__file__).parents[2] / "data-mcp" / "src" / "data_mcp" / "artifact_resource.py"
    ).read_text(encoding="utf-8")
    normalized = " ".join(doc.split())
    assert "bearer credential" in normalized
    assert "Any session holding it can read it" in normalized

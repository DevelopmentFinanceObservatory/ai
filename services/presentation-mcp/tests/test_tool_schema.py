"""Assembly tests: the published tool schemas against the captured release fixture.

The property under test: assembly reduces the platform's six member schemas
to one `anyOf`, keeps only the `$defs` a branch actually references, and
never emits a dangling `$ref`.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from presentation_mcp.tool_schema import (
    describe_component_schema,
    describe_frame_schema,
    write_component_schema,
)

RELEASE: dict[str, Any] = json.loads(
    (Path(__file__).parent / "fixtures" / "release.json").read_text(encoding="utf-8")
)

MEMBERS = RELEASE["intent_schemas_tolerant"]


def _all_refs(node: Any) -> set[str]:
    """Every `#/$defs/<name>` target anywhere in `node`."""
    found: set[str] = set()
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/$defs/"):
            found.add(ref.removeprefix("#/$defs/"))
        for value in node.values():
            found |= _all_refs(value)
    elif isinstance(node, list):
        for item in node:
            found |= _all_refs(item)
    return found


def test_write_component_schema_has_six_anyof_branches() -> None:
    schema = write_component_schema(release=RELEASE)

    branches = schema["properties"]["intent"]["anyOf"]
    assert len(branches) == 6
    assert {branch["properties"]["type"]["const"] for branch in branches} == set(MEMBERS)


def test_write_component_schema_defs_are_exactly_the_reachable_four() -> None:
    schema = write_component_schema(release=RELEASE)

    assert set(schema["$defs"]) == {"frame_ref", "relative_period", "title", "window"}


def test_write_component_schema_offers_a_relative_period_wherever_a_period_is_stated() -> None:
    """A model reads the relative form only from this schema, so each period
    field must offer it beside the label."""
    schema = write_component_schema(release=RELEASE)
    (stat_card,) = [
        branch
        for branch in schema["properties"]["intent"]["anyOf"]
        if branch["properties"]["type"]["const"] == "stat_card"
    ]
    change, _ = stat_card["properties"]["change"]["anyOf"]
    window = schema["$defs"]["window"]["properties"]

    for forms in (
        window["start"]["oneOf"],
        window["end"]["oneOf"],
        change["properties"]["basis"]["items"]["oneOf"],
    ):
        assert {"$ref": "#/$defs/relative_period"} in forms
    assert schema["$defs"]["relative_period"]["properties"] == {
        "anchor": {"const": "latest"},
        "offset": {"type": "integer", "maximum": 0},
    }


def test_write_component_schema_refs_all_resolve_within_the_document() -> None:
    schema = write_component_schema(release=RELEASE)

    intent = schema["properties"]["intent"]
    refs = _all_refs(intent["anyOf"])
    assert refs <= set(schema["$defs"])
    assert refs  # the branches do reference $defs; an empty set would be a vacuous pass


def _resolve_pointer(root: dict[str, Any], ref: str) -> Any:
    """Resolve a `#/a/b/c` JSON Pointer against `root` by walking dict lookups."""
    assert ref.startswith("#/")
    node: Any = root
    for segment in ref.removeprefix("#/").split("/"):
        node = node[segment]
    return node


def test_write_component_schema_refs_resolve_from_the_document_root() -> None:
    """Every `$ref` in the assembled schema follows its JSON Pointer from the
    schema root — the property a strict resolver would use, not `$defs`
    location by convention."""
    schema = write_component_schema(release=RELEASE)

    refs = _all_refs(schema["properties"]["intent"]["anyOf"])
    assert refs  # the branches do reference $defs; an empty set would be a vacuous pass
    for name in refs:
        resolved = _resolve_pointer(schema, f"#/$defs/{name}")
        assert resolved == schema["$defs"][name]
        # the resolved def body itself must carry no ref this walk can't also resolve
        assert _all_refs(resolved) <= set(schema["$defs"])


def test_write_component_schema_size_is_bounded() -> None:
    schema = write_component_schema(release=RELEASE)

    # 9,682 characters at catalogue release 2026.10.0, which added the relative
    # period, the stat card's change, and their descriptions.
    assert len(json.dumps(schema)) < 10_000


def test_write_component_schema_top_level_shape() -> None:
    schema = write_component_schema(release=RELEASE)

    assert schema["required"] == ["intent"]
    assert set(schema["properties"]) == {
        "intent",
        "component_id",
        "write_token",
        "include_projection",
    }
    assert schema["additionalProperties"] is False


def test_write_component_schema_include_projection_is_a_closed_choice_defaulting_to_auto() -> None:
    schema = write_component_schema(release=RELEASE)

    include_projection = schema["properties"]["include_projection"]
    assert include_projection["enum"] == ["auto", "full", "none"]
    assert include_projection["default"] == "auto"


def test_write_component_schema_component_id_requires_a_non_empty_string() -> None:
    schema = write_component_schema(release=RELEASE)

    assert schema["properties"]["component_id"]["minLength"] == 1
    assert schema["properties"]["component_id"]["pattern"] == r"^cmp_[A-Za-z0-9_-]{1,64}$"


def test_describe_frame_schema_rejects_additional_properties() -> None:
    schema = describe_frame_schema(release=RELEASE)

    assert schema["additionalProperties"] is False


def test_defs_mismatch_raises_instead_of_picking_one() -> None:
    rigged = copy.deepcopy(RELEASE)
    rigged["intent_schemas_tolerant"]["note_narration"]["$defs"]["title"]["maxLength"] = 1

    with pytest.raises(ValueError, match=r"\$defs"):
        write_component_schema(release=rigged)


def test_describe_frame_schema_lifts_frame_ref_with_only_spec_id_required() -> None:
    schema = describe_frame_schema(release=RELEASE)

    frame_ref = schema["properties"]["frame_ref"]
    assert frame_ref["required"] == ["spec_id"]
    assert schema["required"] == ["frame_ref"]


def test_describe_frame_schema_frame_ref_matches_the_shared_def_besides_local_id_patterns() -> None:
    schema = describe_frame_schema(release=RELEASE)

    frame_ref = dict(schema["properties"]["frame_ref"])
    shared = dict(MEMBERS["line_chart"]["$defs"]["frame_ref"])
    for ref in (frame_ref, shared):
        ref.pop("description")
        ref.pop("required")
    frame_ref["properties"] = {
        name: {key: value for key, value in property_schema.items() if key != "pattern"}
        for name, property_schema in frame_ref["properties"].items()
    }

    assert frame_ref == shared


def test_describe_frame_schema_requires_valid_path_segments() -> None:
    schema = describe_frame_schema(release=RELEASE)
    properties = schema["properties"]["frame_ref"]["properties"]

    assert properties["spec_id"]["pattern"] == r"^(?:spec|ops)_[a-z0-9]{1,64}$"
    assert properties["content_version"]["pattern"] == r"^(?:cv|ocv)_[a-z0-9]{1,64}$"


def test_describe_component_schema_requires_a_non_empty_component_id() -> None:
    schema = describe_component_schema()

    assert schema["properties"]["component_id"]["minLength"] == 1
    assert schema["properties"]["component_id"]["pattern"] == r"^cmp_[A-Za-z0-9_-]{1,64}$"
    assert schema["required"] == ["component_id"]


def test_describe_component_schema_revision_is_a_positive_integer() -> None:
    schema = describe_component_schema()

    revision = schema["properties"]["revision"]
    assert revision["type"] == "integer"
    assert revision["minimum"] == 1
    assert "revision" not in schema["required"]


def test_describe_component_schema_include_projection_is_a_closed_choice_defaulting_to_auto() -> (
    None
):
    schema = describe_component_schema()

    include_projection = schema["properties"]["include_projection"]
    assert include_projection["enum"] == ["auto", "full", "none"]
    assert include_projection["default"] == "auto"
    assert "include_projection" not in schema["required"]


def test_describe_component_schema_rejects_additional_properties() -> None:
    schema = describe_component_schema()

    assert schema["additionalProperties"] is False

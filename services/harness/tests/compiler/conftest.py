"""Fixtures for the compiler suite: a pinned tool surface and a discovered source.

The schemas here are cut down from the real `get_data`, `aggregate`, `growth`,
`compare`, `rank`, `share_of`, `quotient_of`, `join` and `filter` -- same required arguments,
same types, same `additionalProperties: false` -- so a test can state a case
against a fixed surface. `test_capabilities.py` is what checks the real
surface still matches what the capability table pins, and `test_surface.py`
that these still resemble the real thing.

`capabilities` re-pins the production table to these fixture schemas, so a
compile test exercises the table's mapping without tripping the drift gate the
table exists to hold against the real servers.

`place_scope_block` and `role_open_block` are real `search` spec blocks that
leave a population open: every country, and every donor of education aid to
Kenya. They serve the cases that rank or group over a population nobody named.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from compiler.capabilities import CAPABILITIES, Phase, ToolCapability
from compiler.surface import ToolSchema

SPEC_KENYA = "spec_" + "a" * 32
SPEC_OTHER = "spec_" + "b" * 32

_FIXTURES = Path(__file__).parent.parent / "fixtures"

FRAME_REF_PARAMETER: dict[str, Any] = {
    "type": "object",
    "additionalProperties": {"type": "string"},
}

GET_DATA_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "spec_id": {"type": "string"},
        "scope": {"type": "object"},
        "stat_vars": {"type": "string", "enum": ["sum", "separate"], "default": "separate"},
        "place_types": {"type": "boolean", "default": False},
        "without_party_totals": {"type": "boolean", "default": False},
        "breakdown": {
            "anyOf": [
                {
                    "type": "object",
                    "properties": {
                        "code_system": {"type": "string", "minLength": 1},
                        "level": {"type": "integer", "minimum": 0},
                    },
                    "required": ["code_system", "level"],
                    "additionalProperties": False,
                },
                {"type": "null"},
            ],
            "default": None,
        },
    },
    "required": ["spec_id"],
    "additionalProperties": False,
}

AGGREGATE_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "frame_ref": FRAME_REF_PARAMETER,
        "aggregation": {"type": "string", "minLength": 1, "maxLength": 64},
        "group_by": {"type": "array", "items": {"type": "string"}, "default": []},
        "unit": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
    },
    "required": ["frame_ref", "aggregation"],
    "additionalProperties": False,
}

GROWTH_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "frame_ref": FRAME_REF_PARAMETER,
        "window_start": {"type": "string", "pattern": r"^\d{4}$"},
        "window_end": {"type": "string", "pattern": r"^\d{4}$"},
        "entity_by": {"type": "array", "items": {"type": "string"}, "default": []},
        "unit": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
        "change": {
            "type": "string",
            "enum": ["relative", "absolute"],
            "default": "relative",
        },
    },
    "required": ["frame_ref", "window_start", "window_end"],
    "additionalProperties": False,
}

# A unit pin: a unit ref, or a converted unit's `{currency, prices, base_year}`
# object, as analysis-mcp's `PinnedUnit` declares it.
_PINNED_UNIT_PARAMETER: dict[str, Any] = {
    "anyOf": [
        {"type": "string"},
        {
            "type": "object",
            "properties": {
                "currency": {"type": "string", "pattern": "^[A-Z]{3}$"},
                "prices": {"enum": ["current", "constant"]},
                "base_year": {"type": "integer"},
            },
            "required": ["currency", "prices"],
            "additionalProperties": False,
        },
        {"type": "null"},
    ],
    "default": None,
}

COMPARE_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "frame_ref": FRAME_REF_PARAMETER,
        "right_frame_ref": {"anyOf": [FRAME_REF_PARAMETER, {"type": "null"}], "default": None},
        "left_entity": {"anyOf": [FRAME_REF_PARAMETER, {"type": "null"}], "default": None},
        "right_entity": {"anyOf": [FRAME_REF_PARAMETER, {"type": "null"}], "default": None},
        "left_unit": _PINNED_UNIT_PARAMETER,
        "right_unit": _PINNED_UNIT_PARAMETER,
    },
    "required": ["frame_ref"],
    "additionalProperties": False,
}

RANK_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "frame_ref": FRAME_REF_PARAMETER,
        "rank_by": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "aggregation": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
        "window_start": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
        "window_end": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
        "unit": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
        "ascending": {"type": "boolean", "default": False},
        "per_period": {"type": "boolean", "default": False},
        "limit": {"anyOf": [{"type": "integer", "minimum": 1}, {"type": "null"}], "default": None},
    },
    "required": ["frame_ref", "rank_by"],
    "additionalProperties": False,
}

SHARE_OF_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "frame_ref": FRAME_REF_PARAMETER,
        "denominator_frame_ref": {
            "anyOf": [FRAME_REF_PARAMETER, {"type": "null"}],
            "default": None,
        },
        "numerator_entity": {"anyOf": [FRAME_REF_PARAMETER, {"type": "null"}], "default": None},
        "denominator_entity": {"anyOf": [FRAME_REF_PARAMETER, {"type": "null"}], "default": None},
        "numerator_unit": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
        "denominator_unit": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
        "numerator_by": {"type": "array", "items": {"type": "string"}, "default": []},
        "denominator_by": {
            "type": "object",
            "additionalProperties": {"type": "string"},
            "default": {},
        },
    },
    "required": ["frame_ref"],
    "additionalProperties": False,
}

QUOTIENT_OF_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "frame_ref": FRAME_REF_PARAMETER,
        "denominator_frame_ref": {
            "anyOf": [FRAME_REF_PARAMETER, {"type": "null"}],
            "default": None,
        },
        "numerator_entity": {"anyOf": [FRAME_REF_PARAMETER, {"type": "null"}], "default": None},
        "denominator_entity": {"anyOf": [FRAME_REF_PARAMETER, {"type": "null"}], "default": None},
        "numerator_unit": _PINNED_UNIT_PARAMETER,
        "denominator_unit": _PINNED_UNIT_PARAMETER,
        "numerator_by": {
            "anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "null"}],
            "default": None,
        },
        "denominator_by": {
            "anyOf": [
                {"type": "object", "additionalProperties": {"type": "string"}},
                {"type": "null"},
            ],
            "default": None,
        },
    },
    "required": ["frame_ref"],
    "additionalProperties": False,
}

JOIN_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "frame_ref": FRAME_REF_PARAMETER,
        "right_frame_ref": FRAME_REF_PARAMETER,
        "join_keys": {"type": "array", "items": {"type": "string"}, "default": []},
        "left_unit": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
        "right_unit": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
    },
    "required": ["frame_ref", "right_frame_ref"],
    "additionalProperties": False,
}

_PREDICATE_ITEM: dict[str, Any] = {
    "anyOf": [
        {
            "type": "object",
            "properties": {
                "kind": {"const": "ordered_threshold"},
                "column": {"type": "string"},
                "op": {"enum": ["gt", "gte", "lt", "lte", "eq", "ne"]},
                "value": {"anyOf": [{"type": "number"}, {"type": "string"}]},
            },
            "required": ["column", "op", "value"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {
                "kind": {"const": "category_match"},
                "column": {"type": "string"},
                "values": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            },
            "required": ["column", "values"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {
                "kind": {"const": "place_level"},
                "column": {"type": "string"},
                "level": {"type": "string"},
                "strict": {"type": "boolean", "default": False},
            },
            "required": ["column", "level"],
            "additionalProperties": False,
        },
    ]
}

FILTER_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "frame_ref": FRAME_REF_PARAMETER,
        "predicates": {"type": "array", "items": _PREDICATE_ITEM, "default": []},
    },
    "required": ["frame_ref"],
    "additionalProperties": False,
}


@pytest.fixture
def tool_schema() -> Callable[..., ToolSchema]:
    """Return a factory for a `ToolSchema` with overridable fields."""

    def _tool_schema(**overrides: Any) -> ToolSchema:
        fields: dict[str, Any] = {
            "server": "analysis-mcp",
            "name": "aggregate",
            "description": "Group a stored frame and reduce each group.",
            "parameters": AGGREGATE_PARAMETERS,
        }
        return ToolSchema(**(fields | overrides))

    return _tool_schema


@pytest.fixture
def schemas(tool_schema: Callable[..., ToolSchema]) -> dict[str, ToolSchema]:
    """A ten-tool surface: retrieval, then the nine analysis verbs the compiler maps."""
    return {
        "get_data": tool_schema(
            server="data-mcp",
            name="get_data",
            description="Retrieve the data behind a spec_id.",
            parameters=GET_DATA_PARAMETERS,
        ),
        "aggregate": tool_schema(),
        "growth": tool_schema(
            name="growth", description="Growth across a window.", parameters=GROWTH_PARAMETERS
        ),
        "compare": tool_schema(
            name="compare", description="Compare two sides.", parameters=COMPARE_PARAMETERS
        ),
        "rank": tool_schema(name="rank", description="Rank entities.", parameters=RANK_PARAMETERS),
        "share_of": tool_schema(
            name="share_of", description="Share of a denominator.", parameters=SHARE_OF_PARAMETERS
        ),
        "quotient_of": tool_schema(
            name="quotient_of",
            description="Divide one side by another.",
            parameters=QUOTIENT_OF_PARAMETERS,
        ),
        "join": tool_schema(
            name="join", description="Join two frames.", parameters=JOIN_PARAMETERS
        ),
        "filter": tool_schema(
            name="filter", description="Filter rows by predicate.", parameters=FILTER_PARAMETERS
        ),
        "sum_of": tool_schema(
            name="sum_of", description="Add two sides.", parameters=COMPARE_PARAMETERS
        ),
        "resolve_anchor": tool_schema(
            name="resolve_anchor",
            description="Find the latest period rolling steps follow.",
            parameters={
                "type": "object",
                "properties": {
                    "populations": {"type": "array", "items": {"type": "object"}},
                    "roots": {"type": "array", "items": FRAME_REF_PARAMETER},
                },
                "required": ["populations", "roots"],
                "additionalProperties": False,
            },
        ),
    }


@pytest.fixture
def capabilities(schemas: dict[str, ToolSchema]) -> tuple[ToolCapability, ...]:
    """The production table, re-pinned to the fixture schemas it is tested against.

    `output_schema_hash`/`implementation_version` are re-pinned too, to
    whatever the fixture schema itself declares (`None`, by default, for
    every fixture tool here) -- the production row's own pins are read off
    the real servers and would otherwise read as drift against a fixture
    surface that declares neither.
    """
    return tuple(
        capability.model_copy(
            update={
                "input_schema_hash": schemas[capability.tool.name].input_schema_hash,
                "output_schema_hash": schemas[capability.tool.name].output_schema_hash,
                "implementation_version": schemas[capability.tool.name].version,
            }
        )
        if capability.phase is Phase.WORKFLOW
        else capability
        for capability in CAPABILITIES
    )


@pytest.fixture
def place_scope_block() -> dict[str, Any]:
    """dev `search`'s spec block for health spending per person, its 193 countries left open."""
    return json.loads((_FIXTURES / "search_spec_place_scope.json").read_text())["spec"]


@pytest.fixture
def role_open_block() -> dict[str, Any]:
    """dev `search`'s spec block for education aid to Kenya, its 55 donors left open on `origin`."""
    return json.loads((_FIXTURES / "search_spec_role_open.json").read_text())["spec"]

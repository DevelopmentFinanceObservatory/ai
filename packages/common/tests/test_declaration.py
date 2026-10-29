"""Tests for `common.declaration`: the five rules a declared input schema follows."""

from __future__ import annotations

import re
from typing import Annotated, Any

import pytest
from common.declaration import declared_schema, declared_tool
from fastmcp import Client, FastMCP
from fastmcp.tools import Tool
from pydantic import BaseModel, Field

_LONG = "This sentence has well over seven words in it."


def test_comments_and_empty_required_go_at_every_depth() -> None:
    schema = {
        "type": "object",
        "$comment": "a note",
        "properties": {
            "a": {"type": "object", "$comment": "x", "properties": {}, "required": []},
        },
        "required": ["a"],
    }

    assert declared_schema(schema, description="") == {
        "type": "object",
        "properties": {"a": {"type": "object", "properties": {}}},
        "required": ["a"],
    }


def test_definitions_are_inlined_with_the_description_beside_the_ref() -> None:
    schema = {
        "type": "object",
        "$defs": {"Level": {"type": "string", "enum": ["a", "b"]}},
        "properties": {"level": {"$ref": "#/$defs/Level", "description": "The level."}},
        "required": ["level"],
    }

    assert declared_schema(schema, description="") == {
        "type": "object",
        "properties": {
            "level": {"type": "string", "enum": ["a", "b"], "description": "The level."}
        },
        "required": ["level"],
    }


@pytest.mark.parametrize(
    ("properties", "defs", "pointer"),
    [
        pytest.param(
            {"a": {"$ref": "#/$defs/missing"}}, {}, "/properties/a", id="missing-definition"
        ),
        pytest.param(
            {"a": {"$ref": "#/$defs/Node"}},
            {
                "Node": {
                    "type": "object",
                    "properties": {"child": {"$ref": "#/$defs/Node"}},
                }
            },
            "/",
            id="recursive-definition",
        ),
        pytest.param(
            {"a": {"type": "object", "properties": {"b": {"$ref": "https://x.test/s.json"}}}},
            {},
            "/properties/a/properties/b",
            id="remote-ref",
        ),
    ],
)
def test_a_ref_that_cannot_be_inlined_refuses_naming_its_pointer(
    properties: dict[str, Any], defs: dict[str, Any], pointer: str
) -> None:
    schema = {"type": "object", "properties": properties, **({"$defs": defs} if defs else {})}

    with pytest.raises(ValueError, match=f"^{re.escape(pointer)} "):
        declared_schema(schema, description="")


def test_an_optional_property_is_its_own_type_with_a_default_that_is_not_null() -> None:
    schema = {
        "type": "object",
        "properties": {
            "none": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
            "set": {"anyOf": [{"type": "integer"}, {"type": "null"}], "default": 3},
            "two": {"anyOf": [{"type": "string"}, {"type": "integer"}, {"type": "null"}]},
            "kept": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        },
        "required": ["kept"],
    }

    assert declared_schema(schema, description="")["properties"] == {
        "none": {"type": "string"},
        "set": {"type": "integer", "default": 3},
        "two": {"anyOf": [{"type": "string"}, {"type": "integer"}]},
        # Required: null is a value the caller must be able to send.
        "kept": {"anyOf": [{"type": "string"}, {"type": "null"}]},
    }


def test_the_propertys_description_wins_over_its_types() -> None:
    schema = {
        "type": "object",
        "properties": {
            "scope": {
                "anyOf": [{"type": "object", "description": "The type's docstring."}, _NULL],
                "default": None,
                "description": "The argument.",
            }
        },
    }

    assert declared_schema(schema, description="")["properties"]["scope"] == {
        "type": "object",
        "description": "The argument.",
    }


_NULL = {"type": "null"}


def test_a_long_sentence_is_kept_once_first_in_the_description() -> None:
    schema = {
        "type": "object",
        "properties": {
            "a": {"type": "string", "description": f"Short one. {_LONG}"},
            "b": {
                "type": "string",
                "description": f"Another sentence with seven words here. {_LONG}",
            },
            "c": {"type": "string", "description": _LONG},
        },
    }

    properties = declared_schema(schema, description=f"Intro. {_LONG}")["properties"]

    assert properties["a"]["description"] == "Short one."
    assert properties["b"]["description"] == "Another sentence with seven words here."
    assert "description" not in properties["c"]


def test_sentences_compare_with_whitespace_collapsed_and_split_before_lowercase() -> None:
    wrapped = "This sentence has well over\n    seven words in it."
    schema = {
        "type": "object",
        "properties": {
            "a": {
                "type": "string",
                "description": f"lowercase starts count as sentences of seven words. {_LONG}",
            },
        },
    }

    properties = declared_schema(
        schema, description=f"Intro. {wrapped} lowercase starts count as sentences of seven words."
    )["properties"]

    assert "description" not in properties["a"]


def test_e_g_and_i_e_end_no_sentence() -> None:
    # Split after "e.g.", the second half would repeat the description's sentence and go.
    text = "Pass a level, e.g. Country names one level of place for the call."
    schema = {"type": "object", "properties": {"a": {"type": "string", "description": text}}}

    properties = declared_schema(
        schema, description="Country names one level of place for the call."
    )["properties"]

    assert properties["a"]["description"] == text


def test_a_description_that_loses_nothing_keeps_its_line_breaks() -> None:
    text = "First line of a description\nthat wraps onto a second line."
    schema = {"type": "object", "properties": {"a": {"type": "string", "description": text}}}

    assert declared_schema(schema, description="")["properties"]["a"]["description"] == text


def test_the_input_schema_is_left_unchanged() -> None:
    schema = {"type": "object", "properties": {"a": {"anyOf": [{"type": "string"}, _NULL]}}}

    declared_schema(schema, description="")

    assert schema == {"type": "object", "properties": {"a": {"anyOf": [{"type": "string"}, _NULL]}}}


class _Window(BaseModel):
    """A window."""

    start: int


async def test_a_hidden_argument_is_unlisted_and_still_parses() -> None:
    received: list[Any] = []

    def read(
        window: Annotated[_Window | None, Field(description="The window.")] = None,
        legacy: bool = True,
    ) -> str:
        received.append((window, legacy))
        return "ok"

    mcp = FastMCP("t")
    mcp.add_tool(declared_tool(Tool.from_function(read), hidden=("legacy",)))

    async with Client(mcp) as client:
        (tool,) = await client.list_tools()
        result = await client.call_tool("read", {"legacy": False}, raise_on_error=False)

    assert tool.input_schema["properties"] == {
        "window": {
            "type": "object",
            "properties": {"start": {"type": "integer"}},
            "required": ["start"],
            "description": "The window.",
        }
    }
    assert result.is_error is False
    assert received == [(None, False)]

"""Tests for the `search` tool: translation, client wiring, and refusals."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Coroutine
from typing import Any

import httpx2
import pytest
from data_mcp import client_holder
from data_mcp.digest import render_answer, render_digest
from data_mcp.search_result import SEARCH_ENVELOPE_META_KEY
from data_mcp.search_tool import SearchEntity, SearchPart, search, to_wire_request
from data_mcp.server import mcp
from fastmcp import Client
from fastmcp.client.client import CallToolResult
from pydantic import TypeAdapter

from .conftest import (
    _reset_client,  # noqa: F401 (re-exported for test_search_tool_definition_store.py)
)
from .test_client import _client, _problem_body, text_of
from .test_digest import _envelope, _spec

Handler = (
    Callable[[httpx2.Request], httpx2.Response]
    | Callable[[httpx2.Request], Coroutine[None, None, httpx2.Response]]
)


async def _call_search(
    handler: Handler, parts: list[dict[str, object]], *, raw_query: str | None = None
) -> CallToolResult:
    arguments: dict[str, object] = {"parts": parts}
    if raw_query is not None:
        arguments["raw_query"] = raw_query
    async with _client(handler) as upstream:
        client_holder.set_client(upstream)
        async with Client(mcp) as client:
            return await client.call_tool("search", arguments, raise_on_error=False)


def _json_handler(status: int, payload: dict[str, Any]) -> Handler:
    def handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(status, json=payload)

    return handler


# to_wire_request translation


def test_to_wire_request_renames_fields_and_nests_correctly() -> None:
    parts = [
        SearchPart(
            variable="malaria grants",
            entities=[
                SearchEntity(text="France", role="from"),
                SearchEntity(text="Togo", role="to"),
            ],
            time="2015 to 2024",
            unit="US dollars",
            source="OECD",
        )
    ]

    wire = to_wire_request(parts, raw_query="malaria grants from France to Togo")

    assert wire == {
        "schema_version": 1,
        "input": {
            "kind": "parsed",
            "parts": [
                {
                    "variable_text": "malaria grants",
                    "entities": [
                        {"text": "France", "role_text": "from"},
                        {"text": "Togo", "role_text": "to"},
                    ],
                    "time_text": "2015 to 2024",
                    "unit_text": "US dollars",
                    "source_text": "OECD",
                }
            ],
            "raw_query": "malaria grants from France to Togo",
        },
        "response_options": {
            "include_sentence": True,
            "include_option_coverage": True,
            "include_skills": True,
        },
    }


def test_to_wire_request_drops_none_fields() -> None:
    wire = to_wire_request([SearchPart(variable="malaria grants")], raw_query=None)

    assert wire["input"] == {"kind": "parsed", "parts": [{"variable_text": "malaria grants"}]}
    assert "raw_query" not in wire["input"]


def test_to_wire_request_entity_with_no_role_omits_role_text() -> None:
    parts = [SearchPart(variable="x", entities=[SearchEntity(text="Togo")])]

    wire = to_wire_request(parts, raw_query=None)

    assert wire["input"]["parts"][0]["entities"] == [{"text": "Togo"}]


def test_to_wire_request_asks_for_skills_and_leaves_include_observations_unset() -> None:
    wire = to_wire_request([SearchPart(variable="x")], raw_query=None)

    assert wire["response_options"] == {
        "include_sentence": True,
        "include_option_coverage": True,
        "include_skills": True,
    }


# The docstring's worked example


def test_docstring_worked_example_validates_against_search_part() -> None:
    assert search.__doc__ is not None
    match = re.search(r"```json\n(.*?)\n```", search.__doc__, re.DOTALL)
    assert match is not None

    payload = json.loads(match.group(1))
    parts = TypeAdapter(list[SearchPart]).validate_python(payload["parts"])

    assert len(parts) == 2
    assert parts[0].entities is not None
    assert parts[0].entities[0].role == "to"


# End-to-end tool wiring of the result's three parts and is_error


@pytest.mark.parametrize(
    ("status", "payload"),
    [
        ("definite", {"spec": _spec()}),
        (
            "candidates",
            {
                "candidates": {
                    "ordering": "intent_rank",
                    "max_candidates": 5,
                    "specs": [_spec(spec_id="a"), _spec(spec_id="b")],
                    "truncated": False,
                }
            },
        ),
        ("no_data", {"no_data": {"reason": "no_observations", "category": "absent"}}),
        (
            "unsupported",
            {
                "unsupported": {
                    "code": "facet_binding_unsupported",
                    "pointer": "/identity/facets/provenance",
                    "detail": "no binding exists",
                    "definition": {},
                }
            },
        ),
    ],
)
async def test_every_semantic_status_is_a_success_of_digest_answer_and_envelope(
    status: str, payload: dict[str, Any]
) -> None:
    envelope = _envelope(status, **payload)
    handler = _json_handler(200, envelope)

    result = await _call_search(handler, [{"variable": "malaria grants"}])

    assert result.is_error is False
    assert text_of(result) == render_digest(envelope)
    assert result.structured_content == render_answer(envelope)
    assert result.meta is not None
    assert result.meta[SEARCH_ENVELOPE_META_KEY] == envelope
    assert "diagnostics" not in (result.structured_content or {})


async def test_a_refusal_carries_its_teaching_payload_and_no_envelope() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("boom", request=request)

    result = await _call_search(handler, [{"variable": "malaria grants"}])

    assert result.is_error is True
    assert result.structured_content is not None
    assert "retry_mode" in result.structured_content
    assert not result.meta or SEARCH_ENVELOPE_META_KEY not in result.meta


async def test_skills_the_envelope_attaches_reach_the_digest() -> None:
    spec = _spec()
    spec["resolved"]["skills"] = ["health-financing"]
    envelope = _envelope("definite", spec=spec)
    envelope["skills"] = {
        "health-financing": {
            "id": "health-financing",
            "name": "health-financing",
            "description": "Health aid purpose codes and deflation choices.",
            "content_version": "sha256:abc",
        }
    }
    sent: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        sent.append(json.loads(request.content))
        return httpx2.Response(200, json=envelope)

    result = await _call_search(handler, [{"variable": "health aid"}])

    assert sent[0]["response_options"]["include_skills"] is True
    assert result.is_error is False
    assert "- health-financing: Health aid purpose codes and deflation choices." in text_of(result)


# Refusal paths


async def test_transport_error_becomes_a_refusal() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("boom", request=request)

    result = await _call_search(handler, [{"variable": "malaria grants"}])

    assert result.is_error is True
    assert result.structured_content is not None
    assert result.structured_content["retry_mode"] == "retry_after_delay"


async def test_request_invalid_problem_body_becomes_a_correct_and_retry_refusal_with_pointer() -> (
    None
):
    problem = json.loads(
        _problem_body(
            code="request_invalid", retryable=False, pointer="/input/parts/0/variable_text"
        )
    )
    handler = _json_handler(400, problem)

    result = await _call_search(handler, [{"variable": "malaria grants"}])

    assert result.is_error is True
    assert result.structured_content is not None
    assert result.structured_content["code"] == "request_invalid"
    assert result.structured_content["retry_mode"] == "correct_and_retry"
    assert result.structured_content["fix"]["pointer"] == "/input/parts/0/variable_text"


# Unusable search responses form the same class of gap `get_data_tool.py` guards its
# own mint response against, applied here to the resolve response.


async def test_non_json_search_response_refuses_rather_than_raising() -> None:
    def handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, content=b"not json")

    result = await _call_search(handler, [{"variable": "malaria grants"}])

    assert result.is_error is True
    assert result.structured_content is not None
    assert result.structured_content["code"] == "upstream_error"
    assert result.structured_content["retry_mode"] == "do_not_retry"


async def test_invalid_utf8_search_response_refuses_rather_than_raising() -> None:
    """`httpx2.Response.json()` raises `UnicodeDecodeError` on invalid UTF-8, a `ValueError`
    sibling of `json.JSONDecodeError` rather than a subclass, so a bare `except
    json.JSONDecodeError` lets this escape uncaught.
    """

    def handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, content=b"\xff\xfe\x00invalid utf8 \xff")

    result = await _call_search(handler, [{"variable": "malaria grants"}])

    assert result.is_error is True
    assert result.structured_content is not None
    assert result.structured_content["code"] == "upstream_error"
    assert result.structured_content["retry_mode"] == "do_not_retry"


async def test_malformed_envelope_refuses_rather_than_raising() -> None:
    """A warning entry that isn't a mapping breaks `_warning_lines`' `.get()` walk in
    `render_digest`. No spec node here gives `_register_definitions` anything to trip
    over first, so this proves the render-side guard specifically.
    """
    envelope = {
        "status": "definite",
        "parts": [{"status": "definite", "variable_text": "x", "spec": _spec()}],
        "diagnostics": {"warnings": ["oops"]},
    }
    handler = _json_handler(200, envelope)

    result = await _call_search(handler, [{"variable": "malaria grants"}])

    assert result.is_error is True
    assert result.structured_content is not None
    assert result.structured_content["code"] == "upstream_error"
    assert result.structured_content["retry_mode"] == "do_not_retry"


async def test_malformed_spec_node_in_register_definitions_refuses_rather_than_raising() -> None:
    """A non-mapping `spec` on a definite part breaks `_register_definitions`'s own `.get()`
    walk in `_spec_nodes`, which runs before `render_digest`, so a guard placed only around
    rendering can't catch this. The exception has to be caught around `_register_definitions`
    itself, not just around the render call that happens to run after it.
    """
    envelope = {
        "status": "definite",
        "parts": [{"status": "definite", "variable_text": "x", "spec": "not-a-spec"}],
    }
    handler = _json_handler(200, envelope)

    result = await _call_search(handler, [{"variable": "malaria grants"}])

    assert result.is_error is True
    assert result.structured_content is not None
    assert result.structured_content["code"] == "upstream_error"
    assert result.structured_content["retry_mode"] == "do_not_retry"


# Local argument validation


async def test_empty_variable_is_rejected_locally() -> None:
    async with Client(mcp) as client:
        result = await client.call_tool(
            "search", {"parts": [{"variable": ""}]}, raise_on_error=False
        )

    assert result.is_error is True


async def test_seven_parts_is_rejected_locally() -> None:
    parts = [{"variable": f"variable {i}"} for i in range(7)]

    async with Client(mcp) as client:
        result = await client.call_tool("search", {"parts": parts}, raise_on_error=False)

    assert result.is_error is True


async def test_raw_query_over_max_chars_is_rejected_locally() -> None:
    arguments = {"parts": [{"variable": "malaria grants"}], "raw_query": "x" * 2001}

    async with Client(mcp) as client:
        result = await client.call_tool("search", arguments, raise_on_error=False)

    assert result.is_error is True


async def test_empty_raw_query_is_rejected_locally() -> None:
    arguments = {"parts": [{"variable": "malaria grants"}], "raw_query": ""}

    async with Client(mcp) as client:
        result = await client.call_tool("search", arguments, raise_on_error=False)

    assert result.is_error is True


async def test_a_catalogue_answer_returns_searchs_reading_with_no_parts_and_no_definitions() -> (
    None
):
    """A `catalogue` answer (the question asked what DFO holds) carries a scope and no
    parts. The tool returns that reading as its result and registers nothing, since
    nothing in it carries a spec_id."""
    body = {
        "schema_version": 1,
        "kind": "catalogue",
        "query_echo": {"entry_path": "parsed", "extract_skipped": False},
        "diagnostics": {"engine_build": "test", "trace_id": "trace-1", "warnings": []},
        "scope": {"place": {"dcid": "country/KEN", "label": "Kenya"}},
    }

    result = await _call_search(_json_handler(200, body), [{"variable": "what data do you have"}])

    assert result.is_error is False
    assert text_of(result).startswith("Search read this question as asking what data DFO holds")
    assert "It named Kenya." in text_of(result)
    assert result.structured_content == {
        "kind": "catalogue",
        "status": None,
        "parts": [],
        "scope": {"place": {"dcid": "country/KEN", "label": "Kenya"}},
    }

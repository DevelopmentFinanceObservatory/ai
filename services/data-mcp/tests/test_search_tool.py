"""Tests for the `search` tool: translation, client wiring, and refusals."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Coroutine
from typing import Any

import httpx2
import pytest
from data_mcp import client_holder, search_tool, server
from data_mcp.deadline import CallDeadline
from data_mcp.digest import render_answer, render_digest
from data_mcp.search_result import SEARCH_ENVELOPE_META_KEY
from data_mcp.search_tool import (
    TIME_FORMS,
    DeclaredDataset,
    SearchEntity,
    SearchPart,
    declared_datasets,
    describe_search,
    fetch_declared_datasets,
    search,
    to_wire_request,
)
from data_mcp.server import mcp
from fastmcp import Client
from fastmcp.client.client import CallToolResult
from pydantic import TypeAdapter, ValidationError

from .conftest import (
    _reset_client,  # noqa: F401 (re-exported for test_search_tool_definition_store.py)
)
from .test_client import _client, _FakeClock, _problem_body, text_of
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
            dataset="dataset/grants",
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
                    "provenance": "dataset/grants",
                }
            ],
            "raw_query": "malaria grants from France to Togo",
        },
        "response_options": {"include_skills": True},
    }


def test_the_time_description_lists_exactly_the_forms_the_door_binds() -> None:
    description = SearchPart.model_json_schema()["properties"]["time"]["description"]

    assert TIME_FORMS == (
        "YYYY",
        "YYYY to YYYY",
        "since YYYY",
        "between YYYY and the latest",
        "latest",
    )
    assert all(f"'{form}'" in description for form in TIME_FORMS)


def test_to_wire_request_drops_none_fields() -> None:
    wire = to_wire_request([SearchPart(variable="malaria grants")], raw_query=None)

    assert wire["input"] == {"kind": "parsed", "parts": [{"variable_text": "malaria grants"}]}
    assert "raw_query" not in wire["input"]


def test_to_wire_request_carries_open_roles_as_given() -> None:
    parts = [
        SearchPart(
            variable="debt service",
            entities=[SearchEntity(text="Nigeria", role="debtor")],
            open_roles=["creditor"],
        )
    ]

    wire = to_wire_request(parts, raw_query=None)

    assert wire["input"]["parts"][0]["open_roles"] == ["creditor"]


def test_to_wire_request_entity_with_no_role_omits_role_text() -> None:
    parts = [SearchPart(variable="x", entities=[SearchEntity(text="Togo")])]

    wire = to_wire_request(parts, raw_query=None)

    assert wire["input"]["parts"][0]["entities"] == [{"text": "Togo"}]


@pytest.mark.parametrize(
    ("members", "expected"),
    [
        (None, {"text": "Africa"}),
        (False, {"text": "Africa"}),
        (True, {"text": "Africa", "members": True}),
    ],
)
def test_to_wire_request_sends_members_only_when_set(
    members: bool | None, expected: dict[str, object]
) -> None:
    parts = [SearchPart(variable="x", entities=[SearchEntity(text="Africa", members=members)])]

    wire = to_wire_request(parts, raw_query=None)

    assert wire["input"]["parts"][0]["entities"] == [expected]


def test_an_entity_with_an_unknown_field_is_refused() -> None:
    with pytest.raises(ValidationError) as refused:
        SearchEntity.model_validate({"text": "Africa", "each_member": True})

    assert [error["type"] for error in refused.value.errors()] == ["extra_forbidden"]


def test_to_wire_request_asks_for_skills_and_leaves_include_observations_unset() -> None:
    wire = to_wire_request([SearchPart(variable="x")], raw_query=None)

    assert wire["response_options"] == {"include_skills": True}


# The docstring's worked example


# The served `/api/search/schema` document, cut to the path the dataset list sits on.
# `provenance` is `ParsedPart`'s field as pydantic emits it, with the `oneOf` dfo-apis'
# `_with_declared_datasets` adds to its string branch, in rank order.
_SERVED_SCHEMA: dict[str, Any] = {
    "schema_version": 1,
    "request": {
        "$defs": {
            "ParsedPart": {
                "properties": {
                    "provenance": {
                        "anyOf": [
                            {
                                "maxLength": 200,
                                "minLength": 1,
                                "type": "string",
                                "oneOf": [
                                    {
                                        "const": "dataset/flows",
                                        "title": "Flows",
                                        "description": "Flows by donor and recipient.",
                                        "x-opt-in": False,
                                    },
                                    {
                                        "const": "dataset/stocks",
                                        "title": "Stocks",
                                        "description": "Stocks by debtor.",
                                        "x-opt-in": True,
                                    },
                                ],
                            },
                            {"type": "null"},
                        ],
                        "default": None,
                        "title": "Provenance",
                    }
                }
            }
        }
    },
    "response": {},
    "capabilities": "/api/search/capabilities",
}

_LISTED_DESCRIPTION = (
    "Base.\n\nSet a part's `dataset` to one of these dcids only when the user asks for "
    "that table:\n"
    "- `dataset/flows`: Flows. Flows by donor and recipient.\n"
    "- `dataset/stocks`: Stocks. Stocks by debtor. Served only when a part names it."
)


def test_describe_search_lists_the_served_datasets_in_rank_order() -> None:
    assert describe_search("Base.", declared_datasets(_SERVED_SCHEMA)) == _LISTED_DESCRIPTION


@pytest.mark.parametrize("datasets", [None, []])
def test_describe_search_without_datasets_says_to_leave_dataset_out(
    datasets: list[DeclaredDataset] | None,
) -> None:
    assert describe_search("Base.", datasets) == (
        "Base.\n\nNo datasets are listed, so leave `dataset` out."
    )


async def test_fetch_declared_datasets_reads_the_served_schema() -> None:
    paths: list[str] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        paths.append(request.url.path)
        return httpx2.Response(200, json=_SERVED_SCHEMA)

    async with _client(handler) as upstream:
        client_holder.set_client(upstream)
        datasets = await fetch_declared_datasets()

    assert paths == ["/api/search/schema"]
    assert datasets == declared_datasets(_SERVED_SCHEMA)


@pytest.mark.parametrize(
    "response",
    [
        httpx2.Response(
            503,
            content=_problem_body(code="upstream_unavailable", retryable=True),
            headers={"content-type": "application/problem+json"},
        ),
        httpx2.Response(200, content=b"not json"),
        httpx2.Response(200, json={"request": {"$defs": {}}}),
        httpx2.Response(
            200,
            json={
                "request": {
                    "$defs": {
                        "ParsedPart": {"properties": {"provenance": {"anyOf": [{"type": "null"}]}}}
                    }
                }
            },
        ),
    ],
    ids=["refused", "not-json", "no-parsed-part", "no-string-branch"],
)
async def test_fetch_declared_datasets_returns_none_when_the_schema_cannot_be_read(
    response: httpx2.Response,
) -> None:
    async with _client(lambda _request: response) as upstream:
        client_holder.set_client(upstream)
        assert await fetch_declared_datasets() is None


async def test_the_lifespan_lists_the_fetched_datasets_in_the_search_tool_description(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fetched() -> list[DeclaredDataset]:
        return declared_datasets(_SERVED_SCHEMA)

    monkeypatch.setattr(server._search, "description", server._search.description)
    monkeypatch.setattr(search_tool, "fetch_declared_datasets", fetched)

    async with Client(mcp) as client:
        tools = {tool.name: tool for tool in await client.list_tools()}

    assert tools["search"].description == describe_search(
        server._SEARCH_DESCRIPTION, declared_datasets(_SERVED_SCHEMA)
    )
    assert tools["search"].description.endswith(_LISTED_DESCRIPTION.removeprefix("Base."))


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


async def test_a_no_data_part_counts_the_stat_vars_it_checked_naming_none() -> None:
    """`elected_count` counts the flagged match's family. For "top donors to guatemala
    since 2020" Search checked 6,310 StatVars, which the digest counts without naming
    one."""
    envelope = _envelope(
        "no_data",
        no_data={
            "reason": "qualifier_not_bound",
            "category": "absent",
            "elected_count": 6_310,
            "flagged_match": _spec(spec_id="spec_flagged", member_count=6_310),
        },
    )

    result = await _call_search(_json_handler(200, envelope), [{"variable": "top donors"}])

    assert result.is_error is False
    assert result.meta is not None
    assert result.meta[SEARCH_ENVELOPE_META_KEY] == envelope
    assert "Checked 6,310 StatVars." in text_of(result)


# Refusal paths


async def test_transport_error_becomes_a_refusal() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("boom", request=request)

    result = await _call_search(handler, [{"variable": "malaria grants"}])

    assert result.is_error is True
    assert result.structured_content is not None
    assert result.structured_content["retry_mode"] == "retry_after_delay"


@pytest.mark.parametrize(
    ("attempt_cost_s", "attempts"),
    [(0.0, 2), (3.0, 1)],
    ids=["a fast 503 is retried", "a 503 that spent the budget is not"],
)
async def test_a_retry_starts_only_while_the_call_budget_has_time_left(
    monkeypatch: pytest.MonkeyPatch, attempt_cost_s: float, attempts: int
) -> None:
    clock = _FakeClock()
    monkeypatch.setattr(
        search_tool, "_call_deadline", lambda _budget_s: CallDeadline.after(2.5, clock=clock)
    )
    envelope = _envelope("no_data", no_data={"reason": "no_observations", "category": "absent"})
    calls = 0

    def handler(_request: httpx2.Request) -> httpx2.Response:
        nonlocal calls
        calls += 1
        clock.now += attempt_cost_s
        if calls == 1:
            return httpx2.Response(
                503, content=_problem_body(code="temporarily_unavailable", retryable=True)
            )
        return httpx2.Response(200, json=envelope)

    result = await _call_search(handler, [{"variable": "malaria grants"}])

    assert calls == attempts
    assert result.is_error is (attempts == 1)


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

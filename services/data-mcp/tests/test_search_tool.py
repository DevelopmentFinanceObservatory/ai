"""Tests for the `search` tool: translation, client wiring, and refusals."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Coroutine
from typing import Any

import httpx2
import pytest
from data_mcp import client_holder, search_tool
from data_mcp.deadline import CallDeadline
from data_mcp.digest import render_answer, render_digest
from data_mcp.result_bound import MAX_RESULT_BYTES
from data_mcp.search_result import SEARCH_ENVELOPE_META_KEY
from data_mcp.search_tool import (
    MAX_STAT_VAR_ITEMS,
    TIME_FORMS,
    SearchEntity,
    SearchPart,
    search,
    to_wire_request,
)
from data_mcp.server import mcp
from fastmcp import Client
from fastmcp.client.client import CallToolResult
from pydantic import TypeAdapter

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


@pytest.mark.parametrize(
    ("served", "listed"),
    [
        # A family Search states a member digest for keeps no items, and a
        # recall sample beside it keeps every member it lists.
        ([(1_454, True)], [0]),
        ([(1_454, True), (40, False)], [0, 40]),
        ([(1_454, True)] * 6, [0] * 6),
        # Listing exactly one, a digest family keeps it: it is the whole list.
        ([(1, True), (3, False)], [1, 3]),
        # Past the transport bound, the recall samples share it.
        ([(40, False)] * 6, [33, 33, 33, 33, 34, 34]),
        ([(MAX_STAT_VAR_ITEMS, False)], [MAX_STAT_VAR_ITEMS]),
    ],
)
async def test_each_spec_block_keeps_stat_var_items_by_whether_search_states_its_member_digest(
    served: list[tuple[int, bool]], listed: list[int]
) -> None:
    specs = []
    for block, (count, digested) in enumerate(served):
        spec = _spec(
            spec_id=f"spec_{block}",
            stat_vars=[
                {"ref": f"Amount_{block}_{index:05d}", "slot_values": []} for index in range(count)
            ],
        )
        spec["resolved"]["shape"] = {"member_count": count}
        if digested:
            spec["resolved"]["shape"]["member_digest"] = f"digest_{block}"
        spec["resolved"]["trace"] = {
            "resolved_stat_vars": [
                {"dcid": f"Amount_{block}_{index:05d}"} for index in range(min(count, 200))
            ],
            "resolved_stat_vars_truncated": count > 200,
        }
        specs.append(spec)
    candidates = {
        "ordering": "intent_rank",
        "max_candidates": 6,
        "specs": specs,
        "truncated": False,
    }
    handler = _json_handler(200, _envelope("candidates", candidates=candidates))

    result = await _call_search(handler, [{"variable": "aid"}])

    assert result.is_error is False
    assert result.meta is not None
    returned = result.meta[SEARCH_ENVELOPE_META_KEY]["parts"][0]["candidates"]["specs"]
    for block, (spec, (count, _), kept) in enumerate(zip(returned, served, listed, strict=True)):
        stat_vars = spec["resolved"]["stat_vars"]
        assert [item["ref"] for item in stat_vars["items"]] == [
            f"Amount_{block}_{index:05d}" for index in range(kept)
        ]
        assert stat_vars["member_count"] == count
        assert stat_vars["truncated"] is (kept < count)
        trace = spec["resolved"]["trace"]
        assert [entry["dcid"] for entry in trace["resolved_stat_vars"]] == [
            f"Amount_{block}_{index:05d}" for index in range(min(kept, 200))
        ]
        assert trace["resolved_stat_vars_truncated"] is (kept < count)


# A CRS StatVar dcid as Search names one, which `elected_stat_vars` repeats as its label.
_CRS_DCID = (
    "Amount_FinancialFlow_Gross_Disbursement_CRSTypeOfFlow10_CRSTypeOfFinance110_"
    "CRSPurposeCode{index:05d}_Nominal"
)


@pytest.mark.parametrize(
    ("digested", "elected"),
    [(True, 6_310), (False, 40)],
    ids=["digest-family", "recall-sample"],
)
async def test_a_no_data_part_counts_the_stat_vars_of_a_digest_family_it_checked(
    digested: bool, elected: int
) -> None:
    """`elected_stat_vars` names the flagged match's family. For "top donors to
    guatemala since 2020" Search lists all 6,310 members of a family with a member
    digest, 1.75 MB that would push the result past the 1 MiB bound; such a list
    becomes `elected_stat_var_count`. A recall sample, at most 40, keeps its list."""
    flagged = _spec(
        spec_id="spec_flagged",
        stat_vars=[{"ref": _CRS_DCID.format(index=index), "slot_values": []} for index in range(3)],
    )
    flagged["resolved"]["stat_vars"]["member_count"] = elected
    flagged["resolved"]["shape"] = {"member_count": elected}
    if digested:
        flagged["resolved"]["shape"]["member_digest"] = "d"
    listed = [
        {"dcid": _CRS_DCID.format(index=index), "label": _CRS_DCID.format(index=index)}
        for index in range(elected)
    ]
    envelope = _envelope(
        "no_data",
        no_data={
            "reason": "qualifier_not_bound",
            "category": "absent",
            "elected_stat_vars": listed,
            "flagged_match": flagged,
        },
    )

    result = await _call_search(_json_handler(200, envelope), [{"variable": "top donors"}])

    assert result.is_error is False
    assert result.meta is not None
    no_data = result.meta[SEARCH_ENVELOPE_META_KEY]["parts"][0]["no_data"]
    if digested:
        assert "elected_stat_vars" not in no_data
        assert no_data["elected_stat_var_count"] == elected
    else:
        assert no_data["elected_stat_vars"] == listed
        assert "elected_stat_var_count" not in no_data
    served = {"content": text_of(result), "meta": result.meta, "sc": result.structured_content}
    assert len(json.dumps(served).encode()) < MAX_RESULT_BYTES
    assert f"Checked {elected:,} StatVars." in text_of(result)
    assert "Amount_" not in text_of(result)


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
    monkeypatch.setattr(search_tool, "_call_deadline", lambda: CallDeadline.after(2.5, clock=clock))
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

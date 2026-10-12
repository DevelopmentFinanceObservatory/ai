"""Tests for `harness.mcp_client.McpResolver` and `McpExecutor`: every
`Client(...)` construction must go through `McpTimeoutPolicy`
(`harness.mcp_deadlines`) rather than FastMCP's own unbound defaults.

Each test aborts inside a patched `Client.__aenter__` -- after the real
`Client` is fully constructed, before any I/O -- and captures the
constructed instance, so the assertion reads the deadline off what the
resolver/executor actually built rather than off an argument passed in.
"""

from __future__ import annotations

import asyncio
import json
import random
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx2
import pytest
from common.auth import apply_gate
from common.caller import conversation_caller
from compiler.caller_clients import CallerClients
from compiler.surface import ProgressReport
from data_mcp import stat_var_list
from data_mcp.client import UpstreamClient
from data_mcp.digest import render_answer
from data_mcp.manifest import Manifest
from data_mcp.search_result import SEARCH_ENVELOPE_META_KEY
from data_mcp.settings import Settings as DataMcpSettings
from data_mcp.stat_var_list import STAT_VARS_OMITTED_KEY
from executor.classify import MAX_ATTEMPTS_PER_STEP
from fastmcp import Client, Context, FastMCP
from fastmcp.tools import ToolResult
from harness.conversation import current_conversation_id
from harness.mcp_client import (
    McpExecutor,
    McpResolver,
    _call_held,
    _call_tool,
    _manifest_summary,
    _McpTimeoutError,
    _refusal_outcome,
)
from harness.mcp_deadlines import McpTimeoutPolicy
from harness.provenance import ProvenanceStore
from harness.settings import Settings
from harness.tool_contracts import ResolverError, graph_envelope
from harness.wire import FactSummary, usable_component
from mcp import MCPError
from mcp.types import INVALID_PARAMS, REQUEST_TIMEOUT, ErrorData
from presentation_mcp.platform_contract import RESOLVED_PAYLOAD_META_KEY

from .resolved_revision import resolved_revision


class _StopAtConnectError(Exception):
    """Raised from the patched `Client.__aenter__` once the client is captured."""


@pytest.fixture
def captured_clients(monkeypatch: pytest.MonkeyPatch) -> list[Client]:
    captured: list[Client] = []

    async def _fake_aenter(self: Client) -> Client:
        captured.append(self)
        raise _StopAtConnectError

    monkeypatch.setattr(Client, "__aenter__", _fake_aenter)
    return captured


def _call_deadline_s(client: Client) -> float | None:
    return client._session_kwargs["read_timeout_seconds"]


async def test_the_resolver_search_client_carries_its_capability_row_deadline(
    captured_clients: list[Client],
) -> None:
    resolver = McpResolver(settings=Settings(mcp_init_timeout_s=7.0))

    with suppress(_StopAtConnectError):
        await resolver(
            question="how much debt does Kenya service",
            parts=[{"variable": "debt service"}],
        )

    (client,) = captured_clients
    assert _call_deadline_s(client) == 60.0  # data-mcp `search`'s pinned row
    assert client._init_timeout == 7.0


# --- _call_tool re-raises a request timeout so `_stream_fatal` can recognise it --


class _RaisingClient:
    """A `Client` stand-in whose `call_tool` always raises, for testing
    `_call_tool`'s timeout handling in isolation from the transport."""

    def __init__(self, error: MCPError) -> None:
        self._error = error

    async def call_tool(
        self,
        tool: str,
        arguments: dict[str, Any],
        *,
        meta: dict[str, str],
        progress_handler: Any,
        raise_on_error: bool,
    ) -> Any:
        raise self._error


async def test_call_tool_tags_a_request_timeout_with_its_tool_name() -> None:
    error = MCPError(code=REQUEST_TIMEOUT, message="Timed out")

    with pytest.raises(MCPError) as excinfo:
        await _call_tool(_RaisingClient(error), "search", {"raw_query": "kenya oda 2023"})  # type: ignore[arg-type]

    assert excinfo.value is not error  # re-raised as its own _McpTimeoutError, not mutated in place
    assert excinfo.value.error == error.error
    assert excinfo.value.timeout_tool == "search"  # type: ignore[attr-defined]
    assert excinfo.value.timeout_key == 'search:{"raw_query":"kenya oda 2023"}'  # type: ignore[attr-defined]


async def test_call_tool_keys_the_timeout_by_its_arguments_too() -> None:
    """Two different calls to the same tool must not collide in the
    conversation's repeat-timeout memory -- only the identical call again
    should."""
    same_a = MCPError(code=REQUEST_TIMEOUT, message="Timed out")
    same_b = MCPError(code=REQUEST_TIMEOUT, message="Timed out")
    different = MCPError(code=REQUEST_TIMEOUT, message="Timed out")

    with pytest.raises(MCPError) as first:
        await _call_tool(_RaisingClient(same_a), "search", {"raw_query": "kenya"})  # type: ignore[arg-type]
    with pytest.raises(MCPError) as second:
        await _call_tool(_RaisingClient(same_b), "search", {"raw_query": "kenya"})  # type: ignore[arg-type]
    with pytest.raises(MCPError) as third:
        await _call_tool(_RaisingClient(different), "search", {"raw_query": "mali"})  # type: ignore[arg-type]

    assert first.value.timeout_key == second.value.timeout_key  # type: ignore[attr-defined]
    assert first.value.timeout_key != third.value.timeout_key  # type: ignore[attr-defined]


async def test_call_tool_leaves_a_non_timeout_mcp_error_untouched() -> None:
    error = MCPError(code=INVALID_PARAMS, message="bad arguments")

    with pytest.raises(MCPError) as excinfo:
        await _call_tool(_RaisingClient(error), "search", {})  # type: ignore[arg-type]

    assert excinfo.value is error
    assert getattr(excinfo.value, "timeout_tool", None) is None


async def test_call_tool_tags_a_real_client_deadline_as_a_timeout() -> None:
    """The tests above construct the timeout error themselves; this one lets
    a real `Client` deadline elapse, so the code `_call_tool` matches on is
    the one the transport raises."""
    server = FastMCP(name="data-mcp")

    @server.tool(name="search")
    async def search() -> str:
        await asyncio.sleep(0.2)
        return "slow"

    async with Client(server, timeout=0.01) as client:
        with pytest.raises(MCPError) as excinfo:
            await _call_tool(client, "search", {})

    assert getattr(excinfo.value, "timeout_tool", None) == "search"


async def test_empty_parts_list_refuses_before_the_search_call(
    captured_clients: list[Client],
) -> None:
    """`search` must not be sent an empty `parts`, which its schema rejects
    with a raw MCP validation failure. The refusal carries a usable code
    instead, and fires before any client opens."""
    resolver = McpResolver(settings=Settings())

    with pytest.raises(ResolverError) as excinfo:
        await resolver(question="how much debt does Kenya service", parts=[])

    assert captured_clients == []
    assert excinfo.value.refusal.get("code") == "no_search_parts"


async def test_a_raw_text_search_sends_the_question_without_parts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`raw_text` sends `search` the question alone, so data-mcp reads it
    through Search's raw-text door instead of refusing for missing parts."""
    sent: list[dict[str, Any]] = []

    async def call_held(clients: Any, client: Any, tool: str, arguments: dict[str, Any], **_: Any):
        sent.append({"tool": tool, **arguments})
        return SimpleNamespace(is_error=True, structured_content={"code": "stop"}, meta=None)

    async def get() -> object:
        return object()

    async def release(client: object) -> None:
        return None

    monkeypatch.setattr("harness.mcp_client._call_held", call_held)
    resolver = McpResolver(settings=Settings())
    monkeypatch.setattr(resolver, "_clients", SimpleNamespace(get=get, release=release))

    with pytest.raises(ResolverError):
        await resolver(question="ODA to Kenya from all official donors", parts=[], raw_text=True)

    assert sent == [{"tool": "search", "raw_query": "ODA to Kenya from all official donors"}]


async def test_the_executor_get_data_client_carries_its_180s_deadline(
    captured_clients: list[Client],
) -> None:
    executor = McpExecutor(settings=Settings())

    with suppress(_StopAtConnectError):
        await executor(tool="get_data", args={"spec_id": "spec_" + "1" * 32})

    (client,) = captured_clients
    assert _call_deadline_s(client) == 180.0


async def test_the_executor_refuses_an_analysis_tool_as_unknown(
    captured_clients: list[Client],
) -> None:
    """Analysis verbs (`aggregate` and friends) reach analysis-mcp through the
    execute node's `ToolSurface`, not through `McpExecutor` -- this module
    never routes them. Asked for one anyway, it refuses cleanly instead of
    opening a client against a URL it no longer holds."""
    executor = McpExecutor(settings=Settings())

    result = await executor(tool="aggregate", args={"frame_ref": {}})

    assert captured_clients == []
    assert result == {
        "outcome": "refusal",
        "refusal": {
            "code": "unknown_tool",
            "retry_mode": "do_not_retry",
            "fix": None,
            "origin": "harness",
            "message": "the plan named a tool this executor does not route: 'aggregate'",
        },
    }


def test_a_platform_refusal_is_stamped_as_a_service_refusal_and_keeps_its_advice_text() -> None:
    """The violated constraint id lives only in presentation-mcp's display-text
    `fix`. It rides along as `advice`, while `fix` stays the mergeable mapping
    (empty here), so nothing that merges mapping fixes sees a string."""
    outcome = _refusal_outcome(
        {
            "error": "constraint_violation",
            "retry": "correct_and_retry",
            "detail": "the chart breaks a display rule",
            "fix": "violates constraint 'comparable_units_required'",
        }
    )

    assert outcome == {
        "outcome": "refusal",
        "refusal": {
            "code": "constraint_violation",
            "retry_mode": "correct_and_retry",
            "origin": "service",
            "fix": None,
            "advice": "violates constraint 'comparable_units_required'",
            "message": "the chart breaks a display rule",
        },
    }


def test_a_platform_trace_id_rides_on_the_refusal_when_the_body_carries_one() -> None:
    """presentation-mcp's wire body for a platform 500 keeps the platform's own
    trace id as `platform_trace_id`, and the executor's refusal carries it on."""
    refusal = _refusal_outcome(
        {
            "error": "internal_error",
            "retry": "do_not_retry",
            "detail": "Internal error",
            "platform_trace_id": "41d94ed2",
        }
    )["refusal"]

    assert refusal == {
        "code": "internal_error",
        "retry_mode": "do_not_retry",
        "origin": "service",
        "fix": None,
        "advice": None,
        "message": "Internal error",
        "platform_trace_id": "41d94ed2",
    }


@pytest.mark.parametrize("platform_trace_id", ["", None, 41, ["41d94ed2"]])
def test_an_unusable_platform_trace_id_is_left_off_the_refusal(platform_trace_id: object) -> None:
    refusal = _refusal_outcome(
        {"error": "internal_error", "retry": "do_not_retry", "platform_trace_id": platform_trace_id}
    )["refusal"]

    assert "platform_trace_id" not in refusal


def test_a_mapping_fix_stays_a_fix_and_carries_no_advice() -> None:
    fix = {"parameter": "entity_by", "columns": ["country"]}

    refusal = _refusal_outcome(
        {"code": "role_ambiguous", "retry_mode": "correct_and_retry", "fix": fix, "message": "m"}
    )["refusal"]

    assert refusal["fix"] == fix
    assert refusal["advice"] is None
    assert refusal["origin"] == "service"


def test_a_refusal_with_no_structured_body_is_still_a_service_refusal() -> None:
    refusal = _refusal_outcome(None, fallback_text="argument validation failed")["refusal"]

    assert refusal["code"] == "unspecified"
    assert refusal["origin"] == "service"
    assert refusal["message"] == "argument validation failed"


@pytest.fixture
def scripted_presentation_client(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[list[Any]], _ScriptedMcpClient]:
    """Makes the executor's presentation client a `_ScriptedMcpClient` that
    answers `script`, one result per call, and returns it."""

    def _install(script: list[Any]) -> _ScriptedMcpClient:
        client = _ScriptedMcpClient(script)
        monkeypatch.setattr(
            McpTimeoutPolicy, "client", lambda self, url, *, auth, caller, tool=None: client
        )
        return client

    return _install


def _tool_result(body: dict[str, Any], *, meta: dict[str, Any] | None = None) -> SimpleNamespace:
    return SimpleNamespace(is_error=False, structured_content=body, content=[], meta=meta)


def _write_result(envelope: dict[str, Any] | None) -> SimpleNamespace:
    """A `write_component` result, with the model vocabulary in structured
    content and the resolved revision (when given) in `_meta`."""
    return _tool_result(
        {"component_id": "cmp_1", "revision": 1, "fact_summary": {}},
        meta=None if envelope is None else {RESOLVED_PAYLOAD_META_KEY: envelope},
    )


async def test_the_executor_puts_the_resolved_revision_in_the_payload_verbatim(
    scripted_presentation_client: Callable[[list[Any]], _ScriptedMcpClient],
) -> None:
    """A revise patch names neither type nor frame. The envelope answers
    both, becomes the wire `payload` unchanged, and no second presentation
    call is made."""
    frame_ref = {"spec_id": "spec_" + "1" * 32, "content_version": "cv_" + "1" * 32}
    envelope = resolved_revision(
        "line_chart", component_id="cmp_1", revision=1, frame_ref=frame_ref
    )
    client = scripted_presentation_client([_write_result(envelope)])
    store = ProvenanceStore()
    store.put(conversation=current_conversation_id.get(), manifest={}, **frame_ref)
    executor = McpExecutor(settings=Settings(), provenance=store)

    result = await executor(
        tool="write_component", args={"component_id": "cmp_1", "intent": {"scale": "log"}}
    )

    assert client.calls == ["write_component"]
    assert result["outcome"] == "ok"
    component = result["component"]
    assert component["payload"] == resolved_revision(
        "line_chart", component_id="cmp_1", revision=1, frame_ref=frame_ref
    )
    assert component["type"] == "line_chart"
    assert component["catalog_version"] == envelope["catalog_version"]
    assert component["provenance"]["inputs"][0]["spec_id"] == frame_ref["spec_id"]
    assert result["summary"] == "line_chart: Malaria grant disbursements to Ethiopia"


@pytest.mark.parametrize(
    ("snapshot", "stored_refs"),
    [
        (
            httpx2.Response(
                200, json={"trace": {"resolved_stat_vars": [{"ref": "dcid:A"}, {"ref": "dcid:B"}]}}
            ),
            ["dcid:A", "dcid:B"],
        ),
        # A list that cannot be read back leaves the manifest as it came.
        (httpx2.Response(404, json={"code": "not_found"}), None),
    ],
)
async def test_the_executor_stores_a_retrievals_stat_var_list_the_result_left_out(
    scripted_presentation_client: Callable[[list[Any]], _ScriptedMcpClient],
    monkeypatch: pytest.MonkeyPatch,
    snapshot: httpx2.Response,
    stored_refs: list[str] | None,
) -> None:
    frame_ref = {"spec_id": "spec_" + "1" * 32, "content_version": "cv_" + "1" * 32}
    result = {
        "definition": {"spec_id": frame_ref["spec_id"]},
        "content_version": frame_ref["content_version"],
        "trace": {},
        "frame_ref": frame_ref,
        STAT_VARS_OMITTED_KEY: 2,
    }
    scripted_presentation_client([_tool_result(result)])
    upstream = UpstreamClient(
        settings=DataMcpSettings(upstream_max_attempts=1),
        transport=httpx2.MockTransport(lambda request: snapshot),
    )
    monkeypatch.setattr(stat_var_list, "get_client", lambda: upstream)
    store = ProvenanceStore()
    executor = McpExecutor(settings=Settings(), provenance=store)

    outcome = await executor(tool="get_data", args={"spec_id": frame_ref["spec_id"]})

    assert outcome["outcome"] == "ok"
    lineage = store.get(conversation=current_conversation_id.get(), **frame_ref)
    assert lineage is not None
    stored = lineage[0].manifest
    listed = stored["trace"].get("resolved_stat_vars")
    assert (None if listed is None else [entry["ref"] for entry in listed]) == stored_refs


def _period_mode_bar_chart(frame_ref: dict[str, Any]) -> dict[str, Any]:
    envelope = resolved_revision("bar_chart", component_id="cmp_1", frame_ref=frame_ref)
    envelope["component"]["category_axis"] = {
        "mode": "period",
        "window": {"start": "2015", "end": "2024"},
    }
    return envelope


def _span_bar_chart(frame_ref: dict[str, Any]) -> dict[str, Any]:
    envelope = resolved_revision("bar_chart", component_id="cmp_1", frame_ref=frame_ref)
    envelope["component"]["category_axis"]["period"] = "2018/2022"
    return envelope


@pytest.mark.parametrize(
    ("envelope_for", "platform_window", "window"),
    [
        pytest.param(
            lambda ref: resolved_revision("bar_chart", component_id="cmp_1", frame_ref=ref),
            None,
            {"start": "2024", "end": "2024"},
            id="entity-bar-draws-its-period",
        ),
        pytest.param(
            _period_mode_bar_chart, None, {"start": "2015", "end": "2024"}, id="period-mode-bar"
        ),
        pytest.param(
            _span_bar_chart, None, {"start": "2018", "end": "2022"}, id="entity-bar-over-a-span"
        ),
        pytest.param(
            lambda ref: resolved_revision("line_chart", component_id="cmp_1", frame_ref=ref),
            None,
            {"start": "2015", "end": "2024"},
            id="line-chart",
        ),
        pytest.param(
            lambda ref: resolved_revision("line_chart", component_id="cmp_1", frame_ref=ref),
            {"start": "2016", "end": "2023"},
            {"start": "2016", "end": "2023"},
            id="the-platforms-window-wins",
        ),
        pytest.param(
            lambda ref: resolved_revision("table", component_id="cmp_1", frame_ref=ref),
            None,
            None,
            id="table-has-none",
        ),
        pytest.param(
            lambda ref: resolved_revision("stat_card", component_id="cmp_1", frame_ref=ref),
            None,
            None,
            id="stat-card-has-none",
        ),
    ],
)
async def test_the_wire_window_is_the_platforms_or_the_periods_the_chart_draws(
    scripted_presentation_client: Callable[[list[Any]], _ScriptedMcpClient],
    envelope_for: Callable[[dict[str, Any]], dict[str, Any]],
    platform_window: dict[str, str] | None,
    window: dict[str, str] | None,
) -> None:
    """The platform states no window for a bar chart of entities at one
    period. The wire fact summary then reads one from the chart's own axis,
    so a client that plots from the fact summary has the period to plot. A
    window the platform states is kept, and a card with no axis has none."""
    frame_ref = {"spec_id": "spec_" + "1" * 32, "content_version": "cv_" + "1" * 32}
    written = _write_result(envelope_for(frame_ref))
    if platform_window is not None:
        written.structured_content["fact_summary"] = {"window": platform_window}
    scripted_presentation_client([written])
    store = ProvenanceStore()
    store.put(conversation=current_conversation_id.get(), manifest={}, **frame_ref)
    executor = McpExecutor(settings=Settings(), provenance=store)

    result = await executor(tool="write_component", args={"intent": {"type": "line_chart"}})

    assert result["component"]["fact_summary"].get("window") == window


async def test_a_card_on_a_retrieval_read_with_place_types_lists_the_pair_dfo_apis_serves(
    scripted_presentation_client: Callable[[list[Any]], _ScriptedMcpClient],
) -> None:
    """A retrieval read with place types is held under a `content_version` of its own,
    which the datasets door does not serve. The card lists the pair its manifest was
    minted as."""
    minted = {"spec_id": "spec_" + "1" * 32, "content_version": "cv_" + "1" * 32}
    typed = {"spec_id": minted["spec_id"], "content_version": "cv_" + "2" * 32}
    envelope = resolved_revision("line_chart", component_id="cmp_1", revision=1, frame_ref=typed)
    scripted_presentation_client([_write_result(envelope)])
    store = ProvenanceStore()
    store.put(
        conversation=current_conversation_id.get(),
        manifest={
            "definition": {"spec_id": minted["spec_id"]},
            "content_version": minted["content_version"],
        },
        **typed,
    )
    executor = McpExecutor(settings=Settings(), provenance=store)

    result = await executor(tool="write_component", args={"component_id": "cmp_1", "intent": {}})

    assert result["outcome"] == "ok"
    assert result["component"]["provenance"]["inputs"] == [minted]
    assert result["component"]["payload"]["provenance"]["inputs"] == [typed]


async def test_a_card_on_a_fetched_place_types_retrieval_is_drawn_on_the_pair_dfo_apis_serves(
    scripted_presentation_client: Callable[[list[Any]], _ScriptedMcpClient],
) -> None:
    """`get_data` names a retrieval read with place types by its manifest's own pair,
    and a card drawn on that pair finds the retrieval's lineage."""
    minted = {"spec_id": "spec_" + "1" * 32, "content_version": "cv_" + "1" * 32}
    typed = {"spec_id": minted["spec_id"], "content_version": "cv_" + "2" * 32}
    manifest = {
        "definition": {"spec_id": minted["spec_id"]},
        "content_version": minted["content_version"],
        "trace": {},
        "frame_ref": typed,
    }
    envelope = resolved_revision("line_chart", component_id="cmp_1", revision=1, frame_ref=minted)
    scripted_presentation_client([_tool_result(manifest), _write_result(envelope)])
    store = ProvenanceStore()
    executor = McpExecutor(settings=Settings(), provenance=store)

    fetched = await executor(tool="get_data", args={"spec_id": minted["spec_id"]})
    drawn = await executor(tool="write_component", args={"component_id": "cmp_1", "intent": {}})

    assert fetched["frame_refs"] == [minted]
    assert drawn["outcome"] == "ok", drawn
    assert drawn["component"]["provenance"]["inputs"] == [minted]
    conversation = current_conversation_id.get()
    assert store.get(conversation=conversation, **typed) == store.get(
        conversation=conversation, **minted
    )


async def test_the_executor_moves_a_creates_write_token_off_the_result(
    scripted_presentation_client: Callable[[list[Any]], _ScriptedMcpClient],
) -> None:
    """The token rides beside the component on the outcome, for the
    recorder to store, and is gone from the result's structured content,
    its content text and the component."""
    frame_ref = {"spec_id": "spec_" + "1" * 32, "content_version": "cv_" + "1" * 32}
    envelope = resolved_revision("line_chart", component_id="cmp_1", frame_ref=frame_ref)
    written = _write_result(envelope)
    written.structured_content["write_token"] = "cmw_secret"
    vocabulary = SimpleNamespace(type="text", text='{"component_id":"cmp_1"}')
    written.content = [
        vocabulary,
        SimpleNamespace(type="text", text="Keep this write_token.\nwrite_token: cmw_secret"),
    ]
    scripted_presentation_client([written])
    store = ProvenanceStore()
    store.put(conversation=current_conversation_id.get(), manifest={}, **frame_ref)
    executor = McpExecutor(settings=Settings(), provenance=store)

    result = await executor(tool="write_component", args={"intent": {"type": "line_chart"}})

    assert result["write_token"] == "cmw_secret"
    assert "write_token" not in written.structured_content
    assert written.content == [vocabulary]
    assert "cmw_" not in json.dumps(result["component"])


def test_a_timeout_key_leaves_the_write_token_out() -> None:
    error = _McpTimeoutError(
        ErrorData(code=REQUEST_TIMEOUT, message="timed out"),
        tool="write_component",
        arguments={"component_id": "cmp_1", "write_token": "cmw_secret"},
    )

    assert "cmw_" not in error.timeout_key


async def test_the_executor_asks_presentation_mcp_for_no_text_projection_on_every_write(
    scripted_presentation_client: Callable[[list[Any]], _ScriptedMcpClient],
) -> None:
    """The event stream carries the render, so every write, create or
    revise, sends `include_projection: none` beside the caller's own args."""
    client = scripted_presentation_client([_write_result(None), _write_result(None)])
    executor = McpExecutor(settings=Settings())
    create = {"intent": {"type": "line_chart", "title": "t"}}
    revise = {"component_id": "cmp_1", "intent": {"scale": "log"}}

    await executor(tool="write_component", args=create)
    await executor(tool="write_component", args=revise)

    assert client.arguments == [
        {**create, "include_projection": "none"},
        {**revise, "include_projection": "none"},
    ]
    assert "include_projection" not in create


async def test_the_executor_asks_presentation_mcp_for_no_projection_or_image_on_a_describe(
    scripted_presentation_client: Callable[[list[Any]], _ScriptedMcpClient],
) -> None:
    """The harness reads a describe's structured content alone, so the text
    lead and the chart image presentation-mcp sends its undeclared client
    are switched off, as they are for a write."""
    client = scripted_presentation_client([_write_result(None)])
    executor = McpExecutor(settings=Settings())
    args = {"component_id": "cmp_1"}

    await executor(tool="describe_component", args=args)

    assert client.arguments == [{**args, "include_projection": "none"}]
    assert "include_projection" not in args


async def test_a_revise_is_held_to_the_basis_the_stored_revision_resolved(
    scripted_presentation_client: Callable[[list[Any]], _ScriptedMcpClient],
) -> None:
    """`describe_component` carries the stored revision's resolved payload in
    `_meta`, and its `change.basis` labels are the periods a title may not
    name when the stored intent states that basis against the latest data."""
    frame_ref = {"spec_id": "spec_" + "1" * 32, "content_version": "cv_" + "1" * 32}
    envelope = resolved_revision("stat_card", component_id="cmp_1", frame_ref=frame_ref)
    latest = envelope["component"]["change"]["basis"][1]
    relative = [{"anchor": "latest", "offset": -1}, {"anchor": "latest", "offset": 0}]
    stored = _tool_result(
        {
            "component_id": "cmp_1",
            "revision": 1,
            "stored_intent": {
                "type": "stat_card",
                "frame_ref": frame_ref,
                "change": {"basis": relative},
            },
        },
        meta={RESOLVED_PAYLOAD_META_KEY: envelope},
    )
    client = scripted_presentation_client([stored])
    executor = McpExecutor(settings=Settings())

    result = await executor(
        tool="write_component",
        args={"component_id": "cmp_1", "intent": {"title": f"Aid in {latest[:4]}"}},
    )

    assert client.calls == ["describe_component"]
    assert result["refusal"]["code"] == "title_quotes_rolling_period"


async def test_a_revise_stating_a_new_basis_is_held_to_the_basis_it_resolved(
    scripted_presentation_client: Callable[[list[Any]], _ScriptedMcpClient],
) -> None:
    """A patch that replaces `change` with a relative basis keeps the stored
    title, and is refused after the write when that title names a period the
    new revision resolved."""
    frame_ref = {"spec_id": "spec_" + "1" * 32, "content_version": "cv_" + "1" * 32}
    envelope = resolved_revision(
        "stat_card", component_id="cmp_1", frame_ref=frame_ref, title="Aid in 2024"
    )
    stored = _tool_result(
        {
            "component_id": "cmp_1",
            "revision": 1,
            "stored_intent": {"type": "stat_card", "frame_ref": frame_ref, "title": "Aid in 2024"},
        }
    )
    scripted_presentation_client([stored, _write_result(envelope)])
    store = ProvenanceStore()
    store.put(conversation=current_conversation_id.get(), manifest={}, **frame_ref)
    executor = McpExecutor(settings=Settings(), provenance=store)
    relative = [{"anchor": "latest", "offset": -1}, {"anchor": "latest", "offset": 0}]

    result = await executor(
        tool="write_component",
        args={"component_id": "cmp_1", "intent": {"change": {"basis": relative}}},
    )

    assert result["outcome"] == "refusal"
    assert result["refusal"]["fix"] == {"period": "2024"}


async def test_a_revise_changing_only_the_select_is_held_to_the_window_it_resolved(
    scripted_presentation_client: Callable[[list[Any]], _ScriptedMcpClient],
) -> None:
    """Another select can move the latest period of a card whose window is
    relative, so the stored title is judged after the write against the window
    the new revision resolved, which the stored one did not reach."""
    frame_ref = {"spec_id": "spec_" + "1" * 32, "content_version": "cv_" + "1" * 32}
    envelope = resolved_revision("line_chart", component_id="cmp_1", frame_ref=frame_ref)
    relative = {
        "start": {"anchor": "latest", "offset": -5},
        "end": {"anchor": "latest", "offset": 0},
    }
    stored = _tool_result(
        {
            "component_id": "cmp_1",
            "revision": 1,
            "stored_intent": {
                "type": "line_chart",
                "frame_ref": frame_ref,
                "title": "Aid to 2023",
                "window": relative,
            },
            "fact_summary": {"window": {"start": "2017", "end": "2022"}},
        },
        meta={RESOLVED_PAYLOAD_META_KEY: envelope},
    )
    written = _write_result(envelope)
    written.structured_content["fact_summary"] = {"window": {"start": "2018", "end": "2023"}}
    client = scripted_presentation_client([stored, written])
    store = ProvenanceStore()
    store.put(conversation=current_conversation_id.get(), manifest={}, **frame_ref)
    executor = McpExecutor(settings=Settings(), provenance=store)

    result = await executor(
        tool="write_component",
        args={"component_id": "cmp_1", "intent": {"select": {"reporter": "FRA"}}},
    )

    assert client.calls == ["describe_component", "write_component"]
    assert result["outcome"] == "refusal"
    assert result["refusal"]["fix"] == {"period": "2023"}


async def test_the_executor_refuses_a_write_whose_result_carries_no_resolved_revision(
    scripted_presentation_client: Callable[[list[Any]], _ScriptedMcpClient],
) -> None:
    """A platform that stores the card but sends no envelope leaves nothing
    to draw, so the card is withheld under the harness's own code."""
    client = scripted_presentation_client([_write_result(None)])
    executor = McpExecutor(settings=Settings())

    result = await executor(
        tool="write_component", args={"intent": {"type": "line_chart", "title": "t"}}
    )

    assert client.calls == ["write_component"]
    assert result["outcome"] == "refusal"
    refusal = result["refusal"]
    assert refusal["code"] == "resolved_revision_unavailable"
    assert refusal["origin"] == "harness"
    assert refusal["retry_mode"] == "do_not_retry"


async def test_the_executor_withholds_a_card_with_no_linked_lineage_as_a_harness_refusal(
    scripted_presentation_client: Callable[[list[Any]], _ScriptedMcpClient],
) -> None:
    """The platform accepts the write, but no source manifest is linked to the
    frame it names, so the card is withheld. The refusal is the harness's own
    decision and carries origin `harness`."""
    frame_ref = {"spec_id": "spec_" + "1" * 32, "content_version": "cv_" + "1" * 32}
    client = scripted_presentation_client(
        [_write_result(resolved_revision("line_chart", frame_ref=frame_ref))]
    )
    executor = McpExecutor(settings=Settings())  # an empty `ProvenanceStore`

    result = await executor(
        tool="write_component",
        args={"intent": {"type": "line_chart", "title": "t", "frame_ref": frame_ref}},
    )

    assert client.calls == ["write_component"]
    assert result["outcome"] == "refusal"
    refusal = result["refusal"]
    assert refusal["code"] == "provenance_unavailable"
    assert refusal["origin"] == "harness"
    assert refusal["retry_mode"] == "do_not_retry"


async def test_the_executor_relays_a_platform_write_refusal_as_a_service_refusal(
    scripted_presentation_client: Callable[[list[Any]], _ScriptedMcpClient],
) -> None:
    scripted_presentation_client(
        [
            _tool_result(
                {
                    "error": "constraint_violation",
                    "retry": "correct_and_retry",
                    "detail": "the chart breaks a display rule",
                    "fix": "violates constraint 'comparable_units_required'",
                }
            )
        ]
    )
    executor = McpExecutor(settings=Settings())

    result = await executor(tool="write_component", args={"intent": {"type": "line_chart"}})

    refusal = result["refusal"]
    assert refusal["code"] == "constraint_violation"
    assert refusal["origin"] == "service"
    assert "comparable_units_required" in refusal["advice"]


async def test_the_executor_presentation_client_carries_a_finite_call_and_init_deadline(
    captured_clients: list[Client],
) -> None:
    executor = McpExecutor(settings=Settings(mcp_call_timeout_s=42.0, mcp_init_timeout_s=7.0))

    with suppress(_StopAtConnectError):
        await executor(
            tool="describe_frame",
            args={"frame_ref": {"spec_id": "s", "content_version": "c"}},
        )

    (client,) = captured_clients
    assert _call_deadline_s(client) == 42.0
    assert client._init_timeout == 7.0


async def test_the_indicator_describe_client_carries_its_30s_deadline(
    captured_clients: list[Client],
) -> None:
    """`_indicators_for`'s own `describe` lookup for the More Info panel, a
    second call site distinct from `_get_data`'s. It catches every exception
    from the client (a describe outage degrades to a fallback name), so the
    connect abort never escapes -- the assertion reads the capture list,
    not an exception."""
    executor = McpExecutor(settings=Settings())

    await executor._indicators_for(["dcid:LifeExpectancy_Person"])

    (client,) = captured_clients
    assert _call_deadline_s(client) == 30.0


# --- a held client is built once per conversation, not once per call -- its --
# --- factory dropping `auth=` would otherwise go unnoticed until every later -
# --- call 401s in production, since nothing here mocks the transport away. ---


def _bearer_token_on(client: Client) -> str | None:
    auth = client.transport.auth
    return auth.token.get_secret_value() if auth is not None else None


async def test_the_resolvers_held_search_client_carries_the_bearer_token(
    captured_clients: list[Client],
) -> None:
    apply_gate(FastMCP(), token="s3cret-token", disabled=False)
    resolver = McpResolver(settings=Settings())

    with suppress(_StopAtConnectError):
        await resolver(question="q", parts=[{"variable": "x"}])

    (client,) = captured_clients
    assert _bearer_token_on(client) == "s3cret-token"


async def test_the_executors_held_get_data_client_carries_the_bearer_token(
    captured_clients: list[Client],
) -> None:
    apply_gate(FastMCP(), token="s3cret-token", disabled=False)
    executor = McpExecutor(settings=Settings())

    with suppress(_StopAtConnectError):
        await executor(tool="get_data", args={"spec_id": "spec_" + "1" * 32})

    (client,) = captured_clients
    assert _bearer_token_on(client) == "s3cret-token"


async def test_the_executors_held_presentation_client_carries_the_bearer_token(
    captured_clients: list[Client],
) -> None:
    apply_gate(FastMCP(), token="s3cret-token", disabled=False)
    executor = McpExecutor(settings=Settings())

    with suppress(_StopAtConnectError):
        await executor(
            tool="describe_frame",
            args={"frame_ref": {"spec_id": "s", "content_version": "c"}},
        )

    (client,) = captured_clients
    assert _bearer_token_on(client) == "s3cret-token"


async def test_the_indicator_describe_clients_held_client_carries_the_bearer_token(
    captured_clients: list[Client],
) -> None:
    apply_gate(FastMCP(), token="s3cret-token", disabled=False)
    executor = McpExecutor(settings=Settings())

    await executor._indicators_for(["dcid:LifeExpectancy_Person"])

    (client,) = captured_clients
    assert _bearer_token_on(client) == "s3cret-token"


# --- client reuse: one held client per conversation, not one per call -------


class _FakeMcpClient:
    """A `Client` stand-in that never touches the network: `call_tool`
    answers a canned refusal (so callers take their error branch without
    needing a real envelope/manifest shape), and `__aenter__`/`__aexit__`
    just count."""

    def __init__(self, caller: str | None = None) -> None:
        self.caller = caller
        self.entered = 0
        self.exited = 0
        self.calls: list[str] = []

    async def __aenter__(self) -> _FakeMcpClient:
        self.entered += 1
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        self.exited += 1

    async def call_tool(
        self,
        tool: str,
        arguments: dict[str, Any],
        *,
        meta: dict[str, str],
        progress_handler: Any,
        raise_on_error: bool,
    ) -> Any:
        self.calls.append(tool)
        return SimpleNamespace(is_error=True, structured_content={"code": "refused"}, content=[])


@pytest.fixture
def fake_policy_clients(monkeypatch: pytest.MonkeyPatch) -> list[_FakeMcpClient]:
    """Every `McpTimeoutPolicy.client(...)` call builds and returns a new
    `_FakeMcpClient`, appended here -- so a test can assert how many times
    the policy's own factory ran, which is exactly what `CallerClients`
    is supposed to bound to one per conversation."""
    built: list[_FakeMcpClient] = []

    def _fake_client(
        self: McpTimeoutPolicy,
        url: str,
        *,
        auth: str | None,
        caller: str | None,
        tool: str | None = None,
    ) -> Any:
        client = _FakeMcpClient(caller)
        built.append(client)
        return client

    monkeypatch.setattr(McpTimeoutPolicy, "client", _fake_client)
    return built


async def test_the_resolver_reuses_one_search_client_across_calls(
    fake_policy_clients: list[_FakeMcpClient],
) -> None:
    resolver = McpResolver(settings=Settings())

    for _ in range(2):
        with pytest.raises(ResolverError):
            await resolver(question="q", parts=[{"variable": "x"}])

    (client,) = fake_policy_clients  # the factory ran once, not once per call
    assert client.entered == 1
    assert client.calls == ["search", "search"]


async def test_the_executor_reuses_one_get_data_client_across_calls(
    fake_policy_clients: list[_FakeMcpClient],
) -> None:
    executor = McpExecutor(settings=Settings())

    for _ in range(2):
        await executor(tool="get_data", args={"spec_id": "spec_" + "1" * 32})

    (client,) = fake_policy_clients
    assert client.entered == 1
    assert client.calls == ["get_data", "get_data"]


async def test_two_conversations_get_two_separate_get_data_clients(
    fake_policy_clients: list[_FakeMcpClient],
) -> None:
    """Client reuse is per conversation, not process-wide: two conversations
    calling `get_data` must not share one client, and each client names its
    own conversation as the caller."""
    executor = McpExecutor(settings=Settings())

    current_conversation_id.set("conv-a")
    await executor(tool="get_data", args={"spec_id": "spec_" + "1" * 32})
    current_conversation_id.set("conv-b")
    await executor(tool="get_data", args={"spec_id": "spec_" + "2" * 32})

    assert len(fake_policy_clients) == 2
    assert fake_policy_clients[0] is not fake_policy_clients[1]
    assert fake_policy_clients[0].calls == ["get_data"]
    assert fake_policy_clients[1].calls == ["get_data"]
    assert [client.caller for client in fake_policy_clients] == [
        conversation_caller("conv-a"),
        conversation_caller("conv-b"),
    ]


async def test_a_call_outside_any_conversation_names_no_caller(
    fake_policy_clients: list[_FakeMcpClient],
) -> None:
    await McpExecutor(settings=Settings())(tool="get_data", args={"spec_id": "spec_" + "1" * 32})

    (client,) = fake_policy_clients
    assert client.caller is None


# --- `_call_held`: a broken session is discarded, a slow call is not -------


class _ScriptedMcpClient:
    """A `Client` stand-in whose `call_tool` runs through a scripted list of
    outcomes, one per call: an exception is raised, anything else is
    returned. `__aenter__`/`__aexit__` just count."""

    def __init__(self, script: list[Any]) -> None:
        self._script = list(script)
        self.entered = 0
        self.exited = 0
        self.calls: list[str] = []
        self.arguments: list[dict[str, Any]] = []

    async def __aenter__(self) -> _ScriptedMcpClient:
        self.entered += 1
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        self.exited += 1

    async def call_tool(
        self,
        tool: str,
        arguments: dict[str, Any],
        *,
        meta: dict[str, str],
        progress_handler: Any,
        raise_on_error: bool,
    ) -> Any:
        self.calls.append(tool)
        self.arguments.append(arguments)
        outcome = self._script.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _held_clients(factory: Callable[[], Any]) -> CallerClients:
    """A registry over `factory`'s stand-in clients, for a caller that never changes."""
    return CallerClients(lambda _caller: factory(), caller=lambda: None)


async def test_call_held_keeps_a_timed_out_client_cached() -> None:
    built: list[_ScriptedMcpClient] = []

    def factory() -> _ScriptedMcpClient:
        client = _ScriptedMcpClient([MCPError(code=REQUEST_TIMEOUT, message="slow")])
        built.append(client)
        return client

    clients = _held_clients(factory)
    first = await clients.get()

    with pytest.raises(MCPError):
        await _call_held(clients, first, "search", {})

    # the timeout is this one call's problem, not the session's -- still cached
    assert await clients.get() is first
    assert len(built) == 1


async def test_call_held_discards_a_client_that_failed_for_another_reason() -> None:
    built: list[_ScriptedMcpClient] = []

    def factory() -> _ScriptedMcpClient:
        client = _ScriptedMcpClient([RuntimeError("transport closed")])
        built.append(client)
        return client

    clients = _held_clients(factory)
    first = await clients.get()

    with pytest.raises(RuntimeError):
        await _call_held(clients, first, "search", {})
    # `_call_held` only marks a failed client for eviction; the caller that
    # fetched it -- this test, standing in for `McpResolver`/`McpExecutor`'s
    # own `finally` -- releases it, which is what actually disconnects it
    # once nothing else is still using it. See `CallerClients.release`.
    await clients.release(first)

    second = await clients.get()  # the broken client was discarded and replaced
    assert second is not first
    assert first.exited == 1
    assert len(built) == 2


async def test_a_non_timeout_failure_does_not_tear_down_a_concurrent_call_on_the_same_client() -> (
    None
):
    """Two calls sharing one conversation's held client, in flight at once:
    `describe` fails for a reason other than its own timeout while `search`
    is still awaiting its own `call_tool`. `_call_held`'s eviction of the
    broken client (`CallerClients.discard`) must not disconnect it
    until `search` has released its own hold -- ADK can dispatch a turn's
    tool calls concurrently, so one call failing must not break a sibling
    still using the same conversation's client. Asserts `exited` itself,
    not just that `search` still gets an answer: a `discard` that tore the
    client down immediately would still let a `call_tool` already past its
    `await` return normally on a fake client with no transport underneath
    it to actually break, so the result alone does not prove teardown was
    deferred -- only watching `exited` does."""
    search_started = asyncio.Event()
    release_search = asyncio.Event()

    class _RacingClient:
        def __init__(self) -> None:
            self.exited = 0

        async def __aenter__(self) -> _RacingClient:
            return self

        async def __aexit__(self, *exc_info: object) -> None:
            self.exited += 1

        async def call_tool(
            self,
            tool: str,
            arguments: dict[str, Any],
            *,
            meta: dict[str, str],
            progress_handler: Any,
            raise_on_error: bool,
        ) -> Any:
            if tool == "search":
                search_started.set()
                await release_search.wait()
                return SimpleNamespace(is_error=False, structured_content={"ok": True}, content=[])
            await search_started.wait()
            raise RuntimeError("transport closed")

    built: list[_RacingClient] = []

    def factory() -> _RacingClient:
        client = _RacingClient()
        built.append(client)
        return client

    clients = _held_clients(factory)

    async def _run_search() -> Any:
        client = await clients.get()
        try:
            return await _call_held(clients, client, "search", {})
        finally:
            await clients.release(client)

    async def _run_describe() -> None:
        client = await clients.get()
        try:
            with pytest.raises(RuntimeError):
                await _call_held(clients, client, "describe", {})
        finally:
            await clients.release(client)

    search_task = asyncio.create_task(_run_search())
    describe_task = asyncio.create_task(_run_describe())

    await describe_task
    # `describe` failed and released its own hold, but `search` is still
    # mid-call on the same client -- the eviction must not have torn it
    # down yet.
    assert built[0].exited == 0

    release_search.set()
    result = await asyncio.wait_for(search_task, timeout=2)

    assert result.structured_content == {"ok": True}
    assert len(built) == 1  # both calls shared the one held client
    assert built[0].exited == 1  # search's own release was the last holder, and tore it down


async def test_the_resolver_keeps_its_client_after_a_read_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built: list[_ScriptedMcpClient] = []

    def _fake_client(
        self: McpTimeoutPolicy,
        url: str,
        *,
        auth: str | None,
        caller: str | None,
        tool: str | None = None,
    ) -> Any:
        client = _ScriptedMcpClient(
            [
                MCPError(code=REQUEST_TIMEOUT, message="slow"),
                SimpleNamespace(is_error=True, structured_content={"code": "refused"}, content=[]),
            ]
        )
        built.append(client)
        return client

    monkeypatch.setattr(McpTimeoutPolicy, "client", _fake_client)
    resolver = McpResolver(settings=Settings())

    with pytest.raises(MCPError):
        await resolver(question="q", parts=[{"variable": "x"}])
    with pytest.raises(ResolverError):
        await resolver(question="q", parts=[{"variable": "x"}])

    (client,) = built  # the factory ran once: the timeout did not discard it
    assert client.calls == ["search", "search"]


async def test_the_resolver_discards_its_client_after_a_non_timeout_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built: list[_ScriptedMcpClient] = []

    def _fake_client(
        self: McpTimeoutPolicy,
        url: str,
        *,
        auth: str | None,
        caller: str | None,
        tool: str | None = None,
    ) -> Any:
        script: list[Any] = (
            [RuntimeError("transport closed")]
            if not built
            else [
                SimpleNamespace(is_error=True, structured_content={"code": "refused"}, content=[])
            ]
        )
        client = _ScriptedMcpClient(script)
        built.append(client)
        return client

    monkeypatch.setattr(McpTimeoutPolicy, "client", _fake_client)
    resolver = McpResolver(settings=Settings())

    with pytest.raises(RuntimeError):
        await resolver(question="q", parts=[{"variable": "x"}])
    with pytest.raises(ResolverError):
        await resolver(question="q", parts=[{"variable": "x"}])

    assert len(built) == 2  # the broken client was discarded; a fresh one replaced it
    assert built[0].exited == 1
    assert built[0].calls == ["search"]
    assert built[1].calls == ["search"]


async def test_executor_aclose_disconnects_every_held_client(
    fake_policy_clients: list[_FakeMcpClient],
) -> None:
    executor = McpExecutor(settings=Settings())
    await executor(tool="get_data", args={"spec_id": "spec_" + "1" * 32})
    await executor._indicators_for(["dcid:LifeExpectancy_Person"])
    await executor._presentation_client()

    await executor.aclose()

    assert len(fake_policy_clients) == 3
    assert all(client.exited == 1 for client in fake_policy_clients)


class _DescribingClient(_FakeMcpClient):
    """Answers `describe` with one node per id asked for."""

    async def call_tool(
        self, tool: str, arguments: dict[str, Any], *, raise_on_error: bool, **_: Any
    ) -> Any:
        self.calls.append(tool)
        self.batches.append(list(arguments["ids"]))
        nodes = [
            {"dcid": dcid, "label": f"label {dcid}", "definition": f"def {dcid}"}
            for dcid in arguments["ids"]
        ]
        return _tool_result({"nodes": nodes})

    batches: list[list[str]]

    def __init__(self, caller: str | None = None) -> None:
        super().__init__(caller)
        self.batches = []


class _PartiallyHeldClient(_FakeMcpClient):
    """Answers `describe` as data-mcp does when one id of the batch is not
    held: a `node_not_held` refusal naming it in `fix.dcids`, with no
    `nodes`. A batch without it gets one node per id."""

    def __init__(self, *, unheld: str, named: bool = True) -> None:
        super().__init__()
        self.unheld = unheld
        self.named = named
        self.batches: list[list[str]] = []

    async def call_tool(
        self, tool: str, arguments: dict[str, Any], *, raise_on_error: bool, **_: Any
    ) -> Any:
        self.calls.append(tool)
        ids = list(arguments["ids"])
        self.batches.append(ids)
        if self.unheld in ids:
            body: dict[str, Any] = {"code": "node_not_held"}
            if self.named:
                body["fix"] = {"dcids": [self.unheld]}
            return SimpleNamespace(is_error=True, structured_content=body, content=[], meta=None)
        nodes = [{"dcid": dcid, "label": f"label {dcid}", "definition": ""} for dcid in ids]
        return _tool_result({"nodes": nodes})


async def test_a_miss_that_names_no_ref_caches_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    describing = _PartiallyHeldClient(unheld="Count_1", named=False)
    monkeypatch.setattr(McpTimeoutPolicy, "client", lambda self, *a, **k: describing)
    executor = McpExecutor(settings=Settings())

    rows = await executor._indicators_for(["dcid:Count_0", "dcid:Count_1"])

    assert describing.batches == [["Count_0", "Count_1"]]
    assert all("label" not in row["name"] for row in rows)
    assert executor._indicator_cache == {}


async def test_an_unheld_id_in_a_describe_batch_leaves_its_siblings_named(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    describing = _PartiallyHeldClient(unheld="Count_1")
    monkeypatch.setattr(McpTimeoutPolicy, "client", lambda self, *a, **k: describing)
    executor = McpExecutor(settings=Settings())

    rows = await executor._indicators_for(["dcid:Count_0", "dcid:Count_1", "dcid:Count_2"])

    assert describing.batches == [["Count_0", "Count_1", "Count_2"], ["Count_0", "Count_2"]]
    assert [row["name"] for row in rows[::2]] == ["label Count_0", "label Count_2"]
    assert rows[1]["name"] != "label Count_1"
    assert executor._indicator_cache["dcid:Count_0"]["name"] == "label Count_0"


def _assembled_summary(
    *, drawn: int, resolved: int, series_cut: int
) -> tuple[ProvenanceStore, _DescribingClient, dict[str, Any], dict[str, Any]]:
    """Assemble a card over a manifest of `resolved` StatVars whose chart
    draws the first `drawn`; returns the store, the describe client, the envelope and the platform facts."""
    refs = [f"dcid:Count_{i}" for i in range(resolved)]
    describing = _DescribingClient()
    frame_ref = {"spec_id": "spec_" + "1" * 32, "content_version": "cv_" + "1" * 32}
    store = ProvenanceStore()
    manifest = {"trace": {"resolved_stat_vars": [{"ref": ref, "roles": []} for ref in refs]}}
    store.put(conversation=current_conversation_id.get(), manifest=manifest, **frame_ref)
    envelope = resolved_revision(
        "line_chart", component_id="cmp_1", revision=1, frame_ref=frame_ref
    )
    facts = {
        "series": [
            {"id": f"s{i}", "coordinates": {"stat_var": refs[i], "unit": "n"}} for i in range(drawn)
        ],
        "series_cut": series_cut,
    }
    return store, describing, envelope, facts


async def test_a_card_over_thousands_of_stat_vars_describes_only_the_drawn_ones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, describing, envelope, facts = _assembled_summary(drawn=5, resolved=3000, series_cut=0)
    monkeypatch.setattr(McpTimeoutPolicy, "client", lambda self, *a, **k: describing)
    executor = McpExecutor(settings=Settings(), provenance=store)

    component = await executor._assemble_component(
        reply={"component_id": "cmp_1", "revision": 1, "fact_summary": facts}, envelope=envelope
    )

    assert component is not None
    assert [len(batch) for batch in describing.batches] == [5]
    summary = component["fact_summary"]
    assert [row["id"] for row in summary["indicators"]] == [f"dcid:Count_{i}" for i in range(5)]
    assert summary["attribution"]["indicator_count"] == 3000
    assert summary["attribution"]["label"].endswith("3000 indicator(s)")


async def test_a_cut_series_list_is_shorter_than_the_indicator_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, describing, envelope, facts = _assembled_summary(drawn=20, resolved=3000, series_cut=40)
    monkeypatch.setattr(McpTimeoutPolicy, "client", lambda self, *a, **k: describing)
    executor = McpExecutor(settings=Settings(), provenance=store)

    component = await executor._assemble_component(
        reply={"component_id": "cmp_1", "revision": 1, "fact_summary": facts}, envelope=envelope
    )

    assert component is not None
    assert sorted(len(batch) for batch in describing.batches) == [4, 8, 8]
    summary = component["fact_summary"]
    assert len(summary["indicators"]) == 20
    assert summary["attribution"]["label"].endswith(" · 3000 indicator(s)")
    assert " of " not in summary["attribution"]["label"]
    assert summary["attribution"]["indicator_count"] == 3000


async def test_a_card_over_a_derived_frame_counts_that_frames_stat_vars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A family frame of 461 StatVars filtered to one: the card over the
    derived frame counts the one StatVar its producing step declared, and a
    card over the retrieval's own frame counts the manifest's 461."""
    store, describing, root_envelope, facts = _assembled_summary(
        drawn=1, resolved=461, series_cut=0
    )
    root = root_envelope["provenance"]["inputs"][0]
    derived = {"spec_id": "ops_" + "2" * 32, "content_version": "ocv_" + "2" * 32}
    linked = store.alias(
        conversation=current_conversation_id.get(),
        sources=[(root["spec_id"], root["content_version"])],
        stat_var_count=1,
        **derived,
    )
    assert linked
    monkeypatch.setattr(McpTimeoutPolicy, "client", lambda self, *a, **k: describing)
    executor = McpExecutor(settings=Settings(), provenance=store)
    reply = {"component_id": "cmp_1", "revision": 1, "fact_summary": facts}

    over_derived = await executor._assemble_component(
        reply=reply,
        envelope=resolved_revision(
            "line_chart", component_id="cmp_1", revision=1, frame_ref=derived
        ),
    )
    over_root = await executor._assemble_component(reply=reply, envelope=root_envelope)

    assert over_derived is not None
    assert over_root is not None
    derived_summary = over_derived["fact_summary"]
    assert derived_summary["attribution"]["indicator_count"] == 1
    assert derived_summary["attribution"]["label"].endswith(" · 1 indicator(s)")
    assert [row["id"] for row in derived_summary["indicators"]] == ["dcid:Count_0"]
    assert over_root["fact_summary"]["attribution"]["indicator_count"] == 461


async def test_a_card_over_a_summed_family_counts_the_stat_vars_the_sum_added(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A summed retrieval of 74 StatVars, with dfo-apis's trace still naming
    all 80 the definition resolved, and its frame aggregated by period: a
    card over either frame counts the 74 the sum added, not the one family
    ref the frame holds or the trace's 80."""
    store, describing, root_envelope, facts = _assembled_summary(drawn=0, resolved=80, series_cut=0)
    root = root_envelope["provenance"]["inputs"][0]
    lineage = store.get(conversation=current_conversation_id.get(), **root)
    assert lineage is not None
    lineage[0].manifest["summed_stat_var_count"] = 74
    derived = {"spec_id": "ops_" + "2" * 32, "content_version": "ocv_" + "2" * 32}
    root_key = (root["spec_id"], root["content_version"])
    assert store.alias(
        conversation=current_conversation_id.get(),
        sources=[root_key],
        stat_var_count=1,
        carried=[root_key],
        **derived,
    )
    monkeypatch.setattr(McpTimeoutPolicy, "client", lambda self, *a, **k: describing)
    executor = McpExecutor(settings=Settings(), provenance=store)
    reply = {"component_id": "cmp_1", "revision": 1, "fact_summary": facts}

    over_derived = await executor._assemble_component(
        reply=reply,
        envelope=resolved_revision(
            "line_chart", component_id="cmp_1", revision=1, frame_ref=derived
        ),
    )
    over_root = await executor._assemble_component(reply=reply, envelope=root_envelope)

    assert over_derived is not None
    assert over_root is not None
    assert over_derived["fact_summary"]["attribution"]["indicator_count"] == 74
    assert over_root["fact_summary"]["attribution"]["indicator_count"] == 74


@pytest.mark.parametrize(
    ("label", "citation", "name"),
    [
        (
            "Official development assistance, Education",
            "OECD CRS",
            "Official development assistance, Education",
        ),
        (None, "OECD CRS", "OECD CRS"),
        (None, None, "Unknown source"),
    ],
    ids=["dfo-apis label", "citation fallback", "no source named"],
)
async def test_a_summed_familys_ref_is_named_as_dfo_apis_names_it_and_never_described(
    monkeypatch: pytest.MonkeyPatch, label: str | None, citation: str | None, name: str
) -> None:
    """The chart draws the summed frame's one ref, `dcid:<spec_id>`. The card
    names it by `get_data`'s `summed_stat_var_label`, or by the source its
    attribution names when dfo-apis stated none, and never asks `describe`
    about it."""
    store, describing, envelope, _ = _assembled_summary(drawn=0, resolved=80, series_cut=0)
    root = envelope["provenance"]["inputs"][0]
    lineage = store.get(conversation=current_conversation_id.get(), **root)
    assert lineage is not None
    lineage[0].manifest.update(
        summed_stat_var_count=74,
        summed_stat_var_label=label,
        citation={"text": citation, "sources": []},
    )
    family_ref = f"dcid:{root['spec_id']}"
    facts = {"series": [{"id": "s0", "coordinates": {"stat_var": family_ref}}], "series_cut": 0}
    monkeypatch.setattr(McpTimeoutPolicy, "client", lambda self, *a, **k: describing)
    executor = McpExecutor(settings=Settings(), provenance=store)

    component = await executor._assemble_component(
        reply={"component_id": "cmp_1", "revision": 1, "fact_summary": facts}, envelope=envelope
    )

    assert component is not None
    assert describing.batches == []
    assert component["fact_summary"]["indicators"] == [
        {"id": family_ref, "name": name, "description": "A total over 74 indicators."}
    ]


async def test_a_sums_ref_is_named_by_the_two_sides_it_adds_and_never_described(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A `sum_of` writes its rows under `dcid:<its spec_id>`. The card names
    that ref by its sides: a summed family by dfo-apis's label, any other
    StatVar by `describe`."""
    conversation = current_conversation_id.get()
    store = ProvenanceStore()
    bilateral = ("spec_" + "1" * 32, "cv_" + "1" * 32)
    imputed = ("spec_" + "2" * 32, "cv_" + "2" * 32)
    store.put(
        conversation=conversation,
        spec_id=bilateral[0],
        content_version=bilateral[1],
        manifest={"summed_stat_var_count": 12, "summed_stat_var_label": "Bilateral ODA"},
    )
    store.put(
        conversation=conversation,
        spec_id=imputed[0],
        content_version=imputed[1],
        manifest={"trace": {"resolved_stat_vars": [{"ref": "dcid:ImputedOda", "roles": []}]}},
    )
    total = {"spec_id": "ops_" + "3" * 32, "content_version": "ocv_" + "3" * 32}
    total_ref = f"dcid:{total['spec_id']}"
    assert store.alias(
        conversation=conversation,
        sources=[bilateral, imputed],
        stat_var_count=1,
        carried=[bilateral, imputed],
        own_stat_vars={total_ref: ("sum_of", (bilateral,), (imputed,))},
        **total,
    )
    describing = _DescribingClient()
    monkeypatch.setattr(McpTimeoutPolicy, "client", lambda self, *a, **k: describing)
    executor = McpExecutor(settings=Settings(), provenance=store)
    facts = {"series": [{"id": "s0", "coordinates": {"stat_var": total_ref}}], "series_cut": 0}

    component = await executor._assemble_component(
        reply={"component_id": "cmp_1", "revision": 1, "fact_summary": facts},
        envelope=resolved_revision("line_chart", component_id="cmp_1", revision=1, frame_ref=total),
    )

    assert component is not None
    assert describing.batches == [["ImputedOda"]]
    (row,) = component["fact_summary"]["indicators"]
    assert row["id"] == total_ref
    assert row["name"] == "Bilateral ODA plus label ImputedOda"


_DIVIDED = "The first divided by the second for each year both report."
_SHARED = "The first as a share of the second, in per cent, for each year both report."


@pytest.mark.parametrize(
    ("verb", "numerator", "denominator", "name", "description", "described"),
    [
        (
            "quotient_of",
            ("bilateral",),
            ("population",),
            "Bilateral ODA per label Count_Person",
            _DIVIDED,
            ["Count_Person"],
        ),
        # Two slices of one frame.
        ("quotient_of", ("bilateral",), ("bilateral",), "Ratio of Bilateral ODA", _DIVIDED, []),
        # A numerator that is itself a sum of two retrievals.
        (
            "quotient_of",
            ("bilateral", "imputed"),
            ("population",),
            "Bilateral ODA and label ImputedOda per label Count_Person",
            _DIVIDED,
            ["ImputedOda", "Count_Person"],
        ),
        (
            "share_of",
            ("imputed",),
            ("population",),
            "label ImputedOda as a share of label Count_Person",
            _SHARED,
            ["ImputedOda", "Count_Person"],
        ),
        # A part of its total, both read from one frame.
        ("share_of", ("bilateral",), ("bilateral",), "Share of Bilateral ODA", _SHARED, []),
        (
            "sum_of",
            ("bilateral",),
            ("bilateral",),
            "Bilateral ODA",
            "The two added together for each year both report.",
            [],
        ),
        # Each side an earlier quotient, named by its own sides.
        (
            "share_of",
            (("quotient_of", ("imputed",), ("population",)),),
            (("quotient_of", ("bilateral",), ("population",)),),
            "label ImputedOda per label Count_Person as a share of "
            "Bilateral ODA per label Count_Person",
            _SHARED,
            ["ImputedOda", "Count_Person"],
        ),
        # A numerator that is an earlier sum of two retrievals.
        (
            "quotient_of",
            (("sum_of", ("bilateral",), ("imputed",)),),
            ("population",),
            "Bilateral ODA plus label ImputedOda per label Count_Person",
            _DIVIDED,
            ["ImputedOda", "Count_Person"],
        ),
    ],
    ids=[
        "two frames",
        "one frame",
        "summed numerator",
        "share",
        "share of one frame",
        "sum of one frame",
        "share of two quotients",
        "quotient of a sum",
    ],
)
async def test_a_verbs_own_ref_is_named_by_its_two_sides_and_cites_both(
    monkeypatch: pytest.MonkeyPatch,
    verb: str,
    numerator: tuple[Any, ...],
    denominator: tuple[Any, ...],
    name: str,
    description: str,
    described: list[str],
) -> None:
    """A `quotient_of`, a `share_of` or a `sum_of` writes its rows under
    `dcid:<its spec_id>`. The card names that ref "<numerator> per
    <denominator>" or "<numerator> as a share of <denominator>"; two sides
    of one StatVar read "Ratio of <side>", "Share of <side>" or "<side>", as
    dfo-apis labels a published frame. Each side is named as a sum's side
    is, and every retrieval under either side is listed as an input (a frame
    resting on one retrieval lists itself alone). A side that is an earlier
    verb's own ref, written here as `(verb, first, second)`, reads as that
    verb's own name."""
    conversation = current_conversation_id.get()
    store = ProvenanceStore()
    manifests: dict[str, dict[str, Any]] = {
        "bilateral": {"summed_stat_var_count": 12, "summed_stat_var_label": "Bilateral ODA"},
        "imputed": {"trace": {"resolved_stat_vars": [{"ref": "dcid:ImputedOda", "roles": []}]}},
        "population": {
            "trace": {"resolved_stat_vars": [{"ref": "dcid:Count_Person", "roles": []}]}
        },
    }
    keys = {
        key: (f"spec_{index}" + "0" * 31, f"cv_{index}" + "0" * 31)
        for index, key in enumerate(manifests)
    }
    for key, manifest in manifests.items():
        store.put(
            conversation=conversation,
            spec_id=keys[key][0],
            content_version=keys[key][1],
            manifest=manifest,
        )
    written: dict[str, tuple[str, tuple[Any, ...], tuple[Any, ...]]] = {}
    retrieved: list[str] = []

    def parts(side: tuple[Any, ...]) -> tuple[Any, ...]:
        out: list[Any] = []
        for part in side:
            if isinstance(part, str):
                retrieved.append(part)
                out.append(keys[part])
                continue
            nested_verb, first, second = part
            nested_ref = f"dcid:ops_{len(written)}" + "8" * 31
            written[nested_ref] = (nested_verb, parts(first), parts(second))
            out.append(nested_ref)
        return tuple(out)

    ratio = {"spec_id": "ops_" + "9" * 32, "content_version": "ocv_" + "9" * 32}
    ratio_ref = f"dcid:{ratio['spec_id']}"
    written[ratio_ref] = (verb, parts(numerator), parts(denominator))
    sources = [keys[key] for key in dict.fromkeys(retrieved)]
    assert store.alias(
        conversation=conversation,
        sources=sources,
        stat_var_count=1,
        carried=sources,
        own_stat_vars=written,
        **ratio,
    )
    describing = _DescribingClient()
    monkeypatch.setattr(McpTimeoutPolicy, "client", lambda self, *a, **k: describing)
    executor = McpExecutor(settings=Settings(), provenance=store)
    facts = {"series": [{"id": "s0", "coordinates": {"stat_var": ratio_ref}}], "series_cut": 0}

    component = await executor._assemble_component(
        reply={"component_id": "cmp_1", "revision": 1, "fact_summary": facts},
        envelope=resolved_revision("line_chart", component_id="cmp_1", revision=1, frame_ref=ratio),
    )

    assert component is not None
    assert sorted(ref for batch in describing.batches for ref in batch) == sorted(described)
    assert component["fact_summary"]["indicators"] == [
        {"id": ratio_ref, "name": name, "description": description}
    ]
    assert [
        (source["spec_id"], source["content_version"])
        for source in component["provenance"]["inputs"]
    ] == [(ratio["spec_id"], ratio["content_version"]), *(sources if len(sources) > 1 else [])]


async def test_a_chart_with_no_stat_var_coordinate_lists_no_indicators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, describing, envelope, facts = _assembled_summary(drawn=0, resolved=12, series_cut=0)
    facts["series"] = [{"id": "s0", "coordinates": {"unit": "n"}}]
    monkeypatch.setattr(McpTimeoutPolicy, "client", lambda self, *a, **k: describing)
    executor = McpExecutor(settings=Settings(), provenance=store)

    component = await executor._assemble_component(
        reply={"component_id": "cmp_1", "revision": 1, "fact_summary": facts}, envelope=envelope
    )

    assert component is not None
    assert describing.batches == []
    summary = component["fact_summary"]
    assert summary["indicators"] == []
    assert summary["attribution"]["indicator_count"] == 12


async def test_the_platform_defaults_reach_the_wire_fact_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, describing, envelope, facts = _assembled_summary(drawn=1, resolved=1, series_cut=0)
    monkeypatch.setattr(McpTimeoutPolicy, "client", lambda self, *a, **k: describing)
    executor = McpExecutor(settings=Settings(), provenance=store)
    reply = {"component_id": "cmp_1", "revision": 1, "fact_summary": facts}

    without = await executor._assemble_component(reply=reply, envelope=envelope)
    facts["defaults"] = [{"pointer": "/intent/select", "detail": "Drew every recipient."}]
    facts["a_member_the_platform_adds_later"] = {"x": 1}
    with_defaults = await executor._assemble_component(reply=reply, envelope=envelope)

    assert without is not None
    assert "defaults" not in without["fact_summary"]
    assert with_defaults is not None
    assert with_defaults["fact_summary"]["defaults"] == [
        {"pointer": "/intent/select", "detail": "Drew every recipient."}
    ]
    # The strict wire model sees only the members `_wire_fact_summary` names, so a
    # member the platform adds later does not break the frame.
    FactSummary.model_validate(with_defaults["fact_summary"])


@pytest.mark.parametrize(
    "unit",
    [
        None,
        "US Dollar",
        {"currency": "USD", "prices": "constant", "base_year": 2023},
        {
            "numerator": {"currency": "USD", "prices": "current"},
            "denominator": {"stat_var": "dcid:Count_Person", "label": "Total population"},
        },
    ],
    ids=["none", "name", "structured", "composed"],
)
async def test_the_platform_unit_reaches_the_wire_fact_summary_as_it_is(
    monkeypatch: pytest.MonkeyPatch, unit: object
) -> None:
    store, describing, envelope, facts = _assembled_summary(drawn=1, resolved=1, series_cut=0)
    monkeypatch.setattr(McpTimeoutPolicy, "client", lambda self, *a, **k: describing)
    executor = McpExecutor(settings=Settings(), provenance=store)
    facts["unit"] = unit

    component = await executor._assemble_component(
        reply={"component_id": "cmp_1", "revision": 1, "fact_summary": facts}, envelope=envelope
    )

    assert component is not None
    assert component["fact_summary"].get("unit") == unit
    assert usable_component(component)


async def test_each_series_states_the_periods_of_its_first_and_last_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, describing, envelope, facts = _assembled_summary(drawn=4, resolved=4, series_cut=0)
    facts["series"][0] |= {"point_count": 3, "first_value": "173.6", "last_value": "60.1"}
    envelope["component"]["series"] = [
        {
            # Missing points at both ends: the first and last values fall inside
            # the window, and only their two periods leave the points.
            "id": "s0",
            "points": [
                {"period": "1960", "status": "missing"},
                {"period": "1964", "value": 173.6},
                {"period": "1990", "value": 125.0},
                {"period": "2023", "value": 60.1},
                {"period": "2024", "status": "missing"},
            ],
        },
        # The first observed point carries no period: its period is withheld,
        # never borrowed from the next point.
        {"id": "s1", "points": [{"value": 10.0}, {"period": "2020", "value": 20.0}]},
        # A repeated id cannot say which series a period belongs to.
        {"id": "s2", "points": [{"period": "2001", "value": 1.0}]},
        {"id": "s2", "points": [{"period": "2002", "value": 2.0}]},
        # No point carries a period.
        {"id": "s3", "points": [{"category": "Health", "value": 4.0}]},
    ]
    monkeypatch.setattr(McpTimeoutPolicy, "client", lambda self, *a, **k: describing)
    executor = McpExecutor(settings=Settings(), provenance=store)

    component = await executor._assemble_component(
        reply={"component_id": "cmp_1", "revision": 1, "fact_summary": facts}, envelope=envelope
    )

    assert component is not None
    gapped, partly_dated, repeated, undated = component["fact_summary"]["series"]
    assert {key: gapped[key] for key in ("first", "first_period", "last", "last_period")} == {
        "first": 173.6,
        "first_period": "1964",
        "last": 60.1,
        "last_period": "2023",
    }
    assert "first_period" not in partly_dated and partly_dated["last_period"] == "2020"
    for series in (repeated, undated):
        assert "first_period" not in series and "last_period" not in series
    FactSummary.model_validate(component["fact_summary"])


# --- progress: a get_data call's MCP progress reaches the caller ------------


def _reporting_data_mcp() -> FastMCP:
    """A data-mcp stand-in whose `get_data` reports twice before it answers."""
    server = FastMCP(name="data-mcp")

    @server.tool(name="get_data")
    async def get_data(spec_id: str, ctx: Context) -> ToolResult:
        await ctx.report_progress(progress=0, total=None, message="submitted")
        await ctx.report_progress(progress=4.0, total=None, message="running")
        return ToolResult(
            content="fetched",
            structured_content={"definition": {"spec_id": spec_id}, "content_version": "v1"},
        )

    return server


async def test_call_tool_hands_the_servers_progress_reports_to_on_progress() -> None:
    reports: list[ProgressReport] = []

    async with Client(_reporting_data_mcp()) as client:
        result = await _call_tool(
            client, "get_data", {"spec_id": "spec_" + "1" * 32}, on_progress=reports.append
        )

    assert result.is_error is False
    assert [report.message for report in reports] == ["submitted", "running"]


async def test_the_executor_hands_a_get_data_steps_progress_to_on_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The direct-fetch path: `McpExecutor` passes `on_progress` through its
    held client to the `get_data` call."""
    server = _reporting_data_mcp()

    def _in_memory_client(
        self: McpTimeoutPolicy,
        url: str,
        *,
        auth: str | None,
        caller: str | None,
        tool: str | None = None,
    ) -> Any:
        return Client(server)

    monkeypatch.setattr(McpTimeoutPolicy, "client", _in_memory_client)
    executor = McpExecutor(settings=Settings())
    reports: list[ProgressReport] = []
    try:
        outcome = await executor(
            tool="get_data", args={"spec_id": "spec_" + "1" * 32}, on_progress=reports.append
        )
    finally:
        await executor.aclose()

    assert outcome["outcome"] == "ok"
    assert reports == [
        ProgressReport(progress=0, total=None, message="submitted"),
        ProgressReport(progress=4.0, total=None, message="running"),
    ]


# --- the Activity feed's one-line get_data summary --------------------------

_MINT_MANIFEST = Path(__file__).parent / "fixtures" / "mint_statvar_set_guatemala.json"


def _wire_manifest() -> dict[str, Any]:
    """The `get_data` structuredContent for the recorded mint response."""
    return Manifest.model_validate(json.loads(_MINT_MANIFEST.read_text())).model_dump(mode="json")


def test_manifest_summary_reads_the_wire_manifests_coverage_and_citation() -> None:
    assert (
        _manifest_summary(_wire_manifest())
        == "15 observations, 2010\u20132024 \u00b7 WorldDevelopmentIndicators"
    )


def test_manifest_summary_names_the_series_count_when_there_are_several() -> None:
    manifest = _wire_manifest() | {"series_count": 3}

    assert (
        _manifest_summary(manifest)
        == "15 observations, 2010\u20132024 \u00b7 WorldDevelopmentIndicators \u00b7 3 series"
    )


def test_manifest_summary_without_a_window_keeps_the_count_and_citation() -> None:
    manifest = _wire_manifest()
    manifest["coverage"] = {**manifest["coverage"], "window": None}

    assert _manifest_summary(manifest) == "15 observations \u00b7 WorldDevelopmentIndicators"


def test_manifest_summary_collapses_a_single_year_window() -> None:
    manifest = _wire_manifest()
    manifest["coverage"] = {
        **manifest["coverage"],
        "window": {"kind": "range", "start_year": 2015, "end_year": 2015},
    }

    assert _manifest_summary(manifest).startswith("15 observations, 2015 \u00b7")


def test_manifest_summary_without_a_citation_keeps_the_count_and_window() -> None:
    manifest = _wire_manifest()
    del manifest["citation"]

    assert _manifest_summary(manifest) == "15 observations, 2010\u20132024"


def test_manifest_summary_is_none_when_the_manifest_carries_nothing_to_say() -> None:
    assert _manifest_summary({}) is None


# --- a `search` success carries its envelope in `_meta`. --------------------------
# `structuredContent` is the small answer a host may show a model, and the
# resolver never reads it.

_SEARCH_FIXTURES = (
    "resolve_population_guatemala",
    "resolve_gdp_per_capita_guatemala",
    "resolve_health_spending_broad",
)
_DATA_MCP_FIXTURES = Path(__file__).parents[2] / "data-mcp" / "tests" / "fixtures"


def _search_success(envelope: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        is_error=False,
        structured_content=render_answer(envelope),
        content=[],
        meta={SEARCH_ENVELOPE_META_KEY: envelope},
    )


@pytest.mark.parametrize("fixture", _SEARCH_FIXTURES)
async def test_the_resolver_reads_a_search_envelope_from_meta_whole(
    fixture: str,
    scripted_presentation_client: Callable[[list[Any]], _ScriptedMcpClient],
) -> None:
    """The digest and the whole `spec` block the planner receives are the ones
    the envelope alone determines, whatever `structuredContent` holds."""
    envelope = json.loads((_DATA_MCP_FIXTURES / f"{fixture}.json").read_text())
    scripted_presentation_client([_search_success(envelope)])
    resolver = McpResolver(settings=Settings())

    digest = await resolver(question="q", parts=[{"variable": "x"}])

    expected = graph_envelope(envelope)
    expected["spec_block"] = envelope["parts"][0]["spec"]
    for part, summary in zip(envelope["parts"], expected["part_digests"], strict=True):
        summary["spec_block"] = part.get("spec") if summary["status"] == "definite" else None
        summary["candidate_blocks"] = {
            spec["spec_id"]: spec for spec in (part.get("candidates") or {}).get("specs") or []
        }
    expected["kind"] = envelope.get("kind") or "resolution"
    assert digest == expected
    assert all("spec_block" in summary for summary in digest["part_digests"])


async def test_the_resolver_puts_each_parts_block_kind_and_skills_on_the_digest(
    scripted_presentation_client: Callable[[list[Any]], _ScriptedMcpClient],
) -> None:
    """A two-part envelope: the definite part's own block rides on its
    digest, a candidates part carries none, the envelope's `kind` passes
    through, and its `skills` map reaches the digest as `graph_envelope`'s
    `{"id", "description"}` list."""
    first = json.loads((_DATA_MCP_FIXTURES / f"{_SEARCH_FIXTURES[0]}.json").read_text())
    definite = first["parts"][0]
    candidates = {
        "variable_text": "GDP",
        "status": "candidates",
        "candidates": {"specs": [{"spec_id": "spec_" + "c" * 32, "resolved": {}}]},
    }
    envelope = {
        **first,
        "kind": "resolution",
        "parts": [definite, candidates],
        "skills": {"oda-measures": {"name": "ODA measures", "description": "What counts."}},
    }
    scripted_presentation_client([_search_success(envelope)])
    resolver = McpResolver(settings=Settings())

    digest = await resolver(question="q", parts=[{"variable": "x"}, {"variable": "GDP"}])

    assert [part["status"] for part in digest["part_digests"]] == ["definite", "candidates"]
    assert digest["part_digests"][0]["spec_block"] == definite["spec"]
    assert digest["part_digests"][1]["spec_block"] is None
    assert digest["part_digests"][1]["candidate_blocks"] == {
        "spec_" + "c" * 32: candidates["candidates"]["specs"][0]
    }
    assert digest["kind"] == "resolution"
    assert digest["skills"] == [{"id": "oda-measures", "description": "What counts."}]
    assert digest["spec_block"] == definite["spec"]


async def test_the_resolver_refuses_a_search_success_that_carries_no_envelope(
    scripted_presentation_client: Callable[[list[Any]], _ScriptedMcpClient],
) -> None:
    only_the_answer = SimpleNamespace(
        is_error=False,
        structured_content={"status": "definite", "parts": []},
        content=[],
        meta=None,
    )
    scripted_presentation_client([only_the_answer])
    resolver = McpResolver(settings=Settings())

    with pytest.raises(ResolverError, match="no envelope"):
        await resolver(question="q", parts=[{"variable": "x"}])


# --- a `retry_after_delay` refusal is retried on the executor's own policy ---


class _Sleeps:
    """Records the waits the executor asks for instead of waiting."""

    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


def _data_refusal(retry_mode: str, *, retry_after_s: float | None = 2.0) -> SimpleNamespace:
    """A data-mcp refusal as the MCP client returns it: an error result
    whose structured content is the refusal body."""
    body: dict[str, Any] = {
        "code": "service_unavailable",
        "retry_mode": retry_mode,
        "message": "the frame row budget is full",
    }
    if retry_after_s is not None:
        body["retry_after_s"] = retry_after_s
    return SimpleNamespace(is_error=True, structured_content=body, content=[], meta=None)


def _presentation_refusal(
    retry: str, *, code: str = "service_unavailable", fix: dict[str, Any] | None = None
) -> SimpleNamespace:
    """A presentation-mcp refusal in its own `{error, retry, detail}` grammar."""
    body: dict[str, Any] = {
        "error": code,
        "retry": retry,
        "detail": "the frame row budget is full",
        "retry_after_s": 2.0,
    }
    if fix is not None:
        body["fix"] = fix
    return SimpleNamespace(
        is_error=True,
        structured_content=body,
        content=[],
        meta=None,
    )


async def test_the_executor_retries_a_get_data_refusal_that_asks_for_a_delay(
    scripted_presentation_client: Callable[[list[Any]], _ScriptedMcpClient],
) -> None:
    # A frame minted with extra columns is held under a content_version of its own,
    # which `get_data` states as `frame_ref` beside the manifest. The result names the
    # manifest's own pair, which dfo-apis serves to describe_frame and charts.
    frame_ref = {"spec_id": "spec_" + "1" * 32, "content_version": "cv_typed"}
    manifest = {
        "definition": {"spec_id": "spec_" + "1" * 32},
        "content_version": "cv_1",
        "frame_ref": frame_ref,
    }
    client = scripted_presentation_client(
        [_data_refusal("retry_after_delay"), _tool_result(manifest)]
    )
    sleeps = _Sleeps()
    heard: list[ProgressReport] = []
    executor = McpExecutor(settings=Settings(), sleep=sleeps, jitter=random.Random(7).uniform)

    result = await executor(
        tool="get_data", args={"spec_id": "spec_" + "1" * 32}, on_progress=heard.append
    )

    assert result["outcome"] == "ok"
    assert result["frame_refs"] == [{"spec_id": "spec_" + "1" * 32, "content_version": "cv_1"}]
    assert client.calls == ["get_data", "get_data"]
    (delay,) = sleeps.delays
    # At least the named 2s, and at most twice it.
    assert 2.0 <= delay <= 4.0
    assert [report.message for report in heard] == [
        f"Retrying in {delay:.3g}s (attempt 2 of {MAX_ATTEMPTS_PER_STEP})"
    ]


async def test_the_executor_retries_a_write_refused_by_the_row_budget(
    scripted_presentation_client: Callable[[list[Any]], _ScriptedMcpClient],
) -> None:
    frame_ref = {"spec_id": "spec_" + "1" * 32, "content_version": "cv_" + "1" * 32}
    envelope = resolved_revision("line_chart", component_id="cmp_1", frame_ref=frame_ref)
    client = scripted_presentation_client(
        [
            _presentation_refusal("retry_after_delay", fix={"rows_requested": 250_000}),
            _write_result(envelope),
        ]
    )
    store = ProvenanceStore()
    store.put(conversation=current_conversation_id.get(), manifest={}, **frame_ref)
    sleeps = _Sleeps()
    executor = McpExecutor(
        settings=Settings(), provenance=store, sleep=sleeps, jitter=random.Random(7).uniform
    )

    result = await executor(tool="write_component", args={"intent": {"type": "line_chart"}})

    assert result["outcome"] == "ok"
    assert result["component"]["type"] == "line_chart"
    assert client.calls == ["write_component", "write_component"]
    assert len(sleeps.delays) == 1


async def test_the_executor_does_not_retry_a_write_the_platform_may_have_committed(
    scripted_presentation_client: Callable[[list[Any]], _ScriptedMcpClient],
) -> None:
    client = scripted_presentation_client(
        [_presentation_refusal("retry_after_delay", code="transient_unavailable"), _tool_result({})]
    )
    sleeps = _Sleeps()
    executor = McpExecutor(settings=Settings(), sleep=sleeps, jitter=random.Random(7).uniform)

    result = await executor(tool="write_component", args={"intent": {}})

    assert result["outcome"] == "refusal"
    assert result["refusal"]["code"] == "transient_unavailable"
    assert client.calls == ["write_component"]
    assert sleeps.delays == []


async def test_the_executor_returns_the_refusal_once_every_attempt_is_spent(
    scripted_presentation_client: Callable[[list[Any]], _ScriptedMcpClient],
) -> None:
    client = scripted_presentation_client(
        [_data_refusal("retry_after_delay")] * MAX_ATTEMPTS_PER_STEP
    )
    sleeps = _Sleeps()
    executor = McpExecutor(settings=Settings(), sleep=sleeps, jitter=random.Random(7).uniform)

    result = await executor(tool="get_data", args={"spec_id": "spec_" + "1" * 32})

    assert result["outcome"] == "refusal"
    assert result["refusal"]["code"] == "service_unavailable"
    assert result["refusal"]["retry_mode"] == "retry_after_delay"
    assert client.calls == ["get_data"] * MAX_ATTEMPTS_PER_STEP
    assert len(sleeps.delays) == MAX_ATTEMPTS_PER_STEP - 1


@pytest.mark.parametrize("retry_mode", ["correct_and_retry", "do_not_retry", "refresh_then_retry"])
@pytest.mark.parametrize("tool", ["get_data", "write_component"])
async def test_the_executor_retries_no_other_retry_mode(
    scripted_presentation_client: Callable[[list[Any]], _ScriptedMcpClient],
    tool: str,
    retry_mode: str,
) -> None:
    refused = _data_refusal(retry_mode) if tool == "get_data" else _presentation_refusal(retry_mode)
    client = scripted_presentation_client([refused, _tool_result({})])
    sleeps = _Sleeps()
    executor = McpExecutor(settings=Settings(), sleep=sleeps, jitter=random.Random(7).uniform)

    result = await executor(tool=tool, args={"spec_id": "spec_" + "1" * 32, "intent": {}})

    assert result["outcome"] == "refusal"
    assert result["refusal"]["retry_mode"] == retry_mode
    assert client.calls == [tool]
    assert sleeps.delays == []


class _Clock:
    """A clock that reads `start` until a test moves it on."""

    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now


class _SlowRefusalClient(_ScriptedMcpClient):
    """`_ScriptedMcpClient`, whose every call moves `clock` on by `seconds`
    before it answers."""

    def __init__(self, script: list[Any], *, clock: _Clock, seconds: float) -> None:
        super().__init__(script)
        self._clock = clock
        self._seconds = seconds

    async def call_tool(
        self,
        tool: str,
        arguments: dict[str, Any],
        *,
        meta: dict[str, str],
        progress_handler: Any,
        raise_on_error: bool,
    ) -> Any:
        self._clock.now += self._seconds
        return await super().call_tool(
            tool,
            arguments,
            meta=meta,
            progress_handler=progress_handler,
            raise_on_error=raise_on_error,
        )


@pytest.mark.parametrize(
    ("call_seconds", "calls"),
    [
        # A row-budget refusal answers at once: 0.2 s spent, and a 2-4 s wait
        # fits the 180 s deadline many times over.
        pytest.param(0.2, 2, id="a-fast-refusal-is-retried"),
        # An unanswered mint answers near the deadline: 179 s spent, and no
        # wait of at least 2 s fits in the second left.
        pytest.param(179.0, 1, id="a-refusal-near-the-deadline-is-not"),
    ],
)
async def test_the_executor_retries_only_within_the_tools_own_call_deadline(
    monkeypatch: pytest.MonkeyPatch, call_seconds: float, calls: int
) -> None:
    clock = _Clock()
    manifest = {"definition": {"spec_id": "spec_" + "1" * 32}, "content_version": "cv_1"}
    client = _SlowRefusalClient(
        [_data_refusal("retry_after_delay"), _tool_result(manifest)],
        clock=clock,
        seconds=call_seconds,
    )
    monkeypatch.setattr(
        McpTimeoutPolicy, "client", lambda self, url, *, auth, caller, tool=None: client
    )
    sleeps = _Sleeps()
    executor = McpExecutor(
        settings=Settings(), sleep=sleeps, jitter=random.Random(7).uniform, clock=clock
    )
    assert McpTimeoutPolicy(settings=Settings()).call_timeout_s("get_data") == 180.0

    result = await executor(tool="get_data", args={"spec_id": "spec_" + "1" * 32})

    assert client.calls == ["get_data"] * calls
    assert len(sleeps.delays) == calls - 1
    assert result["outcome"] == ("ok" if calls == 2 else "refusal")


class _HangingRetryClient(_ScriptedMcpClient):
    """Answers its first call from the script and hangs on every later one."""

    async def call_tool(
        self,
        tool: str,
        arguments: dict[str, Any],
        *,
        meta: dict[str, str],
        progress_handler: Any,
        raise_on_error: bool,
    ) -> Any:
        if self.calls:
            self.calls.append(tool)
            await asyncio.Event().wait()
        return await super().call_tool(
            tool,
            arguments,
            meta=meta,
            progress_handler=progress_handler,
            raise_on_error=raise_on_error,
        )


async def test_a_retry_still_running_at_the_call_deadline_is_cut_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _Clock()
    client = _HangingRetryClient([_data_refusal("retry_after_delay")])
    monkeypatch.setattr(
        McpTimeoutPolicy, "client", lambda self, url, *, auth, caller, tool=None: client
    )

    async def wait(seconds: float) -> None:
        # The wait leaves 0.05 s of the 180 s deadline.
        clock.now = 180.0 - 0.05

    executor = McpExecutor(
        settings=Settings(), sleep=wait, jitter=random.Random(7).uniform, clock=clock
    )

    result = await asyncio.wait_for(
        executor(tool="get_data", args={"spec_id": "spec_" + "1" * 32}), timeout=5
    )

    assert client.calls == ["get_data", "get_data"]
    assert result["outcome"] == "refusal"
    assert result["refusal"]["code"] == "service_unavailable"


async def test_a_write_retry_cut_off_at_the_call_deadline_raises_the_timeout_not_the_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cut-off retry was sent, so the platform may have stored the write: the
    call ends as a first attempt's timeout does, never with the earlier busy refusal."""
    clock = _Clock()
    client = _HangingRetryClient(
        [_presentation_refusal("retry_after_delay", fix={"rows_requested": 250_000})]
    )
    monkeypatch.setattr(
        McpTimeoutPolicy, "client", lambda self, url, *, auth, caller, tool=None: client
    )
    deadline_s = McpTimeoutPolicy(settings=Settings()).call_timeout_s("write_component")

    async def wait(seconds: float) -> None:
        clock.now = deadline_s - 0.05

    executor = McpExecutor(
        settings=Settings(), sleep=wait, jitter=random.Random(7).uniform, clock=clock
    )

    with pytest.raises(_McpTimeoutError) as excinfo:
        await asyncio.wait_for(
            executor(tool="write_component", args={"intent": {"type": "line_chart"}}),
            timeout=5,
        )

    assert client.calls == ["write_component", "write_component"]
    assert excinfo.value.timeout_tool == "write_component"
    assert excinfo.value.error.code == REQUEST_TIMEOUT


async def test_a_retrieval_with_no_observations_reports_its_row_count_and_warnings(
    scripted_presentation_client: Callable[[list[Any]], _ScriptedMcpClient],
) -> None:
    """dfo-apis serves a mint whose places or window hold nothing as 200 with zero rows
    and an `empty_result` warning; the outcome carries both, so a caller can tell it
    from data."""
    frame_ref = {"spec_id": "spec_" + "1" * 32, "content_version": "cv_" + "1" * 32}
    warning = {
        "code": "empty_result",
        "severity": "warn",
        "message": "The materialised frame holds zero observations.",
    }
    result = {
        "definition": {"spec_id": frame_ref["spec_id"]},
        "content_version": frame_ref["content_version"],
        "coverage": {"rows": 0, "dimensions": []},
        "warnings": [warning],
        "trace": {},
        "frame_ref": frame_ref,
    }
    scripted_presentation_client([_tool_result(result)])

    outcome = await McpExecutor(settings=Settings(), provenance=ProvenanceStore())(
        tool="get_data", args={"spec_id": frame_ref["spec_id"]}
    )

    assert outcome["outcome"] == "ok"
    assert outcome["rows"] == 0
    assert outcome["warnings"] == [warning]

"""Tests for `compiler.surface`, including against the real tool surface.

The `test_the_real_surface_*` cases read data-mcp and analysis-mcp through
`fastmcp.Client`'s in-memory transport, so they exercise the MCP protocol
in-process. They fail when a tool the compiler builds calls for is renamed or
changes its required arguments.
"""

from __future__ import annotations

import asyncio
import logging
import time

import pytest
from analysis_mcp.server import mcp as analysis_mcp
from common.auth import apply_gate
from compiler.surface import (
    DuplicateToolNameError,
    FixtureSurface,
    McpSurface,
    ProgressReport,
    ToolCallResult,
    ToolSchema,
    ToolSurfaceUnavailableError,
    UnhashableSchemaError,
    _auth_for,
    schema_hash,
    schema_hash_error,
)
from data_mcp.server import mcp as data_mcp
from fastmcp import Client, Context, FastMCP
from fastmcp.server.middleware import Middleware
from fastmcp.tools import ToolResult
from planner.workflow import ToolId

UNREACHABLE_URL = "http://127.0.0.1:9/mcp"


def _held(surface: McpSurface, server: str, caller: str | None = None) -> Client | None:
    """The client `surface` holds for `server` and `caller`, or `None`."""
    held = surface._call_clients[server]._clients.get(caller)
    return held.client if held is not None else None


def _real_surface() -> McpSurface:
    return McpSurface(sources={"data-mcp": data_mcp, "analysis-mcp": analysis_mcp})


async def test_the_real_surface_covers_retrieval_and_the_analysis_verbs() -> None:
    names = sorted(schema.name for schema in await _real_surface().list_tools())

    assert names == [
        "aggregate",
        "compare",
        "describe",
        "filter",
        "get_data",
        "get_skill",
        "get_skill_file",
        "growth",
        "join",
        "rank",
        "resolve_definition",
        "search",
        "share_of",
        "sources",
        "sum_of",
    ]


async def test_the_real_surface_names_each_tools_server() -> None:
    by_name = {schema.name: schema.server for schema in await _real_surface().list_tools()}

    assert by_name["get_data"] == "data-mcp"
    assert by_name["aggregate"] == "analysis-mcp"


async def test_the_real_surface_carries_usable_json_schemas() -> None:
    schemas = {schema.name: schema for schema in await _real_surface().list_tools()}

    aggregate = schemas["aggregate"].parameters

    # What the compiler relies on: the required arguments to hold out or
    # validate, and the closed-object rule that makes an invented argument fail.
    assert aggregate["required"] == ["frame_ref", "aggregation"]
    assert aggregate["additionalProperties"] is False
    assert "frame_ref" in aggregate["properties"]
    assert schemas["get_data"].parameters["required"] == ["spec_id"]


async def test_the_real_surface_declares_a_version_on_every_tool_and_output_schema_on_the_verbs() -> (
    None
):
    # Every tool states its own implementation version. Only the seven
    # analysis-mcp verbs also declare a typed output schema -- data-mcp's
    # structured result is the full `Manifest`, which `server.py`'s own risk
    # note leaves undeclared: deriving and validating that on every call is
    # a risk left untaken.
    schemas = await _real_surface().list_tools()

    assert all(schema.version is not None for schema in schemas)
    analysis_verbs = {schema.name for schema in schemas if schema.server == "analysis-mcp"}
    assert analysis_verbs == {
        "aggregate",
        "compare",
        "filter",
        "growth",
        "join",
        "rank",
        "share_of",
        "sum_of",
    }
    for schema in schemas:
        if schema.server == "analysis-mcp":
            assert schema.output_schema is not None, schema.name
        else:
            assert schema.output_schema is None, schema.name


def test_a_schema_hash_sees_content_and_not_key_order() -> None:
    assert schema_hash({"a": 1, "b": [1, 2]}) == schema_hash({"b": [1, 2], "a": 1})
    assert schema_hash({"a": 1}) != schema_hash({"a": 2})


def test_a_tool_schema_hashes_its_own_parameters() -> None:
    schema = ToolSchema(server="s", name="t", parameters={"type": "object"})

    assert schema.input_schema_hash == schema_hash({"type": "object"})


def test_a_schema_with_an_out_of_range_integer_cannot_be_hashed() -> None:
    # 2**53 is the first integer IEEE 754 double precision cannot represent
    # exactly, so it has no canonical JCS encoding (RFC 8785).
    with pytest.raises(UnhashableSchemaError):
        schema_hash({"maximum": 2**53})

    assert schema_hash({"maximum": 2**53 - 1})


def test_schema_hash_error_names_the_unhashable_schema() -> None:
    assert schema_hash_error({"maximum": 2**53}) is not None
    assert schema_hash_error({"maximum": 2**53 - 1}) is None
    assert schema_hash_error(None) is None


async def test_a_name_served_by_two_servers_raises() -> None:
    first = FastMCP(name="first")
    second = FastMCP(name="second")
    for server in (first, second):

        @server.tool(name="aggregate")
        async def aggregate(frame_ref: dict[str, str]) -> str:
            """Group a stored frame."""
            return "x"

    with pytest.raises(DuplicateToolNameError, match="aggregate"):
        await McpSurface(sources={"first": first, "second": second}).list_tools()


def test_an_in_memory_server_must_be_keyed_by_its_own_name() -> None:
    # The key is what a `ToolId` names the server by, so a mismatch would
    # compile steps against one name and call another.
    with pytest.raises(ValueError, match="named 'analysis-mcp'"):
        McpSurface(sources={"data-mcp": analysis_mcp})


async def test_a_server_that_cannot_be_read_raises_naming_it() -> None:
    with pytest.raises(ToolSurfaceUnavailableError, match=UNREACHABLE_URL):
        await McpSurface(sources={"data-mcp": UNREACHABLE_URL}).list_tools()


async def test_one_unreadable_server_fails_the_whole_read() -> None:
    # Returning the servers that happened to answer would let a workflow
    # compile against a partial surface, and refuse a step a missing server
    # actually serves.
    with pytest.raises(ToolSurfaceUnavailableError):
        await McpSurface(
            sources={"analysis-mcp": analysis_mcp, "data-mcp": UNREACHABLE_URL}
        ).list_tools()


class _SleepBeforeListing(Middleware):
    """Delays a server's own `tools/list` answer, to prove two sources are read
    concurrently rather than one after the other."""

    def __init__(self, delay_s: float) -> None:
        self._delay_s = delay_s

    async def on_list_tools(self, context, call_next):
        await asyncio.sleep(self._delay_s)
        return await call_next(context)


def _slow_listing_server(name: str, delay_s: float) -> FastMCP:
    server = FastMCP(name=name)
    server.add_middleware(_SleepBeforeListing(delay_s))

    @server.tool(name=f"{name}_tool")
    async def _tool() -> str:
        return "x"

    return server


async def test_list_tools_reads_every_source_concurrently() -> None:
    first = _slow_listing_server("first", 0.1)
    second = _slow_listing_server("second", 0.1)

    start = time.monotonic()
    tools = await McpSurface(sources={"first": first, "second": second}).list_tools()
    elapsed = time.monotonic() - start

    assert sorted(tool.name for tool in tools) == ["first_tool", "second_tool"]
    # Serial reads would take roughly 0.2s; concurrent reads take roughly 0.1s.
    assert elapsed < 0.18


def _surface_with_a_slow_tool() -> FastMCP:
    server = FastMCP(name="data-mcp")

    @server.tool(name="get_data")
    async def get_data() -> str:
        await asyncio.sleep(0.2)
        return "slow"

    @server.tool(name="search")
    async def search() -> str:
        return "fast"

    return server


async def test_a_per_tool_call_timeout_override_reaches_the_client() -> None:
    server = _surface_with_a_slow_tool()
    surface = McpSurface(
        sources={"data-mcp": server},
        call_timeouts={ToolId(server="data-mcp", name="get_data"): 0.01},
    )

    with pytest.raises(ToolSurfaceUnavailableError, match="get_data"):
        await surface.call_tool(tool=ToolId(server="data-mcp", name="get_data"), arguments={})

    # A tool with no override, on the same surface, still answers at the default.
    result = await surface.call_tool(tool=ToolId(server="data-mcp", name="search"), arguments={})
    assert result.is_error is False


async def test_a_per_tool_timeout_does_not_discard_the_held_client() -> None:
    """The previous test proves the held client still answers after a
    timeout; this one asserts on the internal identity directly, so a
    regression that discarded and reconnected -- silently paying back the
    round trips this held client exists to avoid -- would fail here even if
    it still happened to answer correctly."""
    server = _surface_with_a_slow_tool()
    surface = McpSurface(
        sources={"data-mcp": server},
        call_timeouts={ToolId(server="data-mcp", name="get_data"): 0.01},
    )

    with pytest.raises(ToolSurfaceUnavailableError):
        await surface.call_tool(tool=ToolId(server="data-mcp", name="get_data"), arguments={})
    held = _held(surface, "data-mcp")

    await surface.call_tool(tool=ToolId(server="data-mcp", name="search"), arguments={})

    assert held is not None
    assert _held(surface, "data-mcp") is held


async def test_a_non_timeout_failure_discards_the_held_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A session that is actually broken -- not just slow -- must not keep
    answering every later call with the same failure, so `call_tool`
    discards it and the next call reconnects."""
    surface = McpSurface(sources={"data-mcp": _answering_server()})
    await surface.call_tool(tool=ToolId(server="data-mcp", name="search"), arguments={"parts": []})
    broken = _held(surface, "data-mcp")

    real_call_tool = Client.call_tool
    calls = 0

    async def _fails_once(self: Client, *args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("transport closed")
        return await real_call_tool(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Client, "call_tool", _fails_once)

    with pytest.raises(ToolSurfaceUnavailableError, match="transport closed"):
        await surface.call_tool(
            tool=ToolId(server="data-mcp", name="search"), arguments={"parts": []}
        )
    assert _held(surface, "data-mcp") is None  # discarded, not yet replaced

    # The next call reconnects a fresh client and actually serves it.
    result = await surface.call_tool(
        tool=ToolId(server="data-mcp", name="search"), arguments={"parts": []}
    )
    assert result.is_error is False
    assert _held(surface, "data-mcp") is not broken


async def test_a_non_timeout_failure_does_not_tear_down_a_concurrent_call_on_the_same_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two calls to one server in flight at once on the shared held client:
    `describe` fails for a reason other than its own timeout while `search`
    is still awaiting the transport. `_discard_call_client`'s
    `client.__aexit__` must not tear the session down under `search` --
    only once `search` itself is done with it."""
    surface = McpSurface(sources={"data-mcp": _answering_server()})
    # Warm the held client first, so both calls below share one session.
    await surface.call_tool(tool=ToolId(server="data-mcp", name="search"), arguments={"parts": []})

    real_call_tool = Client.call_tool
    search_started = asyncio.Event()
    release_search = asyncio.Event()

    async def _scripted(self: Client, name: str, *args: object, **kwargs: object) -> object:
        if name == "search":
            search_started.set()
            await release_search.wait()
            return await real_call_tool(self, name, *args, **kwargs)  # type: ignore[arg-type]
        if name == "describe":
            await search_started.wait()
            raise RuntimeError("transport closed")
        return await real_call_tool(self, name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Client, "call_tool", _scripted)

    async def _run_describe() -> None:
        with pytest.raises(ToolSurfaceUnavailableError, match="transport closed"):
            await surface.call_tool(
                tool=ToolId(server="data-mcp", name="describe"), arguments={"ids": []}
            )
        # By now the failed call has done whatever teardown it is going to
        # do; only release `search` after that, so a buggy teardown lands
        # while `search` is still mid-call rather than before it starts.
        release_search.set()

    search_task = asyncio.create_task(
        surface.call_tool(tool=ToolId(server="data-mcp", name="search"), arguments={"parts": []})
    )
    describe_task = asyncio.create_task(_run_describe())

    result = await asyncio.wait_for(search_task, timeout=5)
    await describe_task

    assert result.is_error is False


async def test_call_tool_holds_one_client_per_source_across_calls() -> None:
    """`call_tool` reads no staleness benefit from a fresh session -- the
    tool it invokes was already resolved by an earlier `list_tools` -- so it
    holds one `Client` per source instead of opening one per call."""
    surface = McpSurface(sources={"data-mcp": _answering_server()})

    await surface.call_tool(
        tool=ToolId(server="data-mcp", name="search"), arguments={"parts": [{"variable": "x"}]}
    )
    first = _held(surface, "data-mcp")
    await surface.call_tool(
        tool=ToolId(server="data-mcp", name="search"), arguments={"parts": [{"variable": "y"}]}
    )

    assert first is not None
    assert _held(surface, "data-mcp") is first


async def test_call_tool_gives_each_source_its_own_held_client() -> None:
    # `ToolId.server` is a `Literal["data-mcp", "analysis-mcp"]`, so the two
    # sources have to be named that way even though this test only cares
    # that each gets its own client.
    data_server = _slow_listing_server("data-mcp", 0.0)
    analysis_server = _slow_listing_server("analysis-mcp", 0.0)
    surface = McpSurface(sources={"data-mcp": data_server, "analysis-mcp": analysis_server})

    await surface.call_tool(tool=ToolId(server="data-mcp", name="data-mcp_tool"), arguments={})
    await surface.call_tool(
        tool=ToolId(server="analysis-mcp", name="analysis-mcp_tool"), arguments={}
    )

    data_client = _held(surface, "data-mcp")
    analysis_client = _held(surface, "analysis-mcp")
    assert data_client is not None
    assert analysis_client is not None
    assert data_client is not analysis_client


async def test_call_tool_holds_one_client_per_caller() -> None:
    callers = iter(["conversation:a", "conversation:b", "conversation:a"])
    surface = McpSurface(sources={"data-mcp": _answering_server()}, caller=lambda: next(callers))

    for _ in range(3):
        await surface.call_tool(
            tool=ToolId(server="data-mcp", name="search"), arguments={"parts": []}
        )

    a = _held(surface, "data-mcp", "conversation:a")
    b = _held(surface, "data-mcp", "conversation:b")
    assert a is not None
    assert b is not None
    assert a is not b


async def test_aclose_disconnects_and_forgets_every_held_call_client() -> None:
    surface = McpSurface(sources={"data-mcp": _answering_server()})
    await surface.call_tool(tool=ToolId(server="data-mcp", name="search"), arguments={"parts": []})
    client = _held(surface, "data-mcp")

    await surface.aclose()

    assert client is not None
    assert _held(surface, "data-mcp") is None
    assert client.is_connected() is False


def test_a_url_source_is_accepted_without_connecting() -> None:
    assert McpSurface(sources={"data-mcp": UNREACHABLE_URL}) is not None


def test_a_url_source_carries_the_resolved_bearer_token() -> None:
    # A URL source is the real, gated mount (`harness.agents.build_tool_surface`
    # builds one from `settings.data_mcp_url` / `settings.analysis_mcp_url`); it
    # must carry the same token `common.auth.apply_gate` resolved for the process.
    apply_gate(FastMCP(), token="test-token", disabled=False)

    assert _auth_for(UNREACHABLE_URL) == "test-token"


def test_an_in_memory_source_carries_no_auth() -> None:
    # The in-memory transport bypasses HTTP entirely, so it bypasses the gate
    # too; passing it a bearer would only make `fastmcp.Client` raise, since
    # that transport doesn't support `auth=`.
    apply_gate(FastMCP(), token="test-token", disabled=False)

    assert _auth_for(_answering_server()) is None


async def test_the_fixture_surface_serves_what_it_was_given() -> None:
    schema = ToolSchema(server="s", name="aggregate", parameters={"type": "object"})

    served = await FixtureSurface(schemas=[schema]).list_tools()

    assert served == [schema]


def _answering_server() -> FastMCP:
    server = FastMCP(name="data-mcp")

    @server.tool(name="search")
    async def search(parts: list[dict[str, str]]) -> ToolResult:
        """Answer with a digest, a structured answer and a `_meta` envelope."""
        return ToolResult(
            content="one reading",
            structured_content={"status": "definite", "n": len(parts)},
            meta={"org.example/envelope": {"parts": parts}},
        )

    @server.tool(name="describe")
    async def describe(ids: list[str]) -> ToolResult:
        """Refuse, the way a data-mcp tool does."""
        return ToolResult(
            content="refused", structured_content={"code": "node_not_held"}, is_error=True
        )

    return server


async def test_call_tool_returns_the_digest_the_structured_content_and_the_meta() -> None:
    surface = McpSurface(sources={"data-mcp": _answering_server()})

    result = await surface.call_tool(
        tool=ToolId(server="data-mcp", name="search"), arguments={"parts": [{"variable": "x"}]}
    )

    assert result.is_error is False
    assert result.structured_content == {"status": "definite", "n": 1}
    assert result.text == "one reading"
    assert result.meta is not None
    assert result.meta["org.example/envelope"] == {"parts": [{"variable": "x"}]}


async def test_call_tool_returns_a_refusal_rather_than_raising() -> None:
    # A refusal is the tool's answer; its typed detail is what a caller acts on.
    surface = McpSurface(sources={"data-mcp": _answering_server()})

    result = await surface.call_tool(
        tool=ToolId(server="data-mcp", name="describe"), arguments={"ids": ["x"]}
    )

    assert result.is_error is True
    assert result.structured_content == {"code": "node_not_held"}
    assert "org.example/envelope" not in (result.meta or {})


async def test_call_tool_on_an_uncovered_server_raises_naming_the_covered_ones() -> None:
    surface = McpSurface(sources={"data-mcp": _answering_server()})

    with pytest.raises(
        ToolSurfaceUnavailableError, match=r"no server named 'analysis-mcp'.*'data-mcp'"
    ):
        await surface.call_tool(tool=ToolId(server="analysis-mcp", name="aggregate"), arguments={})


async def test_call_tool_on_an_unreachable_server_raises_naming_the_tool() -> None:
    surface = McpSurface(sources={"data-mcp": UNREACHABLE_URL})

    with pytest.raises(ToolSurfaceUnavailableError, match="could not call 'search'"):
        await surface.call_tool(tool=ToolId(server="data-mcp", name="search"), arguments={})


async def test_the_fixture_surface_answers_from_its_canned_responses() -> None:
    canned = ToolCallResult(is_error=False, structured_content={"status": "definite"})
    surface = FixtureSurface(schemas=[], responses={"search": canned})

    assert (
        await surface.call_tool(tool=ToolId(server="data-mcp", name="search"), arguments={})
        == canned
    )
    with pytest.raises(ToolSurfaceUnavailableError, match="no canned answer for 'describe'"):
        await surface.call_tool(tool=ToolId(server="data-mcp", name="describe"), arguments={})


def _reporting_server() -> FastMCP:
    """A data-mcp stand-in whose `get_data` reports twice while it waits, then answers."""
    server = FastMCP(name="data-mcp")

    @server.tool(name="get_data")
    async def get_data(ctx: Context) -> ToolResult:
        await ctx.report_progress(progress=0, total=None, message="submitted")
        await ctx.report_progress(progress=12.5, total=None, message="running")
        return ToolResult(content="fetched", structured_content={"status": "ok"})

    return server


async def test_call_tool_hands_the_servers_progress_reports_to_on_progress() -> None:
    surface = McpSurface(sources={"data-mcp": _reporting_server()})
    reports: list[ProgressReport] = []

    result = await surface.call_tool(
        tool=ToolId(server="data-mcp", name="get_data"),
        arguments={},
        on_progress=reports.append,
    )

    assert reports == [
        ProgressReport(progress=0, total=None, message="submitted"),
        ProgressReport(progress=12.5, total=None, message="running"),
    ]
    assert result.structured_content == {"status": "ok"}


async def test_a_raising_progress_callback_does_not_fail_the_call(
    caplog: pytest.LogCaptureFixture,
) -> None:
    surface = McpSurface(sources={"data-mcp": _reporting_server()})
    heard: list[ProgressReport] = []

    def _raises(report: ProgressReport) -> None:
        heard.append(report)
        raise RuntimeError("relay closed")

    with caplog.at_level(logging.WARNING, logger="compiler.surface"):
        result = await surface.call_tool(
            tool=ToolId(server="data-mcp", name="get_data"), arguments={}, on_progress=_raises
        )

    assert result.is_error is False
    assert result.structured_content == {"status": "ok"}
    assert [report.message for report in heard] == ["submitted", "running"]
    # The surface's own guard caught both, not the MCP session's fallback.
    assert [record.getMessage() for record in caplog.records] == [
        "a progress callback raised; dropping its report"
    ] * 2


async def test_a_call_without_on_progress_leaves_the_clients_default_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    surface = McpSurface(sources={"data-mcp": _reporting_server()})
    real_call_tool = Client.call_tool
    handlers: list[object] = []

    async def _spy(self: Client, *args: object, **kwargs: object) -> object:
        handlers.append(kwargs.get("progress_handler"))
        return await real_call_tool(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Client, "call_tool", _spy)

    result = await surface.call_tool(tool=ToolId(server="data-mcp", name="get_data"), arguments={})

    assert handlers == [None]
    assert result.is_error is False
    assert result.structured_content == {"status": "ok"}

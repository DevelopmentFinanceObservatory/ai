"""What Pydantic AI's MCP client does that arm B's `McpSurface` path does not, or does
differently: the tool list it caches, the capabilities it declares, the `_meta` and
headers it sends.
"""

from __future__ import annotations

from typing import Any

import pytest
from common.telemetry import SESSION_META_KEY
from fastmcp.server.dependencies import fastmcp_request_ctx, get_http_headers
from fastmcp.server.middleware import Middleware
from pai_plan_mcp import PlanDeps, open_plan, plan_agent, plan_toolsets
from pydantic_ai.exceptions import UserError
from rig import Rig, Script, call, say, serve

CALLER = "x-dfo-caller"


class Spy(Middleware):
    """What plan-mcp's server sees of the client: its `tools/list` requests, and for each
    `tools/call` the request `_meta`, the headers and the capabilities the client declared."""

    def __init__(self) -> None:
        self.active = True
        self.lists = 0
        self.calls: list[dict[str, Any]] = []

    async def on_list_tools(self, context: Any, call_next: Any) -> Any:
        self.lists += self.active
        return await call_next(context)

    async def on_call_tool(self, context: Any, call_next: Any) -> Any:
        if not self.active:
            return await call_next(context)
        session = context.fastmcp_context.session
        capabilities = getattr(getattr(session, "client_params", None), "capabilities", None)
        request = fastmcp_request_ctx.get()
        self.calls.append(
            {
                # The wire `_meta`, as `common.mcp_telemetry` reads the call's session.
                "meta": dict(request.meta) if request is not None and request.meta else None,
                "caller": get_http_headers(include={CALLER}).get(CALLER),
                "capabilities": capabilities.model_dump(exclude_none=True) if capabilities else {},
            }
        )
        return await call_next(context)


async def propose_twice(rig: Rig, *, hold: bool, **options: Any) -> Spy:
    """Two runs, each making one `propose_plan` call, with or without the connection held
    open across them."""
    spy = Spy()
    rig.server.mcp.add_middleware(spy)
    mcp, view = plan_toolsets(rig.source, **options)
    for _ in range(2):
        script = Script([call("propose_plan", {}, id="c"), say("done")])
        agent = plan_agent(script.model, view, instruction="")
        if hold and not mcp.is_running:
            await mcp.__aenter__()
        await agent.run("brief", deps=PlanDeps("ps_none"))
    if hold:
        await mcp.__aexit__(None, None, None)
    spy.active = False
    return spy


async def test_the_tool_list_is_cached_while_the_connection_is_held(rig: Rig) -> None:
    held = await propose_twice(rig, hold=True)
    reread = await propose_twice(rig, hold=False)
    print(
        rig.transport,
        "tools/list requests over two runs: held",
        held.lists,
        "not held",
        reread.lists,
    )
    if rig.transport == "inproc":
        assert (held.lists, reread.lists) == (1, 2)
    else:
        # The FastMCP client reads the list again before each call it makes over HTTP.
        assert held.lists >= len(held.calls) and reread.lists >= len(reread.calls)


async def test_by_default_it_sends_no_session_meta_no_caller_and_declares_no_capabilities(
    rig: Rig,
) -> None:
    spy = await propose_twice(rig, hold=True)
    print(rig.transport, "server saw:", spy.calls[0])
    assert all(SESSION_META_KEY not in (c["meta"] or {}) for c in spy.calls)
    assert all(c["caller"] is None for c in spy.calls)
    assert all(not c["capabilities"] for c in spy.calls)


async def test_the_meta_and_the_caller_header_can_be_sent(rig: Rig) -> None:
    options: dict[str, Any] = {"meta": lambda: {SESSION_META_KEY: "langfuse-session"}}
    if rig.transport == "http":
        options["headers"] = {CALLER: "caller-1"}
    spy = await propose_twice(rig, hold=True, **options)
    print(rig.transport, "server saw:", spy.calls[0])
    assert all(c["meta"][SESSION_META_KEY] == "langfuse-session" for c in spy.calls)
    assert all(c["caller"] == ("caller-1" if rig.transport == "http" else None) for c in spy.calls)


async def test_a_caller_header_taken_from_the_run_cannot_open_the_session_outside_a_run() -> None:
    """A `headers` function of the run's deps gives each run its own connection, which the
    harness's `open_plan` and `record_answers` (outside any run) cannot use."""
    async with serve("http") as rig:
        mcp, _ = plan_toolsets(rig.source, headers=lambda ctx: {CALLER: "caller-1"})
        with pytest.raises(
            UserError, match="can only connect to the server as part of an agent run"
        ):
            await open_plan(mcp, brief="b", asks=[], searches=[])

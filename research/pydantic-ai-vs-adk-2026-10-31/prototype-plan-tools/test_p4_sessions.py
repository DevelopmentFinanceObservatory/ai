"""P4: plan sessions stay apart. Two sessions interleaved on one client, and two
sessions on two clients, never see each other's sources, plans or runs.

The two-client case runs over Streamable HTTP in process, each client naming
its own caller in `x-dfo-caller`, the header the public `/mcp` keys callers on.
The control keys sessions on the MCP transport session id."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import httpx2
import pytest
from checks import KeyedSearch, call, client_of, scripted_server
from fastmcp import Client, Context
from fastmcp.client.transports import StreamableHttpTransport
from plan_tools import SESSION_UNKNOWN, VERSION_NOT_READY, VERSION_SUPERSEDED
from planner.plan import load_plan, pin_sources
from scripted import PROPOSAL, amended
from server import CALLER_HEADER, PlanMcp, SessionKey
from tests.orchestrator_fakes import definite_envelope

KENYA = "spec_" + "a" * 32
OTHER = "spec_" + "b" * 32
# Two questions whose searches return two datasets, one each.
OPENINGS = {
    "kenya": {
        "brief": "Life expectancy in Kenya since 1960",
        "asks": ["Life expectancy in Kenya since 1960"],
        "searches": [
            {
                "query": "Life expectancy in Kenya",
                "parts": [{"variable": "Life expectancy in Kenya"}],
            }
        ],
    },
    "other": {
        "brief": "Kenya's life expectancy at birth since 1960",
        "asks": ["Kenya's life expectancy at birth since 1960"],
        "searches": [
            {"query": "Kenya's life expectancy", "parts": [{"variable": "Kenya's life expectancy"}]}
        ],
    },
}
RESOLVER = KeyedSearch(
    {
        "Life expectancy in Kenya": definite_envelope(spec_id=KENYA),
        "Kenya's life expectancy": definite_envelope(
            spec_id=OTHER, sentence="Kenya's life expectancy"
        ),
    }
)


def _proposal(name: str) -> dict[str, Any]:
    return {**PROPOSAL, "question": OPENINGS[name]["brief"]}


async def _interleaved(server: PlanMcp, kenya: Client, other: Client) -> dict[str, Any]:
    """Open, propose, amend and run two sessions, one step of each in turn."""
    a = (await call(kenya, "start_plan", OPENINGS["kenya"]))["plan_session"]
    b = (await call(other, "start_plan", OPENINGS["other"]))["plan_session"]
    seen: dict[str, Any] = {"a": a, "b": b}
    seen["a1"] = await call(kenya, "propose_plan", {"plan_session": a, **_proposal("kenya")})
    seen["b1"] = await call(other, "propose_plan", {"plan_session": b, **_proposal("other")})
    seen["b2"] = await call(other, "propose_plan", {"plan_session": b, **amended(1)})
    seen["a_run"] = await call(
        kenya, "run_plan", {"plan_session": a, "version": 1, "approved": True}
    )
    seen["b_run1"] = await call(
        other, "run_plan", {"plan_session": b, "version": 1, "approved": True}
    )
    seen["a_run2"] = await call(
        kenya, "run_plan", {"plan_session": a, "version": 2, "approved": True}
    )
    seen["b_run"] = await call(
        other, "run_plan", {"plan_session": b, "version": 2, "approved": True}
    )
    return seen


def _apart_faults(server: PlanMcp, seen: dict[str, Any]) -> list[str]:
    """Every way the two sessions `_interleaved` drove saw each other's work."""
    faults = []
    sessions = server.service.sessions
    a, b = sessions.get(seen["a"]), sessions.get(seen["b"])
    if a is None or b is None or a is b:
        return [f"sessions {seen['a']} and {seen['b']} are not two sessions"]
    if list(pin_sources(a.state)) != [KENYA] or list(pin_sources(b.state)) != [OTHER]:
        faults.append(f"sources {list(pin_sources(a.state))} {list(pin_sources(b.state))}")
    plan_a, plan_b = load_plan(a.state), load_plan(b.state)
    if plan_a is None or (plan_a.version, plan_a.question) != (1, OPENINGS["kenya"]["brief"]):
        faults.append(f"session a holds {plan_a and (plan_a.version, plan_a.question)}")
    if plan_b is None or (plan_b.version, plan_b.question) != (2, OPENINGS["other"]["brief"]):
        faults.append(f"session b holds {plan_b and (plan_b.version, plan_b.question)}")
    if OTHER in a.first_message or KENYA in b.first_message:
        faults.append("a planner message names the other session's source")
    if {e.id for e in a.events} & {e.id for e in b.events}:
        faults.append("the sessions share events")
    if seen["a_run2"].get("code") != VERSION_NOT_READY:
        faults.append(f"session a ran b's version 2: {seen['a_run2']}")
    if seen["b_run1"].get("code") != VERSION_SUPERSEDED:
        faults.append(f"session b's version 1: {seen['b_run1']}")
    for run, own, other in (("a_run", KENYA, OTHER), ("b_run", OTHER, KENYA)):
        record = (seen[run].get("hand_back") or {}).get("plan_record", "")
        if own not in record or other in record:
            faults.append(f"{run} hands back {record[:80]!r}")
    if len(a.runs) != 1 or len(b.runs) != 1:
        faults.append(f"runs {list(a.runs)} {list(b.runs)}")
    return faults


async def test_p4_two_sessions_interleaved_on_one_client_stay_apart() -> None:
    server = scripted_server(resolver=RESOLVER).server
    async with client_of(server) as client:
        seen = await _interleaved(server, client, client)
    assert _apart_faults(server, seen) == []


def _http_client(app: Any, caller: str) -> Client:
    def factory(**kwargs: Any) -> httpx2.AsyncClient:
        return httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app=app), base_url="http://plan.test", **kwargs
        )

    return Client(
        StreamableHttpTransport(
            url="http://plan.test/mcp",
            headers={CALLER_HEADER: caller},
            httpx_client_factory=cast(Any, factory),
        )
    )


async def test_p4_two_sessions_on_two_clients_stay_apart(
    running_lifespan: Callable[[Any], Any],
) -> None:
    server = scripted_server(resolver=RESOLVER).server
    app = server.mcp.http_app(path="/mcp")
    async with (
        running_lifespan(app),
        _http_client(app, "caller-kenya") as kenya,
        _http_client(app, "caller-other") as other,
    ):
        seen = await _interleaved(server, kenya, other)
        borrowed = await call(other, "propose_plan", {"plan_session": seen["a"], **amended(1)})
        borrowed_run = await call(
            other, "run_plan", {"plan_session": seen["a"], "version": 1, "approved": True}
        )
    assert _apart_faults(server, seen) == []
    assert borrowed["code"] == SESSION_UNKNOWN
    assert borrowed_run["code"] == SESSION_UNKNOWN
    assert load_plan(server.service.sessions[seen["a"]].state).version == 1  # type: ignore[union-attr]


async def test_p4_control_sessions_keyed_on_the_transport_session_lose_their_work() -> None:
    # FastMCP 4 answers a new `ctx.session_id` on every request of an in-memory
    # client too, so the call after `start_plan` finds no session.
    server = scripted_server(resolver=RESOLVER, key=SessionKey.TRANSPORT).server
    async with client_of(server) as client:
        seen = await _interleaved(server, client, client)
    faults = _apart_faults(server, seen)
    print("P4 control, one client:", faults, "transport ids:", len(server.transport_keys))
    assert faults
    assert seen["a1"]["code"] == SESSION_UNKNOWN


async def test_p4_control_a_transport_session_stable_per_connection_mixes_two_sessions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A transport whose session id holds for the connection, as FastMCP's did
    # before 4, keys both of one client's plan sessions on one id.
    monkeypatch.setattr(Context, "session_id", property(lambda self: "one-connection"))
    server = scripted_server(resolver=RESOLVER, key=SessionKey.TRANSPORT).server
    async with client_of(server) as client:
        seen = await _interleaved(server, client, client)
    faults = _apart_faults(server, seen)
    print("P4 control, stable transport id:", faults)
    assert any(fault.startswith("session a holds") for fault in faults)


async def test_p4_control_sessions_keyed_on_the_transport_session_lose_their_work_over_http(
    running_lifespan: Callable[[Any], Any],
) -> None:
    server = scripted_server(resolver=RESOLVER, key=SessionKey.TRANSPORT).server
    app = server.mcp.http_app(path="/mcp")
    async with running_lifespan(app), _http_client(app, "caller-kenya") as kenya:
        opened = await call(kenya, "start_plan", OPENINGS["kenya"])
        proposed = await call(
            kenya, "propose_plan", {"plan_session": opened["plan_session"], **_proposal("kenya")}
        )
    print("P4 control, HTTP:", proposed, "transport ids:", list(server.transport_keys))
    assert proposed["code"] == SESSION_UNKNOWN

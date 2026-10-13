"""Fixtures shared across the harness suites.

`decision` and `requirement` build a valid `Decision` and `DatasetRequirement`
with any field overridden, so a test that cares about one field states only
that field. `pinning_decision` builds a `Decision`, resolved by the user by
default, whose `fixes` records a pinned `source_spec_id`, as a stored plan
states it. `derived_requirement` builds a valid
`AggregateRequirement` on top of `requirement`'s dataset, and `requirements`
pairs the two into a plan's minimal, valid `evidence_requirements`. None of
these are trivially constructible -- a `Decision`'s `chosen` value has to be
one of `options`, a requirement has to cite a decision by slot, and a plan's
dataset requirements each have to be read by a derived one -- so an inline
literal in every test would repeat those pairings.

The autouse fixtures run every test with the studio shared secret unset, and
reset `common.auth`'s process-wide bearer-token policy after every test.
`Settings` reads the repository's `.env`, so a developer who has set
`STUDIO_API_TOKEN` there would otherwise have every test request's bearer
token refused. The environment variable takes precedence over the env file,
and a blank value is the setting unset.

The workspace-root `conftest.py` carries the same `common.auth` reset, but
`services/harness/pyproject.toml` sets `[tool.pytest.ini_options]`, which
makes pytest's rootdir `services/harness` whenever this suite is invoked
from that directory without `--confcutdir` pointed at the workspace root.
conftest.py collection stops at the rootdir in that case, so the
workspace-root fixture never runs and `common.auth`'s first-caller-wins
policy leaks from whichever test resolves it first into every test that
follows in the same process. Resetting it here too makes this suite
self-isolating regardless of how it is invoked.

`source` builds the `SelectedSource` a discovered dataset resolves to, for the
compiler, translator and executor suites alike.

`scripted_llm` and `run_to_completion` drive an agent through a real ADK
`Runner` with a stand-in model, so a test can assert on what ADK itself does --
the root-agent mode check, or the contents a second turn would carry -- rather
than restating ADK's rules and hoping they hold. `running_lifespan` and
`mounted_mcp_transport` drive the composed FastAPI app's lifespan by hand and
reach a mounted MCP server through it, for tests that cross the seam between
services. All are fixtures rather than importable helpers because a test module
importing its own conftest by name is ambiguous when the workspace root has one
too.
"""

from __future__ import annotations

import asyncio
from collections.abc import (
    AsyncGenerator,
    AsyncIterator,
    Awaitable,
    Callable,
    Mapping,
    MutableMapping,
    Sequence,
)
from contextlib import asynccontextmanager
from typing import Any, cast

import httpx2
import pytest
from common import auth as mcp_auth
from compiler.sources import member_digest
from data_mcp import resolution
from fastapi import FastAPI
from fastmcp.client.transports import StreamableHttpTransport
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from harness.conversation import (
    NO_CONVERSATION,
    current_conversation_id,
    current_conversation_key,
)
from mcp.shared._httpx_utils import McpHttpClientFactory
from planner.plan import (
    AggregateRequirement,
    DatasetRequirement,
    Decision,
    DecisionFixes,
    Entity,
    EvidenceRequirement,
    GroupBy,
    Period,
    Reduction,
    ResolvedBy,
    SpecId,
    Unit,
)
from planner.workflow import ResolvedEntity, SelectedSource
from pydantic import Field

SPEC_KENYA = "spec_" + "a" * 32


@pytest.fixture(autouse=True)
def _no_shared_secret_from_the_developer_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run every test with `STUDIO_API_TOKEN` unset."""
    monkeypatch.setenv("STUDIO_API_TOKEN", "")


@pytest.fixture(autouse=True)
def _no_graph_resolve(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolve no entity name through the graph unless a test says how.

    The translator resolves names that match no label in process
    (`translator.discovery.resolve_entity_refs`), which would otherwise reach
    whatever upstream the process holds. A test that wants outcomes patches
    `data_mcp.resolution.resolve_entities` itself.
    """

    async def resolve_nothing(
        scope: Mapping[str, Sequence[str]],
    ) -> dict[str, dict[str, resolution.ResolutionOutcome]]:
        return {
            role: {name: resolution.NotFound(near_matches=[]) for name in names}
            for role, names in scope.items()
        }

    monkeypatch.setattr(resolution, "resolve_entities", resolve_nothing)


@pytest.fixture(autouse=True)
def _reset_mcp_auth_gate() -> Any:
    """Give every test a known starting state for `common.auth`'s policy.

    `common.auth` resolves its bearer-token policy once per process, first
    caller wins, and this suite (like every workspace member's) can run in
    the same process as every other test file. A test that resolves it
    (directly, or by building any FastMCP app that calls `apply_gate`) must
    not leave that resolution for a later test to see -- see this module's
    docstring for why the workspace-root conftest's own reset doesn't
    reliably cover this suite.
    """
    yield
    mcp_auth.reset()


@pytest.fixture(autouse=True)
def _reset_conversation_identity() -> Any:
    """Leave no conversation set after a test.

    A test function that is not a coroutine runs in the session's own context,
    so a `current_conversation_id.set(...)` it makes would name that
    conversation on every MCP call of the tests after it.
    """
    yield
    current_conversation_id.set(NO_CONVERSATION)
    current_conversation_key.set(None)


@pytest.fixture
def decision() -> Callable[..., Decision]:
    """Return a factory for a valid `Decision` with overridable fields."""

    def _decision(**overrides: Any) -> Decision:
        fields: dict[str, Any] = {
            "slot": "measure",
            "description": "What to measure Kenya's dependence by.",
            "options": ["annual debt service", "total debt stock"],
            "chosen": "annual debt service",
            "resolved_by": ResolvedBy.PLANNER_DEFAULT,
            "rationale": "Debt service is the standard measure for near-term dependence.",
        }
        return Decision(**(fields | overrides))

    return _decision


@pytest.fixture
def pinning_decision(decision: Callable[..., Decision]) -> Callable[..., Decision]:
    """Return a factory for a `Decision`, resolved by the user by default, that fixes a pinned `source_spec_id`.

    A pin needs no decision; this one records the pin in `fixes`, the way
    a stored plan or `propose_plan`'s own measure decision states it.
    A `DecisionDraft` never names `source_spec_id`, so `decision_drafts.drafts`
    drops it.
    """

    def _pinning_decision(spec_id: SpecId, **overrides: Any) -> Decision:
        fields: dict[str, Any] = {
            "resolved_by": ResolvedBy.USER,
            "fixes": DecisionFixes(source_spec_id=spec_id),
        }
        return decision(**(fields | overrides))

    return _pinning_decision


@pytest.fixture
def requirement() -> Callable[..., DatasetRequirement]:
    """Return a factory for a valid `DatasetRequirement` citing the `measure` decision."""

    def _requirement(**overrides: Any) -> DatasetRequirement:
        fields: dict[str, Any] = {
            "id": "annual_debt_service",
            "decision_ids": ["measure"],
            "measure": "external debt service paid",
            "entities": [Entity(name="Kenya", role="debtor")],
            "period": Period(start=2015, end=2024),
            "unit": Unit(currency="USD", prices="current"),
        }
        return DatasetRequirement(**(fields | overrides))

    return _requirement


@pytest.fixture
def derived_requirement() -> Callable[..., AggregateRequirement]:
    """Return a factory for a valid `AggregateRequirement` reading `requirement`'s dataset."""

    def _derived_requirement(**overrides: Any) -> AggregateRequirement:
        fields: dict[str, Any] = {
            "id": "annual_totals",
            "decision_ids": ["measure"],
            "input_id": "annual_debt_service",
            "reduction": Reduction.SUM,
            "group_by": (GroupBy.PERIOD,),
        }
        return AggregateRequirement(**(fields | overrides))

    return _derived_requirement


@pytest.fixture
def requirements(
    requirement: Callable[..., DatasetRequirement],
    derived_requirement: Callable[..., AggregateRequirement],
) -> tuple[EvidenceRequirement, ...]:
    """Return a minimal, valid pair: a dataset requirement and one that reads it."""
    return (requirement(), derived_requirement())


class ScriptedLlm(BaseLlm):
    """Answers with a scripted function call, then with a final text response.

    Leaving `tool_name` unset answers with the text straight away, for a run
    that needs no tool call to reach the end. Every `LlmRequest` lands in
    `captured_requests`, so a test can inspect exactly what the model would see
    on its next turn.
    """

    model: str = "scripted-llm"
    tool_name: str | None = None
    tool_args: dict[str, Any] = Field(default_factory=dict)
    captured_requests: list[Any] = Field(default_factory=list)

    async def generate_content_async(
        self, llm_request: Any, stream: bool = False
    ) -> AsyncGenerator[LlmResponse]:
        self.captured_requests.append(llm_request)
        if self.tool_name is not None and len(self.captured_requests) == 1:
            yield LlmResponse(
                content=types.Content(
                    role="model",
                    parts=[types.Part.from_function_call(name=self.tool_name, args=self.tool_args)],
                )
            )
        else:
            yield LlmResponse(
                content=types.Content(role="model", parts=[types.Part.from_text(text="done")])
            )


@pytest.fixture
def scripted_llm() -> type[ScriptedLlm]:
    """Return the stand-in model class, constructed per test with its own script."""
    return ScriptedLlm


@pytest.fixture
def run_to_completion() -> Callable[..., Awaitable[list[Any]]]:
    """Return a callable driving `agent` through a real `Runner` to the end.

    Hands back every event the run produced, so a caller can assert the run
    happened at all: ADK enforces its root-agent rules inside `run_async`
    rather than at construction, so only a real run exercises them.
    """

    async def _run(
        *, agent: Any, session_service: InMemorySessionService, app_name: str
    ) -> list[Any]:
        await session_service.create_session(app_name=app_name, user_id="user-1", session_id="s-1")
        runner = Runner(agent=agent, app_name=app_name, session_service=session_service)
        return [
            event
            async for event in runner.run_async(
                user_id="user-1",
                session_id="s-1",
                new_message=types.Content(role="user", parts=[types.Part.from_text(text="go")]),
            )
        ]

    return _run


KENYA = ResolvedEntity(role="destination", ref="country/KEN", label="Kenya")
GHANA = ResolvedEntity(role="destination", ref="country/GHA", label="Ghana")


@pytest.fixture
def stored_with_member_list() -> Callable[[SelectedSource], dict[str, Any]]:
    """Return a function dumping a source as the release that listed members stored it.

    That release's `SelectedSource` held `stat_vars`, `stat_var_truncated`
    and `pinned`, which the current one rejects, and no `served_count`.
    """

    def _stored(source: SelectedSource) -> dict[str, Any]:
        stored = source.model_dump(mode="json")
        del stored["served_count"]
        return stored | {
            "stat_vars": ["DebtServicePaid"],
            "stat_var_truncated": False,
            "pinned": None,
        }

    return _stored


@pytest.fixture
def source() -> Callable[..., SelectedSource]:
    """Return a factory for a discovered source resolving Kenya alone, 2010-2024, in USD.

    Kenya alone because the shared `requirement` names Kenya alone: the
    compiler reports a source resolving entities the requirement does not
    name (`source_wider_than_requirement`), so the default source has to be
    exactly as wide as the default requirement. `wide_source` is the one
    that resolves Ghana too.
    """

    def _source(**overrides: Any) -> SelectedSource:
        fields: dict[str, Any] = {
            "spec_id": SPEC_KENYA,
            "sentence": "external debt service paid by Kenya, 2010 to 2024",
            "confidence": "high",
            "has_data": True,
            "served_count": 1,
            "shape_member_count": 1,
            "shape_member_digest": member_digest(["DebtServicePaid"]),
            "entities": (KENYA,),
            "unit_ref": "usd",
            "unit_label": "US dollars",
            "coverage_window": Period(start=2010, end=2024),
            "identity": {"stat_var": "DebtServicePaid"},
        }
        return SelectedSource(**(fields | overrides))

    return _source


@pytest.fixture
def wide_source(source: Callable[..., SelectedSource]) -> Callable[..., SelectedSource]:
    """Return a factory for a source resolving Kenya and Ghana: wider than the default requirement."""

    def _wide_source(**overrides: Any) -> SelectedSource:
        fields: dict[str, Any] = {
            "sentence": "external debt service paid by Kenya and Ghana, 2010 to 2024",
            "entities": (KENYA, GHANA),
        }
        return source(**(fields | overrides))

    return _wide_source


@pytest.fixture
def running_lifespan() -> Callable[[FastAPI], Any]:
    """Return an async context manager driving `app`'s ASGI lifespan by hand.

    `httpx2.ASGITransport` never forwards lifespan events, but a mounted FastMCP
    app's session manager only starts inside its own lifespan, so an in-process
    MCP handshake needs the lifespan run explicitly.
    """

    @asynccontextmanager
    async def _running(app: FastAPI) -> AsyncIterator[None]:
        receive_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        startup_complete = asyncio.Event()
        shutdown_complete = asyncio.Event()

        async def receive() -> MutableMapping[str, Any]:
            return await receive_queue.get()

        async def send(message: MutableMapping[str, Any]) -> None:
            if message["type"] == "lifespan.startup.complete":
                startup_complete.set()
            elif message["type"] == "lifespan.shutdown.complete":
                shutdown_complete.set()

        task = asyncio.create_task(app({"type": "lifespan", "state": {}}, receive, send))
        await receive_queue.put({"type": "lifespan.startup"})
        await asyncio.wait(
            {task, asyncio.ensure_future(startup_complete.wait())},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if task.done():
            task.result()  # surface a startup failure instead of hanging forever
        try:
            yield
        finally:
            await receive_queue.put({"type": "lifespan.shutdown"})
            await asyncio.wait(
                {task, asyncio.ensure_future(shutdown_complete.wait())},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not task.done():
                await task

    return _running


@pytest.fixture
def mounted_mcp_transport() -> Callable[..., StreamableHttpTransport]:
    """Return a factory for a fastmcp transport that talks MCP to a path mounted on `app`.

    `token`, when given, is carried as the transport's own bearer auth (the
    same form `test_server.py`'s callers pass to `Client(transport,
    auth=...)` directly) -- so a `ToolSurface` built over this transport,
    which never sees a per-call `auth=` of its own, still reaches a mount
    `common.auth.apply_gate` has gated. `statuses`, when given, collects the
    HTTP status of every response: mcp's client raises a non-2xx answer as
    `MCPError(INTERNAL_ERROR)` without its status, so a test reads a 401 off
    this list.
    """

    def _transport(
        app: FastAPI,
        mounted_path: str,
        *,
        token: str | None = None,
        statuses: list[int] | None = None,
    ) -> StreamableHttpTransport:
        recorded = statuses if statuses is not None else []

        async def record(response: httpx2.Response) -> None:
            recorded.append(response.status_code)

        def factory(**kwargs: Any) -> httpx2.AsyncClient:
            return httpx2.AsyncClient(
                transport=httpx2.ASGITransport(app=app),
                base_url="http://testserver",
                event_hooks={"response": [record]},
                **kwargs,
            )

        return StreamableHttpTransport(
            url=f"http://testserver{mounted_path}",
            auth=token,
            httpx_client_factory=cast(McpHttpClientFactory, factory),
        )

    return _transport

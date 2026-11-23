"""The completion gate in the harness: how `McpExecutor.publish_result` sorts a
publish into published, retryable and final, and how a planned turn whose
publication is unfinished hands back `pending` and the next turn carries it on.

The turns run on the turn loop (`tests.turn_driver.TurnDriver`) over the
plan path's nodes (`tests.plan_turns.plan_nodes`); the Studio routes'
own run in `test_run_completion_routes.py`.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Awaitable, Mapping
from contextlib import asynccontextmanager
from typing import Any

import httpx2
import pytest
from analysis_mcp.runner_client import DerivedUnread
from common.live_apis import LiveApisTransportError
from common.materialisation import FrameExpired, FrameIntegrityError, FrameRef, FrameTooLarge
from common.recipe import (
    RecipeAncestorMissingError,
    RecipeAnchorError,
    RecipeRootUnrecoverableError,
)
from executor.completion import PUBLISH_ATTEMPTS, FramePublished, PublishFault
from executor.ledger import (
    MAX_DELIVERY_ATTEMPTS,
    Attempt,
    InMemoryRunLedger,
    RunChange,
    RunRecord,
    RunStatus,
)
from google.api_core.exceptions import Forbidden, ServiceUnavailable, from_http_status
from harness import mcp_client
from harness.conversation import current_conversation_id
from harness.mcp_client import McpExecutor
from harness.provenance import ProvenanceStore
from harness.scripted import ScriptedSurface
from harness.settings import Settings
from harness.studio_store import ConversationStale
from harness.studio_translate import _RecordingSurface
from planner.plan import Plan, PlanStatus
from presentation_mcp import client as presentation_client
from presentation_mcp.assembler_holder import set_assembler
from presentation_mcp.client import PlatformAssembler
from presentation_mcp.frame_publish import PublishBody
from presentation_mcp.platform_contract import NormalFormConflict, Refusal, RetryMode
from presentation_mcp.settings import Settings as PresentationSettings
from sqlalchemy.ext.asyncio import AsyncEngine

from .graph_turns import agent_read, opening_graph
from .plan_turns import approve_turn, plan_nodes
from .scripted_genai import Step, reply, text_part
from .scripted_tools import plan_call, scripted_orchestrator
from .turn_driver import ScriptedModels, TurnDriver, scripted_models

DERIVED = FrameRef(spec_id="ops_" + "2" * 32, content_version="ocv_" + "3" * 32)
REPLAY_KEY = "rpk_" + "e" * 64
PUBLISH_PATH = "/api/datasets/frames/publish"


class _Store:
    """Holds every frame, none, or raises `raises` on every read."""

    def __init__(self, *, holds: bool, raises: Exception | None = None) -> None:
        self.holds = holds
        self.raises = raises

    async def get_trace(self, ref: FrameRef) -> object | None:
        if self.raises is not None:
            raise self.raises
        return object() if self.holds else None


async def _recipe(ref: FrameRef, **_: Any) -> dict[str, Any]:
    return {"replay_key": REPLAY_KEY}


async def _unbuildable(ref: FrameRef, **_: Any) -> dict[str, Any]:
    raise RecipeAncestorMissingError(ref)


def _anchorless(recipe: dict[str, Any] | None) -> Any:
    """A build that finds no anchor for the rolling tree, offering `recipe`."""

    async def build(ref: FrameRef, **_: Any) -> dict[str, Any]:
        exc = RecipeAnchorError(ref, reason="populations_outside_reach")
        exc.recipe = recipe
        raise exc

    return build


_BUSY = Refusal(
    error="service_unavailable", retry=RetryMode.RETRY_AFTER_DELAY, detail="busy", retry_after_s=4
)
_REFUSED = Refusal(error="request_invalid", retry=RetryMode.DO_NOT_RETRY, detail="no")
_PUBLISHED = FramePublished(replay_key=REPLAY_KEY)
_HOLDS = _Store(holds=True)


def _raising(exc: Exception) -> _Store:
    return _Store(holds=True, raises=exc)


@pytest.mark.parametrize(
    ("store", "build", "answer", "expected"),
    [
        (_HOLDS, _recipe, True, _PUBLISHED),
        (_HOLDS, _recipe, False, _PUBLISHED),
        (
            _HOLDS,
            _recipe,
            NormalFormConflict(detail="kept"),
            ("recipe_normal_form_conflict", False),
        ),
        (_raising(FrameExpired(DERIVED)), _recipe, None, ("replay_unavailable", False)),
        (
            _raising(FrameIntegrityError(DERIVED, reason="bad footer")),
            _recipe,
            None,
            ("frame_integrity_failed", False),
        ),
        (_raising(FrameTooLarge(DERIVED, reason="big")), _recipe, None, ("frame_too_large", False)),
        (_raising(ServiceUnavailable("gcs down")), _recipe, None, ("store_unavailable", True)),
        (_raising(ConnectionResetError("reset")), _recipe, None, ("store_unavailable", True)),
        (_raising(from_http_status(408, "slow")), _recipe, None, ("store_unavailable", True)),
        (_raising(Forbidden("no access")), _recipe, None, ("store_refused", False)),
        (_HOLDS, _unbuildable, None, ("recipe_unbuildable", False)),
        (_HOLDS, _anchorless({"replay_key": REPLAY_KEY, "anchor": None}), True, _PUBLISHED),
        (_HOLDS, _anchorless(None), None, ("recipe_unbuildable", False)),
        (_HOLDS, _recipe, _BUSY, ("service_unavailable", True)),
        (_HOLDS, _recipe, _REFUSED, ("request_invalid", False)),
        (_HOLDS, _recipe, ServiceUnavailable("gcs down"), ("store_unavailable", True)),
        (_HOLDS, _recipe, TimeoutError("read"), ("store_unavailable", True)),
    ],
)
async def test_publish_result_sorts_each_publish_into_published_retryable_or_final(
    monkeypatch: pytest.MonkeyPatch, store: _Store, build: Any, answer: Any, expected: Any
) -> None:
    """A fault of the frame store reads the same whether building the recipe or
    reading the frame to publish met it, and a recipe without an anchor of its
    own is published as it is."""
    monkeypatch.setattr(mcp_client, "get_store", lambda: store)
    monkeypatch.setattr(mcp_client, "build_recipe", build)
    sent: list[dict[str, Any]] = []

    async def publish_frame(**kwargs: Any) -> Any:
        sent.append(kwargs)
        if isinstance(answer, Exception):
            raise answer
        return answer

    executor = McpExecutor(settings=Settings(), publish_frame=publish_frame)

    published = await executor.publish_result(DERIVED)

    if isinstance(expected, FramePublished):
        assert published == expected
        assert sent[0]["recipe"]["replay_key"] == REPLAY_KEY
    else:
        assert isinstance(published, PublishFault)
        assert (published.code, published.retryable) == expected
    assert len(sent) == (answer is not None)


def _runner_recipe(answer: dict[str, Any] | Exception | None) -> Any:
    """`derived_recipe` answering `answer`, or raising it."""

    async def read(ref: FrameRef) -> dict[str, Any] | None:
        assert ref == DERIVED
        if isinstance(answer, Exception):
            raise answer
        return answer

    return read


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ({"replay_key": REPLAY_KEY}, _PUBLISHED),
        (None, ("recipe_unbuildable", False)),
        (
            DerivedUnread(code="snapshot_not_found", retryable=False, detail="no such frame"),
            ("replay_unavailable", False),
        ),
        (
            DerivedUnread(code="artifact_expired", retryable=False, detail="gone"),
            ("replay_unavailable", False),
        ),
        (
            DerivedUnread(code="upstream_error", retryable=True, detail="unreachable"),
            ("upstream_error", True),
        ),
        (
            DerivedUnread(code="service_unavailable", retryable=True, detail="busy"),
            ("service_unavailable", True),
        ),
    ],
)
async def test_a_frame_the_store_does_not_hold_is_the_runners_and_is_published_no_more(
    monkeypatch: pytest.MonkeyPatch, answer: Any, expected: Any
) -> None:
    """A derived frame the store holds no trace for is one dfo-apis' analysis
    runner stored with its recipe: its replay key is read back from dfo-apis and
    nothing is published."""
    monkeypatch.setattr(mcp_client, "get_store", lambda: _Store(holds=False))
    monkeypatch.setattr(mcp_client, "derived_recipe", _runner_recipe(answer))
    sent: list[dict[str, Any]] = []

    async def publish_frame(**kwargs: Any) -> bool:
        sent.append(kwargs)
        return True

    executor = McpExecutor(settings=Settings(), publish_frame=publish_frame)

    published = await executor.publish_result(DERIVED)

    if isinstance(expected, FramePublished):
        assert published == expected
    else:
        assert isinstance(published, PublishFault)
        assert (published.code, published.retryable) == expected
    assert sent == []


async def _rootless(ref: FrameRef, *, recover: Any = None, **_: Any) -> dict[str, Any]:
    """A build whose one root has no mint record, rebuilt through `recover`."""
    if recover is None:
        raise RecipeRootUnrecoverableError(ref, reason="no_mint_record")
    await recover(FrameRef(spec_id="spec_" + "a" * 32, content_version="cv_" + "b" * 32))
    return {"replay_key": REPLAY_KEY}


@pytest.mark.parametrize(
    ("versions", "expected"),
    [
        ((), _PUBLISHED),
        (LiveApisTransportError(detail="down"), ("dataset_versions_unavailable", True)),
    ],
)
async def test_a_root_with_no_mint_record_has_it_rebuilt_from_its_datasets_versions(
    monkeypatch: pytest.MonkeyPatch, versions: Any, expected: Any
) -> None:
    async def recover_mint_record(ref: FrameRef, *, dataset_versions: Any, store: Any) -> Any:
        await dataset_versions(spec_id=ref.spec_id)
        return object()

    async def dataset_versions(*, spec_id: str) -> Any:
        if isinstance(versions, Exception):
            raise versions
        return versions

    async def publish_frame(**_: Any) -> bool:
        return True

    monkeypatch.setattr(mcp_client, "get_store", lambda: _HOLDS)
    monkeypatch.setattr(mcp_client, "build_recipe", _rootless)
    monkeypatch.setattr(mcp_client, "recover_mint_record", recover_mint_record)
    executor = McpExecutor(
        settings=Settings(), publish_frame=publish_frame, dataset_versions=dataset_versions
    )

    published = await executor.publish_result(DERIVED)

    if isinstance(expected, FramePublished):
        assert published == expected
    else:
        assert isinstance(published, PublishFault)
        assert (published.code, published.retryable) == expected


@asynccontextmanager
async def _publish_body(**_: Any) -> AsyncIterator[PublishBody]:
    """A body for any pair, read from no store."""

    async def through(work: Awaitable[bytes]) -> bytes:
        return await work

    yield PublishBody(data=b"{}", rows=0, hold_through=through)


async def test_a_frame_the_gate_published_is_not_published_again_when_described(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The gate publishes through the assembler presentation-mcp serves
    `describe_frame` with, so describing the frame finds it published."""
    monkeypatch.setattr(mcp_client, "get_store", lambda: _HOLDS)
    monkeypatch.setattr(mcp_client, "build_recipe", _recipe)
    monkeypatch.setattr(presentation_client, "publish_body_for", _publish_body)
    paths: list[str] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        paths.append(request.url.path)
        if request.url.path == PUBLISH_PATH:
            return httpx2.Response(200, json={"schema_version": 1, "published": {"created": True}})
        return httpx2.Response(200, json={"spec_id": DERIVED.spec_id, "dimensions": {}})

    executor = McpExecutor(settings=Settings())
    unwired = await executor.publish_result(DERIVED)
    assembler = PlatformAssembler(
        settings=PresentationSettings(datasets_first_party_principal="datasets-principal"),
        transport=httpx2.MockTransport(handler),
    )
    set_assembler(assembler)
    try:
        published = await executor.publish_result(DERIVED)
        view = await assembler.get_frame_view(frame_ref=DERIVED.model_dump())
    finally:
        set_assembler(None)
        await assembler.aclose()

    assert isinstance(unwired, PublishFault)
    assert (unwired.code, unwired.retryable) == ("publish_unavailable", True)
    assert published == _PUBLISHED
    assert not isinstance(view, Refusal)
    assert paths.count(PUBLISH_PATH) == 1


class _FlakyPublisher:
    """Refuses with a wait until `fixed`, then publishes."""

    def __init__(self) -> None:
        self.fixed = False
        self.frames: list[FrameRef] = []

    async def __call__(self, frame: FrameRef) -> FramePublished | PublishFault:
        self.frames.append(frame)
        if self.fixed:
            return FramePublished(replay_key=REPLAY_KEY)
        return PublishFault(
            code="service_unavailable", reason="busy", retryable=True, retry_after_s=0.01
        )


async def _describes(*, tool: str, args: dict[str, Any], on_progress: Any = None) -> Any:
    assert tool == "describe_frame", tool
    return {"outcome": "ok", "view": {"dimensions": [{"name": "observationAbout"}]}}


class _LinkingSurface(ScriptedSurface):
    """`ScriptedSurface` that records each frame whose lineage is linked."""

    def __init__(self) -> None:
        super().__init__()
        self.linked: list[Mapping[str, str | None]] = []

    def link_evidence(self, *, frame_ref: Mapping[str, str | None], **_: Any) -> None:
        self.linked.append(frame_ref)


_BRIEF = "Kenya's debt service since 2015"
_QUESTION = "Kenya's debt service?"


def _planned(after_plan: str, *later: str) -> ScriptedModels:
    """The agent hands `_BRIEF` to the planner, reads the plan's hand-back
    with `after_plan`, and answers each later turn in order with `later`."""
    steps: list[Step] = [reply(plan_call(_BRIEF)), reply(text_part(after_plan))]
    steps += [reply(text_part(text)) for text in later]
    return scripted_models(orchestrator=steps)


def _driver(
    engine: AsyncEngine,
    models: ScriptedModels,
    *,
    tool_surface: Any,
    executor: Any,
    run_ledger: Any = None,
    publisher: Any = None,
) -> TurnDriver:
    """One instance running the plan path over `models`: the scripted
    planner, `tool_surface` running the approved plan, `executor`
    describing its frames, its runs kept in `run_ledger` (one of its own by
    default) and published through `publisher` (all at once by default)."""
    extra = {"publisher": publisher} if publisher is not None else {}
    nodes = plan_nodes(
        scripted_orchestrator(),
        models,
        run_ledger=run_ledger or InMemoryRunLedger(scope=current_conversation_id.get),
        tool_surface=tool_surface,
        executor=executor,
        **extra,
    )
    return TurnDriver(engine, graph=opening_graph(nodes))


def _hand_back_body(models: ScriptedModels) -> dict[str, Any]:
    """The structured fields of the plan's hand-back in the agent's latest request."""
    read = agent_read(models.orchestrator.requests[-1])
    fields = (json.loads(line) for line in read.split("\n") if line.startswith("{"))
    return next(body for body in fields if "frames" in body)


async def test_a_pending_publication_is_handed_back_with_its_run_and_carried_on_next_turn(
    pg_engine: AsyncEngine,
) -> None:
    publisher = _FlakyPublisher()
    surface = _LinkingSurface()
    models = _planned("It is pending.", "It is done.")
    driver = _driver(
        pg_engine, models, tool_surface=surface, executor=_describes, publisher=publisher
    )

    await approve_turn(driver, _QUESTION)

    after_plan = agent_read(models.orchestrator.requests[-1])
    body = _hand_back_body(models)
    run = body["run"]
    assert run["status"] == "pending"
    assert body["frames"] == []
    assert f"pending under run {run['run_id']}" in after_plan
    assert len(publisher.frames) == PUBLISH_ATTEMPTS
    assert surface.linked == []

    publisher.fixed = True
    await driver.turn("Is it ready?", turn_index=2)

    (resumed,) = _runs(agent_read(models.orchestrator.requests[-1]))
    assert (resumed["run_id"], resumed["status"]) == (run["run_id"], "completed")
    assert [frame["frame"] for frame in resumed["frames"]]
    assert len(publisher.frames) == PUBLISH_ATTEMPTS + 1
    # Linked as `plan_done`'s frames are, so a card drawn on it keeps its sources.
    assert [ref["spec_id"] for ref in surface.linked] == [publisher.frames[-1].spec_id]


class _DescribeFails:
    """Describes every frame once `fixed`; until then times out as `answer`
    says: a refusal worth retrying, or an exception."""

    def __init__(self, answer: dict[str, Any] | Exception) -> None:
        self.answer = answer
        self.fixed = False

    async def __call__(self, *, tool: str, args: dict[str, Any], on_progress: Any = None) -> Any:
        if self.fixed:
            return await _describes(tool=tool, args=args)
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


def _runs(text: str) -> list[dict[str, Any]]:
    """The run standings of the latest hand-back of carried-on runs in a
    request, or none."""
    lines = [line for line in text.split("\n") if line.startswith('{"runs"')]
    return json.loads(lines[-1])["runs"] if lines else []


@pytest.mark.parametrize(
    ("describe", "not_chartable"),
    [
        (
            {
                "outcome": "refusal",
                "refusal": {
                    "code": "timeout",
                    "retry_mode": "retry_after_delay",
                    "message": "slow",
                },
            },
            [{"step": "aggregate", "reason": "this step's data view is temporarily unavailable"}],
        ),
        (TimeoutError("describe_frame ran past its deadline"), None),
    ],
    ids=["refused", "raised"],
)
async def test_a_resumed_runs_result_is_listed_again_until_its_hand_back_is_recorded(
    describe: dict[str, Any] | Exception,
    not_chartable: list[dict[str, str]] | None,
    pg_engine: AsyncEngine,
) -> None:
    """Publication completes the run on the next turn, but its frames cannot
    be described then: the run stands completed with its result undelivered,
    a refused frame listed not chartable in `plan_done`'s words, and the turn
    after lists it and records it delivered."""
    publisher = _FlakyPublisher()
    executor = _DescribeFails(describe)
    ledger = InMemoryRunLedger(scope=current_conversation_id.get)
    models = _planned("It is pending.", "It is computed.", "It is done.")
    driver = _driver(
        pg_engine,
        models,
        tool_surface=_LinkingSurface(),
        executor=executor,
        publisher=publisher,
        run_ledger=ledger,
    )
    await approve_turn(driver, _QUESTION)
    publisher.fixed = True

    await driver.turn("Is it ready?", turn_index=2)

    (undelivered,) = _runs(agent_read(models.orchestrator.requests[-1]))
    assert (undelivered["status"], undelivered["code"]) == ("completed", "result_undelivered")
    assert undelivered["frames"] == []
    assert undelivered.get("frames_not_chartable") == not_chartable
    run = await ledger.get(undelivered["run_id"])
    assert run is not None and run.status is RunStatus.COMPLETED and not run.delivered

    executor.fixed = True
    await driver.turn("And now?", turn_index=3)

    (delivered,) = _runs(agent_read(models.orchestrator.requests[-1]))
    assert (delivered["run_id"], delivered["status"]) == (undelivered["run_id"], "completed")
    assert "code" not in delivered
    assert [frame["frame"] for frame in delivered["frames"]]
    run = await ledger.get(undelivered["run_id"])
    assert run is not None and run.delivered
    assert await ledger.unfinished() == ()


async def test_a_run_completed_in_its_own_turn_whose_frames_are_unread_is_listed_next_turn(
    pg_engine: AsyncEngine,
) -> None:
    """The run completes in the turn that ran it, but `plan_done` cannot
    describe its frames: the run stands completed with its result
    undelivered, and the next turn lists it and records it delivered."""
    executor = _DescribeFails(
        {
            "outcome": "refusal",
            "refusal": {"code": "timeout", "retry_mode": "retry_after_delay", "message": "slow"},
        }
    )
    ledger = InMemoryRunLedger(scope=current_conversation_id.get)
    models = _planned("It is computed.", "It is done.")
    driver = _driver(
        pg_engine, models, tool_surface=_LinkingSurface(), executor=executor, run_ledger=ledger
    )

    await approve_turn(driver, _QUESTION)

    run = _hand_back_body(models)["run"]
    assert (run["status"], run["code"]) == ("completed", "result_undelivered")
    stored = await ledger.get(run["run_id"])
    assert stored is not None and not stored.delivered

    executor.fixed = True
    await driver.turn("Is it ready?", turn_index=2)

    (listed,) = _runs(agent_read(models.orchestrator.requests[-1]))
    assert (listed["run_id"], listed["status"]) == (run["run_id"], "completed")
    assert [frame["frame"] for frame in listed["frames"]]
    stored = await ledger.get(run["run_id"])
    assert stored is not None and stored.delivered


async def test_a_resumed_run_whose_frames_stay_unread_is_listed_no_more_past_the_bound(
    pg_engine: AsyncEngine,
) -> None:
    """Publication completes the run on the next turn, but its frames are
    never described: each turn lists it undelivered until the delivery
    bound, whose turn says it is not listed again, and no later turn lists
    it. Each later turn has a cursor of its own, as in Studio."""
    publisher = _FlakyPublisher()
    ledger = InMemoryRunLedger(scope=current_conversation_id.get)
    models = _planned("It is pending.", *("Not yet." for _ in range(MAX_DELIVERY_ATTEMPTS + 1)))
    driver = _driver(
        pg_engine,
        models,
        tool_surface=_LinkingSurface(),
        executor=_DescribeFails(TimeoutError("describe_frame ran past its deadline")),
        publisher=publisher,
        run_ledger=ledger,
    )
    await approve_turn(driver, _QUESTION)
    publisher.fixed = True

    standings = []
    for turn in range(MAX_DELIVERY_ATTEMPTS):
        await driver.turn("Is it ready?", turn_index=turn + 2)
        (standing,) = _runs(agent_read(models.orchestrator.requests[-1]))
        standings.append(standing)

    assert {(s["status"], s["code"]) for s in standings} == {("completed", "result_undelivered")}
    assert ["reason" in standing for standing in standings] == [False] * (
        MAX_DELIVERY_ATTEMPTS - 1
    ) + [True]
    assert await ledger.unfinished() == ()

    await driver.turn("And now?", turn_index=MAX_DELIVERY_ATTEMPTS + 2)

    assert _runs(agent_read(models.orchestrator.requests[-1])) == []


async def test_a_run_carried_on_after_a_restart_keeps_its_frames_provenance(
    pg_engine: AsyncEngine,
) -> None:
    """The process that ran the run is gone, and its provenance store with
    it: the run's frames rest on the retrievals it sealed, as a run that
    finished in its own turn does."""
    ledger = InMemoryRunLedger(scope=current_conversation_id.get)
    publisher = _FlakyPublisher()
    driver = _driver(
        pg_engine,
        _planned("It is pending."),
        tool_surface=_RecordingSurface(ScriptedSurface(), provenance=ProvenanceStore()),
        executor=_describes,
        publisher=publisher,
        run_ledger=ledger,
    )
    await approve_turn(driver, _QUESTION)
    publisher.fixed = True

    provenance = ProvenanceStore()
    models = scripted_models(orchestrator=[reply(text_part("It is done."))])
    restarted = _driver(
        pg_engine,
        models,
        tool_surface=_RecordingSurface(ScriptedSurface(), provenance=provenance),
        executor=_describes,
        publisher=publisher,
        run_ledger=ledger,
    )
    await restarted.turn("Is it ready?", turn_index=2)

    (resumed,) = _runs(agent_read(models.orchestrator.requests[-1]))
    assert resumed["status"] == "completed" and resumed["frames"]
    derived = publisher.frames[-1]
    lineage = provenance.get(
        conversation=current_conversation_id.get(),
        spec_id=derived.spec_id,
        content_version=derived.content_version,
    )
    assert lineage is not None
    assert {member.spec_id for member in lineage} == {
        source.spec_id
        for source in (await ledger.get(resumed["run_id"])).manifest.entries  # type: ignore[union-attr]
    }


async def test_a_completed_run_admitted_again_on_a_fresh_surface_keeps_its_provenance(
    pg_engine: AsyncEngine,
) -> None:
    """A later run of the same work under the same key finds the run
    completed, and the process that computed it is gone with its provenance
    store: its frames rest on the retrievals it sealed."""
    ledger = InMemoryRunLedger(scope=current_conversation_id.get)

    async def approved_turn(surface: Any, *, turn_index: int) -> dict[str, Any]:
        models = _planned("It is done.")
        driver = _driver(
            pg_engine, models, tool_surface=surface, executor=_describes, run_ledger=ledger
        )
        await approve_turn(driver, _QUESTION, turn_index=turn_index)
        return _hand_back_body(models)

    first = _RecordingSurface(ScriptedSurface(), provenance=ProvenanceStore())
    ran = await approved_turn(first, turn_index=1)
    provenance = ProvenanceStore()
    second = _RecordingSurface(ScriptedSurface(), provenance=provenance)
    again = await approved_turn(second, turn_index=2)

    assert again["run"] == ran["run"]
    run = await ledger.get(again["run"]["run_id"])
    assert run is not None and run.status is RunStatus.COMPLETED
    derived = next(
        result.frame for result in run.results if result.frame.spec_id.startswith("ops_")
    )
    lineage = provenance.get(
        conversation=current_conversation_id.get(),
        spec_id=derived.spec_id,
        content_version=derived.content_version,
    )
    assert lineage is not None
    assert {member.spec_id for member in lineage} == {
        entry.spec_id
        for entry in run.manifest.entries  # type: ignore[union-attr]
    }


class _Outage:
    """`InMemoryRunLedger` whose store faults once, as a briefly unreachable
    store does: on the first attempt it is asked to record, or with
    `on_get` on its `on_get`th read of a run."""

    def __init__(self, *, on_get: int | None = None) -> None:
        self.alive = InMemoryRunLedger(scope=current_conversation_id.get)
        self.on_get = on_get
        self.gets = 0
        self.faulted = False

    async def admit(self, candidate: RunRecord, *, submission: Any) -> Any:
        return await self.alive.admit(candidate, submission=submission)

    async def get(self, run_id: str) -> RunRecord | None:
        self.gets += 1
        if self.gets == self.on_get and not self.faulted:
            self.faulted = True
            raise ConversationStale("the conversation could not be read")
        return await self.alive.get(run_id)

    async def submission(self, run_id: str) -> Any:
        return await self.alive.submission(run_id)

    async def retrievals(self, run_id: str) -> Any:
        return await self.alive.retrievals(run_id)

    async def store_plan(self, revision: str, plan: Any) -> None:
        await self.alive.store_plan(revision, plan)

    async def plan(self, revision: str) -> Any:
        return await self.alive.plan(revision)

    async def unfinished(self) -> tuple[RunRecord, ...]:
        return await self.alive.unfinished()

    async def record(self, run_id: str, change: RunChange) -> RunRecord:
        if self.on_get is None and isinstance(change, Attempt) and not self.faulted:
            self.faulted = True
            raise ConversationStale("another writer landed first")
        return await self.alive.record(run_id, change)


# The reads of a planned turn's run: the executor's before driving it, the
# plan adapter's once it finishes, and the `execute` node's.
@pytest.mark.parametrize(
    ("on_get", "left_at", "plan_at"),
    [
        (None, RunStatus.RUNNING, PlanStatus.EXECUTING),
        (1, RunStatus.ADMITTED, PlanStatus.EXECUTING),
        (2, RunStatus.COMPLETED, PlanStatus.EXECUTING),
        (3, RunStatus.COMPLETED, PlanStatus.COMPLETED),
    ],
    ids=["attempt", "driver_read", "plan_read", "node_read"],
)
async def test_a_ledger_outage_mid_run_hands_back_an_unfinished_run_carried_on_next_turn(
    on_get: int | None, left_at: RunStatus, plan_at: PlanStatus, pg_engine: AsyncEngine
) -> None:
    """The run could not be recorded or read as it went: the turn closes
    saying so, with no figure, and the run is left for the next turn to
    finish and deliver."""
    ledger = _Outage(on_get=on_get)
    models = _planned("It could not be saved.", "It is done.")
    driver = _driver(
        pg_engine, models, tool_surface=_LinkingSurface(), executor=_describes, run_ledger=ledger
    )

    ran = await approve_turn(driver, _QUESTION)

    body = _hand_back_body(models)
    run = body["run"]
    assert (run["status"], run["code"]) == ("unfinished", "ledger_unavailable")
    assert body["frames"] == [] and body["step_summaries"] == []
    left = await ledger.alive.get(run["run_id"])
    assert left is not None and left.status is left_at and not left.delivered
    assert Plan.model_validate_json(ran.cursor.state["plan"]).status is plan_at

    await driver.turn("Is it ready?", turn_index=2)

    (resumed,) = _runs(agent_read(models.orchestrator.requests[-1]))
    assert (resumed["run_id"], resumed["status"]) == (run["run_id"], "completed")

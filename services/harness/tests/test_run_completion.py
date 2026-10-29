"""The completion gate in the harness: how `McpExecutor.publish_result` sorts a
publish into published, retryable and final, and how a planned turn whose
publication is unfinished hands back `pending` and the next turn carries it on.
"""

from __future__ import annotations

import hashlib
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
    idempotency_key,
)
from google.adk.errors import StaleSessionError
from google.adk.sessions import InMemorySessionService
from google.api_core.exceptions import Forbidden, ServiceUnavailable, from_http_status
from google.genai import types
from harness import mcp_client
from harness.conversation import current_conversation_id, current_conversation_key
from harness.graph import answer as answer_module
from harness.graph.state import AGENT_NAME
from harness.mcp_client import McpExecutor
from harness.provenance import ProvenanceStore
from harness.run_ledger import StudioRunLedger
from harness.scripted import ScriptedPlannerModel, ScriptedSurface
from harness.settings import Settings
from harness.studio_state import StudioStore
from harness.studio_translate import _RecordingSurface
from planner.plan import Plan, PlanStatus
from presentation_mcp import client as presentation_client
from presentation_mcp.assembler_holder import set_assembler
from presentation_mcp.client import PlatformAssembler
from presentation_mcp.frame_publish import PublishBody
from presentation_mcp.platform_contract import NormalFormConflict, Refusal, RetryMode
from presentation_mcp.settings import Settings as PresentationSettings

from .orchestrator_fakes import (
    CONVERSATION,
    approve_answer,
    call,
    create_request_input_response,
    open_session,
    orchestrator_app,
    orchestrator_runner,
    plan_call,
    request_text,
    run_turn,
    single_pause,
    stream,
    studio_client,
    turn_body,
)

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


async def test_a_pending_publication_is_handed_back_with_its_run_and_carried_on_next_turn() -> None:
    publisher = _FlakyPublisher()
    surface = _LinkingSurface()
    runner, model = orchestrator_runner(
        call(
            "plan",
            brief="Kenya's debt service since 2015",
            asks=["Kenya's debt service since 2015"],
        ),
        types.Part.from_text(text="Handed to the planner."),
        types.Part.from_text(text="It is pending."),
        types.Part.from_text(text="It is done."),
        planner=ScriptedPlannerModel(),
        tool_surface=surface,
        executor=_describes,
        publisher=publisher,
    )
    await open_session(runner)
    approval = single_pause(await run_turn(runner, types.UserContent("Kenya's debt service?")))
    approve = create_request_input_response(
        approval.id, {"result": approve_answer(approval.args["payload"]["revision"])}
    )

    await run_turn(runner, types.Content(role="user", parts=[approve]))

    after_plan = request_text(model.requests[-1])
    body = json.loads(next(line for line in after_plan.split("\n") if line.startswith('{"frames"')))
    run = body["run"]
    assert run["status"] == "pending"
    assert body["frames"] == []
    assert f"pending under run {run['run_id']}" in after_plan
    assert len(publisher.frames) == PUBLISH_ATTEMPTS
    assert surface.linked == []

    publisher.fixed = True
    await run_turn(runner, types.UserContent("Is it ready?"))

    next_turn = request_text(model.requests[-1])
    runs = json.loads(next(line for line in next_turn.split("\n") if line.startswith('{"runs"')))
    (resumed,) = runs["runs"]
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
    describe: dict[str, Any] | Exception, not_chartable: list[dict[str, str]] | None
) -> None:
    """Publication completes the run on the next turn, but its frames cannot
    be described then: the run stands completed with its result undelivered,
    a refused frame listed not chartable in `plan_done`'s words, and the turn
    after lists it and records it delivered."""
    publisher = _FlakyPublisher()
    executor = _DescribeFails(describe)
    ledger = InMemoryRunLedger(scope=current_conversation_id.get)
    runner, model = orchestrator_runner(
        call(
            "plan",
            brief="Kenya's debt service since 2015",
            asks=["Kenya's debt service since 2015"],
        ),
        types.Part.from_text(text="Handed to the planner."),
        types.Part.from_text(text="It is pending."),
        types.Part.from_text(text="It is computed."),
        types.Part.from_text(text="It is done."),
        planner=ScriptedPlannerModel(),
        tool_surface=_LinkingSurface(),
        executor=executor,
        publisher=publisher,
        run_ledger=ledger,
    )
    await open_session(runner)
    approval = single_pause(await run_turn(runner, types.UserContent("Kenya's debt service?")))
    approve = create_request_input_response(
        approval.id, {"result": approve_answer(approval.args["payload"]["revision"])}
    )
    await run_turn(runner, types.Content(role="user", parts=[approve]))
    publisher.fixed = True

    await run_turn(runner, types.UserContent("Is it ready?"))

    (undelivered,) = _runs(request_text(model.requests[-1]))
    assert (undelivered["status"], undelivered["code"]) == ("completed", "result_undelivered")
    assert undelivered["frames"] == []
    assert undelivered.get("frames_not_chartable") == not_chartable
    run = await ledger.get(undelivered["run_id"])
    assert run is not None and run.status is RunStatus.COMPLETED and not run.delivered

    executor.fixed = True
    await run_turn(runner, types.UserContent("And now?"))

    (delivered,) = _runs(request_text(model.requests[-1]))
    assert (delivered["run_id"], delivered["status"]) == (undelivered["run_id"], "completed")
    assert "code" not in delivered
    assert [frame["frame"] for frame in delivered["frames"]]
    run = await ledger.get(undelivered["run_id"])
    assert run is not None and run.delivered
    assert await ledger.unfinished() == ()


async def test_a_run_completed_in_its_own_turn_whose_frames_are_unread_is_listed_next_turn() -> (
    None
):
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
    runner, model = orchestrator_runner(
        call(
            "plan",
            brief="Kenya's debt service since 2015",
            asks=["Kenya's debt service since 2015"],
        ),
        types.Part.from_text(text="Handed to the planner."),
        types.Part.from_text(text="It is computed."),
        types.Part.from_text(text="It is done."),
        planner=ScriptedPlannerModel(),
        tool_surface=_LinkingSurface(),
        executor=executor,
        run_ledger=ledger,
    )
    await open_session(runner)
    approval = single_pause(await run_turn(runner, types.UserContent("Kenya's debt service?")))
    approve = create_request_input_response(
        approval.id, {"result": approve_answer(approval.args["payload"]["revision"])}
    )

    await run_turn(runner, types.Content(role="user", parts=[approve]))

    after_plan = request_text(model.requests[-1])
    body = json.loads(next(line for line in after_plan.split("\n") if line.startswith('{"frames"')))
    run = body["run"]
    assert (run["status"], run["code"]) == ("completed", "result_undelivered")
    stored = await ledger.get(run["run_id"])
    assert stored is not None and not stored.delivered

    executor.fixed = True
    await run_turn(runner, types.UserContent("Is it ready?"))

    (listed,) = _runs(request_text(model.requests[-1]))
    assert (listed["run_id"], listed["status"]) == (run["run_id"], "completed")
    assert [frame["frame"] for frame in listed["frames"]]
    stored = await ledger.get(run["run_id"])
    assert stored is not None and stored.delivered


async def test_a_resumed_run_whose_frames_stay_unread_is_listed_no_more_past_the_bound() -> None:
    """Publication completes the run on the next turn, but its frames are
    never described: each turn lists it undelivered until the delivery
    bound, whose turn says it is not listed again, and no later turn lists
    it."""
    publisher = _FlakyPublisher()
    ledger = InMemoryRunLedger(scope=current_conversation_id.get)
    runner, model = orchestrator_runner(
        call(
            "plan",
            brief="Kenya's debt service since 2015",
            asks=["Kenya's debt service since 2015"],
        ),
        types.Part.from_text(text="Handed to the planner."),
        types.Part.from_text(text="It is pending."),
        *(types.Part.from_text(text="Not yet.") for _ in range(MAX_DELIVERY_ATTEMPTS + 1)),
        planner=ScriptedPlannerModel(),
        tool_surface=_LinkingSurface(),
        executor=_DescribeFails(TimeoutError("describe_frame ran past its deadline")),
        publisher=publisher,
        run_ledger=ledger,
    )
    await open_session(runner)
    approval = single_pause(await run_turn(runner, types.UserContent("Kenya's debt service?")))
    approve = create_request_input_response(
        approval.id, {"result": approve_answer(approval.args["payload"]["revision"])}
    )
    await run_turn(runner, types.Content(role="user", parts=[approve]))
    publisher.fixed = True

    standings = []
    for _ in range(MAX_DELIVERY_ATTEMPTS):
        await run_turn(runner, types.UserContent("Is it ready?"))
        (standing,) = _runs(request_text(model.requests[-1]))
        standings.append(standing)

    assert {(s["status"], s["code"]) for s in standings} == {("completed", "result_undelivered")}
    assert ["reason" in standing for standing in standings] == [False] * (
        MAX_DELIVERY_ATTEMPTS - 1
    ) + [True]
    assert await ledger.unfinished() == ()

    await run_turn(runner, types.UserContent("And now?"))

    # Each turn that carries a run on adds one hand-back to the history.
    listed = request_text(model.requests[-1]).count('{"runs"')
    assert listed == MAX_DELIVERY_ATTEMPTS


async def test_a_run_carried_on_after_a_restart_keeps_its_frames_provenance() -> None:
    """The process that ran the run is gone, and its provenance store with
    it: the run's frames rest on the retrievals it sealed, as a run that
    finished in its own turn does."""
    ledger = InMemoryRunLedger(scope=current_conversation_id.get)
    publisher = _FlakyPublisher()
    runner, _ = orchestrator_runner(
        call(
            "plan",
            brief="Kenya's debt service since 2015",
            asks=["Kenya's debt service since 2015"],
        ),
        types.Part.from_text(text="Handed to the planner."),
        types.Part.from_text(text="It is pending."),
        planner=ScriptedPlannerModel(),
        tool_surface=_RecordingSurface(ScriptedSurface(), provenance=ProvenanceStore()),
        executor=_describes,
        publisher=publisher,
        run_ledger=ledger,
    )
    await open_session(runner)
    approval = single_pause(await run_turn(runner, types.UserContent("Kenya's debt service?")))
    approve = create_request_input_response(
        approval.id, {"result": approve_answer(approval.args["payload"]["revision"])}
    )
    await run_turn(runner, types.Content(role="user", parts=[approve]))
    publisher.fixed = True

    provenance = ProvenanceStore()
    restarted, model = orchestrator_runner(
        types.Part.from_text(text="It is done."),
        planner=ScriptedPlannerModel(),
        tool_surface=_RecordingSurface(ScriptedSurface(), provenance=provenance),
        executor=_describes,
        publisher=publisher,
        run_ledger=ledger,
    )
    await open_session(restarted)
    await run_turn(restarted, types.UserContent("Is it ready?"))

    (resumed,) = _runs(request_text(model.requests[-1]))
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


async def test_a_completed_run_admitted_again_on_a_fresh_surface_keeps_its_provenance() -> None:
    """A later run of the same work under the same key finds the run
    completed, and the process that computed it is gone with its provenance
    store: its frames rest on the retrievals it sealed."""
    ledger = InMemoryRunLedger(scope=current_conversation_id.get)

    async def approved_turn(surface: Any) -> dict[str, Any]:
        runner, model = orchestrator_runner(
            call(
                "plan",
                brief="Kenya's debt service since 2015",
                asks=["Kenya's debt service since 2015"],
            ),
            types.Part.from_text(text="Handed to the planner."),
            types.Part.from_text(text="It is done."),
            planner=ScriptedPlannerModel(),
            tool_surface=surface,
            executor=_describes,
            run_ledger=ledger,
        )
        await open_session(runner)
        approval = single_pause(await run_turn(runner, types.UserContent("Kenya's debt service?")))
        approve = create_request_input_response(
            approval.id, {"result": approve_answer(approval.args["payload"]["revision"])}
        )
        await run_turn(runner, types.Content(role="user", parts=[approve]))
        after_plan = request_text(model.requests[-1])
        body = json.loads(
            next(line for line in after_plan.split("\n") if line.startswith('{"frames"'))
        )
        return body

    first = _RecordingSurface(ScriptedSurface(), provenance=ProvenanceStore())
    ran = await approved_turn(first)
    provenance = ProvenanceStore()
    second = _RecordingSurface(ScriptedSurface(), provenance=provenance)
    again = await approved_turn(second)

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
            raise StaleSessionError("the session could not be read")
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
            raise StaleSessionError("another writer landed first")
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
    on_get: int | None, left_at: RunStatus, plan_at: PlanStatus
) -> None:
    """The run could not be recorded or read as it went: the turn closes
    saying so, with no figure, and the run is left for the next turn to
    finish and deliver."""
    ledger = _Outage(on_get=on_get)
    runner, model = orchestrator_runner(
        call(
            "plan",
            brief="Kenya's debt service since 2015",
            asks=["Kenya's debt service since 2015"],
        ),
        types.Part.from_text(text="Handed to the planner."),
        types.Part.from_text(text="It could not be saved."),
        types.Part.from_text(text="It is done."),
        planner=ScriptedPlannerModel(),
        tool_surface=_LinkingSurface(),
        executor=_describes,
        run_ledger=ledger,
    )
    await open_session(runner)
    approval = single_pause(await run_turn(runner, types.UserContent("Kenya's debt service?")))
    approve = create_request_input_response(
        approval.id, {"result": approve_answer(approval.args["payload"]["revision"])}
    )

    await run_turn(runner, types.Content(role="user", parts=[approve]))

    after_plan = request_text(model.requests[-1])
    body = json.loads(next(line for line in after_plan.split("\n") if line.startswith('{"frames"')))
    run = body["run"]
    assert (run["status"], run["code"]) == ("unfinished", "ledger_unavailable")
    assert body["frames"] == [] and body["step_summaries"] == []
    left = await ledger.alive.get(run["run_id"])
    assert left is not None and left.status is left_at and not left.delivered
    session = await runner.session_service.get_session(app_name="t", user_id="u", session_id="s")
    assert session is not None
    assert Plan.model_validate_json(session.state["plan"]).status is plan_at

    await run_turn(runner, types.UserContent("Is it ready?"))

    next_turn = request_text(model.requests[-1])
    runs = json.loads(next(line for line in next_turn.split("\n") if line.startswith('{"runs"')))
    (resumed,) = runs["runs"]
    assert (resumed["run_id"], resumed["status"]) == (run["run_id"], "completed")


# The principal `orchestrator_fakes.AUTH`'s token names.
PRINCIPAL = "u_" + hashlib.sha256(b"test-token").hexdigest()[:16]


def _frame(frames: list[dict[str, Any]], event: str) -> dict[str, Any]:
    return next(frame for frame in frames if frame["event"] == event)


async def test_a_studio_run_left_pending_is_carried_on_from_the_conversation_ledger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Through the studio, over its conversation ledger session: a run whose
    publication is unfinished at the end of its turn is kept under a key that
    names the conversation, not the turn, and the next turn, in a session of
    its own, completes it and hands the agent its frames and the notes read
    from the plan stored with the ledger."""
    total = "The figures are the sum of 3 indicators."
    monkeypatch.setattr(answer_module, "_stat_var_totals", lambda *a, **k: [total])
    store = StudioStore(InMemorySessionService(), app_name=AGENT_NAME)
    ledger = StudioRunLedger(store)
    publisher = _FlakyPublisher()
    app, model = orchestrator_app(
        plan_call("Kenya's debt service since 2015"),
        types.Part.from_text(text="Handed to the planner."),
        types.Part.from_text(text="It is pending."),
        types.Part.from_text(text="It is done."),
        tool_surface=ScriptedSurface(),
        store=store,
        run_ledger=ledger,
        publisher=publisher,
    )
    async with studio_client(app) as client:
        first = await stream(client, "/api/studio/turn", turn_body("Kenya's debt service?"))
        plan = _frame(first, "plan.proposed")
        await stream(
            client,
            "/api/studio/interrupt",
            {
                "kind": "interrupt.response",
                "schema_version": 1,
                "conversation_id": CONVERSATION,
                "interrupt_id": plan["interrupt_id"],
                "decision": "approve",
            },
        )
        after_plan = request_text(model.requests[-1])
        body = json.loads(
            next(line for line in after_plan.split("\n") if line.startswith('{"frames"'))
        )
        assert (body["run"]["status"], body["frames"]) == ("pending", [])

        publisher.fixed = True
        await stream(client, "/api/studio/turn", turn_body("Is it ready?"))

    next_turn = request_text(model.requests[-1])
    runs = json.loads(next(line for line in next_turn.split("\n") if line.startswith('{"runs"')))
    (resumed,) = runs["runs"]
    assert (resumed["run_id"], resumed["status"]) == (body["run"]["run_id"], "completed")
    assert [frame["frame"] for frame in resumed["frames"]]
    assert resumed["totals"] == [total]
    token = current_conversation_key.set((PRINCIPAL, CONVERSATION))
    try:
        run = await ledger.get(resumed["run_id"])
    finally:
        current_conversation_key.reset(token)
    assert run is not None and run.status is RunStatus.COMPLETED
    assert run.idempotency_key == idempotency_key(
        caller=PRINCIPAL,
        conversation_id=f"{PRINCIPAL}/{CONVERSATION}",
        task=f"plan:{plan['revision']}",
    )


async def test_the_default_ledger_keeps_one_conversations_runs_from_another() -> None:
    publisher = _FlakyPublisher()
    app, model = orchestrator_app(
        plan_call("Kenya's debt service since 2015"),
        types.Part.from_text(text="Handed to the planner."),
        types.Part.from_text(text="It is pending."),
        types.Part.from_text(text="Another conversation."),
        tool_surface=ScriptedSurface(),
        publisher=publisher,
    )
    async with studio_client(app) as client:
        first = await stream(client, "/api/studio/turn", turn_body("Kenya's debt service?"))
        await stream(
            client,
            "/api/studio/interrupt",
            {
                "kind": "interrupt.response",
                "schema_version": 1,
                "conversation_id": CONVERSATION,
                "interrupt_id": _frame(first, "plan.proposed")["interrupt_id"],
                "decision": "approve",
            },
        )
        assert '"status": "pending"' in request_text(model.requests[-1])

        publisher.fixed = True
        other = turn_body("Hello", conversation_id="conv_01other000")
        await stream(client, "/api/studio/turn", other)

    assert '{"runs"' not in request_text(model.requests[-1])
    assert len(publisher.frames) == PUBLISH_ATTEMPTS

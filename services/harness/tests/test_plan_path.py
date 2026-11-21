"""A plan call's path past the planner on the turn loop: `approve` pausing on
the plan card and deciding the answer, `execute` running the approved plan,
and `plan_done` handing back what it computed, plus `plan_done`'s own
rendering of a hand-back.

The turns run the real `approve`, `execute` and `plan_done` nodes over the
scripted planner (`tests.plan_turns.plan_nodes`) and the orchestrator on
its real tools over a scripted Gemini client, on Postgres
(`tests.turn_driver.TurnDriver`).
"""

from __future__ import annotations

import dataclasses
import json
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any, cast

import pytest
from analysis_mcp.digest import render_rank_digest
from analysis_mcp.engine.rank import run_rank
from analysis_mcp.execution_identity import current_execution_identity
from compiler.surface import ToolCallResult
from data_mcp.tool_errors import ToolErrorCode as DataCode
from data_mcp.tool_errors import ToolRefusal as DataRefusal
from dfo_frame_grammar import Frame
from executor import classify as classify_module
from executor.ledger import (
    Attempt,
    InMemoryRunLedger,
    RunChange,
    RunRecord,
    RunStatus,
    idempotency_key,
)
from harness.conversation import current_conversation_key, ledger_conversation_id
from harness.graph import plan_done as plan_done_module
from harness.graph.approve import APPROVE_MESSAGE
from harness.graph.events import StepProgress
from harness.graph.plan_done import (
    HandBack,
    StepSummary,
    _ask_status,
    _partial_result,
    _retry_note,
    _stop_causes,
)
from harness.graph.state import (
    ORCHESTRATOR_CALLS_KEY,
    ORCHESTRATOR_FRAMES_KEY,
    ORCHESTRATOR_PLAN_RECORDS_KEY,
    ORCHESTRATOR_RETRY_NOTE_KEY,
    ORCHESTRATOR_UNAVAILABLE_KEY,
)
from harness.plan_display import card_rows
from harness.run_ledger import StudioRunLedger
from harness.scripted import ScriptedPlannerModel, ScriptedSurface
from harness.sentences import DECLINE_REASONS, failure_cause, failure_sentence
from harness.studio_state import ConversationState, SegmentLost
from harness.turn.events import NodeDone, Paused
from harness.turn.loop import Outcome, run_segment
from harness.turn.store import TurnStatus
from harness.turn_records import PlanAsk, PlanRecord
from planner.plan import AskAnswer, Decline, Plan, PlanStatus, UnsupportedOperation, load_asks
from sqlalchemy.ext.asyncio import AsyncEngine

from .graph_turns import agent_read, opening_graph, visited
from .plan_turns import approve_answer, approve_turn, plan_nodes
from .scripted_genai import reply, text_part
from .scripted_tools import plan_call, scripted_orchestrator
from .turn_driver import CONVERSATION, PRINCIPAL, ScriptedModels, TurnDriver, scripted_models

# The spec_id `harness.scripted_plan.scripted_plan` retrieves when no search pinned one.
SCRIPTED_SPEC = "spec_" + "f" * 32

_BRIEF = "Kenya's debt service since 2015"
_QUESTION = "Kenya's debt service?"


class AskingPlanner(ScriptedPlannerModel):
    """`ScriptedPlannerModel` whose plan answers the brief's first ask with
    its final step and declines every other ask as `unsupported_ask`."""

    async def __call__(self, **kwargs: Any) -> dict[str, Any]:
        plan = Plan.model_validate((await super().__call__(**kwargs))["plan"])
        asks = load_asks(kwargs["ctx"].state)
        final = plan.evidence_requirements[-1].id
        plan = plan.model_copy(
            update={
                "asks": asks,
                "answered": (AskAnswer(ask=0, outputs=(final,)),),
                "declined": tuple(
                    Decline(ask=ask, code=UnsupportedOperation.UNSUPPORTED_ASK, ask_ref=number)
                    for number, ask in enumerate(asks)
                    if number
                ),
            }
        )
        return {"plan": plan.model_dump(mode="json"), "summary": {"version": plan.version}}


async def describes(*, tool: str, args: dict[str, Any], on_progress: Any = None) -> Any:
    """An executor describing every frame."""
    assert tool == "describe_frame", tool
    return {"outcome": "ok", "view": {"dimensions": [{"name": "observationAbout"}]}}


async def refuses_to_describe(*, tool: str, args: dict[str, Any], on_progress: Any = None) -> Any:
    """An executor refusing to describe any frame, for good."""
    assert tool == "describe_frame", tool
    return {
        "outcome": "refusal",
        "refusal": {"code": "publish_not_configured", "retry_mode": "do_not_retry"},
    }


class RefusingGetDataSurface(ScriptedSurface):
    """`ScriptedSurface` whose `get_data` answers the refusal `code` names, with
    the retry mode data-mcp gives it, or `do_not_retry` for a code data-mcp
    does not answer."""

    def __init__(self, code: str, fix: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.code = code
        self.fix = fix or {}
        served = code in set(DataCode)
        self.retry_mode = (
            DataRefusal(code=DataCode(code), message="", fix=self.fix).retry_mode.value
            if served
            else "do_not_retry"
        )

    async def call_tool(self, *, tool: Any, arguments: Any, on_progress: Any = None) -> Any:
        if tool.name != "get_data":
            return await super().call_tool(tool=tool, arguments=arguments, on_progress=on_progress)
        refusal = {
            "code": self.code,
            "retry_mode": self.retry_mode,
            "fix": self.fix,
            "message": "this deployment cannot serve the request",
        }
        return ToolCallResult(is_error=True, structured_content=refusal, text=refusal["message"])


class _BreakdownRefusingSurface(ScriptedSurface):
    """`ScriptedSurface` whose `get_data` refuses the breakdown it was asked for,
    as apis refuses a code system it cannot read."""

    async def call_tool(self, **kwargs: Any) -> ToolCallResult:
        if kwargs["tool"].name != "get_data":
            return await super().call_tool(**kwargs)
        return ToolCallResult(
            is_error=True,
            structured_content={
                "code": "capability_unavailable",
                "retry_mode": "do_not_retry",
                "message": "this source cannot be broken down by that code system",
                "fix": {"pointer": "/breakdown/code_system"},
            },
            text="refused",
        )


class _EarlyMints:
    """Records each card offered for early retrieval and each withdrawal."""

    def __init__(self) -> None:
        self.offered: list[str] = []
        self.withdrawn = 0

    def offer(self, plan: Plan) -> None:
        self.offered.append(plan.model_dump_json())

    def withdraw(self) -> None:
        self.withdrawn += 1


def _driver(
    engine: AsyncEngine,
    models: ScriptedModels,
    *,
    planner: ScriptedPlannerModel | None = None,
    tool_surface: Any = None,
    executor: Any = describes,
    run_ledger: Any = None,
    early_mints: Any = None,
) -> TurnDriver:
    """One instance running the plan path over `models`, the plan run against
    `tool_surface` (a `ScriptedSurface` by default)."""
    nodes = plan_nodes(
        scripted_orchestrator(),
        models,
        run_ledger=run_ledger if run_ledger is not None else InMemoryRunLedger(),
        tool_surface=tool_surface if tool_surface is not None else ScriptedSurface(),
        executor=executor,
        planner=planner,
        early_mints=early_mints,
    )
    return TurnDriver(engine, graph=opening_graph(nodes))


def _planned(*after_plan: str, brief: str = _BRIEF, asks: tuple[str, ...] = ()) -> ScriptedModels:
    """The agent hands `brief` and `asks` to the planner and reads the plan's
    hand-back with each of `after_plan` in turn."""
    return scripted_models(
        orchestrator=[
            reply(plan_call(brief, *asks)),
            *(reply(text_part(text)) for text in after_plan),
        ]
    )


async def _take_over(driver: TurnDriver, *, emit: Any = None) -> Any:
    """Claim the conversation's lease on `driver`'s instance and run the turn
    from where its cursor stands, as a later segment carries a running turn
    on."""
    await driver.studio.load(PRINCIPAL, CONVERSATION)
    segment_id = driver.studio.claim_segment(PRINCIPAL, CONVERSATION)
    await driver.studio.save(PRINCIPAL, CONVERSATION)
    try:
        return await run_segment(
            driver.turns,
            driver.studio,
            await driver.open_cursor(),
            principal=PRINCIPAL,
            conversation_id=CONVERSATION,
            segment_id=segment_id,
            graph=driver.graph,
            emit=emit or (lambda event: None),
        )
    finally:
        await driver.studio.drop_lease(PRINCIPAL, CONVERSATION, segment_id)


def _hand_back_body(models: ScriptedModels) -> dict[str, Any]:
    """The structured fields of the plan's hand-back in the agent's latest request."""
    read = agent_read(models.orchestrator.requests[-1])
    fields = (json.loads(line) for line in read.split("\n") if line.startswith("{"))
    return next(body for body in fields if "frames" in body)


async def test_the_plan_card_pauses_the_turn_and_another_instance_runs_the_approved_plan(
    pg_engine: AsyncEngine,
) -> None:
    """`approve` pauses on the card the planner drafted, under kind
    `approve`; the answer on another instance resumes `approve`, which
    approves the revision it names, and `execute` and `plan_done` hand the
    run's frames back to the agent."""
    models = _planned("Charted.")
    first, second = _driver(pg_engine, models), _driver(pg_engine, models)

    paused = await first.turn(_QUESTION)

    card = paused.pause
    assert (card.kind, card.message) == ("approve", APPROVE_MESSAGE)
    assert visited(paused.events) == ["seed", "orchestrate", "plan_start", "planner"]
    assert paused.cursor.node == "approve"
    drafted = Plan.model_validate_json(paused.cursor.state["plan"])
    assert drafted.status is PlanStatus.PENDING_APPROVAL
    assert card.payload["revision"] == paused.events[-2].output["revision"]

    ran = await second.answer(approve_answer(card.payload["revision"]), to=card)

    assert ran.outcome is Outcome.DONE
    assert visited(ran.events) == ["approve", "execute", "plan_done", "orchestrate"]
    approved = ran.events[0]
    assert isinstance(approved, NodeDone) and approved.label == "approved"
    progress = [event for event in ran.events if isinstance(event, StepProgress)]
    assert progress and {event.step_id for event in progress} <= {
        step.id for step in drafted.workflow.steps
    }  # type: ignore[union-attr]
    body = _hand_back_body(models)
    assert body["frames"] and set(ran.cursor.state[ORCHESTRATOR_FRAMES_KEY]) == {
        frame["frame"] for frame in body["frames"]
    }
    assert body["step_summaries"]
    assert Plan.model_validate_json(ran.cursor.state["plan"]).status is PlanStatus.COMPLETED
    answers = [record for record in await second.records(ran.cursor) if record.kind == "answer"]
    assert [record.body["interrupt_id"] for record in answers] == [card.interrupt_id]


@pytest.mark.parametrize(
    "answer",
    [
        {"approved": "another revision", "at": datetime.now(UTC).isoformat()},
        {"approved": None},
        {"approved": "<revision>", "at": "2026-11-18T10:00:00"},
    ],
    ids=["another_revision", "no_approval", "no_offset"],
)
async def test_an_answer_that_approves_nothing_amends_and_runs_nothing(
    pg_engine: AsyncEngine, answer: dict[str, Any]
) -> None:
    """An answer naming another revision, or approving with no
    timezone-aware time, routes `amend` back to the planner, which offers a
    new card, and nothing runs."""
    surface = ScriptedSurface()
    called: list[str] = []
    original = surface.call_tool

    async def counting(**kwargs: Any) -> Any:
        called.append(kwargs["tool"].name)
        return await original(**kwargs)

    surface.call_tool = counting  # type: ignore[method-assign]
    driver = _driver(pg_engine, _planned(), tool_surface=surface)
    card = (await driver.turn(_QUESTION)).pause
    if answer.get("approved") == "<revision>":
        answer = {**answer, "approved": card.payload["revision"]}

    again = await driver.answer(answer, to=card)

    assert again.outcome is Outcome.PAUSED
    assert visited(again.events) == ["approve", "planner"]
    amended = again.events[0]
    assert isinstance(amended, NodeDone) and amended.label == "amend"
    assert amended.output["answer"] == answer
    assert Plan.model_validate(amended.output["plan"]).status is PlanStatus.PENDING_APPROVAL
    assert again.pause.kind == "approve" and again.pause.round == card.round + 1
    assert called == []


async def test_a_card_is_offered_for_early_retrieval_once_its_pause_commits(
    pg_engine: AsyncEngine,
) -> None:
    """The card is offered once its pause is stored, the answer offers
    nothing, and approving withdraws the card's calls still waiting."""
    mints = _EarlyMints()
    driver = _driver(pg_engine, _planned("Charted."), early_mints=mints)

    paused = await driver.turn(_QUESTION)

    assert mints.offered == [paused.cursor.state["plan"]]
    assert mints.withdrawn == 0

    await driver.answer(approve_answer(paused.pause.payload["revision"]), to=paused.pause)

    assert len(mints.offered) == 1
    assert mints.withdrawn == 1


async def test_a_death_before_the_card_pause_commits_offers_the_card_once_on_the_rerun(
    pg_engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The instance that drafted the card loses its lease at the pause's
    write, as a death there does: it offered nothing. The instance that takes
    the turn over runs `approve` again from its start and offers the card
    once, after its own pause commits."""
    dying, taking_over = _EarlyMints(), _EarlyMints()
    first = _driver(pg_engine, _planned(), early_mints=dying)
    second = _driver(pg_engine, _planned(), early_mints=taking_over)

    async def lost(*args: Any, **kwargs: Any) -> Any:
        raise SegmentLost("the lease was taken at the pause's write")

    monkeypatch.setattr(first.turns, "pause", lost)
    died = await first.turn(_QUESTION)

    assert died.outcome is Outcome.LOST
    assert dying.offered == []
    running = await first.open_cursor()
    assert (running.status, running.node) == (TurnStatus.RUNNING, "approve")

    events: list[Any] = []
    segment = await _take_over(second, emit=events.append)

    assert segment.outcome is Outcome.PAUSED
    assert isinstance(events[-1], Paused) and events[-1].kind == "approve"
    assert taking_over.offered == [segment.cursor.state["plan"]]


async def test_the_approved_plans_run_is_kept_in_the_conversations_ledger_under_the_turns_lease(
    pg_engine: AsyncEngine,
) -> None:
    """`execute` admits the run to the stored run ledger, fenced on the
    lease the turn's segment holds, under a key naming the conversation."""
    models = _planned("Charted.")
    driver = _driver(pg_engine, models)
    ledger = StudioRunLedger(driver.studio)
    driver.graph = _driver(pg_engine, models, run_ledger=ledger).graph
    token = current_conversation_key.set((PRINCIPAL, CONVERSATION))
    try:
        card = (await driver.turn(_QUESTION)).pause
        ran = await driver.answer(approve_answer(card.payload["revision"]), to=card)
        run = await ledger.get(_hand_back_body(models)["run"]["run_id"])
    finally:
        current_conversation_key.reset(token)

    assert ran.outcome is Outcome.DONE
    assert run is not None and run.status is RunStatus.COMPLETED
    assert run.idempotency_key == idempotency_key(
        caller=PRINCIPAL,
        conversation_id=ledger_conversation_id(PRINCIPAL, CONVERSATION),
        task=f"plan:{card.payload['revision']}",
    )
    assert ledger_conversation_id(PRINCIPAL, CONVERSATION) == f"{PRINCIPAL}/{CONVERSATION}"


class _ProcessDiedError(BaseException):
    """The process running the turn stopped here: nothing handles it, as
    nothing handles a process dying."""


class _DiesAtItsFirstAttempt(InMemoryRunLedger):
    """An in-memory run ledger whose process dies at the first attempt of a
    step while `dying`, losing that write and every one after it."""

    def __init__(self) -> None:
        super().__init__()
        self.dying = True
        self.dead = False

    async def record(self, run_id: str, change: RunChange) -> RunRecord:
        if self.dying and (self.dead or isinstance(change, Attempt)):
            self.dead = True
            raise _ProcessDiedError
        return await super().record(run_id, change)


async def test_a_rerun_of_execute_after_a_death_carries_the_same_run_on(
    pg_engine: AsyncEngine,
) -> None:
    """The process dies inside `execute` with the run admitted and left
    running. The turn stays at `execute`; the instance that takes it over
    runs the node again, and the run ledger's key gives it the same run,
    which it completes."""
    ledger = _DiesAtItsFirstAttempt()
    models = _planned("Charted.")
    first, second = (
        _driver(pg_engine, models, run_ledger=ledger),
        _driver(pg_engine, models, run_ledger=ledger),
    )
    card = (await first.turn(_QUESTION)).pause

    with pytest.raises(_ProcessDiedError):
        await first.answer(approve_answer(card.payload["revision"]), to=card)

    cut = await first.open_cursor()
    assert (cut.status, cut.node) == (TurnStatus.RUNNING, "execute")
    (left,) = await ledger.unfinished()
    assert left.status is RunStatus.RUNNING

    ledger.dying = False
    segment = await _take_over(second)

    assert segment.outcome is Outcome.DONE
    assert _hand_back_body(models)["run"]["run_id"] == left.run_id
    finished = await ledger.get(left.run_id)
    assert finished is not None and finished.status is RunStatus.COMPLETED
    assert await ledger.unfinished() == ()


@pytest.mark.parametrize("describes_frames", [True, False], ids=["True", "False"])
async def test_the_hand_back_reports_each_ask_with_its_output_status_or_decline(
    describes_frames: bool, pg_engine: AsyncEngine
) -> None:
    models = _planned(
        "Answered.",
        brief="Kenya's debt service since 2015, ranked against East Africa",
        asks=("Kenya's debt service each year", "a ranking of East African countries"),
    )
    driver = _driver(
        pg_engine,
        models,
        planner=AskingPlanner(),
        executor=describes if describes_frames else refuses_to_describe,
    )

    ran = await approve_turn(driver, _QUESTION)

    body = _hand_back_body(models)
    workflow = Plan.model_validate_json(ran.cursor.state["plan"]).workflow
    assert workflow is not None
    (final_step,) = workflow.terminal_steps
    output = (
        {"step": final_step, "status": "data", "frame": body["frames"][0]["frame"]}
        if describes_frames
        else {
            "step": final_step,
            "status": "not_chartable",
            "reason": body["frames_not_chartable"][0]["reason"],
        }
    )
    assert body["asks"] == [
        {"ask": "Kenya's debt service each year", "answered_by": [output]},
        {
            "ask": "a ranking of East African countries",
            "declined": DECLINE_REASONS[UnsupportedOperation.UNSUPPORTED_ASK],
        },
    ]


async def test_a_breakdown_refused_at_run_time_hands_back_a_failed_step_and_ask(
    pg_engine: AsyncEngine,
) -> None:
    models = _planned("It could not be computed.", asks=("Kenya's debt service",))
    driver = _driver(
        pg_engine, models, planner=AskingPlanner(), tool_surface=_BreakdownRefusingSurface()
    )

    ran = await approve_turn(driver, _QUESTION)

    body = _hand_back_body(models)
    assert body["failed_step"]["step_id"] == "retrieve"
    assert body["failed_step"]["code"] == "capability_unavailable"
    # The ask's output reads the refused retrieval, so it failed with it.
    failed = {"status": "failed", "failed_at": "retrieve", "code": "capability_unavailable"}
    assert body["asks"] == [
        {"ask": "Kenya's debt service", "answered_by": [{"step": "totals", **failed}]}
    ]
    # An output the failed step does not feed, and that never ran, was not run.
    stopped = _stop_causes(Plan.model_validate_json(ran.cursor.state["plan"]), "totals")
    assert stopped == {"totals": "totals"}
    upstream = PlanAsk(ask="the debt service series", outputs=("retrieve",))
    assert _ask_status(upstream, {}, {"step_id": "totals"}, stopped)["answered_by"] == [
        {"step": "retrieve", "status": "not_run"}
    ]


@pytest.mark.parametrize(
    ("code", "given_back"),
    [
        ("capability_unavailable", False),
        ("member_not_resolvable", True),
        ("place_levels_mixed", False),
        ("internal_error", False),
    ],
)
async def test_a_stopped_plan_gives_the_plan_call_back_only_for_a_failure_a_brief_avoids(
    code: str, given_back: bool, pg_engine: AsyncEngine
) -> None:
    """A capability the platform lacks fails any brief and any fetch of the
    step's data, so the plan call is not given back and the step's spec_id
    is refused to `fetch`. A fault of the system fails any brief too. Only a
    failure another source or spec avoids gives the plan call back; any
    other ends the turn on the user's sentence for it."""
    models = _planned("That part is unavailable.")
    driver = _driver(pg_engine, models, tool_surface=RefusingGetDataSurface(code))

    ran = await approve_turn(driver, _QUESTION)

    body = _hand_back_body(models)
    assert body["failed_step"]["code"] == code
    state = ran.cursor.state
    assert bool(state[ORCHESTRATOR_CALLS_KEY].get("plan_given_back")) is given_back
    assert bool(state.get(ORCHESTRATOR_RETRY_NOTE_KEY)) is given_back
    missing = code == "capability_unavailable"
    assert (SCRIPTED_SPEC in (state.get(ORCHESTRATOR_UNAVAILABLE_KEY) or {})) is missing
    after_plan = agent_read(models.orchestrator.requests[-1])
    assert (failure_sentence(code) in after_plan) is not given_back


async def test_a_system_fault_ends_the_turn_worded_as_the_systems_fault(
    pg_engine: AsyncEngine,
) -> None:
    """A code the compiler should have ruled out reaches the user as the
    system's fault, whatever its own class says, and offers no second plan."""
    models = _planned("It could not be computed.")
    advice = "Ask about one sector."
    surface = RefusingGetDataSurface("composed_unit_sum_unsupported", {"remedy": advice})
    driver = _driver(pg_engine, models, tool_surface=surface)

    ran = await approve_turn(driver, _QUESTION)

    assert ran.outcome is Outcome.DONE
    assert not ran.cursor.state[ORCHESTRATOR_CALLS_KEY].get("plan_given_back")
    after_plan = agent_read(models.orchestrator.requests[-1])
    assert failure_sentence("internal_error") in after_plan
    assert failure_sentence("composed_unit_sum_unsupported") not in after_plan
    # The tool's remedy is for a question at fault, and none is.
    assert advice not in after_plan


@pytest.mark.parametrize(
    ("code", "fix"),
    [
        ("member_not_resolvable", {}),
        # The user is told data-mcp's remedy only once no new plan is left.
        ("member_not_resolvable", {"remedy": "Ask about one sector."}),
        # Another facet is a plan the model can draft.
        ("facet_not_available", {}),
    ],
)
async def test_an_avoidable_failure_offers_one_new_plan_whose_card_names_the_failed_step(
    code: str, fix: dict[str, Any], pg_engine: AsyncEngine
) -> None:
    """The plan given back drafts a new card that says which step the
    approved plan stopped at and what happened there, the cause the
    refusal's code and `fix` word, with the new plan as the way past it; its
    run stopping the same way ends the turn on the user's whole sentence,
    with no third card."""
    models = scripted_models(
        orchestrator=[
            reply(plan_call(_BRIEF)),
            reply(plan_call(f"{_BRIEF}, from another source")),
            reply(text_part("It could not be computed.")),
        ]
    )
    surface = RefusingGetDataSurface(code, fix)
    driver = _driver(pg_engine, models, tool_surface=surface)
    sentence = failure_sentence(code, surface.retry_mode, fix)

    second = await approve_turn(driver, _QUESTION)

    card = second.pause
    assert card.kind == "approve"
    plan = Plan.model_validate_json(second.cursor.state["plan"])
    (failed,) = (row for row in card_rows(plan) if row["id"] == "retrieve")
    note = card.payload["refusal_note"]
    assert f"“{failed['label']}”" in note
    assert failure_cause(code, fix) in note
    assert sentence not in note
    assert second.cursor.state[ORCHESTRATOR_CALLS_KEY].get("plan_given_back")
    assert second.cursor.state.get(ORCHESTRATOR_RETRY_NOTE_KEY) is None

    ended = await driver.answer(approve_answer(card.payload["revision"]), to=card)

    assert ended.outcome is Outcome.DONE
    # The second stop leaves no note, since the turn's one new plan is spent.
    assert ended.cursor.state.get(ORCHESTRATOR_RETRY_NOTE_KEY) is None
    first, last = (agent_read(request) for request in models.orchestrator.requests[1:])
    if "remedy" in fix:
        # The plan given back is the way past it, so the hand-back states no remedy.
        assert fix["remedy"] not in first
    if "remedy" in fix:
        assert failure_cause(code, fix) in last
        assert fix["remedy"] in last
    else:
        assert sentence in last


@pytest.mark.parametrize(
    ("code", "fix"),
    [
        # Narrowing the topic is the user's call. The words name the ceiling
        # the refusal's `fix` names, and data-mcp's remedy gives the advice.
        (
            "over_ceiling",
            {"ceiling_kind": "member_cardinality", "remedy": "Ask about one sector."},
        ),
        ("place_levels_mixed", {}),
    ],
)
async def test_an_unavoidable_failure_ends_the_turn_on_its_sentence_with_no_second_card(
    code: str, fix: dict[str, Any], pg_engine: AsyncEngine
) -> None:
    """The agent's second plan call is refused for its budget, so the turn
    ends without another card, and the hand-back carries the user's
    sentence for the failure, or only its cause where a remedy gives the
    advice."""
    models = scripted_models(
        orchestrator=[
            reply(plan_call(_BRIEF)),
            reply(plan_call(f"{_BRIEF}, narrower")),
            reply(text_part("It could not be computed.")),
        ]
    )
    surface = RefusingGetDataSurface(code, fix)
    driver = _driver(pg_engine, models, tool_surface=surface)

    ran = await approve_turn(driver, _QUESTION)

    assert ran.outcome is Outcome.DONE
    assert not ran.cursor.state[ORCHESTRATOR_CALLS_KEY].get("plan_given_back")
    after_plan = agent_read(models.orchestrator.requests[-1])
    sentence = failure_sentence(code, surface.retry_mode, fix)
    if "remedy" in fix:
        assert failure_cause(code, fix) in after_plan
        assert sentence not in after_plan
    else:
        assert sentence in after_plan


def test_a_retry_note_survives_a_card_it_cannot_draw_and_words_a_stop_at_no_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A note never fails the turn: a card that cannot be drawn names "a step",
    and a stop at no step says the plan could not run."""

    def undrawable(plan: Plan) -> list[dict[str, object]]:
        raise ValueError("no card")

    monkeypatch.setattr(plan_done_module.plan_display, "card_rows", undrawable)
    plan = cast("Plan", SimpleNamespace(translation=object(), workflow=object()))
    failed = {"step_id": "retrieve", "code": "facet_not_available", "wave_remedy": "replan"}
    cause = failure_cause("facet_not_available")

    assert _retry_note(plan, failed).startswith(f"The approved plan stopped at a step. {cause} ")
    assert _retry_note(plan, {**failed, "step_id": None}).startswith(
        f"The approved plan could not run. {cause} "
    )


async def test_a_plan_whose_frames_cannot_be_described_hands_back_the_reason_not_a_frame(
    pg_engine: AsyncEngine,
) -> None:
    models = _planned("Here is what the plan computed.")
    driver = _driver(pg_engine, models, executor=refuses_to_describe)

    await approve_turn(driver, _QUESTION)

    after_plan = agent_read(models.orchestrator.requests[-1])
    assert "its result cannot be drawn" in after_plan
    body = _hand_back_body(models)
    assert body["frames"] == []
    assert body["frames_not_chartable"] == [
        {"step": "aggregate", "reason": "this step's data view could not be prepared"}
    ]
    assert "publish_not_configured" not in after_plan


def test_an_ask_answered_in_part_hands_back_its_steps_and_the_part_not_computed() -> None:
    code = UnsupportedOperation.CALCULATION_OVER_COMBINED
    ask = PlanAsk(
        ask="bilateral, imputed and their total",
        outputs=("joined",),
        declined=code,
        not_computed="their total",
    )

    assert _ask_status(ask, {}, None, {}) == {
        "ask": "bilateral, imputed and their total",
        "answered_by": [{"step": "joined", "status": "not_run"}],
        "not_computed": {"part": "their total", "reason": DECLINE_REASONS[code]},
    }


def _ranking(
    entities: int,
    *,
    limit: int | None = None,
    label: str = "",
    years: tuple[str, ...] = ("2023",),
    warnings: tuple[str, ...] = (),
) -> StepSummary:
    """The summary of a rank step over `entities` countries named `label` after
    their number, as `analysis_mcp` renders its digest, with the row count of
    the frame it keeps and, under a `limit`, the entries it ranked. Several
    `years` rank each year on its own. `warnings` are the run's."""
    frame = Frame.build(
        role_keys=["observationAbout"],
        rows=[
            {
                "stat_var": "dcid:LifeExpectancy",
                "observationAbout": f"country/C{number:02d}{label}",
                "time_period": year,
                "value": 80.0 - number,
                "unit": "dcid:Years",
            }
            for number in range(entities)
            for year in years
        ],
    )
    result = run_rank(
        frame,
        rank_by=("observationAbout",),
        aggregation=None,
        window_start=None,
        window_end=None,
        ascending=False,
        per_period=len(years) > 1,
        limit=limit,
    )
    digest = render_rank_digest(
        rank_by=("observationAbout",),
        result=dataclasses.replace(result, warnings=warnings),
        op_spec_hash="ops_" + "1" * 32,
        content_version="ocv_" + "1" * 32,
        execution_identity=current_execution_identity(),
    )
    ranked = len(result.entity_ranks) if limit is not None else None
    return StepSummary(digest, "rank", result.frame.num_rows, ranked)


def _handed_back(
    digest: str,
    tool: str | None,
    rows: int | None = None,
    ranked_before_limit: int | None = None,
) -> list[str]:
    """The lines the orchestrator reads of one `tool` step's `digest` in the hand-back."""
    summary = StepSummary(digest, tool, rows, ranked_before_limit)
    summaries = HandBack(refuses_to_describe, summaries=[summary]).listing({}, None)[
        "step_summaries"
    ]
    return summaries[0].split("\n")


def _rows_of(digest: str) -> list[str]:
    return [line for line in digest.split("\n") if line.startswith("  #")]


def test_a_ranking_of_twelve_hands_back_its_first_row_its_last_row_and_its_count() -> None:
    ranking = _ranking(12)
    rows = _rows_of(ranking.digest)

    handed = _handed_back(*ranking)

    assert len(rows) == 12
    assert "12 ranked entit(y/ies):" in handed
    assert [row for row in rows if row in handed] == [rows[0], rows[-1]]


def test_a_ranking_of_four_is_handed_back_whole() -> None:
    ranking = _ranking(4)

    assert _rows_of(ranking.digest) == [
        line for line in _handed_back(*ranking) if line.startswith("  #")
    ]


@pytest.mark.parametrize("limit", [10, 20], ids=["limit-10", "limit-20"])
def test_a_limited_ranking_hands_back_every_row_its_frame_keeps_and_no_others(limit: int) -> None:
    """The digest lists the rows the frame keeps, a `limit` of 20 or fewer of the
    70 ranked entities, and they are handed back whole."""
    ranking = _ranking(70, limit=limit)
    rows = _rows_of(ranking.digest)

    handed = _handed_back(*ranking)

    assert ranking.rows == limit and len(rows) == limit
    assert f"{limit} ranked entit(y/ies):" in handed
    assert f"limit {limit} (kept {limit} of 70 ranked entities)" in handed[0]
    assert _rows_of("\n".join(handed)) == rows


@pytest.mark.parametrize("periods", [3, 20], ids=["3-periods", "20-periods"])
def test_a_limited_ranking_of_each_period_hands_back_its_header_and_closing_lines(
    periods: int,
) -> None:
    """Blocks of 3 rows hold few lines each, yet a line per period would pass the
    hand-back cap and cut the lines after them."""
    years = tuple(str(2000 + offset) for offset in range(periods))
    ranking = _ranking(40, limit=3, years=years, warnings=("the window is sparse",))

    handed = _handed_back(*ranking)

    assert f"kept {3 * periods} of {40 * periods} ranked rows" in handed[0]
    assert not [line for line in handed if "ranked entit(y/ies)" in line or "#" in line]
    assert any(line.startswith("unit: ") for line in handed)
    assert "  - the window is sparse" in handed


@pytest.mark.parametrize(
    ("limit", "kept"), [(None, 70), (24, 24), (30, 30)], ids=["unlimited", "limit-24", "limit-30"]
)
def test_a_ranking_whose_frame_keeps_more_rows_than_the_digest_lists_hands_back_no_closing_row(
    limit: int | None, kept: int
) -> None:
    """The digest prints 20 rows. Rank 20 is the last row only of a frame that
    keeps 20, so only the first row is handed back, with a marker that says
    how many of the frame's rows the digest lists."""
    ranking = _ranking(70, limit=limit)
    rows = _rows_of(ranking.digest)

    handed = _handed_back(*ranking)

    assert ranking.rows == kept and len(rows) == 20
    assert [row for row in rows if row in handed] == [rows[0]]
    assert any(f"20 of the frame's {kept} rows" in line for line in handed)


def test_a_limited_ranking_over_the_card_cap_hands_back_its_first_and_last_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A frame that keeps more rows than a card draws is trimmed even when the
    digest lists them all (here a card draws 9 and the frame keeps 10)."""
    monkeypatch.setattr(plan_done_module, "MAX_SERIES_PER_COMPONENT", 9)
    ranking = _ranking(70, limit=10)
    rows = _rows_of(ranking.digest)

    handed = _handed_back(*ranking)

    assert [row for row in rows if row in handed] == [rows[0], rows[9]]


def test_a_ranking_stored_without_its_limit_is_trimmed_to_its_first_and_last_kept_row() -> None:
    """A summary stored before `ranked_before_limit` was carried has no value."""
    ranking = _ranking(70, limit=10)._replace(ranked_before_limit=None)
    rows = _rows_of(ranking.digest)

    handed = _handed_back(*ranking)

    assert [row for row in rows if row in handed] == [rows[0], rows[9]]
    assert (
        StepSummary.of({"tool": "rank", "digest": ranking.digest, "rows": 10}).ranked_before_limit
        is None
    )


def test_a_limited_ranking_too_long_to_hand_back_whole_is_trimmed_not_cut_mid_row() -> None:
    ranking = _ranking(70, limit=20, label="x" * 150)
    rows = _rows_of(ranking.digest)

    handed = _handed_back(*ranking)

    assert sum(len(line) + 1 for line in rows[:20]) > plan_done_module._WHOLE_RANKING_CHARS
    assert [row for row in rows if row in handed] == [rows[0], rows[19]]


@pytest.mark.parametrize("rows", [None, 0], ids=["unknown", "empty"])
def test_a_ranking_of_unknown_or_empty_frame_hands_back_its_count_and_no_rows(
    rows: int | None,
) -> None:
    ranking = _ranking(12)

    handed = _handed_back(ranking.digest, "rank", rows)

    assert "12 ranked entit(y/ies):" in handed
    assert _rows_of("\n".join(handed)) == []


# The shapes `analysis_mcp.digest` writes for the six other per-entity digests:
# the digest's opening line, the heading that states the count, one line.
_COUNT_ONLY_SHAPES = {
    "aggregate": (
        "aggregate: sum grouped by reporter",
        "6 group(s):",
        "  reporter=dcid:USA [1 row]: 1.0",
    ),
    "growth": (
        "growth: 2020..2023, entity_by reporter",
        "6 entit(y/ies):",
        "  reporter=dcid:USA: +10.0% (1 -> 2)",
    ),
    "filter": (
        "filter: 1 predicate(s) applied, 6/6 row(s) kept (0 dropped)",
        "6 kept row(s):",
        "  reporter=dcid:USA: 1.0 [Count]",
    ),
    "join": (
        "join: on reporter + time_period, 6 matched key(s) (0 left-only, 0 right-only, dropped)",
        "6 matched key(s):",
        "  reporter=dcid:USA [2021]: left=1.0, right=2.0",
    ),
    "quotient_of": (
        "quotient_of: 6 matched period(s), in Count",
        "6 quotient(s), numerator / denominator, in Count:",
        "  2020: 1 / 2 = 0.5",
    ),
    "share_of": (
        "share_of: 6 matched period(s), in per cent (Percent)",
        "6 period share(s):",
        "  2020: 1 / 2 = 50.000%",
    ),
}
_KEYED_SHARE = (
    "share_of: 1 key(s), 6 matched (key, period) pair(s), in per cent (Percent)",
    "reporter=dcid:USA [6 period(s)], first 2020: 50.000%; latest 2025: 50.000%:",
    "  2020: 1 / 2 = 50.000%",
)


def _entity_digest(shape: tuple[str, str, str], *, lines: int) -> str:
    opening, heading, line = shape
    return "\n".join([opening, "", heading, *[line] * lines, "", "unit: Count"])


@pytest.mark.parametrize(
    ("tool", "shape"),
    [*_COUNT_ONLY_SHAPES.items(), ("share_of", _KEYED_SHARE)],
    ids=[*_COUNT_ONLY_SHAPES, "keyed share_of"],
)
def test_a_per_entity_digest_past_four_lines_hands_back_its_count_and_none_of_its_lines(
    tool: str, shape: tuple[str, str, str]
) -> None:
    _, heading, line = shape

    handed = _handed_back(_entity_digest(shape, lines=5), tool)

    assert heading in handed
    assert line not in handed
    assert _handed_back(_entity_digest(shape, lines=4), tool).count(line) == 4


@pytest.mark.parametrize("tool", [None, "compare", "get_data"], ids=["no-tool", "compare", "data"])
def test_a_digest_of_a_step_not_known_to_print_one_line_per_entity_is_handed_back_whole(
    tool: str | None,
) -> None:
    """A summary stored without its tool, as a run paused before the tool was
    kept leaves it, reads whole."""
    digest = _entity_digest(_COUNT_ONLY_SHAPES["growth"], lines=9)

    assert _handed_back(digest, tool) == digest.split("\n")


@pytest.mark.parametrize(
    ("status", "charted"), [({"status": "data"}, True), ({"status": "empty"}, False)]
)
def test_a_partial_result_offers_to_chart_only_frames_that_hold_series(
    status: dict[str, str], charted: bool
):
    asks = [{"ask": "the mean", "answered_by": [{"step": "totals", **status, "frame": "f1"}]}]

    text = _partial_result(asks, frames=True)

    assert ('can be charted: "the mean".' in text) is charted
    assert (
        'holds no series or cannot be drawn, as each step\'s status says: "the mean".' in text
    ) is not charted


async def test_the_plan_hand_back_reaches_the_agent_with_its_fields_in_the_order_written(
    pg_engine: AsyncEngine,
) -> None:
    """The hand-back crosses the node boundary through the cursor, and the
    agent reads its fields in the order `plan_done` wrote them: the result,
    the frames, the step summaries, the totals, the run's standing, then the
    plan record. A store that reorders an object's keys would change the
    bytes of every later request."""
    models = _planned("Charted.")
    driver = _driver(pg_engine, models, run_ledger=InMemoryRunLedger())

    await approve_turn(driver, _QUESTION)

    (response,) = [
        part.function_response.response
        for content in models.orchestrator.requests[-1].content_list
        for part in content.parts or []
        if part.function_response is not None and part.function_response.name == "plan"
    ]
    assert list(response) == [
        "result",
        "frames",
        "step_summaries",
        "totals",
        "run",
        "plan_record",
    ]


async def test_an_approved_plan_runs_and_its_final_frames_come_back_to_the_agent(
    pg_engine: AsyncEngine,
) -> None:
    """The approved plan's frames come back to the agent as its `plan`
    call's result, registered under the ids it charts; the follow-up turn
    reads what the plan ran from the conversation digest the studio builds
    from the closed turn (`ConversationState.conversation_digest_state`)."""
    models = _planned("Charted.", "Uganda next.")
    driver = _driver(pg_engine, models)

    paused = await driver.turn(_QUESTION)
    card = paused.pause
    assert card.kind == "approve"
    revision = card.payload["revision"]
    ran = await driver.answer(approve_answer(revision), to=card)

    assert ran.outcome is Outcome.DONE
    (response,) = [
        part.function_response.response
        for content in models.orchestrator.requests[-1].content_list
        for part in content.parts or []
        if part.function_response is not None and part.function_response.name == "plan"
    ]
    assert response["result"] == "The approved plan ran. Chart its frames and answer."
    assert response["frames"] and all(f["frame"].startswith("f") for f in response["frames"])
    assert response["step_summaries"]
    assert response["plan_record"].startswith(f"Plan {revision[:12]} (approved)")
    assert set(ran.cursor.state[ORCHESTRATOR_FRAMES_KEY]) == {
        f["frame"] for f in response["frames"]
    }

    closed = ConversationState(conversation_id=CONVERSATION)
    closed.turn_user_message = _QUESTION
    closed.turn_plans = list(ran.cursor.state[ORCHESTRATOR_PLAN_RECORDS_KEY])
    closed.close_turn(final_text="Charted.")
    follow_up = await driver.turn(
        "And Uganda?",
        turn_index=2,
        inputs={"conversation_digest": json.dumps(closed.conversation_digest_state())},
    )

    assert follow_up.outcome is Outcome.DONE
    read = agent_read(models.orchestrator.requests[-1])
    (plan,) = closed.turn_plans
    assert PlanRecord.model_validate(plan).digest_line() in read
    assert "Charted." in read


class _UnusableGetDataSurface(ScriptedSurface):
    """`ScriptedSurface` whose `get_data` answers a result that states no frame."""

    async def call_tool(self, *, tool: Any, arguments: Any, on_progress: Any = None) -> Any:
        if tool.name != "get_data":
            return await super().call_tool(tool=tool, arguments=arguments, on_progress=on_progress)
        return ToolCallResult(is_error=False, structured_content={}, text="fetched")


class _RefusingReplanner(ScriptedPlannerModel):
    """`ScriptedPlannerModel` that drafts the first plan and refuses, not
    finally, to replan past the stop of its run."""

    async def __call__(self, **kwargs: Any) -> dict[str, Any]:
        if kwargs.get("refusal") is not None:
            return {"refused": {"text": "No plan gets past that step.", "final": False}}
        return await super().__call__(**kwargs)


@pytest.mark.parametrize(
    ("surface", "told"),
    [
        # A method choice is the planner's to make, so another brief may make it.
        (RefusingGetDataSurface("entity_ambiguous", {"candidates": ["country/KEN"]}), None),
        # A result the run cannot use is the system's fault, which no brief avoids.
        (_UnusableGetDataSurface(), failure_sentence("internal_error")),
        # A service that did not answer is waited out, not planned around, and
        # told as its code tells it.
        (
            RefusingGetDataSurface("service_unavailable"),
            failure_sentence("service_unavailable", "retry_after_delay"),
        ),
    ],
    ids=["method-choice", "system-fault", "wait"],
)
async def test_a_planner_refusal_after_a_stop_gives_the_plan_back_only_when_a_new_plan_remedies_it(
    surface: ScriptedSurface,
    told: str | None,
    pg_engine: AsyncEngine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The run retries a service that did not answer with no backoff between.
    monkeypatch.setattr(classify_module, "BACKOFF_SECONDS", 0.0)
    models = _planned("It could not be computed.")
    driver = _driver(pg_engine, models, planner=_RefusingReplanner(), tool_surface=surface)

    ran = await approve_turn(driver, _QUESTION)

    assert ran.outcome is Outcome.DONE
    after_plan = agent_read(models.orchestrator.requests[-1])
    given_back = told is None
    assert bool(ran.cursor.state[ORCHESTRATOR_CALLS_KEY].get("plan_given_back")) is given_back
    assert bool(ran.cursor.state.get(ORCHESTRATOR_RETRY_NOTE_KEY)) is given_back
    if told is not None:
        assert told in after_plan

"""Tests for `harness.planner_seam` -- the adapter over `planner.agent`
with its `translate` tool.

Model-free: a scripted `BaseLlm` stands in for the planner's own model. Most
scenarios either hand the seam an already-compiled `Plan` (built the way
`tests/planner/test_agent.py`'s `_translate` helper builds one, through the
real `Plan.with_translation`) so the scripted model can go straight to
`submit_plan`, or leave the plan a draft. The referral-stall tests below
script a `translate` call, which runs the real translation loop against a
canned surface. Whether the real model plans WELL is the eval sets' question
(`evals/README.md`).

`PlannerSeam` runs the planner through `ctx.run_node`, in the graph's own
session, so every test below needs a real `Context`, not a bare call.
`_run_seam` drives one, through a minimal one-node `Workflow`, the same
primitive `harness.graph`'s own planner node uses.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from compiler.surface import FixtureSurface, ToolCallResult
from google.adk import Context, Event
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.base_tool import BaseTool
from google.adk.workflow import START, Workflow, node
from google.adk.workflow.utils._workflow_hitl_utils import (
    REQUEST_INPUT_FUNCTION_CALL_NAME,
    create_request_input_response,
)
from google.genai import types
from harness import planner_seam
from harness.graph import (
    ORCHESTRATOR_PRIOR_LEDGER_KEY,
    ORCHESTRATOR_SPEC_WARNINGS_KEY,
    ORCHESTRATOR_SPECS_KEY,
    frame_row_ceiling,
)
from harness.planner_seam import (
    _DEADLINE_REFUSAL_CODE,
    _ENVELOPE_FIELD_CAP,
    _ENVELOPE_SKILL_CAP,
    _ENVELOPE_SKILL_DESCRIPTION_CAP,
    _MAX_CORRECTIONS,
    _NEXT_MODEL_CALL_MARGIN_S,
    _PLANNER_BRANCH,
    PlannerSeam,
    PlannerSeamError,
    _check_branches,
    _clarifying_question_args,
    _envelope_context,
    _exposed_spec_id,
    _pinnable_candidates,
    _plan_context,
    _render_envelope,
    _requirements_without_discovery,
    _RunTracker,
    _seed_of,
)
from harness.settings import Settings
from harness.studio_state import ConversationState
from harness.tool_contracts import graph_envelope
from harness.turn_records import SearchReading
from planner.agent import _elected_spec_ids
from planner.operations import AggregateOperation, RetrieveOperation, Translation
from planner.plan import (
    ASKS_STATE_KEY,
    CUT_MARKER,
    MAX_PLAN_ROUNDS,
    MAX_QUESTION_CHARS,
    UNSUPPORTED_STATE_KEY,
    AggregateRequirement,
    DatasetRequirement,
    Decision,
    Entity,
    JoinRequirement,
    Period,
    PinSource,
    Plan,
    PlanStatus,
    ResolvedBy,
    SeededSource,
    SessionState,
    Unit,
    current_earlier_block,
    discovery_time_left,
    load_plan,
    pin_sources,
    plan_revision,
    save_plan,
    save_seeded_sources,
    windowed_copy,
)
from planner.records import GroupBy, Reduction
from planner.settings import (
    AGENT_THINKING_CONFIG,
    DEFAULT_THINKING_CONFIG,
    THINKING_SAMPLE_STATE_KEY,
    AgentSettings,
    drop_thought_parts,
    override_thinking_on_sampled_turns,
)
from planner.tool_results import Outcome
from planner.workflow import EvidenceBinding, SelectedSource, ToolId, WorkflowStep
from planner.workflow import Workflow as PlanWorkflow
from pydantic import Field
from translator.mechanical import COMPILE_FAILURES_STATE_KEY
from translator.referral import OUTCOME_OF

from .search_results import search_result_parts

SPEC = "spec_" + "a" * 32
SPEC_B = "spec_" + "b" * 32


_APP = "planner_seam_test"
_USER = "u"


class ScriptedPlannerLlm(BaseLlm):
    """Answers each generation from `script`, in order, and records the requests.

    A script entry is either a list of `(tool_name, args)` function calls, a
    string of final text, or a callable that reads the `LlmRequest` and
    returns one of the other two; ADK asks for one more generation after
    every tool response but `submit_plan`'s ok answer, and a text answer is
    what ends a run. Past the end of the script it says "done", so an
    unscripted extra generation stops the agent rather than looping.

    The callable entry is what lets a script compute a function call's args
    from what the model has already been shown and answered -- an
    `amends_version` a `propose_plan` call reads off the version an earlier
    call in the same turn answered with, say.

    `pause_at`, if set, makes the generation at that index (0-based, by
    request order) set `paused` and then wait on `release` before answering
    -- a test's way of suspending this call's own `ctx.run_node` at a
    controlled point, to interleave a second, concurrent call in between.
    """

    model: str = "scripted-planner-v2"
    script: list[Any] = Field(default_factory=list)
    captured_requests: list[Any] = Field(default_factory=list)
    pause_at: int | None = None
    paused: Any = None
    release: Any = None

    async def generate_content_async(self, llm_request: Any, stream: bool = False) -> Any:
        del stream
        index = len(self.captured_requests)
        if self.pause_at is not None and index == self.pause_at:
            if self.paused is not None:
                self.paused.set()
            if self.release is not None:
                await self.release.wait()
        self.captured_requests.append(llm_request)
        step = self.script[index] if index < len(self.script) else "done"
        if callable(step):
            step = step(llm_request)
        if isinstance(step, str):
            yield LlmResponse(
                content=types.Content(role="model", parts=[types.Part.from_text(text=step)])
            )
            return
        yield LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part.from_function_call(name=name, args=args) for name, args in step],
            )
        )


def _prompt_text(request: Any) -> str:
    """Every user-authored word the model was shown, joined."""
    return "\n".join(
        part.text or ""
        for content in request.contents
        if content.role == "user"
        for part in content.parts
    )


def _call_and_response_names(request: Any) -> list[str]:
    """`FC:<tool>` / `FR:<tool>` labels, in order, for every function call and
    function response part `request.contents` carries -- what tool-call
    history a generation actually saw, not just its text.
    """
    names: list[str] = []
    for content in request.contents:
        for part in content.parts or []:
            if part.function_call is not None:
                names.append(f"FC:{part.function_call.name}")
            elif part.function_response is not None:
                names.append(f"FR:{part.function_response.name}")
    return names


def _last_amended_version(request: Any, *, default: int) -> int:
    """The version the most recent `propose_plan` answer in `request` recorded,
    or `default` when this generation has not seen one.

    `propose_plan` answers with `version`, and that number is where a model
    reads the `amends_version` its next call has to state. A request with no
    such answer in it yet is the turn's first proposal: the model has the
    version from the seam brief's own "version N" line, which `default`
    stands in for.
    """
    for content in reversed(request.contents):
        for part in content.parts or []:
            response = part.function_response
            if response is None or response.name != "propose_plan":
                continue
            version = (response.response or {}).get("version")
            assert isinstance(version, int), f"propose_plan answered without a version: {response}"
            return version
    return default


def _seam(
    *,
    agent_settings: AgentSettings | None = None,
    clock: Callable[[], float] = time.monotonic,
    thinking_sample_rng: Callable[[], float] = random.random,
) -> PlannerSeam:
    return PlannerSeam(
        settings=Settings(),
        surface=FixtureSurface(schemas=[]),
        agent_settings=agent_settings,
        clock=clock,
        thinking_sample_rng=thinking_sample_rng,
    )


class _SeamHarness:
    """Drives `PlannerSeam.__call__` through a real, dynamically-scheduled
    `Context` -- the same primitive `harness.graph`'s own planner node
    drives `PlannerSeam` through, and the only way to get a `Context`
    `ctx.run_node` will accept, since this seam has no `Runner` of its
    own to call directly.

    One `_SeamHarness` reuses the same session across every `run` call,
    the way the graph's own session persists across a turn's several
    `_MAX_CORRECTIONS` rounds and across separate turns of one
    conversation alike -- so a test can inspect `ctx.state` (`state()`)
    after a call that raised, without losing what it left behind.
    """

    def __init__(self, state: Mapping[str, Any] | None = None) -> None:
        self._sessions = InMemorySessionService()
        self._session_id: str | None = None
        self._initial_state = dict(state or {})

    async def run(
        self,
        seam: PlannerSeam,
        *,
        resume_inputs: dict[str, Any] | None = None,
        new_message: types.Content | None = None,
        **kwargs: Any,
    ) -> dict[str, Any] | None:
        async def _node(ctx: Context, node_input: Any) -> Any:
            del node_input
            if resume_inputs is not None:
                # `ctx.resume_inputs` reflects a genuine pause-and-resume of
                # THIS node's own interrupt -- `clarify`/`approve`'s own
                # `yield RequestInput(...)` -- and, confirmed directly
                # against the installed ADK (2.9.1), a nested one too: a
                # planner-side `adk_request_input` call (`planner.agent`'s
                # tool) pauses a `ctx.run_node(self._agent, ...)` call
                # several levels down, but `_create_child_context`
                # (`workflow/_node_runner.py`) reconstructs resume state
                # through `_reconstruct_node_states` with
                # `group_by_direct_child=False`
                # (`workflow/utils/_rehydration_utils.py`'s own default),
                # which attributes any event under this node's own path --
                # descendants included -- to this node's own path, not just
                # this node's direct interrupt. So the planner's own ask
                # does resolve into this node's `ctx.resume_inputs` on a
                # real resume (see `test_a_resumed_ask_gets_the_continuation_
                # not_the_full_brief`, which drives one for real). Setting
                # it directly here is only for a test that needs the OTHER
                # resume condition -- "has `__call__` already been through
                # this session once" -- without paying for a real interrupt
                # round trip, the same way a hand-built `Context` elsewhere
                # in this suite stands in for one the framework would
                # otherwise build.
                ctx._resume_inputs = resume_inputs
            return await seam(ctx=ctx, **kwargs)

        node_fn = node(_node, name="seam_under_test", rerun_on_resume=True)
        wf = Workflow(name="seam_test_wf", edges=[(START, node_fn)])
        runner = Runner(node=wf, app_name=_APP, session_service=self._sessions)
        if self._session_id is None:
            session = await self._sessions.create_session(
                app_name=_APP, user_id=_USER, state=self._initial_state
            )
            self._session_id = session.id
        content = new_message or types.Content(role="user", parts=[types.Part(text="probe")])
        output: Any = None
        async for event in runner.run_async(
            user_id=_USER, session_id=self._session_id, new_message=content
        ):
            assert isinstance(event, Event)
            if event.output is not None:
                output = event.output
        return cast(dict[str, Any] | None, output)

    async def state(self) -> SessionState:
        assert self._session_id is not None, "run() has not been called yet"
        session = await self._sessions.get_session(
            app_name=_APP, user_id=_USER, session_id=self._session_id
        )
        assert session is not None
        return cast(SessionState, session.state)

    async def events(self) -> list[Event]:
        """This session's own event history -- what `planner.agent.
        _elected_spec_ids` reads to find what the user actually chose
        answering `adk_request_input`, and what `ctx.run_node`'s own
        pause-and-resume depends on surviving (`harness.planner_seam`'s
        module docstring)."""
        assert self._session_id is not None, "run() has not been called yet"
        session = await self._sessions.get_session(
            app_name=_APP, user_id=_USER, session_id=self._session_id
        )
        assert session is not None
        return session.events

    def seed_state(self, key: str, value: Any) -> None:
        """Write directly into the canonical stored session, for a test that
        needs to set up state no real call would produce -- `get_session`
        (`state`, above) hands back a shallow copy (`InMemorySessionService`'s
        own `_copy_session`), so a write through it is invisible to every
        later `run()`.
        """
        assert self._session_id is not None, "run() has not been called yet"
        self._sessions.sessions[_APP][_USER][self._session_id].state[key] = value


async def _run_seam(seam: PlannerSeam, **kwargs: Any) -> dict[str, Any]:
    """One-shot `_SeamHarness.run`, for a test that only makes one call and
    does not need to inspect `ctx.state` afterward."""
    output = await _SeamHarness().run(seam, **kwargs)
    assert output is not None, "the seam node produced no output -- did it raise silently?"
    return output


def _translated_plan(
    decision: Any, requirement: Any, derived_requirement: Any, *, version: int = 1
) -> Plan:
    """A `PENDING_APPROVAL` plan, built through the real `with_translation`
    transition -- the minimal shape a graph node and `submit_plan` alike
    accept -- mirroring `tests/planner/test_agent.py`'s `_translate` helper.
    """
    made_decision: Decision = decision()
    dataset = requirement()
    derived = derived_requirement()
    draft = Plan(
        question="kenya debt",
        decisions=[made_decision],
        evidence_requirements=[dataset, derived],
        version=version,
    )
    translation = Translation(
        operations=[
            RetrieveOperation(id="debt", requirement_id=dataset.id, spec_id=SPEC),
            AggregateOperation(
                id="totals",
                requirement_id=derived.id,
                input="debt",
                reduction=derived.reduction,
                group_by=derived.group_by,
            ),
        ],
        terminal=["totals"],
    )
    workflow = PlanWorkflow(
        steps=[
            WorkflowStep(
                id="debt",
                operation_id="debt",
                tool=ToolId(server="data-mcp", name="get_data"),
                args={"spec_id": SPEC},
                input_schema_hash="hash",
            ),
            WorkflowStep(
                id="totals",
                operation_id="totals",
                tool=ToolId(server="analysis-mcp", name="aggregate"),
                args={"aggregation": "sum", "group_by": ["time_period"]},
                bindings={"frame_ref": {"from_step": "debt", "output": "frame_ref"}},
                input_schema_hash="hash",
            ),
        ],
        sources=[SelectedSource(spec_id=SPEC, sentence="external debt service", has_data=True)],
        evidence_bindings=[
            EvidenceBinding(requirement_id=dataset.id, step_id="debt", spec_id=SPEC)
        ],
        terminal_steps=["totals"],
    )
    return draft.with_translation(translation=translation, workflow=workflow)


async def test_the_seam_returns_a_plan_a_graph_node_can_validate(
    decision: Any, requirement: Any, derived_requirement: Any
) -> None:
    plan = _translated_plan(decision, requirement, derived_requirement)
    seam = _seam()
    seam._agent.model = ScriptedPlannerLlm(script=[[("submit_plan", {"version": plan.version})]])

    drafted = await _run_seam(
        seam, question=plan.question, envelope={}, prior_plan=plan.model_dump(mode="json")
    )

    # The node's own validation (`Plan.model_validate(drafted["plan"])` in
    # `harness.graph.planner`), repeated here as the contract this seam owes it.
    validated = Plan.model_validate(drafted["plan"])
    assert validated.version == plan.version
    assert validated.status is PlanStatus.PENDING_APPROVAL
    assert drafted["summary"]["revision"] == plan_revision(plan)
    assert drafted["summary"]["version"] == plan.version


async def test_on_progress_reports_the_time_since_the_previous_round_trip(
    decision: Any, requirement: Any, derived_requirement: Any
) -> None:
    """Each `on_progress` report's `duration_ms` covers everything since the
    previous report (or since the run started, for the first) -- the
    number that lets a ten-thousand-token thinking round read differently
    from a one-second lookup on a feed where every line otherwise prints
    the same way the instant it arrives."""
    plan = _translated_plan(decision, requirement, derived_requirement)
    seam = _seam()
    seam._agent.model = ScriptedPlannerLlm(script=[[("submit_plan", {"version": plan.version})]])
    reports: list[dict[str, Any]] = []

    await _run_seam(
        seam,
        question=plan.question,
        envelope={},
        prior_plan=plan.model_dump(mode="json"),
        on_progress=reports.append,
    )

    assert [(report["agent"], report["tools"]) for report in reports] == [
        ("dfo_planner", ["submit_plan"])
    ]
    assert isinstance(reports[0]["duration_ms"], int)
    assert reports[0]["duration_ms"] >= 0


async def test_a_prior_plan_is_what_propose_plan_amends_from(
    decision: Any, requirement: Any, derived_requirement: Any
) -> None:
    """An amendment starts from the graph's own record, not whatever this
    seam last left in the session: seeding `prior_plan` through
    `save_plan` on every call means the second call's version-5 record,
    not the first call's version-1 draft, is what the amendment builds on
    -- even though both calls share the same session.

    The second call states `amends_version=5`, the version the seed put on
    record. `propose_plan` reads intent off the call, never off a question's
    wording, so the seed is what makes that version the right one to name: a
    session still holding the first call's version-1 draft refuses this call
    as amending the wrong version. What records version 6 is therefore the
    graph's own seeded record, not what this seam left in the session.
    """
    harness = _SeamHarness()
    seam = _seam()
    made_decision = decision()
    requirements = [requirement(), derived_requirement()]
    proposal = {
        "question": "kenya debt",
        "answer_mode": "computed",
        "decisions": [made_decision.model_dump(mode="json")],
        "evidence_requirements": [r.model_dump(mode="json") for r in requirements],
    }

    seam._agent.model = ScriptedPlannerLlm(script=[[("propose_plan", proposal)]])
    with pytest.raises(PlannerSeamError):
        # A draft with no translation never reaches submit_plan, so this
        # call never resolves -- it exists only to leave a version-1 plan
        # in the session, to prove the second call does not read it back.
        await harness.run(seam, question="kenya debt", envelope={})
    left_behind = load_plan(await harness.state())
    assert left_behind is not None
    assert left_behind.version == 1

    seeded = _translated_plan(decision, requirement, derived_requirement, version=5)
    seam._agent.model = ScriptedPlannerLlm(
        script=[[("propose_plan", {**proposal, "amends_version": 5})]]
    )
    with pytest.raises(PlannerSeamError):
        await harness.run(
            seam,
            question="kenya debt",
            envelope={},
            prior_plan=seeded.model_dump(mode="json"),
            amendment="only since 2015",
        )

    amended = load_plan(await harness.state())
    assert amended is not None
    assert amended.version == 6
    assert amended.previous_version == 5


async def test_a_model_that_never_submits_is_corrected_then_raises() -> None:
    seam = _seam()
    llm = ScriptedPlannerLlm(script=["Which years did you have in mind?"] * 4)
    seam._agent.model = llm

    with pytest.raises(PlannerSeamError, match="submit_plan"):
        await _run_seam(seam, question="kenya debt", envelope={})

    # One generation per `_propose` call: the first, plus one per
    # correction round (`_MAX_CORRECTIONS`), each its own `ctx.run_node`
    # call carrying the tool's own nudge forward as a short continuing
    # message (`fresh=False`) -- the full brief the first call sent is
    # already in the session's own history.
    assert len(llm.captured_requests) == 3
    for request in llm.captured_requests[1:]:
        assert "submit_plan" in _prompt_text(request)


@pytest.mark.parametrize("resumed", [False, True])
async def test_an_entry_owes_nothing_an_earlier_entry_recorded(resumed: bool) -> None:
    """A new planner entry starts with no `unsupported` answer to decline and no
    compile failure to call repeated; an ask-the-user resume inside one entry
    keeps both."""
    stale = {
        UNSUPPORTED_STATE_KEY: ["calculation_over_combined"],
        COMPILE_FAILURES_STATE_KEY: ["fingerprint"],
    }
    harness = _SeamHarness(state=stale)
    seam = _seam()
    seam._agent.model = ScriptedPlannerLlm(script=["Which years did you have in mind?"] * 4)

    with pytest.raises(PlannerSeamError):
        await harness.run(
            seam,
            question="kenya debt",
            envelope={},
            resume_inputs={"ask": {"result": "since 2015"}} if resumed else None,
        )

    state = await harness.state()
    kept = {key: state.get(key) for key in stale}
    assert kept == (stale if resumed else {key: [] for key in stale})


async def test_a_question_with_nothing_left_after_a_decline_ends_on_the_decline(
    decision: Any, requirement: Any
) -> None:
    """The planner hears `unsupported` for a total over a join, is refused a plan
    that drops the total without a decline, then proposes the decline alone. That
    ends the entry with a final refusal naming the declined ask: no further
    generation, no plan, and no correction text for the user to read."""
    seam = _seam()
    made_decision = decision().model_dump(mode="json")
    bilateral = requirement(id="bilateral").model_dump(mode="json")
    multilateral = requirement(id="multilateral", measure="imputed multilateral aid")
    summed_join = [
        bilateral,
        multilateral.model_dump(mode="json"),
        {
            "kind": "join",
            "id": "joined",
            "decision_ids": ["measure"],
            "left_id": "bilateral",
            "right_id": "multilateral",
        },
        {
            "kind": "aggregate",
            "id": "total",
            "decision_ids": ["measure"],
            "input_id": "joined",
            "reduction": "sum",
            "group_by": ["period"],
        },
    ]
    first_draft = {"question": "total aid", "answer_mode": "computed", "decisions": [made_decision]}
    llm = ScriptedPlannerLlm(
        script=[
            [("propose_plan", {**first_draft, "evidence_requirements": summed_join})],
            [
                (
                    "propose_plan",
                    {**first_draft, "answer_mode": "series", "evidence_requirements": [bilateral]},
                )
            ],
            [
                (
                    "propose_plan",
                    {
                        **first_draft,
                        "declined": [{"ask": "the total aid", "code": "calculation_over_combined"}],
                    },
                )
            ],
        ]
    )
    seam._agent.model = llm
    harness = _SeamHarness()

    result = await harness.run(seam, question="total aid", envelope={})

    assert result is not None
    assert result["refused"]["final"] is True
    assert "the total aid" in result["refused"]["text"]
    assert len(llm.captured_requests) == 3
    answered = [
        part.function_response.response
        for content in llm.captured_requests[-1].contents
        for part in content.parts
        if part.function_response and part.function_response.name == "propose_plan"
    ]
    assert [(r["outcome"], r["code"]) for r in answered] == [
        ("unsupported", "operation_unsupported"),
        ("invalid_input", "decline_missing"),
    ]
    assert load_plan(await harness.state()) is None


async def test_the_brief_lists_the_asks_and_propose_plan_holds_the_plan_to_them(
    decision: Any, requirement: Any
) -> None:
    """The asks in state render numbered after the question; a draft that leaves one
    unaccounted for is refused naming it, a decline no `unsupported` answer earned is
    refused, and the redraft that declines it once one has records the mapping on
    the plan."""
    asks = ["bilateral aid to Guatemala", "the total of bilateral and multilateral aid"]
    seam = _seam()
    first_draft = {
        "question": "total aid",
        "answer_mode": "series",
        "decisions": [decision().model_dump(mode="json")],
        "evidence_requirements": [requirement(id="bilateral").model_dump(mode="json")],
        "answered": [{"ask": 0, "outputs": ["bilateral"]}],
    }
    decline = {"ask": asks[1], "code": "calculation_over_combined", "ask_ref": 1}
    multilateral = requirement(id="multilateral", measure="imputed multilateral aid")
    joined = JoinRequirement(
        id="joined", decision_ids=["measure"], left_id="bilateral", right_id="multilateral"
    )
    summed = AggregateRequirement(
        id="total",
        decision_ids=["measure"],
        input_id="joined",
        reduction=Reduction.SUM,
        group_by=(GroupBy.PERIOD,),
    )
    summed_join = {
        **first_draft,
        "evidence_requirements": [
            *first_draft["evidence_requirements"],
            *(r.model_dump(mode="json") for r in (multilateral, joined, summed)),
        ],
    }
    llm = ScriptedPlannerLlm(
        script=[
            [("propose_plan", first_draft)],
            [("propose_plan", {**first_draft, "declined": [decline]})],
            [("propose_plan", summed_join)],
            [("propose_plan", {**first_draft, "declined": [decline]})],
            "Planned.",
        ]
    )
    seam._agent.model = llm
    harness = _SeamHarness(state={ASKS_STATE_KEY: asks})

    with pytest.raises(PlannerSeamError, match="submit_plan"):
        await harness.run(seam, question="total aid", envelope={})

    assert "0. bilateral aid to Guatemala\n1. the total of bilateral" in _prompt_text(
        llm.captured_requests[0]
    )
    answered = [
        part.function_response.response
        for content in llm.captured_requests[-1].contents
        for part in content.parts
        if part.function_response and part.function_response.name == "propose_plan"
    ]
    assert [(r["outcome"], r.get("code")) for r in answered] == [
        ("invalid_input", "asks_unaccounted"),
        ("invalid_input", "decline_unearned"),
        ("unsupported", "operation_unsupported"),
        ("ok", None),
    ]
    assert answered[0]["detail"] == {"uncovered": [1], "repeated": []}
    recorded = load_plan(await harness.state())
    assert recorded is not None
    assert recorded.asks == tuple(asks)
    assert [(a.ask, a.outputs) for a in recorded.answered] == [(0, ("bilateral",))]
    assert [d.ask_ref for d in recorded.declined] == [1]


def _unit_refusal(code: str) -> dict[str, Any]:
    """A `MethodChoiceNeeded` replan payload (`graph.py`'s `_replan_refusal`)
    for `share_of` refusing a pair of sides on their units."""
    return {
        "kind": "method_choice_needed",
        "step_id": "per_person",
        "tool": "analysis-mcp.share_of",
        "code": code,
        "candidates": None,
        "refusal": {"code": code, "message": "refused", "retry_mode": "correct_and_retry"},
        "also_stopped": [],
    }


@pytest.mark.parametrize(
    ("refusal", "outcome"),
    [
        pytest.param(_unit_refusal("cross_frame_unit_unbound"), ("ok", None), id="no-operation"),
        pytest.param(
            {
                **_unit_refusal("growth_gap_year"),
                "also_stopped": [_unit_refusal("cross_frame_unit_unbound")],
            },
            ("ok", None),
            id="sibling",
        ),
        pytest.param(
            _unit_refusal("cross_frame_unit_mismatch"),
            ("invalid_input", "decline_unearned"),
            id="method-choice",
        ),
        pytest.param(None, ("invalid_input", "decline_unearned"), id="no-refusal"),
    ],
)
async def test_a_replan_over_a_refusal_no_operation_covers_earns_the_decline(
    decision: Any, requirement: Any, refusal: dict[str, Any] | None, outcome: tuple[str, Any]
) -> None:
    """A replan entry whose refusal means no operation computes the step may
    decline the ask it served as `no_operation_covers`. With a refusal that
    offers a choice, or none, the decline is unearned, and the refusal steers to
    no other output."""
    asks = ["GDP", "population", "GDP divided by population"]
    draft = {
        "question": "GDP per person",
        "answer_mode": "series",
        "decisions": [decision().model_dump(mode="json")],
        "evidence_requirements": [
            requirement(id="gdp").model_dump(mode="json"),
            requirement(id="people", measure="population").model_dump(mode="json"),
        ],
        "answered": [{"ask": 0, "outputs": ["gdp"]}, {"ask": 1, "outputs": ["people"]}],
        "declined": [{"ask": asks[2], "code": "no_operation_covers", "ask_ref": 2}],
    }
    seam = _seam()
    llm = ScriptedPlannerLlm(script=[[("propose_plan", draft)], "Planned."])
    seam._agent.model = llm
    harness = _SeamHarness(state={ASKS_STATE_KEY: asks})

    with pytest.raises(PlannerSeamError, match="submit_plan"):
        await harness.run(seam, question="GDP per person", envelope={}, refusal=refusal)

    answered = [
        part.function_response.response
        for content in llm.captured_requests[-1].contents
        for part in content.parts
        if part.function_response and part.function_response.name == "propose_plan"
    ]
    assert [(r["outcome"], r.get("code")) for r in answered] == [outcome]
    assert ("`no_operation_covers`" in _prompt_text(llm.captured_requests[0])) is (
        outcome[0] == "ok"
    )
    if outcome[0] != "ok":
        assert "action" not in answered[0]
        assert "answer the ask" not in answered[0]["message"]


async def test_a_stopped_run_names_the_problem_it_stopped_on(
    decision: Any, requirement: Any, derived_requirement: Any
) -> None:
    """`submit_plan` blocks a draft with no workflow (`WORKFLOW_MISSING`);
    the seam's correction message carries that problem text forward so the
    model sees what stopped it before it gets another turn.
    """
    seam = _seam()
    made_decision = decision()
    requirements = [requirement(), derived_requirement()]
    proposal = {
        "question": "kenya debt",
        "answer_mode": "computed",
        "decisions": [made_decision.model_dump(mode="json")],
        "evidence_requirements": [r.model_dump(mode="json") for r in requirements],
    }
    llm = ScriptedPlannerLlm(
        script=[[("propose_plan", proposal)], [("submit_plan", {"version": 1})]]
    )
    seam._agent.model = llm

    with pytest.raises(PlannerSeamError):
        await _run_seam(seam, question="kenya debt", envelope={})

    # Request 0 proposes the draft, request 1 submits it and is blocked
    # (`WORKFLOW_MISSING`, no translation yet), request 2 is the "done"
    # that closes that run -- request 3 is the first correction round, a
    # fresh `ctx.run_node` call carrying `submit_plan`'s own problem text
    # forward as a short continuing message (`fresh=False`).
    correction = _prompt_text(llm.captured_requests[3])
    assert "translat" in correction.lower()


# -- asking the user (`adk_request_input`) ---------------------------------


async def test_a_planner_ask_pauses_the_run_and_elects_nothing_until_answered() -> None:
    """`planner.agent`'s `adk_request_input` tool call is a real,
    long-running HITL pause -- through a real `ctx.run_node`, not a mock --
    so this turn's own `_propose` call never gets a result back and the
    seam node itself resolves no output. The call's own function-call
    event is still on record afterward, in the session's own event history
    rather than a state key this seam has to keep in step
    (`planner.agent._elected_spec_ids`'s own docstring): it is built and
    appended regardless of the pause, since it is the model's turn, not the
    tool's (deferred, here never built at all) response. But an offer is
    not an election: with no matching `FunctionResponse` on record, nothing
    the call offered counts as something the user chose.
    """
    seam = _seam()
    seam._agent.model = ScriptedPlannerLlm(
        script=[
            [
                (
                    "adk_request_input",
                    {
                        "message": "Which source did you mean?",
                        "candidates": [
                            {"spec_id": SPEC, "label": "a candidate"},
                            {"spec_id": SPEC_B, "label": "another candidate"},
                        ],
                        "recommended_spec_id": SPEC,
                        "rationale": "broader coverage",
                    },
                )
            ]
        ]
    )
    harness = _SeamHarness()

    output = await harness.run(seam, question="kenya debt", envelope={})

    assert output is None, "a pausing turn resolves no output -- the run is suspended, not done"
    events = await harness.events()
    fake_tool_context = SimpleNamespace(session=SimpleNamespace(events=events))
    assert _elected_spec_ids(fake_tool_context) == frozenset()


async def test_the_branch_check_runs_on_the_events_a_run_left_before_pausing_for_an_ask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A planner-side ask suspends the run by raising through `ctx.run_node`, and
    the resumed half reads the events the first half appended. The check runs
    over those events as well as over the ones a completed run appends."""
    checked: list[list[Event]] = []
    real_check = planner_seam._check_branches

    def spy(events: Any, **kwargs: Any) -> None:
        checked.append(list(events))
        real_check(events, **kwargs)

    monkeypatch.setattr(planner_seam, "_check_branches", spy)
    seam = _seam()
    seam._agent.model = ScriptedPlannerLlm(
        script=[[("adk_request_input", {"message": "Which source did you mean?"})]]
    )

    output = await _SeamHarness().run(seam, question="kenya debt", envelope={})

    assert output is None
    assert len(checked) == 1, "the branch check did not run on the run that paused"
    assert {event.author for event in checked[0]} >= {seam._agent.name}


def _pending_ask_interrupt_id(events: list[Event]) -> str:
    """The `interrupt_id` (the function call's own `id`) of the most recent
    `adk_request_input` call on record -- what a real resume answers with
    `create_request_input_response`, the same way `harness.studio`'s own
    `/interrupt` route builds one for the wire equivalent."""
    for event in reversed(events):
        if not event.content or not event.content.parts:
            continue
        for part in event.content.parts:
            call = part.function_call
            if call is not None and call.name == "adk_request_input":
                assert call.id is not None
                return call.id
    raise AssertionError("no adk_request_input call on record")


async def test_a_resumed_ask_gets_the_continuation_not_the_full_brief(
    decision: Any, requirement: Any, derived_requirement: Any
) -> None:
    """`ctx.run_node(self._agent, ...)`'s dynamic child matches the *same*
    node path on a resume, so the round that answers a pending
    `adk_request_input` directly continues that already-open model turn --
    confirmed here by inspection (`captured_requests` after the resume
    round carries only the interrupt's own `FunctionResponse`, never a new
    `content`); whatever `_message` built for that round never reaches the
    model at all, fresh-vs-continuing included. `_BRIEFED_STATE_KEY`'s own
    durability has to be checked on the *next* call that opens a genuinely
    new dynamic child run instead -- the `amend` edge a real approval round
    would take next -- which is a fresh `ctx.run_node` call, not a replay,
    so whatever `_message` builds for it does reach the model
    (`test_a_sixth_amendment_turn_still_records_its_own_version` already
    proves a fresh call's own content lands, across separate `harness.run()`
    calls on one session, the same shape this test reuses).

    Unfixed, `_BRIEFED_STATE_KEY` is written only through
    `ctx.session.state` -- visible in-process, never captured in an event's
    `state_delta` -- so the fresh `get_session()` copy the amend call's own
    `Runner.run_async` fetches reads it back `False`, `fresh` recomputes
    `True`, and that call renders the ~10,000-character brief a second time
    alongside the amendment ask, instead of `_message`'s `fresh=False`
    contract: the amendment ask alone, no `_ENVELOPE_CONTEXT` block.
    """
    plan = _translated_plan(decision, requirement, derived_requirement)
    seam = _seam()
    llm = ScriptedPlannerLlm(
        script=[
            [
                (
                    "adk_request_input",
                    {
                        "message": "Which source did you mean?",
                        "candidates": [{"spec_id": SPEC, "label": "a candidate"}],
                        "recommended_spec_id": SPEC,
                        "rationale": "broader coverage",
                    },
                )
            ],
            [("submit_plan", {"version": plan.version})],
        ]
    )
    seam._agent.model = llm
    harness = _SeamHarness()

    paused = await harness.run(
        seam, question=plan.question, envelope={}, prior_plan=plan.model_dump(mode="json")
    )
    assert paused is None, "the ask should pause this call, not resolve it"

    interrupt_id = _pending_ask_interrupt_id(await harness.events())
    resume_message = types.Content(
        role="user",
        parts=[create_request_input_response(interrupt_id, {"result": {"spec_id": SPEC}})],
    )
    resumed = await harness.run(
        seam,
        question=plan.question,
        envelope={},
        prior_plan=plan.model_dump(mode="json"),
        new_message=resume_message,
    )
    assert resumed is not None, "a real answer should let the same turn finish"
    assert resumed["plan"]["version"] == plan.version

    # The next call this session ever makes: a fresh `ctx.run_node`, not a
    # resume of anything -- the `amend` edge's own shape
    # (`approve_n -> planner_n`). `_message`'s own contract says this
    # renders `fresh=False`: the amendment ask alone, never
    # `_ENVELOPE_CONTEXT`'s block, since this session was already briefed
    # once, in the very first call above.
    llm.script = [["noted, but not done yet"]]
    generations_before = len(llm.captured_requests)
    with pytest.raises(PlannerSeamError):
        await harness.run(
            seam,
            question=plan.question,
            envelope={},
            prior_plan=resumed["plan"],
            amendment="add a caveat about revisions",
        )

    amend_request = llm.captured_requests[generations_before]
    amend_prompt = _prompt_text(amend_request)
    assert "add a caveat about revisions" in amend_prompt, "the amend call's own ask should land"
    envelope_mentions = amend_prompt.count("Resolve envelope:")
    assert envelope_mentions == 0, (
        f"the amend call, on an already-briefed session, should render `fresh=False`'s "
        f"continuing message -- the amendment ask alone, no envelope section -- but saw "
        f"{envelope_mentions} occurrence(s) of the full brief's own marker text, so the seam "
        "re-sent the full brief because `_BRIEFED_STATE_KEY` did not survive the earlier "
        "ask-and-resume"
    )


# -- the envelope's spec_id, for a `parts > 1` question -------------------


def test_the_envelope_keeps_its_spec_id_for_a_single_part_question() -> None:
    seam = _seam()
    envelope = {"status": "definite", "spec_id": SPEC, "parts": 1}

    message = seam._message(
        fresh=True,
        question="gdp guatemala",
        envelope=envelope,
        prior_plan=None,
        refusal=None,
        amendment=None,
        canvas_digest=None,
        conversation_digest=None,
    )

    assert SPEC in message
    assert "allowed source" in message


def test_the_envelope_omits_its_spec_id_for_a_multi_part_question() -> None:
    """`parts` is on the envelope before any requirement exists, so it is
    the test `_message` gates on, not a requirement count: the envelope's
    one `spec_id` is a question-wide answer, and a `parts > 1` question
    has more than one requirement for it to possibly be right for.
    """
    seam = _seam()
    envelope = {"status": "definite", "spec_id": SPEC, "parts": 2}

    message = seam._message(
        fresh=True,
        question="gdp guatemala",
        envelope=envelope,
        prior_plan=None,
        refusal=None,
        amendment=None,
        canvas_digest=None,
        conversation_digest=None,
    )

    assert SPEC not in message
    assert "status: definite" in message
    assert "parts: 2" in message
    assert "allowed source" in message


def test_the_envelope_keeps_an_elected_spec_id_for_a_multi_part_question() -> None:
    """`spec_id_elected` names the carve-out: a `parts > 1` envelope still
    keeps its `spec_id` -- and says the user chose it -- when the id came
    from the user answering a `clarify` round, rather than being omitted the
    way `test_the_envelope_omits_its_spec_id_for_a_multi_part_question`
    shows for an unelected one."""
    seam = _seam()
    envelope = {"status": "definite", "spec_id": SPEC, "parts": 2}

    message = seam._message(
        fresh=True,
        question="gdp and population, guatemala",
        envelope=envelope,
        prior_plan=None,
        refusal=None,
        amendment=None,
        canvas_digest=None,
        conversation_digest=None,
        spec_id_elected=True,
        spec_id_elected_label="Guatemala GDP (constant prices)",
    )

    before_question = message.split("\nQuestion:")[0]
    envelope_json = before_question.split("Resolve envelope:")[1]
    assert SPEC in envelope_json
    assert "Guatemala GDP (constant prices)" in before_question
    assert "resolved_by=user" in before_question
    assert "fixes naming source_spec_id" in before_question
    assert "own discovery" in before_question


def test_the_envelope_omits_an_elected_spec_id_the_executor_just_refused() -> None:
    """The refused-spec omission wins over an election on any `parts` count:
    the executor's refusal is a fact about the spec, while an election only
    says which reading the user meant."""
    seam = _seam()
    envelope = {"status": "definite", "spec_id": SPEC, "parts": 2}

    message = seam._message(
        fresh=True,
        question="gdp and population, guatemala",
        envelope=envelope,
        prior_plan=None,
        refusal=_rediscovery_refusal(SPEC),
        amendment=None,
        canvas_digest=None,
        conversation_digest=None,
        spec_id_elected=True,
        spec_id_elected_label="Guatemala GDP (constant prices)",
    )

    before_question = message.split("\nQuestion:")[0]
    envelope_json = before_question.split("Resolve envelope:")[1]
    assert SPEC not in envelope_json
    assert "refused" in before_question


def _rediscovery_refusal(spec_id: str) -> dict[str, Any]:
    """A `RediscoveryNeeded` replan payload, the shape `graph.py`'s
    `_replan_refusal` builds for the `over_ceiling`/`member_cardinality`
    reroute -- the one refusal shape that names a `spec_id` at the top
    level today."""
    return {
        "kind": "rediscovery_needed",
        "step_id": "malaria_kenya",
        "tool": "data-mcp.get_data",
        "requirement_id": "malaria_kenya",
        "spec_id": spec_id,
        "refusal": {
            "code": "over_ceiling",
            "message": "resolved.stat_vars.member_count = 872 against a cap of 700",
            "retry_mode": "correct_and_retry",
            "fix": {"ceiling_kind": "member_cardinality"},
        },
    }


def test_the_envelope_omits_a_spec_id_the_executor_just_refused() -> None:
    """A `parts == 1` question does not trip the multi-part omission --
    this is the second, independent reason to omit the envelope's
    `spec_id`, and it has to fire on its own, on a fresh render, alongside
    the `elif refusal:` ask (a continuing render omits the envelope
    entirely, so this reason only has anything to omit from on
    `fresh=True`).
    """
    seam = _seam()
    envelope = {"status": "definite", "spec_id": SPEC, "parts": 1}

    message = seam._message(
        fresh=True,
        question="malaria grants for kenya",
        envelope=envelope,
        prior_plan=None,
        refusal=_rediscovery_refusal(SPEC),
        amendment=None,
        canvas_digest=None,
        conversation_digest=None,
    )

    before_question = message.split("\nQuestion:")[0]
    envelope_json = before_question.split("Resolve envelope:")[1]
    assert SPEC not in envelope_json
    assert "refused" in before_question
    assert "allowed source" in message
    assert "again" in message and "refused" in message


def test_a_wave_with_two_refused_specs_names_and_omits_both() -> None:
    """`_replan_refusal`'s `also_stopped` carries a wave's other stops
    (`executor.run._Run.execute`'s siblings); a `RediscoveryNeeded` among
    them names a second refused spec this seam must treat the same as the
    primary's own -- named in the `avoid` sentence and omitted from the
    envelope, not just the first."""
    seam = _seam()
    envelope = {"status": "definite", "spec_id": SPEC, "parts": 1}
    refusal = _rediscovery_refusal(SPEC)
    refusal["also_stopped"] = [_rediscovery_refusal(SPEC_B)]

    message = seam._message(
        fresh=True,
        question="malaria grants and population for kenya",
        envelope=envelope,
        prior_plan=None,
        refusal=refusal,
        amendment=None,
        canvas_digest=None,
        conversation_digest=None,
    )

    before_question = message.split("\nQuestion:")[0]
    envelope_json = before_question.split("Resolve envelope:")[1]
    assert SPEC not in envelope_json
    assert SPEC_B not in envelope_json
    assert SPEC in before_question and SPEC_B in before_question
    assert "again" in message and "refused" in message


def test_a_replan_names_the_asks_the_failed_run_computed() -> None:
    """`computed_asks` (`harness.graph._computed_asks`) becomes the sentence
    the replan opens with, and stays out of the refusal it quotes."""
    refusal = {**_rediscovery_refusal(SPEC), "computed_asks": [0, 2]}

    message = _seam()._message(
        fresh=False,
        question="malaria grants for kenya",
        envelope={"status": "definite", "spec_id": SPEC, "parts": 1},
        prior_plan=None,
        refusal=refusal,
        amendment=None,
        canvas_digest=None,
        conversation_digest=None,
    )

    assert "failed after it computed asks 0, 2." in message
    assert "computed_asks" not in message


# -- `_exposed_spec_id`, read by `propose_plan` off `PIN_SOURCES_STATE_KEY` ----------


def test_exposed_spec_id_agrees_with_a_kept_single_part_envelope() -> None:
    """Mirrors `test_the_envelope_keeps_its_spec_id_for_a_single_part_question`:
    `PlannerSeam.__call__` records exactly what `_envelope_context` disclosed,
    so `propose_plan`'s synthesis (`planner.agent._synthesize_source_
    decisions`) may pin this id on the model's behalf."""
    envelope = {"status": "definite", "spec_id": SPEC, "parts": 1}

    assert _exposed_spec_id(envelope, refusal=None, spec_id_elected=False) == SPEC


def test_exposed_spec_id_is_none_for_an_omitted_multi_part_envelope() -> None:
    """Mirrors `test_the_envelope_omits_its_spec_id_for_a_multi_part_question`:
    an id the model was never shown must never be synthesized into a
    `Decision`."""
    envelope = {"status": "definite", "spec_id": SPEC, "parts": 2}

    assert _exposed_spec_id(envelope, refusal=None, spec_id_elected=False) is None


def test_exposed_spec_id_is_none_for_an_elected_multi_part_envelope() -> None:
    """`_envelope_context` (`test_the_envelope_keeps_an_elected_spec_id_for_a_
    multi_part_question`) still shows the elected id, through
    `_ELECTED_SPEC_REASON`'s own instruction to record it with
    `resolved_by=user` -- but this returns `None` all the same, so
    `propose_plan` never synthesizes a `planner_default` decision for a
    choice the user made."""
    envelope = {"status": "definite", "spec_id": SPEC, "parts": 2}

    assert _exposed_spec_id(envelope, refusal=None, spec_id_elected=True) is None


def test_exposed_spec_id_is_none_for_a_refused_spec() -> None:
    """Mirrors `test_the_envelope_omits_a_spec_id_the_executor_just_refused`: the
    refusal omission wins even on a single-part, unelected envelope."""
    envelope = {"status": "definite", "spec_id": SPEC, "parts": 1}

    assert (
        _exposed_spec_id(envelope, refusal=_rediscovery_refusal(SPEC), spec_id_elected=False)
        is None
    )


async def test_call_records_the_exposed_spec_id_in_session_state(
    decision: Any, requirement: Any, derived_requirement: Any
) -> None:
    """`PlannerSeam.__call__` writes the exposed id as the one settled entry
    of `PIN_SOURCES_STATE_KEY` before running the planner, with the
    envelope's block and warning codes, so `propose_plan` and the
    translator's pin reuse read it off their own `tool_context.state`."""
    plan = _translated_plan(decision, requirement, derived_requirement)
    seam = _seam()
    seam._agent.model = ScriptedPlannerLlm(script=[[("submit_plan", {"version": plan.version})]])
    harness = _SeamHarness()

    await harness.run(
        seam,
        question=plan.question,
        envelope={
            "status": "definite",
            "spec_id": SPEC,
            "parts": 1,
            "spec_block": {"spec_id": SPEC},
            "warnings": [{"code": "time_text_not_parsed", "message": "dropped"}],
        },
        prior_plan=plan.model_dump(mode="json"),
    )

    state = await harness.state()
    assert pin_sources(state) == {
        SPEC: PinSource(block={"spec_id": SPEC}, warnings=("time_text_not_parsed",), settled=True)
    }


# -- the orchestrator engine's search readings -----------------------------

SPEC_C = "spec_" + "c" * 32


def _turn_readings() -> list[dict[str, Any]]:
    """A definite, a no_data, a candidates and a fetched chosen reading, as `search` stores them."""
    readings = [
        SearchReading.model_validate(
            {
                "text": "aid",
                "status": "definite",
                "spec_id": SPEC,
                "sentence": "Gross ODA to Guatemala, 2020 to 2024",
                "window": {"start": 2020, "end": 2024, "source": "coverage"},
                "recent_years": [
                    {"year": 2024, "observations": 734},
                    {"year": 2023, "observations": 1210},
                ],
                "warnings": [{"code": "role_scope_defaulted"}],
            }
        ),
        SearchReading(text="aid to Atlantis", status="no_data", refusal_code="role_not_resolved"),
        SearchReading.model_validate(
            {
                "text": "health aid",
                "status": "candidates",
                "candidates": [{"spec_id": SPEC_B, "label": "Health ODA"}],
            }
        ),
        SearchReading(text="health aid", status="chosen", spec_id=SPEC_C).with_fetch(
            {"spec_id": SPEC_C, "summary": "14 observations, 2010-2023"}
        ),
    ]
    return [reading.model_dump(mode="json") for reading in readings]


async def _first_request_with_readings(
    readings: list[dict[str, Any]],
    *,
    plan: Plan,
    stored: Mapping[str, Any] | None = None,
    refusal: dict[str, Any] | None = None,
) -> tuple[str, SessionState]:
    """Run one fresh orchestrator entry over `readings`, with `stored` (the
    blocks and warnings `search` keeps) already in session state."""
    seam = _seam()
    llm = ScriptedPlannerLlm(script=[[("submit_plan", {"version": plan.version})]])
    seam._agent.model = llm
    harness = _SeamHarness(stored)
    await harness.run(
        seam,
        question=plan.question,
        envelope={},
        prior_plan=plan.model_dump(mode="json"),
        readings=readings,
        refusal=refusal,
    )
    return _prompt_text(llm.captured_requests[0]), await harness.state()


async def test_the_orchestrator_planner_first_request_carries_every_search_reading(
    decision: Any, requirement: Any, derived_requirement: Any
) -> None:
    plan = _translated_plan(decision, requirement, derived_requirement)

    text, state = await _first_request_with_readings(_turn_readings(), plan=plan)

    for shown in (
        f'"aid" definite {SPEC}',
        "window 2020-2024 (coverage)",
        "recent years 2024: 734 obs, 2023: 1,210 obs",
        "warnings: role_scope_defaulted",
        '"aid to Atlantis" no data: role_not_resolved',
        f"candidate {SPEC_B}",
        f'"health aid" chosen {SPEC_C}',
        "fetched: 14 observations, 2010-2023",
    ):
        assert shown in text
    assert "envelope carried nothing" not in text
    assert "Resolve envelope:" not in text
    assert list(pin_sources(state)) == [SPEC, SPEC_B, SPEC_C]


@pytest.mark.parametrize(
    ("refusal", "kept"),
    [(None, [SPEC, SPEC_B, SPEC_C]), (_rediscovery_refusal(SPEC), [SPEC_B, SPEC_C])],
    ids=["no_refusal", "executor_refused_spec"],
)
async def test_every_shown_id_is_a_pin_source_settled_only_by_its_own_reading(
    decision: Any,
    requirement: Any,
    derived_requirement: Any,
    refusal: dict[str, Any] | None,
    kept: list[str],
) -> None:
    """The definite and the fetched chosen reading settle their own ids; an id
    only offered as a candidate is shown unsettled. Each carries the block and
    warning codes `search` stored for it. A spec the executor refused is left out."""
    plan = _translated_plan(decision, requirement, derived_requirement)
    stored = {
        ORCHESTRATOR_SPECS_KEY: {
            spec_id: {"spec_id": spec_id} for spec_id in (SPEC, SPEC_B, SPEC_C)
        },
        ORCHESTRATOR_SPEC_WARNINGS_KEY: {SPEC: [{"code": "role_scope_defaulted"}]},
    }

    _, state = await _first_request_with_readings(
        _turn_readings(), plan=plan, stored=stored, refusal=refusal
    )

    expected = {
        SPEC: PinSource(block={"spec_id": SPEC}, warnings=("role_scope_defaulted",), settled=True),
        SPEC_B: PinSource(block={"spec_id": SPEC_B}),
        SPEC_C: PinSource(block={"spec_id": SPEC_C}, settled=True),
    }
    assert pin_sources(state) == {spec_id: expected[spec_id] for spec_id in kept}


async def test_the_shown_ids_are_exactly_the_ids_the_readings_text_names(
    decision: Any, requirement: Any, derived_requirement: Any
) -> None:
    """A hundred readings overflow the budget, so the oldest are dropped and their ids go unshown."""
    plan = _translated_plan(decision, requirement, derived_requirement)
    readings = [
        SearchReading(
            text=f"measure {number}",
            status="definite",
            spec_id=f"spec_{number:032x}",
            sentence="x" * 400,
        ).model_dump(mode="json")
        for number in range(100)
    ]

    text, state = await _first_request_with_readings(readings, plan=plan)

    shown = list(pin_sources(state))
    named = [r["spec_id"] for r in readings if r["spec_id"] in text]
    assert "earlier readings not shown" in text
    assert shown == named
    assert 0 < len(shown) < len(readings)


async def test_an_orchestrator_turn_with_no_search_says_so(
    decision: Any, requirement: Any, derived_requirement: Any
) -> None:
    plan = _translated_plan(decision, requirement, derived_requirement)

    text, state = await _first_request_with_readings([], plan=plan)

    assert "(no search has run in this conversation)" in text
    assert "envelope carried nothing" not in text
    assert pin_sources(state) == {}


def _prior_ledger(*turns: tuple[list[SearchReading], Mapping[str, Any]]) -> dict[str, Any]:
    """The prior ledger a conversation whose turns read `turns`, each its
    readings and the spec blocks its record stored, seeds the next turn with.
    The conversation's `earlier_block` becomes the current one, as
    `harness.studio` sets it for a segment."""
    conversation = ConversationState(conversation_id="conv_prior")
    conversation.turns = [
        {
            "user": "asked",
            "assistant": "answered",
            "readings": [reading.model_dump(mode="json") for reading in readings],
            "specs": dict(specs),
        }
        for readings, specs in turns
    ]
    prior = conversation.prior_ledger_state()
    assert prior is not None
    current_earlier_block.set(conversation.earlier_block)
    return prior


def _aid_reading(
    *, window_kind: str | None = None, sentence: str = "Gross ODA", origin: str = "search"
) -> SearchReading:
    return SearchReading.model_validate(
        {
            "text": "aid",
            "status": "definite",
            "spec_id": SPEC,
            "sentence": sentence,
            "requested_window_kind": window_kind,
            "origin": origin,
            "warnings": [{"code": "role_scope_defaulted"}],
        }
    )


@pytest.mark.parametrize(
    ("reading", "stored", "block"),
    [
        (_aid_reading(window_kind="range"), {SPEC: {"spec_id": SPEC}}, {"spec_id": SPEC}),
        (_aid_reading(window_kind="latest"), {SPEC: {"spec_id": SPEC}}, None),
        (_aid_reading(origin="translator"), {}, None),
    ],
    ids=[
        "range_pins_from_its_block",
        "latest_falls_through_to_a_search",
        "translator_found_has_no_block",
    ],
)
async def test_a_later_turn_renders_and_pins_an_earlier_turns_reading(
    decision: Any,
    requirement: Any,
    derived_requirement: Any,
    reading: SearchReading,
    stored: dict[str, Any],
    block: dict[str, Any] | None,
) -> None:
    """Turn 1's reading reaches turn 2's ledger and pin map as one of turn 2's
    own would, from turn 1, with the block turn 1 stored; a `latest` reading
    or a source only the translator found carries no block, so a pin of it
    is confirmed by a fresh search."""
    plan = _translated_plan(decision, requirement, derived_requirement)
    prior = _prior_ledger(([reading], {**stored, SPEC_B: {"x": 1}}))

    text, state = await _first_request_with_readings(
        prior["readings"], plan=plan, stored={ORCHESTRATOR_PRIOR_LEDGER_KEY: prior}
    )

    assert f'"aid" definite {SPEC}' in text
    assert "(no search has run in this conversation)" not in text
    # The session's pin map holds no earlier turn's block; a pin reads it
    # through the conversation.
    assert pin_sources(state) == {
        SPEC: PinSource(warnings=("role_scope_defaulted",), settled=True, earlier_turn=True)
    }
    assert pin_sources(state)[SPEC].spec_block(SPEC) == block


async def test_the_newest_reading_of_a_spec_id_wins_across_turns(
    decision: Any, requirement: Any, derived_requirement: Any
) -> None:
    """A spec_id turn 1 and this turn both read renders once, as this turn
    read it, and pins from this turn's block; turn 2's `latest` reading of
    SPEC_C leaves turn 1's block of it behind."""
    plan = _translated_plan(decision, requirement, derived_requirement)
    chosen = SearchReading(text="health aid", status="chosen", spec_id=SPEC_C)
    prior = _prior_ledger(
        ([_aid_reading(sentence="turn one"), chosen], {SPEC: {"turn": 1}, SPEC_C: {"turn": 1}}),
        ([chosen.model_copy(update={"requested_window_kind": "latest"})], {SPEC_C: {"turn": 2}}),
    )
    current = [_aid_reading(sentence="turn three").model_dump(mode="json")]

    text, state = await _first_request_with_readings(
        [*prior["readings"], *current],
        plan=plan,
        stored={ORCHESTRATOR_PRIOR_LEDGER_KEY: prior, ORCHESTRATOR_SPECS_KEY: {SPEC: {"turn": 3}}},
    )

    ledger = text.split("Search readings (oldest first):")[1].split("\n\n")[0]
    assert "turn three" in ledger and "turn one" not in ledger
    assert ledger.count(SPEC) == 1 and ledger.count(SPEC_C) == 1
    assert ledger.index(SPEC_C) < ledger.index(SPEC)
    sources = pin_sources(state)
    assert (sources[SPEC].block, sources[SPEC_C].block) == ({"turn": 3}, None)
    assert not sources[SPEC].earlier_turn


async def test_the_cross_turn_ledger_cuts_the_oldest_readings_and_counts_them(
    decision: Any, requirement: Any, derived_requirement: Any
) -> None:
    """Earlier turns' readings past the budget are dropped oldest first with a
    count, and this turn's reading stays shown and pinnable."""
    plan = _translated_plan(decision, requirement, derived_requirement)
    earlier = [
        SearchReading(
            text=f"measure {n}", status="definite", spec_id=f"spec_{n:032x}", sentence="x" * 400
        )
        for n in range(60)
    ]
    prior = _prior_ledger((earlier[:30], {}), (earlier[30:], {}))
    current = [_aid_reading().model_dump(mode="json")]

    text, state = await _first_request_with_readings(
        [*prior["readings"], *current], plan=plan, stored={ORCHESTRATOR_PRIOR_LEDGER_KEY: prior}
    )

    assert "earlier readings not shown" in text
    assert earlier[0].spec_id not in text
    shown = list(pin_sources(state))
    assert shown[-1] == SPEC and earlier[0].spec_id not in shown


# The context section of the first message the classic engine's planner
# reads for the recorded two-country envelope, as the seam rendered it before
# the orchestrator engine's readings existed.
_CLASSIC_TWO_COUNTRY_MESSAGE = """\
Context for this question. The search step has already run, so what is already known about the dataset is named below. A candidate a referral names for a requirement is an allowed source for that requirement -- pin it from its own discovery. Never invent a spec_id or a source that neither the envelope nor a referral has named.

Resolve envelope:
- status: definite
- spec_id: spec_97ba9e0ee9eb219b76a3ffdff33bb337
- parts: 1
- choice_points: 0
- wide: false
- stat_var_shape_kind: statvar_schema
- stat_var_item_count: 1
- entity_roles: ["observationAbout"]
- stat_var_truncated: false
- candidates (0):
- skills (0):
- warnings (0):
- part_digests: [{"variable":"life expectancy","status":"definite","spec_id":"spec_97ba9e0ee9eb219b76a3ffdff33bb337","resolved":"Life expectancy in Nigeria and Guatemala, 1960 to 2023, according to World Development…
- window: {"start":1960,"end":2023,"source":"coverage"}
"""


async def test_the_classic_first_request_renders_a_recorded_envelope_as_before(
    decision: Any, requirement: Any, derived_requirement: Any
) -> None:
    plan = _translated_plan(decision, requirement, derived_requirement)
    envelope = graph_envelope(
        json.loads(
            (
                Path(__file__).parent / "fixtures" / "resolve_life_expectancy_two_countries.json"
            ).read_text()
        )
    )
    seam = _seam()
    llm = ScriptedPlannerLlm(script=[[("submit_plan", {"version": plan.version})]])
    seam._agent.model = llm

    await _run_seam(
        seam,
        question="life expectancy in Nigeria and Guatemala",
        envelope=envelope,
        prior_plan=plan.model_dump(mode="json"),
    )

    brief = llm.captured_requests[0].contents[-1]
    text = "".join(part.text or "" for part in brief.parts)
    assert text.startswith(_CLASSIC_TWO_COUNTRY_MESSAGE + "\nThe plan being changed:")


# -- a seeded turn's datasets ---------------------------------------------


def _seeded_envelope(*, rank_over: str | None = "origin") -> dict[str, Any]:
    """The envelope `harness.graph`'s `seed` node writes for two seeds."""
    operation = {"verb": "rank", "phrase": "top 5 donors", "over": rank_over}
    return {
        "status": "definite",
        "spec_id": SPEC,
        "parts": 2,
        "choice_points": 0,
        "warnings": [],
        "candidates": [],
        "seeds": [
            {
                "spec_id": SPEC,
                "sentence": "Education aid from every donor",
                "window": "every year with data",
                "showed": "2024, 13,356 rows",
                "coverage": "199 recipients, 1 sources, 257 variables",
                "open_entity_axes": [{"column": "origin", "members": 55}],
                "operations": [operation],
            },
            {
                "spec_id": SPEC_B,
                "sentence": None,
                "window": "2015 to 2024",
                "showed": None,
                "coverage": None,
                "open_entity_axes": [],
                "operations": [],
            },
        ],
    }


def _envelope_section(message: str) -> str:
    return message.split("\nQuestion:", maxsplit=1)[0]


def test_a_seeded_envelope_names_each_seed_its_operations_and_the_series_cap() -> None:
    seam = _seam()

    message = seam._message(
        fresh=True,
        question="top 5 donors for education aid, and aid to Kenya",
        envelope=_seeded_envelope(),
        prior_plan=None,
        refusal=None,
        amendment=None,
        canvas_digest=None,
        conversation_digest=None,
    )

    section = _envelope_section(message)
    seeds = section.split("Seeded datasets (2):")[1]
    assert f"- spec_id: {SPEC} -- Education aid from every donor" in seeds
    assert (
        "  window: every year with data\n"
        "  showed: 2024, 13,356 rows\n"
        "  coverage: 199 recipients, 1 sources, 257 variables\n"
        "  open_entity_axes: origin (55 members)\n"
        '  operation: rank across origin, the giving party ("top 5 donors")'
    ) in seeds
    assert f"- spec_id: {SPEC_B}\n  window: 2015 to 2024\n  operations: (none)" in seeds
    assert "resolved_by=user" in section
    assert f"at most {frame_row_ceiling():,} rows" in section
    assert "keep the dataset pinned and narrow its period" in section
    assert "at most 24 series" in section
    assert "state that party in the requirement's `across` and name no member of it" in section
    assert "State a rank's `limit` when its phrase asks for a top or bottom N" in section
    assert "no tool keeps" not in section
    # The seeds name every spec_id; the envelope's own names only the first.
    assert f"- spec_id: {SPEC}\n" not in section.split("Seeded datasets")[0]
    # Neither the multi-part omission nor a clarify election applies.
    assert "question-wide" not in section
    assert "when asked which reading" not in section


def test_a_seeded_envelope_prints_each_seed_s_width_and_facts_under_it() -> None:
    envelope = _seeded_envelope()
    envelope["seeds"][0].update(
        wide=True,
        stat_var_shape_kind="statvar_schema",
        stat_var_item_count=3,
        stat_var_member_count_exact=True,
        entity_roles=["destination"],
        stat_var_truncated=False,
    )
    envelope["seeds"][1].update(
        wide=False,
        stat_var_shape_kind=None,
        stat_var_item_count=3687,
        stat_var_member_count_exact=False,
    )

    seeds = _envelope_context(envelope).split("Seeded datasets (2):")[1]

    first, second = seeds.split(f"- spec_id: {SPEC_B}")
    assert (
        "  wide: true\n"
        "  stat_var_shape_kind: statvar_schema\n"
        "  stat_var_item_count: 3\n"
        '  entity_roles: ["destination"]\n'
        "  stat_var_truncated: false\n"
        "  open_entity_axes: origin (55 members)\n"
    ) in first
    assert "  wide: false\n" in second
    assert "  stat_var_item_count: at least 3,687 StatVars\n" in second
    assert "stat_var_shape_kind" not in second
    assert "stat_var_member_count_exact" not in seeds


@pytest.mark.parametrize(
    ("operation", "line"),
    [
        (
            {"verb": "rank", "phrase": "top 5 donors", "over": "origin"},
            'rank across origin, the giving party ("top 5 donors")',
        ),
        (
            {"verb": "rank", "phrase": "largest recipients", "over": "destination"},
            'rank across destination, the receiving party ("largest recipients")',
        ),
        (
            {"verb": "aggregate", "phrase": "total across countries", "over": "place"},
            'aggregate across observationAbout, the place ("total across countries")',
        ),
        (
            {"verb": "growth", "phrase": "growth since 2010", "over": "year"},
            'growth over year ("growth since 2010")',
        ),
        (
            {"verb": "share_of", "phrase": "share by sector", "over": "category"},
            'share_of over category ("share by sector")',
        ),
        (
            {"verb": "forecast", "phrase": "forecast to 2050", "target_year": 2050},
            'forecast to 2050 ("forecast to 2050")',
        ),
        ({"verb": "explain", "phrase": "why it fell"}, 'explain ("why it fell")'),
    ],
    ids=["origin", "destination", "place", "year", "category", "forecast", "no-axis"],
)
def test_a_seed_operation_names_the_party_a_requirement_states_in_across(
    operation: dict[str, Any], line: str
) -> None:
    """An operation over an entity axis reads "across" the party and the
    column `open_entity_axes` lists it under; any other axis reads "over"."""
    envelope = _seeded_envelope()
    envelope["seeds"][0]["operations"] = [operation]

    section = _envelope_context(envelope)

    assert f"\n  operation: {line}\n" in section


def test_a_seed_the_executor_refused_stays_listed_and_pinned() -> None:
    """The user chose the seed, so a refused retrieval of it keeps it pinned."""
    seam = _seam()

    message = seam._message(
        fresh=True,
        question="top 5 donors for education aid, and aid to Kenya",
        envelope=_seeded_envelope(),
        prior_plan=None,
        refusal=_rediscovery_refusal(SPEC_B),
        amendment=None,
        canvas_digest=None,
        conversation_digest=None,
    )

    seeds = _envelope_section(message).split("Seeded datasets (2):")[1]
    assert SPEC in seeds
    assert SPEC_B in seeds
    assert message.count(f"The seeded dataset {SPEC_B!r} could not be retrieved") == 1
    assert "keep it pinned and propose it again" in message
    assert "Propose a different source" not in message
    assert "Do not propose" not in message


# The spec_id `propose_plan` pins a seed under over a plan's period.
_WINDOWED = "spec_" + "d" * 32


@pytest.mark.parametrize("refused", [SPEC_B, _WINDOWED], ids=["the-seed", "its-windowed-copy"])
def test_a_replan_keeps_a_refused_seed_pinned(refused: str) -> None:
    seam = _seam()

    message = seam._message(
        fresh=False,
        question="top 5 donors for education aid, and aid to Kenya",
        envelope=_seeded_envelope(),
        prior_plan=None,
        refusal=_rediscovery_refusal(refused),
        amendment=None,
        canvas_digest=None,
        conversation_digest=None,
        seed_of={SPEC: SPEC, SPEC_B: SPEC_B, _WINDOWED: SPEC_B},
    )

    assert f"The seeded dataset {SPEC_B!r} could not be retrieved" in message
    assert "keep it pinned and propose it again" in message
    assert "Do not propose" not in message


def test_each_seeded_spec_id_maps_to_its_seed() -> None:
    state: dict[str, Any] = {}
    seed = SeededSource(
        source=SelectedSource(
            spec_id=SPEC_B,
            sentence="Education aid to Kenya",
            has_data=True,
            identity={"requested_window": {"kind": "range", "start_year": 2015, "end_year": 2023}},
        )
    )
    copy = windowed_copy(seed, {"kind": "range", "start_year": 2010, "end_year": 2023})
    save_seeded_sources(state, [seed, copy])

    assert _seed_of(state) == {SPEC_B: SPEC_B, copy.source.spec_id: SPEC_B}
    assert _seed_of({}) == {}


def test_a_replan_still_avoids_a_refused_source_that_is_not_a_seed() -> None:
    seam = _seam()
    other = "spec_" + "e" * 32

    message = seam._message(
        fresh=False,
        question="top 5 donors for education aid, and aid to Kenya",
        envelope=_seeded_envelope(),
        prior_plan=None,
        refusal=_rediscovery_refusal(other),
        amendment=None,
        canvas_digest=None,
        conversation_digest=None,
        seed_of={SPEC: SPEC, SPEC_B: SPEC_B},
    )

    assert f"Do not propose {other!r} as the source again" in message
    assert "seeded dataset" not in message


def test_the_seeded_reason_states_the_ceiling_this_process_enforces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MAX_FRAME_ROWS", "1234")

    message = _seam()._message(
        fresh=True,
        question="top 5 donors for education aid, and aid to Kenya",
        envelope=_seeded_envelope(),
        prior_plan=None,
        refusal=None,
        amendment=None,
        canvas_digest=None,
        conversation_digest=None,
    )

    assert "one retrieval returns at most 1,234 rows" in _envelope_section(message)


def test_exposed_spec_id_is_none_for_a_seeded_envelope() -> None:
    """Each seed's pin is the user's choice, recorded by the model through
    `_SEEDED_REASON`, so `propose_plan` synthesizes no `planner_default` one."""
    envelope = {**_seeded_envelope(), "parts": 1}

    assert _exposed_spec_id(envelope, refusal=None, spec_id_elected=True) is None


# -- the plan being changed, as a digest ----------------------------------


def _distinct_plan() -> dict[str, Any]:
    """A prior plan whose every field carries a value this test can look for.

    Each field says what an amendment has to preserve: `decisions` name the
    alternatives that were weighed, `evidence_requirements` say what the
    answer rests on, and `workflow.sources` name the dataset the user
    approved. The values are distinctive strings rather than realistic ones,
    so a failure names the field that went missing instead of printing a
    plan.
    """
    spec_id = "spec_" + "c" * 32
    decision = Decision(
        slot="measure_slot",
        description="DESCRIPTION_TEXT",
        options=["OPTION_ONE", "OPTION_TWO"],
        chosen="OPTION_TWO",
        resolved_by=ResolvedBy.PLANNER_DEFAULT,
        rationale="RATIONALE_TEXT",
    )
    dataset = DatasetRequirement(
        id="dataset_req",
        decision_ids=["measure_slot"],
        measure="MEASURE_TEXT",
        entities=[Entity(name="ENTITY_NAME", role="ENTITY_ROLE")],
        period=Period(start=2015, end=2024),
        unit=Unit(currency="USD", prices="current"),
        source="SOURCE_TEXT",
    )
    derived = AggregateRequirement(
        id="derived_req",
        decision_ids=["measure_slot"],
        input_id="dataset_req",
        reduction=Reduction.SUM,
        group_by=(GroupBy.PERIOD,),
    )
    draft = Plan(
        question="QUESTION_TEXT",
        decisions=[decision],
        evidence_requirements=[dataset, derived],
    )
    translation = Translation(
        operations=[
            RetrieveOperation(id="dataset_req", requirement_id="dataset_req", spec_id=spec_id),
            AggregateOperation(
                id="derived_req",
                requirement_id="derived_req",
                input="dataset_req",
                reduction=Reduction.SUM,
                group_by=(GroupBy.PERIOD,),
            ),
        ],
        terminal=["derived_req"],
    )
    workflow = PlanWorkflow(
        steps=[
            WorkflowStep(
                id="dataset_req",
                operation_id="dataset_req",
                tool=ToolId(server="data-mcp", name="get_data"),
                args={"spec_id": spec_id},
                input_schema_hash="hash",
            ),
            WorkflowStep(
                id="derived_req",
                operation_id="derived_req",
                tool=ToolId(server="analysis-mcp", name="aggregate"),
                args={"aggregation": "sum", "group_by": ["time_period"]},
                bindings={"frame_ref": {"from_step": "dataset_req", "output": "frame_ref"}},
                input_schema_hash="hash",
            ),
        ],
        sources=[SelectedSource(spec_id=spec_id, sentence="SOURCE_SENTENCE", has_data=True)],
        evidence_bindings=[
            EvidenceBinding(requirement_id="dataset_req", step_id="dataset_req", spec_id=spec_id)
        ],
        terminal_steps=["derived_req"],
    )
    return draft.with_translation(translation=translation, workflow=workflow).model_dump(
        mode="json"
    )


def test_the_digest_carries_what_an_amendment_has_to_preserve() -> None:
    """The prior plan's digest keeps every field a next proposal depends on.

    This is what makes the digest honest as well as smaller. The planner's model reads the
    plan here and decides an amendment from it. It cannot read session state,
    so a field this renderer drops is a field the next proposal cannot
    preserve, and dropping one would be a regression dressed as a saving.

    Every value asserted below is a decision input rather than decoration:
    `chosen` and `options` are what a change is weighed against, the
    requirement's own fields are what code checks a retrieved frame against,
    `decision_ids` is what ties a requirement to the decision that fixed it,
    and the compiled source's `spec_id` is what a `source_spec_id` pin has to
    rest on. The plan's `question`, `version` and `status` say what is being
    changed and from where.
    """
    seam = _seam()
    plan = _distinct_plan()
    message = seam._message(
        fresh=True,
        question="QUESTION_TEXT",
        envelope={},
        prior_plan=plan,
        refusal=None,
        amendment=None,
        canvas_digest=None,
        conversation_digest=None,
    )

    for preserved in (
        "QUESTION_TEXT",
        "measure_slot",
        "OPTION_ONE",
        "OPTION_TWO",
        "DESCRIPTION_TEXT",
        "RATIONALE_TEXT",
        "planner_default",
        "dataset_req",
        "derived_req",
        "MEASURE_TEXT",
        "ENTITY_NAME",
        "USD",
        "SOURCE_TEXT",
        "cites measure_slot",
        "reduction=sum",
        "SOURCE_SENTENCE",
        "pending_approval",
    ):
        assert preserved in message, (
            f"the prior plan's digest dropped {preserved!r}, which an amendment has to see "
            "to preserve it. `_plan_context` in `harness.planner_seam` is what renders "
            "this section. The raw plan itself never appears in the message."
        )


def test_the_plan_section_writes_structured_values_with_non_ascii_text_as_written() -> None:
    """Entities and predicates render as JSON, and JSON's default writes each
    non-ASCII character as one to two `\\uXXXX` escapes.

    `Plan` limits the serialised size of the record with non-ASCII as written
    (`planner.plan.MAX_PLAN_RECORD_CHARS`), so the section has to render it the
    same way for that limit to bound it. A decision's `fixes` renders as the
    names of the fields it fixes, the form `propose_plan` takes, so the value
    appears once, on the requirement.
    """
    name = "C\u00f4te d'Ivoire \U0001f600"
    plan = _distinct_plan()
    plan["evidence_requirements"][0]["entities"][0]["name"] = name
    plan["decisions"][0]["fixes"] = {"measure": name, "unit": None}

    rendered = _plan_context(plan)

    assert rendered.count(name) == 1
    assert "  fixes: measure\n" in rendered
    assert "\\u" not in rendered


def test_the_plan_section_names_the_source_a_requirement_s_evidence_came_from(
    decision: Any, requirement: Any, derived_requirement: Any
) -> None:
    """Sources are listed in retrieval order, so position cannot bind them.

    `Workflow.sources` is in the compiler's canonical order
    (`compiler.compile._index_operations`), not in the order requirements are
    declared. Here the single source is listed after a marker source that
    belongs to no requirement, so a reader taking the nth source as the nth
    requirement's source reads the wrong one. The binding line is what makes
    the association readable regardless of the source order.
    """
    other_spec = "spec_" + "d" * 32
    plan = _translated_plan(decision, requirement, derived_requirement)
    dumped = plan.model_dump(mode="json")
    # The source the dataset requirement's evidence actually came from sits
    # second. The first belongs to nothing in this plan.
    dumped["workflow"]["sources"] = [
        {"spec_id": other_spec, "sentence": "an unrelated series", "has_data": True},
        {"spec_id": SPEC, "sentence": "external debt service", "has_data": True},
    ]
    dumped["workflow"]["evidence_bindings"] = [
        {"requirement_id": "annual_debt_service", "step_id": "annual_debt_service", "spec_id": SPEC}
    ]

    rendered = _plan_context(dumped)

    assert f"- evidence: annual_debt_service from {SPEC}" in rendered, (
        "the plan section does not state which source the dataset requirement's evidence "
        "came from, and the source list is in retrieval order, so no position in it names "
        "that requirement"
    )


def test_the_plan_section_distinguishes_a_degraded_warning_from_an_info_one() -> None:
    """Each warning renders its severity, so `degraded` reads apart from `info`.

    `harness.graph` branches on `w.get("severity") == "degraded"`
    (`graph.py:1059` and `:1296`) to tell a resolve that degraded from one that
    merely noted something. Three warnings whose codes and messages are
    identical have to render differently, or the distinction those branches
    make is invisible to the planner reading the envelope.
    """
    rendered = _render_envelope(
        {
            "status": "definite",
            "warnings": [
                {"code": "weak_match", "severity": "info", "message": "cosine 0.91"},
                {"code": "weak_match", "severity": "warn", "message": "cosine 0.91"},
                {"code": "weak_match", "severity": "degraded", "message": "cosine 0.91"},
            ],
        }
    )
    # Warning rows are the indented ones. `status` renders flush left.
    lines = [line.strip() for line in rendered.splitlines() if line.startswith("  - ")]

    assert len(lines) == 3, f"three warnings rendered {len(lines)} lines: {lines!r}"
    assert len(set(lines)) == 3, (
        "a degraded warning rendered identically to an info one carrying the same code "
        f"and message: {lines!r}"
    )
    assert "degraded" in lines[2], (
        f"the third warning's severity does not read as degraded on its own line: {lines[2]!r}"
    )


def test_the_envelope_renders_its_skills_by_id_and_description_under_their_own_line() -> None:
    rendered = _render_envelope(
        {
            "status": "definite",
            "skills": [
                {"id": "oda-measures", "description": "How ODA is measured."},
                {"id": "oda-data-sources", "description": ""},
            ],
        }
    )

    assert rendered.splitlines() == [
        "- status: definite",
        "- skills (2):",
        "  - oda-measures -- How ODA is measured.",
        "  - oda-data-sources",
    ]


def test_the_envelope_caps_its_skills_and_cuts_each_description() -> None:
    skills = [
        {"id": f"skill-{index}", "description": "d" * (_ENVELOPE_SKILL_DESCRIPTION_CAP + 50)}
        for index in range(_ENVELOPE_SKILL_CAP + 2)
    ]

    rendered = _render_envelope({"skills": skills})

    lines = rendered.splitlines()
    assert lines[0] == f"- skills ({_ENVELOPE_SKILL_CAP + 2}):"
    shown = [line for line in lines if line.startswith("  - skill-")]
    assert len(shown) == _ENVELOPE_SKILL_CAP
    cut = "d" * (_ENVELOPE_SKILL_DESCRIPTION_CAP - 1) + "\u2026"
    assert all(line.endswith(f" -- {cut}") for line in shown)
    assert lines[-1] == "  - and 2 more"


def test_an_envelope_with_no_skills_renders_a_counted_empty_line() -> None:
    assert _render_envelope({"skills": []}) == "- skills (0):"


def test_an_envelope_field_past_the_field_cap_is_cut_and_says_so() -> None:
    """A `code`, a `severity`, an `entity_roles` list and every other field the
    resolve puts on the envelope render at most `_ENVELOPE_FIELD_CAP` characters."""
    long = "x" * 100_000
    envelope = {
        "status": long,
        "spec_id": long,
        "parts": long,
        "choice_points": long,
        "wide": long,
        "stat_var_shape_kind": long,
        "stat_var_item_count": long,
        "entity_roles": [long, long],
        "stat_var_truncated": long,
        "open_entity_axes": [{"column": long, "members": 1}],
        "candidates": [{"spec_id": long, "label": "a label"}],
        "warnings": [{"code": long, "severity": long, "message": "a message"}],
        "an_unnamed_field": long,
    }

    rendered = _render_envelope(envelope)

    cut = "x" * (_ENVELOPE_FIELD_CAP - 1) + "\u2026"
    assert rendered.count(cut) == 13
    (roles,) = [line for line in rendered.splitlines() if line.startswith("- entity_roles: ")]
    assert len(roles) == len("- entity_roles: ") + _ENVELOPE_FIELD_CAP
    assert roles.endswith("\u2026")
    assert len(rendered) < 16 * (_ENVELOPE_FIELD_CAP + 60)


def test_an_envelope_field_at_the_field_cap_is_rendered_whole() -> None:
    value = "x" * _ENVELOPE_FIELD_CAP

    rendered = _render_envelope(
        {"status": value, "warnings": [{"code": value, "severity": "warn"}]}
    )

    assert rendered.count(value) == 2
    assert "\u2026" not in rendered


def test_the_envelope_names_the_axis_search_left_open_with_its_member_count() -> None:
    # dev `search` for education aid to Kenya: Kenya bound under
    # `destination`, every donor left open on `origin`.
    fixture = Path(__file__).parent / "fixtures" / "search_spec_role_open.json"
    spec = json.loads(fixture.read_text())["spec"]
    envelope = graph_envelope(
        {"status": "definite", "parts": [{"status": "definite", "spec": spec}]}
    )

    lines = _render_envelope(envelope).splitlines()

    assert "- open_entity_axes: origin (55 members)" in lines
    assert not [line for line in lines if line.startswith("- open_entity_axes: [")]


def test_an_open_axis_states_one_member_or_an_uncounted_population() -> None:
    rendered = _render_envelope(
        {
            "status": "definite",
            "open_entity_axes": [
                {"column": "observationAbout", "members": 1},
                {"column": "origin", "members": None},
            ],
        }
    )

    assert (
        "- open_entity_axes: observationAbout (1 member), origin (members not counted)"
        in rendered.splitlines()
    )


@pytest.mark.parametrize(
    ("exact", "line"),
    [
        (True, "- stat_var_item_count: 3687"),
        (False, "- stat_var_item_count: at least 3,687 StatVars"),
    ],
)
def test_the_envelope_counts_a_floor_as_at_least(exact: bool, line: str) -> None:
    rendered = _render_envelope(
        {"status": "definite", "stat_var_item_count": 3687, "stat_var_member_count_exact": exact}
    )

    assert line in rendered.splitlines()
    assert "stat_var_member_count_exact" not in rendered


def test_an_envelope_with_no_open_axis_renders_no_open_entity_axes_line() -> None:
    assert "open_entity_axes" not in _render_envelope(
        {"status": "definite", "open_entity_axes": []}
    )


async def test_an_amendment_still_works_through_the_seam(
    decision: Any, requirement: Any, derived_requirement: Any
) -> None:
    """An amendment proposed against the digest amends the stored plan.

    The digest is what the planner's model reads to decide an amendment, so
    the end-to-end property is that a `propose_plan` call made after a
    digest-rendered brief still amends rather than resets: the stored plan
    moves version 5 -> 6 with `previous_version == 5`, and the requirements
    it does not restate are carried through. That is exactly what
    `Plan.amend` does when the model re-proposes, and it is the behaviour a
    digest that lost the prior plan would break.

    The call states `amends_version=5`, read off the "version 5" line of the
    digest this brief rendered. A brief that lost the prior plan would leave
    the model with no version to name.
    """
    harness = _SeamHarness()
    seam = _seam()
    seeded = _translated_plan(decision, requirement, derived_requirement, version=5).model_dump(
        mode="json"
    )
    proposal = {
        "question": "kenya debt",
        "answer_mode": "computed",
        "decisions": [decision().model_dump(mode="json")],
        "evidence_requirements": [
            requirement().model_dump(mode="json"),
            derived_requirement().model_dump(mode="json"),
        ],
        "amends_version": 5,
    }
    llm = ScriptedPlannerLlm(script=[[("propose_plan", proposal)]])
    seam._agent.model = llm

    with pytest.raises(PlannerSeamError):
        await harness.run(
            seam,
            question="kenya debt",
            envelope={"status": "definite", "spec_id": SPEC, "parts": 1},
            prior_plan=seeded,
            amendment="only since 2015",
        )

    # What the model actually read on its one generation: the digest, not JSON.
    brief = _prompt_text(llm.captured_requests[0])
    for preserved in ("kenya debt", "annual_debt_service", "annual_totals", SPEC):
        assert preserved in brief, (
            f"the brief the amendment ran against is missing {preserved!r}, so the model "
            "had no way to preserve it"
        )

    amended = load_plan(await harness.state())
    assert amended is not None
    assert amended.version == 6
    assert amended.previous_version == 5


def test_a_continuing_message_omits_the_envelope_and_digests_but_keeps_the_ask() -> None:
    """A `fresh=False` render -- a correction round, or an `amend`/`replan`
    re-entry into a session this seam has already briefed -- drops the
    envelope, both digests and the question, and carries the new thing (the
    amendment or refusal ask) verbatim.
    """
    seam = _seam()
    envelope = {"status": "definite", "spec_id": SPEC, "parts": 1}
    canvas_digest = {"blocks": ["a block"]}
    conversation_digest = {"turns": ["an earlier turn"]}

    amend_message = seam._message(
        fresh=False,
        question="gdp guatemala",
        envelope=envelope,
        prior_plan=None,
        refusal=None,
        amendment="only since 2015",
        canvas_digest=canvas_digest,
        conversation_digest=conversation_digest,
    )
    assert "only since 2015" in amend_message
    assert SPEC not in amend_message
    assert "Resolve envelope" not in amend_message
    assert "already on the user's canvas" not in amend_message
    assert "Earlier turns in this conversation" not in amend_message
    assert "Question:" not in amend_message

    refusal_message = seam._message(
        fresh=False,
        question="gdp guatemala",
        envelope=envelope,
        prior_plan=None,
        refusal=_rediscovery_refusal(SPEC),
        amendment=None,
        canvas_digest=canvas_digest,
        conversation_digest=conversation_digest,
    )
    assert "Executing the approved plan failed" in refusal_message
    assert "resolved.stat_vars.member_count = 872" in refusal_message
    assert "Resolve envelope" not in refusal_message
    assert "already on the user's canvas" not in refusal_message
    assert "Earlier turns in this conversation" not in refusal_message
    assert "Question:" not in refusal_message


def test_the_budget_refusal_is_stated_as_a_per_entry_budget_in_the_user_s_terms() -> None:
    """`MAX_PLAN_ROUNDS` counts the proposals of one entry into the planner, restarts included.

    The plan's own version can read higher than the count (it climbs across
    entries), so the sentence names the budget. It makes no claim about what
    can still be approved: the seam has no view of the graph's record, and the
    graph adds that claim where its record supports it.
    """
    text = PlannerSeam._budget_exhausted_text(
        {"detail": {"scope": "plan_rounds", "rounds": 4, "version": 9}}
    )

    assert text == (
        "This request has used its budget of 4 plan drafts, so nothing was "
        "recorded. Your next message starts with a fresh budget."
    )
    assert "version" not in text
    assert "This question" not in text


def test_the_question_fixed_refusal_is_stated_as_asking_the_other_question_separately() -> None:
    """This seam's pair is always built with `question_fixed=True`
    (`PlannerSeam.__init__`), so `propose_plan`'s own `QUESTION_FIXED`
    refusal (a restart while a plan is already recorded) becomes this
    text: what happened, and that the other question goes in its own
    message, never a claim about a budget, since none was spent."""
    text = PlannerSeam._refusal_text({"code": "question_fixed", "outcome": "refused"})

    assert "own message" in text
    assert "budget" not in text


def test_terminal_refusal_covers_budget_exhausted_and_question_fixed_but_not_invalid_plan() -> None:
    """`_is_terminal_refusal` is what `_track_tool_call` uses to stop the
    planner's turn on a `propose_plan` refusal that recorded nothing and
    left nothing to correct within this entry -- `BUDGET_EXHAUSTED`,
    `QUESTION_FIXED` and `NOTHING_TO_PLAN` alike. `INVALID_PLAN` -- a question-changing amendment
    under `question_fixed`, say -- is not among them: it is correctable,
    so the model (or, failing that, this seam's own correction loop) gets
    to retry rather than the turn ending on it."""
    for code in ("budget_exhausted", "question_fixed", "nothing_to_plan"):
        assert PlannerSeam._is_terminal_refusal({"outcome": "refused", "code": code})
    assert not PlannerSeam._is_terminal_refusal({"outcome": "refused", "code": "invalid_plan"})
    assert not PlannerSeam._is_terminal_refusal({"outcome": "ok", "code": "question_fixed"})
    assert not PlannerSeam._is_terminal_refusal(None)


# -- user text in the brief is cut at the question cap ---------------------------


def _render(seam: PlannerSeam, *, fresh: bool, **overrides: Any) -> str:
    """One `_message` render for a bare envelope, with `overrides` for what a test varies."""
    arguments: dict[str, Any] = {
        "fresh": fresh,
        "question": "gdp guatemala",
        "envelope": {},
        "prior_plan": None,
        "refusal": None,
        "amendment": None,
        "canvas_digest": None,
        "conversation_digest": None,
    }
    return seam._message(**{**arguments, **overrides})


def test_a_question_past_the_cap_is_cut_in_the_brief_and_says_so() -> None:
    """`TurnRequest.message` has no length limit, and a clarify reply lengthens the question."""
    message = _render(_seam(), fresh=True, question="q" * 1_000_000)

    (line,) = [line for line in message.splitlines() if line.startswith("Question: ")]
    assert line == "Question: " + "q" * (MAX_QUESTION_CHARS - len(CUT_MARKER)) + CUT_MARKER
    assert len(message) < 2 * MAX_QUESTION_CHARS


def test_a_question_at_the_cap_is_rendered_whole() -> None:
    question = "q" * MAX_QUESTION_CHARS

    message = _render(_seam(), fresh=True, question=question)

    assert f"Question: {question}\n" in message
    assert CUT_MARKER not in message


@pytest.mark.parametrize("fresh", [True, False])
def test_an_amendment_past_the_cap_is_cut_and_says_so(fresh: bool) -> None:
    message = _render(_seam(), fresh=fresh, amendment="a" * 1_000_000)

    (line,) = [
        line
        for line in message.splitlines()
        if line.startswith("The user asks for this change to the plan: ")
    ]
    asked = line.removeprefix("The user asks for this change to the plan: ")
    assert asked == "a" * (MAX_QUESTION_CHARS - len(CUT_MARKER)) + CUT_MARKER


@pytest.mark.parametrize("fresh", [True, False])
def test_a_refusal_past_the_cap_is_cut_and_says_so(fresh: bool) -> None:
    refusal = _rediscovery_refusal(SPEC)
    refusal["refusal"]["message"] = "m" * 1_000_000
    refusal["also_stopped"] = [_rediscovery_refusal(SPEC_B) for _ in range(1_000)]

    message = _render(_seam(), fresh=fresh, refusal=refusal)

    assert "Executing the approved plan failed" in message
    assert message.count(CUT_MARKER) >= 1
    # Three cuts at most: the refusal, the sentence naming the specs to avoid,
    # and, on a fresh render, the envelope's paragraph naming them.
    assert len(message) < 8 * MAX_QUESTION_CHARS


def test_a_short_refusal_is_rendered_whole() -> None:
    message = _render(_seam(), fresh=False, refusal=_rediscovery_refusal(SPEC))

    assert CUT_MARKER not in message
    assert "resolved.stat_vars.member_count = 872" in message


# -- a referral that repeats without new information ----------


def test_stalled_referral_stops_on_the_second_identical_pair_not_the_first() -> None:
    counts: dict[tuple[str, Any], int] = {}
    referred = {
        "outcome": "choice_required",
        "kind": "source_ambiguous",
        "requirement_id": "gdp_gtm",
        "candidates": [{"spec_id": "spec_nominal"}, {"spec_id": "spec_per_capita"}],
    }

    assert PlannerSeam._stalled_referral(referred, counts) is None

    message = PlannerSeam._stalled_referral(referred, counts)

    assert message is not None
    assert "gdp_gtm" in message
    assert "source_ambiguous" in message
    assert "spec_nominal" in message and "spec_per_capita" in message


@pytest.mark.parametrize("kind", ["unsupported_ask", "requirement_unmeetable"])
def test_stalled_referral_over_an_unmeetable_ask_tells_the_planner_to_ask_the_user(
    kind: str,
) -> None:
    """The user decides what to change for these kinds, and `propose_plan` refuses
    until they have answered, so the message sends the planner to them."""
    counts: dict[tuple[str, Any, Any], int] = {}
    referred = {
        "outcome": "invalid_input",
        "kind": kind,
        "requirement_id": "gdp_gtm",
        "reason": "r",
    }

    PlannerSeam._stalled_referral(referred, counts)
    message = PlannerSeam._stalled_referral(referred, counts)

    assert message is not None
    assert "Tell the user what cannot be done and ask" in message
    assert "change what the requirement asks for" not in message


def test_every_stalling_referral_kind_has_its_own_act() -> None:
    """Each kind that can stall names what to do for that kind; a transient kind
    never stalls and an internal one ends the entry before it could."""
    stalling = {
        kind
        for kind, outcome in OUTCOME_OF.items()
        if outcome not in (Outcome.TRANSIENT, Outcome.INTERNAL)
    }

    assert set(planner_seam._STALLED_ACTS) == stalling


def test_a_stalled_repair_referral_names_the_repair() -> None:
    counts: dict[tuple[str, Any], int] = {}
    referred = {
        "outcome": "invalid_input",
        "kind": "requirement_needs_repair",
        "requirement_id": "gdp_gtm",
        "constraint": "entity_required",
        "repair": "name the entity",
    }

    PlannerSeam._stalled_referral(referred, counts)
    message = PlannerSeam._stalled_referral(referred, counts)

    assert message is not None
    assert "name the entity" in message and "entity_required" in message
    assert "candidates" not in message.split(".", 1)[1]


@pytest.mark.parametrize("kind", ["discovery_unavailable", "surface_unavailable"])
def test_a_transient_referral_never_stalls(kind: str) -> None:
    counts: dict[tuple[str, Any], int] = {}
    referred = {
        "outcome": "transient",
        "kind": kind,
        "requirement_id": "gdp_gtm" if kind == "discovery_unavailable" else None,
        "reason": "r",
    }

    for _ in range(3):
        assert PlannerSeam._stalled_referral(referred, counts) is None
    assert counts == {}


def test_stalled_referral_reads_a_referral_with_no_candidates_recorded() -> None:
    """`pinned_source_not_found` carries no `candidates` today
    (`translator/referral.py`'s `_names_what_it_is_about`); this has to
    read that shape without assuming a future relaxation of that
    invariant has landed.
    """
    counts: dict[tuple[str, Any], int] = {}
    referred = {
        "outcome": "choice_required",
        "kind": "pinned_source_not_found",
        "requirement_id": "gdp_gtm",
    }

    PlannerSeam._stalled_referral(referred, counts)
    message = PlannerSeam._stalled_referral(referred, counts)

    assert message is not None
    assert "no candidates are on record" in message


def test_stalled_referral_ignores_a_response_that_is_not_a_referral() -> None:
    counts: dict[tuple[str, Any], int] = {}
    problem_response = {
        "outcome": "invalid_input",
        "code": "requirement_unknown",
        "message": "no such requirement",
    }

    assert PlannerSeam._stalled_referral(problem_response, counts) is None
    assert PlannerSeam._stalled_referral(problem_response, counts) is None
    assert counts == {}


@pytest.mark.parametrize(
    ("referral", "named"),
    [
        (
            {
                "kind": "internal_failure",
                "outcome": "internal",
                "reason": "cycle: loop",
                "reference": "0123456789ab",
            },
            "0123456789ab",
        ),
        (
            {
                "kind": "requirement_needs_repair",
                "outcome": "invalid_input",
                "requirement_id": "cmp",
                "reason": "entity_required",
                "repeated": True,
            },
            "'cmp'",
        ),
        (
            {
                "kind": "discovery_unavailable",
                "outcome": "transient",
                "requirement_id": "gdp_gtm",
                "reason": "used 4 discovery calls",
                "budget_spent": True,
            },
            "'gdp_gtm'",
        ),
    ],
    ids=["internal", "repeated", "budget-spent"],
)
def test_a_closing_referral_ends_the_entry_on_the_translate_response(
    referral: dict[str, Any], named: str
) -> None:
    """No correction round follows an internal failure, a translation that
    failed the way it failed before, or a data service that timed out until
    the discovery budget was spent: the entry ends as a final refusal naming
    what stopped it, as soon as `translate` answers with it."""
    tracker = _RunTracker(deadline_at=0.0)
    PlannerSeam._track_tool_call(
        tracker,
        None,
        agent_name="dfo_planner",
        tool=cast(BaseTool, SimpleNamespace(name="translate")),
        tool_response=referral,
    )

    assert tracker.ready_to_break
    assert tracker.submitted is None
    assert tracker.refused is not None
    assert named in PlannerSeam._refusal_text(tracker.refused)
    assert PlannerSeam._refused(tracker.refused)["refused"]["final"] is True


def test_last_referral_is_cleared_once_a_later_translate_resolves_the_requirement() -> None:
    """`_track_tool_call` writes `tracker.last_referral` from every `translate`
    response carrying an `outcome`, referral or not, replacing whatever the
    last one recorded: a later `ok` response clears it, so a deadline
    landing after that clear never asks the user about candidates the plan
    no longer needs (`_deadline_response`)."""
    tracker = _RunTracker(deadline_at=0.0)
    referred = {
        "outcome": "choice_required",
        "kind": "source_ambiguous",
        "requirement_id": "gdp_gtm",
        "candidates": [{"spec_id": "spec_nominal"}, {"spec_id": "spec_per_capita"}],
    }
    PlannerSeam._track_tool_call(
        tracker,
        None,
        agent_name="dfo_planner",
        tool=cast(BaseTool, SimpleNamespace(name="translate")),
        tool_response=referred,
    )
    assert tracker.last_referral is not None

    resolved = {"outcome": "ok", "requirement_id": "gdp_gtm"}
    PlannerSeam._track_tool_call(
        tracker,
        None,
        agent_name="dfo_planner",
        tool=cast(BaseTool, SimpleNamespace(name="translate")),
        tool_response=resolved,
    )

    assert tracker.last_referral is None


def test_last_referral_survives_a_translate_response_carrying_no_outcome() -> None:
    """A `translate` response missing its own `outcome` key -- not a shape
    `translate` itself ever returns, but the defensive case `_referral_dict`
    already reads for -- leaves `last_referral` as it was, so a deadline
    landing right after still has the last real referral's candidates to
    build its own `adk_request_input` from."""
    tracker = _RunTracker(deadline_at=0.0)
    referred = {
        "outcome": "choice_required",
        "kind": "source_ambiguous",
        "requirement_id": "gdp_gtm",
        "candidates": [{"spec_id": "spec_nominal"}],
    }
    PlannerSeam._track_tool_call(
        tracker,
        None,
        agent_name="dfo_planner",
        tool=cast(BaseTool, SimpleNamespace(name="translate")),
        tool_response=referred,
    )

    PlannerSeam._track_tool_call(
        tracker,
        None,
        agent_name="dfo_planner",
        tool=cast(BaseTool, SimpleNamespace(name="translate")),
        tool_response={"malformed": True},
    )

    assert tracker.last_referral is not None
    assert tracker.last_referral["requirement_id"] == "gdp_gtm"


def test_a_referral_resolved_by_a_later_ok_translate_lets_a_deadline_refuse_instead_of_asking() -> (
    None
):
    """`_before_planner_model`'s deadline branch reads `tracker.last_referral`
    through `_deadline_response`: once a later `ok` `translate` response has
    cleared it (`test_last_referral_is_cleared_once_a_later_translate_
    resolves_the_requirement`), a deadline landing after that clear finds no
    pinnable candidates and refuses with `_DEADLINE_REFUSAL_CODE`, never the
    `adk_request_input` ask a still-open referral would get."""
    seam = _seam(clock=lambda: 10_000.0)
    tracker = _RunTracker(deadline_at=0.0)
    referred = {
        "outcome": "choice_required",
        "kind": "source_ambiguous",
        "requirement_id": "gdp_gtm",
        "candidates": [{"spec_id": "spec_nominal"}, {"spec_id": "spec_per_capita"}],
    }
    PlannerSeam._track_tool_call(
        tracker,
        None,
        agent_name="dfo_planner",
        tool=cast(BaseTool, SimpleNamespace(name="translate")),
        tool_response=referred,
    )
    resolved = {"outcome": "ok", "requirement_id": "gdp_gtm"}
    PlannerSeam._track_tool_call(
        tracker,
        None,
        agent_name="dfo_planner",
        tool=cast(BaseTool, SimpleNamespace(name="translate")),
        tool_response=resolved,
    )

    callback_context = SimpleNamespace(session=SimpleNamespace(state={}))
    token = planner_seam._active_call.set(
        planner_seam._CallState(tracker=tracker, on_progress=None)
    )
    try:
        response = seam._before_planner_model(callback_context, llm_request=None)
    finally:
        planner_seam._active_call.reset(token)

    assert response is not None
    (part,) = response.content.parts
    assert part.function_call is None, (
        "a cleared referral must not surface an adk_request_input ask"
    )
    assert tracker.refused is not None
    assert tracker.refused["code"] == _DEADLINE_REFUSAL_CODE


def test_a_referral_surviving_a_malformed_translate_response_still_gets_asked_at_the_deadline() -> (
    None
):
    """A `translate` response with no `outcome` -- the typed problem
    `run_translate` answers a malformed resume with -- settles nothing
    (`test_last_referral_survives_a_translate_response_carrying_no_outcome`),
    so a deadline landing after it still finds the earlier referral's
    candidates and asks through `adk_request_input`, the same as if the
    malformed response had never arrived."""
    seam = _seam(clock=lambda: 10_000.0)
    tracker = _RunTracker(deadline_at=0.0)
    referred = {
        "outcome": "choice_required",
        "kind": "source_ambiguous",
        "requirement_id": "gdp_gtm",
        "candidates": [{"spec_id": "spec_nominal"}, {"spec_id": "spec_per_capita"}],
    }
    PlannerSeam._track_tool_call(
        tracker,
        None,
        agent_name="dfo_planner",
        tool=cast(BaseTool, SimpleNamespace(name="translate")),
        tool_response=referred,
    )
    PlannerSeam._track_tool_call(
        tracker,
        None,
        agent_name="dfo_planner",
        tool=cast(BaseTool, SimpleNamespace(name="translate")),
        tool_response={"malformed": True},
    )

    callback_context = SimpleNamespace(session=SimpleNamespace(state={}))
    token = planner_seam._active_call.set(
        planner_seam._CallState(tracker=tracker, on_progress=None)
    )
    try:
        response = seam._before_planner_model(callback_context, llm_request=None)
    finally:
        planner_seam._active_call.reset(token)

    assert response is not None
    (part,) = response.content.parts
    assert part.function_call is not None
    assert part.function_call.name == REQUEST_INPUT_FUNCTION_CALL_NAME
    assert {c["spec_id"] for c in part.function_call.args["candidates"]} == {
        "spec_nominal",
        "spec_per_capita",
    }
    assert tracker.refused is None


# A `no_data` `search` answer, canned so every `discover_source` call inside
# `translate`'s own mechanical loop gets the identical answer -- no nearest
# series, no judgement to pause on, so `translate` refers the plan back on
# its own, in Python, the same way every time.
_NO_DATA_SEARCH_RESPONSE_ENVELOPE = {
    "schema_version": 1,
    "status": "no_data",
    "parts": [
        {
            "status": "no_data",
            "variable_text": "asked",
            "warnings": [],
            "no_data": {"reason": "entity_not_resolved"},
        }
    ],
}

_NO_DATA_SEARCH_RESPONSE = ToolCallResult(
    is_error=False, **search_result_parts(_NO_DATA_SEARCH_RESPONSE_ENVELOPE)
)


def _repeating_translate_call_script(rounds: int) -> list[Any]:
    """The planner's calls for `rounds` pairs of `translate` calls.

    `translate` runs the real mechanical loop against
    `_NO_DATA_SEARCH_RESPONSE`, so both calls in a pair refer the same
    `(kind, requirement_id)` back. That is the shape that stops one
    `ctx.run_node` call on its second identical referral, once per correction
    round the seam pays for before it raises.
    """
    call = [("translate", {})]
    script: list[Any] = []
    for _ in range(rounds):
        script.extend([call, call])
    return script


def _translator_seam(
    *,
    responses: dict[str, ToolCallResult],
    agent_settings: AgentSettings | None = None,
    clock: Callable[[], float] = time.monotonic,
    thinking_sample_rng: Callable[[], float] = random.random,
) -> PlannerSeam:
    return PlannerSeam(
        settings=Settings(),
        surface=FixtureSurface(schemas=[], responses=responses),
        agent_settings=agent_settings,
        clock=clock,
        thinking_sample_rng=thinking_sample_rng,
    )


def _gdp_plan(decision: Any, requirement: Any, derived_requirement: Any) -> Plan:
    return Plan(
        question="gdp guatemala",
        decisions=[decision()],
        evidence_requirements=[
            requirement(id="gdp_gtm"),
            derived_requirement(input_id="gdp_gtm"),
        ],
    )


def test_the_planner_keeps_its_own_after_model_callback_alongside_the_seams() -> None:
    """`PlannerSeam.__init__` assigns `after_model_callback` once, directly,
    so a straight overwrite there would silently drop `drop_thought_parts`
    (`create_planner_agent`) on every graph-served turn. `usage_metadata` logging always returns
    `None`, so it belongs first in the list: `after_model_callback` stops at
    the first truthy return, and only `drop_thought_parts` -- which returns
    the trimmed response, a truthy value, on any generation that thought --
    would ever end the list early.
    """
    seam = _translator_seam(responses={})

    assert isinstance(seam._agent.after_model_callback, list)
    assert seam._agent.after_model_callback[0] == seam._after_planner_model
    assert drop_thought_parts in seam._agent.after_model_callback


def test_the_planner_composes_the_thinking_override_ahead_of_the_deadline_check() -> None:
    """A single assignment to `before_model_callback` here would silently
    drop whichever of `override_thinking_on_sampled_turns` and
    `_before_planner_model` lost the slot. `override_thinking_on_sampled_turns`
    always returns `None`, so it runs first without ending
    `_before_planner_model`'s own turn to check the deadline and the
    stalled-referral break -- `test_a_deadline_that_has_already_passed_
    refuses_before_any_generation` and its neighbours exercise that this
    composed list still reaches it.
    """
    seam = _seam()

    assert seam._agent.before_model_callback == [
        override_thinking_on_sampled_turns,
        seam._before_planner_model,
    ]


async def test_a_zero_sample_rate_never_samples(
    decision: Any, requirement: Any, derived_requirement: Any
) -> None:
    """`0` disables sampling regardless of the draw: `0.0 < 0.0` is `False`,
    the smallest value `random.random()` can ever return."""
    seam = _translator_seam(
        responses={"search": _NO_DATA_SEARCH_RESPONSE},
        agent_settings=AgentSettings(thinking_sample_rate=0.0),
        thinking_sample_rng=lambda: 0.0,
    )
    draft = _gdp_plan(decision, requirement, derived_requirement)
    planner_llm = ScriptedPlannerLlm(script=[[("translate", {})]])
    seam._agent.model = planner_llm

    with pytest.raises(PlannerSeamError):
        await _run_seam(
            seam, question=draft.question, envelope={}, prior_plan=draft.model_dump(mode="json")
        )

    assert planner_llm.captured_requests[0].config.thinking_config == AGENT_THINKING_CONFIG


async def test_a_rate_one_always_samples_and_every_generation_carries_the_default_config(
    decision: Any, requirement: Any, derived_requirement: Any
) -> None:
    """`1` always samples: `random.random()` never returns exactly `1.0`, so
    `draw < 1.0` always holds. One draw, made once for this `__call__`
    (`PlannerSeam.__call__`'s own comment), is shared through session state
    (`THINKING_SAMPLE_STATE_KEY`) by every generation's
    `before_model_callback`."""
    seam = _translator_seam(
        responses={"search": _NO_DATA_SEARCH_RESPONSE},
        agent_settings=AgentSettings(thinking_sample_rate=1.0),
        thinking_sample_rng=lambda: 0.5,
    )
    draft = _gdp_plan(decision, requirement, derived_requirement)
    planner_llm = ScriptedPlannerLlm(script=[[("translate", {})]])
    seam._agent.model = planner_llm

    with pytest.raises(PlannerSeamError):
        await _run_seam(
            seam, question=draft.question, envelope={}, prior_plan=draft.model_dump(mode="json")
        )

    assert len(planner_llm.captured_requests) > 1
    assert all(
        request.config.thinking_config == DEFAULT_THINKING_CONFIG
        for request in planner_llm.captured_requests
    )


async def test_the_thinking_sample_is_drawn_once_and_recorded_on_the_turn_span(
    decision: Any, requirement: Any, derived_requirement: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The draw is made once per turn, not once per `_propose` call: every
    generation of one `__call__`, correction rounds included, sees the same
    recorded draw, and the metadata
    `common.telemetry.record_span` gets tags this turn as sampled records
    once, at the draw, not again on every later generation of the same
    turn."""
    spans: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(
        planner_seam, "record_span", lambda name, attrs: spans.append((name, dict(attrs)))
    )
    seam = _translator_seam(
        responses={"search": _NO_DATA_SEARCH_RESPONSE},
        agent_settings=AgentSettings(thinking_sample_rate=1.0),
        thinking_sample_rng=lambda: 0.5,
    )
    draft = _gdp_plan(decision, requirement, derived_requirement)
    planner_llm = ScriptedPlannerLlm(script=[[("translate", {})]])
    seam._agent.model = planner_llm
    harness = _SeamHarness()

    with pytest.raises(PlannerSeamError):
        await harness.run(
            seam, question=draft.question, envelope={}, prior_plan=draft.model_dump(mode="json")
        )

    assert spans == [("planner.thinking_sampled", {"dfo.thinking_sampled": True})]
    state = await harness.state()
    assert state[THINKING_SAMPLE_STATE_KEY] is True


async def test_a_correction_round_carries_the_brief_and_the_earlier_tool_calls(
    decision: Any, requirement: Any, derived_requirement: Any
) -> None:
    """A correction round's message carries only the nudge (`_message`'s
    `fresh=False`), so its first generation has to read the brief and the
    first round's tool calls from history. Run as a workflow node with no
    explicit `include_contents`, ADK defaults the planner to `'none'`, which
    builds every request from the latest user event onwards
    (`_get_current_turn_contents`, `google.adk.flows.llm_flows.contents`),
    which is the nudge alone. `planner.agent.create_planner_agent` pins
    `include_contents='default'` to stop that; reverting the pin fails this
    test on the correction round's first generation.
    """
    seam = _translator_seam(
        responses={
            "search": _NO_DATA_SEARCH_RESPONSE,
            "get_skill": ToolCallResult(is_error=False, text="# DFO FAQ"),
        }
    )
    draft = _gdp_plan(decision, requirement, derived_requirement)
    planner_llm = ScriptedPlannerLlm(
        script=[
            [("get_skill", {"skill_id": "dfo-faq"})],
            [("translate", {})],
            "stop",
        ]
    )
    seam._agent.model = planner_llm

    with pytest.raises(PlannerSeamError):
        await _run_seam(
            seam,
            question="gdp guatemala",
            envelope={},
            prior_plan=draft.model_dump(mode="json"),
        )

    # `captured_requests[0..2]` are the first round: `get_skill`, `translate`,
    # and the closing text that never called `submit_plan`. `[3]` is the
    # correction round's first generation.
    assert len(planner_llm.captured_requests) >= 4
    correction = planner_llm.captured_requests[3]
    assert "Question: gdp guatemala" in _prompt_text(correction)
    names = _call_and_response_names(correction)
    assert "FC:get_skill" in names
    assert "FR:get_skill" in names
    assert "FC:translate" in names
    assert "FR:translate" in names


# An `unsupported` `search` answer: `translate` refers the plan back as
# `unsupported_ask`, a referral only the user's answer settles.
_UNSUPPORTED_SEARCH_RESPONSE_ENVELOPE = {
    "schema_version": 1,
    "status": "unsupported",
    "parts": [
        {
            "status": "unsupported",
            "variable_text": "asked",
            "warnings": [],
            "unsupported": {"code": "out_of_scope", "detail": "not a data question"},
        }
    ],
}

_UNSUPPORTED_SEARCH_RESPONSE = ToolCallResult(
    is_error=False, **search_result_parts(_UNSUPPORTED_SEARCH_RESPONSE_ENVELOPE)
)


async def test_an_answer_that_resumes_a_run_after_a_translate_call_is_read(
    decision: Any, requirement: Any, derived_requirement: Any
) -> None:
    """The user's answer resumes a run whose earlier turn called `translate`.

    The planner calls `translate`, which refers `unsupported_ask`, then
    asks the user through `adk_request_input` and pauses. The answer resumes
    the run, and ADK finishes it without a new generation and without a
    planner-authored event. The seam has to accept that completion, then
    correct the run as it does any run that ends without `submit_plan`: the
    next round reads the answer off the session and `propose_plan` passes
    the referral guard as version 2. The `unsupported_ask` decline it owes
    survives the resume, so the amendment carries it.
    """
    draft = _gdp_plan(decision, requirement, derived_requirement)
    seam = _translator_seam(responses={"search": _UNSUPPORTED_SEARCH_RESPONSE})
    amend = [
        (
            "propose_plan",
            {
                "question": draft.question,
                "amends_version": draft.version,
                "decisions": [draft.decisions[0].model_dump(mode="json")],
                "evidence_requirements": [],
                "declined": [{"ask": draft.question, "code": "unsupported_ask"}],
            },
        )
    ]
    planner_llm = ScriptedPlannerLlm(
        script=[
            [("translate", {})],
            [("adk_request_input", {"message": "what can be measured instead?", "candidates": []})],
            amend,
            "stop",
        ]
    )
    seam._agent.model = planner_llm
    harness = _SeamHarness()
    plan_json = draft.model_dump(mode="json")

    paused = await harness.run(seam, question=draft.question, envelope={}, prior_plan=plan_json)
    assert paused is None, "the ask should pause this call, not resolve it"

    interrupt_id = _pending_ask_interrupt_id(await harness.events())
    answer = types.Content(
        role="user",
        parts=[create_request_input_response(interrupt_id, {"result": {"text": "use debt stock"}})],
    )
    with pytest.raises(PlannerSeamError) as raised:
        await harness.run(
            seam,
            question=draft.question,
            envelope={},
            prior_plan=plan_json,
            new_message=answer,
        )

    assert "submit_plan" in str(raised.value), (
        "the resumed run should end the way any run without `submit_plan` does, on the "
        f"seam's own correction rounds, not on the branch check: {raised.value}"
    )
    proposals = [
        part.function_response.response
        for event in await harness.events()
        for part in (event.content.parts if event.content and event.content.parts else [])
        if part.function_response is not None and part.function_response.name == "propose_plan"
    ]
    assert [(p or {}).get("outcome") for p in proposals] == ["ok"], proposals
    assert (proposals[0] or {})["version"] == draft.version + 1


async def test_a_repeating_referral_stops_before_max_events(
    decision: Any, requirement: Any, derived_requirement: Any
) -> None:
    """Once `_MAX_CORRECTIONS` is spent on referrals that keep repeating,
    `PlannerSeamError` carries a `refusal` a caller must not retry -- the
    same property the guard always had, proved here against real
    `translate` calls.

    Every `translate` call spends a real `discover_source` call unless the
    requirement already settled (`translator.mechanical._resolve_dataset_sources`
    only skips discovery for a requirement `DiscoveryLog.selectable` already
    has an answer for, and this fixture's requirement never settles -- every
    `search` answers `no_data`), so three real rounds of two calls each run
    past `DISCOVERY_CALL_BUDGET_PER_REQUIREMENT` (4): the first two rounds
    stall on a repeating `no_data` referral, and the third stalls on
    `discovery_budget_exhausted` instead -- a different kind, still caught
    by the same guard, since it does not care which kind repeats.
    """
    seam = _translator_seam(responses={"search": _NO_DATA_SEARCH_RESPONSE})
    draft = _gdp_plan(decision, requirement, derived_requirement)

    # One initial run, plus every `_MAX_CORRECTIONS` round -- each one a
    # pair that never settles, so the seam pays for all of them and still
    # raises on the last round's referral, not on the event cap.
    rounds = _MAX_CORRECTIONS + 1
    planner_llm = ScriptedPlannerLlm(script=_repeating_translate_call_script(rounds))
    seam._agent.model = planner_llm

    with pytest.raises(PlannerSeamError) as excinfo:
        await _run_seam(
            seam,
            question="gdp guatemala",
            envelope={},
            prior_plan=draft.model_dump(mode="json"),
        )

    message = str(excinfo.value)
    assert "for the second time" in message
    refusal = excinfo.value.refusal
    assert refusal["retry_mode"] != "retry_after_delay"
    assert refusal["fix"]["detail"] == message

    # Two planner generations per round -- the stop fires on the second
    # `translate` call's referral every time, so the planner is never asked
    # for a third within a round.
    assert len(planner_llm.captured_requests) == 2 * rounds


async def test_a_run_leaves_every_event_it_appended_scoped_to_the_planner(
    decision: Any, requirement: Any, derived_requirement: Any
) -> None:
    """The planner's own events carry `_PLANNER_BRANCH`, and no other agent runs.

    ADK's `_is_event_belongs_to_branch` returns True when either side's branch
    is empty, so keeping the planner's turns out of another agent's request
    (the orchestrator's, on its own branch) rests entirely on the branch the
    run is stamped with. `ctx.run_node` takes it as
    `override_branch=_PLANNER_BRANCH`. This asserts what that bought, rather
    than the argument being passed.

    `translate` runs inside the planner's own tool call, so every function
    call and response the run appended is the planner's: no second agent
    makes a model call in a planner run.
    """
    seam = _translator_seam(responses={"search": _NO_DATA_SEARCH_RESPONSE})
    draft = _gdp_plan(decision, requirement, derived_requirement)
    harness = _SeamHarness()
    seam._agent.model = ScriptedPlannerLlm(
        script=_repeating_translate_call_script(1) + ["done"] * 4
    )

    with pytest.raises(PlannerSeamError):
        await harness.run(
            seam,
            question=draft.question,
            envelope={},
            prior_plan=draft.model_dump(mode="json"),
        )

    events = await harness.events()
    planner_events = [event for event in events if event.author == "dfo_planner"]
    tool_authors = {
        event.author
        for event in events
        if event.get_function_calls() or event.get_function_responses()
    }

    assert any(event.get_function_responses() for event in planner_events), (
        "the run answered no planner tool call, so there is nothing to assert on"
    )
    assert tool_authors == {"dfo_planner"}
    assert {event.branch for event in planner_events} == {_PLANNER_BRANCH}, (
        "a planner event this run appended is not scoped to "
        f"{_PLANNER_BRANCH!r}: "
        f"{sorted({str(event.branch) for event in planner_events})!r}. Without "
        "`override_branch=_PLANNER_BRANCH` on the planner's `ctx.run_node` call, ADK "
        "treats an unbranched event as belonging to every branch."
    )


def _branch_check(*branches: tuple[str, str | None]) -> None:
    """Run `_check_branches` over one event per `(author, branch)` pair."""
    _check_branches(
        [Event(author=author, branch=branch) for author, branch in branches],
        planner="dfo_planner",
    )


@pytest.mark.parametrize("branch", [None, "", "dfo_orchestrator", "dfo_planner.x", "dfo_planner@1"])
def test_a_planner_event_off_the_planner_branch_fails_the_branch_check(
    branch: str | None,
) -> None:
    """The planner's events carry exactly `_PLANNER_BRANCH`. Any other value fails, an empty one included."""
    with pytest.raises(PlannerSeamError, match="other than"):
        _branch_check(("dfo_planner", branch))


@pytest.mark.parametrize(
    "branches",
    [(), (("user", None),), (("dfo_orchestrator", "dfo_orchestrator"),)],
)
def test_a_slice_with_no_planner_event_passes_the_branch_check(
    branches: tuple[tuple[str, str | None], ...],
) -> None:
    """A run may leave no planner-authored event. A resumed run that finishes the
    model turn the user's answer completed appends none, and the check judges the
    events it is given."""
    _branch_check(*branches)


def test_the_branch_check_ignores_events_from_other_authors() -> None:
    """Only the planner's events are judged, so a user event may be bare."""
    _branch_check(
        ("user", None),
        ("dfo_planner", _PLANNER_BRANCH),
    )


async def test_translate_finds_the_plan_with_no_arguments(
    decision: Any, requirement: Any, derived_requirement: Any
) -> None:
    """The plan round-trips through session state alone. Nothing is passed in.

    The planner calls `translate` with no arguments, and `run_translate`
    reaching the draft through `current_draft(tool_context)` is enough to
    conclude: the stub answers `no_data`, and the referral names `gdp_gtm`,
    the plan's own requirement id, which no argument carried. The planner's
    next generation reads that referral off the tool response.
    """
    seam = _translator_seam(responses={"search": _NO_DATA_SEARCH_RESPONSE})
    draft = _gdp_plan(decision, requirement, derived_requirement)
    planner_llm = ScriptedPlannerLlm(script=[[("translate", {})], "stop"])
    seam._agent.model = planner_llm

    with pytest.raises(PlannerSeamError):
        await _run_seam(
            seam,
            question=draft.question,
            envelope={},
            prior_plan=draft.model_dump(mode="json"),
        )

    answers = [
        part.function_response.response
        for request in planner_llm.captured_requests
        for content in request.contents
        for part in content.parts or []
        if part.function_response is not None and part.function_response.name == "translate"
    ]
    assert any(
        isinstance(answer, dict) and answer.get("requirement_id") == "gdp_gtm" for answer in answers
    ), f"translate never reached the plan's own requirement: {answers!r}"


def _candidate_spec(spec_id: str, sentence: str) -> dict[str, Any]:
    """One discovered source, in the shape `search` serves it."""
    return {
        "spec_id": spec_id,
        "resolved": {
            "rendered_sentence": sentence,
            "confidence": "high",
            "coverage": {
                "kind": "exact",
                "has_data": True,
                "window": {"start_year": 2010, "end_year": 2024},
            },
            "stat_vars": {
                "items": [{"ref": "AidDisbursed", "label": "Aid disbursed", "slot_values": []}],
                "member_count": 1,
                "truncated": False,
            },
            "entities": [{"role": "recipient", "label": "Guatemala", "ref": "country/GTM"}],
            "facets": {"unit": {"basis": "declared_default", "ref": "usd", "label": "US dollars"}},
        },
        "definition": {"schema_version": 1, "spec_id": spec_id, "identity": {"stat_var": "x"}},
    }


_TWO_CANDIDATES_SEARCH_RESPONSE_ENVELOPE = {
    "schema_version": 1,
    "status": "candidates",
    "parts": [
        {
            "status": "candidates",
            "variable_text": "asked",
            "warnings": [],
            "candidates": {
                "specs": [
                    _candidate_spec(SPEC, "aid from France"),
                    _candidate_spec(SPEC_B, "aid to Togo"),
                ],
                "truncated": False,
            },
        }
    ],
}

_TWO_CANDIDATES_SEARCH_RESPONSE = ToolCallResult(
    is_error=False, **search_result_parts(_TWO_CANDIDATES_SEARCH_RESPONSE_ENVELOPE)
)


async def test_the_planners_translate_response_is_the_referral_with_its_outcome(
    decision: Any, requirement: Any, derived_requirement: Any
) -> None:
    """On the path the planner is served from, the function response for
    its `translate` call is the referral itself: its `outcome` on top
    (`choice_required` for two candidates), its `kind`, its requirement, and
    both candidates with the `spec_id` each may be pinned by.
    """
    seam = _translator_seam(responses={"search": _TWO_CANDIDATES_SEARCH_RESPONSE})
    draft = Plan(
        question="aid to guatemala",
        decisions=[decision()],
        evidence_requirements=[
            requirement(id="gdp_gtm", measure="aid disbursed"),
            derived_requirement(input_id="gdp_gtm"),
        ],
    )
    planner_llm = ScriptedPlannerLlm(script=[[("translate", {})], "stop"])
    seam._agent.model = planner_llm

    with pytest.raises(PlannerSeamError):
        await _run_seam(
            seam,
            question="aid to guatemala",
            envelope={},
            prior_plan=draft.model_dump(mode="json"),
        )

    (response,) = [
        part.function_response
        for part in planner_llm.captured_requests[1].contents[-1].parts
        if part.function_response is not None
    ]
    assert response.name == "translate"
    answer = cast(dict[str, Any], response.response)
    assert answer["outcome"] == "choice_required", answer
    assert answer["kind"] == "source_ambiguous"
    assert answer["requirement_id"] == "gdp_gtm"
    assert {candidate["spec_id"] for candidate in answer["candidates"]} == {SPEC, SPEC_B}


async def test_two_conversations_do_not_share_the_seams_callbacks(
    decision: Any, requirement: Any, derived_requirement: Any
) -> None:
    """One `PlannerSeam` serves every conversation, and its callbacks are
    assigned once, in `__init__`, so a callback reads the call it belongs to
    off `_active_call`. Conversation A pauses on its first planner generation,
    conversation B runs its whole turn, and only then does A call `translate`:
    A's report has to land in A's own `on_progress` stream, and B's stream
    must not grow once A resumes.
    """
    seam = _translator_seam(responses={"search": _NO_DATA_SEARCH_RESPONSE})
    draft = _gdp_plan(decision, requirement, derived_requirement)
    started_a = asyncio.Event()
    release_a = asyncio.Event()
    reports_a: list[dict[str, Any]] = []
    reports_b: list[dict[str, Any]] = []

    planner_llm_a = ScriptedPlannerLlm(
        script=[
            [("translate", {})],
            "nothing yet",
            "nothing yet",
            "nothing yet",
        ],
        pause_at=0,
        paused=started_a,
        release=release_a,
    )
    planner_llm_b = ScriptedPlannerLlm(script=["nothing at all"] * 4)

    harness_a = _SeamHarness()
    harness_b = _SeamHarness()

    async def _run_a() -> None:
        seam._agent.model = planner_llm_a
        with pytest.raises(PlannerSeamError):
            await harness_a.run(
                seam,
                question="gdp guatemala",
                envelope={},
                prior_plan=draft.model_dump(mode="json"),
                on_progress=reports_a.append,
            )

    task_a = asyncio.create_task(_run_a())
    await started_a.wait()

    # Conversation B's whole turn, in this coroutine's own task, while A
    # sits suspended above, before its `translate` call.
    seam._agent.model = planner_llm_b
    with pytest.raises(PlannerSeamError):
        await harness_b.run(
            seam, question="unrelated question", envelope={}, on_progress=reports_b.append
        )
    reports_b_after_b = list(reports_b)

    release_a.set()
    await task_a

    assert reports_b == reports_b_after_b, (
        "conversation B's on_progress stream grew after B's own turn had already "
        f"finished -- conversation A's own translate call reported into it: {reports_b}"
    )
    assert any(report["tools"] == ["translate"] for report in reports_a), (
        "conversation A's own translate call never reported into A's own on_progress stream"
    )


def test_planner_seam_error_carries_a_do_not_retry_refusal_by_default() -> None:
    exc = PlannerSeamError("the pair did not converge")

    assert exc.refusal["retry_mode"] == "do_not_retry"
    assert exc.refusal["fix"]["detail"] == "the pair did not converge"


# -- the plan-round budget resets per turn, not per call -------------------


def _amendment_proposal(
    decision: Any, requirement: Any, derived_requirement: Any
) -> dict[str, Any]:
    made_decision = decision()
    requirements = [requirement(), derived_requirement()]
    return {
        "question": "kenya debt",
        "answer_mode": "computed",
        "decisions": [made_decision.model_dump(mode="json")],
        "evidence_requirements": [r.model_dump(mode="json") for r in requirements],
    }


def _amend_step(proposal: dict[str, Any], *, start_version: int) -> Any:
    """A script entry that calls `propose_plan` as an amendment of whatever
    version this turn's recorded plan stands at.

    `propose_plan` refuses a call that states neither `amends_version` nor
    `starts_new_plan` once a plan is recorded, and refuses a version that is
    not the recorded one, so a script that proposes round after round within
    one turn has to read each answer's own `version`. `start_version` is the
    version the turn opens on: the version of a plan the test seeded or an
    earlier turn recorded. It goes unconsulted on the first call of a turn
    whose session holds no plan, since that call is a first draft and its
    intent is not in question.
    """

    def step(request: Any) -> list[tuple[str, dict[str, Any]]]:
        version = _last_amended_version(request, default=start_version)
        return [("propose_plan", {**proposal, "amends_version": version})]

    return step


async def test_a_restart_on_the_amend_edge_is_refused_as_question_fixed(
    decision: Any, requirement: Any, derived_requirement: Any
) -> None:
    """The `amend` edge (`approve_n -> planner_n`) seeds `prior_plan` from
    the graph's own record, so a plan is already recorded when this call's
    one generation runs. This seam's pair is always built with
    `question_fixed=True` (`PlannerSeam.__init__`), so a `starts_new_plan`
    call there is refused as `QUESTION_FIXED` -- terminal, recording
    nothing -- rather than beginning a second plan for a different
    question out from under the graph's own `question` state.
    """
    seeded = _translated_plan(decision, requirement, derived_requirement, version=3)
    seam = _seam()
    seam._agent.model = ScriptedPlannerLlm(
        script=[[("propose_plan", {"starts_new_plan": True, "question": "a different question"})]]
    )
    harness = _SeamHarness()

    result = await harness.run(
        seam,
        question="kenya debt",
        envelope={},
        prior_plan=seeded.model_dump(mode="json"),
        amendment="actually, tell me about Kenya's ODA instead",
    )

    assert result is not None and "refused" in result
    assert "own message" in result["refused"]["text"]
    plan = load_plan(await harness.state())
    assert plan is not None
    assert plan.version == 3
    assert plan.question == "kenya debt"


async def test_a_restart_inside_a_first_entry_correction_round_is_refused_as_question_fixed(
    decision: Any, requirement: Any, derived_requirement: Any
) -> None:
    """A first-draft entry -- no amendment, no replan -- can still open its
    own correction round (`__call__`'s own loop, `planner_seam.py:517-550`):
    the first `propose_plan` call records a draft, the model's own turn
    ends there with no `submit_plan`, and this seam sends a nudge on a
    fresh `_propose` call. The question is already the graph's, recorded
    on that very draft, so a restart there has no more of a legitimate use
    than one on the `amend` edge: refused the same way.
    """
    proposal = _amendment_proposal(decision, requirement, derived_requirement)
    seam = _seam()
    seam._agent.model = ScriptedPlannerLlm(
        script=[
            [("propose_plan", proposal)],
            "Let me think about how best to answer this.",
            [
                (
                    "propose_plan",
                    {"starts_new_plan": True, "question": "a different question"},
                )
            ],
        ]
    )
    harness = _SeamHarness()

    result = await harness.run(seam, question="kenya debt", envelope={})

    assert result is not None and "refused" in result
    assert "own message" in result["refused"]["text"]
    plan = load_plan(await harness.state())
    assert plan is not None
    assert plan.version == 1
    assert plan.question == "kenya debt"


async def test_a_second_turn_on_one_session_clears_the_first_drafts_plan_and_briefed_flag(
    decision: Any, requirement: Any, derived_requirement: Any
) -> None:
    """`evals/agents/dfo_analyst_graph` (`just adk-web`) keeps one ADK
    session across turns, so the graph's planner node passes no
    `prior_plan` on a later turn's own first-draft entry even though
    `ctx.state["plan"]` still holds an earlier turn's. This seam reads that
    the same way: `prior_plan is None` clears `planner.plan.PLAN_STATE_KEY`
    and the briefed flag at the top of `__call__` (see its own comment),
    so the second turn both renders the full brief again -- proven here
    against the actual rendered prompt, not merely against a flag -- and
    drafts its own plan rather than being refused for restating no
    `amends_version` against a plan still on record from the first turn.
    """
    harness = _SeamHarness()
    seam = _seam()
    first_plan = _translated_plan(decision, requirement, derived_requirement, version=1)
    first_llm = ScriptedPlannerLlm(script=[[("submit_plan", {"version": first_plan.version})]])
    seam._agent.model = first_llm

    first = await harness.run(
        seam,
        question="kenya debt",
        envelope={"status": "definite"},
        prior_plan=first_plan.model_dump(mode="json"),
    )
    assert first is not None and "plan" in first

    second_proposal = {
        "question": "kenya oda 2023",
        "answer_mode": "computed",
        "decisions": [decision().model_dump(mode="json")],
        "evidence_requirements": [
            requirement().model_dump(mode="json"),
            derived_requirement().model_dump(mode="json"),
        ],
    }
    second_llm = ScriptedPlannerLlm(script=[[("propose_plan", second_proposal)]])
    seam._agent.model = second_llm

    with pytest.raises(PlannerSeamError):
        # This second entry never calls `submit_plan` (the drafted plan
        # carries no translation, `propose_plan` alone cannot compile
        # one), so it ends in `PlannerSeamError` for want of a submission
        # -- the recorded plan `propose_plan` itself left behind, not the
        # raise, is what this test is about.
        await harness.run(seam, question="kenya oda 2023", envelope={"status": "definite"})

    # The second call's own generation was rendered fresh -- the full
    # brief, "Question: kenya oda 2023" line included -- not a continuing
    # message, which a session already marked briefed would have gotten
    # instead (see `test_the_full_brief_a_second_call_inherits_is_byte_
    # identical_to_the_first`, which proves the other side of that split).
    brief = _prompt_text(second_llm.captured_requests[0])
    assert "Question: kenya oda 2023" in brief

    # If the clear had not fired, `propose_plan`'s first-draft call above
    # (stating neither `amends_version` nor `starts_new_plan`) would have
    # been refused as `INVALID_PLAN` against the first turn's still-
    # recorded plan, recording nothing -- so the plan on record would
    # still be the first turn's own, at "kenya debt". It is instead the
    # second turn's own first draft.
    second_plan = load_plan(await harness.state())
    assert second_plan is not None
    assert second_plan.question == "kenya oda 2023"
    assert second_plan.version == 1
    assert second_plan.previous_version is None


async def test_a_sixth_amendment_turn_still_records_its_own_version(
    decision: Any, requirement: Any, derived_requirement: Any
) -> None:
    """Reproduces the production sequence: six separate amendment turns on
    the same question, each its own `PlannerSeam.__call__`. `__call__`
    resets the counter at the top of every fresh turn (never on a resume --
    see the resume test below), so every turn gets its own four-round
    budget and a sixth amendment still lands as its own new version instead
    of silently repeating the fourth.

    No turn here ever reaches `submit_plan` (the script only ever proposes),
    so every turn ends the same way, in `PlannerSeamError` for want of a
    submission -- the version `propose_plan` recorded along the way, not
    the raise, is what this test is about.

    Each turn's one proposal amends the version the last turn recorded
    (`_amend_step`), read off that call's own answer, so the plan walks
    1-2-3-4-5-6 across the six turns rather than any turn resetting it at
    version 1.
    """
    harness = _SeamHarness()
    seam = _seam()
    proposal = _amendment_proposal(decision, requirement, derived_requirement)

    # The version the recorded plan stands at as each turn opens. Turn 1
    # opens on a session with no plan recorded, so its default goes
    # unconsulted: that call is the first draft, recorded at version 1.
    version = 1
    for turn in range(1, 7):
        step = _amend_step(proposal, start_version=version)
        seam._agent.model = ScriptedPlannerLlm(script=[step])
        with pytest.raises(PlannerSeamError):
            await harness.run(seam, question="kenya debt", envelope={}, amendment=f"turn {turn}")
        plan = load_plan(await harness.state())
        assert plan is not None
        version = plan.version

    assert version == 6


async def test_the_full_brief_a_second_call_inherits_is_byte_identical_to_the_first() -> None:
    """The fresh/continuing split rests on one assumption: `question`,
    `envelope`, `canvas_digest` and `conversation_digest` never change
    mid-turn, so a continuing call can safely drop them because the
    session's own history already carries exactly what a fresh call would
    have rendered.

    This drives two `PlannerSeam.__call__`s through the *same* `ctx`,
    inside one node function -- the shape an `amend`/`replan` edge actually
    produces in the real graph, one continuous `Runner.run_async` for the
    whole turn, `planner_n` re-entered as an internal node transition, not
    two separate top-level runs. That distinction matters here:
    `_BRIEFED_STATE_KEY` is a direct, non-delta-tracked write to
    `ctx.session.state` (see the module docstring on why), and such a write
    is only guaranteed visible to a later read within the same
    `Runner.run_async` invocation (`_SeamHarness`'s own `seed_state`
    docstring already says as much for `prior_plan`'s reseed) -- two
    separate `_SeamHarness.run()` calls would not reliably carry it
    forward, and would not exercise what this test is about.

    With the two calls sharing one `ctx`, the second call's own generation
    sees the first call's full brief already in its history (via
    `include_contents="default"`, see the module docstring) -- asserting
    that text is byte-identical to what a fresh render of the *second*
    call's own arguments would produce is exactly the invariant the
    fresh/continuing split rests on: if a future change ever let this seam
    render one of these fields differently between two calls passed the
    same values, this test would catch it before a continuing message went
    stale on the strength of a false assumption.
    """
    question = "kenya debt"
    envelope = {"status": "definite", "spec_id": SPEC, "parts": 1}
    canvas_digest = {"blocks": ["a block"]}
    conversation_digest = {"turns": ["an earlier turn"]}

    seam = _seam()
    first_llm = ScriptedPlannerLlm(script=["still nothing"] * 4)
    second_llm = ScriptedPlannerLlm(script=["still nothing"] * 4)

    expected_first_message = seam._message(
        fresh=True,
        question=question,
        envelope=envelope,
        prior_plan=None,
        refusal=None,
        amendment=None,
        canvas_digest=canvas_digest,
        conversation_digest=conversation_digest,
    )

    async def _node(ctx: Context, node_input: Any) -> Any:
        del node_input
        seam._agent.model = first_llm
        with pytest.raises(PlannerSeamError):
            await seam(
                ctx=ctx,
                question=question,
                envelope=envelope,
                canvas_digest=canvas_digest,
                conversation_digest=conversation_digest,
            )
        seam._agent.model = second_llm
        with pytest.raises(PlannerSeamError):
            await seam(
                ctx=ctx,
                question=question,
                envelope=envelope,
                canvas_digest=canvas_digest,
                conversation_digest=conversation_digest,
                amendment="only since 2015",
            )

    node_fn = node(_node, name="seam_under_test", rerun_on_resume=True)
    wf = Workflow(name="seam_test_wf", edges=[(START, node_fn)])
    sessions = InMemorySessionService()
    runner = Runner(node=wf, app_name=_APP, session_service=sessions)
    session = await sessions.create_session(app_name=_APP, user_id=_USER)
    content = types.Content(role="user", parts=[types.Part(text="probe")])
    async for event in runner.run_async(user_id=_USER, session_id=session.id, new_message=content):
        assert isinstance(event, Event)

    seen_user_texts = [
        part.text
        for request_content in second_llm.captured_requests[0].contents
        if request_content.role == "user"
        for part in request_content.parts or []
        if part.text is not None
    ]
    assert expected_first_message in seen_user_texts


async def test_a_turn_that_loops_past_the_round_cap_is_refused_not_raised(
    decision: Any, requirement: Any, derived_requirement: Any
) -> None:
    """`MAX_PLAN_ROUNDS` bounds the cumulative `propose_plan` calls made
    *within* one turn, since the model can call it more than once inside a
    single `_propose`/`ctx.run_node` call and nothing but the round cap counts
    that. This scripts five `propose_plan` calls in one continuous planner
    turn (one `__call__`, one `_propose`, no correction round and no second
    turn), so the fifth is the one that goes past the cap. The seam returns a
    refusal and does not raise `PlannerSeamError`: raising it would end the
    turn through `_stream_fatal`'s `close_turn(error=...)`, which `begin_turn`
    documents as abandoning the paused session and its plan, so the last
    recorded version could no longer be approved.

    Every call after the first amends what the call before it recorded
    (`_amend_step`), so the rounds actually accrue: the first drafts
    version 1, the next three amend 1, 2 and 3, and the fifth, amending
    version 4, is the one past the cap.
    """
    seam = _seam()
    proposal = _amendment_proposal(decision, requirement, derived_requirement)
    step = _amend_step(proposal, start_version=1)
    seam._agent.model = ScriptedPlannerLlm(
        script=[[("propose_plan", proposal)], step, step, step, step]
    )

    result = await _run_seam(seam, question="kenya debt", envelope={})

    assert "refused" in result
    text = result["refused"]["text"]
    assert f"budget of {MAX_PLAN_ROUNDS} plan drafts" in text
    assert "fresh budget" in text


async def test_a_refused_loop_leaves_the_last_recorded_version_on_record(
    decision: Any, requirement: Any, derived_requirement: Any
) -> None:
    """The fifth `propose_plan` call's own refusal writes nothing
    (`planner.agent.propose_plan` returns its `problem(...)` before ever
    calling `save_plan`), so whatever the fourth call actually recorded
    stays exactly as it was -- proven here directly against session state
    rather than inferred from the refusal text.

    The four calls before it amend what each recorded (`_amend_step`), so
    the plan on record is the fourth: version 4, still a draft.
    """
    harness = _SeamHarness()
    seam = _seam()
    proposal = _amendment_proposal(decision, requirement, derived_requirement)
    step = _amend_step(proposal, start_version=1)
    seam._agent.model = ScriptedPlannerLlm(
        script=[[("propose_plan", proposal)], step, step, step, step]
    )

    result = await harness.run(seam, question="kenya debt", envelope={})

    assert result is not None and "refused" in result
    plan = load_plan(await harness.state())
    assert plan is not None
    assert plan.version == 4
    assert plan.status is PlanStatus.DRAFT


async def test_the_round_budget_is_not_reset_on_a_resume_of_the_same_turn(
    decision: Any, requirement: Any, derived_requirement: Any
) -> None:
    """A mid-turn resume re-enters `__call__` with `ctx.resume_inputs`
    carrying the interrupt's own answer. Driven through a real planner ask
    and a real `create_request_input_response` answer -- not the hand-set
    `ctx._resume_inputs` this test used before `adk_request_input` pausing
    existed (see `_SeamHarness.run`'s own comment on why that condition is
    genuine, confirmed against the installed ADK's `_reconstruct_node_states`)
    -- so this is the same boundary a live pause-and-resume crosses, not a
    double standing in for one.

    Three rounds are spent for real, through `propose_plan`'s own
    persisted write, in the same continuous planner turn that then asks
    and pauses. They are not seeded directly onto the stored session: a
    direct, non-delta-tracked write (what an ungated reset itself would
    do) proves nothing about whether the *reset* fired, only about what
    was written after it. Scripting a resumed turn that calls
    `propose_plan` twice more reads the count the way `propose_plan`
    itself does, through its own persisted write, which is the only
    reliable way to observe it. The fifth call is refused only if the
    three rounds and the fourth survived the resume. An ungated reset
    would land the fifth call within a fresh 1-2 budget, succeeding
    rather than refusing at all.
    """
    harness = _SeamHarness()
    seam = _seam()
    proposal = _amendment_proposal(decision, requirement, derived_requirement)

    # One continuous planner turn: three real `propose_plan` rounds (no
    # `prior_plan` seeded, so the first call starts a brand-new plan and
    # each later one amends the version the call before it durably recorded,
    # through `_amend_step` -- `test_a_sixth_amendment_turn_still_records_
    # its_own_version`'s same pattern), then an ask that pauses mid-turn.
    step = _amend_step(proposal, start_version=1)
    seam._agent.model = ScriptedPlannerLlm(
        script=[
            [("propose_plan", proposal)],
            step,
            step,
            [
                (
                    "adk_request_input",
                    {
                        "message": "Which source did you mean?",
                        "candidates": [{"spec_id": SPEC, "label": "a candidate"}],
                        "recommended_spec_id": SPEC,
                        "rationale": "broader coverage",
                    },
                )
            ],
        ]
    )
    paused = await harness.run(seam, question="kenya debt", envelope={})
    assert paused is None, "the ask should pause this call, not resolve it"

    # The resumed entry: `ctx.resume_inputs` carries the interrupt's own
    # answer, populated by ADK's own resume machinery, not set by hand.
    # `propose_plan` runs twice more -- the fourth and fifth cumulative
    # rounds if the three spent before the ask survived, only the first and
    # second if an ungated reset wiped them. Gated correctly, the fourth is
    # still within the cap and the fifth goes past it. Both calls amend the
    # version the session's own call history last answered with, which is
    # version 3 when the three rounds survived this resume.
    interrupt_id = _pending_ask_interrupt_id(await harness.events())
    resume_message = types.Content(
        role="user",
        parts=[create_request_input_response(interrupt_id, {"result": {"spec_id": SPEC}})],
    )
    seam._agent.model = ScriptedPlannerLlm(script=[step, step])
    result = await harness.run(
        seam,
        question="kenya debt",
        envelope={},
        amendment="one more change",
        new_message=resume_message,
    )

    assert result is not None and "refused" in result
    text = result["refused"]["text"]
    assert f"budget of {MAX_PLAN_ROUNDS} plan drafts" in text
    assert "fresh budget" in text


# -- a planner entry's own wall-clock deadline ----------------------------


class _ScriptedClock:
    """A fake clock returning `values[i]` on its `i`-th call, and its last
    value for every call past the end -- a test's own way of putting a
    wall-clock deadline exactly where a scenario needs it, with no real
    sleep and no dependence on exactly how many `PlannerSeam._clock()` calls
    a scenario happens to make along the way.
    """

    def __init__(self, values: list[float]) -> None:
        assert values, "a scripted clock needs at least one value"
        self._values = values
        self.calls = 0

    def __call__(self) -> float:
        value = self._values[min(self.calls, len(self._values) - 1)]
        self.calls += 1
        return value


def test_pinnable_candidates_filters_to_spec_id_bearing_entries() -> None:
    """A `no_data` referral's own candidates (`translator.discovery.
    compact_source(selectable=False)`) carry no `spec_id`; the same filter
    `_stalled_referral`'s own `named` text applies drops them here too."""
    referral = {
        "kind": "source_ambiguous",
        "candidates": [
            {"spec_id": SPEC, "sentence": "aid from France"},
            {"sentence": "a nearby series with no spec_id"},
            {"spec_id": SPEC_B, "sentence": "aid to Togo"},
        ],
    }

    assert _pinnable_candidates(referral) == [
        {"spec_id": SPEC, "label": "aid from France"},
        {"spec_id": SPEC_B, "label": "aid to Togo"},
    ]


def test_pinnable_candidates_is_empty_for_no_referral_or_no_candidates() -> None:
    assert _pinnable_candidates(None) == []
    assert _pinnable_candidates({"kind": "no_operation_covers"}) == []
    assert _pinnable_candidates({"kind": "no_data", "candidates": "not a list"}) == []


def test_clarifying_question_args_offers_the_candidates_and_recommends_none() -> None:
    candidates = [{"spec_id": SPEC, "label": "a"}, {"spec_id": SPEC_B, "label": "b"}]

    args = _clarifying_question_args(candidates)

    assert args == {"message": args["message"], "candidates": candidates}
    # The message reaches the user: the candidates' labels name the sources.
    assert "spec_" not in args["message"]


def test_requirements_without_discovery_names_every_requirement_with_no_workflow(
    decision: Any, requirement: Any, derived_requirement: Any
) -> None:
    plan = _gdp_plan(decision, requirement, derived_requirement)
    state: SessionState = {}
    save_plan(state, plan)

    assert set(_requirements_without_discovery(state)) == {"gdp_gtm", "annual_totals"}


def test_requirements_without_discovery_names_only_the_unbound_ones(
    decision: Any, requirement: Any, derived_requirement: Any
) -> None:
    """`_translated_plan`'s own workflow binds only the dataset requirement
    -- the derived one is computed from it, not independently discovered --
    so only the derived requirement's id is still unmet."""
    plan = _translated_plan(decision, requirement, derived_requirement)
    state: SessionState = {}
    save_plan(state, plan)

    assert _requirements_without_discovery(state) == ["annual_totals"]


def test_requirements_without_discovery_is_empty_with_no_plan_on_record() -> None:
    assert _requirements_without_discovery({}) == []


def test_refusal_text_routes_a_deadline_code_to_the_deadline_text() -> None:
    refused = {"code": _DEADLINE_REFUSAL_CODE, "detail": {"requirement_ids": ["req_a"]}}

    text = PlannerSeam._refusal_text(refused)

    assert "req_a" in text
    assert "ran out of time" in text
    assert "fresh budget" in text


def test_deadline_text_says_nothing_was_recorded_true_on_a_first_draft_too() -> None:
    """A first-draft entry has no prior plan for anything to be a change
    to, so the text says nothing was recorded rather than naming a change."""
    refused = {"code": _DEADLINE_REFUSAL_CODE, "detail": {"requirement_ids": ["req_a"]}}

    text = PlannerSeam._deadline_text(refused)

    assert "nothing was recorded" in text
    assert "this change" not in text


def test_deadline_text_agrees_its_pronoun_with_several_requirement_ids() -> None:
    """ "requirement(s) X, Y had a source discovered for it" mismatches
    number when more than one id is named; several ids take "them"."""
    refused = {
        "code": _DEADLINE_REFUSAL_CODE,
        "detail": {"requirement_ids": ["req_a", "req_b"]},
    }

    text = PlannerSeam._deadline_text(refused)

    assert "req_a, req_b had a source discovered for them" in text
    assert "for it" not in text


def test_deadline_text_keeps_the_singular_pronoun_for_one_requirement_id() -> None:
    refused = {"code": _DEADLINE_REFUSAL_CODE, "detail": {"requirement_ids": ["req_a"]}}

    text = PlannerSeam._deadline_text(refused)

    assert "req_a had a source discovered for it" in text


async def test_a_deadline_that_has_already_passed_refuses_before_any_generation() -> None:
    """The deadline is checked before the planner's very first generation of
    an entry too -- `_before_planner_model` fires ahead of every generation,
    the opening one included -- so a deadline that has already passed by
    then never lets the model run at all, and the refusal names no
    requirement, since no draft was ever proposed."""
    clock = _ScriptedClock([0.0, 10_000.0])
    seam = _seam(agent_settings=AgentSettings(planner_entry_deadline_seconds=100.0), clock=clock)
    llm = ScriptedPlannerLlm(script=[[("propose_plan", {})]])
    seam._agent.model = llm

    result = await _run_seam(seam, question="kenya debt", envelope={})

    assert result is not None and "refused" in result
    text = result["refused"]["text"]
    assert "ran out of time" in text
    assert "this plan had a source discovered" in text
    assert "fresh budget" in text
    assert llm.captured_requests == []


async def test_discovery_inside_an_entry_gets_the_deadline_less_the_next_model_calls_margin() -> (
    None
):
    """Inside the run a discovery call reads the entry's remaining time less
    the margin its next model call needs; outside it, none is set."""
    seen: list[float | None] = []

    def _read_time_left(_request: Any) -> str:
        seen.append(discovery_time_left())
        return "done"

    seam = _seam(
        agent_settings=AgentSettings(planner_entry_deadline_seconds=100.0), clock=lambda: 5.0
    )
    seam._agent.model = ScriptedPlannerLlm(script=[_read_time_left] * (_MAX_CORRECTIONS + 1))

    with pytest.raises(PlannerSeamError):
        await _run_seam(seam, question="kenya debt", envelope={})

    assert seen
    assert set(seen) == {100.0 - _NEXT_MODEL_CALL_MARGIN_S}
    assert discovery_time_left() is None


async def test_a_deadline_hit_after_a_draft_survives_a_correction_round_and_names_the_unmet_requirements(
    decision: Any, requirement: Any, derived_requirement: Any
) -> None:
    """A deadline that runs out on a draft with no referral refuses,
    naming the requirement(s) with no `Workflow.evidence_bindings` entry yet
    -- never a correction round's own retry, and never the untranslated
    draft itself.

    The clock is scripted so `propose_plan`, the blocked `submit_plan`
    (`WORKFLOW_MISSING`) and the closing text that ends that call --
    `test_a_stopped_run_names_the_problem_it_stopped_on`'s own shape -- each
    pass their own check. Only the correction round's own first check trips
    it. That check reads the same `deadline_at` `__call__` anchored once,
    before either `_propose` call, not a fresh one an incorrectly reset
    deadline would instead compute from this check's own (much later) clock
    value -- which would still be comfortably inside a 100s budget, and let
    the correction round's own model run.
    """
    made_decision = decision()
    requirements = [requirement(), derived_requirement()]
    proposal = {
        "question": "kenya debt",
        "answer_mode": "computed",
        "decisions": [made_decision.model_dump(mode="json")],
        "evidence_requirements": [r.model_dump(mode="json") for r in requirements],
    }
    clock = _ScriptedClock([0.0, 0.0, 0.0, 0.0, 10_000.0])
    seam = _seam(agent_settings=AgentSettings(planner_entry_deadline_seconds=100.0), clock=clock)
    llm = ScriptedPlannerLlm(
        script=[[("propose_plan", proposal)], [("submit_plan", {"version": 1})]]
    )
    seam._agent.model = llm

    result = await _run_seam(seam, question="kenya debt", envelope={})

    assert result is not None and "refused" in result
    text = result["refused"]["text"]
    assert "annual_debt_service" in text
    assert "annual_totals" in text
    assert "ran out of time" in text
    # The three generations the first `_propose` call itself makes
    # (`propose_plan`, the blocked `submit_plan`, the closing "done" text)
    # -- the correction round's own generation never happens.
    assert len(llm.captured_requests) == 3


async def test_a_deadline_hit_on_a_referral_with_candidates_asks_the_user_and_the_answer_elects(
    decision: Any, requirement: Any, derived_requirement: Any
) -> None:
    """A deadline that runs out on a referral carrying
    pinnable candidates ends the entry the same way the planner's own
    `ask_user` does -- a real, long-running `adk_request_input` pause, built
    here from the referral's own candidates rather than composed by a model
    that has just run out of time to compose one. The user's answer resumes
    the run and pairs by id (`planner.agent._elected_spec_ids`), the same
    way a model-issued ask's own answer does.

    The clock is scripted so the deadline has not yet passed when the
    planner's first generation calls `translate` (the real mechanical loop
    runs against `_TWO_CANDIDATES_SEARCH_RESPONSE` and records a genuine
    `source_ambiguous` referral naming `SPEC`/`SPEC_B`, the same fixture
    `test_the_planners_translate_response_is_the_referral_with_its_outcome`
    uses), and has passed by the very next check -- right after `translate`
    returns, before the planner's own
    model is ever asked to react to the referral it carries.
    """
    draft = Plan(
        question="aid to guatemala",
        decisions=[decision()],
        evidence_requirements=[
            requirement(id="gdp_gtm", measure="aid disbursed"),
            derived_requirement(input_id="gdp_gtm"),
        ],
    )
    # The third reading is the search's own (`planner.plan.discovery_time_left`).
    clock = _ScriptedClock([0.0, 0.0, 0.0, 10_000.0])
    seam = _translator_seam(
        responses={"search": _TWO_CANDIDATES_SEARCH_RESPONSE},
        agent_settings=AgentSettings(planner_entry_deadline_seconds=100.0),
        clock=clock,
    )
    planner_llm = ScriptedPlannerLlm(script=[[("translate", {})]])
    seam._agent.model = planner_llm
    harness = _SeamHarness()

    paused = await harness.run(
        seam, question=draft.question, envelope={}, prior_plan=draft.model_dump(mode="json")
    )
    assert paused is None, "the deadline's own clarifying question should pause this call"
    # One generation: the one that calls `translate`. The deadline
    # intercepts before the planner's own reaction to the referral it
    # returns with, so a second generation is never asked for.
    assert len(planner_llm.captured_requests) == 1

    events = await harness.events()
    interrupt_id = _pending_ask_interrupt_id(events)
    (ask_call,) = [
        part.function_call
        for event in events
        for part in (event.content.parts if event.content and event.content.parts else [])
        if part.function_call is not None and part.function_call.id == interrupt_id
    ]
    assert {c["spec_id"] for c in ask_call.args["candidates"]} == {SPEC, SPEC_B}
    assert "recommended_spec_id" not in ask_call.args
    assert "gdp_gtm" not in ask_call.args["message"]

    resume_message = types.Content(
        role="user",
        parts=[create_request_input_response(interrupt_id, {"result": {"spec_id": SPEC_B}})],
    )
    planner_llm.script = [["noted, but not done yet"]]
    with pytest.raises(PlannerSeamError):
        await harness.run(
            seam,
            question=draft.question,
            envelope={},
            prior_plan=draft.model_dump(mode="json"),
            new_message=resume_message,
        )

    fake_tool_context = SimpleNamespace(session=SimpleNamespace(events=await harness.events()))
    assert _elected_spec_ids(fake_tool_context) == frozenset({SPEC_B})


async def test_a_resumed_entry_gets_a_fresh_deadline_not_the_paused_ones(
    decision: Any, requirement: Any, derived_requirement: Any
) -> None:
    """A resume after a user answer re-enters `__call__`, which anchors a
    fresh `deadline_at` there -- not the paused entry's own, already most of
    the way through its own budget by the time the answer arrives.

    The clock's values put the resumed entry's own check comfortably before
    a *fresh* deadline (`100.5 < 100 + 5`) but past the *paused* entry's
    (`100.5 >= 0 + 5`): an implementation that carried the old `deadline_at`
    over instead of anchoring a new one would refuse here, where this one
    finishes the turn.
    """
    plan = _translated_plan(decision, requirement, derived_requirement)
    clock = _ScriptedClock([0.0, 0.0, 100.0, 100.5])
    seam = PlannerSeam(
        settings=Settings(),
        surface=FixtureSurface(schemas=[]),
        agent_settings=AgentSettings(planner_entry_deadline_seconds=5.0),
        clock=clock,
    )
    llm = ScriptedPlannerLlm(
        script=[
            [
                (
                    "adk_request_input",
                    {
                        "message": "Which source did you mean?",
                        "candidates": [{"spec_id": SPEC, "label": "a candidate"}],
                        "recommended_spec_id": SPEC,
                        "rationale": "broader coverage",
                    },
                )
            ],
            [("submit_plan", {"version": plan.version})],
        ]
    )
    seam._agent.model = llm
    harness = _SeamHarness()

    paused = await harness.run(
        seam, question=plan.question, envelope={}, prior_plan=plan.model_dump(mode="json")
    )
    assert paused is None, "the ask should pause this call, not resolve it"

    interrupt_id = _pending_ask_interrupt_id(await harness.events())
    resume_message = types.Content(
        role="user",
        parts=[create_request_input_response(interrupt_id, {"result": {"spec_id": SPEC}})],
    )

    resumed = await harness.run(
        seam,
        question=plan.question,
        envelope={},
        prior_plan=plan.model_dump(mode="json"),
        new_message=resume_message,
    )

    assert resumed is not None, "a fresh deadline should let the resumed entry finish"
    assert resumed["plan"]["version"] == plan.version


async def test_a_submitted_plan_ends_the_planner_run_with_no_further_model_call(
    decision: Any, requirement: Any, derived_requirement: Any
) -> None:
    """`submit_plan`'s ok answer is the planner's last event: ADK asks the
    model for nothing after it, so a deadline that has passed by then has no
    generation to refuse and the submission stands.
    """
    plan = _translated_plan(decision, requirement, derived_requirement)
    clock = _ScriptedClock([0.0, 0.0, 10_000.0])
    seam = _seam(agent_settings=AgentSettings(planner_entry_deadline_seconds=100.0), clock=clock)
    llm = ScriptedPlannerLlm(script=[[("submit_plan", {"version": plan.version})]])
    seam._agent.model = llm

    result = await _run_seam(
        seam, question=plan.question, envelope={}, prior_plan=plan.model_dump(mode="json")
    )

    assert len(llm.captured_requests) == 1
    assert result is not None and result["plan"]["version"] == plan.version


async def test_a_sampled_turn_does_not_refuse_at_a_budget_the_unsampled_one_would(
    decision: Any, requirement: Any, derived_requirement: Any
) -> None:
    """A sampled turn runs the planner at the model's own default thinking
    level, deliberately slower than the ordinary `LOW` turn
    `planner_entry_deadline_seconds` is sized for -- so a sampled entry gets
    `planner_entry_deadline_seconds_sampled` instead, and a check at a wall
    time past the ordinary (unsampled) budget but still inside the sampled
    one must not refuse."""
    plan = _translated_plan(decision, requirement, derived_requirement)
    clock = _ScriptedClock([0.0, 150.0, 10_000.0])
    seam = _seam(
        agent_settings=AgentSettings(
            planner_entry_deadline_seconds=100.0,
            planner_entry_deadline_seconds_sampled=200.0,
            thinking_sample_rate=1.0,
        ),
        clock=clock,
        thinking_sample_rng=lambda: 0.0,
    )
    seam._agent.model = ScriptedPlannerLlm(script=[[("submit_plan", {"version": plan.version})]])

    result = await _run_seam(
        seam, question=plan.question, envelope={}, prior_plan=plan.model_dump(mode="json")
    )

    assert result is not None and "plan" in result, (
        "a sampled turn's own larger deadline should have covered a check at "
        "t=150, which the 100s unsampled deadline would have refused"
    )
    assert result["plan"]["version"] == plan.version


async def test_an_unsampled_turn_still_uses_the_ordinary_deadline() -> None:
    """The same wall time that a sampled turn's larger budget clears still
    refuses an unsampled turn: `thinking_sample_rate=0.0` never samples,
    regardless of the draw, so this entry keeps
    `planner_entry_deadline_seconds` even with a larger sampled deadline
    configured alongside it."""
    clock = _ScriptedClock([0.0, 150.0])
    seam = _seam(
        agent_settings=AgentSettings(
            planner_entry_deadline_seconds=100.0,
            planner_entry_deadline_seconds_sampled=200.0,
            thinking_sample_rate=0.0,
        ),
        clock=clock,
    )
    llm = ScriptedPlannerLlm(script=[[("propose_plan", {})]])
    seam._agent.model = llm

    result = await _run_seam(seam, question="kenya debt", envelope={})

    assert result is not None and "refused" in result
    text = result["refused"]["text"]
    assert "ran out of time" in text
    assert llm.captured_requests == []

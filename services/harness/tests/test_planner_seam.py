"""Tests for `harness.planner_seam` -- the adapter over `planner.agent`
with translation attached to its `propose_plan`.

Model-free: a scripted `BaseLlm` stands in for the planner's own model. Most
scenarios either hand the seam an already-compiled `Plan` (built the way
`tests/planner/test_agent.py`'s `_translate` helper builds one, through the
real `Plan.with_translation`) and give the planner the `propose_plan`
`harness.scripted.use_compiled_proposals` installs, so the scripted model's
one call declares it ready, or leave the plan a draft. The referral-stall tests below
script a bare `propose_plan` amendment, whose translation runs the real
translation loop against a canned surface. Whether the real model plans WELL is the eval sets' question
(`evals/README.md`).

`PlannerSeam` runs the planner through `ctx.run_node`, in the graph's own
session, so every test below needs a real `Context`, not a bare call.
`_run_seam` drives one, through a minimal one-node `Workflow`, the same
primitive `harness.graph`'s own planner node uses.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from compiler.surface import FixtureSurface, ToolCallResult, ToolSurfaceUnavailableError
from data_mcp.definition_store import canonicalize, spec_id_for
from data_mcp.definition_writes import windowed_identity
from data_mcp.search_result import SEARCH_ENVELOPE_META_KEY
from google.adk import Context, Event
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.tool_context import ToolContext
from google.adk.workflow import START, Workflow, node
from google.adk.workflow.utils._workflow_hitl_utils import (
    REQUEST_INPUT_FUNCTION_CALL_NAME,
    create_request_input_response,
)
from google.genai import types
from harness import planner_seam
from harness.graph.state import (
    ORCHESTRATOR_PRIOR_LEDGER_KEY,
    ORCHESTRATOR_SPEC_WARNINGS_KEY,
    ORCHESTRATOR_SPECS_KEY,
)
from harness.planner_seam import (
    _PLANNER_BRANCH,
    _SHARE_UNIT_CODES,
    _WHICH_INSTEAD,
    _WHICH_MEANT,
    PlannerSeam,
    PlannerSeamError,
    _check_branches,
    _clarifying_question_args,
    _pinnable_candidates,
    _plan_context,
    _RunTracker,
)
from harness.scripted import use_compiled_proposals
from harness.sentences import SeamCode, referral_sentence, refusal_sentence
from harness.settings import Settings
from harness.studio_state import ConversationState
from harness.tool_contracts import reading_facts, resolver_digest
from harness.turn_budget import CLOSED_BY_KEY, TurnBudget, bounded, current_turn_budget
from harness.turn_records import SearchReading
from harness.wiring import handing_back_planner_failures
from planner.agent import _elected_spec_ids
from planner.operations import AggregateOperation, RetrieveOperation, Translation
from planner.plan import (
    ASK_BREAKDOWNS_STATE_KEY,
    ASKS_STATE_KEY,
    CUT_MARKER,
    MAX_PLAN_ROUNDS,
    MAX_QUESTION_CHARS,
    OPTIONAL_DECLINES_STATE_KEY,
    UNSUPPORTED_STATE_KEY,
    AggregateRequirement,
    AskBreakdown,
    DatasetRequirement,
    Decision,
    Entity,
    FullHistory,
    GrowthRequirement,
    JoinRequirement,
    Period,
    PinSource,
    Plan,
    PlanStatus,
    ResolvedBy,
    SessionState,
    Unit,
    current_discovery_time_left,
    current_earlier_block,
    discovery_time_left,
    full_histories,
    load_plan,
    pin_sources,
    plan_revision,
    rolls_from_its_start,
)
from planner.records import GroupBy, Reduction, RelativePeriod
from planner.settings import (
    AGENT_THINKING_CONFIG,
    AgentSettings,
    drop_thought_parts,
)
from planner.tool_results import ok
from planner.workflow import EvidenceBinding, SelectedSource, ToolId, WorkflowStep
from planner.workflow import Workflow as PlanWorkflow
from pydantic import Field
from translator.discovery import current_cut_calls
from translator.mechanical import COMPILE_FAILURES_STATE_KEY

from .search_results import search_result_parts

SPEC = "spec_" + "a" * 32
SPEC_B = "spec_" + "b" * 32


# The call a scripted planner makes to have the plan on record declared ready,
# through the `propose_plan` `use_compiled_proposals` installs.
_DECLARE = [("propose_plan", {})]

_APP = "planner_seam_test"
_USER = "u"


class ScriptedPlannerLlm(BaseLlm):
    """Answers each generation from `script`, in order, and records the requests.

    A script entry is either a list of `(tool_name, args)` function calls, a
    string of final text, or a callable that reads the `LlmRequest` and
    returns one of the other two; ADK asks for one more generation after
    every tool response but a compiled proposal declared ready, and a text answer is
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
) -> PlannerSeam:
    return PlannerSeam(
        settings=Settings(),
        surface=FixtureSurface(schemas=[]),
        agent_settings=agent_settings,
    )


class _SeamHarness:
    """Drives `PlannerSeam.__call__` through a real, dynamically-scheduled
    `Context` -- the same primitive `harness.graph`'s own planner node
    drives `PlannerSeam` through, and the only way to get a `Context`
    `ctx.run_node` will accept, since this seam has no `Runner` of its
    own to call directly.

    One `_SeamHarness` reuses the same session across every `run` call,
    the way the graph's own session persists across a turn's several
    entries and across separate turns of one conversation alike -- so a
    test can inspect `ctx.state` (`state()`) after a call, without losing
    what it left behind.
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
    transition -- the minimal shape a graph node and `declare_ready` alike
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
    use_compiled_proposals(seam._agent)
    seam._agent.model = ScriptedPlannerLlm(script=[_DECLARE])

    drafted = await _run_seam(
        seam, question=plan.question, readings=[], prior_plan=plan.model_dump(mode="json")
    )

    # The node's own validation (`Plan.model_validate(drafted["plan"])` in
    # `harness.graph.planning`'s `planner` node), repeated here as the contract this seam owes it.
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
    use_compiled_proposals(seam._agent)
    seam._agent.model = ScriptedPlannerLlm(script=[_DECLARE])
    reports: list[dict[str, Any]] = []

    await _run_seam(
        seam,
        question=plan.question,
        readings=[],
        prior_plan=plan.model_dump(mode="json"),
        on_progress=reports.append,
    )

    assert [(report["agent"], report["tools"]) for report in reports] == [
        ("dfo_planner", ["propose_plan"])
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
    # A draft with no translation is never declared ready, so this call
    # ends with no plan -- it exists only to leave a version-1 plan in the
    # session, to prove the second call does not read it back.
    await harness.run(seam, question="kenya debt", readings=[])
    left_behind = load_plan(await harness.state())
    assert left_behind is not None
    assert left_behind.version == 1

    seeded = _translated_plan(decision, requirement, derived_requirement, version=5)
    seam._agent.model = ScriptedPlannerLlm(
        script=[[("propose_plan", {**proposal, "amends_version": 5})]]
    )
    await harness.run(
        seam,
        question="kenya debt",
        readings=[],
        prior_plan=seeded.model_dump(mode="json"),
        amendment="only since 2015",
    )

    amended = load_plan(await harness.state())
    assert amended is not None
    assert amended.version == 6
    assert amended.previous_version == 5


async def test_a_model_that_stops_without_a_plan_ends_the_entry_on_plan_not_ready() -> None:
    """A run that ends with no plan declared ready and no referral on record
    ends the entry at once, on a refusal another plan may still answer."""
    seam = _seam()
    llm = ScriptedPlannerLlm(script=["Which years did you have in mind?"] * 4)
    seam._agent.model = llm

    output = await _run_seam(seam, question="kenya debt", readings=[])

    assert output["refused"] == {
        "text": refusal_sentence({"code": SeamCode.PLAN_NOT_READY}),
        "final": False,
        "code": SeamCode.PLAN_NOT_READY,
    }
    assert len(llm.captured_requests) == 1


class _FailingSkillSurface(FixtureSurface):
    """A data-mcp surface whose every call raises an exception `get_skill`
    does not anticipate, the way a bug below a planner tool would."""

    async def call_tool(self, **kwargs: Any) -> ToolCallResult:
        del kwargs
        raise RuntimeError("http://internal.example/skills exploded")


async def test_a_planner_tool_that_raises_ends_the_entry_on_a_final_refusal(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The run stops at the failed call and the orchestrator gets a final
    refusal naming the tool, with the traceback logged at ERROR."""
    seam = PlannerSeam(settings=Settings(), surface=_FailingSkillSurface(schemas=[]))
    llm = ScriptedPlannerLlm(script=[[("get_skill", {"skill_id": "aid"})], "unreachable"])
    seam._agent.model = llm

    harness = _SeamHarness()

    with caplog.at_level(logging.ERROR, logger=planner_seam.logger.name):
        output = await harness.run(
            cast(PlannerSeam, handing_back_planner_failures(seam)),
            question="kenya debt",
            readings=[],
        )

    assert output is not None
    assert output["refused"]["final"] is True
    assert "get_skill" in output["refused"]["text"]
    assert len(llm.captured_requests) == 1
    (response,) = [
        part.function_response.response
        for event in await harness.events()
        for part in (event.content.parts if event.content else None) or []
        if part.function_response is not None and part.function_response.name == "get_skill"
    ]
    assert "internal.example" not in json.dumps(response)
    (record,) = [
        r
        for r in caplog.records
        if r.name == planner_seam.logger.name and r.levelno == logging.ERROR
    ]
    assert record.exc_info is not None and record.exc_info[0] is RuntimeError
    assert record.args is not None and record.args[:2] == ("get_skill", "RuntimeError")


async def test_an_unknown_tool_name_keeps_adks_answer_listing_the_tools(
    caplog: pytest.LogCaptureFixture,
) -> None:
    seam = _seam()
    llm = ScriptedPlannerLlm(script=[[("no_such_tool", {})]])
    seam._agent.model = llm

    with caplog.at_level(logging.ERROR, logger=planner_seam.logger.name):
        output = await _run_seam(seam, question="kenya debt", readings=[])

    assert output["refused"]["final"] is False

    (response,) = [
        part.function_response.response
        for content in llm.captured_requests[1].contents
        for part in content.parts or []
        if part.function_response is not None and part.function_response.name == "no_such_tool"
    ]
    assert "The tools you can call are" in response["error"]
    assert not [
        r
        for r in caplog.records
        if r.name == planner_seam.logger.name and r.levelno == logging.ERROR
    ]


@pytest.mark.parametrize("resumed", [False, True])
async def test_an_entry_owes_nothing_an_earlier_entry_recorded(resumed: bool) -> None:
    """A new planner entry starts with no `unsupported` answer to decline, no optional
    decline earned and no compile failure to call repeated; an ask-the-user resume
    inside one entry keeps all three."""
    stale = {
        UNSUPPORTED_STATE_KEY: ["calculation_over_combined"],
        OPTIONAL_DECLINES_STATE_KEY: ["no_operation_covers"],
        COMPILE_FAILURES_STATE_KEY: ["fingerprint"],
    }
    harness = _SeamHarness(state=stale)
    seam = _seam()
    seam._agent.model = ScriptedPlannerLlm(script=["Which years did you have in mind?"] * 4)

    await harness.run(
        seam,
        question="kenya debt",
        readings=[],
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
    generation and no plan."""
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

    result = await harness.run(seam, question="total aid", readings=[])

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

    output = await harness.run(seam, question="total aid", readings=[])

    assert output is not None and output["refused"]["final"] is False

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


def _unit_refusal(code: str, *, tool: str = "compare") -> dict[str, Any]:
    """A `MethodChoiceNeeded` replan payload (`harness.graph.execute`'s `_replan_refusal`)
    for `tool` refusing a pair of sides on their units."""
    return {
        "kind": "method_choice_needed",
        "step_id": "per_person",
        "tool": tool,
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
        pytest.param(
            _unit_refusal("cross_frame_unit_unbound", tool="sum_of"),
            ("ok", None),
            id="no-operation-sum",
        ),
        pytest.param(
            _unit_refusal("cross_frame_unit_unbound", tool="share_of"),
            ("invalid_input", "decline_unearned"),
            id="share-unbound",
        ),
        pytest.param(
            _unit_refusal("cross_frame_unit_mismatch", tool="share_of"),
            ("invalid_input", "decline_unearned"),
            id="share-mismatch",
        ),
        pytest.param(
            {
                **_unit_refusal("cross_frame_unit_unbound", tool="share_of"),
                "also_stopped": [_unit_refusal("cross_frame_unit_unbound")],
            },
            ("ok", None),
            id="share-with-a-compare-sibling",
        ),
        pytest.param(
            _unit_refusal("cross_frame_facet_mismatch", tool="share_of"),
            ("invalid_input", "decline_unearned"),
            id="share-facet-mismatch",
        ),
        pytest.param(
            _unit_refusal("cross_frame_facet_mismatch"),
            ("invalid_input", "decline_unearned"),
            id="compare-facet-mismatch",
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
    no other output. A `share_of` step refusing its sides on their units earns
    none either: the replan names `quotient_of` as the repair. A step reading one
    measure from two sources earns none, and the replan names one source for it."""
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

    await harness.run(seam, question="GDP per person", readings=[], refusal=refusal)

    answered = [
        part.function_response.response
        for content in llm.captured_requests[-1].contents
        for part in content.parts
        if part.function_response and part.function_response.name == "propose_plan"
    ]
    assert [(r["outcome"], r.get("code")) for r in answered] == [outcome]
    prompt = _prompt_text(llm.captured_requests[0])
    assert ("`no_operation_covers`" in prompt) is (outcome[0] == "ok")
    shares = refusal is not None and refusal["tool"] == "share_of"
    assert ("`quotient_of`" in prompt) is (shares and refusal["code"] in _SHARE_UNIT_CODES)
    facets = refusal is not None and refusal["code"] == "cross_frame_facet_mismatch"
    assert ("give every requirement on that measure the same source" in prompt) is facets
    if outcome[0] != "ok":
        assert "action" not in answered[0]
        assert "answer the ask" not in answered[0]["message"]


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

    output = await harness.run(seam, question="kenya debt", readings=[])

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

    output = await _SeamHarness().run(seam, question="kenya debt", readings=[])

    assert output is None
    assert len(checked) == 1, "the branch check did not run on the run that paused"
    assert {event.author for event in checked[0]} >= {seam._agent.name}


def _referring_proposals(agent: Any, referral: dict[str, Any]) -> None:
    """Replace `agent`'s `propose_plan` with one that answers every proposal
    as recorded, its translation referring `referral` back."""

    def propose_plan(tool_context: ToolContext) -> dict[str, object]:
        del tool_context
        return ok(status="draft", version=1, translation=referral)

    agent.tools = [
        FunctionTool(propose_plan) if getattr(tool, "name", None) == "propose_plan" else tool
        for tool in agent.tools
    ]


@pytest.mark.parametrize(
    ("referral", "candidates"),
    [
        (
            {
                "outcome": "invalid_input",
                "kind": "requirement_unmeetable",
                "source_mismatch": True,
                "requirement_id": "debt",
                "reason": "r",
                "candidates": [
                    {"spec_id": SPEC, "sentence": "debt service from IDS"},
                    {"spec_id": SPEC_B, "sentence": "debt service from the IMF"},
                ],
            },
            [
                {"spec_id": SPEC, "label": "debt service from IDS"},
                {"spec_id": SPEC_B, "label": "debt service from the IMF"},
            ],
        ),
        (
            {
                "outcome": "choice_required",
                "kind": "source_name_ambiguous",
                "requirement_id": "debt",
                "reason": "r",
                "ambiguous_sources": [
                    {
                        "text": "the OECD",
                        "candidates": [
                            {"ref": "oecd/crs", "label": "OECD CRS"},
                            {"ref": "oecd/dac1", "label": "OECD DAC1"},
                        ],
                    }
                ],
            },
            [
                {"spec_id": "oecd/crs", "label": "OECD CRS"},
                {"spec_id": "oecd/dac1", "label": "OECD DAC1"},
            ],
        ),
    ],
    ids=["unmeetable-with-candidates", "source-name-ambiguous"],
)
async def test_a_referral_the_user_decides_ends_the_run_on_the_seam_s_own_ask(
    referral: dict[str, Any], candidates: list[dict[str, str]]
) -> None:
    """The translation refers a requirement back with candidates the user
    picks from, so the seam asks them itself: the run pauses on an
    `adk_request_input` call led by the referral's table sentence, and no
    model turn composes it."""
    seam = _seam()
    _referring_proposals(seam._agent, referral)
    llm = ScriptedPlannerLlm(script=[[("propose_plan", {})], "never sent"])
    seam._agent.model = llm
    harness = _SeamHarness()

    output = await harness.run(seam, question="kenya debt", readings=[])

    assert output is None, "the ask should pause this call, not resolve it"
    assert len(llm.captured_requests) == 1
    asks = [
        part.function_call.args
        for event in await harness.events()
        for part in (event.content.parts if event.content and event.content.parts else [])
        if part.function_call is not None
        and part.function_call.name == REQUEST_INPUT_FUNCTION_CALL_NAME
    ]
    mismatch = referral.get("source_mismatch", False)
    lead = referral_sentence(referral["kind"], source_mismatch=mismatch)
    question = _WHICH_INSTEAD if mismatch else _WHICH_MEANT
    assert asks == [{"message": f"{lead} {question}", "candidates": candidates}]


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
    contract: the amendment ask alone, no `_READINGS_CONTEXT` block.
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
            _DECLARE,
        ]
    )
    use_compiled_proposals(seam._agent)
    seam._agent.model = llm
    harness = _SeamHarness()

    paused = await harness.run(
        seam, question=plan.question, readings=[], prior_plan=plan.model_dump(mode="json")
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
        readings=[],
        prior_plan=plan.model_dump(mode="json"),
        new_message=resume_message,
    )
    assert resumed is not None, "a real answer should let the same turn finish"
    assert resumed["plan"]["version"] == plan.version

    # The next call this session ever makes: a fresh `ctx.run_node`, not a
    # resume of anything -- the `amend` edge's own shape
    # (`approve_n -> planner_n`). `_message`'s own contract says this
    # renders `fresh=False`: the amendment ask alone, never
    # `_READINGS_CONTEXT`'s block, since this session was already briefed
    # once, in the very first call above.
    llm.script = [["noted, but not done yet"]]
    generations_before = len(llm.captured_requests)
    await harness.run(
        seam,
        question=plan.question,
        readings=[],
        prior_plan=resumed["plan"],
        amendment="add a caveat about revisions",
    )

    amend_request = llm.captured_requests[generations_before]
    amend_prompt = _prompt_text(amend_request)
    assert "add a caveat about revisions" in amend_prompt, "the amend call's own ask should land"
    brief_mentions = amend_prompt.count("Context for this question.")
    assert brief_mentions == 0, (
        f"the amend call, on an already-briefed session, should render `fresh=False`'s "
        f"continuing message -- the amendment ask alone, no readings section -- but saw "
        f"{brief_mentions} occurrence(s) of the full brief's own marker text, so the seam "
        "re-sent the full brief because `_BRIEFED_STATE_KEY` did not survive the earlier "
        "ask-and-resume"
    )


# -- a replan's refusal ---------------------------------------------------


def _rediscovery_refusal(spec_id: str) -> dict[str, Any]:
    """A `RediscoveryNeeded` replan payload, the shape `harness.graph.execute`'s
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


def test_a_wave_with_two_refused_specs_names_both() -> None:
    """`_replan_refusal`'s `also_stopped` carries a wave's other stops
    (`executor.run._Run.execute`'s siblings); a `RediscoveryNeeded` among
    them names a second refused spec this seam must treat the same as the
    primary's own: named in the `avoid` sentence, not just the first."""
    seam = _seam()
    refusal = _rediscovery_refusal(SPEC)
    refusal["also_stopped"] = [_rediscovery_refusal(SPEC_B)]

    message = seam._message(
        fresh=False,
        question="malaria grants and population for kenya",
        prior_plan=None,
        refusal=refusal,
        amendment=None,
        canvas_digest=None,
        conversation_digest=None,
    )

    avoid = message.split("Call propose_plan again with a plan that avoids it.")[1]
    assert SPEC in avoid and SPEC_B in avoid
    assert "again" in avoid and "refused" in avoid


def test_a_replan_names_the_asks_the_failed_run_computed() -> None:
    """`computed_asks` (`harness.graph.execute._computed_asks`) becomes the sentence
    the replan opens with, and stays out of the refusal it quotes."""
    refusal = {**_rediscovery_refusal(SPEC), "computed_asks": [0, 2]}

    message = _seam()._message(
        fresh=False,
        question="malaria grants for kenya",
        prior_plan=None,
        refusal=refusal,
        amendment=None,
        canvas_digest=None,
        conversation_digest=None,
    )

    assert "failed after it computed asks 0, 2." in message
    assert "computed_asks" not in message


# -- the conversation's search readings ------------------------------------

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
                "asked_roles": ["donor", "recipient"],
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
    llm = ScriptedPlannerLlm(script=[_DECLARE])
    use_compiled_proposals(seam._agent)
    seam._agent.model = llm
    harness = _SeamHarness(stored)
    await harness.run(
        seam,
        question=plan.question,
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
    warning codes `search` stored for it, and the parties its settling reading asked
    for. A spec the executor refused is left out."""
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
        SPEC: PinSource(
            block={"spec_id": SPEC},
            warnings=("role_scope_defaulted",),
            settled=True,
            asked_roles=("donor", "recipient"),
        ),
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


def test_the_plan_section_names_each_other_way_of_counting_a_source_with_its_spec_id(
    decision: Any, requirement: Any, derived_requirement: Any
) -> None:
    """An amendment asking for net disbursements pins the spec_id named here,
    unless the search readings above already name it."""
    net = "spec_" + "e" * 32
    dumped = _translated_plan(decision, requirement, derived_requirement).model_dump(mode="json")
    dumped["workflow"]["sources"] = [
        {
            "spec_id": SPEC,
            "sentence": "official development assistance",
            "elected_measure": {
                "label": "gross disbursements",
                "alternatives": [
                    {"spec_id": net, "label": "net disbursements", "axis": "what"},
                    {"spec_id": "spec_" + "f" * 32, "label": "constant prices", "axis": "unit"},
                ],
            },
        }
    ]

    rendered = _plan_context(dumped)

    assert (
        f"{SPEC} -- official development assistance; counted as gross disbursements; "
        f"other counts: net disbursements [{net}]\n"
    ) in rendered
    assert "other counts" not in _plan_context(dumped, stated={net})


# Search's answer to "Bilateral plus imputed multilateral aid from France to
# Kenya": gross disbursements elected, net disbursements offered beside it.
_FRANCE_KENYA = json.loads(
    (
        Path(__file__).parents[2]
        / "data-mcp"
        / "tests"
        / "fixtures"
        / "resolve_aid_france_kenya.json"
    ).read_text()
)


async def test_a_continuing_amend_call_shows_the_other_count_and_the_plan_pins_it(
    decision: Any, requirement: Any
) -> None:
    """ "Use net disbursements" arrives on a continuing call, whose message is
    the amendment alone. The fresh brief in its history named the net
    sibling's spec_id beside the gross reading, so the model pins it and the
    measure decision follows the pin."""
    (part,) = resolver_digest(_FRANCE_KENYA)["part_digests"]
    block = part["spec_block"]
    gross = block["spec_id"]
    stored = {k: v for k, v in part.items() if k not in ("spec_block", "candidate_blocks")}
    readings = [SearchReading.from_part({**stored, **reading_facts(block)}).model_dump(mode="json")]
    dataset = requirement()
    first_draft = {
        "question": "aid",
        "answer_mode": "series",
        "decisions": [decision().model_dump(mode="json")],
        "evidence_requirements": [
            dataset.model_copy(update={"source_spec_id": gross}).model_dump(mode="json")
        ],
    }

    def _pin_the_net_count(request: Any) -> list[tuple[str, dict[str, Any]]]:
        named = re.search(r"other counts: net disbursements \[(spec_\w+)\]", _prompt_text(request))
        pin = named.group(1) if named else None
        repinned = dataset.model_copy(update={"source_spec_id": pin}).model_dump(mode="json")
        return [("propose_plan", {"evidence_requirements": [repinned], "amends_version": 1})]

    seam = _seam()
    llm = ScriptedPlannerLlm(script=[[("propose_plan", first_draft)]])
    amending_llm = ScriptedPlannerLlm(script=[_pin_the_net_count])
    plans: list[Plan | None] = []

    # Both entries run in one node, as the graph's planner node re-enters the
    # seam over the `amend` edge in the same session. No translation
    # compiles, so each ends with no plan ready, leaving its draft on record.
    async def _node(ctx: Context, node_input: Any) -> Any:
        del node_input
        seam._agent.model = llm
        await seam(ctx=ctx, question="aid", readings=readings)
        first = load_plan(ctx.state)
        assert first is not None
        seam._agent.model = amending_llm
        await seam(
            ctx=ctx,
            question="aid",
            readings=readings,
            prior_plan=first.model_dump(mode="json"),
            amendment="use net disbursements",
        )
        plans.append(load_plan(ctx.state))

    wf = Workflow(
        name="seam_test_wf",
        edges=[(START, node(_node, name="seam_under_test", rerun_on_resume=True))],
    )
    sessions = InMemorySessionService()
    runner = Runner(node=wf, app_name=_APP, session_service=sessions)
    session = await sessions.create_session(
        app_name=_APP, user_id=_USER, state={ORCHESTRATOR_SPECS_KEY: {gross: block}}
    )
    content = types.Content(role="user", parts=[types.Part(text="probe")])
    async for _ in runner.run_async(user_id=_USER, session_id=session.id, new_message=content):
        pass

    net = block["resolved"]["alternative_measures"][0]["definition"]["spec_id"]
    assert f"other counts: net disbursements [{net}]" in _prompt_text(
        amending_llm.captured_requests[0]
    )
    (plan,) = plans
    assert plan is not None and plan.version == 2
    (pinned,) = plan.evidence_requirements
    assert isinstance(pinned, DatasetRequirement)
    assert pinned.source_spec_id == net
    (measure,) = [d for d in plan.decisions if d.fixes and d.fixes.source_spec_id == net]
    assert measure.options == ("gross disbursements", "net disbursements")
    assert measure.chosen == "net disbursements"
    assert measure.slot in pinned.decision_ids


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

    await harness.run(
        seam,
        question="kenya debt",
        readings=[],
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


def test_a_continuing_message_omits_the_readings_and_digests_but_keeps_the_ask() -> None:
    """A `fresh=False` render -- an `amend`/`replan` re-entry into a session this seam has already briefed -- drops the
    search readings, both digests and the question, and carries the new thing (the
    amendment or refusal ask) verbatim.
    """
    seam = _seam()
    canvas_digest = {"blocks": ["a block"]}
    conversation_digest = {"turns": ["an earlier turn"]}

    amend_message = seam._message(
        fresh=False,
        question="gdp guatemala",
        prior_plan=None,
        refusal=None,
        amendment="only since 2015",
        canvas_digest=canvas_digest,
        conversation_digest=conversation_digest,
    )
    assert "only since 2015" in amend_message
    assert SPEC not in amend_message
    assert "Context for this question" not in amend_message
    assert "already on the user's canvas" not in amend_message
    assert "Earlier turns in this conversation" not in amend_message
    assert "Question:" not in amend_message

    refusal_message = seam._message(
        fresh=False,
        question="gdp guatemala",
        prior_plan=None,
        refusal=_rediscovery_refusal(SPEC),
        amendment=None,
        canvas_digest=canvas_digest,
        conversation_digest=conversation_digest,
    )
    assert "Executing the approved plan failed" in refusal_message
    assert "resolved.stat_vars.member_count = 872" in refusal_message
    assert "Context for this question" not in refusal_message
    assert "already on the user's canvas" not in refusal_message
    assert "Earlier turns in this conversation" not in refusal_message
    assert "Question:" not in refusal_message


def test_terminal_refusal_covers_budget_exhausted_and_question_fixed_but_not_invalid_plan() -> None:
    """`_is_terminal_refusal` is what `_track_tool_call` uses to stop the
    planner's turn on a `propose_plan` refusal that recorded nothing and
    left nothing to correct within this entry -- `BUDGET_EXHAUSTED`,
    `QUESTION_FIXED` and `NOTHING_TO_PLAN` alike. `INVALID_PLAN` -- a question-changing amendment
    under `question_fixed`, say -- is not among them: it is correctable,
    so the model gets to retry rather than the turn ending on it."""
    for code in ("budget_exhausted", "question_fixed", "nothing_to_plan"):
        assert PlannerSeam._is_terminal_refusal({"outcome": "refused", "code": code})
    assert not PlannerSeam._is_terminal_refusal({"outcome": "refused", "code": "invalid_plan"})
    assert not PlannerSeam._is_terminal_refusal({"outcome": "ok", "code": "question_fixed"})
    assert not PlannerSeam._is_terminal_refusal(None)


# -- user text in the brief is cut at the question cap ---------------------------


def _render(seam: PlannerSeam, *, fresh: bool, **overrides: Any) -> str:
    """One `_message` render with no readings, with `overrides` for what a test varies."""
    arguments: dict[str, Any] = {
        "fresh": fresh,
        "question": "gdp guatemala",
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
    # Two cuts at most: the refusal and the sentence naming the specs to avoid.
    assert len(message) < 8 * MAX_QUESTION_CHARS


def test_a_short_refusal_is_rendered_whole() -> None:
    message = _render(_seam(), fresh=False, refusal=_rediscovery_refusal(SPEC))

    assert CUT_MARKER not in message
    assert "resolved.stat_vars.member_count = 872" in message


# -- a referral that repeats without new information ----------


def _repair(constraint: str) -> dict[str, Any]:
    return {
        "outcome": "invalid_input",
        "kind": "requirement_needs_repair",
        "requirement_id": "debt",
        "reason": "r",
        "constraint": constraint,
    }


_AMBIGUOUS = {
    "outcome": "choice_required",
    "kind": "source_ambiguous",
    "requirement_id": "gdp_gtm",
    "candidates": [{"spec_id": "spec_nominal"}, {"spec_id": "spec_per_capita"}],
}


@pytest.mark.parametrize(
    ("first", "second", "stalled"),
    [
        (_AMBIGUOUS, _AMBIGUOUS, True),
        (_repair("family_unsummable"), _repair("family_unsummable"), True),
        (_repair("level_filter_on_flow_axis"), _repair("family_unsummable"), False),
    ],
    ids=["same-choice", "same-repair", "different-repairs"],
)
def test_stalled_referral_stops_on_the_second_identical_referral_not_the_first(
    first: dict[str, Any], second: dict[str, Any], stalled: bool
) -> None:
    counts: dict[tuple[str, Any, Any], int] = {}

    assert PlannerSeam._stalled_referral(first, counts) is False
    assert PlannerSeam._stalled_referral(second, counts) is stalled


@pytest.mark.parametrize("kind", ["discovery_unavailable", "surface_unavailable"])
def test_a_transient_referral_never_stalls(kind: str) -> None:
    counts: dict[tuple[str, Any, Any], int] = {}
    referred = {
        "outcome": "transient",
        "kind": kind,
        "requirement_id": "gdp_gtm" if kind == "discovery_unavailable" else None,
        "reason": "r",
    }

    for _ in range(3):
        assert PlannerSeam._stalled_referral(referred, counts) is False
    assert counts == {}


def test_stalled_referral_ignores_a_response_that_is_not_a_referral() -> None:
    counts: dict[tuple[str, Any, Any], int] = {}
    problem_response = {
        "outcome": "invalid_input",
        "code": "requirement_unknown",
        "message": "no such requirement",
    }

    assert PlannerSeam._stalled_referral(problem_response, counts) is False
    assert PlannerSeam._stalled_referral(problem_response, counts) is False
    assert counts == {}


def _track_proposal(tracker: _RunTracker, translation: dict[str, Any]) -> None:
    """Track one accepted `propose_plan` whose translation answered `translation`."""
    PlannerSeam._track_tool_call(
        tracker,
        None,
        agent_name="dfo_planner",
        tool=cast(BaseTool, SimpleNamespace(name="propose_plan")),
        tool_response={
            "outcome": "ok",
            "status": "draft",
            "version": 1,
            "translation": translation,
        },
    )


def test_a_stalled_referral_ends_the_entry_on_plan_not_ready_carrying_its_constraint() -> None:
    """The second identical repair referral ends the entry with no further
    round: the refusal carries the compile issue the referral names, which
    the user's sentence states (`harness.sentences.referral_reason`)."""
    tracker = _RunTracker()
    referred = {
        "outcome": "invalid_input",
        "kind": "requirement_needs_repair",
        "requirement_id": "gdp_gtm",
        "reason": "operation 'gap' takes a whole side",
        "constraint": "entity_required",
        "repair": "name the entity",
    }

    _track_proposal(tracker, referred)
    assert not tracker.ready_to_break
    _track_proposal(tracker, referred)

    assert tracker.ready_to_break
    assert tracker.refused == {
        "code": SeamCode.PLAN_NOT_READY,
        "detail": {
            "kind": "requirement_needs_repair",
            "requirement_id": "gdp_gtm",
            "reason": "operation 'gap' takes a whole side",
            "constraint": "entity_required",
        },
    }
    assert tracker.submitted is None


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
def test_a_closing_referral_ends_the_entry_on_the_proposal_carrying_it(
    referral: dict[str, Any], named: str
) -> None:
    """An internal failure, a translation that failed the way it failed
    before, or a data service that timed out until the discovery budget was
    spent ends the entry as a final refusal naming what stopped it, as soon
    as a proposal's translation answers with it."""
    tracker = _RunTracker()
    _track_proposal(tracker, referral)

    assert tracker.ready_to_break
    assert tracker.submitted is None
    assert tracker.refused is not None
    assert named in refusal_sentence(tracker.refused)
    assert PlannerSeam._refused(tracker.refused)["refused"]["final"] is True


_NOT_CALLED = "'search' was not called: the planner entry has no time left for it"
_CUT_OFF = "'search' did not answer in the 54 s the planner entry had left"


@pytest.mark.parametrize(
    ("marks", "time_left", "code", "reason"),
    [
        ({"not_issued": True}, 0.0, "turn_wall_time_spent", _NOT_CALLED),
        ({}, 0.0, "discovery_unavailable", _CUT_OFF),
        ({}, 5.0, None, _CUT_OFF),
        ({}, None, None, _CUT_OFF),
        ({"budget_spent": True, "not_issued": True}, 0.0, "discovery_unavailable", _NOT_CALLED),
        ({"budget_spent": True}, 0.0, "discovery_unavailable", _CUT_OFF),
        ({"budget_spent": True}, 5.0, "discovery_unavailable", _CUT_OFF),
    ],
    ids=[
        "never-issued",
        "cut-off-no-time-left",
        "cut-off-time-left",
        "outside-an-entry",
        "budget-spent-before-never-issued",
        "budget-spent-no-time-left",
        "budget-spent-time-left",
    ],
)
def test_a_transient_referral_ends_the_entry_as_the_wall_bound_only_for_a_call_never_issued(
    marks: dict[str, bool], time_left: float | None, code: str | None, reason: str
) -> None:
    """Only a call the request's wall-clock bound left no time to make ends
    the entry as that bound spent.

    A search that was made and cut off with no time left means the data
    service did not answer, so the entry ends as the service unavailable;
    a spent budget says the same whatever the clock reads. With time left the
    planner may propose again. The referral's reason stays in the detail.
    """
    referral = {
        "kind": "discovery_unavailable",
        "outcome": "transient",
        "requirement_id": "oda_france",
        "reason": reason,
        **marks,
    }
    tracker = _RunTracker()
    token = current_discovery_time_left.set(None if time_left is None else lambda: time_left)
    try:
        _track_proposal(tracker, referral)
    finally:
        current_discovery_time_left.reset(token)

    if code is None:
        assert tracker.refused is None
        return
    assert tracker.ready_to_break
    assert tracker.refused is not None
    assert tracker.refused["code"] == code
    assert tracker.refused["detail"]["reason"] == reason
    assert tracker.refused["detail"]["requirement_id"] == "oda_france"


def test_last_referral_is_cleared_once_a_later_translation_resolves_the_requirement() -> None:
    """`_track_tool_call` writes `tracker.last_referral` from every translation
    carrying an `outcome`, referral or not, replacing whatever the
    last one recorded: a later `ok` response clears it, so an entry that
    ends after that clear carries no referral the plan no longer has
    (`_not_ready`)."""
    tracker = _RunTracker()
    referred = {
        "outcome": "choice_required",
        "kind": "source_ambiguous",
        "requirement_id": "gdp_gtm",
        "candidates": [{"spec_id": "spec_nominal"}, {"spec_id": "spec_per_capita"}],
    }
    _track_proposal(tracker, referred)
    assert tracker.last_referral is not None

    resolved = {"outcome": "ok", "requirement_id": "gdp_gtm"}
    _track_proposal(tracker, resolved)

    assert tracker.last_referral is None


def test_last_referral_survives_a_translation_carrying_no_outcome() -> None:
    """A translation missing its own `outcome` key -- not a shape
    translation itself ever returns, but the defensive case `_referral_dict`
    already reads for -- leaves `last_referral` as it was, so an entry that
    ends right after still carries the last real referral."""
    tracker = _RunTracker()
    referred = {
        "outcome": "choice_required",
        "kind": "source_ambiguous",
        "requirement_id": "gdp_gtm",
        "candidates": [{"spec_id": "spec_nominal"}],
    }
    _track_proposal(tracker, referred)

    _track_proposal(tracker, {"malformed": True})

    assert tracker.last_referral is not None
    assert tracker.last_referral["requirement_id"] == "gdp_gtm"


# A `no_data` `search` answer, canned so every `discover_source` call inside
# translation's own mechanical loop gets the identical answer -- no nearest
# series, no judgement to pause on, so translation refers the plan back on
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


def _propose_again_step(*, start_version: int = 1) -> Any:
    """A script entry that proposes the recorded plan again unchanged, which
    translates it.

    Reads the `amends_version` off the last `propose_plan` answer, as
    `_amend_step` does, from `start_version` for the turn's first proposal.
    """

    def step(request: Any) -> list[tuple[str, dict[str, Any]]]:
        version = _last_amended_version(request, default=start_version)
        return [("propose_plan", {"amends_version": version})]

    return step


def _repeating_proposal_script(rounds: int) -> list[Any]:
    """The planner's calls for `rounds` pairs of unchanged proposals.

    Each proposal's translation runs the real mechanical loop against
    `_NO_DATA_SEARCH_RESPONSE`, so both proposals in a pair refer the same
    `(kind, requirement_id)` back. That is the shape that stops one
    `ctx.run_node` call on its second identical referral.
    """
    return [_propose_again_step() for _ in range(2 * rounds)]


def _translator_seam(
    *,
    responses: dict[str, ToolCallResult],
    agent_settings: AgentSettings | None = None,
) -> PlannerSeam:
    return PlannerSeam(
        settings=Settings(),
        surface=FixtureSurface(schemas=[], responses=responses),
        agent_settings=agent_settings,
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


async def test_the_planner_runs_on_the_model_and_translation_the_seam_is_given(
    decision: Any, requirement: Any, derived_requirement: Any
) -> None:
    translated: list[Plan | None] = []

    async def translate(tool_context: Any) -> dict[str, object]:
        translated.append(load_plan(tool_context.state))
        return {"translated_by": "the caller"}

    proposal = {
        "question": "kenya debt",
        "answer_mode": "computed",
        "decisions": [decision().model_dump(mode="json")],
        "evidence_requirements": [
            requirement().model_dump(mode="json"),
            derived_requirement().model_dump(mode="json"),
        ],
    }
    llm = ScriptedPlannerLlm(script=[[("propose_plan", proposal)]])
    seam = PlannerSeam(
        settings=Settings(),
        surface=FixtureSurface(schemas=[]),
        translate=translate,
        model=llm,
    )

    result = await _run_seam(seam, question="kenya debt", readings=[])

    # The caller's translation gives no verdict, so the plan is never declared ready.
    assert result["refused"]["code"] == SeamCode.PLAN_NOT_READY
    assert [plan.version if plan else None for plan in translated] == [1]
    (answered,) = [
        part.function_response.response
        for content in llm.captured_requests[1].contents
        for part in content.parts or []
        if part.function_response is not None and part.function_response.name == "propose_plan"
    ]
    assert answered["translation"] == {"translated_by": "the caller"}


class _ThinkingLlm(ScriptedPlannerLlm):
    """`ScriptedPlannerLlm`, opening each answer with a thought part."""

    async def generate_content_async(self, llm_request: Any, stream: bool = False) -> Any:
        async for response in super().generate_content_async(llm_request, stream):
            assert response.content is not None
            parts = [types.Part(text="weighing it", thought=True), *(response.content.parts or [])]
            yield response.model_copy(update={"content": types.Content(role="model", parts=parts)})


async def test_a_callers_model_callbacks_see_only_what_passes_between_planner_and_model(
    decision: Any, requirement: Any, derived_requirement: Any
) -> None:
    """The first proposal restarts the plan on the amend edge and is refused as
    `QUESTION_FIXED`, which ends the entry, so the seam answers the second request itself
    and the caller's before-model callback never sees it. The caller's after-model
    callback sees the thought part `drop_thought_parts` then drops."""
    seeded = _translated_plan(decision, requirement, derived_requirement, version=3)
    requests: list[Any] = []
    responses: list[LlmResponse] = []
    llm = _ThinkingLlm(
        script=[[("propose_plan", {"starts_new_plan": True, "question": "a different question"})]]
    )
    seam = PlannerSeam(
        settings=Settings(),
        surface=FixtureSurface(schemas=[]),
        model=llm,
        before_model_callbacks=[lambda _context, request: requests.append(request)],
        after_model_callbacks=[lambda _context, response: responses.append(response)],
    )

    result = await _run_seam(
        seam,
        question="kenya debt",
        readings=[],
        prior_plan=seeded.model_dump(mode="json"),
        amendment="actually, tell me about Kenya's ODA instead",
    )

    assert result["refused"]["code"] == "question_fixed"
    assert len(llm.captured_requests) == len(requests) == 1
    assert [bool(part.thought) for response in responses for part in response.content.parts] == [
        True,
        False,
    ]


async def test_every_planner_generation_runs_at_the_low_thinking_config(
    decision: Any, requirement: Any, derived_requirement: Any
) -> None:
    seam = _translator_seam(responses={"search": _NO_DATA_SEARCH_RESPONSE})
    draft = _gdp_plan(decision, requirement, derived_requirement)
    planner_llm = ScriptedPlannerLlm(script=[_propose_again_step()])
    seam._agent.model = planner_llm

    await _run_seam(
        seam, question=draft.question, readings=[], prior_plan=draft.model_dump(mode="json")
    )

    assert len(planner_llm.captured_requests) > 1
    assert all(
        request.config.thinking_config == AGENT_THINKING_CONFIG
        for request in planner_llm.captured_requests
    )


async def test_a_later_entry_carries_the_earlier_tool_calls(
    decision: Any, requirement: Any, derived_requirement: Any
) -> None:
    """A later entry on the same session (the `amend` edge) sends only the
    amendment (`_message`'s `fresh=False`), so its first generation has to
    read the first entry's tool calls from history. Run as a
    workflow node with no explicit `include_contents`, ADK defaults the
    planner to `'none'`, which builds every request from the latest user
    event onwards (`_get_current_turn_contents`,
    `google.adk.flows.llm_flows.contents`), which is the amendment alone.
    `planner.agent.create_planner_agent` pins `include_contents='default'` to
    stop that; reverting the pin fails this test on the later entry's first
    generation.
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
            _propose_again_step(),
            "stop",
            "noted",
        ]
    )
    seam._agent.model = planner_llm
    harness = _SeamHarness()
    plan_json = draft.model_dump(mode="json")

    await harness.run(seam, question="gdp guatemala", readings=[], prior_plan=plan_json)
    await harness.run(
        seam,
        question="gdp guatemala",
        readings=[],
        prior_plan=plan_json,
        amendment="only since 2015",
    )

    # `captured_requests[0..2]` are the first entry: `get_skill`,
    # `propose_plan`, and the closing text. `[3]` is the later entry's first
    # generation.
    later = planner_llm.captured_requests[3]
    assert "only since 2015" in _prompt_text(later)
    names = _call_and_response_names(later)
    assert "FC:get_skill" in names
    assert "FR:get_skill" in names
    assert "FC:propose_plan" in names
    assert "FR:propose_plan" in names


# An `unsupported` `search` answer: translation refers the plan back as
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


async def test_an_answer_that_resumes_a_run_after_a_translated_proposal_is_read(
    decision: Any, requirement: Any, derived_requirement: Any
) -> None:
    """The user's answer resumes a run whose earlier proposal was translated.

    The planner proposes the plan again, whose translation refers
    `unsupported_ask`, then asks the user through `adk_request_input` and
    pauses. The answer resumes the same model turn: the next generation reads
    the answer off the session and `propose_plan` passes the referral guard
    as version 2. The `unsupported_ask` decline it owes survives the resume,
    so the amendment carries it. The amendment's translation refers again,
    so the entry ends with no plan ready.
    """
    draft = _gdp_plan(decision, requirement, derived_requirement)
    seam = _translator_seam(responses={"search": _UNSUPPORTED_SEARCH_RESPONSE})
    # The resumed entry seeds the planner from `prior_plan` again, so the
    # amendment patches the seeded version.
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
            _propose_again_step(start_version=draft.version),
            [("adk_request_input", {"message": "what can be measured instead?", "candidates": []})],
            amend,
            "stop",
        ]
    )
    seam._agent.model = planner_llm
    harness = _SeamHarness()
    plan_json = draft.model_dump(mode="json")

    paused = await harness.run(seam, question=draft.question, readings=[], prior_plan=plan_json)
    assert paused is None, "the ask should pause this call, not resolve it"

    interrupt_id = _pending_ask_interrupt_id(await harness.events())
    answer = types.Content(
        role="user",
        parts=[create_request_input_response(interrupt_id, {"result": {"text": "use debt stock"}})],
    )
    output = await harness.run(
        seam,
        question=draft.question,
        readings=[],
        prior_plan=plan_json,
        new_message=answer,
    )
    assert output is not None and output["refused"]["final"] is False
    proposals = [
        part.function_response.response
        for event in await harness.events()
        for part in (event.content.parts if event.content and event.content.parts else [])
        if part.function_response is not None and part.function_response.name == "propose_plan"
    ]
    assert [(p or {}).get("outcome") for p in proposals] == ["ok", "ok"], proposals
    assert (proposals[1] or {})["version"] == draft.version + 1


async def test_a_repeating_referral_ends_the_entry_on_its_second_identical_referral(
    decision: Any, requirement: Any, derived_requirement: Any
) -> None:
    """Each proposal is translated, and every `search` answers `no_data`, so
    the second unchanged proposal refers the same `(kind, requirement_id,
    constraint)` back. That ends the entry with no further generation, on a refusal whose
    sentence states the referral, and another plan may still answer."""
    seam = _translator_seam(responses={"search": _NO_DATA_SEARCH_RESPONSE})
    draft = _gdp_plan(decision, requirement, derived_requirement)
    planner_llm = ScriptedPlannerLlm(script=[*_repeating_proposal_script(2), "done"])
    seam._agent.model = planner_llm

    result = await _run_seam(
        seam,
        question="gdp guatemala",
        readings=[],
        prior_plan=draft.model_dump(mode="json"),
    )

    referred = {"kind": "no_data", "requirement_id": "gdp_gtm"}
    assert result["refused"] == {
        "text": refusal_sentence({"code": SeamCode.PLAN_NOT_READY, "detail": referred}),
        "final": False,
        "code": SeamCode.PLAN_NOT_READY,
    }
    assert len(planner_llm.captured_requests) == 2


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

    Translation runs inside the planner's own `propose_plan` call, so every function
    call and response the run appended is the planner's: no second agent
    makes a model call in a planner run.
    """
    seam = _translator_seam(responses={"search": _NO_DATA_SEARCH_RESPONSE})
    draft = _gdp_plan(decision, requirement, derived_requirement)
    harness = _SeamHarness()
    seam._agent.model = ScriptedPlannerLlm(script=_repeating_proposal_script(1) + ["done"] * 4)

    await harness.run(
        seam,
        question=draft.question,
        readings=[],
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


async def test_translation_finds_the_plan_a_bare_amendment_records(
    decision: Any, requirement: Any, derived_requirement: Any
) -> None:
    """The plan round-trips through session state alone.

    The planner proposes with nothing but `amends_version`, and
    `run_translate` reaching the draft through `current_draft(tool_context)`
    is enough to conclude: the stub answers `no_data`, and the referral
    names `gdp_gtm`, the plan's own requirement id, which no argument
    carried. The planner's next generation reads that referral off the
    proposal's `translation`.
    """
    seam = _translator_seam(responses={"search": _NO_DATA_SEARCH_RESPONSE})
    draft = _gdp_plan(decision, requirement, derived_requirement)
    planner_llm = ScriptedPlannerLlm(script=[_propose_again_step(), "stop"])
    seam._agent.model = planner_llm

    await _run_seam(
        seam,
        question=draft.question,
        readings=[],
        prior_plan=draft.model_dump(mode="json"),
    )

    answers = [
        (part.function_response.response or {}).get("translation")
        for request in planner_llm.captured_requests
        for content in request.contents
        for part in content.parts or []
        if part.function_response is not None and part.function_response.name == "propose_plan"
    ]
    assert any(
        isinstance(answer, dict) and answer.get("requirement_id") == "gdp_gtm" for answer in answers
    ), f"translation never reached the plan's own requirement: {answers!r}"


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


async def test_the_planners_proposal_carries_the_referral_with_its_outcome(
    decision: Any, requirement: Any, derived_requirement: Any
) -> None:
    """On the path the planner is served from, the `translation` of the
    proposal's function response is the referral itself: its `outcome` on top
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
    planner_llm = ScriptedPlannerLlm(script=[_propose_again_step(), "stop"])
    seam._agent.model = planner_llm

    await _run_seam(
        seam,
        question="aid to guatemala",
        readings=[],
        prior_plan=draft.model_dump(mode="json"),
    )

    (response,) = [
        part.function_response
        for part in planner_llm.captured_requests[1].contents[-1].parts
        if part.function_response is not None
    ]
    assert response.name == "propose_plan"
    proposal = cast(dict[str, Any], response.response)
    assert proposal["outcome"] == "ok", proposal
    answer = proposal["translation"]
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
    conversation B runs its whole turn, and only then does A call `propose_plan`:
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
            _propose_again_step(),
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
        await harness_a.run(
            seam,
            question="gdp guatemala",
            readings=[],
            prior_plan=draft.model_dump(mode="json"),
            on_progress=reports_a.append,
        )

    task_a = asyncio.create_task(_run_a())
    await started_a.wait()

    # Conversation B's whole turn, in this coroutine's own task, while A
    # sits suspended above, before its `propose_plan` call.
    seam._agent.model = planner_llm_b
    await harness_b.run(
        seam, question="unrelated question", readings=[], on_progress=reports_b.append
    )
    reports_b_after_b = list(reports_b)

    release_a.set()
    await task_a

    assert reports_b == reports_b_after_b, (
        "conversation B's on_progress stream grew after B's own turn had already "
        f"finished -- conversation A's own propose_plan call reported into it: {reports_b}"
    )
    assert any(report["tools"] == ["propose_plan"] for report in reports_a), (
        "conversation A's own propose_plan call never reported into A's own on_progress stream"
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
        readings=[],
        prior_plan=seeded.model_dump(mode="json"),
        amendment="actually, tell me about Kenya's ODA instead",
    )

    assert result is not None and "refused" in result
    assert "own message" in result["refused"]["text"]
    plan = load_plan(await harness.state())
    assert plan is not None
    assert plan.version == 3
    assert plan.question == "kenya debt"


async def test_a_second_turn_on_one_session_clears_the_first_drafts_plan_and_briefed_flag(
    decision: Any, requirement: Any, derived_requirement: Any
) -> None:
    """The orchestrator can call `plan` again in one session after an
    earlier entry handed back, and the planner node passes no `prior_plan`
    on that later first-draft entry even though `ctx.state["plan"]` still
    holds the earlier entry's. `prior_plan is None` clears
    `planner.plan.PLAN_STATE_KEY` and the briefed flag at the top of
    `__call__` (see its own comment), so the second entry both renders the
    full brief again (checked against the rendered prompt, not a flag) and
    drafts its own plan. It is not refused for restating no `amends_version`
    against a plan still on record from the first entry.
    """
    harness = _SeamHarness()
    seam = _seam()
    first_plan = _translated_plan(decision, requirement, derived_requirement, version=1)
    first_llm = ScriptedPlannerLlm(script=[_DECLARE])
    use_compiled_proposals(seam._agent)
    seam._agent.model = first_llm

    first = await harness.run(
        seam,
        question="kenya debt",
        readings=[],
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
    # A seam with the real `propose_plan`, on the same session.
    seam = _seam()
    seam._agent.model = second_llm

    # This second entry's draft carries no translation, so it is never
    # declared ready and the entry ends with no plan -- the recorded plan
    # `propose_plan` itself left behind is what this test is about.
    await harness.run(seam, question="kenya oda 2023", readings=[])

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

    No turn here compiles (the script only ever proposes), so every turn
    ends the same way, with no plan ready -- the version `propose_plan`
    recorded along the way is what this test is about.

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
        await harness.run(seam, question="kenya debt", readings=[], amendment=f"turn {turn}")
        plan = load_plan(await harness.state())
        assert plan is not None
        version = plan.version

    assert version == 6


async def test_the_full_brief_a_second_call_inherits_is_byte_identical_to_the_first() -> None:
    """The fresh/continuing split rests on one assumption: `question`,
    `readings`, `canvas_digest` and `conversation_digest` never change
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
    readings: list[dict[str, Any]] = []
    canvas_digest = {"blocks": ["a block"]}
    conversation_digest = {"turns": ["an earlier turn"]}

    seam = _seam()
    first_llm = ScriptedPlannerLlm(script=["still nothing"] * 4)
    second_llm = ScriptedPlannerLlm(script=["still nothing"] * 4)

    expected_first_message = seam._message(
        fresh=True,
        question=question,
        prior_plan=None,
        refusal=None,
        amendment=None,
        canvas_digest=canvas_digest,
        conversation_digest=conversation_digest,
        ledger=planner_seam._readings_ledger(readings),
    )

    async def _node(ctx: Context, node_input: Any) -> Any:
        del node_input
        seam._agent.model = first_llm
        await seam(
            ctx=ctx,
            question=question,
            readings=readings,
            canvas_digest=canvas_digest,
            conversation_digest=conversation_digest,
        )
        seam._agent.model = second_llm
        await seam(
            ctx=ctx,
            question=question,
            readings=readings,
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
    turn (one `__call__`, one `_propose`, and no second turn), so the fifth is the one that goes past the cap. The seam returns a
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

    result = await _run_seam(seam, question="kenya debt", readings=[])

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

    result = await harness.run(seam, question="kenya debt", readings=[])

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
    paused = await harness.run(seam, question="kenya debt", readings=[])
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
        readings=[],
        amendment="one more change",
        new_message=resume_message,
    )

    assert result is not None and "refused" in result
    text = result["refused"]["text"]
    assert f"budget of {MAX_PLAN_ROUNDS} plan drafts" in text
    assert "fresh budget" in text


# -- the request's budget ---------------------------------------------------


def _request_budget(**overrides: Any) -> TurnBudget:
    bounds: dict[str, Any] = {
        "model_time_s": 90.0,
        "model_calls_max": 30,
        "wall_s": 600.0,
        "first_chunk_s": 12.0,
    }
    return TurnBudget(**{**bounds, **overrides})


async def test_discovery_inside_an_entry_reads_the_request_s_wall_bound() -> None:
    """Inside the run a discovery call reads what is left of the request's
    wall-clock bound; outside a request, and outside the run, none is set."""
    seen: list[float | None] = []

    def _read_time_left(_request: Any) -> str:
        seen.append(discovery_time_left())
        return "done"

    seam = _seam()
    seam._agent.model = ScriptedPlannerLlm(script=[_read_time_left, _read_time_left])
    budget = _request_budget(wall_s=600.0, clock=lambda: 5.0)

    token = current_turn_budget.set(budget)
    try:
        await _run_seam(seam, question="kenya debt", readings=[])
    finally:
        current_turn_budget.reset(token)
    await _run_seam(seam, question="kenya debt", readings=[])

    assert seen == [600.0, None]
    assert discovery_time_left() is None


async def test_a_call_cut_inside_an_entry_and_still_running_is_cancelled_when_the_entry_ends() -> (
    None
):
    kept: list[asyncio.Task[Any]] = []

    def _keep_a_cut_call(_request: Any) -> str:
        cut_calls = current_cut_calls.get()
        assert cut_calls is not None
        task = asyncio.ensure_future(asyncio.sleep(3600))
        cut_calls.keep(f"call {len(kept)}".encode(), task)
        kept.append(task)
        return "done"

    seam = _seam()
    seam._agent.model = ScriptedPlannerLlm(script=[_keep_a_cut_call])

    await _run_seam(seam, question="kenya debt", readings=[])

    assert kept
    assert all(task.cancelled() for task in kept)
    assert current_cut_calls.get() is None


@pytest.mark.parametrize(
    ("spent", "code"),
    [
        ({"model_time_spent": 90.0}, SeamCode.TURN_MODEL_TIME_SPENT),
        ({"model_calls": 30}, SeamCode.TURN_MODEL_CALLS_SPENT),
        ({"wall_s": 0.0}, SeamCode.TURN_WALL_TIME_SPENT),
    ],
)
async def test_a_run_whose_model_call_the_budget_closed_ends_the_entry_on_that_bound(
    spent: dict[str, float], code: SeamCode
) -> None:
    """The model answers a closed call with the bound's sentence
    (`harness.turn_budget.bounded`) and makes no request. With no referral to
    ask about, that ends the planner's run, and the entry refuses on the
    bound, final for the request, in place of `SeamCode.PLAN_NOT_READY`."""
    budget = _request_budget()
    for name, value in spent.items():
        setattr(budget, name, value)
    seam = _seam()
    llm = _BoundedScriptedPlannerLlm(script=["never sent"])
    seam._agent.model = llm

    token = current_turn_budget.set(budget)
    try:
        result = await _run_seam(seam, question="kenya debt", readings=[])
    finally:
        current_turn_budget.reset(token)

    assert llm.captured_requests == []
    assert result["refused"] == {
        "text": refusal_sentence({"code": code}),
        "final": True,
        "code": code,
    }


class _BoundedScriptedPlannerLlm(ScriptedPlannerLlm):
    """`ScriptedPlannerLlm` kept to the current request budget, as
    `harness.turn_budget.BoundedGemini` keeps Gemini."""

    async def generate_content_async(self, llm_request: Any, stream: bool = False) -> Any:
        scripted = super().generate_content_async
        async for response in bounded(lambda: scripted(llm_request, stream), stream=stream):
            yield response


def _closing_response(code: SeamCode) -> LlmResponse:
    """The final response `harness.turn_budget.bounded` gives a call `code` closed."""
    return LlmResponse(
        content=types.Content(
            role="model", parts=[types.Part(text=refusal_sentence({"code": code}))]
        ),
        custom_metadata={CLOSED_BY_KEY: code.value},
    )


def _after_model_on(seam: PlannerSeam, tracker: _RunTracker, response: LlmResponse) -> Any:
    token = planner_seam._active_call.set(
        planner_seam._CallState(tracker=tracker, on_progress=None)
    )
    try:
        return seam._after_planner_model(SimpleNamespace(), response)
    finally:
        planner_seam._active_call.reset(token)


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

    args = _clarifying_question_args(candidates, lead="Why the user is asked.")

    assert args == {"message": args["message"], "candidates": candidates}
    # The message reaches the user: the candidates' labels name the sources.
    assert "spec_" not in args["message"]


def test_a_referral_resolved_by_a_later_ok_translation_lets_the_budget_refuse_instead_of_asking() -> (
    None
):
    """Once a later `ok` translation has cleared `tracker.last_referral`
    (`test_last_referral_is_cleared_once_a_later_translation_resolves_the_requirement`),
    a call the budget closes finds no pinnable candidates: its closing text
    stands and the entry refuses on the bound, never the `adk_request_input`
    ask a still-open referral would get."""
    seam = _seam()
    tracker = _RunTracker()
    referred = {
        "outcome": "choice_required",
        "kind": "source_ambiguous",
        "requirement_id": "gdp_gtm",
        "candidates": [{"spec_id": "spec_nominal"}, {"spec_id": "spec_per_capita"}],
    }
    _track_proposal(tracker, referred)
    _track_proposal(tracker, {"outcome": "ok", "requirement_id": "gdp_gtm"})

    response = _after_model_on(seam, tracker, _closing_response(SeamCode.TURN_MODEL_TIME_SPENT))

    assert response is None, "a cleared referral must not surface an adk_request_input ask"
    assert tracker.refused == {"code": SeamCode.TURN_MODEL_TIME_SPENT, "detail": {}}


def test_a_referral_surviving_a_malformed_translation_still_gets_asked_when_the_budget_closes() -> (
    None
):
    """A translation with no `outcome` settles nothing
    (`test_last_referral_survives_a_translation_carrying_no_outcome`), so a
    call the budget closes after it still finds the earlier referral's
    candidates and asks through `adk_request_input`, the same as if the
    malformed response had never arrived."""
    seam = _seam()
    tracker = _RunTracker()
    referred = {
        "outcome": "choice_required",
        "kind": "source_ambiguous",
        "requirement_id": "gdp_gtm",
        "candidates": [{"spec_id": "spec_nominal"}, {"spec_id": "spec_per_capita"}],
    }
    _track_proposal(tracker, referred)
    _track_proposal(tracker, {"malformed": True})

    response = _after_model_on(seam, tracker, _closing_response(SeamCode.TURN_MODEL_CALLS_SPENT))

    assert response is not None
    (part,) = response.content.parts
    assert part.function_call is not None
    assert part.function_call.name == REQUEST_INPUT_FUNCTION_CALL_NAME
    assert {c["spec_id"] for c in part.function_call.args["candidates"]} == {
        "spec_nominal",
        "spec_per_capita",
    }
    assert tracker.refused is None


async def test_a_budget_closing_on_a_referral_with_candidates_asks_the_user_and_the_answer_elects(
    decision: Any, requirement: Any, derived_requirement: Any
) -> None:
    """A call the request's budget closes on a referral carrying pinnable
    candidates ends the entry the way the planner's own `ask_user` does: a
    long-running `adk_request_input` pause, built from the referral's own
    candidates with no model call. The user's answer resumes the run in a
    request of its own and pairs by id (`planner.agent._elected_spec_ids`).

    The cap allows one model call: it proposes the plan again, and its
    translation records a `source_ambiguous` referral naming `SPEC` and
    `SPEC_B` (`_TWO_CANDIDATES_SEARCH_RESPONSE`). The next call finds the cap
    spent.
    """
    draft = Plan(
        question="aid to guatemala",
        decisions=[decision()],
        evidence_requirements=[
            requirement(id="gdp_gtm", measure="aid disbursed"),
            derived_requirement(input_id="gdp_gtm"),
        ],
    )
    seam = _translator_seam(responses={"search": _TWO_CANDIDATES_SEARCH_RESPONSE})
    planner_llm = _BoundedScriptedPlannerLlm(script=[_propose_again_step()])
    seam._agent.model = planner_llm
    harness = _SeamHarness()

    token = current_turn_budget.set(_request_budget(model_calls_max=1))
    try:
        paused = await harness.run(
            seam, question=draft.question, readings=[], prior_plan=draft.model_dump(mode="json")
        )
    finally:
        current_turn_budget.reset(token)
    assert paused is None, "the budget's clarifying question should pause this call"
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
    await harness.run(
        seam,
        question=draft.question,
        readings=[],
        prior_plan=draft.model_dump(mode="json"),
        new_message=resume_message,
    )

    fake_tool_context = SimpleNamespace(session=SimpleNamespace(events=await harness.events()))
    assert _elected_spec_ids(fake_tool_context) == frozenset({SPEC_B})


async def test_a_submitted_plan_ends_the_planner_run_with_no_further_model_call(
    decision: Any, requirement: Any, derived_requirement: Any
) -> None:
    """A compiled proposal declared ready is the planner's last event: ADK asks the
    model for nothing after it, so the submission stands."""
    plan = _translated_plan(decision, requirement, derived_requirement)
    seam = _seam()
    llm = ScriptedPlannerLlm(script=[_DECLARE])
    use_compiled_proposals(seam._agent)
    seam._agent.model = llm

    result = await _run_seam(
        seam, question=plan.question, readings=[], prior_plan=plan.model_dump(mode="json")
    )

    assert len(llm.captured_requests) == 1
    assert result is not None and result["plan"]["version"] == plan.version


# -- the first year a pinned dataset holds, read when a plan needs it -------------

# The dataset Search found for "the last five years": answered over 2020-2024.
_AID_2020_2024 = {
    "identity_version": "spec1",
    "measure": "education aid",
    "requested_window": {"kind": "range", "start_year": 2020, "end_year": 2024},
}
_AID_SPEC = spec_id_for(canonicalize(_AID_2020_2024))
_LAST_FIVE_YEARS = "How has education aid to Kenya grown over the last five years?"
_AID_EVERY_YEAR_SPEC, _AID_EVERY_YEAR = windowed_identity(_AID_2020_2024, {"kind": "all"})


class _DefinitionReads:
    """A tool surface whose `resolve_definition` answers `answer()`, recording each call it takes.

    Every other tool is unreachable, so the translation each recorded plan
    gets stops on a transient referral. Translation still resolves a pinned
    copy over the plan's period through `resolve_definition`, so a test
    about the first year reads `first_year_reads`.
    """

    def __init__(self, answer: Callable[[], ToolCallResult]) -> None:
        self._answer = answer
        self.calls: list[Mapping[str, object]] = []

    async def list_tools(self) -> list[Any]:
        return []

    async def call_tool(
        self, *, tool: ToolId, arguments: Mapping[str, object], on_progress: Any = None
    ) -> ToolCallResult:
        del on_progress
        if tool.name != "resolve_definition":
            raise ToolSurfaceUnavailableError(f"{tool.name} is not served here")
        self.calls.append(arguments)
        return self._answer()

    @property
    def first_year_reads(self) -> list[Mapping[str, object]]:
        """The calls resolving a dataset over no window, the read `propose_plan` takes."""
        return [
            call
            for call in self.calls
            if cast(dict[str, Any], call["identity"])["requested_window"] == {"kind": "all"}
        ]


def _held_since_1995() -> ToolCallResult:
    """data-mcp's answer for the dataset over no window: held from 1995, 870 rows."""
    block = {
        "spec_id": _AID_EVERY_YEAR_SPEC,
        "definition": {"identity": _AID_EVERY_YEAR},
        "resolved": {"download": {"rows": 870, "window": {"start_year": 1995, "end_year": 2030}}},
    }
    part = {"status": "definite", "spec": block}
    return ToolCallResult(is_error=False, meta={SEARCH_ENVELOPE_META_KEY: {"parts": [part]}})


def _unreachable() -> ToolCallResult:
    raise ToolSurfaceUnavailableError("data-mcp did not answer")


def _last_five_years_plan() -> dict[str, Any]:
    """What a model proposes for "the last five years" and "the latest year" of the shown dataset.

    The growth and the filter both read from a start that follows the newest data.
    """
    latest = {"anchor": "latest", "offset": 0}
    return {
        "question": _LAST_FIVE_YEARS,
        "answer_mode": "series",
        "evidence_requirements": [
            {
                "kind": "dataset",
                "id": "aid",
                "measure": "education aid",
                "entities": [{"name": "Kenya"}],
                "period": {"start": 2020, "end": 2024},
                "period_end_open": True,
                "source_spec_id": _AID_SPEC,
            },
            {
                "kind": "growth",
                "id": "growth",
                "input_id": "aid",
                "window_start": 2020,
                "window_end": 2024,
                "rolling": {
                    "window_start": {"anchor": "latest", "offset": -4},
                    "window_end": latest,
                },
            },
            {
                "kind": "filter",
                "id": "newest",
                "input_id": "aid",
                "predicates": [
                    {"kind": "period_threshold", "op": "eq", "year": 2024, "rolling": latest}
                ],
            },
        ],
    }


_AID_BLOCK = {"spec_id": _AID_SPEC, "definition": {"identity": _AID_2020_2024}, "resolved": {}}


async def _run_the_planner(
    reads: _DefinitionReads, script: list[Any]
) -> tuple[ScriptedPlannerLlm, SessionState]:
    """One first-turn entry showing the planner the dataset Search answered over
    2020-2024, as a search reading whose block the turn's state holds."""
    seam = PlannerSeam(settings=Settings(), surface=reads)
    llm = ScriptedPlannerLlm(script=script)
    seam._agent.model = llm
    reading = SearchReading.model_validate(
        {
            "text": "education aid over the last five years",
            "status": "definite",
            "spec_id": _AID_SPEC,
            "sentence": "Education aid to Kenya, 2020 to 2024",
            "window": {"start": 2020, "end": 2024, "source": "requested"},
        }
    )
    harness = _SeamHarness({ORCHESTRATOR_SPECS_KEY: {_AID_SPEC: _AID_BLOCK}})
    await harness.run(seam, question=_LAST_FIVE_YEARS, readings=[reading.model_dump(mode="json")])
    return llm, await harness.state()


async def _propose_the_last_five_years(
    reads: _DefinitionReads,
) -> tuple[ScriptedPlannerLlm, SessionState]:
    return await _run_the_planner(reads, [[("propose_plan", _last_five_years_plan())]])


def _the_last_three_years(request: Any) -> list[tuple[str, dict[str, Any]]]:
    """A second proposal of the turn: the growth amended to the last three years."""
    growth = copy.deepcopy(_last_five_years_plan()["evidence_requirements"][1])
    growth["window_start"] = 2021
    growth["rolling"]["window_start"]["offset"] = -3
    amended = {
        "evidence_requirements": [growth],
        "amends_version": _last_amended_version(request, default=1),
    }
    return [("propose_plan", amended)]


async def test_a_last_five_years_plan_reads_the_first_year_once_and_pins_from_it() -> None:
    """Search answered over 2020-2024, which states no earlier year. The model's
    `propose_plan` call is recorded all the same: the dataset is resolved once
    over no window for its first year, 1995, and the pin moves to the dataset
    from 1995 with no end year. The growth and the latest-year filter both
    roll from their start on that one source, and one read serves both, as it
    does the second proposal of the turn."""
    reads = _DefinitionReads(_held_since_1995)

    llm, state = await _run_the_planner(
        reads, [[("propose_plan", _last_five_years_plan())], _the_last_three_years]
    )

    assert reads.first_year_reads == [
        {"spec_id": _AID_EVERY_YEAR_SPEC, "identity": _AID_EVERY_YEAR}
    ]
    assert full_histories(state) == {_AID_SPEC: FullHistory(first_year=1995, rows=870)}
    plan = load_plan(state)
    assert plan is not None, llm.captured_requests[-1].contents[-1]
    assert plan.version == 2
    opened, _ = windowed_identity(_AID_2020_2024, {"kind": "range", "start_year": 1995})
    dataset, growth, newest = plan.evidence_requirements
    assert isinstance(dataset, DatasetRequirement)
    assert (dataset.source_spec_id, dataset.period_end_open) == (opened, True)
    assert dataset.period == Period(start=2020, end=2024)
    assert isinstance(growth, GrowthRequirement)
    assert growth.rolling is not None
    assert growth.rolling.window_start == RelativePeriod(offset=-3)
    assert rolls_from_its_start(newest)


async def test_a_plan_of_fixed_years_over_a_windowed_reading_reads_nothing() -> None:
    """The reading is over 2020-2024 and the plan states those years as fixed,
    so no first year is needed and `resolve_definition` is never called."""
    reads = _DefinitionReads(_held_since_1995)
    fixed = _last_five_years_plan()
    dataset, growth, _ = fixed["evidence_requirements"]
    del dataset["period_end_open"], growth["rolling"]
    fixed["evidence_requirements"] = [dataset, growth]

    _, state = await _run_the_planner(reads, [[("propose_plan", fixed)]])

    assert reads.calls == []
    assert full_histories(state) == {}
    plan = load_plan(state)
    assert plan is not None
    assert plan.evidence_requirements[0].source_spec_id == _AID_SPEC  # type: ignore[union-attr]


def _no_first_year() -> ToolCallResult:
    """data-mcp's answer for the dataset over no window, stating no coverage."""
    block = {
        "spec_id": _AID_EVERY_YEAR_SPEC,
        "definition": {"identity": _AID_EVERY_YEAR},
        "resolved": {},
    }
    part = {"status": "definite", "spec": block}
    return ToolCallResult(is_error=False, meta={SEARCH_ENVELOPE_META_KEY: {"parts": [part]}})


@pytest.mark.parametrize(
    ("answer", "code"),
    [(_unreachable, "surface_unavailable"), (_no_first_year, "invalid_plan")],
    ids=["read-did-not-complete", "answer-states-no-first-year"],
)
async def test_a_first_year_that_cannot_be_read_refuses_saying_the_read_failed(
    answer: Callable[[], ToolCallResult], code: str
) -> None:
    reads = _DefinitionReads(answer)

    llm, state = await _propose_the_last_five_years(reads)

    assert len(reads.calls) == 1
    assert load_plan(state) is None
    (refusal,) = [
        part.function_response.response
        for content in llm.captured_requests[-1].contents
        for part in content.parts or []
        if part.function_response is not None
    ]
    assert refusal["code"] == code
    assert refusal["detail"] == {
        "issue": "rolling_needs_open_period",
        "fault": "first_year_read_failed",
        "requirement_ids": ["aid"],
        "spec_id": _AID_SPEC,
    }


async def test_the_brief_marks_each_ask_with_the_parties_it_breaks_down_by() -> None:
    seam = _seam()
    llm = ScriptedPlannerLlm(script=["Planned."])
    seam._agent.model = llm
    asks = ["the total", "Debt service by creditor"]
    harness = _SeamHarness(
        state={
            ASKS_STATE_KEY: asks,
            ASK_BREAKDOWNS_STATE_KEY: [AskBreakdown(ask=1, by=("creditor",)).model_dump()],
        }
    )

    await harness.run(seam, question="debt service by creditor", readings=[])

    prompt = _prompt_text(llm.captured_requests[0])
    assert "0. the total\n1. Debt service by creditor (by creditor)\n" in prompt
    assert (
        "'breakdown_not_kept' when the source this ask's answer pins cannot hold the party open"
        in prompt
    )

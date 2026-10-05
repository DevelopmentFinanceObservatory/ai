"""Tests for the thinking level the translator's generations run at, and the turn-opening mechanics `_before_translator_model` still owns.

`create_translator_agent` sets `BuiltInPlanner(thinking_config=AGENT_THINKING_CONFIG)`
(`planner.settings`), which writes LOW into every request of this agent's
turn on its own -- there is no longer a per-call level to choose, so what is
asserted here is that every request actually carried LOW, captured off a
recording model through a real `Runner`, not the agent's own config object.
`_before_translator_model` keeps one job: answering the turn's first
generation with `translate`'s own call instead of reaching the model for it.
The rest of this file proves that opener fires exactly once per run and
never re-fires once a tool response already exists for it, using each run's
own captured requests and tool-call counts as the evidence -- not thinking
level, which no longer distinguishes an opened turn from an unopened one.

Driven end to end, through a real `Runner`, because `BuiltInPlanner` writing
the level into the request only happens inside ADK's own flow.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import pytest
from analysis_mcp.server import mcp as analysis_mcp
from compiler.surface import McpSurface
from data_mcp.server import mcp as real_data_mcp
from fastmcp import FastMCP
from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events.event import Event
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.sessions.session import Session
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from planner.plan import (
    PLAN_STATE_KEY,
    AggregateRequirement,
    DatasetRequirement,
    Decision,
    Plan,
    PlanStatus,
    save_plan,
)
from pydantic import Field as PydanticField
from translator.agent import _last_tool_response as _translator_last_tool_response
from translator.agent import create_translator_agent

from ..context_probe import canonical_tools, system_instruction_text

RequirementFactory = Callable[..., DatasetRequirement]
DerivedRequirementFactory = Callable[..., AggregateRequirement]
DraftFactory = Callable[..., Plan]
DecisionFactory = Callable[..., Decision]

SPEC_OTHER = "spec_" + "b" * 32


class _ScriptedLlm(BaseLlm):
    """Answers each generation from `script`, and keeps every request it was handed.

    A script entry is a list of `(tool_name, args)` function calls, a string
    of final text, or a callable reading the request and returning one of
    those. Either shape can also carry a `thought`: a list is wrapped as
    `{"thought": text, "calls": [...]}`, a string as `{"thought": text,
    "text": ...}`, putting a `Part(text=thought, thought=True)` ahead of the
    rest of the response -- the shape a real `include_thoughts=True`
    generation returns, for tests that check what happens to it afterward.
    Past the end of the script it answers "done", so an unscripted extra
    generation ends the turn rather than looping.

    `max_generations` caps the generation count outright: the "done" answer
    only ends a turn when the flow is waiting on the model, and a flow that
    keeps starting steps without ever asking for one would loop forever. A
    turn that reaches the cap raises instead, which fails the test it is in
    rather than hanging it.
    """

    model: str = "scripted-reasoning-test"
    script: list[Any] = PydanticField(default_factory=list)
    captured_requests: list[Any] = PydanticField(default_factory=list)
    max_generations: int = 24

    async def generate_content_async(self, llm_request: Any, stream: bool = False) -> Any:
        del stream
        self.captured_requests.append(llm_request)
        index = len(self.captured_requests) - 1
        if index >= self.max_generations:
            raise AssertionError(
                f"the translator's turn ran {index + 1} model generations without settling"
            )
        step = self.script[index] if index < len(self.script) else "done"
        if callable(step):
            step = step(llm_request)
        thought_parts: list[types.Part] = []
        if isinstance(step, dict):
            thought_parts = [types.Part(text=step["thought"], thought=True)]
            step = step.get("calls", step.get("text"))
        if isinstance(step, str):
            yield LlmResponse(
                content=types.Content(
                    role="model", parts=[*thought_parts, types.Part.from_text(text=step)]
                )
            )
            return
        yield LlmResponse(
            content=types.Content(
                role="model",
                parts=[
                    *thought_parts,
                    *(types.Part.from_function_call(name=name, args=args) for name, args in step),
                ],
            )
        )


@pytest.fixture
async def combined_surface(data_server: FastMCP) -> McpSurface:
    """The canned `data_server` plus the real `get_data` tool object.

    `discover_source` answers from the canned `search`, and
    `propose_operations` needs `get_data`'s real capability schema to compile,
    or it reports `capability_schema_drift` against a stand-in schema no real
    deploy serves.
    """
    data_server.local_provider.remove_tool("get_data")
    data_server.add_tool(await real_data_mcp.get_tool("get_data"))
    return McpSurface(sources={"data-mcp": data_server, "analysis-mcp": analysis_mcp})


def _retype_the_referral_translate_reported(request: Any) -> list[tuple[str, dict[str, Any]]]:
    """Close as a model does: hand `translate`'s own `referral` back through `set_model_response`."""
    (response,) = [
        part.function_response for part in request.contents[-1].parts if part.function_response
    ]
    return [("set_model_response", {"response": response.response["referral"]})]


async def _translator_requests(
    plan: Plan,
    surface: McpSurface,
    *,
    script: list[Any],
    tool_calls: dict[str, int] | None = None,
) -> list[Any]:
    """Run the translator as a sub-agent through a real `Runner`, and return its requests.

    The plan is written into the session at creation, so `translate` finds it.
    `tool_calls`, when given, is filled with how many times each tool ran: the
    callback is listed ahead of the translator's own closing one the way
    `harness.planner_seam.PlannerSeam.__init__` lists its tracking callback,
    so it counts a call as the tool produced it.
    """
    translator = create_translator_agent(model_name="m", surface=surface, mode="single_turn")
    translator_llm = _ScriptedLlm(script=script)
    translator.model = translator_llm
    if tool_calls is not None:

        def _count(tool: Any, args: Any, tool_context: Any, tool_response: Any) -> None:
            del args, tool_context, tool_response
            tool_calls[tool.name] = tool_calls.get(tool.name, 0) + 1

        translator.after_tool_callback = [  # type: ignore[assignment]
            _count,
            *translator.canonical_after_tool_callbacks,
        ]
    root_llm = _ScriptedLlm(script=[[("dfo_translator", {"request": "irrelevant here"})]])
    root = LlmAgent(name="root", model=root_llm, sub_agents=[translator])
    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name="app",
        user_id="u",
        session_id="s",
        state={PLAN_STATE_KEY: plan.model_dump_json()},
    )
    runner = Runner(agent=root, app_name="app", session_service=session_service)
    async for _event in runner.run_async(
        user_id="u",
        session_id="s",
        new_message=types.Content(role="user", parts=[types.Part.from_text(text="go")]),
    ):
        pass
    return cast("list[Any]", translator_llm.captured_requests)


def _levels(requests: list[Any]) -> list[types.ThinkingLevel | None]:
    return [
        request.config.thinking_config.thinking_level
        if request.config.thinking_config is not None
        else None
        for request in requests
    ]


def _last_tool_response(request: Any) -> Any:
    """The tool response the model's next turn would read, from `request`."""
    for content in reversed(request.contents):
        for part in reversed(content.parts or ()):
            if part.function_response is not None:
                return part.function_response
    return None


async def test_a_clean_plan_calls_the_model_once_and_only_to_close(
    draft: DraftFactory, combined_surface: McpSurface
) -> None:
    """The opener carries a clean plan's whole model cost down to the closing turn.

    The generation that would emit `translate` with no arguments never reaches
    the model: `_before_translator_model` answers it with the call itself, ADK
    runs `translate`, and the first request this model sees is the one
    carrying `translate`'s own answer -- proven by the single captured
    request and by that request already holding `translate`'s answer rather
    than a bare, argument-free call the model would still have to make. That
    request runs at LOW, `AGENT_THINKING_CONFIG`'s level for every generation
    of this agent's turn.
    """
    requests = await _translator_requests(draft(), combined_surface, script=['{"outcome": "ok"}'])

    assert len(requests) == 1
    assert _levels(requests) == [types.ThinkingLevel.LOW]
    response = _last_tool_response(requests[0])
    assert response.name == "translate"
    assert response.response["outcome"] == "ok"


async def test_the_opener_carries_a_thought_signature_on_the_request_the_model_reads(
    draft: DraftFactory, combined_surface: McpSurface
) -> None:
    """Gemini rejects a request whose model-authored function call has no thought signature.

    The opener's `translate` call is injected outside a model turn, so nothing
    but `_before_translator_model` can give it one. A scripted model cannot
    see the server's 400, so this asserts the request's shape. Every function
    call the model role authored in the first request the model is handed
    carries a `thought_signature`, and that request holds the opener's call
    and no other model-written content.
    """
    (request,) = await _translator_requests(draft(), combined_surface, script=['{"outcome": "ok"}'])

    calls = [
        part
        for content in request.contents
        if content.role == "model"
        for part in content.parts or ()
        if part.function_call is not None
    ]
    assert [part.function_call.name for part in calls] == ["translate"]
    assert all(part.thought_signature for part in calls)


async def test_the_generation_that_ranks_candidates_also_runs_at_low(
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    decision: DecisionFactory,
    combined_surface: McpSurface,
) -> None:
    """The `needs_decision` answer is the one request that has to weigh candidates.

    A requirement measuring aid answers `source_ambiguous`, so `translate`
    stops with two candidates and a `candidate_ranking` judgement. The
    generation reading that answer ranks them, and, like the generation that
    answers it and the one closing the turn, runs at LOW: nothing in this
    agent's turn singles the ranking generation out for a different level any
    more.
    """
    dataset = requirement(id="gdp_gtm", measure="aid disbursed")
    plan = Plan(
        question="aid to guatemala",
        decisions=[decision()],
        evidence_requirements=[dataset, derived_requirement(id="gdp_total", input_id="gdp_gtm")],
        status=PlanStatus.DRAFT,
    )
    script: list[Any] = [
        [("translate", {"requirement_id": "gdp_gtm", "choice": SPEC_OTHER, "rationale": "closer"})],
        _retype_the_referral_translate_reported,
    ]

    requests = await _translator_requests(plan, combined_surface, script=script)

    assert len(requests) == 2
    assert _levels(requests) == [types.ThinkingLevel.LOW, types.ThinkingLevel.LOW]
    first = _last_tool_response(requests[0])
    assert first.name == "translate"
    assert first.response["outcome"] == "needs_decision"


async def test_the_ranking_turn_keeps_one_cached_prefix_across_its_two_calls(
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    decision: DecisionFactory,
    combined_surface: McpSurface,
) -> None:
    """The ranking generation and the closer share the prefix ADK caches.

    A ranking turn is the translator's one pair of model generations: the
    opener supplies `translate` on a clean turn, so a clean turn reaches the
    model once. `GeminiContextCacheManager._generate_cache_fingerprint` reads
    the system instruction and the canonicalised tools, so those have to
    match across the pair for the closer to reuse the cache. They drift if
    the instruction is ever built per request.
    """
    dataset = requirement(id="gdp_gtm", measure="aid disbursed")
    plan = Plan(
        question="aid to guatemala",
        decisions=[decision()],
        evidence_requirements=[dataset, derived_requirement(id="gdp_total", input_id="gdp_gtm")],
        status=PlanStatus.DRAFT,
    )
    script: list[Any] = [
        [("translate", {"requirement_id": "gdp_gtm", "choice": SPEC_OTHER, "rationale": "closer"})],
        _retype_the_referral_translate_reported,
    ]

    ranking, closer = await _translator_requests(plan, combined_surface, script=script)

    assert system_instruction_text(ranking) == system_instruction_text(closer)
    assert canonical_tools(ranking) == canonical_tools(closer)


async def test_a_generations_thought_never_reaches_the_next_generations_request(
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    decision: DecisionFactory,
    combined_surface: McpSurface,
) -> None:
    """`drop_thought_parts` keeps a generation's own reasoning out of this turn's later requests.

    The ranking generation is scripted to think out loud, the shape a real
    `include_thoughts=True` generation returns
    (`planner.settings.AGENT_THINKING_CONFIG`). Nothing about that response
    ever reaches the model again: the closing generation's own request --
    the ranking turn's next, and last, one -- carries the ranking call and
    its `needs_decision` answer, but no `thought`-marked part, because
    `after_model_callback=drop_thought_parts` (`create_translator_agent`)
    already stripped it from the response before ADK turned it into the
    session event this request replays.
    """
    dataset = requirement(id="gdp_gtm", measure="aid disbursed")
    plan = Plan(
        question="aid to guatemala",
        decisions=[decision()],
        evidence_requirements=[dataset, derived_requirement(id="gdp_total", input_id="gdp_gtm")],
        status=PlanStatus.DRAFT,
    )
    script: list[Any] = [
        {
            "thought": "Weighing the two candidates against the requirement.",
            "calls": [
                (
                    "translate",
                    {"requirement_id": "gdp_gtm", "choice": SPEC_OTHER, "rationale": "closer"},
                )
            ],
        },
        _retype_the_referral_translate_reported,
    ]

    _ranking, closer = await _translator_requests(plan, combined_surface, script=script)

    closer_thoughts = [
        part
        for content in closer.contents
        if content.role == "model"
        for part in content.parts or ()
        if part.thought
    ]
    assert closer_thoughts == []
    # The ranking call itself still reaches the closer -- only the thought is gone.
    calls = [
        part.function_call
        for content in closer.contents
        if content.role == "model"
        for part in content.parts or ()
        if part.function_call is not None
    ]
    assert any(call.name == "translate" for call in calls)


async def test_the_opener_never_fires_once_a_tool_response_exists(
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    decision: DecisionFactory,
    combined_surface: McpSurface,
) -> None:
    """A resumed turn carries `translate`'s answer already, so the model is asked for its ranking.

    Firing the opener here would replace the pending `needs_decision` with a
    fresh `translate()`, dropping the ranking the model was asked for, and
    would keep doing it: the plan never compiles, every generation reads a
    tool response, and the turn only stops when the model's script runs out.

    Bounded on both sides. `tool_calls` pins the number of times `translate`
    ran, so a turn that re-opened itself shows up as extra calls rather than
    as a passing test that merely took longer. `_ScriptedLlm.max_generations`
    turns a turn that cannot settle into an assertion instead of a hang.
    """
    dataset = requirement(id="gdp_gtm", measure="aid disbursed")
    plan = Plan(
        question="aid to guatemala",
        decisions=[decision()],
        evidence_requirements=[dataset, derived_requirement(id="gdp_total", input_id="gdp_gtm")],
        status=PlanStatus.DRAFT,
    )
    seen: list[Any] = []
    closed: list[dict[str, Any]] = []
    tool_calls: dict[str, int] = {}

    def _ranking_call(request: Any) -> list[tuple[str, dict[str, Any]]]:
        seen.append(_last_tool_response(request))
        return [
            (
                "translate",
                {"requirement_id": "gdp_gtm", "choice": SPEC_OTHER, "rationale": "closer"},
            )
        ]

    def _close(request: Any) -> list[tuple[str, dict[str, Any]]]:
        # `_retype_the_referral_translate_reported` reads the last tool
        # response of this request, which is `translate`'s answer to the
        # ranking. A dropped ranking would answer that call with a refusal
        # instead, and this raises rather than reaching the assertion.
        closed.append(_last_tool_response(request).response["referral"])
        return _retype_the_referral_translate_reported(request)

    await _translator_requests(
        plan, combined_surface, script=[_ranking_call, _close], tool_calls=tool_calls
    )

    assert tool_calls["translate"] == 2, tool_calls
    assert len(seen) == 1
    assert seen[0].name == "translate"
    assert seen[0].response["outcome"] == "needs_decision"
    assert seen[0].response["judgement"] == "candidate_ranking"
    assert [candidate["spec_id"] for candidate in seen[0].response["candidates"]]
    (referral,) = closed
    assert referral["kind"] == "source_ambiguous"
    assert referral["recommendation"] == {"spec_id": SPEC_OTHER, "rationale": "closer"}


async def test_a_second_standalone_invocation_reopens_the_turn_on_translate(
    draft: DraftFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    decision: DecisionFactory,
    combined_surface: McpSurface,
) -> None:
    """A standalone, conversational translator (`mode=None`) opens every invocation on `translate`, not only its first.

    `mode=None` runs with `include_contents="default"`, so a second
    `runner.run_async` call on the same session carries the whole prior
    history into its first request, the first invocation's own closing tool
    response included. Scoping the opener's detection to the current
    invocation (`_last_tool_response`) is what keeps that stale response from
    being read as this invocation's own turn having already opened.

    Proven by giving the second invocation a plan `translate` cannot settle
    without a judgement: with the opener firing again, `translate` runs
    fresh against it, finds the ambiguity, and the first real model call of
    the second invocation reads that fresh `needs_decision` answer -- one
    request to read it, a second to close on the ranking it answers.
    Reading the first invocation's stale `ok` instead would reach the model
    directly, with nothing to judge, collapsing the second invocation to one
    request instead of two -- the failure this asserts against.
    """
    translator = create_translator_agent(model_name="m", surface=combined_surface)
    dataset = requirement(id="gdp_gtm", measure="aid disbursed")
    ambiguous_plan = Plan(
        question="aid to guatemala",
        decisions=[decision()],
        evidence_requirements=[dataset, derived_requirement(id="gdp_total", input_id="gdp_gtm")],
        status=PlanStatus.DRAFT,
    )
    script: list[Any] = [
        '{"outcome": "ok"}',
        [("translate", {"requirement_id": "gdp_gtm", "choice": SPEC_OTHER, "rationale": "closer"})],
        _retype_the_referral_translate_reported,
    ]
    translator_llm = _ScriptedLlm(script=script)
    translator.model = translator_llm
    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name="app",
        user_id="u",
        session_id="s",
        state={PLAN_STATE_KEY: draft().model_dump_json()},
    )
    runner = Runner(agent=translator, app_name="app", session_service=session_service)

    async for _event in runner.run_async(
        user_id="u",
        session_id="s",
        new_message=types.Content(role="user", parts=[types.Part.from_text(text="translate this")]),
    ):
        pass
    assert len(translator_llm.captured_requests) == 1

    # A fresh draft, written the way a second turn's plan reaches state:
    # `save_plan`, keyed on `PLAN_STATE_KEY`. `get_session` hands back a
    # shallow copy, so the write goes straight to the service's own stored
    # session.
    canonical_session = session_service.sessions["app"]["u"]["s"]
    save_plan(canonical_session.state, ambiguous_plan)

    async for _event in runner.run_async(
        user_id="u",
        session_id="s",
        new_message=types.Content(
            role="user", parts=[types.Part.from_text(text="translate again")]
        ),
    ):
        pass

    requests = translator_llm.captured_requests
    # One request for turn 1's own closer, then two for turn 2: the
    # generation that reads the fresh `needs_decision` judgement, and the one
    # that closes on the referral it answers.
    assert len(requests) == 3, requests
    second_turn_opener_response = _last_tool_response(requests[1])
    assert second_turn_opener_response is not None
    assert second_turn_opener_response.name == "translate"
    assert second_turn_opener_response.response["outcome"] == "needs_decision"


async def test_a_second_translator_call_in_one_planner_invocation_reopens_on_translate(
    draft: DraftFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    decision: DecisionFactory,
    combined_surface: McpSurface,
) -> None:
    """The graph path: the planner can call `dfo_translator` more than once inside one invocation.

    `single_turn`'s `include_contents="none"` already scopes each request to
    the current run, but `_SingleTurnAgentTool.run_async` runs the translator
    through `tool_context.run_node`, inside the planner's own
    `InvocationContext` -- not a nested `Runner` with a session of its own --
    so a second `dfo_translator` call in the same invocation (a correction
    round) shares the first call's `invocation_id`. Detection scoped to
    invocation id alone would read the first call's own closing response as
    the second call's opener already having fired, the same failure `mode=None`
    has, moved into the graph path a correctly invocation-scoped-only check
    would still carry.

    Proven the way the standalone case above is: the plan behind the second
    `dfo_translator` call needs a judgement `translate` cannot make alone, so
    a correctly reopened run reads a fresh `needs_decision` answer, over two
    requests; a stale read of the first call's `ok` would reach the model
    directly, with nothing to judge, over one.
    """
    translator = create_translator_agent(
        model_name="m", surface=combined_surface, mode="single_turn"
    )
    dataset = requirement(id="gdp_gtm", measure="aid disbursed")
    ambiguous_plan = Plan(
        question="aid to guatemala",
        decisions=[decision()],
        evidence_requirements=[dataset, derived_requirement(id="gdp_total", input_id="gdp_gtm")],
        status=PlanStatus.DRAFT,
    )
    translator_script: list[Any] = [
        '{"outcome": "ok"}',
        [("translate", {"requirement_id": "gdp_gtm", "choice": SPEC_OTHER, "rationale": "closer"})],
        _retype_the_referral_translate_reported,
    ]
    translator_llm = _ScriptedLlm(script=translator_script)
    translator.model = translator_llm

    async def write_plan(tool_context: ToolContext, plan_json: str) -> dict[str, str]:
        """Stand in for the planner recording an amended plan mid-invocation."""
        save_plan(tool_context.state, Plan.model_validate_json(plan_json))
        return {"outcome": "ok"}

    # The scripted planner: calls `dfo_translator`, rewrites the plan (the
    # way an amendment would) once it reads that answer, then calls
    # `dfo_translator` again -- all inside this one `run_async` invocation.
    root_script: list[Any] = [
        [("dfo_translator", {"request": "first"})],
        [("write_plan", {"plan_json": ambiguous_plan.model_dump_json()})],
        [("dfo_translator", {"request": "second"})],
        "done",
    ]
    root_llm = _ScriptedLlm(script=root_script)
    root = LlmAgent(
        name="root",
        model=root_llm,
        sub_agents=[translator],
        tools=[FunctionTool(write_plan)],
    )
    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name="app",
        user_id="u",
        session_id="s",
        state={PLAN_STATE_KEY: draft().model_dump_json()},
    )
    runner = Runner(agent=root, app_name="app", session_service=session_service)
    async for _event in runner.run_async(
        user_id="u",
        session_id="s",
        new_message=types.Content(role="user", parts=[types.Part.from_text(text="go")]),
    ):
        pass

    requests = translator_llm.captured_requests
    # One request closes the first call; two more read the second call's own
    # fresh `needs_decision` judgement and close on the referral it answers.
    assert len(requests) == 3, requests
    second_call_response = _last_tool_response(requests[1])
    assert second_call_response is not None
    assert second_call_response.name == "translate"
    assert second_call_response.response["outcome"] == "needs_decision"


def _branch_event(
    *,
    author: str,
    invocation_id: str,
    branch: str | None,
    response: dict[str, object] | None = None,
) -> Event:
    """A real `Event`, carrying either `translate`'s call or its response, on `branch`."""
    content = (
        types.Content(
            role="model", parts=[types.Part.from_function_call(name="translate", args={})]
        )
        if response is None
        else types.Content(
            role="user",
            parts=[
                types.Part(
                    function_response=types.FunctionResponse(name="translate", response=response)
                )
            ],
        )
    )
    return Event(author=author, invocation_id=invocation_id, branch=branch, content=content)


def test_two_translator_runs_in_one_invocation_do_not_cross_read_each_others_response() -> None:
    """Two `dfo_translator` calls in one planner response run on sibling branches, and their events interleave.

    `_SingleTurnAgentTool.run_async` puts each call on its own sub-branch,
    `_BranchPath.create_sub_branch(base_branch, name=agent.name,
    run_id=function_call_id)`, keyed on that call's own function-call id --
    confirmed against the installed ADK, 2.9.1, by driving two such calls
    through a real `Runner` and reading the branch each run's own events
    landed on. Both runs share one `invocation_id`, so the invocation/author
    boundary alone cannot tell them apart; without a branch check, the
    second run's later response reads as the first run's own, since it is
    the last one in session order carrying that invocation id and author.
    This builds that interleaving directly -- both runs' own call and
    response events, second run's landing after the first's in session
    order -- and reads each run's `_last_tool_response` off its own
    `CallbackContext`, one built on each run's own branch.
    """
    session = Session(id="s", app_name="app", user_id="u")
    session.events = [
        _branch_event(author="planner", invocation_id="inv", branch=None),
        _branch_event(author="translator", invocation_id="inv", branch="root.translator@call1"),
        _branch_event(
            author="translator",
            invocation_id="inv",
            branch="root.translator@call1",
            response={"outcome": "ok", "call": "first"},
        ),
        _branch_event(author="translator", invocation_id="inv", branch="root.translator@call2"),
        _branch_event(
            author="translator",
            invocation_id="inv",
            branch="root.translator@call2",
            response={"outcome": "ok", "call": "second"},
        ),
    ]
    agent = LlmAgent(name="translator")
    session_service = InMemorySessionService()

    def context_on(branch: str) -> CallbackContext:
        return CallbackContext(
            InvocationContext(
                session_service=session_service,
                invocation_id="inv",
                branch=branch,
                agent=agent,
                session=session,
            )
        )

    first_run_response = _translator_last_tool_response(context_on("root.translator@call1"))
    second_run_response = _translator_last_tool_response(context_on("root.translator@call2"))

    assert first_run_response is not None
    assert first_run_response.response["call"] == "first"
    assert second_run_response is not None
    assert second_run_response.response["call"] == "second"

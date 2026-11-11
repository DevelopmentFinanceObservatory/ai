"""The Pydantic AI planner's agent and run, on a scripted google-genai client.

The run tests drive `run_planner` over small tool specs whose bodies say
what each call does, so each test reads the run loop alone. The request test
builds the ADK planner and the Pydantic AI planner for the same turn and
compares the first request each sends.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Any, cast

import pytest
from compiler.surface import ToolSurface
from google.genai import types
from harness.turn_events import PLANNER_ASK
from planner.call import PlannerCall
from planner.pydantic_agent import (
    PLANNER_PROFILE,
    PlannerAgent,
    PlannerDeps,
    PlannerItem,
    PlannerResponse,
    PlannerToolAnswer,
    planner_agent,
    run_planner,
)
from planner.specs import PROPOSE_PLAN, planner_tool_specs
from planner.tool_spec import Parameter, ToolSpec
from planner.tools import ASK_ACCEPTED, EndRun, load_pending_ask
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

from ..scripted_genai import (
    Reply,
    Request,
    ScriptedGenai,
    call_part,
    reply,
    text_part,
    thought_part,
    usage_metadata,
)
from .planner_pairs import BRIEF, QUESTION, adk_run, assert_pinned_wire_differences, pydantic_run

_MODEL = "gemini-3.8-flash"
_QUESTION = QUESTION
_BRIEF = BRIEF
_ACCEPTED = {"outcome": "ok", "status": "ready"}
_REFUSED = {"outcome": "refused", "code": "plan_invalid", "message": "Pin a source."}


class _Run:
    """What the run's callbacks were told, and the answer `tool_failed` gives."""

    def __init__(self) -> None:
        self.answered: list[tuple[str, dict[str, Any]]] = []
        self.failed: list[tuple[str, BaseException]] = []

    def tool_answered(self, name: str, answer: dict[str, Any]) -> None:
        self.answered.append((name, answer))

    def tool_failed(self, name: str, error: BaseException) -> dict[str, Any]:
        self.failed.append((name, error))
        return {"error": f"{name} failed"}


def _propose(*answers: dict[str, Any], writes: dict[str, Any] | None = None) -> ToolSpec:
    """A `propose_plan` answering `answers` in turn, ending the run on an `ok` one
    and writing `writes` to state on every call."""
    left = list(answers)

    def body(call: PlannerCall, end_run: EndRun, arguments: dict[str, Any]) -> object:
        del arguments
        for key, value in (writes or {}).items():
            call.state[key] = value
        answer = left.pop(0)
        if answer.get("outcome") == "ok":
            end_run()
        return answer

    return ToolSpec(
        name=PROPOSE_PLAN,
        description="Propose the plan.",
        parameters=(Parameter("question", str | None, None),),
        body=body,
    )


def _write(key: str, value: object) -> ToolSpec:
    """A tool that writes `value` to state under `key` and answers `ok`."""

    def body(call: PlannerCall, end_run: EndRun, arguments: dict[str, Any]) -> object:
        del end_run, arguments
        call.state[key] = value
        return {"outcome": "ok"}

    return ToolSpec(name=f"write_{key}", description="Write.", parameters=(), body=body)


def _ask_spec() -> ToolSpec:
    return next(spec for spec in planner_tool_specs() if spec.name == PLANNER_ASK)


def _google_model(scripted: ScriptedGenai) -> GoogleModel:
    return GoogleModel(
        _MODEL, provider=GoogleProvider(client=scripted.client), profile=PLANNER_PROFILE
    )


async def _run(
    specs: Sequence[ToolSpec],
    replies: Sequence[Reply],
    *,
    state: dict[str, Any] | None = None,
    run: _Run | None = None,
) -> tuple[list[PlannerItem], ScriptedGenai]:
    scripted = ScriptedGenai(replies)
    agent = planner_agent(specs, instruction="Plan.", model=_google_model(scripted))
    deps = _deps(state, run)
    items = [item async for item in run_planner(agent, deps, question=_QUESTION, brief=_BRIEF)]
    return items, scripted


def _deps(state: dict[str, Any] | None = None, run: _Run | None = None) -> PlannerDeps:
    run = run or _Run()
    return PlannerDeps(
        state={} if state is None else state,
        earlier=(),
        tool_answered=run.tool_answered,
        tool_failed=run.tool_failed,
    )


def _kinds(items: Sequence[PlannerItem]) -> list[str]:
    return [
        "response" if isinstance(item, PlannerResponse) else f"answer:{item.name}" for item in items
    ]


async def test_an_accepted_proposal_ends_the_run_after_one_model_request() -> None:
    signed = thought_part("Weighing the sources.", signature=b"sig-1")
    call = call_part(PROPOSE_PLAN, {"question": "q"}, id="p1", signature=b"sig-2")
    run = _Run()

    items, scripted = await _run(
        [_propose(_ACCEPTED)],
        [
            reply(signed, call, usage=usage_metadata(prompt=40, candidates=5)),
            reply(text_part("x")),
        ],
        run=run,
    )

    assert len(scripted.requests) == 1
    assert _kinds(items) == ["response", f"answer:{PROPOSE_PLAN}"]
    response, answer = items
    assert isinstance(response, PlannerResponse)
    assert response.parts == (signed, call)
    assert answer == PlannerToolAnswer(PROPOSE_PLAN, "p1", _ACCEPTED, {}, ended=True)
    assert run.answered == [(PROPOSE_PLAN, _ACCEPTED)]


async def test_a_refused_proposal_goes_on_to_a_second_request_that_sees_the_refusal() -> None:
    items, scripted = await _run(
        [_propose(_REFUSED, _ACCEPTED)],
        [
            reply(call_part(PROPOSE_PLAN, {}, id="p1")),
            reply(call_part(PROPOSE_PLAN, {"question": "q"}, id="p2")),
        ],
    )

    assert len(scripted.requests) == 2
    assert _kinds(items) == ["response", f"answer:{PROPOSE_PLAN}"] * 2
    returned = scripted.requests[1].content_list[-1].parts[0].function_response
    assert (returned.id, returned.name, returned.response) == ("p1", PROPOSE_PLAN, _REFUSED)


async def test_an_accepted_ask_ends_the_run_and_leaves_the_pending_ask_in_state() -> None:
    state: dict[str, Any] = {}
    args = {"message": "Which donor do you mean?"}

    items, scripted = await _run(
        [_ask_spec()],
        [reply(call_part(PLANNER_ASK, args, id="a1")), reply(text_part("x"))],
        state=state,
    )

    assert len(scripted.requests) == 1
    assert _kinds(items) == ["response", f"answer:{PLANNER_ASK}"]
    assert items[-1] == PlannerToolAnswer(
        PLANNER_ASK,
        "a1",
        dict(ASK_ACCEPTED),
        {"planner:pending_ask": state["planner:pending_ask"]},
        ended=True,
    )
    pending = load_pending_ask(state)
    assert pending is not None
    assert (pending.call_id, pending.args["message"]) == ("a1", args["message"])


async def test_each_answer_carries_the_state_its_own_call_wrote_and_nothing_else() -> None:
    state: dict[str, Any] = {"kept": "as it was"}

    items, _ = await _run(
        [_write("a", 1), _write("b", 2), _propose(_REFUSED, writes={"a": 3})],
        [
            reply(call_part("write_a", id="w1"), call_part("write_b", id="w2")),
            reply(call_part(PROPOSE_PLAN, {}, id="p1")),
            reply(text_part("Nothing more to do.")),
        ],
        state=state,
    )

    answers = [item for item in items if isinstance(item, PlannerToolAnswer)]
    assert sorted((a.call_id, a.writes) for a in answers) == [
        ("p1", {"a": 3}),
        ("w1", {"a": 1}),
        ("w2", {"b": 2}),
    ]
    assert state == {"kept": "as it was", "a": 3, "b": 2}
    assert _kinds(items)[-1] == "response"


async def test_a_raising_tool_is_answered_through_tool_failed_and_the_run_goes_on() -> None:
    error = RuntimeError("broken")

    def body(call: PlannerCall, end_run: EndRun, arguments: dict[str, Any]) -> object:
        raise error

    broken = ToolSpec(name="broken", description="Breaks.", parameters=(), body=body)
    run = _Run()

    items, scripted = await _run(
        [broken], [reply(call_part("broken", id="b1")), reply(text_part("done"))], run=run
    )

    assert run.failed == [("broken", error)]
    assert run.answered == [("broken", {"error": "broken failed"})]
    assert len(scripted.requests) == 2
    returned = scripted.requests[1].content_list[-1].parts[0].function_response
    assert returned.response == {"error": "broken failed"}
    assert _kinds(items) == ["response", "answer:broken", "response"]


def _with_tool(agent: PlannerAgent, function: Callable[..., Any]) -> PlannerAgent:
    """`agent` with `function` registered as a Pydantic AI tool of its own, as
    the planner's tools are, to reach answers no planner tool gives. Pydantic
    AI validates its typed arguments and lets what the function raises through."""
    agent.agent.tool_plain(function)
    return PlannerAgent(agent=agent.agent, tools=(*agent.tools, function.__name__))


async def test_a_call_pydantic_ai_answers_with_a_retry_still_gets_its_answer() -> None:
    """A call whose arguments fail validation is answered by Pydantic AI with a
    retry. That answer is yielded as an error, as Gemini is sent it, so every
    answer of the response is yielded, its sibling's write with them."""

    def counted(n: int) -> dict[str, Any]:
        return {"outcome": "ok", "n": n}

    scripted = ScriptedGenai(
        [
            reply(call_part("write_a", id="w1"), call_part("counted", {"n": "many"}, id="c1")),
            reply(text_part("done")),
        ]
    )
    agent = _with_tool(
        planner_agent([_write("a", 1)], instruction="Plan.", model=_google_model(scripted)),
        counted,
    )

    items = [item async for item in run_planner(agent, _deps(), question=_QUESTION, brief=_BRIEF)]

    assert _kinds(items)[0] == "response" and _kinds(items)[-1] == "response"
    answers = {item.call_id: item for item in items[1:-1] if isinstance(item, PlannerToolAnswer)}
    assert answers.keys() == {"w1", "c1"}
    assert answers["w1"] == PlannerToolAnswer("write_a", "w1", {"outcome": "ok"}, {"a": 1})
    assert answers["c1"].name == "counted"
    assert set(answers["c1"].answer) == {"error"}


async def test_a_run_pydantic_ai_ends_after_a_call_answered_yields_that_answer() -> None:
    """A call raising `UnexpectedModelBehavior` ends the run without raising, and
    the answer of the sibling that answered before it is yielded."""
    answered = asyncio.Event()
    run = _Run()

    def tool_answered(name: str, answer: dict[str, Any]) -> None:
        run.tool_answered(name, answer)
        answered.set()

    async def misbehaves() -> None:
        await answered.wait()
        raise UnexpectedModelBehavior("the model misbehaved")

    scripted = ScriptedGenai(
        [reply(call_part("write_a", id="w1"), call_part("misbehaves", id="m1"))]
    )
    agent = _with_tool(
        planner_agent([_write("a", 1)], instruction="Plan.", model=_google_model(scripted)),
        misbehaves,
    )
    deps = PlannerDeps(
        state={}, earlier=(), tool_answered=tool_answered, tool_failed=run.tool_failed
    )

    items = [item async for item in run_planner(agent, deps, question=_QUESTION, brief=_BRIEF)]

    assert items[1:] == [PlannerToolAnswer("write_a", "w1", {"outcome": "ok"}, {"a": 1})]


# --- The first request equals the ADK planner's --------------------------------------


async def _translate(call: PlannerCall) -> dict[str, Any]:
    del call
    return {}


def _surface() -> ToolSurface:
    # Neither planner calls a tool before its first request, so nothing reads it.
    return cast(ToolSurface, SimpleNamespace())


async def _first_requests(surface: ToolSurface | None) -> tuple[Request, Request]:
    script = [reply(text_part("done"))]
    (adk,) = (await adk_run(script, surface=surface, translate=_translate)).requests
    (pydantic,) = (await pydantic_run(script, surface=surface, translate=_translate)).requests
    return adk, pydantic


@pytest.mark.parametrize("with_skills", [False, True], ids=["no-skill-tool", "skill-tool"])
async def test_the_first_request_equals_the_adk_planners_apart_from_the_pinned_wire_differences(
    with_skills: bool,
) -> None:
    surface = _surface() if with_skills else None
    adk_request, pydantic_request = await _first_requests(surface)

    assert_pinned_wire_differences(adk_request, pydantic_request)
    assert adk_request.content_list == [
        types.Content(role="user", parts=[types.Part(text=_QUESTION)]),
        types.Content(role="user", parts=[types.Part(text=_BRIEF)]),
    ]
    (adk_tool,) = adk_request.generate_config.tools or []
    assert [d.name for d in adk_tool.function_declarations or []] == [
        *(["get_skill"] if with_skills else []),
        PLANNER_ASK,
        PROPOSE_PLAN,
        "reject_plan",
    ]
    thinking = adk_request.generate_config.thinking_config
    assert thinking is not None
    assert thinking.model_dump(mode="json", exclude_none=True) == {
        "thinking_level": "LOW",
        "include_thoughts": True,
    }

"""The Pydantic AI planner's agent and run, on a scripted google-genai client.

The run tests drive `run_planner` over small tool specs whose bodies say
what each call does, so each test reads the run loop alone. The request test
builds the ADK planner and the Pydantic AI planner for the same turn and
compares the first request each sends.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any, cast

import pytest
from compiler.surface import ToolSurface
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.models import Gemini
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.workflow import START, Workflow, node
from google.genai import types
from harness.turn_events import PLANNER_ASK
from planner.agent import BRIEF_STATE_KEY, create_planner_agent, planner_instruction
from planner.call import PlannerCall
from planner.pydantic_agent import (
    PlannerCallWrites,
    PlannerDeps,
    PlannerItem,
    PlannerResponse,
    PlannerToolAnswer,
    planner_agent,
    planner_profile,
    run_planner,
)
from planner.specs import PROPOSE_PLAN, planner_tool_specs, skill_tool_spec
from planner.tool_spec import Parameter, ToolSpec
from planner.tools import ASK_ACCEPTED, EndRun, load_pending_ask
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

from ..scripted_genai import (
    Reply,
    ScriptedGenai,
    call_part,
    reply,
    text_part,
    thought_part,
    usage_metadata,
)

_MODEL = "gemini-3.8-flash"
_QUESTION = "How much aid did Ghana receive?"
_BRIEF = "Context for this question: the readings."
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
        _MODEL, provider=GoogleProvider(client=scripted.client), profile=planner_profile(_MODEL)
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
        [reply(signed, call, usage=usage_metadata(prompt=40, candidates=5)), reply(text_part("x"))],
        run=run,
    )

    assert len(scripted.requests) == 1
    assert _kinds(items) == ["response", f"answer:{PROPOSE_PLAN}"]
    response, answer = items
    assert isinstance(response, PlannerResponse)
    assert response.parts == (signed, call)
    assert (response.usage.input_tokens, response.usage.output_tokens) == (40, 5)
    assert answer == PlannerToolAnswer(PROPOSE_PLAN, "p1", _ACCEPTED, {})
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
        PLANNER_ASK, "a1", dict(ASK_ACCEPTED), {"planner:pending_ask": state["planner:pending_ask"]}
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


async def test_a_cancelled_run_has_yielded_the_finished_call_and_holds_the_running_ones_writes() -> (
    None
):
    async def stuck(call: PlannerCall, end_run: EndRun, arguments: dict[str, Any]) -> object:
        del end_run, arguments
        call.state["stuck"] = "started"
        await asyncio.Event().wait()
        return {"outcome": "ok"}

    scripted = ScriptedGenai(
        [reply(call_part("write_a", id="w1"), call_part("stuck", id="s1")), reply(text_part("x"))]
    )
    agent = planner_agent(
        [_write("a", 1), ToolSpec(name="stuck", description="Hangs.", parameters=(), body=stuck)],
        instruction="Plan.",
        model=_google_model(scripted),
    )
    deps = _deps()
    items: list[PlannerItem] = []
    answered = asyncio.Event()

    async def consume() -> None:
        async for item in run_planner(agent, deps, question=_QUESTION, brief=_BRIEF):
            items.append(item)
            if isinstance(item, PlannerToolAnswer):
                answered.set()

    task = asyncio.create_task(consume())
    await asyncio.wait_for(answered.wait(), timeout=5)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    assert items[-1] == PlannerToolAnswer("write_a", "w1", {"outcome": "ok"}, {"a": 1})
    assert deps.unyielded_writes() == [PlannerCallWrites("stuck", "s1", {"stuck": "started"})]


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


# --- The first request equals the ADK planner's --------------------------------------


async def _translate(call: PlannerCall) -> dict[str, Any]:
    del call
    return {}


def _surface() -> ToolSurface:
    # Neither planner calls a tool before its first request, so nothing reads it.
    return cast(ToolSurface, SimpleNamespace())


async def _adk_first_request(surface: ToolSurface | None) -> Any:
    scripted = ScriptedGenai([reply(text_part("done"))])
    agent = create_planner_agent(
        model_name=_MODEL,
        translate=_translate,
        surface=surface,
        question_fixed=True,
        reads_user_words=False,
    )
    agent.model = Gemini(model=_MODEL, client=scripted.client)

    async def _planner(ctx: Any, node_input: Any) -> None:
        del node_input
        ctx.state[BRIEF_STATE_KEY] = _BRIEF
        await ctx.run_node(agent, None, override_branch="dfo_planner")

    workflow = Workflow(
        name="turn", edges=[(START, node(_planner, name="planner", rerun_on_resume=True))]
    )
    sessions = InMemorySessionService()
    session = await sessions.create_session(app_name="app", user_id="user")
    runner = Runner(node=workflow, app_name="app", session_service=sessions)
    async for _ in runner.run_async(
        user_id="user",
        session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(text=_QUESTION)]),
        run_config=RunConfig(streaming_mode=StreamingMode.SSE),
    ):
        pass
    (request,) = scripted.requests
    return request


async def _pydantic_first_request(surface: ToolSurface | None) -> Any:
    scripted = ScriptedGenai([reply(text_part("done"))])
    specs = planner_tool_specs(
        translate=_translate, question_fixed=True, reads_user_words=False, surface=surface
    )
    agent = planner_agent(
        [*([skill_tool_spec(surface)] if surface is not None else []), *specs],
        instruction=planner_instruction(
            has_translator=True, has_skills=surface is not None, question_fixed=True
        ),
        model=_google_model(scripted),
    )
    async for _ in run_planner(agent, _deps(), question=_QUESTION, brief=_BRIEF):
        pass
    (request,) = scripted.requests
    return request


def _dumped(request: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = request.generate_config.model_dump(mode="json", exclude_none=True)
    contents = [
        content.model_dump(mode="json", exclude_none=True) for content in request.content_list
    ]
    return config, contents


@pytest.mark.parametrize("with_skills", [False, True], ids=["no-skill-tool", "skill-tool"])
async def test_the_first_request_equals_the_adk_planners_apart_from_the_pinned_wire_differences(
    with_skills: bool,
) -> None:
    surface = _surface() if with_skills else None
    adk, adk_contents = _dumped(await _adk_first_request(surface))
    pydantic, pydantic_contents = _dumped(await _pydantic_first_request(surface))

    assert (
        pydantic_contents
        == adk_contents
        == [
            {"role": "user", "parts": [{"text": _QUESTION}]},
            {"role": "user", "parts": [{"text": _BRIEF}]},
        ]
    )
    differing = {key for key in adk.keys() | pydantic.keys() if adk.get(key) != pydantic.get(key)}
    assert differing == {
        "system_instruction",
        "tools",
        "tool_config",
        "response_modalities",
        "automatic_continuation",
        "http_options",
    }
    # The instruction as one user `Content` where the ADK planner sends text.
    assert pydantic["system_instruction"] == {
        "role": "user",
        "parts": [{"text": adk["system_instruction"]}],
    }
    # One `Tool` per declaration where the ADK planner sends one `Tool` of them all, and
    # `reject_plan`, which takes nothing, declared with an empty object schema.
    (adk_tool,) = adk["tools"]
    declared = adk_tool["function_declarations"]
    assert [d["name"] for d in declared] == [
        *(["get_skill"] if with_skills else []),
        PLANNER_ASK,
        PROPOSE_PLAN,
        "reject_plan",
    ]
    assert pydantic["tools"] == [
        {
            "function_declarations": [
                declaration
                if "parameters_json_schema" in declaration
                else {**declaration, "parameters_json_schema": {"type": "object", "properties": {}}}
            ]
        }
        for declaration in declared
    ]
    # An explicit `AUTO` where the ADK planner leaves the mode unset, which is `AUTO`.
    assert pydantic["tool_config"] == {"function_calling_config": {"mode": "AUTO"}}
    assert "tool_config" not in adk
    assert pydantic["response_modalities"] == ["TEXT"]
    assert "response_modalities" not in adk
    # Unset, which google-genai reads as on, where the ADK planner sets it on.
    assert adk["automatic_continuation"] is True
    assert "automatic_continuation" not in pydantic
    # Each framework's own client headers.
    assert adk["http_options"].keys() == pydantic["http_options"].keys() == {"headers"}
    assert adk["thinking_config"] == {"thinking_level": "LOW", "include_thoughts": True}

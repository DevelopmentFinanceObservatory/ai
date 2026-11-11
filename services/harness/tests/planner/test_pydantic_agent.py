"""The Pydantic AI planner's agent and run, on a scripted google-genai client.

The run tests drive `run_planner` over small tool specs whose bodies say
what each call does, so each test reads the run loop alone. The request test
builds the ADK planner and the Pydantic AI planner for the same turn and
compares the first request each sends.
"""

from __future__ import annotations

from collections.abc import MutableMapping, Sequence
from types import SimpleNamespace
from typing import Any, cast

import pytest
from compiler.surface import ToolSurface
from google.genai import types
from harness.turn_events import PLANNER_ASK
from planner.call import PlannerCall
from planner.pydantic_agent import (
    PLANNER_PROFILE,
    PlannerAnswers,
    PlannerDeps,
    PlannerItem,
    PlannerResponse,
    UnsentQuestionPartError,
    planner_agent,
    run_planner,
)
from planner.session import SessionCall
from planner.specs import PROPOSE_PLAN, planner_tool_specs
from planner.tool_spec import Parameter, ToolSpec
from planner.tools import ASK_ACCEPTED, EndRun, load_pending_ask
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

from ..planner_expectations import (
    STORING_RUNNER,
    expect,
    expectation_name,
    expected,
    on_each_runner,
    shared,
)
from ..planner_seam_runners import PlannerRunner as Runner
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
from ..test_planner_node import _as_stored, _request_dumped, _stored_request
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


def _no_events(state: MutableMapping[str, Any], call_id: str | None) -> SessionCall:
    """A tool call over a session with no events."""
    return SessionCall([], state, call_id)


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
        call_for=_no_events,
        tool_answered=run.tool_answered,
        tool_failed=run.tool_failed,
    )


def _kinds(items: Sequence[PlannerItem]) -> list[str]:
    return [
        "response"
        if isinstance(item, PlannerResponse)
        else "answers:" + ",".join(answer.name or "" for answer in item.answers)
        for item in items
    ]


def _answers(*answers: tuple[str, str, dict[str, Any]], state: dict[str, Any]) -> PlannerAnswers:
    """The answers of a response's calls, each given as its tool, id and answer."""
    return PlannerAnswers(
        answers=tuple(
            types.FunctionResponse(id=call_id, name=name, response=answer)
            for name, call_id, answer in answers
        ),
        state=state,
    )


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
    assert _kinds(items) == ["response", f"answers:{PROPOSE_PLAN}"]
    response, answers = items
    assert isinstance(response, PlannerResponse)
    assert response.parts == (signed, call)
    assert answers == _answers((PROPOSE_PLAN, "p1", _ACCEPTED), state={})
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
    assert _kinds(items) == ["response", f"answers:{PROPOSE_PLAN}"] * 2
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
    assert _kinds(items) == ["response", f"answers:{PLANNER_ASK}"]
    assert items[-1] == _answers(
        (PLANNER_ASK, "a1", dict(ASK_ACCEPTED)),
        state={"planner:pending_ask": state["planner:pending_ask"]},
    )
    pending = load_pending_ask(state)
    assert pending is not None
    assert (pending.call_id, pending.args["message"]) == ("a1", args["message"])


async def test_each_responses_answers_carry_the_state_its_calls_wrote_and_nothing_else() -> None:
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

    answers = [item for item in items if isinstance(item, PlannerAnswers)]
    assert [(len(a.answers), a.state) for a in answers] == [(2, {"a": 1, "b": 2}), (1, {"a": 3})]
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
    assert _kinds(items) == ["response", "answers:broken", "response"]


@pytest.mark.parametrize(
    "part",
    [
        types.Part(file_data=types.FileData(file_uri="gs://b/f.pdf", mime_type="x/pdf")),
        types.Part(inline_data=types.Blob(mime_type="x/pdf")),
    ],
    ids=["file-by-uri", "inline-data-without-bytes"],
)
async def test_a_question_part_the_planner_cannot_send_is_refused_before_any_request(
    part: types.Part,
) -> None:
    scripted = ScriptedGenai([reply(text_part("done"))])
    agent = planner_agent([], instruction="Plan.", model=_google_model(scripted))
    question = types.Content(role="user", parts=[types.Part(text="kenya debt"), part])

    with pytest.raises(UnsentQuestionPartError):
        async for _ in run_planner(agent, _deps(), question=question, brief=_BRIEF):
            pass

    assert scripted.requests == []


# --- The first request equals the ADK planner's --------------------------------------


async def _translate(call: PlannerCall) -> dict[str, Any]:
    del call
    return {}


def _surface() -> ToolSurface:
    # Neither planner calls a tool before its first request, so nothing reads it.
    return cast(ToolSurface, SimpleNamespace())


async def _first_request(runner: Runner, surface: ToolSurface | None) -> Request:
    script = [reply(text_part("done"))]
    run = adk_run if runner == "adk" else pydantic_run
    (request,) = (await run(script, surface=surface, translate=_translate)).requests
    return request


@on_each_runner
@pytest.mark.parametrize("with_skills", [False, True], ids=["no-skill-tool", "skill-tool"])
async def test_the_first_request_is_the_same_on_both_planners_apart_from_the_pinned_wire_differences(
    with_skills: bool, runner: Runner, request: pytest.FixtureRequest
) -> None:
    surface = _surface() if with_skills else None
    sent = await _first_request(runner, surface)

    name = expectation_name(request, runner=runner)
    if runner == STORING_RUNNER:
        dumped = _request_dumped(sent)
        if with_skills:
            dumped["config"] = shared(dumped["config"], "request_config_recorded_entry")
        expect(name, dumped)
    else:
        assert_pinned_wire_differences(_as_stored(sent), _stored_request(expected(name)))
    assert sent.content_list == [
        types.Content(role="user", parts=[types.Part(text=_QUESTION)]),
        types.Content(role="user", parts=[types.Part(text=_BRIEF)]),
    ]
    (tool,) = sent.generate_config.tools or []
    assert [d.name for d in tool.function_declarations or []] == [
        *(["get_skill"] if with_skills else []),
        PLANNER_ASK,
        PROPOSE_PLAN,
        "reject_plan",
    ]
    thinking = sent.generate_config.thinking_config
    assert thinking is not None
    assert thinking.model_dump(mode="json", exclude_none=True) == {
        "thinking_level": "LOW",
        "include_thoughts": True,
    }

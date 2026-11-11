"""`PlannerSeam` on the Pydantic AI planner produces each entry's stored expectation.

Each entry runs on one script of a scripted google-genai client, through the
workflow node `tests.test_planner_seam._SeamHarness` drives, in a session of its
own, under Studio's streaming run config, and is checked against the entry's
stored expectation (`tests.planner_expectations`) on what the seam answers, the
progress it reports, the requests the model received, the events the session
stores, the steps `session_steps` reads from them, and the stored state. The
The Studio tests run a whole turn through
`/api/studio/turn` and check the frames it streams.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import gc
import itertools
import json
import logging
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast
from unittest import mock

import aiohttp
import httpx
import pytest
from compiler.surface import FixtureSurface
from google import genai
from google.adk import Context, Event
from google.adk.agents.invocation_context import LlmCallsLimitExceededError
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.runners import Runner as AdkRunner
from google.adk.workflow import START, Workflow, node
from google.adk.workflow.utils._workflow_hitl_utils import create_request_input_response
from google.genai import errors, types
from harness import planner_node as planner_node_module
from harness import planner_run, planner_seam, usage
from harness.planner_run import PlannerRun, PlannerSeamError
from harness.planner_seam import PlannerSeam
from harness.scripted import _compiled_proposal
from harness.sentences import SeamCode, refusal_sentence
from harness.settings import Settings
from harness.turn_budget import TurnBudget, current_turn_budget
from harness.turn_events import PLANNER_ASK
from planner.call import PlannerCall, elected_spec_ids
from planner.plan import (
    DatasetRequirement,
    PinSource,
    Plan,
    PlanStatus,
    SessionState,
    load_plan,
    save_pin_sources,
    save_plan,
)
from planner.session import SessionCall, session_steps
from planner.settings import AGENT_NAME
from planner.specs import PROPOSE_PLAN
from planner.tool_results import Outcome, ProblemCode
from planner.tool_spec import ToolSpec
from planner.tools import EndRun
from submission.service import scope_choice_decisions
from translator.tool import Translate, build_translate

from .orchestrator_fakes import plan_call, stream, studio_client, turn_body
from .planner.planner_pairs import seam_specs
from .planner_expectations import (
    BRIEF_OPENING,
    expect,
    expectation_name,
    shared,
    versionless_headers,
)
from .scripted_genai import (
    Reply,
    Request,
    ScriptedGenai,
    blocked_reply,
    call_part,
    empty_reply,
    live_call_reply,
    malformed_call_reply,
    reply,
    text_part,
    thought_part,
    usage_metadata,
)
from .scripted_planner import MODEL as _MODEL
from .scripted_planner import SIGNED as _SIGNED
from .scripted_planner import SSE as _SSE
from .scripted_planner import Tools, scripted_planner
from .scripted_planner import numbered_call_ids as _numbered_call_ids
from .test_planner_model import _HangingGenai
from .test_planner_seam import (
    _APP,
    _USER,
    _pending_ask_interrupt_id,
    _SeamHarness,
    _sector_switched,
    _translated_plan,
)
from .test_studio import _app, _interrupt_body, _pick, _text
from .test_turn_records import _switchable_scope

_USAGE = usage_metadata(prompt=120, cached=40, candidates=9, thoughts=4)
_QUESTION = "kenya debt"
_REFUSED = {"outcome": "refused", "code": "plan_invalid", "message": "Pin a source."}
_SPEC = "spec_" + "c" * 32
_SPEC_NEW = "spec_" + "d" * 32
# Where the test's `propose_plan` records the spec_ids the user picked.
_ELECTED_KEY = "planner:elected_spec_ids"
# Where the test's `propose_plan` records the steps and response calls it read.
_READ_KEY = "planner:read"


@pytest.fixture
def plan(decision: Any, requirement: Any, derived_requirement: Any) -> Plan:
    return _translated_plan(decision, requirement, derived_requirement)


# --- The planner's tools and runs -------------------------------------------------


def _proposals(*answers: dict[str, Any], records: Plan | None = None) -> Tools:
    """The planner's tools with a `propose_plan` that answers `answers` in turn,
    and then as a proposal whose translation compiled the plan on record
    (`harness.scripted.use_compiled_proposals`), recording `records` first when
    the session holds no plan. Every call records the spec_ids the user picked,
    read from its steps, under `_ELECTED_KEY`, and the steps and response calls
    it read under `_READ_KEY`. Each seam built gets `answers` afresh."""

    def tools(specs: Sequence[ToolSpec]) -> list[ToolSpec]:
        left = list(answers)

        def body(call: PlannerCall, end_run: EndRun, arguments: dict[str, Any]) -> object:
            call.state[_ELECTED_KEY] = sorted(elected_spec_ids(call))
            call.state[_READ_KEY] = {
                "steps": [[step.kind, step.call_id] for step in call.steps()],
                "response_calls": [list(response) for response in call.response_calls()],
            }
            if left:
                return left.pop(0)
            if records is not None and load_plan(call.state) is None:
                save_plan(call.state, records)
            return _compiled_proposal(call, end_run, arguments)

        return _serving_proposals(specs, body)

    return tools


def _serving_proposals(specs: Sequence[ToolSpec], body: Callable[..., object]) -> list[ToolSpec]:
    """`specs` with a `propose_plan` that runs `body`."""
    proposal = ToolSpec(
        name=PROPOSE_PLAN, description="Propose the plan.", parameters=(), body=body
    )
    return [proposal if spec.name == PROPOSE_PLAN else spec for spec in specs]


@dataclass
class _Entry:
    """What an entry's runs answered, reported, sent and left."""

    outputs: list[dict[str, Any] | None]
    reports: list[dict[str, Any]]
    requests: list[Request]
    events: list[Event]
    state: dict[str, Any]


@dataclass
class _Script:
    """An entry's script: the model's replies, the seam's arguments for each run,
    which the user's `answer` to the pending ask resumes, how the seam is set up,
    and the budget each run is given."""

    replies: Sequence[Reply]
    runs: Sequence[Mapping[str, Any]]
    setup: Tools
    budget: Callable[[], TurnBudget] | None = None
    translate: Translate | None = None


async def _entry(
    script: _Script,
    *,
    then: Callable[[PlannerSeam, _SeamHarness], Any] | None = None,
) -> _Entry:
    """Run `script`'s runs in one session on one seam, ADK's call ids numbered
    (`_numbered_call_ids`)."""
    scripted = ScriptedGenai(script.replies)
    seam = scripted_planner(
        scripted,
        translate=script.translate,
        tools=script.setup,
    )
    harness = _SeamHarness()
    reports: list[dict[str, Any]] = []
    outputs: list[dict[str, Any] | None] = []
    with _numbered_call_ids():
        for arguments in script.runs:
            kwargs = {"question": _QUESTION, "readings": [], **arguments}
            answer = kwargs.pop("answer", None)
            if answer is not None:
                interrupt_id = _pending_ask_interrupt_id(await harness.events())
                kwargs["new_message"] = types.Content(
                    role="user", parts=[create_request_input_response(interrupt_id, answer)]
                )
            token = current_turn_budget.set(script.budget() if script.budget is not None else None)
            try:
                outputs.append(
                    await harness.run(seam, on_progress=reports.append, run_config=_SSE, **kwargs)
                )
            finally:
                current_turn_budget.reset(token)
        if then is not None:
            outputs.append(await then(seam, harness))
    return _Entry(
        outputs=outputs,
        reports=[{k: v for k, v in report.items() if k != "duration_ms"} for report in reports],
        requests=scripted.requests,
        events=await harness.events(),
        state=dict(await harness.state()),
    )


def _read(events: Sequence[Event]) -> list[dict[str, Any]]:
    """What the session's readers read of `events`, in order.

    `planner.session.session_steps`, Studio's translator and the seam read a
    planner event's author, branch, invocation, node path, whether it is
    partial, its content (role and parts: text, thought and signature, each
    call's name, arguments and id, each answer's name, id and response) and
    its state delta. A planner event with neither content nor state delta
    leaves them nothing to read and is left out. Every other event is read
    whole, less its id and timestamp. Each invocation is numbered in the order
    it first appears, since its id differs between any two runs."""
    invocations: dict[str, int] = {}
    read: list[dict[str, Any]] = []
    for event in events:
        invocation = invocations.setdefault(event.invocation_id, len(invocations))
        if event.author != AGENT_NAME:
            dumped = event.model_dump(
                mode="json", exclude_none=True, exclude={"id", "timestamp", "invocation_id"}
            )
            read.append({**dumped, "invocation": invocation})
            continue
        if event.content is None and not event.actions.state_delta:
            continue
        read.append(
            {
                "author": event.author,
                "branch": event.branch,
                "invocation": invocation,
                "path": event.node_info.path,
                "partial": bool(event.partial),
                "content": (
                    event.content.model_dump(mode="json", exclude_none=True)
                    if event.content is not None
                    else None
                ),
                "state_delta": event.actions.state_delta,
            }
        )
    return read


def _steps(events: Sequence[Event]) -> list[dict[str, Any]]:
    """The steps `session_steps` reads from `events`, each by its fields."""
    return [step._asdict() for step in session_steps(events)]


def _request_dumped(request: Request) -> dict[str, Any]:
    """`request`'s model, config and contents, dumped, each package version in the
    config's HTTP headers put as `<version>` (`versionless_headers`)."""
    return {
        "model": request.model,
        "config": versionless_headers(
            request.generate_config.model_dump(mode="json", exclude_none=True)
        ),
        "contents": [
            content.model_dump(mode="json", exclude_none=True) for content in request.content_list
        ],
    }


def _briefs(request: Request) -> list[str]:
    """The text of each part of `request` that holds the seam's brief."""
    return [
        part.text
        for content in request.content_list
        if content.role == "user"
        for part in content.parts or ()
        if part.text is not None and part.text.startswith(BRIEF_OPENING)
    ]


def _view(entry: _Entry) -> dict[str, Any]:
    """What `_assert_expected` checks of `entry`."""
    assert all(len(_briefs(request)) == 1 for request in entry.requests), "one brief a request"
    return {
        "outputs": entry.outputs,
        "reports": entry.reports,
        "requests": [_request_dumped(request) for request in entry.requests],
        "events": _read(entry.events),
        "steps": _steps(entry.events),
        "state": entry.state,
    }


def _assert_expected(
    request: pytest.FixtureRequest,
    entry: _Entry,
    *,
    also: Mapping[str, Any] | None = None,
    config: str = "request_config",
) -> None:
    """Assert that `entry`, with `also` beside it, equals the stored expectation of
    the test `request` runs (`expectation_name`). Each request's config is stored
    once (`planner_expectations.shared`) under the name `config`."""
    view = {**_view(entry), **(also or {})}
    for dumped in view["requests"]:
        dumped["config"] = shared(dumped["config"], config)
    expect(expectation_name(request), view)


def _planner_events(entry: _Entry) -> list[Event]:
    return [event for event in entry.events if event.author == AGENT_NAME]


def _planned(plan: Plan) -> dict[str, Any]:
    return {"question": plan.question, "prior_plan": plan.model_dump(mode="json")}


def _ask_call(id: str, **args: Any) -> types.Part:
    return call_part(
        PLANNER_ASK, {"message": "Which source did you mean?", **args}, id=id, signature=_SIGNED
    )


# --- Entries ------------------------------------------------------------------------------


async def test_a_planned_turn_lands_the_plan_and_stores_two_planner_events(
    plan: Plan, request: pytest.FixtureRequest
) -> None:
    entry = await _entry(
        _Script(
            replies=[reply(call_part(PROPOSE_PLAN, {}, id="p1", signature=_SIGNED), usage=_USAGE)],
            runs=[_planned(plan)],
            setup=_proposals(),
        ),
    )

    assert entry.outputs[0] is not None and "plan" in entry.outputs[0]
    assert len(_planner_events(entry)) == 2
    _assert_expected(request, entry)


# Replies streamed over several chunks, as Gemini streams every call.
_STREAMED = {
    "thought-then-signed-call": Reply(
        chunks=[
            [thought_part("Weighing the debt sources.", signature=None)],
            [call_part(PROPOSE_PLAN, {}, id="p1", signature=_SIGNED)],
            [],
        ],
        usage=_USAGE,
    ),
    "text-over-three-chunks": Reply(
        chunks=[
            [text_part("I will ")],
            [text_part("propose ")],
            [text_part("the plan.")],
            [call_part(PROPOSE_PLAN, {}, id="p1", signature=_SIGNED)],
        ],
        usage=_USAGE,
    ),
    "thoughts-text-and-call": Reply(
        chunks=[
            [thought_part("Weighing the sources", signature=_SIGNED)],
            [thought_part(" and the years.", signature=None)],
            [text_part("Proposing ")],
            [text_part("now.")],
            [call_part(PROPOSE_PLAN, {}, id="p1")],
            [],
        ],
        usage=_USAGE,
    ),
}


@pytest.mark.parametrize("streamed", list(_STREAMED))
async def test_a_reply_streamed_in_chunks_is_stored_and_replayed_as_one_response(
    plan: Plan, streamed: str, request: pytest.FixtureRequest
) -> None:
    """A reply split over several chunks is stored and replayed as one response."""
    entry = await _entry(
        _Script(
            replies=[_STREAMED[streamed]],
            runs=[_planned(plan)],
            setup=_proposals(),
        ),
    )

    assert entry.outputs[0] is not None and "plan" in entry.outputs[0]
    (response,) = [event for event in _planner_events(entry) if event.get_function_calls()]
    assert not response.partial
    _assert_expected(request, entry)


async def test_a_refused_proposal_streamed_in_chunks_is_replayed_as_one_model_turn(
    plan: Plan, request: pytest.FixtureRequest
) -> None:
    """The request after a reply streamed over several chunks replays that reply
    as one model turn."""
    entry = await _entry(
        _Script(
            replies=[
                _STREAMED["thoughts-text-and-call"],
                reply(call_part(PROPOSE_PLAN, {}, id="p2", signature=_SIGNED), usage=_USAGE),
            ],
            runs=[_planned(plan)],
            setup=_proposals(_REFUSED),
        ),
    )

    assert len(entry.requests) == 2
    _assert_expected(request, entry)


async def test_a_question_in_two_parts_is_sent_in_two_parts(
    plan: Plan, request: pytest.FixtureRequest
) -> None:
    """The turn's question goes to the model with its parts as the user sent them."""
    question = types.Content(
        role="user", parts=[types.Part(text="Kenya's debt"), types.Part(text="since 2010")]
    )
    scripted = ScriptedGenai([reply(call_part(PROPOSE_PLAN, {}, id="p1", signature=_SIGNED))])
    seam = scripted_planner(scripted, tools=_proposals())
    await _SeamHarness().run(
        seam, new_message=question, run_config=_SSE, readings=[], **_planned(plan)
    )
    (sent,) = scripted.requests

    assert sent.content_list[0].parts == question.parts
    expect(expectation_name(request), _request_dumped(sent)["contents"])


async def test_a_question_part_the_planner_cannot_send_ends_the_entry_refused(
    plan: Plan,
) -> None:
    """A question holding a file by URI ends the entry before any request on a
    refusal the seam answers with, as it answers any refusal the run ends on."""
    question = types.Content(
        role="user",
        parts=[
            types.Part(text="Kenya's debt"),
            types.Part(file_data=types.FileData(file_uri="gs://b/f.pdf", mime_type="x/pdf")),
        ],
    )
    scripted = ScriptedGenai([reply(call_part(PROPOSE_PLAN, {}, id="p1", signature=_SIGNED))])
    seam = scripted_planner(scripted, tools=_proposals())

    output = await _SeamHarness().run(
        seam,
        new_message=question,
        run_config=_SSE,
        readings=[],
        **_planned(plan),
    )

    assert scripted.requests == []
    assert output is not None and output["refused"]["code"] == "internal_error"


async def test_a_refused_proposal_then_an_accepted_one_lands_the_plan(
    plan: Plan, request: pytest.FixtureRequest
) -> None:
    entry = await _entry(
        _Script(
            replies=[
                reply(call_part(PROPOSE_PLAN, {}, id="p1", signature=_SIGNED), usage=_USAGE),
                reply(call_part(PROPOSE_PLAN, {}, id="p2", signature=_SIGNED), usage=_USAGE),
            ],
            runs=[_planned(plan)],
            setup=_proposals(_REFUSED),
        ),
    )

    assert entry.outputs[0] is not None and "plan" in entry.outputs[0]
    assert len(entry.requests) == 2
    _assert_expected(request, entry)


async def test_a_live_shaped_call_reply_is_replayed_without_an_empty_text_part(
    plan: Plan, request: pytest.FixtureRequest
) -> None:
    """Replies streamed as live Gemini streams a function call (`live_call_reply`):
    the refused proposal's model turn is replayed and stored with no empty text
    part after the call, and Studio's frames equal the stored ones."""
    # Imported here: `tests.test_planner_events` imports this module.
    from .test_planner_events import _frames as _event_frames  # noqa: PLC0415

    entry = await _entry(
        _Script(
            replies=[
                live_call_reply(
                    call_part(PROPOSE_PLAN, {}, id="p1", signature=_SIGNED), usage=_USAGE
                ),
                live_call_reply(
                    call_part(PROPOSE_PLAN, {}, id="p2", signature=_SIGNED), usage=_USAGE
                ),
            ],
            runs=[_planned(plan)],
            setup=_proposals(_REFUSED),
        ),
    )

    assert entry.outputs[0] is not None and "plan" in entry.outputs[0]
    (replayed,) = [c for c in entry.requests[1].content_list if c.role == "model"]
    assert replayed.parts is not None
    assert replayed.parts[-1] != types.Part(text="")
    _assert_expected(request, entry, also={"frames": _event_frames(entry.events)})


async def test_an_ask_and_its_answer_land_the_plan_in_the_next_run(
    plan: Plan, request: pytest.FixtureRequest
) -> None:
    entry = await _entry(
        _Script(
            replies=[
                reply(_ask_call("a1"), usage=_USAGE),
                reply(call_part(PROPOSE_PLAN, {}, id="p1", signature=_SIGNED)),
            ],
            runs=[_planned(plan), {**_planned(plan), "answer": {"result": "2020"}}],
            setup=_proposals(),
        ),
    )

    assert entry.outputs[0] is None, "the ask pauses the first run"
    assert entry.outputs[1] is not None and "plan" in entry.outputs[1]
    _assert_expected(request, entry)


@pytest.mark.parametrize(
    "sibling",
    [call_part(PROPOSE_PLAN, {}, id="p1"), _ask_call("a2")],
    ids=["proposal", "ask"],
)
async def test_an_ask_beside_a_call_stores_one_answers_event(
    plan: Plan, sibling: types.Part, request: pytest.FixtureRequest
) -> None:
    """The answers of a response's calls are stored as one event once every call
    has answered. A proposal beside the ask reads
    no step of the ask, which has no answer yet, and a second ask beside it is
    refused, read off its response's calls."""
    entry = await _entry(
        _Script(
            replies=[reply(_ask_call("a1"), sibling, usage=_USAGE)],
            runs=[_planned(plan)],
            setup=_proposals(),
        ),
    )

    assert sibling.function_call is not None
    sibling_id = sibling.function_call.id
    (answers,) = [event for event in _planner_events(entry) if event.get_function_responses()]
    assert [response.id for response in answers.get_function_responses()] == ["a1", sibling_id]
    if sibling_id == "p1":
        assert entry.state[_READ_KEY] == {
            "steps": [["message", None]],
            "response_calls": [[PLANNER_ASK, "a1"], [PROPOSE_PLAN, "p1"]],
        }
    else:
        first, second = answers.get_function_responses()
        assert (first.response or {}).get("outcome") == Outcome.OK.value
        assert (second.response or {}).get("code") == ProblemCode.INVALID_ARGUMENTS.value
    _assert_expected(request, entry)


async def test_a_call_reads_its_response_in_the_session_and_no_answer_of_its_sibling(
    plan: Plan,
) -> None:
    """A planner tool reads the session as it stands while the tool runs. The
    event holding its own response's calls is already there, and its sibling's
    answer is not, though the sibling ran first."""
    seen: list[list[tuple[list[str | None], list[str | None]]]] = []

    def tools(specs: Sequence[ToolSpec]) -> list[ToolSpec]:
        def body(call: PlannerCall, end_run: EndRun, arguments: dict[str, Any]) -> object:
            assert isinstance(call, SessionCall)
            seen.append(
                [
                    (
                        [c.id for c in event.get_function_calls()],
                        [r.id for r in event.get_function_responses()],
                    )
                    for event in call.events
                    if event.author == AGENT_NAME
                ]
            )
            return _REFUSED if len(seen) < 2 else _compiled_proposal(call, end_run, arguments)

        return _serving_proposals(specs, body)

    replies = [
        reply(call_part(PROPOSE_PLAN, {}, id="p1"), call_part(PROPOSE_PLAN, {}, id="p2")),
    ]
    await _entry(_Script(replies=replies, runs=[_planned(plan)], setup=tools))

    assert seen == [[(["p1", "p2"], [])]] * 2


async def test_two_calls_writing_one_key_store_the_value_the_last_to_finish_left(
    plan: Plan,
) -> None:
    """The stored answers of a response hold the session's value of each key its
    calls wrote once they have all answered, so the call that finishes last decides
    a key two calls wrote, whatever their order in the response. The answers are
    stored in call order."""
    started: list[str] = []

    def tools(specs: Sequence[ToolSpec]) -> list[ToolSpec]:
        async def body(call: PlannerCall, end_run: EndRun, arguments: dict[str, Any]) -> object:
            # The first call in call order is the last to finish.
            started.append("call")
            first = len(started) == 1
            if first:
                await asyncio.sleep(0.05)
            call.state["test:scalar"] = "first" if first else "second"
            call.state["test:dict"] = {"writer": "first" if first else "second"}
            return _REFUSED

        return _serving_proposals(specs, body)

    replies = [
        reply(call_part(PROPOSE_PLAN, {}, id="p1"), call_part(PROPOSE_PLAN, {}, id="p2")),
        reply(text_part("Done.")),
    ]
    entry = await _entry(_Script(replies=replies, runs=[_planned(plan)], setup=tools))

    (answers,) = [e for e in _planner_events(entry) if e.get_function_responses()]
    assert [r.id for r in answers.get_function_responses()] == ["p1", "p2"]
    live = {"test:scalar": "first", "test:dict": {"writer": "first"}}
    delta = {k: v for k, v in (answers.actions.state_delta or {}).items() if k.startswith("test:")}
    assert delta == live
    assert {k: v for k, v in entry.state.items() if k.startswith("test:")} == live


# --- Turns put in the model's place ------------------------------------------------------


async def test_a_proposal_refused_for_good_ends_the_run_on_the_stops_sentence(
    plan: Plan, request: pytest.FixtureRequest
) -> None:
    """A refusal that ends the entry answers the next request with the stop's
    sentence in place of the model, which is never called again."""
    refused = {
        "outcome": "refused",
        "code": ProblemCode.BUDGET_EXHAUSTED.value,
        "message": "No more rounds.",
    }
    entry = await _entry(
        _Script(
            replies=[reply(call_part(PROPOSE_PLAN, {}, id="p1"), usage=_USAGE)],
            runs=[_planned(plan)],
            setup=_proposals(refused),
        ),
    )

    assert len(entry.requests) == 1
    stops = [
        event
        for event in _planner_events(entry)
        if event.content and any(part.text for part in event.content.parts or ())
    ]
    assert len(stops) == 1
    _assert_expected(request, entry)


async def test_a_referral_the_user_decides_asks_in_place_of_the_model(
    plan: Plan, request: pytest.FixtureRequest
) -> None:
    """A referral whose pick is the user's answers the next request with the
    question in place of the model."""
    referral = _referral({"spec_id": _SPEC, "sentence": "external debt"})
    referral["translation"]["kind"] = "source_name_ambiguous"
    referral["translation"]["ambiguous_sources"] = [
        {"candidates": [{"ref": _SPEC, "label": "external debt"}]}
    ]
    entry = await _entry(
        _Script(
            replies=[reply(call_part(PROPOSE_PLAN, {}, id="p1"), usage=_USAGE)],
            runs=[_planned(plan)],
            setup=_proposals(referral),
        ),
    )

    assert entry.outputs == [None], "the question pauses the run"
    assert len(entry.requests) == 1
    _assert_expected(request, entry)


# --- The turn's budget ------------------------------------------------------------------


def _one_call_budget() -> TurnBudget:
    """A budget whose call cap closes every call after the first."""
    return TurnBudget(model_time_s=60.0, model_calls_max=1, wall_s=60.0, first_chunk_s=30.0)


def _referral(*candidates: dict[str, Any]) -> dict[str, Any]:
    """A proposal whose translation referred the plan back to choose a source."""
    return {
        "outcome": Outcome.INVALID_INPUT.value,
        "translation": {
            "outcome": Outcome.CHOICE_REQUIRED.value,
            "kind": "source_ambiguous",
            "requirement_id": "debt",
            "candidates": list(candidates),
        },
    }


async def test_a_budget_closing_a_call_with_pinnable_candidates_asks(
    plan: Plan, request: pytest.FixtureRequest
) -> None:
    """The second call is past the cap. The referral before it names a source to
    pin, so the closed call is answered with a question."""
    entry = await _entry(
        _Script(
            replies=[reply(call_part(PROPOSE_PLAN, {}, id="p1", signature=_SIGNED), usage=_USAGE)],
            runs=[_planned(plan)],
            setup=_proposals(_referral({"spec_id": _SPEC, "sentence": "external debt"})),
            budget=_one_call_budget,
        ),
    )

    assert entry.outputs == [None], "the closed call's question pauses the run"
    (asked,) = [
        call
        for event in _planner_events(entry)
        for call in event.get_function_calls()
        if call.name == PLANNER_ASK
    ]
    assert asked.id == "adk-1"
    assert (asked.args or {})["candidates"] == [{"spec_id": _SPEC, "label": "external debt"}]
    _assert_expected(request, entry)


async def test_a_budget_closing_a_call_with_nothing_to_pin_refuses(
    plan: Plan, request: pytest.FixtureRequest
) -> None:
    entry = await _entry(
        _Script(
            replies=[reply(call_part(PROPOSE_PLAN, {}, id="p1", signature=_SIGNED), usage=_USAGE)],
            runs=[_planned(plan)],
            setup=_proposals(_referral({"sentence": "a nearby series"})),
            budget=_one_call_budget,
        ),
    )

    code = SeamCode.TURN_MODEL_CALLS_SPENT
    assert entry.outputs == [
        {"refused": {"text": refusal_sentence({"code": code}), "final": True, "code": code}}
    ]
    _assert_expected(request, entry)


async def test_a_call_a_bound_closes_logs_the_usage_its_chunks_reported(
    plan: Plan, caplog: pytest.LogCaptureFixture
) -> None:
    """The stream stops after a chunk that reported usage, and the model-time
    bound closes the call. Operators cost a turn from its `token usage` lines,
    so the closed call logs the usage it reported."""
    unfinished = Reply(chunks=[[text_part("Partly")]], usage=_USAGE, finish_reason=None)
    scripted = _HangingGenai([unfinished], hang_after=1)
    seam = scripted_planner(scripted, tools=_proposals())
    budget = TurnBudget(model_time_s=0.05, model_calls_max=30, wall_s=60.0, first_chunk_s=30.0)
    token = current_turn_budget.set(budget)
    try:
        with caplog.at_level(logging.INFO, logger=usage.logger.name):
            output = await _SeamHarness().run(seam, readings=[], **_planned(plan))
    finally:
        current_turn_budget.reset(token)

    code = SeamCode.TURN_MODEL_TIME_SPENT
    assert output is not None and output["refused"]["code"] == code
    assert [record.args for record in caplog.records if record.name == usage.logger.name] == [
        (AGENT_NAME, 120, 40, 9, 4, 133)
    ]


# --- Responses that are no turn, and a run that raises ----------------------------------


@pytest.mark.parametrize(
    "response",
    [
        reply(text_part("Cut off"), usage=_USAGE, finish_reason=types.FinishReason.MAX_TOKENS),
        blocked_reply(),
        empty_reply(),
    ],
    ids=["max-tokens", "blocked", "empty"],
)
async def test_a_response_that_errs_or_holds_nothing_ends_the_run_with_no_plan(
    plan: Plan, response: Reply, request: pytest.FixtureRequest
) -> None:
    """A response cut at its token limit stores its text, and a blocked prompt
    or an empty response stores nothing a reader reads. Each ends the run with
    no plan."""
    entry = await _entry(
        _Script(replies=[response], runs=[_planned(plan)], setup=_proposals()),
    )

    assert (
        entry.outputs[0] is not None
        and entry.outputs[0]["refused"]["code"] == SeamCode.PLAN_NOT_READY
    )
    _assert_expected(request, entry)


async def test_a_function_call_gemini_could_not_parse_ends_the_run_with_no_plan(
    plan: Plan, request: pytest.FixtureRequest
) -> None:
    """A reply streamed as live Gemini streams a function call it could not
    parse (`malformed_call_reply`) ends the run on `SeamCode.PLAN_NOT_READY`
    with no further model call. The response is stored as its signed empty text
    part."""
    entry = await _entry(
        _Script(
            replies=[malformed_call_reply(usage=_USAGE)],
            runs=[_planned(plan)],
            setup=_proposals(),
        ),
    )

    assert (
        entry.outputs[0] is not None
        and entry.outputs[0]["refused"]["code"] == SeamCode.PLAN_NOT_READY
    )
    assert len(entry.requests) == 1
    _assert_expected(request, entry)


async def _raising(plan: Plan) -> _Entry:
    """An entry whose second tool answer of one response is past the run's event
    cap (`harness.planner_run._MAX_EVENTS`, set to 1), which raises out of the run."""
    scripted = ScriptedGenai(
        [reply(call_part(PROPOSE_PLAN, {}, id="p1", signature=_SIGNED), _ask_call("a1"))]
    )
    seam = scripted_planner(scripted, tools=_proposals(_REFUSED))
    harness = _SeamHarness()
    with (
        _numbered_call_ids(),
        mock.patch.object(planner_run, "_MAX_EVENTS", 1),
        pytest.raises(PlannerSeamError),
    ):
        await harness.run(seam, run_config=_SSE, readings=[], **_planned(plan))
    return _Entry(
        outputs=[],
        reports=[],
        requests=scripted.requests,
        events=await harness.events(),
        state=dict(await harness.state()),
    )


async def test_a_run_that_raises_while_calls_run_stores_none_of_the_responses_answers(
    plan: Plan,
) -> None:
    """A run that raises while a response's calls run fails the turn. The planner
    stores the response and none of its answers, though the call that finished
    before its sibling raised had answered."""
    entry = await _raising(plan)

    events = _planner_events(entry)
    assert [call.id for event in events for call in event.get_function_calls()] == ["p1", "a1"]
    assert [r.id for event in events for r in event.get_function_responses()] == []


async def _stored(harness: _SeamHarness) -> _Entry:
    return _Entry(outputs=[], reports=[], requests=[], events=await harness.events(), state={})


@dataclass
class _SpentFirst:
    """`seam`, called after `spent` model calls of the invocation, as the
    orchestrator's calls before the planner's spend the invocation's allowance."""

    seam: PlannerSeam
    spent: int

    async def __call__(self, *, ctx: Context, **kwargs: Any) -> dict[str, Any]:
        invocation = ctx.get_invocation_context()
        for _ in range(self.spent):
            invocation.increment_llm_call_count()
        return await self.seam(ctx=ctx, **kwargs)


async def _capped(plan: Plan, *, spent: int) -> _Entry:
    """An entry under the invocation's cap of three model calls
    (`RunConfig.max_llm_calls`), `spent` of them spent before the planner runs,
    whose proposals are all refused, so the planner calls until the cap trips."""
    scripted = ScriptedGenai(
        [
            reply(call_part(PROPOSE_PLAN, {}, id=f"p{n}", signature=_SIGNED), usage=_USAGE)
            for n in (1, 2, 3, 4)
        ]
    )
    seam = scripted_planner(scripted, tools=_proposals(_REFUSED, _REFUSED, _REFUSED))
    harness = _SeamHarness()
    capped = RunConfig(streaming_mode=StreamingMode.SSE, max_llm_calls=3)
    with pytest.raises(LlmCallsLimitExceededError):
        await harness.run(
            cast(PlannerSeam, _SpentFirst(seam, spent)),
            run_config=capped,
            readings=[],
            **_planned(plan),
        )
    return _Entry(
        outputs=[],
        reports=[],
        requests=scripted.requests,
        events=await harness.events(),
        state=dict(await harness.state()),
    )


@pytest.mark.parametrize("spent", [0, 1], ids=["nothing-spent", "one-call-spent-before"])
async def test_a_run_past_the_invocations_model_call_cap_stores_the_error_event(
    plan: Plan, spent: int, request: pytest.FixtureRequest
) -> None:
    """The call past `RunConfig.max_llm_calls` raises before it is sent, counted
    with the invocation's calls before the planner ran, and the session stores
    an error event."""
    entry = await _capped(plan, spent=spent)

    assert len(entry.requests) == 3 - spent
    assert entry.events[-1].error_code == "LlmCallsLimitExceededError"
    _assert_expected(request, entry)


async def test_two_entries_at_once_on_one_pydantic_ai_seam_keep_to_their_own_runs(
    plan: Plan,
) -> None:
    """One seam's Pydantic AI planner serves two entries running at once, each in
    its own session. The planner is built once, each request carries its own
    entry's question and answers, and each session stores the same events as
    the other."""
    barrier = asyncio.Barrier(2)

    async def meets(call: PlannerCall) -> object:
        call.state["planner:met"] = True
        await asyncio.wait_for(barrier.wait(), timeout=5)
        return _REFUSED

    def body(call: PlannerCall, end_run: EndRun, arguments: dict[str, Any]) -> object:
        if "planner:met" not in call.state:
            return meets(call)
        return _compiled_proposal(call, end_run, arguments)

    scripted = ScriptedGenai(
        [
            reply(call_part(PROPOSE_PLAN, {}, id=call_id, signature=_SIGNED))
            for call_id in ("p1", "p1", "p2", "p2")
        ]
    )
    questions = ("kenya debt", "ghana aid")
    harnesses = [_SeamHarness() for _ in questions]
    with mock.patch.object(
        planner_node_module, "planner_agent", wraps=planner_node_module.planner_agent
    ) as built:
        seam = scripted_planner(scripted, tools=lambda specs: _serving_proposals(specs, body))
        outputs = await asyncio.gather(
            *(
                harness.run(
                    seam,
                    new_message=types.Content(role="user", parts=[types.Part(text=question)]),
                    run_config=_SSE,
                    readings=[],
                    **_planned(plan),
                )
                for harness, question in zip(harnesses, questions, strict=True)
            )
        )

    assert built.call_count == 1
    assert all(output is not None and "plan" in output for output in outputs)
    asked = [
        [part.text for part in request.content_list[0].parts or ()] for request in scripted.requests
    ]
    assert sorted(asked) == sorted([[question] for question in questions] * 2)
    for request in scripted.requests[2:]:
        (refusal,) = [
            part.function_response
            for content in request.content_list
            for part in content.parts or ()
            if part.function_response is not None
        ]
        assert refusal.id == "p1"
    first, second = [_planner_events(await _stored(harness)) for harness in harnesses]
    assert _read(first) == _read(second)


# --- A Stop ------------------------------------------------------------------------------


async def _stopped(plan: Plan) -> _Entry:
    """An entry stopped while `propose_plan` hangs, its first call answered."""
    started = asyncio.Event()

    async def hangs(call: PlannerCall, end_run: EndRun, arguments: dict[str, Any]) -> object:
        del end_run, arguments
        call.state["planner:hanging"] = "started"
        started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    calls = itertools.count()

    def body(call: PlannerCall, end_run: EndRun, arguments: dict[str, Any]) -> object:
        if next(calls) == 0:
            call.state["planner:first"] = "answered"
            return _REFUSED
        return hangs(call, end_run, arguments)

    scripted = ScriptedGenai(
        [
            reply(call_part(PROPOSE_PLAN, {}, id="p1", signature=_SIGNED)),
            reply(call_part(PROPOSE_PLAN, {}, id="p2", signature=_SIGNED), _ask_call("a1")),
        ]
    )
    seam = scripted_planner(scripted, tools=lambda specs: _serving_proposals(specs, body))
    harness = _SeamHarness()
    running = asyncio.create_task(harness.run(seam, run_config=_SSE, readings=[], **_planned(plan)))
    await asyncio.wait_for(started.wait(), timeout=5)
    running.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await running
    return _Entry(
        outputs=[],
        reports=[],
        requests=scripted.requests,
        events=await harness.events(),
        state=dict(await harness.state()),
    )


async def test_a_stop_while_a_planner_tool_hangs_stores_the_answered_call_and_not_the_hanging_one(
    plan: Plan,
    request: pytest.FixtureRequest,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Stop's cancellation reaches the hanging tool as a cancellation. The
    run logs no tool failure and records no refusal."""
    runs: list[PlannerRun] = []

    def recorded(**kwargs: Any) -> PlannerRun:
        runs.append(PlannerRun(**kwargs))
        return runs[-1]

    monkeypatch.setattr(planner_seam, "PlannerRun", recorded)
    with caplog.at_level(logging.ERROR):
        entry = await _stopped(plan)

    assert entry.state["planner:first"] == "answered"
    assert "planner:hanging" not in entry.state
    _assert_expected(request, entry)
    assert [
        record.getMessage() for record in caplog.records if record.levelno >= logging.ERROR
    ] == []
    (run,) = runs
    assert run.refused is None


async def test_a_stop_while_the_node_waits_on_its_event_closes_the_run_in_its_task(
    plan: Plan, caplog: pytest.LogCaptureFixture
) -> None:
    """A Stop while ADK's node runner stores the planner's response, the node
    suspended at its yield, closes the Pydantic AI run in the node's task. Its
    spans and context variables end where they began, with nothing logged."""
    scripted = ScriptedGenai([reply(call_part(PROPOSE_PLAN, {}, id="p1", signature=_SIGNED))])
    seam = scripted_planner(scripted, tools=_proposals())
    harness = _SeamHarness()
    storing = asyncio.Event()
    append = harness._sessions.append_event

    async def append_event(session: Any, event: Event) -> Event:
        if event.author == AGENT_NAME and event.get_function_calls():
            storing.set()
            await asyncio.Event().wait()
        return await append(session, event)

    with mock.patch.object(harness._sessions, "append_event", append_event):
        running = asyncio.create_task(
            harness.run(seam, run_config=_SSE, readings=[], **_planned(plan))
        )
        await asyncio.wait_for(storing.wait(), timeout=5)
        running.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await running
    gc.collect()
    for _ in range(5):
        await asyncio.sleep(0)

    assert [record for record in caplog.records if record.levelno >= logging.ERROR] == []


# --- A revision after the entry -----------------------------------------------------------


def _pinned_plan(plan: Plan) -> tuple[Plan, DatasetRequirement, dict[str, PinSource]]:
    """`plan` with its dataset pinned to `_SPEC`, whose scope offers a sector
    switch, the requirement re-pinned to `_SPEC_NEW` with the sector switched,
    and the pin sources of both."""
    scope = _switchable_scope()
    switched = _sector_switched(scope)
    blocks = {
        _SPEC: {"spec_id": _SPEC, "resolved": {"scope_choices": scope}},
        _SPEC_NEW: {"spec_id": _SPEC_NEW, "resolved": {"scope_choices": switched}},
    }
    shown = {spec_id: PinSource(block=block, settled=True) for spec_id, block in blocks.items()}
    scoped = scope_choice_decisions({_SPEC: shown[_SPEC]})[_SPEC]
    dataset = next(r for r in plan.evidence_requirements if isinstance(r, DatasetRequirement))
    pinned = dataset.model_copy(
        update={
            "source_spec_id": _SPEC,
            "decision_ids": (*dataset.decision_ids, *(d.slot for d in scoped)),
        }
    )
    pinned_plan = Plan.model_validate(
        {
            **plan.model_dump(),
            "decisions": [*plan.decisions, *scoped],
            "evidence_requirements": [
                pinned if r.id == dataset.id else r for r in plan.evidence_requirements
            ],
        }
    )
    return pinned_plan, pinned.model_copy(update={"source_spec_id": _SPEC_NEW}), shown


async def test_a_revision_after_an_entry_reads_the_steps_of_its_stored_ask_and_answer(
    plan: Plan, request: pytest.FixtureRequest
) -> None:
    """The user picked `_SPEC` answering the planner's ask. A sector switch then
    re-pins it to `_SPEC_NEW` with no model call, and the decision the switch
    makes is the user's, read from the entry's stored ask and
    answer."""
    pinned, repinned, shown = _pinned_plan(plan)

    async def revise(seam: PlannerSeam, harness: _SeamHarness) -> dict[str, Any]:
        async def _node(ctx: Context, node_input: Any) -> Any:
            del node_input
            save_pin_sources(cast(SessionState, ctx.state), shown)
            return await seam.revise(
                ctx=ctx,
                prior_plan=pinned.model_dump(mode="json"),
                evidence_requirements=[repinned.model_dump(mode="json")],
                inherits={_SPEC_NEW: _SPEC},
            )

        revising = AdkRunner(
            node=Workflow(name="revise_wf", edges=[(START, node(_node, name="revise"))]),
            app_name=_APP,
            session_service=harness._sessions,
        )
        assert harness._session_id is not None
        (output,) = [
            event.output
            async for event in revising.run_async(
                user_id=_USER,
                session_id=harness._session_id,
                new_message=types.Content(role="user", parts=[types.Part(text="switch")]),
            )
            if event.output is not None
        ]
        return cast(dict[str, Any], output)

    async def translate(call: PlannerCall) -> dict[str, object]:
        draft = load_plan(cast(SessionState, call.state))
        assert draft is not None
        assert plan.translation is not None and plan.workflow is not None
        save_plan(
            cast(SessionState, call.state),
            draft.with_translation(translation=plan.translation, workflow=plan.workflow),
        )
        return {"outcome": "ok"}

    candidates = [{"spec_id": _SPEC, "label": "external debt"}]
    entry = await _entry(
        _Script(
            replies=[
                reply(_ask_call("a1", candidates=candidates), usage=_USAGE),
                reply(call_part(PROPOSE_PLAN, {}, id="p1", signature=_SIGNED)),
            ],
            runs=[_planned(pinned), {**_planned(pinned), "answer": {"result": {"spec_id": _SPEC}}}],
            setup=_proposals(),
            translate=translate,
        ),
        then=revise,
    )

    assert entry.state[_ELECTED_KEY] == [_SPEC]
    revised = entry.outputs[-1]
    assert revised is not None and "plan" in revised, revised
    revision = Plan.model_validate(revised["plan"])
    assert revision.status is PlanStatus.PENDING_APPROVAL
    sector = next(d for d in revision.decisions if d.slot == f"scope_sector_{_SPEC_NEW}")
    assert sector.resolved_by is not None and sector.resolved_by.value == "user"
    _assert_expected(request, entry)


# --- A Studio turn ------------------------------------------------------------------------


# The frame fields Studio fills with ids of its own drawing.
_DRAWN_IDS = ("interrupt_id", "plan_version_id")


def _frames(frames: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The frames, less each line's duration, with each id Studio draws numbered in
    the order it first appears."""
    numbered: dict[str, str] = {}
    shown = []
    for frame in frames:
        kept = {k: v for k, v in frame.items() if k != "duration_ms"}
        for key in _DRAWN_IDS:
            if key in kept:
                kept[key] = numbered.setdefault(kept[key], f"{key}-{len(numbered)}")
        shown.append(kept)
    return shown


async def _studio(
    plan: Plan, replies: Sequence[Reply], *, answer: dict[str, Any] | None = None
) -> list[list[dict[str, Any]]]:
    """The frames of a Studio turn whose orchestrator hands the question to the
    planner, and of the answer that resumes it when `answer` is given."""
    scripted = ScriptedGenai(replies)
    seam = scripted_planner(scripted, tools=_proposals(records=plan))
    app = _app(plan_call(_QUESTION), _text("Planned."), planner=seam)
    async with studio_client(app) as client:
        with _numbered_call_ids():
            first = await stream(client, "/api/studio/turn", turn_body(_QUESTION))
        streamed = [_frames(first)]
        if answer is not None:
            (elicitation,) = [f for f in first if f["event"] == "elicitation.request"]
            second = await stream(
                client,
                "/api/studio/interrupt",
                _interrupt_body(elicitation["interrupt_id"], "answer", answers={"q1": answer}),
            )
            streamed.append(_frames(second))
    return streamed


@pytest.mark.parametrize(
    "replied",
    [
        pytest.param(
            reply(call_part(PROPOSE_PLAN, {}, id="p1", signature=_SIGNED), usage=_USAGE),
            id="one-chunk",
        ),
        *(pytest.param(streamed, id=name) for name, streamed in _STREAMED.items()),
    ],
)
async def test_a_planned_studio_turn_streams_the_stored_frames(
    plan: Plan, replied: Reply, request: pytest.FixtureRequest
) -> None:
    frames = await _studio(plan, [replied])

    assert "plan.proposed" in [frame["event"] for frame in frames[0]]
    expect(expectation_name(request), frames)


async def test_a_studio_ask_and_its_answer_stream_the_stored_frames(
    plan: Plan, request: pytest.FixtureRequest
) -> None:
    candidates = [
        {"spec_id": "spec_a", "label": "reading A"},
        {"spec_id": "spec_b", "label": "reading B"},
    ]
    replies = [
        reply(_ask_call("a1", candidates=candidates), usage=_USAGE),
        reply(call_part(PROPOSE_PLAN, {}, id="p1", signature=_SIGNED)),
    ]

    frames = await _studio(plan, replies, answer=_pick("spec_b"))

    assert frames[0][-1]["event"] == "elicitation.request"
    assert "plan.proposed" in [frame["event"] for frame in frames[1]]
    expect(expectation_name(request), frames)


# --- A Gemini transport failure reaching Studio --------------------------------------


def _error_frame(frames: Sequence[Mapping[str, Any]]) -> tuple[str, str]:
    """The machine-readable code and retry mode of the turn's stream-fatal frame."""
    (error,) = [frame["error"] for frame in frames if frame["event"] == "error"]
    return error["error"], error["retry"]


@pytest.mark.parametrize(
    "failed",
    [
        pytest.param(
            Reply(open_error=errors.ServerError(503, {"error": {"code": 503}})),
            id="unavailable-at-open",
        ),
        pytest.param(
            Reply(
                chunks=[[thought_part("Weighing.")], [call_part(PROPOSE_PLAN, {}, id="p1")]],
                cut_after=1,
                cut_with=aiohttp.ClientPayloadError("Response payload is not completed"),
            ),
            id="payload-cut-mid-stream",
        ),
    ],
)
async def test_a_dropped_gemini_call_ends_the_studio_turn_as_retry_later(
    plan: Plan, failed: Reply
) -> None:
    (frames,) = await _studio(plan, [failed])

    assert _error_frame(frames) == ("transient_unavailable", "retry_after_delay")


def _sse_chunk(
    *parts: dict[str, Any],
    response_id: str,
    token: bytes | None = None,
    finish: str | None = None,
) -> bytes:
    """One `streamGenerateContent?alt=sse` event as the Gemini API sends it."""
    candidate: dict[str, Any] = {"content": {"role": "model", "parts": list(parts)}}
    if token is not None:
        candidate["continuationToken"] = base64.b64encode(token).decode()
    if finish is not None:
        candidate["finishReason"] = finish
    body = {"candidates": [candidate], "responseId": response_id, "modelVersion": _MODEL}
    return f"data: {json.dumps(body)}\r\n\r\n".encode()


class _Dropping(httpx.AsyncByteStream):
    """A response body that sends `events`, then drops the connection when
    `dropped` is set."""

    def __init__(self, events: Sequence[bytes], *, dropped: bool) -> None:
        self._events = events
        self._dropped = dropped

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for event in self._events:
            yield event
        if self._dropped:
            raise httpx.RemoteProtocolError("peer closed connection without sending complete body")


async def test_a_stream_google_genai_resumed_over_a_repeated_call_ends_the_turn_as_retry_later(
    plan: Plan,
) -> None:
    """The real google-genai client streams a checkpointed thought, then a
    function call no continuation token covers, then the connection drops. The
    SDK resumes from the thought's token and the resumed request repeats the
    call. Pydantic AI refuses the repeat, so the tool runs no time at all, and
    the turn ends as the dropped connection it is: retry later."""
    token = b"checkpoint-1"
    call = {"functionCall": {"name": PROPOSE_PLAN, "args": {}}, "thoughtSignature": "c2ln"}
    hops = [
        _Dropping(
            [
                _sse_chunk({"text": "Weighing.", "thought": True}, response_id="r1", token=token),
                _sse_chunk(call, response_id="r1"),
            ],
            dropped=True,
        ),
        _Dropping([_sse_chunk(call, response_id="r2", finish="STOP")], dropped=False),
    ]
    sent: list[bytes] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request.content)
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=hops[len(sent) - 1]
        )

    client = genai.Client(
        api_key="scripted",
        http_options=types.HttpOptions(
            async_client_args={"transport": httpx.MockTransport(handler)}
        ),
    )
    ran: list[str] = []

    def body(*_: Any) -> object:
        ran.append(PROPOSE_PLAN)
        return {"outcome": "ok"}

    surface = FixtureSurface(schemas=[])
    translate = build_translate(surface=surface)
    seam = PlannerSeam(
        settings=Settings(),
        surface=surface,
        translate=translate,
        specs=_serving_proposals(seam_specs(surface=surface, translate=translate), body),
        client=client,
    )
    app = _app(plan_call(_QUESTION), _text("Planned."), planner=seam)
    async with studio_client(app) as studio:
        frames = await stream(studio, "/api/studio/turn", turn_body(_QUESTION))

    assert len(sent) == 2
    assert base64.b64encode(token) in sent[1], "the SDK resumed from the thought's checkpoint"
    assert ran == []
    assert _error_frame(frames) == ("transient_unavailable", "retry_after_delay")

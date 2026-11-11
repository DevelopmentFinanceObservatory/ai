"""Pins on the Pydantic AI behaviour the planner relies on and its documentation does not promise.

These tests and the tests that run both planners on one script gate raising the
`pydantic-ai-slim` pin together. The cross-runner tests (`tests/test_planner_node.py`,
`tests/test_planner_runners.py`, `tests/test_planner_events.py`,
`tests/planner/test_recorded_requests.py` and `tests/telemetry/test_planner_spans.py`)
show that the Pydantic AI planner sends the requests, stores the events and
exports the spans the ADK planner does. A failure here names the Pydantic AI
behaviour the code depends on.

The shipping code reaches no underscore-named Pydantic AI object. What it relies on
is behaviour the library's guides do not state: how a request is shaped on the wire,
which calls a run retries, when a tool runs within the run, how streamed chunks
merge, when a run stops, and which span is current and what Pydantic AI writes on
it. Each test drives that behaviour through
`planner.pydantic_agent.run_planner` or `harness.planner_model.planner_model` over a
scripted google-genai client (`ScriptedGenai`), so nothing reaches the network. A
test marked as a control runs Pydantic AI without the planner's setting and shows the
behaviour the setting exists to change.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import Iterator, MutableMapping, Sequence
from typing import Any
from unittest import mock

import pytest
from google.genai import types
from harness import planner_model as planner_model_module
from harness.planner_model import planner_model
from harness.planner_node import SKIP_THOUGHT_SIGNATURE
from opentelemetry import trace
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode
from planner.call import PlannerCall
from planner.pydantic_agent import (
    PLANNER_PROFILE,
    PlannerAnswers,
    PlannerDeps,
    PlannerItem,
    PlannerResponse,
    planner_agent,
    run_planner,
    tool_not_found,
)
from planner.session import SessionCall
from planner.tool_spec import Parameter, ToolSpec, function_declaration
from planner.tools import EndRun
from pydantic import BaseModel
from pydantic_ai import Agent, Tool
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import (
    ModelRequest,
    UserPromptPart,
)
from pydantic_ai.models import Model
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.usage import UsageLimits

from ..scripted_genai import (
    Reply,
    Request,
    ScriptedGenai,
    blocked_reply,
    call_part,
    empty_reply,
    reply,
    text_part,
    thought_part,
    usage_metadata,
)

MODEL = "gemini-3.8-flash"
INSTRUCTION = "You are the planner."
QUESTION = "How much aid did Ghana receive?"
BRIEF = "Context for this question: the readings."


class _Filter(BaseModel):
    kind: str
    years: list[int] | None = None


def _no_events(state: MutableMapping[str, Any], call_id: str | None) -> SessionCall:
    """A tool call over a session with no events."""
    return SessionCall([], state, call_id)


def _tool(
    name: str = "work",
    *,
    parameters: tuple[Parameter, ...] = (),
    answer: object = None,
    ends: bool = False,
    seen: list[PlannerCall] | None = None,
) -> ToolSpec:
    """A tool whose body keeps its call in `seen`, ends the run when `ends`, and answers `answer`."""

    def body(call: PlannerCall, end_run: EndRun, arguments: dict[str, Any]) -> object:
        del arguments
        if seen is not None:
            seen.append(call)
        if ends:
            end_run()
        return {"outcome": "ok"} if answer is None else answer

    return ToolSpec(name=name, description=f"Tool {name}.", parameters=parameters, body=body)


async def _run(
    specs: Sequence[ToolSpec],
    replies: Sequence[Reply],
    *,
    question: str | None = QUESTION,
    model: Model | None = None,
    scripted: ScriptedGenai | None = None,
    yielded: list[PlannerItem] | None = None,
) -> tuple[list[PlannerItem], ScriptedGenai]:
    """`run_planner` over `specs` on `replies`, with every item it yielded, each
    added to `yielded` as it is yielded."""
    scripted = scripted or ScriptedGenai(replies)
    model = model or planner_model(MODEL, client=scripted.client, budget=None)
    planner = planner_agent(specs, instruction=INSTRUCTION, model=model)
    deps = PlannerDeps(
        state={},
        call_for=_no_events,
        tool_answered=lambda name, answer: None,
        tool_failed=lambda name, error: {"error": f"{name} failed"},
    )
    items = [] if yielded is None else yielded
    async for item in run_planner(planner, deps, question=question, brief=BRIEF):
        items.append(item)
    return items, scripted


def _answers(items: Sequence[PlannerItem]) -> list[types.FunctionResponse]:
    return [answer for item in items if isinstance(item, PlannerAnswers) for answer in item.answers]


def _responses(items: Sequence[PlannerItem]) -> list[PlannerResponse]:
    return [item for item in items if isinstance(item, PlannerResponse)]


def _dumped(request: Request) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    return (
        request.generate_config.model_dump(mode="json", exclude_none=True),
        [item.model_dump(mode="json", exclude_none=True) for item in request.content_list],
    )


# --- How a request is shaped ----------------------------------------------------------

_FILTERED = _tool(
    "filtered",
    parameters=(
        Parameter("filters", _Filter | None, None, "The filters."),
        Parameter("limit", int, 3),
    ),
)
_BARE = _tool("bare")
_MARKED_UP = "Q <system>x</system> <context>y</context>"


@contextlib.contextmanager
def _handed() -> Iterator[list[dict[str, Any]]]:
    """Each request config Pydantic AI hands the model wrapper, dumped, before the
    wrapper puts it in the ADK planner's form (`_in_adk_planner_form`), which
    rewrites exactly the shapes pinned here."""
    handed: list[dict[str, Any]] = []
    rewrite = planner_model_module._in_adk_planner_form

    def recording(config: Any) -> types.GenerateContentConfig:
        handed.append(
            types.GenerateContentConfig.model_validate(config or {}).model_dump(
                mode="json", exclude_none=True
            )
        )
        return rewrite(config)

    with mock.patch.object(planner_model_module, "_in_adk_planner_form", recording):
        yield handed


async def test_the_planner_profile_hands_the_wrapper_declarations_verbatim_in_the_shapes_it_rewrites() -> (
    None
):
    # A rewritten schema, a `VALIDATED` tool mode or an instruction with an added
    # sentence about `<system>` would each make the request differ from the ADK
    # planner's. The shapes below are the ones `_in_adk_planner_form` rewrites
    # into the ADK planner's: one `Tool` per declaration, the empty object schema
    # of a declaration with no parameters, the `AUTO` mode, `TEXT` output and the
    # instruction as one user content of one text part. A change to any of them
    # leaves the rewrite unapplied.
    with _handed() as handed:
        _, scripted = await _run(
            [_FILTERED, _BARE], [reply(text_part("done"))], question=_MARKED_UP
        )

    config = handed[0]
    _, contents = _dumped(scripted.requests[0])
    assert [len(tool["function_declarations"]) for tool in config["tools"]] == [1, 1]
    declared = [tool["function_declarations"][0] for tool in config["tools"]]
    assert declared[0]["parameters_json_schema"] == (
        function_declaration(_FILTERED).parameters_json_schema
    )
    assert "$defs" in declared[0]["parameters_json_schema"]
    assert declared[1]["parameters_json_schema"] == {"type": "object", "properties": {}}
    assert config["tool_config"] == {"function_calling_config": {"mode": "AUTO"}}
    assert config["response_modalities"] == ["TEXT"]
    assert config["system_instruction"] == {"role": "user", "parts": [{"text": INSTRUCTION}]}
    assert contents[0]["parts"] == [{"text": _MARKED_UP}]


async def test_control_gemini_own_profile_rewrites_the_request_the_planner_profile_keeps() -> None:
    # Shows the profile keys are what the test above depends on.
    scripted = ScriptedGenai([reply(text_part("done"))])
    own = GoogleModel(MODEL, provider=GoogleProvider(client=scripted.client))
    await _run([_FILTERED], [], question=_MARKED_UP, model=own, scripted=scripted)

    config, contents = _dumped(scripted.requests[0])
    declared = config["tools"][0]["function_declarations"][0]
    assert (
        declared["parameters_json_schema"] != function_declaration(_FILTERED).parameters_json_schema
    )
    assert config["system_instruction"]["parts"] != [{"text": INSTRUCTION}]
    assert contents[0]["parts"] != [{"text": _MARKED_UP}]


async def _stream_run(agent: Agent[Any, Any], prompt: str = BRIEF, **options: Any) -> Any:
    """A run of `agent` in which every request streams, as `run_planner`'s do."""
    async with agent.iter(prompt, usage_limits=UsageLimits(request_limit=None), **options) as run:
        async for node in run:
            if Agent.is_model_request_node(node):
                async with node.stream(run.ctx) as stream:
                    async for _event in stream:
                        pass
    assert run.result is not None
    return run.result


def _plain_agent(scripted: ScriptedGenai, **options: Any) -> Agent[Any, Any]:
    """An agent on the planner's model profile with none of the planner's other settings."""
    model = GoogleModel(
        MODEL, provider=GoogleProvider(client=scripted.client), profile=PLANNER_PROFILE
    )
    options.setdefault("output_type", str | None)
    return Agent(model, instructions=INSTRUCTION, **options)


async def test_control_a_tool_left_strict_by_default_puts_gemini_in_validated_mode() -> None:
    # `planner_agent` declares each tool `strict=False` so the mode is `AUTO`.
    scripted = ScriptedGenai([reply(text_part("done"))])
    schema = function_declaration(_FILTERED).parameters_json_schema or {}
    tool = Tool.from_schema(
        lambda **raw: raw, name="filtered", description="d", json_schema=dict(schema)
    )

    await _stream_run(_plain_agent(scripted, tools=[tool]))

    config, _ = _dumped(scripted.requests[0])
    assert config["tool_config"]["function_calling_config"]["mode"] == "VALIDATED"


async def test_a_call_with_arguments_the_schema_refuses_reaches_the_body_unvalidated() -> None:
    # Pydantic AI validates arguments against a tool's schema unless the function takes
    # them raw. The planner answers a missing argument itself (`run_tool`), where
    # validation would send the model a retry prompt.
    seen: list[PlannerCall] = []
    spec = _tool("needs", parameters=(Parameter("name", str),), seen=seen)
    stray = call_part("needs", {"name": 7, "stray": True}, id="c1")
    missing = call_part("needs", {}, id="c2")

    items, scripted = await _run([spec], [reply(stray, missing), reply(text_part("done"))])

    answers = {answer.id: answer.response or {} for answer in _answers(items)}
    assert answers["c1"] == {"outcome": "ok"}
    assert answers["c2"].get("outcome") != "ok" and "name" in str(answers["c2"])
    assert len(scripted.requests) == 2


# --- A tool's call ids ----------------------------------------------------------------


async def test_a_call_keeps_the_id_the_model_sent() -> None:
    # Answers are paired with calls by id, and a stored step names its call. An id
    # the model did not send is given in ADK's form by `harness.planner_model`
    # before Pydantic AI reads the response, and `tests.test_planner_runners` covers that.
    items, _ = await _run(
        [_BARE],
        [reply(call_part("bare", id="sent"), call_part("bare", id="also")), reply(text_part("ok"))],
    )

    called = [part.function_call.id for part in _responses(items)[0].parts if part.function_call]
    assert called == ["sent", "also"]
    assert [answer.id for answer in _answers(items)] == ["sent", "also"]


# --- Unknown tools and the retry allowance --------------------------------------------


def _unknown_calls(count: int) -> list[Reply]:
    return [reply(call_part("nope", id=f"c{i}")) for i in range(count)] + [reply(text_part("done"))]


async def test_a_call_to_an_unknown_tool_is_answered_as_the_adk_planner_answers_it() -> None:
    # Without the processor that rewrites the answer, the model reads Pydantic AI's
    # retry prompt, and without the lifted retry allowance the run ends in an error.
    items, scripted = await _run([_BARE], _unknown_calls(3))

    expected = tool_not_found("nope", ["bare"])
    assert [answer.response for answer in _answers(items)] == [expected] * 3
    sent = scripted.requests[-1].content_list[-1].parts or []
    assert [part.function_response.response for part in sent if part.function_response] == [
        expected
    ]
    assert len(scripted.requests) == 4


async def test_control_a_plain_agent_charges_each_unknown_call_against_its_retries() -> None:
    # `planner_agent` lifts the retry allowance. A plain agent ends on the second unknown call.
    scripted = ScriptedGenai(_unknown_calls(2))

    with pytest.raises(UnexpectedModelBehavior):
        await _stream_run(_plain_agent(scripted, retries=1))


# --- Where a run stops ----------------------------------------------------------------


async def test_an_empty_response_ends_the_run_with_no_second_request() -> None:
    # Pydantic AI retries a response with no parts unless the output type allows None.
    items, scripted = await _run([_BARE], [empty_reply()])

    assert len(scripted.requests) == 1
    assert [response.parts for response in _responses(items)] == [()]


async def test_a_blocked_prompt_ends_the_run_and_raises_nothing() -> None:
    # Pydantic AI hands over a blocked prompt as a response with no parts, which the
    # node stores no event for.
    items, scripted = await _run([_BARE], [blocked_reply(types.BlockedReason.SAFETY)])

    assert len(scripted.requests) == 1
    assert [r.parts for r in _responses(items)] == [()]


async def test_a_stream_with_no_chunk_ends_the_run_and_raises_nothing() -> None:
    # Pydantic AI refuses a response that streamed nothing with `UnexpectedModelBehavior`,
    # which `run_planner` treats as the end of the run.
    items, scripted = await _run([_BARE], [Reply(chunks=())])

    assert items == []
    assert len(scripted.requests) == 1


async def test_a_run_ends_before_the_request_after_a_call_that_ended_it() -> None:
    # The model request node that follows the returns is reached before its request is
    # sent, which is where the run stops on an ask or a ready plan.
    items, scripted = await _run(
        [_tool("finish", ends=True)], [reply(call_part("finish", id="c1")), reply(text_part("x"))]
    )

    assert len(scripted.requests) == 1
    assert [answer.id for answer in _answers(items)] == ["c1"]


async def test_a_run_goes_past_the_default_cap_of_fifty_requests() -> None:
    # Pydantic AI stops a run at 50 requests unless the limit is lifted.
    replies = [reply(call_part("bare", id=f"c{i}")) for i in range(52)] + [reply(text_part("x"))]

    _, scripted = await _run([_BARE], replies)

    assert len(scripted.requests) == 53


# --- When a tool runs within the run --------------------------------------------------


def _texts(content: types.Content) -> str | None:
    return (content.parts or [types.Part()])[0].text


async def test_a_responses_calls_run_after_it_is_yielded_and_before_the_next_request() -> None:
    # A planner tool reads the session's events (`planner.session.SessionCall`), which
    # `harness.planner_node` appends to as `run_planner` yields. So each call of a
    # response runs once the response is yielded, and no request is sent until every
    # call has answered: Pydantic AI runs no tool while a model request streams.
    yielded: list[PlannerItem] = []
    seen: list[list[str]] = []

    def body(call: PlannerCall, end_run: EndRun, arguments: dict[str, Any]) -> object:
        del call, end_run, arguments
        seen.append([type(item).__name__ for item in yielded])
        return {"outcome": "ok"}

    await _run(
        [ToolSpec(name="bare", description="d", parameters=(), body=body)],
        [
            reply(call_part("bare", id="a"), call_part("bare", id="b")),
            reply(call_part("bare", id="c")),
            reply(text_part("done")),
        ],
        yielded=yielded,
    )

    first = ["PlannerResponse"]
    assert seen == [first, first, [*first, "PlannerAnswers", *first]]


async def test_the_answers_of_a_response_come_in_call_order_whatever_order_they_finish_in() -> None:
    # `run_planner` takes a response's answers from the tool returns Pydantic AI
    # collects into the next request, which holds them in the order of the calls.
    async def slow(call: PlannerCall, end_run: EndRun, arguments: dict[str, Any]) -> object:
        await asyncio.sleep(0.05)
        return {"which": "slow"}

    spec = ToolSpec(name="slow", description="d", parameters=(), body=slow)

    items, _ = await _run(
        [spec, _BARE],
        [reply(call_part("slow", id="a"), call_part("bare", id="b")), reply(text_part("done"))],
    )

    assert [answer.id for answer in _answers(items)] == ["a", "b"]


# --- Request shape across messages ----------------------------------------------------


async def test_the_users_message_and_the_brief_reach_the_model_as_two_contents() -> None:
    # Pydantic AI merges consecutive requests into one content unless their
    # `instructions` differ, and the ADK planner sends two.
    _, scripted = await _run([_BARE], [reply(text_part("done"))])
    _, alone = await _run([_BARE], [reply(text_part("done"))], question=None)

    assert [_texts(content) for content in scripted.requests[0].content_list] == [
        QUESTION,
        BRIEF,
    ]
    assert [_texts(content) for content in alone.requests[0].content_list] == [BRIEF]


@pytest.mark.parametrize(
    ("instructions", "contents"),
    [(None, 1), ("The user's message.", 2)],
    ids=["no instructions merge into one content", "differing instructions stay two"],
)
async def test_control_consecutive_requests_merge_unless_their_instructions_differ(
    instructions: str | None, contents: int
) -> None:
    scripted = ScriptedGenai([reply(text_part("done"))])
    history = [ModelRequest(parts=[UserPromptPart(QUESTION)], instructions=instructions)]

    await _stream_run(_plain_agent(scripted), message_history=history)

    assert len(scripted.requests[0].content_list) == contents


# --- Thoughts and their signatures ----------------------------------------------------


async def test_a_signed_thought_and_a_signed_call_replay_with_their_signatures() -> None:
    # Gemini rejects a request whose history drops a signature it issued.
    signed = reply(
        thought_part("plan", signature=b"thought-sig"),
        call_part("bare", id="c", signature=b"call-sig"),
    )

    items, scripted = await _run([_BARE], [signed, reply(text_part("done"))])

    thought, call = _responses(items)[0].parts
    assert (thought.thought, thought.thought_signature) == (True, b"thought-sig")
    assert call.thought_signature == b"call-sig"
    replayed = scripted.requests[1].content_list[2].parts or []
    assert [(part.thought, part.thought_signature) for part in replayed] == [
        (True, b"thought-sig"),
        (None, b"call-sig"),
    ]


async def test_the_first_unsigned_call_of_a_response_replays_with_the_skip_signature() -> None:
    # Gemini answers 400 to a replayed first call with no signature. Pydantic AI sends
    # its documented placeholder, which `assert_pinned_wire_differences` expects.
    items, scripted = await _run(
        [_BARE],
        [reply(call_part("bare", id="a"), call_part("bare", id="b")), reply(text_part("done"))],
    )

    first, second = scripted.requests[1].content_list[2].parts or []
    assert first.thought_signature == SKIP_THOUGHT_SIGNATURE
    assert second.thought_signature is None
    assert _responses(items)[0].parts[0].thought_signature is None


async def test_consecutive_thought_chunks_make_one_part_that_keeps_the_signature() -> None:
    # The planner drops unsigned thoughts from the parts this merge makes
    # (`planner.pydantic_agent._without_unsigned_thoughts`), so a run of thought
    # chunks with a signature on one chunk stays whole, as on the ADK planner.
    chunks = Reply(
        chunks=[
            [thought_part("a", signature=None)],
            [thought_part("b", signature=b"sig")],
            [thought_part("c", signature=None)],
            [call_part("bare", id="c")],
        ]
    )

    items, scripted = await _run([_BARE], [chunks, reply(text_part("done"))])

    thought, _call = _responses(items)[0].parts
    assert (thought.text, thought.thought_signature) == ("abc", b"sig")
    replayed = scripted.requests[1].content_list[2].parts or []
    assert [part.text for part in replayed if part.thought] == ["abc"]


async def test_an_unsigned_thought_summary_is_left_out_of_the_next_request() -> None:
    # The ADK planner's session keeps no unsigned thought summary, so the next request
    # carries none (`planner.settings.without_unsigned_thoughts`).
    unsigned = reply(thought_part("musing", signature=None), call_part("bare", id="c"))

    items, scripted = await _run([_BARE], [unsigned, reply(text_part("done"))])

    replayed = scripted.requests[1].content_list[2].parts or []
    assert [part.thought for part in replayed] == [None]
    assert [part.thought for part in _responses(items)[0].parts] == [None]


# --- The client the model calls -------------------------------------------------------


async def test_every_request_reaches_the_clients_streamed_call_with_three_keywords() -> None:
    # `harness.planner_model` answers the client's streamed call with a keyword-only
    # signature of exactly these three, and refuses a call that does not stream.
    scripted = ScriptedGenai([reply(call_part("bare", id="c")), reply(text_part("done"))])
    model = planner_model(MODEL, client=scripted.client, budget=None)
    streamed = model.client.aio.models.generate_content_stream
    sent: list[set[str]] = []

    async def spy(**kwargs: Any) -> Any:
        sent.append(set(kwargs))
        return await streamed(**kwargs)

    model.client.aio.models.generate_content_stream = spy  # ty: ignore[invalid-assignment]
    planner = planner_agent([_BARE], instruction=INSTRUCTION, model=model)
    deps = PlannerDeps(
        state={}, call_for=_no_events, tool_answered=lambda *_: None, tool_failed=lambda *_: {}
    )

    [_ async for _ in run_planner(planner, deps, question=QUESTION, brief=BRIEF)]

    assert sent == [{"model", "contents", "config"}] * 2


# --- Spans ----------------------------------------------------------------------------


async def test_the_clients_call_runs_inside_the_chat_span_of_the_conversation(
    exported_spans: InMemorySpanExporter,
) -> None:
    # `harness.turn_budget.bounded_by` stamps `dfo.model.*` on the span current when the
    # call is made. That span has to be the generation's, which Pydantic AI opens
    # under the planner's run span.
    scripted = ScriptedGenai([reply(text_part("done"))])
    model = planner_model(MODEL, client=scripted.client, budget=None)
    streamed = model.client.aio.models.generate_content_stream
    current: list[tuple[bool, str]] = []

    async def spy(**kwargs: Any) -> Any:
        span = trace.get_current_span()
        current.append((span.is_recording(), getattr(span, "name", "")))
        return await streamed(**kwargs)

    model.client.aio.models.generate_content_stream = spy  # ty: ignore[invalid-assignment]
    planner = planner_agent([_BARE], instruction=INSTRUCTION, model=model, name="dfo_planner")
    deps = PlannerDeps(
        state={}, call_for=_no_events, tool_answered=lambda *_: None, tool_failed=lambda *_: {}
    )

    [
        _
        async for _ in run_planner(
            planner, deps, question=QUESTION, brief=BRIEF, conversation_id="conv"
        )
    ]

    assert [(recording, name.partition(" ")[0]) for recording, name in current] == [(True, "chat")]
    spans = {span.name.split()[0]: span for span in exported_spans.get_finished_spans()}
    assert (spans["chat"].attributes or {})["gen_ai.conversation.id"] == "conv"
    assert spans["chat"].parent is not None
    assert spans["chat"].parent.span_id == spans["invoke_agent"].context.span_id


async def _traced_run(specs: Sequence[ToolSpec], replies: Sequence[Reply]) -> list[PlannerItem]:
    """`run_planner` over `specs` on `replies`, its spans exported."""
    scripted = ScriptedGenai(replies)
    model = planner_model(MODEL, client=scripted.client, budget=None)
    planner = planner_agent(specs, instruction=INSTRUCTION, model=model, name="dfo_planner")
    deps = PlannerDeps(
        state={}, call_for=_no_events, tool_answered=lambda *_: None, tool_failed=lambda *_: {}
    )
    return [
        item
        async for item in run_planner(
            planner, deps, question=QUESTION, brief=BRIEF, conversation_id="conv"
        )
    ]


async def test_a_tools_body_runs_with_its_execute_tool_span_current(
    exported_spans: InMemorySpanExporter,
) -> None:
    # `planner.pydantic_agent` marks a refused call's span failed through the span
    # current while the tool function runs. That span has to be the call's own.
    current: list[str] = []

    def body(call: PlannerCall, end_run: EndRun, arguments: dict[str, Any]) -> object:
        del call, end_run, arguments
        current.append(getattr(trace.get_current_span(), "name", ""))
        return {"outcome": "ok"}

    spec = ToolSpec(name="work", description="Tool work.", parameters=(), body=body)

    await _traced_run([spec], [reply(call_part("work", id="c1")), reply(text_part("done"))])

    assert current == ["execute_tool work"]
    (span,) = [s for s in exported_spans.get_finished_spans() if s.name == "execute_tool work"]
    assert (span.attributes or {})["gen_ai.tool.call.id"] == "c1"


async def test_a_refused_calls_failed_mark_stands_and_no_span_is_marked_ok(
    exported_spans: InMemorySpanExporter,
) -> None:
    # `planner.pydantic_agent` marks a refused call's span ERROR while it runs. A
    # status of OK set after it would override it, so Pydantic AI must set none.
    spec = _tool("needs", parameters=(Parameter("name", str),))

    await _traced_run([spec], [reply(call_part("needs", id="c1")), reply(text_part("done"))])

    spans = exported_spans.get_finished_spans()
    (refused,) = [span for span in spans if span.name == "execute_tool needs"]
    assert (refused.status.status_code, refused.status.description) == (
        StatusCode.ERROR,
        "TOOL_ERROR",
    )
    assert (refused.attributes or {})["error.type"] == "TOOL_ERROR"
    assert [span.name for span in spans if span.status.status_code is StatusCode.OK] == []


async def test_the_usage_the_model_records_on_its_span_is_left_as_it_is(
    exported_spans: InMemorySpanExporter,
) -> None:
    # `harness.planner_model` writes the thought tokens under ADK's attribute and
    # the usage Langfuse reads ahead of the rest while the call runs. Pydantic AI
    # writes its own usage when the span ends and must leave those two alone.
    usage = usage_metadata(prompt=120, cached=40, candidates=9, thoughts=4)

    await _traced_run([_BARE], [reply(text_part("done"), usage=usage)])

    (chat,) = [
        span.attributes or {}
        for span in exported_spans.get_finished_spans()
        if span.name.startswith("chat ")
    ]
    assert chat["gen_ai.usage.reasoning.output_tokens"] == 4
    assert json.loads(str(chat["langfuse.observation.usage_details"])) == {
        "input": 80,
        "output": 9,
        "input_cached_tokens": 40,
        "output_reasoning_tokens": 4,
    }

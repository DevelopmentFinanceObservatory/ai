"""What a Langfuse reader sees of a planner entry on each runner.

Each entry runs through the seam on the ADK planner and on the Pydantic AI
planner over the same scripted google-genai client (`tests.test_planner_node`),
and the planner's spans are read the way Langfuse's OTLP ingestion reads them
(`_seen`): the observation type, name, model, session, input, output, usage,
level and status message, the completion start time, and the attributes the
turn budget writes. The two runners must show a reader the same thing, apart
from these differences, each asserted exactly (`_as_adk_names_it`):

- a model call's span is `chat <model>` where ADK's is `generate_content <model>`;
- the run's span is named `dfo_planner run` where ADK's is
  `invoke_agent dfo_planner`, and it carries the run's messages and final text
  as its input and output where ADK's carries none;
- in a model call's input, a replayed model turn carries its finish reason and
  a tool's answer is a `tool_call_response` part with `name` and `result`
  where ADK's has `response`;
- a tool declaration with no parameters has an empty object schema where
  ADK's has none;
- a tool call's span names the conversation where ADK's names none;
- ADK writes each streamed response of a call as an output message of its
  own, ending on the whole response, which is the one message Pydantic AI
  writes (`_as_one_message`);
- on a reply streamed over several chunks, ADK writes one output message per
  chunk, a thought as plain text, with an empty finish reason on each and no
  whole response, where Pydantic AI writes one merged message, its thought
  typed as thinking, finishing on `stop`
  (`test_a_reply_streamed_in_chunks_is_one_message_on_pydantic_ai_and_one_per_chunk_on_adk`);
- a call the turn budget closed, with no model call behind it, ends its
  output on a `stop` finish reason where ADK's has none, and gets a
  completion start time from Pydantic AI's time to first chunk;
- a turn put in place of a request (a stop) is a span of its own, which ADK
  opens none for, marked a plain span with no usage.

The last test runs the real Langfuse exporter against a stand-in server
(`fake_langfuse`) to show the session and the trace name a turn sets reach the
planner's spans on both runners.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import Any

import pytest
from common.telemetry import (
    TURN_SPAN_NAME,
    flush_telemetry,
    init_telemetry,
    start_turn_span,
    use_turn_span,
)
from harness.turn_budget import COMPLETION_START_TIME, TurnBudget
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode
from planner.plan import Plan
from planner.settings import AGENT_NAME
from planner.specs import PROPOSE_PLAN
from planner.tool_spec import Parameter, ToolSpec

from ..scripted_genai import Reply, call_part, reply, thought_part
from ..test_planner_node import (
    _MODEL,
    _REFUSED,
    _SIGNED,
    _STREAMED,
    _USAGE,
    Runner,
    _entries,
    _numbered_call_ids,
    _one_call_budget,
    _planned,
    _proposals,
    _referral,
    _Script,
    _serving_proposals,
)
from ..test_planner_seam import _translated_plan

_RUNNERS: tuple[Runner, Runner] = ("adk", "pydantic_ai")
_SESSION = "conv_01planner_spans"
# Where a span's conversation is the session the run's workflow ran in.
_THE_SESSION = "<the session>"

# Langfuse's `LangfuseOtelSpanAttributes` names a span carries.
_OBSERVATION_TYPE = "langfuse.observation.type"
_USAGE_DETAILS = "langfuse.observation.usage_details"

# The observation type Langfuse gives each GenAI operation
# (`ObservationTypeMapper`).
_TYPES = {
    "chat": "generation",
    "generate_content": "generation",
    "invoke_agent": "agent",
    "execute_tool": "tool",
}

# The `gen_ai.usage.*` keys Langfuse folds into input, output and the cache
# and reasoning buckets (`OtelIngestionProcessor.extractGenericGenAiUsageDetails`);
# any other key is a usage detail of its own, `details.` dropped.
_CACHE_READ_KEYS = ("cache_read.input_tokens", "details.cache_read_tokens")
_FOLDED_KEYS = {"input_tokens", "output_tokens", "reasoning.output_tokens", *_CACHE_READ_KEYS}


@pytest.fixture
def plan(decision: Any, requirement: Any, derived_requirement: Any) -> Plan:
    return _translated_plan(decision, requirement, derived_requirement)


@pytest.fixture(autouse=True)
def _span_content(monkeypatch: pytest.MonkeyPatch) -> None:
    """Message content on the spans, as `common.telemetry` sets it when a backend comes up."""
    monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "SPAN_ONLY")


def _budget() -> TurnBudget:
    """A budget no entry here reaches, so each call runs under it and is timed."""
    return TurnBudget(model_time_s=60.0, model_calls_max=10, wall_s=60.0, first_chunk_s=30.0)


# --- Reading a span as Langfuse does ----------------------------------------------------


def _json(value: Any) -> Any:
    return json.loads(value) if isinstance(value, str) else value


def _generic_usage(attributes: Mapping[str, Any]) -> dict[str, int]:
    """The usage Langfuse derives from a span's `gen_ai.usage.*` attributes."""
    raw = {
        key.removeprefix("gen_ai.usage."): int(value)
        for key, value in attributes.items()
        if key.startswith("gen_ai.usage.")
    }
    if not raw:
        return {}
    usage = {key.removeprefix("details."): v for key, v in raw.items() if key not in _FOLDED_KEYS}
    cached = next((raw[key] for key in _CACHE_READ_KEYS if key in raw), None)
    reasoning = raw.get("reasoning.output_tokens")
    if "input_tokens" in raw:
        usage["input"] = max(raw["input_tokens"] - (cached or 0), 0)
    if "output_tokens" in raw:
        usage["output"] = max(raw["output_tokens"] - (reasoning or 0), 0)
    if cached is not None:
        usage["input_cached_tokens"] = cached
    if reasoning is not None:
        usage["output_reasoning_tokens"] = reasoning
    return usage


def _usage(attributes: Mapping[str, Any]) -> dict[str, int]:
    """The usage Langfuse records for a span: its usage details when it carries
    them, else what it derives from `gen_ai.usage.*`, with the total it adds as
    the sum of the rest."""
    usage = (
        _json(attributes[_USAGE_DETAILS])
        if _USAGE_DETAILS in attributes
        else _generic_usage(attributes)
    )
    return {**usage, "total": sum(usage.values())} if usage else {}


def _input_output(span: ReadableSpan) -> tuple[Any, Any]:
    """A span's input and output, by Langfuse's order of sources."""
    attributes = span.attributes or {}
    scope = span.instrumentation_scope.name if span.instrumentation_scope else ""
    if scope == "pydantic-ai" and (
        "pydantic_ai.all_messages" in attributes or "final_result" in attributes
    ):
        return _json(attributes.get("pydantic_ai.all_messages")), attributes.get("final_result")
    if "gen_ai.input.messages" in attributes or "gen_ai.output.messages" in attributes:
        system = _json(attributes.get("gen_ai.system_instructions"))
        messages = _json(attributes.get("gen_ai.input.messages"))
        return (system, messages), _json(attributes.get("gen_ai.output.messages"))
    if "gen_ai.tool.call.arguments" in attributes or "gen_ai.tool.call.result" in attributes:
        return (
            _json(attributes.get("gen_ai.tool.call.arguments")),
            _json(attributes.get("gen_ai.tool.call.result")),
        )
    return None, None


def _completion_start(attributes: Mapping[str, Any]) -> str | None:
    """Where Langfuse takes a span's completion start time from, if anywhere."""
    if COMPLETION_START_TIME in attributes:
        return "set"
    if "gen_ai.client.operation.time_to_first_chunk" in attributes:
        return "time to first chunk"
    return None


def _seen(span: ReadableSpan, session: str) -> dict[str, Any]:
    """What Langfuse shows a reader of `span`, a planner span of a run whose
    workflow ran in the conversation `session`."""
    attributes = span.attributes or {}
    conversation = attributes.get("gen_ai.conversation.id")
    input_, output = _input_output(span)
    return {
        "type": attributes.get(_OBSERVATION_TYPE)
        or _TYPES[str(attributes["gen_ai.operation.name"])],
        "name": attributes.get("gen_ai.tool.name") or attributes.get("logfire.msg") or span.name,
        "model": attributes.get("gen_ai.response.model") or attributes.get("gen_ai.request.model"),
        "session": _THE_SESSION if conversation == session else conversation,
        "agent": attributes.get("gen_ai.agent.name"),
        "input": input_,
        "output": output,
        "usage": _usage(attributes),
        "level": "ERROR" if span.status.status_code is StatusCode.ERROR else None,
        "status_message": span.status.description,
        "error_type": attributes.get("error.type"),
        "completion_start": _completion_start(attributes),
        "tool_definitions": _json(attributes.get("gen_ai.tool.definitions")),
        "budget": {
            key: (value if key == "dfo.model.thoughts_token_count" else "timed")
            for key, value in attributes.items()
            if key.startswith("dfo.model.")
        },
    }


# --- The pinned differences -------------------------------------------------------------


def _as_adk_part(part: dict[str, Any]) -> dict[str, Any]:
    """A part of a Pydantic AI model call's input in the shape ADK writes it."""
    if part["type"] != "tool_call_response":
        return part
    assert set(part) == {"type", "id", "name", "result"}, part
    return {"type": part["type"], "id": part["id"], "response": part["result"]}


def _as_adk_input(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A Pydantic AI model call's input messages in the shape ADK writes them."""
    shaped = []
    for message in messages:
        if message["role"] == "assistant":
            assert "finish_reason" in message
        shaped.append(
            {
                **{key: value for key, value in message.items() if key != "finish_reason"},
                "parts": [_as_adk_part(part) for part in message["parts"]],
            }
        )
    return shaped


def _as_adk_names_it(seen: dict[str, Any]) -> dict[str, Any]:
    """`seen`, a Pydantic AI planner span's, with each pinned difference put the
    way the ADK planner's span has it."""
    seen = dict(seen)
    if seen["type"] == "generation":
        assert seen["name"] == f"chat {_MODEL}"
        seen["name"] = f"generate_content {_MODEL}"
        system, messages = seen["input"]
        seen["input"] = (system, _as_adk_input(messages))
        seen["tool_definitions"] = [
            {**declaration, "parameters": None}
            if declaration["parameters"] == {"type": "object", "properties": {}}
            else declaration
            for declaration in seen["tool_definitions"]
        ]
        if not seen["usage"]:
            (message,) = seen["output"]
            assert message["finish_reason"] == "stop"
            seen["output"] = [{**message, "finish_reason": ""}]
            assert seen["completion_start"] == "time to first chunk"
            seen["completion_start"] = None
    if seen["type"] == "tool":
        assert seen["session"] == _THE_SESSION
        seen["session"] = None
    if seen["type"] == "agent":
        assert seen["name"] == f"{AGENT_NAME} run"
        seen["name"] = f"invoke_agent {AGENT_NAME}"
        assert isinstance(seen["input"], list), "the run's messages are its input"
        seen["input"] = seen["output"] = None
    return seen


def _as_one_message(seen: dict[str, Any]) -> dict[str, Any]:
    """`seen`, an ADK planner span's, with a call's output as the whole
    response alone, the message its streamed partial responses lead up to."""
    if seen["type"] != "generation":
        return seen
    *partials, whole = seen["output"]
    assert all(message["finish_reason"] == "" for message in partials)
    return {**seen, "output": [whole]}


# --- Running an entry -------------------------------------------------------------------


async def _spans(runner: Runner, script: _Script, exporter: InMemorySpanExporter) -> list[Any]:
    """What Langfuse shows of the planner's spans in `script`'s entry on
    `runner`, in the order they started."""
    exporter.clear()
    with _numbered_call_ids():
        await _entries(runner, script)
    spans = sorted(exporter.get_finished_spans(), key=lambda span: span.start_time or 0)
    (workflow,) = [span for span in spans if span.name.startswith("invoke_workflow")]
    session = str((workflow.attributes or {})["gen_ai.conversation.id"])
    return [
        _seen(span, session)
        for span in spans
        if (span.attributes or {}).get("gen_ai.agent.name") == AGENT_NAME
    ]


def _needing_a_note(specs: Sequence[ToolSpec]) -> list[ToolSpec]:
    """The planner's tools with a `propose_plan` that requires a `note`, and
    otherwise answers as `_proposals()` does."""
    return [
        replace(spec, parameters=(Parameter("note", str),)) if spec.name == PROPOSE_PLAN else spec
        for spec in _proposals()(specs)
    ]


def _raising(specs: Sequence[ToolSpec]) -> list[ToolSpec]:
    """The planner's tools with a `propose_plan` that raises, which ends the run on a stop."""

    def body(*_: Any) -> object:
        raise RuntimeError("propose_plan broke")

    return _serving_proposals(specs, body)


def _proposal(id: str, **args: Any) -> Reply:
    return reply(call_part(PROPOSE_PLAN, args, id=id, signature=_SIGNED), usage=_USAGE)


_ENTRIES: dict[str, Callable[[Plan], _Script]] = {
    # A refused proposal, then an accepted one.
    "refused-then-accepted": lambda plan: _Script(
        replies=[_proposal("p1", draft=1), _proposal("p2")],
        runs=[_planned(plan)],
        setup=_proposals(_REFUSED),
        budget=_budget,
    ),
    # A call missing a required argument, refused and marked failed, then a proposal.
    "missing-argument": lambda plan: _Script(
        replies=[_proposal("p1"), _proposal("p2", note="pin it")],
        runs=[_planned(plan)],
        setup=_needing_a_note,
        budget=_budget,
    ),
    # The second call is past the turn's call cap and is closed with no model call.
    "budget-closed": lambda plan: _Script(
        replies=[_proposal("p1")],
        runs=[_planned(plan)],
        setup=_proposals(_referral({"sentence": "a nearby series"})),
        budget=_one_call_budget,
    ),
    # A tool that raises ends the run on a stop put in place of the next request.
    "stopped": lambda plan: _Script(
        replies=[_proposal("p1")],
        runs=[_planned(plan)],
        setup=_raising,
        budget=_budget,
    ),
}


# --- Tests ------------------------------------------------------------------------------


@pytest.mark.parametrize("entry", list(_ENTRIES))
async def test_langfuse_shows_the_same_planner_on_both_runners(
    plan: Plan, exported_spans: InMemorySpanExporter, entry: str
) -> None:
    script = _ENTRIES[entry](plan)
    adk = await _spans("adk", script, exported_spans)
    pydantic = await _spans("pydantic_ai", script, exported_spans)

    put_in_place = [seen for seen in pydantic if seen["type"] == "span"]
    assert len(put_in_place) == (1 if entry == "stopped" else 0)
    for seen in put_in_place:
        assert seen["usage"] == {}
        assert seen["output"][0]["parts"][0]["content"].startswith("Stopping this turn:")
    observed = [_as_adk_names_it(seen) for seen in pydantic if seen["type"] != "span"]
    assert observed == [_as_one_message(seen) for seen in adk]
    assert {seen["session"] for seen in adk if seen["type"] != "tool"} == {_THE_SESSION}


async def test_a_reply_streamed_in_chunks_is_one_message_on_pydantic_ai_and_one_per_chunk_on_adk(
    plan: Plan, exported_spans: InMemorySpanExporter
) -> None:
    """A reply streamed as a signed and an unsigned thought chunk, two text
    chunks, a call and a usage-only chunk. Apart from the generation's output,
    a reader sees the same spans on both runners. The output differs in shape
    only. ADK writes each chunk with parts as a message of its own, the thought
    as plain text, and Pydantic AI one merged message. Read in order, both say
    the same words and make the same call."""
    script = _Script(
        replies=[_STREAMED["thoughts-text-and-call"]],
        runs=[_planned(plan)],
        setup=_proposals(),
        budget=_budget,
    )
    adk = await _spans("adk", script, exported_spans)
    pydantic = await _spans("pydantic_ai", script, exported_spans)

    (adk_call,) = [seen for seen in adk if seen["type"] == "generation"]
    (pydantic_call,) = [_as_adk_names_it(seen) for seen in pydantic if seen["type"] == "generation"]
    call = {"type": "tool_call", "id": "p1", "name": "propose_plan", "arguments": {}}
    assert adk_call["output"] == [
        {"role": "assistant", "parts": [part], "finish_reason": ""}
        for part in [
            {"type": "text", "content": "Weighing the sources"},
            {"type": "text", "content": " and the years."},
            {"type": "text", "content": "Proposing "},
            {"type": "text", "content": "now."},
            call,
        ]
    ]
    assert pydantic_call["output"] == [
        {
            "role": "assistant",
            "parts": [
                {"type": "thinking", "content": "Weighing the sources and the years."},
                {"type": "text", "content": "Proposing now."},
                call,
            ],
            "finish_reason": "stop",
        }
    ]
    (merged,) = pydantic_call["output"]
    assert "".join(
        part["content"]
        for message in adk_call["output"]
        for part in message["parts"]
        if "content" in part
    ) == "".join(part["content"] for part in merged["parts"] if "content" in part)
    for key in ("usage", "model", "session", "level"):
        assert pydantic_call[key] == adk_call[key], key
    without_output = [
        {**_as_adk_names_it(seen), "output": None} for seen in pydantic if seen["type"] != "span"
    ]
    assert without_output == [{**seen, "output": None} for seen in adk]


async def test_a_refused_call_is_marked_failed_and_usage_is_counted_once_on_both(
    plan: Plan, exported_spans: InMemorySpanExporter
) -> None:
    """The values behind the comparison above, for the entry with a refused
    call. The refusal marks the call's span failed, and each generation's usage
    is Gemini's, split as Langfuse splits it. Read from the `gen_ai.usage.*`
    attributes alone, the Pydantic AI planner's total would count the cached
    and thought tokens twice."""
    script = _ENTRIES["missing-argument"](plan)
    adk = await _spans("adk", script, exported_spans)
    exported_spans.clear()
    with _numbered_call_ids():
        await _entries("pydantic_ai", script)
    calls = [
        span.attributes or {}
        for span in exported_spans.get_finished_spans()
        if (span.attributes or {}).get("gen_ai.operation.name") == "chat"
    ]

    refused = [seen for seen in adk if seen["level"] == "ERROR"]
    assert [(seen["type"], seen["error_type"]) for seen in refused] == [("tool", "TOOL_ERROR")]
    usage = {
        "input": 80,
        "output": 9,
        "input_cached_tokens": 40,
        "output_reasoning_tokens": 4,
        "total": 133,
    }
    assert [seen["usage"] for seen in adk if seen["type"] == "generation"] == [usage, usage]
    assert [_usage(call) for call in calls] == [usage, usage]
    assert sum(_generic_usage(calls[0]).values()) == usage["total"] + 40 + 4


@pytest.mark.parametrize("capture", ["SPAN_ONLY", "EVENT_ONLY", "NO_CONTENT", None])
async def test_a_dropped_thought_summary_reaches_a_span_only_with_span_content(
    plan: Plan,
    exported_spans: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
    capture: str | None,
) -> None:
    """An unsigned thought summary, which neither planner keeps of a response,
    is in its generation's output on both runners when message content is on
    the spans: ADK's as a text part, Pydantic AI's as a thinking part. With
    content kept off the spans (in events only, off, or the variable unset) no
    span of either runner carries it."""
    if capture is None:
        monkeypatch.delenv("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT")
    else:
        monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", capture)
    thought = "I weigh the two debt sources, sentinel-7f3a."
    script = _Script(
        replies=[
            reply(
                thought_part(thought, signature=None),
                call_part(PROPOSE_PLAN, {}, id="p1", signature=_SIGNED),
                usage=_USAGE,
            )
        ],
        runs=[_planned(plan)],
        setup=_proposals(),
        budget=_budget,
    )
    outputs: dict[Runner, Any] = {}
    carrying: dict[Runner, list[tuple[str, str]]] = {}
    for runner in _RUNNERS:
        exported_spans.clear()
        with _numbered_call_ids():
            await _entries(runner, script)
        spans = exported_spans.get_finished_spans()
        (call,) = [
            span.attributes or {}
            for span in spans
            if (span.attributes or {}).get("gen_ai.operation.name") in {"chat", "generate_content"}
        ]
        outputs[runner] = _json(call.get("gen_ai.output.messages"))
        carrying[runner] = [
            (span.name, key)
            for span in spans
            for key, value in (span.attributes or {}).items()
            if "sentinel-7f3a" in str(value)
        ]

    if capture != "SPAN_ONLY":
        assert carrying == {"adk": [], "pydantic_ai": []}
        return
    assert outputs["adk"][0]["parts"][0] == {"type": "text", "content": thought}
    assert outputs["pydantic_ai"][0]["parts"][0] == {"type": "thinking", "content": thought}
    assert outputs["pydantic_ai"][0]["parts"][1:] == outputs["adk"][0]["parts"][1:]


def _attribute(span: Any, key: str) -> Any:
    for attribute in span.attributes:
        if attribute.key == key:
            return getattr(attribute.value, attribute.value.WhichOneof("value"))
    return None


async def test_the_turns_session_and_trace_name_reach_the_planners_spans_on_both(
    plan: Plan, fake_langfuse: Any
) -> None:
    """Exported through the real Langfuse SDK and its span filter, each runner's
    planner spans reach the wire, from ADK's scope and Pydantic AI's
    respectively, with the turn's session and trace name."""
    init_telemetry(service_name="harness")
    script = _ENTRIES["refused-then-accepted"](plan)
    exported: dict[Runner, list[tuple[str, Any]]] = {}
    for runner in _RUNNERS:
        fake_langfuse.exports.clear()
        handle = start_turn_span(f"trace-{runner}", session_id=_SESSION)
        with use_turn_span(handle), _numbered_call_ids():
            await _entries(runner, script)
        await flush_telemetry()
        exported[runner] = [
            (scope, span)
            for scope, span in fake_langfuse.scoped_spans()
            if _attribute(span, "gen_ai.agent.name") == AGENT_NAME
        ]

    assert {scope for scope, _ in exported["adk"]} == {"gcp.vertex.agent"}
    assert {scope for scope, _ in exported["pydantic_ai"]} == {"pydantic-ai"}
    names = {runner: sorted(span.name for _, span in spans) for runner, spans in exported.items()}
    assert names["adk"] == [
        "execute_tool propose_plan",
        "execute_tool propose_plan",
        f"generate_content {_MODEL}",
        f"generate_content {_MODEL}",
        f"invoke_agent {AGENT_NAME}",
    ]
    assert names["pydantic_ai"] == sorted(
        name.replace("generate_content ", "chat ") for name in names["adk"]
    )
    for spans in exported.values():
        for _, span in spans:
            assert _attribute(span, "session.id") == _SESSION, span.name
            assert _attribute(span, "langfuse.trace.name") == TURN_SPAN_NAME, span.name

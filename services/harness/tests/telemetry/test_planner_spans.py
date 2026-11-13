"""What a Langfuse reader sees of a planner entry.

Each entry runs through the adapter on the Pydantic AI planner over a scripted
google-genai client (`tests.test_planner_node`), and the planner's spans are
read the way Langfuse's OTLP ingestion reads them (`_seen`): the observation
type, name, model, session, input, output, usage, level and status message, the
completion start time, and the attributes the turn budget writes. What a
reader sees is checked against each entry's stored expectation
(`tests.planner_expectations`).

The last test runs the real Langfuse exporter against a stand-in server
(`fake_langfuse`) to show the session and the trace name a turn sets reach the
planner's spans.
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

from ..planner_expectations import expect, expectation_name, shared
from ..scripted_genai import Reply, call_part, malformed_call_reply, reply, thought_part
from ..test_planner_adapter import _translated_plan
from ..test_planner_node import (
    _MODEL,
    _REFUSED,
    _SIGNED,
    _STREAMED,
    _USAGE,
    _entry,
    _one_call_budget,
    _planned,
    _proposals,
    _referral,
    _Script,
    _serving_proposals,
)

_SESSION = "conv_01planner_spans"
# Where a span's conversation is the session the run's workflow ran in.
_THE_SESSION = "<the session>"

# Langfuse's `LangfuseOtelSpanAttributes` names a span carries.
_OBSERVATION_TYPE = "langfuse.observation.type"
_USAGE_DETAILS = "langfuse.observation.usage_details"
# The number of a call's attempts discarded for a malformed function call
# (`harness.planner_model`).
_MALFORMED_DISCARDED = "dfo.model.malformed_attempts_discarded"

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


def _stored(seen: Sequence[dict[str, Any]], tools: str = "span_tools") -> list[dict[str, Any]]:
    """`seen`, the Pydantic AI planner's spans, as their stored expectation holds
    them: the system instruction of each model call, and of each turn put in a
    call's place, stored once as `span_instruction`, and its tool definitions
    as `tools` (`planner_expectations.shared`)."""
    stored = []
    for span in seen:
        if span["type"] not in {"generation", "span"} or span["input"] is None:
            stored.append(span)
            continue
        system, messages = span["input"]
        definitions = span["tool_definitions"]
        stored.append(
            {
                **span,
                "input": [shared(system, "span_instruction"), messages],
                "tool_definitions": None if definitions is None else shared(definitions, tools),
            }
        )
    return stored


# --- Running an entry -------------------------------------------------------------------


async def _spans(script: _Script, exporter: InMemorySpanExporter) -> list[Any]:
    """What Langfuse shows of the planner's spans in `script`'s entry, in the order
    they started."""
    exporter.clear()
    await _entry(script)
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


# The name an entry's tool definitions are stored under when they are not the
# planner's own (`span_tools`).
_TOOLS = {"missing-argument": "span_tools_note_required"}

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
async def test_langfuse_shows_each_entry_as_stored(
    plan: Plan,
    exported_spans: InMemorySpanExporter,
    entry: str,
    request: pytest.FixtureRequest,
) -> None:
    shown = await _spans(_ENTRIES[entry](plan), exported_spans)

    put_in_place = [seen for seen in shown if seen["type"] == "span"]
    assert len(put_in_place) == (1 if entry == "stopped" else 0)
    for seen in put_in_place:
        assert seen["usage"] == {}
        assert seen["output"][0]["parts"][0]["content"].startswith("Stopping this turn:")
    expect(expectation_name(request), _stored(shown, _TOOLS.get(entry, "span_tools")))


async def test_a_reply_streamed_in_chunks_is_one_merged_message(
    plan: Plan, exported_spans: InMemorySpanExporter, request: pytest.FixtureRequest
) -> None:
    """A reply streamed as a signed and an unsigned thought chunk, two text
    chunks, a call and a usage-only chunk is one output message: the thoughts
    merged as one thinking part, the text as one text part, the call, and the
    finish reason `stop`."""
    script = _Script(
        replies=[_STREAMED["thoughts-text-and-call"]],
        runs=[_planned(plan)],
        setup=_proposals(),
        budget=_budget,
    )
    shown = await _spans(script, exported_spans)

    (generation,) = [seen for seen in shown if seen["type"] == "generation"]
    assert generation["output"] == [
        {
            "role": "assistant",
            "parts": [
                {"type": "thinking", "content": "Weighing the sources and the years."},
                {"type": "text", "content": "Proposing now."},
                {"type": "tool_call", "id": "p1", "name": "propose_plan", "arguments": {}},
            ],
            "finish_reason": "stop",
        }
    ]
    expect(expectation_name(request), _stored(shown))


async def test_a_refused_call_is_marked_failed_and_usage_is_counted_once(
    plan: Plan, exported_spans: InMemorySpanExporter
) -> None:
    """The values behind the stored expectation, for the entry with a refused
    call. The refusal marks the call's span failed, and each generation's usage
    is Gemini's, split as Langfuse splits it. Read from the `gen_ai.usage.*`
    attributes alone, the planner's total would count the cached and thought
    tokens twice."""
    shown = await _spans(_ENTRIES["missing-argument"](plan), exported_spans)
    calls = [
        span.attributes or {}
        for span in exported_spans.get_finished_spans()
        if (span.attributes or {}).get("gen_ai.operation.name") == "chat"
    ]

    refused = [seen for seen in shown if seen["level"] == "ERROR"]
    assert [(seen["type"], seen["error_type"]) for seen in refused] == [("tool", "TOOL_ERROR")]
    usage = {
        "input": 80,
        "output": 9,
        "input_cached_tokens": 40,
        "output_reasoning_tokens": 4,
        "total": 133,
    }
    assert [seen["usage"] for seen in shown if seen["type"] == "generation"] == [usage, usage]
    assert [_usage(call) for call in calls] == [usage, usage]
    assert sum(_generic_usage(calls[0]).values()) == usage["total"] + 40 + 4
    assert [_MALFORMED_DISCARDED in call for call in calls] == [False, False]


async def test_a_resent_calls_span_carries_both_attempts_usage(
    plan: Plan, exported_spans: InMemorySpanExporter
) -> None:
    """A reply Gemini ended on a function call it could not parse is discarded and
    the request sent again within one call. The call's span carries the usage
    summed over both replies and counts the one discarded attempt."""
    script = _Script(
        replies=[malformed_call_reply(usage=_USAGE), _proposal("p1")],
        runs=[_planned(plan)],
        setup=_proposals(),
        budget=_budget,
    )
    exported_spans.clear()
    await _entry(script)
    (call,) = [
        span.attributes or {}
        for span in exported_spans.get_finished_spans()
        if (span.attributes or {}).get("gen_ai.operation.name") == "chat"
    ]

    assert _usage(call) == {
        "input": 160,
        "output": 18,
        "input_cached_tokens": 80,
        "output_reasoning_tokens": 8,
        "total": 266,
    }
    assert call[_MALFORMED_DISCARDED] == 1


@pytest.mark.parametrize("capture", ["SPAN_ONLY", "EVENT_ONLY", "NO_CONTENT", None])
async def test_a_dropped_thought_summary_reaches_a_span_only_with_span_content(
    plan: Plan,
    exported_spans: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
    capture: str | None,
    request: pytest.FixtureRequest,
) -> None:
    """An unsigned thought summary, which the planner keeps from no response, is in
    its generation's output as a thinking part when message content is on the
    spans. With content kept off the spans (in events only, off, or the variable
    unset) no span carries it."""
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
    exported_spans.clear()
    await _entry(script)
    spans = exported_spans.get_finished_spans()
    (call,) = [
        span.attributes or {}
        for span in spans
        if (span.attributes or {}).get("gen_ai.operation.name") == "chat"
    ]
    outputs = _json(call.get("gen_ai.output.messages"))
    carrying = [
        (span.name, key)
        for span in spans
        for key, value in (span.attributes or {}).items()
        if "sentinel-7f3a" in str(value)
    ]

    if capture != "SPAN_ONLY":
        assert carrying == []
        return
    assert outputs[0]["parts"][0] == {"type": "thinking", "content": thought}
    expect(expectation_name(request), outputs)


def _attribute(span: Any, key: str) -> Any:
    for attribute in span.attributes:
        if attribute.key == key:
            return getattr(attribute.value, attribute.value.WhichOneof("value"))
    return None


async def test_the_turns_session_and_trace_name_reach_the_planners_spans(
    plan: Plan, fake_langfuse: Any
) -> None:
    """Exported through the real Langfuse SDK and its span filter, the planner's
    spans reach the wire from the `pydantic-ai` scope, with the turn's session and
    trace name."""
    init_telemetry(service_name="harness")
    fake_langfuse.exports.clear()
    handle = start_turn_span("trace-planner", session_id=_SESSION)
    with use_turn_span(handle):
        await _entry(_ENTRIES["refused-then-accepted"](plan))
    await flush_telemetry()
    exported = [
        (scope, span)
        for scope, span in fake_langfuse.scoped_spans()
        if _attribute(span, "gen_ai.agent.name") == AGENT_NAME
    ]

    assert {scope for scope, _ in exported} == {"pydantic-ai"}
    tools = ["execute_tool propose_plan"] * 2
    assert sorted(span.name for _, span in exported) == [
        f"chat {_MODEL}",
        f"chat {_MODEL}",
        *tools,
        f"invoke_agent {AGENT_NAME}",
    ]
    for _, span in exported:
        assert _attribute(span, "session.id") == _SESSION, span.name
        assert _attribute(span, "langfuse.trace.name") == TURN_SPAN_NAME, span.name

"""Unit tests for `harness.studio_translate`'s pure helpers.

`_refusal_detail` builds the Activity feed's one-line reason for a failed
step; these tests pin its truncation behaviour directly, without a full
studio round trip. The `_interrupt_frames` tests below drive its elicitation
branch directly with plain `args` dicts -- what `_request_input_call` would
have already pulled off a real ADK event -- rather than building one, since
that branch's own logic (which shape's `candidates` to read, which option to
mark `recommended`) needs no real event to exercise.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from typing import Any

import pytest
from compiler.surface import ProgressCallback, ToolCallResult, ToolSchema
from executor.run import current_step_id
from google.adk import Context
from google.adk.events import Event, EventActions
from google.adk.events.event import NodeInfo
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.workflow import START, Workflow, node
from google.adk.workflow.utils._workflow_hitl_utils import REQUEST_INPUT_FUNCTION_CALL_NAME
from google.genai import types
from harness.conversation import current_conversation_id, current_conversation_key
from harness.graph import APPROVE_INTERRUPT, STEP_PROGRESS_KEY
from harness.planner_seam import PlannerSeamError
from harness.scripted_plan import scripted_plan
from harness.studio_state import ConversationState, StudioStore, _next_manifest
from harness.studio_translate import (
    _PLANNER_FAILED_DETAIL,
    _STEP_DETAIL_CAP,
    _ExecutorRecord,
    _RecordingExecutor,
    _RecordingSurface,
    _refusal_detail,
    _stream_fatal,
    _Translator,
)
from harness.wire import CompositionActor, CompositionBlock, StepProduced, StepStatus
from planner.plan import plan_revision
from planner.workflow import ToolId

from .resolved_revision import resolved_revision
from .test_planner_seam import ScriptedPlannerLlm, _seam


async def _dummy_executor(*, tool: str, args: dict[str, object]) -> dict[str, object]:
    del tool, args
    return {"outcome": "ok"}


def _translator() -> _Translator:
    return _Translator(
        conversation=ConversationState(conversation_id="c1"),
        recorder=_RecordingExecutor(_dummy_executor, store=StudioStore(InMemorySessionService())),
    )


async def _translator_with_store() -> tuple[_Translator, _RecordingExecutor, StudioStore]:
    """A translator whose recorder is bound to the same conversation it
    translates for -- what `land_component` needs (the (store, contextvar)
    pair), as opposed to `_translator()`'s unbound recorder, which is fine
    for the tests that drive `_component_frames` directly with an explicit
    `action` and never call `land_component` at all."""
    store = StudioStore(InMemorySessionService())
    current_conversation_key.set(("alice", "c1"))
    conversation = await store.create("alice", "c1")
    recorder = _RecordingExecutor(_dummy_executor, store=store)
    return _Translator(conversation=conversation, recorder=recorder), recorder, store


def test_a_short_refusal_passes_through_unmodified() -> None:
    refusal = {"code": "unsupported_component_type", "message": "no such component"}
    assert _refusal_detail(refusal) == "unsupported_component_type: no such component"


def test_the_platforms_trace_id_is_not_part_of_the_activity_detail() -> None:
    refusal = {
        "code": "internal_error",
        "message": "Internal error",
        "platform_trace_id": "41d94ed2",
    }

    assert _refusal_detail(refusal) == "internal_error: Internal error"


def test_a_real_member_cardinality_refusal_fits_whole() -> None:
    """The advice sentence -- scope can't narrow it, ask a narrower question --
    fits whole under the 400-character cap."""
    detail = (
        "member_cardinality is 480 for spec_ed5b608a610cdc5a4bc703cf7d1bc2d0, over the 400 ceiling"
    )
    message = (
        f"Over the coverage ceiling: {detail}. This spec_id's definition "
        "has more member series than one request can serve. scope.entities and "
        "scope.window do not change that. A narrower question, such as one naming "
        "a sector for aid flows, resolves to a smaller definition."
    )
    refusal = {"code": "over_ceiling", "message": message}

    result = _refusal_detail(refusal)

    assert result == f"over_ceiling: {message}"
    assert result.endswith("resolves to a smaller definition.")


def test_a_refusal_longer_than_the_cap_cuts_on_a_word_boundary() -> None:
    long_message = "word " * 200  # far past the cap, and never ends mid-sentence by accident
    refusal = {"code": "refused", "message": long_message}

    result = _refusal_detail(refusal)

    assert len(result) <= _STEP_DETAIL_CAP + len("...")
    assert result.endswith("...")
    assert not result[: -len("...")].endswith(" ")
    # what precedes the ellipsis is whole words, never a fragment
    body = result[len("refused: ") : -len("...")]
    assert body.strip().rsplit(" ", 1)[-1] == "word"


def test_a_non_string_fix_is_dumped_to_json_before_capping() -> None:
    refusal = {"code": "facet_not_available", "fix": {"issues": [{"axis": "gender"}]}}

    result = _refusal_detail(refusal)

    assert result.startswith("facet_not_available: ")
    assert '"axis": "gender"' in result


def test_a_planner_ask_marks_the_recommended_option_and_carries_its_rationale() -> None:
    # `planner.agent`'s `adk_request_input` tool call has no `payload`
    # wrapper -- `candidates`, `recommended_spec_id` and `rationale` are its
    # own call args directly (`_interrupt_frames`'s own comment).
    args = {
        "message": "Which source did you mean?",
        "candidates": [
            {"spec_id": "spec_a", "label": "reading A"},
            {"spec_id": "spec_b", "label": "reading B"},
        ],
        "recommended_spec_id": "spec_b",
        "rationale": "broader coverage of the requested years",
    }

    (frame,) = _translator()._interrupt_frames("call_1", args)

    assert frame.event == "elicitation.request"
    assert [(o.id, o.recommended) for o in frame.options] == [
        ("spec_a", False),
        ("spec_b", True),
    ]
    assert frame.rationale == "broader coverage of the requested years"


def test_a_clarify_ask_marks_no_option_recommended_and_carries_no_rationale() -> None:
    # The graph's own `clarify` node raises a `RequestInput` whose `payload`
    # carries `candidates` and nothing about a recommendation -- the shape
    # `_request_input_call` hands over for `CLARIFY_INTERRUPT`.
    args = {
        "message": "Which reading did you mean?",
        "payload": {
            "candidates": [
                {"spec_id": "spec_a", "label": "reading A"},
                {"spec_id": "spec_b", "label": "reading B"},
            ]
        },
    }

    (frame,) = _translator()._interrupt_frames("clarify", args)

    assert [o.recommended for o in frame.options] == [False, False]
    assert frame.rationale is None


def _plan_payload(**extra: object) -> dict[str, object]:
    """The planner node's `"ok"`-routed payload shape (`harness.graph`'s
    `planner` node), built from the same scripted plan the graph itself
    drafts in scripted mode."""
    plan = scripted_plan("kenya life expectancy", spec_id=None, parts=1)
    return {
        **plan.model_dump(mode="json"),
        "summary": "Kenya life expectancy, 1960-2023.",
        "revision": plan_revision(plan),
        **extra,
    }


def test_a_plan_proposed_ask_carries_the_depth_verdict_and_its_notice() -> None:
    notice = "This asks for more than one thing, so I am proposing a plan before fetching anything."
    args = {"payload": _plan_payload(depth="standard", depth_notice=notice)}

    (frame,) = _translator()._interrupt_frames(APPROVE_INTERRUPT, args)

    assert frame.event == "plan.proposed"
    assert frame.depth == "standard"
    assert frame.depth_notice == notice


def test_a_plan_proposed_ask_without_a_depth_leaves_it_unset() -> None:
    args = {"payload": _plan_payload()}

    (frame,) = _translator()._interrupt_frames(APPROVE_INTERRUPT, args)

    assert frame.depth is None
    assert frame.depth_notice is None


def test_a_regressed_version_is_logged_and_does_not_move_latest_backwards(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Versions only rise within a turn once the amend/replan guard lands;
    a version-1 approve arriving after the conversation has already seen
    version 3 is the case that guard is meant to prevent, exercised here
    directly against `_interrupt_frames`."""
    conversation = ConversationState(conversation_id="c1")
    translator = _Translator(
        conversation=conversation,
        recorder=_RecordingExecutor(_dummy_executor, store=StudioStore(InMemorySessionService())),
    )
    (first,) = translator._interrupt_frames(
        APPROVE_INTERRUPT, {"payload": _plan_payload(version=1)}
    )
    translator._interrupt_frames(APPROVE_INTERRUPT, {"payload": _plan_payload(version=2)})
    translator._interrupt_frames(APPROVE_INTERRUPT, {"payload": _plan_payload(version=3)})

    with caplog.at_level(logging.ERROR, logger="harness.studio_translate"):
        (regressed,) = translator._interrupt_frames(
            APPROVE_INTERRUPT, {"payload": _plan_payload(version=1)}
        )

    assert any("regress" in record.message.lower() for record in caplog.records)
    assert conversation.latest_plan_version == 3
    assert regressed.event == "plan.proposed"
    assert regressed.plan_version_id != first.plan_version_id


class _UnreachableSurface:
    """A `ToolSurface` `_RecordingSurface` never actually calls here: this
    test appends `_ExecutorRecord`s to it directly, the way a wave's own
    concurrent steps (`executor.run._Run.execute`) would have appended them
    out of the plan's own order."""

    async def list_tools(self) -> list[ToolSchema]:
        raise AssertionError("not called")

    async def call_tool(
        self,
        *,
        tool: ToolId,
        arguments: Mapping[str, Any],
        on_progress: ProgressCallback | None = None,
    ) -> ToolCallResult:
        raise AssertionError("not called")


def test_out_of_order_surface_records_drain_in_the_plan_s_own_step_order() -> None:
    """A wave's independent steps finish, and get recorded, in whatever
    order the event loop schedules them -- here step B is recorded before
    step A even though the plan lists A first. `_drain_recorder` must still
    number them by the plan's own order: A as `s1`, B as `s2`."""
    current_conversation_id.set("c1")
    plan = scripted_plan("kenya life expectancy", spec_id=None, parts=1)
    assert plan.workflow is not None
    step_a, step_b = plan.workflow.steps
    payload = {
        **plan.model_dump(mode="json"),
        "summary": "Kenya life expectancy, 1960-2023.",
        "revision": plan_revision(plan),
    }
    surface_recorder = _RecordingSurface(_UnreachableSurface())
    translator = _Translator(
        conversation=ConversationState(conversation_id="c1"),
        recorder=_RecordingExecutor(_dummy_executor, store=StudioStore(InMemorySessionService())),
        surface_recorder=surface_recorder,
    )

    translator._interrupt_frames(APPROVE_INTERRUPT, {"payload": payload})

    surface_recorder._records.append(
        _ExecutorRecord(
            tool="tool_b",
            args={},
            result={"outcome": "ok"},
            duration_ms=1,
            conversation="c1",
            step_id=step_b.id,
        )
    )
    surface_recorder._records.append(
        _ExecutorRecord(
            tool="tool_a",
            args={},
            result={"outcome": "ok"},
            duration_ms=1,
            conversation="c1",
            step_id=step_a.id,
        )
    )

    frames = translator._drain_recorder()

    activity_steps = [frame for frame in frames if frame.event == "activity.step"]
    assert [(step.step_id, step.tool) for step in activity_steps] == [
        ("s1", "tool_a"),
        ("s2", "tool_b"),
    ]


def _progress_event(step_id: str, message: str) -> Event:
    """What `harness.graph._step_progress_event` yields for one relayed report."""
    return Event(
        custom_metadata={
            STEP_PROGRESS_KEY: {"step_id": step_id, "tool": "get_data", "message": message}
        }
    )


def _surface_record(step_id: str, result: dict[str, Any]) -> _ExecutorRecord:
    return _ExecutorRecord(
        tool="get_data", args={}, result=result, duration_ms=1, conversation="c1", step_id=step_id
    )


def _progress_translator() -> tuple[_Translator, _RecordingSurface]:
    current_conversation_id.set("c1")
    surface_recorder = _RecordingSurface(_UnreachableSurface())
    translator = _Translator(
        conversation=ConversationState(conversation_id="c1"),
        recorder=_RecordingExecutor(_dummy_executor, store=StudioStore(InMemorySessionService())),
        surface_recorder=surface_recorder,
    )
    return translator, surface_recorder


def test_a_step_keeps_one_wire_id_from_its_first_report_to_its_final_frame() -> None:
    """Studio keeps one Activity line per `step_id`, so a step's running
    reports, a failed attempt the executor retried, and its final frame all
    land on the one line its first report opened; a concurrent step gets a
    line of its own."""
    translator, surface_recorder = _progress_translator()

    (first,) = translator.translate(_progress_event("debt", "Preparing the dataset (queued, 3s)"))
    (other,) = translator.translate(_progress_event("pop", "Preparing the dataset (queued, 1s)"))
    (later,) = translator.translate(_progress_event("debt", "Retrying in 7s (attempt 2 of 3)"))
    surface_recorder._records.append(
        _surface_record(
            "debt",
            {"outcome": "refusal", "refusal": {"code": "service_unavailable", "message": "busy"}},
        )
    )
    surface_recorder._records.append(_surface_record("debt", {"outcome": "ok", "summary": "done"}))
    surface_recorder._records.append(_surface_record("pop", {"outcome": "ok", "summary": "done"}))
    final = translator._drain_recorder()

    assert [(f.status, f.detail) for f in (first, other, later)] == [
        (StepStatus.RUNNING, "Preparing the dataset (queued, 3s)"),
        (StepStatus.RUNNING, "Preparing the dataset (queued, 1s)"),
        (StepStatus.RUNNING, "Retrying in 7s (attempt 2 of 3)"),
    ]
    assert first.step_id == later.step_id != other.step_id
    assert first.label == final[0].label == "Fetch the data"
    assert [(f.step_id, f.status) for f in final] == [
        (first.step_id, StepStatus.FAILED),
        (first.step_id, StepStatus.COMPLETE),
        (other.step_id, StepStatus.COMPLETE),
    ]


def test_a_mid_run_report_leaves_a_finished_steps_record_for_the_run_to_amend() -> None:
    """A step's record is written when its call returns, before the
    executor's post-call checks; `execute` amends it once the run is over
    (`_RecordingSurface.mark_step_failed`). A report from a sibling still
    running must not drain that record early, or the amendment finds
    nothing and the step reads complete forever."""
    translator, surface_recorder = _progress_translator()
    surface_recorder._records.append(_surface_record("debt", {"outcome": "ok", "summary": "done"}))

    (running,) = translator.translate(_progress_event("pop", "Preparing the dataset (queued, 1s)"))
    surface_recorder.mark_step_failed(
        step_id="debt", tool="get_data", code="source_drift", detail="drifted"
    )
    (final,) = translator._drain_recorder()

    assert running.status is StepStatus.RUNNING
    assert final.step_id != running.step_id
    assert (final.status, final.detail) == (StepStatus.FAILED, "source_drift: drifted")


class _HangingSurface(_UnreachableSurface):
    """A `ToolSurface` whose call never returns, until cancelled."""

    async def call_tool(
        self,
        *,
        tool: ToolId,
        arguments: Mapping[str, Any],
        on_progress: ProgressCallback | None = None,
    ) -> ToolCallResult:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


async def _cancel_mid_call(surface_recorder: _RecordingSurface, *, step_id: str) -> None:
    """Start `step_id`'s call through `surface_recorder`, then cancel it
    before it returns, as a step budget, a Stop or a disconnect does."""

    async def call() -> None:
        current_step_id.set(step_id)
        await surface_recorder.call_tool(
            tool=ToolId(server="data-mcp", name="get_data"), arguments={}
        )

    task = asyncio.create_task(call())
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_a_step_cut_off_mid_call_closes_its_line_as_failed() -> None:
    """A step's budget cancels its call before it returns, so the call
    writes no record. `mark_step_failed` records the failure itself, and the
    step's final frame closes the line its running report opened."""
    translator, surface_recorder = _progress_translator()
    surface_recorder._surface = _HangingSurface()

    (running,) = translator.translate(_progress_event("debt", "Preparing the dataset (queued, 3s)"))
    await _cancel_mid_call(surface_recorder, step_id="debt")
    surface_recorder.mark_step_failed(
        step_id="debt", tool="get_data", code="transport_failed", detail="ran past its budget"
    )
    (final,) = translator._drain_recorder()

    assert (final.step_id, final.status, final.tool) == (
        running.step_id,
        StepStatus.FAILED,
        "get_data",
    )
    assert final.detail == "transport_failed: ran past its budget"
    assert final.duration_ms is not None


async def test_a_call_abandoned_mid_flight_leaves_nothing_once_its_segment_drains() -> None:
    """A Stop or a disconnect cancels the call and ends the segment, whose
    `finally` drains the recorder. Nothing of the abandoned call is left for
    a later segment to report."""
    surface_recorder = _RecordingSurface(_HangingSurface())
    current_conversation_id.set("c1")

    await _cancel_mid_call(surface_recorder, step_id="debt")

    assert surface_recorder.drain("c1") == []
    assert surface_recorder._in_flight == {}


def test_a_step_that_never_called_gets_a_failed_record_and_a_refused_one_keeps_its_own() -> None:
    """A step stopped before its first call still gets a line that closes
    as failed. A step whose last call already refused, with none in flight,
    is left alone: its own refusal is its final frame."""
    translator, surface_recorder = _progress_translator()
    refused = {"outcome": "refusal", "refusal": {"code": "service_unavailable", "message": "busy"}}
    surface_recorder._records.append(_surface_record("pop", refused))

    for step_id in ("debt", "pop"):
        surface_recorder.mark_step_failed(
            step_id=step_id, tool="get_data", code="transport_failed", detail="out of time"
        )
    frames = translator._drain_recorder()

    assert [(f.status, f.detail, f.duration_ms) for f in frames] == [
        (StepStatus.FAILED, "service_unavailable: busy", 1),
        (StepStatus.FAILED, "transport_failed: out of time", 0),
    ]


# --- the recorder's `revises`: swapping a card's block in place --------------


def _written_component(
    component_id: str = "cmp_test0001",
    *,
    revision: int = 1,
    title: str = "Life expectancy in Nigeria",
    labels: tuple[str, ...] = ("Nigeria",),
) -> dict[str, Any]:
    """A platform-shaped `write_component` result: the wire fields of
    `component.created` minus event/seq, one `FactSeries` per label."""
    series = [{"id": label.lower(), "label": label, "point_count": 64} for label in labels]
    return {
        "component_id": component_id,
        "revision": revision,
        "type": "line_chart",
        "catalog_version": "2026.8.0",
        "min_catalog_version": "2026.8.0",
        "payload": resolved_revision(
            "line_chart", component_id=component_id, revision=revision, title=title
        ),
        "fact_summary": {
            "component_id": component_id,
            "revision": revision,
            "series": series,
            "indicators": [
                {"id": "dcid:LifeExpectancy_Person", "name": "Life Expectancy", "description": ""}
            ],
            "attribution": {
                "label": "WDI · 1 indicator(s)",
                "source_count": 1,
                "indicator_count": 1,
            },
            "window": {"start": "1960", "end": "2023"},
        },
        "provenance": {
            "inputs": [{"spec_id": "spec_" + "a" * 32, "content_version": "cv_" + "1" * 32}],
            "sources": [{"id": "src_test", "name": "Test source", "licence": "CC BY 4.0"}],
            "freshness": {
                "data_through": "2023",
                "last_refreshed": "2026-06-30",
                "partial_period": False,
            },
            "warnings": [],
            "generated_at": "2026-01-01T00:00:00Z",
            "citation": "Test source (2026).",
        },
    }


# --- `_component_frames`: the one place validation, landing and framing meet --


def test_component_frames_lands_and_replaces_the_revise_target() -> None:
    """`land=True` plus a `revises` id that is really on the canvas lands
    the component in place of it -- validates, then calls
    `harness.studio_state.land_component` itself, then streams
    `component.created` plus `composition.updated` reading the manifest
    that call just produced."""
    current_conversation_id.set("c1")
    translator = _translator()
    translator._conversation.composition = _next_manifest(
        None,
        [
            CompositionBlock(
                component_id="cmp_two", revision=1, type="line_chart", title="x", pinned=False
            )
        ],
        actor=CompositionActor.MODEL,
    )

    frames = translator._component_frames(
        _written_component("cmp_two"), step_id="s1", land=True, revises="cmp_two"
    )

    events = [getattr(f, "event", None) for f in frames]
    assert events == ["component.created", "composition.updated"]
    assert translator._conversation.turn_components[-1]["action"] == "replaced"


def test_component_frames_lands_and_appends_with_no_revise_target() -> None:
    """`land=True`, `revises=None`: an append, which also streams both
    frames."""
    current_conversation_id.set("c1")
    translator = _translator()
    translator._conversation.composition = _next_manifest(None, [], actor=CompositionActor.MODEL)

    frames = translator._component_frames(
        _written_component("cmp_two"), step_id="s1", land=True, revises=None
    )

    events = [getattr(f, "event", None) for f in frames]
    assert events == ["component.created", "composition.updated"]
    assert translator._conversation.turn_components[-1]["action"] == "created"


def test_component_frames_land_true_holds_when_auto_add_is_off() -> None:
    """`land=True` against a conversation with `auto_add` off resolves to
    `harness.studio_state.land_component`'s `"held"`: no block for
    `composition.updated` to report."""
    current_conversation_id.set("c1")
    translator = _translator()
    translator._conversation.auto_add = False

    frames = translator._component_frames(
        _written_component("cmp_two"), step_id="s1", land=True, revises=None
    )

    events = [getattr(f, "event", None) for f in frames]
    assert events == ["component.created"]


def test_component_frames_land_false_emits_only_component_created() -> None:
    """`land=False` -- a write `harness.graph`'s `compose` judged unusable,
    or a bare executor with no `land_component` capability at all --
    skips `harness.studio_state.land_component` entirely: the component
    streamed, but nothing landed a block for it, and the canvas is
    untouched."""
    current_conversation_id.set("c1")
    translator = _translator()
    before = translator._conversation.composition

    frames = translator._component_frames(
        _written_component("cmp_two"), step_id="s1", land=False, revises="cmp_two"
    )

    events = [getattr(f, "event", None) for f in frames]
    assert events == ["component.created"]
    assert translator._conversation.composition is before
    assert translator._conversation.turn_components == []


def test_component_frames_logs_error_for_a_stamped_component_that_fails_validation(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """`land=True` means `harness.graph`'s `compose` already ran
    `harness.wire.usable_component` on this exact component and it passed,
    so a validation failure here means the wire check and this method's
    own contract have diverged -- a real bug, worth an ERROR record.
    Nothing streams and the canvas stays untouched, rather than raising
    out of `translate` and failing the whole segment."""
    current_conversation_id.set("c1")
    translator = _translator()
    before = translator._conversation.composition
    invalid = {**_written_component("cmp_two")}
    del invalid["fact_summary"]

    with caplog.at_level(logging.ERROR, logger="harness.studio_translate"):
        frames = translator._component_frames(invalid, step_id="s1", land=True, revises=None)

    assert frames == []
    assert translator._conversation.composition is before
    assert translator._conversation.turn_components == []
    assert any("validation" in record.message.lower() for record in caplog.records)


def test_component_frames_logs_nothing_for_an_unstamped_component_that_fails_validation(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """`land=False` means this write was never stamped usable in the first
    place -- `harness.graph`'s `compose` already refused it
    (`component_invalid`, in `refused_writes`) because this same component
    failed `harness.wire.usable_component`. Failing validation here again
    is the expected outcome of that refusal, not a divergence, so no
    frames stream and nothing is logged at ERROR."""
    current_conversation_id.set("c1")
    translator = _translator()
    before = translator._conversation.composition
    invalid = {**_written_component("cmp_two")}
    del invalid["fact_summary"]

    with caplog.at_level(logging.DEBUG, logger="harness.studio_translate"):
        frames = translator._component_frames(invalid, step_id="s1", land=False, revises=None)

    assert frames == []
    assert translator._conversation.composition is before
    assert translator._conversation.turn_components == []
    assert not any(record.levelno >= logging.ERROR for record in caplog.records)


# --- `_RecordingExecutor.land_component`: stamps the record, decides nothing -


def test_land_component_stamps_land_and_revises_on_the_matching_record() -> None:
    """`land_component` only stamps the record whose result carries this
    exact component object (found by identity) -- it never resolves a
    conversation and never mutates a canvas; `_component_frames`, draining
    later, is what does the real work from these two fields."""
    recorder = _RecordingExecutor(_dummy_executor, store=StudioStore(InMemorySessionService()))
    component = _written_component("cmp_two", labels=("Nigeria", "Guatemala"))
    recorder._records.append(
        _ExecutorRecord(
            tool="write_component",
            args={"intent": {}},
            result={"outcome": "ok", "component": component},
            duration_ms=1,
            conversation="c1",
        )
    )

    recorder.land_component(component, land=True, revises="cmp_nigeria")

    assert recorder._records[0].land is True
    assert recorder._records[0].revises == "cmp_nigeria"


def test_land_component_is_a_silent_no_op_with_no_matching_record() -> None:
    """A component with no matching record (this capability called
    directly, not through `__call__` first) leaves nothing to stamp and
    raises nothing."""
    recorder = _RecordingExecutor(_dummy_executor, store=StudioStore(InMemorySessionService()))

    recorder.land_component(_written_component("cmp_two"), land=True, revises=None)


# --- `_RecordingExecutor.__call__`: write tokens --------------------------


def _returning(result: dict[str, object]) -> Any:
    async def _executor(
        *, tool: str, args: dict[str, object], on_progress: ProgressCallback | None = None
    ) -> dict[str, object]:
        del tool, args, on_progress
        return dict(result)

    return _executor


async def test_a_create_that_returns_no_write_token_is_logged_at_error_level(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """That card is rebuilt as a new card on its next edit, so the missing
    token is a defect an operator must see, once per create."""
    store = StudioStore(InMemorySessionService())
    current_conversation_key.set(("alice", "c1"))
    current_conversation_id.set("alice/c1")
    await store.create("alice", "c1")
    component = _written_component("cmp_tokenless")
    recorder = _RecordingExecutor(
        _returning({"outcome": "ok", "component": component}), store=store
    )

    with caplog.at_level(logging.ERROR, logger="harness.studio_translate"):
        await recorder(tool="write_component", args={"intent": {"title": "x"}})

    errors = [record for record in caplog.records if record.levelno == logging.ERROR]
    assert len(errors) == 1
    assert "cmp_tokenless" in errors[0].getMessage()
    assert "alice/c1" in errors[0].getMessage()


async def test_a_revise_returning_no_write_token_logs_nothing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Only a create is issued a token; a revise's reply never carries one."""
    store = StudioStore(InMemorySessionService())
    current_conversation_key.set(("alice", "c1"))
    await store.create("alice", "c1")
    recorder = _RecordingExecutor(
        _returning({"outcome": "ok", "component": _written_component("cmp_one", revision=2)}),
        store=store,
    )

    with caplog.at_level(logging.ERROR, logger="harness.studio_translate"):
        await recorder(
            tool="write_component",
            args={"intent": {"title": "x"}, "component_id": "cmp_one", "write_token": "cmw_x"},
        )

    assert not [record for record in caplog.records if record.levelno == logging.ERROR]


async def test_the_record_of_a_revise_leaves_the_write_token_out() -> None:
    store = StudioStore(InMemorySessionService())
    current_conversation_key.set(("alice", "c1"))
    current_conversation_id.set("alice/c1")
    await store.create("alice", "c1")
    recorder = _RecordingExecutor(
        _returning({"outcome": "ok", "component": _written_component("cmp_one", revision=2)}),
        store=store,
    )
    args = {"intent": {"title": "x"}, "component_id": "cmp_one", "write_token": "cmw_secret"}

    await recorder(tool="write_component", args=args)

    [record] = recorder.drain("alice/c1")
    assert record.args == {"intent": {"title": "x"}, "component_id": "cmp_one"}
    assert args["write_token"] == "cmw_secret"


# --- `_RecordingExecutor.components_replaced`: reads the turn's own outcome -


async def test_components_replaced_is_true_after_a_real_replace() -> None:
    """Resolves the conversation through the (store, contextvar) pair
    `harness.studio.StudioRunner` binds for the whole turn, and reads
    `turn_components` for an actual `"replaced"` action -- the outcome
    `harness.studio_state.land_component` recorded, at drain time, for
    this turn."""
    store = StudioStore(InMemorySessionService())
    current_conversation_key.set(("alice", "c1"))
    conversation = await store.create("alice", "c1")
    conversation.turn_components.append(
        {"component_id": "cmp_two", "type": "line_chart", "title": "x", "action": "replaced"}
    )
    recorder = _RecordingExecutor(_dummy_executor, store=store)

    assert recorder.components_replaced() is True


async def test_components_replaced_is_false_with_no_replaced_action() -> None:
    """A turn whose only landed actions are `"created"` (or nothing at
    all) reads as `False` -- narrate's fallback, and the same value a bare
    executor with no capability implies."""
    store = StudioStore(InMemorySessionService())
    current_conversation_key.set(("alice", "c1"))
    conversation = await store.create("alice", "c1")
    conversation.turn_components.append(
        {"component_id": "cmp_two", "type": "line_chart", "title": "x", "action": "created"}
    )
    recorder = _RecordingExecutor(_dummy_executor, store=store)

    assert recorder.components_replaced() is False


def test_components_replaced_raises_with_no_conversation_key_bound() -> None:
    """A caller outside a studio turn -- `current_conversation_key` never
    set -- gets a loud failure: `components_replaced` has no
    store-independent fallback."""
    current_conversation_key.set(None)
    recorder = _RecordingExecutor(_dummy_executor, store=StudioStore(InMemorySessionService()))

    with pytest.raises(RuntimeError):
        recorder.components_replaced()


# --- `_drain_recorder`: performs the landing `land_component` only stamped -


async def test_drain_recorder_lands_a_component_stamped_for_replace() -> None:
    """The real call sequence `harness.graph`'s `compose` node makes: write
    through the recorder, then stamp synchronously (no `await` between,
    deciding and mutating nothing yet), then drain, later, on the next ADK
    event -- draining is what actually calls `harness.studio_state.
    land_component` and produces both the frames and the turn record."""
    component = _written_component("cmp_two", labels=("Nigeria", "Guatemala"))

    async def _write(
        *, tool: str, args: dict[str, object], on_progress: ProgressCallback | None = None
    ) -> dict[str, object]:
        del tool, args
        return {"outcome": "ok", "component": component, "summary": "created"}

    store = StudioStore(InMemorySessionService())
    current_conversation_key.set(("alice", "c1"))
    conversation = await store.create("alice", "c1")
    conversation.composition = _next_manifest(
        None,
        [
            CompositionBlock(
                component_id="cmp_nigeria", revision=1, type="line_chart", title="x", pinned=False
            )
        ],
        actor=CompositionActor.MODEL,
    )
    recorder = _RecordingExecutor(_write, store=store)
    translator = _Translator(conversation=conversation, recorder=recorder)
    current_conversation_id.set("c1")

    written = await recorder(tool="write_component", args={"intent": {}})
    recorder.land_component(written["component"], land=True, revises="cmp_nigeria")
    # Stamping alone decides and mutates nothing: the canvas is untouched
    # until the drain below actually lands it.
    assert [b.component_id for b in conversation.composition.blocks] == ["cmp_nigeria"]

    frames = translator._drain_recorder()

    activity = [f for f in frames if f.event == "activity.step"]
    assert activity[0].produced == StepProduced(component_id="cmp_two", revision=1)
    assert any(getattr(f, "event", None) == "composition.updated" for f in frames)
    assert [b.component_id for b in conversation.composition.blocks] == ["cmp_two"]
    assert conversation.turn_components[-1]["action"] == "replaced"


# --- a refused write never reaches `_component_frames` -----------------------


def test_a_refused_write_never_reaches_component_frames() -> None:
    """A refusal carries no `component`, so `_drain_recorder` never calls
    `_component_frames` for it -- no `component.created` goes out, and the
    canvas is untouched."""
    current_conversation_id.set("c1")
    translator = _translator()
    before = translator._conversation.composition

    translator._recorder._records.append(
        _ExecutorRecord(
            tool="write_component",
            args={"intent": {}},
            result={
                "outcome": "refusal",
                "refusal": {"code": "constraint_violation", "message": "no commensurable unit"},
            },
            duration_ms=1,
            conversation="c1",
        )
    )
    frames = translator._drain_recorder()

    assert translator._conversation.composition is before
    assert not any(getattr(f, "event", None) == "component.created" for f in frames)
    activity = [f for f in frames if f.event == "activity.step"]
    assert activity and activity[0].status == StepStatus.FAILED


# --- the recorded action survives into the conversation digest ---------------


async def test_the_replaced_action_reaches_the_conversation_digest() -> None:
    """`close_turn` folds `turn_components` into the turn record
    `conversation_digest_state` later serves converse. A replace's
    `action`, written by `harness.studio_state.land_component` when
    `_drain_recorder` lands it, has to survive that trip, so a later
    turn's digest can name the card as changed."""
    translator, recorder, store = await _translator_with_store()
    conversation = store.current("alice", "c1")
    current_conversation_id.set("c1")

    def _land(component: dict[str, Any], *, revises: str | None) -> None:
        """The write, stamp, drain sequence `harness.graph`'s `compose`
        node and the translator's own drain perform together for one
        component."""
        recorder._records.append(
            _ExecutorRecord(
                tool="write_component",
                args={"intent": {}},
                result={"outcome": "ok", "component": component},
                duration_ms=1,
                conversation="c1",
            )
        )
        recorder.land_component(component, land=True, revises=revises)
        translator._drain_recorder()

    nigeria = _written_component("cmp_nigeria")
    _land(nigeria, revises=None)
    translator._conversation.close_turn(final_text="Here is Nigeria.")

    translator._conversation.begin_turn()
    two_series = _written_component("cmp_two", labels=("Nigeria", "Guatemala"))
    _land(two_series, revises="cmp_nigeria")
    translator._conversation.close_turn(final_text="Updated the chart.")

    assert conversation.turns[-1]["cards"] == [
        {
            "component_id": "cmp_two",
            "type": "line_chart",
            "title": "Life expectancy in Nigeria",
            "action": "replaced",
            "window": "1960-2023",
        }
    ]


def _node_event(
    node: str, *, route: str | None = None, output: Any = None, **metadata: Any
) -> Event:
    return Event(
        author="dfo_analyst_graph",
        node_info=NodeInfo(path=f"dfo_analyst_graph/{node}@1"),
        actions=EventActions(route=route),
        output=output,
        custom_metadata=metadata or None,
    )


def _planner_call(name: str, **args: Any) -> Event:
    return Event(
        author="dfo_planner",
        content=types.Content(
            role="model", parts=[types.Part(function_call=types.FunctionCall(name=name, args=args))]
        ),
    )


def _status_lines(frames: list[Any]) -> list[tuple[str, str, StepStatus]]:
    return [(f.step_id, f.label, f.status) for f in frames]


def test_each_planner_call_becomes_a_status_line_as_it_happens() -> None:
    """The planner's function-call events arrive while it drafts, so each
    one opens its line then, and completes the line before it."""
    translator = _translator()

    working = translator.translate(_node_event("planner", planner_started={"revising": False}))
    drafting = translator.translate(_planner_call("propose_plan", question="q"))
    checking = translator.translate(_planner_call("dfo_translator", request="translate"))
    ready = translator.translate(_planner_call("submit_plan", version=1))
    done = translator.translate(_node_event("planner", route="ok", output={"version": 1}))

    assert _status_lines(working) == [("s0", "Working on a plan", StepStatus.RUNNING)]
    assert _status_lines(drafting) == [
        ("s0", "Working on a plan", StepStatus.COMPLETE),
        ("s2", "Drafting a plan", StepStatus.RUNNING),
    ]
    assert _status_lines(checking) == [
        ("s2", "Drafting a plan", StepStatus.COMPLETE),
        ("s4", "Checking the plan against the data", StepStatus.RUNNING),
    ]
    assert _status_lines(ready) == [
        ("s4", "Checking the plan against the data", StepStatus.COMPLETE),
        ("s6", "Getting the plan ready for you", StepStatus.RUNNING),
    ]
    assert _status_lines(done) == [("s6", "Getting the plan ready for you", StepStatus.COMPLETE)]
    # Studio shows a line as done only once a frame carries `duration_ms`.
    completed = [f for f in drafting + checking + ready + done if f.status is StepStatus.COMPLETE]
    assert all(f.duration_ms is not None for f in completed)
    assert all(f.duration_ms is None for f in working + drafting if f.status is StepStatus.RUNNING)


def test_a_repeated_planner_call_keeps_its_line_and_other_calls_add_none() -> None:
    translator = _translator()
    opened = translator.translate(_planner_call("propose_plan", question="q"))

    assert _status_lines(opened) == [("s0", "Drafting a plan", StepStatus.RUNNING)]
    assert translator.translate(_planner_call("propose_plan", question="q")) == []
    assert translator.translate(_planner_call("get_skill", skill_id="oda-measures")) == []


@pytest.mark.parametrize(
    "event",
    [
        _planner_call("propose_plan", amends_version=1),
        _node_event("planner", planner_started={"revising": True}),
    ],
)
def test_an_amendment_or_a_replan_shows_revising_the_plan(event: Event) -> None:
    """The planner node starts a revising draft on an amendment or a
    replan, and the planner's own call names the version it amends."""
    frames = _translator().translate(event)

    assert _status_lines(frames) == [("s0", "Revising the plan", StepStatus.RUNNING)]


def test_the_planner_starting_a_first_draft_opens_working_on_a_plan() -> None:
    """Whatever routes the turn to the planner, the line opens as the
    planner node starts its draft."""
    frames = _translator().translate(_node_event("planner", planner_started={"revising": False}))

    assert _status_lines(frames) == [("s0", "Working on a plan", StepStatus.RUNNING)]


def test_the_seams_late_progress_reports_are_not_shown() -> None:
    """The node relays the seam's reports only after the draft is done, so
    they would repeat the live lines out of order."""
    translator = _translator()
    report = {"agent": "dfo_planner", "tools": ["propose_plan"], "duration_ms": 5}

    assert translator.translate(_node_event("planner", progress=report)) == []


def test_a_pause_completes_the_running_line_before_its_card() -> None:
    translator = _translator()
    translator.translate(_planner_call("propose_plan", question="q"))
    ask = Event(
        author="dfo_planner",
        content=types.Content(
            role="model",
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        id="ask-1",
                        name=REQUEST_INPUT_FUNCTION_CALL_NAME,
                        args={"message": "Which source?", "candidates": []},
                    )
                )
            ],
        ),
    )

    frames = translator.translate(ask)

    assert [f.event for f in frames] == ["activity.step", "elicitation.request"]
    assert frames[0].status is StepStatus.COMPLETE
    assert translator.finish() == []


def _planner_ask(message: str) -> Event:
    """The planner's `adk_request_input` call asking `message`."""
    call = types.FunctionCall(
        id="ask-1", name=REQUEST_INPUT_FUNCTION_CALL_NAME, args={"message": message}
    )
    return Event(
        author="dfo_planner",
        content=types.Content(role="model", parts=[types.Part(function_call=call)]),
    )


def test_a_question_the_planner_asks_opens_with_the_drops_its_draft_started_with() -> None:
    """The `planner` node starts a draft naming seeds the turn dropped and the user has
    not been told of. The planner's question opens with them, once."""
    translator = _translator()
    drops = '"dataset 1 from Search" could not be retrieved: Search finds no data for it.'
    translator.translate(
        _node_event("planner", planner_started={"revising": False, "untold_drops": drops})
    )

    (*_, asked) = translator.translate(_planner_ask("Which donors?"))
    (again,) = translator.translate(_planner_ask("Which years?"))

    assert asked.question == f"{drops} Which donors?"
    assert again.question == "Which years?"


def test_a_question_after_a_draft_that_named_no_drop_is_asked_as_is() -> None:
    translator = _translator()
    translator.translate(_node_event("planner", planner_started={"revising": False}))

    (*_, asked) = translator.translate(_planner_ask("Which donors?"))

    assert asked.question == "Which donors?"


def test_the_segment_end_completes_a_line_still_running() -> None:
    translator = _translator()
    translator.translate(_planner_call("submit_plan", version=1))

    frames = translator.finish()

    assert (frames[0].label, frames[0].status) == (
        "Getting the plan ready for you",
        StepStatus.COMPLETE,
    )


async def test_a_planner_call_becomes_its_line_while_the_draft_is_still_running() -> None:
    """A real `PlannerSeam`, driven through `ctx.run_node` the way the
    graph's planner node drives it, with its model held after its first
    call. The runner hands the translator that call while the draft is
    still waiting, so its line reaches the wire before the draft ends."""
    seam = _seam()
    paused = asyncio.Event()
    release = asyncio.Event()
    seam._agent.model = ScriptedPlannerLlm(
        script=[[("propose_plan", {"question": "kenya debt"})]],
        pause_at=1,
        paused=paused,
        release=release,
    )

    async def planner(ctx: Context, node_input: Any) -> Any:
        del node_input
        return await seam(ctx=ctx, question="kenya debt", envelope={})

    workflow = Workflow(
        name="status_test_wf", edges=[(START, node(planner, name="planner", rerun_on_resume=True))]
    )
    sessions = InMemorySessionService()
    session = await sessions.create_session(app_name="status_test", user_id="u")
    runner = Runner(node=workflow, app_name="status_test", session_service=sessions)
    translator = _translator()
    frames: list[Any] = []

    async def run() -> None:
        with pytest.raises(PlannerSeamError):
            async for event in runner.run_async(
                user_id="u",
                session_id=session.id,
                new_message=types.Content(role="user", parts=[types.Part(text="probe")]),
            ):
                frames.extend(translator.translate(event))

    task = asyncio.create_task(run())
    await asyncio.wait_for(paused.wait(), timeout=10)
    mid_draft = _status_lines(frames)
    release.set()
    await task

    assert mid_draft == [("s0", "Drafting a plan", StepStatus.RUNNING)]


def test_a_planner_seam_error_reaches_the_user_without_its_model_facing_text() -> None:
    conversation = ConversationState(conversation_id="c1")
    conversation.begin_turn()
    exc = PlannerSeamError(
        "This plan still needs translating and compiling into a workflow before it can "
        "execute. Call `dfo_translator` first."
    )

    frame = _stream_fatal(conversation, exc)

    assert frame.error.error.value == "capability_unavailable"
    assert frame.error.retry.value == "do_not_retry"
    assert "dfo_translator" not in frame.error.detail
    assert frame.error.detail == _PLANNER_FAILED_DETAIL.format(trace_id=conversation.trace_id)

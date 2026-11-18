"""Studio's recorded turns, replayed through the translator as turn events.

Each file under `fixtures/translator_baselines/` holds one Studio scenario
(the parity baselines: a plain answer, an ask, a planned turn, a switch, a
Stop, a card edit, ...) as Studio's translator saw it, one entry per stream
segment: the conversation as the segment opened on it, the plan card a first
answer approves, the plan's step labels, and each event of the segment with
the calls recorded before it, the canvas and removal flag the event found,
and the frames the translator sent for it. The events are the ones the
graph's nodes and the turn loop emit (`harness.graph.events`,
`harness.turn.events`); the frames are those the recorded turn streamed.

Each segment is replayed through a fresh `_Translator`: the recorded calls
reach it through stand-in recorders, and each event through `emit`. Before
the segment's end it takes the turn's records as the turn's state holds
them, as the stream does once the segment is done. The frames it sends must
be the recorded ones, with each plan version id and duration read by order.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
from google.genai import types
from harness.conversation import current_conversation_id, current_conversation_key
from harness.graph.events import AgentStatus, PlannerStarted, StepProgress
from harness.studio_state import ConversationState
from harness.studio_translate import _ExecutorRecord, _Translator, keep_turn_records
from harness.turn.events import NodeDone, Paused
from harness.wire import CompositionManifest, PlanProposed
from orchestrator.run import ModelResponded, TextDelta, ToolsAnswered
from planner.pydantic_agent import PlannerAnswers, PlannerResponse

FIXTURES = Path(__file__).parent / "fixtures" / "translator_baselines"
SCENARIOS = sorted(path.stem for path in FIXTURES.glob("*.json"))

# The prefixes of the ids a stream mints as it runs: plan versions and canvases.
_MINTED = ("plan_", "cps_")


def _parts(encoded: Sequence[Mapping[str, Any]]) -> tuple[types.Part, ...]:
    return tuple(types.Part.model_validate(part) for part in encoded)


def _answers(encoded: Sequence[Mapping[str, Any]]) -> tuple[types.FunctionResponse, ...]:
    return tuple(types.FunctionResponse.model_validate(answer) for answer in encoded)


# Each turn event's type, and how to build it from its recorded fields.
_EVENTS: dict[str, Callable[..., Any]] = {
    "AgentStatus": AgentStatus,
    "TextDelta": TextDelta,
    "ModelResponded": lambda parts: ModelResponded(parts=_parts(parts)),
    "ToolsAnswered": lambda answers, state: ToolsAnswered(answers=_answers(answers), state=state),
    "PlannerStarted": PlannerStarted,
    "PlannerResponse": lambda parts: PlannerResponse(parts=_parts(parts)),
    "PlannerAnswers": lambda answers, state: PlannerAnswers(answers=_answers(answers), state=state),
    "StepProgress": StepProgress,
    "NodeDone": NodeDone,
    "Paused": Paused,
}


def _event(encoded: Mapping[str, Any]) -> Any:
    """The turn event `encoded` names by its `type`."""
    fields = {key: value for key, value in encoded.items() if key != "type"}
    return _EVENTS[encoded["type"]](**fields)


class _Store:
    """The canvas writes a landing card makes, applied to the segment's
    conversation as the store applies them to the one it holds."""

    def __init__(self, conversation: ConversationState) -> None:
        self._conversation = conversation

    async def change_canvas(
        self, principal: str, conversation_id: str, change: Callable[[ConversationState], Any]
    ) -> Any:
        del principal, conversation_id
        return change(self._conversation)


class _Recorder:
    """The executor recorder: hands each drain the calls recorded since the last."""

    def __init__(self, store: _Store) -> None:
        self.store = store
        self.recorded: list[_ExecutorRecord] = []

    def drain(self, conversation: str) -> list[_ExecutorRecord]:
        del conversation
        drained, self.recorded = self.recorded, []
        return drained


class _Surface:
    """The surface recorder: the plan's step labels, and each retrieval's
    manifest as the recorded turn read it. Its calls arrive through
    `_Recorder`, already in the plan's step order."""

    def __init__(self, labels: Mapping[str, str], manifests: Sequence[Mapping[str, Any]]) -> None:
        self._labels = dict(labels)
        self._manifests = [(entry["frame_ref"], entry["manifest"]) for entry in manifests]

    def drain(self, conversation: str) -> list[_ExecutorRecord]:
        del conversation
        return []

    def step_order(self, conversation: str) -> list[str]:
        del conversation
        return []

    def step_labels(self, conversation: str) -> dict[str, str]:
        del conversation
        return self._labels

    def set_step_order(
        self, conversation: str, order: Sequence[str], labels: Mapping[str, str] | None = None
    ) -> None:
        del conversation, order
        self._labels = dict(labels or {})

    def retrieval_manifest(self, *, frame_ref: Mapping[str, Any]) -> dict[str, Any] | None:
        return next((manifest for ref, manifest in self._manifests if ref == frame_ref), None)


# The conversation fields whose JSON form is not the field's own type.
_FIELDS: dict[str, Callable[[Any], Any]] = {
    "composition": lambda kept: CompositionManifest.model_validate(kept) if kept else None,
    "plan_version_ids": lambda kept: {int(version): wire_id for version, wire_id in kept.items()},
    "timed_out_calls": set,
}


def _conversation(conversation_id: str, recorded: Mapping[str, Any]) -> ConversationState:
    """The conversation as the segment opened on it."""
    conversation = ConversationState(conversation_id=conversation_id)
    for name, value in recorded.items():
        setattr(conversation, name, _FIELDS.get(name, lambda kept: kept)(value))
    return conversation


def _world(conversation: ConversationState, world: Mapping[str, Any]) -> None:
    """The canvas and removal flag the tools left before an event. The
    canvas keeps the id the replay minted for it."""
    recorded = world["composition"]
    composition = CompositionManifest.model_validate(recorded) if recorded else None
    if composition is not None and conversation.composition is not None:
        composition = composition.model_copy(
            update={"composition_id": conversation.composition.composition_id}
        )
    conversation.composition = composition
    conversation._removal_unreported = world["removal"]


def _records(encoded: Sequence[Mapping[str, Any]]) -> list[_ExecutorRecord]:
    return [_ExecutorRecord(**record) for record in encoded]


async def _replayed(segment: Mapping[str, Any]) -> list[Any]:
    """Every frame a fresh translator sends for `segment`'s events and end."""
    current_conversation_id.set(segment["conversation_id"])
    current_conversation_key.set(tuple(segment["conversation_key"]))
    conversation = _conversation(segment["conversation_id"], segment["conversation"])
    recorder = _Recorder(_Store(conversation))
    surface = (
        _Surface(segment["step_labels"], segment["manifests"])
        if segment["step_labels"] is not None
        else None
    )
    approving = segment["approving"]
    translator = _Translator(
        conversation=conversation,
        recorder=recorder,  # type: ignore[arg-type]
        surface_recorder=surface,  # type: ignore[arg-type]
        approving=PlanProposed.model_validate(approving) if approving else None,
    )
    frames: list[Any] = []
    for item in segment["items"]:
        _world(conversation, item["world"])
        recorder.recorded.extend(_records(item["records"]))
        for event in item["events"]:
            frames.extend(await translator.translate(_event(event)))
    finish = segment["finish"]
    if finish is None:
        return frames
    world = finish["world"]
    _world(conversation, world)
    keep_turn_records(
        conversation,
        {
            "orchestrator:plan_records": world["turn_plans"],
            "orchestrator:searches": world["turn_readings"],
            "orchestrator:specs": world["turn_specs"],
            **(
                {"orchestrator:applied_choice": world["turn_applied_choice"]}
                if world["turn_applied_choice"] is not None
                else {}
            ),
        },
    )
    recorder.recorded.extend(_records(finish["records"]))
    frames.extend(await translator.finish())
    return frames


def _recorded(segment: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
    for item in segment["items"]:
        yield from item["frames"]
    if segment["finish"] is not None:
        yield from segment["finish"]["frames"]


def _by_order(frames: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """`frames` with each plan version and canvas id numbered by first
    appearance and every duration zero, since all three are minted as the
    stream runs."""
    names: dict[str, str] = {}

    def read(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: 0 if key == "duration_ms" else read(item) for key, item in value.items()}
        if isinstance(value, list):
            return [read(item) for item in value]
        if isinstance(value, str) and value.startswith(_MINTED):
            return names.setdefault(value, f"{value[:4]}<{len(names) + 1}>")
        return value

    return [read(frame) for frame in frames]


@pytest.mark.parametrize("scenario", SCENARIOS)
async def test_each_recorded_segment_streams_its_recorded_frames(scenario: str) -> None:
    segments = json.loads((FIXTURES / f"{scenario}.json").read_text())

    for index, segment in enumerate(segments):
        replayed = [
            frame.model_dump(mode="json", exclude_none=True) for frame in await _replayed(segment)
        ]

        assert _by_order(replayed) == _by_order(list(_recorded(segment))), f"segment {index}"

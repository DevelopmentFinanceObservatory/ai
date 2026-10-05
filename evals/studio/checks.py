"""Wire-contract conformance checks over studio frame streams.

The contract layer of harness-studio-v1: mechanical, model-free assertions
that any recorded stream — a scripted run, a live run, or a transcript in
`docs/transcripts/` — obeys the invariants `docs/frontend-interface.md`
promises the frontend:

- every frame validates against `harness.wire.STUDIO_FRAME_ADAPTER`
- `seq` starts at 0 per turn and strictly increases across resumed segments
- a turn ends on `done` or `error`, exactly once, with nothing after it
  (which also rules out `done` following a stream-fatal `error`)
- composition revisions strictly increase across `composition.updated`
  frames, and the `composition_id` never changes within a conversation
- `component.revised` only ever names a component the conversation has seen
- every number in a `message.complete` is grounded in the turn's fact
  frames — fact summaries, step results, plan text — or a prior turn's
  fact summaries (the canvas digest narration may cite); ids, revisions,
  and timestamps are structure and never ground anything
- a live runner additionally holds each SSE response to `check_segment`:
  exactly one terminal frame, last (a flat transcript has lost its
  segment boundaries, so `check_transcript` cannot assert this)

Frames are plain dicts (as decoded from SSE `data:` lines or a transcript
JSONL). All checks return violation strings; an empty list is a pass.
"""

from __future__ import annotations

import itertools
import re
from typing import Any

from harness.wire import STUDIO_FRAME_ADAPTER
from pydantic import ValidationError

TERMINAL_EVENTS = frozenset({"done", "error", "plan.proposed", "elicitation.request"})
TURN_FINAL_EVENTS = frozenset({"done", "error"})

# Measured or structural fields inside otherwise-fact structures; excluded
# from the grounding corpus so a coincidental match can't launder an
# ungrounded number.
_NON_FACT_KEYS = frozenset({"duration_ms", "seq", "message_seq"})

_NUMBER_TOKEN = re.compile(r"\d[\d,]*(?:\.\d+)?")

# A scale word after a number multiplies it: "7.6 million" is 7,600,000.
_SCALE_WORD = re.compile(r"\s*(thousand|million|billion|trillion)\b", re.IGNORECASE)
_SCALES = {"thousand": 1e3, "million": 1e6, "billion": 1e9, "trillion": 1e12}

# Small counts ("2 series", "3 steps") are derivable from structure; only
# larger integers, years, and decimals must ground.
_GROUNDING_INT_FLOOR = 12


def validate_frames(frames: list[dict[str, Any]]) -> list[str]:
    violations = []
    for index, frame in enumerate(frames):
        try:
            STUDIO_FRAME_ADAPTER.validate_python(frame)
        except ValidationError as error:
            event = frame.get("event", "<missing event>")
            violations.append(f"frame {index} ({event}): fails contract validation: {error}")
    return violations


def split_turns(frames: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """The contract says a new turn starts at seq 0 — that reset is the turn
    boundary. Splitting on the boundary lets `check_turn` catch frames
    trailing a terminal: they stay in the same turn and fail the
    terminal-is-last rule."""
    turns: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for frame in frames:
        if current and frame.get("seq") == 0:
            turns.append(current)
            current = []
        current.append(frame)
    if current:
        turns.append(current)
    return turns


def check_turn(turn: list[dict[str, Any]], *, label: str = "turn") -> list[str]:
    violations = []

    seqs = [frame.get("seq") for frame in turn]
    if seqs and seqs[0] != 0:
        violations.append(f"{label}: first frame has seq {seqs[0]}, contract says 0")
    for previous, current in itertools.pairwise(seqs):
        if previous is None or current is None or current <= previous:
            violations.append(f"{label}: seq not strictly increasing ({previous} -> {current})")
            break

    events = [frame.get("event") for frame in turn]
    if events and events[-1] not in TURN_FINAL_EVENTS:
        violations.append(f"{label}: ends on '{events[-1]}', not done/error (incomplete turn)")
    for index, event in enumerate(events[:-1]):
        if event in TURN_FINAL_EVENTS:
            violations.append(
                f"{label}: {event} at position {index} is followed by more frames — "
                "done/error end the turn"
            )
    if events.count("done") > 1:
        violations.append(f"{label}: more than one done frame")
    return violations


def check_conversation(
    frames: list[dict[str, Any]], *, complete_conversation: bool = False
) -> list[str]:
    """Cross-turn invariants over one conversation's frames, in stream order.

    `complete_conversation=True` additionally requires every
    `component.revised` to name a component this stream created — sound only
    when the frames cover the conversation from its first turn (the runner's
    case; a transcript may record a later slice).
    """
    violations = []

    composition_id: str | None = None
    last_updated_revision = 0
    seen_components: set[str] = set()
    for index, frame in enumerate(frames):
        event = frame.get("event")
        manifest = None
        if event == "composition.updated":
            manifest = frame.get("manifest")
        elif event == "done":
            manifest = frame.get("composition")
        if manifest:
            if composition_id is None:
                composition_id = manifest.get("composition_id")
            elif manifest.get("composition_id") != composition_id:
                violations.append(
                    f"frame {index}: composition_id changed mid-conversation "
                    f"({composition_id} -> {manifest.get('composition_id')})"
                )
            revision = manifest.get("revision", 0)
            if event == "composition.updated":
                if revision <= last_updated_revision:
                    violations.append(
                        f"frame {index}: composition revision not strictly increasing "
                        f"({last_updated_revision} -> {revision})"
                    )
                last_updated_revision = revision
            elif revision < last_updated_revision:
                violations.append(
                    f"frame {index}: done carries composition revision {revision}, "
                    f"older than the last composition.updated ({last_updated_revision})"
                )
            # Blocks may reference components streamed before this recording
            # (a resumed conversation), so only frame-level references are
            # held to the seen set below.
        if event == "component.created":
            seen_components.add(frame.get("component_id", ""))
        if event == "component.revised":
            if complete_conversation and frame.get("component_id") not in seen_components:
                violations.append(
                    f"frame {index}: component.revised for {frame.get('component_id')!r}, "
                    "never streamed as component.created in this conversation"
                )
            seen_components.add(frame.get("component_id", ""))
    return violations


def _numbers_in_text(text: str) -> list[tuple[str, float, float]]:
    """Numeric claims in narration text, as (written form, number, scale):
    "7.6 million" is ("7.6 million", 7.6, 1e6). Integers up to the floor
    are exempt (structural counts) -- unless written as a percentage or
    with a scale word, which are always quantitative claims. Known residual
    gaps, accepted: scientific notation, and malformed thousands separators
    ("1,2" reads as 12)."""
    values = []
    for match in _NUMBER_TOKEN.finditer(text):
        token = match.group()
        try:
            value = float(token.replace(",", ""))
        except ValueError:
            continue
        scale_word = _SCALE_WORD.match(text, match.end())
        scale = _SCALES[scale_word.group(1).lower()] if scale_word else 1.0
        written = text[match.start() : scale_word.end()] if scale_word else token
        is_percent = text[match.end() : match.end() + 1] == "%"
        if (
            not is_percent
            and not scale_word
            and value == int(value)
            and abs(value) <= _GROUNDING_INT_FLOOR
        ):
            continue
        values.append((written, value, scale))
    return values


def _numbers_in_prose(text: str) -> set[float]:
    """Numbers in a fact STRING. Identifiers are excluded: a string with no
    whitespace that is not itself a number (component ids, dcids, ISO
    timestamps, catalog versions) contributes nothing — otherwise the
    digits inside `cmp_01M13ABC2026` would ground an invented "2026"."""
    stripped = text.strip()
    if not stripped:
        return set()
    if " " not in stripped:
        try:
            return {float(stripped.replace(",", ""))}
        except ValueError:
            return set()
    found: set[float] = set()
    for token in _NUMBER_TOKEN.findall(stripped):
        try:
            found.add(float(token.replace(",", "")))
        except ValueError:
            continue
    return found


def _numbers_in_data(node: Any) -> set[float]:
    """Every quantitative value reachable in a fact structure."""
    found: set[float] = set()
    if isinstance(node, bool):
        return found
    if isinstance(node, (int, float)):
        found.add(float(node))
    elif isinstance(node, str):
        found |= _numbers_in_prose(node)
    elif isinstance(node, dict):
        for key, value in node.items():
            if key in _NON_FACT_KEYS:
                continue
            found |= _numbers_in_data(value)
    elif isinstance(node, list):
        for item in node:
            found |= _numbers_in_data(item)
    return found


def _grounded(value: float, corpus: set[float], scale: float = 1.0) -> bool:
    """Exact, or a rounding of a corpus fact at the precision the narration
    used (54.462 grounds "54.5" and "54"; 12.6 grounds "13"; 7,640,000
    grounds "7.6 million")."""
    if value * scale in corpus:
        return True
    text = f"{value}"
    decimals = len(text.split(".")[1]) if "." in text and not text.endswith(".0") else 0
    return any(round(fact / scale, decimals) == value for fact in corpus)


def _fact_corpus(frame: dict[str, Any]) -> set[float]:
    """What narration may cite from one frame — the fact summary, step
    results, plan text, and error/elicitation prose. Deliberately NOT the
    whole frame: ids, revisions, catalog versions, provenance timestamps,
    and composition counts are structure and must not launder
    an invented number."""
    event = frame.get("event")
    if event in ("component.created", "component.revised"):
        summary = dict(frame.get("fact_summary") or {})
        summary.pop("component_id", None)
        summary.pop("revision", None)
        return _numbers_in_data(summary)
    if event == "activity.step":
        return _numbers_in_data([frame.get("label"), frame.get("detail")])
    if event == "plan.proposed":
        return _numbers_in_data([frame.get("title"), frame.get("steps"), frame.get("decisions")])
    if event == "error":
        return _numbers_in_data(frame.get("error"))
    if event == "elicitation.request":
        return _numbers_in_data([frame.get("question"), frame.get("options")])
    return set()


def check_grounding(turns: list[list[dict[str, Any]]]) -> list[str]:
    """Numbers in `message.complete` must exist in the turn's own fact
    frames or in an earlier turn's fact summaries (the canvas digest)."""
    violations = []
    digest_corpus: set[float] = set()
    for number, turn in enumerate(turns, start=1):
        corpus = set(digest_corpus)
        for frame in turn:
            corpus |= _fact_corpus(frame)
        for frame in turn:
            if frame.get("event") != "message.complete":
                continue
            ungrounded = [
                written
                for written, value, scale in _numbers_in_text(frame.get("text", ""))
                if not _grounded(value, corpus, scale)
            ]
            if ungrounded:
                violations.append(
                    f"turn {number}: message.complete cites number(s) {ungrounded} "
                    "found in no fact frame of this conversation"
                )
        for frame in turn:
            if frame.get("event") in ("component.created", "component.revised"):
                digest_corpus |= _fact_corpus(frame)
    return violations


def check_segment(frames: list[dict[str, Any]], *, label: str = "segment") -> list[str]:
    """One SSE response's frames: it must end on exactly one terminal frame
    (done | error | plan.proposed | elicitation.request) with nothing after
    it. Only a live runner can apply this — a flat transcript has lost its
    segment boundaries."""
    if not frames:
        return [f"{label}: stream carried no frames"]
    violations = []
    events = [frame.get("event") for frame in frames]
    if events[-1] not in TERMINAL_EVENTS:
        violations.append(f"{label}: stream ended on '{events[-1]}', not a terminal frame")
    violations.extend(
        f"{label}: terminal frame {event} at position {index} is followed by more "
        "frames on the same stream"
        for index, event in enumerate(events[:-1])
        if event in TERMINAL_EVENTS
    )
    return violations


def check_transcript(
    frames: list[dict[str, Any]], *, complete_conversation: bool = False
) -> list[str]:
    """All checks over one conversation's frames. Empty list = conformant.

    A validation failure short-circuits: the structural checks assume
    contract-shaped frames, and running them over arbitrary dicts would
    crash."""
    violations = validate_frames(frames)
    if violations:
        return violations
    turns = split_turns(frames)
    for number, turn in enumerate(turns, start=1):
        violations += check_turn(turn, label=f"turn {number}")
    violations += check_conversation(frames, complete_conversation=complete_conversation)
    violations += check_grounding(turns)
    return violations

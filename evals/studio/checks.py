"""Wire-contract conformance checks over studio frame streams.

The contract layer of harness-studio-v1: mechanical, model-free assertions
that any recorded stream (a scripted run or a live run) obeys the
invariants `docs/frontend-interface.md` promises the frontend:

- every frame validates against `harness.wire.STUDIO_FRAME_ADAPTER`
- `seq` starts at 0 per turn and strictly increases across resumed segments
- a turn ends on `done` or `error`, exactly once, with nothing after it
  (which also rules out `done` following a stream-fatal `error`)
- composition revisions strictly increase across `composition.updated`
  frames, and the `composition_id` never changes within a conversation
- `component.revised` only ever names a component the conversation has seen
- every number in a `message.complete` is grounded in the turn's fact
  frames — fact summaries, step results, plan text, the periods a component
  draws — or a prior turn's fact summaries (the canvas digest narration may
  cite), or is a value a component draws (cited to four or more
  significant figures); ids, revisions, and timestamps are structure and
  never ground anything
- a live runner additionally holds each SSE response to `check_segment`:
  exactly one terminal frame, last (a flat transcript has lost its
  segment boundaries, so `check_transcript` cannot assert this)

Beyond the contract, a case names the properties it holds in its `checks`
list (`CASE_CHECKS`, run by `run_case_checks`). Those read a case's
`Evidence`: the frames of each turn and, when the studio runs scripted
(`orchestrator_scripted`), the log of every model request and tool call, so a
property of what a model was shown is asserted on the request itself.

Frames are plain dicts (as decoded from SSE `data:` lines). All checks return
violation strings; an empty list is a pass.
"""

from __future__ import annotations

import itertools
import json
import re
from collections import Counter
from collections.abc import Callable, Iterator, Set
from dataclasses import dataclass
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

# Fewest significant figures at which a drawn chart point grounds a citation.
_DRAWN_MIN_SIGNIFICANT_FIGURES = 4


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


@dataclass(frozen=True)
class _Citation:
    """One number narration cites. `decimals` and `figures` come from the
    written digits, so "1.260" keeps the zero a float would drop."""

    written: str
    value: float
    scale: float
    decimals: int
    figures: int


def _numbers_in_text(text: str) -> list[_Citation]:
    """Numeric claims in narration text: "7.60 million" is value 7.6 at scale
    1e6, written to 2 decimals and 3 significant figures. Integers up to the
    floor are exempt (structural counts) -- unless written as a percentage or
    with a scale word, which are always quantitative claims. Known residual
    gaps, accepted: scientific notation, and malformed thousands separators
    ("1,2" reads as 12)."""
    citations = []
    for match in _NUMBER_TOKEN.finditer(text):
        token = match.group()
        digits = token.replace(",", "")
        try:
            value = float(digits)
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
        whole, _, fraction = digits.partition(".")
        citations.append(
            _Citation(
                written=written,
                value=value,
                scale=scale,
                decimals=len(fraction),
                figures=len((whole + fraction).lstrip("0")),
            )
        )
    return citations


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
            return {abs(float(stripped.replace(",", "")))}
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
        found.add(abs(float(node)))
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


def _drawn_fields(node: Any) -> Iterator[tuple[str, Any]]:
    """Every `value` number and `period` string a component draws, whatever
    block holds them (stat card, chart point, table row)."""
    if isinstance(node, dict):
        for key, child in node.items():
            if (
                key == "value" and isinstance(child, (int, float)) and not isinstance(child, bool)
            ) or (key == "period" and isinstance(child, str)):
                yield key, child
            else:
                yield from _drawn_fields(child)
    elif isinstance(node, list):
        for child in node:
            yield from _drawn_fields(child)


def _drawn_values(frame: dict[str, Any]) -> set[float]:
    """The magnitudes a component draws. A chart holds hundreds of them, and a
    citation of up to three figures lands within half a unit of one by
    chance, so `_grounded` accepts these only for longer citations."""
    component = (frame.get("payload") or {}).get("component")
    return {abs(float(child)) for key, child in _drawn_fields(component) if key == "value"}


def _drawn_periods(frame: dict[str, Any]) -> set[float]:
    """Years at period endpoints; month and day digits are not quantitative facts."""
    component = (frame.get("payload") or {}).get("component")
    return {
        float(year)
        for key, child in _drawn_fields(component)
        if key == "period"
        for year in re.findall(r"(?:^|/)([0-9]{4})(?=$|-|/)", child)
    }


def _matches(citation: _Citation, facts: Set[float]) -> bool:
    if citation.value * citation.scale in facts:
        return True
    return any(round(fact / citation.scale, citation.decimals) == citation.value for fact in facts)


def _grounded(citation: _Citation, corpus: Set[float], *, drawn: Set[float] = frozenset()) -> bool:
    """Exact, or a rounding of a corpus fact at the decimals the narration
    wrote (54.462 grounds "54.5" and "54"; 12.6 grounds "13"; 7,640,000
    grounds "7.6 million"; 7,600,000 grounds "7.60 million" and 7,640,000
    does not). A drawn value grounds only a citation written with at least
    `_DRAWN_MIN_SIGNIFICANT_FIGURES` significant figures, zeros of a whole
    number and trailing decimal zeros included."""
    if _matches(citation, corpus):
        return True
    return citation.figures >= _DRAWN_MIN_SIGNIFICANT_FIGURES and _matches(citation, drawn)


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
        return _numbers_in_data(summary) | _drawn_periods(frame)
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
    frames or in an earlier turn's fact summaries (the canvas digest), or
    match a value a component draws. Facts compare by magnitude: narration
    says "an outflow of $223 million" for -223e6."""
    violations = []
    digest_corpus: set[float] = set()
    digest_drawn: set[float] = set()
    for number, turn in enumerate(turns, start=1):
        corpus = set(digest_corpus)
        drawn = set(digest_drawn)
        for frame in turn:
            corpus |= _fact_corpus(frame)
            drawn |= _drawn_values(frame)
        for frame in turn:
            if frame.get("event") != "message.complete":
                continue
            ungrounded = [
                citation.written
                for citation in _numbers_in_text(frame.get("text", ""))
                if not _grounded(citation, corpus, drawn=drawn)
            ]
            if ungrounded:
                violations.append(
                    f"turn {number}: message.complete cites number(s) {ungrounded} "
                    "found in no fact frame of this conversation"
                )
        for frame in turn:
            if frame.get("event") in ("component.created", "component.revised"):
                digest_corpus |= _fact_corpus(frame)
                digest_drawn |= _drawn_values(frame)
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


# ---------------------------------------------------------------------------
# Case checks: properties a case names in its `checks` list
# ---------------------------------------------------------------------------

# ADK's fence around another agent's quoted turn
# (`google.adk.flows.llm_flows._fencing.QUOTED_CONTENT_BEGIN`), spelled out so a
# change to the marker shows here instead of passing unnoticed.
RELAY_MARKER = "<<<BEGIN_QUOTED_AGENT_CONTENT>>>"
# The function name ADK gives a pause for the user's input
# (`google.adk.workflow.utils._workflow_hitl_utils.REQUEST_INPUT_FUNCTION_CALL_NAME`).
REQUEST_INPUT_NAME = "adk_request_input"

# ADK opens each relayed payload with the relaying agent's name in brackets; its
# own framing sentence carries the marker with no name.
_AUTHOR_TAG = re.compile(r"^\[([^\]\n]+)\]")


@dataclass(frozen=True)
class Evidence:
    """What a case's checks read: each message turn's frames, and the scripted
    orchestrator's log of model requests and tool calls (`None` against a live
    server, which records neither)."""

    turn_frames: list[list[dict[str, Any]]]
    log: list[dict[str, Any]] | None = None


def _requests(evidence: Evidence, *, agent: str | None = None) -> list[dict[str, Any]]:
    """The log's model requests, in order, optionally one agent's."""
    return [
        entry
        for entry in evidence.log or []
        if entry["kind"] == "request" and agent in (None, entry["agent"])
    ]


def _contents_text(request: dict[str, Any]) -> str:
    """Everything the model was shown besides its system instruction."""
    return json.dumps(request["contents"], ensure_ascii=False)


def _label(request: dict[str, Any], log: list[dict[str, Any]]) -> str:
    ordinal = sum(
        1
        for entry in log[: log.index(request)]
        if entry["kind"] == "request" and entry["agent"] == request["agent"]
    )
    return f"{request['agent']} request {ordinal + 1}"


def _need_log(evidence: Evidence, check: str) -> list[str]:
    return (
        []
        if evidence.log is not None
        else [f"{check}: needs the scripted orchestrator's request log"]
    )


def approval_payload_absent_after_approval(evidence: Evidence) -> list[str]:
    """No model request made after the plan's first step ran mentions a pause for
    the user's input. The approval pause carries the whole plan and the user's
    answer to it; the browser reads both from the stream, and no model is
    meant to."""
    if missing := _need_log(evidence, "approval_payload_absent_after_approval"):
        return missing
    log = evidence.log or []
    started = next(
        (
            index
            for index, entry in enumerate(log)
            if entry["kind"] == "call" and entry["layer"] == "step"
        ),
        None,
    )
    if started is None:
        return ["approval_payload_absent_after_approval: no plan step ran, so nothing was approved"]
    return [
        f"{_label(entry, log)} after approval carries a pause for input ({REQUEST_INPUT_NAME})"
        for entry in log[started:]
        if entry["kind"] == "request" and REQUEST_INPUT_NAME in _contents_text(entry)
    ]


def _fenced_authors(request: dict[str, Any]) -> list[str]:
    """The author named on each fenced block in `request`'s contents, `untagged`
    for ADK's own framing sentence, which names none."""
    authors = []
    for content in request["contents"]:
        for part in content.get("parts", []):
            text = part.get("text") or ""
            if RELAY_MARKER in text:
                tag = _AUTHOR_TAG.match(text)
                authors.append(tag.group(1) if tag else "untagged")
    return authors


def _relay_marker_violations(evidence: Evidence, *, check: str, agent: str | None) -> list[str]:
    if missing := _need_log(evidence, check):
        return missing
    log = evidence.log or []
    return [
        f"{_label(entry, log)} holds fenced quoted blocks by "
        f"{dict(sorted(Counter(authors).items()))}"
        for entry in _requests(evidence, agent=agent)
        if (authors := _fenced_authors(entry))
    ]


def no_relay_marker(evidence: Evidence) -> list[str]:
    """No model request holds ADK's fence around another agent's quoted turn.
    The orchestrator and the planner read their own hand-backs and answers;
    a fenced block is the framework presenting a graph node's output as another
    agent's speech."""
    return _relay_marker_violations(evidence, check="no_relay_marker", agent=None)


def planner_sees_no_relay_marker(evidence: Evidence) -> list[str]:
    """`no_relay_marker` over the planner's requests alone: a planner entered a
    second time in a turn reads the turn's session, and the orchestrator's
    hand-back from the first plan is in it."""
    return _relay_marker_violations(evidence, check="planner_sees_no_relay_marker", agent="planner")


_SPEC_ID = re.compile(r"spec_[0-9a-f]{32}")


def _search_evidence(evidence: Evidence) -> dict[str, set[str]]:
    """What the orchestrator's `search` answers named: spec ids, warning codes
    and the years each part's window covers."""
    found: dict[str, set[str]] = {"spec_id": set(), "warning code": set(), "window year": set()}
    for request in _requests(evidence, agent="orchestrator"):
        for content in request["contents"]:
            for part in content.get("parts", []):
                response = part.get("function_response")
                if not response or response["name"] != "search":
                    continue
                for reading in (response.get("response") or {}).get("parts", []):
                    if reading.get("spec_id"):
                        found["spec_id"].add(reading["spec_id"])
                    found["warning code"].update(w["code"] for w in reading.get("warnings") or [])
                    found["window year"].update(
                        str(year)
                        for year in (reading.get("window") or {}).values()
                        if isinstance(year, int)
                    )
    return found


def planner_first_message_carries_search_evidence(evidence: Evidence) -> list[str]:
    """The planner's first request names every spec id, warning code and window
    year the turn's searches returned. A planner that is not shown what
    search found cannot pin a source, heed a warning, or state a period
    against the data's coverage."""
    if missing := _need_log(evidence, "planner_first_message_carries_search_evidence"):
        return missing
    planner = _requests(evidence, agent="planner")
    searched = _search_evidence(evidence)
    if not planner:
        return ["planner_first_message_carries_search_evidence: the planner never ran"]
    if not any(searched.values()):
        return ["planner_first_message_carries_search_evidence: no search answer to carry"]
    shown = _contents_text(planner[0])
    return [
        f"the planner's first request names no {kind} {value!r} the search returned"
        for kind, values in searched.items()
        for value in sorted(values)
        if value not in shown
    ]


def previous_final_message_in_view(evidence: Evidence) -> list[str]:
    """Each turn's first orchestrator request shows the previous turn's closing
    message whole. A reply of "yes" answers the offer that message ends on;
    a message cut short has lost it."""
    if missing := _need_log(evidence, "previous_final_message_in_view"):
        return missing
    violations = []
    turns: list[list[dict[str, Any]]] = []
    for entry in evidence.log or []:
        if entry["kind"] == "turn":
            turns.append([])
        elif turns and entry["kind"] == "request" and entry["agent"] == "orchestrator":
            turns[-1].append(entry)
    for number in range(2, len(turns) + 1):
        closing = [
            frame["text"]
            for frame in evidence.turn_frames[number - 2]
            if frame["event"] == "message.complete"
        ]
        if not closing:
            violations.append(f"turn {number - 1} closed with no message to carry forward")
        elif not turns[number - 1]:
            violations.append(f"turn {number} made no orchestrator request")
        else:
            first = turns[number - 1][0]
            shown = first["system"] + _contents_text(first)
            escaped = json.dumps(closing[-1])[1:-1]
            if closing[-1] not in shown and escaped not in shown:
                violations.append(
                    f"turn {number}'s first orchestrator request does not show turn "
                    f"{number - 1}'s closing message whole"
                )
    return violations


def earlier_plan_in_view(evidence: Evidence) -> list[str]:
    """Each turn's first orchestrator request names, by spec_id, every source a
    plan step retrieved in an earlier turn. A question about a plan ("why does
    the data stop in 2023?") comes a turn after the plan ran, in a session that
    holds none of that turn's events."""
    if missing := _need_log(evidence, "earlier_plan_in_view"):
        return missing
    turns: list[tuple[list[dict[str, Any]], list[str]]] = []
    for entry in evidence.log or []:
        if entry["kind"] == "turn":
            turns.append(([], []))
        elif turns and entry["kind"] == "request" and entry["agent"] == "orchestrator":
            turns[-1][0].append(entry)
        elif (
            turns
            and entry["kind"] == "call"
            and entry["layer"] == "step"
            and entry["tool"] == "get_data"
        ):
            turns[-1][1].append(str(entry["arguments"].get("spec_id")))
    if not any(spec_ids for _, spec_ids in turns[:-1]):
        return ["earlier_plan_in_view: no plan step retrieved anything before the last turn"]
    violations = []
    earlier: dict[str, None] = {}
    for number, (requests, spec_ids) in enumerate(turns, start=1):
        if earlier and not requests:
            violations.append(f"turn {number} made no orchestrator request")
        elif earlier:
            shown = requests[0]["system"] + _contents_text(requests[0])
            violations += [
                f"turn {number}'s first orchestrator request does not name {spec_id}, "
                "which an earlier plan retrieved"
                for spec_id in earlier
                if spec_id not in shown
            ]
        earlier.update(dict.fromkeys(spec_ids))
    return violations


def no_repeated_failed_step(evidence: Evidence) -> list[str]:
    """A plan step that failed is not run again with the same tool, label and
    outcome text in one conversation. A failure a retry cannot change comes
    back identical, so a repeat spends a turn to learn nothing."""
    seen: set[tuple[Any, ...]] = set()
    violations = []
    for number, turn in enumerate(evidence.turn_frames, start=1):
        for frame in turn:
            if frame["event"] != "activity.step" or frame["status"] != "failed":
                continue
            key = (frame.get("tool"), frame["label"], frame.get("detail"))
            if key in seen:
                violations.append(
                    f"turn {number}: step {frame['label']!r} failed again with the outcome it failed with before"
                )
            seen.add(key)
    return violations


def translation_stops_on_a_repeated_failure(evidence: Evidence) -> list[str]:
    """Once a translation fails the same way it failed before, nothing is
    translated again in that planner entry. The plan did not change where it
    failed, so a further attempt reaches the same stop. A translation is the
    `translation` an accepted proposal answers with, `propose_plan`'s or
    `submit_analysis`'s, read off the response as `harness.planner_seam` reads it."""
    if missing := _need_log(evidence, "translation_stops_on_a_repeated_failure"):
        return missing
    translations = [
        entry["response"]["translation"]
        for entry in evidence.log or []
        if entry["kind"] == "call"
        and entry["layer"] == "planner"
        and isinstance(entry["response"].get("translation"), dict)
    ]
    repeated = next(
        (index for index, answer in enumerate(translations) if answer.get("repeated")), None
    )
    if repeated is None:
        return ["translation_stops_on_a_repeated_failure: no translation failed the same way twice"]
    return [
        f"translation {index + 1} ran after a repeated failure"
        for index in range(repeated + 1, len(translations))
    ]


# The search warning codes under which part of the ask was dropped
# (`translator.discovery.BLOCKING_WARNING_CODES`).
_BLOCKING_WARNING_CODES = frozenset({"role_text_not_recognised", "time_text_not_parsed"})


def blocked_reading_is_searched_again(evidence: Evidence) -> list[str]:
    """A source whose search reading carries a blocking warning is searched
    again before a plan retrieves it. The warning says the search dropped part
    of the ask, so its source answers a narrower question than the plan's."""
    if missing := _need_log(evidence, "blocked_reading_is_searched_again"):
        return missing
    blocked = set()
    for request in _requests(evidence, agent="orchestrator"):
        for content in request["contents"]:
            for part in content.get("parts", []):
                response = part.get("function_response")
                if not response or response["name"] != "search":
                    continue
                for reading in (response.get("response") or {}).get("parts", []):
                    codes = {w["code"] for w in reading.get("warnings") or []}
                    if reading.get("spec_id") and codes & _BLOCKING_WARNING_CODES:
                        blocked.add(reading["spec_id"])
    if not blocked:
        return ["blocked_reading_is_searched_again: no search reading carried a blocking warning"]
    log = evidence.log or []
    searched = any(entry.get("layer") == "discovery" for entry in log)
    retrieved = {
        str(entry["arguments"].get("spec_id"))
        for entry in log
        if entry["kind"] == "call" and entry["layer"] == "step" and entry["tool"] == "get_data"
    }
    if not retrieved:
        return ["blocked_reading_is_searched_again: no plan step retrieved anything"]
    return [
        f"the plan retrieved {spec_id} on a reading with a blocking warning, unsearched"
        for spec_id in sorted(blocked & retrieved)
        if not searched
    ]


def _listed_asks(evidence: Evidence) -> list[str]:
    """The asks of the orchestrator's last `plan` call, as its requests record it."""
    asks: list[str] = []
    for request in _requests(evidence, agent="orchestrator"):
        for content in request["contents"]:
            for part in content.get("parts", []):
                call = part.get("function_call")
                if call and call["name"] == "plan":
                    asks = list((call.get("args") or {}).get("asks") or [])
    return asks


def _handed_back_asks(evidence: Evidence) -> dict[str, dict[str, Any]]:
    """Each ask a plan's hand-back to the orchestrator reports, by its text."""
    reported: dict[str, dict[str, Any]] = {}
    for request in _requests(evidence, agent="orchestrator"):
        for content in request["contents"]:
            for part in content.get("parts", []):
                for line in (part.get("text") or "").split("\n"):
                    try:
                        body = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(body, dict):
                        reported.update({ask["ask"]: ask for ask in body.get("asks", [])})
    return reported


def every_ask_accounted_for(evidence: Evidence) -> list[str]:
    """Each output the orchestrator listed as an ask when it called `plan` is on
    the plan card, answered by steps the card shows, declined with a reason,
    or answered in part with the part not computed named, and comes back in the plan's hand-back with the status of each step that
    answers it, or its decline. An ask listed and then dropped is the output
    the user asked for and never got."""
    if missing := _need_log(evidence, "every_ask_accounted_for"):
        return missing
    listed = _listed_asks(evidence)
    if not listed:
        return ["every_ask_accounted_for: no plan call listed an ask"]
    cards = [f for turn in evidence.turn_frames for f in turn if f["event"] == "plan.proposed"]
    if not cards:
        return ["every_ask_accounted_for: no plan card was proposed"]
    card = cards[-1]
    on_card = {ask["ask"]: ask for ask in card.get("asks") or []}
    step_ids = {step["id"] for step in card["steps"]}
    handed_back = _handed_back_asks(evidence)
    violations = []
    for ask in listed:
        shown = on_card.get(ask)
        if shown is None:
            violations.append(f"the plan card does not show the ask {ask!r}")
        elif not (shown.get("answered_by") or shown.get("declined")):
            violations.append(f"the plan card neither answers nor declines {ask!r}")
        elif shown.get("answered_by") and shown.get("declined") and not shown.get("not_computed"):
            violations.append(f"the plan card answers {ask!r} in part without naming the rest")
        elif set(shown.get("answered_by") or ()) - step_ids:
            violations.append(f"the plan card answers {ask!r} with steps it does not show")
        reported = handed_back.get(ask)
        if reported is None:
            violations.append(f"the plan's hand-back does not report the ask {ask!r}")
        elif not reported.get("declined") and not all(
            entry.get("status") for entry in reported.get("answered_by") or [{}]
        ):
            violations.append(f"the plan's hand-back gives no status for {ask!r}")
    return violations


def _direct_answers(evidence: Evidence) -> list[dict[str, Any]]:
    """Every answer the orchestrator's `submit_analysis` gave, in order, each
    once, as the requests after it show it."""
    answers: list[dict[str, Any]] = []
    for request in _requests(evidence, agent="orchestrator"):
        for content in request["contents"]:
            for part in content.get("parts", []):
                response = part.get("function_response")
                if response and response["name"] == "submit_analysis":
                    answer = response.get("response") or {}
                    if answer not in answers:
                        answers.append(answer)
    return answers


def _retrievals(evidence: Evidence) -> list[str]:
    return [
        str(entry["arguments"].get("spec_id"))
        for entry in evidence.log or []
        if entry["kind"] == "call" and entry.get("layer") == "step" and entry["tool"] == "get_data"
    ]


def direct_run_completed(evidence: Evidence) -> list[str]:
    """A `submit_analysis` call answered `completed`, its run completed, with a
    frame the orchestrator can chart, and no plan was proposed."""
    if missing := _need_log(evidence, "direct_run_completed"):
        return missing
    answers = _direct_answers(evidence)
    if not answers:
        return ["direct_run_completed: the orchestrator never called submit_analysis"]
    completed = [
        answer
        for answer in answers
        if answer.get("outcome") == "completed"
        and (answer.get("run") or {}).get("status") == "completed"
        and answer.get("frames")
    ]
    violations = (
        []
        if completed
        else [f"direct_run_completed: no submit_analysis answer completed with frames: {answers}"]
    )
    proposed = sum(
        1 for turn in evidence.turn_frames for frame in turn if frame["event"] == "plan.proposed"
    )
    if proposed:
        violations.append(f"direct_run_completed: {proposed} plan card(s) were proposed")
    return violations


def unverified_source_refused(evidence: Evidence) -> list[str]:
    """A `submit_analysis` call naming a spec_id no search settled on is
    answered `needs_fields` for that spec_id, and nothing is retrieved."""
    if missing := _need_log(evidence, "unverified_source_refused"):
        return missing
    answers = _direct_answers(evidence)
    if not answers:
        return ["unverified_source_refused: the orchestrator never called submit_analysis"]
    refused = [
        answer
        for answer in answers
        if answer.get("outcome") == "needs_fields"
        and any(
            error.get("field") == "spec_id" and error.get("code") == "unverified_source"
            for error in answer.get("errors") or []
        )
    ]
    violations = (
        [] if refused else [f"unverified_source_refused: no answer refused the spec_id: {answers}"]
    )
    violations.extend(
        f"unverified_source_refused: {spec_id} was retrieved" for spec_id in _retrievals(evidence)
    )
    return violations


CASE_CHECKS: dict[str, Callable[[Evidence], list[str]]] = {
    "approval_payload_absent_after_approval": approval_payload_absent_after_approval,
    "no_relay_marker": no_relay_marker,
    "planner_sees_no_relay_marker": planner_sees_no_relay_marker,
    "planner_first_message_carries_search_evidence": planner_first_message_carries_search_evidence,
    "previous_final_message_in_view": previous_final_message_in_view,
    "earlier_plan_in_view": earlier_plan_in_view,
    "no_repeated_failed_step": no_repeated_failed_step,
    "translation_stops_on_a_repeated_failure": translation_stops_on_a_repeated_failure,
    "blocked_reading_is_searched_again": blocked_reading_is_searched_again,
    "every_ask_accounted_for": every_ask_accounted_for,
    "direct_run_completed": direct_run_completed,
    "unverified_source_refused": unverified_source_refused,
}
# The checks that read the scripted orchestrator's request log.
REQUEST_CHECKS = frozenset(CASE_CHECKS) - {"no_repeated_failed_step"}


def check_names(names: object, *, label: str) -> list[str]:
    """A case's `checks` list is a list of names this module defines."""
    if not isinstance(names, list) or not names:
        return [f"{label}: checks must be a non-empty list of check names"]
    return [
        f"{label}: {name!r} is not a check checks.py defines"
        for name in names
        if name not in CASE_CHECKS
    ]


def run_case_checks(names: list[str], evidence: Evidence) -> list[str]:
    """Every named check over `evidence`; the violations, empty when all hold."""
    return [violation for name in names for violation in CASE_CHECKS[name](evidence)]

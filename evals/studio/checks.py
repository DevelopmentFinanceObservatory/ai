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
  draws, the first and last value of each series it draws, its series count
  and the counts of a cut card's title, or a prior
  turn's fact summaries (the canvas digest narration may cite), or is a
  value a component draws (cited to four or more significant figures); ids,
  revisions, and timestamps are structure and never ground anything
- a live runner additionally holds each SSE response to `check_segment`:
  exactly one terminal frame, last (a flat transcript has lost its
  segment boundaries, so `check_transcript` cannot assert this)

`check_reply` holds a turn's closing message to the figures its cards
supplied: the values a `make_charts` result lists under `reply_figures`,
each beside its own series and its own period.

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
import math
import re
from collections import Counter
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence, Set
from dataclasses import dataclass
from typing import Any

from harness.reply_figures import (
    DrawnSeries,
    card_result,
    drawn_series,
    point_figure,
    series_ends,
)
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

# A reply rounds a figure to this many significant figures, as the platform's
# fact summary does. A number written to fewer states a figure only when it
# is the figure's whole value.
_REPLY_SIGNIFICANT_FIGURES = 3


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
    return [citation for _, citation, structural in _cited(text) if not structural]


def _cited(text: str) -> Iterator[tuple[int, _Citation, bool]]:
    """Every number `text` writes, with where it starts and whether it is a
    structural count (a bare integer up to the grounding floor)."""
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
        structural = (
            not is_percent
            and not scale_word
            and value == int(value)
            and abs(value) <= _GROUNDING_INT_FLOOR
        )
        whole, _, fraction = digits.partition(".")
        citation = _Citation(
            written=written,
            value=value,
            scale=scale,
            decimals=len(fraction),
            figures=len((whole + fraction).lstrip("0")),
        )
        yield match.start(), citation, structural


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


def _round_significant(value: float, figures: int) -> float:
    if value == 0:
        return 0.0
    return round(value, figures - 1 - math.floor(math.log10(abs(value))))


def _drawn_ends(frame: dict[str, Any]) -> set[float]:
    """The first and last value by period (`series_ends`) of every series a
    component draws, rounded as the platform's fact summary rounds them. The summary lists twelve series
    of a card and `make_charts` gives a reply those two values of any series
    drawn, so each grounds the citation that writes it at that precision,
    whichever series it is of."""
    component = (frame.get("payload") or {}).get("component") or {}
    return {
        abs(_round_significant(value, _REPLY_SIGNIFICANT_FIGURES))
        for series in drawn_series(component, frame.get("fact_summary") or {})
        for _, value in series_ends(series) or ()
    }


# The suffix `make_charts` writes on the title of a card cut to its largest
# members: how many it draws, of how many with a value.
_CUT_TITLE = re.compile(r"\((\d+) largest of (\d+) in ")


def _card_counts(frame: dict[str, Any]) -> set[float]:
    """The counts a `make_charts` result states of a card in `holds`, read
    where the stream carries them: `series_count` is the number of series
    the payload's component draws, and a cut card's `cut.drawn` and
    `cut.with_a_value` are the two numbers of its title's suffix."""
    component = (frame.get("payload") or {}).get("component") or {}
    drawn = drawn_series(component, frame.get("fact_summary") or {})
    title = component.get("title")
    cut = _CUT_TITLE.search(str(title.get("text") if isinstance(title, Mapping) else title or ""))
    return {float(len(drawn)), *(float(count) for count in (cut.groups() if cut else ()))}


def _matches(citation: _Citation, facts: Set[float]) -> bool:
    if citation.value * citation.scale in facts:
        return True
    return any(round(fact / citation.scale, citation.decimals) == citation.value for fact in facts)


def _grounded(
    citation: _Citation,
    corpus: Set[float],
    *,
    drawn: Set[float] = frozenset(),
    ends: Set[float] = frozenset(),
) -> bool:
    """Exact, or a rounding of a corpus fact at the decimals the narration
    wrote (54.462 grounds "54.5" and "54"; 12.6 grounds "13"; 7,640,000
    grounds "7.6 million"; 7,600,000 grounds "7.60 million" and 7,640,000
    does not). A drawn value grounds only a citation written with at least
    `_DRAWN_MIN_SIGNIFICANT_FIGURES` significant figures, zeros of a whole
    number and trailing decimal zeros included. One of `ends`, a series'
    first or last drawn value at three figures (`_drawn_ends`), grounds the
    citation that writes exactly that."""
    if _matches(citation, corpus):
        return True
    written = citation.value * citation.scale
    if any(math.isclose(written, end, rel_tol=1e-9) for end in ends):
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
        return _numbers_in_data(summary) | _drawn_periods(frame) | _card_counts(frame)
    if event == "activity.step":
        return _numbers_in_data([frame.get("label"), frame.get("detail")])
    if event == "plan.proposed":
        return _numbers_in_data([frame.get("title"), frame.get("steps"), frame.get("decisions")])
    if event == "error":
        return _numbers_in_data(frame.get("error"))
    if event == "elicitation.request":
        return _numbers_in_data([frame.get("question"), frame.get("options")])
    return set()


def check_grounding(
    turns: list[list[dict[str, Any]]], *, stated: Sequence[Set[float]] = ()
) -> list[str]:
    """Numbers in `message.complete` must exist in the turn's own fact
    frames or in an earlier turn's fact summaries (the canvas digest), or
    match a value a component draws. Facts compare by magnitude: narration
    says "an outflow of $223 million" for -223e6. `stated` adds, turn by
    turn, the figures the turn's cards gave its reply (`stated_values`)."""
    violations = []
    digest_corpus: set[float] = set()
    digest_drawn: set[float] = set()
    digest_ends: set[float] = set()
    for number, turn in enumerate(turns, start=1):
        corpus = set(digest_corpus)
        if number <= len(stated):
            corpus |= stated[number - 1]
        drawn = set(digest_drawn)
        ends = set(digest_ends)
        for frame in turn:
            corpus |= _fact_corpus(frame)
            drawn |= _drawn_values(frame)
            ends |= _drawn_ends(frame)
        for frame in turn:
            if frame.get("event") != "message.complete":
                continue
            ungrounded = [
                citation.written
                for citation in _numbers_in_text(frame.get("text", ""))
                if not _grounded(citation, corpus, drawn=drawn, ends=ends)
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
                digest_ends |= _drawn_ends(frame)
    return violations


# ---------------------------------------------------------------------------
# The closing message against the figures its cards supplied
# ---------------------------------------------------------------------------

# The rules `check_reply` reports a fault under.
REPLY_RULES = (
    "figure_missing",
    "number_in_no_frame",
    "value_outside_reply_figures",
    "figure_without_period",
    "series_outside_reply_figures",
)
# The rule of a finding that is no fault: a statement whose values the check
# cannot pair with one series and one year.
REPLY_UNDECIDED = "pairing_undecided"
# What the findings of a closing message come to (`reply_verdict`).
REPLY_VERDICTS = ("pass", "fail", "undecided")

# The points of a series a case may require a reply to state. `largest` and
# `smallest` are the marks of a card whose series share one period.
STATED_WHICH = ("first", "last", "at", "largest", "smallest")
_MARKS = ("largest", "smallest")
_REPLY_EXPECT_KEYS = frozenset({"states"})
_FIGURE_SPEC_KEYS = frozenset({"series", "which", "period"})

_MAGNITUDES = {"thousands": 1e3, "millions": 1e6, "billions": 1e9, "trillions": 1e12}
# A reply names a series of a card of more series than this only for a figure.
_NAMED_SERIES_CAP = 4

_LIST_ITEM = re.compile(r"\s*(?:[-*•]|\d+[.)])\s+")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_YEAR = re.compile(r"(?<![0-9])[12][0-9]{3}(?![0-9])")
# What tells a number written as a year from an amount (`_is_year`).
_CURRENCY_BEFORE = re.compile(r"[$€£¥]\s?$")
_NEXT_WORD = re.compile(r"\s+([A-Za-z]+)")
_YEAR_WORD_BEFORE = re.compile(r"\b(?:in|for|since|from|to|between)\s+$", re.IGNORECASE)
_RANGE_BEFORE = re.compile(r"[12][0-9]{3}\s*(?:-|\u2013|\u2014|to)\s*$")
_RANGE_AFTER = re.compile(r"\s*(?:-|\u2013|\u2014|to)\s*[12][0-9]{3}(?![0-9])")
# The words that make the number they come before a bound, not a value.
_BOUND = re.compile(
    r"\b(?:above|below|over|under|at least|more than|less than)\s+[^\w\s]?$", re.IGNORECASE
)


@dataclass(frozen=True)
class ReplyFault:
    """One finding of `check_reply`. A fault is a way the closing message
    departs from the figures supplied: the rule of `REPLY_RULES` it breaks,
    and what in the message breaks it. A finding under `REPLY_UNDECIDED` is
    no fault: its detail quotes a statement the check cannot decide."""

    rule: str
    detail: str

    @property
    def undecided(self) -> bool:
        return self.rule == REPLY_UNDECIDED

    def __str__(self) -> str:
        return f"reply [{self.rule}] {self.detail}"


@dataclass(frozen=True)
class ReplyCard:
    """One card a turn drew: every series it draws with every point, and
    `figures`, the values of it a reply may state, each shaped as an entry of
    a `make_charts` result's `reply_figures`."""

    title: str
    series: tuple[DrawnSeries, ...]
    figures: tuple[Mapping[str, Any], ...]
    # False when `figures` are every point drawn, for want of the ones given.
    listed: bool = True


def card_title(frame: Mapping[str, Any]) -> str:
    """The title a `component.created` frame's card is drawn under."""
    title = ((frame.get("payload") or {}).get("component") or {}).get("title")
    return str(title.get("text") if isinstance(title, Mapping) else title or "")


def reply_card(
    frame: Mapping[str, Any],
    *,
    returned: Mapping[str, Any] | None = None,
    on_canvas: Collection[str] = (),
    any_point: bool = False,
) -> ReplyCard:
    """The card a `component.created` frame drew, its series read from the
    drawn payload as `make_charts` reads them. Its figures are the first of:

    - the `reply_figures` of `returned`, the card's entry in the `make_charts`
      result, when the caller holds it;
    - every point the card draws, when `any_point`: nothing says which of
      them the model was given, so any may be stated;
    - what `harness.reply_figures.card_result` gives a card none of whose
      series the model named: the latest value of a lone series, the largest
      and smallest of one period, and nothing for a card whose every series
      is `on_canvas`, the ids drawn in earlier turns.
    """
    component = (frame.get("payload") or {}).get("component") or {}
    series = drawn_series(component, frame.get("fact_summary") or {})
    if returned is not None:
        figures = list(returned.get("reply_figures") or [])
    elif any_point:
        figures = [point_figure(one, point) for one in series for point in one.points]
    else:
        figures = card_result(series, on_canvas=on_canvas)["reply_figures"]
    return ReplyCard(
        title=card_title(frame),
        series=tuple(series),
        figures=tuple(figures),
        listed=returned is not None or not any_point,
    )


def reply_expect_problems(expect: Any) -> list[str]:
    """Why a turn's `reply` expectation is malformed. It is an object of
    `states`, a list of `{series, which, period?}` figures the closing message
    must state. `which` is
    one of `STATED_WHICH` and `period` goes with `at` alone. `series` names
    the series by its label or a coordinate; left out, any drawn series
    answers. A mark takes none, since a card has one largest and one smallest."""
    if not isinstance(expect, dict) or "states" not in expect:
        return ["expect.reply must be an object with a 'states' list"]
    problems = [
        f"expect.reply key {key!r} is not one the runner checks"
        for key in expect
        if key not in _REPLY_EXPECT_KEYS
    ]
    states = expect["states"]
    if not (isinstance(states, list) and all(isinstance(spec, dict) for spec in states)):
        return [*problems, "expect.reply.states must be a list of objects"]
    for spec in states:
        which = spec.get("which")
        if unknown := sorted(set(spec) - _FIGURE_SPEC_KEYS):
            problems.append(f"expect.reply.states key(s) {unknown} are not checked")
        if which not in STATED_WHICH:
            problems.append(f"expect.reply.states 'which' must be one of {list(STATED_WHICH)}")
        if "series" in spec and (
            which in _MARKS or not (isinstance(spec["series"], str) and spec["series"])
        ):
            problems.append(
                "expect.reply.states 'series' is a non-empty string, and no mark takes one"
            )
        if (which == "at") != ("period" in spec):
            problems.append("expect.reply.states 'period' is for 'at', and 'at' needs it")
    return problems


def _spec_text(spec: Mapping[str, Any]) -> str:
    return " ".join(str(spec[key]) for key in ("series", "which", "period") if key in spec)


def _series_names(series: DrawnSeries) -> set[str]:
    """What a case may call a series: its label or any of its coordinates."""
    return {name.casefold() for name in (series.label, *(series.coordinates or {}).values())}


def _required(spec: Mapping[str, Any], cards: Sequence[ReplyCard]) -> list[tuple[int, Any]]:
    """The figures that answer `spec`, each with the index of its card: the
    named point of every drawn series the spec's `series` names, of every
    drawn series when it names none, or the mark of every card that has one.
    A series' first and last are its ends by period (`series_ends`). A reply
    stating any of them states the figure."""
    found: list[tuple[int, Any]] = []
    which = spec["which"]
    for index, card in enumerate(cards):
        if which in _MARKS:
            marked = card_result(card.series)["reply_figures"]
            found += [(index, figure) for figure in marked if figure.get("mark") == which]
            continue
        for one in card.series:
            if "series" in spec and spec["series"].casefold() not in _series_names(one):
                continue
            ends = series_ends(one)
            points = {
                "first": ends[:1] if ends else [],
                "last": ends[1:] if ends else [],
                "at": [point for point in one.points if point[0] == spec.get("period")][:1],
            }[which]
            found += [(index, point_figure(one, point)) for point in points]
    return found


def _writes(citation: _Citation, value: float, *, figures: int) -> bool:
    """Whether `citation` is `value` as a reply writes it: whole, rounded to
    three significant figures, or to as many decimals as the citation writes
    when it writes `figures` figures or more."""
    written = citation.value * citation.scale
    if written == value or math.isclose(
        written, _round_significant(value, _REPLY_SIGNIFICANT_FIGURES), rel_tol=1e-9
    ):
        return True
    return (
        citation.figures >= figures
        and round(value / citation.scale, citation.decimals) == citation.value
    )


def _states_value(
    citation: _Citation, value: float, unit: Any, *, figures: int = _REPLY_SIGNIFICANT_FIGURES
) -> bool:
    """`_writes` on the magnitude of `value`, read as drawn and, under a unit
    of a magnitude ("millions"), at that scale too: 441.9 in millions is
    "441.9 million"."""
    scale = _magnitude(unit)
    return _writes(citation, abs(value), figures=figures) or (
        scale is not None and _writes(citation, abs(value) * scale, figures=figures)
    )


def _magnitude(unit: Any) -> float | None:
    """The scale a structured unit's magnitude puts on the values drawn in it."""
    return _MAGNITUDES.get(str(unit.get("magnitude"))) if isinstance(unit, Mapping) else None


def stated_values(
    cards: Sequence[ReplyCard], states: Sequence[Mapping[str, Any]] = (), *, message: str = ""
) -> set[float]:
    """The magnitudes a reply over `cards` may state, for `check_grounding`:
    the figures each card lists and those `states` requires, whole and as a
    reply rounds them. A card whose figures are every point it draws adds
    only the points `message` states beside their own series and year, since
    any number near a point would otherwise ground."""
    points = _read(message, cards, states)[1] if message else []
    return _stated_values(cards, states, points)


def _stated_values(
    cards: Sequence[ReplyCard],
    states: Sequence[Mapping[str, Any]],
    points: Sequence[Mapping[str, Any]],
) -> set[float]:
    figures = [figure for card in cards if card.listed for figure in card.figures]
    figures += [figure for spec in states for _, figure in _required(spec, cards)]
    values: set[float] = set()
    for figure in (*figures, *points):
        value = abs(float(figure["value"]))
        for scaled in {value, value * (_magnitude(figure.get("unit")) or 1.0)}:
            values |= {scaled, _round_significant(scaled, _REPLY_SIGNIFICANT_FIGURES)}
    return values


def _is_year(
    statement: str, start: int, citation: _Citation, cards: Sequence[ReplyCard]
) -> bool | None:
    """Whether the number of `statement` at `start` is a year rather than an
    amount written in four bare digits, such as 1500, or None where nothing
    decides.

    A number written as a year is one unless its form makes it an amount: a
    currency sign before it, a per cent sign or the first word of a drawn
    series' unit after it ("2500 tonnes"). Otherwise the cards decide: a
    number that is a year the cards draw a point at and no value they draw
    is a year, and one that is a drawn value and no drawn year is an amount.
    Where it is both or neither, its place decides: a year after "in",
    "for", "since", "from", "to" or "between", or within a range of years
    ("2000-2023"). Anywhere else it is undecided."""
    written = citation.written.rstrip(",")
    if _YEAR.fullmatch(written) is None:
        return False
    after = statement[start + len(written) :]
    units = {
        str(one.unit).split()[0].casefold()
        for card in cards
        for one in card.series
        if isinstance(one.unit, str) and one.unit.strip()
    }
    following = _NEXT_WORD.match(after)
    if (
        _CURRENCY_BEFORE.search(statement, 0, start)
        or after.startswith("%")
        or (following and following.group(1).casefold().removesuffix("s") in units)
    ):
        return False
    year = int(citation.value)
    period = any(
        year in _period_years(at) for card in cards for one in card.series for at, _ in one.points
    )
    value = _draws(cards, citation)
    if period != value:
        return period
    placed = (
        _YEAR_WORD_BEFORE.search(statement, 0, start)
        or _RANGE_BEFORE.search(statement, 0, start)
        or _RANGE_AFTER.match(after)
    )
    return True if placed else None


def _draws(cards: Sequence[ReplyCard], citation: _Citation) -> bool:
    """Whether `citation` writes a value some card draws."""
    return any(
        _states_value(citation, drawn, one.unit)
        for card in cards
        for one in card.series
        for _, drawn in one.points
    )


def _period_years(period: Any) -> set[int]:
    return {int(year) for year in _YEAR.findall(str(period))} if period is not None else set()


def _statements(message: str) -> Iterator[str]:
    """The sentences of a closing message, a list item or a heading each
    its own, without its list marker."""
    for line in message.splitlines():
        item = _LIST_ITEM.match(line)
        yield from filter(None, _SENTENCE_END.split(line[item.end() :] if item else line.strip()))


def _mentions(text: str, labels: Collection[str]) -> list[tuple[int, str]]:
    """Where `text` names each of `labels` (casefolded), as whole words, the
    longer label winning where two overlap ("South Sudan" over "Sudan")."""
    taken: list[tuple[int, int, str]] = []
    for label in sorted(labels, key=len, reverse=True):
        pattern = rf"(?<!\w){re.escape(label)}(?!\w)"
        for match in re.finditer(pattern, text, re.IGNORECASE):
            if all(match.end() <= start or end <= match.start() for start, end, _ in taken):
                taken.append((match.start(), match.end(), label))
    return [(start, label) for start, _, label in sorted(taken)]


def _beside[T](
    at: int, anchors: Sequence[tuple[int, T]], breaks: Collection[int], *, direct: bool = False
) -> set[T]:
    """The anchors a value at `at` sits beside: on each side, the nearest run
    of anchors with none of `breaks`, the positions of the statement's other
    values, inside it. "68.5 years in Uganda and 63.8 years in Kenya" puts
    63.8 beside both names and 68.5 beside Uganda alone. A side whose
    nearest run lies past another value still gives that run ("from 56.1 to
    63.6 in Kenya" puts 56.1 beside Kenya), unless `direct`."""
    marks = sorted(
        [(position, anchor) for position, anchor in anchors] + [(b, None) for b in breaks],
        key=lambda mark: mark[0],
    )
    found: set[T] = set()
    for side in (reversed([m for m in marks if m[0] < at]), (m for m in marks if m[0] > at)):
        run: list[T] = []
        for _, anchor in side:
            if anchor is not None:
                run.append(anchor)
            elif run or direct:
                break
        found.update(run)
    return found


def _blocks[T](
    values: Sequence[int], anchors: Sequence[tuple[int, T]], before: Sequence[T] = ()
) -> list[tuple[list[T], list[int]]]:
    """A statement's values in blocks, each the values that follow one
    another with no anchor between them, with the run of anchors directly
    before it. `before` is the run the statement opens under."""
    marks: list[tuple[int, T | None]] = sorted(
        [*anchors, *((at, None) for at in values)], key=lambda mark: mark[0]
    )
    blocks: list[tuple[list[T], list[int]]] = []
    run = list(before)
    block: list[int] | None = None
    for at, anchor in marks:
        if anchor is None:
            if block is None:
                block = []
                blocks.append((run, block))
            block.append(at)
        else:
            run = [anchor] if block is not None else [*run, anchor]
            block = None
    return blocks


def _paired[T](
    run: Sequence[T], block: Sequence[int], fits: Callable[[int, T], bool], *, shared: Sequence[T]
) -> list[set[T]] | None:
    """The anchors each value of `block` takes from `run`, the several anchors
    written before it, or None where nothing decides. Where the run divides
    the block, names and then values pair in order, the run repeating
    ("Uganda and Kenya stood at 68.5 and 63.8"), or each name takes its
    values in turn ("Kenya and Uganda rose from 60.1 to 63.8 and from 58.0 to
    68.5"). Or every value of the block takes one anchor of `shared`
    ("compared with Kenya, South Sudan rose from 47.6 to 57.6"). The block
    is decided when `fits` accepts one of those readings for every value,
    and a value takes its anchor under each reading that fits. When none
    fits and the run divides the block, the values are held to their order.
    When none fits and the run does not divide the block, nothing decides."""
    size, repeats = len(block), len(block) // len(run)
    readings = [[anchor] * size for anchor in shared]
    if size % len(run) == 0:
        readings[:0] = [
            [run[index % len(run)] for index in range(size)],
            [run[index // repeats] for index in range(size)],
        ]
    fitting = [
        reading
        for reading in readings
        if all(fits(at, anchor) for at, anchor in zip(block, reading, strict=True))
    ]
    if fitting:
        return [set(anchors) for anchors in zip(*fitting, strict=True)]
    return None if size % len(run) else [{anchor} for anchor in readings[0]]


@dataclass(frozen=True)
class _Stated:
    """A number of a statement that writes a value of a card: where it
    starts, the figures the reply may state that it writes, and the labels
    of the series that draw it. `rough` marks a number of two figures, which
    states no figure and is still held to the series and year beside it."""

    start: int
    citation: _Citation
    figures: list[tuple[int, Any]]
    drawn: list[str]
    rough: bool = False


def _stated(
    statement: str,
    cards: Sequence[ReplyCard],
    allowed: Sequence[tuple[int, Any]],
    named: Collection[str],
    years: Collection[int],
) -> list[_Stated]:
    """The values `statement` states. A number of two figures is one when it
    rounds a figure of a series among `named`, the series the statement is
    about. One that rounds only figures of other series comes back with no
    figure and those series as `drawn`: it may be a value or a count ("the
    24 largest"), and nothing decides. One written as a bound ("all above
    60") states no series' value."""
    values = []
    for start, citation, structural in _cited(statement):
        if start in years:
            continue
        figures = [
            entry
            for entry in allowed
            if _states_value(citation, entry[1]["value"], entry[1].get("unit"))
        ]
        drawn = sorted(
            {
                one.label
                for card in cards
                for one in card.series
                if any(_states_value(citation, value, one.unit) for _, value in one.points)
            }
        )
        rough = (
            not (figures or drawn or structural)
            and citation.figures == 2
            and not _BOUND.search(statement, 0, start)
        )
        if rough:
            rounded = [
                entry
                for entry in allowed
                if _states_value(citation, entry[1]["value"], entry[1].get("unit"), figures=2)
            ]
            figures = [entry for entry in rounded if entry[1]["series"].casefold() in named]
            drawn = sorted({figure["series"] for _, figure in rounded})
        if figures or drawn:
            values.append(_Stated(start, citation, figures, drawn, rough))
    return values


def _read(
    message: str, cards: Sequence[ReplyCard], states: Sequence[Mapping[str, Any]]
) -> tuple[list[ReplyFault], list[Mapping[str, Any]]]:
    """Every fault of `check_reply` but `number_in_no_frame`, and the points
    of the cards whose figures are every point they draw that `message`
    states beside their own series and year."""
    faults: list[ReplyFault] = []

    def key(entry: tuple[int, Any]) -> tuple[Any, ...]:
        index, figure = entry
        return index, figure["series"], figure["value"], figure.get("period")

    allowed = [(index, figure) for index, card in enumerate(cards) for figure in card.figures]
    listed = {key(entry) for entry in allowed}
    required = []
    for spec in states:
        found = _required(spec, cards)
        if not found:
            faults.append(
                ReplyFault("figure_missing", f"no card of the turn draws {_spec_text(spec)!r}")
            )
        required.append((spec, found))
        for entry in found:
            if key(entry) not in listed:
                listed.add(key(entry))
                allowed.append(entry)
    labels = {one.label.casefold() for card in cards for one in card.series if one.label}
    drawn_years = {
        year
        for card in cards
        for one in card.series
        for period, _ in one.points
        for year in _period_years(period)
    }

    stated: dict[tuple[Any, ...], set[str]] = {}
    maybe_stated: set[tuple[Any, ...]] = set()
    points: list[Mapping[str, Any]] = []
    statements = list(_statements(message))
    named_anywhere = {label for text in statements for _, label in _mentions(text, labels)}
    earlier_years: list[int] = []
    awaiting: str | None = None
    for statement in statements:
        named = _mentions(statement, labels)
        subject = None if named else awaiting
        read = [
            (start, citation, _is_year(statement, start, citation, cards))
            for start, citation, _ in _cited(statement)
        ]
        years = [(start, int(citation.value)) for start, citation, year in read if year]
        unsure = [citation for _, citation, year in read if year is None]
        values = _stated(
            statement,
            cards,
            allowed,
            {label for _, label in named} | {subject or ""},
            {start for start, _, year in read if year is not False},
        )
        by_start = {value.start: value for value in values}
        starts = list(by_start)
        # What of the statement nothing decides: the number, and why.
        open_readings: list[str] = []
        if values or any(_draws(cards, citation) for citation in unsure):
            open_readings += [
                f"{citation.written} may be a year or an amount" for citation in unsure
            ]

        def of_series(at: int, label: str, by_start: dict[int, _Stated] = by_start) -> bool:
            figures = by_start[at].figures
            return not figures or any(f["series"].casefold() == label for _, f in figures)

        held = {starts[0]: {subject}} if subject and starts else {}
        skipped: set[int] = set()
        for run, block in _blocks(starts, named):
            if len(run) < 2 or len(block) < 2:
                continue
            reading = _paired(run, block, of_series, shared=run)
            if reading is None:
                skipped.update(block)
                written = ", ".join(by_start[at].citation.written for at in block)
                open_readings.append(
                    f"{written} follow {', '.join(run)}, and no one way of pairing "
                    f"{len(block)} values with {len(run)} series fits"
                )
            else:
                held.update(zip(block, reading, strict=True))

        own_of: dict[int, list[tuple[int, Any]]] = {}
        beside_of: dict[int, set[str]] = {}
        for value in values:
            if value.start in skipped:
                continue
            written = value.citation.written
            others = [other for other in starts if other != value.start]
            if value.rough and not value.figures:
                open_readings.append(
                    f"{written} rounds a figure of {', '.join(value.drawn)}, which the "
                    "statement does not name, and may be a count"
                )
                continue
            if not value.figures:
                faults.append(
                    ReplyFault(
                        "value_outside_reply_figures",
                        f"{written} is a value drawn for "
                        f"{', '.join(value.drawn)} that its card's reply_figures do not list",
                    )
                )
                continue
            beside = (
                held[value.start] if value.start in held else _beside(value.start, named, others)
            )
            own = [e for e in value.figures if e[1]["series"].casefold() in beside]
            of = sorted({figure["series"] for _, figure in value.figures})
            reached = value.start in held or bool(_beside(value.start, named, others, direct=True))
            if beside and not own and not reached:
                open_readings.append(
                    f"{written} is a figure of {', '.join(of)}, and the nearest series "
                    f"named, {', '.join(sorted(beside))}, sits past another value"
                )
                continue
            if (
                beside
                and not own
                and (
                    any(cards[index].listed for index, _ in value.figures)
                    or any(series.casefold() in named_anywhere for series in of)
                )
            ):
                faults.append(
                    ReplyFault(
                        "value_outside_reply_figures",
                        f"{written} is stated beside "
                        f"{', '.join(sorted(beside))} and is a figure of {', '.join(of)}",
                    )
                )
                continue
            if not beside and len(labels) > 1:
                open_readings.append(f"{written} sits in a statement that names no series")
                continue
            if len(beside) > 1 and value.start not in held:
                open_readings.append(
                    f"{written} sits beside {', '.join(sorted(beside))}, and nothing says which"
                )
                continue
            if beside and not own:
                open_readings.append(
                    f"{written} sits beside {', '.join(sorted(beside))} and is a point of "
                    f"{', '.join(of)}, which the message spells out nowhere"
                )
                continue
            beside_of[value.start] = beside
            own_of[value.start] = own or value.figures

        def at_year(at: int, year: int, own_of: dict[int, list[Any]] = own_of) -> bool:
            return at not in own_of or any(
                not (period := _period_years(figure.get("period"))) or year in period
                for _, figure in own_of[at]
            )

        # A year no card draws a point at dates no value: the newer year a
        # period note names beside the year served, for one.
        carried = [year for year in dict.fromkeys(earlier_years) if year in drawn_years]
        year_of: dict[int, set[int]] = {}
        for written_years, block in _blocks(starts, years, () if years else carried):
            run = [year for year in dict.fromkeys(written_years) if year in drawn_years]
            dated_here = [at for at in block if at in own_of]
            if len(run) < 2 or not dated_here or (years and len(block) < 2):
                continue
            reading = _paired(run, block, at_year, shared=run[-1:])
            if reading is not None:
                year_of.update(zip(block, reading, strict=True))
            elif not years:
                for at in dated_here:
                    del own_of[at]
                cited = ", ".join(by_start[at].citation.written for at in dated_here)
                open_readings.append(
                    f"{cited} follow a statement of {', '.join(map(str, run))}, and no one "
                    "way of pairing them with those years fits"
                )

        for start, own in own_of.items():
            value = by_start[start]
            if start in year_of:
                near = year_of[start]
            elif years:
                near = _beside(start, years, [other for other in starts if other != start])
            else:
                near = set(earlier_years)
            dated = [
                entry
                for entry in own
                if not (period := _period_years(entry[1].get("period"))) or period & near
            ]
            if dated and len(near & drawn_years) > 1 and start not in year_of:
                open_readings.append(
                    f"{value.citation.written} sits beside "
                    f"{', '.join(str(year) for year in sorted(near))}, and nothing says which"
                )
                continue
            others = [other for other in starts if other != start]
            if (
                not dated
                and len({year for _, year in years}) > 1
                and start not in year_of
                and not _beside(start, years, others, direct=True)
            ):
                _, figure = own[0]
                open_readings.append(
                    f"{value.citation.written} ({figure['series']}, {figure['period']}) is "
                    f"nearest {', '.join(str(year) for year in sorted(near))}, which sits "
                    "past another value"
                )
                continue
            if not dated and years:
                _, figure = own[0]
                faults.append(
                    ReplyFault(
                        "figure_without_period",
                        f"{value.citation.written} ({figure['series']}, {figure['period']}) "
                        f"is stated beside {', '.join(str(year) for year in sorted(near))}",
                    )
                )
            if value.rough:
                continue
            for entry in dated or own:
                stated.setdefault(key(entry), set()).add(
                    "dated" if dated else "misdated" if years else "undated"
                )
            points += [
                figure
                for index, figure in dated
                if not cards[index].listed
                and figure["series"].casefold() in beside_of[start]
                and _period_years(figure.get("period")) & near
            ]
        if open_readings:
            faults.append(ReplyFault(REPLY_UNDECIDED, f"{statement!r}: {'; '.join(open_readings)}"))
            # A figure an undecided statement writes may be stated there.
            maybe_stated.update(
                key(entry) for value in values if not value.rough for entry in value.figures
            )
        earlier_years = [year for _, year in years] or earlier_years
        awaiting = named[0][1] if len({label for _, label in named}) == 1 and not values else None

    for entry in allowed:
        if stated.get(key(entry)) == {"undated"}:
            _, figure = entry
            faults.append(
                ReplyFault(
                    "figure_without_period",
                    f"{figure['series']}'s {figure['value']} is stated without its period "
                    f"{figure['period']}",
                )
            )
    faults += [
        ReplyFault("figure_missing", f"the message does not state {_spec_text(spec)!r}")
        for spec, found in required
        if found and not any(key(entry) in stated or key(entry) in maybe_stated for entry in found)
    ]
    for index, card in enumerate(cards):
        if len(card.series) <= _NAMED_SERIES_CAP:
            continue
        may = {figure["series"].casefold() for at, figure in allowed if at == index}
        outside = [one.label for one in card.series if one.label.casefold() in named_anywhere - may]
        if outside:
            faults.append(
                ReplyFault(
                    "series_outside_reply_figures",
                    f"the message names {len(outside)} of the {len(card.series)} series of "
                    f"{card.title!r} that it may state no figure of: {', '.join(outside)}",
                )
            )
    return faults, points


def check_reply(
    message: str,
    cards: Sequence[ReplyCard],
    *,
    states: Sequence[Mapping[str, Any]] = (),
    turns: Sequence[list[dict[str, Any]]] | None = None,
) -> list[ReplyFault]:
    """A turn's closing `message` against the figures its `cards` supplied:
    its findings, which `reply_verdict` sums up.

    A reply may state the `figures` of its cards and the figures `states`
    lists (valid per `reply_expect_problems`, each a point of a drawn series),
    which it must state. A number states a figure when it writes the figure's
    value as a reply rounds it. The faults, by rule:

    - `figure_missing`: a figure of `states` the message does not state.
    - `value_outside_reply_figures`: a value a card draws that is no figure of
      it, or a figure stated beside another series' name.
    - `figure_without_period`: a figure stated beside a year that is not its
      period's, or stated nowhere in the message with its period.
    - `pairing_undecided`, a finding and no fault: a sentence, quoted, with a
      value the readings below pair with no one series or no one year. Such
      a value never passes by default. A figure of `states` that it writes
      is not reported missing, since it may be stated there.
    - `series_outside_reply_figures`: a card of more than four series, one of
      which the message names though no figure of it may be stated.
    - `number_in_no_frame`: with `turns`, the conversation's frames by turn
      up to this one, a number `check_grounding` finds in no fact frame and
      among no figure the cards supplied (`stated_values`).

    A series is recognised by its label alone. The readings that decide:

    - Several names and then a multiple of as many values pair in order, or
      each name takes its values in turn, or the values are all one named
      series' (`_paired`). Several years and then values pair the same way.
    - Any other value is of the one series named beside it with no other
      value between (`_beside`), or of the one series the cards draw. It is
      at the one year beside it, or, in a sentence that names no year, at
      the year of the last sentence before it that names one, as a list item
      is under the line that introduces its list. A year no card draws a
      point at dates no value.
    - A sentence that names one series and states no value gives that name
      to the first value of the next sentence when that names none ("Uganda
      leads. It stood at 68.5 years.").
    - A number of two figures that rounds a figure of a series its sentence
      names is held to the series and year beside it like any other value,
      and states no figure, unless it is written as a bound ("above 60").
    - A four-digit number is a year or an amount by `_is_year`.

    Left undecided: a value beside several names or several years, a value
    in a sentence that names no series on a card of several, a value beside
    a label that is not its own where its own is spelt out nowhere or the
    label sits past another value, values under several carried years they
    do not pair with, a two-figure number that rounds a figure of a series
    the sentence does not name, and a four-digit number `_is_year` cannot
    place.
    """
    faults, points = _read(message, cards, states)
    if turns:
        here = f"turn {len(turns)}: "
        supplied = [*([set()] * (len(turns) - 1)), _stated_values(cards, states, points)]
        faults += [
            ReplyFault("number_in_no_frame", violation.removeprefix(here))
            for violation in check_grounding(list(turns), stated=supplied)
            if violation.startswith(here)
        ]
    return faults


def reply_verdict(findings: Iterable[ReplyFault]) -> str:
    """What the findings of one or more closing messages come to: `fail`
    when any is a fault, `undecided` when none is and a statement is
    undecided, else `pass`."""
    undecided = {finding.undecided for finding in findings}
    return "fail" if False in undecided else "undecided" if undecided else "pass"


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
    frames: list[dict[str, Any]],
    *,
    complete_conversation: bool = False,
    stated: Sequence[Set[float]] = (),
) -> list[str]:
    """All checks over one conversation's frames. Empty list = conformant.
    `stated` is `check_grounding`'s.

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
    violations += check_grounding(turns, stated=stated)
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
    `translation` a `propose_plan` answer carries, read off the response as
    `harness.planner_seam` reads it."""
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

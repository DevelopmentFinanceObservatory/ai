"""One planner run's bookkeeping, apart from the framework that drives the model.

A `PlannerRun` is told what each tool answered (`PlannerRun.tool_answered`),
which tool raised (`PlannerRun.tool_failed`), and which turn-budget bound
closed a model call (`PlannerRun.model_answered`). Before each model request
it says whether the run asks the user, stops, or leaves the request to the
model (`PlannerRun.next_request`). The answer is a `Directive`: `Ask` with the
`PLANNER_ASK` call's arguments, or `Stop` with the refusal the run ends on.
`harness.planner_seam` drives one through the ADK planner's callbacks and
`harness.planner_node` through the Pydantic AI planner's model hooks, and
`harness.planner_node.directive_turn` builds the model turn each directive
stands for.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from executor.classify import internal_fault
from planner.plan import discovery_time_left, truncate
from planner.tool_results import Outcome
from translator.referral import USER_DECIDES

from harness.sentences import TERMINAL_PROPOSE_CODES, SeamCode, referral_sentence

# What `_stalled_referral` counts a referral by: its kind, requirement,
# constraint, and the (name, role) of each entity it leaves unsettled.
type StallKey = tuple[str, str | None, str | None, tuple[tuple[str, str | None], ...]]

# Insurance against a runaway tool loop inside one agent run. A healthy plan
# takes well under this many tool calls (skill reads and proposals).
_MAX_EVENTS = 60

# A referral candidate's `label` is the resolve's own `rendered_sentence`,
# which can run to a paragraph. Capped where `_pinnable_candidates` shapes
# it for a question to the user.
_CANDIDATE_LABEL_CAP = 120

# What the question a request's closing budget asks opens with.
_BUDGET_CLOSED_LEAD = "This request reached its limit before the plan settled on a source."

# What the question a referral's candidates put to the user asks: which one
# they meant, or, when the source they named holds none of the data, which
# other source to use.
_WHICH_MEANT = "Which of these did you mean?"
_WHICH_INSTEAD = "Which of these should be used instead?"


class PlannerSeamError(RuntimeError):
    """The planner produced no plan ready for the user's answer.

    Raised so that a plan the user approves is a plan
    `planner.tools.declare_ready` declared ready.

    `refusal` is the shape `harness.studio_translate._stream_fatal` reads
    (`code`, `retry_mode`), so the user reads the sentence of its code. The
    default code is `internal_error`, because the seam stopping on its own
    account is a fault of the system, whatever the question. The wire
    grammar has no member for it, so the frame states
    `capability_unavailable`. Every way this seam fails on its own account
    replays the same way against the same state, so the default is
    `do_not_retry`. A caller told to retry a deterministic failure pays for
    the same stall again. `fix.detail` holds the message for the log.
    """

    def __init__(self, message: str, *, refusal: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.refusal = refusal or internal_fault(message)


@dataclass(frozen=True, slots=True)
class Ask:
    """Answer the next model request with one `PLANNER_ASK` call taking `args`."""

    args: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Stop:
    """Answer the next model request with the end of the run on `refusal`."""

    refusal: dict[str, Any]


type Directive = Ask | Stop


class PlannerRun:
    """What one planner run has settled so far, and how its next request is answered.

    One per run of the planner (`harness.planner_seam.PlannerSeam._propose`),
    held for that run alone. The resume of an entry by the user's answer to
    an ask is a run of its own, so it starts with no stall counts, a fresh
    event cap and no last referral. Its brief shows the referral the asking
    run saw last (`harness.planner_seam._answer_text`).

    `submitted` is the answer of the call that declared the plan ready
    (`planner.tools.declare_ready`), `refused` the refusal the run ends on,
    and `last_referral` the referral the last translation with an `outcome`
    answered, or `None` once one of those answers something other than a
    referral. An entry that ends with no plan ready carries it (`not_ready`).
    `ready_to_break` says the run stops before its next generation, on
    `refused`.

    `on_progress`, when given, receives one report per tool answer, naming
    `agent` and the tool. `on_tool_answered`, when given, receives each
    tool's name and the answer the model reads (`tool_answered`).
    """

    def __init__(
        self,
        *,
        agent: str,
        on_progress: Callable[[dict[str, Any]], None] | None = None,
        on_tool_answered: Callable[[str, dict[str, Any] | None], None] | None = None,
    ) -> None:
        self._agent = agent
        self._on_progress = on_progress
        self._on_tool_answered = on_tool_answered
        self.submitted: dict[str, Any] | None = None
        self.refused: dict[str, Any] | None = None
        self.ready_to_break = False
        self.last_referral: dict[str, Any] | None = None
        self._referral_counts: dict[StallKey, int] = {}
        # The `PLANNER_ASK` call args the next generation is replaced with,
        # for a referral the user decides (`_user_decided_ask`).
        self._ask: dict[str, Any] | None = None
        self._events = 0
        # The clock a progress report's `duration_ms` measures against. It
        # resets each time a report fires, so a report covers everything
        # since the last one, including the thinking the model call that
        # produced this one spent before emitting a function call. Nothing
        # here sees a call start, so it cannot time one call alone.
        self._last_progress = time.monotonic()

    def tool_answered(self, name: str, answer: dict[str, Any] | None) -> None:
        """Record what tool `name` answered: progress, then what the answer settles.

        A terminal refusal (`_is_terminal_refusal`) ends the turn, a
        `translation` is the verdict on the plan the call recorded, and
        `summary` says the call declared that plan ready
        (`planner.tools.declare_ready`).

        The run's `on_tool_answered` observer is told the name and answer before
        any of that, on every answer within `_MAX_EVENTS`: a tool's own, a
        refusal of its arguments, and `tool_failed`'s for a tool that raised.
        What it raises propagates, as what `on_progress` raises does.

        Raises:
            PlannerSeamError: On the answer past `_MAX_EVENTS` in one run.
        """
        self._events += 1
        if self._events > _MAX_EVENTS:
            raise PlannerSeamError(
                f"the planner seam ran past {_MAX_EVENTS} events without settling"
            )
        if self._on_tool_answered is not None:
            self._on_tool_answered(name, answer)
        if self._on_progress is not None:
            now = time.monotonic()
            self._on_progress(
                {
                    "agent": self._agent,
                    "tools": [name],
                    "duration_ms": int((now - self._last_progress) * 1000),
                }
            )
            self._last_progress = now
        if _is_terminal_refusal(answer):
            # The proposal refused with a code that records nothing and
            # leaves nothing correctable within this entry: the round cap
            # (`planner.plan.MAX_PLAN_ROUNDS`, `BUDGET_EXHAUSTED`), or a
            # restart while the graph holds the question fixed
            # (`QUESTION_FIXED`, `planner.agent`'s `question_fixed` mode, which
            # the seam's own planner always uses, see
            # `harness.planner_seam.PlannerSeam.__init__`). The cap bounds the
            # proposals within one entry into the planner, since
            # `PlannerSeam.__call__` resets the count on each entry. A restart
            # is refused however fresh that budget is. `NOTHING_TO_PLAN`
            # declines every ask, and `PRICES_UNSERVED` needs a search the
            # planner cannot make (`submission.prices_choice`). Stop the
            # planner's turn here in every case.
            self.ready_to_break = True
            self.refused = answer
        elif isinstance(answer, dict):
            translation = answer.get("translation")
            if isinstance(translation, dict):
                self._track_translation(translation)
            if "summary" in answer:
                self.submitted = answer

    def tool_failed(self, name: str, error: BaseException) -> dict[str, Any]:
        """End the run on `SeamCode.PLANNER_TOOL_FAILED` for tool `name`, which
        raised `error`, unless a refusal already ends it, and answer what the
        model reads as that call's response (`failed_tool_answer`).

        The answer carries no `outcome`, so `tool_answered` reads it as neither
        a terminal refusal nor a translation. The model never reads it, since
        `next_request` stops the run before the next generation.
        """
        if self.refused is None:
            self.ready_to_break = True
            self.refused = {"code": SeamCode.PLANNER_TOOL_FAILED, "detail": {"tool": name}}
        return failed_tool_answer(name, error)

    def model_answered(self, closed_by: str | None) -> Directive | None:
        """Answer a model response whose call the turn budget's bound
        `closed_by` closed, or `None` for a call no bound closed.

        A call the budget closed is answered with a question to the user when
        the last referral names sources to choose from. The question is the
        same `PLANNER_ASK` call the planner's model makes
        (`ask_user_question`, `planner.specs`), built here from the referral's
        candidates (`_pinnable_candidates`), since no model call is left to
        compose it, and its tool accepts it as any other, ending the run on
        it. The answer arrives in a request of its own, with a fresh budget.
        With nothing pinnable (no referral, or a `no_data` referral, whose
        nearest-series candidates carry no `spec_id`) the closing text stands
        and this answers `None`. The run refuses on the bound, as the
        `SeamCode` `closed_by` names, and ends on it.
        """
        if closed_by is None:
            return None
        candidates = _pinnable_candidates(self.last_referral)
        if not candidates:
            self.ready_to_break = True
            self.refused = {"code": SeamCode(closed_by), "detail": {}}
            return None
        return Ask(_clarifying_question_args(candidates, lead=_BUDGET_CLOSED_LEAD))

    def next_request(self) -> Directive | None:
        """What replaces the next model request, or `None` to send it.

        A referral the user decides (`_user_decided_ask`) is answered first,
        with the `PLANNER_ASK` call it asks, which ends the run on the
        question in place of a model turn composing it. Once the run is ready
        to break it stops on the refusal it recorded.
        """
        if self._ask is not None:
            ask, self._ask = self._ask, None
            return Ask(ask)
        if not self.ready_to_break:
            return None
        assert self.refused is not None, "every break sets the refusal it ends on"
        return Stop(self.refused)

    def _track_translation(self, translation: dict[str, Any]) -> None:
        """Read one translation (the `translation` of an accepted `propose_plan`)
        for the referral it answered and whether it ends the entry.

        `last_referral` is written from every translation that carries an
        `outcome`, referral or not. Translation runs the whole plan, so each
        outcome-bearing verdict states what is open across it right now, and
        replaces whatever the last one recorded. `ok` means every requirement
        compiled, which clears a referral the same way. A response with no
        `outcome` settles nothing and is left alone. An entry that ends with no
        plan ready carries `last_referral` whichever way it landed
        (`not_ready`).

        An internal, repeated, budget-spent or never-issued failure, or a
        transient one with no discovery time left (`closing_refusal`), ends
        the entry on its own refusal, since the next translation would stop
        where this one did. A referral that recurs with nothing new
        (`_stalled_referral`) ends it on `SeamCode.PLAN_NOT_READY`, carrying
        that referral. Either way the planner's turn stops before its next
        generation. A referral the user decides, with candidates to pick from
        (`_user_decided_ask`), replaces that next generation with the
        question it asks.
        """
        if "outcome" in translation:
            self.last_referral = referral_of(translation)
        closing = closing_refusal(self.last_referral)
        if closing is not None:
            self.ready_to_break = True
            self.refused = closing
            return
        ask = _user_decided_ask(referral_of(translation))
        if ask is not None:
            self._ask = ask
            return
        if _stalled_referral(translation, self._referral_counts):
            self.ready_to_break = True
            self.refused = not_ready(self.last_referral)


def failed_tool_answer(name: str, error: BaseException) -> dict[str, Any]:
    """The function response for tool `name`, which raised `error`."""
    return {"error": f"{name} failed inside DFO ({type(error).__name__}). Do not repeat this call."}


def not_ready(referral: Mapping[str, Any] | None) -> dict[str, Any]:
    """The refusal for an entry that ended with no plan ready, carrying `referral` if any."""
    return {
        "code": SeamCode.PLAN_NOT_READY,
        "detail": _referral_detail(referral) if referral is not None else {},
    }


def referral_of(response: dict[str, Any] | None) -> dict[str, Any] | None:
    """The referral a translation answers with, or `None` for any other answer.

    `_stalled_referral` and `PlannerRun`'s `last_referral` bookkeeping both need
    this check, kept in one function so the two cannot drift apart. A referral
    is the one answer that names a `kind`, and `ok` and a problem name none.
    """
    if not isinstance(response, dict) or "kind" not in response:
        return None
    return response


def closing_refusal(referral: dict[str, Any] | None) -> dict[str, Any] | None:
    """The refusal a referral ends the planner entry on, or `None` when the
    planner may still settle it in this entry.

    An `internal` referral ends it with its reference. One marked
    `budget_spent` ends it as the data service unavailable, whatever else
    it carries. One marked `not_issued`, for a call the request's
    wall-clock bound left no time to make, ends it as that bound spent.
    Any other `transient` referral once no discovery time is left
    (`planner.plan.discovery_time_left`) ends it as the data service
    unavailable. The call was made and did not answer, and no discovery
    can run again in this request. Each of these carries the requirement
    the referral is about and its reason. One marked `repeated` ends it
    with what it is about, and the compile issue a
    `requirement_needs_repair` one names as its `constraint`.
    """
    if referral is None:
        return None
    if referral.get("outcome") == Outcome.INTERNAL.value:
        return {
            "code": SeamCode.TRANSLATION_INTERNAL_FAILURE,
            "detail": {"reference": str(referral.get("reference"))},
        }
    requirement_id = referral.get("requirement_id")
    reason = referral.get("reason")
    if referral.get("not_issued") and not referral.get("budget_spent"):
        return {
            "code": SeamCode.TURN_WALL_TIME_SPENT,
            "detail": {"requirement_id": requirement_id, "reason": reason},
        }
    time_left = discovery_time_left()
    out_of_time = (
        referral.get("outcome") == Outcome.TRANSIENT.value
        and time_left is not None
        and time_left <= 0
    )
    if referral.get("budget_spent") or out_of_time:
        return {
            "code": SeamCode.DISCOVERY_UNAVAILABLE,
            "detail": {"requirement_id": requirement_id, "reason": reason},
        }
    if referral.get("repeated"):
        return {
            "code": SeamCode.TRANSLATION_REPEATED,
            "detail": _referral_detail(referral),
        }
    return None


def _is_terminal_refusal(tool_response: dict[str, Any] | None) -> bool:
    """Whether a tool response refuses with one of
    `harness.sentences.TERMINAL_PROPOSE_CODES`, which only a proposal (`propose_plan`)
    answers: an outcome that records no new version and ends the entry.
    """
    return (
        isinstance(tool_response, dict)
        and tool_response.get("outcome") == "refused"
        and tool_response.get("code") in TERMINAL_PROPOSE_CODES
    )


def _stalled_referral(response: dict[str, Any] | None, counts: dict[StallKey, int]) -> bool:
    """Count one referral by its kind, its requirement, the compile issue it
    names as its `constraint` and the entities its `ambiguous_entities` leave
    unsettled, and answer whether that exact referral has now come back a
    second time, which stops the run. Two different repairs on one requirement
    count apart, so a planner that fixes one compile issue per round keeps
    going, and an `entity_name_ambiguous` referral covering fewer entities
    counts apart, so a planner that settles one end of "aid from Congo to
    Congo" per round keeps going too. A third repetition would carry nothing a
    model could act on that the second one did not already say. A referral
    marked `repeated` ends the entry instead (`closing_refusal`). A transient
    referral is never counted. The same plan is worth proposing again.
    """
    referral = referral_of(response)
    if referral is None or referral.get("outcome") == Outcome.TRANSIENT.value:
        return False
    kind = referral.get("kind")
    if kind is None:
        return False
    unsettled = tuple(
        (reading["entity"]["name"], reading["entity"].get("role"))
        for reading in referral.get("ambiguous_entities") or ()
    )
    key = (kind, referral.get("requirement_id"), referral.get("constraint"), unsettled)
    counts[key] = counts.get(key, 0) + 1
    return counts[key] >= 2


def _referral_detail(referral: Mapping[str, Any]) -> dict[str, Any]:
    """What a refusal ending on `referral` carries for its sentence and its reader."""
    return {key: referral.get(key) for key in ("kind", "requirement_id", "reason", "constraint")}


def _pinnable_candidates(referral: dict[str, Any] | None) -> list[dict[str, str]]:
    """The `spec_id`-bearing candidates a referral names, shaped as the
    `candidates` argument of `ask_user_question` (`planner.specs`), each
    `{"spec_id": ..., "label": ...}` with `label` the candidate's `sentence`.

    A `no_data` referral's candidates (`translator.discovery.compact_source
    (selectable=False)`) carry no `spec_id`, since they describe what search
    found nearby rather than a source pinnable without a fresh search, so
    they never appear here. `referral` is `None`, or carries no candidates,
    for a translation this entry never saw, or one whose kind is
    not among `translator.referral.MAY_CARRY_CANDIDATES`.
    """
    if not isinstance(referral, dict):
        return []
    if referral.get("kind") == "source_name_ambiguous":
        return [
            {"spec_id": str(candidate["ref"]), "label": str(candidate["label"])}
            for group in referral.get("ambiguous_sources") or []
            for candidate in group.get("candidates") or []
        ]
    candidates = referral.get("candidates")
    if not isinstance(candidates, list):
        return []
    return [
        {
            "spec_id": str(candidate["spec_id"]),
            "label": truncate(str(candidate.get("sentence") or ""), _CANDIDATE_LABEL_CAP),
        }
        for candidate in candidates
        if isinstance(candidate, dict) and candidate.get("spec_id")
    ]


def _clarifying_question_args(
    candidates: list[dict[str, str]], *, lead: str, question: str = _WHICH_MEANT
) -> dict[str, Any]:
    """The `PLANNER_ASK` call args that put a referral's pinnable candidates to
    the user, in the shape `ask_user_question` declares (`planner.specs`),
    built here from the candidates with no model call composing it. `lead`
    says why the user is asked and `question` what they are asked. The call
    recommends no candidate, since nothing has weighed them. The message
    reaches the user, so it names no requirement id, referral kind or
    spec_id. Each candidate's label says what it is.
    """
    return {"message": f"{lead} {question}", "candidates": candidates}


def _user_decided_ask(referral: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """The `PLANNER_ASK` call args for a referral the user decides, or
    `None` for any other answer.

    A referral of a kind in `translator.referral.USER_DECIDES` carrying
    pinnable candidates (`_pinnable_candidates`) leaves the pick to the user,
    so the seam asks it, led by the referral's sentence
    (`harness.sentences.referral_sentence`). One marked `source_mismatch`
    asks which other source to use instead. One with no candidates leaves
    the planner to say what cannot be done and ask what to change.
    """
    if referral is None or referral.get("kind") not in USER_DECIDES:
        return None
    candidates = _pinnable_candidates(dict(referral))
    if not candidates:
        return None
    source_mismatch = bool(referral.get("source_mismatch"))
    lead = referral_sentence(
        referral.get("kind"), referral.get("constraint"), source_mismatch=source_mismatch
    )
    return _clarifying_question_args(
        candidates, lead=lead, question=_WHICH_INSTEAD if source_mismatch else _WHICH_MEANT
    )

"""What one planner tool call reads of its turn, with no agent framework in view.

A `PlannerCall` is the call's session state and the steps of what the user
was asked and answered in its session, in order (`Step`). The readers here
answer from those steps: the spec_ids the user elected (`elected_spec_ids`),
what they answered since the translator's referral
(`answers_since_referral`), the combinations they were asked to add up
(`combination_picks`), whether they spoke since a plan version was recorded
(`user_spoke_since_recorded`), and the prices they chose (`prices_chosen`).
`CallSession` is the `submission.service.SessionView` over a call, through
those readers. `planner.session.SessionCall` is the call the planner serves a
tool with.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, NamedTuple, Protocol

from compiler.evidence import PriceBasis
from harness.graph.orchestrate import counts_chosen
from harness.graph.state import (
    ORCHESTRATOR_SPECS_KEY,
    ServedUnit,
    resume_answers,
    stored_mapping,
    unit_reads,
    units_served,
)
from harness.plan_display import DECISION_CHOICE_PREFIX, decision_option_id
from harness.turn_events import PLANNER_ASK
from submission.prices_choice import (
    PRICES_BY_LABEL,
    PRICES_OPTIONS,
    PRICES_PICKS,
    PRICES_SLOT_PREFIX,
    UNIT_ROWS_STATE_KEY,
    USER_PRICES_SLOT_PREFIXES,
    PricesAnswers,
    PricesChoice,
    measure_key,
    prices_slot,
    save_absorbed_rows,
)
from submission.service import (
    COMBINATION_OPTIONS,
    COMBINE_OPTION,
    Admission,
    Combination,
    ReferralAnswers,
)
from submission.service import PROPOSE_PLAN as PROPOSE_PLAN_PROPOSER
from translator.discovery import load_log
from translator.referral import clear_referral

from planner.plan import (
    DecisionPick,
    SessionState,
    Unit,
    UnsupportedOperation,
    decision_pick,
    keep_found_sources,
    load_plan,
    pin_sources,
    record_optional_decline,
    record_reduction_refused_asks,
    record_unsupported,
    save_alignments,
    save_pin_verdicts,
    save_plan,
    save_plan_rounds,
    save_windowed_pins,
)
from planner.stated_scope import (
    Pair,
    RefusedRequirement,
    record_keep_offered,
    record_unattributed,
)
from planner.tool_results import ProblemCode

# The session-state slot holding how many records of the turn's transcript
# the call that recorded the plan on record read (`CallSession.record_admission`).
# A step from a later record came after the plan (`prices_answers`). The
# state is the turn's own, so the count is always of this turn's transcript.
PLAN_READ_STATE_KEY = "planner:plan_read"


class Step(NamedTuple):
    """One step of a session's history, as the readers of what the user answered walk it.

    `message` is a message the user typed into the session, which the
    planner adapter's own messages are not (`planner.session._user_message`).
    `call` is an ask, the planner's own (`PLANNER_ASK`) or a pause the turn
    put to the user, with its id and `args`. `answer` is a response to one,
    with its id, the `args` of the nearest preceding call sharing that id
    (`None` when none precedes it) and the response body.
    `response` is any other tool's response, with its `name` and body.
    `record` is the position in the turn's transcript of the record the step
    was read from (`planner.session.session_steps`), which places it before
    or after the plan on record was recorded (`PLAN_READ_STATE_KEY`).
    """

    kind: Literal["message", "call", "answer", "response"]
    call_id: str | None = None
    args: dict[str, Any] | None = None
    body: Mapping[str, Any] = MappingProxyType({})
    name: str | None = None
    record: int = 0


class ResponseCall(NamedTuple):
    """One function call of a model response: the tool's `name` and the call's id."""

    name: str
    call_id: str | None


class PlannerCall(Protocol):
    """One planner tool call, as the readers of what the user answered read it.

    `state` is the call's session state, `call_id` the id of the tool call
    being served (`None` outside one), and `steps` the session's history in
    order (`Step`). The answer arrives on a later invocation than the ask
    that paused the run, so the steps span the whole session, across
    invocations. `response_calls` gives the function calls of the model
    response that holds the call being served, in the order the model wrote
    them. `steps` cannot give that order, because an unanswered ask and an ask
    of the same response stand side by side in the steps.
    """

    @property
    def state(self) -> SessionState:
        """The call's session state."""
        ...

    @property
    def call_id(self) -> str | None:
        """The id of the tool call being served, or `None` outside one."""
        ...

    def steps(self) -> Sequence[Step]:
        """The steps of the call's session history, in order."""
        ...

    def response_calls(self) -> Sequence[ResponseCall]:
        """The calls of the model response that holds the call being served, in order,
        or none when the call has no id or no response holds it."""
        ...

    def records_read(self) -> int:
        """How many records of the turn's transcript the call reads."""
        ...


def candidate_ids(candidates: Any) -> frozenset[str]:
    """The spec_ids of a list of `{"spec_id", "label"}` candidates."""
    return frozenset(
        str(candidate["spec_id"])
        for candidate in candidates or []
        if isinstance(candidate, dict) and candidate.get("spec_id")
    )


def _offered_candidates(args: dict[str, Any]) -> dict[str, frozenset[str]]:
    """The candidates an ask's own `args` offered, by the id of the question
    that offered them, from either shape an ask takes:

    - the planner's own `PLANNER_ASK` call: `args` is exactly the kwargs the
      model supplied, flat -- `args["candidates"]`, the one question `q1`.
    - the pause the graph's `ask` node puts to the user: its `args` hold the
      pause's message and its `payload`, which nests under `args["payload"]`
      (`planner.session.session_steps` says how the transcript records it).
      Its `questions` each carry their own `id` and `candidates`, whose ids
      are scoped to that question. A `payload` with `candidates` and no
      `questions` holds the one question `q1`.
    """
    return {
        question_id: candidate_ids(candidates)
        for question_id, candidates in _question_candidates(args).items()
    }


def _question_candidates(args: dict[str, Any]) -> dict[str, Any]:
    """The raw candidates list of each question an ask's
    `args` offers, by question id, read as `_offered_candidates` reads them."""
    candidates = args.get("candidates")
    payload = args.get("payload")
    payload = payload if isinstance(payload, dict) else {}
    if not candidates and payload.get("questions"):
        return {
            str(question["id"]): question.get("candidates")
            for question in payload["questions"]
            if isinstance(question, dict) and question.get("id")
        }
    return {"q1": candidates or payload.get("candidates")}


def prices_chosen(call: PlannerCall) -> dict[str, PricesChoice]:
    """The prices the user picked answering asks, by the
    part each prices, read off the call's steps.

    The `ask` node marks each prices option's candidate with its `part` and
    `prices` (`harness.graph.state.ask_candidates`), so a pick states them
    whether or not the option carries a spec_id. Each response pairs with the
    nearest preceding call sharing its `id`, as `elected_spec_ids` pairs
    them. A choice stands until the user answers the same question again,
    a question pricing the same part or another window of its measure
    (`submission.prices_choice.measure_key`): a pick of other prices
    replaces it, and any other answer, a skip aside, releases it, so one
    measure holds one choice however many asks priced it.

    A prices refusal that sends the planner to the user (`detail.part`,
    `submission.prices_choice`) binds the planner's next ask offering
    `PRICES_OPTIONS` that reached the user (a refused ask leaves no step,
    `planner.session.session_steps`) to its part, and the user's pick there
    sets the part's prices, marked `asked` so the planner does not ask it
    again (`submission.prices_choice`). The user's own words there release
    the part's choice, as they do on the orchestrator's question, and a skip
    keeps it, marking a constant choice `asked`. A switch of the part's
    prices row on the plan card (`_prices_switched`) is a new answer for the
    part, marked `asked` too, while its replan is under way or once the plan
    on record states it (`_switch_stands`). A switch of the row of a whole
    unit a unit question chose is no answer here: the `switch` node re-reads
    the series the plan pins in the unit picked (`harness.graph.switch`), and
    admission reads that row's unit and series off the plan
    (`submission.service._user_prices`).
    Any other `propose_plan` response, the plan landing included, clears the
    refusal, so no later ask of another kind is read as that one. A message the user typed
    (a `message` step) releases every choice and clears the refusal, and an
    amendment of the approval card (`_is_amendment`) releases every choice.
    An answer to any other question leaves them standing. A Studio turn is
    one session, so a choice holds for the turn it was made in.
    """
    return dict(prices_answers(call).chosen)


def prices_answers(call: PlannerCall) -> PricesAnswers:
    """`prices_chosen` over `call`, with the parts whose choice a step after
    the plan on record was recorded set (`fresh`), and whether a message the
    user typed or an amendment of the approval card came after it
    (`released`), in one walk of its steps.

    The plan on record was recorded once the call recording it had read the
    first `plan_read` records of the turn's transcript, so a step from a
    later record comes after it. Every admission records that count, a
    revision of the plan in code included (`CallSession.record_admission`).
    With no plan recorded this turn, every step comes after it: every part
    is fresh and any such step releases. A skip of the planner's prices
    question that marks a choice `asked` sets nothing
    (`submission.prices_choice.plan_held_choices`)."""
    measures = {
        spec_id: measure_key(block)
        for spec_id, source in pin_sources(call.state).items()
        if (block := source.spec_block(spec_id)) is not None
    }
    reads = unit_reads(call.state)
    served = units_served(call.state)
    since = plan_read(call.state)
    chosen: dict[str, PricesChoice] = {}
    refused: str | None = None
    asked: dict[str, str] = {}
    offered = [dict(option) for option in PRICES_OPTIONS]
    fresh: set[str] = set()
    released = False
    after = False

    def choose(part: str, choice: PricesChoice) -> None:
        # A choice set again, even to what it was, is a new answer.
        chosen[part] = choice
        if after:
            fresh.add(part)

    for step in call.steps():
        after = step.record >= since
        if step.kind == "message":
            chosen.clear()
            refused = None
            released = released or after
        elif step.kind == "response":
            if step.name == PROPOSE_PLAN_PROPOSER.call:
                refused = _prices_refused(step.body)
        elif step.kind == "call":
            if refused is not None and (step.args or {}).get("options") == offered:
                asked[str(step.call_id)], refused = refused, None
        elif _is_amendment(step.body):
            chosen.clear()
            released = released or after
        elif (switched := _prices_switched(step.body)) is not None:
            part, basis = switched
            if _switch_stands(call.state, part, basis):
                _release_measure(chosen, part, measures)
                choose(part, PricesChoice(basis, asked=True))
        elif step.call_id in asked:
            part = asked.pop(step.call_id)
            answer = _answers(step.body.get("result")).get("q1") or {}
            basis = PRICES_PICKS.get(str(answer.get("spec_id")))
            if answer.get("kind") == "option" and basis is not None:
                _release_measure(chosen, part, measures)
                choose(part, PricesChoice(basis, asked=True))
            elif _said(answer):
                _release_measure(chosen, part, measures)
            elif part in chosen and chosen[part].prices == "constant":
                chosen[part] = chosen[part]._replace(asked=True)
        else:
            _priced_answers(
                step.args or {},
                step.body,
                chosen,
                measures,
                reads=reads,
                served=served,
                choose=choose,
            )
    return PricesAnswers(chosen, frozenset(fresh.intersection(chosen)), released=released)


def plan_read(state: SessionState) -> int:
    """How many records of the turn's transcript the call recording the plan
    on record read (`PLAN_READ_STATE_KEY`), 0 when no plan was recorded this
    turn."""
    read = state.get(PLAN_READ_STATE_KEY)
    return read if isinstance(read, int) else 0


def _prices_switched(body: Mapping[str, Any]) -> tuple[str, PriceBasis] | None:
    """The part and prices a switch of a part's prices row on the plan card
    picks (`harness.studio`'s `{"result": {"switch": {"choice_id",
    "option_id"}}}` answer to the approval card), or `None` for any other answer."""
    result = body.get("result")
    switch = result.get("switch") if isinstance(result, dict) else None
    if not isinstance(switch, dict):
        return None
    slot = str(switch.get("choice_id")).removeprefix(DECISION_CHOICE_PREFIX)
    if not slot.startswith(PRICES_SLOT_PREFIX):
        return None
    basis = next(
        (
            basis
            for label, basis in PRICES_BY_LABEL.items()
            if decision_option_id(label) == switch.get("option_id")
        ),
        None,
    )
    return (slot.removeprefix(PRICES_SLOT_PREFIX), basis) if basis is not None else None


def _switch_stands(state: Any, part: str, basis: PriceBasis) -> bool:
    """Whether a switch of `part`'s prices row to `basis` holds: its replan is
    under way (`planner.plan.decision_pick` names the row), or the plan on
    record states the switched prices. A switch whose replan was refused or
    stopped holds neither, so the prices stay as the card shows them."""
    slot = prices_slot(part)
    pick = decision_pick(state)
    if pick is not None and pick.slot == slot:
        return True
    plan = load_plan(state)
    return plan is not None and any(
        d.slot == slot and d.fixes is not None and d.fixes.prices == basis for d in plan.decisions
    )


def _release_measure(
    chosen: dict[str, PricesChoice], part: str, measures: Mapping[str, str | None]
) -> None:
    """Take out of `chosen` the choice for `part` and for every part measuring
    what it measures (equal `measures`), as a new answer pricing it replaces them."""
    key = measures.get(part)
    for priced in [p for p in chosen if p == part or (key is not None and measures.get(p) == key)]:
        del chosen[priced]


def _prices_refused(body: Mapping[str, Any]) -> str | None:
    """The part a `propose_plan` refusal of the user's prices sending the planner
    to the user names (`detail.part`), or `None`."""
    detail = body.get("detail")
    action = body.get("action")
    if not (
        isinstance(detail, dict)
        and isinstance(action, dict)
        and action.get("call") == PLANNER_ASK
        and str(detail.get("slot")).startswith(USER_PRICES_SLOT_PREFIXES)
    ):
        return None
    part = detail.get("part")
    return part if isinstance(part, str) else None


def _priced_answers(
    args: dict[str, Any],
    body: Mapping[str, Any],
    chosen: dict[str, PricesChoice],
    measures: Mapping[str, str | None],
    *,
    reads: Mapping[str, Sequence[str]],
    served: Mapping[str, ServedUnit],
    choose: Callable[[str, PricesChoice], None],
) -> None:
    """Record in `chosen`, through `choose`, what one clarify response says of
    each part its call priced, each answer replacing the choices of its
    part's measure (`prices_chosen`).

    A unit question (`harness.graph.state.ask_questions`, `unit`) sets the
    unit of every part it covers to the one the series the agent fetches for
    it is served in (`served`, `harness.graph.state.units_served`): the unit
    the user chose, or the one its chips started on for a skip, unless
    Search refused it, when the part keeps the unit it is served in. The
    choice is keyed by the part alone, names that series (`spec_id`) and
    the unit row the part sits on (`ServedUnit.row`), none for a part Search
    refused to read in the unit chosen, and the series' currency may differ
    from the part's; the series reaches the part
    through `submission.prices_choice.Parts.fetched`. A part whose
    served unit cannot be stated takes no choice. A part with no served unit
    recorded, answered in a turn paused before the `ask` node recorded one,
    takes the prices of the answer, as does each series the part was re-read
    as (`reads`, `harness.graph.state.unit_reads`). A skip or dismissal of a
    whole pause of several questions sets none, as the `ask` node hands it
    back unanswered; of a lone unit question it is that question's skip."""
    result = body.get("result")
    answers = _answers(result)
    units = _unit_questions(args)
    lone = len(_question_candidates(args)) == 1
    answered = result is not None and (resume_answers(result) is not None or lone)
    for question_id, unit in units.items() if answered else ():
        prices = _unit_prices(unit, answers.get(question_id))
        parts = [str(part) for part in unit.get("parts") or ()]
        # The parts share one measure (`unit_asks`), so release them all before
        # setting any, or releasing one part would drop the choice just set for another.
        for part in parts:
            _release_measure(chosen, part, measures)
        for part in parts:
            if (fetched := served.get(part)) is not None:
                if fetched.unit is not None:
                    choose(
                        part,
                        PricesChoice(
                            fetched.unit.prices,
                            fetched.spec_id,
                            unit=fetched.unit,
                            group=fetched.row,
                        ),
                    )
            elif prices is not None:
                for priced in (part, *reads.get(part, ())):
                    choose(priced, PricesChoice(prices))
    for question_id, candidates in _question_candidates(args).items():
        marked = [c for c in candidates or () if isinstance(c, dict) and c.get("part")]
        answer = answers.get(question_id)
        if not marked or not isinstance(answer, dict) or not _said(answer):
            continue
        part = str(marked[0]["part"])
        picked = next(
            (
                choice
                for candidate in marked
                if answer.get("kind") == "option"
                and candidate.get("spec_id") == answer.get("spec_id")
                and (choice := PricesChoice.of_candidate(candidate)) is not None
            ),
            None,
        )
        _release_measure(chosen, part, measures)
        if picked is not None:
            choose(part, picked)


def _unit_questions(args: dict[str, Any]) -> dict[str, Mapping[str, Any]]:
    """The `unit` entry of each unit question an ask's `args` put, by question id."""
    payload = args.get("payload")
    questions = payload.get("questions") if isinstance(payload, dict) else None
    return {
        str(question["id"]): question["unit"]
        for question in questions or ()
        if isinstance(question, dict)
        and question.get("id")
        and isinstance(question.get("unit"), dict)
    }


def _unit_prices(unit: Mapping[str, Any], answer: object) -> PriceBasis | None:
    """The prices a unit question's `answer` settles: a unit answer's, else, as
    for a skip, the prices its offer's chips started on."""
    if isinstance(answer, dict) and answer.get("kind") == "unit":
        prices = answer.get("prices")
    else:
        prices = ((unit.get("offer") or {}).get("start") or {}).get("prices")
    match prices:
        case "current":
            return "current"
        case "constant":
            return "constant"
    return None


def _answers(result: object) -> dict[str, dict[str, Any]]:
    """An ask's response's `result` as one answer per question id
    (`harness.graph.state.resume_answers`). A missing result and a dismissal of the
    whole pause hold none."""
    return {} if result is None else resume_answers(result) or {}


def _picks(result: object, offered: dict[str, frozenset[str]]) -> tuple[str, ...]:
    """The spec_ids `result` picks, in question order, each as an option its own
    question offered.

    A skipped or free-text answer picks nothing, and neither does an id another
    question offered."""
    return tuple(
        str(answer["spec_id"])
        for question_id, answer in _answers(result).items()
        if isinstance(answer, dict)
        and answer.get("kind") == "option"
        and str(answer.get("spec_id")) in offered.get(question_id, frozenset())
    )


def _answered_calls(call: PlannerCall) -> Iterator[tuple[dict[str, Any], object]]:
    """Each interrupt answer with the nearest preceding call sharing its id."""
    for step in call.steps():
        if step.kind == "answer" and step.args is not None:
            yield step.args, step.body.get("result")


def elected_spec_ids(call: PlannerCall) -> frozenset[str]:
    """Every spec_id the user actually chose answering an ask, read off the
    call's steps -- the record of what
    the user was asked and what they answered, which
    makes a measure decision the user's (`_user_settled_measures`).

    Membership in a call's offered `candidates` is not by itself an
    election: an ask that offers `{A, B}` records only
    a question, not a choice, and a plan pinning `B` after the user picked
    `A` -- or after no answer has arrived at all -- is not a user election
    either. What counts as one is a call's `answer` step, a call paired
    with the response, if any, that answers it. The answer's `result` is read
    as one answer per question id (`harness.graph.state.resume_answers`), a
    chosen candidate as `{"kind": "option", "spec_id": ...}`. A call with no
    paired response yet -- the run is still paused there, or the answer
    hasn't arrived -- contributes nothing: its candidates were offered, not
    elected.

    One response's picks elect the counts they choose, read against the
    turn's search blocks (`harness.graph.orchestrate.counts_chosen`): a pick
    of other prices alone chooses no count, and a count picked with other
    prices elects the listed reading measuring both, the one the `ask`
    node tells the agent to fetch.

    The steps walk the turn's transcript once, in order, and pair each
    answer with the call whose offer was seen most recently at that point (the
    nearest preceding call sharing its id), so an answer resolves against the
    offer it answered, whatever a later offer under the same id holds.

    The call carrying its offered candidates, and the answer once it arrives,
    are both in the turn's transcript (`planner.session.session_steps` says
    how it holds them). The transcript is read whole, since the ask pauses the
    turn and the answer resumes it in a later segment.
    """
    specs = stored_mapping(call.state, ORCHESTRATOR_SPECS_KEY)
    elected: set[str] = set()
    for args, result in _answered_calls(call):
        elected |= counts_chosen(_picks(result, _offered_candidates(args)), specs)
    return frozenset(elected)


def _said(answer: object) -> str:
    """What one answer says: the candidate picked or the user's own words, else nothing."""
    if not isinstance(answer, dict):
        return ""
    key = {"option": "spec_id", "text": "text"}.get(answer.get("kind"))
    value = answer.get(key) if key else None
    return value.strip() if isinstance(value, str) else ""


def _is_answer(body: Mapping[str, Any]) -> bool:
    """Whether an ask's response body carries something the user said.

    A user's answer is `{"result": ...}` (`planner.session.session_steps` wraps
    it), read as `harness.graph.state.resume_answers` reads it: a pick of a
    candidate, or free text for a reply outside the offered options. A body
    with no `result` is a call the tool rejected. A dismissal of the pause and
    a question left to the planner are the user declining to choose, and blank
    text says nothing.
    """
    return any(_said(answer) for answer in _answers(body.get("result")).values())


def _is_amendment(body: Mapping[str, Any]) -> bool:
    """Whether an ask's response body is the user amending the approval card.

    `harness.studio`'s `/interrupt` route answers the approval card
    (`harness.graph.state.APPROVE_PAUSE`) with `{"result": {"amendment": ...}}`
    when the user types a change there.
    """
    result = body.get("result")
    amendment = result.get("amendment") if isinstance(result, dict) else None
    return isinstance(amendment, str) and bool(amendment.strip())


def user_spoke_since_recorded(call: PlannerCall, version: int) -> bool:
    """Whether the user has said something since plan `version` was recorded.

    Read off the call's steps. The first `propose_plan` response carrying `version`
    marks where the plan was recorded. A later response repeating the version records nothing
    (`_transient_retry` answers a bare retry so) and moves no mark. After the mark, the user
    speaks with an ask's response they answered (`_is_answer`: a pick, free text),
    an amendment of the approval card (`_is_amendment`), or a message of their own
    (a `message` step). A
    refusal, a skip, a referral and the planner adapter's own messages are not.

    Every Studio turn has a transcript of its own (`planner.session`), and
    every turn starts with the user's message. A plan this transcript holds
    no recording response for was recorded before that message, so this
    answers `True`. The planner adapter seeds such a plan from the graph's
    record (`harness.planner_adapter.PlannerAdapter`).
    """
    last: object = None
    recorded = False
    spoke = False
    for step in call.steps():
        if step.kind == "message":
            if recorded:
                spoke = True
        elif step.kind == "response":
            if (
                step.name == PROPOSE_PLAN_PROPOSER.call
                and "version" in step.body
                and step.body["version"] != last
            ):
                last = step.body["version"]
                recorded, spoke = last == version, False
        elif step.kind == "answer" and recorded:
            spoke = spoke or _is_answer(step.body) or _is_amendment(step.body)
    return spoke or not recorded


def answered_since_referral(call: PlannerCall) -> bool:
    """Whether the user has answered an ask the planner raised after seeing the
    translator's referral (`answers_since_referral`)."""
    return answers_since_referral(call).answered


def answers_since_referral(call: PlannerCall) -> ReferralAnswers:
    """Whether the user has answered an ask the planner raised after seeing the
    translator's referral, the spec_ids they picked answering those asks, and the
    spec_ids offered by those asks still unanswered, read off the call's steps as
    `elected_spec_ids` reads them.

    The referral reaches the planner in the `translation` of the response that recorded the
    plan, `propose_plan`'s, carrying its `kind` and `requirement_id`. This walks the steps
    once, in order. An ask after that response is one the
    model raised knowing the referral, and its `id` is remembered. A response counts as the
    user's answer when it pairs by `id` with one of those asks and `_is_answer` holds. An ask
    issued before the referral, a call the tool rejected, and a skip do not count. A skip
    is the user declining to choose, which the planner may not substitute for, so the planner
    ends the turn by telling them what cannot be done.

    A response that recorded a version clears the referral in session
    state and translates the new version, so it ends the referral this walk is
    following and starts the one its `translation` names, if any. A refused
    proposal leaves both alone, which keeps the user's answer standing for the
    retry.

    With no referral on record the result is no answer. The guard refuses when
    it cannot say where the referral began, so an answer given for something
    else settles nothing.

    A spec_id counts as picked when an answer names it and the ask it answers,
    the nearest preceding one with its `id`, offered it, as `elected_spec_ids`
    pairs them. A pick made before the referral is not counted. Picks keep the
    order the answers came in. An ask with no answer, a skip included, leaves
    every id it offered unanswered. An ask whose response has no `result`,
    one the tool refused, never reached the user, so it offered nothing. A
    pending ask has no response and stays unanswered.

    `picked_ever` holds the user's standing picks since the first referral,
    which a later translation does not clear. An answered ask, picked or in
    the user's own words, replaces every earlier pick among the ids it
    offered, so the latest answer about a reading stands and a free-text
    answer releases it to the planner. A message the user typed
    (a `message` step) or an amendment of the approval card (`_is_amendment`)
    releases them all.
    """
    answered = False
    referral: tuple[object, object] | None = None
    asks: dict[str, dict[str, frozenset[str]]] = {}
    replied: set[str] = set()
    picked: list[str] = []
    picked_ever: list[str] = []
    for step in call.steps():
        if step.kind == "message":
            picked_ever.clear()
        elif step.kind == "call":
            if referral is not None:
                asks[str(step.call_id)] = _offered_candidates(step.args or {})
        elif step.kind == "answer":
            if _is_amendment(step.body):
                picked_ever.clear()
            call_id = step.call_id
            if call_id is None or call_id not in asks:
                continue
            if "result" not in step.body:
                del asks[call_id]
                continue
            if not _is_answer(step.body):
                continue
            answered = True
            replied.add(call_id)
            offered = {spec_id for ids in asks[call_id].values() for spec_id in ids}
            picks = _picks(step.body.get("result"), asks[call_id])
            picked.extend(picks)
            picked_ever = [pick for pick in picked_ever if pick not in offered]
            picked_ever.extend(picks)
        elif "translation" in step.body:
            translation = step.body["translation"]
            referral = (
                (translation.get("kind"), translation.get("requirement_id"))
                if isinstance(translation, dict) and "kind" in translation
                else None
            )
            answered = False
            asks.clear()
            replied.clear()
            picked.clear()
    unanswered = frozenset(
        spec_id
        for call_id, offered in asks.items()
        if call_id not in replied
        for spec_ids in offered.values()
        for spec_id in spec_ids
    )
    return ReferralAnswers(
        answered=answered,
        picked=tuple(picked),
        unanswered=unanswered,
        picked_ever=tuple(picked_ever),
    )


def _refused_combination(body: Mapping[str, Any]) -> Combination | None:
    """The combination a `combination_unasked` refusal names in its `detail`, or `None`."""
    if body.get("code") != ProblemCode.COMBINATION_UNASKED.value:
        return None
    detail = body.get("detail")
    if not isinstance(detail, dict):
        return None
    requirement_id = detail.get("requirement_id")
    datasets = detail.get("datasets")
    if not isinstance(requirement_id, str) or not isinstance(datasets, list):
        return None
    named = [dataset for dataset in datasets if isinstance(dataset, str)]
    if len(named) != len(datasets):
        return None
    return requirement_id, frozenset(named)


def combination_picks(
    call: PlannerCall,
) -> tuple[dict[Combination, str | None], bool]:
    """The option id the user last picked for each combination they were asked about, and a flag.

    A combination ask is a `PLANNER_ASK` call whose `options` are `COMBINATION_OPTIONS`.
    It asks about the combination the latest `combination_unasked` refusal before it names, read
    off that refusal's own response, so the model cannot point an ask at another sum. A
    refusal binds the first ask after it that reached the user and no other, so an ask
    with no refusal of its own before it asks about nothing. An ask the tool refused
    leaves no step (`planner.session.session_steps`), so it binds nothing and leaves the
    refusal for the next ask. Each response pairs with the nearest preceding call sharing
    its `id`, as `elected_spec_ids` pairs them. The combination ask is one question, `q1`
    (`harness.graph.state.resume_answers`). Each answer about a combination replaces the one
    before it, and free text or a skip records `None`, so a user who changes their mind, or
    answers outside the options, ends with no pick for it. The flag says whether any answer to
    any combination ask, about a combination or about nothing, was other than `COMBINE_OPTION`.
    """
    offered = [dict(option) for option in COMBINATION_OPTIONS]
    ids = {option["id"] for option in COMBINATION_OPTIONS}
    refused: Combination | None = None
    asks: dict[str, Combination | None] = {}
    picks: dict[Combination, str | None] = {}
    declined = False
    for step in call.steps():
        if step.kind == "call":
            if (step.args or {}).get("options") == offered:
                asks[str(step.call_id)], refused = refused, None
            else:
                asks.pop(str(step.call_id), None)
        elif step.kind == "response":
            refused = _refused_combination(step.body) or refused
        elif step.kind == "answer" and step.call_id in asks:
            answer = _answers(step.body.get("result")).get("q1")
            option = answer.get("spec_id") if answer and answer.get("kind") == "option" else None
            declined = declined or option != COMBINE_OPTION
            combination = asks[step.call_id]
            if combination is not None:
                picks[combination] = option if option in ids else None
    return picks, declined


@dataclass(frozen=True)
class CallSession:
    """The `submission.service.SessionView` over one planner tool call.

    The state is the call's own. What the user was asked and answered is
    read off the call's steps (`elected_spec_ids`,
    `answers_since_referral`, `combination_picks`,
    `user_spoke_since_recorded`, `prices_chosen`). Recording an admitted
    plan writes its windowed pins, the plan, its round count, and clears the
    referral the new proposal supersedes.
    """

    call: PlannerCall

    @property
    def state(self) -> SessionState:
        """The call's session state."""
        return self.call.state

    def elected_spec_ids(self) -> frozenset[str]:
        """`elected_spec_ids` over the call."""
        return elected_spec_ids(self.call)

    def answers_since_referral(self) -> ReferralAnswers:
        """`answers_since_referral` over the call."""
        return answers_since_referral(self.call)

    def combination_picks(self) -> tuple[dict[Combination, str | None], bool]:
        """`combination_picks` over the call."""
        return combination_picks(self.call)

    def user_spoke_since(self, version: int) -> bool:
        """`user_spoke_since_recorded` over the call."""
        return user_spoke_since_recorded(self.call, version)

    def prices_answers(self) -> PricesAnswers:
        """`prices_answers` over the call."""
        return prices_answers(self.call)

    def switched_units(self) -> Mapping[str, Unit]:
        """None: a planner call revises no unit row a switch moved."""
        return {}

    def decision_pick(self) -> DecisionPick | None:
        """The pick on the plan card the call's planner entry holds (`planner.plan.decision_pick`)."""
        return decision_pick(self.call.state)

    def record_unsupported(self, operation: UnsupportedOperation) -> None:
        """Record `operation` as answered `unsupported` in this planner entry."""
        record_unsupported(self.call.state, operation)

    def record_optional_decline(self, operation: UnsupportedOperation) -> None:
        """Permit `operation` as an optional decline in this planner entry."""
        record_optional_decline(self.call.state, operation)

    def record_reduction_refused(self, asks: Iterable[int]) -> None:
        """Record `asks` as named by a `PERIOD_REDUCTION_UNSTATED` refusal in this planner entry."""
        record_reduction_refused_asks(self.call.state, asks)

    def record_scope_unattributed(
        self, option_id: str, refused: Iterable[RefusedRequirement]
    ) -> None:
        """Record a `STATED_SCOPE_UNATTRIBUTED` refusal for `option_id` in the call's
        state, with `refused`: the id and `measure` of each dataset requirement whose pin
        offers the option."""
        record_unattributed(self.call.state, option_id, refused)

    def record_keep_offered(self, pairs: Iterable[Pair]) -> None:
        """Record that a refusal listed the datasets serving `pairs`
        (`planner.stated_scope.record_keep_offered`) in the call's state."""
        record_keep_offered(self.call.state, pairs)

    def record_admission(self, admission: Admission) -> None:
        """Record `admission` in the call's state, with how many records of
        the turn's transcript the call read (`PLAN_READ_STATE_KEY`)."""
        state = self.call.state
        state[PLAN_READ_STATE_KEY] = self.call.records_read()
        if admission.windowed_pins is not None:
            save_windowed_pins(state, admission.windowed_pins)
        save_alignments(state, admission.alignments)
        save_pin_verdicts(state, admission.pin_verdicts)
        # Each admission replaces the record; a state that never held one
        # gains none until a row absorbs a part.
        if admission.unit_rows or state.get(UNIT_ROWS_STATE_KEY) is not None:
            save_absorbed_rows(state, admission.unit_rows)
        save_plan(state, admission.plan)
        keep_found_sources(
            state,
            admission.plan,
            alignments=admission.alignments,
            windowed_pins=admission.windowed_pins or (),
            offered=load_log(state).offered_for(admission.plan),
        )
        save_plan_rounds(state, admission.rounds)
        # A new proposal supersedes whatever the translator referred back.
        clear_referral(state)

"""The `orchestrate` and `ask` nodes: the agent's runs of the turn, and its
question to the user.

`orchestrate` runs the agent (`orchestrator.run.run_orchestrator`) from the
turn's opening, or from the `AgentHandBack` that answers the `plan` or
`ask_user` call its last run handed off on, and streams the run's events.
The action the hand-off recorded is the node's edge: `plan` or `ask`, with
the action as the next node's input, or `answered` when the agent replied.
"""

from __future__ import annotations

from collections.abc import (
    Callable,
    Collection,
    Mapping,
    Sequence,
)
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from compiler.surface import ToolSurface
from orchestrator.agent import Orchestrator
from orchestrator.run import (
    Answered,
    HandBack,
    HandedOff,
    Opening,
    StoredHistory,
    run_orchestrator,
)
from planner.plan import SessionState
from pydantic import BaseModel, Field, model_validator
from pydantic_ai.models import Model

from harness.gemini_agent import RecordingState
from harness.graph.events import AgentStatus, GraphEvent
from harness.graph.state import (
    ORCHESTRATOR_GIVEN_BACK_ASKS_KEY,
    ORCHESTRATOR_HISTORY_KEY,
    ORCHESTRATOR_PENDING_KEY,
    ORCHESTRATOR_SEARCHES_KEY,
    ORCHESTRATOR_SPECS_KEY,
    AskOption,
    AskQuestion,
    PendingAsk,
    TurnInput,
    UnitAsk,
    ask_candidates,
    ask_questions,
    pending_action,
    priced_options,
    record_unit_read,
    record_unit_served,
    resume_answers,
    settle_unit,
    store_search,
    stored_mapping,
    stored_sequence,
    unit_asks,
)
from harness.scope_choices import mark_planner_axes
from harness.tool_contracts import resolver_digest
from harness.turn.node import Next, Node, NodeCtx, Pause, TurnKey
from harness.turn_records import SearchReading
from harness.unit_offer import served_unit, unit_words
from harness.unit_switch import UnitRefused, UnitResubmitted, resubmit_unit
from harness.wire import UnitChoice, UnitOffer


class AgentHandBack(BaseModel):
    """What a node hands the agent through `orchestrate`, and what it answers.

    `answers` names the agent's call the hand-back is the result of: `plan`
    for `plan_done`, `ask_user` for `ask`. None for `seed`'s hand-back, the
    runs, datasets or chip a turn opens with, which answers no call. `text`
    leads the result, and `body` holds its structured fields, which the agent
    reads beside `text` in the call's result (`response`); a seed hand-back
    is text alone. `status` is the line Studio shows while the agent reads
    it, and `failure` the refusal a plan failed with, when one did.
    `written` names the turn state keys the node wrote in reading the
    answer, which reach Studio with the answer (`orchestrator.run.HandBack`);
    the agent reads none of it.
    """

    answers: Literal["plan", "ask_user"] | None
    status: str
    text: str
    body: dict[str, Any] = Field(default_factory=dict)
    failure: dict[str, Any] | None = None
    written: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _body_answers_a_call(self) -> AgentHandBack:
        if self.body and self.answers is None:
            raise ValueError("a seed hand-back is text alone")
        if "result" in self.body:
            raise ValueError("a hand-back's body cannot hold `result`, which holds its text")
        return self

    def response(self) -> dict[str, Any]:
        """The result of the call the hand-back answers, as the agent reads it."""
        return {"result": self.text, **self.body}


def turn_id(key: TurnKey) -> str:
    """The id the orchestrator's tools know the turn by
    (`orchestrator.call.OrchestratorCall.turn_id`), the same on every run of
    the turn."""
    return f"{key.conversation_id}/{key.turn_index}"


# The edge each hand-off takes, by the tool whose call handed off.
HAND_OFF_EDGES = {"plan": "plan", "ask_user": "ask"}


def orchestrate_node(*, orchestrator: Orchestrator, model: Callable[[], Model]) -> Node[GraphEvent]:
    """The `orchestrate` node, running `orchestrator` on the model `model`
    builds for each run (`orchestrator.model.orchestrator_model`, on the
    turn's budget)."""

    async def orchestrate(ctx: NodeCtx[GraphEvent], node_input: Any) -> Next:
        """One run of the agent. Reached from `seed` with nothing or a seed
        hand-back, and again after `ask` and `plan_done` with the
        `AgentHandBack` that answers the agent's `ask_user` or `plan` call.

        The entry first shows its status line: "Reading your question", or
        the hand-back's own. A turn's first run opens on the user's message,
        the date and the digests (`TurnInput`), with a seed hand-back after
        the message; a later run resumes the conversation the last one
        handed off with (`ORCHESTRATOR_HISTORY_KEY`), its waiting call
        answered by the hand-back. The run's first `ToolsAnswered` carries
        that answer, so the node emits nothing else for it. Every event of
        the run is emitted as it comes.

        A run that hands off stores its conversation and takes the action
        its call recorded (`ORCHESTRATOR_PENDING_KEY`) as the edge and the
        next node's input. Anything the run raises fails the turn: a model
        response Gemini blocked (`pydantic_ai.exceptions.ContentFilterError`)
        and one that held no answer (`harness.gemini_agent.EmptyResponseError`)
        included.

        Raises:
            RuntimeError: A hand-back answers a call but no conversation
                waits on one, or a run handed off with no action recorded.
        """
        hand_back = AgentHandBack.model_validate(node_input) if node_input is not None else None
        if hand_back is None:
            ctx.emit(AgentStatus(text="Reading your question"))
        else:
            ctx.emit(AgentStatus(text=hand_back.status, failure=hand_back.failure))
        opening: Opening | None = None
        history: StoredHistory | None = None
        resumed: HandBack | None = None
        if hand_back is None or hand_back.answers is None:
            inputs = TurnInput.model_validate(ctx.inputs)
            opening = Opening(
                question=inputs.message,
                today=datetime.now(UTC).date(),
                canvas_digest=inputs.canvas_digest,
                conversation_digest=inputs.conversation_digest,
                seeded=hand_back.text if hand_back is not None else None,
            )
        else:
            stored = ctx.state.pop(ORCHESTRATOR_HISTORY_KEY, None)
            if stored is None:
                raise RuntimeError(f"no conversation waits on the {hand_back.answers} hand-back")
            history = StoredHistory.model_validate(stored)
            resumed = HandBack(
                answers=hand_back.answers,
                result=hand_back.response(),
                written=tuple(hand_back.written),
            )
        run = run_orchestrator(
            orchestrator,
            state=ctx.state,
            turn_id=turn_id(ctx.key),
            model=model(),
            opening=opening,
            history=history,
            hand_back=resumed,
            conversation_id=ctx.key.conversation_id,
        )
        async for event in run:
            if isinstance(event, HandedOff):
                return _handed_off(ctx, event)
            if isinstance(event, Answered):
                return Next("answered")
            ctx.emit(event)
        raise RuntimeError("the orchestrator's run ended without its reply or a hand-off")

    return Node(run=orchestrate)


def _handed_off(ctx: NodeCtx[GraphEvent], handed: HandedOff) -> Next:
    """Store the conversation `handed` waits with, and go on along the edge
    of the action its call recorded, removing that action from the state."""
    pending = pending_action(ctx.state)
    ctx.state.pop(ORCHESTRATOR_PENDING_KEY, None)
    if pending is None or pending.kind != HAND_OFF_EDGES.get(handed.tool):
        raise RuntimeError(f"the {handed.tool} call handed off with no action of its own recorded")
    ctx.state[ORCHESTRATOR_HISTORY_KEY] = handed.history.model_dump(mode="json")
    return Next(pending.kind, pending.model_dump(mode="json"))


_SETTLED = "Each answered choice is settled. Do not ask it again."


async def _ask(ctx: NodeCtx[GraphEvent], node_input: dict[str, Any]) -> Pause:
    """Put the agent's questions to the user in one pause, each under its id
    with its options as candidates (`ask_questions`). Each prices option
    carries the part and prices code reads for it (`priced_options`), and a
    unit question the parts it covers and its offer (`unit_asks`), which the
    turn's search results do not change while the pause is open."""
    questions = PendingAsk.model_validate(node_input).questions
    return Pause(
        kind="ask",
        message=questions[0].question,
        payload={
            "questions": ask_questions(
                questions,
                priced_options(questions, ctx.state),
                unit_asks(questions, ctx.state),
            )
        },
    )


def ask_node(*, definition_surface: ToolSurface) -> Node[GraphEvent]:
    """The `ask` node: the agent's question to the user, a pause, and the answer
    handed back, re-reading a unit answer's parts through `definition_surface`."""

    async def answered(
        ctx: NodeCtx[GraphEvent], node_input: dict[str, Any], answer: dict[str, Any]
    ) -> Next:
        """Hand the user's `answer` back in the options' own words
        (`answer_replies`), as the result of the agent's `ask_user` call
        (`AgentHandBack`).

        A unit question's answer, and its skip, which goes on with the unit
        the question started on, are read in code first: each part the
        question covers is re-read in that unit (`_unit_reply`), and the
        reply names the spec_id to fetch. Skipping the whole pause skips a
        lone unit question the same way. The node writes the state through
        one view, and the hand-back names the keys it wrote, so Studio
        reads a re-read's block before the agent fetches it."""
        state = RecordingState(ctx.state)
        questions = PendingAsk.model_validate(node_input).questions
        answers = resume_answers(answer)
        units = unit_asks(questions, state)
        if answers is None and len(questions) == 1 and units[0] is not None:
            answers = {"q1": {"kind": "skipped"}}
        if answers is None:
            replies = [""] * len(questions)
            them = (
                "the question. Do not answer it or search for it"
                if len(questions) == 1
                else "the questions. Do not answer them or search for them"
            )
            closing = (
                f"The user skipped {them}: close the turn in a sentence and ask what else you "
                "can help with."
            )
        else:
            picked = _picked_spec_ids(questions, answers)
            settled = {
                index: await _unit_reply(
                    state,
                    definition_surface,
                    ask,
                    answers.get(f"q{index + 1}") or {"kind": "skipped"},
                    picked=picked,
                )
                for index, ask in enumerate(units)
                if ask is not None
            }
            replies = answer_replies(questions, answers, settled=settled)
            closing = _SETTLED
            # The user answered, and may ask for other outputs than a plan call
            # given back was held to.
            state[ORCHESTRATOR_GIVEN_BACK_ASKS_KEY] = None
        if len(questions) == 1:
            lines = [f"You asked the user: {questions[0].question}", *filter(None, replies)]
        else:
            lines = ["You asked the user:"]
            lines += [
                f"{number}. {question.question} {reply}".rstrip()
                for number, (question, reply) in enumerate(zip(questions, replies, strict=True), 1)
            ]
        if answers is not None and (
            paired := measure_pair_reply(
                questions,
                answers,
                stored_mapping(state, ORCHESTRATOR_SPECS_KEY),
                settled={part for ask in units if ask is not None for part in ask.parts},
            )
        ):
            lines.append(paired)
        hand_back = AgentHandBack(
            answers="ask_user",
            status="Reading your answer",
            text="\n".join([*lines, closing]),
            written=list(state.written),
        )
        return Next("next", hand_back.model_dump(mode="json"))

    return Node(run=_ask, resume=answered)


def answered_unit(offer: UnitOffer, answer: Mapping[str, Any]) -> UnitChoice:
    """The unit `answer` to a unit question names; any other answer, a skip
    included, goes on with the unit the question's chips started on."""
    if answer.get("kind") != "unit":
        return offer.start
    return UnitChoice.model_validate(
        {key: answer.get(key) for key in ("currency", "prices", "base_year")}
    )


async def _unit_reply(
    state: SessionState,
    surface: ToolSurface,
    ask: UnitAsk,
    answer: Mapping[str, Any],
    *,
    picked: Sequence[str],
) -> str:
    """The reply to a unit question: the unit the user chose, or the starting
    unit when they left it to the agent, then a sentence for each part it
    covers (`_part_in_unit`)."""
    unit = answered_unit(ask.offer, answer)
    words = _in_sentence(unit_words(unit, ask.offer))
    lead = (
        f"The user chose {words}."
        if answer.get("kind") == "unit"
        else f"The user left this to you: go on in {words}."
    )
    offers = _offers(stored_mapping(state, ORCHESTRATOR_SPECS_KEY))
    sentences = [lead]
    for part in ask.parts:
        count = next(
            (
                spec_id
                for spec_id in picked
                if spec_id != part
                and (offer := offers.get(spec_id)) is not None
                and offer.block == part
                and offer.axis == "what"
            ),
            None,
        )
        sentences.append(
            await _part_in_unit(state, surface, part, unit, offer=ask.offer, count=count)
        )
    return " ".join(sentences)


def _in_sentence(words: str) -> str:
    """`harness.unit_offer.unit_words` inside a sentence: "constant 2023 US dollars"."""
    return words[:1].lower() + words[1:]


def _latest_reading(state: Any, spec_id: str) -> SearchReading | None:
    """The turn's newest reading of `spec_id`."""
    for stored in reversed(stored_sequence(state, ORCHESTRATOR_SEARCHES_KEY)):
        if stored.get("spec_id") == spec_id:
            return SearchReading.model_validate(stored)
    return None


def _count_block(block: Mapping[str, Any], count: str) -> dict[str, Any] | None:
    """`block` with the definition of `count`, another count it offers
    (`resolved.alternative_measures`), in place of its own; `None` when it offers none."""
    for alternative in (block.get("resolved") or {}).get("alternative_measures") or ():
        definition = alternative.get("definition") if isinstance(alternative, Mapping) else None
        if isinstance(definition, Mapping) and definition.get("spec_id") == count:
            return {**block, "definition": dict(definition)}
    return None


@dataclass(frozen=True)
class UnitRead:
    """What `read_part_in_unit` did for a part: `label`, the part's name in
    words; `target`, the series it read, the part or the count picked beside
    it; `served`, the unit the part is served in; and `result`, Search's
    answer, `None` when the part was served in the unit as it stands."""

    label: str
    target: str
    served: UnitChoice | None
    result: UnitResubmitted | UnitRefused | None


async def read_part_in_unit(
    state: SessionState,
    surface: ToolSurface,
    part: str,
    unit: UnitChoice,
    *,
    count: str | None = None,
) -> UnitRead:
    """Settle `part` in `unit`: the part as it stands when served in that
    unit, else the part Search re-reads in it
    (`harness.unit_switch.resubmit_unit`), stored as the turn's newest reading
    of it. With `count`, another count of the part the user picked in the same
    pause, the count's definition is re-read in place of the part's, with the
    part's reading for its roles and named sets; the part's scope base is
    left out, since it compiles the part's own count.

    The part, and the count, are settled on every path, so the question is
    not asked of them again, and the series fetched for the part is recorded
    with the unit it is served in and the count read in the part's place
    (`record_unit_served`): `unit`, unless Search refuses it, when the series
    keeps the unit it is served in."""
    block = stored_mapping(state, ORCHESTRATOR_SPECS_KEY).get(part) or {}
    reading = _latest_reading(state, part)
    label = (reading.text if reading is not None else None) or part
    target, target_block, resubmitted = part, block, reading
    if count is not None and (counted := _count_block(block, count)) is not None:
        target, target_block = count, counted
        if reading is not None:
            resubmitted = reading.model_copy(update={"scope": None})
    settle_unit(state, part, target)
    served = served_unit(block)
    read_count = target if target != part else None
    if count is None and served == unit:
        record_unit_served(state, part, part, unit)
        return UnitRead(label, target, served, None)
    result = await resubmit_unit(
        surface,
        reading=resubmitted,
        block=target_block,
        unit=unit.model_dump(mode="json", exclude_none=True),
        label=label,
    )
    if isinstance(result, UnitRefused):
        record_unit_served(state, part, target, served_unit(target_block), count=read_count)
        return UnitRead(label, target, served, result)
    settle_unit(state, result.spec_id)
    record_unit_read(state, part, result.spec_id)
    record_unit_served(state, part, result.spec_id, unit, count=read_count)
    digest = resolver_digest({"parts": [dict(result.part)]})
    for read in digest.get("part_digests") or ():
        read["variable"] = label
    if reading is not None:
        mark_planner_axes(digest, reading.planner_axes)
    store_search(
        state,
        digest,
        asked=[reading.asked_roles] if reading is not None else (),
        held_open=[reading.open_roles] if reading is not None else (),
        share_sides=[reading.share_over] if reading is not None else (),
    )
    return UnitRead(label, target, served, result)


async def _part_in_unit(
    state: SessionState,
    surface: ToolSurface,
    part: str,
    unit: UnitChoice,
    *,
    offer: UnitOffer,
    count: str | None,
) -> str:
    """What the agent fetches for `part` in `unit`, in a sentence: the part
    as it stands, the series Search re-read it as (`read_part_in_unit`), or,
    for a unit Search does not serve, the part as served, said so."""
    words = _in_sentence(unit_words(unit, offer))
    read = await read_part_in_unit(state, surface, part, unit, count=count)
    if read.result is None:
        return f"{read.label} ({part}) is served in {words}: fetch it as it stands."
    if isinstance(read.result, UnitRefused):
        served = read.served
        as_served = f", in {_in_sentence(unit_words(served, offer))}," if served is not None else ""
        return (
            f"Search could not serve {words} for {read.label}: {read.result.reason} Fetch "
            f"{read.target} as served{as_served} and say so."
        )
    return (
        f"Search read {read.label} in {words} as {read.result.spec_id}: fetch it in place of "
        f"{read.target}."
    )


def answer_replies(
    questions: list[AskQuestion],
    answers: dict[str, dict[str, Any]],
    *,
    settled: Mapping[int, str] | None = None,
) -> list[str]:
    """What the user answered to each of `questions`, in order, from `answers`
    keyed by question id (`resume_answers`): a pick in the option's own words
    with its spec_id when it carries one, the labels of several picks of a
    `multi_select` question joined by "; ", the user's own words, or, for a question
    the user skipped or left unanswered, the first option as the agent's to
    go on with. `settled` holds the reply to each unit question by place,
    which code wrote (`_unit_reply`)."""
    replies = []
    for index, question in enumerate(questions):
        if settled and index in settled:
            replies.append(settled[index])
            continue
        answer = answers.get(f"q{index + 1}") or {"kind": "skipped"}
        if answer.get("kind") == "text":
            replies.append(f"The user answered in their own words: {answer['text']}")
            continue
        if answer.get("kind") == "options" and question.options:
            ids = [candidate["spec_id"] for candidate in ask_candidates(question.options)]
            labels = [
                question.options[ids.index(picked)].label
                for picked in answer.get("spec_ids") or ()
                if picked in ids
            ]
            if labels:
                replies.append(f"The user chose: {'; '.join(labels)}")
                continue
        if answer.get("kind") != "option" or not question.options:
            first = question.options[0].label if question.options else None
            replies.append(
                f"The user left this to you. Go on with {first}."
                if first
                else "The user left this to you."
            )
            continue
        place = _picked_place(question, answer)
        option = question.options[place] if place is not None else None
        if option is None:
            replies.append(f"The user chose: {answer.get('spec_id')}")
        elif option.spec_id:
            replies.append(f"The user chose: {option.label} ({option.spec_id})")
        else:
            replies.append(f"The user chose: {option.label}")
    return replies


def _picked_place(question: AskQuestion, answer: Mapping[str, Any]) -> int | None:
    """The place of the option of `question` that `answer` picked by its candidate id
    (`ask_candidates`); None for an answer that picked none of them."""
    if answer.get("kind") != "option":
        return None
    ids = [candidate["spec_id"] for candidate in ask_candidates(question.options)]
    picked = answer.get("spec_id")
    return ids.index(picked) if picked in ids else None


def _picked(question: AskQuestion, answer: Mapping[str, Any]) -> AskOption | None:
    """The option of `question` that `answer` picked (`_picked_place`)."""
    place = _picked_place(question, answer)
    return question.options[place] if place is not None else None


# The measure axis that holds a reading's price basis, and each basis in words.
_PRICE_AXIS = "measurementQualifier"
_PRICES = {"Nominal": "current prices", "RealValue": "constant prices"}


@dataclass(frozen=True)
class _Offer:
    """One reading a search block offers: the block's own, a count on axis
    "what" with the block's measure, or one of its `alternative_measures`."""

    block: str
    axis: str
    label: str | None
    values: dict[str, Any]


def _offers(specs: Mapping[str, Any]) -> dict[str, _Offer]:
    """Each spec_id the turn's search blocks offer, by the first block in
    stored order to offer it. A search's block precedes the block a later
    `fetch` of one of its siblings stores, so a sibling reads as the
    search offered it."""
    offers: dict[str, _Offer] = {}
    for spec_id, block in specs.items():
        resolved = (block or {}).get("resolved") or {}
        offers.setdefault(
            spec_id,
            _Offer(
                block=spec_id,
                axis="what",
                label=resolved.get("measure_label"),
                values=_axis_values(resolved),
            ),
        )
        for alternative in resolved.get("alternative_measures") or ():
            if offered := (alternative.get("definition") or {}).get("spec_id"):
                offers.setdefault(
                    offered,
                    _Offer(
                        block=spec_id,
                        axis=alternative.get("axis") or "",
                        label=alternative.get("label"),
                        values=_axis_values(alternative),
                    ),
                )
    return offers


def _picked_spec_ids(questions: list[AskQuestion], answers: dict[str, dict[str, Any]]) -> list[str]:
    """The spec_ids the user picked answering `questions`, in question order."""
    return [
        option.spec_id
        for index, question in enumerate(questions)
        if (option := _picked(question, answers.get(f"q{index + 1}") or {}))
        and option.spec_id is not None
    ]


def _block_picks(picked: Sequence[str], offers: Mapping[str, _Offer]) -> dict[str, dict[str, str]]:
    """`picked` by the block that offered each, then by axis ("what" for a
    count, the block's own reading included, "unit" for other prices), each
    axis holding its first pick in `picked` order."""
    by_block: dict[str, dict[str, str]] = {}
    for spec_id in picked:
        if (offer := offers.get(spec_id)) is not None:
            by_block.setdefault(offer.block, {}).setdefault(offer.axis, spec_id)
    return by_block


def _paired(picks: Mapping[str, str], offers: Mapping[str, _Offer]) -> tuple[str, str | None]:
    """The reading a count pick and a prices pick of one block name together,
    in words, and the spec_id of the block's offer measuring exactly that,
    None when the block lists none. `picks` holds both axes."""
    count, prices = offers[picks["what"]], offers[picks["unit"]]
    reading = f"{count.label or picks['what']} in {prices.label or picks['unit']}"
    basis = prices.values.get(_PRICE_AXIS)
    if not (count.values and basis):
        return reading, None
    wanted = {**count.values, _PRICE_AXIS: basis}
    listed = next(
        (
            spec_id
            for spec_id, offer in offers.items()
            if offer.block == prices.block and offer.values == wanted
        ),
        None,
    )
    return reading, listed


def counts_chosen(picked: Sequence[str], specs: Mapping[str, Any]) -> frozenset[str]:
    """The spec_ids whose count the user chose with `picked`, one pause's
    picks in question order, read against the turn's search blocks
    (`ORCHESTRATOR_SPECS_KEY`) as `measure_pair_reply` reads them.

    A pick of other prices alone leaves the count to search: it counts only
    beside a pick of the reading that offered it. A count picked with other
    prices of the same block adds the listed reading measuring both, the
    one the agent is told to fetch. Every other pick counts as it stands."""
    offers = _offers(specs)
    chosen = set(picked)
    for block, picks in _block_picks(picked, offers).items():
        prices = picks.get("unit")
        if prices is None:
            continue
        if "what" not in picks:
            chosen.discard(prices)
            continue
        if picks["what"] != block:
            chosen.discard(prices)
        if (listed := _paired(picks, offers)[1]) is not None:
            chosen.add(listed)
    return frozenset(chosen)


def measure_pair_reply(
    questions: list[AskQuestion],
    answers: dict[str, dict[str, Any]],
    specs: Mapping[str, Any],
    *,
    settled: Collection[str] = (),
) -> str | None:
    """The one reading the count and prices answers name together, read
    against `specs`, the turn's search blocks by spec_id
    (`ORCHESTRATOR_SPECS_KEY`).

    A count (the block's own reading or another count it offers, axis
    "what") and other prices (axis "unit") picked from one block name the
    reading with the count's `measure_axis_values` and the prices' basis:
    the line names the block's offer measuring exactly that to fetch, or,
    when the block lists none, asks for one search. A count picked alone
    that the block lists only at another price basis than its own reading
    gets a line saying so. None for any other answers: each answer's own
    line then stands. Blocks in `settled`, the parts a unit question of the
    pause covers, get no line: code has re-read them in the chosen unit.
    """
    offers = _offers(specs)
    for block, picks in _block_picks(_picked_spec_ids(questions, answers), offers).items():
        if "what" not in picks or block in settled:
            continue
        if "unit" in picks:
            reading, listed = _paired(picks, offers)
            if listed is not None:
                return (
                    f"Together the answers ask for {reading}. Fetch {listed}, which this "
                    "search listed with both."
                )
            return (
                f"Together the answers ask for {reading}, which this search did not list. "
                "Search once for that reading."
            )
        count = offers[picks["what"]]
        basis = count.values.get(_PRICE_AXIS)
        if basis and basis != offers[block].values.get(_PRICE_AXIS):
            return (
                f"This search listed {count.label or picks['what']} only in "
                f"{_PRICES.get(basis, basis)}. For other prices, search once for that reading."
            )
    return None


def _axis_values(reading: Mapping[str, Any]) -> dict[str, Any]:
    """What `reading` measures, as each `measure_axis_values` axis's ref."""
    return {
        value["axis"]: (value.get("value") or {}).get("ref")
        for value in reading.get("measure_axis_values") or ()
        if isinstance(value, Mapping) and value.get("axis")
    }

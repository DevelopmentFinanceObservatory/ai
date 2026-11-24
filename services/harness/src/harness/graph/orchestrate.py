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
    Mapping,
    Sequence,
)
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from orchestrator.agent import Orchestrator
from orchestrator.run import (
    Answered,
    HandBack,
    HandedOff,
    Opening,
    StoredHistory,
    run_orchestrator,
)
from pydantic import BaseModel, Field, model_validator
from pydantic_ai.models import Model

from harness.graph.events import AgentStatus, GraphEvent
from harness.graph.state import (
    ORCHESTRATOR_HISTORY_KEY,
    ORCHESTRATOR_PENDING_KEY,
    ORCHESTRATOR_SPECS_KEY,
    AskOption,
    AskQuestion,
    PendingAsk,
    PricedOption,
    TurnInput,
    ask_candidates,
    ask_questions,
    pending_action,
    priced_options,
    resume_answers,
    stored_mapping,
)
from harness.turn.node import Next, Node, NodeCtx, Pause, TurnKey


class AgentHandBack(BaseModel):
    """What a node hands the agent through `orchestrate`, and what it answers.

    `answers` names the agent's call the hand-back is the result of: `plan`
    for `plan_done`, `ask_user` for `ask`. None for `seed`'s hand-back, the
    runs, datasets or chip a turn opens with, which answers no call. `text`
    leads the result, and `body` holds its structured fields, which the agent
    reads beside `text` in the call's result (`response`); a seed hand-back
    is text alone. `status` is the line Studio shows while the agent reads
    it, and `failure` the refusal a plan failed with, when one did.
    """

    answers: Literal["plan", "ask_user"] | None
    status: str
    text: str
    body: dict[str, Any] = Field(default_factory=dict)
    failure: dict[str, Any] | None = None

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
            resumed = HandBack(answers=hand_back.answers, result=hand_back.response())
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
    carries the part and prices code reads for it (`priced_options`), which
    the turn's search results do not change while the pause is open."""
    questions = PendingAsk.model_validate(node_input).questions
    return Pause(
        kind="ask",
        message=questions[0].question,
        payload={"questions": ask_questions(questions, priced_options(questions, ctx.state))},
    )


async def _answered(
    ctx: NodeCtx[GraphEvent], node_input: dict[str, Any], answer: dict[str, Any]
) -> Next:
    """Hand the user's `answer` back in the options' own words
    (`answer_replies`), as the result of the agent's `ask_user` call
    (`AgentHandBack`)."""
    questions = PendingAsk.model_validate(node_input).questions
    answers = resume_answers(answer)
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
        replies = answer_replies(questions, answers, priced_options(questions, ctx.state))
        closing = _SETTLED
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
            questions, answers, stored_mapping(ctx.state, ORCHESTRATOR_SPECS_KEY)
        )
    ):
        lines.append(paired)
    hand_back = AgentHandBack(
        answers="ask_user", status="Reading your answer", text="\n".join([*lines, closing])
    )
    return Next("next", hand_back.model_dump(mode="json"))


# The agent's question to the user: a pause, and the answer handed back.
ask: Node[GraphEvent] = Node(run=_ask, resume=_answered)


def answer_replies(
    questions: list[AskQuestion],
    answers: dict[str, dict[str, Any]],
    priced: Sequence[Mapping[int, PricedOption]] = (),
) -> list[str]:
    """What the user answered to each of `questions`, in order, from `answers`
    keyed by question id (`resume_answers`): a pick in the option's own words
    with its spec_id when it carries one, the user's own words, or, for a question
    the user skipped or left unanswered, the first option as the agent's to
    go on with. A pick of constant prices no search listed, an option of
    `priced` (`priced_options`) in constant prices with no spec_id, asks
    for one search of its part in those prices."""
    replies = []
    for index, question in enumerate(questions):
        prices = priced[index] if index < len(priced) else {}
        answer = answers.get(f"q{index + 1}") or {"kind": "skipped"}
        if answer.get("kind") == "text":
            replies.append(f"The user answered in their own words: {answer['text']}")
            continue
        if answer.get("kind") != "option":
            first = question.options[0].label
            replies.append(f"The user left this to you. Go on with {first}.")
            continue
        place = _picked_place(question, answer)
        option = question.options[place] if place is not None else None
        if option is None:
            replies.append(f"The user chose: {answer.get('spec_id')}")
        elif option.spec_id:
            replies.append(f"The user chose: {option.label} ({option.spec_id})")
        elif (stamp := prices.get(place)) is not None and stamp.prices == "constant":
            replies.append(
                f"The user chose: {option.label}, which no search listed. Search once for "
                f"that part ({stamp.part}) in constant prices."
            )
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
    line then stands.
    """
    offers = _offers(specs)
    for block, picks in _block_picks(_picked_spec_ids(questions, answers), offers).items():
        if "what" not in picks:
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

"""The `orchestrate`, `route` and `ask` nodes: the agent's turn, the one action
it asked for, and its question to the user.
"""

from __future__ import annotations

from collections.abc import (
    AsyncIterator,
    Callable,
    Iterator,
    Mapping,
    Sequence,
)
from dataclasses import dataclass
from typing import Any, Literal

from google.adk import Context, Event
from google.adk.agents import LlmAgent
from google.adk.events import RequestInput
from google.genai import types
from planner.plan import SessionState

from harness.graph.state import (
    CLARIFY_INTERRUPT,
    ORCHESTRATOR_ASKED_KEY,
    ORCHESTRATOR_PENDING_KEY,
    ORCHESTRATOR_SEARCHES_KEY,
    ORCHESTRATOR_SPECS_KEY,
    ORCHESTRATOR_STATUS_FAILURE_KEY,
    ORCHESTRATOR_STATUS_KEY,
    AskOption,
    AskQuestion,
    PendingAsk,
    PricedOption,
    ask_candidates,
    ask_questions,
    interrupt_event,
    pending_action,
    resume_answers,
)
from harness.turn_records import SearchReading, open_prices


def orchestrate_node(*, orchestrator: LlmAgent) -> Callable[[Context, Any], AsyncIterator[Event]]:
    """The `orchestrate` node, running `orchestrator`."""

    async def orchestrate(ctx: Context, node_input: Any) -> AsyncIterator[Event]:
        """The orchestrator's turn: the agent runs with its tools on its own
        branch, so it sees the whole turn so far -- the user's message, its
        own calls, every hand-back -- and none of the planner's internal
        events. Reached from START with the user's message (already in the
        session), and again after `ask` and `plan_done` with text to hand
        back, which goes in as a user message of its own so it is saved
        with the session and still in view after a later pause.

        The hand-back is authored `user` on the agent's branch: ADK quotes an
        event another author wrote as that agent's data, and an unbranched
        event reaches every agent, the planner on a later `plan` call
        included. The author is restored before the agent runs, since a
        child context inherits it. An event's explicit branch also becomes
        its node's branch, which ADK hands to every node downstream; the
        status event that follows the hand-back carries the empty branch,
        which ADK reads as "clear it", so this node's output and every later
        node of the turn stay unbranched, as they are before any hand-back."""
        if isinstance(node_input, dict):
            author, ctx.event_author = ctx.event_author, "user"
            try:
                yield Event(
                    content=types.Content(
                        role="user", parts=[types.Part.from_text(text=node_input["text"])]
                    ),
                    branch=orchestrator.name,
                )
            finally:
                ctx.event_author = author
            status = {ORCHESTRATOR_STATUS_KEY: node_input["status"]}
            if "failure" in node_input:
                status[ORCHESTRATOR_STATUS_FAILURE_KEY] = node_input["failure"]
            yield Event(custom_metadata=status, branch="")
        else:
            yield Event(custom_metadata={ORCHESTRATOR_STATUS_KEY: "Reading your question"})
        answer = await ctx.run_node(orchestrator, None, override_branch=orchestrator.name)
        # A turn that ends on an `ask_user` or `plan` hand-off ends on the
        # tool's result, with no text (`orchestrator_tools._pend`), so the
        # agent's output is None. ADK records no output for an event whose
        # output is None, and a resume could not replay past this node, so
        # the node outputs empty text instead.
        yield Event(output=answer if answer is not None else "")

    return orchestrate


def route(ctx: Context, node_input: Any) -> Iterator[Event]:
    """The one action the agent asked for, or none: it answered."""
    del node_input
    pending = pending_action(ctx.state)
    ctx.state[ORCHESTRATOR_PENDING_KEY] = None
    if pending is None:
        yield Event(output=None, route="answered")
    else:
        yield Event(output=pending.model_dump(), route=pending.kind)


_SETTLED = "Each answered choice is settled. Do not ask it again."


def ask(ctx: Context, node_input: dict[str, Any]) -> Iterator[Any]:
    """Put the agent's questions to the user through one clarify interrupt,
    each under its id with its options as candidates (`ask_questions`), and
    hand the answers back in the options' own words (`answer_replies`). Each
    prices option carries the part and prices code reads for it
    (`priced_options`), which the turn's search results do not change while
    the interrupt is open. The `PendingAsk` waits under
    `ORCHESTRATOR_ASKED_KEY` while the interrupt is open, so a resume is told
    from a fresh entry."""
    result = (ctx.resume_inputs or {}).get(CLARIFY_INTERRUPT)
    held = ctx.state.get(ORCHESTRATOR_ASKED_KEY)
    if not (held and result is not None):
        ctx.state[ORCHESTRATOR_ASKED_KEY] = node_input
        questions = PendingAsk.model_validate(node_input).questions
        yield interrupt_event(
            RequestInput(
                interrupt_id=CLARIFY_INTERRUPT,
                message=questions[0].question,
                payload={
                    "questions": ask_questions(questions, priced_options(questions, ctx.state))
                },
            )
        )
        return
    ctx.state[ORCHESTRATOR_ASKED_KEY] = None
    questions = PendingAsk.model_validate(held).questions
    answers = resume_answers(result)
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
            questions, answers, ctx.state.get(ORCHESTRATOR_SPECS_KEY) or {}
        )
    ):
        lines.append(paired)
    yield Event(output={"status": "Reading your answer", "text": "\n".join([*lines, closing])})


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
        option = _picked(question, answer)
        if option is None:
            replies.append(f"The user chose: {answer.get('spec_id')}")
        elif option.spec_id:
            replies.append(f"The user chose: {option.label} ({option.spec_id})")
        elif (
            stamp := prices.get(_candidate_ids(question).index(option_id(answer)))
        ) is not None and stamp.prices == "constant":
            replies.append(
                f"The user chose: {option.label}, which no search listed. Search once for "
                f"that part ({stamp.part}) in constant prices."
            )
        else:
            replies.append(f"The user chose: {option.label}")
    return replies


def option_id(answer: Mapping[str, Any]) -> str:
    """The candidate id an option answer picked."""
    return str(answer.get("spec_id"))


def _candidate_ids(question: AskQuestion) -> list[str]:
    """The candidate id of each of `question`'s options, in order (`ask_candidates`)."""
    return [candidate["spec_id"] for candidate in ask_candidates(question.options)]


def _picked(question: AskQuestion, answer: Mapping[str, Any]) -> AskOption | None:
    """The option of `question` that `answer` picked by its candidate id
    (`ask_candidates`); None for an answer that picked none of them."""
    if answer.get("kind") != "option":
        return None
    options = dict(zip(_candidate_ids(question), question.options, strict=True))
    return options.get(answer.get("spec_id"))


# The measure axis that holds a reading's price basis, and each basis in words.
_PRICE_AXIS = "measurementQualifier"
_PRICES = {"Nominal": "current prices", "RealValue": "constant prices"}
_BASES: dict[str, Literal["current", "constant"]] = {"Nominal": "current", "RealValue": "constant"}


def priced_options(
    questions: list[AskQuestion], state: SessionState
) -> list[dict[int, PricedOption]]:
    """For each of `questions`, the place of each option offering prices, with
    the part it prices and the prices, read off the turn's search results.

    A part is a reading whose prices are open (`harness.turn_records.
    open_prices`, over the readings of `ORCHESTRATOR_SEARCHES_KEY` and the
    blocks of `ORCHESTRATOR_SPECS_KEY`). An option prices a part when it
    carries the spec_id of one of the part's entries, or carries no spec_id
    and is labelled as an entry carrying none is, case aside. An option
    carrying the spec_id of another prices a search block offers (axis
    `unit`) prices that block at its `measurementQualifier`. A question
    prices one part, the one every such option of it can price; a question
    whose options could price several parts, or none, has none.
    """
    parts = _open_parts(state)
    offers = _offers(state.get(ORCHESTRATOR_SPECS_KEY) or {})
    priced: list[dict[int, PricedOption]] = []
    for question in questions:
        found = [_option_parts(option, parts, offers) for option in question.options]
        shared = set.intersection(*(set(f) for f in found if f)) if any(found) else set()
        part = next(iter(shared)) if len(shared) == 1 else None
        priced.append(
            {}
            if part is None
            else {
                index: PricedOption(part, options[part])
                for index, options in enumerate(found)
                if part in options
            }
        )
    return priced


def _open_parts(state: SessionState) -> dict[str, list[dict[str, Any]]]:
    """The `open_prices` entries of each reading of the turn whose prices are open, by its spec_id."""
    specs = state.get(ORCHESTRATOR_SPECS_KEY) or {}
    parts: dict[str, list[dict[str, Any]]] = {}
    for stored in state.get(ORCHESTRATOR_SEARCHES_KEY) or ():
        reading = SearchReading.model_validate(stored)
        block = specs.get(reading.spec_id) if reading.spec_id is not None else None
        if block is None or reading.spec_id is None:
            continue
        entries = open_prices(
            block,
            window=reading.window.model_dump() if reading.window is not None else None,
            other_prices=[offer.model_dump() for offer in reading.other_prices],
        )
        if entries:
            parts[reading.spec_id] = entries
    return parts


def _option_parts(
    option: AskOption,
    parts: Mapping[str, list[dict[str, Any]]],
    offers: Mapping[str, _Offer],
) -> dict[str, Literal["current", "constant"]]:
    """Each part `option` can price, with the prices it offers (`priced_options`)."""
    found: dict[str, Literal["current", "constant"]] = {}
    label = option.label.strip().casefold()
    for part, entries in parts.items():
        for entry in entries:
            if (
                entry.get("spec_id") == option.spec_id
                if option.spec_id is not None
                else "spec_id" not in entry and entry["label"].casefold() == label
            ):
                found[part] = entry["prices"]
    offer = offers.get(option.spec_id) if option.spec_id is not None else None
    if not found and offer is not None and offer.axis == "unit":
        basis = _BASES.get(offer.values.get(_PRICE_AXIS) or "")
        if basis is not None:
            found[offer.block] = basis
    return found


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

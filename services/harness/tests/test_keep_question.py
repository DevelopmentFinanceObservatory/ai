"""The question whether to keep a dataset counted as published, through the planner adapter.

G7 total ODA on DAC1, whose `measure` states imputed multilateral aid as the user's
words do, cannot be counted that way. `propose_plan` pauses the planner's run on a
question built in code (`planner.tools.pause_on_keep_question`), and the entry the
user's answer resumes records it (`harness.planner_adapter.PlannerAdapter`), which
admission reads from the turn's state.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

import pytest
from harness.conversation import current_user_message
from harness.graph.state import ORCHESTRATOR_SEARCHES_KEY, ORCHESTRATOR_SPECS_KEY
from harness.planner_adapter import PlannerAdapter, _keep_text
from harness.sentences import TERMINAL_PROPOSE_CODES, refusal_sentence
from harness.settings import Settings
from harness.studio_translate import _planner_question
from harness.turn_events import PLANNER_ASK
from harness.turn_records import SearchReading
from planner.plan import DatasetRequirement, load_plan, windowed_pins
from planner.specs import PROPOSE_PLAN, REJECT_PLAN, planner_tool_specs
from planner.stated_scope import KeepAnswer, StatedScopeReader, TextReading, keep_question
from planner.tool_results import Outcome, ProblemCode
from submission.service import COUNT_AS_ASKED_OPTION, KEEP_PUBLISHED_OPTION, STATED_SCOPE_OPTIONS

from .planner.test_stated_provider_scope import (
    _DAC1_STATED,
    _G7,
    _G7_AMENDED,
    _INCLUSIVE,
    _STATED_INCLUSIVE,
    DAC1,
    _block,
    _dac1_total,
    _dataset,
    _Reader,
    _Search,
    _spec,
)
from .planner_host import PlannerHarness
from .scripted_genai import ScriptedGenai, call_part, reply, text_part
from .session_events import one_answer


def _searched(block: Mapping[str, Any]) -> dict[str, Any]:
    """The turn's search reading of `block`, as `search` stores it."""
    return SearchReading.from_part(
        {
            "variable": "G7 ODA",
            "status": "definite",
            "spec_id": block["spec_id"],
            "asked_roles": ["donor"],
        }
    ).model_dump(mode="json")


def _propose(**changes: Any) -> tuple[str, dict[str, Any]]:
    """The `propose_plan` call pinning DAC1 for G7 total ODA stated as imputed
    multilateral aid."""
    return (
        PROPOSE_PLAN,
        {
            "question": _G7,
            "answer_mode": "series",
            "decisions": [],
            "evidence_requirements": [_dac1_total(_DAC1_STATED, **changes)],
            **({} if "period" not in changes else {"amends_version": 1}),
        },
    )


_OPEN_ASK = (PLANNER_ASK, {"message": "Which years should the chart cover?"})


def _not_offering(*, titled: bool = True) -> dict[str, Any]:
    """The recorded DAC1 total with no option counting imputed multilateral aid in its
    provider choice, and with no dataset title unless `titled`."""
    block = _block("dac1")
    (choice,) = [
        c for c in block["resolved"]["scope_choices"]["choices"] if c["axis"] == "provider"
    ]
    choice["options"] = [o for o in choice["options"] if o["option_id"] != _INCLUSIVE]
    if not titled:
        del block["resolved"]["facets"]["provenance"]
    return block


class _Turn:
    """One Studio turn of the G7 question, over the blocks `names` give shown on
    record (a name, or a block), with the planner adapter its model calls answered by
    `replies` in order, its compiles answered by `_Search(**search)`, and the
    stated-scope line read at entry by `entry_reader` where given. `answers` holds
    each `propose_plan` answer, in order."""

    def __init__(
        self,
        *names: str | dict[str, Any],
        replies: list[Any],
        search: Mapping[str, Any] | None = None,
        entry_reader: StatedScopeReader | None = None,
    ) -> None:
        blocks = [_block(name) if isinstance(name, str) else name for name in names]
        self.searched = [_searched(block) for block in blocks]
        self.harness = PlannerHarness(
            {
                ORCHESTRATOR_SEARCHES_KEY: self.searched,
                ORCHESTRATOR_SPECS_KEY: {block["spec_id"]: block for block in blocks},
            }
        )
        self.scripted = ScriptedGenai(replies)
        self.answers: list[dict[str, Any]] = []
        reader = _Reader().reader()
        surface = _Search(*blocks, **(search or {}))
        self.adapter = PlannerAdapter(
            settings=Settings(),
            surface=surface,
            specs=planner_tool_specs(surface=surface, stated_scope=reader),
            client=self.scripted.client,
            on_tool_answered=self._answered,
            stated_scope=entry_reader if entry_reader is not None else reader,
        )

    def _answered(self, name: str, answer: dict[str, Any] | None) -> None:
        if name == PROPOSE_PLAN and answer is not None:
            self.answers.append(answer)

    async def run(
        self, *, words: str = _G7, answer: Any = None, **entry: Any
    ) -> dict[str, Any] | None:
        with _words(words):
            return await self.harness.run(
                self.adapter, question=_G7, readings=self.searched, answer=answer, **entry
            )

    async def state(self) -> Mapping[str, Any]:
        return await self.harness.state()

    def brief(self, index: int) -> str:
        """The text of the model request `index` of the turn."""
        return "".join(
            part.text or ""
            for content in self.scripted.requests[index].content_list
            for part in content.parts or []
        )


@contextmanager
def _words(words: str) -> Iterator[None]:
    token = current_user_message.set(words)
    try:
        yield
    finally:
        current_user_message.reset(token)


def _paused(brief: str, opening: str) -> list[dict[str, Any]]:
    """The requirements of the paused proposal the brief's line beginning `opening`
    renders, as `propose_plan` was given them."""
    (line,) = [text for text in brief.splitlines() if text.startswith(opening)]
    proposal = json.loads(line[line.index("{") :])
    return [
        {key: value for key, value in requirement.items() if value is not None}
        for requirement in proposal["evidence_requirements"]
    ]


def _calls(*calls: tuple[str, dict[str, Any]]) -> Any:
    return reply(*(call_part(name, args, id=f"{name}-{i}") for i, (name, args) in enumerate(calls)))


async def test_the_question_reaches_the_user_closed_with_two_options() -> None:
    """DAC1 alone on record: `propose_plan` pauses the run on the question, which
    names DAC1 by its catalogue title in one clean sentence and Studio shows with the
    two options and no free text, under the call's own id."""
    turn = _Turn("dac1", replies=[_calls(_propose())])

    paused = await turn.run()

    assert paused is None
    pause = turn.harness.pause
    assert pause is not None
    message = str(pause.payload["message"])
    assert ", and OECD DAC1 is not counted that way." in message
    assert "\u00b7" not in message
    assert ". is" not in message
    question = _planner_question(dict(pause.payload))
    assert [(o.id, o.label) for o in question.options] == [
        (option["id"], option["label"]) for option in STATED_SCOPE_OPTIONS
    ]
    assert question.allow_free_text is False
    (refused,) = turn.answers
    assert refused["code"] == ProblemCode.STATED_PROVIDER_DROPPED.value
    assert keep_question(dict(await turn.state())).asked == ((DAC1, _INCLUSIVE),)


async def test_keeping_the_dataset_admits_it_and_an_amendment_keeps_it() -> None:
    """The user keeps DAC1 as published: proposed again, it is admitted on its own
    count, and an amendment later in the turn keeps it with no question."""
    turn = _Turn(
        "dac1",
        replies=[
            _calls(_propose()),
            _calls(_propose()),
            reply(text_part("done")),
            _calls(_propose(period=(1980, 2024))),
            reply(text_part("done")),
        ],
    )

    await turn.run()
    await turn.run(answer=one_answer({"spec_id": KEEP_PUBLISHED_OPTION}))
    resumed = turn.brief(1)
    assert f"They kept {DAC1} counted as published" in resumed
    assert _paused(resumed, "They kept") == _propose()[1]["evidence_requirements"]
    plan = load_plan(dict(await turn.state()))
    amended = await turn.run(
        words=_G7_AMENDED,
        amendment="Start in 1980.",
        prior_plan=plan.model_dump(mode="json") if plan is not None else None,
    )

    assert [answer.get("outcome") for answer in turn.answers[1:]] == [Outcome.OK.value] * 2
    assert amended is not None
    assert turn.harness.pause is None
    state = dict(await turn.state())
    plan = load_plan(state)
    assert plan is not None
    origins = {copy.spec_id: copy.windowed_from for copy in windowed_pins(state).values()}
    assert [
        origins.get(r.source_spec_id, r.source_spec_id)
        for r in plan.evidence_requirements
        if isinstance(r, DatasetRequirement)
    ] == [DAC1]


@pytest.mark.parametrize(
    "said",
    [
        pytest.param({"spec_id": COUNT_AS_ASKED_OPTION}, id="count-as-asked"),
        pytest.param({"answers": {"q1": {"kind": "skipped"}}}, id="skip"),
        pytest.param({"skip": True}, id="dismissal"),
    ],
)
async def test_any_other_answer_releases_and_the_next_proposal_asks_nothing(said: Any) -> None:
    """The user answers anything but keeping DAC1 as published: the resumed brief says
    DAC1 was not kept, to be pinned again or declined, beside the paused proposal;
    proposed again, DAC1 is refused with no question, and the run goes on."""
    turn = _Turn(
        "dac1",
        replies=[_calls(_propose()), _calls(_propose()), reply(text_part("done"))],
    )

    await turn.run()
    resumed = await turn.run(answer=one_answer(said))

    brief = turn.brief(1)
    assert "nothing it asked is settled" not in brief
    assert f"They did not keep {DAC1} counted as published" in brief
    assert "`declined`" in brief
    assert _paused(brief, "They did not keep") == _propose()[1]["evidence_requirements"]
    assert "They kept" not in brief
    assert turn.harness.pause is None
    assert resumed is not None
    declined = turn.answers[-1]
    assert declined["code"] == ProblemCode.STATED_PROVIDER_DROPPED.value
    assert declined["action"]["call"] == PROPOSE_PLAN
    assert keep_question(dict(await turn.state())).released == ((DAC1, _INCLUSIVE),)


async def test_a_planner_ask_beside_the_proposal_asks_it_to_propose_alone() -> None:
    """The planner calls `propose_plan` and asks the user in one response: the
    planner's question pauses the run, and the proposal, which would put a second
    question, asks the planner to propose again alone."""
    turn = _Turn("dac1", replies=[_calls(_propose(), _OPEN_ASK)])

    paused = await turn.run()

    assert paused is None
    pause = turn.harness.pause
    assert pause is not None
    assert pause.payload["message"] == _OPEN_ASK[1]["message"]
    (refused,) = turn.answers
    assert refused["code"] == ProblemCode.STATED_PROVIDER_DROPPED.value
    assert "Propose again in a response with no other call" in refused["message"]
    assert keep_question(dict(await turn.state())).asked_call_id is None


@pytest.mark.parametrize("between", [None, REJECT_PLAN], ids=["after-a-refusal", "after-reject"])
async def test_an_open_planner_ask_is_never_blocked(between: str | None) -> None:
    """With a dataset on record counting the option, DAC1 is refused listing it, with
    no question. The planner's own open question after that refusal, a `reject_plan`
    between them or not, pauses the run on it."""
    replies = [
        _calls(_propose()),
        *([_calls((between, {}))] if between is not None else []),
        _calls(_OPEN_ASK),
    ]
    turn = _Turn("dac1", "g7_bilateral", replies=replies)

    paused = await turn.run()

    assert paused is None
    pause = turn.harness.pause
    assert pause is not None
    assert pause.payload["message"] == _OPEN_ASK[1]["message"]
    (listed,) = turn.answers
    assert listed["detail"]["serving"] == [_spec("g7_bilateral")]


async def test_the_brief_names_the_phrase_and_the_datasets_counting_it() -> None:
    """The planner's first brief carries the user's phrase, the option's label and
    the dataset on record offering it, read through the stated-scope reader at entry."""
    turn = _Turn("dac1", "g7_bilateral", replies=[_calls(_propose()), _calls(_OPEN_ASK)])

    await turn.run()

    brief = "".join(
        part.text or ""
        for content in turn.scripted.requests[0].content_list
        for part in content.parts or []
    )
    (line,) = [text for text in brief.splitlines() if text.startswith("The user's words")]
    assert _spec("g7_bilateral") in line
    assert DAC1 not in line
    assert "`measure`" in line


@pytest.mark.parametrize("titled", [True, False], ids=["named", "nameless"])
async def test_the_question_names_the_option_and_the_dataset_in_words(titled: bool) -> None:
    """A DAC1 whose provider choice has no option counting imputed multilateral aid:
    the question the user sees names the option by its label and the dataset by its
    name where it has one, and no id of either."""
    turn = _Turn(_not_offering(titled=titled), replies=[_calls(_propose())])

    await turn.run()

    pause = turn.harness.pause
    assert pause is not None
    message = str(pause.payload["message"])
    assert "bilateral aid plus imputed multilateral aid" in message
    assert ("OECD DAC1" in message) is titled
    assert _INCLUSIVE not in message
    assert DAC1 not in message
    assert "provider:" not in message


async def test_the_brief_names_an_option_no_dataset_on_record_offers_in_words() -> None:
    """No dataset on record can be counted as imputed multilateral aid: the brief's
    line names the option by its label, not its id."""
    turn = _Turn(_not_offering(), replies=[reply(text_part("done"))])

    await turn.run()

    (line,) = [t for t in turn.brief(0).splitlines() if t.startswith("The user's words")]
    assert "bilateral aid plus imputed multilateral aid" in line
    assert "provider:" not in line


async def test_a_reading_list_short_of_the_texts_renders_no_line() -> None:
    """The reader answers fewer readings than texts at entry: the brief carries no
    line on the stated scope, and the entry runs on."""

    async def short(texts: Sequence[str]) -> Sequence[TextReading]:
        return []

    turn = _Turn("dac1", replies=[reply(text_part("done"))], entry_reader=short)

    done = await turn.run()

    assert done is not None
    assert "The user's words" not in turn.brief(0)


async def test_a_stated_compile_that_cannot_reach_search_ends_the_entry_as_final() -> None:
    """Africa's bilateral aid stated as imputed multilateral aid, whose compile cannot
    reach Search for good: the entry ends on the code's own sentence, final for the
    turn, as an unreadable stated scope's does."""
    africa = _block("africa_bilateral")
    proposal = {
        "question": _G7,
        "answer_mode": "series",
        "decisions": [],
        "evidence_requirements": [
            _dataset("aid", africa["spec_id"], _STATED_INCLUSIVE, recipient="Africa")
        ],
    }
    turn = _Turn(
        "africa_bilateral",
        replies=[_calls((PROPOSE_PLAN, proposal))],
        search={"unreachable_after": 0},
    )

    ended = await turn.run()

    code = ProblemCode.STATED_COMPILE_UNREACHABLE
    assert code in TERMINAL_PROPOSE_CODES
    assert ended is not None
    assert ended["refused"]["code"] == code.value
    assert ended["refused"]["final"] is True
    sentence = refusal_sentence({"code": code.value, "detail": {}})
    assert ended["refused"]["text"] == sentence
    assert sentence != refusal_sentence(
        {"code": ProblemCode.STATED_SCOPE_UNREADABLE.value, "detail": {}}
    )


async def test_the_brief_names_the_summing_option_for_summed_words() -> None:
    """Words summing the bilateral and multilateral lines, with DAC1 on record: the
    brief's line names the option DAC1 sums them on, core contributions, which is not
    the imputed count Search read the words as."""
    turn = _Turn("dac1", replies=[reply(text_part("done"))])

    await turn.run(words="G7 bilateral plus multilateral ODA")

    (line,) = [t for t in turn.brief(0).splitlines() if t.startswith("The user's words")]
    assert DAC1 in line
    assert "core contributions" in line
    assert "imputed" not in line


def test_the_paused_proposal_renders_its_requirements_and_decisions_as_written() -> None:
    """The resumed brief carries the paused proposal's requirements and decisions, and
    nothing else of its arguments, with the user's characters as written."""
    proposal = {
        "question": "Aide du G7",
        "evidence_requirements": [{"id": "aid", "measure": "aide à l'Afrique"}],
        "decisions": [],
        "removed_asks": [],
    }

    text = _keep_text(KeepAnswer(pairs=((DAC1, _INCLUSIVE),), kept=False, proposal=proposal))

    assert "aide à l'Afrique" in text
    assert json.loads(text[text.index("{") :]) == {
        "evidence_requirements": proposal["evidence_requirements"]
    }

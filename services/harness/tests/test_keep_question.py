"""The question whether to keep a dataset counted as published, through the planner adapter.

G7 total ODA on DAC1, whose `measure` states imputed multilateral aid as the user's
words do, cannot be counted that way. `propose_plan` pauses the planner's run on a
question built in code (`planner.tools.pause_on_keep_question`), and the entry the
user's answer resumes records it (`harness.planner_adapter.PlannerAdapter`), which
admission reads from the turn's state.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import pytest
from harness.conversation import current_user_message
from harness.graph.state import ORCHESTRATOR_SEARCHES_KEY, ORCHESTRATOR_SPECS_KEY
from harness.planner_adapter import PlannerAdapter
from harness.settings import Settings
from harness.studio_translate import _planner_question
from harness.turn_events import PLANNER_ASK
from harness.turn_records import SearchReading
from planner.plan import DatasetRequirement, load_plan, windowed_pins
from planner.specs import PROPOSE_PLAN, REJECT_PLAN, planner_tool_specs
from planner.stated_scope import keep_question
from planner.tool_results import Outcome, ProblemCode
from submission.service import COUNT_AS_ASKED_OPTION, KEEP_PUBLISHED_OPTION, STATED_SCOPE_OPTIONS

from .planner.test_stated_provider_scope import (
    _DAC1_STATED,
    _G7,
    _G7_AMENDED,
    _INCLUSIVE,
    DAC1,
    _block,
    _dac1_total,
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


class _Turn:
    """One Studio turn of the G7 question, over `names` shown on record, with the
    planner adapter its model calls answered by `replies` in order. `answers` holds
    each `propose_plan` answer, in order."""

    def __init__(self, *names: str, replies: list[Any]) -> None:
        blocks = [_block(name) for name in names]
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
        surface = _Search(*blocks)
        self.adapter = PlannerAdapter(
            settings=Settings(),
            surface=surface,
            specs=planner_tool_specs(surface=surface, stated_scope=reader),
            client=self.scripted.client,
            on_tool_answered=self._answered,
            stated_scope=reader,
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


@contextmanager
def _words(words: str) -> Iterator[None]:
    token = current_user_message.set(words)
    try:
        yield
    finally:
        current_user_message.reset(token)


def _calls(*calls: tuple[str, dict[str, Any]]) -> Any:
    return reply(*(call_part(name, args, id=f"{name}-{i}") for i, (name, args) in enumerate(calls)))


async def test_the_question_reaches_the_user_closed_with_two_options() -> None:
    """DAC1 alone on record: `propose_plan` pauses the run on the question, which
    Studio shows with the two options and no free text, under the call's own id."""
    turn = _Turn("dac1", replies=[_calls(_propose())])

    paused = await turn.run()

    assert paused is None
    pause = turn.harness.pause
    assert pause is not None
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
    """The user answers anything but keeping DAC1 as published: proposed again, DAC1
    is refused with no question, and the run goes on."""
    turn = _Turn(
        "dac1",
        replies=[_calls(_propose()), _calls(_propose()), reply(text_part("done"))],
    )

    await turn.run()
    resumed = await turn.run(answer=one_answer(said))

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

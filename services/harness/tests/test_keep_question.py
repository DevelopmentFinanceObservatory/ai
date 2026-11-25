"""The brief's stated-scope line, and a turn paused on the keep question.

The planner reads the provider option the user's words state in its brief
(`submission.service.stated_scope_line`), read through the stated-scope reader at
entry. A turn paused on a `propose_plan` question whether to keep a dataset counted as
published (`harness.planner_adapter._PAUSED_KEEP_STATE_KEY`) resumes on the user's
answer (`harness.planner_adapter._paused_keep`), which the resumed brief states.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

import pytest
from harness.conversation import current_user_message
from harness.graph.state import ORCHESTRATOR_SEARCHES_KEY, ORCHESTRATOR_SPECS_KEY
from harness.planner_adapter import (
    _PAUSED_KEEP_STATE_KEY,
    PlannerAdapter,
    _keep_text,
    _KeepAnswer,
)
from harness.sentences import TERMINAL_PROPOSE_CODES, refusal_sentence
from harness.settings import Settings
from harness.turn_records import SearchReading
from planner.specs import PROPOSE_PLAN, planner_tool_specs
from planner.stated_scope import StatedScopeReader, TextReading
from planner.tool_results import ProblemCode
from planner.tools import PENDING_ASK_STATE_KEY

from .planner.test_stated_provider_scope import (
    _G7,
    _INCLUSIVE,
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
    """The `propose_plan` call pinning DAC1 for G7 total ODA."""
    return (
        PROPOSE_PLAN,
        {
            "question": _G7,
            "answer_mode": "series",
            "decisions": [],
            "evidence_requirements": [
                _dac1_total("official development assistance including imputed multilateral aid")
            ],
            **changes,
        },
    )


def _not_offering() -> dict[str, Any]:
    """The recorded DAC1 total with no option counting imputed multilateral aid in its
    provider choice."""
    block = _block("dac1")
    (choice,) = [
        c for c in block["resolved"]["scope_choices"]["choices"] if c["axis"] == "provider"
    ]
    choice["options"] = [o for o in choice["options"] if o["option_id"] != _INCLUSIVE]
    return block


class _Turn:
    """One Studio turn of the G7 question, over the blocks `names` give shown on
    record (a name, or a block), with the planner adapter its model calls answered by
    `replies` in order, its compiles answered by `_Search(**search)`, and the
    stated-scope line read at entry by `entry_reader` where given."""

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
        reader = _Reader().reader()
        surface = _Search(*blocks, **(search or {}))
        self.adapter = PlannerAdapter(
            settings=Settings(),
            surface=surface,
            specs=planner_tool_specs(surface=surface, stated_scope=reader),
            client=self.scripted.client,
            stated_scope=entry_reader if entry_reader is not None else reader,
        )

    async def run(self, *, words: str = _G7, answer: Any = None) -> dict[str, Any] | None:
        with _words(words):
            return await self.harness.run(
                self.adapter, question=_G7, readings=self.searched, answer=answer
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


# The id of the `propose_plan` call the turn is paused on.
_PAUSED_CALL = f"{PROPOSE_PLAN}-0"


def paused_on_keep(turn: _Turn) -> None:
    """Leave `turn` paused on the keep question about DAC1
    counted as published: the ask waiting under the paused `propose_plan` call's id,
    and the question's record holding that call's arguments."""
    turn.harness.seed_state(
        PENDING_ASK_STATE_KEY,
        {
            "call_id": _PAUSED_CALL,
            "args": {
                "message": "You asked for bilateral aid plus imputed multilateral aid, and OECD "
                "DAC1 is not counted that way. Use the data as published, or only data "
                "counted the way you asked?",
                "candidates": [],
                "options": [
                    {"id": "keep as published", "label": "Use this data as published"},
                    {"id": "count it as asked", "label": "Use only data counted the way I asked"},
                ],
                "recommended_spec_id": None,
                "rationale": None,
            },
        },
    )
    turn.harness.seed_state(
        _PAUSED_KEEP_STATE_KEY,
        {
            "offered": [],
            "asked_call_id": _PAUSED_CALL,
            "asked": [[DAC1, _INCLUSIVE]],
            "proposal": _propose()[1],
            "kept": [],
            "released": [],
        },
    )


def _paused(brief: str, opening: str) -> list[dict[str, Any]]:
    """The requirements of the paused proposal the brief's line beginning `opening`
    renders, as `propose_plan` was given them."""
    (line,) = [text for text in brief.splitlines() if text.startswith(opening)]
    return json.loads(line[line.index("{") :])["evidence_requirements"]


async def test_a_turn_paused_on_the_keep_question_resumes_on_its_answer() -> None:
    """A turn paused on the keep question, the user keeping DAC1:
    the resumed brief says `propose_plan` asked, that DAC1 was kept, and carries the
    paused proposal to propose again; the question's record is cleared."""
    turn = _Turn("dac1", replies=[reply(text_part("done"))])
    paused_on_keep(turn)

    resumed = await turn.run(answer=one_answer({"spec_id": "keep as published"}))

    assert resumed is not None
    brief = turn.brief(0)
    assert f"`{PROPOSE_PLAN}` put to them" in brief
    assert f"They kept {DAC1} counted as published" in brief
    assert _paused(brief, "They kept") == _propose()[1]["evidence_requirements"]
    assert (await turn.state()).get(_PAUSED_KEEP_STATE_KEY) is None


@pytest.mark.parametrize(
    "said",
    [
        pytest.param({"spec_id": "count it as asked"}, id="count-as-asked"),
        pytest.param({"answers": {"q1": {"kind": "skipped"}}}, id="skip"),
        pytest.param({"skip": True}, id="dismissal"),
    ],
)
async def test_any_other_answer_to_a_paused_keep_releases_the_dataset(said: Any) -> None:
    """Any answer but keeping DAC1: the resumed brief says DAC1 was not kept, to be
    pinned again declaring the option asked for, or declined, beside the paused
    proposal."""
    turn = _Turn("dac1", replies=[reply(text_part("done"))])
    paused_on_keep(turn)

    await turn.run(answer=one_answer(said))

    brief = turn.brief(0)
    assert f"They did not keep {DAC1} counted as published" in brief
    assert "`scope_options`" in brief
    assert "`declined`" in brief
    assert _paused(brief, "They did not keep") == _propose()[1]["evidence_requirements"]
    assert "They kept" not in brief


async def test_an_answer_to_another_ask_reads_no_keep() -> None:
    """A planner ask waiting under another call's id: the resumed brief reads as any
    answer to the planner's own question, and the stale record is cleared."""
    turn = _Turn("dac1", replies=[reply(text_part("done"))])
    paused_on_keep(turn)
    turn.harness.seed_state(
        PENDING_ASK_STATE_KEY,
        {"call_id": "ask_user_question-0", "args": {"message": "Which years?", "options": []}},
    )

    await turn.run(answer=one_answer("2010 to 2020"))

    brief = turn.brief(0)
    assert "They kept" not in brief
    assert "They did not keep" not in brief
    assert (await turn.state()).get(_PAUSED_KEEP_STATE_KEY) is None


async def test_the_brief_names_the_phrase_and_the_datasets_counting_it() -> None:
    """The planner's first brief carries the user's phrase, the option's label and id,
    and the dataset on record offering it, read through the stated-scope reader at
    entry, and asks for the option in `scope_options`."""
    turn = _Turn("dac1", "g7_bilateral", replies=[reply(text_part("done"))])

    await turn.run()

    (line,) = [text for text in turn.brief(0).splitlines() if text.startswith("The user's words")]
    assert _spec("g7_bilateral") in line
    assert DAC1 not in line
    assert f"({_INCLUSIVE})" in line
    assert "`scope_options`" in line


async def test_the_brief_names_an_option_no_dataset_on_record_offers_in_words() -> None:
    """No dataset on record can be counted as imputed multilateral aid: the brief's
    line names the option by its label beside its id."""
    turn = _Turn(_not_offering(), replies=[reply(text_part("done"))])

    await turn.run()

    (line,) = [t for t in turn.brief(0).splitlines() if t.startswith("The user's words")]
    assert "bilateral aid plus imputed multilateral aid" in line
    assert "which no dataset on record" in line


async def test_a_reading_list_short_of_the_texts_renders_no_line() -> None:
    """The reader answers fewer readings than texts at entry: the brief carries no
    line on the stated scope, and the entry runs on."""

    async def short(texts: Sequence[str]) -> Sequence[TextReading]:
        return []

    turn = _Turn("dac1", replies=[reply(text_part("done"))], entry_reader=short)

    done = await turn.run()

    assert done is not None
    assert "The user's words" not in turn.brief(0)


async def test_a_declared_compile_that_cannot_reach_search_ends_the_entry_as_final() -> None:
    """Africa's bilateral aid declaring imputed multilateral aid, whose compile cannot
    reach Search for good: the entry ends on the code's own sentence, final for the
    turn, as an unreadable stated scope's does."""
    africa = _block("africa_bilateral")
    proposal = {
        "question": _G7,
        "answer_mode": "series",
        "decisions": [],
        "evidence_requirements": [
            _dataset("aid", africa["spec_id"], "net ODA", recipient="Africa", declares=_INCLUSIVE)
        ],
    }
    turn = _Turn(
        "africa_bilateral",
        replies=[reply(call_part(PROPOSE_PLAN, proposal, id=_PAUSED_CALL))],
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

    text = _keep_text(_KeepAnswer(pins=(DAC1,), kept=False, proposal=proposal))

    assert "aide à l'Afrique" in text
    assert json.loads(text[text.index("{") :]) == {
        "evidence_requirements": proposal["evidence_requirements"]
    }

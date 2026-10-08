"""Grep-level content checks for the generative seams' prompts, plus
behavioral tests for `GeminiCompose`'s own parsing of a drafted intent.

The prompt checks are not behavioral tests -- there is no cheap way to
assert a model's output from a prompt alone. These pin the specific
grammar and binding guidance text so a future edit that silently drops it
fails loudly, the same role a snapshot test plays elsewhere. The rules
pinned here were measured against the deployed platform: a `stat_card`
whose select leaves a plural dimension unpinned is refused
`binding_invalid`, and an array value where the member requires a string
is refused `schema_invalid`.

The `GeminiCompose` and `GeminiConverse` tests run the real classes against
a fake `genai` client that returns a canned `response.text`, so `_generate`'s
JSON parse and each `__call__`'s own handling of the drafted object both run
for real.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
from compiler.surface import ToolCallResult, ToolSurfaceUnavailableError
from harness.dfo_faq import CANNOT_DESCRIBE_DFO, DFO_FAQ_NOT_READ
from harness.generative import (
    _COMPOSE_INSTRUCTION,
    _CONVERSE_INSTRUCTION,
    _DFO_REPLY_INSTRUCTION,
    _EDIT_INSTRUCTION,
    _NARRATE_INSTRUCTION,
    GeminiCompose,
    GeminiConverse,
    GeminiDenominatorSplit,
    GeminiEdit,
)


def test_narrate_instruction_states_the_difference_between_facts_and_question() -> None:
    assert "Compare the measure, unit, places, direction and period" in _NARRATE_INSTRUCTION
    assert "state the specific difference in the first sentence" in _NARRATE_INSTRUCTION
    assert "never raw\nidentifiers or codes" in _NARRATE_INSTRUCTION
    assert "an unproven conclusion is\nnot missing data" in _NARRATE_INSTRUCTION
    assert "When the facts match the question, add no mismatch\ncaveat" in _NARRATE_INSTRUCTION


def test_converse_instruction_states_the_role_grammar() -> None:
    assert "which side of the flow" in _CONVERSE_INSTRUCTION
    assert "donor" in _CONVERSE_INSTRUCTION
    assert "creditor" in _CONVERSE_INSTRUCTION
    assert "exporter" in _CONVERSE_INSTRUCTION
    assert "recipient" in _CONVERSE_INSTRUCTION
    assert "debtor" in _CONVERSE_INSTRUCTION
    assert "importer" in _CONVERSE_INSTRUCTION
    assert 'means "from"' in _CONVERSE_INSTRUCTION
    assert 'means "to"' in _CONVERSE_INSTRUCTION


def test_converse_instruction_keeps_a_description_of_dfo_out_of_the_first_draft() -> None:
    assert 'When about is "reply"\n  and the question is about DFO itself' in _CONVERSE_INSTRUCTION
    assert "set about_dfo and leave the\n  description out of text" in _CONVERSE_INSTRUCTION
    assert 'what data exists\n  on a topic stays about "data"' in _CONVERSE_INSTRUCTION


def test_dfo_reply_instruction_allows_only_the_figures_dfo_faq_gives() -> None:
    assert "State only the figures dfo_faq itself gives" in _DFO_REPLY_INSTRUCTION
    assert "Never state a figure" not in _DFO_REPLY_INSTRUCTION


def test_converse_instruction_states_the_time_grammar() -> None:
    assert "since 2015" in _CONVERSE_INSTRUCTION
    assert "2015 to 2020" in _CONVERSE_INSTRUCTION
    assert "2015-2020" in _CONVERSE_INSTRUCTION
    assert "latest" in _CONVERSE_INSTRUCTION


def test_converse_instruction_passes_through_unparseable_time_rather_than_omitting() -> None:
    assert "pass the user's phrase through" in _CONVERSE_INSTRUCTION
    assert "omit time" not in _CONVERSE_INSTRUCTION


def test_converse_instruction_still_uses_the_users_own_words_for_variable_and_entities() -> None:
    assert "user's own words for variable, entities, and unit" in _CONVERSE_INSTRUCTION
    assert "do not translate to" in _CONVERSE_INSTRUCTION
    assert "canonical names" in _CONVERSE_INSTRUCTION


def test_converse_instruction_states_one_part_per_variable_with_all_entities() -> None:
    assert "ONE entry per" in _CONVERSE_INSTRUCTION
    assert "one entry with two entities, not two entries" in _CONVERSE_INSTRUCTION
    assert "Two entries are for two" in _CONVERSE_INSTRUCTION
    assert "life expectancy in\n  Nigeria and Guatemala" in _CONVERSE_INSTRUCTION
    assert "a comparison of two things is two parts" not in _CONVERSE_INSTRUCTION


def test_compose_instruction_states_the_one_series_rule() -> None:
    assert "stat_card must resolve to exactly one series" in _COMPOSE_INSTRUCTION
    assert "pin EVERY dimension the" in _COMPOSE_INSTRUCTION
    assert "marks plural to one value" in _COMPOSE_INSTRUCTION
    assert "except the facet columns unit,\n    measurementMethod and provenance" in (
        _COMPOSE_INSTRUCTION
    )
    assert "refuses only when one series holds several of their values" in _COMPOSE_INSTRUCTION
    assert "series_count" in _COMPOSE_INSTRUCTION


def test_compose_instruction_routes_several_entities_over_time_to_a_line_chart() -> None:
    assert "bar_chart for a single period read across entities" in _COMPOSE_INSTRUCTION
    assert "line_chart whose selected entity dimension is the array" in _COMPOSE_INSTRUCTION
    assert "bar_chart for comparison across" not in _COMPOSE_INSTRUCTION


def test_compose_instruction_states_the_array_rules() -> None:
    assert "stat_card select carries no array" in _COMPOSE_INSTRUCTION
    assert "at most one dimension" in _COMPOSE_INSTRUCTION
    assert "line_chart up to 8 values" in _COMPOSE_INSTRUCTION
    assert "bar_chart up to 24" in _COMPOSE_INSTRUCTION
    assert "cross-product is at most 24 series" in _COMPOSE_INSTRUCTION
    assert "table may use arrays" in _COMPOSE_INSTRUCTION
    assert "more than one dimension" in _COMPOSE_INSTRUCTION


def test_edit_instruction_states_the_window_grammar() -> None:
    assert "window is an object of period labels" in _EDIT_INSTRUCTION
    assert '{"start": "2000", "end": "2023"}' in _EDIT_INSTRUCTION
    assert "Either side may be omitted" in _EDIT_INSTRUCTION
    assert '"since 2000" is {"start": "2000"}' in _EDIT_INSTRUCTION


def test_edit_instruction_updates_a_title_that_names_what_changed() -> None:
    assert "A title that names something you change" in _EDIT_INSTRUCTION
    assert '"... (2000-2023)"' in _EDIT_INSTRUCTION
    assert "open side read from the frame's coverage" in _EDIT_INSTRUCTION


def test_edit_instruction_states_the_relative_period_grammar() -> None:
    assert '{"anchor": "latest", "offset": N}' in _EDIT_INSTRUCTION
    assert '"The last three years" is {"start": {"anchor": "latest",' in _EDIT_INSTRUCTION
    assert "A window with a\n  relative side states both sides" in _EDIT_INSTRUCTION
    assert (
        '"Compare with\n  the previous year" is {"basis": [{"anchor": "latest", "offset": -1}'
        in (_EDIT_INSTRUCTION)
    )
    # Years a person names stay labels, in a window and in a change alike.
    assert '"2015\n  to 2020" is {"start": "2015", "end": "2020"}' in _EDIT_INSTRUCTION
    assert '"compare 2015 with 2020" is\n  {"basis": ["2015", "2020"]}' in _EDIT_INSTRUCTION
    assert "its title names no\n  year and no year range" in _EDIT_INSTRUCTION


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text
        self.usage_metadata = None


class _FakeModels:
    def __init__(self, text: str) -> None:
        self._text = text
        self.contents: list[Any] = []

    async def generate_content(self, **kwargs: Any) -> _FakeResponse:
        self.contents.append(kwargs["contents"])
        return _FakeResponse(self._text)


class _FakeAio:
    def __init__(self, text: str) -> None:
        self.models = _FakeModels(text)


class _FakeClient:
    """Stands in for `google.genai.Client`: only the `.aio.models.
    generate_content` path `_ModelCall._generate` calls is faked."""

    def __init__(self, text: str) -> None:
        self.aio = _FakeAio(text)


_LATEST = {"anchor": "latest", "offset": 0}


def _drafted_intent(*, select_json: str) -> dict[str, Any]:
    return {
        "intents": [
            {
                "frame_index": 0,
                "type": "stat_card",
                "select_json": select_json,
                "title": "population",
            }
        ]
    }


async def _compose_result(*, select_json: str) -> dict[str, Any]:
    client = _FakeClient(json.dumps(_drafted_intent(select_json=select_json)))
    compose = GeminiCompose(model_name="test-model", client=client)
    return await compose(
        question="population",
        frame_refs=[{"spec_id": "spec_" + "a" * 32, "content_version": "cv_" + "1" * 32}],
        frame_views=[{"available": True, "view": {}}],
        frame_provenance=[{"step": 0, "tool": "get_data", "is_final_step": True}],
    )


@pytest.mark.parametrize(
    "period",
    [
        pytest.param(
            {"window": {"start": {"anchor": "latest", "offset": -2}, "end": _LATEST}},
            id="relative-window",
        ),
        pytest.param({"window": {"start": "2015", "end": "2020"}}, id="fixed-window"),
        pytest.param(
            {"change": {"basis": [{"anchor": "latest", "offset": -1}, _LATEST]}},
            id="relative-change",
        ),
        pytest.param({"change": {"basis": ["2015", "2020"]}}, id="fixed-change"),
    ],
)
async def test_edit_passes_a_recreates_period_through_as_the_model_wrote_it(
    period: dict[str, Any],
) -> None:
    recreate = {"type": "stat_card", "title": "Aid", "select": {"donor": "country/FRA"}, **period}
    client = _FakeClient(
        json.dumps({"text": "Done.", "patch_json": None, "recreate_json": json.dumps(recreate)})
    )

    drafted = await GeminiEdit(model_name="test-model", client=client)(
        message="change the period",
        stored_intent={"type": "stat_card", "title": "Aid", "select": {"donor": "country/FRA"}},
        frame_view={"dimensions": [], "rolling": True},
    )

    assert drafted["recreate"] == recreate
    (content,) = client.aio.models.contents
    assert json.loads(content)["frame_view"]["rolling"] is True


async def test_compose_skips_an_intent_whose_select_json_is_a_json_array(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A model that emits a JSON array for select_json parses fine into a
    `list`; downstream `_select_binding_ok` calls `.items()` on it, which
    would raise. The intent is skipped instead of crashing the node."""
    with caplog.at_level("WARNING", logger="harness.generative"):
        result = await _compose_result(select_json="[1, 2]")
    assert result["intents"] == []
    messages = [r.getMessage() for r in caplog.records]
    assert any("not a JSON object" in message for message in messages)
    assert not any("did not parse" in message for message in messages)


async def _converse_result(drafted: dict[str, Any]) -> dict[str, Any]:
    client = _FakeClient(json.dumps(drafted))
    converse = GeminiConverse(model_name="test-model", client=client)
    return await converse(question="life expectancy")


def _two_part_draft() -> dict[str, Any]:
    return {
        "text": "looking into life expectancy",
        "about": "data",
        "recall": None,
        "search_parts": [
            {"variable": "life expectancy", "entities": [{"text": "Nigeria"}]},
            {"variable": "life expectancy", "entities": [{"text": "Guatemala"}]},
        ],
    }


async def test_converse_merges_two_parts_naming_one_variable_with_two_entities() -> None:
    """Two entities on one measured variable are one search part. The model
    states them as two parts often enough that the seam folds them, since
    two parts would trip the depth policy's compound floor and send a
    plain comparison to the planner."""
    result = await _converse_result(_two_part_draft())
    assert result["search_parts"] == [
        {
            "variable": "life expectancy",
            "entities": [{"text": "Nigeria"}, {"text": "Guatemala"}],
        }
    ]


async def test_converse_merge_keeps_first_seen_order_and_drops_repeats() -> None:
    """The union is in first-seen order, and an entity the second part
    repeats is not listed twice."""
    draft = _two_part_draft()
    draft["search_parts"][1]["entities"] = [
        {"text": "Guatemala"},
        {"text": "Nigeria"},
        {"text": "Benin"},
    ]
    result = await _converse_result(draft)
    assert result["search_parts"][0]["entities"] == [
        {"text": "Nigeria"},
        {"text": "Guatemala"},
        {"text": "Benin"},
    ]


async def test_converse_merge_gives_a_repeated_text_one_members_reading() -> None:
    """A text named with and without `members` is one entity asking for its
    members, in the first occurrence's position."""
    draft = _two_part_draft()
    draft["search_parts"][0]["entities"] = [{"text": "Africa"}, {"text": "Nigeria"}]
    draft["search_parts"][1]["entities"] = [
        {"text": "Africa", "members": True},
        {"text": "Africa", "members": False},
    ]
    result = await _converse_result(draft)
    assert result["search_parts"][0]["entities"] == [
        {"text": "Africa", "members": True},
        {"text": "Nigeria"},
    ]


async def test_converse_merge_drops_a_repeat_already_in_the_first_part() -> None:
    """A repeat inside the held part's own initial entities, not only a
    repeat arriving later, folds away."""
    draft = _two_part_draft()
    draft["search_parts"][0]["entities"] = [{"text": "Nigeria"}, {"text": "Nigeria"}]
    result = await _converse_result(draft)
    assert result["search_parts"][0]["entities"] == [
        {"text": "Nigeria"},
        {"text": "Guatemala"},
    ]


async def test_converse_merge_drops_a_repeat_within_one_incoming_part() -> None:
    """Both copies of a name repeated inside a single incoming part must not
    land -- only computing the seen set from the held part's entities before
    the loop would let both through."""
    draft = _two_part_draft()
    draft["search_parts"][1]["entities"] = [{"text": "Guatemala"}, {"text": "Guatemala"}]
    result = await _converse_result(draft)
    assert result["search_parts"][0]["entities"] == [
        {"text": "Nigeria"},
        {"text": "Guatemala"},
    ]


async def test_converse_keeps_parts_apart_when_their_keys_or_time_differ() -> None:
    """Only an identical question asked of two entity lists merges. A
    different time phrase or a different key set is a different part."""
    draft = _two_part_draft()
    draft["search_parts"][1]["time"] = "2020"
    result = await _converse_result(draft)
    assert len(result["search_parts"]) == 2


async def test_converse_keeps_role_bearing_parts_apart() -> None:
    """A role marks each entity's side of a flow. Swapping donor and
    recipient would reverse it, so the two lists never merge."""
    result = await _converse_result(
        {
            "text": "looking into aid",
            "about": "data",
            "recall": None,
            "search_parts": [
                {"variable": "oda", "entities": [{"text": "Kenya", "role": "to"}]},
                {"variable": "oda", "entities": [{"text": "France", "role": "from"}]},
            ],
        }
    )
    assert len(result["search_parts"]) == 2


async def test_converse_merges_parts_whose_entities_carry_a_null_role() -> None:
    """The converse schema declares `role` an optional string, so Gemini may
    emit `"role": null` for a role-free entity. That must merge exactly
    like an entity that omits the key outright, not block the merge the
    way a bare `"role" not in entity` check would."""
    result = await _converse_result(
        {
            "text": "looking into life expectancy",
            "about": "data",
            "recall": None,
            "search_parts": [
                {"variable": "life expectancy", "entities": [{"text": "Nigeria", "role": None}]},
                {
                    "variable": "life expectancy",
                    "entities": [{"text": "Guatemala", "role": None}],
                },
            ],
        }
    )
    assert result["search_parts"] == [
        {
            "variable": "life expectancy",
            "entities": [
                {"text": "Nigeria", "role": None},
                {"text": "Guatemala", "role": None},
            ],
        }
    ]


async def test_converse_keeps_a_role_free_part_out_of_a_role_bearing_held_part() -> None:
    """A role-free part arriving second could merge into an earlier
    role-bearing held part and swap its sides, if only the incoming part's
    entities were checked for a role. The held part's own entities must
    block the merge too."""
    result = await _converse_result(
        {
            "text": "looking into aid",
            "about": "data",
            "recall": None,
            "search_parts": [
                {"variable": "oda", "entities": [{"text": "Kenya", "role": "to"}]},
                {"variable": "oda", "entities": [{"text": "France"}]},
            ],
        }
    )
    assert len(result["search_parts"]) == 2


async def test_converse_keeps_a_role_bearing_part_out_of_a_role_free_held_part() -> None:
    """A role-bearing part arriving after a role-free held part never
    merges into it either, whichever side of the pair carries the role."""
    result = await _converse_result(
        {
            "text": "looking into aid",
            "about": "data",
            "recall": None,
            "search_parts": [
                {"variable": "oda", "entities": [{"text": "France"}]},
                {"variable": "oda", "entities": [{"text": "Kenya", "role": "to"}]},
            ],
        }
    )
    assert len(result["search_parts"]) == 2


def _revising_draft() -> dict[str, Any]:
    """The shape a "add Guatemala to the chart" wording drafts: the existing
    card's component_id, and one part naming both entities."""
    return {
        "text": "adding Guatemala to the chart",
        "about": "data",
        "recall": None,
        "revises": "cmp_01test0001",
        "search_parts": [
            {
                "variable": "life expectancy",
                "entities": [{"text": "Nigeria"}, {"text": "Guatemala"}],
            }
        ],
    }


async def test_converse_passes_a_drafted_revises_through_as_a_component_id() -> None:
    result = await _converse_result(_revising_draft())

    assert result["revises"] == "cmp_01test0001"
    assert result["search_parts"][0]["entities"] == [{"text": "Nigeria"}, {"text": "Guatemala"}]


async def test_converse_leaves_revises_unset_for_a_new_chart() -> None:
    """A question about a country with no reference to an existing card
    drafts no `revises`. The seam normalises that absence to None, which is
    the only value the node checks for."""
    draft = _revising_draft()
    draft["revises"] = None
    draft["search_parts"] = [{"variable": "life expectancy", "entities": [{"text": "Guatemala"}]}]

    result = await _converse_result(draft)

    assert result["revises"] is None


async def test_converse_revises_rejects_a_non_string_draft() -> None:
    """Constrained decoding can return a placeholder. The seam hands the
    node a string or nothing, so it never has to guess at a value."""
    draft = _revising_draft()
    draft["revises"] = ["cmp_01test0001"]

    result = await _converse_result(draft)

    assert result["revises"] is None


def test_converse_instruction_explains_when_to_set_revises() -> None:
    assert "revises" in _CONVERSE_INSTRUCTION
    assert "set `revises` to the" in _CONVERSE_INSTRUCTION
    assert "EXISTING entities together with the new" in _CONVERSE_INSTRUCTION
    assert "`revises` null for a new, separate chart" in _CONVERSE_INSTRUCTION
    assert '"now show me Guatemala"' in _CONVERSE_INSTRUCTION


async def test_compose_skips_an_intent_whose_select_json_does_not_parse(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Malformed (non-JSON) select_json raises from `json.loads` itself at
    the same site -- also skipped, also logged, never a crash."""
    with caplog.at_level("WARNING", logger="harness.generative"):
        result = await _compose_result(select_json="{not json")
    assert result["intents"] == []
    assert any("did not parse" in r.getMessage() for r in caplog.records)


# --- converse's reply about DFO itself ---------------------------------------


class _DraftsInTurn:
    """A `google.genai.Client` stand-in that answers each call with the next
    draft and keeps every call's `contents`."""

    def __init__(self, *drafts: dict[str, Any]) -> None:
        self._drafts = [json.dumps(draft) for draft in drafts]
        self.contents: list[str] = []
        self.aio = SimpleNamespace(models=SimpleNamespace(generate_content=self._generate))

    async def _generate(self, **kwargs: Any) -> _FakeResponse:
        self.contents.append(kwargs["contents"])
        return _FakeResponse(self._drafts[len(self.contents) - 1])


class _SkillSurface:
    """A tool surface whose every call answers `outcome`, or raises it."""

    def __init__(self, outcome: ToolCallResult | Exception) -> None:
        self.outcome = outcome
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def list_tools(self) -> list[Any]:
        return []

    async def call_tool(self, *, tool: Any, arguments: Any, on_progress: Any = None) -> Any:
        self.calls.append((tool.name, dict(arguments)))
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


_ABOUT_DFO = {"about": "reply", "about_dfo": True, "text": "", "search_parts": [], "recall": None}


@pytest.mark.parametrize(
    ("drafted", "reply"),
    [
        ("DFO is ONE Data's platform.", "DFO is ONE Data's platform."),
        # An empty second draft would close the turn blank.
        ("", DFO_FAQ_NOT_READ),
    ],
)
async def test_converse_drafts_a_reply_about_dfo_from_the_dfo_faq_skill(
    drafted: str, reply: str
) -> None:
    client = _DraftsInTurn(_ABOUT_DFO, {"text": drafted})
    surface = _SkillSurface(ToolCallResult(is_error=False, text="# The DFO\nBuilt by ONE Data."))
    converse = GeminiConverse(model_name="test-model", client=client, surface=surface)

    result = await converse(question="what is DFO?")

    assert surface.calls == [("get_skill", {"skill_id": "dfo-faq"})]
    assert json.loads(client.contents[1]) == {
        "question": "what is DFO?",
        "dfo_faq": "# The DFO\nBuilt by ONE Data.",
    }
    assert result["about"] == "reply"
    assert result["text"] == reply
    assert result["search_parts"] == []


@pytest.mark.parametrize(
    ("outcome", "reply"),
    [
        (
            ToolCallResult(is_error=True, structured_content={"code": "capability_unavailable"}),
            CANNOT_DESCRIBE_DFO,
        ),
        (
            ToolCallResult(is_error=True, structured_content={"code": "skill_not_found"}),
            CANNOT_DESCRIBE_DFO,
        ),
        (
            ToolCallResult(is_error=True, structured_content={"code": "service_unavailable"}),
            DFO_FAQ_NOT_READ,
        ),
        (ToolSurfaceUnavailableError("data-mcp unreachable"), DFO_FAQ_NOT_READ),
    ],
)
async def test_converse_says_plainly_when_dfo_faq_cannot_be_read(
    outcome: ToolCallResult | Exception, reply: str
) -> None:
    """No skill, no second draft: the reply is the sentence that says why,
    and the turn still closes on it."""
    client = _DraftsInTurn(_ABOUT_DFO)
    converse = GeminiConverse(
        model_name="test-model", client=client, surface=_SkillSurface(outcome)
    )

    result = await converse(question="what is DFO?")

    assert len(client.contents) == 1
    assert result["about"] == "reply"
    assert result["text"] == reply


async def test_converse_reads_no_skill_for_a_greeting() -> None:
    client = _DraftsInTurn({"about": "reply", "about_dfo": False, "text": "Hello!"})
    surface = _SkillSurface(ToolCallResult(is_error=False, text="unused"))
    converse = GeminiConverse(model_name="test-model", client=client, surface=surface)

    result = await converse(question="hi")

    assert surface.calls == []
    assert result["text"] == "Hello!"


async def test_the_denominator_split_sends_the_part_and_returns_the_draft() -> None:
    part = {"variable": "aid per head", "entities": [{"text": "Kenya", "role": "to"}]}
    drafted = {
        "measure": "aid",
        "measure_unit": None,
        "denominator": "population",
        "denominator_unit": None,
        "denominator_entities": [0],
        "denominator_keeps_roles": False,
    }
    client = _DraftsInTurn(drafted)
    split = GeminiDenominatorSplit(model_name="test-model", client=client)

    result = await split(question="aid per head to Kenya", part=part, message="none held")

    assert result == drafted
    assert json.loads(client.contents[0]) == {
        "question": "aid per head to Kenya",
        "part": part,
        "message": "none held",
    }

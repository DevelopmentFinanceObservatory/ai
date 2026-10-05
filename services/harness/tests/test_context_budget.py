"""Ceilings on what the planner and the translator are handed, and their baseline.

This file measures what the planner and the translator are handed and holds
each part of it to a ceiling. The bounds themselves live in production code:
`planner.plan` limits the plan and cuts user text (`MAX_PLAN_RECORD_CHARS`,
`cut_user_text`), `harness.studio_state` limits the two digests, and
`harness.planner_seam` caps the envelope and cuts the question, the amendment
and a refusal. This file's ceilings are derived from those limits or measured
from a recorded scenario, and each control test shows its ceiling can fail.

What is bounded is a segment of a request `context_probe.compose` already
measures: the system instruction, the tool declarations together, each named
declaration on its own, and the text the planner's or translator's `contents`
carries. Each realistic-input ceiling is a named constant at the top of this
file with the reason it sits where it does.

`RELAY_BY_AUTHOR_CEILINGS` is the exception to every other ceiling's rounding:
its values are exact, and zero. The bucket it bounds is the fenced
foreign-agent relay, ADK relaying the translator's own turn into the planner's
request (`google.adk.flows.llm_flows._fencing._present_other_agent_message`).
`harness.planner_seam._PLANNER_BRANCH` stamps the planner's own run with a
branch, which keeps that relay out, so the bucket measures zero and this
constant holds it there: a change that lets a fenced part reach the planner
again fails this test. A run whose events lack the stamp stops earlier,
because `harness.planner_seam._check_branches` raises when a planner event the
run appended carries a branch other than that stamp.

Every realistic-input ceiling is proved live by `test_every_ceiling_fails_when_its_segment_grows`.
The relay ceiling's own proof is `test_no_relayed_foreign_agent_text_reaches_the_planner`
paired with `test_the_relay_check_measures_a_relay_that_is_there`, since a
zero ceiling that cannot go non-zero would pass whatever the code did. A guard
test needs a fixture that can fail, or it proves nothing. The proof mutates a
captured request past the ceiling and re-checks it, leaving production code
untouched.

Those ceilings bound a realistic input and move when a renderer or a field
description grows. The legal-maximum tier, the last sections of the file, holds
the largest input the schema, the caps and the digest maxima allow to a ceiling
derived from them, each with its own control test, and `BOUNDED_CONTEXT` records
a planner call carrying every bounded input at its maximum.

The comparison is chars, not tokens, for the reasons `context_probe`'s own
docstring gives. Realistic-input ceilings are 25% above the measurement unless
the comment on the constant says otherwise, and the legal-maximum ceilings are
derived from the limits. A failure message names `just context-baseline`
because a deliberate change means re-measuring as well as raising a number. The
diff of `context_baseline.json` is the review artifact, and it shows which
bucket moved and by how much.

Three gates, following `services/data-mcp/tests/test_digest_budget.py`'s own
shape, because a checked-in baseline on its own rots. Regenerating it is a single
command that will quietly absorb any drift. So there is the pairing gate (every
scenario has a row and every row has a scenario), the ceilings as named constants
in this file that regeneration cannot move, and named mechanism assertions that
state a property, not a number. The mechanism assertions are what survives
someone running `just context-baseline` carelessly.

One of those mechanism assertions is about stability, not size, and it is
the one this whole budget is least able to feel. The planner's prefix is cached
in production: across the 44 Langfuse traces recorded since the planner and
translator began sharing one session, 234 of 237 planner calls after a turn's
first hit the cache, and 46.5% of the planner's input tokens were served from it.
That saving depends on the prefix being byte-identical from call to call, which is exactly what
`google.adk.models.gemini_context_cache_manager._generate_cache_fingerprint`
hashes, and a mismatch there is logged at DEBUG level and silently recreates the
cache. A change that made the instruction or the tool list vary per request would
cost the cached share with nothing in the log to say so, which is why
`test_the_cached_prefix_is_stable_across_a_run` exists.

The scenario is one plan-then-translate round. The seam is handed an already
drafted plan as `prior_plan`, the way `harness.graph`'s own planner node hands
it one. The planner's model loads a skill with `get_skill`, answered with one
full page, then hands that plan to `dfo_translator`. The
translator's own model calls `translate`, which runs the real mechanical loop
against a canned `no_data` search and refers the plan back. The planner then has
no `submit_plan` to reach, so the seam raises `PlannerSeamError` after its
correction rounds: an unfinished turn, which is what leaves the largest
planner request on record.
"""

from __future__ import annotations

import copy
import json
import os
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from compiler.surface import FixtureSurface, ToolCallResult
from data_mcp.definition_store import canonicalize, spec_id_for
from data_mcp.skills_client import PAGE_CHARS, continuation_line, page_of, with_closing_line
from google.adk.agents import LlmAgent
from google.adk.models.llm_request import LlmRequest
from google.genai import types
from harness import planner_seam, studio_state
from harness.graph import frame_row_ceiling
from harness.planner_seam import (
    _ENVELOPE_CANDIDATE_CAP,
    _ENVELOPE_CONTEXT,
    _ENVELOPE_FIELD_CAP,
    _ENVELOPE_KEYS,
    _ENVELOPE_LABEL_CAP,
    _ENVELOPE_OPERATION_CAP,
    _ENVELOPE_SKILL_CAP,
    _ENVELOPE_SKILL_DESCRIPTION_CAP,
    _ENVELOPE_WARNING_CAP,
    _ENVELOPE_WARNING_TEXT_CAP,
    _PLAN_SOURCE_SENTENCE_CAP,
    _SEEDED_REASON,
    PlannerSeam,
    PlannerSeamError,
    _envelope_context,
    _plan_context,
    _section,
)
from harness.seeds import seeded_envelope
from harness.settings import Settings
from harness.studio_state import (
    CANVAS_DIGEST_MAX_BYTES,
    CONVERSATION_DIGEST_MAX_BYTES,
    ConversationState,
)
from harness.wire import (
    MAX_OPERATION_PHRASE_CHARS,
    MAX_SEED_COUNT,
    MAX_SEED_LIST_ITEMS,
    MAX_SEED_OPERATIONS,
    MAX_SEED_RECORD_CHARS,
    MAX_SEED_ROLES,
    MAX_SEED_SENTENCE_CHARS,
    MAX_SEED_SOURCE_CHARS,
    MAX_SEED_STAT_VARS,
    MAX_SEED_TEXT_CHARS,
    MAX_SEEDS,
    MAX_TARGET_YEAR,
    ComponentType,
    CompositionActor,
    CompositionBlock,
)
from planner.agent import build_planner_tools
from planner.operations import AggregateOperation, RetrieveOperation, Translation
from planner.plan import (
    CUT_MARKER,
    MAX_DECISION_DESCRIPTION_CHARS,
    MAX_DECISION_OPTIONS,
    MAX_DECISION_RATIONALE_CHARS,
    MAX_ENTITIES,
    MAX_ENTITY_NAME_CHARS,
    MAX_ENTITY_ROLE_CHARS,
    MAX_MEASURE_CHARS,
    MAX_OPTION_CHARS,
    MAX_PLAN_DECISIONS,
    MAX_PLAN_RECORD_CHARS,
    MAX_PLAN_REQUIREMENTS,
    MAX_PLAN_ROUNDS,
    MAX_PREDICATES,
    MAX_QUESTION_CHARS,
    MAX_SOURCE_CHARS,
    SEEDED_SOURCES_STATE_KEY,
    AggregateRequirement,
    DatasetRequirement,
    Decision,
    Entity,
    Period,
    Plan,
    ResolvedBy,
    Unit,
    save_plan_rounds,
    save_seeded_sources,
    seeded_sources,
)
from planner.records import GroupBy, Reduction
from planner.settings import AgentSettings
from planner.workflow import EvidenceBinding, SelectedSource, ToolId, WorkflowStep
from planner.workflow import Workflow as PlanWorkflow
from translator.discovery import compact_source

from .bounded_seeds import (
    bounded_seed,
    fullest_described_seed,
    fullest_recorded_seed,
    most_recorded_entities,
    read_seed,
    recorded_seed,
)
from .context_probe import (
    BOUNDING_PATTERNS,
    RELAY_MARKER,
    SEAM_BUCKET,
    SEAM_MARKER,
    RecordingLlm,
    bucket_chars,
    canonical_tools,
    compose,
    contents_leading,
    plan_record,
    relay_buckets,
    system_instruction_text,
    unbounded_paths,
)
from .context_probe import (
    record_chars as _record_chars,
)
from .test_planner_seam import _NO_DATA_SEARCH_RESPONSE, _gdp_plan, _SeamHarness

# The root conftest.py chdirs every test into a tmp_path, so this must be absolute.
_BASELINE_PATH = Path(__file__).resolve().parent / "context_baseline.json"

_REGENERATE = "just context-baseline"


# Ceilings -----------------------------------------------------------------------
#
# Chars, 25% above today's measurement unless the comment says otherwise.

# The planner's system instruction: the role, the tool protocol, the closing
# rules, the skills paragraph and the translator paragraph, assembled by
# `planner.agent.planner_instruction`. Measured 14,161 with the skill tools and
# the translator attached. The margin is 8.5% and covers a hand edit of
# model-facing wording, since these paragraphs are rewritten for model
# behaviour a sentence at a time.
PLANNER_SYSTEM_INSTRUCTION_CEILING = 15_400

# Every planner tool declaration together: 42,002 (`propose_plan` 35,883,
# `dfo_translator` 916, and 5,203 across the smaller ones). The margin is 10%.
PLANNER_DECLARATIONS_TOTAL_CEILING = 46_300

# `propose_plan`'s own declaration: the `Decision` and `EvidenceRequirement`
# schemas with their `$defs` tree, and the patch and intent arguments. Measured
# 29,578, of which the plan's limits (`maxLength` and `maxItems` on 31 fields)
# are about 520. It sits in the static prefix ADK caches, so it is billed once
# per cache creation. Its field descriptions are what the planner writes a plan
# from. The margin is 23%.
PROPOSE_PLAN_DECLARATION_CEILING = 36_300

# `dfo_translator`'s declaration, built by ADK from the sub-agent's name,
# description and wrapped `request` argument. Measured 916.
TRANSLATOR_DECLARATION_CEILING = 1_200

# The smaller planner declarations at once: `adk_request_input` 2,370,
# `get_skill_file` 1,084, `submit_plan` 840, `get_skill` 651 and `reject_plan`
# 258.
MINOR_DECLARATIONS_TOTAL_CEILING = 6_400

# The translator's own request: its single `translate` declaration (1,998) plus
# ADK's closing `set_model_response` (3,666).
TRANSLATOR_OWN_DECLARATIONS_CEILING = 7_100

# The planner's `contents` history on its largest captured generation, less
# the skill text `SKILL_PAGE_CEILING` bounds on its own (1,849 chars, the
# fifth generation of the realistic scenario).
PLANNER_CONTENTS_CEILING = 2_300

# The `get_skill` answer in the planner's `contents`, bucketed as
# `functionResponse:get_skill`: one full page of a skill file
# (`data_mcp.skills_client.PAGE_CHARS`, 32,000 chars) plus the closing line
# naming the next call, as ADK serialises it. Measured 32,676 on a page of
# short Markdown lines, where escaping each line break adds a char. This bounds
# one call. Nothing caps how many skill pages one planner entry reads besides
# the seam's event limit and entry deadline.
SKILL_PAGE_CEILING = 40_800

# The seam's own brief, bucketed as `text(seam-brief)` on its opening words
# (`harness.planner_seam._ENVELOPE_CONTEXT`): the resolve envelope, the two
# digests, the plan being changed and the question. A session sends it on its
# first planner call and only the ask thereafter, so this bounds what that one
# call is billed in full before ADK's cache covers it.
# `harness.planner_seam._render_envelope` and `_plan_context` render the
# envelope and the prior plan as line-rendered digests, which is what this
# bounds.
#
# Measured 1,150 here, where the fixture's plan is two requirements and the
# envelope is `{}`. That is the wrong end of the range to size a ceiling from:
# the seam's own docstring probe records the realistic upper end at nine
# requirements, and the section that dominates it is the prior plan, re-sent on
# every fresh render. `test_the_seam_brief_holds_a_nine_requirement_plan_under_its_ceiling`
# measures that upper end, with a `candidates` envelope at every cap
# `harness.planner_seam` sets: 9,941 chars for a draft plan and 11,437 for a
# compiled one (the plan section alone is 8,723 -> 6,942 raw-to-digest for the
# draft, 22,609 -> 8,438 for the compiled one). The ceiling is 25% above the
# compiled case, as this file's own convention says, not above the fixture's
# two-requirement number.
SEAM_BRIEF_CEILING = 14_300

# The translator's `contents` history on its largest captured generation
# (343 chars). The translator starts fresh on every `dfo_translator` call, so
# this stays near one turn's own tool exchange.
TRANSLATOR_CONTENTS_CEILING = 500

# The fenced foreign-agent relay in the planner's `contents`, ceiling per author,
# not one for the bucket. `dfo_translator`'s payload was 509 chars here
# and 17,243 on the reference trace. ADK's own preamble is on its own row.
RELAY_BY_AUTHOR_CEILINGS: dict[str, int] = {
    # `harness.planner_seam._PLANNER_BRANCH` stamps the planner's own run with
    # a branch, which removes the translator's relay. Zero is the guarantee
    # itself, not a rounded measurement: any non-zero reading means ADK is
    # relaying another agent's turn into the planner again. Without that
    # branch stamp it measured 509 chars of payload here.
    "dfo_translator": 0,
    # ADK's own preamble, about 409 chars emitted once per relayed event, so it
    # scales with the count of relays, not with what they said. It measured
    # 818 here (2 events) without the branch stamp. Kept on its own row and
    # not merged into the translator's, because a fix that moved the payloads
    # while leaving the framing would otherwise report a partial win against a
    # number the preamble kept up.
    "untagged": 0,
}

# Authors this scenario is expected to see at all. A new author is a new row and
# must be added deliberately, with a ceiling: `_by_author_type` is what keeps an
# unexpected one from appearing as an unceilinged number.
RELAY_AUTHORS = frozenset(RELAY_BY_AUTHOR_CEILINGS)

# The agent name `context_probe` gives ADK's own relay preamble, which carries the
# markers without an `[author]` tag.
_UNTAGGED_RELAY = "untagged"


# The measured scenario ----------------------------------------------------------

# The third recorded scenario: the same round, with the seam's brief carrying a
# maximal resolve envelope and the canvas and conversation digests at their
# maxima. The other two run with an empty envelope and no digests.
BOUNDED_CONTEXT = "bounded-context"


def _seam() -> PlannerSeam:
    """The seam the recorded round runs on, over a canned data-mcp surface.

    Built here, not reused from `test_planner_seam`, whose own
    `_translator_seam` is private to the scenarios it serves.
    `thinking_sample_rate=0` pins every request
    this drives to `AGENT_THINKING_CONFIG` -- the default `random.random`
    draw would otherwise occasionally swap in `DEFAULT_THINKING_CONFIG`
    (planner.settings), which moves `thinking_level` on and off the
    serialised request and, with it, this file's own `total_chars`.
    """
    return PlannerSeam(
        settings=Settings(),
        surface=FixtureSurface(
            schemas=[],
            responses={"search": _NO_DATA_SEARCH_RESPONSE, "get_skill": _skill_page()},
        ),
        agent_settings=AgentSettings(thinking_sample_rate=0.0),
    )


def _skill_page() -> ToolCallResult:
    """A `get_skill` answer as long as data-mcp serves: one full page and its closing line.

    The skill text is short Markdown lines, longer than one page, so
    `data_mcp.skills_client.page_of` cuts a full page and the answer ends on the
    line naming the next call.
    """
    line = "- A caveat the answer has to state, with the definition it rests on.\n"
    text = line * (2 * PAGE_CHARS // len(line))
    page = page_of(text, 0)
    closing = continuation_line(
        skill_id="dfo-faq",
        path="SKILL.md",
        page=page,
        length=len(text),
        digest="sha256:" + "0" * 64,
    )
    return ToolCallResult(is_error=False, text=with_closing_line(page.text, closing))


def _scripts() -> tuple[list[Any], list[Any]]:
    """The planner's script and the translator's, for one round.

    The planner loads a skill, hands the plan to the translator, and then
    produces text, which the seam corrects twice before raising. The translator calls `translate` once per its own generation. The
    later entries only exist to close a turn the seam's own stall guard may
    already have ended.
    """
    planner_script: list[Any] = [
        [("get_skill", {"skill_id": "dfo-faq"})],
        [("dfo_translator", {"request": "find debt service for kenya"})],
    ]
    planner_script.extend(["nothing to submit"] * 4)
    # `translator.agent._before_translator_model` opens each translator turn on
    # `translate` itself, so the model is asked only to close it.
    translator_script: list[Any] = ["no data", "no data"]
    return planner_script, translator_script


async def _recorded_round(
    *,
    scenario: str,
    decision: Callable[..., Any],
    requirement: Callable[..., Any],
    derived_requirement: Callable[..., Any],
    context: dict[str, Any] | None = None,
) -> dict[str, tuple[list[dict[str, Any]], list[Any]]]:
    """One plan-then-translate round, with every request both models were handed.

    `context` is what the seam's brief carries besides the plan: the resolve
    `envelope` and the `canvas_digest` and `conversation_digest`. Left out, the
    envelope is empty and there are no digests, the smallest brief the seam
    renders.

    Returns `{"planner": (records, requests), "translator": (records, requests)}`:
    each agent's `context_probe.compose` records in call order, and the raw
    `LlmRequest` objects the stability assertions compare byte for byte.
    """
    draft = _gdp_plan(decision, requirement, derived_requirement)
    seam = _seam()
    planner_script, translator_script = _scripts()
    planner_llm = RecordingLlm(script=planner_script, agent_name="planner", scenario=scenario)
    translator_llm = RecordingLlm(
        script=translator_script, agent_name="translator", scenario=scenario
    )
    seam._agent.model = planner_llm
    cast(LlmAgent, seam._agent.sub_agents[0]).model = translator_llm

    with pytest.raises(PlannerSeamError):
        await _SeamHarness().run(
            seam,
            question=draft.question,
            prior_plan=draft.model_dump(mode="json"),
            **({"envelope": {}} if context is None else context),
        )

    return {
        "planner": (planner_llm.record, planner_llm.captured),
        "translator": (translator_llm.record, translator_llm.captured),
    }


async def _all_records(
    *,
    decision: Callable[..., Any],
    requirement: Callable[..., Any],
    derived_requirement: Callable[..., Any],
) -> list[dict[str, Any]]:
    """Every record from the realistic scenario, without `BOUNDED_CONTEXT`."""
    measured = await _recorded_round(
        scenario="realistic",
        decision=decision,
        requirement=requirement,
        derived_requirement=derived_requirement,
    )
    return [record for agent in ("planner", "translator") for record in measured[agent][0]]


# Baseline -----------------------------------------------------------------------


def _compose_baseline(records: list[dict[str, Any]]) -> dict[str, Any]:
    """The baseline payload: every record, ordered so a diff reads by scenario."""
    return {
        "records": sorted(
            records, key=lambda item: (item["scenario"], item["agent"], item["call_index"])
        )
    }


def _baseline_key(record: dict[str, Any]) -> str:
    return f"{record['scenario']}/{record['agent']}[{record['call_index']}]"


def _write_baseline(records: list[dict[str, Any]]) -> None:
    # `indent=2` plus a trailing newline matches the `pretty-format-json`
    # pre-commit hook's own formatting (`--indent=2 --no-sort-keys`), so it
    # leaves this file alone, the same way `digest_baseline.json` relies on.
    _BASELINE_PATH.write_text(json.dumps(_compose_baseline(records), indent=2) + "\n")


@pytest.fixture
async def measured(
    decision: Callable[..., Any],
    requirement: Callable[..., Any],
    derived_requirement: Callable[..., Any],
) -> dict[tuple[str, str], tuple[list[dict[str, Any]], list[Any]]]:
    """Every round at once, keyed `(scenario, agent)` -> `(records, raw requests)`.

    Both loads come from one run: a test that reads a record and a test that
    reads a raw request are looking at the same capture, so the stability
    assertions and the size assertions can never describe different rounds.
    The second scenario, `BOUNDED_CONTEXT`, hands the seam a maximal resolve
    envelope and both digests at their maxima.

    The baseline rewrite is env-gated the way `test_digest_budget.py`'s own is
    and wrapped by the `context-baseline` `just` recipe. It happens here, in the
    fixture every test reads through, so the ceilings are checked against
    the freshly written file, not a skipped gate.
    """
    measured: dict[tuple[str, str], tuple[list[dict[str, Any]], list[Any]]] = {}
    scenarios: list[tuple[str, dict[str, Any] | None]] = [
        ("realistic", None),
        (BOUNDED_CONTEXT, _bounded_context()),
    ]
    for scenario, context in scenarios:
        round_measured = await _recorded_round(
            scenario=scenario,
            decision=decision,
            requirement=requirement,
            derived_requirement=derived_requirement,
            context=context,
        )
        for agent in ("planner", "translator"):
            measured[(scenario, agent)] = round_measured[agent]

    if os.environ.get("CONTEXT_BASELINE_UPDATE") == "1":
        _write_baseline([record for records, _requests in measured.values() for record in records])
    return measured


@pytest.fixture
def records(
    measured: dict[tuple[str, str], tuple[list[dict[str, Any]], list[Any]]],
) -> list[dict[str, Any]]:
    """Every composition record from every scenario, sizes only."""
    return [record for records, _requests in measured.values() for record in records]


def test_the_checked_in_baseline_matches_today_s_measurement(
    records: list[dict[str, Any]],
) -> None:
    """Gate 1 (pairing), then gate 2's numbers: the baseline and the tree agree.

    Pairing first, both ways: every measured request has a baseline row and every
    row has a measured request. Neither can drift out of the other, so a scenario
    that stopped running cannot leave a stale row behind looking like agreement.

    Then the numbers, field by field so a failure names the segment that moved.
    This gate is the one `just context-baseline` rewrites, which is exactly why
    the mechanism assertions above it exist.
    """
    baseline = json.loads(_BASELINE_PATH.read_text())
    expected = {_baseline_key(record): record for record in baseline["records"]}
    measured = {_baseline_key(record): record for record in records}

    missing = sorted(set(measured) - set(expected))
    stale = sorted(set(expected) - set(measured))
    assert not missing, (
        f"requests with no baseline row: {missing}. "
        f"A deliberate change means {_REGENERATE}, which re-measures every row."
    )
    assert not stale, (
        f"baseline rows for requests no longer made: {stale}. "
        f"A deliberate change means {_REGENERATE}, which drops what the code no longer sends."
    )

    for key in sorted(measured):
        before, after = expected[key], copy.deepcopy(measured[key])
        # Field by field, not a whole-record comparison, so a failure names
        # the segment that moved, without printing two records to diff by eye.
        assert before["total_chars"] == after["total_chars"], (
            f"{key}: total_chars {before['total_chars']} -> {after['total_chars']}. "
            f"Run {_REGENERATE} to record a deliberate change."
        )
        assert _segments(before) == _segments(after), (
            f"{key}: a segment moved without the total moving. "
            f"before={_segments(before)} after={_segments(after)}. Run {_REGENERATE}."
        )
        assert before["contents"]["buckets"] == after["contents"]["buckets"], (
            f"{key}: contents buckets changed. "
            f"before={before['contents']['buckets']} after={after['contents']['buckets']}. "
            f"Run {_REGENERATE}."
        )


def _segments(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "system_instruction_chars": record["system_instruction_chars"],
        "declaration_total_chars": record["declaration_total_chars"],
        "declarations": record["declarations"],
    }


# The ceilings -------------------------------------------------------------------


def _largest(records: list[dict[str, Any]], *, scenario: str, agent: str) -> dict[str, Any]:
    """The record with the most `contents` chars in one scenario and agent,
    the request worth bounding.
    """
    matching = [
        record for record in records if record["scenario"] == scenario and record["agent"] == agent
    ]
    assert matching, f"no {agent} request captured for {scenario} (the round did not run)"
    return max(matching, key=lambda item: item["contents"]["total_chars"])


def _declaration(record: dict[str, Any], name: str) -> int:
    for item in record["declarations"]:
        if item["name"] == name:
            return item["chars"]
    raise AssertionError(f"no declaration named {name!r} on {_baseline_key(record)}")


def _declare(record: dict[str, Any]) -> list[tuple[str, int, int]]:
    """`(label, measured, ceiling)` for every boundary one request can be held to."""
    return [
        (
            "planner system instruction",
            record["system_instruction_chars"],
            PLANNER_SYSTEM_INSTRUCTION_CEILING,
        ),
        (
            "planner declarations",
            record["declaration_total_chars"],
            PLANNER_DECLARATIONS_TOTAL_CEILING,
        ),
        (
            "propose_plan declaration",
            _declaration(record, "propose_plan"),
            PROPOSE_PLAN_DECLARATION_CEILING,
        ),
        (
            "dfo_translator declaration",
            _declaration(record, "dfo_translator"),
            TRANSLATOR_DECLARATION_CEILING,
        ),
        (
            "minor planner declarations",
            _minor_declarations(record),
            MINOR_DECLARATIONS_TOTAL_CEILING,
        ),
        (
            "planner contents",
            record["contents"]["total_chars"] - bucket_chars(record, SKILL_BUCKET),
            PLANNER_CONTENTS_CEILING,
        ),
        ("seam brief", bucket_chars(record, SEAM_BUCKET), SEAM_BRIEF_CEILING),
        ("skill page", bucket_chars(record, SKILL_BUCKET), SKILL_PAGE_CEILING),
    ]


def _declare_for_translator(record: dict[str, Any]) -> list[tuple[str, int, int]]:
    """The same, for the translator's own request."""
    return [
        (
            "translator declarations",
            record["declaration_total_chars"],
            TRANSLATOR_OWN_DECLARATIONS_CEILING,
        ),
        ("translator contents", record["contents"]["total_chars"], TRANSLATOR_CONTENTS_CEILING),
    ]


# The bucket `context_probe` files a `get_skill` answer under.
SKILL_BUCKET = "functionResponse:get_skill"


# Everything a planner request declares other than `propose_plan` and
# `dfo_translator`: the two declarations bounded individually above, not
# lumped into this total.
_NAMED_DECLARATIONS = ("propose_plan", "dfo_translator")


def _minor_declarations(record: dict[str, Any]) -> int:
    return sum(
        item["chars"] for item in record["declarations"] if item["name"] not in _NAMED_DECLARATIONS
    )


def test_no_segment_grows_past_its_ceiling(records: list[dict[str, Any]]) -> None:
    """Every bound holds on the largest request each agent was handed.

    The planner's request has to carry the full skill page the scenario serves,
    so a round that stopped calling `get_skill` cannot clear the skill ceiling
    by measuring nothing.
    """
    checked: list[tuple[str, int, int]] = []
    planner = _largest(records, scenario="realistic", agent="planner")
    assert bucket_chars(planner, SKILL_BUCKET) > PAGE_CHARS, (
        f"{_baseline_key(planner)} carries no full `get_skill` page: the round did not load one"
    )
    checked.extend(_declare(planner))
    translator = _largest(records, scenario="realistic", agent="translator")
    checked.extend(_declare_for_translator(translator))

    over = [
        f"{label}: {measured} chars > ceiling {ceiling}"
        for label, measured, ceiling in checked
        if measured > ceiling
    ]
    assert not over, (
        "a segment of the planner or translator request outgrew its stated bound:\n  "
        + "\n  ".join(over)
        + f"\nA deliberate change means {_REGENERATE}, then this file's ceiling constant."
    )


def test_no_relayed_foreign_agent_text_reaches_the_planner(
    records: list[dict[str, Any]],
) -> None:
    """No author's relayed text reaches the planner, each author checked on its own.

    `harness.planner_seam._PLANNER_BRANCH` stamps the planner's own run with a
    branch, which keeps ADK from relaying the translator's turn in. This is
    the guard on that mechanism.

    Per author, not per bucket: the translator's payloads and ADK's own preamble
    are separate numbers, so a change that silenced one while leaving the other
    cannot report as progress. The preamble is the one to watch, since it only
    exists to introduce a relay and scales with the number of them, not
    with what any said.
    """
    planner_records = [
        record
        for record in records
        if record["scenario"] == "realistic" and record["agent"] == "planner"
    ]
    assert planner_records, "no realistic planner request captured (the round did not run)"
    for record in planner_records:
        found = relay_buckets(record)
        unexpected = sorted(set(found) - RELAY_AUTHORS)
        assert not unexpected, (
            f"{_baseline_key(record)}: relayed text authored by {unexpected}, which "
            "has no ceiling here. The guarantee covers `dfo_translator` and the "
            "`untagged` preamble. A third author is a new surface and needs its own "
            "entry in `RELAY_BY_AUTHOR_CEILINGS`."
        )
        over = [
            f"{author}: {chars} chars > ceiling {RELAY_BY_AUTHOR_CEILINGS[author]}"
            for author, chars in sorted(found.items())
            if chars > RELAY_BY_AUTHOR_CEILINGS[author]
        ]
        assert not over, (
            f"{_baseline_key(record)}: the planner is carrying relayed foreign-agent "
            "text again:\n  "
            + "\n  ".join(over)
            + "\nADK is relaying another agent's turn into this request. The fix that "
            "removed it is the branch stamp in `harness.planner_seam`. Reverting that "
            "stamp reproduces exactly this failure."
        )


def test_the_relay_check_measures_a_relay_that_is_there() -> None:
    """The zero ceilings are only meaningful if the bucket can be non-zero.

    The control for `RELAY_BY_AUTHOR_CEILINGS`: a bucket that always read zero,
    or a marker match that had drifted from what `_fencing` emits, would pass
    `test_no_relayed_foreign_agent_text_reaches_the_planner` for the wrong reason
    and would pass most convincingly on the day the relay came back. The relay
    built here is ADK's own framing: one tagged part naming the author, then
    the fenced payload, and every author's bound must reject it.
    """
    request = _request(
        tool_names=["propose_plan"],
        texts=[],
        relay=("dfo_translator", 4_000),
    )
    record = compose(request, agent="planner", scenario="realistic", call_index=0)

    found = relay_buckets(record)
    assert set(found) == {"dfo_translator", "untagged"}, found
    over = [
        author for author, chars in found.items() if chars > RELAY_BY_AUTHOR_CEILINGS.get(author, 0)
    ]
    assert sorted(over) == ["dfo_translator", "untagged"], (
        f"a zero ceiling did not fail on a relay built to exceed it: {found}"
    )


# Mechanism assertions -----------------------------------------------------------
#
# Properties, not numbers, so `just context-baseline` cannot absorb a
# regression by rewriting them. Modelled on `test_digest_budget.py`'s named
# digest-size regressions (a digest that leaks frame size, a candidate list that
# grows without a cap, a grounding ref that restates its sentence).


def test_the_cached_prefix_is_stable_across_a_run(measured: Any) -> None:
    """Each consecutive pair of requests carries a byte-identical cached prefix.

    Between two requests in one run, `config.system_instruction` is byte-identical
    and `config.tools` is byte-identical after canonicalisation. Those are the
    inputs to ADK's own cache fingerprint:
    `google.adk.models.gemini_context_cache_manager._generate_cache_fingerprint`
    (`gemini_context_cache_manager.py:310-384`) hashes the model, the cache scope,
    the system instruction, the canonicalised tools, the tool config and the
    first N contents. On a mismatch its caller (`:305`) logs
    `"Cache content fingerprint mismatch"` at **debug** level and silently
    recreates the cache: a fail-open degradation with no loud path, which is
    why this assertion is not redundant with ADK's own behaviour.

    What it protects is measured: across the 44 Langfuse traces recorded since the planner and translator
    began sharing one session, 234 of 237 planner calls after a turn's first
    hit the cache, and 46.5% of the planner's input tokens were served from
    it. The change most
    likely to break it is setting `static_instruction` or moving to an
    `InstructionProvider`, either of which can make the instruction vary per
    request and destroy that saving with nothing in the log to say so.

    Two things ADK already handles, so nothing here tests for them: it sorts each
    tool's declarations by name and the tool list by its canonical JSON before
    hashing, so reordering is safe; and the declarations only have to be
    invariant within one run, not between conversations, so growth between runs
    is safe.

    The translator's pair is checked in
    `tests/translator/test_reasoning_effort.py`, on a ranking turn. A clean
    translation reaches the translator's model once (the opener supplies
    `translate`), so this scenario captures one translator request and no
    pair. A ranking turn is where the translator makes two generations.
    """
    _records, requests = measured[("realistic", "planner")]
    assert len(requests) >= 2, (
        f"realistic/planner: only {len(requests)} request(s) captured, so there is no "
        "pair to compare. Two consecutive requests are what the cache prefix needs."
    )
    for index in range(len(requests) - 1):
        before, after = requests[index], requests[index + 1]
        first, second = f"request {index}", f"request {index + 1}"

        instruction_before = system_instruction_text(before)
        instruction_after = system_instruction_text(after)
        assert instruction_before == instruction_after, (
            f"realistic/planner: the system instruction differs between {first} and "
            f"{second} ({len(instruction_before)} vs {len(instruction_after)} chars). "
            "ADK's cache fingerprint covers it and a mismatch is only logged at debug "
            "level, so this silently costs the cached share."
        )

        tools_before = canonical_tools(before)
        tools_after = canonical_tools(after)
        assert tools_before == tools_after, (
            f"realistic/planner: the tool declarations differ between {first} and "
            f"{second} ({len(tools_before)} vs {len(tools_after)} canonical chars). "
            "ADK's cache fingerprint covers them, and a mismatch is only logged at debug "
            "level, so this silently costs the cached share."
        )


def test_the_leading_contents_of_a_run_are_only_ever_appended_to(measured: Any) -> None:
    """Each planner request's `contents` begins with the previous one's, unchanged.

    The cache fingerprint also covers the first N contents, so a run whose
    earlier turn was rewritten, not merely appended to, would miss the cache
    from the rewrite onward. Every request ADK assembles here is the session's
    whole history, so this is a real property of the served path and not an
    artefact of the fixture: `harness.planner_seam._propose` runs each
    correction round in the same session.

    The guarantee this protects: a correction round's supersession edit stays
    transient and is never persisted, and a version that persisted it would
    show up here as a leading content that changed.

    The planner is only checkable this way when the run makes more than one
    generation. Where it does not, the test reports that directly, so it
    never passes vacuously.
    """
    _records, requests = measured[("realistic", "planner")]
    assert len(requests) >= 2, "realistic/planner: no pair of requests to compare"
    signatures = [contents_leading(request) for request in requests]
    for index in range(len(signatures) - 1):
        before, after = signatures[index], signatures[index + 1]
        assert after[: len(before)] == before, (
            f"realistic/planner: request {index + 1}'s contents do not begin with request "
            f"{index}'s. {len(before)} leading contents compared. The first difference is at "
            f"index {next(i for i, (a, b) in enumerate(zip(before, after, strict=False)) if a != b)}. "
            "ADK's cache fingerprint covers the leading contents, and a rewrite here "
            "invalidates the cache silently."
        )


# Failable fixtures --------------------------------------------------------------
#
# Each ceiling above is proved live by handing the same comparison a request
# built to exceed it and confirming every bound reports. A ceiling this could
# not fail would be one that passes whatever the code does, which is the failure
# mode a guard test with no failable fixture always risks. The request is
# fabricated here, not by editing production code to inflate the real scenario.

# Size of every segment in the deliberately inflated request: past every ceiling
# above, `propose_plan`'s own 29,578 included.
_INFLATED = 50_000


def _request(
    *,
    tool_names: list[str],
    texts: list[str],
    relay: tuple[str, int] | None = None,
    seam: int | None = None,
    skill: int | None = None,
) -> Any:
    """A `LlmRequest` carrying the named declarations and text parts, oversized.

    `relay`, when given, prepends an ADK-shaped fenced relay of
    `(author, chars)`: the framing `_fencing._present_other_agent_message`
    emits, so the relay bucket is exercised against the real shape, not
    one this test invented. `seam`, when given, prepends a seam-brief part of
    that many chars, `harness.planner_seam._ENVELOPE_CONTEXT`'s own opening
    words, which is what `context_probe` buckets on, so the seam ceiling is
    exercised against a part that really is one. `skill`, when given, appends
    a `get_skill` answer of that many chars.
    """
    declarations = [
        types.FunctionDeclaration(name=name, description="x" * _INFLATED) for name in tool_names
    ]
    contents = [types.Content(role="user", parts=[types.Part(text=text)]) for text in texts]
    if seam is not None:
        contents.insert(
            0,
            types.Content(role="user", parts=[types.Part(text=SEAM_MARKER + "\n" + "z" * seam)]),
        )
    if skill is not None:
        contents.append(
            types.Content(
                role="user",
                parts=[
                    types.Part(
                        function_response=types.FunctionResponse(
                            name="get_skill", response={"result": "k" * skill}
                        )
                    )
                ],
            )
        )
    if relay is not None:
        author, chars = relay
        contents.insert(
            0,
            types.Content(
                role="user",
                parts=[
                    types.Part(
                        text=(
                            f"[{author}] called tool `translate` with parameters:\n"
                            f"{RELAY_MARKER}\n{{}}\n<<<END_QUOTED_AGENT_CONTENT>>>"
                        )
                    ),
                    types.Part(text=RELAY_MARKER + "\n" + "y" * chars),
                ],
            ),
        )
    return LlmRequest(
        model="probe",
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=types.Content(parts=[types.Part(text="s" * _INFLATED)]),
            tools=[types.Tool(function_declarations=declarations)],
        ),
    )


def test_every_ceiling_fails_when_its_segment_grows() -> None:
    """Each planner bound rejects a request past it, so no ceiling is dead.

    One oversized declaration per name `_declare` looks up, so `_declaration`
    finds each of them, plus enough text to clear the contents ceiling and a
    `get_skill` answer past its own.
    """
    request = _request(
        tool_names=["propose_plan", "dfo_translator", "adk_request_input", "submit_plan"],
        texts=["t" * _INFLATED] * 3,
        seam=_INFLATED,
        skill=SKILL_PAGE_CEILING + 1,
    )
    inflated = compose(request, agent="planner", scenario="realistic", call_index=0)
    reported = {label for label, measured, ceiling in _declare(inflated) if measured > ceiling}
    expected = {label for label, _, _ in _declare(inflated)}
    assert reported == expected, (
        "a ceiling did not fail on a request built to exceed it, so that bound is not "
        f"live. Missing from the reported set: {sorted(expected - reported)}"
    )

    translator_request = _request(
        tool_names=["translate", "set_model_response"], texts=["t" * _INFLATED]
    )
    inflated_translator = compose(
        translator_request, agent="translator", scenario="realistic", call_index=0
    )
    translator_reported = {
        label
        for label, measured, ceiling in _declare_for_translator(inflated_translator)
        if measured > ceiling
    }
    assert translator_reported == {
        "translator declarations",
        "translator contents",
    }, f"a translator ceiling is not live: {sorted(translator_reported)}"


# The seam brief at its realistic upper end --------------------------------------
#
# The recorded scenario above runs a two-requirement plan with an empty
# envelope, which is the smallest brief this seam ever renders and the wrong
# end to size a ceiling from. `planner_seam.py`'s own docstring probe records
# nine requirements as the realistic upper end, so the ceiling above is sized
# from that and this test measures it directly.

# Requirements in the plan the seam probe records as the realistic upper end.
_SEAM_PROBE_REQUIREMENTS = 9


def _nine_requirement_plan() -> Any:
    """A nine-requirement plan, the size `planner_seam`'s own probe records.

    Each dataset requirement feeds one aggregate, so the plan is nine of each
    and a decision per indicator: the shape a question about a handful of
    measures produces, not a synthetic one.
    """
    decisions = [
        Decision(
            slot=f"slot_{i}",
            description=f"Which measure to use for indicator {i}, and at what units.",
            options=[f"annual measure {i}", f"total measure {i}", f"per-capita measure {i}"],
            chosen=f"annual measure {i}",
            resolved_by=ResolvedBy.PLANNER_DEFAULT,
            rationale=f"Annual measure {i} is the standard published frequency for this indicator.",
        )
        for i in range(_SEAM_PROBE_REQUIREMENTS)
    ]
    requirements: list[Any] = []
    for i in range(_SEAM_PROBE_REQUIREMENTS):
        requirements.append(
            DatasetRequirement(
                id=f"req_{i}",
                decision_ids=[f"slot_{i}"],
                measure=f"indicator {i} total value in current US dollars",
                entities=[
                    Entity(name="Guatemala", role="country"),
                    Entity(name="Kenya", role="country"),
                ],
                period=Period(start=2000, end=2024),
                unit=Unit(currency="USD", prices="current"),
                source=f"World Bank WDI indicator {i}",
            )
        )
        requirements.append(
            AggregateRequirement(
                id=f"agg_{i}",
                decision_ids=[f"slot_{i}"],
                input_id=f"req_{i}",
                reduction=Reduction.SUM,
                group_by=(GroupBy.ENTITY, GroupBy.PERIOD),
            )
        )
    return Plan(
        question=(
            "compare population and gdp for guatemala since the year 2000, with growth "
            "and shares across nine indicators"
        ),
        decisions=decisions,
        evidence_requirements=requirements,
    )


def _capped_envelope() -> dict[str, Any]:
    """A `candidates` envelope at every cap `harness.planner_seam` sets.

    The candidates branch carries more than the `definite` branch does: six
    candidates each with a label, plus the pin-predicate facts on each, so it
    is the envelope shape the ceiling has to hold on.
    """
    return {
        "status": "candidates",
        "spec_id": None,
        "choice_points": 6,
        "parts": 2,
        "wide": True,
        "stat_var_shape_kind": "statvar_schema",
        "stat_var_item_count": 0,
        "entity_roles": ["observationAbout"],
        "stat_var_truncated": False,
        "candidates": [
            {
                "spec_id": f"spec_{index:032x}",
                "label": (
                    "Population of Guatemala, annual, from the World Bank's World "
                    "Development Indicators, measured as the total resident population"
                ),
                "wide": index % 2 == 0,
            }
            for index in range(1, 7)
        ],
        "warnings": [
            {"code": f"warn_{i}", "severity": "warn", "message": "m" * 250} for i in range(6)
        ],
    }


def _seam_brief(
    *,
    fresh: bool,
    envelope: dict[str, Any],
    prior_plan: dict[str, Any] | None,
    canvas_digest: dict[str, Any] | None = None,
    conversation_digest: dict[str, Any] | None = None,
    question: str = "compare population and gdp for guatemala since the year 2000",
    refusal: dict[str, Any] | None = None,
    amendment: str | None = None,
) -> str:
    """One `_message` render, the way `PlannerSeam.__call__` asks for it."""
    seam = PlannerSeam(
        settings=Settings(),
        surface=FixtureSurface(schemas=[], responses={}),
    )
    return seam._message(
        fresh=fresh,
        question=question,
        envelope=envelope,
        prior_plan=prior_plan,
        refusal=refusal,
        amendment=amendment,
        canvas_digest=canvas_digest,
        conversation_digest=conversation_digest,
    )


def test_the_seam_brief_holds_a_nine_requirement_plan_under_its_ceiling() -> None:
    """The realistic upper end, measured: a nine-requirement plan in the brief.

    Both a draft plan and a compiled one, since a compiled plan carries the
    `translation` and `workflow` the digest renders a summary of. Each is
    rendered against a `candidates` envelope at every cap, which is the larger
    envelope shape, and held to `SEAM_BRIEF_CEILING` on its own.

    This is the test the ceiling is sized from. The recorded scenario's own
    brief is two requirements and an empty envelope, so a ceiling taken from it
    would pass a change that let the brief grow with the plan it carries.
    """
    draft = _nine_requirement_plan()
    dumped = draft.model_dump(mode="json")
    envelope = _capped_envelope()

    for label, plan in (("draft", dumped), ("compiled", _compiled(draft))):
        render = _seam_brief(fresh=True, envelope=envelope, prior_plan=plan)
        assert len(render) <= SEAM_BRIEF_CEILING, (
            f"a nine-requirement {label} plan renders a {len(render)}-char seam brief, "
            f"past the {SEAM_BRIEF_CEILING} ceiling this test derives from that same case"
        )
        # A guard on the guard: a render that dropped the plan entirely would
        # clear the ceiling for the wrong reason.
        assert dumped["question"] in render
        assert all(decision["slot"] in render for decision in dumped["decisions"])


def _compiled(plan: Any) -> dict[str, Any]:
    """`plan` with a compiled translation and workflow, as `harness.graph`'s
    execute edge leaves it: the largest shape a prior plan ever reaches."""
    spec_ids = [f"spec_{index + 1:032x}" for index in range(_SEAM_PROBE_REQUIREMENTS)]
    operations: list[Any] = []
    steps: list[Any] = []
    for index, spec_id in enumerate(spec_ids):
        operations.append(
            RetrieveOperation(id=f"req_{index}", requirement_id=f"req_{index}", spec_id=spec_id)
        )
        operations.append(
            AggregateOperation(
                id=f"agg_{index}",
                requirement_id=f"agg_{index}",
                input=f"req_{index}",
                reduction=Reduction.SUM,
                group_by=(GroupBy.ENTITY, GroupBy.PERIOD),
            )
        )
        steps.append(
            WorkflowStep(
                id=f"req_{index}",
                operation_id=f"req_{index}",
                tool=ToolId(server="data-mcp", name="get_data"),
                args={"spec_id": spec_id},
                input_schema_hash="a" * 16,
            )
        )
        steps.append(
            WorkflowStep(
                id=f"agg_{index}",
                operation_id=f"agg_{index}",
                tool=ToolId(server="analysis-mcp", name="aggregate"),
                args={"aggregation": "sum", "group_by": ["time_period", "entity"]},
                bindings={"frame_ref": {"from_step": f"req_{index}", "output": "frame_ref"}},
                input_schema_hash="b" * 16,
            )
        )
    translation = Translation(operations=operations, terminal=[f"agg_{i}" for i in range(9)])
    workflow = PlanWorkflow(
        steps=steps,
        sources=[
            SelectedSource(
                spec_id=spec_id,
                sentence=f"indicator {index} total value in current US dollars",
                has_data=True,
            )
            for index, spec_id in enumerate(spec_ids)
        ],
        evidence_bindings=[
            EvidenceBinding(requirement_id=f"req_{index}", step_id=f"req_{index}", spec_id=spec_id)
            for index, spec_id in enumerate(spec_ids)
        ],
        terminal_steps=[f"agg_{i}" for i in range(9)],
    )
    return plan.with_translation(translation=translation, workflow=workflow).model_dump(mode="json")


def test_the_seam_brief_ceiling_fails_when_the_plan_section_grows() -> None:
    """The control for the seam ceiling: a brief past it is rejected.

    `SEAM_BRIEF_CEILING` bounds a rendering this seam builds, so a test that
    only ever measured the real scenario would pass whatever the renderer did
    to the real scenario. This hands `_declare` a record built the same way
    `test_every_ceiling_fails_when_its_segment_grows` builds one (a seam part
    past the ceiling), and requires the bound to report.
    """
    request = _request(
        tool_names=["propose_plan", "dfo_translator", "adk_request_input", "submit_plan"],
        texts=["t" * _INFLATED] * 3,
        seam=_INFLATED,
    )
    record = compose(request, agent="planner", scenario="realistic", call_index=0)
    measured = bucket_chars(record, SEAM_BUCKET)
    assert measured > SEAM_BRIEF_CEILING, (
        f"a fabricated seam part of {_INFLATED} chars measured {measured}, which does not "
        "exceed the ceiling (the bucket is not reading the seam part it is meant to)"
    )
    reported = {label for label, value, ceiling in _declare(record) if value > ceiling}
    assert "seam brief" in reported, (
        "the seam-brief ceiling did not report on a request carrying an oversized seam "
        f"part. Reported set was {sorted(reported)}"
    )


# The legal maximum ---------------------------------------------------------------
#
# Everything above sizes a ceiling from realistic inputs, and drifts when a
# renderer or a field description grows. It proves nothing about an input the
# schema allows and nobody writes. This tier measures the largest legal input to
# each part of the planner's brief and holds it to a ceiling derived from the
# limits that make it legal: `planner.plan.MAX_PLAN_RECORD_CHARS` for the plan,
# `planner_seam`'s caps for the envelope, `studio_state`'s maxima for the two
# digests. The ceilings are computed in this file from those constants, so a
# limit that moves changes its ceiling with it.
#
# The tier covers the model-authored fields at their bound plus the renderer's
# own caps. What the compiler and data-mcp put on a plan is bounded in the
# render: a workflow's source `sentence` is cut at `_PLAN_SOURCE_SENTENCE_CAP`,
# and its other rendered strings are ids that the `Identifier` and `SpecId`
# patterns bound. The live question, the amendment text and a refusal are
# unbounded on the wire, so `PlannerSeam._message` cuts each at
# `MAX_QUESTION_CHARS` (`cut_user_text`) and the tier feeds each one far past it.

_FIXED_IDENTIFIER_CHARS = BOUNDING_PATTERNS[r"^[a-z][a-z0-9_]{0,63}$"]
_SPEC_ID_CHARS = BOUNDING_PATTERNS[r"^spec_[0-9a-f]{32}$"]


def _plan_framing_chars() -> int:
    """What `_plan_context` adds to the model-authored record's serialised size, at most.

    `MAX_PLAN_RECORD_CHARS` bounds the plan as compact JSON, and every rendered
    value is a part of that JSON: a string is written raw or as JSON, a
    structure as the same compact JSON, and a list of strings joined by `, `
    costs no more than its quoted and comma-separated form. So the render adds
    only the labels around those values and the compiled workflow. Each term
    below counts the literal text of one line in full, though the JSON's own
    keys already cover most of it, and the render cuts `description` and
    `rationale` below their JSON size.

    The workflow is bounded by one source and one evidence binding per
    requirement, the source's `sentence` cut at `_PLAN_SOURCE_SENTENCE_CAP`.
    """
    opening = (
        len("\nThe plan being changed:\nQuestion: ")
        + len("\nversion 999999 (amended from version 999999)")
        + len("\nstatus: pending_approval")
        + len("\napproved_revision: ")
        + 64
        + len("\nDecisions (24):")
        + len("\nEvidence requirements (24):")
        + len("\n")
    )
    decision = (
        len("\n- ")
        + len(": ")
        + len("\n  description: ")
        + len("\n  options: ")
        + len("\n  rationale: ")
        + len("\n  resolved_by: ")
        + len("\n  fixes: ")
    )
    requirement = (
        len("\n- ")
        + len(" (comparison) cites ")
        + len("\n  ")
        # `key=value` and the space before the next field, for the widest kind's fields.
        + 12 * len("=")
        + 12 * len(" ")
    )
    workflow = (
        len("\nCompiled sources:")
        + MAX_PLAN_REQUIREMENTS
        * (len("\n  ") + _SPEC_ID_CHARS + len(" -- ") + _PLAN_SOURCE_SENTENCE_CAP)
        + MAX_PLAN_REQUIREMENTS
        * (
            len("\n- evidence: ")
            + _FIXED_IDENTIFIER_CHARS
            + len(" from ")
            + _SPEC_ID_CHARS
            + len(" (step ")
            + _FIXED_IDENTIFIER_CHARS
            + len(")")
        )
    )
    return opening + MAX_PLAN_DECISIONS * decision + MAX_PLAN_REQUIREMENTS * requirement + workflow


# The plan section at its largest: the model-authored record at its size limit
# (16,000) and the framing (12,708, of which the compiled workflow is 9,522), so
# 28,708.
LEGAL_MAXIMUM_PLAN_SECTION_CEILING = MAX_PLAN_RECORD_CHARS + _plan_framing_chars()


def _with_workflow(record: dict[str, Any]) -> dict[str, Any]:
    """`record` with a compiled workflow added: one source and one binding per requirement.

    Each source sentence is three times the render's cap, so the render's own
    cut is what sets that part of the size.
    """
    ids = [requirement["id"] for requirement in record["evidence_requirements"]]
    return {
        **record,
        "workflow": {
            "sources": [
                {"spec_id": _spec(index), "sentence": "S" * (_PLAN_SOURCE_SENTENCE_CAP * 3)}
                for index in range(len(ids))
            ],
            "evidence_bindings": [
                {
                    "requirement_id": requirement_id,
                    "spec_id": _spec(index),
                    "step_id": f"step_{index}",
                }
                for index, requirement_id in enumerate(ids)
            ],
        },
    }


def _spec(index: int) -> str:
    return "spec_" + f"{index:032x}"


def _authored_chars(plan: dict[str, Any]) -> int:
    """The serialised size `MAX_PLAN_RECORD_CHARS` limits, of a plan dict with a workflow."""
    return _record_chars(
        {key: plan[key] for key in ("question", "decisions", "evidence_requirements")}
    )


def _legal_maximum_plan_record() -> dict[str, Any]:
    """A plan at its size limit exactly, in the worst case the render meets, compiled."""
    return _with_workflow(
        Plan.model_validate(plan_record(MAX_PLAN_RECORD_CHARS)).model_dump(mode="json")
    )


def test_the_largest_legal_plan_renders_under_the_ceiling_its_limit_derives() -> None:
    """The plan's size limit bounds the plan section, and every item still renders.

    `Plan` limits the model-authored record to `MAX_PLAN_RECORD_CHARS`, and
    `_plan_context` renders all of it, because `propose_plan` patches a plan by
    `slot` and `id` and the planner has to see what it is patching. So the size
    of this section is a fact about the largest legal plan, and the ceiling is
    computed from the limit. The last decision and the last requirement are
    checked by name, so a renderer that dropped items to fit under the ceiling
    still fails. The second plan is the one whose framing is largest: every
    requirement cites every decision, and every decision states `MAX_DECISION_OPTIONS`
    options.
    """
    plan = _legal_maximum_plan_record()
    framing_heavy = _with_workflow(plan_record(options=MAX_DECISION_OPTIONS, cites_all=True))

    section = _plan_context(plan)
    heavy_section = _plan_context(framing_heavy)

    assert len(section) <= LEGAL_MAXIMUM_PLAN_SECTION_CEILING, (
        f"a plan at its size limit renders a {len(section):,}-char plan section, past the "
        f"{LEGAL_MAXIMUM_PLAN_SECTION_CEILING:,} its limit derives. The limit or the renderer "
        "moved: re-derive `_plan_framing_chars` against it."
    )
    assert len(heavy_section) <= _authored_chars(framing_heavy) + _plan_framing_chars(), (
        f"the framing-heavy plan renders {len(heavy_section):,} chars, past its own "
        f"{_authored_chars(framing_heavy):,} plus {_plan_framing_chars():,} of framing"
    )
    # A ceiling far above what the largest plan renders bounds nothing: it must
    # be within reach of the plan it was derived for.
    assert len(section) > LEGAL_MAXIMUM_PLAN_SECTION_CEILING * 0.5, (
        f"the largest legal plan renders {len(section):,} chars, under half of the derived "
        f"{LEGAL_MAXIMUM_PLAN_SECTION_CEILING:,}: the maximal plan or the derivation is off"
    )
    last_decision = plan["decisions"][-1]
    last_requirement = plan["evidence_requirements"][-1]
    assert f"- {last_decision['slot']}: {last_decision['chosen']}" in section
    assert f"- {last_requirement['id']} ({last_requirement['kind']}) cites" in section


def test_the_plan_ceiling_fails_when_one_field_outgrows_the_limit() -> None:
    """The control for the derived plan ceiling: a field past the size limit breaks it.

    A plan record with one field made far longer than the limit allows, rendered
    as a stored record would be (`Plan` would refuse to construct it), has to go
    over. A ceiling that could not fail would pass whatever the limit allowed.
    """
    plan = _legal_maximum_plan_record()
    plan["evidence_requirements"][0]["measure"] = "m" * (LEGAL_MAXIMUM_PLAN_SECTION_CEILING + 1)

    assert len(_plan_context(plan)) > LEGAL_MAXIMUM_PLAN_SECTION_CEILING


def _maximal_envelope() -> dict[str, Any]:
    """An envelope with every field the renderer names far past its cap.

    Fifty candidates against a cap of `_ENVELOPE_CANDIDATE_CAP`, each label
    100,000 characters against `_ENVELOPE_LABEL_CAP`, fifty skills against
    `_ENVELOPE_SKILL_CAP`, each description 100,000 characters against
    `_ENVELOPE_SKILL_DESCRIPTION_CAP`, and fifty warnings against
    `_ENVELOPE_WARNING_CAP`, each message 100,000 characters against
    `_ENVELOPE_WARNING_TEXT_CAP`. Every other field, a `spec_id`, a skill's
    `id`, a warning's `code` and `severity`, the pin-predicate facts, eight
    entity roles and eight open axes, is 100,000 characters against
    `_ENVELOPE_FIELD_CAP`. `parts` is a string here,
    so the render does not add the multi-part reason paragraph, which is fixed
    text. Fifty seeds against `MAX_SEEDS`, each with its own width facts, eight
    open axes and fifty operations against `_ENVELOPE_OPERATION_CAP`, and every
    field 100,000 characters, make the envelope a seeded one, which adds
    `_SEEDED_REASON`, also fixed text.
    """
    long = "x" * _BIG
    return {
        "status": long,
        "spec_id": long,
        "choice_points": long,
        "parts": long,
        "wide": long,
        "stat_var_shape_kind": long,
        "stat_var_item_count": long,
        "entity_roles": [long] * 8,
        "stat_var_truncated": long,
        "open_entity_axes": [{"column": long, "members": _BIG}] * 8,
        "candidates": [{"spec_id": long, "label": long, "wide": True} for _ in range(50)],
        "skills": [{"id": long, "description": long} for _ in range(50)],
        "warnings": [{"code": long, "severity": long, "message": long} for _ in range(50)],
        "seeds": [
            {
                "spec_id": f"{index}{long}",
                "sentence": long,
                "window": long,
                "showed": long,
                "coverage": long,
                "wide": long,
                "stat_var_shape_kind": long,
                "stat_var_item_count": long,
                "entity_roles": [long] * 8,
                "stat_var_truncated": long,
                "open_entity_axes": [{"column": long, "members": _BIG}] * 8,
                "operations": [
                    {"verb": long, "phrase": long, "over": long, "target_year": long}
                    for _ in range(50)
                ],
            }
            for index in range(50)
        ],
    }


def _envelope_ceiling() -> int:
    """What `_envelope_context` renders for an envelope with these keys, at most.

    The frame, then one line for each of the ten scalar fields at
    `_ENVELOPE_FIELD_CAP`, then the three capped lists: `_ENVELOPE_CANDIDATE_CAP`
    candidate lines of a `spec_id` at the field cap, `(wide)` and a label of
    `_ENVELOPE_LABEL_CAP`, `_ENVELOPE_SKILL_CAP` skill lines of an `id` at the
    field cap and a description of `_ENVELOPE_SKILL_DESCRIPTION_CAP`, and
    `_ENVELOPE_WARNING_CAP` warning lines of a `severity` and a `code` at the
    field cap and a message of `_ENVELOPE_WARNING_TEXT_CAP`. Each list ends
    with an "and N more" line. The seeded block follows
    (`_seeded_section_ceiling`). The
    keys the renderer does not name are outside it (see
    `test_the_maximal_envelope_names_every_key_the_renderer_does`), and so is the
    reason paragraph a searched resolve that is not one part adds.
    """
    frame = len(_ENVELOPE_CONTEXT.format(envelope="", reasons=""))
    scalar_lines = 10 * (len("\n- stat_var_shape_kind: ") + _ENVELOPE_FIELD_CAP)
    candidates = len("\n- candidates (999):") + _ENVELOPE_CANDIDATE_CAP * (
        len("\n  - ") + _ENVELOPE_FIELD_CAP + len(" (wide)") + len(" -- ") + _ENVELOPE_LABEL_CAP
    )
    more = len("\n  - and 999 more")
    skills = len("\n- skills (999):") + _ENVELOPE_SKILL_CAP * (
        len("\n  - ") + _ENVELOPE_FIELD_CAP + len(" -- ") + _ENVELOPE_SKILL_DESCRIPTION_CAP
    )
    warnings = len("\n- warnings (999):") + _ENVELOPE_WARNING_CAP * (
        len("\n  - [")
        + _ENVELOPE_FIELD_CAP
        + len("] ")
        + _ENVELOPE_FIELD_CAP
        + len(": ")
        + _ENVELOPE_WARNING_TEXT_CAP
    )
    return (
        frame
        + scalar_lines
        + candidates
        + more
        + skills
        + more
        + warnings
        + more
        + _seeded_section_ceiling()
    )


def _seeded_section_ceiling() -> int:
    """What a seeded envelope adds to its render, at most: `_SEEDED_REASON` and
    the seeds block (`harness.planner_seam._render_seeds`) at every cap. An
    operation line is longest over an axis that names no party, with a target
    year."""
    operation = (
        len("\n  operation: ")
        + _ENVELOPE_FIELD_CAP
        + len(" over ")
        + _ENVELOPE_FIELD_CAP
        + len(" to ")
        + _ENVELOPE_FIELD_CAP
        + len(' ("')
        + _ENVELOPE_LABEL_CAP
        + len('")')
    )
    seed = (
        len("\n- spec_id: ")
        + _ENVELOPE_FIELD_CAP
        + len(" -- ")
        + _ENVELOPE_LABEL_CAP
        + len("\n  window: ")
        + _ENVELOPE_FIELD_CAP
        + len("\n  showed: ")
        + _ENVELOPE_FIELD_CAP
        + len("\n  coverage: ")
        + _ENVELOPE_FIELD_CAP
        + 5 * (len("\n  stat_var_shape_kind: ") + _ENVELOPE_FIELD_CAP)
        + len("\n  open_entity_axes: ")
        + _ENVELOPE_FIELD_CAP
        + _ENVELOPE_OPERATION_CAP * operation
        + len("\n  and 999 more")
    )
    more = len("\n- and 999 more")
    reason = _SEEDED_REASON.format(ceiling=frame_row_ceiling())
    return len(reason) + len("\nSeeded datasets (999):") + MAX_SEEDS * seed + more


MAXIMAL_ENVELOPE_CEILING = _envelope_ceiling()


def test_the_maximal_envelope_names_every_key_the_renderer_does() -> None:
    """The envelope ceiling is derived for the keys `_render_envelope` names.

    A key the renderer starts to name has to enter the derivation, and a key it
    drops leaves a line in the ceiling that nothing renders. Either way the
    fixture and the renderer's key set disagree, and this fails.
    """
    assert set(_maximal_envelope()) == set(_ENVELOPE_KEYS)


def test_the_resolve_envelope_at_far_past_its_caps_renders_under_the_ceiling() -> None:
    """The envelope's caps bound its section however much the resolve returned."""
    envelope = _maximal_envelope()

    section = _envelope_context(
        envelope, refusal=None, spec_id_elected=False, spec_id_elected_label=None
    )

    assert len(section) <= MAXIMAL_ENVELOPE_CEILING, (
        f"a resolve envelope of 50 candidates, 50 skills and 50 warnings renders {len(section):,} chars, "
        f"past the {MAXIMAL_ENVELOPE_CEILING:,} its caps derive"
    )
    assert len(section) > MAXIMAL_ENVELOPE_CEILING * 0.9, (
        f"the maximal envelope renders {len(section):,} chars, under 90% of the derived "
        f"{MAXIMAL_ENVELOPE_CEILING:,}: the envelope or the derivation is off"
    )
    assert "and 44 more" in section
    assert section.count("and 45 more") == 2


@pytest.mark.parametrize(
    "cap",
    [
        "_ENVELOPE_LABEL_CAP",
        "_ENVELOPE_FIELD_CAP",
        "_ENVELOPE_SKILL_DESCRIPTION_CAP",
        "_ENVELOPE_WARNING_TEXT_CAP",
    ],
)
def test_the_envelope_ceiling_fails_when_a_cap_is_removed(
    cap: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The control for the envelope ceiling: with one cap gone, it is exceeded."""
    monkeypatch.setattr(planner_seam, cap, 10**9)

    section = _envelope_context(
        _maximal_envelope(), refusal=None, spec_id_elected=False, spec_id_elected_label=None
    )

    assert len(section) > MAXIMAL_ENVELOPE_CEILING


# -- what a seed puts outside the brief ------------------------------------------------
#
# A seed's text also reaches the model in tool results and stays in session
# state. Each is measured here with seeds at their bounds
# (`harness.wire`'s `MAX_SEED_*`), against a ceiling derived from those bounds.


# The description a model reads of one seed is `compact_source` of its source.
# It is held to `MAX_SEED_SOURCE_CHARS` where the seed is read, and discovery
# stamps the requirement's measure on it as `searched_as`, at most
# `MAX_MEASURE_CHARS`. What a whole `translate` result and the `planner:plan`
# state hold for a plan that retrieves `MAX_SEEDS` such seeds through
# `MAX_PLAN_REQUIREMENTS` requirements is measured on `run_translate` itself,
# in `tests/translator/test_mechanical.py`.
SEEDED_SOURCE_DESCRIPTION_CEILING = MAX_SEED_SOURCE_CHARS + len('""') + MAX_MEASURE_CHARS


def test_a_seed_s_description_stays_under_its_ceiling() -> None:
    seed, _ = fullest_described_seed(0)

    described = compact_source(
        read_seed(0, seed).source.source.model_copy(update={"searched_as": "m" * MAX_MEASURE_CHARS})
    )

    one = len(json.dumps(described))
    assert one <= SEEDED_SOURCE_DESCRIPTION_CEILING
    assert one > SEEDED_SOURCE_DESCRIPTION_CEILING * 0.99, (
        f"the fullest seed describes itself in {one:,} chars, under 99% of the cap: "
        "the seed is not at its bound"
    )


def test_a_seed_described_past_the_ceiling_is_refused_not_cut() -> None:
    """The control: one more entity than fits refuses the seed, with the size."""
    seed, entities = fullest_described_seed(0)
    over = bounded_seed(0, entities=entities + 1, stat_vars=MAX_SEED_STAT_VARS)

    read_seed(0, seed)
    with pytest.raises(ValueError, match="characters to describe, past the cap of 65,536"):
        read_seed(0, over)


# The session-state keys that hold a seed's text, each for `MAX_SEEDS` seeds.
#
# `planner:seeded_sources` holds one record per seed, each held to
# `MAX_SEED_RECORD_CHARS` where the seed is read. `propose_plan` holds a copy
# of a seed for each dataset requirement of the plan it records that pins it
# over another period, at most `MAX_PLAN_REQUIREMENTS` more, and lets go of a
# copy only a plan it replaced pinned, so no count of drafts or amendments
# adds to that. A drop moves records to `planner:dropped_seeds` without adding
# any. A copy is stored as the seed it copies and its window, never its
# identity (`planner.plan._store_sources`).
_STORED_COPY_CHARS = (
    len(
        json.dumps(
            {
                "windowed_from": "",
                "window": {"kind": "range", "start_year": 9999, "end_year": 9999},
            }
        )
    )
    + _SPEC_ID_CHARS
    + len(", ")
)
SEEDED_SOURCES_STATE_CEILING = (
    MAX_SEEDS * (MAX_SEED_RECORD_CHARS + len(", "))
    + MAX_PLAN_REQUIREMENTS * _STORED_COPY_CHARS
    + len('{"seeds": [], "copies": []}')
)


def test_a_seed_whose_record_is_past_the_ceiling_is_refused_not_cut() -> None:
    """The control for the state ceiling: one more entity than fits refuses the seed."""
    entities = most_recorded_entities(0)

    read_seed(0, recorded_seed(0, entities=entities))
    with pytest.raises(ValueError, match=r"its record takes .* past the cap of 131,072"):
        read_seed(0, recorded_seed(0, entities=entities + 1))


class _ToolContext:
    """What `propose_plan` reads of an ADK `ToolContext`: its state, and its session's events."""

    def __init__(self) -> None:
        self.state: dict[str, Any] = {}
        self.session = SimpleNamespace(events=[])


# The most dataset requirements, each with its own decision pinning a seed, a
# plan of the shortest fields fits under `MAX_PLAN_RECORD_CHARS`: 21 take
# 16,303 characters. So no plan pins `MAX_PLAN_REQUIREMENTS` copies at once,
# and the ceiling above holds them with room.
_MOST_PINNED_REQUIREMENTS = 20


def test_the_seeded_sources_state_stays_under_its_ceiling_across_drafts_and_amendments(
    pinning_decision: Callable[..., Decision],
    requirement: Callable[..., DatasetRequirement],
) -> None:
    """`MAX_SEEDS` seeds at their record bound, and a plan of
    `_MOST_PINNED_REQUIREMENTS` requirements, each pinning a seed over a year
    of its own, drafted and then amended over new years across two planner
    entries. Every proposal copies each seed it pins over new years; the
    state holds the seeds and the copies the recorded plan pins, the same
    size after every proposal and under the ceiling."""
    reads = [read_seed(number, fullest_recorded_seed(number)) for number in range(MAX_SEEDS)]
    one = len(json.dumps(reads[0].source.model_dump(mode="json")))
    assert one <= MAX_SEED_RECORD_CHARS
    assert one > MAX_SEED_RECORD_CHARS * 0.99
    propose = next(tool for tool in build_planner_tools() if tool.name == "propose_plan")
    ctx = _ToolContext()
    save_seeded_sources(ctx.state, [read.source for read in reads])
    version: int | None = None
    held: list[int] = []
    for entry in range(2):
        # What `harness.planner_seam` does on each fresh entry into the planner.
        save_plan_rounds(ctx.state, 0)
        for draft in range(MAX_PLAN_ROUNDS):
            first_year = 1900 + (entry * MAX_PLAN_ROUNDS + draft) * _MOST_PINNED_REQUIREMENTS
            pins = [
                (index, reads[index % MAX_SEEDS].source.source.spec_id)
                for index in range(_MOST_PINNED_REQUIREMENTS)
            ]
            result = propose.func(
                question="q",
                answer_mode="series",
                decisions=[
                    pinning_decision(
                        spec_id,
                        slot=f"d{index}",
                        description="d",
                        options=["a", "b"],
                        chosen="a",
                        rationale="r",
                    )
                    for index, spec_id in pins
                ],
                evidence_requirements=[
                    requirement(
                        id=f"r{index}",
                        decision_ids=[f"d{index}"],
                        source_spec_id=spec_id,
                        period=Period(start=first_year + index, end=first_year + index),
                        measure="m",
                        entities=[Entity(name="K")],
                        unit=None,
                    )
                    for index, spec_id in pins
                ],
                amends_version=version,
                tool_context=ctx,
            )
            assert result["outcome"] == "ok", result
            version = int(result["version"])
            assert len(seeded_sources(ctx.state)) == MAX_SEEDS + _MOST_PINNED_REQUIREMENTS
            held.append(len(ctx.state[SEEDED_SOURCES_STATE_KEY]))

    assert len(held) > MAX_PLAN_ROUNDS
    assert set(held) == {held[0]}
    assert held[0] <= SEEDED_SOURCES_STATE_CEILING


def _envelope_seed_ceiling() -> int:
    """What one seed adds to `resolve_envelope`, at most (`harness.seeds._envelope_seed`).

    The frame is the entry of a seed with nothing in it. To that: the
    sentence at its cap; the window, two years of `MAX_SEED_COUNT`'s digits;
    the year and rows Search showed, the rows with thousands separators; the
    coverage text, `MAX_SEED_LIST_ITEMS` counted labels; the shape kind; the
    identity's roles, `MAX_SEED_ROLES` strings; the open entity axes, one per
    role fact and one for the scope; and `MAX_SEED_OPERATIONS` operations.
    """
    digits = len(str(MAX_SEED_COUNT))
    grouped = len(f"{MAX_SEED_COUNT:,}")
    empty = read_seed(0, _EMPTY_SEED).planned
    frame = len(json.dumps(empty)) - len(json.dumps(empty["window"])) + len('""')
    axis = len(json.dumps({"column": "", "members": 0})) - 1 + MAX_SEED_TEXT_CHARS + digits
    operation = (
        len(json.dumps({"verb": "forecast", "phrase": "", "over": "destination", "target_year": 0}))
        - 1
        + MAX_OPERATION_PHRASE_CHARS
        + len(str(MAX_TARGET_YEAR))
    )
    return (
        frame
        + MAX_SEED_SENTENCE_CHARS
        + len(" to ")
        + 2 * digits
        + len('""')
        + digits
        + len(", ")
        + grouped
        + len(" rows")
        + len('""')
        + MAX_SEED_LIST_ITEMS * (grouped + len(" ") + MAX_SEED_TEXT_CHARS + len(", "))
        + len('""')
        + MAX_SEED_TEXT_CHARS
        + MAX_SEED_ROLES * (len('"", ') + MAX_SEED_TEXT_CHARS)
        + (MAX_SEED_LIST_ITEMS + 1) * (axis + len(", "))
        + MAX_SEED_OPERATIONS * (operation + len(", "))
    )


_EMPTY_SEED = {
    "spec": {
        "spec_id": spec_id_for(canonicalize({"identity_version": "spec1"})),
        "definition": {
            "schema_version": 1,
            "spec_id": spec_id_for(canonicalize({"identity_version": "spec1"})),
            "identity": {"identity_version": "spec1"},
        },
        "resolved": {},
    },
    "operations": [],
}

RESOLVE_ENVELOPE_SEED_CEILING = _envelope_seed_ceiling()


def test_the_seeded_resolve_envelope_and_the_elected_label_stay_under_their_ceilings() -> None:
    """`resolve_envelope` rides every state write of the turn and every node's output."""
    reads = [
        read_seed(number, bounded_seed(number, open_members=True)) for number in range(MAX_SEEDS)
    ]

    envelope = seeded_envelope([read.planned for read in reads])

    one = len(json.dumps(reads[0].planned))
    assert one <= RESOLVE_ENVELOPE_SEED_CEILING
    assert one > RESOLVE_ENVELOPE_SEED_CEILING * 0.95, (
        f"one seed's envelope entry is {one:,} chars, under 95% of the derived "
        f"{RESOLVE_ENVELOPE_SEED_CEILING:,}: the seed or the derivation is off"
    )
    frame = len(json.dumps(seeded_envelope([read_seed(0, _EMPTY_SEED).planned])))
    assert len(json.dumps(envelope)) <= frame + MAX_SEEDS * (RESOLVE_ENVELOPE_SEED_CEILING + 2)
    # `spec_id_elected_label` is the first seed's sentence.
    assert len(reads[0].planned["sentence"]) == MAX_SEED_SENTENCE_CHARS


# The two digests, built by `harness.studio_state`'s own functions.

_CONVERSATION = "conv_01budget0000"
_ASTRAL = "\U0001f600"
_BIG = 100_000


def _canvas_state(*, title: str, fact_summary: Any) -> ConversationState:
    """A conversation whose composition holds three times the block cap of components,
    each naming three times the show cap of 100,000-character variables and entities."""
    state = ConversationState(conversation_id=_CONVERSATION)
    entries = {
        f"cmp_{index:016x}": {
            "type": "line_chart",
            "title": title,
            "revision": 999_999_999,
            "fact_summary": fact_summary,
            "variable": [_ASTRAL * _BIG] * (3 * studio_state._DIGEST_SHOW_CAP),
            "entities": [_ASTRAL * _BIG] * (3 * studio_state._DIGEST_SHOW_CAP),
            "time": _ASTRAL * _BIG,
        }
        for index in range(3 * studio_state._DIGEST_BLOCK_CAP)
    }
    state.components.update(entries)
    state.composition = studio_state._next_manifest(
        None,
        [
            CompositionBlock(
                component_id=component_id,
                revision=entry["revision"],
                type=ComponentType(entry["type"]),
                title="block",
            )
            for component_id, entry in entries.items()
        ],
        actor=CompositionActor.MODEL,
    )
    return state


def _canvas_digest(*, title: str, fact_summary: Any, focus: str | None = None) -> dict[str, Any]:
    digest = _canvas_state(title=title, fact_summary=fact_summary).canvas_digest_state(focus=focus)
    assert digest is not None
    return digest


def _largest_canvas_digest() -> dict[str, Any]:
    """The largest digest: every block's fact summary just inside its cap.

    The titles are 100,000 characters, so they come out cut to their cap. A
    summary any longer than the cap is replaced by a short stub, so the largest
    digest is the one whose summaries fit.
    """
    summary = "s" * (studio_state._FACT_SUMMARY_BYTE_CAP - 2)
    return _canvas_digest(title="t" * _BIG, fact_summary=summary)


def _conversation_digest() -> dict[str, Any]:
    """Ten turns at the window's cap, each with 100,000-character texts and a
    list of two hundred cards, about 100 KB of them."""
    cards = [
        {"component_id": f"cmp_{index:016x}", "type": "line_chart", "title": "c" * 500}
        for index in range(200)
    ]
    state = ConversationState(conversation_id=_CONVERSATION)
    state.turns = [
        {
            "user": f"turn {index} " + _ASTRAL * _BIG,
            "assistant": _ASTRAL * _BIG,
            "cards": cards,
            "error": _ASTRAL * _BIG,
        }
        for index in range(studio_state._HISTORY_TURN_CAP)
    ]
    digest = state.conversation_digest_state()
    assert digest is not None
    return digest


_CANVAS_TITLE = "What is already on the user's canvas"
_CONVERSATION_TITLE = "Earlier turns in this conversation"


def _section_ceiling(title: str, body_bytes: int) -> int:
    """The section `_section` renders for a body of at most `body_bytes` characters."""
    return len(f"\n{title}:\n\n") + body_bytes


CANVAS_SECTION_CEILING = _section_ceiling(_CANVAS_TITLE, CANVAS_DIGEST_MAX_BYTES)
CONVERSATION_SECTION_CEILING = _section_ceiling(_CONVERSATION_TITLE, CONVERSATION_DIGEST_MAX_BYTES)


@pytest.mark.parametrize(
    "digest",
    [
        pytest.param(
            lambda: _canvas_digest(title="t" * _BIG, fact_summary={"series": ["x" * _BIG]}),
            id="canvas-inputs-100kb",
        ),
        pytest.param(_largest_canvas_digest, id="canvas-largest-digest"),
        pytest.param(
            lambda: _canvas_digest(title="t", fact_summary="s", focus="F" * 1_000_000),
            id="canvas-focus-names-no-component",
        ),
    ],
)
def test_the_canvas_digest_holds_its_maximum_however_large_its_inputs(
    digest: Callable[[], dict[str, Any]],
) -> None:
    built = digest()

    assert len(built["blocks"]) == studio_state._DIGEST_BLOCK_CAP
    assert len(json.dumps(built)) <= CANVAS_DIGEST_MAX_BYTES


def test_the_conversation_digest_holds_its_maximum_however_large_its_inputs() -> None:
    built = _conversation_digest()

    assert len(json.dumps(built)) <= CONVERSATION_DIGEST_MAX_BYTES
    # The newest turn is kept, so the digest was cut to size.
    assert built["turns"][-1]["user"].startswith(f"turn {studio_state._HISTORY_TURN_CAP - 1} ")


def test_the_digest_maxima_fail_when_the_digest_is_not_bounded() -> None:
    """The control: a digest as the state stored it, uncut, is past both maxima.

    The canvas digest is what `canvas_digest_state` returns for a state whose
    block fields are each 100,000 characters, but without `_digest_block`'s
    cut. This is the unbounded digest, and it has to exceed the maxima the
    bounded one is held to.
    """
    state = _canvas_state(title="t" * _BIG, fact_summary="s" * _BIG)
    uncut = {
        "blocks": [{"component_id": cid, **entry} for cid, entry in state.components.items()][
            : studio_state._DIGEST_BLOCK_CAP
        ]
    }
    assert len(json.dumps(uncut)) > CANVAS_DIGEST_MAX_BYTES

    turns = [{"user": "u" * _BIG, "assistant": "a" * _BIG}]
    assert len(json.dumps({"turns": turns, "total_turns": 1})) > CONVERSATION_DIGEST_MAX_BYTES


# The composed brief.

# The user's question and amendment are 100,000 characters, far past the cut.
_BRIEF_QUESTION = "q" * _BIG
_BRIEF_AMENDMENT = "a" * _BIG
_ASKS = ("amendment", "refusal")


def _maximal_refusal() -> dict[str, Any]:
    """An executor refusal far past the cut: one 100,000-character message and a
    hundred wave siblings that each carry the same."""

    def refused(index: int) -> dict[str, Any]:
        return {
            "kind": "rediscovery_needed",
            "spec_id": _spec(index),
            "refusal": {"code": "over_ceiling", "message": "m" * _BIG},
        }

    return {**refused(1), "also_stopped": [refused(index) for index in range(2, 102)]}


def _ask_kwargs(ask: str) -> dict[str, Any]:
    """`_seam_brief`'s arguments for the amendment ask or the refusal ask."""
    if ask == "refusal":
        return {"refusal": _maximal_refusal()}
    return {"amendment": _BRIEF_AMENDMENT}


def _maximal_brief_parts(ask: str) -> dict[str, str]:
    """Every section of the largest brief `PlannerSeam._message` builds, apart.

    The parts are rendered by the functions `_message` calls, so the sum of
    their lengths can be checked against the whole. The question is the cut
    the seam makes of a 100,000-character one, and the ask is what a continuing
    `_message` renders for the amendment or the refusal: an ask stands in for
    the one the fresh brief ends with, and a fresh brief carries one or the
    other.
    """
    refusal = _maximal_refusal() if ask == "refusal" else None
    cut_question = _BRIEF_QUESTION[: MAX_QUESTION_CHARS - len(CUT_MARKER)] + CUT_MARKER
    return {
        "envelope": _envelope_context(
            _maximal_envelope(), refusal=refusal, spec_id_elected=False, spec_id_elected_label=None
        ),
        "canvas digest": _section(_CANVAS_TITLE, _largest_canvas_digest()),
        "conversation digest": _section(_CONVERSATION_TITLE, _conversation_digest()),
        "plan": _plan_context(_legal_maximum_plan_record()),
        "question": f"\nQuestion: {cut_question}\n",
        "ask": _seam_brief(fresh=False, envelope={}, prior_plan=None, **_ask_kwargs(ask)),
    }


def _continuing_chars(**kwargs: Any) -> int:
    """Characters of the continuing message for `kwargs`, an ask and nothing else."""
    return len(_seam_brief(fresh=False, envelope={}, prior_plan=None, **kwargs))


def _refused_sentence_ceiling(sentence: Callable[[list[tuple[str, str]]], str]) -> int:
    """The most a sentence naming refused specs renders, whatever it is told.

    The seam cuts the reason for one spec, or the whole list of them, at
    `MAX_QUESTION_CHARS`. The sentence's own wording around it is measured by
    rendering it for placeholders. One spec adds its `spec_id`, which is a
    `SpecId` in a refusal the executor builds.
    """
    one = len(sentence([("s", "d")])) - len("d") + (_SPEC_ID_CHARS - len("s"))
    many_text = "'s' (d); 's' (d)"
    many = len(sentence([("s", "d"), ("s", "d")])) - len(many_text)
    return max(one, many) + MAX_QUESTION_CHARS


def _maximal_brief_ceilings(ask: str) -> dict[str, int]:
    """What each part of `_maximal_brief_parts(ask)` can be, from the caps that bound it.

    The question and the amendment are cut at `MAX_QUESTION_CHARS`. A refusal
    is cut there as JSON, then again in the sentence naming the specs to avoid,
    and, on a fresh render, in the envelope's paragraph naming the specs it left
    out. The wording around each cut is measured from the renderer.
    """
    envelope = MAXIMAL_ENVELOPE_CEILING
    if ask == "refusal":
        framing = _continuing_chars(refusal={"kind": "x"}) - len(json.dumps({"kind": "x"}))
        ask_ceiling = (
            framing + MAX_QUESTION_CHARS + _refused_sentence_ceiling(planner_seam._avoid_sentence)
        )
        envelope += _refused_sentence_ceiling(planner_seam._refused_spec_reason)
    else:
        ask_ceiling = _continuing_chars(amendment="x") - 1 + MAX_QUESTION_CHARS
    return {
        "envelope": envelope,
        "canvas digest": CANVAS_SECTION_CEILING,
        "conversation digest": CONVERSATION_SECTION_CEILING,
        "plan": LEGAL_MAXIMUM_PLAN_SECTION_CEILING,
        "question": len("\nQuestion: \n") + MAX_QUESTION_CHARS,
        "ask": ask_ceiling,
    }


# The largest brief the seam builds: the sum of what each section can be.
LEGAL_MAXIMUM_BRIEF_CEILING = max(sum(_maximal_brief_ceilings(ask).values()) for ask in _ASKS)


@pytest.mark.parametrize("ask", _ASKS)
def test_the_largest_legal_brief_is_within_the_sum_of_its_parts_maxima(ask: str) -> None:
    """Every section of the fresh brief at its maximum at once, held to the sum.

    The maximal envelope, the two digests through their own functions, the
    largest legal plan, a question and the amendment or refusal ask, rendered by
    `PlannerSeam._message`. The message names each part with its size and
    ceiling, so a failure says which one moved.
    """
    parts = _maximal_brief_parts(ask)
    ceilings = _maximal_brief_ceilings(ask)
    brief = _seam_brief(
        fresh=True,
        envelope=_maximal_envelope(),
        prior_plan=_legal_maximum_plan_record(),
        canvas_digest=_largest_canvas_digest(),
        conversation_digest=_conversation_digest(),
        question=_BRIEF_QUESTION,
        **_ask_kwargs(ask),
    )

    table = "\n".join(
        f"  {name}: {len(text):,} chars, ceiling {ceilings[name]:,}" for name, text in parts.items()
    )
    assert len(brief) == sum(len(text) for text in parts.values()), (
        f"the brief is not its sections in order (a section was added, or one dropped):\n{table}"
    )
    assert len(brief) <= sum(ceilings.values()), (
        f"the largest legal brief is {len(brief):,} chars, past {sum(ceilings.values()):,}:\n"
        f"{table}"
    )
    over = [name for name, text in parts.items() if len(text) > ceilings[name]]
    assert not over, f"a section outgrew its own ceiling: {over}\n{table}"
    # Each ceiling is within reach of what its part renders at the caps, so a
    # part far under its own ceiling is a derivation that stopped matching.
    slack = [name for name, text in parts.items() if len(text) < 0.5 * ceilings[name]]
    assert not slack, f"a section is under half of its own ceiling: {slack}\n{table}"


def test_the_brief_ceiling_fails_when_a_part_is_inflated() -> None:
    """The control for the composed ceiling: one inflated part puts the brief over it."""
    brief = _seam_brief(
        fresh=True,
        envelope=_maximal_envelope(),
        prior_plan=_legal_maximum_plan_record(),
        canvas_digest={"blocks": [{"title": "x" * LEGAL_MAXIMUM_BRIEF_CEILING}]},
        conversation_digest=_conversation_digest(),
        question=_BRIEF_QUESTION,
    )

    assert len(brief) > LEGAL_MAXIMUM_BRIEF_CEILING


@pytest.mark.parametrize("ask", _ASKS)
def test_the_question_and_ask_ceilings_fail_when_the_seam_stops_cutting(
    ask: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The control for the user-text ceilings: with the cut gone, they are exceeded."""
    monkeypatch.setattr(planner_seam, "cut_user_text", lambda text, cap: text)
    ceilings = _maximal_brief_ceilings(ask)

    question = _seam_brief(fresh=True, envelope={}, prior_plan=None, question=_BRIEF_QUESTION)
    asked = _seam_brief(fresh=False, envelope={}, prior_plan=None, **_ask_kwargs(ask))

    assert len(question) > ceilings["question"]
    assert len(asked) > ceilings["ask"]


# The bounded production context, through the seam and ADK --------------------------
#
# The two recorded scenarios above run with an empty envelope and no digests. A
# session's first planner call carries all three in the brief, and each is
# bounded at its source, so this scenario records that call with every one at
# its maximum, as the request ADK assembles it.


def _bounded_context() -> dict[str, Any]:
    """What `PlannerSeam.__call__` is handed besides the plan and the question."""
    return {
        "envelope": _maximal_envelope(),
        "canvas_digest": _largest_canvas_digest(),
        "conversation_digest": _conversation_digest(),
    }


# The seam's brief in the `BOUNDED_CONTEXT` scenario: `SEAM_BRIEF_CEILING`, which
# already covers a capped searched envelope, a nine-requirement plan and the
# question, plus the two digests at their maxima (`CANVAS_SECTION_CEILING` and
# `CONVERSATION_SECTION_CEILING`), plus what a seeded envelope adds at its caps
# (`_seeded_section_ceiling`).
BOUNDED_CONTEXT_BRIEF_CEILING = (
    SEAM_BRIEF_CEILING
    + CANVAS_SECTION_CEILING
    + CONVERSATION_SECTION_CEILING
    + _seeded_section_ceiling()
)

# The planner's whole `contents` in that scenario: the realistic history's
# ceiling plus the brief's, which the realistic ceiling already counts a small
# brief of, plus the one skill page the round reads.
BOUNDED_CONTEXT_CONTENTS_CEILING = (
    PLANNER_CONTENTS_CEILING + BOUNDED_CONTEXT_BRIEF_CEILING + SKILL_PAGE_CEILING
)


def test_the_bounded_production_context_stays_under_its_derived_ceilings(
    records: list[dict[str, Any]],
) -> None:
    """A brief carrying every bounded input at its maximum, as ADK assembles it.

    The brief is held to `BOUNDED_CONTEXT_BRIEF_CEILING`, and the planner's
    whole `contents` to the ceiling that adds the realistic history to it. The
    brief has to contain the digests, so a scenario that silently stopped
    passing them cannot clear the ceiling by measuring a brief without them.
    """
    planner = _largest(records, scenario=BOUNDED_CONTEXT, agent="planner")

    brief = bucket_chars(planner, SEAM_BUCKET)
    contents = planner["contents"]["total_chars"]

    assert brief > CANVAS_DIGEST_MAX_BYTES * 0.9, (
        f"the bounded-context brief is {brief:,} chars, less than the canvas digest alone "
        f"({CANVAS_DIGEST_MAX_BYTES:,} max): the scenario is not carrying its digests"
    )
    assert brief <= BOUNDED_CONTEXT_BRIEF_CEILING, (
        f"the bounded-context seam brief is {brief:,} chars, past {BOUNDED_CONTEXT_BRIEF_CEILING:,} "
        f"(SEAM_BRIEF_CEILING {SEAM_BRIEF_CEILING:,} + canvas {CANVAS_SECTION_CEILING:,} + "
        f"conversation {CONVERSATION_SECTION_CEILING:,} + seeds {_seeded_section_ceiling():,})"
    )
    assert contents <= BOUNDED_CONTEXT_CONTENTS_CEILING, (
        f"the planner's contents in the bounded-context scenario are {contents:,} chars, "
        f"past {BOUNDED_CONTEXT_CONTENTS_CEILING:,}"
    )


def test_the_bounded_context_ceilings_fail_when_the_brief_grows() -> None:
    """The control for the bounded-context ceilings: a seam part past them reports."""
    request = _request(
        tool_names=["propose_plan"],
        texts=[],
        seam=BOUNDED_CONTEXT_BRIEF_CEILING + 1,
    )
    record = compose(request, agent="planner", scenario=BOUNDED_CONTEXT, call_index=0)

    assert bucket_chars(record, SEAM_BUCKET) > BOUNDED_CONTEXT_BRIEF_CEILING
    assert record["contents"]["total_chars"] > BOUNDED_CONTEXT_BRIEF_CEILING


def test_the_propose_plan_declaration_carries_the_plan_limits(measured: Any) -> None:
    """The declaration ADK sends the model states every limit, in `maxLength` and `maxItems`.

    The limits live on the schema alone, and no field description restates
    one, so the model reads them from here. The declaration is read off the
    request the recorded round captured, and every model in its `$defs` is
    walked for a string or array with no bound.
    """
    _records, requests = measured[("realistic", "planner")]
    declaration = next(
        declaration
        for tool in requests[0].config.tools
        for declaration in tool.function_declarations or []
        if declaration.name == "propose_plan"
    )
    definitions = declaration.parameters_json_schema["$defs"]

    def properties(model: str) -> dict[str, Any]:
        return definitions[model]["properties"]

    assert properties("Decision")["description"].get("maxLength") == MAX_DECISION_DESCRIPTION_CHARS
    assert properties("Decision")["rationale"].get("maxLength") == MAX_DECISION_RATIONALE_CHARS
    assert properties("Decision")["chosen"].get("maxLength") == MAX_OPTION_CHARS
    assert properties("Decision")["options"].get("maxItems") == MAX_DECISION_OPTIONS
    assert properties("Decision")["options"]["items"].get("maxLength") == MAX_OPTION_CHARS
    assert properties("DatasetRequirement")["measure"].get("maxLength") == MAX_MEASURE_CHARS
    assert (
        properties("DatasetRequirement")["source"]["anyOf"][0].get("maxLength") == MAX_SOURCE_CHARS
    )
    assert properties("DatasetRequirement")["entities"].get("maxItems") == MAX_ENTITIES
    assert properties("Entity")["name"].get("maxLength") == MAX_ENTITY_NAME_CHARS
    assert properties("Entity")["role"]["anyOf"][0].get("maxLength") == MAX_ENTITY_ROLE_CHARS
    assert properties("EntityIn")["names"].get("maxItems") == MAX_ENTITIES
    assert properties("FilterRequirement")["predicates"].get("maxItems") == MAX_PREDICATES
    unbounded = [
        path
        for name, model in definitions.items()
        for path in unbounded_paths(model, definitions, name)
    ]
    assert unbounded == []

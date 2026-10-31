"""Ceilings on what the planner is handed, and their baseline.

This file measures what the planner is handed and holds
each part of it to a ceiling. The bounds themselves live in production code:
`planner.plan` limits the plan and cuts user text (`MAX_PLAN_RECORD_CHARS`,
`cut_user_text`), `harness.studio_state` limits the two digests, and
`harness.planner_seam` holds the search readings to a budget and cuts the question, the amendment
and a refusal. This file's ceilings are derived from those limits or measured
from a recorded scenario, and each control test shows its ceiling can fail.

What is bounded is a segment of a request `context_probe.compose` already
measures: the system instruction, the tool declarations together, each named
declaration on its own, and the text the planner's `contents` carries. Each realistic-input ceiling is a named constant at the top of this
file with the reason it sits where it does.

The fenced foreign-agent relay is the exception to every other ceiling's
rounding: no relayed text may reach the planner at all. ADK relays another
agent's turn into a request as fenced user text
(`google.adk.flows.llm_flows.context._fencing._present_other_agent_message`), and no
other agent runs inside a planner run, so any relayed part is a regression.

Every realistic-input ceiling is proved live by `test_every_ceiling_fails_when_its_segment_grows`.
The relay guard's own proof is `test_no_relayed_foreign_agent_text_reaches_the_planner`
paired with `test_the_relay_check_measures_a_relay_that_is_there`, since a
check for zero that cannot go non-zero would pass whatever the code did. A guard
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
the one this whole budget is least able to feel. The planner's prefix is cached:
`harness.prefix_cache` keeps one explicit cache per prefix, keyed by the hash of
the model, the system instruction, the tool declarations and the tool config, and
Gemini's implicit caching matches whatever follows it. Either saving depends on
the prefix being byte-identical from call to call. A change that made the
instruction or the tool list vary per request would create a new cache on every
call, at the price of an uncached call each, with nothing but an INFO line per
create to say so, which is why `test_the_cached_prefix_is_stable_across_a_run`
exists.

The scenario is one plan-then-translate round. The seam is handed an already
drafted plan as `prior_plan`, the way `harness.graph`'s own planner node hands
it one, and the realistic round's brief carries the search readings, the two
digests, the asks and the question a deployed planner call recorded
(`_RECORDED_BRIEF_PATH`). The planner's model loads a skill with `get_skill`,
answered with one full page, then proposes the drafted plan again unchanged
(`propose_plan` with only `amends_version`), whose translation runs the real
mechanical loop against a canned `no_data` search and refers the plan back. The
planner then answers in text, which ends the entry with no plan ready: the
largest planner request on record is that last generation, which carries the
brief, the skill page and the referral.
"""

from __future__ import annotations

import copy
import json
import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
from compiler.surface import FixtureSurface, ToolCallResult
from data_mcp.skills_client import PAGE_CHARS, continuation_line, page_of, with_closing_line
from google.adk.models.llm_request import LlmRequest
from google.genai import types
from harness import planner_seam, studio_state
from harness.planner_seam import (
    _PLAN_MEASURE_CHOICE_CAP,
    _PLAN_SOURCE_SENTENCE_CAP,
    _READINGS_BUDGET,
    _READINGS_CONTEXT,
    PlannerSeam,
    _plan_context,
    _readings_ledger,
    _section,
)
from harness.prefix_cache import prefix_key
from harness.settings import Settings
from harness.studio_state import (
    CANVAS_DIGEST_MAX_BYTES,
    CONVERSATION_DIGEST_MAX_BYTES,
    ConversationState,
)
from harness.turn_records import CandidateReading, SearchReading, WarningReading
from harness.wire import (
    ComponentType,
    CompositionActor,
    CompositionBlock,
)
from planner.operations import AggregateOperation, RetrieveOperation, Translation
from planner.plan import (
    ASKS_STATE_KEY,
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
    MAX_PREDICATES,
    MAX_QUESTION_CHARS,
    MAX_SOURCE_CHARS,
    AggregateRequirement,
    DatasetRequirement,
    Decision,
    Entity,
    Period,
    Plan,
    ResolvedBy,
    Unit,
)
from planner.records import GroupBy, Reduction
from planner.workflow import EvidenceBinding, SelectedSource, ToolId, WorkflowStep
from planner.workflow import Workflow as PlanWorkflow

from .context_probe import (
    BOUNDING_PATTERNS,
    RELAY_MARKER,
    SEAM_BUCKET,
    SEAM_MARKER,
    RecordingLlm,
    bucket_chars,
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
# rules, the skills paragraph and the translating paragraph, assembled by
# `planner.agent.planner_instruction`. Measured 19,875 with the skill tools and
# translation attached. The margin is 1.4% and covers a hand edit of
# model-facing wording, since these paragraphs are rewritten for model
# behaviour a sentence at a time.
PLANNER_SYSTEM_INSTRUCTION_CEILING = 20_150

# Every planner tool declaration together measures 41,324 (`propose_plan` 37,293 and
# 4,031 across the smaller ones). The margin is 1.6%.
PLANNER_DECLARATIONS_TOTAL_CEILING = 42_000

# `propose_plan`'s own declaration: the `DecisionDraft` and `EvidenceRequirement`
# schemas with their `$defs` tree, and the patch, decline, answer and intent
# arguments. Measured 37,293, with every field that refers to a definition holding a
# copy of it under the field's own description. It sits in the static prefix every planner call
# repeats, which Gemini's implicit cache can serve. Its field descriptions are what
# the planner writes a plan from. The margin is 1.1%.
PROPOSE_PLAN_DECLARATION_CEILING = 37_700

# The smaller planner declarations at once: `adk_request_input`,
# `get_skill_file`, `get_skill` and `reject_plan`, measured 4,031 together.
MINOR_DECLARATIONS_TOTAL_CEILING = 5_070

# The planner's `contents` history on its largest captured generation, less
# the skill text `SKILL_PAGE_CEILING` and the brief `RECORDED_BRIEF_CEILING`
# bound on their own (1,158 chars, the third and last generation of the
# realistic scenario). Most of it is the `propose_plan` answer, since it
# carries the translation's referral beside the plan's own summary. The
# margin is 5%.
PLANNER_CONTENTS_CEILING = 1_220

# The `get_skill` answer in the planner's `contents`, bucketed as
# `functionResponse:get_skill`: one full page of a skill file
# (`data_mcp.skills_client.PAGE_CHARS`, 32,000 chars) plus the closing line
# naming the next call, as ADK serialises it. Measured 32,676 on a page of
# short Markdown lines, where escaping each line break adds a char. This bounds
# one call. Nothing caps how many skill pages one planner entry reads besides
# the seam's event limit and entry deadline.
SKILL_PAGE_CEILING = 40_800

# The seam's own brief, bucketed as `text(seam-brief)` on its opening words
# (`harness.planner_seam._READINGS_CONTEXT`): the search readings, the two
# digests, the plan being changed and the question. A session sends it on its
# first planner call and only the ask thereafter, so this bounds what that one
# call is billed in full before a cache can serve it.
# `harness.planner_seam._plan_context` renders the prior plan as a
# line-rendered digest, and `_READINGS_BUDGET` holds the readings.
#
# A two-requirement plan is the wrong end of the range to size this from: the
# seam's own docstring probe records the realistic upper end at nine
# requirements, and the section that dominates it is the prior plan, re-sent on
# every amend and replan call. `test_the_seam_brief_holds_a_nine_requirement_plan_under_its_ceiling`
# measures that upper end, with readings at their whole budget: 11,131 chars for
# a draft plan and 12,627 for a compiled one (the plan section alone is
# 8,723 -> 6,942 raw-to-digest for the draft, 22,609 -> 8,438 for the compiled
# one, and the readings section 3,486). The ceiling is 25% above the compiled
# case, as this file's own convention says.
SEAM_BRIEF_CEILING = 15_800

# The seam's brief in the realistic scenario, which carries the brief a
# deployed planner call recorded (`_RECORDED_BRIEF_PATH`): three search
# readings, four canvas blocks, one earlier turn, two asks and the question,
# plus the scenario's two-requirement plan. Measured 7,568.
RECORDED_BRIEF_CEILING = 9_460


# The measured scenario ----------------------------------------------------------

# The second recorded scenario: the same round, with the seam's brief carrying
# search readings far past their budget and the canvas and conversation digests
# at their maxima.
BOUNDED_CONTEXT = "bounded-context"


def _seam() -> PlannerSeam:
    """The seam the recorded round runs on, over a canned data-mcp surface.

    Built here, not reused from `test_planner_seam`, whose own
    `_translator_seam` is private to the scenarios it serves.
    """
    return PlannerSeam(
        settings=Settings(),
        surface=FixtureSurface(
            schemas=[],
            responses={"search": _NO_DATA_SEARCH_RESPONSE, "get_skill": _skill_page()},
        ),
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


def _script() -> list[Any]:
    """The planner's script for one round.

    The planner loads a skill, proposes the drafted plan again unchanged (which
    translates it and refers it back), and then produces text, which ends the
    entry.
    """
    return [
        [("get_skill", {"skill_id": "dfo-faq"})],
        [("propose_plan", {"amends_version": 1})],
        "nothing to submit",
    ]


# The brief the deployed planner's first call of a recorded turn carried: the
# second turn of a Studio conversation about Guatemala's GDP and population,
# with three search readings, four canvas blocks, one earlier turn and two asks.
_RECORDED_BRIEF_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "brief_guatemala_gdp_population.json"
)


def _recorded_brief() -> dict[str, Any]:
    """The realistic round's `context` and session `state`, from `_RECORDED_BRIEF_PATH`."""
    recorded = json.loads(_RECORDED_BRIEF_PATH.read_text())
    return {
        "context": {
            "question": recorded["question"],
            "readings": recorded["readings"],
            "canvas_digest": recorded["canvas_digest"],
            "conversation_digest": recorded["conversation_digest"],
        },
        "state": {ASKS_STATE_KEY: recorded["asks"]},
    }


async def _recorded_round(
    *,
    scenario: str,
    decision: Callable[..., Any],
    requirement: Callable[..., Any],
    derived_requirement: Callable[..., Any],
    context: dict[str, Any],
    state: dict[str, Any] | None = None,
) -> dict[str, tuple[list[dict[str, Any]], list[Any]]]:
    """One plan-then-translate round, with every request the planner's model was handed.

    `context` is what the seam's brief carries besides the plan: the search
    `readings`, the `canvas_digest` and
    `conversation_digest`, and the `question` when it is not the draft's own.
    `state` is what the session holds before the round, such as the asks.

    Returns `{"planner": (records, requests)}`: the `context_probe.compose`
    records in call order, and the raw `LlmRequest` objects the stability
    assertions compare byte for byte.
    """
    draft = _gdp_plan(decision, requirement, derived_requirement)
    seam = _seam()
    planner_llm = RecordingLlm(script=_script(), agent_name="planner", scenario=scenario)
    seam._agent.model = planner_llm

    arguments: dict[str, Any] = {"question": draft.question, **context}
    output = await _SeamHarness(state).run(
        seam, prior_plan=draft.model_dump(mode="json"), **arguments
    )
    assert output is not None and "refused" in output, output

    return {"planner": (planner_llm.record, planner_llm.captured)}


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
    The second scenario, `BOUNDED_CONTEXT`, hands the seam readings far past
    their budget and both digests at their maxima.

    The baseline rewrite is env-gated the way `test_digest_budget.py`'s own is
    and wrapped by the `context-baseline` `just` recipe. It happens here, in the
    fixture every test reads through, so the ceilings are checked against
    the freshly written file, not a skipped gate.
    """
    measured: dict[tuple[str, str], tuple[list[dict[str, Any]], list[Any]]] = {}
    recorded = _recorded_brief()
    scenarios: list[tuple[str, dict[str, Any], dict[str, Any] | None]] = [
        ("realistic", recorded["context"], recorded["state"]),
        (BOUNDED_CONTEXT, _bounded_context(), None),
    ]
    for scenario, context, state in scenarios:
        round_measured = await _recorded_round(
            scenario=scenario,
            decision=decision,
            requirement=requirement,
            derived_requirement=derived_requirement,
            context=context,
            state=state,
        )
        measured[(scenario, "planner")] = round_measured["planner"]

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
            "minor planner declarations",
            _minor_declarations(record),
            MINOR_DECLARATIONS_TOTAL_CEILING,
        ),
        (
            "planner contents",
            record["contents"]["total_chars"]
            - bucket_chars(record, SKILL_BUCKET)
            - bucket_chars(record, SEAM_BUCKET),
            PLANNER_CONTENTS_CEILING,
        ),
        ("seam brief", bucket_chars(record, SEAM_BUCKET), RECORDED_BRIEF_CEILING),
        ("skill page", bucket_chars(record, SKILL_BUCKET), SKILL_PAGE_CEILING),
    ]


# The bucket `context_probe` files a `get_skill` answer under.
SKILL_BUCKET = "functionResponse:get_skill"


# Everything a planner request declares other than `propose_plan`, the one
# declaration bounded individually above, not lumped into this total.
def _minor_declarations(record: dict[str, Any]) -> int:
    return sum(item["chars"] for item in record["declarations"] if item["name"] != "propose_plan")


def test_no_segment_grows_past_its_ceiling(records: list[dict[str, Any]]) -> None:
    """Every bound holds on the largest request the planner was handed.

    The planner's request has to carry the full skill page the scenario serves,
    so a round that stopped calling `get_skill` cannot clear the skill ceiling
    by measuring nothing.
    """
    planner = _largest(records, scenario="realistic", agent="planner")
    assert bucket_chars(planner, SKILL_BUCKET) > PAGE_CHARS, (
        f"{_baseline_key(planner)} carries no full `get_skill` page: the round did not load one"
    )
    checked = _declare(planner)

    over = [
        f"{label}: {measured} chars > ceiling {ceiling}"
        for label, measured, ceiling in checked
        if measured > ceiling
    ]
    assert not over, (
        "a segment of the planner request outgrew its stated bound:\n  "
        + "\n  ".join(over)
        + f"\nA deliberate change means {_REGENERATE}, then this file's ceiling constant."
    )


def test_no_relayed_foreign_agent_text_reaches_the_planner(
    records: list[dict[str, Any]],
) -> None:
    """No relayed text reaches the planner, tagged with an author or not.

    Translation runs inside the planner's own `propose_plan` call, and
    `harness.planner_seam._PLANNER_BRANCH` keeps another agent's turns off the
    planner's branch, so no fenced relay belongs in a planner request.
    """
    planner_records = [
        record
        for record in records
        if record["scenario"] == "realistic" and record["agent"] == "planner"
    ]
    assert planner_records, "no realistic planner request captured (the round did not run)"
    for record in planner_records:
        found = relay_buckets(record)
        assert not found, (
            f"{_baseline_key(record)}: the planner is carrying relayed foreign-agent "
            f"text: {found}. ADK is relaying another agent's turn into this request."
        )


def test_the_relay_check_measures_a_relay_that_is_there() -> None:
    """The relay guard is only meaningful if the bucket can be non-empty.

    The control for `test_no_relayed_foreign_agent_text_reaches_the_planner`: a
    bucket that always read empty, or a marker match that had drifted from
    what `_fencing` emits, would pass that test for the wrong reason. The
    relay built here is ADK's own framing: one tagged part naming the author,
    then the fenced payload.
    """
    request = _request(
        tool_names=["propose_plan"],
        texts=[],
        relay=("dfo_orchestrator", 4_000),
    )
    record = compose(request, agent="planner", scenario="realistic", call_index=0)

    found = relay_buckets(record)
    assert set(found) == {"dfo_orchestrator", "untagged"}, found
    assert all(found.values()), found


# Mechanism assertions -----------------------------------------------------------
#
# Properties, not numbers, so `just context-baseline` cannot absorb a
# regression by rewriting them. Modelled on `test_digest_budget.py`'s named
# digest-size regressions (a digest that leaks frame size, a candidate list that
# grows without a cap, a grounding ref that restates its sentence).


def test_the_cached_prefix_is_stable_across_a_run(measured: Any) -> None:
    """Each consecutive pair of requests carries a byte-identical cached prefix.

    Between two requests in one run, `config.system_instruction` is byte-identical
    and so is the `harness.prefix_cache.prefix_key` the shared cache files it
    under: the hash of the model, the system instruction, the tool declarations
    as sent and the tool config. A mismatch creates a second cache, which costs
    about one uncached call, with one INFO line to show for it.

    A stable prefix is what the shared explicit cache serves and what Gemini's
    implicit caching matches across requests. The change most likely to break
    it is setting `static_instruction` or moving to an `InstructionProvider`,
    either of which can make the instruction vary per request and destroy that
    saving with nothing in the log to say so.

    The comparison is within one run. A cache is shared between conversations
    only while every conversation builds the same prefix. A deploy that changes
    the prompt starts a new cache, as it should.
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
            "The shared prefix cache is keyed on it, so each variant creates a cache."
        )
        key_before = prefix_key(before.model or "", before.config)
        assert key_before == prefix_key(after.model or "", after.config), (
            f"realistic/planner: the tool declarations or tool config differ between "
            f"{first} and {second}. The shared prefix cache is keyed on them, so each "
            "variant creates a cache."
        )


def test_the_leading_contents_of_a_run_are_only_ever_appended_to(measured: Any) -> None:
    """Each planner request's `contents` begins with the previous one's, unchanged.

    Gemini's implicit caching matches a leading run of contents, so a run whose
    earlier turn was rewritten, not merely appended to, would miss the cache
    from the rewrite onward. Every request ADK assembles here is the session's
    whole history, so this is a real property of the served path and not an
    artefact of the fixture: `harness.planner_seam._propose` runs every
    generation of an entry in the same session.

    The guarantee this protects: a supersession edit stays transient and is
    never persisted, and a version that persisted it would show up here as a
    leading content that changed.

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
            "Implicit caching matches the leading contents, and a rewrite here "
            "loses the cached share silently."
        )


# Failable fixtures --------------------------------------------------------------
#
# Each ceiling above is proved live by handing the same comparison a request
# built to exceed it and confirming every bound reports. A ceiling this could
# not fail would be one that passes whatever the code does, which is the failure
# mode a guard test with no failable fixture always risks. The request is
# fabricated here, not by editing production code to inflate the real scenario.

# Size of every segment in the deliberately inflated request: past every ceiling
# above.
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
    that many chars, `harness.planner_seam._READINGS_CONTEXT`'s own opening
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
                            f"[{author}] called tool `propose_plan` with parameters:\n"
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
        tool_names=["propose_plan", "adk_request_input", "reject_plan"],
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


# The seam brief at its realistic upper end --------------------------------------
#
# The recorded scenario above runs a two-requirement plan, the small end of the
# plans this seam renders and the wrong end to size a ceiling from.
# `planner_seam.py`'s own docstring probe records nine requirements as the
# realistic upper end, so `SEAM_BRIEF_CEILING` is sized from that and this test
# measures it directly.

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


def _seam_brief(
    *,
    prior_plan: dict[str, Any] | None,
    readings: Sequence[Mapping[str, Any]] = (),
    canvas_digest: dict[str, Any] | None = None,
    conversation_digest: dict[str, Any] | None = None,
    question: str = "compare population and gdp for guatemala since the year 2000",
    refusal: dict[str, Any] | None = None,
    amendment: str | None = None,
    refused: Sequence[str] = (),
) -> str:
    """One `_message` render, the way `PlannerSeam.__call__` asks for it."""
    seam = PlannerSeam(
        settings=Settings(),
        surface=FixtureSurface(schemas=[], responses={}),
    )
    return seam._message(
        question=question,
        prior_plan=prior_plan,
        refusal=refusal,
        amendment=amendment,
        canvas_digest=canvas_digest,
        conversation_digest=conversation_digest,
        ledger=_readings_ledger(readings),
        refused=refused,
    )


def test_the_seam_brief_holds_a_nine_requirement_plan_under_its_ceiling() -> None:
    """The realistic upper end, measured: a nine-requirement plan in the brief.

    Both a draft plan and a compiled one, since a compiled plan carries the
    `translation` and `workflow` the digest renders a summary of. Each is
    rendered against readings far past their budget, so the readings section
    is at its largest, and held to `SEAM_BRIEF_CEILING` on its own.

    This is the test the ceiling is sized from. The recorded scenario's own
    brief is two requirements and three readings, so a ceiling taken from it
    would pass a change that let the brief grow with the plan it carries.
    """
    draft = _nine_requirement_plan()
    dumped = draft.model_dump(mode="json")
    readings = _maximal_readings()

    for label, plan in (("draft", dumped), ("compiled", _compiled(draft))):
        render = _seam_brief(readings=readings, prior_plan=plan)
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
        tool_names=["propose_plan", "adk_request_input", "reject_plan"],
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
# `planner_seam`'s budget for the readings, `studio_state`'s maxima for the two
# digests. The ceilings are computed in this file from those constants, so a
# limit that moves changes its ceiling with it.
#
# The tier covers the model-authored fields at their bound plus the renderer's
# own caps. What the compiler and data-mcp put on a plan is bounded in the
# render: a workflow's source `sentence` is cut at `_PLAN_SOURCE_SENTENCE_CAP`,
# its measure and other counts at `_PLAN_MEASURE_CHOICE_CAP`, and its other
# rendered strings are ids that the `Identifier` and `SpecId` patterns
# bound. The live question, the amendment text and a refusal are
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
    requirement, the source's `sentence` cut at `_PLAN_SOURCE_SENTENCE_CAP`
    and its measure and other counts at `_PLAN_MEASURE_CHOICE_CAP`.
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
        * (
            len("\n  ")
            + _SPEC_ID_CHARS
            + len(" -- ")
            + _PLAN_SOURCE_SENTENCE_CAP
            + _PLAN_MEASURE_CHOICE_CAP
        )
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
# (16,000) and the framing (18,468, of which the compiled workflow is 15,282), so
# 34,468.
LEGAL_MAXIMUM_PLAN_SECTION_CEILING = MAX_PLAN_RECORD_CHARS + _plan_framing_chars()


def _with_workflow(record: dict[str, Any]) -> dict[str, Any]:
    """`record` with a compiled workflow added: one source and one binding per requirement.

    Each source sentence is three times the render's cap, and each source
    offers more other counts than its measure line fits, so the render's own
    cuts are what set those parts of the size.
    """
    ids = [requirement["id"] for requirement in record["evidence_requirements"]]
    return {
        **record,
        "workflow": {
            "sources": [
                {
                    "spec_id": _spec(index),
                    "sentence": "S" * (_PLAN_SOURCE_SENTENCE_CAP * 3),
                    "elected_measure": {
                        "label": "gross disbursements",
                        "alternatives": [
                            {"spec_id": _spec(other), "label": "net disbursements", "axis": "what"}
                            for other in range(_PLAN_MEASURE_CHOICE_CAP // _SPEC_ID_CHARS + 1)
                        ],
                    },
                }
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


def _maximal_readings() -> list[dict[str, Any]]:
    """Fifty readings, each far past what `_READINGS_BUDGET` holds.

    Each is a `definite` part with a 100,000-character sentence, ten candidates
    with 100,000-character labels and a warning, so `render_readings` sheds
    sentences and labels, folds every reading and then drops the oldest until
    the text fits.
    """
    long = "x" * _BIG
    return [
        SearchReading(
            text=f"aid to country {index}",
            status="definite",
            spec_id=_spec(index),
            sentence=long,
            candidates=tuple(
                CandidateReading(spec_id=_spec(1_000 + 10 * index + rank), label=long)
                for rank in range(10)
            ),
            warnings=(WarningReading(code="role_scope_defaulted"),),
        ).model_dump(mode="json")
        for index in range(50)
    ]


# The readings section of the brief, at most: `_READINGS_CONTEXT`'s own wording
# around a ledger `render_readings` holds to `_READINGS_BUDGET`.
READINGS_SECTION_CEILING = len(_READINGS_CONTEXT.format(readings="")) + _READINGS_BUDGET


def _readings_section(readings: Sequence[Mapping[str, Any]]) -> str:
    """The readings section `PlannerSeam._message` opens every brief with."""
    return _READINGS_CONTEXT.format(readings=_readings_ledger(readings).text)


def test_readings_far_past_the_budget_render_under_the_ceiling() -> None:
    """The budget bounds the readings section however many readings the conversation holds."""
    section = _readings_section(_maximal_readings())

    assert len(section) <= READINGS_SECTION_CEILING, (
        f"fifty oversized readings render {len(section):,} chars, past the "
        f"{READINGS_SECTION_CEILING:,} the budget derives"
    )
    assert len(section) > READINGS_SECTION_CEILING * 0.9, (
        f"the maximal readings render {len(section):,} chars, under 90% of the derived "
        f"{READINGS_SECTION_CEILING:,}: the readings or the derivation is off"
    )
    assert "earlier readings not shown" in section


def test_the_readings_ceiling_fails_when_the_budget_is_removed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The control for the readings ceiling: with the budget gone, it is exceeded."""
    monkeypatch.setattr(planner_seam, "_READINGS_BUDGET", 10**9)

    assert len(_readings_section(_maximal_readings())) > READINGS_SECTION_CEILING


# The two digests, built by `harness.studio_state`'s own functions.

_CONVERSATION = "conv_01budget0000"
_ASTRAL = "\U0001f600"
_BIG = 100_000


def _canvas_state(*, title: str, fact_summary: Any) -> ConversationState:
    """A conversation whose composition holds three times the block cap of components,
    each naming three times the show cap of 100,000-character entities."""
    state = ConversationState(conversation_id=_CONVERSATION)
    entries = {
        f"cmp_{index:016x}": {
            "type": "line_chart",
            "title": title,
            "revision": 999_999_999,
            "fact_summary": fact_summary,
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
    hundred wave siblings that each carry the same.

    Every stop names a refused spec, and is also a step refusing its sides on their units under a
    100,000-character step id: a `share_of` on the odd stops, which gets the
    repair, and a `compare` on the even ones, which earns a decline. The run
    computed 100,000 asks first. So the ask carries every sentence
    `PlannerSeam._message` can add to a refusal, each at its largest.
    """

    def refused(index: int) -> dict[str, Any]:
        return {
            "kind": "rediscovery_needed",
            "spec_id": _spec(index),
            "refusal": {"code": "over_ceiling", "message": "m" * _BIG},
            **_share_units_stop("s" * _BIG),
            "tool": "share_of" if index % 2 else "compare",
        }

    return {
        **refused(1),
        "also_stopped": [refused(index) for index in range(2, 102)],
        "computed_asks": list(range(_BIG)),
    }


def _share_units_stop(step_id: str) -> dict[str, Any]:
    """The members that make a stop a `share_of` step refusing its sides on their units."""
    return {"tool": "share_of", "code": "cross_frame_unit_unbound", "step_id": step_id}


def _share_units_repair_ceiling() -> int:
    """The most the sentence repairing refused `share_of` steps renders.

    The seam cuts its list of steps at `MAX_QUESTION_CHARS`, and the wording
    around the list is measured by rendering it for a placeholder.
    """
    one = planner_seam._share_units_repair_sentence(_share_units_stop("s"))
    return len(one) - len(repr("s")) + MAX_QUESTION_CHARS


def _failed_run_ceiling() -> int:
    """The most the sentence on what a failed run computed adds to the one for a plan held to no asks.

    The seam cuts its list of ask numbers at `MAX_QUESTION_CHARS`, and the
    sentence is longest when the asks section marks an ask the user removed.
    """
    one = planner_seam._failed_run_sentence([0], removed=True)
    listed = len(one) - len("0") + MAX_QUESTION_CHARS
    return listed - len(planner_seam._failed_run_sentence(None))


def _earned_decline_ceiling() -> int:
    """The sentence naming every decline a run-time refusal can earn."""
    earnable = set(planner_seam._UNSUPPORTED_BY_RUN_TIME_CODE.values())
    return len(planner_seam._earned_decline_sentence(earnable))


def _ask_kwargs(ask: str) -> dict[str, Any]:
    """`_seam_brief`'s arguments for the amendment ask or the refusal ask. The
    amendment follows replans that refused a hundred and one specs, whose
    sentence it carries."""
    if ask == "refusal":
        return {"refusal": _maximal_refusal()}
    return {"amendment": _BRIEF_AMENDMENT, "refused": [_spec(index) for index in range(1, 102)]}


def _ask_only(ask: str) -> str:
    """The amendment ask or the refusal ask a brief closes on, alone."""
    asked = {"amendment": None, "refusal": None, **_ask_kwargs(ask)}
    return planner_seam._ask_text(pick=None, **asked)


def _maximal_brief_parts(ask: str) -> dict[str, str]:
    """Every section of the largest brief `PlannerSeam._message` builds, apart.

    The parts are rendered by the functions `_message` calls, so the sum of
    their lengths can be checked against the whole. The question is the cut
    the seam makes of a 100,000-character one, and the ask is the amendment
    or refusal ask the brief closes on (`planner_seam._ask_text`).
    """
    cut_question = _BRIEF_QUESTION[: MAX_QUESTION_CHARS - len(CUT_MARKER)] + CUT_MARKER
    return {
        "readings": _readings_section(_maximal_readings()),
        "canvas digest": _section(_CANVAS_TITLE, _largest_canvas_digest()),
        "conversation digest": _section(_CONVERSATION_TITLE, _conversation_digest()),
        "plan": _plan_context(_legal_maximum_plan_record()),
        "question": f"\nQuestion: {cut_question}\n",
        "ask": _ask_only(ask),
    }


def _ask_chars(*, amendment: str | None = None, refusal: dict[str, Any] | None = None) -> int:
    """Characters of the ask a brief closes on (`planner_seam._ask_text`)."""
    return len(planner_seam._ask_text(amendment=amendment, refusal=refusal, pick=None))


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

    The question and the amendment are cut at `MAX_QUESTION_CHARS`, and so
    is the sentence naming the specs refused before the amendment. A refusal
    is cut there as JSON, then again in each sentence the ask adds to it: the
    asks the failed run computed, the specs to avoid, and the `share_of` steps
    to repair. The declines it earned are a closed set. The wording around each
    cut is measured from the renderer.
    """
    if ask == "refusal":
        framing = _ask_chars(refusal={"kind": "x"}) - len(json.dumps({"kind": "x"}))
        ask_ceiling = (
            framing
            + MAX_QUESTION_CHARS
            + _failed_run_ceiling()
            + _refused_sentence_ceiling(planner_seam._avoid_sentence)
            + _earned_decline_ceiling()
            + _share_units_repair_ceiling()
        )
    else:
        # The specs refused before the amendment render on a line of their own.
        ask_ceiling = (
            _ask_chars(amendment="x")
            - 1
            + MAX_QUESTION_CHARS
            + len("\n\n")
            + _refused_sentence_ceiling(planner_seam._avoid_sentence)
        )
    return {
        "readings": READINGS_SECTION_CEILING,
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
    """Every section of the brief at its maximum at once, held to the sum.

    Readings far past their budget, the two digests through their own functions, the
    largest legal plan, a question and the amendment or refusal ask, rendered by
    `PlannerSeam._message`. The message names each part with its size and
    ceiling, so a failure says which one moved.
    """
    parts = _maximal_brief_parts(ask)
    ceilings = _maximal_brief_ceilings(ask)
    brief = _seam_brief(
        readings=_maximal_readings(),
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
        readings=_maximal_readings(),
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

    question = _seam_brief(prior_plan=None, question=_BRIEF_QUESTION)
    asked = _ask_only(ask)

    assert len(question) > ceilings["question"]
    assert len(asked) > ceilings["ask"]


# The bounded production context, through the seam and ADK --------------------------
#
# The realistic round above runs with three readings and small digests. A
# session's first planner call can carry far more of all three in the brief, and
# each is bounded at its source, so this scenario records that call with every
# one at its maximum, as the request ADK assembles it.


def _bounded_context() -> dict[str, Any]:
    """What `PlannerSeam.__call__` is handed besides the plan and the question."""
    return {
        "readings": _maximal_readings(),
        "canvas_digest": _largest_canvas_digest(),
        "conversation_digest": _conversation_digest(),
    }


# The seam's brief in the `BOUNDED_CONTEXT` scenario: `SEAM_BRIEF_CEILING`, which
# already covers readings at their budget, a nine-requirement plan and the
# question, plus the two digests at their maxima (`CANVAS_SECTION_CEILING` and
# `CONVERSATION_SECTION_CEILING`).
BOUNDED_CONTEXT_BRIEF_CEILING = (
    SEAM_BRIEF_CEILING + CANVAS_SECTION_CEILING + CONVERSATION_SECTION_CEILING
)

# The planner's whole `contents` in that scenario: the realistic history's
# ceiling plus the brief's, plus the one skill page the round reads.
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
        f"conversation {CONVERSATION_SECTION_CEILING:,})"
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

    assert (
        properties("DecisionDraft")["description"].get("maxLength")
        == MAX_DECISION_DESCRIPTION_CHARS
    )
    assert properties("DecisionDraft")["rationale"].get("maxLength") == MAX_DECISION_RATIONALE_CHARS
    assert properties("DecisionDraft")["chosen"].get("maxLength") == MAX_OPTION_CHARS
    assert properties("DecisionDraft")["options"].get("maxItems") == MAX_DECISION_OPTIONS
    assert properties("DecisionDraft")["options"]["items"].get("maxLength") == MAX_OPTION_CHARS
    assert properties("DatasetRequirement")["measure"].get("maxLength") == MAX_MEASURE_CHARS
    assert properties("DatasetRequirement")["source"].get("maxLength") == MAX_SOURCE_CHARS
    assert properties("DatasetRequirement")["entities"].get("maxItems") == MAX_ENTITIES
    assert properties("Entity")["name"].get("maxLength") == MAX_ENTITY_NAME_CHARS
    assert properties("Entity")["role"].get("maxLength") == MAX_ENTITY_ROLE_CHARS
    assert properties("EntityIn")["names"].get("maxItems") == MAX_ENTITIES
    assert properties("FilterRequirement")["predicates"].get("maxItems") == MAX_PREDICATES
    unbounded = [
        path
        for name, model in definitions.items()
        for path in unbounded_paths(model, definitions, name)
    ]
    assert unbounded == []

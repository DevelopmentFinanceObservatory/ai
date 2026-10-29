"""An aligned input's compile across a Studio turn close, and a plan paused on an aligned input.

Turn 1 compares the US total with France's multilateral ODA. Alignment moves the
total onto DAC1's multilateral row, which Search answers under the spec_id it gives
US multilateral ODA. Turn 2, a fresh planner session after the turn closes, reads
US multilateral ODA as a share of the total, pinning that spec_id directly.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from analysis_mcp.server import mcp as analysis_mcp
from common.materialisation import FrameRef
from compiler.compile import compile_plan
from compiler.sources import source_from_spec_block
from compiler.surface import McpSurface, ToolCallResult, ToolSurface, ToolSurfaceUnavailableError
from data_mcp.server import mcp as data_mcp
from executor.outcomes import Accepted, EvidenceRecord
from harness.graph.execute import record_served_scopes
from harness.graph.plan_done import record_plan
from harness.graph.state import (
    ORCHESTRATOR_PLAN_RECORDS_KEY,
    ORCHESTRATOR_PRIOR_LEDGER_KEY,
    ORCHESTRATOR_SEARCHES_KEY,
    ORCHESTRATOR_SPECS_KEY,
)
from harness.planner_seam import PlannerSeam
from harness.scope_choices import planner_axes
from harness.settings import Settings
from harness.studio_state import ConversationState
from harness.turn_records import PlanRecord, SearchReading, basis_line, loose_match_notes
from planner.plan import (
    PLAN_STATE_KEY,
    DatasetRequirement,
    Period,
    Plan,
    ResolvedBy,
    current_earlier_block,
    load_plan,
    pin_sources,
    plan_revision,
)
from planner.workflow import ToolId
from translator.mechanical import build_translation

from .basis_alignment import WholeCompiles, us_2023_block
from .orchestrator_fakes import approve_answer
from .planner.test_agent import (
    _US_MULTILATERAL,
    _US_TOTAL,
    DecisionFactory,
    _another_part,
    _dac_countries,
    _difference,
    _FakeToolContext,
    _french_dac1,
    _share,
    _show_blocks,
    _tools_reading,
    _us_2023,
)
from .planner.test_aligned_identity import _multilateral_compiles
from .test_graph import (
    APPROVE_INTERRUPT,
    _ManifestSurface,
    _output_of,
    _plan_runner,
    _resume,
    _Seam,
    _start,
)
from .test_planner_seam import ScriptedPlannerLlm, _SeamHarness

_FRANCE = _french_dac1(_US_MULTILATERAL)
_MULTILATERAL = _US_MULTILATERAL["spec_id"]
_TOTAL = _US_TOTAL["spec_id"]
_US = "United States of America"
_MCP = McpSurface(sources={"data-mcp": data_mcp, "analysis-mcp": analysis_mcp})


def _searched(block: dict[str, Any], words: str) -> dict[str, Any]:
    """The turn's search reading of `block` under `words`, as `search` stores it."""
    return SearchReading.from_part(
        {
            "variable": words,
            "status": "definite",
            "spec_id": block["spec_id"],
            "scope": block["resolved"]["scope_choices"],
            "asked_roles": ["donor"],
        }
    ).model_dump(mode="json")


def _proposal(question: str, decision: DecisionFactory, *requirements: Any) -> Any:
    """A `propose_plan` call stating `requirements`, as the planner model sends it."""
    return (
        "propose_plan",
        {
            "question": question,
            "answer_mode": "computed",
            "decisions": [decision().model_dump(mode="json", exclude={"fixes"})],
            "evidence_requirements": [r.model_dump(mode="json") for r in requirements],
        },
    )


async def _compiled(plan: Plan, blocks: dict[str, dict[str, Any]]) -> Plan:
    """`plan` translated and compiled against data-mcp's and analysis-mcp's schemas,
    each input read from the block of its pin in `blocks`, awaiting approval."""
    pins = {
        r.id: r.source_spec_id
        for r in plan.evidence_requirements
        if isinstance(r, DatasetRequirement) and r.source_spec_id is not None
    }
    discovered = {rid: source_from_spec_block(cast(Any, blocks[pin])) for rid, pin in pins.items()}
    translation = build_translation(plan, {rid: s.spec_id for rid, s in discovered.items()})
    compiled = compile_plan(
        plan=plan,
        translation=translation,
        schemas={tool.name: tool for tool in await _MCP.list_tools()},
        discovered=discovered,
    )
    assert compiled.workflow is not None, compiled.issues
    return plan.with_translation(translation=translation, workflow=compiled.workflow)


_COMPARISON = "How does US ODA compare with France's multilateral ODA?"


async def _turn_one(
    decision: DecisionFactory, *, us: dict[str, Any] | None = None
) -> tuple[dict[str, Any], Plan]:
    """Run turn 1, which compares the US total, read as `us` (its reading under
    "US ODA" by default), with France's multilateral ODA and so aligns the total
    onto US multilateral ODA's spec_id. Returns the turn's state and its plan,
    compiled."""
    surface = cast(ToolSurface, _multilateral_compiles())
    searched = [
        us or _searched(_US_TOTAL, "US ODA"),
        _searched(_FRANCE, "France's multilateral ODA"),
    ]
    turn_one = _SeamHarness(
        {
            ORCHESTRATOR_SEARCHES_KEY: searched,
            ORCHESTRATOR_SPECS_KEY: {_TOTAL: _US_TOTAL, _FRANCE["spec_id"]: _FRANCE},
        }
    )
    seam = PlannerSeam(settings=Settings(), surface=surface)
    seam._agent.model = ScriptedPlannerLlm(
        script=[
            [
                _proposal(
                    _COMPARISON,
                    decision,
                    _dac_countries("second", _US_TOTAL),
                    _dac_countries("other", _FRANCE),
                    _difference("second", "other"),
                )
            ]
        ]
    )
    await turn_one.run(seam, question=_COMPARISON, readings=searched)
    state = await turn_one.state()
    compared = load_plan(state)
    assert compared is not None
    assert {
        r.id: r.source_spec_id
        for r in compared.evidence_requirements
        if isinstance(r, DatasetRequirement)
    } == {"second": _MULTILATERAL, "other": _FRANCE["spec_id"]}
    recorded = await _compiled(
        compared, {_MULTILATERAL: _US_MULTILATERAL, _FRANCE["spec_id"]: _FRANCE}
    )
    return state, recorded


async def _turn_one_closed(decision: DecisionFactory) -> dict[str, Any]:
    """Run turn 1 (`_turn_one`), record its plan, whose compile source carries the
    count of the total moved onto it (`PlanSource.aligned_basis`), and close it as
    the Studio does. Returns the ledger turn 2 opens with, with the closed turn's
    blocks set as the earlier blocks (`current_earlier_block`)."""
    state, recorded = await _turn_one(decision)
    conversation = ConversationState(conversation_id="conv_aligned")
    conversation.turn_user_message = _COMPARISON
    conversation.turn_readings = list(state[ORCHESTRATOR_SEARCHES_KEY])
    conversation.turn_specs = dict(state[ORCHESTRATOR_SPECS_KEY])
    record_plan(state, recorded)
    conversation.turn_plans = list(state[ORCHESTRATOR_PLAN_RECORDS_KEY])
    (aligned,) = [
        source
        for source in PlanRecord.model_validate(conversation.turn_plans[0]).sources
        if source.spec_id == _MULTILATERAL
    ]
    assert aligned.aligned_basis is not None
    conversation.close_turn(final_text="The US gave more.")
    prior = conversation.prior_ledger_state()
    assert prior is not None
    current_earlier_block.set(conversation.earlier_block)
    return prior


async def test_an_aligned_inputs_loose_stated_series_keeps_its_note(
    decision: DecisionFactory,
) -> None:
    """Search answered "US ODA" with the US total only as its closest match, and
    alignment moved the total onto the multilateral row. The plan retrieves the
    compile alone, yet the answer still tells the user the series their words got
    is a loose match: the compile moved only the provider."""
    us = _searched(_US_TOTAL, "US ODA")
    us["departures"] = [{"code": "closest_match", "text": "You asked for X. This shows Y."}]
    state, recorded = await _turn_one(decision, us=us)
    record_plan(state, recorded)

    (note,) = loose_match_notes(
        state[ORCHESTRATOR_SEARCHES_KEY],
        specs=state[ORCHESTRATOR_SPECS_KEY],
        plans=state[ORCHESTRATOR_PLAN_RECORDS_KEY],
    )
    assert note.startswith('The closest series to "US ODA" is a loose match')


async def test_an_aligned_input_is_counted_on_its_compiles_basis(
    decision: DecisionFactory,
) -> None:
    """The total's own reading counts DAC1's total, and alignment moved it onto the
    multilateral row. Read with France's part left out, the basis line states the
    total's count on the provider of the compile the plan retrieved, core
    contributions, where its own reading holds bilateral plus core contributions."""
    state, recorded = await _turn_one(decision)
    record_plan(state, recorded)
    (us,) = [r for r in state[ORCHESTRATOR_SEARCHES_KEY] if r["spec_id"] == _TOTAL]

    line = basis_line(
        [us], specs=state[ORCHESTRATOR_SPECS_KEY], plans=state[ORCHESTRATOR_PLAN_RECORDS_KEY]
    )

    assert (
        line
        == "Counted as: gross disbursements, current USD · core contributions to multilaterals."
    )


async def test_an_aligned_inputs_count_states_how_its_compile_was_served(
    decision: DecisionFactory,
) -> None:
    """The plan's retrieval of the compile the total was moved onto was served as
    the OECD headline, net to 2017 and grant equivalents from 2018. No reading holds
    the compile, yet the basis line states the served count in place of the
    compile's gross option, as it does for a reading's retrieval."""
    state, recorded = await _turn_one(decision)
    frame = FrameRef(spec_id=_MULTILATERAL, content_version="cv_" + "1" * 32)
    outcome = Accepted(
        revision="rev",
        evidence=(
            EvidenceRecord(
                requirement_id="second",
                step_id="second_op",
                frame_ref=frame,
                spec_id=_MULTILATERAL,
                stat_vars=(),
                window=None,
                gaps=(),
                citation="DAC1",
            ),
        ),
        steps=(),
        results={},
    )
    identity = _US_MULTILATERAL["resolved"]["scope_choices"]["compiled"]["identity"]
    manifest = {
        "definition": {"schema_version": 1, "spec_id": _MULTILATERAL, "identity": identity},
        "coverage": {"window": {"start_year": 2015, "end_year": 2024}},
        "trace": {
            "components": [
                {"role": role, "periods": spans, "dropped_years": [], "rows": 10, "zero_rows": 0}
                for role, spans in (
                    ("headline_net", [[2015, 2017]]),
                    ("headline_ge", [[2018, 2024]]),
                )
            ]
        },
    }
    record_served_scopes(state, cast(Any, _ManifestSurface(frame, manifest)), outcome)
    record_plan(state, recorded)
    (us,) = [r for r in state[ORCHESTRATOR_SEARCHES_KEY] if r["spec_id"] == _TOTAL]

    line = basis_line(
        [us], specs=state[ORCHESTRATOR_SPECS_KEY], plans=state[ORCHESTRATOR_PLAN_RECORDS_KEY]
    )

    assert line == (
        "Counted as: net to 2017, grant equivalents from 2018 · "
        "core contributions to multilaterals."
    )


async def test_a_turn_two_direct_pin_of_a_turn_one_compile_reads_no_planner_marks(
    decision: DecisionFactory,
) -> None:
    """Turn 1 aligns the US total onto US multilateral ODA's spec_id, and the Studio
    closes the turn. Turn 2, a fresh planner session, pins that spec_id directly as
    the numerator of a share of the total. The direct pin reads no block and no
    planner marks, since the planner settled nothing of it, though turn 1's plan
    record holds how the compile counted the total. The share keeps the total on its
    own pin, so it does not divide one series by itself."""
    prior = await _turn_one_closed(decision)
    surface = cast(ToolSurface, _multilateral_compiles())

    turn_two = _SeamHarness({ORCHESTRATOR_PRIOR_LEDGER_KEY: prior})
    seam = PlannerSeam(settings=Settings(), surface=surface)
    share = "What share of US ODA goes to multilaterals?"
    seam._agent.model = ScriptedPlannerLlm(
        script=[
            [
                _proposal(
                    share,
                    decision,
                    _dac_countries("first", _US_MULTILATERAL),
                    _dac_countries("second", _US_TOTAL),
                    _share("first", "second"),
                )
            ]
        ]
    )
    await turn_two.run(seam, question=share, readings=prior["readings"])
    state = await turn_two.state()

    plan = load_plan(state)
    assert plan is not None
    assert {
        r.id: r.source_spec_id
        for r in plan.evidence_requirements
        if isinstance(r, DatasetRequirement)
    } == {"first": _MULTILATERAL, "second": _TOTAL}
    (reading,) = [
        SearchReading.model_validate(r) for r in prior["readings"] if r["spec_id"] == _MULTILATERAL
    ]
    assert reading.planner_axes == ()
    direct = pin_sources(state)[_MULTILATERAL]
    assert direct.spec_block(_MULTILATERAL) is None
    assert planner_axes(direct.spec_block(_MULTILATERAL)) == frozenset()


async def test_a_turn_two_restatement_of_the_comparison_moves_the_total_again(
    decision: DecisionFactory,
) -> None:
    """Turn 1 aligns the US total onto US multilateral ODA's spec_id. Turn 2 asks for
    the same comparison over 2020. Each turn opens a fresh planner session with no
    plan on record, so the planner restates both inputs on the sources it reads: the
    total on its own pin, whose block turn 1 kept as the pin alignment moved. Alignment
    moves it onto the multilateral row again, and the card's provider row for it is
    the planner's, keyed by the total's id."""
    prior = await _turn_one_closed(decision)
    surface = _multilateral_compiles()
    turn_two = _SeamHarness({ORCHESTRATOR_PRIOR_LEDGER_KEY: prior})
    seam = PlannerSeam(settings=Settings(), surface=cast(ToolSurface, surface))
    question = "And in 2020?"
    seam._agent.model = ScriptedPlannerLlm(
        script=[
            [
                _proposal(
                    question,
                    decision,
                    _dac_countries("second", _US_TOTAL).model_copy(
                        update={"period": Period(start=2020, end=2020)}
                    ),
                    _dac_countries("other", _FRANCE).model_copy(
                        update={"period": Period(start=2020, end=2020)}
                    ),
                    _difference("second", "other"),
                )
            ]
        ]
    )
    await turn_two.run(seam, question=question, readings=prior["readings"])
    state = await turn_two.state()

    plan = load_plan(state)
    assert plan is not None
    assert {
        r.id: r.source_spec_id
        for r in plan.evidence_requirements
        if isinstance(r, DatasetRequirement)
    } == {"second": _MULTILATERAL, "other": _FRANCE["spec_id"]}
    assert len(surface.calls) == 1
    provider = [
        (d.slot, d.resolved_by)
        for d in plan.decisions
        if d.slot in plan.requirement("second").decision_ids
        and d.scope is not None
        and d.scope.axis == "provider"
    ]
    assert provider == [("scope_provider_of_second", ResolvedBy.PLANNER_DEFAULT)]


async def test_a_declared_wholes_dataset_reaches_a_later_turn_under_the_wholes_own_words(
    decision: DecisionFactory,
) -> None:
    """Turn 1 reads US aid to health as a share of US total ODA, declared the health
    part over every sector, so admission compiles the whole from the part. The
    compile is no reading of turn 1: it reaches turn 2's ledger as a source the
    translator found, under the words the whole's own requirement states, never
    the part's, and the conversation digest names it by its source."""
    ctx = _FakeToolContext()
    health = us_2023_block("health")
    whole = us_2023_block("whole")
    _show_blocks(ctx, health)
    result = await _us_2023(
        _tools_reading(cast(ToolSurface, WholeCompiles()))["propose_plan"].func,
        ctx,
        decision(),
        part="health",
        whole=None,
        declared=True,
    )
    assert result["outcome"] == "ok", result
    plan = load_plan(ctx.state)
    assert plan is not None
    # The recorded whole states no coverage window and names the donor as the
    # graph labels it; the compile needs a window and the part's name.
    covered = json.loads(json.dumps(whole).replace(_US, "United States"))
    covered["resolved"].setdefault("coverage", {})["window"] = {
        "start_year": 2023,
        "end_year": 2023,
    }
    recorded = await _compiled(plan, {health["spec_id"]: health, whole["spec_id"]: covered})

    conversation = ConversationState(conversation_id="conv_whole")
    conversation.turn_user_message = plan.question
    conversation.turn_readings = [_searched(health, "US aid to health")]
    conversation.turn_specs = {health["spec_id"]: health}
    conversation.turn_plans = [
        PlanRecord.from_plan(recorded, searched=[health["spec_id"]]).model_dump(mode="json")
    ]
    conversation.close_turn(final_text="About a tenth.")

    prior = conversation.prior_ledger_state()
    assert prior is not None
    (reading,) = [
        SearchReading.model_validate(r)
        for r in prior["readings"]
        if r["spec_id"] == whole["spec_id"]
    ]
    assert (reading.text, reading.origin) == ("US total ODA", "translator")
    digest = conversation.conversation_digest_state()
    assert digest is not None
    (line,) = digest["turns"][-1]["plans"]
    assert whole["spec_id"] in line
    assert "US aid to health" not in line


# The France block `fixtures/aligned_total_plan_stored.json` was recorded against.
_STORED_FRANCE = _another_part(_US_MULTILATERAL)


async def test_a_paused_plan_stored_before_the_change_approves_on_its_revision() -> None:
    """`fixtures/aligned_total_plan_stored.json` holds the session state the comparison
    above leaves, stored before a requirement recorded what alignment did: the plan,
    its total moved onto the multilateral row, and that compile held as an aligned
    pin. The plan loads under the revision it was stored with, and, paused on its
    card, approves under the revision the card shows."""
    recorded = json.loads(
        (Path(__file__).parent / "fixtures" / "aligned_total_plan_stored.json").read_text()
    )
    stored = {
        **recorded["state"],
        PLAN_STATE_KEY: json.dumps(recorded["state"][PLAN_STATE_KEY]),
    }
    plan = load_plan(stored)
    assert plan is not None
    assert plan_revision(plan) == recorded["revision"]
    paused = await _compiled(
        plan,
        {_MULTILATERAL: _US_MULTILATERAL, _STORED_FRANCE["spec_id"]: _STORED_FRANCE},
    )
    seam = _Seam([{"plan": paused.model_dump(mode="json"), "summary": {}}])
    surface = _Unreachable()
    runner, _ = _plan_runner(seam, tool_surface=surface)
    events = await _start(runner)
    offered = _output_of(events, "planner")
    assert offered["revision"] == plan_revision(paused)

    await _resume(runner, APPROVE_INTERRUPT, approve_answer(offered["revision"]))

    assert ("get_data", _MULTILATERAL) in surface.calls


class _Unreachable:
    """A tool surface whose servers cannot be reached, recording each call as its tool
    name and the spec_id it reads."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    async def list_tools(self) -> list[Any]:
        return await _MCP.list_tools()

    async def call_tool(
        self, *, tool: ToolId, arguments: Mapping[str, Any], on_progress: Any = None
    ) -> ToolCallResult:
        del on_progress
        self.calls.append((tool.name, arguments.get("spec_id")))
        raise ToolSurfaceUnavailableError(f"{tool.name} unreachable")

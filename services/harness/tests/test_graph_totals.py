"""Tests for what `harness.graph` says about a summed family of StatVars.

`_carried_retrievals` picks the retrievals whose StatVars a terminal frame
carries, which the card counts its indicators from. `_stat_var_totals`
writes the sentence `plan_done` hands the agent for each summed family an
answer frame carries, and `_dropped_key_sentences` the one for each step
under an answer frame that left keys out.

The plans are compiled by `compile_plan` against the real tool schemas, over
the spec blocks recorded from dev apis in
`tests/fixtures/family_spec_blocks.json` where a test quotes a user question.
Each `get_data` result declares the role keys and coverage dimensions the
live frame declares: a CRS frame keys on `financingProvider` (origin axis) and
`financingRecipient` (destination axis), whatever roles the source resolves.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest
from analysis_mcp.server import mcp as analysis_mcp
from common.materialisation import FrameRef
from common.period_note import period_note
from compiler.compile import compile_plan
from compiler.sources import source_from_spec_block
from compiler.surface import McpSurface, ToolSchema, ToolSurface
from data_mcp.server import mcp as data_mcp
from executor.outcomes import Accepted, EvidenceRecord, StepRecord
from harness.conversation import current_conversation_id
from harness.graph.answer import (
    _carried_parts,
    _carried_retrievals,
    _dropped_key_sentences,
    _group_column,
    _stat_var_totals,
    answer_notes,
    link_evidence_to_terminal_frames,
)
from harness.provenance import ProvenanceStore
from harness.studio_translate import _RecordingSurface
from planner.operations import (
    AggregateOperation,
    CompareOperation,
    FilterOperation,
    JoinOperation,
    Operation,
    QuotientOfOperation,
    RankOperation,
    RetrieveOperation,
    ShareOfOperation,
    Side,
    Translation,
)
from planner.plan import (
    AggregateRequirement,
    AnswerMode,
    ComparisonRequirement,
    DatasetRequirement,
    Decision,
    DecisionDraft,
    DecisionFixes,
    Entity,
    EvidenceRequirement,
    FilterRequirement,
    JoinRequirement,
    Plan,
    PlanStatus,
    QuotientRequirement,
    RankAmong,
    RankDirection,
    RankRequirement,
    RequirementSide,
    ResolvedBy,
    ShareOfRequirement,
    settle_decisions,
)
from planner.records import GroupBy, Period, PeriodThreshold, Reduction, StatVarReading
from planner.workflow import (
    EvidenceBinding,
    OpenAxis,
    ResolvedEntity,
    ResolvedGroup,
    SelectedSource,
    ToolId,
    Workflow,
    WorkflowStep,
)

_BLOCKS: dict[str, Any] = json.loads(
    (Path(__file__).parent / "fixtures" / "family_spec_blocks.json").read_text()
)
CRS = "OECD Creditor Reporting System (CRS)"
CRS_ROLE_KEYS = ("financingProvider", "financingRecipient")
CRS_DIMENSIONS = (
    {"label": "financingProvider", "count": 1, "axis": "origin"},
    {"label": "financingRecipient", "count": 157, "axis": "destination"},
    {"label": "years", "count": 27},
)
REGIONAL = " They include regional and unallocated flows as well as flows to individual countries."
_WHAT = Decision(
    slot="measure",
    description="Which measure answers the question.",
    options=["the CRS flow", "another source"],
    chosen="the CRS flow",
    resolved_by=ResolvedBy.PLANNER_DEFAULT,
    rationale="The CRS reports the flow by donor and recipient.",
)


@pytest.fixture
async def schemas() -> dict[str, ToolSchema]:
    """The real servers' tool schemas, which the capability table is pinned to."""
    tools = await McpSurface(
        sources={"data-mcp": data_mcp, "analysis-mcp": analysis_mcp}
    ).list_tools()
    return {tool.name: tool for tool in tools}


def _recorded(name: str) -> SelectedSource:
    return source_from_spec_block(_BLOCKS[name]["parts"][0]["spec"])


def _family(spec_char: str, label: str) -> SelectedSource:
    """A Kenya family of three StatVars open on one summable axis, which the rule sums."""
    return SelectedSource(
        spec_id="spec_" + spec_char * 32,
        sentence=f"{label} of Kenya, 2015 to 2024",
        has_data=True,
        served_count=3,
        entities=(ResolvedEntity(role="recipient", ref="country/KEN", label="Kenya"),),
        unit_ref="usd",
        unit_label="US dollars",
        coverage_window=Period(start=2015, end=2024),
        identity={"stat_var_schema": label},
        open_axes=(OpenAxis(property="crsPurpose", summable=True),),
    )


def _single(spec_char: str, stat_var: str) -> SelectedSource:
    """A Kenya source binding one StatVar, which no reading applies to."""
    return _family(spec_char, stat_var).model_copy(
        update={"served_count": 1, "open_axes": (), "identity": {"stat_var": stat_var}}
    )


def _dataset(
    requirement_id: str, entities: Sequence[Entity], period: Period, **fields: Any
) -> DatasetRequirement:
    return DatasetRequirement(
        id=requirement_id,
        decision_ids=fields.pop("decision_ids", ["measure"]),
        measure=fields.pop("measure", "official development assistance"),
        entities=entities,
        period=period,
        **fields,
    )


def _kenya(requirement_id: str, measure: str) -> DatasetRequirement:
    return _dataset(
        requirement_id,
        [Entity(name="Kenya", role="recipient")],
        Period(start=2015, end=2024),
        measure=measure,
    )


def _yearly_total(
    input_id: str, operation_input: str, *, reduction: Reduction = Reduction.SUM
) -> tuple[AggregateRequirement, Operation]:
    requirement = AggregateRequirement(
        id="yearly_total",
        decision_ids=["measure"],
        input_id=input_id,
        reduction=reduction,
        group_by=(GroupBy.PERIOD,),
    )
    operation = AggregateOperation(
        id="yearly_total",
        requirement_id="yearly_total",
        input=operation_input,
        reduction=reduction,
        group_by=(GroupBy.PERIOD,),
    )
    return requirement, operation


def _compiled(
    schemas: Mapping[str, ToolSchema],
    requirements: Sequence[EvidenceRequirement],
    operations: Sequence[Operation],
    *,
    terminal: Sequence[str],
    discovered: Mapping[str, SelectedSource],
    decisions: Sequence[Decision] = (_WHAT,),
    answer_mode: AnswerMode = AnswerMode.COMPUTED,
) -> Plan:
    draft = Plan(
        question="a question about a family of StatVars",
        answer_mode=answer_mode,
        decisions=decisions,
        evidence_requirements=requirements,
        status=PlanStatus.DRAFT,
    )
    translation = Translation(operations=operations, terminal=terminal)
    compiled = compile_plan(
        plan=draft, translation=translation, schemas=dict(schemas), discovered=dict(discovered)
    )
    assert compiled.workflow is not None, compiled.issues
    return draft.with_translation(translation=translation, workflow=compiled.workflow).approve(
        at=datetime.now(UTC)
    )


def _digest(step_id: str) -> str:
    return hashlib.sha256(step_id.encode()).hexdigest()[:32]


def _frame(step_id: str) -> FrameRef:
    return FrameRef(spec_id="ops_" + _digest(step_id), content_version="ocv_" + _digest(step_id))


def _retrieved(spec_id: str, step_id: str) -> FrameRef:
    return FrameRef(spec_id=spec_id, content_version="cv_" + _digest(step_id))


def _manifest(
    frame: FrameRef,
    *,
    summed: int | None,
    label: str | None = None,
    role_keys: Sequence[str] = CRS_ROLE_KEYS,
    dimensions: Sequence[Mapping[str, Any]] = CRS_DIMENSIONS,
) -> dict[str, Any]:
    """A `get_data` result as data-mcp returns it, cut to the fields the totals sentences read."""
    return {
        "schema_version": 1,
        "definition": {"schema_version": 1, "spec_id": frame.spec_id, "identity": {}},
        "content_version": frame.content_version,
        "coverage": {"rows": 688, "dimensions": list(dimensions)},
        "citation": {"text": CRS, "sources": []},
        "trace": {},
        "warnings": [],
        "artifacts": [],
        "role_keys": list(role_keys),
        "stat_var_count": 1,
        "summed_stat_var_count": summed,
        "summed_stat_var_label": label,
        "frame_ref": frame.model_dump(),
    }


class _Surface:
    """Answers `retrieval_manifest` from the `get_data` results it was given."""

    def __init__(self, manifests: Sequence[dict[str, Any]]) -> None:
        self.manifests = {
            (manifest["definition"]["spec_id"], manifest["content_version"]): manifest
            for manifest in manifests
        }

    def retrieval_manifest(self, *, frame_ref: Mapping[str, str | None]) -> dict[str, Any] | None:
        return self.manifests.get((frame_ref["spec_id"], frame_ref["content_version"]))


def _run(
    plan: Plan,
    *,
    summed: Mapping[str, int | None],
    role_keys: Mapping[str, tuple[str, ...] | None],
    labels: Mapping[str, str] | None = None,
    retrieval_role_keys: Sequence[str] = CRS_ROLE_KEYS,
) -> tuple[_Surface, Accepted]:
    """The surface and outcome of `plan` having run: each retrieval's `get_data`
    result sums `summed[step_id]` StatVars, and each step declares
    `role_keys[step_id]` (a retrieval its frame's own, `retrieval_role_keys`)."""
    workflow = plan.workflow
    assert workflow is not None
    frames = {
        binding.step_id: _retrieved(binding.spec_id, binding.step_id)
        for binding in workflow.evidence_bindings
    }
    manifests = [
        _manifest(
            frames[step_id],
            summed=summed.get(step_id),
            label=(labels or {}).get(step_id),
            role_keys=retrieval_role_keys,
        )
        for step_id in frames
    ]
    results = {step_id: frames.get(step_id, _frame(step_id)) for step_id in workflow.terminal_steps}
    outcome = Accepted(
        revision="rev",
        evidence=tuple(
            EvidenceRecord(
                requirement_id=binding.requirement_id,
                step_id=binding.step_id,
                frame_ref=frames[binding.step_id],
                spec_id=binding.spec_id,
                stat_vars=(),
                window=None,
                gaps=(),
                citation=CRS,
            )
            for binding in workflow.evidence_bindings
        ),
        steps=tuple(
            StepRecord(
                step_id=step.id,
                tool=step.tool,
                attempts=1,
                frame_ref=frames.get(step.id) or results.get(step.id) or _frame(step.id),
                digest="",
                role_keys=(
                    tuple(retrieval_role_keys) if step.id in frames else role_keys.get(step.id)
                ),
                stat_var_count=1,
            )
            for step in workflow.steps
        ),
        results=results,
    )
    return _Surface(manifests), outcome


def _germany_bilateral_oda(
    schemas: Mapping[str, ToolSchema],
    *,
    then: str = "aggregate",
    reduction: Reduction = Reduction.SUM,
    **dataset_fields: Any,
) -> Plan:
    """ "Chart Germany's bilateral ODA since 2000" over the recorded source,
    which resolves Germany alone, so the compiler predicts the retrieval's
    role keys as `("origin",)`. `then` is the step the answer reads: the
    yearly `reduction` across recipients, or a filter keeping each recipient."""
    source = _recorded("germany_bilateral_oda")
    decisions = dataset_fields.pop("decisions", (_WHAT,))
    oda = _dataset(
        "oda",
        [Entity(name="Germany", role="donor")],
        Period(start=2000, end=2026),
        **dataset_fields,
    )
    retrieve = RetrieveOperation(id="oda_op", requirement_id="oda", spec_id=source.spec_id)
    if then == "aggregate":
        total, total_op = _yearly_total("oda", "oda_op", reduction=reduction)
        return _compiled(
            schemas,
            [oda, total],
            [retrieve, total_op],
            terminal=["yearly_total"],
            discovered={"oda": source},
            decisions=decisions,
        )
    from_2005 = FilterRequirement(
        id="from_2005",
        decision_ids=["measure"],
        input_id="oda",
        predicates=(PeriodThreshold(op="gte", year=2005),),
    )
    return _compiled(
        schemas,
        [oda, from_2005],
        [
            retrieve,
            FilterOperation(
                id="from_2005_op",
                requirement_id="from_2005",
                input="oda_op",
                predicates=(PeriodThreshold(op="gte", year=2005),),
            ),
        ],
        terminal=["from_2005_op"],
        discovered={"oda": source},
        decisions=decisions,
        answer_mode=AnswerMode.SERIES,
    )


def _by_default(count: str, name: str = CRS) -> str:
    return f'By default, the figures for "{name}" sum the family\'s {count} indicators.'


@pytest.mark.parametrize(
    ("reduction", "regional"),
    [
        (Reduction.SUM, REGIONAL),
        (
            Reduction.MEAN,
            " They are computed from regional and unallocated flows as well as flows to "
            "individual countries.",
        ),
    ],
    ids=["summed", "averaged"],
)
async def test_germany_s_bilateral_oda_totalled_each_year_includes_regional_flows(
    schemas: dict[str, ToolSchema], reduction: Reduction, regional: str
) -> None:
    plan = _germany_bilateral_oda(schemas, reduction=reduction)
    assert plan.workflow is not None
    (retrieval,) = (step for step in plan.workflow.steps if step.id == "oda_op")
    # The compiler predicts only the role Germany resolves under; the frame
    # declares both, and the yearly total keeps neither.
    assert retrieval.expected_role_keys == ("origin",)
    assert retrieval.args["stat_vars"] == "sum"

    surface, outcome = _run(plan, summed={"oda_op": 1554}, role_keys={"yearly_total": ()})

    assert _stat_var_totals(surface, outcome, plan=plan) == [_by_default("1,554") + regional]


async def test_a_filter_keeping_each_recipient_states_the_family_count_alone(
    schemas: dict[str, ToolSchema],
) -> None:
    plan = _germany_bilateral_oda(schemas, then="filter")

    surface, outcome = _run(plan, summed={"oda_op": 74}, role_keys={"from_2005_op": CRS_ROLE_KEYS})

    assert _stat_var_totals(surface, outcome, plan=plan) == [_by_default("74")]


async def test_germany_s_oda_to_kenya_names_its_recipient_and_states_no_regional_flows(
    schemas: dict[str, ToolSchema],
) -> None:
    source = _recorded("germany_oda_to_kenya")
    oda = _dataset(
        "oda",
        [Entity(name="Germany", role="donor"), Entity(name="Kenya", role="recipient")],
        Period(start=2000, end=2024),
    )
    total, total_op = _yearly_total("oda", "oda_op")
    plan = _compiled(
        schemas,
        [oda, total],
        [RetrieveOperation(id="oda_op", requirement_id="oda", spec_id=source.spec_id), total_op],
        terminal=["yearly_total"],
        discovered={"oda": source},
    )

    surface, outcome = _run(plan, summed={"oda_op": 12}, role_keys={"yearly_total": ()})

    assert _stat_var_totals(surface, outcome, plan=plan) == [_by_default("12")]


async def test_a_reading_the_user_settled_says_so_and_the_planners_own_says_nothing(
    schemas: dict[str, ToolSchema],
) -> None:
    """The user's decision is built as `propose_plan` builds it: the draft
    names `stat_vars` and its value comes off the requirement citing it."""
    draft = DecisionDraft(
        slot="reading",
        description="Read the aid family summed or as separate series",
        options=["sum", "separate"],
        chosen="sum",
        resolved_by=ResolvedBy.USER,
        rationale="The user asked for one total",
        fixes=("stat_vars",),
    )
    citing = _dataset(
        "oda",
        [Entity(name="Germany", role="donor")],
        Period(start=2000, end=2026),
        decision_ids=["measure", "reading"],
        stat_vars=StatVarReading.SUM,
    )
    (settled,) = settle_decisions([draft], [citing])
    user_plan = _germany_bilateral_oda(
        schemas,
        decision_ids=["measure", "reading"],
        stat_vars=StatVarReading.SUM,
        decisions=(_WHAT, settled),
    )
    planner_plan = _germany_bilateral_oda(schemas, stat_vars=StatVarReading.SUM)

    for plan, expected in (
        (user_plan, f'As you asked, the figures for "{CRS}" sum the family\'s 74 indicators.'),
        (planner_plan, f'The figures for "{CRS}" sum the family\'s 74 indicators.'),
    ):
        surface, outcome = _run(plan, summed={"oda_op": 74}, role_keys={"yearly_total": ()})
        assert _stat_var_totals(surface, outcome, plan=plan) == [expected + REGIONAL]


def _two_sided(
    schemas: Mapping[str, ToolSchema], kind: str, left: SelectedSource, right: SelectedSource
) -> Plan:
    """`left` and `right` retrieved for Kenya and read by one `kind` step, `combined`."""
    left_req, right_req = _kenya("left", "grants"), _kenya("right", "loans")
    derived: EvidenceRequirement
    operation: Operation
    match kind:
        case "join":
            derived = JoinRequirement(
                id="combined", decision_ids=["measure"], left_id="left", right_id="right"
            )
            operation = JoinOperation(
                id="combined", requirement_id="combined", left="left_op", right="right_op"
            )
        case "compare":
            derived = ComparisonRequirement(
                id="combined",
                decision_ids=["measure"],
                minuend=RequirementSide(input_id="right"),
                subtrahend=RequirementSide(input_id="left"),
            )
            operation = CompareOperation(
                id="combined",
                requirement_id="combined",
                left=Side(input="left_op"),
                right=Side(input="right_op"),
            )
        case "share_of":
            derived = ShareOfRequirement(
                id="combined",
                decision_ids=["measure"],
                numerator=RequirementSide(input_id="left"),
                denominator=RequirementSide(input_id="right"),
            )
            operation = ShareOfOperation(
                id="combined",
                requirement_id="combined",
                numerator=Side(input="left_op"),
                denominator=Side(input="right_op"),
            )
        case _:
            raise AssertionError(kind)
    return _compiled(
        schemas,
        [left_req, right_req, derived],
        [
            RetrieveOperation(id="left_op", requirement_id="left", spec_id=left.spec_id),
            RetrieveOperation(id="right_op", requirement_id="right", spec_id=right.spec_id),
            operation,
        ],
        terminal=["combined"],
        discovered={"left": left, "right": right},
    )


KENYA_ROLE_KEYS = ("financingRecipient",)
KENYA_DIMENSIONS = ({"label": "financingRecipient", "count": 1, "axis": "destination"},)


@pytest.mark.parametrize("kind", ["join", "share_of", "compare"])
async def test_two_summed_families_on_both_sides_get_one_sentence_each_naming_its_family(
    schemas: dict[str, ToolSchema], kind: str
) -> None:
    """A join, a share and a compare carry both sides' StatVars, so each
    summed side's family gets its own sentence."""
    plan = _two_sided(schemas, kind, _family("a", "Grants"), _family("b", "Loans"))

    surface, outcome = _run(
        plan,
        summed={"left_op": 3, "right_op": 5},
        labels={"left_op": "Grants to Kenya", "right_op": "Loans to Kenya"},
        role_keys={"combined": KENYA_ROLE_KEYS},
        retrieval_role_keys=KENYA_ROLE_KEYS,
    )

    assert _stat_var_totals(surface, outcome, plan=plan) == [
        _by_default("3", "Grants to Kenya"),
        _by_default("5", "Loans to Kenya"),
    ]


async def test_two_retrievals_of_one_spec_each_state_their_own_reading(
    schemas: dict[str, ToolSchema],
) -> None:
    """One family read twice, once by the rule and once as the plan asked:
    each retrieval's sentence names who set its own reading."""
    family = _family("a", "Grants")
    by_rule = _kenya("by_rule", "grants")
    by_plan = _dataset(
        "by_plan",
        [Entity(name="Kenya", role="recipient")],
        Period(start=2015, end=2024),
        measure="grants",
        stat_vars=StatVarReading.SUM,
    )
    plan = _compiled(
        schemas,
        [by_rule, by_plan],
        [
            RetrieveOperation(id="by_rule_op", requirement_id="by_rule", spec_id=family.spec_id),
            RetrieveOperation(id="by_plan_op", requirement_id="by_plan", spec_id=family.spec_id),
        ],
        terminal=["by_rule_op", "by_plan_op"],
        discovered={"by_rule": family, "by_plan": family},
        answer_mode=AnswerMode.SERIES,
    )

    surface, outcome = _run(
        plan,
        summed={"by_rule_op": 3, "by_plan_op": 3},
        role_keys={},
        retrieval_role_keys=KENYA_ROLE_KEYS,
    )

    # Steps run in canonical order, `by_plan_op` first.
    assert _stat_var_totals(surface, outcome, plan=plan) == [
        f'The figures for "{CRS}" sum the family\'s 3 indicators.',
        _by_default("3"),
    ]


async def test_a_retrieval_that_summed_nothing_states_nothing(
    schemas: dict[str, ToolSchema],
) -> None:
    plan = _germany_bilateral_oda(schemas)
    surface, outcome = _run(plan, summed={}, role_keys={"yearly_total": ()})

    assert _stat_var_totals(surface, outcome, plan=plan) == []
    assert _stat_var_totals(None, outcome, plan=plan) == []


_G7_MEMBERS = (("country/FRA", "France"), ("country/JPN", "Japan"))


@pytest.mark.parametrize(
    ("donors", "total", "answer_role_keys", "notes", "reduction"),
    [
        (
            (("GroupOfSeven", "G7 countries, Total"),),
            "GroupOfSeven",
            (),
            [
                "The figures for G7 are the total the source publishes for the group as a whole "
                "(G7 countries, Total), not its members' figures added up."
            ],
            Reduction.SUM,
        ),
        (
            _G7_MEMBERS,
            None,
            (),
            [
                "The figure for G7 adds up its members' figures and leaves out amounts the data "
                "allocates to no single country."
            ],
            Reduction.SUM,
        ),
        (
            _G7_MEMBERS,
            None,
            (),
            [
                "The figure for G7 is computed from its members' figures and leaves out amounts "
                "the data allocates to no single country."
            ],
            Reduction.MEAN,
        ),
        (
            _G7_MEMBERS,
            None,
            ("financingProvider",),
            ["The figures for G7 are its members' figures, each on its own."],
            Reduction.SUM,
        ),
        (
            _G7_MEMBERS,
            None,
            None,
            [
                "The figures for G7 are its members' figures, not a total the source publishes "
                "for the group."
            ],
            Reduction.SUM,
        ),
    ],
    ids=["total", "members-summed", "members-averaged", "members-each", "members-unknown-answer"],
)
async def test_a_retrieval_of_a_named_group_states_how_it_reads_the_group(
    schemas: dict[str, ToolSchema],
    donors: tuple[tuple[str, str], ...],
    total: str | None,
    answer_role_keys: tuple[str, ...] | None,
    notes: list[str],
    reduction: Reduction,
) -> None:
    source = _single("c", "Oda").model_copy(
        update={
            "entities": (
                *(ResolvedEntity(role="origin", ref=ref, label=label) for ref, label in donors),
                ResolvedEntity(role="destination", ref="country/KEN", label="Kenya"),
            ),
            "groups": (ResolvedGroup(role="origin", label="G7", total=total),),
        }
    )
    oda = _dataset(
        "oda",
        [Entity(name="G7", role="donor"), Entity(name="Kenya", role="recipient")],
        Period(start=2015, end=2024),
    )
    retrieve = RetrieveOperation(id="oda_op", requirement_id="oda", spec_id=source.spec_id)
    yearly, yearly_op = _yearly_total("oda", "oda_op", reduction=reduction)
    plan = _compiled(
        schemas,
        [oda, yearly],
        [retrieve, yearly_op],
        terminal=["yearly_total"],
        discovered={"oda": source},
    )
    surface, outcome = _run(plan, summed={}, role_keys={"yearly_total": answer_role_keys})

    assert answer_notes(surface, outcome, plan=plan) == notes


@pytest.mark.parametrize(
    ("role", "role_keys", "column"),
    [
        ("origin", CRS_ROLE_KEYS, "financingProvider"),
        ("origin", ("financingRecipient",), None),
        ("place", ("observationAbout", "observationAbout_type"), "observationAbout"),
        ("place", ("observationAbout", "sector"), None),
        ("place", None, None),
    ],
    ids=["flow-side", "flow-side-absent", "lone-place", "two-columns", "unknown"],
)
def test_a_group_sits_on_its_flow_side_or_the_frames_one_role_column(
    role: str, role_keys: tuple[str, ...] | None, column: str | None
) -> None:
    group = ResolvedGroup(role=role, label="Africa")

    assert _group_column(group, role_keys) == column


_EAST_AFRICA = ("Kenya", "Uganda", "Tanzania")
_AMONG = _WHAT.model_copy(
    update={
        "slot": "among",
        "description": "Who the ranking compares.",
        "options": ("every recipient the source reports", "the three named"),
        "chosen": "every recipient the source reports",
        "rationale": "The question asks who received the most.",
        "fixes": DecisionFixes(among=RankAmong.POPULATION),
    }
)


def _east_africa(spec_char: str, stat_var: str, role: str, unit_ref: str | None) -> SelectedSource:
    """A source binding one StatVar for three countries under `role`."""
    return _single(spec_char, stat_var).model_copy(
        update={
            "entities": tuple(
                ResolvedEntity(role=role, ref=f"country/{name[:3].upper()}", label=name)
                for name in _EAST_AFRICA
            ),
            "unit_ref": unit_ref,
            "unit_label": "US dollars" if unit_ref else None,
        }
    )


def _aid_per_head(schemas: Mapping[str, ToolSchema], *, ranked: bool = False) -> Plan:
    """Aid to three recipients divided by each one's population, the
    quotient step `per_head` keyed by `recipient` against the population's
    `country`, and read by the rank `ranked_op` when `ranked`."""
    period = Period(start=2015, end=2024)
    aid = _dataset(
        "aid",
        [Entity(name=name, role="recipient") for name in _EAST_AFRICA],
        period,
        measure="aid received",
    )
    population = _dataset(
        "population",
        [Entity(name=name, role="country") for name in _EAST_AFRICA],
        period,
        measure="population",
    )
    keys: dict[str, Any] = {"axis": "recipient", "denominator_axis": "country"}
    requirements: list[EvidenceRequirement] = [
        aid,
        population,
        QuotientRequirement(
            id="per_head",
            decision_ids=["measure"],
            numerator=RequirementSide(input_id="aid"),
            denominator=RequirementSide(input_id="population"),
            **keys,
        ),
    ]
    operations: list[Operation] = [
        RetrieveOperation(id="aid_op", requirement_id="aid", spec_id="spec_" + "a" * 32),
        RetrieveOperation(
            id="population_op", requirement_id="population", spec_id="spec_" + "b" * 32
        ),
        QuotientOfOperation(
            id="per_head",
            requirement_id="per_head",
            numerator=Side(input="aid_op"),
            denominator=Side(input="population_op"),
            **keys,
        ),
    ]
    if ranked:
        order: dict[str, Any] = {
            "direction": RankDirection.HIGHEST_FIRST,
            "among": RankAmong.POPULATION,
            "reduction": Reduction.MEAN,
        }
        requirements.append(
            RankRequirement(
                id="ranked", decision_ids=["measure", "among"], input_id="per_head", **order
            )
        )
        operations.append(
            RankOperation(id="ranked_op", requirement_id="ranked", input="per_head", **order)
        )
    return _compiled(
        schemas,
        requirements,
        operations,
        terminal=["ranked_op" if ranked else "per_head"],
        decisions=(_WHAT, _AMONG) if ranked else (_WHAT,),
        discovered={
            "aid": _east_africa("a", "Aid", "recipient", "usd"),
            "population": _east_africa("b", "Population", "country", None),
        },
    )


def _with_dropped_keys(outcome: Accepted, **dropped: int) -> Accepted:
    """`outcome` with each named step's record declaring that many dropped keys."""
    return outcome.model_copy(
        update={
            "steps": tuple(
                record.model_copy(update={"dropped_key_count": dropped.get(record.step_id, 0)})
                for record in outcome.steps
            )
        }
    )


@pytest.mark.parametrize(
    ("step_id", "count", "stated"),
    [
        pytest.param("per_head", 0, (), id="none"),
        pytest.param("per_head", 1, ("1 recipient entry", "population", "was"), id="one"),
        pytest.param(
            "per_head", 1204, ("1,204 recipient entries", "population", "were"), id="several"
        ),
        # A step the plan gives no quotient's words for counts plain entries.
        pytest.param("aid_op", 2, ("2 entries", "were"), id="no-nouns"),
    ],
)
async def test_a_step_that_left_keys_out_states_how_many_in_the_plans_words(
    schemas: dict[str, ToolSchema], step_id: str, count: int, stated: tuple[str, ...]
) -> None:
    plan = _aid_per_head(schemas)
    _surface, outcome = _run(plan, summed={}, role_keys={})

    sentences = _dropped_key_sentences(_with_dropped_keys(outcome, **{step_id: count}), plan=plan)

    assert len(sentences) == bool(count)
    for sentence in sentences:
        assert all(words in sentence for words in stated), sentence


async def test_a_rank_over_a_quotient_that_left_keys_out_still_says_so(
    schemas: dict[str, ToolSchema],
) -> None:
    """The rank's frame is the answer and the quotient under it lost two of
    its keys: the sentence travels with the family totals and is confined to
    the terminals it sits under."""
    plan = _aid_per_head(schemas, ranked=True)
    surface, outcome = _run(plan, summed={"aid_op": 3}, role_keys={})
    outcome = _with_dropped_keys(outcome, per_head=2)

    notes = answer_notes(surface, outcome, plan=plan)

    assert len(notes) == 2
    assert notes[:-1] == _stat_var_totals(surface, outcome, plan=plan)
    assert "2 recipient entries" in notes[-1]
    assert answer_notes(surface, outcome, plan=plan, terminals=["ranked_op"]) == notes
    assert answer_notes(surface, outcome, plan=plan, terminals=[]) == []


async def test_a_rank_states_the_period_note_of_the_retrieval_under_it(
    schemas: dict[str, ToolSchema],
) -> None:
    """The rank's frame is an analysis output with no manifest of its own.
    The note comes from the manifest of each retrieval that feeds it, once
    per distinct sentence, and is confined to the terminals it sits under.
    The captured manifest states no noun for its dimensions, so the note
    counts members."""
    plan = _aid_per_head(schemas, ranked=True)
    surface, outcome = _run(plan, summed={}, role_keys={})
    coverage = json.loads(
        (Path(__file__).parent / "fixtures" / "mint_manifest_recent_years.json").read_text()
    )["coverage"]
    assert answer_notes(surface, outcome, plan=plan) == []
    for manifest in surface.manifests.values():
        manifest["coverage"] = coverage

    notes = answer_notes(surface, outcome, plan=plan)

    assert len(surface.manifests) == 2
    assert notes == [period_note(coverage)]
    assert all(stated in notes[0] for stated in ("2024", "8 members", "2025", "2 members"))
    assert answer_notes(surface, outcome, plan=plan, terminals=["ranked_op"]) == notes
    assert answer_notes(surface, outcome, plan=plan, terminals=[]) == []


def _step(step_id: str, tool: str, *, frame_bindings: dict[str, str] | None = None) -> WorkflowStep:
    return WorkflowStep(
        id=step_id,
        operation_id=step_id,
        tool=ToolId(server="data-mcp" if tool == "get_data" else "analysis-mcp", name=tool),
        bindings={
            name: {"from_step": source, "output": "frame_ref"}
            for name, source in (frame_bindings or {}).items()
        },
        input_schema_hash="sha256:" + "0" * 64,
    )


def test_a_join_sum_or_compare_carries_both_sides() -> None:
    """A compare writes a StatVar of its own, so a frame over it carries that
    one StatVar in place of the retrievals under its sides."""
    steps = {
        step.id: step
        for step in (
            _step("left", "get_data"),
            _step("right", "get_data"),
            _step(
                "joined", "join", frame_bindings={"frame_ref": "left", "right_frame_ref": "right"}
            ),
            _step(
                "compared",
                "compare",
                frame_bindings={"frame_ref": "left", "right_frame_ref": "right"},
            ),
            _step("yearly", "aggregate", frame_bindings={"frame_ref": "compared"}),
            _step(
                "summed", "sum_of", frame_bindings={"frame_ref": "left", "right_frame_ref": "right"}
            ),
        )
    }

    def carried(step_id: str) -> frozenset[str]:
        return _carried_retrievals(
            step_id, steps_by_id=steps, retrieval_step_ids=frozenset({"left", "right"})
        )

    assert carried("joined") == {"left", "right"}
    assert carried("summed") == {"left", "right"}
    assert carried("compared") == {"left", "right"}
    assert carried("yearly") == {"left", "right"}
    assert carried("right") == {"right"}
    assert _carried_parts(
        "yearly", steps_by_id=steps, retrieval_step_ids=frozenset({"left", "right"})
    ) == (frozenset(), frozenset({"compared"}))


def test_a_quotient_and_a_share_carry_both_sides() -> None:
    """A quotient and a share name their denominator's frame under
    `denominator_frame_ref` and carry its StatVars."""
    two_frames = {"frame_ref": "left", "denominator_frame_ref": "right"}
    steps = {
        step.id: step
        for step in (
            _step("left", "get_data"),
            _step("right", "get_data"),
            _step("divided", "quotient_of", frame_bindings=two_frames),
            _step("one_frame", "quotient_of", frame_bindings={"frame_ref": "left"}),
            _step("yearly", "aggregate", frame_bindings={"frame_ref": "divided"}),
            _step("shared", "share_of", frame_bindings=two_frames),
        )
    }

    def carried(step_id: str) -> frozenset[str]:
        return _carried_retrievals(
            step_id, steps_by_id=steps, retrieval_step_ids=frozenset({"left", "right"})
        )

    assert carried("divided") == {"left", "right"}
    assert carried("yearly") == {"left", "right"}
    assert carried("one_frame") == {"left"}
    assert carried("shared") == {"left", "right"}


_ODA_FAMILY = tuple(f"ODA_{index}" for index in range(12))


@pytest.mark.parametrize(
    ("left", "right", "terminal", "counted"),
    [
        # A twelve-member family summed by its retrieval, plus a singleton.
        ((_ODA_FAMILY, 12), (("Imputed_ODA",), None), "summed", 13),
        # A verb over the sum, and a compare whose left side is the sum.
        ((_ODA_FAMILY, 12), (("Imputed_ODA",), None), "yearly", 13),
        ((_ODA_FAMILY, 12), (("Imputed_ODA",), None), "compared", 13),
        # A sum of two slices of the family's one frame.
        ((_ODA_FAMILY, 12), (("Imputed_ODA",), None), "one_frame", 12),
        # One StatVar retrieved for two places and summed.
        ((("ODA",), None), (("ODA",), None), "summed", 1),
        ((("ODA",), None), (("ODA",), None), "yearly", 1),
        # A quotient of the family over a singleton, a verb over it, and a
        # quotient of two slices of the family's one frame.
        ((_ODA_FAMILY, 12), (("Population",), None), "divided", 13),
        ((_ODA_FAMILY, 12), (("Population",), None), "divided_yearly", 13),
        ((_ODA_FAMILY, 12), (("Population",), None), "one_frame_divided", 12),
        # A share of the family in a singleton.
        ((_ODA_FAMILY, 12), (("GDP",), None), "shared", 13),
    ],
)
def test_a_card_over_a_sum_counts_the_distinct_stat_vars_under_both_sides(
    left: tuple[tuple[str, ...], int | None],
    right: tuple[tuple[str, ...], int | None],
    terminal: str,
    counted: int,
) -> None:
    """A sum or a quotient declares one StatVar of its own; the card counts
    the distinct StatVars under its two sides, read from each retrieval's
    trace."""
    workflow_steps = (
        _step("left", "get_data"),
        _step("right", "get_data"),
        _step("summed", "sum_of", frame_bindings={"frame_ref": "left", "right_frame_ref": "right"}),
        _step("yearly", "aggregate", frame_bindings={"frame_ref": "summed"}),
        _step(
            "compared",
            "compare",
            frame_bindings={"frame_ref": "summed", "right_frame_ref": "right"},
        ),
        _step("one_frame", "sum_of", frame_bindings={"frame_ref": "left"}),
        _step(
            "divided",
            "quotient_of",
            frame_bindings={"frame_ref": "left", "denominator_frame_ref": "right"},
        ),
        _step("divided_yearly", "aggregate", frame_bindings={"frame_ref": "divided"}),
        _step("one_frame_divided", "quotient_of", frame_bindings={"frame_ref": "left"}),
        _step(
            "shared",
            "share_of",
            frame_bindings={"frame_ref": "left", "denominator_frame_ref": "right"},
        ),
    )
    sources = {
        step_id: SelectedSource(
            spec_id="spec_" + spec_char * 32,
            sentence="Aid to Kenya",
            has_data=True,
            served_count=len(stat_vars),
        )
        for step_id, spec_char, (stat_vars, _) in (("left", "a", left), ("right", "b", right))
    }
    frames = {
        **{step_id: _retrieved(source.spec_id, step_id) for step_id, source in sources.items()},
        **{step.id: _frame(step.id) for step in workflow_steps[2:]},
    }
    store = ProvenanceStore()
    for step_id, (stat_vars, summed) in (("left", left), ("right", right)):
        store.put(
            conversation=current_conversation_id.get(),
            spec_id=frames[step_id].spec_id,
            content_version=frames[step_id].content_version,
            manifest={
                "trace": {
                    "resolved_stat_vars": [
                        {"ref": ref, "observation_count": 5} for ref in stat_vars
                    ]
                },
                "summed_stat_var_count": summed,
            },
        )
    outcome = Accepted(
        revision="r",
        evidence=tuple(
            EvidenceRecord(
                requirement_id=step_id,
                step_id=step_id,
                frame_ref=frames[step_id],
                spec_id=frames[step_id].spec_id,
                stat_vars=(),
                window=None,
                gaps=(),
                citation="c",
            )
            for step_id in sources
        ),
        steps=tuple(
            StepRecord(
                step_id=step.id,
                tool=step.tool,
                attempts=1,
                frame_ref=frames[step.id],
                digest="d",
                stat_var_count=1,
            )
            for step in workflow_steps
        ),
        results={terminal: frames[terminal]},
    )
    workflow = Workflow(
        steps=list(workflow_steps),
        sources=list(sources.values()),
        evidence_bindings=[
            EvidenceBinding(requirement_id=step_id, step_id=step_id, spec_id=source.spec_id)
            for step_id, source in sources.items()
        ],
        terminal_steps=[terminal],
    )

    link_evidence_to_terminal_frames(
        _RecordingSurface(cast("ToolSurface", None), provenance=store), outcome, workflow=workflow
    )

    assert (
        store.stat_var_count(
            conversation=current_conversation_id.get(),
            spec_id=frames[terminal].spec_id,
            content_version=frames[terminal].content_version,
        )
        == counted
    )


def test_a_retrieval_s_source_is_the_one_at_its_binding_s_position() -> None:
    """Two retrievals of one spec, read differently: each step's source is
    its own, and a workflow listing one source for both names it by spec."""
    family = _family("a", "Grants")
    steps = [_step("one", "get_data"), _step("two", "get_data")]
    bindings = [
        EvidenceBinding(requirement_id="one", step_id="one", spec_id=family.spec_id),
        EvidenceBinding(requirement_id="two", step_id="two", spec_id=family.spec_id),
    ]
    summed = family.model_copy(update={"stat_var_reading": StatVarReading.SUM})
    separate = family.model_copy(update={"stat_var_reading": StatVarReading.SEPARATE})

    paired = Workflow(
        steps=steps, sources=[summed, separate], evidence_bindings=bindings, terminal_steps=["one"]
    )
    shared = Workflow(
        steps=steps, sources=[summed], evidence_bindings=bindings, terminal_steps=["one"]
    )

    assert paired.source_of("one") is summed
    assert paired.source_of("two") is separate
    assert shared.source_of("two") is summed
    assert paired.source_of("absent") is None

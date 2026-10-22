"""Compile a plan's translation into the workflow that runs it.

`compile_plan` is code, not judgement. It proves that every requirement the
plan states is carried out, exactly once, by an operation that repeats its
calculation fields (`planner.proof.mismatches`); checks each retrieval names
the source discovery selected for its own requirement; maps each operation to
a tool through the capability table; writes every tool argument itself -- a
column name from the frame grammar, an entity binding from the source's own
roles and values, a frame reference as a `Binding` -- and validates the
result against the live tool surface. It reports every issue it finds in one
pass, so a failed attempt comes back with all of them, and it returns a
`Workflow` only when it found none.

A retrieval is a legal terminal when its frame answers a question that asks
to see a level or a trend for named entities, carrying one series per entity
for a chart to draw: under `planner.plan.AnswerMode.SERIES` whether or not a
calculation reads it, and under `AnswerMode.COMPUTED` only when one does and
an ask names it in `Plan.answered`. Any terminal may also be read by another
operation. Those gates are the plan's own. `Plan._references_resolve`
rejects an unread dataset requirement under `AnswerMode.COMPUTED` before a
translation naming it as terminal ever reaches here, so this module states no
`answer_mode` check of its own, and its operation graph mirrors the requirement
graph the plan already validated. Nothing here reads a claim.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum, StrEnum
from typing import Any, Literal, NamedTuple

from common.place_type import FLOW_ROLE_COLUMNS, PLACE_ROLE, type_column
from common.rolling import PopulationEmptyError, PopulationLevelColumnMissingError
from common.unit_refs import PERCENT_UNIT
from dfo_frame_grammar.grammar import TIME_PERIOD_COLUMN, VALUE_COLUMN, normalize_role_keys
from planner.operations import (
    AggregateOperation,
    CompareOperation,
    FilterOperation,
    GroupBy,
    GrowthOperation,
    JoinOperation,
    Operation,
    QuotientOfOperation,
    RankOperation,
    RetrieveOperation,
    ShareOfOperation,
    Side,
    SumOfOperation,
    Translation,
)
from planner.parties import across_party
from planner.plan import DatasetRequirement, Entity, Plan, Unit, keeps_one_level
from planner.proof import mismatches
from planner.records import (
    DenominatorScope,
    EntityIn,
    EvidenceIssueCode,
    GrowthChange,
    Period,
    PeriodThreshold,
    PlaceLevel,
    RankAmong,
    RankDirection,
    StatVarReading,
    StrictModel,
    ValueThreshold,
)
from planner.shapes import PLACE_PHRASE, ValueKind, place_role, role_named, roles_named
from planner.workflow import (
    ArtifactKind,
    Binding,
    EvidenceBinding,
    OpenEntityAxis,
    ResolvedEntity,
    SelectedSource,
    ToolId,
    Workflow,
    WorkflowStep,
)
from pydantic import JsonValue, ValidationError, model_validator

from compiler.arguments import (
    literal_argument_errors,
    op_spec_errors,
    undeclared_parameters,
    unusable_schema_reason,
)
from compiler.capabilities import (
    CAPABILITIES,
    SchemaDrift,
    ToolCapability,
    capabilities_for,
    schema_drift,
)
from compiler.evidence import (
    EvidenceIssue,
    UnitReading,
    check_evidence,
    errors,
    evidence_from_source,
    named_members,
    named_refs,
    open_axes_for,
    opens_place_axis,
    read_source_unit,
    read_unit_dcid,
)
from compiler.frame_columns import ROLLING_PARAMETER
from compiler.rolling import (
    Root,
    anchor_groups,
    rolling_ref,
    step_populations,
    window_overlay,
    with_population,
)
from compiler.shapes import (
    BREAKDOWN_CODE_SYSTEMS,
    ResolvedRole,
    ResolvedShape,
    operation_inputs,
    rolls,
    shapes_for,
    unsummable_axes,
)
from compiler.sources import (
    FamilyOverlap,
    entity_scopes_disjoint,
    family_overlap,
    pinned_stat_var,
    stat_var_reading,
)
from compiler.surface import ToolSchema, schema_hash_error


class CompileIssueCode(StrEnum):
    """What stopped a translation from compiling."""

    DUPLICATE_ID = "duplicate_id"
    REQUIREMENT_UNKNOWN = "requirement_unknown"
    REQUIREMENT_UNSATISFIED = "requirement_unsatisfied"
    DUPLICATE_SATISFACTION = "duplicate_satisfaction"
    KIND_MISMATCH = "kind_mismatch"
    OPERATION_MISMATCH = "operation_mismatch"
    DANGLING_INPUT = "dangling_input"
    INPUT_IS_TERMINAL = "input_is_terminal"
    INPUT_SHAPE_UNSUPPORTED = "input_shape_unsupported"
    LEVEL_FILTER_ON_FLOW_AXIS = "level_filter_on_flow_axis"
    RANK_AXIS_REQUIRED = "rank_axis_required"
    SHARE_AXIS_REQUIRED = "share_axis_required"
    SHARE_KEY_UNKNOWN = "share_key_unknown"
    QUOTIENT_AXIS_REQUIRED = "quotient_axis_required"
    QUOTIENT_KEY_UNKNOWN = "quotient_key_unknown"
    KEYS_DISJOINT = "keys_disjoint"
    CYCLE = "cycle"
    UNKNOWN_TERMINAL = "unknown_terminal"
    OPERATION_UNREACHABLE = "operation_unreachable"
    SOURCE_UNSELECTED = "source_unselected"
    SOURCE_NOT_DISCOVERED_FOR_REQUIREMENT = "source_not_discovered_for_requirement"
    SOURCE_PIN_VIOLATED = "source_pin_violated"
    SOURCE_UNUSABLE = "source_unusable"
    SOURCE_WIDER_THAN_REQUIREMENT = "source_wider_than_requirement"
    REQUIREMENT_UNMEETABLE = "requirement_unmeetable"
    WINDOW_OUTSIDE_PERIOD = "window_outside_period"
    ROLLING_NEEDS_OPEN_PERIOD = "rolling_needs_open_period"
    ROLLING_POPULATION_UNUSABLE = "rolling_population_unusable"
    ENTITY_REQUIRED = "entity_required"
    UNKNOWN_ENTITY = "unknown_entity"
    ENTITY_AMBIGUOUS = "entity_ambiguous"
    ENTITY_UNRESOLVED = "entity_unresolved"
    ENTITY_NOT_ON_FRAME = "entity_not_on_frame"
    DEGENERATE_SIDES = "degenerate_sides"
    SUM_SIDES_OVERLAP = "sum_sides_overlap"
    SUM_OVERLAP_UNDECIDED = "sum_overlap_undecided"
    FAMILY_UNSUMMABLE = "family_unsummable"
    SUM_UNITS_DIFFER = "sum_units_differ"
    SHARE_UNITS_DIFFER = "share_units_differ"
    FRAME_UNITS_DIFFER = "frame_units_differ"
    FRAME_UNIT_UNBOUND = "frame_unit_unbound"
    FRAME_UNIT_FORMS_DIFFER = "frame_unit_forms_differ"
    FRAME_PRICE_BASES_DIFFER = "frame_price_bases_differ"
    FRAME_UNITS_IRRECONCILABLE = "frame_units_irreconcilable"
    PERIODS_DISJOINT = "periods_disjoint"
    JOIN_NO_COMMON_KEY_COLUMNS = "join_no_common_key_columns"
    NO_CAPABILITY = "no_capability"
    AMBIGUOUS_CAPABILITY = "ambiguous_capability"
    UNKNOWN_TOOL = "unknown_tool"
    BAD_TOOL_SCHEMA = "bad_tool_schema"
    CAPABILITY_SCHEMA_DRIFT = "capability_schema_drift"
    CAPABILITY_UNPINNED = "capability_unpinned"
    ARTIFACT_MISMATCH = "artifact_mismatch"
    BAD_ARGS = "bad_args"


class CompileIssue(StrictModel):
    """One reason a translation does not compile, and what it is about.

    `finding` is the evidence finding a `SOURCE_UNUSABLE` issue reports, when
    `check_evidence` found it.
    """

    code: CompileIssueCode
    reason: str
    operation_id: str | None = None
    requirement_id: str | None = None
    tool: ToolId | None = None
    finding: EvidenceIssueCode | None = None


class CompileResult(StrictModel):
    """A workflow, or every issue that stopped one being built.

    `warnings` are evidence findings that do not block compiling -- a
    requirement the source could not confirm -- for the approval summary to
    carry.
    """

    workflow: Workflow | None = None
    issues: tuple[CompileIssue, ...] = ()
    warnings: tuple[EvidenceIssue, ...] = ()

    @property
    def ok(self) -> bool:
        """Whether a workflow was built."""
        return self.workflow is not None

    @model_validator(mode="after")
    def _a_workflow_or_issues(self) -> CompileResult:
        if (self.workflow is None) == (not self.issues):
            raise ValueError("a compile result carries a workflow or issues, and exactly one")
        return self


def compile_plan(
    *,
    plan: Plan,
    translation: Translation,
    schemas: Mapping[str, ToolSchema],
    discovered: Mapping[str, SelectedSource],
    capabilities: Sequence[ToolCapability] = CAPABILITIES,
) -> CompileResult:
    """Compile `translation` against `plan`, the live `schemas`, and the discovered sources.

    Args:
        plan: The plan whose requirements the translation resolves.
        translation: The translator's proposal.
        schemas: The live tool surface, keyed by tool name.
        discovered: The source discovery recorded for this plan, keyed by
            *requirement id* -- the one source discovery selected for that
            requirement's retrieval.
        capabilities: The mapping table; the module default in production.

    Returns:
        A `CompileResult` carrying the workflow, or every issue found.
    """
    build = _Build(plan, translation, schemas, discovered, capabilities)
    build.run()
    # Sorted, like the steps, so that two translations naming the same
    # operations compile to the same workflow and so the same revision.
    warnings = tuple(sorted(build.warnings, key=_warning_key))
    if build.issues:
        return CompileResult(issues=tuple(build.issues), warnings=warnings)
    unbuilt = sorted(set(build.order) - set(build.steps))
    if unbuilt:
        # Every path that skips a step reports an issue first; reaching here
        # means one did not, which is a fault in this module.
        raise RuntimeError(f"no step was built for {unbuilt!r} and no issue explains why")
    workflow = build.workflow(warnings)
    # One population is written on every rolling step of a group. A group whose
    # steps disagree is a fault in this module, so it is raised here and not reported.
    anchor_groups(workflow)
    return CompileResult(workflow=workflow, warnings=warnings)


# What `_Build` writes for one step: its literal arguments and its bindings.
Arguments = tuple[dict[str, JsonValue], dict[str, Binding]]

# What a single-input chain rests on: the source discovery selected for its
# retrieval, and the dataset requirement that retrieval carries out.
Upstream = tuple[SelectedSource, DatasetRequirement]

# Operations that read exactly one other operation, and so can be walked
# through to find the dataset requirement (or source) a chain rests on.
_SINGLE_INPUT = (AggregateOperation, GrowthOperation, RankOperation, FilterOperation)


class _Ratio(Enum):
    """The ratio a frame holds (`_Build._ratio_held`), a share in `PERCENT_UNIT` or one with no unit."""

    SHARE = "share"
    UNITLESS = "unitless"


class _PerRowUnit(Enum):
    """A frame whose unit `_frame_unit` cannot know ahead: one per row or per series."""

    PER_ROW = "per_row"


@dataclass(frozen=True, slots=True)
class _StatVarPart:
    """The part of a composed unit a side declaring no unit gives: its StatVar."""

    stat_var: str


@dataclass(frozen=True, slots=True)
class _ComposedUnit:
    """The unit of a `quotient_of` result: each side's unit, or its StatVar when it declares none.

    A part is never `None`.
    """

    numerator: JsonValue | _StatVarPart
    denominator: JsonValue | _StatVarPart


# The unit cells of a frame, as `_frame_unit` reads them ahead of the fetch:
# one value, `None` when its source declares no unit, or a unit per row.
type FrameUnit = JsonValue | Literal[_PerRowUnit.PER_ROW]


class _Build:
    """One compilation's working state: what has been checked and what was built."""

    def __init__(
        self,
        plan: Plan,
        translation: Translation,
        schemas: Mapping[str, ToolSchema],
        discovered: Mapping[str, SelectedSource],
        capabilities: Sequence[ToolCapability],
    ) -> None:
        self.translation = translation
        self.schemas = schemas
        self.discovered = discovered
        self.capabilities = capabilities
        self.requirements = {r.id: r for r in plan.evidence_requirements}
        self.issues: list[CompileIssue] = []
        self.warnings: list[EvidenceIssue] = []
        self.operations: dict[str, Operation] = {}
        self.order: list[str] = []
        self.satisfied_by: dict[str, str] = {}
        self.sources: dict[str, SelectedSource] = {}
        self.shapes: dict[str, ResolvedShape] = {}
        self._retrieve_by_requirement: dict[str, str] = {}
        self.tools: dict[str, tuple[ToolCapability, ToolSchema]] = {}
        self.steps: dict[str, WorkflowStep] = {}

    def run(self) -> None:
        self._index_operations()
        self._check_requirements_satisfied()
        self._check_graph()
        self._check_sources()
        self._check_requirements_meetable()
        self._build_shapes()
        self._check_level_filters()
        self._check_operations()
        self._check_source_width()
        self._check_capabilities()
        self._build_steps()

    def workflow(self, warnings: tuple[EvidenceIssue, ...]) -> Workflow:
        """Build the workflow, every sequence of it in the canonical order `self.order` fixes."""
        retrieves = [
            op
            for op_id in self.order
            if isinstance(op := self.operations[op_id], RetrieveOperation)
        ]
        terminal = set(self.translation.terminal)
        return Workflow(
            steps=tuple(self.steps[op_id] for op_id in self.order),
            sources=tuple(self.sources[op.id] for op in retrieves),
            evidence_bindings=tuple(
                EvidenceBinding(
                    requirement_id=op.requirement_id,
                    step_id=op.id,
                    spec_id=self.sources[op.id].spec_id,
                )
                for op in retrieves
            ),
            terminal_steps=tuple(op_id for op_id in self.order if op_id in terminal),
            warnings=warnings,
        )

    # -- checks ------------------------------------------------------------------

    def _index_operations(self) -> None:
        """Index the operations by id, in canonical order rather than the translator's.

        Everything downstream that walks `self.operations` -- the topological
        sort above all -- inherits this order, so two translations naming the
        same operations in a different order compile to the same `Workflow`,
        and so to the same approved revision.
        """
        for operation in self.translation.operations:
            if operation.id in self.operations:
                self._issue(
                    CompileIssueCode.DUPLICATE_ID,
                    f"operation id {operation.id!r} is used more than once",
                    operation_id=operation.id,
                )
                continue
            self.operations[operation.id] = operation
        self.operations = dict(
            sorted(self.operations.items(), key=lambda item: _canonical_key(item[1]))
        )

    def _check_requirements_satisfied(self) -> None:
        """Reject a translation where a requirement is not carried out exactly once, rightly.

        Buckets every operation (not only retrievals) by the requirement it
        names: one named by no operation this plan's requirements admit is
        `requirement_unknown`; a requirement no operation names is
        `requirement_unsatisfied`; one named by several is
        `duplicate_satisfaction`. `self.satisfied_by` then maps each
        requirement carried out exactly once to that operation, and
        `planner.proof.mismatches` checks the two agree: a kind that does not
        correspond is `kind_mismatch`, any other disagreement is
        `operation_mismatch`, naming the differing fields.
        """
        satisfied: dict[str, list[str]] = {rid: [] for rid in self.requirements}
        for operation in self.operations.values():
            if operation.requirement_id not in self.requirements:
                self._issue(
                    CompileIssueCode.REQUIREMENT_UNKNOWN,
                    f"operation {operation.id!r} carries out {operation.requirement_id!r}, "
                    "which is not a requirement of this plan",
                    operation_id=operation.id,
                    requirement_id=operation.requirement_id,
                )
                continue
            satisfied[operation.requirement_id].append(operation.id)

        self.satisfied_by = {
            rid: op_ids[0] for rid, op_ids in satisfied.items() if len(op_ids) == 1
        }
        for rid, op_ids in satisfied.items():
            if not op_ids:
                self._issue(
                    CompileIssueCode.REQUIREMENT_UNSATISFIED,
                    f"no operation carries out requirement {rid!r}",
                    requirement_id=rid,
                )
            elif len(op_ids) > 1:
                self._issue(
                    CompileIssueCode.DUPLICATE_SATISFACTION,
                    f"requirement {rid!r} is carried out by {op_ids!r}; exactly one operation "
                    "carries out each requirement",
                    requirement_id=rid,
                )

        for rid, op_id in self.satisfied_by.items():
            requirement = self.requirements[rid]
            operation = self.operations[op_id]
            problems = mismatches(requirement, operation, self.satisfied_by)
            if problems == ["kind"]:
                self._issue(
                    CompileIssueCode.KIND_MISMATCH,
                    f"operation {op_id!r} is a {operation.kind.value!r}, which does not carry "
                    f"out requirement {rid!r}",
                    operation_id=op_id,
                    requirement_id=rid,
                )
            elif problems:
                self._issue(
                    CompileIssueCode.OPERATION_MISMATCH,
                    f"operation {op_id!r} disagrees with requirement {rid!r} on {problems!r}",
                    operation_id=op_id,
                    requirement_id=rid,
                )

    def _check_graph(self) -> None:
        for operation in self.operations.values():
            for input_id in operation_inputs(operation):
                if input_id not in self.operations:
                    self._issue(
                        CompileIssueCode.DANGLING_INPUT,
                        f"operation {operation.id!r} reads {input_id!r}, which is not an operation",
                        operation_id=operation.id,
                    )
                elif isinstance(self.operations[input_id], CompareOperation | JoinOperation):
                    # A comparison or a join is terminal by contract: each
                    # rests on two requirements, so no single requirement or
                    # source stands behind it for a reader to be checked
                    # against, and a join's own downstream need -- a per-key
                    # ratio -- is keyed share_of, not a further read of the join.
                    kind = self.operations[input_id].kind.value
                    self._issue(
                        CompileIssueCode.INPUT_IS_TERMINAL,
                        f"operation {operation.id!r} reads the {kind} {input_id!r}; a {kind} "
                        "answers the question and feeds no further operation",
                        operation_id=operation.id,
                    )
        self.order = self._topological_order()
        known = set(self.operations)
        # A terminal another operation reads is legal, since an ask the plan
        # answers with that requirement's result (`Plan.answered`) gets it.
        for terminal in self.translation.terminal:
            if terminal not in known:
                self._issue(
                    CompileIssueCode.UNKNOWN_TERMINAL,
                    f"terminal {terminal!r} is not an operation",
                    operation_id=terminal,
                )
        reachable = self._reachable_from_terminals()
        for op_id in self.operations:
            if op_id not in reachable:
                self._issue(
                    CompileIssueCode.OPERATION_UNREACHABLE,
                    f"operation {op_id!r} feeds none of the terminal operations",
                    operation_id=op_id,
                )

    def _check_sources(self) -> None:
        for operation in self._retrieves():
            requirement = self.requirements.get(operation.requirement_id)
            if operation.spec_id is None:
                self._issue(
                    CompileIssueCode.SOURCE_UNSELECTED,
                    f"operation {operation.id!r} names no spec_id; discover a source first",
                    operation_id=operation.id,
                    requirement_id=operation.requirement_id,
                )
                continue
            source = self.discovered.get(operation.requirement_id)
            if source is None or source.spec_id != operation.spec_id:
                self._issue(
                    CompileIssueCode.SOURCE_NOT_DISCOVERED_FOR_REQUIREMENT,
                    f"operation {operation.id!r} names spec_id {operation.spec_id!r} for "
                    f"requirement {operation.requirement_id!r}, which discovery did not record "
                    "for it",
                    operation_id=operation.id,
                    requirement_id=operation.requirement_id,
                )
                continue
            if (
                isinstance(requirement, DatasetRequirement)
                and requirement.source_spec_id is not None
                and requirement.source_spec_id != operation.spec_id
            ):
                self._issue(
                    CompileIssueCode.SOURCE_PIN_VIOLATED,
                    f"requirement {requirement.id!r} pins source {requirement.source_spec_id!r}, "
                    f"but operation {operation.id!r} names {operation.spec_id!r}",
                    operation_id=operation.id,
                    requirement_id=requirement.id,
                )
                continue
            if not isinstance(requirement, DatasetRequirement):
                # A retrieval of a derived requirement is already `kind_mismatch`.
                self.sources[operation.id] = source
                continue
            # The reading is settled here, before shapes count measures from it
            # and before the retrieval's arguments are written, so the
            # approved workflow's source states it and who set it. A breakdown
            # is read separately: `get_data` refuses it beside a summed family.
            requested = (
                StatVarReading.SEPARATE
                if requirement.breakdown is not None
                else requirement.stat_vars
            )
            source = stat_var_reading(source, requested)
            self.sources[operation.id] = source
            self._check_summable(operation, requirement, source)
            self._check_breakdown(operation, requirement, source)
            findings = check_evidence(requirement, evidence_from_source(source))
            for finding in errors(findings):
                self._issue(
                    CompileIssueCode.SOURCE_UNUSABLE,
                    f"source {source.spec_id} cannot serve requirement {requirement.id!r}: "
                    f"{finding.detail}",
                    operation_id=operation.id,
                    requirement_id=requirement.id,
                    finding=finding.code,
                )
            self.warnings.extend(f for f in findings if f not in errors(findings))

    def _check_summable(
        self, operation: RetrieveOperation, requirement: DatasetRequirement, source: SelectedSource
    ) -> None:
        """Report a family read summed over an axis dfo-apis may not sum over.

        `stat_var_reading` reads a family summed by the rule only when every
        open axis is summable, but the plan may ask for `sum` over any family;
        `get_data` would refuse that retrieval `rollup_not_summable`.
        """
        if source.stat_var_reading is not StatVarReading.SUM:
            return
        axes = unsummable_axes(source)
        if not axes:
            return
        one_value = "one value for it" if len(axes) == 1 else "one value for each"
        self._issue(
            CompileIssueCode.FAMILY_UNSUMMABLE,
            f"requirement {requirement.id!r} reads source {source.spec_id}'s family summed "
            f"(stat_vars='sum'), but it cannot be summed over {', '.join(axes)}; read it "
            f"separately (stat_vars='separate') or have the question name {one_value}",
            operation_id=operation.id,
            requirement_id=requirement.id,
        )

    def _check_breakdown(
        self, operation: RetrieveOperation, requirement: DatasetRequirement, source: SelectedSource
    ) -> None:
        """Report a breakdown by a code system the source's family does not vary along.

        `get_data` sums the family within each code of the breakdown's code
        system, so a source whose members state no value on it has no codes
        to break its total down by.
        """
        if requirement.breakdown is None:
            return
        code_system = BREAKDOWN_CODE_SYSTEMS[requirement.breakdown.dimension]
        if any(axis.property == code_system for axis in source.open_axes):
            return
        self._issue(
            CompileIssueCode.SOURCE_UNUSABLE,
            f"source {source.spec_id} cannot serve requirement {requirement.id!r}: its "
            f"StatVars do not vary by {requirement.breakdown.dimension.value} "
            f"({code_system}), so it has no breakdown by it",
            operation_id=operation.id,
            requirement_id=requirement.id,
        )

    def _build_shapes(self) -> None:
        """Resolve one `ResolvedShape` per operation, reporting `INPUT_SHAPE_UNSUPPORTED` on a refusal.

        Drives `compiler.shapes.shapes_for` over `self.operations` in
        `self.order`, with `self.sources` for the source behind each
        retrieval. Also indexes `self._retrieve_by_requirement`, which every
        `_upstream`-family lookup below uses to walk from a shape's own
        `roots` back to the retrieval that produced it.
        """
        self._retrieve_by_requirement = {op.requirement_id: op.id for op in self._retrieves()}
        self.shapes, reasons = shapes_for(
            self.operations,
            requirements=self.requirements,
            sources=self.sources,
            order=self.order,
        )
        for reason in reasons:
            operation_id, _, detail = reason.partition(": ")
            self._issue(CompileIssueCode.INPUT_SHAPE_UNSUPPORTED, detail, operation_id=operation_id)

    def _check_level_filters(self) -> None:
        """Report `LEVEL_FILTER_ON_FLOW_AXIS` for a place-level filter whose places sit on a flow axis.

        A requirement whose `across` names the place may keep one level with a
        `PlaceLevel` filter, which reads `observationAbout_type`. Only a
        source that leaves `observationAbout` open over several places mixes
        levels (`compiler.evidence.opens_place_axis`). Where the place word
        covers a flow axis instead (`compiler.evidence.open_axes_for`), as
        IDS holds its debtors on `destination`, the axis holds parties, and a
        flow retrieval serves them without their group totals
        (`_retrieve_arguments`), so the filter has no level to keep and the
        planner drops it.
        """
        for operation in self.operations.values():
            if not isinstance(operation, FilterOperation) or not keeps_one_level(
                operation.predicates
            ):
                continue
            retrieve = self.operations.get(self._through_filters(operation.input))
            source = self.sources.get(retrieve.id) if retrieve is not None else None
            requirement = (
                self.requirements.get(retrieve.requirement_id) if retrieve is not None else None
            )
            if (
                source is None
                or not isinstance(requirement, DatasetRequirement)
                or opens_place_axis(source.open_entity_axes)
            ):
                continue
            axes = open_axes_for(requirement.across, source.open_entity_axes)
            if isinstance(axes, str):
                continue
            for word, axis in zip(requirement.across, axes, strict=True):
                if across_party(word) != "place" or axis.column == PLACE_ROLE:
                    continue
                self._issue(
                    CompileIssueCode.LEVEL_FILTER_ON_FLOW_AXIS,
                    f"source {source.spec_id} holds every {word!r} on its flow axis "
                    f"{axis.column!r}, whose parties have no place level to keep, and the "
                    "retrieval already leaves the parties' group totals out; drop the "
                    "place_level predicate",
                    operation_id=operation.id,
                    requirement_id=requirement.id,
                )

    def _check_requirements_meetable(self) -> None:
        for requirement in self.requirements.values():
            if not isinstance(requirement, DatasetRequirement):
                continue
            if requirement.unit is not None and requirement.unit.prices == "constant":
                self._issue(
                    CompileIssueCode.REQUIREMENT_UNMEETABLE,
                    f"requirement {requirement.id!r} asks for constant {requirement.unit.base_year} "
                    f"prices, and no operation restates a series in another year's prices",
                    requirement_id=requirement.id,
                )

    def _check_operations(self) -> None:
        for operation in self.operations.values():
            if isinstance(operation, RetrieveOperation):
                continue
            if isinstance(operation, AggregateOperation):
                self._check_aggregate(operation)
            elif isinstance(operation, GrowthOperation):
                self._check_growth(operation)
            elif isinstance(operation, CompareOperation):
                self._check_two_sided(operation.id, operation.left, operation.right)
            elif isinstance(operation, SumOfOperation):
                self._check_two_sided(operation.id, operation.left, operation.right)
                self._check_sum_of(operation)
            elif isinstance(operation, RankOperation):
                self._check_rank(operation)
            elif isinstance(operation, ShareOfOperation):
                self._check_two_sided(
                    operation.id,
                    operation.numerator,
                    operation.denominator,
                    keyed=_keyed_axes(operation),
                )
                self._check_key_axes(operation)
            elif isinstance(operation, QuotientOfOperation):
                self._check_two_sided(
                    operation.id,
                    operation.numerator,
                    operation.denominator,
                    keyed=_keyed_axes(operation),
                    any_units=True,
                )
                self._check_key_axes(operation)
                self._check_quotient_parts(operation)
            elif isinstance(operation, JoinOperation):
                self._check_join(operation)
            else:
                self._check_filter(operation)
            self._check_rolling_reads_open_periods(operation)

    def _check_rolling_reads_open_periods(self, operation: Operation) -> None:
        """Report `ROLLING_NEEDS_OPEN_PERIOD` for a rolling operation over a dataset with a fixed end.

        A dataset retrieved over fixed years never holds a later one, so a
        period that follows the newest data would stop at the plan's own end
        year. `planner.plan` refuses such a plan. This check covers a translation
        whose operation rolls where its requirement does not, and an
        approved source whose identity's `requested_window` states an end
        year, as Search may answer "since 2015" with one. A window of kind
        `all`, or a range with no end year, reaches every later year.
        """
        if not rolls(operation):
            return
        for requirement_id in sorted(self._root_requirement_ids(operation.id)):
            requirement = self.requirements.get(requirement_id)
            if not isinstance(requirement, DatasetRequirement):
                continue
            source = self._source_of(requirement_id)
            window = ((source.identity if source is not None else None) or {}).get(
                "requested_window"
            )
            closed = isinstance(window, dict) and window.get("end_year") is not None
            if requirement.period_end_open and not closed:
                continue
            self._issue(
                CompileIssueCode.ROLLING_NEEDS_OPEN_PERIOD,
                f"operation {operation.id!r} states a period that follows the newest data, "
                f"over requirement {requirement.id!r}, retrieved with a fixed end year",
                operation_id=operation.id,
                requirement_id=requirement.id,
            )

    def _check_growth(self, operation: GrowthOperation) -> None:
        requirement = self._requirement_upstream(operation.input)
        if requirement is None:
            return
        window = Period(start=operation.window_start, end=operation.window_end)
        if not requirement.period.covers(window):
            self._issue(
                CompileIssueCode.WINDOW_OUTSIDE_PERIOD,
                f"operation {operation.id!r} takes growth over {window.start}-{window.end}, "
                f"outside requirement {requirement.id!r}'s {requirement.period.start}-"
                f"{requirement.period.end}",
                operation_id=operation.id,
                requirement_id=requirement.id,
            )

    def _check_aggregate(self, operation: AggregateOperation) -> None:
        if GroupBy.ENTITY not in operation.group_by:
            return
        source = self._source_upstream(operation.input)
        if source is not None and not source.entities and not self._states_across(operation.input):
            self._issue(
                CompileIssueCode.ENTITY_UNRESOLVED,
                f"operation {operation.id!r} groups by entity, but source {source.spec_id} "
                "resolves no entity columns to group by",
                operation_id=operation.id,
            )

    def _check_sum_of(self, operation: SumOfOperation) -> None:
        """Report a sum whose sides could count one flow twice, or add different units.

        Each side is read as the full set of retrievals its shape rests on,
        so a side that is itself a sum is checked through to its sources.
        `SUM_SIDES_OVERLAP` when a source behind one side reads the same
        definition as a source behind the other, they elected a StatVar
        in common, or one's StatVar family holds the other's, and the
        entities they bind do not keep their rows apart: the sum would add
        some values to themselves. `SUM_OVERLAP_UNDECIDED` when a side
        lists no members and the two identities cannot show the families
        apart (`compiler.sources.family_overlap`). Whether two
        distinct sources are additive is a methodological fact the plan's
        decisions state, not one code can check. `SUM_UNITS_DIFFER` when the
        dataset requirements behind the sum do not all state the same unit,
        including one stating a unit and another not: the currency and price
        basis sit in the StatVar's identity, where analysis-mcp's own unit
        check does not see them.
        """
        left_roots = self._root_requirement_ids(operation.left.input)
        right_roots = self._root_requirement_ids(operation.right.input)
        found = {
            overlap
            for left_id in left_roots
            for right_id in right_roots
            if (overlap := self._sources_overlap(left_id, right_id)) is not None
        }
        if overlaps := sorted(
            what for code, what in found if code is CompileIssueCode.SUM_SIDES_OVERLAP
        ):
            self._issue(
                CompileIssueCode.SUM_SIDES_OVERLAP,
                f"operation {operation.id!r} adds two sides that both read "
                f"{'; '.join(overlaps)}, so the sum would count those values twice",
                operation_id=operation.id,
            )
        if undecided := sorted(
            what for code, what in found if code is CompileIssueCode.SUM_OVERLAP_UNDECIDED
        ):
            self._issue(
                CompileIssueCode.SUM_OVERLAP_UNDECIDED,
                f"operation {operation.id!r} adds {'; '.join(undecided)}, and the search "
                "results cannot rule out a StatVar both families hold, so the sum could count "
                "those values twice",
                operation_id=operation.id,
            )
        roots = [
            requirement
            for requirement_id in sorted({*left_roots, *right_roots})
            if isinstance(requirement := self.requirements.get(requirement_id), DatasetRequirement)
        ]
        if len({requirement.unit for requirement in roots}) > 1:
            stated = ", ".join(
                f"requirement {requirement.id!r} ({_unit_words(requirement)})"
                for requirement in roots
            )
            self._issue(
                CompileIssueCode.SUM_UNITS_DIFFER,
                f"operation {operation.id!r} adds {stated}; a sum needs every side in one "
                "currency and price basis",
                operation_id=operation.id,
            )

    def _sources_overlap(self, left_id: str, right_id: str) -> tuple[CompileIssueCode, str] | None:
        """What the sources two requirements' retrievals selected read in common, if anything.

        Nothing when the entities they bind keep their rows apart
        (`compiler.sources.entity_scopes_disjoint`), whatever StatVars they share.

        `SUM_SIDES_OVERLAP` with what both read when they share a definition
        or a member digest, or one family holds the other, decided from the
        two identities (`compiler.sources.family_overlap`):
        `SUM_OVERLAP_UNDECIDED`, with the two definitions, when they cannot
        show the families apart.
        """
        left = self._source_of(left_id)
        right = self._source_of(right_id)
        if left is None or right is None or entity_scopes_disjoint(left, right):
            return None
        if left.spec_id == right.spec_id:
            return CompileIssueCode.SUM_SIDES_OVERLAP, f"definition {left.spec_id}"
        if left.shape_member_digest is not None and (
            left.shape_member_digest == right.shape_member_digest
        ):
            return (
                CompileIssueCode.SUM_SIDES_OVERLAP,
                f"StatVar family with member digest {left.shape_member_digest}",
            )
        return _family_overlap_issue(left, right)

    def _check_rank(self, operation: RankOperation) -> None:
        source = self._source_upstream(operation.input)
        if source is not None and not source.entities and not self._states_across(operation.input):
            self._issue(
                CompileIssueCode.ENTITY_UNRESOLVED,
                f"operation {operation.id!r} ranks entities, but source {source.spec_id} "
                "resolves no entity columns to rank by",
                operation_id=operation.id,
            )
        upstream = self.shapes.get(operation.input)
        if upstream is not None and operation.axis is None and len(upstream.shape.roles) > 1:
            names = _axis_phrases(upstream.columns)
            self._issue(
                CompileIssueCode.RANK_AXIS_REQUIRED,
                f"operation {operation.id!r} ranks a frame carrying several roles {names!r} and "
                "names no axis; say which one to rank on",
                operation_id=operation.id,
            )

    def _check_key_axes(self, operation: ShareOfOperation | QuotientOfOperation) -> None:
        """Report a side whose key cannot resolve to a literal column.

        `SHARE_KEY_UNKNOWN` for a share, `QUOTIENT_KEY_UNKNOWN` for a quotient.

        Each side is looked up under the phrase `_keyed_axes` keys it by.
        `planner.shapes.rejects` refuses the same fact at the shape level
        (`INPUT_SHAPE_UNSUPPORTED`, by plan-vocabulary role phrase); this
        states it at the column-resolution level `_key_arguments` depends
        on.

        With no `axis` at all, `_check_needs_axis` takes over: an un-keyed
        division still has to make sense on its own terms. With both sides
        keyed, `_check_keys_overlap` reads the members the two keys hold.
        """
        if operation.axis is None:
            self._check_needs_axis(operation)
            return
        self._check_keys_overlap(operation)
        sides = (("numerator", operation.numerator), ("denominator", operation.denominator))
        for (name, side), axis in zip(sides, _keyed_axes(operation), strict=True):
            resolved = self.shapes.get(side.input)
            if axis is None or resolved is None:
                continue
            if role_named(resolved.columns, axis) is not None:
                continue
            roles = _axis_phrases(resolved.columns)
            self._issue(
                CompileIssueCode.SHARE_KEY_UNKNOWN
                if isinstance(operation, ShareOfOperation)
                else CompileIssueCode.QUOTIENT_KEY_UNKNOWN,
                f"operation {operation.id!r} keys its {name} by {axis!r}, which the {name}'s "
                f"own frame does not carry; its roles are {roles!r}",
                operation_id=operation.id,
            )

    def _check_keys_overlap(self, operation: ShareOfOperation | QuotientOfOperation) -> None:
        """Report `KEYS_DISJOINT` when two keyed sides hold no entity in common.

        A division keyed on both sides pairs each numerator key with the
        denominator's entry for the same entity, so two key columns sharing
        none pair nothing (`quotient_of_no_key_matched`). Each side's members
        are the refs its source resolves under its key column that still
        reach the side (`_key_members`). A side whose key column is open, or
        whose members cannot be read, is left to analysis-mcp.
        """
        numerator_axis, denominator_axis = _keyed_axes(operation)
        if numerator_axis is None or denominator_axis is None:
            return
        numerator = self._key_members(operation.numerator, numerator_axis)
        denominator = self._key_members(operation.denominator, denominator_axis)
        if not numerator or not denominator or numerator.keys() & denominator.keys():
            return
        self._issue(
            CompileIssueCode.KEYS_DISJOINT,
            f"operation {operation.id!r} divides key by key, and its numerator holds "
            f"{sorted(numerator.values())!r} under {numerator_axis!r} while its denominator "
            f"holds {sorted(denominator.values())!r} under {denominator_axis!r}: no entity is "
            "on both sides, so nothing would be divided",
            operation_id=operation.id,
        )

    def _key_members(self, side: Side, axis: str) -> dict[str, str]:
        """The label of each entity `side` holds under the column `axis` keys it by, by ref.

        Empty when the key resolves to no column, the column is open, or its
        source is missing. The members are those the column's source
        resolves, narrowed by any upstream filter (`_retained_labels`) and by
        the entity the side itself names.
        """
        role = self._key_column(side, axis)
        source = self._role_source(role) if role is not None else None
        if role is None or role.open or source is None:
            return {}
        retained = _narrowed_by_side(side, role, source, self._retained_labels(side.input, role))
        return {
            entity.ref: entity.label
            for entity in source.entities
            if entity.role == role.column and entity.label.casefold() in retained
        }

    def _check_needs_axis(self, operation: ShareOfOperation | QuotientOfOperation) -> None:
        """Report an un-keyed numerator that still carries several members.

        `SHARE_AXIS_REQUIRED` for a share, `QUOTIENT_AXIS_REQUIRED` for a
        quotient.

        `numerator.entity` narrows a whole side to one series by itself, so
        this reads only a numerator left unnarrowed that way. It counts the
        members the numerator's own chain still carries under each role
        (`_retained_member_count`), after any upstream filter, and not the
        role dimensions: a numerator can carry one column, `observationAbout`
        say, and still hold several members under it. A numerator filtered
        down to one entity needs no axis.

        Left with more than one, an un-keyed division sums every member into
        one figure for nobody in particular. Over an equally un-keyed
        denominator of the same members a share is 1 at every period. Naming
        an axis lets the numerator answer for each member.
        """
        if operation.numerator.entity is not None:
            return
        numerator = self.shapes.get(operation.numerator.input)
        if numerator is None:
            return
        pooled = [
            role
            for role in numerator.columns
            if self._retained_member_count(operation.numerator.input, role) > 1
        ]
        if not pooled:
            return
        roles = _axis_phrases(pooled)
        code, verb, noun = (
            (CompileIssueCode.SHARE_AXIS_REQUIRED, "shares", "share")
            if isinstance(operation, ShareOfOperation)
            else (CompileIssueCode.QUOTIENT_AXIS_REQUIRED, "divides", "quotient")
        )
        self._issue(
            code,
            f"operation {operation.id!r} {verb} a frame with no axis, but its numerator still "
            f"carries several members under {roles!r}; name one to key the {noun} on",
            operation_id=operation.id,
        )

    def _check_quotient_parts(self, operation: QuotientOfOperation) -> None:
        """Report `FRAME_UNIT_UNBOUND` for a side that has no part in a composed unit.

        A quotient's unit names each side by its unit or, for a side that
        declares none, by its StatVar. A `sum_of` result carries a StatVar of
        its own that no graph node or stored definition names, so a sum of
        two unit-less sources has neither, and analysis-mcp refuses it
        (`quotient_of_side_unit_unsupported`). The sum is found through any
        single-input chain the side reads it by, and is unit-less when every
        source behind it declares no unit (`_frame_unit`). A sum whose
        sources declare a unit, or whose unit cannot be known ahead, is left
        to analysis-mcp.

        The other sides with no part, a share, a relative growth, a quotient
        and a rank position, are refused by their value kind
        (`planner.shapes.rejects`).
        """
        sides = (("numerator", operation.numerator), ("denominator", operation.denominator))
        for name, side in sides:
            summed = self._sum_behind(side.input)
            if summed is None:
                continue
            sources = [self._source_of(root) for root in self._root_requirement_ids(summed)]
            if not sources or any(
                source is None or _frame_unit(source) is not None for source in sources
            ):
                continue
            self._issue(
                CompileIssueCode.FRAME_UNIT_UNBOUND,
                f"operation {operation.id!r} takes its {name} from the sum {summed!r}, whose "
                "sources declare no unit. quotient_of names a unit-less side by its StatVar, "
                "and a sum has none a reader could name, so no operation computes this from "
                "these sources",
                operation_id=operation.id,
            )

    def _sum_behind(self, op_id: str) -> str | None:
        """The `sum_of` whose result `op_id` reads, itself or through single-input operations."""
        return self._behind(op_id, SumOfOperation)

    def _behind(self, op_id: str, kind: type[Operation]) -> str | None:
        """The operation of `kind` whose result `op_id` reads, itself or through single-input operations."""
        seen: set[str] = set()
        while op_id not in seen:
            seen.add(op_id)
            operation = self.operations.get(op_id)
            if isinstance(operation, kind):
                return op_id
            if not isinstance(operation, _SINGLE_INPUT):
                return None
            op_id = operation.input
        return None

    def _retained_member_count(self, op_id: str, role: ResolvedRole) -> int:
        """How many distinct members of `role` still reach `op_id`, after any upstream filter.

        Starts from every label `role`'s own source resolves under its
        column, then walks the chain back to its retrieval, through each
        single-input operation and from a `share_of` or `quotient_of` to its
        numerator, whose key column the result carries. It narrows
        that set by the entity a dividing verb's numerator names and by the
        tightest `EntityIn` predicate any
        `FilterOperation` along the way names for it -- by the predicate's
        own `role` phrase, or, unqualified, by resolving each of its names
        against `role`'s source the way `compiler.shapes.
        _resolved_narrowed_roles` does. A predicate that still leaves
        several labels standing pools exactly as meaninglessly as no filter
        at all, so this counts what remains rather than trusting a filter's
        mere presence.

        An open entity axis counts the members its source states, which
        `compiler.evidence.open_axes_for` admits only when counted. No
        filter narrows one: an `EntityIn` may only name the requirement's
        own entities, and none sits on an open axis.
        """
        open_axis = self._open_axis_of(role)
        if open_axis is not None:
            return open_axis.member_count or 0
        return len(self._retained_labels(op_id, role))

    def _retained_labels(self, op_id: str, role: ResolvedRole) -> set[str]:
        """The case-folded labels of `role` that still reach `op_id`; see `_retained_member_count`."""
        source = self._role_source(role)
        if source is None:
            return set()
        retained = {e.label.casefold() for e in source.entities if e.role == role.column}
        seen: set[str] = set()
        while op_id in self.operations and op_id not in seen:
            seen.add(op_id)
            operation = self.operations[op_id]
            if isinstance(operation, RetrieveOperation):
                break
            if isinstance(operation, FilterOperation):
                retained = _narrowed_by_filter(operation, role, source, retained)
            if isinstance(operation, ShareOfOperation | QuotientOfOperation):
                # A dividing verb's result carries its numerator's key column.
                retained = _narrowed_by_side(operation.numerator, role, source, retained)
                op_id = operation.numerator.input
                continue
            if not isinstance(operation, _SINGLE_INPUT):
                break
            op_id = operation.input
        return retained

    def _states_across(self, op_id: str) -> bool:
        """Whether the dataset requirement `op_id`'s chain rests on states `across`."""
        requirement = self._requirement_upstream(op_id)
        return requirement is not None and bool(requirement.across)

    def _role_source(self, role: ResolvedRole) -> SelectedSource | None:
        """The source the retrieval of `role`'s own requirement selected."""
        retrieve_id = self._retrieve_by_requirement.get(role.requirement_id)
        return self.sources.get(retrieve_id) if retrieve_id is not None else None

    def _role_upstream(self, role: ResolvedRole) -> Upstream | None:
        """The source and dataset requirement `role`'s column came from, or `None` if either is missing."""
        source = self._role_source(role)
        requirement = self.requirements.get(role.requirement_id)
        if source is None or not isinstance(requirement, DatasetRequirement):
            return None
        return source, requirement

    def _named_behind(self, op_id: str) -> list[Upstream] | None:
        """The source and dataset requirement behind the entities the frame at `op_id` holds, each once.

        A frame holds entities under the role columns of its own resolved
        shape, and each column carries the requirement it came from
        (`ResolvedRole.requirement_id`). So a keyed `share_of` or
        `quotient_of` result, which rests on two retrievals, answers with its
        numerator's, the side its key column came from. A frame with no role
        column that rests on one retrieval answers with that retrieval's, so
        a name its source does not resolve is still reported against it.
        Empty for a frame with no role column resting on several retrievals:
        it is one series and holds no entity to name.

        `None` when `op_id` has no shape, or a source or dataset requirement
        behind it is missing, each of which an issue already reports.
        """
        resolved = self.shapes.get(op_id)
        if resolved is None:
            return None
        requirement_ids = list(dict.fromkeys(role.requirement_id for role in resolved.columns))
        if not requirement_ids and len(resolved.shape.roots) == 1:
            requirement_ids = list(resolved.shape.roots)
        behind: list[Upstream] = []
        for requirement_id in requirement_ids:
            source = self._source_of(requirement_id)
            requirement = self.requirements.get(requirement_id)
            if source is None or not isinstance(requirement, DatasetRequirement):
                return None
            behind.append((source, requirement))
        return behind

    def _check_names_known(
        self, op_id: str, input_id: str, names: Sequence[str], *, role: str | None = None
    ) -> bool:
        """Report `UNKNOWN_ENTITY` for each of `names` no requirement behind `input_id`'s frame states.

        `role` is the role phrase the names are stated under, when the
        operation gives one: a name the requirement states under another
        role is as unknown as one it does not state. Returns whether every
        name is known, `False` too when the frame's requirements cannot be
        read (`_named_behind`).
        """
        behind = self._named_behind(input_id)
        if behind is None:
            return False
        unknown = [
            name
            for name in names
            if not any(_entity_candidates(requirement, name, role) for _, requirement in behind)
        ]
        if not unknown:
            return True
        stated = sorted(
            {
                entity.name
                for _, requirement in behind
                for entity in requirement.entities
                if _stated_under(entity, role)
            }
        )
        under = f" under role {role!r}" if role is not None else ""
        self._issue(
            CompileIssueCode.UNKNOWN_ENTITY,
            f"operation {op_id!r} names {unknown!r}{under}, and the frame it reads from "
            f"{input_id!r} holds none by that name; the entities its requirements state"
            f"{under} are {stated!r}",
            operation_id=op_id,
            requirement_id=behind[0][1].id if len(behind) == 1 else None,
        )
        return False

    def _open_axis_of(self, role: ResolvedRole) -> OpenEntityAxis | None:
        """The open entity axis of `role`'s source on `role`'s column, or `None` when it is not open."""
        source = self._role_source(role)
        if source is None:
            return None
        open_entity_axes = source.open_entity_axes or ()
        return next((axis for axis in open_entity_axes if axis.column == role.column), None)

    def _check_join(self, operation: JoinOperation) -> None:
        left = self._columns_of(operation.left)
        right = self._columns_of(operation.right)
        if left is None or right is None:
            return
        left_columns = set(left)
        right_columns = set(right)
        if not (left_columns & right_columns):
            self._issue(
                CompileIssueCode.JOIN_NO_COMMON_KEY_COLUMNS,
                f"operation {operation.id!r} joins sources that share no role column: "
                f"{sorted(left_columns)!r} and {sorted(right_columns)!r}",
                operation_id=operation.id,
            )

    def _check_filter(self, operation: FilterOperation) -> None:
        for predicate in operation.predicates:
            if not isinstance(predicate, EntityIn):
                continue
            if not self._check_names_known(
                operation.id, operation.input, predicate.names, role=predicate.role
            ):
                continue
            for name in predicate.names:
                found = self._frame_entity(operation.id, operation.input, name, role=predicate.role)
                if isinstance(found, CompileIssue):
                    self.issues.append(found)

    def _check_two_sided(
        self,
        op_id: str,
        left: Side,
        right: Side,
        *,
        keyed: tuple[str | None, str | None] = (None, None),
        any_units: bool = False,
    ) -> None:
        """Check the two sides of `op_id` against each other.

        `keyed` is the role phrase each side is keyed by (`_keyed_axes`),
        `None` for a side read whole. `any_units` is for a verb that pairs
        sides in any two units, which skips `_check_frame_units`.
        """
        requirements = [
            self._check_side(op_id, side, keyed=axis)
            for side, axis in zip((left, right), keyed, strict=True)
        ]
        left_identity, right_identity = (self._side_identity(side) for side in (left, right))
        if left_identity == right_identity:
            frame, entity = left_identity
            about = f"{entity!r} of {frame!r}" if entity else f"the whole of {frame!r}"
            self._issue(
                CompileIssueCode.DEGENERATE_SIDES,
                f"operation {op_id!r} sets one side against itself: both sides are {about}",
                operation_id=op_id,
            )
            return
        if left.input == right.input:
            # One requirement behind both sides: its period overlaps itself.
            return
        left_req, right_req = requirements
        if (
            left_req is not None
            and right_req is not None
            and not left_req.period.overlaps(right_req.period)
        ):
            self._issue(
                CompileIssueCode.PERIODS_DISJOINT,
                f"operation {op_id!r} sets sides against each other whose periods share no year",
                operation_id=op_id,
            )
        if not any_units:
            self._check_frame_units(op_id, left, right)

    def _check_frame_units(self, op_id: str, left: Side, right: Side) -> None:
        """Report two sides whose frames declare their units differently.

        analysis-mcp pairs two slices only when their `unit` cells are equal
        (`cross_frame_unit_mismatch`), and each side's cells are what its
        source declares (`_frame_unit`): the request object for a definition
        bound to a unit by request, the source's unit ref for any other.
        A frame downstream of a `quotient_of` declares the composed pair of
        its two sides' units, and one holding a `share_of` result declares
        `PERCENT_UNIT` (`_side_unit`). A side whose unit cannot be known
        ahead is left to analysis-mcp.

        `FRAME_UNIT_UNBOUND` when exactly one side's source declares no unit:
        `compare`, `share_of` and `sum_of` each refuse that pair
        (`cross_frame_unit_unbound`), and neither `compare` nor `sum_of` has
        another operation that sets a unit-less count against a value in a
        unit. Two unit-less sides pair.

        `FRAME_UNIT_FORMS_DIFFER` when both denote one currency and price
        basis (`_denoted_unit`) and only the form differs, a ref against a
        request: every side whose source binds no unit is named (both, when
        both are refs), with what to change so it is retrieved by request too.
        dfo-apis writes a natively satisfied request as the request object and
        an unbound definition as its ref (ONEcampaign/dfo-apis#1179).

        No requirement may ask for constant prices (`_check_requirements_meetable`),
        so the one unit two sides can both be stated in is a current-price
        one. `FRAME_UNITS_IRRECONCILABLE` when a side's requirement pins its
        source and that source cannot serve current prices, as `check_evidence`
        reads its unit: no requirement change pairs the two, and the issue
        names the pinned side's requirement. `FRAME_PRICE_BASES_DIFFER` when
        either side states constant prices otherwise, whose repair is current
        prices on both sides. `FRAME_UNITS_DIFFER` for any other pair.

        A `share_of` reports `SHARE_UNITS_DIFFER` in place of
        `FRAME_UNIT_UNBOUND` and `FRAME_UNITS_DIFFER`: `quotient_of` divides
        the same two sides and states the result in a composed unit. A pair
        with a composed side is `FRAME_UNITS_DIFFER` for every verb, a share
        included, since a quotient is not divided again by `quotient_of`.

        A side holding a ratio (`_ratio_held`) is read too. A share is in
        `PERCENT_UNIT`, and a relative growth or a rank position has no unit.
        A pair holding a ratio whose two units differ is `FRAME_UNIT_UNBOUND`
        for every verb (`_report_ratio_side_units`). analysis-mcp refuses it
        (`cross_frame_unit_unbound`, `cross_frame_unit_mismatch`), and
        `quotient_of` divides no ratio, so no operation computes it.
        """
        shares = isinstance(self.operations[op_id], ShareOfOperation)
        units: list[JsonValue | _ComposedUnit] = []
        ratios: list[_Ratio | None] = []
        for side in (left, right):
            resolved = self.shapes.get(side.input)
            if resolved is None:
                return
            ratio = self._ratio_held(side.input)
            ratios.append(ratio)
            if ratio is _Ratio.UNITLESS:
                units.append(None)
                continue
            if (
                ratio is None
                and resolved.shape.value not in (ValueKind.MEASURED, ValueKind.DIFFERENCE)
            ) or (unit := self._side_unit(side.input)) is _PerRowUnit.PER_ROW:
                return
            units.append(unit)
        if units[0] == units[1]:
            return
        if any(ratio is not None for ratio in ratios):
            self._report_ratio_side_units(
                op_id, list(zip((left, right), ratios, units, strict=True))
            )
            return
        if any(isinstance(unit, _ComposedUnit) for unit in units):
            self._issue(
                CompileIssueCode.FRAME_UNITS_DIFFER,
                f"operation {op_id!r} pairs a side in {self._unit_words(left.input, units[0])} "
                f"with one in {self._unit_words(right.input, units[1])}, and two frames in "
                "different units cannot be set against each other",
                operation_id=op_id,
            )
            return
        self._report_source_units(op_id, left, right, shares=shares)

    def _report_ratio_side_units(
        self, op_id: str, sides: list[tuple[Side, _Ratio | None, JsonValue | _ComposedUnit]]
    ) -> None:
        """Report `FRAME_UNIT_UNBOUND` for two sides in different units, one holding a ratio.

        `sides` holds each side with the ratio it holds (`_ratio_held`) and
        its unit. The issue is about the side holding no ratio, when there is
        one. A source for its requirement in the ratio side's unit would
        pair, such as a published share in `PERCENT_UNIT` against a computed
        one.
        """
        (ratio_side, ratio, _), (other, other_ratio, other_unit) = (
            sides if sides[0][1] is not None else sides[::-1]
        )
        held = (
            "a share, in per cent"
            if ratio is _Ratio.SHARE
            else "a relative growth or a rank position, which has no unit"
        )
        tool = self.operations[op_id].kind.value
        requirement = self._requirement_upstream(other.input) if other_ratio is None else None
        other_side = (
            f"requirement {requirement.id!r}" if requirement is not None else repr(other.input)
        )
        clears = (
            f"; a source for {other_side} that declares "
            f"{PERCENT_UNIT if ratio is _Ratio.SHARE else 'no unit'} would also clear it"
            if requirement is not None
            else ""
        )
        self._issue(
            CompileIssueCode.FRAME_UNIT_UNBOUND,
            f"operation {op_id!r} sets {other_side}, in "
            f"{self._unit_words(other.input, other_unit)}, against {ratio_side.input!r}, "
            f"{held}. {tool} pairs two sides in one unit or two with none, and quotient_of "
            f"divides no share, relative growth or rank position, so no operation computes "
            f"this from these sources{clears}",
            operation_id=op_id,
            requirement_id=requirement.id if requirement is not None else None,
        )

    def _holds_share(self, op_id: str) -> bool:
        """Whether the frame at `op_id` holds a `share_of` result, or its absolute change."""
        return self._ratio_held(op_id) is _Ratio.SHARE

    def _ratio_held(self, op_id: str) -> _Ratio | None:
        """The ratio the frame at `op_id` holds, or `None` when it holds none.

        `_Ratio.UNITLESS` for a rank position, or a frame reading a relative
        growth through single-input operations. `_Ratio.SHARE` for one
        reading a `share_of` that way. An aggregate, a filter or a rank with
        a limit keeps the share's value, and an absolute growth reads it in
        percentage points, all in `PERCENT_UNIT`.
        """
        resolved = self.shapes.get(op_id)
        if resolved is None:
            return None
        if resolved.shape.value is ValueKind.ORDINAL:
            return _Ratio.UNITLESS
        seen: set[str] = set()
        while op_id not in seen:
            seen.add(op_id)
            operation = self.operations.get(op_id)
            if isinstance(operation, ShareOfOperation):
                return _Ratio.SHARE
            if isinstance(operation, GrowthOperation) and operation.change is GrowthChange.RELATIVE:
                return _Ratio.UNITLESS
            if not isinstance(operation, _SINGLE_INPUT):
                return None
            op_id = operation.input
        return None

    def _report_source_units(self, op_id: str, left: Side, right: Side, *, shares: bool) -> None:
        """Report two sides whose sources declare different units, for `_check_frame_units`.

        Each side reads one retrieval through single-input operations, and
        the two units differ.
        """
        read: list[_SideUnit] = []
        for side in (left, right):
            source = self._source_upstream(side.input)
            assert source is not None
            unit = _frame_unit(source)
            assert unit is not _PerRowUnit.PER_ROW
            denoted = _denoted_unit(source, unit)
            read.append(
                _SideUnit(
                    source=source,
                    unit=unit,
                    denoted=denoted,
                    reading=_unit_reading(source, denoted),
                    requirement=self._requirement_upstream(side.input),
                )
            )
        left_unit, left_denoted, left_req = read[0].unit, read[0].denoted, read[0].requirement
        right_unit, right_denoted, right_req = read[1].unit, read[1].denoted, read[1].requirement
        if shares and (left_unit is None or right_unit is None):
            self._report_share_units(
                op_id,
                *(
                    "no unit" if unit is None else _bound_unit_words(unit, denoted)
                    for unit, denoted in ((left_unit, left_denoted), (right_unit, right_denoted))
                ),
            )
            return
        if left_unit is None or right_unit is None:
            unbound, bound, bound_unit, bound_denoted = (
                (left_req, right_req, right_unit, right_denoted)
                if left_unit is None
                else (right_req, left_req, left_unit, left_denoted)
            )
            self._report_unit_unbound(
                op_id,
                unbound=unbound,
                bound=bound,
                bound_unit=_bound_unit_words(bound_unit, bound_denoted),
            )
            return
        unbound = [
            req
            for unit, req in ((left_unit, left_req), (right_unit, right_req))
            if not isinstance(unit, dict) and req is not None
        ]
        if left_denoted is not None and left_denoted == right_denoted and unbound:
            subject = (
                f"requirement {unbound[0].id!r}"
                if len(unbound) == 1
                else "requirements " + " and ".join(repr(req.id) for req in unbound)
            )
            self._issue(
                CompileIssueCode.FRAME_UNIT_FORMS_DIFFER,
                f"operation {op_id!r} pairs two sides in {_denoted_words(left_denoted)}, but "
                f"the unit of {subject} is not bound by request, so the frames state it "
                "differently and the two are not paired; state unit "
                f"{left_denoted.model_dump_json(exclude_none=True)} on {subject} and leave "
                "source_spec_id unset, so the search runs again and retrieves the data bound "
                "to that unit",
                operation_id=op_id,
                requirement_id=unbound[0].id,
            )
            return
        if shares:
            self._report_share_units(
                op_id,
                _unit_cell(left_unit, left_denoted),
                _unit_cell(right_unit, right_denoted),
            )
            return
        if self._report_irreconcilable_units(op_id, read[0], read[1]):
            return
        constant = any(side.reading.prices == "constant" for side in read)
        self._issue(
            CompileIssueCode.FRAME_PRICE_BASES_DIFFER
            if constant
            else CompileIssueCode.FRAME_UNITS_DIFFER,
            f"operation {op_id!r} pairs a side in {read[0].words()} with one in "
            f"{read[1].words()}, and two frames in different units cannot be set against "
            "each other",
            operation_id=op_id,
        )

    def _report_irreconcilable_units(self, op_id: str, left: _SideUnit, right: _SideUnit) -> bool:
        """Report `FRAME_UNITS_IRRECONCILABLE` for a pinned side whose source cannot serve current prices.

        A current-price unit is the one unit two sides can both be stated
        in, and a pinned requirement is always served by its own source.
        Each pinned side's source unit is read as `check_evidence` reads it
        (`read_source_unit`). A source unit that reads as constant prices
        cannot be stated at current prices in any currency, so no
        requirement change pairs the two. A pinned side bound to a unit by
        request is skipped, because its frame is converted on fetch and its
        own reading already states the basis it serves. Returns whether it
        reported.
        """
        for pinned, other in ((left, right), (right, left)):
            requirement = pinned.requirement
            if (
                requirement is None
                or requirement.source_spec_id is None
                or isinstance(pinned.unit, dict)
                or read_source_unit(pinned.source).prices != "constant"
            ):
                continue
            self._issue(
                CompileIssueCode.FRAME_UNITS_IRRECONCILABLE,
                f"operation {op_id!r} pairs requirement {requirement.id!r}, whose pinned source "
                f"{pinned.source.spec_id} serves {pinned.words()}, with a side in "
                f"{other.words()}. No operation restates a series in other prices and no "
                "requirement may ask for constant prices, so the two sides can never be "
                "stated in one unit",
                operation_id=op_id,
                requirement_id=requirement.id,
            )
            return True
        return False

    def _side_unit(self, op_id: str) -> FrameUnit | _ComposedUnit:
        """The unit the frame at `op_id` declares in its `unit` cells, as far as it is known ahead.

        A frame holding a `share_of` result declares `PERCENT_UNIT`
        (`_holds_share`). A frame read from one retrieval through
        single-input operations declares its source's unit (`_frame_unit`).
        A `quotient_of` result, and every frame read from one through
        single-input operations, declares the composed pair of its two
        sides' units (`_composed_part`). `_PerRowUnit.PER_ROW` when no
        reading applies or a part of the pair cannot be named.
        """
        if self._holds_share(op_id):
            return PERCENT_UNIT
        quotient = self.operations.get(self._behind(op_id, QuotientOfOperation) or "")
        if not isinstance(quotient, QuotientOfOperation):
            source = self._source_upstream(op_id)
            return _frame_unit(source) if source is not None else _PerRowUnit.PER_ROW
        numerator = self._composed_part(quotient.numerator.input)
        denominator = self._composed_part(quotient.denominator.input)
        if numerator is None or denominator is None:
            return _PerRowUnit.PER_ROW
        return _ComposedUnit(numerator=numerator, denominator=denominator)

    def _composed_part(self, op_id: str) -> JsonValue | _StatVarPart:
        """What a quotient names the side read from `op_id` by: its unit, or its StatVar when it declares none.

        `None` when the side has no shape, its unit cannot be known ahead, it
        is itself in a composed unit, or it declares no unit and its source
        pins no one StatVar. analysis-mcp decides each of those.
        """
        if op_id not in self.shapes:
            return None
        unit = self._side_unit(op_id)
        if unit is _PerRowUnit.PER_ROW or isinstance(unit, _ComposedUnit):
            return None
        if unit is not None:
            return unit
        source = self._source_upstream(op_id)
        stat_var = pinned_stat_var(source.identity) if source is not None else None
        return _StatVarPart(stat_var) if stat_var is not None else None

    def _unit_words(self, op_id: str, unit: JsonValue | _ComposedUnit) -> str:
        """`unit`, the unit of the frame at `op_id`, in words for an issue's reason."""
        if isinstance(unit, _ComposedUnit):
            return f"{_part_words(unit.numerator)} per {_part_words(unit.denominator)}"
        if unit is None:
            return "no unit"
        if self._holds_share(op_id):
            return "per cent, as a share"
        source = self._source_upstream(op_id)
        return _unit_cell(unit, _denoted_unit(source, unit) if source is not None else None)

    def _report_share_units(self, op_id: str, numerator: str, denominator: str) -> None:
        self._issue(
            CompileIssueCode.SHARE_UNITS_DIFFER,
            f"operation {op_id!r} takes a share of a numerator in {numerator} over a "
            f"denominator in {denominator}. share_of divides two sides in one unit or two "
            "with none; quotient_of divides one measure by another in any two units",
            operation_id=op_id,
        )

    def _report_unit_unbound(
        self,
        op_id: str,
        *,
        unbound: DatasetRequirement | None,
        bound: DatasetRequirement | None,
        bound_unit: str,
    ) -> None:
        tool = self.operations[op_id].kind.value
        unbound_side = f"requirement {unbound.id!r}" if unbound is not None else "one side"
        bound_side = f"requirement {bound.id!r}" if bound is not None else "the other"
        self._issue(
            CompileIssueCode.FRAME_UNIT_UNBOUND,
            f"operation {op_id!r} sets {unbound_side}, whose source declares no unit, against "
            f"{bound_side} in {bound_unit}. {tool} pairs two sides in one unit or two with "
            "none, so no operation computes this from these sources; a source for "
            f"{unbound_side} that declares {bound_unit} would also clear it",
            operation_id=op_id,
            requirement_id=unbound.id if unbound is not None else None,
        )

    def _check_side(
        self, op_id: str, side: Side, *, keyed: str | None = None
    ) -> DatasetRequirement | None:
        """Check what `side` reads, and return the one dataset requirement it rests on, if one."""
        if side.entity is None:
            self._check_whole_side(op_id, side, keyed=keyed)
        else:
            self._check_names_known(op_id, side.input, [side.entity])
        return self._requirement_upstream(side.input)

    def _check_whole_side(self, op_id: str, side: Side, *, keyed: str | None) -> None:
        """Report `ENTITY_REQUIRED` when a side naming no entity still holds several series.

        A side keyed by an axis (`_keyed_axes`) reads one series per member
        of the column `keyed` names, so that column is left out and every
        other column is held to the rule.

        Reads the side's own resolved shape, the frame at `side.input`, as
        `planner.plan._check_side_entity_is_named` reads the plan's. A role
        column an aggregate or a rank collapsed is gone. A filter's narrowed
        flag says only that it named entities, so each column reads the
        labels still reaching the side (`_retained_labels`): a filter keeping
        one leaves one series, a filter keeping two leaves two, and the issue
        names those two. A breakdown's column holds codes and is left to the
        operation. Each column is read against the requirement and source it
        came from (`_role_upstream`), so a side reading a keyed `share_of`
        or `quotient_of` result, which rests on two retrievals, is held to
        the rule on that result's key column. One surviving column makes the
        side ambiguous, checked in this order: a column on which several of
        the entities its requirement names still reach the side, an open
        column, whose members nobody named, or a column on which several of
        the labels its source resolves still reach it.
        """
        resolved = self.shapes.get(side.input)
        if resolved is None:
            return
        key = role_named(resolved.columns, keyed) if keyed is not None else None
        columns = [
            (column, upstream)
            for column in resolved.columns
            if not column.breakdown
            and column is not key
            and (upstream := self._role_upstream(column)) is not None
        ]
        held = [
            (column, upstream, self._retained_labels(side.input, column))
            for column, upstream in columns
            if not column.open
        ]
        pooled = [
            (requirement, names)
            for column, (source, requirement), kept in held
            if len(names := _names_on_column(requirement, source, column.column, kept)) > 1
        ]
        labels = [
            (source, requirement, column_labels)
            for column, (source, requirement), kept in held
            if len(column_labels := _labels_on_column(source, column.column, kept)) > 1
        ]
        open_column = next(
            ((column, requirement) for column, (_, requirement) in columns if column.open), None
        )
        if pooled:
            names = [name for _, column_names in pooled for name in column_names]
            self._issue(
                CompileIssueCode.ENTITY_REQUIRED,
                f"operation {op_id!r} takes a whole side from {side.input!r}, whose "
                f"requirement names several entities on one side of the flow {names!r}; "
                "say which",
                operation_id=op_id,
                requirement_id=pooled[0][0].id,
            )
        elif open_column is not None:
            # A side can name only an entity its requirement names, so it has no
            # way to say which open members it reads.
            column, requirement = open_column
            self._issue(
                CompileIssueCode.ENTITY_REQUIRED,
                f"operation {op_id!r} takes a whole side from {side.input!r}, whose "
                f"requirement covers every {column.role!r} and names none of them, "
                "so the side cannot say which it reads; name them in the requirement's "
                "`entities`, or collapse them with an aggregate grouped by period first",
                operation_id=op_id,
                requirement_id=requirement.id,
            )
        elif labels:
            source, requirement, _ = labels[0]
            resolved_labels = [label for _, _, column_labels in labels for label in column_labels]
            self._issue(
                CompileIssueCode.ENTITY_REQUIRED,
                f"operation {op_id!r} takes a whole side from {side.input!r}, whose source "
                f"{source.spec_id} resolves several entities {resolved_labels!r}; say which",
                operation_id=op_id,
                requirement_id=requirement.id,
            )

    def _check_source_width(self) -> None:
        """Report a source resolving entities the requirement never named, folded in or ranked alongside.

        A retrieval writes no `scope` (see `_retrieve_arguments`), so the
        frame carries every entity the definition holds; the surplus is
        counted within the requirement's own role columns
        (`_surplus_labels`). Every `ResolvedRole` on a shape already carries
        its own `surplus` and `narrowed` flag, carried forward by every
        transform (`compiler.shapes._carry_columns`) from the retrieval that
        first resolved it to wherever the frame travels -- narrowed once a
        filter has kept only named entities on it -- so a reader only has to
        read its own immediate input's shape, with no walk of its own. Two
        readers report: a period-only aggregate, or a whole side of a
        two-sided operation (`_pooled_sides`), folds a surplus role into a
        figure presented as the asked entity's; a `named` rank claims to
        rank exactly the entities a filter narrowed its axis to, so a
        surplus role elsewhere on the same frame (an axis the rank does not
        use, left un-narrowed) leaks an uninvited entity into the ranking
        it reduces over. An aggregate grouping by entity, a `population`
        rank (the wider field is the point), a growth and a filter each
        keep one row per entity or state the wider claim themselves, so
        none of them are reported. One issue per reporting operation,
        however many of its inputs carry a surplus role.
        """
        for operation in self.operations.values():
            if isinstance(operation, RankOperation) and operation.among is RankAmong.NAMED:
                self._report_source_width(
                    operation.id,
                    ((operation.input, None),),
                    verb="ranks them alongside the entities it names",
                )
                continue
            self._report_source_width(
                operation.id, _pooled_sides(operation), verb="pools them with the entities it does"
            )

    def _report_source_width(
        self, operation_id: str, sides: Sequence[tuple[str, str | None]], *, verb: str
    ) -> None:
        role = self._first_unnarrowed_surplus_role(sides)
        if role is None:
            return
        requirement = self.requirements.get(role.requirement_id)
        retrieve_id = self._retrieve_by_requirement.get(role.requirement_id)
        source = self.sources.get(retrieve_id) if retrieve_id is not None else None
        if not isinstance(requirement, DatasetRequirement) or source is None:
            return
        surplus = _surplus_labels(requirement, source)
        self._issue(
            CompileIssueCode.SOURCE_WIDER_THAN_REQUIREMENT,
            f"source {source.spec_id} resolves {surplus!r}, which requirement "
            f"{requirement.id!r} does not name, and operation {operation_id!r} {verb}; narrow "
            "the frame first",
            operation_id=operation_id,
            requirement_id=requirement.id,
        )

    def _first_unnarrowed_surplus_role(
        self, sides: Sequence[tuple[str, str | None]]
    ) -> ResolvedRole | None:
        """The first surplus, un-narrowed role dimension a reader pools among `sides`' own shapes.

        Each of `sides` is an input and the role phrase its reader keys it
        by (`_pooled_sides`): the reader keeps one series per member of that
        column, so a surplus member there stays a row of its own. Reads
        `self.shapes[input_id].columns` directly, in the order `sides` names
        them -- deterministic, and enough for one issue naming one
        requirement, since a pooling operation with several surplus sides
        still gets exactly one.
        """
        for input_id, keyed in sides:
            resolved = self.shapes.get(input_id)
            if resolved is None:
                continue
            key = role_named(resolved.columns, keyed) if keyed is not None else None
            for role in resolved.columns:
                if role.surplus and not role.narrowed and role is not key:
                    return role
        return None

    def _check_capabilities(self) -> None:
        drift_by_tool: dict[ToolId, list[SchemaDrift]] = {}
        for d in schema_drift(self.capabilities, self.schemas):
            drift_by_tool.setdefault(d.tool, []).append(d)
        for operation in self.operations.values():
            candidates = capabilities_for(operation.kind, self.capabilities)
            if not candidates:
                self._issue(
                    CompileIssueCode.NO_CAPABILITY,
                    f"no tool implements {operation.kind.value!r} operations",
                    operation_id=operation.id,
                )
                continue
            if len(candidates) > 1:
                self._issue(
                    CompileIssueCode.AMBIGUOUS_CAPABILITY,
                    f"{len(candidates)} tools implement {operation.kind.value!r} operations",
                    operation_id=operation.id,
                )
                continue
            (capability,) = candidates
            schema = self.schemas.get(capability.tool.name)
            if schema is None or schema.server != capability.tool.server:
                self._issue(
                    CompileIssueCode.UNKNOWN_TOOL,
                    f"{capability.tool.server} serves no tool named {capability.tool.name!r}",
                    operation_id=operation.id,
                    tool=capability.tool,
                )
                continue
            unusable = unusable_schema_reason(schema.parameters)
            if unusable is not None:
                self._issue(
                    CompileIssueCode.BAD_TOOL_SCHEMA,
                    f"{capability.tool.name!r} declares an unusable argument schema: {unusable}",
                    operation_id=operation.id,
                    tool=capability.tool,
                )
                continue
            hash_error = schema_hash_error(schema.parameters) or schema_hash_error(
                schema.output_schema
            )
            if hash_error is not None:
                self._issue(
                    CompileIssueCode.BAD_TOOL_SCHEMA,
                    f"{capability.tool.name!r} declares a schema that cannot be hashed: "
                    f"{hash_error}",
                    operation_id=operation.id,
                    tool=capability.tool,
                )
                continue
            tool_drift = drift_by_tool.get(capability.tool, [])
            drifted = [d for d in tool_drift if d.pinned is not None]
            unpinned = [d for d in tool_drift if d.pinned is None]
            if drifted:
                d = drifted[0]
                self._issue(
                    CompileIssueCode.CAPABILITY_SCHEMA_DRIFT,
                    f"{capability.tool.name!r} no longer matches its capability row's {d.field} "
                    f"(pinned {d.pinned}, live {d.live}); the row needs re-pinning",
                    operation_id=operation.id,
                    tool=capability.tool,
                )
                continue
            if unpinned:
                d = unpinned[0]
                self._issue(
                    CompileIssueCode.CAPABILITY_UNPINNED,
                    f"{capability.tool.name!r} declares a {d.field} its capability row does not "
                    f"pin (live {d.live}); the row needs pinning",
                    operation_id=operation.id,
                    tool=capability.tool,
                )
                continue
            self.tools[operation.id] = (capability, schema)

    # -- building --------------------------------------------------------------------

    def _build_steps(self) -> None:
        """Build a step for each operation whose arguments can be written.

        A rolling step's arguments name the population the newest period is
        taken over, which rests on what the steps above and below it keep, so
        the rolling steps are completed once every step's arguments are
        written.
        """
        written: dict[str, Arguments] = {}
        for op_id in self.order:
            operation = self.operations[op_id]
            if op_id not in self.tools:
                continue
            capability, _schema = self.tools[op_id]
            built = self._arguments(operation)
            if built is None:
                continue
            written[op_id] = built
            self._check_bindings(operation, capability, built[1])
            if ROLLING_PARAMETER not in built[0]:
                self._build_step(op_id, *built)
        rolling = [op_id for op_id, (args, _) in written.items() if ROLLING_PARAMETER in args]
        if not rolling:
            return
        populations = self._populations(written)
        if populations is None:
            return
        for op_id in rolling:
            args, bindings = written[op_id]
            self._build_step(op_id, with_population(args, populations[op_id]), bindings)

    def _build_step(
        self, op_id: str, args: dict[str, JsonValue], bindings: dict[str, Binding]
    ) -> None:
        capability, schema = self.tools[op_id]
        self._check_arguments(self.operations[op_id], capability, schema, args, bindings)
        self.steps[op_id] = WorkflowStep(
            id=op_id,
            operation_id=op_id,
            tool=capability.tool,
            args=args,
            bindings=bindings,
            implementation_version=capability.implementation_version,
            input_schema_hash=schema.input_schema_hash,
            output_schema_hash=schema.output_schema_hash,
            expected_role_keys=self._expected_role_keys(op_id),
        )

    def _populations(
        self, written: Mapping[str, Arguments]
    ) -> dict[str, list[dict[str, Any]]] | None:
        """The population each rolling step carries, or `None` when it cannot be built.

        The steps that stand on one newest period carry one population: one
        entry for each chain from a terminal step, through one of them, down
        to a retrieval (`compiler.rolling.step_populations`). `None` when a
        step has no arguments, which an issue already explains, or when the
        chains keep no series of a retrieval, reported as
        `ROLLING_POPULATION_UNUSABLE`.
        """
        if set(written) != set(self.order):
            return None
        roots = {
            op_id: self._root(operation, written[op_id][0])
            for op_id in self.order
            if isinstance(operation := self.operations[op_id], RetrieveOperation)
        }
        frame_slots = {
            op_id: [
                declared.parameter
                for declared in self.tools[op_id][0].inputs
                if declared.artifact is ArtifactKind.FRAME_REF
            ]
            for op_id in self.order
            if op_id not in roots
        }
        terminal = set(self.translation.terminal)
        try:
            return step_populations(
                written,
                terminal=[op_id for op_id in self.order if op_id in terminal],
                roots=roots,
                frame_slots=frame_slots,
            )
        except (PopulationEmptyError, PopulationLevelColumnMissingError) as exc:
            retrieve = next(
                self.operations[op_id]
                for op_id, root in roots.items()
                if root.spec_id == exc.spec_id
            )
            self._issue(
                CompileIssueCode.ROLLING_POPULATION_UNUSABLE,
                f"a period of this plan follows the newest data, and {exc}",
                operation_id=retrieve.id,
                requirement_id=retrieve.requirement_id,
            )
            return None

    def _root(self, operation: RetrieveOperation, args: Mapping[str, JsonValue]) -> Root:
        """`operation`'s retrieval as a population reads it.

        Its columns are the ones a binding the compiler writes may name, which
        are every role the source resolves an entity under, the columns of the
        retrieval's own shape, and the place-type column when the retrieval
        asks for one. The frame's own spelling of them is known once it is
        retrieved (`compiler.frame_columns`).
        """
        source = self.sources[operation.id]
        requirement = self.requirements.get(operation.requirement_id)
        columns = {entity.role for entity in source.entities}
        columns.update(self._columns_of(operation.id) or ())
        if self._read_by_level_filter(operation.id):
            columns.add(type_column(PLACE_ROLE))
        named: dict[str, tuple[str, ...]] = {}
        if isinstance(requirement, DatasetRequirement):
            for entity in requirement.entities:
                for member in named_members(entity, source):
                    if member.ref not in named.get(member.role, ()):
                        named[member.role] = (*named.get(member.role, ()), member.ref)
        return Root(spec_id=source.spec_id, columns=frozenset(columns), named=named)

    def _expected_role_keys(self, op_id: str) -> tuple[str, ...]:
        """The prediction frozen onto `op_id`'s step: its own resolved shape's role columns.

        Sorted the way `Frame.build` sorts them, through
        `grammar.normalize_role_keys` -- an unsorted prediction would fail
        against a correct frame, which is exactly the drift the executor's
        `SHAPE_MISMATCH` check exists to catch. The default (`()`) applies
        to an operation with no resolved shape, which `_arguments` has
        already refused to build a step for unless an issue explains why.
        """
        resolved = self.shapes.get(op_id)
        if resolved is None:
            return ()
        return normalize_role_keys(role.column for role in resolved.columns)

    def _arguments(self, operation: Operation) -> Arguments | None:
        """Write the tool arguments for `operation`, or `None` if a dependency was refused."""
        writers: dict[type, Callable[[Any], Arguments | None]] = {
            RetrieveOperation: self._retrieve_arguments,
            AggregateOperation: self._aggregate_arguments,
            GrowthOperation: self._growth_arguments,
            CompareOperation: self._compare_arguments,
            SumOfOperation: self._sum_of_arguments,
            RankOperation: self._rank_arguments,
            ShareOfOperation: self._divided_arguments,
            QuotientOfOperation: self._quotient_of_arguments,
            JoinOperation: self._join_arguments,
            FilterOperation: self._filter_arguments,
        }
        return writers[type(operation)](operation)

    def _retrieve_arguments(self, operation: RetrieveOperation) -> Arguments | None:
        """Write `spec_id` and `stat_vars` for a family.

        `stat_vars` is the source's `stat_var_reading`: `sum` asks dfo-apis
        to sum the family's members into one series, `separate` keeps one
        per member. It sits outside the definition's identity, so it
        re-mints nothing.

        `breakdown` is the requirement's, as its code system and level: dfo-apis
        sums the family within each code rolled up to that level, so the frame
        carries one series per code under the family's own ref.

        `place_types` is written for a retrieval a place-level filter reads
        (`_read_by_level_filter`): it adds the `observationAbout_type`
        column that filter keeps one level by. Rendering only, so it
        re-mints nothing either.

        `without_party_totals` is written for every flow retrieval (a source
        with a flow role column among its stated or open entity axes). A flow
        side its source leaves open holds single parties and group totals of
        them, so the frame of a flow retrieval holds each open side's parties
        without their group totals, for every verb that reads it. It names a
        frame of its own, and re-mints no definition.

        No `scope` is written, even when the source resolves more entities
        than the requirement names. `get_data`'s contract is that any scope
        re-mints the definition under a new identity, and the executor's
        drift check refuses a manifest whose identity differs from the
        approved source's -- so a scoped retrieval would fail its own
        approval on every run. A source wider than its requirement is
        instead reported by `_check_source_width` where an operation would
        pool the surplus.
        """
        source = self.sources.get(operation.id)
        if source is None:
            return None
        args: dict[str, JsonValue] = {"spec_id": source.spec_id}
        if source.stat_var_reading is not None:
            args["stat_vars"] = source.stat_var_reading.value
        requirement = self.requirements.get(operation.requirement_id)
        if isinstance(requirement, DatasetRequirement) and requirement.breakdown is not None:
            args["breakdown"] = {
                "code_system": BREAKDOWN_CODE_SYSTEMS[requirement.breakdown.dimension],
                "level": requirement.breakdown.level,
            }
        if self._read_by_level_filter(operation.id):
            args["place_types"] = True
        if self._is_flow_source(source):
            args["without_party_totals"] = True
        return args, {}

    @staticmethod
    def _is_flow_source(source: SelectedSource) -> bool:
        """Whether `source` has a flow role column among its stated or open entity axes."""
        axes = {entity.role for entity in source.entities}
        axes.update(axis.column for axis in source.open_entity_axes or ())
        return bool(axes & FLOW_ROLE_COLUMNS.keys())

    def _read_by_level_filter(self, retrieve_id: str) -> bool:
        """Whether a filter keeping one level of place reads `retrieve_id`, directly or through other filters."""
        return any(
            isinstance(operation, FilterOperation)
            and keeps_one_level(operation.predicates)
            and self._through_filters(operation.input) == retrieve_id
            for operation in self.operations.values()
        )

    def _through_filters(self, op_id: str) -> str:
        """The operation `op_id` reads through filters alone: `op_id` itself when it is no filter."""
        seen = set[str]()
        while isinstance(read := self.operations.get(op_id), FilterOperation) and op_id not in seen:
            seen.add(op_id)
            op_id = read.input
        return op_id

    def _aggregate_arguments(self, operation: AggregateOperation) -> Arguments | None:
        resolved = self.shapes.get(operation.input)
        if resolved is None:
            return None
        args: dict[str, JsonValue] = {
            "aggregation": operation.reduction.value,
            "group_by": _group_columns(operation.group_by, resolved.columns),
        }
        return args, {"frame_ref": _frame_from(operation.input)}

    def _growth_arguments(self, operation: GrowthOperation) -> Arguments | None:
        columns = self._columns_of(operation.input)
        if columns is None:
            return None
        args: dict[str, JsonValue] = {
            "window_start": str(operation.window_start),
            "window_end": str(operation.window_end),
            "entity_by": columns,
        }
        if operation.change is not GrowthChange.RELATIVE:
            args["change"] = operation.change.value
        if operation.rolling is not None:
            args[ROLLING_PARAMETER] = window_overlay(operation.rolling)
        return args, {"frame_ref": _frame_from(operation.input)}

    def _compare_arguments(self, operation: CompareOperation) -> Arguments | None:
        return self._two_sided_arguments(operation.id, operation.left, operation.right)

    def _two_sided_arguments(self, op_id: str, left: Side, right: Side) -> Arguments | None:
        """The frames and entity bindings of a two-slice verb, `compare` or `sum_of`."""
        args: dict[str, JsonValue] = {}
        bindings = {"frame_ref": _frame_from(left.input)}
        if right.input != left.input:
            bindings["right_frame_ref"] = _frame_from(right.input)
        for parameter, side in (("left_entity", left), ("right_entity", right)):
            if side.entity is None:
                continue
            resolved = self._side_entity(op_id, side)
            if resolved is None:
                return None
            args[parameter] = {resolved.role: resolved.ref}
        return args, bindings

    def _sum_of_arguments(self, operation: SumOfOperation) -> Arguments | None:
        """`_compare_arguments`' sides, plus each side's unit pin (`_unit_pins`).

        analysis-mcp checks the two resolved units agree whether or not a
        side is pinned.
        """
        built = self._two_sided_arguments(operation.id, operation.left, operation.right)
        if built is None:
            return None
        args, bindings = built
        args.update(
            self._unit_pins((("left_unit", operation.left), ("right_unit", operation.right)))
        )
        return args, bindings

    def _unit_pins(self, sides: Sequence[tuple[str, Side]]) -> dict[str, JsonValue]:
        """Each side's unit parameter, pinned to the unit its fetched frame declares (`_frame_unit`).

        A pin narrows a side to the rows in that unit, so a frame holding
        several units is read in the one its source resolved. A side whose
        source declares no unit, whose frame unit cannot be known ahead, or
        whose frame is in a composed unit, is left unpinned.
        """
        pins: dict[str, JsonValue] = {}
        for parameter, side in sides:
            unit = self._side_unit(side.input)
            if unit is not None and not isinstance(unit, _PerRowUnit | _ComposedUnit):
                pins[parameter] = unit
        return pins

    def _divided_arguments(
        self, operation: ShareOfOperation | QuotientOfOperation
    ) -> Arguments | None:
        """The frames, entity bindings and keys of a dividing verb, `share_of` or `quotient_of`."""
        args: dict[str, JsonValue] = {}
        bindings = {"frame_ref": _frame_from(operation.numerator.input)}
        if operation.denominator.input != operation.numerator.input:
            bindings["denominator_frame_ref"] = _frame_from(operation.denominator.input)
        for parameter, side in (
            ("numerator_entity", operation.numerator),
            ("denominator_entity", operation.denominator),
        ):
            if side.entity is None:
                continue
            resolved = self._side_entity(operation.id, side)
            if resolved is None:
                return None
            args[parameter] = {resolved.role: resolved.ref}
        keyed = self._key_arguments(operation)
        if keyed is None:
            return None
        args.update(keyed)
        return args, bindings

    def _quotient_of_arguments(self, operation: QuotientOfOperation) -> Arguments | None:
        """`_divided_arguments`, plus each side's unit pin (`_unit_pins`).

        The two pins may differ: a quotient takes any pair of units and
        composes the result's from them.
        """
        built = self._divided_arguments(operation)
        if built is None:
            return None
        args, bindings = built
        args.update(
            self._unit_pins(
                (
                    ("numerator_unit", operation.numerator),
                    ("denominator_unit", operation.denominator),
                )
            )
        )
        return args, bindings

    def _key_arguments(
        self, operation: ShareOfOperation | QuotientOfOperation
    ) -> dict[str, JsonValue] | None:
        """`numerator_by`/`denominator_by`, keyed by name so canonicalisation cannot mispair them.

        Resolves the literal column each side's key (`_keyed_axes`) names
        from that side's own resolved shape; `None` when a lookup fails,
        which `_check_key_axes` has already reported as `SHARE_KEY_UNKNOWN`.
        An un-keyed numerator writes neither argument. A denominator read
        whole writes no `denominator_by`, so every key divides by its one
        total, the engine's own default.
        """
        numerator_axis, denominator_axis = _keyed_axes(operation)
        if numerator_axis is None:
            return {}
        numerator_role = self._key_column(operation.numerator, numerator_axis)
        if numerator_role is None:
            return None
        keyed: dict[str, JsonValue] = {"numerator_by": [numerator_role.column]}
        if denominator_axis is not None:
            denominator_role = self._key_column(operation.denominator, denominator_axis)
            if denominator_role is None:
                return None
            keyed["denominator_by"] = {numerator_role.column: denominator_role.column}
        return keyed

    def _key_column(self, side: Side, axis: str) -> ResolvedRole | None:
        """The column of `side`'s own frame that `axis` names, or `None` when it carries none."""
        resolved = self.shapes.get(side.input)
        return None if resolved is None else role_named(resolved.columns, axis)

    def _side_entity(self, op_id: str, side: Side) -> ResolvedEntity | None:
        """Resolve `side.entity` on the frame the side reads, or `None` if no argument can be built for it.

        Only called once `side.entity is not None`. Reports the issue that
        stops it (`_frame_entity`); a name no requirement behind the frame
        states is already `unknown_entity` from `_check_side`.
        """
        assert side.entity is not None
        found = self._frame_entity(op_id, side.input, side.entity)
        if isinstance(found, CompileIssue):
            self.issues.append(found)
            return None
        if found is None:
            return None
        if len(found) > 1:
            self._issue(
                CompileIssueCode.ENTITY_AMBIGUOUS,
                f"operation {op_id!r} names {side.entity!r} on a side, a set whose members "
                f"{[member.label for member in found]!r} are each a series of their own, and "
                "a side reads one; filter the frame to the set and aggregate it by period "
                "first",
                operation_id=op_id,
            )
            return None
        return found[0]

    def _rank_arguments(self, operation: RankOperation) -> Arguments | None:
        upstream = self.shapes.get(operation.input)
        if upstream is None:
            return None
        axis_column = _axis_column(upstream.columns, operation.axis)
        if axis_column is None:
            # Already reported: `RANK_AXIS_REQUIRED` (several roles, none
            # named) or `INPUT_SHAPE_UNSUPPORTED` (an axis naming nothing
            # the input carries).
            return None
        args: dict[str, JsonValue] = {
            "rank_by": [axis_column.column],
            "ascending": operation.direction is RankDirection.LOWEST_FIRST,
        }
        if operation.reduction is not None:
            args["aggregation"] = operation.reduction.value
        if operation.window_start is not None:
            args["window_start"] = str(operation.window_start)
        if operation.window_end is not None:
            args["window_end"] = str(operation.window_end)
        if operation.per_period:
            args["per_period"] = True
        if operation.limit is not None:
            args["limit"] = operation.limit
        if operation.rolling is not None:
            args[ROLLING_PARAMETER] = window_overlay(operation.rolling)
        return args, {"frame_ref": _frame_from(operation.input)}

    def _join_arguments(self, operation: JoinOperation) -> Arguments | None:
        left = self._columns_of(operation.left)
        right = self._columns_of(operation.right)
        if left is None or right is None:
            return None
        right_columns = set(right)
        join_keys = [column for column in left if column in right_columns]
        args: dict[str, JsonValue] = {"join_keys": join_keys}
        bindings = {
            "frame_ref": _frame_from(operation.left),
            "right_frame_ref": _frame_from(operation.right),
        }
        return args, bindings

    def _filter_arguments(self, operation: FilterOperation) -> Arguments | None:
        predicates: list[JsonValue] = []
        for predicate in operation.predicates:
            if isinstance(predicate, ValueThreshold):
                predicates.append(
                    {
                        "kind": "ordered_threshold",
                        "column": VALUE_COLUMN,
                        "op": predicate.op,
                        "value": predicate.value,
                    }
                )
            elif isinstance(predicate, PeriodThreshold):
                threshold: dict[str, JsonValue] = {
                    "kind": "ordered_threshold",
                    "column": TIME_PERIOD_COLUMN,
                    "op": predicate.op,
                    "value": str(predicate.year),
                }
                if predicate.rolling is not None:
                    threshold[ROLLING_PARAMETER] = rolling_ref(predicate.rolling)
                predicates.append(threshold)
            elif isinstance(predicate, PlaceLevel):
                predicates.append(
                    {
                        "kind": "place_level",
                        "column": type_column(PLACE_ROLE),
                        "level": predicate.level.value,
                    }
                )
            else:
                written = self._entity_in_argument(operation, predicate)
                if written is None:
                    return None
                predicates.append(written)
        args: dict[str, JsonValue] = {"predicates": predicates}
        if rolls(operation):
            # The population every rolling step carries is written in `_build_steps`.
            args[ROLLING_PARAMETER] = {}
        return args, {"frame_ref": _frame_from(operation.input)}

    def _entity_in_argument(
        self, operation: FilterOperation, predicate: EntityIn
    ) -> dict[str, JsonValue] | None:
        columns: set[str] = set()
        refs: list[str] = []
        for name in predicate.names:
            found = self._frame_entity(operation.id, operation.input, name, role=predicate.role)
            if not isinstance(found, tuple):
                # Unknown, ambiguous, unresolved or on no column of the
                # frame -- already reported by `_check_filter`.
                return None
            for member in found:
                columns.add(member.role)
                if member.ref not in refs:
                    refs.append(member.ref)
        if len(columns) > 1:
            self._issue(
                CompileIssueCode.ENTITY_AMBIGUOUS,
                f"operation {operation.id!r} names entities resolving to different role columns "
                f"{sorted(columns)!r}",
                operation_id=operation.id,
            )
            return None
        (column,) = columns
        return {"kind": "category_match", "column": column, "values": refs}

    def _frame_entity(
        self, op_id: str, input_id: str, name: str, *, role: str | None = None
    ) -> tuple[ResolvedEntity, ...] | CompileIssue | None:
        """Resolve `name` to the entities it names on the frame at `input_id`, one role column.

        Each requirement behind the frame's role columns that states `name`
        (`_named_behind`), under `role` when the operation gives one,
        resolves it against its own source (`_stated_entity`): one entity,
        or a named set's members. So a name on a keyed `share_of`
        or `quotient_of` result resolves on that result's key column, through
        the requirement and source the column came from.

        Returns the issue that stops it, for the caller to report once:
        `entity_ambiguous` when a requirement states `name` under several
        roles and `role` does not narrow it to one, or two requirements
        resolve it differently; `entity_unresolved` when the source does not
        resolve it; `entity_not_on_frame` when it resolves under a column the
        frame does not carry, as a donor does on a result keyed by recipient.
        `None` when no requirement behind the frame states `name`, which
        `_check_names_known` reports as `unknown_entity`, or when the frame's
        requirements cannot be read.
        """
        resolved = self.shapes.get(input_id)
        found: dict[tuple[tuple[str, str], ...], tuple[ResolvedEntity, ...]] = {}
        for source, requirement in self._named_behind(input_id) or ():
            members = self._stated_entity(op_id, name, role, source, requirement)
            if isinstance(members, CompileIssue):
                return members
            if members is not None:
                found[tuple(sorted((one.role, one.ref) for one in members))] = members
        if resolved is None or not found:
            return None
        if len(found) > 1:
            return CompileIssue(
                code=CompileIssueCode.ENTITY_AMBIGUOUS,
                reason=f"operation {op_id!r} names entity {name!r}, which the requirements "
                f"behind {input_id!r} resolve differently {sorted(found)!r}; say which",
                operation_id=op_id,
            )
        (members,) = found.values()
        column = members[0].role
        carried = [one.column for one in resolved.columns]
        if column not in carried:
            return CompileIssue(
                code=CompileIssueCode.ENTITY_NOT_ON_FRAME,
                reason=f"operation {op_id!r} names entity {name!r}, which its source holds "
                f"under column {column!r}, and the frame it reads from {input_id!r} "
                f"carries {carried!r}: that frame holds no series per {column!r} to "
                "pick one from",
                operation_id=op_id,
            )
        return members

    def _stated_entity(
        self,
        op_id: str,
        name: str,
        role: str | None,
        source: SelectedSource,
        requirement: DatasetRequirement,
    ) -> tuple[ResolvedEntity, ...] | CompileIssue | None:
        """The entities of `source` that the entity `requirement` states as `name` names, for `_frame_entity`.

        One entity, or a named set's members on its role column
        (`compiler.evidence.named_members`). `None` when `requirement`
        states no such entity, under `role` when one is given.
        """
        candidates = _entity_candidates(requirement, name, role)
        if not candidates:
            return None
        if len(candidates) > 1:
            return CompileIssue(
                code=CompileIssueCode.ENTITY_AMBIGUOUS,
                reason=f"operation {op_id!r} names entity {name!r}, which requirement "
                f"{requirement.id!r} names under several roles "
                f"{[c.role for c in candidates]!r}; say which",
                operation_id=op_id,
                requirement_id=requirement.id,
            )
        members = named_members(candidates[0], source)
        if not members:
            return CompileIssue(
                code=CompileIssueCode.ENTITY_UNRESOLVED,
                reason=f"operation {op_id!r} names entity {name!r}, which source "
                f"{source.spec_id} does not resolve, so no entity argument can be written",
                operation_id=op_id,
            )
        return members

    def _check_bindings(
        self, operation: Operation, capability: ToolCapability, bindings: Mapping[str, Binding]
    ) -> None:
        declared = {i.parameter: i.artifact for i in capability.inputs}
        for parameter, binding in bindings.items():
            producer = self.tools.get(binding.from_step)
            produced = {o.artifact for o in producer[0].outputs} if producer else set()
            expected = declared.get(parameter)
            if expected is not binding.output or (producer and binding.output not in produced):
                self._issue(
                    CompileIssueCode.ARTIFACT_MISMATCH,
                    f"operation {operation.id!r} binds {parameter!r} to a {binding.output.value} "
                    f"from {binding.from_step!r}; the tool takes {expected and expected.value!r} "
                    f"there and the producer yields {sorted(a.value for a in produced)!r}",
                    operation_id=operation.id,
                    tool=capability.tool,
                )

    def _check_arguments(
        self,
        operation: Operation,
        capability: ToolCapability,
        schema: ToolSchema,
        args: Mapping[str, JsonValue],
        bindings: Mapping[str, Binding],
    ) -> None:
        """Report `BAD_ARGS` for arguments the tool refuses.

        The arguments are checked against the tool's JSON Schema first and,
        once they fit it, for an analysis-mcp verb, against the verb's own
        op-spec checks (`compiler.arguments.op_spec_errors`). Two entity
        filters naming one entity, one under its role and one under none,
        both write the same `category_match`, which analysis-mcp refuses as
        a predicate stated twice.
        """
        undeclared = undeclared_parameters(bindings, schema.parameters)
        for name in undeclared:
            self._issue(
                CompileIssueCode.BAD_ARGS,
                f"{capability.tool.name!r} takes no argument named {name!r}",
                operation_id=operation.id,
                tool=capability.tool,
            )
        messages = literal_argument_errors(arguments=args, schema=schema.parameters, bound=bindings)
        if not (undeclared or messages) and capability.tool.server == "analysis-mcp":
            messages = [
                f"operation {operation.id!r} writes arguments {capability.tool.name!r} refuses: "
                f"{message}"
                for message in op_spec_errors(
                    tool=capability.tool.name, arguments=args, bound=bindings
                )
            ]
        for message in messages:
            self._issue(
                CompileIssueCode.BAD_ARGS, message, operation_id=operation.id, tool=capability.tool
            )

    # -- helpers ---------------------------------------------------------------------

    def _issue(
        self,
        code: CompileIssueCode,
        reason: str,
        *,
        operation_id: str | None = None,
        requirement_id: str | None = None,
        tool: ToolId | None = None,
        finding: EvidenceIssueCode | None = None,
    ) -> None:
        self.issues.append(
            CompileIssue(
                code=code,
                reason=reason,
                operation_id=operation_id,
                requirement_id=requirement_id,
                tool=tool,
                finding=finding,
            )
        )

    def _retrieves(self) -> Iterator[RetrieveOperation]:
        return (op for op in self.operations.values() if isinstance(op, RetrieveOperation))

    def _root_requirement_id(self, op_id: str) -> str | None:
        """The one dataset requirement `op_id`'s shape ultimately rests on, or `None` if it rests on more than one.

        Reads `self.shapes[op_id].shape.roots`, which every transform in
        `planner.shapes` carries or merges forward, so this resolves for a
        single-input chain and for a two-sided operation whose sides happen
        to rest on the one retrieval.
        """
        resolved = self.shapes.get(op_id)
        if resolved is None or len(resolved.shape.roots) != 1:
            return None
        (requirement_id,) = resolved.shape.roots
        return requirement_id

    def _requirement_upstream(self, op_id: str) -> DatasetRequirement | None:
        requirement_id = self._root_requirement_id(op_id)
        if requirement_id is None:
            return None
        requirement = self.requirements.get(requirement_id)
        return requirement if isinstance(requirement, DatasetRequirement) else None

    def _source_upstream(self, op_id: str) -> SelectedSource | None:
        requirement_id = self._root_requirement_id(op_id)
        return self._source_of(requirement_id) if requirement_id is not None else None

    def _root_requirement_ids(self, op_id: str) -> tuple[str, ...]:
        """Every dataset requirement `op_id`'s shape rests on, or none if it has no shape."""
        resolved = self.shapes.get(op_id)
        return resolved.shape.roots if resolved is not None else ()

    def _source_of(self, requirement_id: str) -> SelectedSource | None:
        retrieve_id = self._retrieve_by_requirement.get(requirement_id)
        return self.sources.get(retrieve_id) if retrieve_id is not None else None

    def _upstream(self, op_id: str) -> Upstream | None:
        """The source and dataset requirement a chain rests on, or `None` if either is missing.

        Either can be missing only once an issue has been reported for it:
        the retrieval was refused a source, the requirement it carries out
        is not a dataset one, or `op_id`'s shape rests on more than one
        retrieval (a compare, sum_of, share_of, quotient_of or join reading two sides
        that do not coincide).
        """
        source = self._source_upstream(op_id)
        requirement = self._requirement_upstream(op_id)
        if source is None or requirement is None:
            return None
        return source, requirement

    def _columns_of(self, op_id: str) -> list[str] | None:
        """The literal columns `op_id`'s own resolved shape carries, or `None` when it has none.

        Reads `self.shapes[op_id].columns` directly -- the operation's own
        immediate shape, not walked back to a root retrieval -- so a rank or
        a growth reading a keyed `share_of` writes that share's own columns
        rather than failing for want of a single retrieval behind it.
        """
        resolved = self.shapes.get(op_id)
        return None if resolved is None else [role.column for role in resolved.columns]

    def _side_identity(self, side: Side) -> tuple[str, str | None]:
        """What a side is really of: the retrieval its chain rests on, and its effective entity.

        Two sides with the same identity set a series against itself however
        their inputs are named -- `retrieve_a` against `filter(retrieve_a)`.
        The effective entity is the side's own, or the requirement's when it
        names exactly one; a chain resting on no single retrieval identifies
        by its input alone.
        """
        requirement_id = self._root_requirement_id(side.input)
        retrieve_id = (
            self._retrieve_by_requirement.get(requirement_id)
            if requirement_id is not None
            else None
        )
        if retrieve_id is None:
            return side.input, side.entity
        entity = side.entity
        requirement = self.requirements.get(requirement_id)
        if entity is None and isinstance(requirement, DatasetRequirement):
            names = {one.name for one in requirement.entities}
            if len(names) == 1:
                (entity,) = names
        return retrieve_id, entity

    def _topological_order(self) -> list[str]:
        """Order operations so every input precedes its reader, reporting any cycle once.

        A post-order walk driven, and tie-broken, by the canonical order
        `_index_operations` fixed, so the result depends on which operations
        the translation names and not on the order it listed them in.
        """
        order: list[str] = []
        state: dict[str, int] = {}  # 1 = on the current path, 2 = placed

        def visit(op_id: str, path: list[str]) -> None:
            if state.get(op_id) == 2:
                return
            if state.get(op_id) == 1:
                self._issue(
                    CompileIssueCode.CYCLE,
                    f"operations read each other in a cycle: {[*path, op_id]!r}",
                    operation_id=op_id,
                )
                return
            state[op_id] = 1
            inputs = [i for i in operation_inputs(self.operations[op_id]) if i in self.operations]
            for input_id in sorted(inputs, key=lambda i: _canonical_key(self.operations[i])):
                visit(input_id, [*path, op_id])
            state[op_id] = 2
            order.append(op_id)

        for op_id in self.operations:
            visit(op_id, [])
        return order

    def _reachable_from_terminals(self) -> set[str]:
        reachable: set[str] = set()
        pending = [t for t in self.translation.terminal if t in self.operations]
        while pending:
            op_id = pending.pop()
            if op_id in reachable:
                continue
            reachable.add(op_id)
            pending.extend(
                i for i in operation_inputs(self.operations[op_id]) if i in self.operations
            )
        return reachable


def _entity_candidates(
    requirement: DatasetRequirement, name: str, role: str | None
) -> list[Entity]:
    """The entities of `requirement` called `name`, under `role` when one is given (`_stated_under`)."""
    return [
        entity
        for entity in requirement.entities
        if entity.name == name and _stated_under(entity, role)
    ]


def _stated_under(entity: Entity, role: str | None) -> bool:
    """Whether the phrase `role` names the dimension `entity` sits on; any does when `role` is `None`.

    The rule is `planner.shapes.roles_named`'s: a phrase names the role it
    equals, and a place word names the place dimension, whose entities
    state no role or a place phrase.
    """
    if role is None or entity.role == role:
        return True
    return across_party(role) == "place" and place_role(entity.role)


def _axis_phrases(columns: Sequence[ResolvedRole]) -> list[str]:
    """The phrase the plan writes to name each of `columns`, sorted, for an issue's reason.

    A column's phrase is its role. A place column whose entities state no
    role carries its literal column as `role`, which `Plan` refuses as an
    axis, so it reads as `PLACE_PHRASE`.
    """
    return sorted(
        {
            PLACE_PHRASE if column.place and column.role == column.column else column.role
            for column in columns
            if column.role is not None
        }
    )


def _part_words(part: JsonValue | _StatVarPart) -> str:
    """One part of a composed unit, in words for an issue's reason."""
    if isinstance(part, _StatVarPart):
        return f"StatVar {part.stat_var!r}"
    return f"unit {part!r}"


def _keyed_axes(
    operation: ShareOfOperation | QuotientOfOperation,
) -> tuple[str | None, str | None]:
    """The role phrase each side of a dividing verb is keyed by: the numerator's, then the denominator's.

    A keyed side holds one series per member of the column its phrase names
    and divides member by member, so it reads every member of that column and
    pools none of them: several named entities there, an open axis, or
    members the requirement does not name are each one row of the result. On
    every other column the side is a whole side like any other. `None` is a
    side read whole on every column, which has to be one series.

    Both verbs key the numerator by `axis`. A quotient keys its denominator
    by `denominator_axis`, the denominator's own phrase for the same
    entities; without one, every key divides by the denominator's one total.
    A share keys its denominator by `axis` when it divides `per_axis`; a
    `pooled` share divides every key by one total.
    """
    if isinstance(operation, QuotientOfOperation):
        return operation.axis, operation.denominator_axis
    per_axis = operation.denominator_scope is DenominatorScope.PER_AXIS
    return operation.axis, operation.axis if per_axis else None


def _frame_unit(source: SelectedSource) -> FrameUnit:
    """The unit a frame fetched from `source` declares in its `unit` cells.

    A definition that binds its unit by request (`facets.unit.request` in its
    identity) is converted on fetch, so its frame declares that request
    object; one whose request names its currency by entity role resolves a
    currency per row, which no single pin covers, so it is
    `_PerRowUnit.PER_ROW`, as is a source whose unit varies per series
    (`unit_per_series`): analysis-mcp reads each side's unit from its own rows
    once sliced. Any other definition serves its source's own unit ref, `None`
    when the source declares no unit.
    """
    facets = (source.identity or {}).get("facets")
    unit = facets.get("unit") if isinstance(facets, dict) else None
    if not isinstance(unit, dict) or "request" not in unit:
        return _PerRowUnit.PER_ROW if source.unit_per_series else source.unit_ref
    try:
        converted = Unit.model_validate(unit["request"])
    except ValidationError:
        return _PerRowUnit.PER_ROW
    return converted.model_dump(exclude_none=True)


def _denoted_unit(source: SelectedSource, unit: JsonValue) -> Unit | None:
    """The currency and price basis a frame unit `_frame_unit` returns denotes; `None` when unknown.

    A request object states both. A unit ref states its currency by its
    structure (`compiler.evidence.read_unit_dcid`), and its price basis
    either the same way or, for a bare currency such as `CurrencyUSD`,
    through the StatVar shape's `measurement_qualifier`: `Nominal` is
    current prices, which is how dfo-apis decides a request for current
    prices is natively satisfied.
    """
    if isinstance(unit, dict):
        try:
            return Unit.model_validate(unit)
        except ValidationError:
            return None
    if not isinstance(unit, str):
        return None
    reading = read_unit_dcid(unit)
    prices, base_year = reading.prices, reading.base_year
    if prices is None:
        shape = (source.identity or {}).get("shape")
        qualifier = shape.get("measurement_qualifier") if isinstance(shape, dict) else None
        prices = "current" if qualifier == "Nominal" else None
    if reading.currency is None or prices is None:
        return None
    try:
        return Unit(currency=reading.currency, prices=prices, base_year=base_year)
    except ValidationError:
        return None


class _SideUnit(NamedTuple):
    """One side of a pair, as `_report_source_units` reads its unit."""

    source: SelectedSource
    unit: JsonValue
    denoted: Unit | None
    reading: UnitReading
    requirement: DatasetRequirement | None

    def words(self) -> str:
        """The side's unit in words for an issue's reason: what it denotes, else
        what its source's unit states, else `_unit_cell`'s words for it."""
        if self.denoted is None and self.reading.prices is not None:
            return _reading_words(self.reading)
        return _unit_cell(self.unit, self.denoted)


def _unit_reading(source: SelectedSource, denoted: Unit | None) -> UnitReading:
    """What a side's frame unit states: what it denotes (`_denoted_unit`), else what
    `source`'s own unit states, as `check_evidence` reads it."""
    if denoted is not None:
        return UnitReading(denoted.currency, denoted.prices, denoted.base_year)
    return read_source_unit(source)


def _reading_words(reading: UnitReading) -> str:
    """A unit reading that states a price basis, in words for an issue's reason."""
    currency = reading.currency or "an unstated currency"
    if reading.prices == "current":
        return f"{currency} at current prices"
    if reading.base_year is not None:
        return f"{currency} at constant {reading.base_year} prices"
    return f"{currency} at constant prices of an unstated base year"


def _denoted_words(unit: Unit) -> str:
    """`unit` in words for an issue's reason."""
    if unit.base_year is not None:
        return f"{unit.currency} at constant {unit.base_year} prices"
    return f"{unit.currency} at {unit.prices} prices"


def _bound_unit_words(unit: JsonValue, denoted: Unit | None) -> str:
    """A bound frame unit `_frame_unit` returns, named exactly, for an issue's reason."""
    return _denoted_words(denoted) if denoted is not None else f"unit {unit!r}"


def _unit_cell(unit: JsonValue, denoted: Unit | None) -> str:
    """A frame unit `_frame_unit` returns, in words for an issue's reason."""
    if denoted is not None:
        return _denoted_words(denoted)
    if isinstance(unit, dict):
        return "a unit requested by role"
    return "its source's own unit, whose price basis the source does not state"


def _family_overlap_issue(
    left: SelectedSource, right: SelectedSource
) -> tuple[CompileIssueCode, str] | None:
    """The sum issue two StatVar families raise, judged from their identities alone."""
    definitions = f"the StatVar families of definitions {left.spec_id} and {right.spec_id}"
    match family_overlap(left, right):
        case FamilyOverlap.DISJOINT:
            return None
        case FamilyOverlap.NESTED:
            return (
                CompileIssueCode.SUM_SIDES_OVERLAP,
                f"{definitions}, one of which holds the other",
            )
        case FamilyOverlap.UNDECIDED:
            return CompileIssueCode.SUM_OVERLAP_UNDECIDED, definitions


def _unit_words(requirement: DatasetRequirement) -> str:
    unit = requirement.unit
    if unit is None:
        return "no unit stated"
    if unit.base_year is None:
        return f"{unit.prices} {unit.currency}"
    return f"constant {unit.base_year} {unit.currency}"


def _frame_from(op_id: str) -> Binding:
    return Binding(from_step=op_id, output=ArtifactKind.FRAME_REF)


def _canonical_key(operation: Operation) -> tuple[str, str]:
    """The order operations compile in: by requirement, then by id, which is unique once indexed."""
    return operation.requirement_id, operation.id


def _warning_key(warning: EvidenceIssue) -> tuple[str, str, str, str]:
    return warning.requirement_id, warning.code.value, warning.detail, warning.model_dump_json()


def _names_on_column(
    requirement: DatasetRequirement, source: SelectedSource, column: str, kept: set[str]
) -> list[str]:
    """The names `requirement` states that name a member under `column` with a label in `kept`.

    `kept` holds the case-folded labels still reaching the reader. One
    entity per role column, a donor and its recipient say, is one series,
    which a whole side reads as it is. Two on one column are two series a
    whole side would pool. A set's name counts once
    (`compiler.evidence.named_members`); its members still reaching the
    reader are `_labels_on_column`'s to count.
    """
    return [
        entity.name
        for entity in requirement.entities
        if any(
            member.role == column and member.label.casefold() in kept
            for member in named_members(entity, source)
        )
    ]


def _labels_on_column(source: SelectedSource, column: str, kept: set[str]) -> list[str]:
    """The labels `source` resolves under `column` that are in case-folded `kept`, duplicates once."""
    labels: dict[str, str] = {}
    for entity in source.entities:
        if entity.role == column and entity.label.casefold() in kept:
            labels.setdefault(entity.label.casefold(), entity.label)
    return list(labels.values())


def _surplus_labels(requirement: DatasetRequirement, source: SelectedSource) -> list[str]:
    """The labels of the entities `source` resolves under `requirement`'s role columns that it does not name.

    Named is by ref (`compiler.evidence.named_refs`), so a set's members are
    named by the set. Distinct case-folded, in the source's order. Entities
    under the other role columns -- the creditor alongside the debtor -- are
    left out, as `_role_columns` leaves their columns out: pooling a debtor
    with its creditor pools nothing the plan did not state, and naming the
    creditor to narrow it would be `unknown_entity`.
    """
    columns = set(_role_columns(source, requirement))
    named = named_refs(requirement.entities, source)
    labels: dict[str, str] = {}
    for entity in source.entities:
        if entity.role in columns and entity.ref not in named:
            labels.setdefault(entity.label.casefold(), entity.label)
    return list(labels.values())


def _pooled_sides(operation: Operation) -> tuple[tuple[str, str | None], ...]:
    """Each input `operation` folds into one figure, with the role phrase it keys that input by.

    A period-only aggregate pools its one input, and a two-sided operation
    whichever of its sides takes no `entity`. A side of a dividing verb keyed
    by an axis (`_keyed_axes`) keeps one row per member of that column and
    pools the rest, so its phrase rides along; `None` pools every column.

    A rank and a growth write the requirement's role columns as `rank_by`
    and `entity_by`, an aggregate grouping by entity writes them as
    `group_by`, and a filter and a join keep the rows: each leaves one row
    per entity, so a surplus entity stays visible to the reader and none of
    them is listed. A pooled total hides it behind a figure presented as the
    asked entity's. A rank over a wider set still changes an entity's
    position, which this does not cover.
    """
    if isinstance(operation, AggregateOperation):
        return ((operation.input, None),) if GroupBy.ENTITY not in operation.group_by else ()
    if isinstance(operation, CompareOperation | SumOfOperation):
        sides = ((operation.left, None), (operation.right, None))
    elif isinstance(operation, ShareOfOperation | QuotientOfOperation):
        numerator_axis, denominator_axis = _keyed_axes(operation)
        sides = ((operation.numerator, numerator_axis), (operation.denominator, denominator_axis))
    else:
        return ()
    return tuple((side.input, keyed) for side, keyed in sides if side.entity is None)


def _narrowed_by_filter(
    operation: FilterOperation, role: ResolvedRole, source: SelectedSource, retained: set[str]
) -> set[str]:
    """`retained`, narrowed by `operation`'s own `EntityIn` predicates that name `role`.

    A predicate stating its own `role` narrows `retained` only when that
    phrase names `role` (`planner.shapes.roles_named`); an unqualified one narrows it only for the names
    that resolve onto `role.column` -- a column-for-column mirror of
    `compiler.shapes._resolved_narrowed_roles`, which only proves *that* a
    predicate narrows a column, not *by how much*. A predicate naming
    nothing that resolves here leaves `retained` untouched.
    """
    for predicate in operation.predicates:
        if not isinstance(predicate, EntityIn):
            continue
        if predicate.role is not None:
            if not roles_named((role,), predicate.role):
                continue
            named = {name.casefold() for name in predicate.names} | _member_labels(
                predicate.names, predicate.role, role, source
            )
        else:
            named = _member_labels(predicate.names, None, role, source)
            if not named:
                continue
        retained = retained & named
    return retained


def _member_labels(
    names: Sequence[str], phrase: str | None, role: ResolvedRole, source: SelectedSource
) -> set[str]:
    """The case-folded labels of the members `names` name on `role`'s column (`named_members`)."""
    return {
        member.label.casefold()
        for name in names
        for member in named_members(Entity(name=name, role=phrase), source)
        if member.role == role.column
    }


def _narrowed_by_side(
    side: Side, role: ResolvedRole, source: SelectedSource, retained: set[str]
) -> set[str]:
    """`retained`, narrowed to the entity `side` names when `source` resolves it onto `role`'s column."""
    if side.entity is None:
        return retained
    named = _member_labels((side.entity,), None, role, source)
    return retained & named if named else retained


def _axis_column(columns: Sequence[ResolvedRole], axis: str | None) -> ResolvedRole | None:
    """The literal column `axis` names among `columns`, or the lone one when `axis` is `None`.

    A column-for-column mirror of `planner.shapes._select_role` over the
    compiler's own `ResolvedRole`s, by the same `planner.shapes.role_named`:
    `None` when `axis` does not resolve,
    which every caller reads as "already reported" -- `_check_rank`'s
    `RANK_AXIS_REQUIRED` when several columns and none named, or
    `planner.shapes.rejects`'s own `INPUT_SHAPE_UNSUPPORTED` when `axis`
    names nothing the input carries.
    """
    if axis is None:
        return columns[0] if len(columns) == 1 else None
    return role_named(columns, axis)


def _role_columns(source: SelectedSource, requirement: DatasetRequirement) -> list[str]:
    """The role columns of `source` that `requirement`'s entities name members on, in the source's order.

    `compiler.shapes.shapes_for` computes this same set once, into a fresh
    retrieval's own `ResolvedShape.columns`, for every argument an operation
    writes off its shape (`compile._columns_of`). This copy remains for
    `_surplus_labels`, which `_check_source_width` reads
    off the root `(source, requirement)` pair directly.

    A source often resolves entities under roles the requirement never
    asked about -- a creditor alongside the debtor -- and grouping, ranking
    or joining by those would split the answer along lines the plan did not
    state. Falls back to every role column when the requirement resolves
    none, which `entity_unresolved` reports separately.
    """
    every: list[str] = []
    for entity in source.entities:
        if entity.role not in every:
            every.append(entity.role)
    wanted = {
        member.role for entity in requirement.entities for member in named_members(entity, source)
    }
    return [column for column in every if column in wanted] or every


def _group_columns(group_by: Sequence[GroupBy], roles: Sequence[ResolvedRole]) -> list[str]:
    """`group_by` written as literal columns: every one of `roles` but the open ones for
    `ENTITY`, the breakdown's alone for `BREAKDOWN`, the frame's own period column for
    `PERIOD`. An open column is left out of every group, so the aggregate sums it."""
    columns: list[str] = []
    for key in group_by:
        if key is GroupBy.PERIOD:
            new = [TIME_PERIOD_COLUMN]
        elif key is GroupBy.BREAKDOWN:
            new = [role.column for role in roles if role.breakdown]
        else:
            new = [role.column for role in roles if not role.open]
        columns.extend(column for column in new if column not in columns)
    return columns

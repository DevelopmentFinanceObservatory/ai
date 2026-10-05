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

A retrieval is a legal terminal only under `planner.plan.AnswerMode.SERIES`:
its frame answers a question that asks to see a level or a trend for named
entities, and it carries one series per entity for a chart to draw. That gate
is the plan's own -- `Plan._references_resolve` rejects an unread dataset
requirement under `AnswerMode.COMPUTED` before a translation naming it as
terminal ever reaches here, so this module states no `answer_mode` check of
its own; its operation graph mirrors the requirement graph the plan already
validated. Nothing here reads a claim.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from enum import StrEnum
from typing import Any

from common.place_type import PLACE_ROLE, type_column
from dfo_frame_grammar.grammar import TIME_PERIOD_COLUMN, VALUE_COLUMN, normalize_role_keys
from planner.operations import (
    AggregateOperation,
    CompareOperation,
    FilterOperation,
    GroupBy,
    GrowthOperation,
    JoinOperation,
    Operation,
    RankOperation,
    RetrieveOperation,
    ShareOfOperation,
    Side,
    SumOfOperation,
    Translation,
)
from planner.parties import flow_side
from planner.plan import DatasetRequirement, Entity, Plan, Unit, keeps_one_level
from planner.proof import mismatches
from planner.records import (
    DenominatorScope,
    EntityIn,
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
from planner.shapes import ValueKind
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
    check_evidence,
    errors,
    evidence_from_source,
    read_unit_dcid,
    resolve_entity,
)
from compiler.shapes import BREAKDOWN_CODE_SYSTEMS, ResolvedRole, ResolvedShape, shapes_for
from compiler.sources import (
    FamilyOverlap,
    entity_scopes_disjoint,
    family_overlap,
    posted_stat_vars,
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
    RANK_AXIS_REQUIRED = "rank_axis_required"
    SHARE_AXIS_REQUIRED = "share_axis_required"
    SHARE_KEY_UNKNOWN = "share_key_unknown"
    CYCLE = "cycle"
    UNKNOWN_TERMINAL = "unknown_terminal"
    TERMINAL_IS_READ = "terminal_is_read"
    OPERATION_UNREACHABLE = "operation_unreachable"
    SOURCE_UNSELECTED = "source_unselected"
    SOURCE_NOT_DISCOVERED_FOR_REQUIREMENT = "source_not_discovered_for_requirement"
    SOURCE_PIN_VIOLATED = "source_pin_violated"
    SOURCE_UNUSABLE = "source_unusable"
    SOURCE_WIDER_THAN_REQUIREMENT = "source_wider_than_requirement"
    REQUIREMENT_UNMEETABLE = "requirement_unmeetable"
    WINDOW_OUTSIDE_PERIOD = "window_outside_period"
    ENTITY_REQUIRED = "entity_required"
    UNKNOWN_ENTITY = "unknown_entity"
    ENTITY_AMBIGUOUS = "entity_ambiguous"
    ENTITY_UNRESOLVED = "entity_unresolved"
    DEGENERATE_SIDES = "degenerate_sides"
    SUM_SIDES_OVERLAP = "sum_sides_overlap"
    SUM_OVERLAP_UNDECIDED = "sum_overlap_undecided"
    SUM_UNITS_DIFFER = "sum_units_differ"
    FRAME_UNITS_DIFFER = "frame_units_differ"
    FRAME_UNIT_FORMS_DIFFER = "frame_unit_forms_differ"
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
    """One reason a translation does not compile, and what it is about."""

    code: CompileIssueCode
    reason: str
    operation_id: str | None = None
    requirement_id: str | None = None
    tool: ToolId | None = None


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
    return CompileResult(workflow=build.workflow(warnings), warnings=warnings)


# What `_Build` writes for one step: its literal arguments and its bindings.
Arguments = tuple[dict[str, JsonValue], dict[str, Binding]]

# What a single-input chain rests on: the source discovery selected for its
# retrieval, and the dataset requirement that retrieval carries out.
Upstream = tuple[SelectedSource, DatasetRequirement]

# Operations that read exactly one other operation, and so can be walked
# through to find the dataset requirement (or source) a chain rests on.
_SINGLE_INPUT = (AggregateOperation, GrowthOperation, RankOperation, FilterOperation)


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
            for input_id in _inputs_of(operation):
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
        readers = self._readers()
        for terminal in self.translation.terminal:
            if terminal not in known:
                self._issue(
                    CompileIssueCode.UNKNOWN_TERMINAL,
                    f"terminal {terminal!r} is not an operation",
                    operation_id=terminal,
                )
            elif readers.get(terminal):
                # A terminal is the answer; a frame another operation reads is
                # an intermediate. One that is both leaks into the run's results.
                self._issue(
                    CompileIssueCode.TERMINAL_IS_READ,
                    f"terminal {terminal!r} is read by {readers[terminal]!r}; a terminal is the "
                    "answer and an operation another reads is an intermediate, so its frame "
                    "would leak into the run's results",
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
            self._check_breakdown(operation, requirement, source)
            findings = check_evidence(requirement, evidence_from_source(source))
            for finding in errors(findings):
                self._issue(
                    CompileIssueCode.SOURCE_UNUSABLE,
                    f"source {source.spec_id} cannot serve requirement {requirement.id!r}: "
                    f"{finding.detail}",
                    operation_id=operation.id,
                    requirement_id=requirement.id,
                )
            self.warnings.extend(f for f in findings if f not in errors(findings))

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
                self._check_two_sided(operation.id, operation.numerator, operation.denominator)
                self._check_share_of_axis(operation)
            elif isinstance(operation, JoinOperation):
                self._check_join(operation)
            else:
                self._check_filter(operation)

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

        `SUM_SIDES_OVERLAP` with what both read when they share a definition,
        a member digest or a listed StatVar, or one family holds the other.
        Two sources that both list StatVars are compared by those lists
        alone. A side listing none is a family `search` states by digest,
        decided from the two identities (`compiler.sources.family_overlap`):
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
        if shared := set(left.stat_vars) & set(right.stat_vars):
            return CompileIssueCode.SUM_SIDES_OVERLAP, f"StatVars both elected ({len(shared)})"
        if left.stat_vars and right.stat_vars:
            return None
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
            names = sorted(role.role for role in upstream.shape.roles if role.role is not None)
            self._issue(
                CompileIssueCode.RANK_AXIS_REQUIRED,
                f"operation {operation.id!r} ranks a frame carrying several roles {names!r} and "
                "names no axis; say which one to rank on",
                operation_id=operation.id,
            )

    def _check_share_of_axis(self, operation: ShareOfOperation) -> None:
        """Report `SHARE_KEY_UNKNOWN` when `axis` cannot resolve to a literal column on a side.

        `planner.shapes.rejects` already refuses this at the shape level
        (`INPUT_SHAPE_UNSUPPORTED`, by plan-vocabulary role phrase), so this
        restates the same fact at the column-resolution level the argument
        writer itself depends on -- the same "verified, not merely
        predicted" discipline the shape check applies elsewhere.

        With no `axis` at all, `_check_share_of_needs_axis` takes over: an
        un-keyed share still has to make sense on its own terms.
        """
        if operation.axis is None:
            self._check_share_of_needs_axis(operation)
            return
        numerator = self.shapes.get(operation.numerator.input)
        if numerator is not None and _matching_role(numerator.columns, operation.axis) is None:
            roles = sorted(r.role for r in numerator.columns if r.role is not None)
            self._issue(
                CompileIssueCode.SHARE_KEY_UNKNOWN,
                f"operation {operation.id!r} keys its numerator by axis {operation.axis!r}, "
                f"which the numerator's own frame does not carry; its roles are {roles!r}",
                operation_id=operation.id,
            )
        if operation.denominator_scope is DenominatorScope.PER_AXIS:
            denominator = self.shapes.get(operation.denominator.input)
            if (
                denominator is not None
                and _matching_role(denominator.columns, operation.axis) is None
            ):
                roles = sorted(r.role for r in denominator.columns if r.role is not None)
                self._issue(
                    CompileIssueCode.SHARE_KEY_UNKNOWN,
                    f"operation {operation.id!r} divides per axis {operation.axis!r}, which the "
                    f"denominator's own frame does not carry; its roles are {roles!r}",
                    operation_id=operation.id,
                )

    def _check_share_of_needs_axis(self, operation: ShareOfOperation) -> None:
        """Report `SHARE_AXIS_REQUIRED` when an un-keyed share's numerator still carries several members.

        `numerator.entity` already narrows a whole side to one row by
        itself -- `_check_side`'s own concern -- so this only reads a side
        left unnarrowed that way. What it asks of the numerator's own
        resolved shape is not `len(shape.roles) > 1`, `_check_rank`'s own
        test: a share_of's numerator can carry exactly one role dimension
        (one column, `observationAbout` in the reproduction this guards)
        and still pool several members under it, which `_check_rank`'s
        count of role *dimensions* would never see. `_retained_member_count`
        instead counts the members the numerator's own chain actually still
        carries for that role, after any upstream filter has narrowed it --
        a numerator already filtered down to one entity needs no axis
        either, the case pinned by the sibling test whose numerator reads a
        filter that narrowed a two-country source down to one. Left with
        more than one, dividing by an equally unkeyed denominator sums
        every member into the same total on both sides, so the quotient is
        1 at every period; naming an axis is what lets the numerator
        answer for each member instead.
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
        roles = sorted(role.role for role in pooled if role.role is not None)
        self._issue(
            CompileIssueCode.SHARE_AXIS_REQUIRED,
            f"operation {operation.id!r} shares a frame with no axis, but its numerator still "
            f"carries several members under {roles!r}; name one to key the share on",
            operation_id=operation.id,
        )

    def _retained_member_count(self, op_id: str, role: ResolvedRole) -> int:
        """How many distinct members of `role` still reach `op_id`, after any upstream filter.

        Starts from every label `role`'s own source resolves under its
        column, then walks the single-input chain `_chain_narrows_entities`
        walks, narrowing that set by the tightest `EntityIn` predicate any
        `FilterOperation` along the way names for it -- by the predicate's
        own `role` phrase, or, unqualified, by resolving each of its names
        against `role`'s source the way `compiler.shapes.
        _resolved_narrowed_roles` does. A predicate that still leaves
        several labels standing pools exactly as meaninglessly as no filter
        at all, so this counts what remains rather than trusting a filter's
        mere presence.

        An open entity axis counts the members its source states, which
        `compiler.evidence.open_axis_for` admits only when counted. No
        filter narrows one: an `EntityIn` may only name the requirement's
        own entities, and none sits on an open axis.
        """
        open_axis = self._open_axis_of(role)
        if open_axis is not None:
            return open_axis.member_count or 0
        source = self._role_source(role)
        if source is None:
            return 0
        retained = {e.label.casefold() for e in source.entities if e.role == role.column}
        seen: set[str] = set()
        while op_id in self.operations and op_id not in seen:
            seen.add(op_id)
            operation = self.operations[op_id]
            if isinstance(operation, RetrieveOperation):
                break
            if isinstance(operation, FilterOperation):
                retained = _narrowed_by_filter(operation, role, source, retained)
            if not isinstance(operation, _SINGLE_INPUT):
                break
            op_id = operation.input
        return len(retained)

    def _states_across(self, op_id: str) -> bool:
        """Whether the dataset requirement `op_id`'s chain rests on states `across`."""
        requirement = self._requirement_upstream(op_id)
        return requirement is not None and requirement.across is not None

    def _role_source(self, role: ResolvedRole) -> SelectedSource | None:
        """The source the retrieval of `role`'s own requirement selected."""
        retrieve_id = self._retrieve_by_requirement.get(role.requirement_id)
        return self.sources.get(retrieve_id) if retrieve_id is not None else None

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
        requirement = self._requirement_upstream(operation.input)
        if requirement is None:
            return
        source = self._source_upstream(operation.input)
        names = {entity.name for entity in requirement.entities}
        for predicate in operation.predicates:
            if not isinstance(predicate, EntityIn):
                continue
            unknown = [name for name in predicate.names if name not in names]
            if unknown:
                self._issue(
                    CompileIssueCode.UNKNOWN_ENTITY,
                    f"operation {operation.id!r} names entities {unknown!r}, which requirement "
                    f"{requirement.id!r} does not; its entities are {sorted(names)!r}",
                    operation_id=operation.id,
                    requirement_id=requirement.id,
                )
                continue
            if source is None:
                continue
            for name in predicate.names:
                self._resolve_named_entity(
                    operation.id, requirement, name, source, role=predicate.role
                )

    def _check_two_sided(self, op_id: str, left: Side, right: Side) -> None:
        requirements = [self._check_side(op_id, side) for side in (left, right)]
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
        self._check_frame_units(op_id, left, right)

    def _check_frame_units(self, op_id: str, left: Side, right: Side) -> None:
        """Report two sides whose frames declare their units differently.

        analysis-mcp pairs two slices only when their `unit` cells are equal
        (`cross_frame_unit_mismatch`), and each side's cells are what its
        source declares (`_frame_unit`): the request object for a definition
        bound to a unit by request, the source's unit ref for any other.
        Only a side holding a value in its source's unit, measured or an
        absolute difference, is read; a side whose unit no single pin covers
        is left to analysis-mcp.

        `FRAME_UNIT_FORMS_DIFFER` when both denote one currency and price
        basis (`_denoted_unit`) and only the form differs, a ref against a
        request: the side whose source binds no unit is named, with what to
        change so it is retrieved by request too. dfo-apis writes a natively
        satisfied request as the request object and an unbound definition as
        its ref (ONEcampaign/dfo-apis#1179). `FRAME_UNITS_DIFFER` otherwise.
        """
        read: list[tuple[JsonValue, Unit | None, DatasetRequirement | None]] = []
        for side in (left, right):
            resolved = self.shapes.get(side.input)
            source = self._source_upstream(side.input)
            if (
                resolved is None
                or resolved.shape.value not in (ValueKind.MEASURED, ValueKind.DIFFERENCE)
                or source is None
                or (unit := _frame_unit(source)) is None
            ):
                return
            read.append((unit, _denoted_unit(source, unit), self._requirement_upstream(side.input)))
        (left_unit, left_denoted, left_req), (right_unit, right_denoted, right_req) = read
        if left_unit == right_unit:
            return
        unbound = left_req if not isinstance(left_unit, dict) else right_req
        if left_denoted is not None and left_denoted == right_denoted and unbound is not None:
            self._issue(
                CompileIssueCode.FRAME_UNIT_FORMS_DIFFER,
                f"operation {op_id!r} pairs two sides in {_denoted_words(left_denoted)}, but "
                f"requirement {unbound.id!r} reads a source that binds no unit, so its frame "
                "states the unit differently and the two are not paired; state unit "
                f"{left_denoted.model_dump_json(exclude_none=True)} on requirement "
                f"{unbound.id!r} and leave its source_spec_id unset, so it is searched again "
                "and retrieved bound to that unit",
                operation_id=op_id,
                requirement_id=unbound.id,
            )
            return
        self._issue(
            CompileIssueCode.FRAME_UNITS_DIFFER,
            f"operation {op_id!r} pairs a side in {_unit_cell(left_unit, left_denoted)} with "
            f"one in {_unit_cell(right_unit, right_denoted)}, and two frames in different "
            "units cannot be set against each other",
            operation_id=op_id,
        )

    def _check_side(self, op_id: str, side: Side) -> DatasetRequirement | None:
        requirement = self._requirement_upstream(side.input)
        if requirement is None:
            return None
        names = [entity.name for entity in requirement.entities]
        if side.entity is None:
            source = self._source_upstream(side.input)
            # What the side reads is the frame at `side.input`, not the frame
            # the retrieval produced: a filter naming entities on the way has
            # already kept only those, so the source's own list says nothing
            # about how many the side would pool. `_check_source_width` reads
            # the chain the same way.
            narrowed = self._chain_narrows_entities(side.input)
            labels = (
                _pooled_labels(source, requirement) if source is not None and not narrowed else []
            )
            pooled = _pooled_names(requirement, source)
            if pooled:
                self._issue(
                    CompileIssueCode.ENTITY_REQUIRED,
                    f"operation {op_id!r} takes a whole side from {side.input!r}, whose "
                    f"requirement names several entities on one side of the flow {pooled!r}; "
                    "say which",
                    operation_id=op_id,
                    requirement_id=requirement.id,
                )
            elif requirement.across is not None:
                # The frame holds every member of the open axis, whether or not
                # an entity on another axis is named, and a side can name only
                # an entity its requirement names: keyed by an axis or not, it
                # has no way to say which members it reads.
                self._issue(
                    CompileIssueCode.ENTITY_REQUIRED,
                    f"operation {op_id!r} takes a whole side from {side.input!r}, whose "
                    f"requirement covers every {requirement.across!r} and names none of them, "
                    "so the side cannot say which it reads; name them in the requirement's "
                    "`entities`",
                    operation_id=op_id,
                    requirement_id=requirement.id,
                )
            elif labels and source is not None:
                # The requirement names one entity per role column, but the
                # frame carries several on one: a whole side would pool them
                # into one series.
                self._issue(
                    CompileIssueCode.ENTITY_REQUIRED,
                    f"operation {op_id!r} takes a whole side from {side.input!r}, whose source "
                    f"{source.spec_id} resolves several entities {labels!r}; say which",
                    operation_id=op_id,
                    requirement_id=requirement.id,
                )
        elif side.entity not in names:
            self._issue(
                CompileIssueCode.UNKNOWN_ENTITY,
                f"operation {op_id!r} names entity {side.entity!r}, which requirement "
                f"{requirement.id!r} does not; its entities are {names!r}",
                operation_id=op_id,
                requirement_id=requirement.id,
            )
        return requirement

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
        two-sided operation (`_pools_entities`), folds a surplus role into a
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
                    (operation.input,),
                    verb="ranks them alongside the entities it names",
                )
                continue
            if not _pools_entities(operation):
                continue
            self._report_source_width(
                operation.id, _pooled_inputs(operation), verb="pools them with the entities it does"
            )

    def _report_source_width(
        self, operation_id: str, input_ids: Sequence[str], *, verb: str
    ) -> None:
        role = self._first_unnarrowed_surplus_role(input_ids)
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

    def _first_unnarrowed_surplus_role(self, input_ids: Sequence[str]) -> ResolvedRole | None:
        """The first surplus, un-narrowed role dimension among `input_ids`' own resolved shapes.

        Reads `self.shapes[input_id].columns` directly, in the order
        `input_ids` names them -- deterministic, and enough for one issue
        naming one requirement, since a pooling operation with several
        surplus sides still gets exactly one.
        """
        for input_id in input_ids:
            resolved = self.shapes.get(input_id)
            if resolved is None:
                continue
            for role in resolved.columns:
                if role.surplus and not role.narrowed:
                    return role
        return None

    def _chain_narrows_entities(self, op_id: str) -> bool:
        """Whether anything between `op_id` and its retrieval keeps only named entities."""
        seen: set[str] = set()
        while op_id in self.operations and op_id not in seen:
            seen.add(op_id)
            operation = self.operations[op_id]
            if isinstance(operation, RetrieveOperation):
                return False
            if _narrows_entities(operation):
                return True
            if not isinstance(operation, _SINGLE_INPUT):
                return False
            op_id = operation.input
        return False

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
        for op_id in self.order:
            operation = self.operations[op_id]
            if op_id not in self.tools:
                continue
            capability, schema = self.tools[op_id]
            built = self._arguments(operation)
            if built is None:
                continue
            args, bindings = built
            self._check_bindings(operation, capability, bindings)
            self._check_arguments(operation, capability, schema, args, bindings)
            expected_role_keys = self._expected_role_keys(op_id)
            self.steps[op_id] = WorkflowStep(
                id=op_id,
                operation_id=op_id,
                tool=capability.tool,
                args=args,
                bindings=bindings,
                implementation_version=capability.implementation_version,
                input_schema_hash=schema.input_schema_hash,
                output_schema_hash=schema.output_schema_hash,
                expected_role_keys=expected_role_keys,
            )

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
            ShareOfOperation: self._share_of_arguments,
            JoinOperation: self._join_arguments,
            FilterOperation: self._filter_arguments,
        }
        return writers[type(operation)](operation)

    def _retrieve_arguments(self, operation: RetrieveOperation) -> Arguments | None:
        """Write `spec_id`, `stat_vars` for a family, and the StatVars the source elected.

        `stat_vars` is the source's `stat_var_reading`: `sum` asks dfo-apis
        to sum the family's members into one series, `separate` keeps one
        per member. It sits outside the definition's identity, so it
        re-mints nothing.

        The StatVars the source posts (`compiler.sources.posted_stat_vars`)
        go as `resolved_stat_var_dcids`, left out when it posts none.
        dfo-apis checks they are still members of the dataset, refusing a
        stale list with a remedy to search again, and a family's root among
        them proves a truncated member walk complete. They sit outside the
        identity too.

        `breakdown` is the requirement's, as its code system and level: dfo-apis
        sums the family within each code rolled up to that level, so the frame
        carries one series per code under the family's own ref.

        `place_types` is written for a retrieval a place-level filter reads
        (`_read_by_level_filter`): it adds the `observationAbout_type`
        column that filter keeps one level by. Rendering only, so it re-mints
        nothing either.

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
        if posted := posted_stat_vars(source):
            args["resolved_stat_var_dcids"] = list(posted)
        requirement = self.requirements.get(operation.requirement_id)
        if isinstance(requirement, DatasetRequirement) and requirement.breakdown is not None:
            args["breakdown"] = {
                "code_system": BREAKDOWN_CODE_SYSTEMS[requirement.breakdown.dimension],
                "level": requirement.breakdown.level,
            }
        if self._read_by_level_filter(operation.id):
            args["place_types"] = True
        return args, {}

    def _read_by_level_filter(self, retrieve_id: str) -> bool:
        """Whether a filter keeping one level of place reads `retrieve_id`, directly or through other filters."""
        for operation in self.operations.values():
            if not isinstance(operation, FilterOperation) or not keeps_one_level(
                operation.predicates
            ):
                continue
            op_id, seen = operation.input, set[str]()
            while (
                isinstance(read := self.operations.get(op_id), FilterOperation)
                and op_id not in seen
            ):
                seen.add(op_id)
                op_id = read.input
            if op_id == retrieve_id:
                return True
        return False

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
        """`_compare_arguments`' sides, plus each side's unit pinned to the one
        its fetched frame declares (`_frame_unit`).

        A pin narrows a side to the rows in that unit, so a frame holding
        several units is summed in the one its source resolved. A side whose
        frame unit cannot be known ahead is left
        unpinned, and analysis-mcp checks the two resolved units agree either
        way.
        """
        built = self._two_sided_arguments(operation.id, operation.left, operation.right)
        if built is None:
            return None
        args, bindings = built
        for parameter, side in (("left_unit", operation.left), ("right_unit", operation.right)):
            source = self._source_upstream(side.input)
            unit = _frame_unit(source) if source is not None else None
            if unit is not None:
                args[parameter] = unit
        return args, bindings

    def _share_of_arguments(self, operation: ShareOfOperation) -> Arguments | None:
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
        if operation.axis is not None:
            keyed = self._share_of_key_arguments(operation)
            if keyed is None:
                return None
            args.update(keyed)
        return args, bindings

    def _share_of_key_arguments(self, operation: ShareOfOperation) -> dict[str, JsonValue] | None:
        """`numerator_by`/`denominator_by`, keyed by name so canonicalisation cannot mispair them.

        Only called once `operation.axis is not None`. Resolves the literal
        column `axis` names on each side from that side's own resolved
        shape; `None` when either lookup fails, which
        `_check_share_of_axis` has already reported as `SHARE_KEY_UNKNOWN`.
        A pooled denominator writes no `denominator_by` at all, matching the
        engine's own default of an empty mapping.
        """
        assert operation.axis is not None
        numerator_shape = self.shapes.get(operation.numerator.input)
        if numerator_shape is None:
            return None
        numerator_role = _matching_role(numerator_shape.columns, operation.axis)
        if numerator_role is None:
            return None
        keyed: dict[str, JsonValue] = {"numerator_by": [numerator_role.column]}
        if operation.denominator_scope is DenominatorScope.PER_AXIS:
            denominator_shape = self.shapes.get(operation.denominator.input)
            if denominator_shape is None:
                return None
            denominator_role = _matching_role(denominator_shape.columns, operation.axis)
            if denominator_role is None:
                return None
            keyed["denominator_by"] = {numerator_role.column: denominator_role.column}
        return keyed

    def _side_entity(self, op_id: str, side: Side) -> ResolvedEntity | None:
        """Resolve `side.entity`, or `None` if no argument can be built for it.

        Only called once `side.entity is not None`.
        """
        assert side.entity is not None
        source = self._source_upstream(side.input)
        requirement = self._requirement_upstream(side.input)
        if source is None or requirement is None:
            return None
        if side.entity not in {entity.name for entity in requirement.entities}:
            # Already reported as `unknown_entity` by `_check_side`.
            return None
        return self._resolve_named_entity(op_id, requirement, side.entity, source)

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
        requirement = self._requirement_upstream(operation.input)
        source = self._source_upstream(operation.input)
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
                predicates.append(
                    {
                        "kind": "ordered_threshold",
                        "column": TIME_PERIOD_COLUMN,
                        "op": predicate.op,
                        "value": str(predicate.year),
                    }
                )
            elif isinstance(predicate, PlaceLevel):
                predicates.append(
                    {
                        "kind": "place_level",
                        "column": type_column(PLACE_ROLE),
                        "level": predicate.level.value,
                    }
                )
            else:
                written = self._entity_in_argument(operation.id, requirement, source, predicate)
                if written is None:
                    return None
                predicates.append(written)
        return {"predicates": predicates}, {"frame_ref": _frame_from(operation.input)}

    def _entity_in_argument(
        self,
        op_id: str,
        requirement: DatasetRequirement | None,
        source: SelectedSource | None,
        predicate: EntityIn,
    ) -> dict[str, JsonValue] | None:
        if requirement is None or source is None:
            return None
        columns: set[str] = set()
        refs: list[str] = []
        for name in predicate.names:
            candidates = _entity_candidates(requirement, name, predicate.role)
            if len(candidates) != 1:
                # Unknown or ambiguous -- already reported by `_check_filter`.
                return None
            resolved = resolve_entity(candidates[0], source.entities)
            if resolved is None:
                # Unresolved -- already reported by `_check_filter`.
                return None
            columns.add(resolved.role)
            refs.append(resolved.ref)
        if len(columns) > 1:
            self._issue(
                CompileIssueCode.ENTITY_AMBIGUOUS,
                f"operation {op_id!r} names entities resolving to different role columns "
                f"{sorted(columns)!r}",
                operation_id=op_id,
            )
            return None
        (column,) = columns
        return {"kind": "category_match", "column": column, "values": refs}

    def _resolve_named_entity(
        self,
        op_id: str,
        requirement: DatasetRequirement,
        name: str,
        source: SelectedSource,
        *,
        role: str | None = None,
    ) -> ResolvedEntity | None:
        """Resolve `name` against `requirement`'s own entities and `source`.

        Reports `entity_ambiguous` when `requirement` names `name` under more
        than one role and `role` does not narrow it to one, and
        `entity_unresolved` when `source` cannot resolve it. Returns `None`
        either way, so a caller with nothing to bind can stop cleanly.
        """
        candidates = _entity_candidates(requirement, name, role)
        if len(candidates) > 1:
            self._issue(
                CompileIssueCode.ENTITY_AMBIGUOUS,
                f"operation {op_id!r} names entity {name!r}, which requirement "
                f"{requirement.id!r} names under several roles {[c.role for c in candidates]!r}; "
                "say which",
                operation_id=op_id,
                requirement_id=requirement.id,
            )
            return None
        entity = candidates[0] if candidates else Entity(name=name, role=role)
        resolved = resolve_entity(entity, source.entities)
        if resolved is None:
            self._issue(
                CompileIssueCode.ENTITY_UNRESOLVED,
                f"operation {op_id!r} names entity {name!r}, which source {source.spec_id} does "
                "not resolve, so no entity argument can be written",
                operation_id=op_id,
            )
        return resolved

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
        for name in undeclared_parameters(bindings, schema.parameters):
            self._issue(
                CompileIssueCode.BAD_ARGS,
                f"{capability.tool.name!r} takes no argument named {name!r}",
                operation_id=operation.id,
                tool=capability.tool,
            )
        for message in literal_argument_errors(
            arguments=args, schema=schema.parameters, bound=bindings
        ):
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
    ) -> None:
        self.issues.append(
            CompileIssue(
                code=code,
                reason=reason,
                operation_id=operation_id,
                requirement_id=requirement_id,
                tool=tool,
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
        retrieval (a compare, sum_of, share_of or join reading two sides that do not
        coincide).
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

    def _readers(self) -> dict[str, list[str]]:
        """Map each operation id to the operations reading it, in canonical order."""
        readers: dict[str, list[str]] = {}
        for operation in self.operations.values():
            for input_id in _inputs_of(operation):
                reading = readers.setdefault(input_id, [])
                if operation.id not in reading:
                    reading.append(operation.id)
        return readers

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
            inputs = [i for i in _inputs_of(self.operations[op_id]) if i in self.operations]
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
            pending.extend(i for i in _inputs_of(self.operations[op_id]) if i in self.operations)
        return reachable


def _entity_candidates(
    requirement: DatasetRequirement, name: str, role: str | None
) -> list[Entity]:
    return [
        entity
        for entity in requirement.entities
        if entity.name == name and (role is None or entity.role == role)
    ]


def _inputs_of(operation: Operation) -> tuple[str, ...]:
    if isinstance(operation, RetrieveOperation):
        return ()
    if isinstance(operation, CompareOperation | SumOfOperation):
        return (operation.left.input, operation.right.input)
    if isinstance(operation, ShareOfOperation):
        return (operation.numerator.input, operation.denominator.input)
    if isinstance(operation, JoinOperation):
        return (operation.left, operation.right)
    return (operation.input,)


def _frame_unit(source: SelectedSource) -> JsonValue:
    """The unit a frame fetched from `source` declares in its `unit` cells.

    A definition that binds its unit by request (`facets.unit.request` in its
    identity) is converted on fetch, so its frame declares that request
    object; one whose request names its currency by entity role resolves a
    currency per row, which no single pin covers, so it is `None`. Any
    other definition serves its source's own unit ref.
    """
    facets = (source.identity or {}).get("facets")
    unit = facets.get("unit") if isinstance(facets, dict) else None
    if not isinstance(unit, dict) or "request" not in unit:
        return source.unit_ref
    try:
        converted = Unit.model_validate(unit["request"])
    except ValidationError:
        return None
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


def _denoted_words(unit: Unit) -> str:
    """`unit` in words for an issue's reason."""
    if unit.base_year is not None:
        return f"{unit.currency} at constant {unit.base_year} prices"
    return f"{unit.currency} at {unit.prices} prices"


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


def _distinct_labels(source: SelectedSource, requirement: DatasetRequirement) -> list[str]:
    """The distinct entity labels `source` resolves under `requirement`'s own role columns.

    Case-folded, in the source's order. Entities under the other role
    columns -- the creditor alongside the debtor -- are left out, as
    `_role_columns` leaves their columns out: pooling a debtor with its
    creditor pools nothing the plan did not state, and naming the creditor
    to narrow it would be `unknown_entity`.
    """
    columns = set(_role_columns(source, requirement))
    labels: list[str] = []
    seen: set[str] = set()
    for entity in source.entities:
        if entity.role in columns and entity.label.casefold() not in seen:
            seen.add(entity.label.casefold())
            labels.append(entity.label)
    return labels


def _pooled_names(requirement: DatasetRequirement, source: SelectedSource | None) -> list[str]:
    """The names `requirement` states that share a role column with another it states.

    One entity per role column, a donor and its recipient say, is one
    series, which a whole side reads as it is. Two on one column are two
    series a whole side would pool. A name's column is the one `source`
    resolves it under, else its role; either is compared by its side of the
    flow, so `origin` and `donor` are one column. Names neither places
    share one column, since nothing tells them apart.
    """
    columns: dict[str | None, list[str]] = {}
    for entity in requirement.entities:
        resolved = resolve_entity(entity, source.entities) if source is not None else None
        column = resolved.role if resolved is not None else entity.role
        key = (flow_side(column) or column) if column is not None else None
        columns.setdefault(key, []).append(entity.name)
    return [name for names in columns.values() if len(names) > 1 for name in names]


def _pooled_labels(source: SelectedSource, requirement: DatasetRequirement) -> list[str]:
    """The labels `source` resolves on any of `requirement`'s role columns that holds several.

    Case-folded duplicates are counted once, as `_distinct_labels` counts
    them; a column holding one label is one series and is left out.
    """
    columns = set(_role_columns(source, requirement))
    by_column: dict[str, dict[str, str]] = {}
    for entity in source.entities:
        if entity.role in columns:
            by_column.setdefault(entity.role, {}).setdefault(entity.label.casefold(), entity.label)
    return [label for labels in by_column.values() if len(labels) > 1 for label in labels.values()]


def _surplus_labels(requirement: DatasetRequirement, source: SelectedSource) -> list[str]:
    """The labels `source` resolves under `requirement`'s role columns that it does not name."""
    named = {entity.name.casefold() for entity in requirement.entities}
    return [
        label for label in _distinct_labels(source, requirement) if label.casefold() not in named
    ]


def _narrows_entities(operation: Operation) -> bool:
    """Whether `operation` keeps only named entities of the frame it reads."""
    return isinstance(operation, FilterOperation) and any(
        isinstance(predicate, EntityIn) for predicate in operation.predicates
    )


def _pools_entities(operation: Operation) -> bool:
    """Whether `operation` folds every entity of some input's frame into one figure.

    A period-only aggregate does; so does a whole side of a two-sided
    operation. A rank and a growth write the requirement's role columns as
    `rank_by` and `entity_by`, an aggregate grouping by entity writes them
    as `group_by`, and a filter and a join keep the rows: each leaves one
    row per entity, so a surplus entity stays visible to the reader. A
    pooled total hides it behind a figure presented as the asked entity's,
    which is what this rule refuses. A rank over a wider set still changes
    an entity's position; this rule does not catch that, a future one might.
    """
    if isinstance(operation, AggregateOperation):
        return GroupBy.ENTITY not in operation.group_by
    if isinstance(operation, CompareOperation | SumOfOperation):
        sides = (operation.left, operation.right)
    elif isinstance(operation, ShareOfOperation):
        sides = (operation.numerator, operation.denominator)
    else:
        return False
    return any(side.entity is None for side in sides)


def _pooled_inputs(operation: Operation) -> tuple[str, ...]:
    """The inputs `operation` pools when `_pools_entities` says it does; `()` otherwise.

    A period-only aggregate pools its one input; a two-sided operation
    pools whichever of its sides takes no `entity` -- the same sides
    `_pools_entities` checks, named rather than merely counted.
    """
    if isinstance(operation, AggregateOperation):
        return (operation.input,) if GroupBy.ENTITY not in operation.group_by else ()
    if isinstance(operation, CompareOperation | SumOfOperation):
        sides = (operation.left, operation.right)
    elif isinstance(operation, ShareOfOperation):
        sides = (operation.numerator, operation.denominator)
    else:
        return ()
    return tuple(side.input for side in sides if side.entity is None)


def _narrowed_by_filter(
    operation: FilterOperation, role: ResolvedRole, source: SelectedSource, retained: set[str]
) -> set[str]:
    """`retained`, narrowed by `operation`'s own `EntityIn` predicates that name `role`.

    A predicate stating its own `role` narrows `retained` only when that
    phrase is `role.role`; an unqualified one narrows it only for the names
    that resolve onto `role.column` -- a column-for-column mirror of
    `compiler.shapes._resolved_narrowed_roles`, which only proves *that* a
    predicate narrows a column, not *by how much*. A predicate naming
    nothing that resolves here leaves `retained` untouched.
    """
    for predicate in operation.predicates:
        if not isinstance(predicate, EntityIn):
            continue
        if predicate.role is not None:
            if predicate.role != role.role:
                continue
            named = {name.casefold() for name in predicate.names}
        else:
            named = {
                name.casefold()
                for name in predicate.names
                if (resolved := resolve_entity(Entity(name=name, role=None), source.entities))
                is not None
                and resolved.role == role.column
            }
            if not named:
                continue
        retained = retained & named
    return retained


def _matching_role(columns: Sequence[ResolvedRole], role: str) -> ResolvedRole | None:
    """The first of `columns` whose plan-vocabulary `role` is `role`, or `None` when none is."""
    for column in columns:
        if column.role == role:
            return column
    return None


def _axis_column(columns: Sequence[ResolvedRole], axis: str | None) -> ResolvedRole | None:
    """The literal column `axis` names among `columns`, or the lone one when `axis` is `None`.

    A column-for-column mirror of `planner.shapes._select_role` over the
    compiler's own `ResolvedRole`s: `None` when `axis` does not resolve,
    which every caller reads as "already reported" -- `_check_rank`'s
    `RANK_AXIS_REQUIRED` when several columns and none named, or
    `planner.shapes.rejects`'s own `INPUT_SHAPE_UNSUPPORTED` when `axis`
    names nothing the input carries.
    """
    if axis is None:
        return columns[0] if len(columns) == 1 else None
    return _matching_role(columns, axis)


def _role_columns(source: SelectedSource, requirement: DatasetRequirement) -> list[str]:
    """The role columns of `source` that `requirement`'s entities resolve under, in the source's order.

    `compiler.shapes.shapes_for` computes this same set once, into a fresh
    retrieval's own `ResolvedShape.columns`, for every argument an operation
    writes off its shape (`compile._columns_of`). This copy remains for
    `_distinct_labels`/`_surplus_labels`, which `_check_source_width` reads
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
        resolved.role
        for entity in requirement.entities
        if (resolved := resolve_entity(entity, source.entities)) is not None
    }
    return [column for column in every if column in wanted] or every


def _group_columns(group_by: Sequence[GroupBy], roles: Sequence[ResolvedRole]) -> list[str]:
    """`group_by` written as literal columns: every one of `roles` for `ENTITY`, the
    breakdown's alone for `BREAKDOWN`, the frame's own period column for `PERIOD`."""
    columns: list[str] = []
    for key in group_by:
        if key is GroupBy.PERIOD:
            new = [TIME_PERIOD_COLUMN]
        elif key is GroupBy.BREAKDOWN:
            new = [role.column for role in roles if role.breakdown]
        else:
            new = [role.column for role in roles]
        columns.extend(column for column in new if column not in columns)
    return columns

"""The compiled workflow: exactly what runs once a plan is approved.

`compiler.compile` builds one from a `Translation`. Each `WorkflowStep` names
its tool, the literal arguments the compiler wrote, and the `Binding`s that
take an earlier step's output where a value exists only once the plan runs --
a frame reference, above all. `Workflow.sources` records what discovery
selected for each retrieval, `evidence_bindings` which step's result is the
evidence for which requirement, so the executor knows where to check, and
`warnings` the caveats the compiler found and let through.

This is a record, not a program: the executor reads it in order and resolves
each binding from the results it has. Everything in it is hashed into the
plan's revision, so approving a plan approves these exact calls and caveats.
"""

from __future__ import annotations

from typing import Annotated, Literal

from common.hashed_fields import HASHED_WHEN_SET
from pydantic import Field, JsonValue, model_validator

from planner.records import (
    EvidenceIssue,
    Identifier,
    Period,
    ResolvedSourceMention,
    SpecId,
    StatVarReading,
    StrictModel,
    TitlelessEnum,
)


class ArtifactKind(TitlelessEnum):
    """A kind of value that passes between steps."""

    SPEC_ID = "spec_id"
    FRAME_REF = "frame_ref"


class ToolId(StrictModel):
    """A tool, by the server that serves it and its name there."""

    server: Literal["data-mcp", "analysis-mcp"]
    name: str = Field(min_length=1)


class Binding(StrictModel):
    """Take an earlier step's output as this argument's value at run time."""

    from_step: Identifier
    output: ArtifactKind


class ResolvedEntity(StrictModel):
    """An entity as the selected source resolved it.

    `role` is the source's role column for this entity and `ref` its value
    there, which is what a tool's entity binding takes. `label` is the name
    the source gives it, which is what a requirement's `Entity.name` is
    matched against. A name that resolves under several role columns is
    told apart by `role` alone.
    """

    role: str = Field(min_length=1)
    ref: str = Field(min_length=1)
    label: str = Field(min_length=1)


class ResolvedGroup(StrictModel):
    """A named set a source binds on one role column, read whole or as its members.

    `search` names the set it bound, keyed by the role column it fills
    (`resolved.entity_groups`). `label` is the set's name, and `ref` its own
    graph node, `None` when the graph holds none (a resolver group such as
    "G7"). `total` is the party the dataset publishes as the set's own total
    (`GroupOfSeven`, `DACAfrica`) when the role binds that one party; `None`
    when the role binds the set's members, one entity each. A requirement's
    `Entity` naming the label, case-folded, on the set's side of the flow is
    met by the source under either reading (`compiler.evidence.resolve_group`),
    except that one asking for the members is not met by the total.
    """

    role: str = Field(min_length=1)
    label: str = Field(min_length=1)
    ref: str | None = None
    # Left out of every dump while unset, so a source saved without it keeps
    # its serialised form.
    total: Annotated[str | None, HASHED_WHEN_SET] = Field(
        default=None, exclude_if=lambda total: total is None
    )


class OpenAxis(StrictModel):
    """One constraint property a source's family of StatVars varies along.

    `property` is the slot's constraint property dcid, as `search` resolved
    it. `summable` is whether dfo-apis may sum the family's members over it:
    the slot declares it summable, and every member resolved a value on it.
    """

    property: str = Field(min_length=1)
    summable: bool


class ReadingSetBy(TitlelessEnum):
    """Who set a source's `StatVarReading`.

    `plan` is the dataset requirement's own `stat_vars`. `rule` is
    `compiler.sources.default_reading`, applied where the requirement leaves
    it unset.
    """

    PLAN = "plan"
    RULE = "rule"


class OpenEntityAxis(StrictModel):
    """An entity axis the selected source covers in full, with no member named.

    `column` is the source's role column for the axis (`observationAbout`,
    `origin` or `destination`), the column a rank or a grouping over the
    population reads. `member_count` is how many entities the source holds
    on it: 193 for every country, 1 for a worldwide total, `None` when
    `search` could not count them.
    """

    column: str = Field(min_length=1)
    member_count: int | None = Field(default=None, ge=0)


class MeasureAlternative(StrictModel):
    """One other measure `search` offers for a source, from `resolved.alternative_measures`.

    `spec_id` is the alternative's own definition, minted from the source's
    identity with its measure axes swapped. `label` names the measure without
    its price basis ("net disbursements") on axis `what`, or names the prices
    ("constant prices") on axis `unit`. `axis` is a plain string, since
    the vocabulary belongs to dfo-apis and an unknown axis must not fail
    validation. `identity` preserves the sibling definition for approval
    amendments and later turns, so choosing it resolves the recorded
    dataset without a fresh search.
    """

    spec_id: SpecId
    label: str = Field(min_length=1)
    axis: str = Field(min_length=1)
    identity: Annotated[dict[str, JsonValue] | None, HASHED_WHEN_SET] = Field(
        default=None, exclude_if=lambda identity: identity is None
    )


class ElectedMeasure(StrictModel):
    """The measure `search` elected for a source, and the others it could answer with.

    `label` is `resolved.measure_label`, the measure partition the source's
    served rows exhibit, named the way each `what` alternative is. Empty
    `alternatives` means `search` labelled no other partition.
    """

    label: str = Field(min_length=1)
    alternatives: tuple[MeasureAlternative, ...] = ()

    def other_counts(self) -> tuple[MeasureAlternative, ...]:
        """Each alternative that counts the flow another way (axis `what`).

        A price alternative (axis `unit`) is the same count in other prices,
        so it is not one.
        """
        return tuple(alternative for alternative in self.alternatives if alternative.axis == "what")


class SelectedSource(StrictModel):
    """The dataset definition discovery selected for one retrieval.

    Copied from the search result at compile time, so a plan carries what
    was known about the source when it was approved: its identity, the
    measures and entities it resolves, its unit, and the years it reports
    covering.

    `searched_as` is the phrase actually sent to `search` to find this
    source, recorded here so the words the source answered are inside the
    approved revision. `coverage_kind` is the envelope's
    `resolved.coverage.kind` -- `exact`, `breadth` or `bare` -- and only
    `exact` reports a real series window, so `coverage_window` is read
    against it. It is a plain string, not a closed set: the value comes from
    another service and an unknown kind must not fail validation. `has_data`
    is `None` when the envelope states no coverage, rather than an optimistic
    default. `identity` is the canonical identity document `spec_id` hashes;
    `None` when the search result carried none, which is not the same as an
    empty one -- drift detection compares an approved identity against a
    manifest's and cannot report drift from a value it never held.

    `measure_label` is `resolved.shape.label`, the name `search` gives what
    the definition measures; `None` when the envelope states none.

    `elected_measure` is the measure partition `search` elected among those
    the definition's family holds, with the others it offers in its place
    (`ElectedMeasure`); `None` when the envelope states no
    `resolved.measure_label`, as for a source with no measure family.

    `unit_per_series` is `True` when `resolved.facets.unit` selects a unit
    per series (`selection: per_series`), so each series states its own unit
    and the source names no one `unit_ref`; `None` otherwise.

    `provenance_account` is `resolved.facets.provenance.account`:
    `"provider"` (`compiler.sources.PROVIDER_ACCOUNT`) for a table of the
    flows a provider reports, as the OECD and ONE aid tables are; `None`
    when the envelope states none.

    `stat_var_member_count` is `resolved.stat_vars.member_count`, the members
    of the definition's binding that `search` confirmed at the resolve's
    place, and `stat_var_member_count_exact` says whether that count is exact
    or a floor (`False`). `served_count` is `resolved.stat_vars.served_count`,
    how many StatVars the mint serves for the definition, which `search`
    states only once the membership settled; one makes the source a single
    series (`compiler.sources.is_family`). Each is `None` when the envelope
    did not state it.

    The binding itself is the population `resolved.shape.member_count`
    counts, kept as `shape_member_count`, which can exceed the confirmed
    members. `shape_member_digest` is `resolved.shape.member_digest`, the
    digest of that population's sorted bare dcids
    (`compiler.sources.member_digest`), which `search` states only when the
    population is exact; `None` when it is inexact or the envelope carries
    none. The membership check compares it against the binding a retrieval
    traces.

    `open_axes` are the slots the family's members vary along, from
    `resolved.slots`: the ones with a constraint property that is not
    inapplicable, bound to no value, to a set, or to a value whose code
    system admitted its descendants. `stat_var_reading` is how the
    retrieval reads a family (`compiler.sources.is_family`), `None` for a
    source binding one StatVar, and
    `stat_var_reading_set_by` who set it; the compiler states both
    (`compiler.sources.stat_var_reading`).

    `slot_codes` holds, for each constraint property the binding bounds,
    every code it admits: a value with the descendants the code index widened
    it to (the slot's `expansion` in `resolved.slots`; Search states one
    only when the walk added a code), or each value of a set with the codes
    its expansion added. A member StatVar carries one of those codes on that
    property. A set with no expansion maps to `None`: nothing states whether
    its values widen, so the codes it admits are unknown. An unbound
    property is absent. The whole field is `None` when the block states no
    slots or a binding this reader does not know
    (`compiler.sources.family_overlap` then cannot decide what two families
    share).

    `open_entity_axes` are the entity axes the source left open, one per column and
    sorted by column: the axis `search` defaulted for a part that named no
    entity on it, which the definition covers in full. A column a resolved
    entity binds is never among them. A source built from a `search` block
    carries a tuple, empty when it leaves no axis open. `None` means the
    axes were never read, as on a source recorded without the field, and
    `translator.discovery.DiscoveryLog.selectable` does not offer such a
    source to a requirement that states `across`.

    `groups` are the named sets the source binds, one per role column at
    most, each read as its published total or as its members
    (`ResolvedGroup`); empty when the block names none.

    `entity_refs` maps a requirement's entity name, case-folded, that
    matches no label of `entities` to the graph ref it resolves to
    (`translator.discovery.resolve_entity_refs`), so "United States" finds
    the entity labelled "United States of America" by ref
    (`compiler.evidence.named_members`). Empty until that resolve runs, and
    for a source whose names all match by label.

    `definition` is the source's own definition, read from data-mcp's
    `describe` rather than `search`; for a StatVar it is the graph's
    constraint encoding ("mp=count,pt=Person"), not prose.
    `translator.discovery.fetch_definitions`
    populates it, at compile time, for a genuinely `definite`-origin source
    only. `None` before that fetch runs, and left `None` after a fetch that
    timed out, was refused, or never named this source's own stat var: a
    missing or unusable description does not fail the compile it would have
    enriched.
    """

    spec_id: SpecId
    sentence: str
    searched_as: str | None = None
    confidence: str | None = None
    has_data: bool | None
    measure_label: Annotated[str | None, HASHED_WHEN_SET] = None
    # Left out of every dump while unset, so a source saved without it keeps
    # its serialised form.
    elected_measure: Annotated[ElectedMeasure | None, HASHED_WHEN_SET] = Field(
        default=None, exclude_if=lambda measure: measure is None
    )
    stat_var_member_count: int | None = None
    stat_var_member_count_exact: Annotated[bool | None, HASHED_WHEN_SET] = None
    served_count: Annotated[int | None, HASHED_WHEN_SET] = None
    shape_member_count: int | None = None
    shape_member_digest: str | None = None
    entities: tuple[ResolvedEntity, ...] = ()
    groups: Annotated[tuple[ResolvedGroup, ...], HASHED_WHEN_SET] = ()
    # Left out of every dump while empty, so a source saved without it keeps
    # its serialised form.
    entity_refs: Annotated[dict[str, str], HASHED_WHEN_SET] = Field(
        default_factory=dict, exclude_if=lambda refs: not refs
    )
    open_entity_axes: Annotated[tuple[OpenEntityAxis, ...] | None, HASHED_WHEN_SET] = None
    unit_ref: str | None = None
    unit_label: str | None = None
    unit_per_series: Annotated[bool | None, HASHED_WHEN_SET] = None
    provenance_ref: str | None = None
    provenance_label: str | None = None
    provenance_account: Annotated[str | None, HASHED_WHEN_SET] = None
    resolved_sources: Annotated[tuple[ResolvedSourceMention, ...], HASHED_WHEN_SET] = ()
    coverage_kind: str | None = None
    coverage_window: Period | None = None
    identity: dict[str, JsonValue] | None = None
    definition: str | None = None
    open_axes: tuple[OpenAxis, ...] = ()
    slot_codes: Annotated[dict[str, tuple[str, ...] | None] | None, HASHED_WHEN_SET] = None
    stat_var_reading: StatVarReading | None = None
    stat_var_reading_set_by: ReadingSetBy | None = None

    @model_validator(mode="after")
    def _open_entity_axes_are_distinct_and_unbound(self) -> SelectedSource:
        columns = [axis.column for axis in self.open_entity_axes or ()]
        if len(set(columns)) != len(columns):
            raise ValueError(f"every open entity axis must name its own column: {columns!r}")
        if bound := sorted(set(columns) & {entity.role for entity in self.entities}):
            raise ValueError(f"a column a resolved entity binds is not open: {bound!r}")
        return self


class WorkflowStep(StrictModel):
    """One tool call, with its literal arguments and its bindings kept apart.

    `input_schema_hash` is the hash of the tool's argument schema this step
    was compiled against, so the executor can tell when the tool has changed
    underneath an approved plan. `output_schema_hash` is the same, for the
    tool's declared output schema, when it declares one.

    `expected_role_keys` is the compiler's own prediction of this step's
    output role columns, sorted the way `Frame.build` itself sorts role keys.
    It and the role columns in `args` are spelled in the approved source's
    vocabulary (a flow axis such as `origin`). At execution the executor
    binds them to the columns of the frame the step reads
    (`compiler.frame_columns`): it spells the arguments the way that frame
    spells them, and compares a successful call's declared `role_keys`
    against `expected_role_keys` in that spelling, or, for a filter, against
    the role keys its input declared, since a filter keeps every role column
    of its input. A difference refuses `SHAPE_MISMATCH`, and a role the frame
    carries no column for refuses `COLUMN_UNBOUND` before the call. Approving
    a plan approves the roles each step reads and produces; the column names
    those roles take, and the StatVars each frame holds, come from the
    retrieved frame (`compiler.frame_columns.bound_stat_var_count`).

    `id` must equal `operation_id`. The compiler resolves a frame binding by
    naming the operation that produces it (`compile._frame_from`), and that
    binding only finds its step because the two are the same value. The day
    one operation compiles to several steps, `_frame_from` has to name the
    producing step and this validator comes off.
    """

    id: Identifier
    operation_id: Identifier
    tool: ToolId
    args: dict[str, JsonValue] = Field(default_factory=dict)
    bindings: dict[str, Binding] = Field(default_factory=dict)
    implementation_version: str | None = None
    input_schema_hash: str
    output_schema_hash: str | None = None
    expected_role_keys: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _an_argument_is_literal_or_bound(self) -> WorkflowStep:
        both = sorted(set(self.args) & set(self.bindings))
        if both:
            raise ValueError(f"arguments given both literally and by binding: {both!r}")
        return self

    @model_validator(mode="after")
    def _names_its_operation(self) -> WorkflowStep:
        if self.id != self.operation_id:
            raise ValueError(
                f"step id {self.id!r} must equal its operation_id {self.operation_id!r}"
            )
        return self


class EvidenceBinding(StrictModel):
    """Which step's result is the evidence for which requirement.

    `spec_id` is the source the step was approved to retrieve for that
    requirement, so the executor binds a step to its source by identity
    rather than by whatever the tool happened to return.
    """

    requirement_id: Identifier
    step_id: Identifier
    spec_id: SpecId


class Workflow(StrictModel):
    """The compiled steps, in the order they run, with what they rest on.

    `warnings` are the evidence findings that did not block compiling -- a
    coverage window short of the period where the requirement reports gaps,
    a unit that could not be verified. They are kept on the workflow so that
    they enter the plan's revision hash: approving a plan approves its
    caveats, not only its calls.
    """

    steps: tuple[WorkflowStep, ...] = Field(min_length=1)
    sources: tuple[SelectedSource, ...]
    evidence_bindings: tuple[EvidenceBinding, ...]
    terminal_steps: tuple[Identifier, ...] = Field(min_length=1)
    warnings: tuple[EvidenceIssue, ...] = ()

    @model_validator(mode="after")
    def _references_resolve(self) -> Workflow:
        step_ids = [step.id for step in self.steps]
        if len(set(step_ids)) != len(step_ids):
            raise ValueError("every step id must be unique")
        known = set(step_ids)
        unknown = sorted(
            {binding.step_id for binding in self.evidence_bindings} - known
            | set(self.terminal_steps) - known
            | {b.from_step for step in self.steps for b in step.bindings.values()} - known
        )
        if unknown:
            raise ValueError(f"the workflow refers to steps it does not hold: {unknown!r}")
        return self

    def source_of(self, step_id: str) -> SelectedSource | None:
        """The source retrieval `step_id` was compiled from, or `None` for a step that retrieves nothing.

        `compiler.compile` lists `sources` one per evidence binding, in binding
        order, so the source at the binding's own position is the step's even
        where two retrievals share a `spec_id` and read it differently. A
        workflow listing them otherwise names the source by `spec_id` alone.
        """
        for index, binding in enumerate(self.evidence_bindings):
            if binding.step_id != step_id:
                continue
            if index < len(self.sources) and self.sources[index].spec_id == binding.spec_id:
                return self.sources[index]
            return next((s for s in self.sources if s.spec_id == binding.spec_id), None)
        return None

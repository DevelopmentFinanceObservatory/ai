"""Compiled plans of shares of ODA, each source the caller's.

`us_share_of_kenya_oda` is the United States' share of ODA to Kenya, each
year: `us` reads the United States' ODA to Kenya and `all_oda` every
donor's, which `all_by_year` totals by year; `share` divides the first by
that total, and any steps `then` names read the share. The source
`all_oda` reads is one binding the total for official donors (DAC)
(`DAC_DONORS`), one leaving every donor open (`ALL_DONORS`), or one of
named donors only (`NAMED_DONORS`), to Kenya or to more recipients
(`UGANDA`).

`us_share_per_recipient` is the United States' share of ODA to each
recipient, keyed by recipient, over sources leaving the recipients open:
pooled, of every donor's ODA to all of them; per recipient, of official
donors' (DAC) ODA to each.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from planner.operations import (
    AggregateOperation,
    FilterOperation,
    GrowthOperation,
    Operation,
    RankOperation,
    RetrieveOperation,
    ShareOfOperation,
    Side,
    Translation,
)
from planner.plan import (
    AggregateRequirement,
    DatasetRequirement,
    Decision,
    DecisionFixes,
    Entity,
    EvidenceRequirement,
    Period,
    Plan,
    RequirementSide,
    ShareOfRequirement,
)
from planner.records import DenominatorScope, GroupBy, RankAmong, Reduction
from planner.workflow import (
    ArtifactKind,
    Binding,
    EvidenceBinding,
    OpenEntityAxis,
    ResolvedEntity,
    ResolvedGroup,
    SelectedSource,
    ToolId,
    Workflow,
    WorkflowStep,
)

SPEC_US = "spec_" + "c" * 32
SPEC_ALL = "spec_" + "d" * 32

_KENYA = ResolvedEntity(role="destination", ref="country/KEN", label="Kenya")
_USA = ResolvedEntity(role="origin", ref="country/USA", label="United States")
_GBR = ResolvedEntity(role="origin", ref="country/GBR", label="United Kingdom")
UGANDA = ResolvedEntity(role="destination", ref="country/UGA", label="Uganda")
# The United States as a recipient, which a source resolving it on both
# sides of the flow holds beside the donor.
USA_RECEIVING = ResolvedEntity(role="destination", ref="country/USA", label="United States")
_PERIOD = Period(start=2015, end=2024)

# The DAC2a source of the United States' ODA to Kenya.
US_DONOR: dict[str, Any] = {"entities": (_KENYA, _USA)}
# The DAC2a source of ODA to Kenya from official donors (DAC), the total the
# source publishes for every official donor, as Search binds it when no donor is named.
DAC_DONORS: dict[str, Any] = {
    "entities": (
        _KENYA,
        ResolvedEntity(role="origin", ref="DACOfficialDonors", label="official donors (DAC)"),
    ),
    "groups": (
        ResolvedGroup(
            role="origin",
            label="official donors (DAC)",
            total="DACOfficialDonors",
            whole_of=("official",),
            side_classes=("official", "private", "residual"),
        ),
    ),
}
# The DAC2a source of ODA to Kenya from each of its 140 providers, private ones included.
ALL_DONORS: dict[str, Any] = {
    "entities": (_KENYA,),
    "open_entity_axes": (OpenEntityAxis(column="origin", member_count=140),),
}

# The DAC2a source of ODA to Kenya from the United Kingdom and the United States.
NAMED_DONORS: dict[str, Any] = {"entities": (_KENYA, _GBR, _USA)}

# A step reading the share: its requirement and its operation, each reading `share`.
type Then = tuple[EvidenceRequirement, Operation]


def us_share_of_kenya_oda(
    decision: Callable[..., Decision],
    *,
    whole: Mapping[str, Any],
    numerator: Mapping[str, Any],
    donor: str = "United States",
    whole_across: tuple[str, ...] = ("donor",),
    whole_donors: Sequence[str] = (),
    whole_recipients: Sequence[str] = ("Kenya",),
    us_donors: Sequence[str] | None = None,
    whole_reduction: Reduction = Reduction.SUM,
    every_donor: bool = False,
    narrowed_from: str | None = None,
    narrowing: Sequence[Then] = (),
    then: Sequence[Then] = (),
) -> Plan:
    """The first plan of the module docstring, `all_oda` read from a source
    stating `whole`'s fields for the donors named `whole_donors` and the
    recipients named `whole_recipients`, and over those `whole_across`
    covers, and narrowed by `narrowing` before its total, which reduces by
    `whole_reduction`, and `us` from one stating `numerator`'s for the
    donors named `us_donors`, by default `donor` alone. With `every_donor`,
    `us` reads every donor's ODA and totals it by year, as a share of one
    sector's ODA does. With `narrowed_from`, the share's numerator is
    `donor` narrowed out of that retrieval, `us` or `all_oda`."""
    kenya = Entity(name="Kenya", role="recipient")
    named = (donor,) if us_donors is None else us_donors
    donors = (
        [kenya] if every_donor else [*(Entity(name=name, role="donor") for name in named), kenya]
    )
    return _compiled(
        decision,
        question="What share of ODA to Kenya came from the United States each year since 2015?",
        numerator=DatasetRequirement(
            id="us",
            decision_ids=["measure"],
            measure="ODA gross disbursements",
            entities=donors,
            across=("donor",) if every_donor else (),
            period=_PERIOD,
        ),
        whole=DatasetRequirement(
            id="all_oda",
            decision_ids=["measure"],
            measure="ODA gross disbursements",
            entities=[
                *(Entity(name=name, role="donor") for name in whole_donors),
                *(Entity(name=name, role="recipient") for name in whole_recipients),
            ],
            across=whole_across,
            period=_PERIOD,
        ),
        group_by=(GroupBy.PERIOD,),
        reduction=whole_reduction,
        sources={"us": numerator, "all_oda": whole},
        numerator_by_year=every_donor,
        narrowed_to=None if narrowed_from is None else (narrowed_from, donor),
        narrowing=narrowing,
        then=then,
    )


def us_share_per_recipient(
    decision: Callable[..., Decision],
    *,
    scope: DenominatorScope,
    numerator_recipients: int,
    whole_recipients: int,
    then: Sequence[Then] = (),
) -> Plan:
    """The second plan of the module docstring, dividing per recipient by
    `scope`, its numerator source holding `numerator_recipients` recipients
    and its denominator's `whole_recipients`, then read by each of `then`."""

    pooled = scope is DenominatorScope.POOLED

    def recipients(count: int) -> tuple[OpenEntityAxis, ...]:
        return (OpenEntityAxis(column="destination", member_count=count),)

    return _compiled(
        decision,
        question="What share of each recipient's ODA came from the United States?",
        numerator=DatasetRequirement(
            id="us",
            decision_ids=["measure"],
            measure="ODA gross disbursements",
            entities=[Entity(name="United States", role="donor")],
            across=("recipient",),
            period=_PERIOD,
        ),
        whole=DatasetRequirement(
            id="all_oda",
            decision_ids=["measure"],
            measure="ODA gross disbursements",
            across=("donor", "recipient") if pooled else ("recipient",),
            period=_PERIOD,
        ),
        group_by=(GroupBy.PERIOD,) if pooled else None,
        sources={
            "us": {"entities": (_USA,), "open_entity_axes": recipients(numerator_recipients)},
            "all_oda": (
                {
                    "open_entity_axes": (
                        OpenEntityAxis(column="origin", member_count=140),
                        *recipients(whole_recipients),
                    )
                }
                if pooled
                else {
                    **DAC_DONORS,
                    "entities": (),
                    "open_entity_axes": recipients(whole_recipients),
                }
            ),
        },
        axis="recipient",
        scope=scope,
        then=then,
    )


def _compiled(
    decision: Callable[..., Decision],
    *,
    question: str,
    numerator: DatasetRequirement,
    whole: DatasetRequirement,
    group_by: tuple[GroupBy, ...] | None,
    reduction: Reduction = Reduction.SUM,
    sources: Mapping[str, Mapping[str, Any]],
    axis: str | None = None,
    scope: DenominatorScope | None = None,
    numerator_by_year: bool = False,
    narrowed_to: tuple[str, str] | None = None,
    narrowing: Sequence[Then] = (),
    then: Sequence[Then] = (),
) -> Plan:
    """`numerator`'s share of `whole`, narrowed by each of `narrowing` in turn
    and totalled by `group_by` (`all_by_year`), or of `whole` itself when
    `group_by` is None, reducing by `reduction`. The numerator is totalled
    by year first (`us_by_year`) when `numerator_by_year`, or with
    `narrowed_to`, an input id and an entity, is that entity narrowed out of
    that input. The share is keyed by `axis` and divided by `scope` when
    given, then read by each of `then`, and the plan compiled with each
    dataset read from a source stating `sources`' fields."""
    steps: list[Then] = [*narrowing]
    divisor = narrowing[-1][0].id if narrowing else whole.id
    if group_by is not None:
        steps.append(_total("all_by_year", divisor, group_by, reduction))
        divisor = "all_by_year"
    share_of = numerator.id
    if numerator_by_year:
        steps.append(_total("us_by_year", numerator.id, (GroupBy.PERIOD,), Reduction.SUM))
        share_of = "us_by_year"
    numerator_entity = None
    if narrowed_to is not None:
        share_of, numerator_entity = narrowed_to
    decisions = [decision()]
    if scope is not None:
        decisions.append(decision(slot="scope", fixes=DecisionFixes(denominator_scope=scope)))
    if any("population" in requirement.decision_ids for requirement, _ in [*steps, *then]):
        decisions.append(
            decision(slot="population", fixes=DecisionFixes(among=RankAmong.POPULATION))
        )
    draft = Plan(
        question=question,
        decisions=decisions,
        evidence_requirements=[
            numerator,
            whole,
            *(requirement for requirement, _ in steps),
            ShareOfRequirement(
                id="share",
                decision_ids=["measure", *(["scope"] if scope is not None else [])],
                numerator=RequirementSide(input_id=share_of, entity=numerator_entity),
                denominator=RequirementSide(input_id=divisor),
                axis=axis,
                denominator_scope=scope,
            ),
            *(requirement for requirement, _ in then),
        ],
    )
    operations: list[Operation] = [
        RetrieveOperation(id=numerator.id, requirement_id=numerator.id, spec_id=SPEC_US),
        RetrieveOperation(id=whole.id, requirement_id=whole.id, spec_id=SPEC_ALL),
        *(operation for _, operation in steps),
        ShareOfOperation(
            id="share",
            requirement_id="share",
            numerator=Side(input=share_of, entity=numerator_entity),
            denominator=Side(input=divisor),
            axis=axis,
            denominator_scope=scope,
        ),
        *(operation for _, operation in then),
    ]
    terminal = operations[-1].id
    translation = Translation(operations=operations, terminal=[terminal])
    specs = {numerator.id: SPEC_US, whole.id: SPEC_ALL}
    workflow = Workflow(
        steps=[_step(operation, specs) for operation in operations],
        sources=[
            SelectedSource(
                spec_id=specs[requirement_id],
                sentence="Gross disbursements, 2015 to 2024.",
                has_data=True,
                **fields,
            )
            for requirement_id, fields in sources.items()
        ],
        evidence_bindings=[
            EvidenceBinding(requirement_id=requirement_id, step_id=requirement_id, spec_id=spec)
            for requirement_id, spec in specs.items()
        ],
        terminal_steps=[terminal],
    )
    return draft.with_translation(translation=translation, workflow=workflow)


def _total(
    step_id: str, input_id: str, group_by: tuple[GroupBy, ...], reduction: Reduction
) -> Then:
    """`input_id` reduced by `reduction` grouped by `group_by`, as a requirement
    and its operation."""
    return (
        AggregateRequirement(
            id=step_id,
            decision_ids=["measure"],
            input_id=input_id,
            reduction=reduction,
            group_by=group_by,
        ),
        AggregateOperation(
            id=step_id,
            requirement_id=step_id,
            input=input_id,
            reduction=reduction,
            group_by=group_by,
        ),
    )


def _step(operation: Operation, specs: Mapping[str, str]) -> WorkflowStep:
    """The workflow step carrying out `operation`, bound to the frames it reads."""
    if isinstance(operation, RetrieveOperation):
        return WorkflowStep(
            id=operation.id,
            operation_id=operation.id,
            tool=ToolId(server="data-mcp", name="get_data"),
            args={"spec_id": specs[operation.id]},
            input_schema_hash="hash",
        )
    if isinstance(operation, ShareOfOperation):
        inputs = {
            "numerator": operation.numerator.input,
            "denominator": operation.denominator.input,
        }
    else:
        assert isinstance(
            operation, AggregateOperation | FilterOperation | GrowthOperation | RankOperation
        )
        inputs = {"frame_ref": operation.input}
    return WorkflowStep(
        id=operation.id,
        operation_id=operation.id,
        tool=ToolId(server="analysis-mcp", name=operation.kind.value),
        bindings={
            name: Binding(from_step=step, output=ArtifactKind.FRAME_REF)
            for name, step in inputs.items()
        },
        input_schema_hash="hash",
    )

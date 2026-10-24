"""The translation `propose_plan` runs, and the translator core it drives.

The translator finds the data each dataset requirement needs (see
`translator.discovery`), proposes the operations that carry the plan's
requirements out, and hands both to `compiler.compile_plan`, which writes the
workflow and checks it. None of this needs a model:
`translator.mechanical.run_translate` drives it as code, and `build_translate`
wraps it in the function the planner's `propose_plan` calls on every plan it
records, reading that plan back from session state. The planner's model never
calls translation itself, so it spends no generation on it.

Translation answers with one verdict from a closed set, which `propose_plan`
returns under `translation`: `ok` once the plan compiled, or the `Referral`
the run stopped on, flattened so its own `outcome`
(`translator.referral.OUTCOME_OF`) is the verdict's: `choice_required` with
the candidate sources, `invalid_input` for a requirement the planner repairs,
`unsupported`, `transient` or `internal`. The referral is also saved in
session state, for the planner seam's stall watchdog and `declare_ready`'s
gate. A plan that is not a translatable draft answers with the typed problem
that says so.

Several candidate sources for one requirement close on a `source_ambiguous`
referral carrying them. The planner records the choice as a decision or puts
it to the user.

None of the translator's work is confirmation-gated. Discovery reads
metadata, and compiling writes a workflow that nothing runs before the user
approves it. The one read of observations is a workflow's newest years. A
workflow with a rolling step has them read through its own retrievals and
`resolve_anchor`, as its run reads them (`executor.run.read_anchors`), so the
years the user approves are the years the run reads. That read shows the user
nothing and records nothing. The approval gate is the harness workflow's
`approve` node.

An unreadable surface comes back as a typed problem the planner can act on.
These run inside ADK tool calls, where a raise ends the planner entry on a
final `planner_tool_failed` refusal, set in `harness.planner_seam`.
`DuplicateToolNameError` propagates, since a tool name served twice is a
misconfiguration retrying cannot resolve.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import suppress
from enum import StrEnum
from typing import assert_never, cast

from common.telemetry import record_event
from compiler.compile import CompileIssue, CompileIssueCode, CompileResult, compile_plan
from compiler.rolling import anchor_groups
from compiler.surface import ToolSurface, ToolSurfaceUnavailableError
from executor.outcomes import (
    AnchorResult,
    MethodChoiceNeeded,
    RediscoveryNeeded,
    Refused,
    StopCause,
    failed_step,
    stop_cause,
)
from executor.run import AnchorRead, read_anchors
from google.adk.tools.tool_context import ToolContext
from planner.operations import Translation
from planner.plan import (
    DatasetRequirement,
    HistorySplit,
    Plan,
    Restatement,
    UnsupportedOperation,
    authored_text,
    discovery_time_left,
    load_plan,
    origin_of,
    plan_revision,
    record_optional_decline,
    record_unsupported,
    restated_to_anchors,
    restated_translation,
    restatements,
    save_plan,
    split_of,
)
from planner.records import EvidenceIssueCode, RollingPlanError
from planner.shapes import pooling_remedy
from planner.tool_results import Action, Outcome, ProblemCode, ok, problem, referred
from planner.workflow import SelectedSource, Workflow
from pydantic import JsonValue, ValidationError

from translator.discovery import (
    DiscoveryLog,
    apply_definitions,
    build_discovery_tools,
    compact_source,
    current_draft,
    dataset_keys,
    described_sources,
    discovery_lock,
    fetch_definitions,
    load_log,
    name_pinned_groups,
    requirement_revision,
    resolve_entity_refs,
    resolving_once,
)
from translator.mechanical import TranslatorCore, run_translate
from translator.referral import (
    MAY_CARRY_CANDIDATES,
    Referral,
    ReferralKind,
    StalePin,
    WithheldSource,
    load_referral,
    save_referral,
)

logger = logging.getLogger(__name__)

# What `build_translate` builds: the planner's tool context in, the verdict
# `propose_plan` answers under `translation` out.
type Translate = Callable[[ToolContext], Awaitable[dict[str, object]]]


def _other_stale_pins(plan: Plan, log: DiscoveryLog, requirement_id: str) -> tuple[StalePin, ...]:
    """Every dataset requirement of `plan` but `requirement_id` whose pin discovery did not find.

    Read off `log` (`DiscoveryLog.stale_pin_sources`), each with the sources
    found in its place, in the plan's order, less those withheld for
    dropping its pinned source's scope (`DiscoveryLog.scope_kept`).
    """
    stale: list[StalePin] = []
    for requirement in plan.evidence_requirements:
        if (
            not isinstance(requirement, DatasetRequirement)
            or requirement.id == requirement_id
            or requirement.source_spec_id is None
        ):
            continue
        found = log.stale_pin_sources(
            requirement_revision(requirement), requirement.id, requirement.source_spec_id
        )
        if found is not None:
            scope = log.scope_kept(requirement, found)
            stale.append(
                StalePin(
                    requirement_id=requirement.id,
                    candidates=scope.offered,
                    withheld=scope.withheld,
                    repair=scope.repair,
                )
            )
    return tuple(stale)


def _referral_payload(referral: Referral) -> dict[str, object]:
    """Return `referral` as the planner reads it: its `outcome`, and each candidate,
    its own and each stale pin's, rendered by `compact_source`.

    `no_data`'s candidates are the nearest series discovery found, not a
    source the planner may pin without a fresh search: rendered without
    `spec_id` so the only way one reaches a pinnable choice is a real
    `source_ambiguous` or `pinned_source_not_found` referral. `budget_spent`
    and `not_issued` appear only when set, the cases the planner seam ends
    the entry on, and `source_mismatch` and `stop_code` likewise. `withheld`
    appears only when a source is withheld, each rendered without its
    `spec_id` beside the scope it drops, since no pin may name it, and
    `repair` only when one is stated.
    """
    selectable = referral.kind is not ReferralKind.NO_DATA
    excluded = {"candidates", "stale_pins", "withheld"}
    if referral.repair is None:
        excluded.add("repair")
    if not referral.budget_spent:
        excluded.add("budget_spent")
    if not referral.not_issued:
        excluded.add("not_issued")
    if not referral.source_mismatch:
        excluded.add("source_mismatch")
    if referral.stop_code is None:
        excluded.add("stop_code")
    payload = referral.model_dump(mode="json", exclude=excluded) | {
        "outcome": referral.outcome.value,
        "candidates": [
            compact_source(source, selectable=selectable) for source in referral.candidates
        ],
    }
    if referral.withheld:
        payload["withheld"] = _withheld_payload(referral.withheld)
    if referral.stale_pins:
        payload["stale_pins"] = [
            {
                "requirement_id": pin.requirement_id,
                "candidates": [
                    compact_source(source, selectable=True) for source in pin.candidates
                ],
                **({"withheld": _withheld_payload(pin.withheld)} if pin.withheld else {}),
                **({"repair": pin.repair} if pin.repair else {}),
            }
            for pin in referral.stale_pins
        ]
    return payload


def _withheld_payload(withheld: Sequence[WithheldSource]) -> list[dict[str, object]]:
    """Each withheld source as the planner reads it: described without its `spec_id`, beside the scope it drops."""
    return [
        compact_source(one.source, selectable=False) | {"dropped": list(one.dropped)}
        for one in withheld
    ]


class IssueResolution(StrEnum):
    """What a compile issue asks for, listed in the order `_compile_failure_action` prefers.

    `DISCOVER_AGAIN` is a fix the loop applies itself, for a translation that
    names a source discovery has not settled for its requirement. `INTERNAL`
    is a fault no plan change repairs. `build_translation` copies a valid
    plan's requirements verbatim, so a structural issue in the translation is
    a translator fault, and a capability or schema issue is the services'
    configuration. `UNSUPPORTED` is an operation no tool performs, including
    one that sets a unit-less side against a side in a unit.
    `UNMEETABLE` is a requirement the user decides how to change, and
    `REPAIR` one the planner repairs itself, as `_PERMITTED_REPAIRS` names.
    An `UNMEETABLE` code in `_DECLINES_EARNED` also earns a decline.
    """

    DISCOVER_AGAIN = "discover_again"
    INTERNAL = "internal"
    UNSUPPORTED = "unsupported"
    UNMEETABLE = "unmeetable"
    REPAIR = "repair"


def issue_resolution(code: CompileIssueCode) -> IssueResolution:
    """The one resolution every compile issue code maps to."""
    match code:
        case (
            CompileIssueCode.SOURCE_UNSELECTED
            | CompileIssueCode.SOURCE_NOT_DISCOVERED_FOR_REQUIREMENT
            | CompileIssueCode.SOURCE_PIN_VIOLATED
        ):
            return IssueResolution.DISCOVER_AGAIN
        case CompileIssueCode.SOURCE_UNUSABLE | CompileIssueCode.FRAME_UNITS_IRRECONCILABLE:
            return IssueResolution.UNMEETABLE
        case CompileIssueCode.NO_CAPABILITY | CompileIssueCode.FRAME_UNIT_UNBOUND:
            return IssueResolution.UNSUPPORTED
        case (
            CompileIssueCode.INPUT_SHAPE_UNSUPPORTED
            | CompileIssueCode.RANK_AXIS_REQUIRED
            | CompileIssueCode.SHARE_AXIS_REQUIRED
            | CompileIssueCode.SHARE_KEY_UNKNOWN
            | CompileIssueCode.QUOTIENT_AXIS_REQUIRED
            | CompileIssueCode.QUOTIENT_KEY_UNKNOWN
            | CompileIssueCode.KEYS_DISJOINT
            | CompileIssueCode.SOURCE_WIDER_THAN_REQUIREMENT
            | CompileIssueCode.WINDOW_OUTSIDE_PERIOD
            | CompileIssueCode.ROLLING_NEEDS_OPEN_PERIOD
            | CompileIssueCode.ROLLING_POPULATION_UNUSABLE
            | CompileIssueCode.ROLLING_ANCHOR_BEHIND
            | CompileIssueCode.ENTITY_REQUIRED
            | CompileIssueCode.UNKNOWN_ENTITY
            | CompileIssueCode.ENTITY_AMBIGUOUS
            | CompileIssueCode.ENTITY_UNRESOLVED
            | CompileIssueCode.ENTITY_NOT_ON_FRAME
            | CompileIssueCode.DEGENERATE_SIDES
            | CompileIssueCode.SUM_SIDES_OVERLAP
            | CompileIssueCode.SUM_OVERLAP_UNDECIDED
            | CompileIssueCode.FAMILY_UNSUMMABLE
            | CompileIssueCode.SUM_UNITS_DIFFER
            | CompileIssueCode.SHARE_UNITS_DIFFER
            | CompileIssueCode.FRAME_UNITS_DIFFER
            | CompileIssueCode.FRAME_UNIT_FORMS_DIFFER
            | CompileIssueCode.FRAME_PRICE_BASES_DIFFER
            | CompileIssueCode.PERIODS_DISJOINT
            | CompileIssueCode.BAD_ARGS
        ):
            return IssueResolution.REPAIR
        case (
            CompileIssueCode.DUPLICATE_ID
            | CompileIssueCode.REQUIREMENT_UNKNOWN
            | CompileIssueCode.REQUIREMENT_UNSATISFIED
            | CompileIssueCode.DUPLICATE_SATISFACTION
            | CompileIssueCode.KIND_MISMATCH
            | CompileIssueCode.OPERATION_MISMATCH
            | CompileIssueCode.DANGLING_INPUT
            | CompileIssueCode.INPUT_IS_TERMINAL
            | CompileIssueCode.CYCLE
            | CompileIssueCode.UNKNOWN_TERMINAL
            | CompileIssueCode.OPERATION_UNREACHABLE
            | CompileIssueCode.AMBIGUOUS_CAPABILITY
            | CompileIssueCode.UNKNOWN_TOOL
            | CompileIssueCode.BAD_TOOL_SCHEMA
            | CompileIssueCode.CAPABILITY_SCHEMA_DRIFT
            | CompileIssueCode.CAPABILITY_UNPINNED
            | CompileIssueCode.ARTIFACT_MISMATCH
        ):
            return IssueResolution.INTERNAL
        case _:
            assert_never(code)


# What the planner may change for each `IssueResolution.REPAIR` code. The
# issue's own reason says what was found.
_PERMITTED_REPAIRS: Mapping[CompileIssueCode, str] = {
    CompileIssueCode.INPUT_SHAPE_UNSUPPORTED: (
        "change the calculation, or the requirement it reads, so its input has a shape the "
        "calculation takes"
    ),
    CompileIssueCode.RANK_AXIS_REQUIRED: "set the rank's `axis` to the role it ranks on",
    CompileIssueCode.SHARE_AXIS_REQUIRED: (
        "set the share's `axis` to the role it is taken per, or name the numerator's "
        f"`entity`, or {pooling_remedy(one_figure=True, each_side=True)}"
    ),
    CompileIssueCode.SHARE_KEY_UNKNOWN: "set the share's `axis` to a role both sides carry",
    CompileIssueCode.QUOTIENT_AXIS_REQUIRED: (
        "set the quotient's `axis` to the role phrase its numerator is keyed by, or name the "
        "numerator's `entity`; set `denominator_axis` to the role phrase the denominator's "
        "dataset gives the same entities, or leave it out to divide every key by one total"
    ),
    CompileIssueCode.QUOTIENT_KEY_UNKNOWN: (
        "set the quotient's `axis` to a role phrase its numerator carries, and "
        "`denominator_axis` to the role phrase the denominator's dataset gives the same "
        "entities, or leave `denominator_axis` out to divide every key by one total"
    ),
    CompileIssueCode.KEYS_DISJOINT: (
        "state at least one entity both keyed sides hold on each side's requirement; a "
        "division can instead divide every key by one total: a quotient with no "
        "`denominator_axis`, a share with a `pooled` denominator"
    ),
    CompileIssueCode.SOURCE_WIDER_THAN_REQUIREMENT: (
        "name the entities the requirement needs, or filter the series to them before the "
        "calculation pools them"
    ),
    CompileIssueCode.WINDOW_OUTSIDE_PERIOD: (
        "move the growth window inside the requirement's period, or widen the period"
    ),
    CompileIssueCode.ROLLING_NEEDS_OPEN_PERIOD: (
        "state `period_end_open` on the dataset requirement the rolling period reads, or drop "
        "`rolling` for fixed years"
    ),
    CompileIssueCode.ROLLING_POPULATION_UNUSABLE: (
        "make the filters and entities along each calculation keep at least one series of "
        "the dataset, at one level of place, or drop `rolling` for fixed years"
    ),
    CompileIssueCode.ROLLING_ANCHOR_BEHIND: (
        "state the rolling years from the newest year the reason names, changing what the "
        "rule it quotes refuses, or drop `rolling` for fixed years"
    ),
    # The calculation is unknown here, so this offers the per-year pooling every
    # reader takes; the issue's reason adds one figure where its reader takes one.
    CompileIssueCode.ENTITY_REQUIRED: (
        "name the entity each side reads, among those its requirement names on one side of "
        f"the flow, or {pooling_remedy(one_figure=False)}"
    ),
    CompileIssueCode.UNKNOWN_ENTITY: (
        "name only entities the requirement it reads states, or add the entity to that requirement"
    ),
    CompileIssueCode.ENTITY_AMBIGUOUS: "give the entity's role, so the name matches one column",
    CompileIssueCode.ENTITY_UNRESOLVED: (
        "name an entity the source resolves, or state the requirement's `across` axis"
    ),
    CompileIssueCode.ENTITY_NOT_ON_FRAME: (
        "name an entity under a role the frame the calculation reads still carries, or read "
        "the entity from the requirement that retrieves it"
    ),
    CompileIssueCode.DEGENERATE_SIDES: (
        "make the two sides differ, by entity or by the requirement each reads"
    ),
    CompileIssueCode.SUM_SIDES_OVERLAP: (
        "add two sources that hold different flows; to total entities of one series, aggregate it"
    ),
    CompileIssueCode.SUM_OVERLAP_UNDECIDED: (
        "add two sources whose families bind one property to different codes, such as two "
        "sectors; to total entities of one series, aggregate it"
    ),
    CompileIssueCode.FAMILY_UNSUMMABLE: (
        "set the requirement's `stat_vars` to 'separate', or ask for a measure that names one "
        "value on each axis the reason lists"
    ),
    CompileIssueCode.SUM_UNITS_DIFFER: (
        "state the same unit on both sides' requirements, or on neither"
    ),
    CompileIssueCode.SHARE_UNITS_DIFFER: (
        "for one measure per unit of another, replace the share with a `quotient_of`, which "
        "divides the two and states the result in a composed unit; for a part of its whole, "
        "state the same unit on both sides' requirements"
    ),
    CompileIssueCode.FRAME_UNITS_DIFFER: (
        "state the same unit on both sides' requirements, so each source is retrieved in it"
    ),
    CompileIssueCode.FRAME_PRICE_BASES_DIFFER: (
        "state one unit on both sides' requirements, either one currency at current prices or "
        "one currency at constant prices of one base year, so each source is retrieved in it, "
        "and keep a pinned source's own unit on its side"
    ),
    CompileIssueCode.FRAME_UNIT_FORMS_DIFFER: (
        "on each requirement the reason names, state the unit it gives and leave its "
        "`source_spec_id` unset"
    ),
    CompileIssueCode.PERIODS_DISJOINT: (
        "give the two sides' requirements periods that share at least one year"
    ),
    CompileIssueCode.BAD_ARGS: (
        "change the requirement field the refused argument is written from to a value the "
        "tool accepts"
    ),
}


# The `unsupported` answer an `IssueResolution.UNMEETABLE` code also stands
# for. Two sides that can never share a unit leave the user the choice of
# what to change, and if the answer is to show them apart, the plan declines
# the operation that paired them.
_DECLINES_EARNED: Mapping[CompileIssueCode, UnsupportedOperation] = {
    CompileIssueCode.FRAME_UNITS_IRRECONCILABLE: UnsupportedOperation.NO_OPERATION_COVERS,
}


def permitted_repair(code: CompileIssueCode) -> str:
    """What the planner may change for `code`, an `IssueResolution.REPAIR` code."""
    return _PERMITTED_REPAIRS[code]


def _referral_action(
    issue: CompileIssue, resolution: IssueResolution, requirement_of: Mapping[str, str]
) -> Action:
    """The `return_to_planner` call that closes on `issue`.

    A repair or a gap names the requirement the failing operation carries
    out (`requirement_of`), or else the requirement the issue names. An issue
    naming neither is a fault, and closes as one. An unmeetable requirement
    whose stated source the source found does not name is marked
    `source_mismatch`, so the referral offers the sources discovery found.
    An unmeetable issue that also earns a decline (`_DECLINES_EARNED`) says
    so in its reason, the one field the planner's instruction keys that
    decline on.
    """
    about = requirement_of.get(issue.operation_id or "") or issue.requirement_id
    if resolution is IssueResolution.UNMEETABLE and issue.requirement_id is not None:
        earned = _DECLINES_EARNED.get(issue.code)
        arguments: dict[str, JsonValue] = {
            "kind": ReferralKind.REQUIREMENT_UNMEETABLE.value,
            "requirement_id": issue.requirement_id,
            "reason": issue.reason
            if earned is None
            else f"{issue.reason}. This earns the decline `{earned.value}`, which the plan "
            "may make if the user would rather see the two apart",
        }
        if issue.finding is EvidenceIssueCode.SOURCE_MISMATCH:
            arguments["source_mismatch"] = True
    elif resolution is IssueResolution.UNSUPPORTED and about is not None:
        arguments = {
            "kind": ReferralKind.NO_OPERATION_COVERS.value,
            "requirement_id": about,
            "reason": issue.reason,
        }
    elif resolution is IssueResolution.REPAIR and about is not None:
        arguments = {
            "kind": ReferralKind.REQUIREMENT_NEEDS_REPAIR.value,
            "requirement_id": about,
            "reason": issue.reason,
            "constraint": issue.code.value,
            "repair": _PERMITTED_REPAIRS[issue.code],
        }
    else:
        arguments = {
            "kind": ReferralKind.INTERNAL_FAILURE.value,
            "reason": f"{issue.code.value}: {issue.reason}",
        }
    return Action(call="return_to_planner", arguments=arguments)


def _compile_failure_action(
    issues: Sequence[CompileIssue], *, requirement_of: Mapping[str, str]
) -> Action:
    """Return what to call next to move past a failed `propose_operations`.

    Every issue maps to one `IssueResolution`. The first issue of the most
    preferred resolution present decides, so a source to discover again is
    fixed before anything is referred, and a fault is reported before a
    repair the fault would make moot. `requirement_of` maps each operation
    id of the translation to the requirement it carries out.
    """
    resolution, issue = _deciding_issue(issues)
    if resolution is IssueResolution.DISCOVER_AGAIN:
        return Action(call="discover_source", arguments={"requirement_id": issue.requirement_id})
    return _referral_action(issue, resolution, requirement_of)


def _deciding_issue(issues: Sequence[CompileIssue]) -> tuple[IssueResolution, CompileIssue]:
    """The issue `_compile_failure_action` closes on, with its resolution.

    The first issue of the most preferred resolution present, where an
    issue that earns a decline (`_DECLINES_EARNED`) goes before others of
    its resolution, so the decline is recorded with the referral.
    """
    order = list(IssueResolution)
    resolved = [(issue_resolution(issue.code), issue) for issue in issues]
    return min(
        resolved,
        key=lambda pair: (order.index(pair[0]), pair[1].code not in _DECLINES_EARNED),
    )


def _earned_decline(issues: Sequence[CompileIssue]) -> UnsupportedOperation | None:
    """The `unsupported` answer the referral `issues` close on also stands for, if any."""
    return _DECLINES_EARNED.get(_deciding_issue(issues)[1].code)


def _not_compiled(
    result: CompileResult, translation: Translation, tool_context: ToolContext
) -> dict[str, object]:
    """The `COMPILE_FAILED` problem for `result`'s issues, with the action that moves past them.

    A decline the closing issue earns (`_earned_decline`) is recorded as
    optional, for the rest of the planner entry.
    """
    if (declinable := _earned_decline(result.issues)) is not None:
        record_optional_decline(tool_context.state, declinable)
    return problem(
        ProblemCode.COMPILE_FAILED,
        f"This translation did not compile: {len(result.issues)} issue(s) found.",
        outcome=Outcome.INVALID_INPUT,
        action=_compile_failure_action(
            result.issues,
            requirement_of={op.id: op.requirement_id for op in translation.operations},
        ),
        detail={
            "issues": [issue.model_dump(mode="json", exclude_none=True) for issue in result.issues],
            "warnings": [warning.model_dump(mode="json") for warning in result.warnings],
        },
    )


def restated_to_the_data(
    plan: Plan, translation: Translation, workflow: Workflow, read: Sequence[AnchorRead]
) -> tuple[Plan, Translation] | CompileResult:
    """`plan` and `translation` stating each group's rolling years from the newest year `read` found before it.

    A group whose newest year is the one its rolling years state, or later,
    keeps its years, since every plan rule holds of them moved later and the
    run reads them so (`executor.run._Run._check_anchor_held`). So does a group
    `read` holds a stop for. A group behind moves to its newest year
    (`planner.plan.restated_to_anchors`, `restated_translation`). A plan
    rule that refuses the moved plan comes back as a `ROLLING_ANCHOR_BEHIND`
    issue on the requirement the rule names, in the first group whose move
    alone the rules refuse (`_refused_move`).
    """
    steps = {step.id: step for step in workflow.steps}
    requirement_of = {op.id: op.requirement_id for op in translation.operations}

    def anchors(moving: Sequence[AnchorRead]) -> dict[str, int]:
        years: dict[str, int] = {}
        for group in read:
            moved = group in moving and isinstance(group.result, AnchorResult)
            year = group.result.period if moved else group.planned
            for step_id in group.steps:
                years[requirement_of[steps[step_id].operation_id]] = int(year)
        return years

    behind = [group for group in read if group.behind]
    try:
        return (
            restated_to_anchors(plan, anchors=anchors(behind)),
            restated_translation(translation, plan=plan, anchors=anchors(behind)),
        )
    except ValueError as exc:
        group, refusal = _refused_move(plan, behind, exc, anchors=anchors)
        requirements = [requirement_of[steps[step_id].operation_id] for step_id in group.steps]
        named = _named_requirements(refusal)
        requirement_id = next((r for r in requirements if r in named), requirements[0])
        return CompileResult(
            issues=(
                _anchor_behind(
                    group,
                    _rule_refusal(refusal),
                    operation_id=next(
                        op.id
                        for op in translation.operations
                        if op.requirement_id == requirement_id
                    ),
                    requirement_id=requirement_id,
                ),
            )
        )


def _refused_move(
    plan: Plan,
    behind: Sequence[AnchorRead],
    refused: ValueError,
    *,
    anchors: Callable[[Sequence[AnchorRead]], Mapping[str, int]],
) -> tuple[AnchorRead, ValueError]:
    """The first group of `behind` whose move alone a plan rule refuses, with the refusal.

    Where every group moves alone and only their moves together are
    refused, `refused` stands, on the first group behind.
    """
    for group in behind:
        try:
            restated_to_anchors(plan, anchors=anchors([group]))
        except ValueError as alone:
            return group, alone
    return behind[0], refused


def _named_requirements(refusal: ValueError) -> tuple[str, ...]:
    """The requirements a plan rule's `RollingPlanError` under `refusal` names, if one raised it."""
    raised: list[object] = [refusal]
    if isinstance(refusal, ValidationError):
        raised = [error.get("ctx", {}).get("error") for error in refusal.errors(include_url=False)]
    return tuple(
        requirement_id
        for cause in raised
        if isinstance(cause, RollingPlanError)
        for requirement_id in cause.requirement_ids
    )


def _anchor_behind(
    group: AnchorRead, refusal: str, *, operation_id: str, requirement_id: str
) -> CompileIssue:
    """The `ROLLING_ANCHOR_BEHIND` issue for `group`'s years stated from its newest year, refused for `refusal`."""
    assert isinstance(group.result, AnchorResult)
    newest = group.result.period
    return CompileIssue(
        code=CompileIssueCode.ROLLING_ANCHOR_BEHIND,
        reason=(
            f"the newest year at least half of the series these rolling years read hold is "
            f"{newest}, before {group.planned}, the year they state, and the plan stated from "
            f"{newest} is refused: {refusal}"
        ),
        operation_id=operation_id,
        requirement_id=requirement_id,
    )


def _restatement_not_compiled(
    result: CompileResult, translation: Translation, workflow: Workflow, read: Sequence[AnchorRead]
) -> CompileResult:
    """`result`, the failed compile of a restated translation, its issues told as `ROLLING_ANCHOR_BEHIND`.

    Only years moved, so each issue is the restatement's. It is told on the
    group behind holding its operation, or else the first group behind,
    with that group's newest year and the compiler's own words, on the
    requirement its operation carries out.
    """
    operation_of = {step.id: step.operation_id for step in workflow.steps}
    requirement_of = {op.id: op.requirement_id for op in translation.operations}
    behind = [group for group in read if group.behind]
    issues: list[CompileIssue] = []
    for issue in result.issues:
        group = next(
            (g for g in behind if issue.operation_id in {operation_of[s] for s in g.steps}),
            behind[0],
        )
        operation_id = (
            issue.operation_id
            if issue.operation_id in requirement_of
            else operation_of[group.steps[0]]
        )
        issues.append(
            _anchor_behind(
                group,
                issue.reason,
                operation_id=operation_id,
                requirement_id=requirement_of[operation_id],
            )
        )
    return CompileResult(issues=tuple(issues), warnings=result.warnings)


def _run_would_stop(
    stop: Refused, *, workflow: Workflow, translation: Translation
) -> dict[str, object]:
    """The `RUN_WOULD_STOP` problem for a read of the newest year that stopped where the run would.

    `detail` holds the step the run's caller would be told it failed at
    (`executor.outcomes.failed_step`): its `step_id`, the run's own `code`
    and `message`, and the requirement that step carries out. Its `action`
    refers the plan back with that code as `stop_code`, by whose stop it is
    (`executor.outcomes.stop_cause`): as `surface_unavailable` when a later
    try can get past it, as an internal failure when it is the harness's,
    and as `requirement_unmeetable` on that requirement when it is the
    data's, for the user to decide otherwise. A data stop that names no
    requirement is an internal failure too. The stop is logged with its
    code.
    """
    failed = failed_step(stop)
    code, step_id = str(failed["code"]), failed["step_id"]
    requirement_id = stop.requirement_id
    if requirement_id is None and step_id is not None:
        requirement_of = {op.id: op.requirement_id for op in translation.operations}
        operation_of = {step.id: step.operation_id for step in workflow.steps}
        requirement_id = requirement_of.get(operation_of.get(step_id, ""))
    cause = stop_cause(stop)
    logger.warning(
        "translation: the newest-year read stopped code=%s step_id=%s requirement_id=%s cause=%s",
        code,
        step_id,
        requirement_id,
        cause,
    )
    reason = (
        "reading the newest year the rolling years count from stopped where the run would: "
        f"{failed['message']}"
    )
    arguments: dict[str, JsonValue] = {"reason": reason, "stop_code": code}
    if cause is StopCause.TRANSIENT:
        arguments["kind"] = ReferralKind.SURFACE_UNAVAILABLE.value
    elif cause is StopCause.DATA and requirement_id is not None:
        arguments["kind"] = ReferralKind.REQUIREMENT_UNMEETABLE.value
        arguments["requirement_id"] = requirement_id
    else:
        arguments["kind"] = ReferralKind.INTERNAL_FAILURE.value
    return problem(
        ProblemCode.RUN_WOULD_STOP,
        f"Reading the newest year the data holds stopped where the run would: {code}.",
        outcome=Outcome.REFUSED,
        action=Action(call="return_to_planner", arguments=arguments),
        detail={
            "code": code,
            "message": failed["message"],
            "step_id": step_id,
            "requirement_id": requirement_id,
        },
    )


def _rule_refusal(exc: ValueError) -> str:
    """What the plan rule that raised `exc` says, without pydantic's framing."""
    if isinstance(exc, ValidationError):
        return "; ".join(error["msg"] for error in exc.errors(include_url=False))
    return str(exc)


def _translatable_draft(tool_context: ToolContext) -> Plan | dict[str, object]:
    """Return the draft plan `propose_operations` may compile, or the problem that stops it.

    Needs a draft plan with no referral pending on it and at least one
    evidence requirement to carry out. No translator tool settles a pending
    referral, so there is no action to point at. The planner reads it and
    proposes or rejects.
    """
    plan = current_draft(tool_context)
    if isinstance(plan, dict):
        return plan
    pending = load_referral(tool_context.state)
    if pending is not None:
        return problem(
            ProblemCode.REFERRAL_PENDING,
            f"A {pending.kind.value} referral is pending on this plan; the planner settles "
            "it before the plan is translated again.",
            outcome=Outcome.REFUSED,
            detail={"kind": pending.kind.value, "requirement_id": pending.requirement_id},
        )
    if not plan.evidence_requirements:
        return problem(
            ProblemCode.INVALID_PLAN,
            "This plan states no evidence requirements; the planner records them "
            "before a plan can be translated.",
            outcome=Outcome.REFUSED,
        )
    return plan


def _logged_reference(reason: str) -> str:
    """Log an internal translation failure and return the reference it is logged under."""
    reference = uuid.uuid4().hex[:12]
    logger.error("translation internal failure %s: %s", reference, reason)
    return reference


async def _read_within_entry(
    workflow: Workflow, plan: Plan, *, surface: ToolSurface
) -> tuple[AnchorRead, ...] | MethodChoiceNeeded | Refused | RediscoveryNeeded | None:
    """`executor.run.read_anchors` over `workflow`, or `None` when the planner entry has no time for it.

    Inside a planner entry the read may take the time the entry has left
    (`planner.plan.discovery_time_left`). A read that time cuts short, or
    one there is no time to start, is logged and recorded on the trace as
    skipped, with its groups, the time it had and the time it took.
    """
    time_left = discovery_time_left()
    began = time.monotonic()
    if time_left is None or time_left > 0:
        deadline = asyncio.timeout(time_left)
        try:
            async with deadline:
                return await read_anchors(
                    workflow,
                    revision=plan_revision(plan),
                    requirements=[
                        r for r in plan.evidence_requirements if isinstance(r, DatasetRequirement)
                    ],
                    surface=surface,
                )
        except TimeoutError:
            if not deadline.expired():
                raise
    metadata: dict[str, str | int | float | bool] = {
        "groups": len(anchor_groups(workflow)),
        "time_left_s": round(time_left or 0.0, 3),
        "elapsed_s": round(time.monotonic() - began, 3),
    }
    logger.warning(
        "translation: the newest-year read was skipped for want of time groups=%s "
        "time_left_s=%s elapsed_s=%s",
        metadata["groups"],
        metadata["time_left_s"],
        metadata["elapsed_s"],
    )
    record_event("translation.anchor_read_skipped", metadata=metadata)
    return None


def _groups_read(
    read: tuple[AnchorRead, ...] | MethodChoiceNeeded | Refused | RediscoveryNeeded | None,
) -> tuple[AnchorRead, ...] | Refused:
    """The groups `read` resolved, or the stop that answers the translation.

    A refusal answers it, the whole read's or the first group's. A read
    skipped, or one stopping on a method choice or a source to discover
    again, resolves no group, and the run meets those itself.
    """
    if read is None or isinstance(read, MethodChoiceNeeded | RediscoveryNeeded):
        return ()
    if isinstance(read, Refused):
        return read
    for group in read:
        if isinstance(group.result, Refused):
            return group.result
    return read


def _restated_answer(restated: Sequence[Restatement]) -> dict[str, object]:
    """`restated` as the `ok` answer carries it, under `restated`, or nothing when no year moved."""
    if not restated:
        return {}
    return {"restated": [restatement.model_dump(mode="json") for restatement in restated]}


def build_translator_core(*, surface: ToolSurface) -> TranslatorCore:
    """Build the translator's three core functions, for `build_translate` or a test to drive.

    `build_translate` drives the three in Python through
    `translator.mechanical.run_translate`. They are built here so a test can
    exercise `propose_operations`'s and `return_to_planner`'s own logic
    without going through `build_translate`.
    """

    async def stated_from_the_data(
        plan: Plan,
        translation: Translation,
        result: CompileResult,
        *,
        compiled: Callable[[Plan, Translation], CompileResult],
        tool_context: ToolContext,
    ) -> tuple[Plan, Translation, CompileResult] | dict[str, object]:
        """`plan`, `translation` and `result`, its compiled workflow, with each group's rolling years stated from the newest year the data holds.

        A workflow with a rolling step has its newest years read as its run
        reads them (`executor.run.read_anchors`), over its own retrievals,
        before the user approves anything, within the time the planner
        entry has left (`planner.plan.discovery_time_left`). Every group on
        or after the year its rolling years state keeps them, and so does
        every group when the read runs out of time, and a group whose read
        meets a method choice or a source to discover again, since the run
        meets those as it does any plan and refuses a newest year before the
        stated one (`executor.run._Run._check_anchor_held`). Otherwise the
        restated plan and translation (`restated_to_the_data`) are compiled
        again. A read that stops otherwise answers `_run_would_stop`, and a
        restatement a plan rule refuses, or one that does not compile,
        answers `_not_compiled` with `ROLLING_ANCHOR_BEHIND` issues.
        """
        workflow = result.workflow
        assert workflow is not None
        if not anchor_groups(workflow):
            return plan, translation, result
        read = _groups_read(await _read_within_entry(workflow, plan, surface=surface))
        if isinstance(read, Refused):
            return _run_would_stop(read, workflow=workflow, translation=translation)
        if not any(group.behind for group in read):
            return plan, translation, result
        restated = restated_to_the_data(plan, translation, workflow, read)
        if isinstance(restated, CompileResult):
            return _not_compiled(restated, translation, tool_context)
        moved_plan, moved_translation = restated
        recompiled = compiled(moved_plan, moved_translation)
        if recompiled.workflow is None:
            return _not_compiled(
                _restatement_not_compiled(recompiled, moved_translation, workflow, read),
                moved_translation,
                tool_context,
            )
        return moved_plan, moved_translation, recompiled

    async def propose_operations(
        translation: Translation, tool_context: ToolContext
    ) -> dict[str, object]:
        """Compile `translation` against the recorded draft, the sources discovery
        recorded and the live tool surface, and record it on the plan if it
        compiles, its rolling years stated from the newest year the data
        holds (`stated_from_the_data`). Answers `ok` with the plan's new
        status, version, revision, sources and steps, and under `restated`
        each requirement whose years moved (`planner.plan.restatements`), or
        the problem that stopped it, carrying every compile issue and, when
        known, the action that moves past it; the plan is then left
        untouched, and a decline the closing issue earns
        (`_earned_decline`) is recorded as optional, for the rest of the
        planner entry."""
        plan = _translatable_draft(tool_context)
        if isinstance(plan, dict):
            return plan
        try:
            schemas = await surface.list_tools()
        except ToolSurfaceUnavailableError as exc:
            return problem(ProblemCode.SURFACE_UNAVAILABLE, str(exc), outcome=Outcome.REFUSED)

        requirement_keys, pinned, across = dataset_keys(plan)

        def settled(log: DiscoveryLog) -> dict[str, SelectedSource]:
            """The sources `log` settles for this plan."""
            return log.selectable(
                requirement_keys=requirement_keys,
                pinned=pinned,
                across=across,
            )

        # The log read that decides what to describe takes `discovery_lock`,
        # the same discipline `discover_source`'s own body follows; the
        # `describe` round trip `fetch_definitions` runs does not, so the
        # lock is released for it and re-taken only to apply what it found.
        async with discovery_lock(tool_context.state):
            log = load_log(tool_context.state)
            discovered = settled(log)
            definite_ids = log.definite_source_ids(requirement_keys=requirement_keys, pinned=pinned)
            definite_sources = {
                requirement_id: source
                for requirement_id, source in discovered.items()
                if requirement_id in definite_ids
            }
        if definite_sources:
            pairs = await fetch_definitions(
                surface,
                log,
                revision=plan_revision(plan),
                requirement_keys=requirement_keys,
                sources=definite_sources,
            )
            log = await apply_definitions(tool_context.state, pairs)
            discovered = settled(log)
        resolved = await resolve_entity_refs(
            log,
            revision=plan_revision(plan),
            requirements=[
                r for r in plan.evidence_requirements if isinstance(r, DatasetRequirement)
            ],
            requirement_keys=requirement_keys,
            sources=discovered,
        )
        if resolved:
            log = await apply_definitions(tool_context.state, resolved)
            discovered = settled(log)
        by_name = {schema.name: schema for schema in schemas}

        def compiled(plan: Plan, translation: Translation) -> CompileResult:
            return compile_plan(
                plan=plan, translation=translation, schemas=by_name, discovered=discovered
            )

        result = compiled(plan, translation)
        if result.workflow is None:
            return _not_compiled(result, translation, tool_context)
        stated = await stated_from_the_data(
            plan, translation, result, compiled=compiled, tool_context=tool_context
        )
        if isinstance(stated, dict):
            return stated
        restated = restatements(plan, stated[0])
        plan, translation, result = stated
        assert result.workflow is not None
        warnings = [warning.model_dump(mode="json") for warning in result.warnings]

        plan = plan.with_translation(translation=translation, workflow=result.workflow)
        save_plan(tool_context.state, plan)
        return ok(
            **_restated_answer(restated),
            status=plan.status.value,
            version=plan.version,
            revision=plan_revision(plan),
            # Each dataset once, with every spec_id it is retrieved under and
            # the phrase each was searched as: the workflow is the record of
            # what was approved.
            sources=described_sources(result.workflow.sources),
            steps=[
                {"id": step.id, "tool": f"{step.tool.server}.{step.tool.name}"}
                for step in result.workflow.steps
            ],
            warnings=warnings,
        )

    def return_to_planner(
        kind: ReferralKind,
        reason: str,
        tool_context: ToolContext,
        requirement_id: str | None = None,
        constraint: str | None = None,
        repair: str | None = None,
        reference: str | None = None,
        repeated: bool = False,
        budget_spent: bool = False,
        not_issued: bool = False,
        source_mismatch: bool = False,
        stop_code: str | None = None,
    ) -> dict[str, object]:
        """Save a referral of the draft back to the planner and answer with it.

        The referral is saved to session state for the stall watchdog and
        `declare_ready`'s gate, and the plan is left untouched. A
        referral that may carry candidates takes them from the discovery
        log; `no_data`'s are the nearest real sources found instead, with
        the reason code, and a `requirement_unmeetable` one carries the
        sources a pin may be taken from only when marked `source_mismatch`.
        An `internal_failure` is logged at error level under a fresh reference, unless it restates one already logged
        under `reference`. A plan that is no longer a draft, an unknown
        `requirement_id`, or a referral that does not validate answers the
        problem instead.
        """
        # A raw string arrives here from a `discover_source` or compile
        # action's arguments. `kind` is coerced on the way into `Referral`
        # below, but the checks here run first, so a kind the vocabulary does
        # not hold is left as it came and refused by `Referral`, which says so
        # better than anything written here.
        with suppress(ValueError):
            kind = ReferralKind(kind)
        # A draft, not any plan: a plan a successful compile already moved to
        # PENDING_APPROVAL cannot have a referral written over it.
        plan = current_draft(tool_context)
        if isinstance(plan, dict):
            return plan
        requirement: DatasetRequirement | None = None
        if requirement_id is not None:
            try:
                found = plan.requirement(requirement_id)
            except KeyError:
                known = [requirement.id for requirement in plan.evidence_requirements]
                return problem(
                    ProblemCode.REQUIREMENT_UNKNOWN,
                    f"This plan records no requirement {requirement_id!r}.",
                    outcome=Outcome.INVALID_INPUT,
                    detail={"known": known},
                )
            requirement = found if isinstance(found, DatasetRequirement) else None
        candidates: tuple[SelectedSource, ...] = ()
        stale_pins: tuple[StalePin, ...] = ()
        withheld: tuple[WithheldSource, ...] = ()
        scope_repair: str | None = None
        no_data_reason: str | None = None
        if kind in MAY_CARRY_CANDIDATES and requirement_id is not None and requirement is not None:
            key = requirement_revision(requirement)
            log = load_log(tool_context.state)
            if kind is ReferralKind.NO_DATA:
                candidates = log.nearest_for(key, requirement_id)
                no_data_reason = log.no_data_reason(key, requirement_id)
            elif kind is ReferralKind.PINNED_SOURCE_NOT_FOUND:
                candidates, withheld, scope_repair = log.scope_kept(
                    requirement, log.pinnable_for(key, requirement_id)
                )
                stale_pins = _other_stale_pins(plan, log, requirement_id)
            elif kind is ReferralKind.REQUIREMENT_UNMEETABLE:
                # One of the sources found can meet a stated source they do not
                # name. None can meet a period or coverage they miss.
                if source_mismatch:
                    candidates = log.pinnable_for(key, requirement_id)
            else:
                candidates = log.candidates_for(key, requirement_id)
                if kind is ReferralKind.SOURCE_AMBIGUOUS and len(candidates) < 2:
                    return problem(
                        ProblemCode.NO_CANDIDATES_RECORDED,
                        f"Discovery recorded no candidate sources for {requirement_id!r} in this "
                        "revision; a source is ambiguous only when `discover_source` answered "
                        "with candidates for it.",
                        outcome=Outcome.INVALID_INPUT,
                    )
            # `pinned_source_not_found`, `no_data` and a `source_mismatch`
            # `requirement_unmeetable` each carry whatever the log holds, even
            # none: the latest search found no source a pin may be taken
            # from, or the nearest-series relaxation found nothing, and a
            # referral naming no candidates there reports exactly that.
        ambiguous_sources = ()
        if kind is ReferralKind.SOURCE_NAME_AMBIGUOUS and requirement is not None:
            key = requirement_revision(requirement)
            log = load_log(tool_context.state)
            matching = [
                record
                for record in log.records
                if record.requirement_id == requirement.id and record.requirement_key == key
            ]
            ambiguous_sources = matching[-1].ambiguous_sources if matching else ()
        try:
            referral = Referral(
                kind=kind,
                reason=reason,
                requirement_id=requirement_id,
                candidates=candidates,
                ambiguous_sources=ambiguous_sources,
                no_data_reason=no_data_reason,
                constraint=constraint,
                repair=repair or scope_repair,
                reference=(
                    reference or _logged_reference(reason)
                    if kind is ReferralKind.INTERNAL_FAILURE
                    else reference
                ),
                repeated=repeated,
                budget_spent=budget_spent,
                not_issued=not_issued,
                source_mismatch=source_mismatch,
                stop_code=stop_code,
                stale_pins=stale_pins,
                withheld=withheld,
            )
        except ValidationError as exc:
            return problem(
                ProblemCode.INVALID_REFERRAL,
                f"This referral does not have the shape a referral needs: {exc}",
                outcome=Outcome.INVALID_INPUT,
            )
        save_referral(tool_context.state, referral)
        return referred(_referral_payload(referral))

    (discover_source_tool,) = build_discovery_tools(surface=surface)
    return TranslatorCore(
        discover_source=discover_source_tool.func,
        propose_operations=propose_operations,
        return_to_planner=return_to_planner,
    )


def _planner_answer(result: dict[str, object]) -> dict[str, object]:
    """Return the verdict the planner reads, given `run_translate`'s `result`.

    A compiled plan answers `ok`, with the requirements whose rolling years
    translation restated under `restated` when it moved any.
    `declare_ready` reads the version, revision and warnings off the plan's
    record, and the user reads the sources and steps on the approval card. A
    referral answers as the referral itself, its `outcome` one of
    `OUTCOME_OF`'s. A problem answers unchanged.
    """
    if result.get("outcome") == Outcome.OK.value:
        return ok(**({"restated": result["restated"]} if "restated" in result else {}))
    if result.get("outcome") == Outcome.REFERRED.value:
        return cast("dict[str, object]", result["referral"])
    return result


def authored_answer(answer: dict[str, object], split: HistorySplit) -> dict[str, object]:
    """`answer` naming each requirement by the id the model wrote, under the plan's `split`.

    `answer` is a referral as the planner reads it, or a translation verdict
    carrying `named_in_place_of_across`. The draft translation runs on holds
    the copies the split of whole-history reads added
    (`Plan.rolling_copies`). Discovery, the compiler and the referral saved
    in session state name a copy by its own id, which the model did not write
    and cannot patch or pin, so the answer it reads names the copy's origin
    (`planner.plan.origin_of`) in `requirement_id`, each stale pin's and each
    named group's, and in the
    prose of `reason` and each `repair`. Discovery and the compiler build
    that prose with ids interpolated at dozens of sites, so it is rewritten
    here (`planner.plan.authored_text`).

    The model pins one source on the origin, and the split gives the copy
    its own window of it, so what the referral offers or withholds for the
    copy it offers or withholds for the origin
    (`translator.referral.Referral.offered`). The referral and its stale
    pins name each origin once: entries naming the same origin once mapped
    join, the origin's own entry first, each candidate and withheld source
    listed once, and the first `repair` stated kept. A named group
    repeating one before it is dropped.
    """
    if not split.copies:
        return answer

    def prose(entry: dict[str, object]) -> dict[str, object]:
        return entry | {
            field: authored_text(text, split.origins())
            for field in ("reason", "repair")
            if isinstance(text := entry.get(field), str)
        }

    def named(entries: object) -> list[dict[str, object]]:
        return [
            prose(entry | {"requirement_id": origin_of(split, str(entry["requirement_id"]))})
            for entry in cast("list[dict[str, object]]", entries)
        ]

    def own_first(entries: list[tuple[str, dict[str, object]]]) -> list[dict[str, object]]:
        """`entries`, each keyed by the id it named, the origin's own before its copies'."""
        copies = split.origins()
        return [entry for _, entry in sorted(entries, key=lambda pair: pair[0] in copies)]

    def joined(entries: list[dict[str, object]]) -> dict[str, object]:
        """The candidates and withheld sources of `entries`, each listed once, and the first repair."""
        fields: dict[str, object] = {}
        for field in ("candidates", "withheld"):
            listed: list[object] = []
            for entry in entries:
                listed += [
                    item
                    for item in cast("list[object]", entry.get(field) or [])
                    if item not in listed
                ]
            if listed:
                fields[field] = listed
        if repairs := [entry["repair"] for entry in entries if entry.get("repair")]:
            fields["repair"] = repairs[0]
        return fields

    mapped = prose(answer)
    if isinstance(requirement_id := answer.get("requirement_id"), str):
        mapped["requirement_id"] = origin_of(split, requirement_id)
    if "stale_pins" in answer:
        by_origin: dict[str, list[tuple[str, dict[str, object]]]] = {}
        for pin in cast("list[dict[str, object]]", answer["stale_pins"]):
            step_id = str(pin["requirement_id"])
            by_origin.setdefault(origin_of(split, step_id), []).append((step_id, prose(pin)))
        if isinstance(requirement_id, str) and (
            shared := by_origin.pop(origin_of(split, requirement_id), None)
        ):
            mapped |= joined(own_first([(requirement_id, mapped), *shared]))
        stale_pins: list[dict[str, object]] = []
        for origin, entries in by_origin.items():
            ordered = own_first(entries)
            stale_pins.append(ordered[0] | joined(ordered) | {"requirement_id": origin})
        mapped["stale_pins"] = stale_pins
    if "named_in_place_of_across" in answer:
        groups: list[dict[str, object]] = []
        for group in named(answer["named_in_place_of_across"]):
            if group not in groups:
                groups.append(group)
        mapped["named_in_place_of_across"] = groups
    return mapped


def build_translate(*, surface: ToolSurface) -> Translate:
    """Build the translation `propose_plan` runs on every plan it records, closing over `surface`.

    The function takes the planner's tool context and reads the plan
    `propose_plan` just recorded from session state, so the model never calls
    it and spends no generation on it.
    """
    core = build_translator_core(surface=surface)

    async def translate(tool_context: ToolContext) -> dict[str, object]:
        """Find the data the recorded plan needs and compile the workflow the user approves.

        First names, on the recorded draft, each group a pin holds a
        requirement's `across` side to, in place of that `across`
        (`translator.discovery.name_pinned_groups`). Each identity is
        resolved at most once in the whole translation
        (`translator.discovery.resolving_once`).

        Returns:
            `{"outcome": "ok"}` once the plan compiled, with `restated` when
            its rolling years were stated from the data, or the referral the
            plan goes back on, whose `outcome` says what it asks of the
            planner. Either carries `named_in_place_of_across` when a group
            was named, each entry the requirement, the entity named and the
            `across` dropped. Each names a requirement by the id the model
            wrote (`authored_answer`).
        """
        with resolving_once():
            named = await name_pinned_groups(tool_context, surface=surface)
            answer = _planner_answer(await run_translate(tool_context, core=core))
        if answer.get("outcome") == Outcome.UNSUPPORTED.value:
            # Every kind `OUTCOME_OF` maps to `unsupported` is an
            # `UnsupportedOperation`, so the plan has to decline it.
            record_unsupported(tool_context.state, UnsupportedOperation(answer["kind"]))
        if named:
            answer = answer | {
                "named_in_place_of_across": [
                    {
                        "requirement_id": group.requirement_id,
                        "entity": group.entity.model_dump(mode="json"),
                        "across": group.across,
                    }
                    for group in named
                ]
            }
        plan = load_plan(tool_context.state)
        return answer if plan is None else authored_answer(answer, split_of(plan))

    return translate

"""What a plan's run says about its answer: the retrievals each answer frame's
evidence comes from (`link_evidence_to_terminal_frames`), the answer frames
(`answer_frames`), and the notes on how the figures were computed
(`answer_notes`). The `execute` node (`harness.graph.execute`) calls it,
and `harness.graph.resume` links a completed run carried on from an earlier
turn through `link_accepted_run` and reads its notes from the plan stored
before the run (`store_run_plan`, `stored_run_plan`).
"""

from __future__ import annotations

import logging
from collections.abc import (
    Callable,
    Collection,
    Mapping,
    Sequence,
)
from functools import partial
from typing import Any, NamedTuple

from common.materialisation import FrameRef
from common.period_note import period_note
from common.place_type import FLOW_ROLE_COLUMNS, enrichment_role
from compiler.frame_columns import FRAME_PARAMETER
from compiler.sources import MembersAnswer, members_reading, total_reading
from compiler.surface import ToolSurface
from executor.artifacts import declared_shape, parse_manifest
from executor.ledger import RunLedger, store_faults
from executor.outcomes import (
    Accepted,
    RunOutcome,
    StepRecord,
)
from planner.plan import (
    DatasetRequirement,
    Decision,
    EvidenceRequirement,
    QuotientRequirement,
    ResolvedBy,
    stored_plan,
)
from planner.plan import Plan as PlannerPlan
from planner.records import Reduction
from planner.workflow import ReadingSetBy, ResolvedGroup, SelectedSource, WorkflowStep
from planner.workflow import Workflow as PlanWorkflow

from harness.provenance import BOTH_SIDES_VERBS, summed_stat_var_count

logger = logging.getLogger(__name__)


def _upstream_retrievals(
    step_id: str,
    *,
    steps_by_id: dict[str, WorkflowStep],
    retrieval_step_ids: frozenset[str],
    memo: dict[str, frozenset[str]],
) -> frozenset[str]:
    """The retrieval step ids that transitively feed `step_id`, via `WorkflowStep.bindings`.

    A retrieval step (one `Workflow.evidence_bindings` names) feeds itself;
    any other step's set is the union of what feeds each step its own
    bindings point `from_step` at -- so a two-hop chain (retrieve ->
    aggregate -> filter) still resolves to the one retrieval underneath it,
    and a join resolves to every retrieval either side rests on. Memoized
    per call: a DAG's shared ancestor is walked once per descendant it
    feeds, not once per edge on the path to it.
    """
    if step_id in memo:
        return memo[step_id]
    if step_id in retrieval_step_ids:
        result = frozenset({step_id})
    else:
        step = steps_by_id.get(step_id)
        upstream: set[str] = set()
        for binding in step.bindings.values() if step is not None else ():
            upstream |= _upstream_retrievals(
                binding.from_step,
                steps_by_id=steps_by_id,
                retrieval_step_ids=retrieval_step_ids,
                memo=memo,
            )
        result = frozenset(upstream)
    memo[step_id] = result
    return result


def _carried_frame_parameters(step: WorkflowStep) -> list[str]:
    """The arguments naming the frames whose StatVars `step`'s frame carries:
    `FRAME_PARAMETER`, and the second frame's for a verb that carries both
    sides (`harness.provenance.BOTH_SIDES_VERBS`)."""
    names = [FRAME_PARAMETER]
    if (verb := BOTH_SIDES_VERBS.get(step.tool.name)) is not None:
        names.append(verb.second_frame)
    return names


def _writes_own_stat_var(step: WorkflowStep) -> bool:
    """Whether `step`'s verb writes its result under a StatVar of its own."""
    verb = BOTH_SIDES_VERBS.get(step.tool.name)
    return verb is not None and verb.writes_own_stat_var


def _carried_retrievals(
    step_id: str,
    *,
    steps_by_id: dict[str, WorkflowStep],
    retrieval_step_ids: frozenset[str],
) -> frozenset[str]:
    """The retrieval step ids whose StatVars `step_id`'s frame carries.

    A verb writes the StatVars of the frame bound to `FRAME_PARAMETER`, and
    `join` also those of its right side. `compare`, `sum_of`, `quotient_of`
    and `share_of` each write a StatVar of their own that stands for both
    sides' StatVars, so they carry both too (`_carried_frame_parameters`).
    A subset of `_upstream_retrievals`.
    """
    if step_id in retrieval_step_ids:
        return frozenset({step_id})
    step = steps_by_id.get(step_id)
    if step is None:
        return frozenset()
    carried: set[str] = set()
    for name in _carried_frame_parameters(step):
        if (binding := step.bindings.get(name)) is not None:
            carried |= _carried_retrievals(
                binding.from_step, steps_by_id=steps_by_id, retrieval_step_ids=retrieval_step_ids
            )
    return frozenset(carried)


def _carried_parts(
    step_id: str,
    *,
    steps_by_id: dict[str, WorkflowStep],
    retrieval_step_ids: frozenset[str],
) -> tuple[frozenset[str], frozenset[str]]:
    """The retrieval steps, and the steps writing a StatVar of their own,
    whose StatVars `step_id`'s frame carries.

    The walk `_carried_retrievals` takes, stopping at each `compare`,
    `sum_of`, `quotient_of` or `share_of` (`_writes_own_stat_var`). The frame
    carries that step's one StatVar, which stands for every StatVar under its
    sides, in place of the retrievals beneath it.
    """
    if step_id in retrieval_step_ids:
        return frozenset({step_id}), frozenset()
    step = steps_by_id.get(step_id)
    if step is None:
        return frozenset(), frozenset()
    if _writes_own_stat_var(step):
        return frozenset(), frozenset({step_id})
    retrievals: set[str] = set()
    writers: set[str] = set()
    for name in _carried_frame_parameters(step):
        if (binding := step.bindings.get(name)) is not None:
            found, written = _carried_parts(
                binding.from_step, steps_by_id=steps_by_id, retrieval_step_ids=retrieval_step_ids
            )
            retrievals |= found
            writers |= written
    return frozenset(retrievals), frozenset(writers)


def _ran(outcome: RunOutcome) -> tuple[StepRecord, ...]:
    """Every step of `outcome`'s run that finished, in the plan's own step order."""
    return outcome.steps if isinstance(outcome, Accepted) else outcome.completed


def presentable_steps(outcome: RunOutcome, *, published: bool = True) -> tuple[StepRecord, ...]:
    """Every finished step of `outcome`'s run whose frame and digest may reach
    the agent, in the plan's own step order. A retrieved (`spec_`) frame is
    always presentable. A derived (`ops_`) frame is presentable only once
    published with its replay recipe (`executor.completion`), since an
    unpublished figure is never quoted or charted: an accepted run's are
    when `published` says its run completed, and a stopped run's never are,
    whatever `published` says. The answer frames (`answer_frames`) and the
    step summaries `execute` hands back both read this one selection."""
    if isinstance(outcome, Accepted) and published:
        return outcome.steps
    return tuple(
        record for record in _ran(outcome) if not record.frame_ref.spec_id.startswith("ops_")
    )


class AnswerFrame(NamedTuple):
    """One answer frame's two pairs: `presented`, which it is described and
    drawn by (`StepRecord.presented_ref`), and `store`, which the
    materialisation store holds it under (`StepRecord.frame_ref`). They are
    the same pair unless the frame is a retrieval minted with extra columns."""

    presented: FrameRef
    store: FrameRef


def answer_frames(outcome: RunOutcome, *, published: bool = True) -> dict[str, AnswerFrame]:
    """Each terminal frame that stands as an answer, by step id, in the
    workflow's terminal order: each terminal step's on an accepted run
    (`Accepted.results`), and on a stopped run each finished terminal step's
    that passed the same answer check (`answers`, `executor.run._Run.
    _answer_refusal`), when it is presentable (`presentable_steps`, given
    `published`).

    Each frame comes with both its pairs (`AnswerFrame`): a final retrieval
    minted with extra columns is described and drawn by its manifest's own
    pair, which apis serves, while the store holds it under its
    `frame_ref`, the pair the steps reading it were bound to."""
    records = {record.step_id: record for record in presentable_steps(outcome, published=published)}
    answers = outcome.results if isinstance(outcome, Accepted) else outcome.answers
    return {
        step_id: AnswerFrame(
            presented=records[step_id].presented_ref, store=records[step_id].frame_ref
        )
        for step_id in answers
        if step_id in records
    }


def link_evidence_to_terminal_frames(
    tool_surface: ToolSurface | None,
    outcome: RunOutcome,
    *,
    workflow: PlanWorkflow,
    published: bool = True,
) -> None:
    """Link each terminal frame's provenance to every retrieval that feeds it.

    A terminal step is either a bare `get_data` -- a plan answering a
    question that asks to see a series names the retrieved frame as its own
    terminal -- or an analysis-mcp transform (`aggregate`, `filter`, `join`,
    ...) chained onto one or more `get_data` steps. A transform mints its own
    `ops_` frame identity, distinct from the retrieval's `spec_` one that
    `harness.provenance.ProvenanceStore` was written under
    (`harness.studio_translate._RecordingSurface._remember_manifest`).
    Without this, `McpExecutor._assemble_component` would never find a
    terminal frame's provenance on the approved-plan path.

    A stopped run, or an accepted run whose derived frames are not yet
    `published`, links the terminal frames it hands back (`answer_frames`)
    the same way, from the evidence its finished retrievals recorded, so a
    partial answer's cards keep their sources.

    Attribution is per frame: `_upstream_retrievals` walks each terminal
    step's own `bindings` back to the retrieval(s) under it, so two
    independent branches in the same run (a debt-service step and a
    population step, each aggregated on its own) each link to their own
    source, and a frame fed by several retrievals (a join, a compare, a
    share of two datasets) links to all of them, in `workflow.steps` order.
    The frame's lineage names exactly the sources its card rests on. The
    link also carries the distinct StatVars the terminal step declared its
    frame carries (`StepRecord.stat_var_count`), which the card counts its
    indicators from, and the retrievals and sums whose StatVars the frame
    carries (`_carried_parts`), each sum with the retrievals under it, so a
    summed family's ref counts as the StatVars it sums and a sum's ref as the
    distinct StatVars under its sides, and a quotient's ref the same way.

    All-or-nothing: a frame with no retrieval under it (a workflow defect,
    never a valid one), or one whose retrieval has no `EvidenceRecord`, is
    not linked and a warning names it. `McpExecutor._assemble_component`
    withholds the card for a frame with no lineage rather than state a
    provenance narrower than the frame has. `link_evidence` is looked up via
    `getattr`, the same duck-typed opt-in `_RecordingExecutor.aclose` uses.
    """
    link = getattr(tool_surface, "link_evidence", None)
    if link is None:
        return
    retrieval_step_ids = frozenset(binding.step_id for binding in workflow.evidence_bindings)
    steps_by_id = {step.id: step for step in workflow.steps}
    step_order = {step.id: index for index, step in enumerate(workflow.steps)}
    evidence_by_step = {record.step_id: record for record in outcome.evidence}
    stat_var_counts = {record.step_id: record.stat_var_count for record in _ran(outcome)}
    memo: dict[str, frozenset[str]] = {}
    for step_id, frame in answer_frames(outcome, published=published).items():
        frame_ref = frame.presented
        sources = sorted(
            _upstream_retrievals(
                step_id,
                steps_by_id=steps_by_id,
                retrieval_step_ids=retrieval_step_ids,
                memo=memo,
            ),
            key=lambda source: step_order[source],
        )
        unrecorded = [source for source in sources if source not in evidence_by_step]
        if unrecorded or not sources:
            logger.warning(
                "provenance: frame for step %r not linked; retrieval step(s) without an "
                "evidence record: %s",
                step_id,
                unrecorded or "no retrieval feeds it",
            )
            continue
        carried, writers = _carried_parts(
            step_id, steps_by_id=steps_by_id, retrieval_step_ids=retrieval_step_ids
        )
        link(
            frame_ref=frame_ref.model_dump(),
            evidence_frame_refs=[
                evidence_by_step[source].frame_ref.model_dump() for source in sources
            ],
            stat_var_count=stat_var_counts.get(step_id),
            carried_frame_refs=[
                evidence_by_step[source].frame_ref.model_dump()
                for source in sources
                if source in carried
            ],
            summed_frame_refs=[
                [
                    evidence_by_step[source].frame_ref.model_dump()
                    for source in sources
                    if source in under
                ]
                for under in (
                    _carried_retrievals(
                        writer, steps_by_id=steps_by_id, retrieval_step_ids=retrieval_step_ids
                    )
                    for writer in sorted(writers, key=step_order.__getitem__)
                )
            ],
        )


async def link_accepted_run(
    tool_surface: ToolSurface | None,
    outcome: Accepted,
    *,
    workflow: PlanWorkflow,
    ledger: RunLedger,
    published: bool = True,
) -> None:
    """Link accepted run `outcome`'s terminal frames to their retrievals
    (`link_evidence_to_terminal_frames`, given `published`: false for a run
    whose publication is unfinished, which links only its retrieved
    terminals), once the surface holds the lineage of every `get_data`
    result the run sealed its inputs from (`RunLedger.retrievals`). A run
    computed in another turn or another process left none in this process's
    provenance store, so its frames would otherwise link to nothing. `restore_retrievals` is looked up via
    `getattr`, as `link_evidence` is.

    Raises:
        LedgerUnavailableError: the ledger's store could not be read.
    """
    restore = getattr(tool_surface, "restore_retrievals", None)
    if restore is not None and outcome.run_id is not None:
        async with store_faults(outcome.run_id):
            retrievals = await ledger.retrievals(outcome.run_id)
        await restore(retrievals or [])
    link_evidence_to_terminal_frames(tool_surface, outcome, workflow=workflow, published=published)


async def store_run_plan(ledger: RunLedger, *, revision: str, plan: PlannerPlan) -> None:
    """Store `plan`, which compiled the work `revision` names, in `ledger`
    before that work's run is admitted, so the turn that completes the run,
    in whichever session it runs, reads its notes (`answer_notes`) from it
    (`stored_run_plan`). A store that cannot take it is logged, and such a
    run completed by a later turn lists no notes."""
    try:
        await ledger.store_plan(revision, plan.model_dump_json())
    except Exception:
        logger.warning(
            "the plan for revision %s could not be stored; a later turn that completes its "
            "run lists no notes",
            revision,
            exc_info=True,
        )


async def stored_run_plan(ledger: RunLedger, *, run_id: str, revision: str) -> PlannerPlan | None:
    """The plan `store_run_plan` stored for `revision`, the work of run
    `run_id`, or `None` when none was stored or this release cannot read it.

    Raises:
        LedgerUnavailableError: the ledger's store could not be read.
    """
    async with store_faults(run_id):
        raw = await ledger.plan(revision)
    return stored_plan(raw if isinstance(raw, str) else None, state_key=f"run plan {revision}")


# The flow axis a recipient's role column sits on (`get_data`'s coverage
# dimensions).
_RECIPIENT_AXIS = "destination"


def _stat_var_totals(
    tool_surface: ToolSurface | None,
    outcome: RunOutcome,
    *,
    plan: PlannerPlan,
    terminals: Collection[str] | None = None,
) -> list[str]:
    """One sentence per summed family an answer frame carries, over every
    answer frame (`answer_frames`) or only those of `terminals`.

    Walks the retrievals whose StatVars each answer frame carries
    (`_carried_retrievals`, the walk the card counts its indicators by), so
    both sides of a `compare` get their sentence. A retrieval is summed when
    its recorded `get_data` result (`_RecordingSurface.retrieval_manifest`,
    looked up via `getattr` like `link_evidence`) names the StatVars
    apis added into the family total
    (`summed_stat_var_count`). The sentence names the family, states that
    count and who set the reading (`_reading_set_by`), and says when the
    figures add up, or are computed from, recipients the question did not
    name (`_totals_across_recipients`). Identical sentences are stated once.
    """
    manifest_of = getattr(tool_surface, "retrieval_manifest", None)
    workflow = plan.workflow
    if manifest_of is None or workflow is None:
        return []
    retrieval_step_ids = frozenset(binding.step_id for binding in workflow.evidence_bindings)
    steps_by_id = {step.id: step for step in workflow.steps}
    step_order = {step.id: index for index, step in enumerate(workflow.steps)}
    requirement_ids = {b.step_id: b.requirement_id for b in workflow.evidence_bindings}
    requirements = {requirement.id: requirement for requirement in plan.evidence_requirements}
    decisions = {decision.slot: decision for decision in plan.decisions}
    evidence_by_step = {record.step_id: record for record in outcome.evidence}
    role_keys_by_step = {record.step_id: record.role_keys for record in _ran(outcome)}
    memo: dict[str, frozenset[str]] = {}
    sentences: list[str] = []
    for terminal_id in answer_frames(outcome):
        if terminals is not None and terminal_id not in terminals:
            continue
        carried = _carried_retrievals(
            terminal_id, steps_by_id=steps_by_id, retrieval_step_ids=retrieval_step_ids
        )
        for retrieval_id in sorted(carried, key=step_order.__getitem__):
            record = evidence_by_step.get(retrieval_id)
            manifest = manifest_of(frame_ref=record.frame_ref.model_dump()) if record else None
            count = summed_stat_var_count(manifest) if manifest is not None else None
            if manifest is None or count is None:
                continue
            source = workflow.source_of(retrieval_id)
            sentence = _total_sentence(
                _family_name(manifest, source),
                count,
                set_by=_reading_set_by(
                    source, requirements.get(requirement_ids[retrieval_id]), decisions
                ),
                across_recipients=_totals_across_recipients(
                    manifest,
                    source,
                    answer_role_keys=role_keys_by_step.get(terminal_id),
                    reads=partial(
                        _members_answer,
                        retrieval_id=retrieval_id,
                        terminal_id=terminal_id,
                        steps_by_id=steps_by_id,
                        retrieval_step_ids=retrieval_step_ids,
                        role_keys_by_step=role_keys_by_step,
                        memo=memo,
                    ),
                ),
            )
            if sentence not in sentences:
                sentences.append(sentence)
    return sentences


def answer_notes(
    tool_surface: ToolSurface | None,
    outcome: RunOutcome,
    *,
    plan: PlannerPlan,
    terminals: Collection[str] | None = None,
) -> list[str]:
    """The notes on how an answer's figures were computed, over every
    answer frame (`answer_frames`) or only those of `terminals`: one per
    summed family (`_stat_var_totals`), one per named set a retrieval reads
    (`_group_reading_sentences`), one per step that left keys out
    (`_dropped_key_sentences`), then one per one-year retrieval when a newer
    year holds data, the year served is not the elected year, or a fuller
    earlier year is named (`_period_notes`)."""
    return [
        *_stat_var_totals(tool_surface, outcome, plan=plan, terminals=terminals),
        *_group_reading_sentences(outcome, plan=plan, terminals=terminals),
        *_dropped_key_sentences(outcome, plan=plan, terminals=terminals),
        *_period_notes(tool_surface, outcome, plan=plan, terminals=terminals),
    ]


def _period_notes(
    tool_surface: ToolSurface | None,
    outcome: RunOutcome,
    *,
    plan: PlannerPlan,
    terminals: Collection[str] | None = None,
) -> list[str]:
    """The period note (`common.period_note.period_note`) of each retrieval
    under an answer frame, over every answer frame (`answer_frames`) or only
    those of `terminals`.

    An answer frame an analysis step wrote has no manifest of its own, so
    the note is read from the recorded `get_data` result
    (`_RecordingSurface.retrieval_manifest`) of every retrieval that feeds
    it (`_upstream_retrievals`). Retrievals are taken in the plan's step
    order, and identical sentences are stated once.
    """
    manifest_of = getattr(tool_surface, "retrieval_manifest", None)
    workflow = plan.workflow
    if manifest_of is None or workflow is None:
        return []
    retrieval_step_ids = frozenset(binding.step_id for binding in workflow.evidence_bindings)
    steps_by_id = {step.id: step for step in workflow.steps}
    step_order = {step.id: index for index, step in enumerate(workflow.steps)}
    evidence_by_step = {record.step_id: record for record in outcome.evidence}
    memo: dict[str, frozenset[str]] = {}
    sentences: list[str] = []
    for terminal_id in answer_frames(outcome):
        if terminals is not None and terminal_id not in terminals:
            continue
        under = _upstream_retrievals(
            terminal_id, steps_by_id=steps_by_id, retrieval_step_ids=retrieval_step_ids, memo=memo
        )
        for retrieval_id in sorted(under, key=step_order.__getitem__):
            record = evidence_by_step.get(retrieval_id)
            manifest = manifest_of(frame_ref=record.frame_ref.model_dump()) if record else None
            note = period_note((manifest or {}).get("coverage") or {})
            if note and note not in sentences:
                sentences.append(note)
    return sentences


def _group_reading_sentences(
    outcome: RunOutcome,
    *,
    plan: PlannerPlan,
    terminals: Collection[str] | None = None,
) -> list[str]:
    """How each named set a retrieval under an answer frame binds is read,
    over every answer frame (`answer_frames`) or only those of `terminals`.

    One sentence per set, giving the group's published total
    (`compiler.sources.total_reading`) or its members
    (`compiler.sources.members_reading`), each on its own when the answer
    frame keeps the column the group sits on (`_group_column`), added up
    when every step that leaves that column out sums across it
    (`_members_answer`), and computed from them otherwise. Retrievals are
    taken in the plan's step order, and identical sentences are stated once.
    """
    workflow = plan.workflow
    if workflow is None:
        return []
    retrieval_step_ids = frozenset(binding.step_id for binding in workflow.evidence_bindings)
    steps_by_id = {step.id: step for step in workflow.steps}
    step_order = {step.id: index for index, step in enumerate(workflow.steps)}
    role_keys_by_step = {record.step_id: record.role_keys for record in _ran(outcome)}
    memo: dict[str, frozenset[str]] = {}
    sentences: list[str] = []
    for terminal_id in answer_frames(outcome):
        if terminals is not None and terminal_id not in terminals:
            continue
        answer_role_keys = role_keys_by_step.get(terminal_id)
        under = _upstream_retrievals(
            terminal_id, steps_by_id=steps_by_id, retrieval_step_ids=retrieval_step_ids, memo=memo
        )
        for retrieval_id in sorted(under, key=step_order.__getitem__):
            source = workflow.source_of(retrieval_id)
            if source is None:
                continue
            labels = {entity.ref: entity.label for entity in source.entities}
            for group in source.groups:
                column = _group_column(group, role_keys_by_step.get(retrieval_id))
                answer = (
                    None
                    if column is None or answer_role_keys is None
                    else _members_answer(
                        column,
                        retrieval_id=retrieval_id,
                        terminal_id=terminal_id,
                        steps_by_id=steps_by_id,
                        retrieval_step_ids=retrieval_step_ids,
                        role_keys_by_step=role_keys_by_step,
                        memo=memo,
                    )
                )
                sentence = total_reading(group, labels) or members_reading(group, answer=answer)
                if sentence is not None and sentence not in sentences:
                    sentences.append(sentence)
    return sentences


def _members_answer(
    column: str,
    *,
    retrieval_id: str,
    terminal_id: str,
    steps_by_id: dict[str, WorkflowStep],
    retrieval_step_ids: frozenset[str],
    role_keys_by_step: Mapping[str, tuple[str, ...] | None],
    memo: dict[str, frozenset[str]],
) -> MembersAnswer:
    """How the answer frame of `terminal_id` reads the members `column` of
    retrieval `retrieval_id` holds: a named group's members
    (`_group_reading_sentences`) or every recipient a source reports
    (`_totals_across_recipients`).

    `EACH` when the answer frame keeps `column`. `SUMMED` when every step
    between the two whose input frame carries `column` and whose own frame
    does not is an `aggregate` with `aggregation='sum'`, and there is one.
    `COMBINED` when such a step does anything else (a mean, an extreme, a
    ratio, a rank), when none is found, or when a step's role keys on the
    way are unknown.
    """
    if column in (role_keys_by_step.get(terminal_id) or ()):
        return MembersAnswer.EACH
    summed = False
    for step_id in _upstream_steps(terminal_id, steps_by_id=steps_by_id):
        step = steps_by_id.get(step_id)
        if step is None or step_id in retrieval_step_ids:
            continue
        inputs = [
            binding.from_step
            for binding in step.bindings.values()
            if retrieval_id
            in _upstream_retrievals(
                binding.from_step,
                steps_by_id=steps_by_id,
                retrieval_step_ids=retrieval_step_ids,
                memo=memo,
            )
        ]
        if not inputs:
            continue
        output = role_keys_by_step.get(step_id)
        carried = [role_keys_by_step.get(from_step) for from_step in inputs]
        if output is None or None in carried:
            return MembersAnswer.COMBINED
        if column in output or not any(column in keys for keys in carried if keys is not None):
            continue
        if step.tool.name != "aggregate" or step.args.get("aggregation") != Reduction.SUM:
            return MembersAnswer.COMBINED
        summed = True
    return MembersAnswer.SUMMED if summed else MembersAnswer.COMBINED


def _group_column(group: ResolvedGroup, role_keys: Sequence[str] | None) -> str | None:
    """The column of a retrieval frame carrying `role_keys` that `group` sits on.

    A flow side's column (`common.place_type.FLOW_ROLE_COLUMNS`), and on any
    other role the frame's one role column besides its type and kind
    columns. `None` when the frame's role keys are unknown, or it carries
    no such column or several.
    """
    if role_keys is None:
        return None
    column = FLOW_ROLE_COLUMNS.get(group.role)
    if column is not None:
        return column if column in role_keys else None
    roles = [key for key in role_keys if enrichment_role(key, roles=role_keys) is None]
    return roles[0] if len(roles) == 1 else None


def _dropped_key_sentences(
    outcome: RunOutcome,
    *,
    plan: PlannerPlan,
    terminals: Collection[str] | None = None,
) -> list[str]:
    """One sentence per step under an answer frame that left keys out of its
    result (`StepRecord.dropped_key_count`), over every answer frame
    (`answer_frames`) or only those of `terminals`.

    A keyed `quotient_of` drops each numerator key it has nothing to divide
    by, and every step reading its frame, a `rank` among them, works over
    the keys that remain. So the walk covers every step an answer frame
    rests on (`_upstream_steps`), in the plan's step order. Identical
    sentences are stated once.
    """
    workflow = plan.workflow
    if workflow is None:
        return []
    dropped = {
        record.step_id: record.dropped_key_count
        for record in _ran(outcome)
        if record.dropped_key_count
    }
    if not dropped:
        return []
    steps_by_id = {step.id: step for step in workflow.steps}
    sentences: list[str] = []
    for terminal_id in answer_frames(outcome):
        if terminals is not None and terminal_id not in terminals:
            continue
        under = _upstream_steps(terminal_id, steps_by_id=steps_by_id)
        for step_id, count in dropped.items():
            if step_id not in under:
                continue
            sentence = _dropped_key_sentence(count, *_dropped_key_nouns(step_id, plan))
            if sentence not in sentences:
                sentences.append(sentence)
    return sentences


def _upstream_steps(step_id: str, *, steps_by_id: dict[str, WorkflowStep]) -> frozenset[str]:
    """`step_id` and every step that transitively feeds it, via `WorkflowStep.bindings`."""
    found: set[str] = set()
    pending = [step_id]
    while pending:
        current = pending.pop()
        if current in found:
            continue
        found.add(current)
        if (step := steps_by_id.get(current)) is not None:
            pending.extend(binding.from_step for binding in step.bindings.values())
    return frozenset(found)


def _dropped_key_nouns(step_id: str, plan: PlannerPlan) -> tuple[str | None, str | None]:
    """The plan's own words for the keys step `step_id` divides and for what
    it divides them by: the quotient requirement's `axis`, and the measure
    of the dataset requirement under its denominator. `None` for either the
    plan does not state.

    A step carries out the operation it is named for (`WorkflowStep.id`),
    and that operation names its requirement.
    """
    operations = plan.translation.operations if plan.translation is not None else ()
    requirement_id = next(
        (operation.requirement_id for operation in operations if operation.id == step_id), None
    )
    requirements = {requirement.id: requirement for requirement in plan.evidence_requirements}
    requirement = requirements.get(requirement_id or "")
    if not isinstance(requirement, QuotientRequirement):
        return None, None
    divisor = requirements.get(requirement.denominator.input_id)
    # A denominator read through a filter or an aggregate keeps its
    # dataset's measure.
    while divisor is not None and not isinstance(divisor, DatasetRequirement):
        divisor = requirements.get(getattr(divisor, "input_id", None) or "")
    return requirement.axis, divisor.measure if divisor is not None else None


def _dropped_key_sentence(count: int, axis: str | None, divisor: str | None) -> str:
    """The sentence for `count` keys a quotient left out: how many, of which
    role when the plan names one (`axis`), and that each had nothing to
    divide by, named by its measure when the plan states one (`divisor`).

    A key is left out when the denominator holds no entry for it, none in a
    period the key reports, or only zeros where it does, and the one wording
    is true of all three.

    The role phrase is the planner's own word and is not inflected: it
    qualifies "entry" or "entries", which carries the number.
    """
    noun = "entry" if count == 1 else "entries"
    subject = f"{count:,} {axis} {noun}" if axis else f"{count:,} {noun}"
    lacked = f"no {divisor} figure to divide by" if divisor else "no figure to divide by"
    verb = "was" if count == 1 else "were"
    return f"{subject} had {lacked} and {verb} left out."


def _family_name(manifest: Mapping[str, Any], source: SelectedSource | None) -> str:
    """The name narration gives a summed family: the name apis gave the
    family (`summed_stat_var_label`), else the name of its source."""
    label = manifest.get("summed_stat_var_label")
    if isinstance(label, str) and label:
        return label
    if source is not None and source.provenance_label:
        return source.provenance_label
    return parse_manifest(manifest).citation.text


def _totals_across_recipients(
    manifest: Mapping[str, Any],
    source: SelectedSource | None,
    *,
    answer_role_keys: tuple[str, ...] | None,
    reads: Callable[[str], MembersAnswer],
) -> MembersAnswer | None:
    """How the answer combines recipients of a summed retrieval the question did not name.

    `reads` (`_members_answer` for the retrieval and the answer frame) says
    how the answer reads a role column of the retrieval's frame. The answer
    combines recipients when the frame carries a role column on the
    recipient axis (its declared `role_keys`, each placed on its axis by the
    manifest's coverage dimensions) that the answer does not keep, and the
    source resolves no recipient, so the frame held every recipient the
    source reports. `None` when it keeps them, and when either frame's role
    keys are unknown.
    """
    _rows, role_keys, _stat_var_count = declared_shape(manifest)
    if role_keys is None or answer_role_keys is None:
        return None
    if source is not None and any(entity.role == _RECIPIENT_AXIS for entity in source.entities):
        return None
    for dimension in parse_manifest(manifest).coverage.dimensions:
        if dimension.axis != _RECIPIENT_AXIS or dimension.label not in role_keys:
            continue
        answer = reads(dimension.label)
        if answer is not MembersAnswer.EACH:
            return answer
    return None


def _reading_set_by(
    source: SelectedSource | None,
    requirement: EvidenceRequirement | None,
    decisions: Mapping[str, Decision],
) -> ResolvedBy | ReadingSetBy | None:
    """Who set a summed retrieval's reading, when narration names them.

    `ReadingSetBy.RULE` when the compiler applied the source's default
    (`compiler.sources.default_reading`); `ResolvedBy.USER` when the
    requirement cites a decision the user settled that fixes the reading.
    `None` when the planner read it off the question, which needs no
    mention.
    """
    if source is not None and source.stat_var_reading_set_by is ReadingSetBy.RULE:
        return ReadingSetBy.RULE
    slots = requirement.decision_ids if requirement is not None else ()
    cited = (decisions[slot] for slot in slots if slot in decisions)
    if any(
        decision.resolved_by is ResolvedBy.USER
        and decision.fixes is not None
        and decision.fixes.stat_vars is not None
        for decision in cited
    ):
        return ResolvedBy.USER
    return None


def _total_sentence(
    name: str,
    count: int,
    *,
    set_by: ResolvedBy | ReadingSetBy | None,
    across_recipients: MembersAnswer | None,
) -> str:
    """The sentence stated for one summed family.

    `count` is the StatVars apis summed, named indicators as the card's
    attribution names them. It describes the family, so it stays true of
    figures a later filter or verb drew from the sum. `across_recipients`
    is how the answer combines recipients the question did not name
    (`_totals_across_recipients`), `None` when it does not.
    """
    sentence = f'the figures for "{name}" sum the family\'s {count:,} indicators.'
    if set_by is ReadingSetBy.RULE:
        sentence = f"By default, {sentence}"
    elif set_by is ResolvedBy.USER:
        sentence = f"As you asked, {sentence}"
    else:
        sentence = sentence[0].upper() + sentence[1:]
    if across_recipients is MembersAnswer.SUMMED:
        sentence += (
            " They include regional and unallocated flows as well as flows to individual countries."
        )
    elif across_recipients is MembersAnswer.COMBINED:
        sentence += (
            " They are computed from regional and unallocated flows as well as flows to "
            "individual countries."
        )
    return sentence

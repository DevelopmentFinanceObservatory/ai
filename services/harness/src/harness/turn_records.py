"""What one Studio turn planned and what its searches read, as records and text.

`PlanRecord` is a compiled `planner.plan.Plan` reduced to what a model needs
to answer for the plan afterwards: the question, the period and whether the
planner assumed it, each source by name and spec_id with its coverage and
where it came from, each decision with who settled it and why, the
revision, the outputs, and one line per step. `SearchReading` is one part of
a `search` result as `harness.tool_contracts._part_summary` builds it, plus
the facts a later `fetch` of its spec_id adds.

A family of StatVars appears as a count. Neither record holds a StatVar
dcid, so no rendering of either can inline one.

`PlanRecord.render` and `render_readings` are deterministic and take a
character budget. Over budget they shed detail in the fixed order each
docstring states (shortening text, folding steps, sources or readings into
a count) and say what they folded, so a shed field leaves a trace. A text
still over budget after the last level is cut with
`planner.plan.CUT_MARKER`.
"""

from __future__ import annotations

from collections.abc import Collection, Iterator, Mapping, Sequence
from datetime import datetime
from typing import Any, Literal, NamedTuple, cast

from common.place_type import FLOW_ROLE_COLUMNS
from compiler.evidence import read_unit_currency
from compiler.sources import PROVIDER_ACCOUNT, stat_var_count
from executor.ledger import RunStatus
from planner.parties import PARTY_ROLES, column_party_roles
from planner.plan import (
    DatasetRequirement,
    Decline,
    Plan,
    PlanStatus,
    ResolvedBy,
    UnsupportedOperation,
    cut_user_text,
    plan_revision,
)
from planner.records import (
    AmbiguousSourceReading,
    ResolvedSourceMention,
    StatVarReading,
    StrictModel,
)
from planner.workflow import ElectedMeasure, MeasureAlternative, ResolvedGroup
from pydantic import ConfigDict, Field, model_validator
from translator.discovery import BLOCKING_WARNING_CODES

from harness import plan_display
from harness.tool_contracts import requested_range, series_description

# Where a source or reading came from: the orchestrator's `search`, the
# translator's own discovery, or a dataset the user brought from Search
# (`harness.graph`'s `seed` node).
Origin = Literal["search", "translator", "seed"]

# How many characters of a decision rationale or warning detail, a reading's
# sentence, or a candidate's label the first shortening level keeps.
_RATIONALE_CUT = 80
_SENTENCE_CUT = 160
_LABEL_CUT = 60
_REFUSED_PHRASES_CUT = 80
_FLAGGED_MATCH_CUT = 160
_TIGHT_REFUSED_PHRASES_CUT = 40
_TIGHT_FLAGGED_MATCH_CUT = 80
_FOLDED_REFUSED_PHRASES_CUT = 30
_FOLDED_FLAGGED_MATCH_CUT = 60

# A revision is a SHA-256 hex digest; this many characters name it in text.
_REVISION_CHARS = 12


class YearSpan(StrictModel):
    """An inclusive range of years."""

    start: int
    end: int

    def text(self) -> str:
        return f"{self.start}-{self.end}"


class PlanPeriod(StrictModel):
    """A period the plan's dataset requirements ask for.

    `decision` is the slot of the cited decision that fixes the period, and
    `assumed` holds when that decision is a `planner_default`. With no such
    decision the period is the one the requirements state.
    """

    start: int
    end: int
    assumed: bool
    decision: str | None = None
    requirement_ids: tuple[str, ...]


class PlanSource(StrictModel):
    """One source the compiled plan retrieves.

    `name` is the provenance label. `members` counts the StatVars a family
    reads (`compiler.sources.stat_var_count`). `None` for a source that binds
    one StatVar.

    `coverage_source` says what bounds `coverage`: `requested` when the
    identity's own range (`harness.tool_contracts.requested_range`) holds
    it, since a spec minted for a range reports coverage only within the
    years asked; `coverage` when it is the span the source holds data
    across. `None` when no coverage is stated.

    `identity` is the identity document `spec_id` hashes from
    (`planner.workflow.SelectedSource.identity`), `None` when the source
    carried none. It is kept so a later turn's pin of a source only the
    translator found has its definition on record
    (`harness.studio_state.ConversationState.recorded_identity`). No
    rendering of the record reads it. `groups` are the sets its places were
    asked for as, kept beside it for the same pin, and left out when none.
    `elected_measure` is the source's `SelectedSource.elected_measure`, so a
    later turn's reading of the source states how it is counted
    (`source_readings`); left out when the source carried none.

    A record ignores keys it does not declare, since the conversation ledger
    holds records written before a field was removed: `stat_vars`, the
    StatVars Search listed, which records stored before Search stopped
    listing members still carry.
    """

    model_config = ConfigDict(extra="ignore")

    spec_id: str
    name: str | None
    measure: str | None
    coverage_kind: str | None
    coverage: YearSpan | None
    coverage_source: Literal["requested", "coverage"] | None
    members: int | None
    origin: Origin
    identity: dict[str, Any] | None = None
    groups: tuple[ResolvedGroup, ...] = Field(default=(), exclude_if=lambda groups: not groups)
    resolved_sources: tuple[ResolvedSourceMention, ...] = Field(
        default=(), exclude_if=lambda mentions: not mentions
    )
    provenance_ref: str | None = Field(default=None, exclude_if=lambda ref: ref is None)
    provenance_account: str | None = Field(default=None, exclude_if=lambda account: account is None)
    elected_measure: ElectedMeasure | None = Field(
        default=None, exclude_if=lambda elected: elected is None
    )


class RecordedIdentity(StrictModel):
    """A source's definition as a later turn resubmits it to Search.

    `identity` is the identity document its spec_id hashes from. Search
    derives the population from it. `groups` are the sets its places were
    asked for as, which a resubmit posts beside it; left out of the record
    when there are none. Like `PlanSource`, it ignores the
    `stat_vars` that ledger records stored before Search stopped listing
    members carry.
    """

    model_config = ConfigDict(extra="ignore")

    identity: dict[str, Any]
    resolved_sources: tuple[ResolvedSourceMention, ...] = Field(
        default=(), exclude_if=lambda mentions: not mentions
    )
    groups: tuple[ResolvedGroup, ...] = Field(default=(), exclude_if=lambda groups: not groups)


class PlanDecision(StrictModel):
    """One decision: its slot, the option chosen, who settled it, and the first line of why.

    A record ignores keys it does not declare, so a ledger record written by
    a release that adds a field still loads in one that does not.
    """

    model_config = ConfigDict(extra="ignore")

    slot: str
    chosen: str
    resolved_by: ResolvedBy
    rationale: str


class PlanWarning(StrictModel):
    """One evidence finding the compiled workflow carries."""

    code: str
    requirement_id: str
    detail: str


class PlanStep(StrictModel):
    """One workflow step: its id, the card's label, and the source and reading of a retrieval."""

    id: str
    label: str
    spec_id: str | None = None
    reading: StatVarReading | None = None
    failed: bool = False


class PlanAsk(StrictModel):
    """One output the user asked for and how the plan accounts for it.

    `outputs` are the workflow steps whose results answer it. `declined` is
    the code a decline of it names and `not_computed` the part that decline
    leaves out, in the user's words. An ask carries outputs, a decline, or
    both when the plan computes part of it.
    """

    ask: str
    outputs: tuple[str, ...] = ()
    declined: UnsupportedOperation | None = None
    not_computed: str | None = None


class PlanRecord(StrictModel):
    """A compiled plan as the agents that act on it after approval read it.

    The `Plan` itself stays in session state as the execution authority;
    this record is a projection of it. `revision` is `plan_revision(plan)`
    and `approved_revision` the revision the user approved, `None` before
    approval. `decisions` are those `plan_display.shown_decisions` lists.
    Periods, sources, warnings and steps name the requirements the model
    wrote, with the copies the split added folded into them
    (`plan_display.shown_plan`).
    `outputs` are the workflow's terminal step ids. `asks` are the outputs
    the user asked for, in the brief's order, each with the steps that
    answer it, its decline, or both. `declined` are the other parts of the
    question the plan does not compute. `run_id` is the run of the run
    ledger the plan executed as, left out when the plan admitted none, so a
    later turn reads an `EXECUTING` record's status from that run
    (`with_run_status`).

    A record ignores keys it does not declare, so a ledger record written by
    a release that adds a field still loads in one that does not.
    """

    model_config = ConfigDict(extra="ignore")

    question: str
    status: PlanStatus
    revision: str
    approved_revision: str | None
    periods: tuple[PlanPeriod, ...]
    sources: tuple[PlanSource, ...]
    decisions: tuple[PlanDecision, ...]
    asks: tuple[PlanAsk, ...] = ()
    declined: tuple[Decline, ...] = ()
    warnings: tuple[PlanWarning, ...]
    outputs: tuple[str, ...]
    steps: tuple[PlanStep, ...]
    run_id: str | None = Field(default=None, exclude_if=lambda run_id: run_id is None)

    @classmethod
    def from_plan(
        cls,
        plan: Plan,
        *,
        searched: Collection[str] = (),
        failed_steps: Collection[str] = (),
        run_id: str | None = None,
    ) -> PlanRecord:
        """Build the record of compiled `plan`.

        Args:
            plan: A plan carrying a translation and a workflow.
            searched: Spec ids the turn's `search` calls returned.
            failed_steps: The ids of the steps the run's stop landed on, if any.
            run_id: The run the plan executed as, if the ledger admitted one.

        A source's origin is `search` when `searched` names it, else
        `translator`.
        """
        assert plan.workflow is not None, "PlanRecord needs a compiled plan"
        shown = plan_display.shown_plan(plan)
        assert shown.workflow is not None
        steps = tuple(
            PlanStep(
                id=str(row["id"]),
                label=str(row["label"]),
                spec_id=_str_or_none(_args(row).get("spec_id")),
                reading=_reading(_args(row).get("stat_vars")),
                failed=row["id"] in failed_steps,
            )
            for row in plan_display.step_rows(plan)
        )
        return cls(
            question=plan.question,
            status=plan.status,
            revision=plan_revision(plan),
            approved_revision=plan.approved_revision,
            periods=_periods(shown),
            sources=_sources(shown, searched=searched),
            decisions=tuple(
                PlanDecision(
                    slot=decision.slot,
                    chosen=plan_display.with_source_names(decision.chosen, plan),
                    resolved_by=decision.resolved_by,
                    rationale=plan_display.with_source_names(_first_line(decision.rationale), plan),
                )
                for decision in plan_display.shown_decisions(plan)
            ),
            asks=_asks(plan),
            declined=tuple(decline for decline in plan.declined if decline.ask_ref is None),
            warnings=tuple(
                PlanWarning(
                    code=str(issue.code.value),
                    requirement_id=issue.requirement_id,
                    detail=_first_line(issue.detail),
                )
                for issue in shown.workflow.warnings
            ),
            outputs=tuple(shown.workflow.terminal_steps),
            steps=steps,
            run_id=run_id,
        )

    def with_run_status(self, runs: Mapping[str, RunStatus]) -> PlanRecord:
        """The record with an `EXECUTING` status replaced by its run's.

        `runs` maps run ids to their status in the run ledger. A run that an
        earlier turn left unfinished and a later one carried on finishes
        after its record was written, so the ledger states it now
        (`plan_status_of`). A record with another status, or whose run
        `runs` does not hold, is returned as it is.
        """
        run = runs.get(self.run_id) if self.run_id is not None else None
        if self.status is not PlanStatus.EXECUTING or run is None:
            return self
        return self.model_copy(update={"status": plan_status_of(run)})

    def digest_line(self) -> str:
        """The record as one line for the conversation digest.

        Segments, separated by ` | `: each period and its basis; each
        decision the planner assumed, with its rationale cut to 80
        characters; each source by spec_id and name with its coverage and
        origin; each decline with its code. Unbounded, since the digest cuts it.
        """
        segments = [
            "; ".join(f"period {p.start}-{p.end}, {_period_basis(p)}" for p in self.periods)
        ]
        assumed = [
            f"{d.slot} = {d.chosen}: {_cut(d.rationale, _RATIONALE_CUT)}"
            for d in self.decisions
            if d.resolved_by is ResolvedBy.PLANNER_DEFAULT
        ]
        if assumed:
            segments.append("assumed: " + "; ".join(assumed))
        segments.append(
            "sources: " + ", ".join(_source_text(s, measure=False) for s in self.sources)
        )
        not_computed = [f"{d.ask} [{d.code.value}]" for d in self.declined] + [
            f"{a.not_computed if a.outputs and a.not_computed else a.ask} [{a.declined.value}]"
            for a in self.asks
            if a.declined is not None
        ]
        if not_computed:
            segments.append("not computed: " + "; ".join(not_computed))
        return " | ".join(segment for segment in segments if segment)

    def source_readings(self, *, observed_at: datetime) -> tuple[SearchReading, ...]:
        """A reading for each source no `search` of the turn returned.

        A source the translator discovered is otherwise in no ledger, so a
        later turn's planner would never be shown it
        (`harness.studio_state.ConversationState.prior_ledger_state`). The
        reading carries no spec block, so a later pin of it is confirmed by
        a fresh search. The reading states the source's measure, spec_id,
        coverage kind, origin, how it is counted, the other counts it
        offers (`ElectedMeasure.other_counts`) and the other prices it offers
        (its alternatives on axis `unit`) and, when the record states
        coverage, its window.
        """
        return tuple(
            SearchReading(
                text=source.measure,
                status="definite",
                spec_id=source.spec_id,
                counted_as=source.elected_measure.label if source.elected_measure else None,
                alternatives=tuple(
                    CandidateReading(spec_id=other.spec_id, label=other.label)
                    for other in source.elected_measure.other_counts()
                )
                if source.elected_measure
                else (),
                other_prices=tuple(
                    CandidateReading(spec_id=other.spec_id, label=other.label)
                    for other in source.elected_measure.alternatives
                    if other.axis == "unit"
                )
                if source.elected_measure
                else (),
                window=(
                    WindowReading(
                        start=source.coverage.start,
                        end=source.coverage.end,
                        source=source.coverage_source or "coverage",
                    )
                    if source.coverage is not None
                    else None
                ),
                coverage_kind=source.coverage_kind,
                provenance_ref=source.provenance_ref,
                provenance_account=source.provenance_account,
                origin=source.origin,
                observed_at=observed_at,
            )
            for source in self.sources
            if source.origin != "search"
        )

    def render(self, *, budget: int) -> str:
        """The record as text of at most `budget` characters.

        Sections run header, periods, asks, declines, decisions, warnings,
        sources, outputs, steps. Asks and declines are never shed. Over
        budget the record sheds detail in `_SHED` order, one more each try, until it
        fits: steps other than outputs and the failed step folded into a
        count; outputs folded too; source measures dropped; sources folded
        into a count; rationales and warning details cut to 80 characters;
        then dropped, leaving each decision's choice and each warning's
        code. Text still over budget is cut to `budget`, which takes the
        steps, outputs and sources line before any warning, decision,
        decline or ask.
        """
        text = ""
        for level in range(len(_SHED) + 1):
            text = "\n".join(self._lines(frozenset(_SHED[:level])))
            if len(text) <= budget:
                return text
        return cut_user_text(text, budget)

    def _lines(self, shed: frozenset[str]) -> Iterator[str]:
        if self.approved_revision is None:
            approval = "not approved"
        elif self.approved_revision == self.revision:
            approval = "approved"
        else:
            approval = f"approved revision {self.approved_revision[:_REVISION_CHARS]}"
        yield (
            f"Plan {self.revision[:_REVISION_CHARS]} ({approval}), status {self.status.value}: "
            f'"{self.question}"'
        )
        for period in self.periods:
            yield f"period {period.start}-{period.end}, {_period_basis(period)}"
        if self.asks:
            yield "asks:"
        for number, ask in enumerate(self.asks):
            accounts = []
            if ask.outputs:
                accounts.append(f"answered by {', '.join(ask.outputs)}")
            if ask.declined is not None:
                part = f' "{ask.not_computed}"' if ask.outputs and ask.not_computed else ""
                accounts.append(f"not computed{part} [{ask.declined.value}]")
            yield f'- {number} "{ask.ask}": {"; ".join(accounts)}'
        if self.declined:
            yield "not computed, the system cannot do it:"
        for decline in self.declined:
            yield f"- {decline.ask} [{decline.code.value}]"
        if self.decisions:
            yield "decisions:"
        for decision in self.decisions:
            line = f"- {decision.slot} = {decision.chosen} [{decision.resolved_by.value}]"
            if "drop_details" not in shed:
                yield f"{line}: {_detail(decision.rationale, shed)}"
            else:
                yield line
        if self.warnings:
            yield "warnings:"
        for warning in self.warnings:
            line = f"- {warning.code} on {warning.requirement_id}"
            if "drop_details" not in shed:
                yield f"{line}: {_detail(warning.detail, shed)}"
            else:
                yield line
        yield "sources:"
        if "fold_sources" in shed:
            yield f"- {_count(len(self.sources), 'source')}"
        else:
            for source in self.sources:
                yield f"- {_source_text(source, measure='drop_measures' not in shed)}"
        yield f"outputs: {', '.join(self.outputs)}"
        yield "steps:"
        shown = [
            step
            for step in self.steps
            if "fold_steps" not in shed
            or step.failed
            or ("fold_outputs" not in shed and step.id in self.outputs)
        ]
        yield from (f"- {_step_text(step)}" for step in shown)
        if folded := len(self.steps) - len(shown):
            yield f"- and {_count(folded, 'more step')}" if shown else f"- {_count(folded, 'step')}"


def plan_status_of(status: RunStatus) -> PlanStatus:
    """The plan status a record states for a run at `status`: `COMPLETED`
    or `FAILED` once the run is, else `EXECUTING`."""
    return _PLAN_STATUS_OF_RUN.get(status, PlanStatus.EXECUTING)


_PLAN_STATUS_OF_RUN = {
    RunStatus.COMPLETED: PlanStatus.COMPLETED,
    RunStatus.FAILED: PlanStatus.FAILED,
}


# What `PlanRecord.render` sheds over budget, in order. Each try sheds one more.
_SHED = (
    "fold_steps",
    "fold_outputs",
    "drop_measures",
    "fold_sources",
    "cut_details",
    "drop_details",
)


def _detail(text: str, shed: frozenset[str]) -> str:
    return _cut(text, _RATIONALE_CUT) if "cut_details" in shed else text


def _count(number: int, noun: str) -> str:
    return f"{number} {noun}{'' if number == 1 else 's'}"


class WindowReading(StrictModel):
    """The window a search part states and which side settled it.

    `source` is `requested` when the window echoes the question's own range,
    `coverage` when it is the span the resolved items hold data across, and
    `elected` when it is the one year Search chose for a question naming no
    period (`harness.tool_contracts._window`).
    """

    start: int
    end: int
    source: str


class WarningReading(StrictModel):
    """One search warning, by code."""

    code: str
    severity: str | None = None
    message: str | None = None


class DepartureReading(StrictModel):
    """One way a search part's answer differs from what was asked: its code and its text."""

    code: str
    text: str


class CandidateReading(StrictModel):
    """One spec a `candidates` part offers."""

    spec_id: str
    label: str


class SiblingReading(StrictModel):
    """The reading that offered a reading's spec_id as another reading of its flow.

    `spec_id` is the offering reading's own. `axis` is `what` for another
    count (`SearchReading.alternatives`) and `unit` for the same count in
    other prices (`SearchReading.other_prices`). `label` is the offer's own
    label ("net disbursements", "constant prices"), `None` when the
    stored reading carries no label.
    """

    spec_id: str
    axis: Literal["what", "unit"]
    label: str | None = None


class OpenAxisReading(StrictModel):
    """An entity axis a search left open, and how many members it holds when counted."""

    column: str
    members: int | None = None


class YearCount(StrictModel):
    """Observations, and series when stated, in one recent year."""

    year: int
    observations: int
    series: int | None = None


class FetchReading(StrictModel):
    """What a `fetch` of the reading's spec_id returned: its summary, or a refusal code."""

    summary: str | None = None
    refusal_code: str | None = None


# The reading statuses whose own spec_id a search settled on: a `definite`
# part, and a candidate or nearest series the orchestrator fetched (`chosen`).
SETTLED_STATUSES = frozenset({"definite", "chosen"})


class SearchReading(StrictModel):
    """One search part as the turn read it.

    `text` is the part's variable text and `sentence` the resolved sentence
    of a `definite` part. `separate` preserves the part's comparison intent
    when search keeps it independent of other parts. `departures` are the ways search says the answer
    differs from what was asked. `skills` are the ids of the skills the search
    result named. `coverage_kind` (`exact`, `breadth`, `bare`),
    `requested_window_kind` (`range`, `all`, `latest`), `roles` and
    `open_axes` are empty when the part did not state them. `recent_years`
    is read from a stored conversation only: no reading gains it and it is
    not rendered. `roles` and `open_axes` are the series' own;
    `asked_roles` are the flow parties the search part asked for, each a
    `planner.parties.PARTY_ROLES` word, empty when it named none, and
    `open_roles` those of them it asked to hold open, every member listed,
    rather than name: each a `PARTY_ROLES` word, or the role column
    (`origin`, `destination`) for a word that names a side of the flow
    alone (`planner.parties.open_party`). `share_over` is the role column
    of each side a `share_of` operation of the part runs over, whose whole
    spans every member of it. `refusal_code` is the reason code a `no_data`
    part gave. `unbound_qualifiers` and `flagged_match` explain a
    `qualifier_not_bound` refusal: its rejected phrase and the broader match
    search left unused. `fetch` is what a later `fetch` of `spec_id` returned.
    `counted_as` is the measure the part's block names, `alternatives`
    each other way of counting the flow it offers and `other_prices` the
    same count in each other prices it offers, each with its spec_id
    (`harness.tool_contracts.measure_choice`); all empty when it offers
    none. `sibling_of` names the reading that offered this one's spec_id
    among its `alternatives` or `other_prices`, set on the reading a `fetch`
    of such a spec_id gains (`harness.orchestrator_tools`), whose
    `counted_as` is then the count it reads: the offered count, or the
    offering reading's own in other prices. `observed_at` is stored and not
    rendered.
    """

    text: str | None
    status: str
    separate: bool = Field(default=False, exclude_if=lambda separate: not separate)
    spec_id: str | None = None
    sentence: str | None = None
    warnings: tuple[WarningReading, ...] = ()
    departures: tuple[DepartureReading, ...] = ()
    candidates: tuple[CandidateReading, ...] = ()
    ambiguous_sources: tuple[AmbiguousSourceReading, ...] = Field(
        default=(), exclude_if=lambda groups: not groups
    )
    provenance_ref: str | None = Field(default=None, exclude_if=lambda ref: ref is None)
    provenance_account: str | None = Field(default=None, exclude_if=lambda account: account is None)
    window: WindowReading | None = None
    wide: bool | None = None
    skills: tuple[str, ...] = ()
    coverage_kind: str | None = None
    requested_window_kind: str | None = None
    roles: tuple[str, ...] = ()
    open_axes: tuple[OpenAxisReading, ...] = ()
    asked_roles: tuple[str, ...] = ()
    open_roles: tuple[str, ...] = Field(default=(), exclude_if=lambda roles: not roles)
    share_over: tuple[str, ...] = Field(default=(), exclude_if=lambda sides: not sides)
    recent_years: tuple[YearCount, ...] = ()
    refusal_code: str | None = None
    # Search may refuse a same-variable match rather than silently widen the
    # request. These carry the rejected user language and the broader match it
    # deliberately left unused; neither is a source a later turn may fetch.
    unbound_qualifiers: tuple[str, ...] = ()
    flagged_match: str | None = None
    fetch: FetchReading | None = None
    counted_as: str | None = None
    alternatives: tuple[CandidateReading, ...] = ()
    other_prices: tuple[CandidateReading, ...] = Field(
        default=(), exclude_if=lambda prices: not prices
    )
    sibling_of: SiblingReading | None = Field(default=None, exclude_if=lambda link: link is None)
    origin: Origin = "search"
    observed_at: datetime | None = None

    @model_validator(mode="before")
    @classmethod
    def _stored_count_link(cls, data: Any) -> Any:
        """A stored `other_count_of`, the bare spec_id of the offering reading,
        reads as a `sibling_of` on axis `what`."""
        if not isinstance(data, Mapping) or "other_count_of" not in data:
            return data
        data = dict(data)
        offering = data.pop("other_count_of")
        if offering is not None and "sibling_of" not in data:
            data["sibling_of"] = {"spec_id": offering, "axis": "what"}
        return data

    @property
    def other_count_of(self) -> str | None:
        """The spec_id of the reading that offered this one as another count of
        its flow (`sibling_of` on axis `what`): search elected that reading's
        measure for the part, not this one's `counted_as`. `None` for any other
        reading, a fetched price sibling included, whose count is the one search
        elected."""
        link = self.sibling_of
        return link.spec_id if link is not None and link.axis == "what" else None

    @classmethod
    def from_part(
        cls,
        part: Mapping[str, Any],
        *,
        skills: Sequence[str] = (),
        origin: Origin = "search",
        observed_at: datetime | None = None,
    ) -> SearchReading:
        """Read one `_part_summary` entry.

        Besides the keys `_part_summary` writes, reads the keys
        `harness.tool_contracts.reading_facts` writes and `asked_roles` and
        `open_roles` (`harness.orchestrator_tools`) when present. A `no_data` part's
        `reason` is the reading's `refusal_code`, and `counted_as`,
        `alternatives` and `other_prices` are
        `harness.tool_contracts.measure_choice`'s.
        """
        window = part.get("window")
        return cls(
            text=_str_or_none(part.get("variable")),
            status=str(part.get("status") or "unsupported"),
            separate=part.get("separate") is True,
            spec_id=_str_or_none(part.get("spec_id")),
            sentence=_str_or_none(part.get("resolved")),
            warnings=tuple(
                WarningReading(
                    code=str(warning["code"]),
                    severity=_str_or_none(warning.get("severity")),
                    message=_str_or_none(warning.get("message")),
                )
                for warning in part.get("warnings") or ()
                if isinstance(warning, Mapping) and warning.get("code")
            ),
            departures=tuple(
                DepartureReading.model_validate(departure)
                for departure in part.get("departures") or ()
            ),
            candidates=tuple(
                CandidateReading(spec_id=str(c["spec_id"]), label=str(c.get("label") or ""))
                for c in part.get("candidates") or ()
                if isinstance(c, Mapping) and c.get("spec_id")
            ),
            ambiguous_sources=tuple(
                AmbiguousSourceReading.model_validate(group)
                for group in part.get("ambiguous_sources") or ()
            ),
            provenance_ref=_str_or_none(part.get("provenance_ref")),
            provenance_account=_str_or_none(part.get("provenance_account")),
            window=WindowReading.model_validate(window) if isinstance(window, Mapping) else None,
            wide=part.get("wide"),
            skills=tuple(skills),
            coverage_kind=_str_or_none(part.get("coverage_kind")),
            requested_window_kind=_str_or_none(part.get("requested_window_kind")),
            roles=tuple(str(role) for role in part.get("entity_roles") or ()),
            open_axes=tuple(
                OpenAxisReading(column=str(axis["column"]), members=axis.get("members"))
                for axis in part.get("open_entity_axes") or ()
                if isinstance(axis, Mapping) and axis.get("column")
            ),
            asked_roles=tuple(str(role) for role in part.get("asked_roles") or ()),
            open_roles=tuple(str(role) for role in part.get("open_roles") or ()),
            share_over=tuple(str(side) for side in part.get("share_over") or ()),
            refusal_code=_str_or_none(part.get("reason")),
            unbound_qualifiers=tuple(
                dict.fromkeys(
                    qualifier
                    for qualifier in part.get("unbound_qualifiers") or ()
                    if isinstance(qualifier, str) and qualifier
                )
            ),
            flagged_match=(
                flagged_match
                if isinstance(flagged_match := part.get("flagged_match"), str)
                else None
            ),
            counted_as=_str_or_none(part.get("counted_as")),
            alternatives=_offers(part.get("alternatives")),
            other_prices=_offers(part.get("other_prices")),
            origin=origin,
            observed_at=observed_at,
        )

    def with_fetch(self, result: Mapping[str, Any]) -> SearchReading:
        """This reading with the outcome of a `fetch` result for its spec_id."""
        if result.get("code"):
            fetched = FetchReading(refusal_code=str(result["code"]))
        else:
            fetched = FetchReading(summary=_str_or_none(result.get("summary")))
        return self.model_copy(update={"fetch": fetched})


def _offers(offers: Sequence[Any] | None) -> tuple[CandidateReading, ...]:
    """A `measure_choice` list of `{spec_id, label}` offers, each with a spec_id."""
    return tuple(
        CandidateReading(spec_id=str(offer["spec_id"]), label=str(offer.get("label") or ""))
        for offer in offers or ()
        if isinstance(offer, Mapping) and offer.get("spec_id")
    )


def newest_per_spec_id(readings: Sequence[SearchReading]) -> list[SearchReading]:
    """`readings`, oldest first, with each spec_id held only by its last reading.

    A reading with no spec_id of its own (a `no_data` or `candidates` part)
    is kept wherever it stands.
    """
    last = {reading.spec_id: index for index, reading in enumerate(readings) if reading.spec_id}
    return [
        reading
        for index, reading in enumerate(readings)
        if reading.spec_id is None or last[reading.spec_id] == index
    ]


def measure_choice_text(
    counted_as: str,
    alternatives: Sequence[CandidateReading | MeasureAlternative],
    *,
    other_prices: Sequence[CandidateReading] = (),
    cap: int | None = None,
) -> str:
    """How a source is counted, each other count and each other prices, with its spec_id.

    For example "counted as gross disbursements; other counts: net
    disbursements [spec_…]; other prices: constant prices [spec_…]", at
    most `cap` characters when set. An offer past the cap is left out
    whole, since a cut spec_id pins nothing, and so is every offer after it.
    Empty when none is offered or none fits.
    """
    text = f"counted as {counted_as}"
    offered = False
    for lead, offers in (("other counts", alternatives), ("other prices", other_prices)):
        listed: list[str] = []
        for offer in offers:
            entry = f"{offer.label} [{offer.spec_id}]"
            section = f"; {lead}: " + ", ".join([*listed, entry])
            if cap is not None and len(text) + len(section) > cap:
                break
            listed.append(entry)
        if listed:
            text += f"; {lead}: " + ", ".join(listed)
            offered = True
        if len(listed) < len(offers):
            break
    return text if offered else ""


# What marks a search part as a loose match: search's weak-score warning on a
# reading search left as candidates, or its departure saying the answer shows
# the closest measure it holds. Search attaches the weak-score warning to
# `definite` readings too, and every recorded one bound the measure asked for
# ("official development assistance" to gross ODA disbursements), so on a
# `definite` reading the warning alone marks nothing.
RETRIEVAL_SCORE_WEAK = "retrieval_score_weak"
CLOSEST_MATCH = "closest_match"
# What a loose match whose scope is the one place `Earth` lacks.
_WORLDWIDE = "a worldwide total with no breakdown by country"


def loose_match_notes(
    readings: Sequence[Mapping[str, Any]],
    *,
    specs: Mapping[str, Mapping[str, Any]],
    plans: Sequence[Mapping[str, Any]],
) -> list[str]:
    """One sentence for each spec_id the turn used as a loose match, on what its series covers and lacks.

    A reading is a loose match when it carries a `closest_match` departure,
    or `retrieval_score_weak` and is no `definite` reading. The turn used it
    when a fetch of its spec_id returned data or a plan record names it as a
    source. The sentence
    names what was asked and the series' heading and byline
    (`harness.tool_contracts.series_description` of its block in `specs`,
    the reading's resolved sentence when the block is missing), then what
    the series lacks, read from its block:

    - "a worldwide total with no breakdown by country" when the block's
      scope (`resolved.scope.members`) is the one place `Earth`;
    - "no donor or recipient detail" when the search part asked for a flow
      party (`SearchReading.asked_roles`), the series measures money, and
      neither flow side is one of its entity roles or open axes. The clause
      names the two parties of each kind of flow asked about, so a creditor
      asked gives "no creditor or debtor detail". A block
      measures money when its unit (`resolved.facets.unit`) names a
      currency (`compiler.evidence.read_unit_currency`) or its provenance
      (`resolved.facets.provenance`) states account `provider`
      (`compiler.sources.PROVIDER_ACCOUNT`).

    A reading with no block gets neither clause.

    Args:
        readings: The turn's search readings, as `SearchReading` JSON.
        specs: The spec blocks the turn's searches returned, by spec_id.
        plans: The turn's plan records, as `PlanRecord` JSON.
    """
    planned = {
        source.get("spec_id")
        for plan in plans
        for source in plan.get("sources") or ()
        if isinstance(source, Mapping)
    }
    notes: dict[str, str] = {}
    for reading in (SearchReading.model_validate(r) for r in readings):
        if reading.spec_id is None or reading.spec_id in notes or not _loose(reading):
            continue
        fetched = reading.fetch is not None and reading.fetch.refusal_code is None
        if fetched or reading.spec_id in planned:
            notes[reading.spec_id] = _loose_match_note(reading, specs.get(reading.spec_id))
    return list(notes.values())


def _loose(reading: SearchReading) -> bool:
    weak = reading.status != "definite" and any(
        w.code == RETRIEVAL_SCORE_WEAK for w in reading.warnings
    )
    return weak or any(d.code == CLOSEST_MATCH for d in reading.departures)


def _loose_match_note(reading: SearchReading, block: Mapping[str, Any] | None) -> str:
    asked = f' to "{reading.text}"' if reading.text else ""
    described = series_description(block) if block else None
    if described is not None:
        heading, byline = described
        series = f"{heading} ({', '.join(byline)})" if byline else heading
    else:
        series = (reading.sentence or "").strip().removesuffix(".")
    lead = f"The closest series{asked} is a loose match"
    lacks = _lacks(reading, block) if block else ""
    return f"{lead}: {series}{lacks}." if series else f"{lead}{lacks}."


# Each `PARTY_ROLES` word's flow kind, as its giving and receiving words:
# `PartyRole` lists one giving and one receiving word per kind, in that order.
_FLOW_PARTIES: dict[str, tuple[str, str]] = {
    role: pair for pair in zip(PARTY_ROLES[::2], PARTY_ROLES[1::2], strict=True) for role in pair
}


def _lacks(reading: SearchReading, block: Mapping[str, Any]) -> str:
    """The clause `loose_match_notes` closes a note with, on what the series lacks."""
    worldwide = _worldwide(block)
    sides = set(reading.roles) | {axis.column for axis in reading.open_axes}
    parties = list(
        dict.fromkeys(word for role in reading.asked_roles for word in _FLOW_PARTIES.get(role, ()))
    )
    no_parties = bool(parties) and _measures_money(block) and not sides & FLOW_ROLE_COLUMNS.keys()
    detail = f"no {_either(parties)} detail"
    if worldwide and no_parties:
        return f", {_WORLDWIDE} and {detail}"
    if worldwide:
        return f", {_WORLDWIDE}"
    return f", with {detail}" if no_parties else ""


def _either(words: Sequence[str]) -> str:
    """`words` as alternatives: "donor or recipient", "donor, recipient, creditor, or debtor"."""
    if len(words) <= 2:
        return " or ".join(words)
    return f"{', '.join(words[:-1])}, or {words[-1]}"


def _worldwide(block: Mapping[str, Any]) -> bool:
    """Whether a spec block's scope is the one place `Earth`."""
    members = ((block.get("resolved") or {}).get("scope") or {}).get("members") or ()
    return (
        len(members) == 1 and isinstance(members[0], Mapping) and members[0].get("ref") == "Earth"
    )


def _measures_money(block: Mapping[str, Any]) -> bool:
    """Whether a spec block's unit names a currency or its dataset is a table of a provider's flows."""
    facets = (block.get("resolved") or {}).get("facets") or {}
    unit = facets.get("unit") or {}
    ref, label = unit.get("ref"), unit.get("label")
    currency = read_unit_currency(
        ref if isinstance(ref, str) else None, label if isinstance(label, str) else None
    )
    return (
        currency is not None or (facets.get("provenance") or {}).get("account") == PROVIDER_ACCOUNT
    )


def named_spec_ids(reading: SearchReading) -> list[str]:
    """The reading's own spec_id, when it has one, and each candidate's."""
    own = [reading.spec_id] if reading.spec_id else []
    return own + [candidate.spec_id for candidate in reading.candidates]


class RenderedReadings(NamedTuple):
    """`render_readings`' text, the spec_ids it names, and the other counts it names.

    `spec_ids` are the readings' own and their candidates', in the order the
    text names them. `other_counts` are the spec_ids of the other ways of
    counting a reading's flow the text names (`measure_choice_text`). A pin
    of one is honoured through the measure choice of the reading that
    offered it (`submission.service.measure_choices`), so it needs no pin source
    of its own.
    """

    text: str
    spec_ids: tuple[str, ...]
    other_counts: tuple[str, ...] = ()


def render_readings(
    readings: Sequence[SearchReading], *, budget: int, skills_read: Collection[str] = ()
) -> RenderedReadings:
    """Number `readings` oldest first, as text of at most `budget` characters.

    Every warning code shows once per reading, a blocking one
    (`BLOCKING_WARNING_CODES`) marked `!`. The skills every reading named
    close the text, once each, those in `skills_read` marked `*`.

    Levels, tried in order until one fits: everything, then sentences cut to
    160 characters and candidate labels to 60, then sentences and labels
    dropped, then the oldest readings folded one at a time to their first
    line and whether a fetch ran, then the oldest readings dropped one at a
    time with a count of how many. The last level's text is cut to `budget`
    if it still does not fit, and by then it names no reading.

    `spec_ids` holds each reading's own spec_id and, unless the reading is
    folded, each of its candidates', for every reading the text shows.
    `other_counts` holds the spec_id of each other count an unfolded
    reading names. A folded reading drops its measure choice.
    """
    rendered = RenderedReadings("", ())
    for detail in range(3):
        rendered = _ledger(readings, detail=detail, folded=0, dropped=0, skills_read=skills_read)
        if len(rendered.text) <= budget:
            return rendered
    for folded in range(1, len(readings) + 1):
        rendered = _ledger(readings, detail=2, folded=folded, dropped=0, skills_read=skills_read)
        if len(rendered.text) <= budget:
            return rendered
    for dropped in range(1, len(readings) + 1):
        rendered = _ledger(
            readings, detail=2, folded=len(readings), dropped=dropped, skills_read=skills_read
        )
        if len(rendered.text) <= budget:
            return rendered
    return rendered._replace(text=cut_user_text(rendered.text, budget))


def _ledger(
    readings: Sequence[SearchReading],
    *,
    detail: int,
    folded: int,
    dropped: int,
    skills_read: Collection[str],
) -> RenderedReadings:
    lines = ["Search readings (oldest first):"]
    named: dict[str, None] = {}
    counts: dict[str, None] = {}
    if dropped:
        lines.append(f"({dropped} earlier reading{'s' if dropped > 1 else ''} not shown)")
    for number, reading in enumerate(readings, 1):
        if number <= dropped:
            continue
        lines.extend(_reading_lines(number, reading, detail=detail, folded=number <= folded))
        if reading.spec_id:
            named[reading.spec_id] = None
        if number > folded:
            named.update(dict.fromkeys(candidate.spec_id for candidate in reading.candidates))
            if _measure_choice(reading):
                counts.update(dict.fromkeys(other.spec_id for other in reading.alternatives))
    skills = dict.fromkeys(skill for reading in readings for skill in reading.skills)
    if skills:
        listed = ", ".join(skill + ("*" if skill in skills_read else "") for skill in skills)
        lines.append(f"Skills named (* = already read): {listed}")
    return RenderedReadings("\n".join(lines), tuple(named), tuple(counts))


def _reading_lines(number: int, reading: SearchReading, *, detail: int, folded: bool) -> list[str]:
    status = f"no data: {reading.refusal_code}" if reading.refusal_code else reading.status
    head = f'{number}. "{reading.text}" {status}' if reading.text else f"{number}. {status}"
    if reading.spec_id:
        head += f" {reading.spec_id}"
    if reading.origin != "search":
        head += f" (from {reading.origin})"
    codes = _warning_codes(reading)
    if folded:
        if reading.fetch is not None:
            head += "; fetch refused" if reading.fetch.refusal_code else "; fetched"
        if refusal := _refusal_context(reading, detail=3):
            head += f"; {refusal}"
        return [head + (f"; warnings: {codes}" if codes else "")]
    lines = [head]
    if reading.sentence and detail < 2:
        sentence = reading.sentence if detail == 0 else _cut(reading.sentence, _SENTENCE_CUT)
        lines.append(f"   {sentence}")
    if refusal := _refusal_context(reading, detail=detail):
        lines.append(f"   {refusal}")
    for candidate in reading.candidates:
        label = "" if detail == 2 else f" {_cut(candidate.label, _LABEL_CUT if detail else None)}"
        lines.append(f"   candidate {candidate.spec_id}{label}")
    for group in reading.ambiguous_sources:
        lines.append(f"   source {group.text!r} could mean:")
        lines.extend(
            f"   {candidate.label} [{candidate.ref}] ({candidate.kind})"
            for candidate in group.candidates
        )
    facts = _reading_facts(reading)
    if facts:
        lines.append(f"   {facts}")
    if choice := _measure_choice(reading):
        lines.append(f"   {choice}")
    if sibling := _sibling_text(reading):
        lines.append(f"   {sibling}")
    if codes:
        lines.append(f"   warnings: {codes}")
    if reading.fetch is not None:
        if reading.fetch.refusal_code:
            lines.append(f"   fetch refused: {reading.fetch.refusal_code}")
        else:
            lines.append(f"   fetched: {reading.fetch.summary or 'no summary'}")
    return lines


def _measure_choice(reading: SearchReading) -> str:
    if reading.counted_as is None:
        return ""
    return measure_choice_text(
        reading.counted_as, reading.alternatives, other_prices=reading.other_prices
    )


def _sibling_text(reading: SearchReading) -> str:
    """What a fetched sibling reads and the reading that offered it.

    "counted as net disbursements, another count of spec_…" on axis
    `what`; "counted as gross disbursements in constant prices, other prices
    of spec_…" on axis `unit`. Empty for a reading no other offered.
    """
    link = reading.sibling_of
    if link is None:
        return ""
    if link.axis == "what":
        measure = f"counted as {reading.counted_as}, " if reading.counted_as else ""
        return f"{measure}another count of {link.spec_id}"
    prices = f" in {link.label}" if link.label else ""
    measure = f"counted as {reading.counted_as}{prices}, " if reading.counted_as else ""
    return f"{measure}other prices of {link.spec_id}"


def _reading_facts(reading: SearchReading) -> str:
    facts: list[str] = []
    if reading.window is not None:
        facts.append(
            f"window {reading.window.start}-{reading.window.end} ({reading.window.source})"
        )
    if reading.coverage_kind:
        facts.append(f"coverage {reading.coverage_kind}")
    if reading.requested_window_kind:
        facts.append(f"asked {reading.requested_window_kind}")
    if reading.wide:
        facts.append("wide")
    if reading.roles:
        facts.append(f"roles {', '.join(column_party_roles(role) for role in reading.roles)}")
    for axis in reading.open_axes:
        members = f" ({axis.members:,} members)" if axis.members is not None else ""
        facts.append(f"open {column_party_roles(axis.column)}{members}")
    return "; ".join(facts)


def _refusal_context(reading: SearchReading, *, detail: int) -> str:
    """The user language a `qualifier_not_bound` refusal must keep visible.

    The flagged sentence is explanatory only. It has no spec id and is not a
    candidate, so rendering it cannot make the broader match fetchable.
    """
    facts: list[str] = []
    if detail == 0:
        phrase_cap, match_cap = None, None
    elif detail == 1:
        phrase_cap, match_cap = _REFUSED_PHRASES_CUT, _FLAGGED_MATCH_CUT
    elif detail == 2:
        phrase_cap, match_cap = _TIGHT_REFUSED_PHRASES_CUT, _TIGHT_FLAGGED_MATCH_CUT
    else:
        phrase_cap, match_cap = _FOLDED_REFUSED_PHRASES_CUT, _FOLDED_FLAGGED_MATCH_CUT
    if reading.unbound_qualifiers:
        phrases = ", ".join(reading.unbound_qualifiers)
        facts.append(f"refused phrases: {_cut(phrases, phrase_cap)}")
    if reading.flagged_match:
        facts.append(f"broader unused match: {_cut(reading.flagged_match, match_cap)}")
    return "; ".join(facts)


def _warning_codes(reading: SearchReading) -> str:
    codes = dict.fromkeys(warning.code for warning in reading.warnings)
    return ", ".join(("!" if code in BLOCKING_WARNING_CODES else "") + code for code in codes)


def _periods(plan: Plan) -> tuple[PlanPeriod, ...]:
    """One entry per distinct period the dataset requirements ask for, with the decision fixing it."""
    decisions = {decision.slot: decision for decision in plan.decisions}
    grouped: dict[tuple[int, int, str | None], list[str]] = {}
    for requirement in plan.evidence_requirements:
        if not isinstance(requirement, DatasetRequirement):
            continue
        fixing = next(
            (
                slot
                for slot in requirement.decision_ids
                if (fixes := decisions[slot].fixes) is not None and fixes.period is not None
            ),
            None,
        )
        key = (requirement.period.start, requirement.period.end, fixing)
        grouped.setdefault(key, []).append(requirement.id)
    return tuple(
        PlanPeriod(
            start=start,
            end=end,
            assumed=slot is not None and decisions[slot].resolved_by is ResolvedBy.PLANNER_DEFAULT,
            decision=slot,
            requirement_ids=tuple(ids),
        )
        for (start, end, slot), ids in grouped.items()
    )


def _sources(plan: Plan, *, searched: Collection[str]) -> tuple[PlanSource, ...]:
    """One entry per distinct spec_id the workflow retrieves, in binding order."""
    assert plan.workflow is not None
    measures = {
        requirement.id: requirement.measure
        for requirement in plan.evidence_requirements
        if isinstance(requirement, DatasetRequirement)
    }
    seen: dict[str, PlanSource] = {}
    for binding in plan.workflow.evidence_bindings:
        source = plan.workflow.source_of(binding.step_id)
        if source is None or source.spec_id in seen:
            continue
        family = source.stat_var_reading is not None
        window = source.coverage_window
        asked = requested_range(source.identity or {})
        origin: Origin = "search" if source.spec_id in searched else "translator"
        seen[source.spec_id] = PlanSource(
            spec_id=source.spec_id,
            name=source.provenance_label,
            provenance_ref=source.provenance_ref,
            provenance_account=source.provenance_account,
            resolved_sources=source.resolved_sources,
            measure=measures.get(binding.requirement_id),
            coverage_kind=source.coverage_kind,
            coverage=YearSpan(start=window.start, end=window.end) if window else None,
            coverage_source=(
                None
                if window is None
                else "requested"
                if asked is not None and asked[0] <= window.start and window.end <= asked[1]
                else "coverage"
            ),
            members=stat_var_count(source) if family else None,
            origin=origin,
            identity=source.identity,
            groups=source.groups,
            elected_measure=source.elected_measure,
        )
    return tuple(seen.values())


def _period_basis(period: PlanPeriod) -> str:
    if period.decision is None:
        return "as the requirements state it, no period decision"
    if period.assumed:
        return f"assumed by the planner (decision {period.decision})"
    return f"chosen by the user (decision {period.decision})"


def _source_text(source: PlanSource, *, measure: bool) -> str:
    text = source.spec_id
    if source.name:
        text += f" {source.name}"
    if measure and source.measure:
        text += f" for {source.measure}"
    facts = []
    if source.coverage is not None:
        kind = f" ({source.coverage_kind})" if source.coverage_kind else ""
        if source.coverage_source == "requested":
            facts.append(f"data {source.coverage.text()}{kind} within the years asked")
        else:
            facts.append(f"coverage {source.coverage.text()}{kind}")
    else:
        facts.append("coverage not stated")
    if source.members is not None:
        facts.append(f"family of {source.members:,} StatVars")
    facts.append(f"from {source.origin}")
    return f"{text}; {'; '.join(facts)}"


def _step_text(step: PlanStep) -> str:
    text = f"{step.id}: {step.label}"
    retrieval = ", ".join(
        part for part in (step.spec_id, step.reading.value if step.reading else None) if part
    )
    if retrieval:
        text += f" ({retrieval})"
    if step.failed:
        text += " [failed]"
    return text


def _args(row: Mapping[str, object]) -> dict[str, Any]:
    return cast("dict[str, Any]", row["args"])


def _reading(value: object) -> StatVarReading | None:
    return StatVarReading(value) if value is not None else None


def _asks(plan: Plan) -> tuple[PlanAsk, ...]:
    """Each ask `plan` carries, with the steps answering it or its decline (`plan_display.ask_rows`)."""
    return tuple(
        PlanAsk(
            ask=str(row["ask"]),
            outputs=tuple(cast("list[str]", row["answered_by"])),
            declined=cast("UnsupportedOperation | None", row["declined"]),
            not_computed=cast("str | None", row["not_computed"]),
        )
        for row in plan_display.ask_rows(plan)
    )


def _str_or_none(value: object) -> str | None:
    return str(value) if value is not None else None


def _first_line(text: str) -> str:
    return text.strip().splitlines()[0] if text.strip() else ""


def _cut(text: str, cap: int | None) -> str:
    """`text` cut to `cap` characters ending in `CUT_MARKER`, or whole when `cap` is `None`."""
    return text if cap is None else cut_user_text(text, cap)

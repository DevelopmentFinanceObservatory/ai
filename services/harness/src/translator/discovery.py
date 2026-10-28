"""The translator's read-only look at what data exists, before anything is retrieved.

One tool, built over the same `ToolSurface` the compiler reads and the
executor will call: `discover_source` wraps data-mcp's `search`, one part per
call. `describe` is not a model tool; `fetch_definitions` calls it from code,
enriching each source `propose_operations` is about to compile with its own
definition before the plan reaches the user. Nothing here wraps `get_data`
or an analysis verb, so observations and computation are out of reach by
construction rather than by instruction.

`discover_source` takes only a requirement id: the search itself -- the
variable, the entities and their roles, the time window, the unit and its
price basis -- is built from the requirement the plan already states, never
from fields a model could supply that contradict it. A `no_data` or
`unsupported` answer refers straight back to the planner: rewording the
search is a second model judgement, and the planner is where the requirement
itself changes instead, carrying the reason and, for `no_data`, the nearest
real sources search found. Whatever phrase is sent is stamped on every source
the call returns as `searched_as`, so the workflow the user approves shows
the words the source answered.

`search` may drop a phrase it does not recognise -- a role, a time window --
and answer the narrower ask, saying so only in a warning. An answer carrying
one of those warnings is refused as `ask_not_understood` and its record is
never selectable, pinned or not: the source found is a source for a different
question.

Every `search` call is counted against `DISCOVERY_CALL_BUDGET` for the plan
*revision* it serves, and a second, narrower ceiling,
`DISCOVERY_CALL_BUDGET_PER_REQUIREMENT`, bounds what any one requirement may
spend of it, so a single requirement repeatedly searching cannot drain the
budget a whole revision's several requirements share. A `describe` call is
recorded the same way but does not count against either ceiling: it is
`fetch_definitions` enriching a source the compile loop already chose, not a
model spending the budget, and `_tried_source_ids` already stops it
repeating for the same source. Both budgets reset together when the plan is
amended into a fresh revision. Selectability and a requirement's candidates
are scoped
differently: by `requirement_revision`, a content hash of the requirement's
own searched fields (`measure`, `entities`,
`period`, `unit`, `source_spec_id`), not by the plan's own revision. An edit
to a different requirement or to a decision changes the plan's `revision` --
and so gives the whole revision a fresh call budget -- without changing an
untouched requirement's own key, so what discovery already found for it stays
selectable rather than needing to be re-found. A retrieval may name only a
`spec_id` a `definite` record named for its own requirement, and a referral's
candidates come from the log, so a model can neither invent a source nor pick
one quietly from several, nor borrow a source discovered for a different
requirement -- the log keeps each requirement's own view of a source, even one
two requirements both found.

ADK dispatches several function calls from one turn with `asyncio.gather` over
one shared `session.state` dict, so two concurrent `discover_source` calls can
each want to spend the same slice of `DISCOVERY_CALL_BUDGET` before either has
written a record for it. `discovery_lock` serialises only the budget check,
the reservation it makes, and the log's own read-modify-write; the `search`
call to data-mcp, and `fetch_definitions`'s own `describe` round trip, both
run with the lock released, so several calls reach the server at once. A
reservation is in-process bookkeeping, never persisted: it holds a caller's
claim on the budget between the moment its check passes and the moment its
record settles that claim, so a sibling's own check inside the same wave
counts it too and the two cannot both spend the same last slot. The lock and
its reservations are per session and live only in the process, the same
place mid-turn session state already lives, so neither one serialises a
call across processes. A call whose deadline elapses raises
`ToolSurfaceUnavailableError` (`compiler.surface`'s per-tool timeout), and
that call is always recorded, marked `transport_failed`, so a search can
never make a source selectable that way and a describe never leaves a
definition behind that way. A search's own timeout is counted against
`DISCOVERY_CALL_BUDGET` exactly like an answered search, so a persistently
slow server cannot buy the model unbounded retries; a describe's timeout is
recorded the same way but, like every
describe record, spends neither budget -- `_tried_source_ids` is what
stops a failing describe being retried on every `COMPILE_FAILED` pass within
the same revision. The worst case the lock and the search budget buy
together is one `search` deadline, 60 s (the `call_timeout_s` of the `search`
capability row), per wave of concurrent calls
the budget admits, rather than one deadline per call. A turn discovering
`DISCOVERY_CALL_BUDGET` requirements at once waits out that one deadline
together. Inside a planner entry a call is also cut off at the request's
wall-clock bound (`call_within_entry`), and recorded as a
timeout. A search the entry had no time left to issue (`NoTimeLeftError`)
is not recorded and spends nothing, and refers the requirement back marked
`not_issued`. A cut `resolve_definition` keeps running for the entry's next
proposal to be served (`CutCalls`). Once the search budget is spent, a caller sees the same
`budget_exhausted` refusal (`_budget_error`, action `return_to_planner`)
whether it was spent on answers or on timeouts. The referral it names is
the transient `discovery_unavailable` when the most recent call that spent
it timed out and `discovery_budget_exhausted` otherwise (`_budget_referral`).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import uuid
from collections import Counter
from collections.abc import Awaitable, Callable, Collection, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from enum import StrEnum
from typing import Any, Literal, NamedTuple, cast
from weakref import WeakValueDictionary

import rfc8785
from common.hashed_fields import revision_content
from compiler.evidence import (
    named_members,
    named_on_open_axes,
    named_refs,
    open_axes_for,
    source_requirement_matches,
)
from compiler.shapes import BREAKDOWN_CODE_SYSTEMS
from compiler.sources import (
    EXACT_COVERAGE,
    SOURCE_CODE_REFUSALS,
    DiscoveryStatus,
    PartWarning,
    bound_codes,
    outcome_from_envelope,
    pinned_stat_var,
    source_from_spec_block,
    stat_var_count,
    stat_var_count_is_floor,
    wire_groups,
)
from compiler.surface import ToolCallResult, ToolSurface, ToolSurfaceUnavailableError
from data_mcp import node_lookup, resolution
from data_mcp.definition_store import IdentityMismatch
from data_mcp.definition_writes import IdentityUnusable, caller_definition
from data_mcp.errors import UpstreamError
from data_mcp.graph_wire import CodeNode
from data_mcp.node_lookup import described_variable_text
from data_mcp.search_result import SEARCH_ENVELOPE_META_KEY
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.tool_context import ToolContext
from planner.parties import FlowSide, across_party, flow_side, party_role
from planner.plan import (
    AggregateRequirement,
    DatasetRequirement,
    Entity,
    Plan,
    PlanStatus,
    QuotientRequirement,
    RankRequirement,
    SessionState,
    ShareOfRequirement,
    Unit,
    authored,
    dataset_of,
    discovery_time_left,
    identity_on_record,
    inputs_of,
    load_plan,
    pin_sources,
    plan_revision,
    retrieves_period,
    save_plan,
    side_entities,
    split_of,
    windowed_pins,
    with_authored_requirements,
)
from planner.records import (
    AmbiguousEntityReading,
    AmbiguousSourceReading,
    EntityKey,
    GroupBy,
    Identifier,
    ResolvedSourceMention,
    SpecId,
    StrictModel,
    warn_left_out,
)
from planner.tool_results import Action, Outcome, ProblemCode, ok, problem
from planner.workflow import ResolvedGroup, SelectedSource, ToolId
from pydantic import Field, JsonValue, ValidationError

from translator.referral import (
    ReferralKind,
    WithheldSource,
    closing_group,
    dropped_scope,
    scope_repair,
)

logger = logging.getLogger(__name__)

# How many `search` calls one plan revision may make in total; a `describe`
# record never spends this budget, only `search` does (`DiscoveryLog.calls`).
# Typical plans carry one to three requirements, each needing one search, so
# 8 leaves headroom for a requirement's own referral rounds -- a
# `candidates`, `no_data` or `unsupported` answer sends the planner back to
# rephrase and search again, and each round is its own spend whether it runs
# alone or alongside a sibling requirement's own search.
DISCOVERY_CALL_BUDGET = 8

# How many of `DISCOVERY_CALL_BUDGET`'s `search` calls any one requirement
# may spend within the same revision -- half, so a requirement stuck in
# referral rounds, rephrasing and searching again, cannot spend what the
# revision's other requirements still need of their own share.
DISCOVERY_CALL_BUDGET_PER_REQUIREMENT = 4

# The session-state slot holding the `DiscoveryLog`. Session-scoped, so what
# one revision discovered stays known to a later one. `calls` is counted per
# plan revision; `selectable`, `candidates_for`, `pinnable_for` and `nearest_for` are counted
# per requirement instead -- see `requirement_revision`.
DISCOVERY_STATE_KEY = "translator:discovery"

# The session-state slot holding this session's lock token. See `discovery_lock`.
DISCOVERY_LOCK_TOKEN_KEY = "translator:discovery_lock_token"

# One `asyncio.Lock` per session token. Weak, so a session's lock is collected
# once no call holds it; a waiter keeps its lock alive through `async with`,
# so a lock is never collected out from under a blocked caller.
_LOCKS: WeakValueDictionary[str, asyncio.Lock] = WeakValueDictionary()

SEARCH = ToolId(server="data-mcp", name="search")
DESCRIBE = ToolId(server="data-mcp", name="describe")
RESOLVE_DEFINITION = ToolId(server="data-mcp", name="resolve_definition")

# `describe`'s own batch cap: "one to eight identifiers" (data-mcp's node door).
DESCRIBE_ID_BATCH_MAX = 8

# The `search` warning codes under which part of the ask was dropped rather
# than refused: the phrase is absent from what was searched, so the answer
# is to a narrower ask than the requirement states. Every other code --
# `facet_binding_unsupported`, `default_not_declared` -- is advisory and
# rides out with the answer.
BLOCKING_WARNING_CODES = frozenset({"role_text_not_recognised", "time_text_not_parsed"})

# The answers of the translator's own search a pin may be honoured from. A
# pin the planner chose among a `source_ambiguous` referral's candidates is
# honoured from the `candidates` answer a search for the pinned requirement
# gives again. A `no_data` answer's nearest real series are offered instead
# of the ask, not as it, so a pin among them is not the source that search
# found for the requirement. A pin of a series the planner was shown, a
# nearest series the orchestrator fetched included, is honoured from its
# block or its identity before any search (`_source_from_resolve_pin`).
_PIN_HONOURING = frozenset({DiscoveryStatus.DEFINITE, DiscoveryStatus.CANDIDATES})

# The referral kind to point at for each answer that is not `definite`. A
# `definite` answer carries no hint: it names the source.
_HINT_KIND: dict[DiscoveryStatus, ReferralKind] = {
    DiscoveryStatus.CANDIDATES: ReferralKind.SOURCE_AMBIGUOUS,
    DiscoveryStatus.NO_DATA: ReferralKind.NO_DATA,
    DiscoveryStatus.UNSUPPORTED: ReferralKind.UNSUPPORTED_ASK,
}


class DiscoveryRecord(StrictModel):
    """One discovery call: what it asked about and what came back.

    `revision` is the `plan_revision` of the draft the call served, which is
    what scopes `calls` and `DISCOVERY_CALL_BUDGET` to one revision of the
    plan. `requirement_key` is the same call's requirement's own
    `requirement_revision`, which is what scopes `selectable`,
    `candidates_for`, `pinnable_for` and `nearest_for` instead -- unlike `revision`, it does
    not change when a different requirement or a decision is edited, only
    when this requirement's own searched content or pin does. Both are
    carried because they answer different questions: `revision` bounds cost
    for the whole plan revision, `requirement_key` says whether a past
    answer still speaks to the requirement as it stands now. `variable_sent`
    is the phrase actually sent, the requirement's own `measure`. `reason`
    is a `no_data` answer's own
    closed reason code (`compiler.sources.DiscoveryOutcome.reason`), `None`
    for every other status. `warnings` are every warning the answer carried,
    and `blocked_by` the codes among them under which `search` dropped part
    of the ask; a record with any is never selectable. `transport_failed`
    marks a call whose deadline elapsed before `data-mcp` answered at all:
    `status` stays `None` on such a record, same as a `describe` call's own
    record, so a dedicated field is what tells the two apart rather than
    overloading `status`'s absence to mean two different things.

    `id` and `sequence` exist for the persisted log, not for a record's own
    meaning: `id` is this record's key in `save_log`'s dict, minted once at
    construction so two records never collide there; `sequence` is this
    record's position in append order, assigned by `DiscoveryLog.with_record`
    from the log it is appended to, not by the caller constructing the
    record. `load_log` sorts by `sequence` to rebuild `records` in append
    order after a union of several calls' own snapshots, which does not
    otherwise carry that order.

    `tool == "pin"` records a pin honoured with no `search` call
    (`_source_from_resolve_pin`): a requirement pinning a dataset the planner
    was shown, or one whose identity is on record over the requirement's
    period. Like a `describe` record, it spends neither budget
    (`calls`/`requirement_calls` count only `search`). It carries
    `status == DEFINITE` and one `spec_ids` entry. A `pin` record with no
    `status` and one `spec_ids` entry records the pin's source read and set
    aside, because it holds closed the side the requirement's `across` asks for every member of: no
    retrieval may take it, and the sources the fresh search finds in its
    place are offered only when they keep its scope or a reader reads that
    side per member (`DiscoveryLog.scope_kept`).

    `tool == "resolve"` records the graph resolve of the requirement's
    entity names that match no label of its one `spec_ids` source
    (`resolve_entity_refs`), which spends no budget either; the refs it
    found are on that source's `entity_refs`. A resolve that failed, in
    transport or with an unreadable answer, is `transport_failed` too, with
    the error as its `detail`.

    `tool == "code_paths"` records the read of the hierarchy above the codes
    the requirement's one `spec_ids` source names on its breakdown's code
    system (`fetch_code_paths`), which spends no budget either; the paths
    found are on that source's `code_paths`, and a failed read is
    `transport_failed` with the error as its `detail`.
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    sequence: int = 0
    revision: str
    tool: Literal["search", "describe", "pin", "resolve", "code_paths"]
    requirement_id: Identifier | None = None
    requirement_key: str | None = None
    status: DiscoveryStatus | None = None
    spec_ids: tuple[SpecId, ...] = ()
    detail: str | None = None
    reason: str | None = None
    variable_sent: str | None = None
    warnings: tuple[PartWarning, ...] = ()
    ambiguous_sources: tuple[AmbiguousSourceReading, ...] = ()
    ambiguous_entities: tuple[AmbiguousEntityReading, ...] = Field(
        default=(), exclude_if=lambda groups: not groups
    )
    blocked_by: tuple[str, ...] = ()
    transport_failed: bool = False

    @property
    def blocked(self) -> bool:
        """Whether this record's answer is one no retrieval may take a source from."""
        return bool(self.blocked_by)


def _select_definite(
    records: Sequence[DiscoveryRecord], *, pinned_spec_id: str | None
) -> DiscoveryRecord | None:
    """The latest of `records` that names the source `selectable` and `definite_source_ids` use.

    Scans backward -- `records` in call order, latest last -- stopping at the
    first (latest) record that qualifies, so an unqualifying record in
    between (a `None`-status re-search, say) is passed over rather than
    ending the scan. With `pinned_spec_id` set, a record qualifies by naming
    it among a `_PIN_HONOURING` status's own `spec_ids`; unset, only an
    outright `DiscoveryStatus.DEFINITE` record qualifies. Neither branch
    considers a `blocked` record: `search` answered a narrower ask than the
    requirement states, so what it found is not a source for the requirement
    as it stands.
    """
    for record in reversed(records):
        if pinned_spec_id is not None:
            if (
                record.status in _PIN_HONOURING
                and not record.blocked
                and pinned_spec_id in record.spec_ids
            ):
                return record
        elif record.status is DiscoveryStatus.DEFINITE and not record.blocked:
            return record
    return None


class ScopeKept(NamedTuple):
    """The sources found in place of a pin, as a `pinned_source_not_found` referral carries them (`DiscoveryLog.scope_kept`)."""

    offered: tuple[SelectedSource, ...]
    withheld: tuple[WithheldSource, ...] = ()
    repair: str | None = None


class _RecordedSourceDefinition(NamedTuple):
    identity: Mapping[str, Any]
    groups: tuple[ResolvedGroup, ...]
    resolved_sources: tuple[ResolvedSourceMention, ...] = ()


class DiscoveryLog(StrictModel):
    """Every discovery call a session made, and every source those calls returned.

    `sources` is keyed by requirement id, then `spec_id`: a source two
    requirements both found is kept once per requirement, each with the
    phrase that requirement searched it as.
    """

    records: tuple[DiscoveryRecord, ...] = ()
    sources: dict[str, dict[str, SelectedSource]] = Field(default_factory=dict)

    def calls(self, revision: str) -> int:
        """How many `search` calls `revision` has made in total.

        Counts only `tool == "search"`: a `describe` record is
        `fetch_definitions` enriching a source the compile loop already
        chose, not a model spending the budget, and a `pin` record spent no
        call at all, so neither is counted here.
        """
        return sum(
            1 for record in self.records if record.revision == revision and record.tool == "search"
        )

    def requirement_calls(self, revision: str, requirement_id: str) -> int:
        """How many of `revision`'s `search` calls were spent on `requirement_id`.

        Scoped by `requirement_id`, not `requirement_key`: this bounds one
        plan slot's own share of one revision's budget, and two distinct
        requirements that happen to search for the same content (an edge
        case, not the common one) each still get their own share rather than
        being counted as if they were one. Counts only `tool == "search"`,
        for the same reason `calls` does: a `describe` record is not a
        model's own spend, and a `pin` record spent no call at all.
        """
        return sum(
            1
            for record in self.records
            if record.revision == revision
            and record.requirement_id == requirement_id
            and record.tool == "search"
        )

    def last_call_timed_out(self, revision: str, requirement_id: str | None = None) -> str | None:
        """The requirement of `revision`'s most recent `search` call, or of its
        most recent one on `requirement_id`, when that call timed out; `None`
        when it was answered or there is none."""
        last = next(
            (
                record
                for record in reversed(self.records)
                if record.revision == revision
                and record.tool == "search"
                and requirement_id in (None, record.requirement_id)
            ),
            None,
        )
        return last.requirement_id if last is not None and last.transport_failed else None

    def selectable(
        self,
        *,
        requirement_keys: Mapping[str, str],
        pinned: Mapping[str, str],
        across: Collection[str] = (),
    ) -> dict[str, SelectedSource]:
        """The sources a retrieval may name, one entry per requirement `requirement_keys` names.

        `requirement_keys` maps each dataset requirement's id to its own
        `requirement_revision`: a record counts toward a requirement only when
        its `requirement_key` matches the entry for its `requirement_id`, so
        what a past plan revision found still applies as long as this
        requirement's own searched content and pin have not changed, whatever
        happened elsewhere in the plan since.

        For a requirement id not in `pinned`, the single source of the
        latest matching `definite` record. For a requirement id in `pinned`,
        instead the source with that pinned `spec_id`, taken from the latest
        matching `definite` or `candidates` record that names that
        `spec_id` -- nothing when none does, even if a `definite` record
        named a different source, and nothing from a `no_data` answer whose
        nearest real series happen to include it. A record `search` answered
        with part of the ask dropped (`blocked`) is passed over in both
        cases: a pin cannot resurrect it.

        `across` names the requirement ids that state `across`. Such a
        requirement reads its source's open entity axes, so a source
        recorded without them (`SelectedSource.open_entity_axes` of `None`)
        is left out for it and the requirement is discovered again.
        `requirement_revision` is the same with and without an `across` that
        opens no role (the place), so the record's key cannot tell the two
        apart.
        """
        by_requirement: dict[str, list[DiscoveryRecord]] = {}
        for record in self.records:
            if record.requirement_id is None:
                continue
            key = requirement_keys.get(record.requirement_id)
            if key is not None and record.requirement_key == key:
                by_requirement.setdefault(record.requirement_id, []).append(record)

        result: dict[str, SelectedSource] = {}
        for requirement_id, records in by_requirement.items():
            pinned_spec_id = pinned.get(requirement_id)
            record = _select_definite(records, pinned_spec_id=pinned_spec_id)
            if record is None:
                continue
            if pinned_spec_id is not None:
                source = self.sources[requirement_id][pinned_spec_id]
            else:
                (spec_id,) = record.spec_ids
                source = self.sources[requirement_id][spec_id]
            if requirement_id in across and source.open_entity_axes is None:
                continue
            result[requirement_id] = source
        return result

    def definite_source_ids(
        self, *, requirement_keys: Mapping[str, str], pinned: Mapping[str, str]
    ) -> frozenset[str]:
        """The subset of `selectable`'s own requirement ids whose source is genuinely `definite`-origin.

        `selectable` returns one `SelectedSource` per requirement id whether
        it came from an outright `definite` answer or from a pin honoured out
        of a `candidates` answer, with nothing in its own return value saying
        which. `fetch_definitions` reads only the former here: a pinned
        candidate got no definition-based confirmation either, only a
        referral the user answered without seeing the source's own
        definition. Takes the same `requirement_keys`/`pinned` arguments as
        `selectable`, and mirrors its own two branches, narrowed to
        `DiscoveryStatus.DEFINITE`.
        """
        result: set[str] = set()
        for requirement_id, key in requirement_keys.items():
            matching = [
                record
                for record in self.records
                if record.requirement_id == requirement_id and record.requirement_key == key
            ]
            record = _select_definite(matching, pinned_spec_id=pinned.get(requirement_id))
            # `_select_definite` also returns a pin honoured out of a
            # `CANDIDATES` record; narrowed to `DEFINITE` here, per this
            # method's own docstring.
            if record is not None and record.status is DiscoveryStatus.DEFINITE:
                result.add(requirement_id)
        return frozenset(result)

    def candidates_for(
        self, requirement_key: str, requirement_id: str
    ) -> tuple[SelectedSource, ...]:
        """The candidates the latest matching `candidates` answer for `requirement_id` returned."""
        for record in reversed(self.records):
            if (
                record.requirement_key == requirement_key
                and record.requirement_id == requirement_id
                and record.status is DiscoveryStatus.CANDIDATES
            ):
                found = self.sources.get(requirement_id, {})
                return tuple(found[spec_id] for spec_id in record.spec_ids)
        return ()

    def pinnable_for(self, requirement_key: str, requirement_id: str) -> tuple[SelectedSource, ...]:
        """The sources the latest matching answer a pin may be taken from returned for `requirement_id`.

        Same shape as `candidates_for`, over every `_PIN_HONOURING` status
        and skipping a `blocked` record, as `_select_definite` does: a
        `definite` answer's one source as well as a `candidates` answer's
        several. A `pinned_source_not_found` referral offers these.
        """
        for record in reversed(self.records):
            if (
                record.requirement_key == requirement_key
                and record.requirement_id == requirement_id
                and record.status in _PIN_HONOURING
                and not record.blocked
            ):
                found = self.sources.get(requirement_id, {})
                return tuple(found[spec_id] for spec_id in record.spec_ids)
        return ()

    def set_aside_pin(self, requirement_key: str, requirement_id: str) -> SelectedSource | None:
        """The pin's source `_source_from_resolve_pin` read and set aside for `requirement_id`, or `None`.

        Read off the latest `pin` record at `requirement_key`, which sets a
        source aside when it states no `status` (`DiscoveryRecord`).
        """
        for record in reversed(self.records):
            if (
                record.tool == "pin"
                and record.requirement_key == requirement_key
                and record.requirement_id == requirement_id
            ):
                if record.status is not None or not record.spec_ids:
                    return None
                return self.sources.get(requirement_id, {}).get(record.spec_ids[0])
        return None

    def scope_kept(
        self, plan: Plan, requirement: DatasetRequirement, sources: Sequence[SelectedSource]
    ) -> ScopeKept:
        """`sources`, found in place of `requirement`'s pin in `plan`, split into those a referral offers and those it withholds.

        The pin's source was set aside (`set_aside_pin`) because it holds
        closed a side `across` asks for every member of. When every
        requirement of `plan` reading `requirement` sums that side away
        (`_sums_its_sides_away`), the pin's scope is the figure the plan asks
        for: a source that drops it (`translator.referral.dropped_scope`) is
        withheld, and the change that keeps that scope is stated when the
        requirement can name it (`translator.referral.scope_repair`). A group
        the user named as a share's whole, as G7 in "Germany's share of G7
        ODA", is that case.

        A plan whose every reader ranks over that side, where the pin holds
        it to the members of one group (`_ranks_its_group_members`), keeps
        the scope too: "the largest DAC donors" ranks the DAC members, and a
        source holding every provider open would rank providers outside the
        group. Its repair names the group for its members.

        The pin gives way, and every source is offered, the same sources a
        search with the pin dropped finds, in two cases. Any other reader,
        such as a filter or a rank over a side held to one member, reads the
        side per member, and naming the pin's scope would leave one member
        where the reader needs every one `across` asks for. A sum that a share_of or quotient_of
        divides by, where the pin binds that side to the numerator's own
        members alone (`_divides_its_member_by_itself`), would divide the
        member by itself: "the United States' share of ODA to Kenya" with its
        total pinned to the United States' figure is that case. With no
        source set aside, every source is offered too.
        """
        held = self.set_aside_pin(requirement_revision(requirement), requirement.id)
        if held is None or _divides_its_member_by_itself(plan, requirement, held):
            return ScopeKept(offered=tuple(sources))
        summed = _sums_its_sides_away(plan, requirement.id)
        if not summed and not _ranks_its_group_members(plan, requirement, held):
            return ScopeKept(offered=tuple(sources))
        offered: list[SelectedSource] = []
        withheld: list[WithheldSource] = []
        for source in sources:
            if dropped := dropped_scope(held, source):
                withheld.append(WithheldSource(source=source, dropped=dropped))
            else:
                offered.append(source)
        repairs = [
            repair
            for word in requirement.across
            if (repair := scope_repair(held, word, members=not summed))
        ]
        repair = "; ".join(repairs) if withheld and repairs else None
        return ScopeKept(offered=tuple(offered), withheld=tuple(withheld), repair=repair)

    def recorded_source(self, spec_id: str) -> SelectedSource | None:
        """The source a record of this log last named `spec_id` with, or `None` when none did."""
        for record in reversed(self.records):
            if record.requirement_id is not None and spec_id in record.spec_ids:
                found = self.sources.get(record.requirement_id, {}).get(spec_id)
                if found is not None:
                    return found
        return None

    def recorded_definition(self, spec_id: str) -> _RecordedSourceDefinition | None:
        """The verified definition of `spec_id` a source this log holds states, or `None`.

        The source is `spec_id`'s own, such as a candidate a
        `pinned_source_not_found` referral offered, or one offering it as a
        sibling measure (`elected_measure.alternatives`). Its identity must
        hash to `spec_id` (`caller_definition`).
        """
        for own in reversed(tuple(self.sources.values())):
            for source in reversed(tuple(own.values())):
                identities = (
                    {a.spec_id: a.identity for a in source.elected_measure.alternatives}
                    if source.elected_measure is not None
                    else {}
                )
                if source.identity is not None:
                    identities[source.spec_id] = source.identity
                identity = identities.get(spec_id)
                if identity is None:
                    continue
                try:
                    caller_definition(
                        {"schema_version": 1, "spec_id": spec_id, "identity": identity}
                    )
                except (IdentityMismatch, IdentityUnusable):
                    continue
                return _RecordedSourceDefinition(identity, source.groups, source.resolved_sources)
        return None

    def stale_pin_sources(
        self, requirement_key: str, requirement_id: str, pinned_spec_id: str
    ) -> tuple[SelectedSource, ...] | None:
        """The sources found in place of `pinned_spec_id`, or `None` when the pin is not stale.

        The pin is stale when the latest `search` answer for `requirement_id`
        at `requirement_key` is an unblocked one a pin may be taken from
        that does not name it: the answer `discover_source` refers back as
        `pinned_source_not_found`.
        """
        for record in reversed(self.records):
            if (
                record.tool != "search"
                or record.requirement_key != requirement_key
                or record.requirement_id != requirement_id
                or record.status is None
            ):
                continue
            if (
                record.status not in _PIN_HONOURING
                or record.blocked
                or pinned_spec_id in record.spec_ids
            ):
                return None
            found = self.sources.get(requirement_id, {})
            return tuple(found[spec_id] for spec_id in record.spec_ids)
        return None

    def nearest_for(self, requirement_key: str, requirement_id: str) -> tuple[SelectedSource, ...]:
        """The nearest real sources the latest matching `no_data` answer for `requirement_id` named.

        Same shape as `candidates_for`, filtering on `DiscoveryStatus.NO_DATA`
        instead: a `no_data` answer's own sources are the nearest real series
        `search` offered instead of the ask, recorded the same way a
        `candidates` answer's sources are.
        """
        for record in reversed(self.records):
            if (
                record.requirement_key == requirement_key
                and record.requirement_id == requirement_id
                and record.status is DiscoveryStatus.NO_DATA
            ):
                found = self.sources.get(requirement_id, {})
                return tuple(found[spec_id] for spec_id in record.spec_ids)
        return ()

    def no_data_reason(self, requirement_key: str, requirement_id: str) -> str | None:
        """The reason code the latest matching `no_data` answer for `requirement_id` carried."""
        for record in reversed(self.records):
            if (
                record.requirement_key == requirement_key
                and record.requirement_id == requirement_id
                and record.status is DiscoveryStatus.NO_DATA
            ):
                return record.reason
        return None

    def with_record(
        self, record: DiscoveryRecord, sources: Sequence[SelectedSource] = ()
    ) -> DiscoveryLog:
        """Return a new log carrying `record` and `sources`; `self` is unchanged.

        Assigns `record.sequence` from this log's own high-water mark,
        whatever the caller built `record` with, so append order survives
        the persisted log's own union of several calls' snapshots (see
        `save_log`).

        Raises:
            ValueError: If `sources` is given for a record naming no requirement;
                a source is recorded against the requirement that found it.
        """
        next_sequence = max((r.sequence for r in self.records), default=-1) + 1
        record = record.model_copy(update={"sequence": next_sequence})
        if not sources:
            return DiscoveryLog(records=(*self.records, record), sources=self.sources)
        if record.requirement_id is None:
            raise ValueError("sources are recorded against the requirement that found them")
        own = self.sources.get(record.requirement_id, {})
        return DiscoveryLog(
            records=(*self.records, record),
            sources={
                **self.sources,
                record.requirement_id: {**own, **{source.spec_id: source for source in sources}},
            },
        )


class DatasetKeys(NamedTuple):
    """What `DiscoveryLog.selectable` matches a plan's dataset requirements by,
    each keyed by requirement id: its `requirement_revision`, its pin, and
    whether it states `across`."""

    requirement_keys: dict[str, str]
    pinned: dict[str, str]
    across: frozenset[str]


def dataset_keys(plan: Plan) -> DatasetKeys:
    """`plan`'s dataset requirements as `DiscoveryLog.selectable` matches them."""
    datasets = [r for r in plan.evidence_requirements if isinstance(r, DatasetRequirement)]
    return DatasetKeys(
        requirement_keys={r.id: requirement_revision(r) for r in datasets},
        pinned={r.id: r.source_spec_id for r in datasets if r.source_spec_id is not None},
        across=frozenset(r.id for r in datasets if r.across),
    )


def load_log(state: SessionState) -> DiscoveryLog:
    """Return the session's discovery log, empty when none has been written.

    Rebuilds `records` sorted by `DiscoveryRecord.sequence`: the persisted
    dict (see `save_log`) is a union of several calls' own snapshots once ADK
    has merged their state deltas, and a dict has no order of its own, so
    `sequence` is what recovers "call order, latest last" for `_select_definite`
    and every other scan that relies on it.

    A `str` value under `DISCOVERY_STATE_KEY` is `DiscoveryLog.model_dump_json()`,
    the form an earlier release wrote here; every record inside it defaults
    `sequence` to 0, so it is re-stamped by position, the same append order
    the string form's own `records` tuple already carries.

    A source or a record stored in a shape its model rejects, one a release
    with other fields wrote, invalidates its requirement: every source and
    every record stored for that requirement id is left out, and the
    omission is logged. A record names its sources by spec_id and the scans
    read the latest record that matches, so leaving out the one entry would
    leave a record naming a source the log does not hold, or an earlier
    answer standing in for the latest. The requirement then holds nothing
    and is discovered again. A rejected record that names no requirement is
    left out alone.
    """
    raw = state.get(DISCOVERY_STATE_KEY)
    if not raw:
        return DiscoveryLog()
    if isinstance(raw, str):
        stored = json.loads(raw)
        stored_records = [
            {**value, "sequence": i} for i, value in enumerate(stored.get("records", ()))
        ]
        stored_sources = stored.get("sources", {})
    else:
        stored_records = list(raw["records"].values())
        stored_sources = raw["sources"]

    errors: list[ValidationError] = []
    invalidated: set[str] = set()
    sources: dict[str, dict[str, SelectedSource]] = {}
    for requirement_id, by_spec in stored_sources.items():
        try:
            sources[requirement_id] = {
                spec_id: SelectedSource.model_validate(value) for spec_id, value in by_spec.items()
            }
        except ValidationError as exc:
            errors.append(exc)
            invalidated.add(requirement_id)
    records: list[DiscoveryRecord] = []
    for value in stored_records:
        try:
            records.append(DiscoveryRecord.model_validate(value))
        except ValidationError as exc:
            errors.append(exc)
            requirement_id = value.get("requirement_id")
            if isinstance(requirement_id, str):
                invalidated.add(requirement_id)
    kept = [record for record in records if record.requirement_id not in invalidated]
    if errors:
        dropped_sources = sum(len(stored_sources[key]) for key in invalidated & set(stored_sources))
        dropped_records = len(stored_records) - len(kept)
        warn_left_out(
            logger,
            DISCOVERY_STATE_KEY,
            errors,
            entries=dropped_records + dropped_sources,
            detail="%d records, %d sources, for %d requirements",
            counts=(dropped_records, dropped_sources, len(invalidated)),
        )
    return DiscoveryLog(
        records=tuple(sorted(kept, key=lambda record: record.sequence)),
        sources={key: found for key, found in sources.items() if key not in invalidated},
    )


def save_log(state: SessionState, log: DiscoveryLog) -> None:
    """Record `log` as the session's discovery log.

    Persisted as a dict keyed by each record's own `id`, with `sources`
    keyed the same way `DiscoveryLog.sources` already is -- never as one JSON
    string. ADK merges two concurrent tool calls' state deltas with
    `deep_merge_dicts` (`google.adk.flows.llm_flows.tools._batch_executor`),
    which recurses into a dict shared by both deltas but replaces a string
    leaf outright with whichever delta is listed last. A record keyed by its
    own `id` under a shared `"records"` dict is exactly such a shared dict,
    so the merge unions the two calls' records instead of one replacing the
    other's.
    """
    state[DISCOVERY_STATE_KEY] = {
        "records": {record.id: record.model_dump(mode="json") for record in log.records},
        "sources": {
            requirement_id: {
                spec_id: source.model_dump(mode="json") for spec_id, source in by_spec.items()
            }
            for requirement_id, by_spec in log.sources.items()
        },
    }


def _session_token(state: SessionState) -> str:
    """This session's own token, minting one when absent.

    Synchronous on purpose: it reads the session's token and writes one back
    when there is none, and only the absence of an `await` between those two
    steps stops two concurrent calls minting two different tokens. Do not
    make this a coroutine. `discovery_lock` and `_reservations` both mint
    their own per-session object keyed by this same token, so either one
    minting it first fixes it for the other.
    """
    token = state.get(DISCOVERY_LOCK_TOKEN_KEY)
    if token is None:
        token = uuid.uuid4().hex
        state[DISCOVERY_LOCK_TOKEN_KEY] = token
    return token


def discovery_lock(state: SessionState) -> asyncio.Lock:
    """The lock serialising this session's budget reservations and log read-modify-writes."""
    token = _session_token(state)
    lock = _LOCKS.get(token)
    if lock is None:
        lock = asyncio.Lock()
        _LOCKS[token] = lock
    return lock


class _Reservations:
    """One session's own in-flight `search` counters, per revision and per requirement.

    Not persisted: an in-flight call has reserved a slot of
    `DISCOVERY_CALL_BUDGET` nobody's record yet reflects, so `_budget_error`
    and `_requirement_budget_error` count these counters alongside
    `DiscoveryLog.calls` and `DiscoveryLog.requirement_calls`. `reserve` and
    `release` are called with `discovery_lock` held, the same discipline the
    log's own read-modify-write follows, so two concurrent calls' checks
    never both pass for the budget's last slot.
    """

    def __init__(self) -> None:
        self._by_revision: Counter[str] = Counter()
        self._by_requirement: Counter[tuple[str, str]] = Counter()

    def reserve(self, revision: str, requirement_id: str) -> None:
        self._by_revision[revision] += 1
        self._by_requirement[(revision, requirement_id)] += 1

    def release(self, revision: str, requirement_id: str) -> None:
        self._by_revision[revision] -= 1
        self._by_requirement[(revision, requirement_id)] -= 1
        if self._by_revision[revision] <= 0:
            del self._by_revision[revision]
        if self._by_requirement[(revision, requirement_id)] <= 0:
            del self._by_requirement[(revision, requirement_id)]

    def in_flight(self, revision: str) -> int:
        return self._by_revision[revision]

    def requirement_in_flight(self, revision: str, requirement_id: str) -> int:
        return self._by_requirement[(revision, requirement_id)]


# One `_Reservations` per session token, the same weak, per-process discipline
# `_LOCKS` follows: a session's reservations are collected once no call holds
# a reference to them, and a waiter keeps its own reference alive regardless.
_RESERVATIONS: WeakValueDictionary[str, _Reservations] = WeakValueDictionary()


def _reservations(state: SessionState) -> _Reservations:
    """This session's in-flight search counters; see `discovery_lock` and `_Reservations`."""
    token = _session_token(state)
    reservations = _RESERVATIONS.get(token)
    if reservations is None:
        reservations = _Reservations()
        _RESERVATIONS[token] = reservations
    return reservations


def compact_source(source: SelectedSource, *, selectable: bool = True) -> dict[str, object]:
    """What a model needs to know about a source: enough to judge it, and to name
    it only when it may.

    Refs and identity documents stay out: the model never writes them, and the
    compiler reads them from the log. A source that is not selectable -- a
    candidate among several, a nearest match to an ask with no data -- is
    described without its `spec_id`, so the only way it reaches a referral is
    through the code that attaches candidates from the log. `searched_as` is
    always shown, so a reader sees the phrase that found the source without
    comparing it against the requirement. `measure` is what the source
    measures, by `search`'s label; `measure_count` says how many StatVars
    it holds (`compiler.sources.stat_var_count`), "at least" that many when
    the count is a floor; and `varies_by`, present only when the source has
    open axes, names the slots its StatVars vary along. The description
    names no StatVar.
    `open_entity_axes`, present only when
    the source leaves an entity axis open, names each one and how many members
    it holds, the population a requirement's `across` covers. `definition` carries
    whatever `fetch_definitions` has already fetched for this source --
    `None` until it has, never fetched here.
    """
    open_entity_axes = [
        {"column": axis.column, "members": axis.member_count}
        for axis in source.open_entity_axes or ()
    ]
    return {
        **({"spec_id": source.spec_id} if selectable else {}),
        "sentence": source.sentence,
        "searched_as": source.searched_as,
        "confidence": source.confidence,
        "has_data": source.has_data,
        "measure": source.measure_label,
        "measure_count": _measure_count(source),
        **({"varies_by": [axis.property for axis in source.open_axes]} if source.open_axes else {}),
        "entities": [{"label": e.label, "role": e.role} for e in source.entities],
        **({"open_entity_axes": open_entity_axes} if open_entity_axes else {}),
        "unit": source.unit_label or source.unit_ref,
        "coverage_window": (
            source.coverage_window.model_dump() if source.coverage_window else None
        ),
        "definition": source.definition,
    }


def _measure_count(source: SelectedSource) -> str | None:
    """How many StatVars `source` holds, in words: "at least" that many when `search` states the count as a floor.

    A served count is exact. Without one, the confirmed members are a floor
    when `search` says so (`stat_var_member_count_exact`).
    """
    count = stat_var_count(source)
    if count == 0:
        return None
    return stat_var_count_text(count, floor=stat_var_count_is_floor(source))


def stat_var_count_text(count: int, *, floor: bool) -> str:
    """`count` StatVars in words, "at least" that many when the count is a `floor`."""
    if floor:
        return f"at least {count:,} StatVars"
    return "1 StatVar" if count == 1 else f"{count:,} StatVars"


# The members of a source's description that name one retrieval of it: the
# spec_id it is retrieved under and the phrase its requirement was searched as.
_RETRIEVAL_KEYS = ("spec_id", "searched_as")


def _retrieval(compact: Mapping[str, object], first: Mapping[str, object]) -> dict[str, object]:
    """One retrieval of the dataset `first` describes: its `_RETRIEVAL_KEYS`, the
    coverage Search states for its period, and `has_data` when it differs from `first`'s."""
    return {
        **{key: compact[key] for key in _RETRIEVAL_KEYS},
        "coverage_window": compact["coverage_window"],
        **({"has_data": compact["has_data"]} if compact["has_data"] != first["has_data"] else {}),
    }


def described_sources(sources: Sequence[SelectedSource]) -> list[dict[str, object]]:
    """`sources` as a model reads them, each dataset described once.

    A workflow holds one source per retrieval, so a dataset several
    requirements retrieve, under one spec_id or under a copy of it over each
    requirement's own period, appears once per requirement there. Here two
    sources are one dataset when their identities are equal apart from the
    years (`planner.plan.dataset_of`), whatever sentence and coverage Search
    states for each period. Two sources that carry no identity are one
    dataset when `compact_source` describes them alike apart from
    `_RETRIEVAL_KEYS` and `definition`, and bind the same `shape_member_digest`
    (both `None` when `search` states none). Each dataset is listed once, as
    its first retrieval, in the order given, with the first definition any of
    its retrievals fetched. Every further retrieval adds its own
    `_RETRIEVAL_KEYS` and `coverage_window`, and its `has_data` when that
    differs from the first's, under `also_retrieved_as`, so the result names
    every spec_id and phrase the workflow holds and the coverage of each
    period, and grows with the datasets, not with the requirements.
    """
    described: list[dict[str, object]] = []
    first_of: dict[bytes, dict[str, object]] = {}
    for source in sources:
        compact = compact_source(source)
        dataset = rfc8785.dumps(
            {"identity": dataset_of(source.identity)}
            if source.identity is not None
            else {
                "described": {
                    key: value
                    for key, value in compact.items()
                    if key not in (*_RETRIEVAL_KEYS, "definition")
                },
                "members": source.shape_member_digest,
            }
        )
        first = first_of.get(dataset)
        if first is None:
            first_of[dataset] = compact
            described.append(compact)
            continue
        if first["definition"] is None:
            first["definition"] = compact["definition"]
        retrieval = _retrieval(compact, first)
        others = cast(list[object], first.setdefault("also_retrieved_as", []))
        if retrieval != _retrieval(first, first) and retrieval not in others:
            others.append(retrieval)
    for entry in described:
        if entry.get("also_retrieved_as") == []:
            del entry["also_retrieved_as"]
    return described


def current_draft(tool_context: ToolContext) -> Plan | dict[str, object]:
    """Return the draft plan in session state, or the typed problem a tool answers with."""
    plan = load_plan(tool_context.state)
    if plan is None:
        return problem(
            ProblemCode.PLAN_MISSING,
            "No plan has been proposed yet.",
            outcome=Outcome.STATE_MISSING,
        )
    if plan.status is not PlanStatus.DRAFT:
        return problem(
            ProblemCode.PLAN_NOT_DRAFT,
            f"This plan is {plan.status.value}; only a draft plan is translated.",
            outcome=Outcome.REFUSED,
            detail={"status": plan.status.value},
        )
    return plan


def requirement_revision(requirement: DatasetRequirement) -> str:
    """Return the content hash identifying this dataset requirement's own searched content.

    Covers exactly what `search_part` sends to `search` -- `measure`,
    `entities`, `period`, `unit`, the open role `across` asks for
    (`requirement_open_roles`), the `source` the question names and the
    `source_code`, each stated only when there is one -- plus `source_spec_id`,
    the pin `selectable` honours. Never `decision_ids`, `coverage` or `id`, none of
    which affects what is searched or what a pin resolves to. Canonical JSON
    (RFC 8785) under SHA-256, the same discipline `plan_revision` uses for the
    whole plan: stable across an edit to a different requirement or a
    decision, and changed exactly when this requirement's own searched
    content or pin changes.
    """
    content = revision_content(
        requirement, include={"measure", "entities", "period", "unit", "source_spec_id"}
    )
    if open_roles := requirement_open_roles(requirement):
        content["open_roles"] = open_roles
    if requirement.source is not None:
        content["source"] = requirement.source
    if requirement.source_ref is not None:
        content["source_ref"] = requirement.source_ref
    if requirement.source_exclude:
        content["source_exclude"] = True
    if requirement.source_code is not None:
        content["source_code"] = requirement.source_code
    return hashlib.sha256(rfc8785.dumps(content)).hexdigest()


def _budget_error(
    log: DiscoveryLog, revision: str, requirement_id: str, reservations: _Reservations
) -> dict[str, object] | None:
    """The problem a call is refused with once `revision` has spent every call its budget allows.

    Counts `log.calls(revision)` -- committed search records -- plus
    `reservations.in_flight(revision)`, every concurrent call's own claim on
    the budget that has passed this check but not yet written its record, so
    two calls racing for the budget's last slot cannot both pass.
    """
    used = log.calls(revision) + reservations.in_flight(revision)
    if used < DISCOVERY_CALL_BUDGET:
        return None
    message = (
        f"This plan revision has used every one of its {DISCOVERY_CALL_BUDGET} discovery calls."
    )
    return problem(
        ProblemCode.BUDGET_EXHAUSTED,
        message,
        outcome=Outcome.REFUSED,
        action=_budget_referral(message, timed_out_requirement=log.last_call_timed_out(revision)),
        detail={"budget": DISCOVERY_CALL_BUDGET, "used": used},
    )


def _requirement_budget_error(
    log: DiscoveryLog, revision: str, requirement_id: str, reservations: _Reservations
) -> dict[str, object] | None:
    """The problem a call is refused with once `requirement_id` alone has spent half the revision's budget.

    Counts committed calls plus this requirement's own in-flight
    reservations, the same reasoning `_budget_error` applies to the
    revision's total.
    """
    used = log.requirement_calls(revision, requirement_id) + reservations.requirement_in_flight(
        revision, requirement_id
    )
    if used < DISCOVERY_CALL_BUDGET_PER_REQUIREMENT:
        return None
    message = (
        f"{requirement_id!r} has used {DISCOVERY_CALL_BUDGET_PER_REQUIREMENT} discovery calls in "
        f"this plan revision, half of the revision's own {DISCOVERY_CALL_BUDGET}-call budget; no "
        "single requirement may spend more."
    )
    return problem(
        ProblemCode.BUDGET_EXHAUSTED,
        message,
        outcome=Outcome.REFUSED,
        action=_budget_referral(
            message, timed_out_requirement=log.last_call_timed_out(revision, requirement_id)
        ),
        detail={
            "budget": DISCOVERY_CALL_BUDGET_PER_REQUIREMENT,
            "used": used,
            "requirement_id": requirement_id,
        },
    )


def _budget_referral(message: str, *, timed_out_requirement: str | None) -> Action:
    """The referral a spent budget makes, by how the most recent `search` call that spent it ended.

    When that call timed out (`timed_out_requirement` names its
    requirement), the referral is `discovery_unavailable` about that
    requirement, marked `budget_spent`. A retry of the same revision is
    refused before it searches, so the planner entry ends on it. When that call was answered, the referral is
    `discovery_budget_exhausted`, whatever earlier calls did, and the
    planner may correct the plan. The budget refuses the call either way,
    which keeps the number of `search` calls a revision makes bounded.
    """
    if timed_out_requirement is None:
        return Action(
            call="return_to_planner",
            arguments={"kind": ReferralKind.DISCOVERY_BUDGET_EXHAUSTED.value, "reason": message},
        )
    return Action(
        call="return_to_planner",
        arguments={
            "kind": ReferralKind.DISCOVERY_UNAVAILABLE.value,
            "reason": message,
            "requirement_id": timed_out_requirement,
            "budget_spent": True,
        },
    )


def _unit_phrase(unit: Unit) -> str:
    """The unit as `search` takes it: currency and price basis, with the base year when constant."""
    if unit.prices == "constant":
        return f"constant {unit.base_year} {unit.currency}"
    return f"current {unit.currency}"


def search_part(requirement: DatasetRequirement, variable: str) -> dict[str, object]:
    """One `search` part, built from `requirement`'s own fields plus `variable`.

    Sends the named entities, and the side of a flow `across` covers as an
    open role (`requirement_open_roles`). A `source`, the publisher or dataset
    the question names in the user's own words, goes as one included
    `sources` entry, with its chosen reference and exclusion flag when stated.
    A publisher's series code goes as the part's `source_code`.

    An entity's role goes as the `PARTY_ROLES` word it stands for
    (`party_role`), each a word of `search`'s role lexicon. A role standing
    for none, which only a plan read back from session state can hold, goes
    unstated.

    An entity with `members` set asks for each member of the set it names
    (`"members": true`). A set named without it goes as one entity, which
    search binds to the set's own published figure where the source has one.

    An entity with a `dcid` sends it beside its words, so search binds that
    graph node and keeps two entities written alike apart.
    """
    entities = []
    for entity in requirement.entities:
        sent: dict[str, object] = {"text": entity.name}
        if entity.dcid is not None:
            sent["dcid"] = entity.dcid
        if entity.role is not None and (role := party_role(entity.role)) is not None:
            sent["role"] = role
        if entity.members:
            sent["members"] = True
        entities.append(sent)
    part: dict[str, object] = {"variable": variable}
    if entities:
        part["entities"] = entities
    # "since YYYY" is the form Search binds as a range with no end year.
    part["time"] = (
        f"since {requirement.period.start}"
        if requirement.period_end_open
        else f"{requirement.period.start} to {requirement.period.end}"
    )
    if requirement.unit is not None:
        part["unit"] = _unit_phrase(requirement.unit)
    if open_roles := requirement_open_roles(requirement):
        part["open_roles"] = open_roles
    if requirement.source is not None:
        mention: dict[str, object] = {"text": requirement.source}
        if requirement.source_ref is not None:
            mention["ref"] = requirement.source_ref
        if requirement.source_exclude:
            mention["exclude"] = True
        part["sources"] = [mention]
    if requirement.source_code is not None:
        part["source_code"] = requirement.source_code
    return part


def requirement_open_roles(requirement: DatasetRequirement) -> list[str]:
    """The roles `search` is asked to hold open for `requirement`, one per `across` word naming a side of a flow.

    Each role goes as the `PARTY_ROLES` word its `across` word stands for, in
    `across` order, so "lenders" asks for every creditor. Search then elects a
    source that breaks the measure down by those roles over one that only
    totals them, and counts each role's members. A word naming the place, a
    population narrower than a party ("bilateral donor"), or two words of one
    side names no role `search` holds open, and nothing is sent for it. No
    entity binds a side sent, because `DatasetRequirement` refuses an entity
    in a party its `across` covers.
    """
    return [
        role
        for word in requirement.across
        if across_party(word) in ("giving", "receiving") and (role := party_role(word)) is not None
    ]


def _dropped_phrases(
    requirement: DatasetRequirement, part: Mapping[str, object], blocked_by: Sequence[str]
) -> dict[str, list[str]]:
    """The phrases of `part` that `search` dropped, keyed by the warning code that says so."""
    dropped: dict[str, list[str]] = {}
    if "role_text_not_recognised" in blocked_by:
        dropped["role_text_not_recognised"] = [
            entity.role for entity in requirement.entities if entity.role is not None
        ]
    if "time_text_not_parsed" in blocked_by:
        dropped["time_text_not_parsed"] = [str(part["time"])]
    return dropped


def _dataset_requirement(plan: Plan, requirement_id: str) -> DatasetRequirement | dict[str, object]:
    """Return the dataset requirement `requirement_id` names, or the problem that says why not."""
    try:
        requirement = plan.requirement(requirement_id)
    except KeyError:
        known = [r.id for r in plan.evidence_requirements]
        return problem(
            ProblemCode.REQUIREMENT_UNKNOWN,
            f"This plan records no requirement {requirement_id!r}.",
            outcome=Outcome.INVALID_INPUT,
            detail={"known": known},
        )
    if not isinstance(requirement, DatasetRequirement):
        return problem(
            ProblemCode.REQUIREMENT_NOT_A_DATASET,
            f"Requirement {requirement_id!r} is a {requirement.kind!r} requirement; only a "
            "dataset requirement is discovered.",
            outcome=Outcome.INVALID_INPUT,
            detail={"kind": requirement.kind},
        )
    return requirement


def _bare_dcid(ref: str) -> str:
    """`ref` with a `dcid:` prefix stripped, so a reference compares the same with or without it."""
    return ref.removeprefix("dcid:")


def _tried_source_ids(
    log: DiscoveryLog,
    requirement_key: str,
    requirement_id: str,
    *,
    tool: Literal["describe", "resolve"],
    counting_failed: bool,
) -> frozenset[str]:
    """The `spec_id`s `requirement_id` already tried a `tool` call for, at `requirement_key`.

    With `counting_failed`, a failed or timed-out attempt counts the same as a
    successful one: `fetch_definitions` tries each source once per
    `requirement_key`, whatever the outcome, so a failing `describe` is not
    re-hit on every `COMPILE_FAILED` retry `propose_operations` makes within
    the same revision. Without it, a `transport_failed` attempt is left out, so
    the next call tries again.
    """
    return frozenset(
        spec_id
        for record in log.records
        if record.tool == tool
        and record.requirement_id == requirement_id
        and record.requirement_key == requirement_key
        and (counting_failed or not record.transport_failed)
        for spec_id in record.spec_ids
    )


def _nodes_of(answer: ToolCallResult) -> Sequence[Mapping[str, object]]:
    """`answer`'s own `nodes`, filtered to mappings; empty on a refusal or a malformed answer."""
    if answer.is_error:
        return ()
    raw = (answer.structured_content or {}).get("nodes")
    if not isinstance(raw, Sequence):
        return ()
    return cast("list[Mapping[str, object]]", [node for node in raw if isinstance(node, Mapping)])


async def _describe_ids(
    surface: ToolSurface, ids: Sequence[str]
) -> tuple[Sequence[Mapping[str, object]], bool]:
    """Call `describe` on `ids`, dropping an unheld dcid it names and retrying without it.

    A chunk mixes dcids across several sources' and requirements' own
    stat vars (`fetch_definitions`'s own docstring), so one unheld dcid must
    not zero every other id's definition: `describe`'s own `node_not_held`
    refusal carries `is_error=True` for the *whole* call
    (`data_mcp.describe_tool.describe`) but names every offending dcid in
    `structured_content["fix"]["dcids"]`
    (`data_mcp.tool_errors.ToolRefusal.node_not_held`) -- read here and
    retried without them, rather than left unread.

    Bounded by `len(ids)` attempts: each retry drops at least one id, so the
    remaining ids shrink to nothing within that many rounds even in the
    worst case. A retry that names no offender not already dropped -- no
    progress -- or that refuses any other way, stops immediately rather than
    looping.

    Returns the nodes a resolving call answered (empty if none ever did) and
    whether the ids that never got an answer were refused rather than merely
    named unheld; a caller reads that only to choose `DiscoveryRecord.detail`.
    """
    remaining = list(ids)
    for _ in range(len(ids)):
        if not remaining:
            return (), True
        answer = await surface.call_tool(tool=DESCRIBE, arguments={"ids": remaining})
        if not answer.is_error:
            return _nodes_of(answer), False
        structured = answer.structured_content or {}
        if structured.get("code") != "node_not_held":
            return (), True
        fix = structured.get("fix")
        raw_dcids = fix.get("dcids") if isinstance(fix, Mapping) else None
        named = raw_dcids if isinstance(raw_dcids, Sequence) else ()
        offenders = {_bare_dcid(str(dcid)) for dcid in named}
        retry = [dcid for dcid in remaining if _bare_dcid(dcid) not in offenders]
        if len(retry) == len(remaining):
            return (), True
        remaining = retry
    return (), True


def _definition_of(nodes: Sequence[Mapping[str, object]], dcid: str) -> str | None:
    """`dcid`'s own planner line, read out of a `describe` answer's `nodes`; `None` if `dcid` is absent.

    The line is the variable's description, else one composed from its definition
    (`data_mcp.node_lookup.variable_text`). Reads by `ref`, `dcid:`-stripped, rather than by
    position: `describe` may answer fewer ids than asked, or in a different order.
    """
    target = _bare_dcid(dcid)
    for node in nodes:
        if _bare_dcid(str(node.get("ref", ""))) == target:
            return described_variable_text(node)
    return None


DescribedPair = tuple[DiscoveryRecord, tuple[SelectedSource, ...]]


async def fetch_definitions(
    surface: ToolSurface,
    log: DiscoveryLog,
    *,
    revision: str,
    requirement_keys: Mapping[str, str],
    sources: Mapping[str, SelectedSource],
) -> tuple[DescribedPair, ...]:
    """Fetch each of `sources`' own `describe` definition, one pair per source described.

    `sources` names, per requirement id, the one source to describe. The
    caller chooses which sources those are -- `propose_operations` passes
    `DiscoveryLog.definite_source_ids`'s narrowing of `selectable`'s own
    output, so only a genuinely-`definite` answer is enriched -- and this
    function enriches whichever ones it is handed, so that a caller wanting
    definitions for a `candidates` answer's own entries can pass those
    instead.

    `log` is read only, for `_tried_source_ids`'s skip check: a source
    already tried this revision's `requirement_key` is left out, success or
    failure alike. This function neither reads nor writes session state and
    takes no lock -- it returns the `(DiscoveryRecord, sources)` pair each
    described source produced, in the order fetched, and it is the caller's
    job to apply them with `apply_definitions`, under `discovery_lock`, the
    same discipline `discover_source`'s own closing phase follows. Held apart
    this way, the `describe` round trip below runs with the lock released.

    Only a source whose identity pins one StatVar
    (`compiler.sources.pinned_stat_var`) is described: a family's definition
    would be one member's and a model would read it as the family's, and a
    described identity names no dcid to describe. Every
    source still needing a fetch is batched into as few `describe` calls as
    `DESCRIBE_ID_BATCH_MAX` allows, one call per chunk of the sources' own
    stat var each, so a plan's single-digit definite-source count costs one
    round trip rather than one per source.

    A `describe` that times out, is refused, or never names a source's own
    stat var leaves that source's `definition` unset (`None` on first
    discovery): a missing or unusable description must not fail the compile
    it enriches, so it is returned as tried and left at that.
    """
    pending = [
        (requirement_id, requirement_keys[requirement_id], source, dcid)
        for requirement_id, source in sources.items()
        if requirement_id in requirement_keys
        and (dcid := pinned_stat_var(source.identity)) is not None
        and source.spec_id
        not in _tried_source_ids(
            log,
            requirement_keys[requirement_id],
            requirement_id,
            tool="describe",
            counting_failed=True,
        )
    ]
    pairs: list[DescribedPair] = []
    for start in range(0, len(pending), DESCRIBE_ID_BATCH_MAX):
        chunk = pending[start : start + DESCRIBE_ID_BATCH_MAX]
        ids = [dcid for _, _, _, dcid in chunk]
        try:
            nodes, refused = await _describe_ids(surface, ids)
        except ToolSurfaceUnavailableError:
            for requirement_id, key, source, _ in chunk:
                pairs.append(
                    (
                        DiscoveryRecord(
                            revision=revision,
                            tool="describe",
                            requirement_id=requirement_id,
                            requirement_key=key,
                            spec_ids=(source.spec_id,),
                            transport_failed=True,
                        ),
                        (),
                    )
                )
            continue
        for requirement_id, key, source, dcid in chunk:
            record = DiscoveryRecord(
                revision=revision,
                tool="describe",
                requirement_id=requirement_id,
                requirement_key=key,
                spec_ids=(source.spec_id,),
                detail="describe refused" if refused else None,
            )
            definition = _definition_of(nodes, dcid)
            if definition is None:
                pairs.append((record, ()))
            else:
                pairs.append((record, (source.model_copy(update={"definition": definition}),)))
    return tuple(pairs)


async def apply_definitions(state: SessionState, pairs: Sequence[DescribedPair]) -> DiscoveryLog:
    """Reload the log fresh, apply every `fetch_definitions` pair, and save it.

    The shared closing phase both `fetch_definitions` callers
    (`propose_operations` and `_enrich_candidates`) run under
    `discovery_lock`: reloading here rather than reusing whatever the caller
    read before `fetch_definitions`'s own network calls, since a sibling may
    have written its own record in between, and only a fresh read plus an
    in-lock write keeps both.
    """
    async with discovery_lock(state):
        log = load_log(state)
        for record, sources in pairs:
            log = log.with_record(record, sources)
        save_log(state, log)
        return log


def _unmatched_names(requirement: DatasetRequirement, source: SelectedSource) -> tuple[str, ...]:
    """The names `requirement` gives entities that name nothing `source` holds (`named_members`).

    An entity with a `dcid` already names its node, and resolving its name
    again would key `entity_refs` once for two entities written alike.
    """
    return tuple(
        entity.name
        for entity in requirement.entities
        if entity.dcid is None and not named_members(entity, source)
    )


# The graph's resolve door as `resolve_entity_refs` calls it.
type ResolveEntities = Callable[
    [Mapping[str, Sequence[str]]], Awaitable[dict[str, dict[str, resolution.ResolutionOutcome]]]
]


async def graph_resolve(
    scope: Mapping[str, Sequence[str]],
) -> dict[str, dict[str, resolution.ResolutionOutcome]]:
    """`data_mcp.resolution.resolve_entities`, the door the served translation resolves through.

    Looked up when called, so a patch of `data_mcp.resolution` reaches it.
    """
    return await resolution.resolve_entities(scope)


async def resolve_entity_refs(
    log: DiscoveryLog,
    *,
    revision: str,
    requirements: Sequence[DatasetRequirement],
    requirement_keys: Mapping[str, str],
    sources: Mapping[str, SelectedSource],
    resolve_entities: ResolveEntities = graph_resolve,
) -> tuple[DescribedPair, ...]:
    """Resolve the entity names each source matches by no label to graph refs, one pair per source.

    `sources` names, per requirement id, the source the requirement settled
    on. A name that names nothing the source holds (`named_members`), such
    as "United States" beside an entity labelled "United States of
    America", is resolved through the graph's resolve door in process
    (`resolve_entities`, `graph_resolve` when served), every requirement's
    names in one call. Only a `resolved` outcome counts: a region, a group, an
    ambiguous name or one the graph lacks maps to nothing, and the
    evidence check reports it as missing. The refs found are
    recorded on the source (`SelectedSource.entity_refs`), so compile and
    the run-time evidence check both compare refs.

    Each source is resolved once per `requirement_key` (`_tried_source_ids`).
    A resolve that fails, in transport or with a response of an unknown
    shape, is logged with the names and requirements, records the attempt as
    `transport_failed` with the error as its `detail`, and adds no ref; the
    next translate asks again. Like `fetch_definitions`, this reads `log`
    only and returns pairs for `apply_definitions` to apply under
    `discovery_lock`.
    """
    pending = [
        (requirement, requirement_keys[requirement.id], source, names)
        for requirement in requirements
        if requirement.id in requirement_keys
        and (source := sources.get(requirement.id)) is not None
        and (names := _unmatched_names(requirement, source))
        and source.spec_id
        not in _tried_source_ids(
            log,
            requirement_keys[requirement.id],
            requirement.id,
            tool="resolve",
            counting_failed=False,
        )
    ]
    if not pending:
        return ()
    asked = list(dict.fromkeys(name for *_, names in pending for name in names))
    requirement_ids = [requirement.id for requirement, *_ in pending]
    failure: str | None = None
    outcomes: dict[str, resolution.ResolutionOutcome] = {}
    try:
        outcomes = (await resolve_entities({"names": asked})).get("names", {})
    except UpstreamError as exc:
        failure = str(exc)
        logger.warning(
            "entity names %s of requirements %s unresolved: %s", asked, requirement_ids, exc
        )
    except (KeyError, TypeError, ValueError) as exc:
        # A resolve answer of a shape `resolve_entities` does not read: a missing
        # `results`, a body that is not JSON, a status it does not know, or one result
        # too few.
        failure = f"{type(exc).__name__}: {exc}"
        logger.error(
            "entity names %s of requirements %s unresolved: the resolve answer was unreadable (%s)",
            asked,
            requirement_ids,
            failure,
        )
    refs = {
        name: outcome.dcid
        for name, outcome in outcomes.items()
        if isinstance(outcome, resolution.ResolvedEntity)
    }
    pairs: list[DescribedPair] = []
    for requirement, key, source, names in pending:
        record = DiscoveryRecord(
            revision=revision,
            tool="resolve",
            requirement_id=requirement.id,
            requirement_key=key,
            spec_ids=(source.spec_id,),
            detail=failure,
            transport_failed=failure is not None,
        )
        found = {name.casefold(): refs[name] for name in names if name in refs}
        if not found:
            pairs.append((record, ()))
            continue
        updated = source.model_copy(update={"entity_refs": {**source.entity_refs, **found}})
        pairs.append((record, (updated,)))
    return tuple(pairs)


# The graph's node door as `fetch_code_paths` calls it: dcids to node envelopes.
type DescribeNodes = Callable[[Sequence[str]], Awaitable[Mapping[str, node_lookup.NodeOutcome]]]


async def graph_describe(dcids: Sequence[str]) -> Mapping[str, node_lookup.NodeOutcome]:
    """`data_mcp.node_lookup.describe_nodes`, the door the served translation reads codes through.

    Looked up when called, so a patch of `data_mcp.node_lookup` reaches it.
    """
    return await node_lookup.describe_nodes(dcids)


async def _paths_to_root(
    codes: Collection[str], describe_nodes: DescribeNodes
) -> tuple[dict[str, tuple[str, ...]], frozenset[str]]:
    """Each of `codes`' path from its hierarchy's root down to itself, by `specializationOf`
    parent, and those of `codes` with no children.

    Reads one batch of nodes per level climbed, each node once. A code
    whose climb meets a node that is no code, or that the graph does not
    hold, has no path. A code is a leaf when its node states no `children`
    link.
    """
    parents: dict[str, str | None] = {}
    leaves: set[str] = set()
    asked: set[str] = set()
    frontier = set(codes)
    while frontier:
        nodes = await describe_nodes(sorted(frontier))
        asked |= frontier
        for dcid in frontier:
            if isinstance(node := nodes.get(dcid), CodeNode):
                parent = node.code.parent
                parents[dcid] = None if parent is None else parent.ref
                if "children" not in node.links:
                    leaves.add(dcid)
        frontier = {parent for parent in parents.values() if parent is not None} - asked
    paths: dict[str, tuple[str, ...]] = {}
    for code in codes:
        path: list[str] = []
        at: str | None = code
        while at is not None and at in parents and at not in path:
            path.append(at)
            at = parents[at]
        if at is None:
            paths[code] = tuple(reversed(path))
    return paths, frozenset(leaves & set(codes))


async def fetch_code_paths(
    *,
    revision: str,
    requirements: Sequence[DatasetRequirement],
    requirement_keys: Mapping[str, str],
    sources: Mapping[str, SelectedSource],
    describe_nodes: DescribeNodes = graph_describe,
) -> tuple[DescribedPair, ...]:
    """Read the hierarchy above the codes each breakdown's source names, one pair per source.

    A requirement stating a breakdown reads the level its source's named
    codes set (`compiler.shapes.breakdown_level`), which needs each named
    code's path to its root. The codes are those the source's identity
    binds on the breakdown's code system (`compiler.sources.bound_codes`);
    every source's codes are read together, through the graph's node door
    (`describe_nodes`, `graph_describe` when served), and the paths, with
    the codes found to have no children, are recorded on the source
    (`SelectedSource.code_paths`, `leaf_codes`). A source that names no code
    there, or holds every path already, is left out, so a source a fresh
    search recorded again is read again.

    Inside a planner entry the whole climb may take
    `planner.plan.discovery_time_left` seconds, as every discovery read may:
    with no time left no node is asked for, and a climb that time cuts short
    is a failed read. A read that fails is logged, records the attempt as
    `transport_failed` with the error as its `detail`, and adds no path; `propose_operations`
    answers it `surface_unavailable` before compiling, and the next
    translate reads again. A code the graph holds no path for adds none and
    fails nothing, and compile refuses the breakdown. Like
    `resolve_entity_refs`, this returns pairs for `apply_definitions` to
    apply under `discovery_lock`.
    """
    pending = [
        (requirement, requirement_keys[requirement.id], source, codes)
        for requirement in requirements
        if requirement.breakdown is not None
        and requirement.id in requirement_keys
        and (source := sources.get(requirement.id)) is not None
        and (
            codes := tuple(
                code
                for code in bound_codes(
                    source, BREAKDOWN_CODE_SYSTEMS[requirement.breakdown.dimension]
                )
                if code not in source.code_paths
            )
        )
    ]
    if not pending:
        return ()
    asked = list(dict.fromkeys(code for *_, codes in pending for code in codes))
    failure: str | None = None
    paths: dict[str, tuple[str, ...]] = {}
    leaves: frozenset[str] = frozenset()
    time_left = discovery_time_left()
    try:
        if time_left is not None and time_left <= 0:
            failure = "the node door was not read: the planner entry has no time left for it"
        else:
            async with asyncio.timeout(time_left):
                paths, leaves = await _paths_to_root(asked, describe_nodes)
    except TimeoutError:
        failure = (
            f"the node door did not answer in the {time_left:.0f} s the planner entry had left"
        )
    except (UpstreamError, ValidationError, ValueError) as exc:
        failure = f"{type(exc).__name__}: {exc}"
    if failure is not None:
        logger.warning("the hierarchy above codes %s was unread: %s", asked, failure)
    pairs: list[DescribedPair] = []
    for requirement, key, source, codes in pending:
        record = DiscoveryRecord(
            revision=revision,
            tool="code_paths",
            requirement_id=requirement.id,
            requirement_key=key,
            spec_ids=(source.spec_id,),
            detail=failure,
            transport_failed=failure is not None,
        )
        found = {code: paths[code] for code in codes if code in paths}
        if not found:
            pairs.append((record, ()))
            continue
        updated = source.model_copy(
            update={
                "code_paths": {**source.code_paths, **found},
                "leaf_codes": tuple(sorted({*source.leaf_codes, *(leaves & found.keys())})),
            }
        )
        pairs.append((record, (updated,)))
    return tuple(pairs)


def _ready_to_search(
    tool_context: ToolContext, requirement_id: str, reservations: _Reservations
) -> tuple[Plan, DatasetRequirement, str, str] | dict[str, object]:
    """Return the draft plan, the requirement to search for, its revision, and its key; or the problem.

    Needs a draft plan, a dataset requirement the plan states by
    `requirement_id`, and budget left for the plan's revision and for this
    requirement's own share of it, counting `reservations` alongside the
    committed log so two concurrent calls cannot both pass for the same
    slot.
    """
    plan = current_draft(tool_context)
    if isinstance(plan, dict):
        return plan
    requirement = _dataset_requirement(plan, requirement_id)
    if isinstance(requirement, dict):
        return requirement
    revision = plan_revision(plan)
    key = requirement_revision(requirement)
    log = load_log(tool_context.state)
    blocked = _budget_error(log, revision, requirement_id, reservations)
    if blocked is not None:
        return blocked
    blocked = _requirement_budget_error(log, revision, requirement_id, reservations)
    if blocked is not None:
        return blocked
    return plan, requirement, revision, key


async def _record_search(
    tool_context: ToolContext,
    record: DiscoveryRecord,
    sources: Sequence[SelectedSource] = (),
    *,
    reservations: _Reservations,
    revision: str,
    requirement_id: str,
) -> DiscoveryLog:
    """Reload the log fresh, append `record` and `sources`, save, and release this call's reservation -- a search's closing phase.

    Reloads under the lock rather than reusing whatever `_ready_to_search`
    read before the network call: a sibling may have written its own record
    in between, and only a fresh read plus an in-lock write keeps both.
    Releasing the reservation this call made happens after `save_log`, inside
    the same `discovery_lock` acquisition: a sibling's own budget check
    between the release and the commit would count the slot as free while it
    is still unrecorded, so the two are one critical section rather than two.
    The `BaseException` path, where nothing is ever recorded, releases alone
    under the lock instead of calling this function.
    """
    async with discovery_lock(tool_context.state):
        log = load_log(tool_context.state)
        log = log.with_record(record, sources)
        save_log(tool_context.state, log)
        reservations.release(revision, requirement_id)
    return log


class DefinitionNotResolvedError(RuntimeError):
    """data-mcp's `resolve_definition` gave no part for an identity, for the reason the message states.

    `retryable` is whether the same call can succeed later: data-mcp or the
    Search API could not be reached, or the definition store could not take
    the write; the message is the refusal's. Otherwise Search refused the
    identity (`refused`) or answered it with no part, and the message says
    so in words a user reads, naming the refusal's code when it states one.
    `code` is that code, `None` when the refusal states none or nothing was
    refused. `not_issued` is whether the call was never made, the planner
    entry having no time left for it (`NoTimeLeftError`).
    """

    def __init__(
        self,
        message: str,
        *,
        retryable: bool,
        refused: bool = False,
        code: str | None = None,
        not_issued: bool = False,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.refused = refused
        self.code = code
        self.not_issued = not_issued


class CutCalls:
    """The calls a planner entry cut off for lack of time and left running, by tool and arguments.

    A `resolve_definition` the request's wall-clock bound cuts is shielded rather than
    cancelled, since Search often answers it moments later (a cold member
    walk) and the definition it stores is the same whichever call stores it.
    The next call with the same tool and arguments in the same entry takes
    the kept task (`call_within_entry`) and is served its answer once it
    has one. A call is kept from the moment it is issued until it finishes,
    so a caller cancelled mid-await leaves it here too. `close` cancels
    whatever is still kept, so no call outlives the entry.
    `harness.planner_seam.PlannerSeam.__call__` opens one per entry through
    `current_cut_calls`.
    """

    def __init__(self) -> None:
        self._tasks: dict[bytes, asyncio.Task[ToolCallResult]] = {}

    def keep(self, key: bytes, task: asyncio.Task[ToolCallResult]) -> None:
        """Hold `task` for the next call `key` names, cancelling another task held there."""
        held = self._tasks.get(key)
        if held is not None and held is not task:
            held.cancel()
        self._tasks[key] = task

    def release(self, key: bytes, task: asyncio.Task[ToolCallResult]) -> None:
        """Stop holding `task` under `key`; a different task held there stays."""
        if self._tasks.get(key) is task:
            del self._tasks[key]

    def take(self, key: bytes) -> asyncio.Task[ToolCallResult] | None:
        """The task kept for `key`, no longer kept; `None` when none is."""
        return self._tasks.pop(key, None)

    async def close(self) -> None:
        """Cancel every kept task and wait for each to finish."""
        tasks = list(self._tasks.values())
        self._tasks.clear()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


# The `CutCalls` of the planner entry the current call runs in; unset outside one.
current_cut_calls: ContextVar[CutCalls | None] = ContextVar("cut_calls", default=None)


async def call_within_entry(
    surface: ToolSurface,
    *,
    tool: ToolId,
    arguments: Mapping[str, object],
    keep_when_cut: bool = False,
) -> ToolCallResult:
    """Call `tool` on `surface`, cut off when the planner entry it serves needs the time.

    Inside a planner entry the call may take `planner.plan.discovery_time_left`
    seconds, and the surface's own timeout, the capability row's
    `call_timeout_s`, caps it as everywhere else. With no time left the call is
    not issued. Outside an entry it takes the row's timeout alone.

    With `keep_when_cut` and an entry holding `current_cut_calls`, a cut call
    keeps running there, and a later call with the same `tool` and
    `arguments` waits on it within the time then left. A kept call that has
    answered by then serves that answer even with no time left.

    Raises:
        ToolSurfaceUnavailableError: If the call did not complete, including
            one cut off for, or never issued for lack of, the entry's time.
    """
    time_left = discovery_time_left()
    if time_left is None:
        return await surface.call_tool(tool=tool, arguments=arguments)
    cut_calls = current_cut_calls.get() if keep_when_cut else None
    if cut_calls is None:
        if time_left <= 0:
            raise _no_time_left(tool)
        try:
            async with asyncio.timeout(time_left):
                return await surface.call_tool(tool=tool, arguments=arguments)
        except TimeoutError as exc:
            raise _cut_off(tool, time_left) from exc

    key = rfc8785.dumps({"tool": tool.name, "arguments": dict(arguments)})
    task = cut_calls.take(key)
    if task is not None and task.done():
        return task.result()
    if time_left <= 0:
        if task is not None:
            cut_calls.keep(key, task)
        raise _no_time_left(tool)
    if task is None:
        task = asyncio.ensure_future(surface.call_tool(tool=tool, arguments=arguments))
    # Held while awaited, so a cut or a cancelled caller leaves it for the
    # entry's `CutCalls.close`; a finished call is served or raised here.
    cut_calls.keep(key, task)
    try:
        async with asyncio.timeout(time_left):
            return await asyncio.shield(task)
    except TimeoutError as exc:
        raise _cut_off(tool, time_left) from exc
    finally:
        if task.done():
            cut_calls.release(key, task)


class NoTimeLeftError(ToolSurfaceUnavailableError):
    """A call `call_within_entry` never issued, the planner entry having no time left for it."""


def _no_time_left(tool: ToolId) -> NoTimeLeftError:
    return NoTimeLeftError(
        f"{tool.name!r} was not called: the planner entry has no time left for it"
    )


def _cut_off(tool: ToolId, time_left: float) -> ToolSurfaceUnavailableError:
    return ToolSurfaceUnavailableError(
        f"{tool.name!r} did not answer in the {time_left:.0f} s the planner entry had left"
    )


class ResolvedAnswers:
    """The `resolve_definition` answers one translation has had, by arguments.

    `translator.tool.build_translate`'s translation opens one
    (`resolving_once`), so the group rewrite (`name_pinned_groups`) and
    discovery after it read a pin's identity off one call. An answer is kept
    whatever it says, a refusal included; a call that did not complete is
    not kept, so the next asks again. Concurrent callers with the same
    arguments wait on the first.
    """

    def __init__(self) -> None:
        self._answers: dict[bytes, ToolCallResult] = {}
        self._locks: dict[bytes, asyncio.Lock] = {}

    async def answer(
        self, key: bytes, call: Callable[[], Awaitable[ToolCallResult]]
    ) -> ToolCallResult:
        """The answer kept under `key`, or `call`'s, kept once it completes."""
        async with self._locks.setdefault(key, asyncio.Lock()):
            if key not in self._answers:
                self._answers[key] = await call()
            return self._answers[key]


# The `ResolvedAnswers` of the translation the current call runs in; unset outside one.
current_resolved_answers: ContextVar[ResolvedAnswers | None] = ContextVar(
    "resolved_answers", default=None
)


@contextmanager
def resolving_once() -> Iterator[ResolvedAnswers]:
    """Serve each identity `resolve_definition` is asked for once, inside the block."""
    answers = ResolvedAnswers()
    token = current_resolved_answers.set(answers)
    try:
        yield answers
    finally:
        current_resolved_answers.reset(token)


def refusal_retryable(answer: ToolCallResult) -> bool:
    """Whether `answer`, a data-mcp refusal, can succeed if the same call is made later.

    data-mcp marks such a refusal `retry_after_delay`: Search or the
    definition store could not be reached or could not take the call. Any
    other refusal is Search's answer to the call as made.
    """
    return (answer.structured_content or {}).get("retry_mode") == "retry_after_delay"


async def resolve_definition(
    surface: ToolSurface,
    *,
    spec_id: str,
    identity: Mapping[str, Any],
    groups: Sequence[ResolvedGroup] = (),
    scope_entries: Sequence[Mapping[str, Any]] = (),
) -> Mapping[str, Any]:
    """The part the Search API resolves `identity` to now, through data-mcp's `resolve_definition`.

    Inside `resolving_once` an identity already answered is served that
    answer (`ResolvedAnswers`).

    `groups`, the sets a role's places were asked for as, are posted beside
    the identity (`compiler.sources.wire_groups`), so Search names them again.
    Search keeps a posted set when the role binds the one party the dataset
    publishes as its total, and answers with that `total`; otherwise only
    when it names a graph node whose members are the identity's places on
    that role (apis `resubmit_request.resubmit_named_groups`). A
    members-bound set with no node, such as the G7 read as its members, is
    dropped: its places still bind, and its name is lost.

    `scope_entries`, the explicit and carried entries of a part's stored
    scope choices (`data_mcp.scope.ScopeEntry` as JSON), are posted beside
    a part's base identity, and Search answers with the identity they
    compile to, under that identity's own spec_id.

    data-mcp stores the definition the part carries, on the service `surface`
    reaches, which is the one whose `get_data` retrieves it. The part is
    Search's own, and `resolved_spec` judges whether it is the dataset
    `spec_id` names.

    Raises:
        DefinitionNotResolvedError: If the call did not complete, data-mcp
            refused it, or the envelope holds no part.
    """
    arguments: dict[str, object] = {"spec_id": spec_id, "identity": dict(identity)}
    if groups:
        arguments["entity_groups"] = wire_groups(groups)
    if scope_entries:
        arguments["scope_entries"] = [dict(entry) for entry in scope_entries]

    def call() -> Awaitable[ToolCallResult]:
        return call_within_entry(
            surface, tool=RESOLVE_DEFINITION, arguments=arguments, keep_when_cut=True
        )

    answers = current_resolved_answers.get()
    try:
        if answers is None:
            answer = await call()
        else:
            answer = await answers.answer(rfc8785.dumps(arguments), call)
    except ToolSurfaceUnavailableError as exc:
        raise DefinitionNotResolvedError(
            str(exc), retryable=True, not_issued=isinstance(exc, NoTimeLeftError)
        ) from exc
    if answer.is_error:
        refusal = answer.structured_content or {}
        if refusal_retryable(answer):
            raise DefinitionNotResolvedError(
                str(refusal.get("message") or answer.text), retryable=True
            )
        stated = refusal.get("upstream_code") or refusal.get("code")
        code = str(stated) if stated else None
        raise DefinitionNotResolvedError(
            f"Search no longer resolves it ({code})" if code else "Search no longer resolves it",
            retryable=False,
            refused=True,
            code=code,
        )
    envelope = (answer.meta or {}).get(SEARCH_ENVELOPE_META_KEY)
    parts = envelope.get("parts") if isinstance(envelope, Mapping) else None
    if not isinstance(parts, list) or not parts or not isinstance(parts[0], Mapping):
        raise DefinitionNotResolvedError("Search returns nothing for it", retryable=False)
    return cast(Mapping[str, Any], parts[0])


class PartFault(StrEnum):
    """How a part Search answered for an identity falls short of the dataset its spec_id names."""

    NOT_DEFINITE = "not_definite"
    ANOTHER_DATASET = "another_dataset"
    NO_DEFINITION = "no_definition"


class PartNotResolvedError(Exception):
    """A part Search answered for an identity is not the dataset its spec_id names.

    `fault` says how, `part` is Search's own part, and `answered` is the
    spec_id its spec names, `None` when it names none as a string.
    """

    def __init__(
        self, fault: PartFault, *, part: Mapping[str, Any], answered: str | None = None
    ) -> None:
        super().__init__(fault.value)
        self.fault = fault
        self.part = part
        self.answered = answered


def resolved_spec(part: Mapping[str, Any], *, spec_id: str) -> Mapping[str, Any]:
    """The Spec block of `part`, when Search resolves an identity to the one dataset `spec_id` names.

    `part` is what `resolve_definition` returns. It is that dataset when it
    is `definite`, its spec names `spec_id`, and the spec carries the
    identity it hashes from (`definition.identity`), which is the definition
    data-mcp stored as it answered, so a `get_data` of `spec_id` finds it.

    Raises:
        PartNotResolvedError: If the part is not `definite`
            (`PartFault.NOT_DEFINITE`), carries no spec block or a block
            without `definition.identity` (`PartFault.NO_DEFINITION`), or
            its spec names a spec_id other than `spec_id`, or none
            (`PartFault.ANOTHER_DATASET`).
    """
    if part.get("status") != "definite":
        raise PartNotResolvedError(PartFault.NOT_DEFINITE, part=part)
    spec = part.get("spec")
    if not isinstance(spec, Mapping):
        raise PartNotResolvedError(PartFault.NO_DEFINITION, part=part)
    answered = spec.get("spec_id")
    if answered != spec_id:
        raise PartNotResolvedError(
            PartFault.ANOTHER_DATASET,
            part=part,
            answered=answered if isinstance(answered, str) else None,
        )
    definition = spec.get("definition")
    if not isinstance(definition, Mapping) or not isinstance(definition.get("identity"), Mapping):
        raise PartNotResolvedError(PartFault.NO_DEFINITION, part=part)
    return spec


async def _recorded_source(
    tool_context: ToolContext,
    *,
    requirement: DatasetRequirement,
    first_year: int | None,
    surface: ToolSurface,
) -> SelectedSource | dict[str, object] | None:
    """The source of the dataset `requirement` pins, read from Search's answer for its identity on record.

    The identity is the copy over the requirement's period that
    `planner.agent.propose_plan` pinned (`planner.plan.windowed_pins`), or
    the one on record for the pin itself (`planner.plan.identity_on_record`:
    a search in the conversation, this session's discovery, or the block of
    the pin the planner was shown) when its window is that period, or
    a range with no end year for a requirement stating `period_end_open`,
    from `first_year` where the plan reads its dataset's whole history
    (`planner.plan.retrieves_period`). It
    is resolved through data-mcp (`resolve_definition`), which stores the
    definition for the plan's `get_data`, and the source is read from
    Search's part (`resolved_spec`), so its sentence, coverage and member
    counts are the ones Search states for that period.

    `None` when no such identity is on record, or when Search does not
    answer it with that one dataset: the caller then searches as for a pin
    with nothing on record. When data-mcp or Search could not be reached it
    is `SURFACE_UNAVAILABLE` with no `action`, which the translator refers
    back as `discovery_unavailable`, since the same plan can succeed later.
    """
    spec_id = requirement.source_spec_id or ""
    recorded = windowed_pins(tool_context.state).get(spec_id) or identity_on_record(
        spec_id,
        shown=pin_sources(tool_context.state),
        end_open=True,
        definition_lookup=load_log(tool_context.state).recorded_definition,
    )
    identity = recorded.identity if recorded is not None else None
    if (
        recorded is None
        or identity is None
        or not retrieves_period(
            identity.get("requested_window"), requirement, first_year=first_year
        )
    ):
        return None
    try:
        part = await resolve_definition(
            surface, spec_id=spec_id, identity=identity, groups=recorded.groups
        )
        resolved_spec(part, spec_id=spec_id)
        source = outcome_from_envelope({"parts": [part]}).sources[0]
        # Resolving an immutable pin may restate the canonical source label.
        # Keep the original aliases/exclusions that produced this identity.
        return source.model_copy(
            update={
                "resolved_sources": source.resolved_sources
                + tuple(getattr(recorded, "resolved_sources", ()))
            }
        )
    except DefinitionNotResolvedError as exc:
        if not exc.retryable:
            return None
        return _unavailable(
            requirement,
            f"Search could not be asked for {spec_id!r} over {requirement.period.start} to "
            f"{requirement.period.end}: {exc}",
            exc.not_issued,
        )
    except (PartNotResolvedError, ValueError):
        return None


def _resolve_pin_block(
    tool_context: ToolContext, *, plan: Plan, requirement: DatasetRequirement
) -> Mapping[str, object] | None:
    """The spec block of the dataset `requirement` pins, when the planner was shown it; `None` otherwise.

    The pin is the planner's assertion that the dataset answers the
    requirement, so a dataset it was shown this entry
    (`planner.plan.pin_sources`) is reused when `requirement` is the only
    dataset requirement in `plan` pinning it. With several, nothing tells
    which one the search that returned it was about. `None` too when the
    search part that returned it carries a code in `BLOCKING_WARNING_CODES`,
    since one means the search dropped part of the ask and answered a
    narrower one, and when no block was carried forward for it. An earlier
    turn's source reads the block that turn stored (`PinSource.spec_block`).
    The compiler checks the reused source's entities and window against the requirement.
    """
    pin = requirement.source_spec_id
    if pin is None:
        return None
    source = pin_sources(tool_context.state).get(pin)
    if source is None or BLOCKING_WARNING_CODES.intersection(source.warnings):
        return None
    pinning = [
        item
        for item in plan.evidence_requirements
        if isinstance(item, DatasetRequirement) and item.source_spec_id == pin
    ]
    if len(pinning) != 1:
        return None
    return source.spec_block(pin)


def _block_source(
    tool_context: ToolContext, *, plan: Plan, requirement: DatasetRequirement
) -> SelectedSource | None:
    """The source of the pinned dataset's block as the planner was shown it, when it covers `requirement.period` exactly; `None` otherwise."""
    block = _resolve_pin_block(tool_context, plan=plan, requirement=requirement)
    if block is None:
        return None
    try:
        source = source_from_spec_block(block)
    except ValueError:
        return None
    if source.coverage_kind != EXACT_COVERAGE or source.coverage_window != requirement.period:
        return None
    return source


async def _source_from_resolve_pin(
    tool_context: ToolContext,
    *,
    plan: Plan,
    requirement: DatasetRequirement,
    requirement_id: str,
    surface: ToolSurface,
) -> dict[str, object] | None:
    """Build and record `requirement`'s source from what this turn already holds, no `search` call.

    The source is the pinned dataset's block as the planner was
    shown it (`_block_source`), when it parses as a source whose
    `coverage_kind` is `exact` and whose `coverage_window` equals
    `requirement.period`. That check exists because a search's identity
    carries the question's own window and the
    plan's retrieval writes no `scope` (`compiler.compile._retrieve_arguments`),
    so a requirement over other years needs another spec. A `breadth`,
    `bare` or `latest` block states no window to compare.

    Otherwise the source is Search's answer for the pin's identity on record
    over the requirement's period (`_recorded_source`), which is how a pin
    `propose_plan` moved to a copy over that period is honoured. Returns
    `None`, not a raised problem, when none of these holds a source. The
    caller falls through to a fresh `search`, the path for a pin with
    nothing on record, which `search` itself must confirm.

    A requirement whose `across` asks for every member of a side of a flow
    (`requirement_open_roles`) takes the source only when it leaves open every
    party `across` covers (`compiler.evidence.open_axes_for`). Otherwise the source is recorded
    as set aside (`DiscoveryLog.set_aside_pin`) and this returns `None` too,
    and the fresh `search` asks for the side as an open role, which elects
    a source broken down by it where one exists. A source that search finds
    in the pin's place and that drops the set-aside source's scope is
    withheld from the referral when every reader sums the side away
    (`DiscoveryLog.scope_kept`).

    A source that leaves open the side an entity `requirement` names sits on,
    and so binds none of those names (`compiler.evidence.named_on_open_axes`),
    is not honoured either. The fresh `search` sends the names with their
    roles (`search_part`), so the source it finds binds them, and the plan
    approved holds the entities its run resolves. "ODA of the United States
    and Germany" pinned to a source holding every donor open is that case.

    When none of the readers above holds the pin's source, the source a
    search recorded for its `spec_id` (`DiscoveryLog.recorded_source`) is
    read for that check alone, so a pin of a source the translator found
    itself is set aside too. It never honours the pin: its identity carries
    the window of the search that found it.
    """
    read = await _pin_source(
        tool_context,
        plan=plan,
        requirement=requirement,
        first_year=split_of(plan).whole_history.get(requirement.id),
        surface=surface,
    )
    if read is None or isinstance(read, dict):
        return read
    source, honoured = read
    holds_closed = _holds_closed(requirement, source)
    if not holds_closed and (not honoured or named_on_open_axes(requirement, source)):
        return None
    if holds_closed:
        set_aside = DiscoveryRecord(
            revision=plan_revision(plan),
            tool="pin",
            requirement_id=requirement_id,
            requirement_key=requirement_revision(requirement),
            spec_ids=(source.spec_id,),
            variable_sent=requirement.measure,
            detail=f"set aside: it holds closed a side `across` {list(requirement.across)!r} asks for",
        )
        async with discovery_lock(tool_context.state):
            log = load_log(tool_context.state)
            save_log(tool_context.state, log.with_record(set_aside, (source,)))
        return None
    log = await _record_honoured_pin(
        tool_context, plan=plan, requirement=requirement, source=source
    )
    return ok(
        requirement_id=requirement_id,
        status=DiscoveryStatus.DEFINITE.value,
        searched_as=requirement.measure,
        sources=[compact_source(source, selectable=True)],
        detail=None,
        warnings=[],
        budget_remaining=DISCOVERY_CALL_BUDGET - log.calls(plan_revision(plan)),
    )


async def _pin_source(
    tool_context: ToolContext,
    *,
    plan: Plan,
    requirement: DatasetRequirement,
    first_year: int | None,
    surface: ToolSurface,
) -> tuple[SelectedSource, bool] | dict[str, object] | None:
    """The source `requirement`'s pin names, and whether it is honoured, read as `_source_from_resolve_pin` describes.

    `first_year` is the year `plan`'s split reads `requirement` from
    (`HistorySplit.whole_history`), `None` when it reads the stated period.
    The source is stamped with the requirement's `measure` as `searched_as`.
    Honoured means the block or identity reader holds it over the
    requirement's period. A source read only off the discovery log
    (`DiscoveryLog.recorded_source`) is not honoured, and serves the scope
    check alone. A source that does not confirm the `source` or the
    `source_code` the requirement states
    (`compiler.evidence.source_requirement_matches`) is not honoured either,
    so the caller searches again with them. `None` when no reader holds a
    source; a problem a reader met is returned as it is.
    """
    source: SelectedSource | dict[str, object] | None = _block_source(
        tool_context, plan=plan, requirement=requirement
    )
    if source is None:
        source = await _recorded_source(
            tool_context,
            requirement=requirement,
            first_year=first_year,
            surface=surface,
        )
    if isinstance(source, dict):
        return source
    honoured = source is not None
    if source is None and requirement.source_spec_id is not None:
        source = load_log(tool_context.state).recorded_source(requirement.source_spec_id)
    if source is None:
        return None
    if (
        requirement.source is not None or requirement.source_code is not None
    ) and not source_requirement_matches(requirement=requirement, source=source):
        # Refresh an unconfirmed source constraint or series code while
        # retaining the pin's scope, so a replacement cannot silently widen a
        # closed group.
        honoured = False
    return source.model_copy(update={"searched_as": requirement.measure}), honoured


def _holds_closed(requirement: DatasetRequirement, source: SelectedSource) -> bool:
    """Whether `source` fails to leave open a side of a flow `requirement`'s `across` asks every member of."""
    return bool(requirement_open_roles(requirement)) and isinstance(
        open_axes_for(requirement.across, source.open_entity_axes), str
    )


async def _record_honoured_pin(
    tool_context: ToolContext,
    *,
    plan: Plan,
    requirement: DatasetRequirement,
    source: SelectedSource,
) -> DiscoveryLog:
    """Record `source` as the `definite` source of `requirement`'s pin in `plan`, no `search` spent."""
    record = DiscoveryRecord(
        revision=plan_revision(plan),
        tool="pin",
        requirement_id=requirement.id,
        requirement_key=requirement_revision(requirement),
        status=DiscoveryStatus.DEFINITE,
        spec_ids=(source.spec_id,),
        variable_sent=requirement.measure,
    )
    async with discovery_lock(tool_context.state):
        log = load_log(tool_context.state).with_record(record, (source,))
        save_log(tool_context.state, log)
    return log


class NamedGroup(NamedTuple):
    """A group `name_pinned_groups` named on a requirement in place of one `across` word."""

    requirement_id: str
    entity: Entity
    across: str


async def name_pinned_groups(
    tool_context: ToolContext, *, surface: ToolSurface
) -> tuple[NamedGroup, ...]:
    """Name the group a pin holds a requirement's `across` side to, in place of that `across` word, on the draft.

    A dataset requirement whose `across` asks for every member of a side of
    a flow (`requirement_open_roles`), and whose pinned source holds that side closed
    to one group (`translator.referral.closing_group`), names the group in
    `entities` under the side's `PARTY_ROLES` word and drops the word from
    `across`. The pin stays. A source a search could find in the pin's place either leaves
    the side open, dropping the group, or holds it to the same group
    (`translator.referral.dropped_scope`), so no search is spent to learn
    that. A named group is read as the pinned source binds it, as its
    members or as its published total (`compiler.evidence.named_members`),
    and every reader sums the side away, so either reading gives the
    group's figure. Where every reader ranks over the side instead, the
    group is named for its members, so the rank orders the members the pin
    held. A side held to named
    members alone is left as it is: widening it changes the scope, which the
    planner decides (`DiscoveryLog.scope_kept`). So is the total a share or
    quotient divides by when the group is the numerator's own
    (`_divides_its_member_by_itself`): "G7's share of ODA to Kenya" named G7
    on its total would divide G7 by itself, so the pin is set aside and the
    total stays the whole.

    The requirements read are those the model wrote (`authored`), so a
    dataset read beside a copy of its whole history
    (`Plan.rolling_copies`) is judged with the readers of both, and the
    copies name the group as the split of the rewritten plan rebuilds them
    (`planner.plan.with_authored_requirements`), each keeping its own pin.

    The side turns from open to named, so a requirement is rewritten only
    when every requirement reading it sums that side away
    (`_summed_by_every_reader`) or ranks over it (`_ranked_sides`), which
    then reads the group's members, and only when the plan still validates
    with it rewritten: a decision the requirement cites may fix its `across` or
    `entities`. A requirement left as it is, with the reason logged, takes
    discovery's set-aside path, and its referral withholds the sources that
    drop the pin's scope only when every reader sums the side away or ranks
    the members of a group the pin holds it to (`DiscoveryLog.scope_kept`).

    The draft is saved in place, at its version, so the plan the user
    approves names the group. A pin whose source is honoured is recorded
    for the rewritten requirement (`_record_honoured_pin`), so discovery
    reads nothing again. A copy pins its own dataset, over its whole
    history, which the origin's source does not describe, so discovery
    reads the copy's pin as it reads any other. Returns each group named,
    in the plan's order; empty when there is no draft or nothing to name.
    """
    plan = current_draft(tool_context)
    if isinstance(plan, dict):
        return ()
    whole_history = split_of(plan).whole_history
    written = authored(plan)
    named: list[NamedGroup] = []
    honoured: list[tuple[DatasetRequirement, SelectedSource]] = []
    for requirement in written.evidence_requirements:
        if not isinstance(requirement, DatasetRequirement) or requirement.source_spec_id is None:
            continue
        found = await _groups_named(
            tool_context,
            plan=written,
            requirement=requirement,
            first_year=whole_history.get(requirement.id),
            surface=surface,
        )
        if found is None:
            continue
        groups, source = found
        try:
            plan, written, rewritten = _with_groups_named(plan, written, requirement, groups)
        except ValueError as exc:
            logger.info(
                "left %r with across %r: naming %r in its place does not validate: %s",
                requirement.id,
                [group.across for group in groups],
                [group.entity.name for group in groups],
                exc,
            )
            continue
        named.extend(groups)
        if source is not None:
            honoured.append((rewritten, source))
    if not named:
        return ()
    save_plan(tool_context.state, plan)
    for requirement, source in honoured:
        await _record_honoured_pin(tool_context, plan=plan, requirement=requirement, source=source)
    return tuple(named)


def _origin_id(plan: Plan, requirement_id: str) -> str:
    """The requirement `requirement_id` copies over its whole history (`Plan.rolling_copies`), else `requirement_id`."""
    origin = {copy.copy_id: copy.origin_id for copy in plan.rolling_copies}
    return origin.get(requirement_id, requirement_id)


def _ranks_its_group_members(
    plan: Plan, requirement: DatasetRequirement, held: SelectedSource
) -> bool:
    """Whether every reader of `requirement` ranks over a side `held` holds to the members of one group.

    Judged on the plan as the model wrote it (`authored`), as
    `_sums_its_sides_away` judges it. The group is the one `_group_named`
    names for its members on an `across` word of that side.
    """
    ranked = _ranked_sides(authored(plan), _origin_id(plan, requirement.id))
    return any(
        flow_side(word) in ranked
        and (role := party_role(word)) is not None
        and _group_named(held, word, role, members=True) is not None
        for word in requirement.across
    )


def _sums_its_sides_away(plan: Plan, requirement_id: str) -> bool:
    """Whether every reader of `requirement_id` in `plan` sums its sides away (`_summed_by_every_reader`).

    Judged on the plan as the model wrote it (`authored`), so a copy of a
    requirement's whole history (`Plan.rolling_copies`) is judged with the
    readers of the requirement it copies, as `name_pinned_groups` judges it.
    """
    return _summed_by_every_reader(authored(plan), _origin_id(plan, requirement_id))


def _divides_its_member_by_itself(
    plan: Plan, requirement: DatasetRequirement, pinned: SelectedSource
) -> bool:
    """Whether `pinned`, `requirement`'s pinned source, binds an open side to the members a numerator dividing by `requirement` names, and to no others.

    True when a share_of or quotient_of of `plan` reads `requirement` as its
    denominator, directly or through aggregates (`_dividers`), and on a side
    `requirement.across` opens every entity `pinned` binds is one its
    numerator names (`planner.plan.side_entities`). Both are compared by graph ref:
    the numerator's names are resolved against `pinned` by label, by the refs
    Search bound for them or by the refs the graph resolved them to
    (`compiler.evidence.named_refs`), and a group the numerator names stands
    for every member `pinned` binds on its side. Keeping that scope on the
    denominator, by withholding the sources that drop it
    (`DiscoveryLog.scope_kept`) or by naming its group (`name_pinned_groups`),
    would divide the numerator by itself. A pin binding that side to other
    entities too, as a group whose members include the numerator's, keeps
    a scope the user may have named as the whole, so it is not this case.
    Judged on the plan as the model wrote it (`authored`), as
    `_sums_its_sides_away` judges it.
    """
    written = authored(plan)
    by_id = {item.id: item for item in written.evidence_requirements}
    sides = {side for word in requirement.across if (side := flow_side(word)) is not None}
    for divider in _dividers(written, _origin_id(plan, requirement.id)):
        members = named_refs(side_entities(by_id, divider.numerator), pinned)
        for side in sides:
            bound = {entity.ref for entity in pinned.entities if flow_side(entity.role) == side}
            if bound and bound <= members:
                return True
    return False


def _dividers(
    plan: Plan, requirement_id: str
) -> tuple[ShareOfRequirement | QuotientRequirement, ...]:
    """Each share_of or quotient_of of `plan` reading `requirement_id` as its denominator, directly or through aggregates."""
    found: list[ShareOfRequirement | QuotientRequirement] = []
    for reader in plan.evidence_requirements:
        if requirement_id not in inputs_of(reader):
            continue
        if isinstance(reader, ShareOfRequirement | QuotientRequirement):
            if reader.denominator.input_id == requirement_id:
                found.append(reader)
        elif isinstance(reader, AggregateRequirement):
            found.extend(_dividers(plan, reader.id))
    return tuple(found)


def _summed_by_every_reader(plan: Plan, requirement_id: str) -> bool:
    """Whether every requirement reading `requirement_id` is an aggregate grouped by no entity.

    Such an aggregate sums every role away, so it computes the same figure
    whether a side is open or names a group's members. Any other reader
    treats the two apart: an aggregate grouped by entity keeps a named side
    per member where it summed an open one, and a rank, a share, a quotient,
    a join or a filter keys or narrows by the side. True when nothing reads
    `requirement_id`.
    """
    return all(
        isinstance(reader, AggregateRequirement) and GroupBy.ENTITY not in reader.group_by
        for reader in plan.evidence_requirements
        if requirement_id in inputs_of(reader)
    )


def _ranked_sides(plan: Plan, requirement_id: str) -> frozenset[FlowSide]:
    """The sides of a flow every requirement reading `requirement_id` ranks over, empty unless each is such a rank.

    A rank orders the members of its `axis` side one by one, whether that
    side is open or names a group's members, so naming the group a pin holds
    the side to ranks the members the pin held. Empty when nothing reads
    `requirement_id`, or when a reader is not a rank stating an `axis` that
    names a side of a flow.
    """
    sides: set[FlowSide] = set()
    for reader in plan.evidence_requirements:
        if requirement_id not in inputs_of(reader):
            continue
        if not isinstance(reader, RankRequirement) or reader.axis is None:
            return frozenset()
        if (side := flow_side(reader.axis)) is None:
            return frozenset()
        sides.add(side)
    return frozenset(sides)


def _group_named(source: SelectedSource, word: str, role: str, *, members: bool) -> Entity | None:
    """The entity naming the one group `source` holds the `across` word's side to, or `None`.

    The group is `translator.referral.closing_group`'s. Named whole, it is
    met by the source under either reading (`compiler.evidence.named_members`).
    Named for its `members`, it is met only where the source binds the
    members, so a group the source binds as a published total
    (`ResolvedGroup.reads_total`) gives `None`.
    """
    label = closing_group(source, word)
    if label is None:
        return None
    if not members:
        return Entity(name=label, role=role)
    side = flow_side(word)
    if any(
        group.reads_total
        for group in source.groups
        if group.label == label and flow_side(group.role) == side
    ):
        return None
    return Entity(name=label, role=role, members=True)


def _with_groups_named(
    plan: Plan, written: Plan, requirement: DatasetRequirement, groups: Sequence[NamedGroup]
) -> tuple[Plan, Plan, DatasetRequirement]:
    """`plan`, and `written` as the model wrote it, with `requirement` naming each group's entity in place of its `across` word, and that requirement.

    `written` is `authored(plan)`, and the rewritten `plan` the split of the
    rewritten `written` (`planner.plan.with_authored_requirements`).

    Raises:
        ValueError: If the rewritten requirement or plan does not validate.
    """
    dropped = {group.across for group in groups}
    rewritten = DatasetRequirement.model_validate(
        {
            **requirement.model_dump(),
            "entities": [*requirement.entities, *(group.entity for group in groups)],
            "across": [word for word in requirement.across if word not in dropped],
        }
    )
    requirements = [
        rewritten if item.id == requirement.id else item for item in written.evidence_requirements
    ]
    rebuilt = with_authored_requirements(plan, requirements)
    return rebuilt, authored(rebuilt), rewritten


async def _groups_named(
    tool_context: ToolContext,
    *,
    plan: Plan,
    requirement: DatasetRequirement,
    first_year: int | None,
    surface: ToolSurface,
) -> tuple[tuple[NamedGroup, ...], SelectedSource | None] | None:
    """The group `requirement`'s pin holds each `across` side to, and the pin's source when it is honoured.

    `plan` is the plan as the model wrote it, and `first_year` as
    `_pin_source` takes it.

    One group per `across` word whose side the pin's source holds to one
    group (`_group_named`). A plan whose every reader ranks the requirement
    over that side (`_ranked_sides`) names the group's members
    (`Entity.members`), so "the largest DAC donors" pinned to the DAC
    members ranks those members and no other provider. `None` when a
    requirement reading it keeps the sides apart otherwise (neither
    `_summed_by_every_reader` nor `_ranked_sides`), when a
    share or quotient divides by it and the pin's source binds the side to
    the numerator's own members alone (`_divides_its_member_by_itself`), as
    in "G7's share of ODA to Kenya" with the total pinned to G7, or when
    the pin's source cannot be read, leaves every side `across` asks for
    open, or holds none of them to one group.
    """
    summed = _summed_by_every_reader(plan, requirement.id)
    ranked = frozenset() if summed else _ranked_sides(plan, requirement.id)
    roles = [
        (word, role)
        for word in requirement.across
        if (role := party_role(word)) and (summed or flow_side(word) in ranked)
    ]
    if not roles:
        return None
    read = await _pin_source(
        tool_context, plan=plan, requirement=requirement, first_year=first_year, surface=surface
    )
    if read is None or isinstance(read, dict):
        return None
    source, honoured = read
    if not _holds_closed(requirement, source) or _divides_its_member_by_itself(
        plan, requirement, source
    ):
        return None
    groups = tuple(
        NamedGroup(requirement.id, entity, word)
        for word, role in roles
        if (entity := _group_named(source, word, role, members=not summed)) is not None
    )
    if not groups:
        return None
    return groups, source if honoured else None


async def _discovery_start(
    tool_context: ToolContext,
    requirement_id: str,
    reservations: _Reservations,
    *,
    surface: ToolSurface,
) -> dict[str, object] | tuple[Plan, DatasetRequirement, str, str]:
    """Ready to search: the draft plan, the requirement, its revision, and its key, with
    this call's reservation already made; the source already honoured from
    this turn's own resolve pin, with no `search` spent; or the problem
    that stopped either.

    The pin check (`_source_from_resolve_pin`) runs before the budget check
    below: it spends no call, so a pin naming this turn's own resolve id
    must never be refused for a budget it does not touch. The budget check
    and the reservation it makes are one critical section once a fresh
    `search` is needed: two concurrent calls' checks must never both pass
    for the same slot. The `search` call itself runs with the lock released.
    """
    plan = current_draft(tool_context)
    if isinstance(plan, dict):
        return plan
    requirement = _dataset_requirement(plan, requirement_id)
    if isinstance(requirement, dict):
        return requirement
    pinned = await _source_from_resolve_pin(
        tool_context,
        plan=plan,
        requirement=requirement,
        requirement_id=requirement_id,
        surface=surface,
    )
    if pinned is not None:
        return pinned
    async with discovery_lock(tool_context.state):
        ready = _ready_to_search(tool_context, requirement_id, reservations)
        if isinstance(ready, dict):
            return ready
        plan, requirement, revision, key = ready
        reservations.reserve(revision, requirement_id)
    return plan, requirement, revision, key


def _referred_answer(
    requirement: DatasetRequirement,
    *,
    kind: ReferralKind,
    code: ProblemCode,
    message: str,
    detail: Mapping[str, Any],
) -> dict[str, object]:
    """The refusal for a `search` answer no retrieval may take a source from,
    referring `requirement` back to the planner as `kind` with `message`."""
    return problem(
        code,
        message,
        outcome=Outcome.REFUSED,
        action=Action(
            call="return_to_planner",
            arguments={"kind": kind.value, "requirement_id": requirement.id, "reason": message},
        ),
        detail=dict(detail),
    )


def _unavailable(
    requirement: DatasetRequirement,
    message: str,
    not_issued: bool,
    *,
    detail: dict[str, JsonValue] | None = None,
) -> dict[str, object]:
    """The problem for a data-service call about `requirement` that did not complete.

    A call the planner entry had no time left to issue (`not_issued`)
    refers the requirement back as `discovery_unavailable` marked
    `not_issued`, which the planner seam takes as the request's wall-clock
    bound spent and ends the entry on. Any other failure names no `action`, and `translator.mechanical` refers
    it back as `discovery_unavailable` unmarked.
    """
    action = None
    if not_issued:
        action = Action(
            call="return_to_planner",
            arguments={
                "kind": ReferralKind.DISCOVERY_UNAVAILABLE.value,
                "requirement_id": requirement.id,
                "reason": message,
                "not_issued": True,
            },
        )
    return problem(
        ProblemCode.SURFACE_UNAVAILABLE,
        message,
        outcome=Outcome.REFUSED,
        action=action,
        detail=detail,
    )


def _ask_not_understood(
    requirement: DatasetRequirement,
    part: Mapping[str, object],
    *,
    blocked_by: Sequence[str],
    warnings: list[dict[str, Any]],
    budget_remaining: int,
) -> dict[str, object]:
    """The refusal for an answer `search` gave with part of the ask dropped
    (`blocked_by`, the warning codes it dropped it under)."""
    dropped = _dropped_phrases(requirement, part, blocked_by)
    return _referred_answer(
        requirement,
        kind=ReferralKind.ASK_NOT_UNDERSTOOD,
        code=ProblemCode.ASK_NOT_UNDERSTOOD,
        message=(
            f"search did not understand part of the ask for {requirement.id!r} and "
            f"answered without it: {dropped!r}. The answer is to a narrower ask than "
            "the requirement states, so it is not selectable."
        ),
        detail={"dropped": dropped, "warnings": warnings, "budget_remaining": budget_remaining},
    )


def build_discovery_tools(*, surface: ToolSurface) -> list[FunctionTool]:
    """Build the one discovery tool, closing over `surface` for the call."""

    async def discover_source(
        requirement_id: str,
        tool_context: ToolContext,
    ) -> dict[str, object]:
        """Find what data exists for one dataset requirement.

        The search is built from the requirement itself: its measure, its
        entities and their roles, its period, and its unit if it states one.
        Call this once per dataset requirement. A `definite` answer names the
        one source a retrieval of that requirement may use. `candidates`,
        `no_data` and `unsupported` are not yours to resolve: follow the
        `action` this returns, which names `return_to_planner` and the kind
        of referral to make -- unless the requirement already pins a
        `source_spec_id` and this answer confirms it among its sources, in
        which case there is no `action` at all: that pin is the source,
        settled, the same as a `definite` answer's. A pin naming the exact id
        this turn's own resolve settled on is honoured straight from that
        resolve (`_source_from_resolve_pin`), with no `search` call at all,
        and so is a pin naming a dataset the user handed over from Search.
        Any other pin is confirmed the usual way, from a fresh search's own
        answer.

        Args:
            requirement_id: The dataset requirement this search serves.

        Returns:
            The answer's status, the sources it names, and how many discovery
            calls this plan revision has left; or the problem that stopped it.
        """
        reservations = _reservations(tool_context.state)
        started = await _discovery_start(
            tool_context, requirement_id, reservations, surface=surface
        )
        if isinstance(started, dict):
            return started
        plan, requirement, revision, key = started

        variable = requirement.measure
        part = search_part(requirement, variable)

        async def _finish(
            record: DiscoveryRecord, sources: Sequence[SelectedSource] = ()
        ) -> DiscoveryLog:
            """Record `record` (and any `sources`) and release this call's reservation.

            Binds `_record_search`'s `reservations`/`revision`/`requirement_id`
            to this call, the same three for every record this call may write.
            """
            return await _record_search(
                tool_context,
                record,
                sources,
                reservations=reservations,
                revision=revision,
                requirement_id=requirement_id,
            )

        try:
            answer = await call_within_entry(surface, tool=SEARCH, arguments={"parts": [part]})
        except ToolSurfaceUnavailableError as exc:
            if isinstance(exc, NoTimeLeftError):
                # Never issued, so nothing is spent: no record, and the
                # reservation is released alone, under the lock.
                async with discovery_lock(tool_context.state):
                    reservations.release(revision, requirement_id)
                    log = load_log(tool_context.state)
            else:
                # A timeout still spends the call: recorded, so it counts
                # against the budget exactly like an answer, and marked
                # `transport_failed` so it can never be selected. The
                # reservation this call made is released in the same locked
                # section as that record's commit, inside `_record_search`.
                log = await _finish(
                    DiscoveryRecord(
                        revision=revision,
                        tool="search",
                        requirement_id=requirement_id,
                        requirement_key=key,
                        variable_sent=variable,
                        detail=str(exc),
                        transport_failed=True,
                    )
                )
            return _unavailable(
                requirement,
                str(exc),
                isinstance(exc, NoTimeLeftError),
                detail={"budget_remaining": DISCOVERY_CALL_BUDGET - log.calls(revision)},
            )
        except BaseException:
            # An unexpected exception spends nothing: no record is ever
            # written for it, so the reservation it made must not linger
            # either, or a following call would be refused for a budget
            # nothing was actually spent from. There is no record to commit
            # here, so this releases alone, under the lock.
            async with discovery_lock(tool_context.state):
                reservations.release(revision, requirement_id)
            raise

        record = DiscoveryRecord(
            revision=revision,
            tool="search",
            requirement_id=requirement_id,
            requirement_key=key,
            variable_sent=variable,
        )
        if answer.is_error:
            log = await _finish(record.model_copy(update={"detail": "search refused"}))
            return problem(
                ProblemCode.SEARCH_REFUSED,
                "data-mcp refused this search.",
                outcome=Outcome.REFUSED,
                detail={
                    "refusal": answer.structured_content,
                    "budget_remaining": DISCOVERY_CALL_BUDGET - log.calls(revision),
                },
            )
        try:
            envelope = (answer.meta or {}).get(SEARCH_ENVELOPE_META_KEY)
            if not isinstance(envelope, dict):
                raise ValueError(
                    f"the search result carries no envelope under _meta[{SEARCH_ENVELOPE_META_KEY!r}]"
                )
            outcome = outcome_from_envelope(
                cast("Mapping[str, object]", envelope),
                entities=[
                    EntityKey(name=entity.name, role=entity.role) for entity in requirement.entities
                ],
            )
        except ValueError as exc:
            log = await _finish(record.model_copy(update={"detail": str(exc)}))
            return problem(
                ProblemCode.SEARCH_REFUSED,
                "data-mcp answered with an envelope that could not be read.",
                outcome=Outcome.REFUSED,
                detail={
                    "parse_error": str(exc),
                    "budget_remaining": DISCOVERY_CALL_BUDGET - log.calls(revision),
                },
            )

        sources = tuple(
            source.model_copy(update={"searched_as": variable}) for source in outcome.sources
        )
        blocked_by = tuple(
            dict.fromkeys(w.code for w in outcome.warnings if w.code in BLOCKING_WARNING_CODES)
        )
        log = await _finish(
            record.model_copy(
                update={
                    "status": outcome.status,
                    "spec_ids": tuple(s.spec_id for s in sources),
                    "detail": outcome.detail,
                    "reason": outcome.reason,
                    "ambiguous_sources": outcome.ambiguous_sources,
                    "ambiguous_entities": outcome.ambiguous_entities,
                    "warnings": outcome.warnings,
                    "blocked_by": blocked_by,
                }
            ),
            sources,
        )
        warnings = [w.model_dump(mode="json", exclude_none=True) for w in outcome.warnings]
        budget_remaining = DISCOVERY_CALL_BUDGET - log.calls(revision)

        if blocked_by:
            return _ask_not_understood(
                requirement,
                part,
                blocked_by=blocked_by,
                warnings=warnings,
                budget_remaining=budget_remaining,
            )

        # A pin is only honoured from an answer a pin may be taken from, the
        # same statuses `selectable` accepts it from. A pinned spec_id that
        # turns up among a `no_data` answer's nearest series has not been
        # found: saying otherwise sends the model back for a source discovery
        # will not hand it.
        honoured = {s.spec_id for s in sources} if outcome.status in _PIN_HONOURING else set[str]()
        pin_honoured = (
            requirement.source_spec_id is not None and requirement.source_spec_id in honoured
        )
        # A refused `source_code` is not a missing pin: the pin's source was
        # read, and the code cannot be served under any dataset, so the
        # answer's own hint reaches the planner. Nor is a pin judged while
        # Search cannot tell which node an entity names: the entity is
        # settled first, and the pin read again with it.
        code_refused = (
            requirement.source_code is not None and outcome.reason in SOURCE_CODE_REFUSALS
        )
        pinned_not_found = (
            requirement.source_spec_id is not None
            and not pin_honoured
            and not code_refused
            and not outcome.ambiguous_entities
        )

        # An honoured pin has already settled this requirement --
        # `selectable` (this module) resolves it from the log the same
        # way a `definite` answer would -- so no hint is due even for a
        # `candidates` or other non-`definite` status. Setting one here
        # unconditionally on `outcome.status` alone would re-pin a settled
        # requirement to the same spec_id, get the same answer again, and
        # refer it back forever, with only the guard downstream ever
        # stopping the loop.
        action: Action | None = None
        if not pin_honoured:
            hint_kind = (
                ReferralKind.SOURCE_NAME_AMBIGUOUS
                if outcome.ambiguous_sources
                else ReferralKind.ENTITY_NAME_AMBIGUOUS
                if outcome.ambiguous_entities
                else _HINT_KIND.get(outcome.status)
            )
            if hint_kind is not None:
                reason = f"discover_source answered {outcome.status.value} for {requirement_id!r}"
                if outcome.detail:
                    reason = f"{reason}: {outcome.detail}"
                action = Action(
                    call="return_to_planner",
                    arguments={
                        "kind": hint_kind.value,
                        "requirement_id": requirement_id,
                        "reason": reason,
                    },
                )
        if pinned_not_found:
            # `honoured` is what this call actually found among the
            # statuses a pin may be taken from -- the same sources
            # `return_to_planner` attaches to the referral as
            # `candidates` (`DiscoveryLog.pinnable_for`). Naming how many here tells the planner this
            # question can still be answered: a `candidates` answer's
            # two-plus sources are the real alternative worth naming when
            # this action replaces the `source_ambiguous` hint.
            instead = f" This search found {len(honoured)} source(s) instead." if honoured else ""
            scope = log.scope_kept(plan, requirement, [s for s in sources if s.spec_id in honoured])
            if scope.withheld:
                instead += (
                    f" {len(scope.withheld)} of them drop the scope the pinned source held and "
                    "are withheld; "
                    + (
                        f"{len(scope.offered)} keep it."
                        if scope.offered
                        else f"none keeps it, so {scope.repair}."
                        if scope.repair
                        else "none keeps it, so ask the user how to proceed."
                    )
                )
            action = Action(
                call="return_to_planner",
                arguments={
                    "kind": ReferralKind.PINNED_SOURCE_NOT_FOUND.value,
                    "requirement_id": requirement_id,
                    "reason": (
                        f"{requirement_id!r} pins {requirement.source_spec_id!r}, which "
                        f"this revision's discovery did not find.{instead}"
                    ),
                },
            )

        fields: dict[str, object] = {
            "requirement_id": requirement_id,
            "status": outcome.status.value,
            "searched_as": variable,
            "sources": [
                compact_source(
                    s,
                    selectable=(
                        outcome.status is DiscoveryStatus.DEFINITE
                        or (pin_honoured and s.spec_id == requirement.source_spec_id)
                    ),
                )
                for s in sources
            ],
            "detail": outcome.detail,
            "ambiguous_sources": [
                group.model_dump(mode="json") for group in outcome.ambiguous_sources
            ],
            "ambiguous_entities": [
                group.model_dump(mode="json") for group in outcome.ambiguous_entities
            ],
            "warnings": warnings,
            "budget_remaining": budget_remaining,
        }
        if action is not None:
            fields["action"] = action.model_dump(mode="json", exclude_none=True)
        return ok(**fields)

    return [FunctionTool(discover_source)]

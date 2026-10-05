"""Reads a `search` result into the sources a plan can select.

data-mcp's `search` answers with an envelope carrying one entry in `parts` per
part asked, and each part's own `status` says what it found: one reading
(`definite`), several (`candidates`), none (`no_data`), or an ask it cannot
serve (`unsupported`). Discovery asks one part per call, so
`outcome_from_envelope` reads that one part into a `DiscoveryOutcome`, and each
spec block it carries into a `SelectedSource`, the record a `RetrieveOperation`
is compiled against and a workflow keeps.

The envelope is untrusted input from another service. A block that does not
carry what a source needs raises `ValueError`, for the caller to report. What
it merely leaves out -- `has_data`, a `definition.identity`, a series window
-- is read as unknown (`None`), never as fine: data-mcp's own digest reads
those fields with no default, and an optimistic default here would disagree
with the producer.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from enum import StrEnum
from typing import Any, cast

import rfc8785
from data_mcp.substitution import MAX_RESOLVED_STAT_VAR_DCIDS, root_member
from planner.records import Period, StatVarReading, StrictModel
from planner.workflow import (
    OpenAxis,
    OpenEntityAxis,
    ReadingSetBy,
    ResolvedEntity,
    SelectedSource,
)
from pydantic import JsonValue

# The `coverage.kind` under which `search` reports a real series window.
EXACT_COVERAGE = "exact"

# The `shape.kind` of an identity that describes its StatVars by schema and
# slot bindings, where a pinned one names them.
DESCRIBED_SHAPE_KIND = "statvar_schema"

# The prefix dfo-apis's mint refuses on a posted StatVar (`SpecIdInput.
# _dcids_are_not_dcid_prefixed`): each is a bare dcid.
_DCID_PREFIX = "dcid:"


class DiscoveryStatus(StrEnum):
    """What a `search` call found, in the envelope's own terms."""

    DEFINITE = "definite"
    CANDIDATES = "candidates"
    NO_DATA = "no_data"
    UNSUPPORTED = "unsupported"


class PartWarning(StrictModel):
    """One warning `search` attached to its answer.

    `code` and `severity` are plain strings, not closed sets: they come from
    another service, and an unknown value must not fail the whole envelope.
    `spec_id` is set when the warning speaks about one spec block.
    """

    code: str
    severity: str
    message: str
    spec_id: str | None = None


class DiscoveryOutcome(StrictModel):
    """One `search` call's answer: its status, the sources it names, and why.

    `sources` holds one entry per spec block the part carries: the single
    reading of a `definite` answer, every candidate of a `candidates` one,
    and the nearest real series a `no_data` answer offers instead. `detail`
    is the envelope's own explanation for `no_data` and `unsupported`. `reason`
    is `no_data`'s own closed reason code (`NoDataReason` in dfo-apis'
    `search.models.vocab`, e.g. `"no_observations"`), kept as a plain string
    rather than a closed set of our own: the vocabulary belongs to another
    service, and a value it does not yet recognise is still data to carry,
    not a reason to fail the whole envelope. `None` for every status but
    `no_data`. `warnings` are the envelope's request-level
    `diagnostics.warnings` followed by the part's own -- the one upstream
    signal that a role or time phrase was dropped from the search. Discovery
    sends one part per call, so a request-level warning bears on it.
    """

    status: DiscoveryStatus
    sources: tuple[SelectedSource, ...] = ()
    detail: str | None = None
    reason: str | None = None
    warnings: tuple[PartWarning, ...] = ()


def outcome_from_envelope(envelope: Mapping[str, object]) -> DiscoveryOutcome:
    """Read a single-part `search` envelope into a `DiscoveryOutcome`.

    Raises:
        ValueError: If the envelope does not carry exactly one part, the part's
            status is not one `search` documents, a `definite` part names no
            spec, or a spec block it carries is not usable as a source.
    """
    parts = _sequence(envelope.get("parts"))
    if len(parts) != 1:
        raise ValueError(f"search answered {len(parts)} parts; discovery asks one part per call")
    part = parts[0]
    raw_status = part.get("status")
    try:
        status = DiscoveryStatus(str(raw_status))
    except ValueError as exc:
        raise ValueError(f"search answered with an unrecognised status {raw_status!r}") from exc
    warnings = _warnings(envelope, part)

    if status is DiscoveryStatus.DEFINITE:
        spec = _mapping(part.get("spec"))
        if not spec:
            raise ValueError("a definite search part names no spec")
        return DiscoveryOutcome(
            status=status, sources=(source_from_spec_block(spec),), warnings=warnings
        )
    if status is DiscoveryStatus.CANDIDATES:
        blocks = _sequence(_mapping(part.get("candidates")).get("specs"))
        return DiscoveryOutcome(status=status, sources=_sources(blocks), warnings=warnings)
    if status is DiscoveryStatus.NO_DATA:
        no_data = _mapping(part.get("no_data"))
        nearest = [block for block in _sequence(no_data.get("nearest_real")) if "spec_id" in block]
        reason = _text(no_data.get("reason"))
        return DiscoveryOutcome(
            status=status,
            sources=_sources(nearest),
            detail=reason,
            reason=reason,
            warnings=warnings,
        )
    unsupported = _mapping(part.get("unsupported"))
    detail = " ".join(
        part
        for part in (_text(unsupported.get("detail")), _text(unsupported.get("pointer")))
        if part
    )
    return DiscoveryOutcome(status=status, detail=detail or None, warnings=warnings)


def source_from_spec_block(block: Mapping[str, object]) -> SelectedSource:
    """Read one spec block of a `search` envelope into a `SelectedSource`.

    Takes the block's `spec_id`, what it `resolved` -- sentence, confidence,
    coverage, measures, the size and digest of the binding its `shape`
    states, entities, the entity axes it leaves open, unit, provenance, the
    axes its family varies along -- and the identity document of its
    `definition` sidecar when the envelope carries one, with whether a
    described identity mints as one StatVar (`pinned_stat_var`).
    A `shape` without `member_count` or `member_digest` leaves them `None`.
    `coverage.window` is a series window only when `coverage.kind` is
    `exact`; a `breadth` or `bare` coverage carries none.

    Raises:
        ValueError: If the block lacks a usable `spec_id`, an entity it
            resolves lacks its role, value or label, or the StatVars it
            elected are not a list `get_data` takes (`_stat_vars`).
    """
    resolved = _mapping(block.get("resolved"))
    coverage = _mapping(resolved.get("coverage"))
    coverage_kind = _text(coverage.get("kind"))
    facets = _mapping(resolved.get("facets"))
    unit = _mapping(facets.get("unit"))
    provenance = _mapping(facets.get("provenance"))
    shape = _mapping(resolved.get("shape"))
    identity = _identity(_mapping(block.get("definition")).get("identity"))
    items = _sequence(_mapping(resolved.get("stat_vars")).get("items"))
    try:
        stat_vars, stat_var_member_count, stat_var_truncated = _stat_vars(resolved.get("stat_vars"))
        return SelectedSource(
            spec_id=str(block.get("spec_id")),
            sentence=_text(resolved.get("rendered_sentence")) or "",
            confidence=_text(resolved.get("confidence")),
            has_data=_flag(coverage.get("has_data")),
            stat_vars=stat_vars,
            stat_var_member_count=stat_var_member_count,
            stat_var_truncated=stat_var_truncated,
            pinned=_pinned(identity, items, slots=resolved.get("slots")),
            shape_member_count=_count(shape.get("member_count")),
            shape_member_digest=_text(shape.get("member_digest")),
            entities=tuple(_entity(entity) for entity in _sequence(resolved.get("entities"))),
            open_entity_axes=open_entity_axes_of(resolved),
            unit_ref=_text(unit.get("ref")),
            unit_label=_text(unit.get("label")),
            provenance_ref=_text(provenance.get("ref")),
            provenance_label=_text(provenance.get("label")),
            coverage_kind=coverage_kind,
            coverage_window=(
                _window(_mapping(coverage.get("window")))
                if coverage_kind == EXACT_COVERAGE
                else None
            ),
            identity=identity,
            open_axes=_open_axes(resolved),
        )
    except ValueError as exc:
        # pydantic's ValidationError is a ValueError too, so one clause covers
        # a malformed field and a missing one alike.
        raise ValueError(f"the spec block is not usable as a source: {exc}") from exc


def _stat_vars(value: object) -> tuple[tuple[str, ...], int | None, bool | None]:
    """The dcids `resolved.stat_vars` names, how many members it confirmed, and whether `items` was capped.

    `search` serialises `resolved.stat_vars` as an object -- `items`,
    `member_count`, `truncated` -- never a bare list, so a value that is not
    a mapping (an absent field, or the old wrong list shape) states none of
    the three: `member_count` and `truncated` come back `None`, unknown
    rather than complete, and `items` comes back empty.

    Every item the Search API serves is an object whose `ref` is a dcid, and
    a retrieval posts these dcids to `get_data` as `resolved_stat_var_dcids`,
    so an item of any other form raises, and so does a list `get_data` would
    refuse: one past `MAX_RESOLVED_STAT_VAR_DCIDS`, or a `ref` with the `dcid:` prefix.

    Raises:
        ValueError: If `items` is not a list, holds more than `MAX_RESOLVED_STAT_VAR_DCIDS`
            entries, or one of its items is not an object with a non-empty
            string `ref` that is a bare dcid.
    """
    obj = _mapping(value)
    raw_items = obj.get("items") if obj.get("items") is not None else []
    if not isinstance(raw_items, list):
        raise ValueError(f"resolved.stat_vars.items is not a list: {raw_items!r}")
    if len(raw_items) > MAX_RESOLVED_STAT_VAR_DCIDS:
        raise ValueError(
            f"resolved.stat_vars.items holds {len(raw_items)} StatVars, past the "
            f"{MAX_RESOLVED_STAT_VAR_DCIDS} get_data takes"
        )
    items: list[str] = []
    for index, item in enumerate(raw_items):
        ref = item.get("ref") if isinstance(item, Mapping) else None
        if not isinstance(ref, str) or not ref:
            raise ValueError(
                f"resolved.stat_vars.items[{index}] is not an object with a string ref: {item!r}"
            )
        if ref.startswith(_DCID_PREFIX):
            raise ValueError(
                f"resolved.stat_vars.items[{index}].ref {ref!r} is not a bare dcid: "
                f"it carries the {_DCID_PREFIX} prefix"
            )
        items.append(ref)
    truncated = obj.get("truncated")
    return (
        tuple(items),
        _count(obj.get("member_count")),
        truncated if isinstance(truncated, bool) else None,
    )


def elected_stat_vars(resolved: Mapping[str, object]) -> tuple[str, ...]:
    """The StatVars a spec block's `resolved` elects, as `get_data` takes them.

    The list `source_from_spec_block` records as `SelectedSource.stat_vars`,
    for a reader that offers a block without building a source from it
    (`harness.tool_contracts`, for each clarify candidate).

    Raises:
        ValueError: If the list is not one `get_data` takes (`_stat_vars`).
    """
    return _stat_vars(resolved.get("stat_vars"))[0]


def _count(value: object) -> int | None:
    """`value` as a count, or `None` when it is not a whole number of zero or more.

    A JSON `true` is no count of one, though Python reads `bool` as an `int`.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _open_axes(resolved: Mapping[str, object]) -> tuple[OpenAxis, ...]:
    """The slots `resolved.slots` leaves open, each with whether the family sums over it.

    A slot with no constraint property (the place, time window or source)
    names no StatVar member, and one listed in `inapplicable_slots` is
    declared by none of the resolved StatVars, so neither is an axis. A slot
    is closed only when bound to one value whose code system admitted no
    descendants (`expansion` absent); an unbound or set binding, an expanded
    value, and a binding kind this reader does not know are open. An open
    slot is summable when it says so (`summable`, read as false when
    absent) and is not listed in `unresolved_slots`, where some member
    carries a value dfo-apis could not decode and so could not sum.
    """
    inapplicable = set(_strings(resolved.get("inapplicable_slots")))
    unresolved = set(_strings(resolved.get("unresolved_slots")))
    axes: list[OpenAxis] = []
    for slot in _sequence(resolved.get("slots")):
        prop = _text(_mapping(_mapping(slot.get("key")).get("property")).get("dcid"))
        if prop is None or prop in inapplicable:
            continue
        binding_kind = _mapping(slot.get("binding")).get("kind")
        expanded = bool(_sequence_of_any(_mapping(slot.get("expansion")).get("added")))
        if binding_kind == "value" and not expanded:
            continue
        axes.append(
            OpenAxis(
                property=prop, summable=slot.get("summable") is True and prop not in unresolved
            )
        )
    return tuple(axes)


def pinned_stat_var(
    identity: Mapping[str, Any], items: Sequence[Mapping[str, Any]], *, slots: object
) -> str | None:
    """The one StatVar a described `identity` mints as, or `None`.

    `items` are the block's `resolved.stat_vars.items` and `slots` its
    `resolved.slots`. The rule is data-mcp's
    (`data_mcp.substitution.root_member`): a described identity whose only
    entity role is `observationAbout`, for which `search` elected exactly one
    StatVar that is the family's root. dfo-apis reduces a family to the
    members no other member dominates, and a root dominates every member
    with further constraints, so the mint returns that one StatVar under the
    identity's own spec_id. An identity naming any other role, one electing
    several StatVars or one constrained beyond its bound slots, one whose
    elected StatVar states no `constraint_properties`, and one whose shape
    names its StatVar are minted as they stand.
    """
    return root_member(identity, items, slots=slots)


def _pinned(
    identity: Mapping[str, Any] | None, items: Sequence[Mapping[str, Any]], *, slots: object
) -> bool | None:
    """Whether a described `identity` mints as one StatVar; `None` for any other."""
    shape = (identity or {}).get("shape")
    if not isinstance(shape, Mapping) or shape.get("kind") != DESCRIBED_SHAPE_KIND:
        return None
    return pinned_stat_var(identity or {}, items, slots=slots) is not None


def is_family(source: SelectedSource) -> bool:
    """Whether a retrieval of `source` holds a family of StatVars.

    A described identity is minted with every member of its binding unless
    the one StatVar `search` elected is the family's root
    (`SelectedSource.pinned`), so an unpinned one is a family however many
    members `search` elected: an aid flow keyed on provider and recipient is
    never pinned, and one member elected for it still mints them all. The
    exception is a binding `search` states holds exactly one member
    (`_states_one_member`): minted unpinned, it still holds that one StatVar.
    A source that states no pin verdict (one whose identity names its
    StatVars, one carrying no identity, and a record that does not state the
    field) binds the StatVars `stat_vars` lists, capped upstream well above
    one, so two or more is a family.
    """
    if source.pinned is None:
        return len(source.stat_vars) > 1
    return not source.pinned and not _states_one_member(source)


def member_digest(dcids: Iterable[str]) -> str:
    """Lowercase SHA-256 hex of the RFC 8785 canonical JSON of the sorted `dcids`.

    The digest `search` states as `resolved.shape.member_digest` (dfo-apis
    `statvar_members.digest.member_digest`). Callers pass bare dcids, without
    the `dcid:` prefix.
    """
    return hashlib.sha256(rfc8785.dumps(sorted(dcids))).hexdigest()


def _states_one_member(source: SelectedSource) -> bool:
    """Whether `search` states that the binding of `source` holds exactly the one StatVar it lists.

    `shape_member_digest` is `search`'s digest of the whole binding, stated
    only where `shape_member_count` is a true total; without it the count is
    a floor over a capped recall sample, which says the binding holds at
    least that many. The binding is the one listed StatVar exactly when the
    digest equals `member_digest` of that StatVar alone, so a digest of any
    other population, and an absent one, leave the source a family. The
    counts have to agree with it: a binding count of one, and a listing
    whose `stat_var_member_count` is one and `stat_var_truncated` false. A
    count the envelope left out and a `stat_var_truncated` it did not state
    say nothing, so the source stays a family.
    """
    if len(source.stat_vars) != 1:
        return False
    (listed,) = source.stat_vars
    return (
        source.shape_member_count == 1
        and source.shape_member_digest == member_digest([listed.removeprefix("dcid:")])
        and source.stat_var_member_count == 1
        and source.stat_var_truncated is False
    )


def default_reading(source: SelectedSource) -> StatVarReading:
    """How a family is read when the plan does not say: summed when every open axis is summable.

    A family with no open axis has nothing to sum over, so it reads
    separately; so does one with any axis dfo-apis may not sum over (a sex,
    a netting basis), since its members are different measures.
    """
    if source.open_axes and all(axis.summable for axis in source.open_axes):
        return StatVarReading.SUM
    return StatVarReading.SEPARATE


def stat_var_reading(source: SelectedSource, requested: StatVarReading | None) -> SelectedSource:
    """`source` stating how its retrieval reads it, and who said so.

    A source binding one StatVar has no reading. A family takes the
    requirement's `requested` reading when it states one, and
    `default_reading` otherwise.
    """
    if not is_family(source):
        reading, set_by = None, None
    elif requested is not None:
        reading, set_by = requested, ReadingSetBy.PLAN
    else:
        reading, set_by = default_reading(source), ReadingSetBy.RULE
    return source.model_copy(
        update={"stat_var_reading": reading, "stat_var_reading_set_by": set_by}
    )


def open_entity_axes_of(resolved: Mapping[str, object]) -> tuple[OpenEntityAxis, ...]:
    """The entity axes a spec block's `resolved` leaves open, one per column, sorted by column.

    Two fields state one: `scope` lists every member of the axis `search`
    defaulted for a part that named no entity on it (the one member `Earth`
    for a worldwide total), and each `role_axis_facts` entry counts the members of
    a flow side the part left open. A count either field cannot state, such
    as a `members` that is not a list, is `None`, never 0. When both state
    one column, the scope's count wins, since it lists the members it
    counts. A column a `resolved.entities` entry binds is not open. Never
    raises: `harness.tool_contracts` reads the planner's envelope with it
    too, where a malformed field must not fail the digest.
    """
    bound = {_text(entity.get("role")) for entity in _sequence(resolved.get("entities"))}
    counts: dict[str, int | None] = {}
    for fact in _sequence(resolved.get("role_axis_facts")):
        if (role := _text(fact.get("role"))) is not None:
            counts[role] = _count(fact.get("member_count"))
    scope = _mapping(resolved.get("scope"))
    if (axis := _text(scope.get("axis"))) is not None:
        members = scope.get("members")
        counts[axis] = len(members) if isinstance(members, list) else None
    return tuple(
        OpenEntityAxis(column=column, member_count=count)
        for column, count in sorted(counts.items())
        if column not in bound
    )


def _entity(entity: Mapping[str, object]) -> ResolvedEntity:
    role, ref = _text(entity.get("role")), _text(entity.get("ref"))
    if role is None or ref is None:
        raise ValueError(f"a resolved entity lacks its role or ref: {dict(entity)!r}")
    return ResolvedEntity(role=role, ref=ref, label=_text(entity.get("label")) or ref)


def _sources(blocks: Sequence[Mapping[str, object]]) -> tuple[SelectedSource, ...]:
    return tuple(source_from_spec_block(block) for block in blocks)


def _warnings(
    envelope: Mapping[str, object], part: Mapping[str, object]
) -> tuple[PartWarning, ...]:
    """The request-level warnings, then the part's own, in the order stated."""
    request_level = _sequence(_mapping(envelope.get("diagnostics")).get("warnings"))
    return tuple(
        _warning(warning) for warning in (*request_level, *_sequence(part.get("warnings")))
    )


def _warning(warning: Mapping[str, object]) -> PartWarning:
    # A field the producer left out reads as empty rather than failing the
    # envelope: the warning's presence is the signal, its wording is detail.
    return PartWarning(
        code=_text(warning.get("code")) or "",
        severity=_text(warning.get("severity")) or "",
        message=_text(warning.get("message")) or "",
        spec_id=_text(warning.get("spec_id")),
    )


def _flag(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def _window(window: Mapping[str, object]) -> Period | None:
    start, end = window.get("start_year"), window.get("end_year")
    if isinstance(start, int) and isinstance(end, int):
        return Period(start=start, end=end)
    return None


def _mapping(value: object) -> Mapping[str, object]:
    return cast(Mapping[str, object], value) if isinstance(value, Mapping) else {}


def _identity(value: object) -> dict[str, JsonValue] | None:
    """The envelope's identity document as the JSON object it is; `None` when there is none."""
    if not isinstance(value, Mapping):
        return None
    return cast(dict[str, JsonValue], dict(cast(Mapping[str, object], value)))


def _sequence(value: object) -> list[Mapping[str, object]]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        return []
    return [cast(Mapping[str, object], item) for item in value if isinstance(item, Mapping)]


def _sequence_of_any(value: object) -> list[object]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        return []
    return list(value)


def _strings(value: object) -> list[str]:
    return [item for item in _sequence_of_any(value) if isinstance(item, str)]


def _text(value: object) -> str | None:
    return value if isinstance(value, str) and value else None

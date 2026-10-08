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
from typing import cast

import rfc8785
from planner.records import Period, StatVarReading, StrictModel
from planner.workflow import (
    OpenAxis,
    OpenEntityAxis,
    ReadingSetBy,
    ResolvedEntity,
    ResolvedGroup,
    SelectedSource,
)
from pydantic import JsonValue

# The `coverage.kind` under which `search` reports a real series window.
EXACT_COVERAGE = "exact"

# The `shape.kind` of an identity that describes its StatVars by schema and
# slot bindings, where a pinned one names them.
DESCRIBED_SHAPE_KIND = "statvar_schema"


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
    coverage, the counts of its members and how many it serves, the label,
    size and digest of the binding its `shape` states, entities and the named
    sets whose members it binds, the entity
    axes it leaves open, unit (or that it varies per series), provenance, the
    axes its family varies along -- and the identity document of its
    `definition` sidecar when the envelope carries one. A `shape` without
    `member_count` or `member_digest` leaves them `None`, and a
    `served_count` Search states as null (membership not settled) leaves it
    `None`.
    `coverage.window` is a series window only when `coverage.kind` is
    `exact`; a `breadth` or `bare` coverage carries none.

    Raises:
        ValueError: If the block lacks a usable `spec_id`, or an entity it
            resolves lacks its role, value or label.
    """
    resolved = _mapping(block.get("resolved"))
    coverage = _mapping(resolved.get("coverage"))
    coverage_kind = _text(coverage.get("kind"))
    facets = _mapping(resolved.get("facets"))
    unit = _mapping(facets.get("unit"))
    provenance = _mapping(facets.get("provenance"))
    shape = _mapping(resolved.get("shape"))
    stat_vars = _mapping(resolved.get("stat_vars"))
    try:
        return SelectedSource(
            spec_id=str(block.get("spec_id")),
            sentence=_text(resolved.get("rendered_sentence")) or "",
            confidence=_text(resolved.get("confidence")),
            has_data=_flag(coverage.get("has_data")),
            measure_label=_text(shape.get("label")),
            stat_var_member_count=_count(stat_vars.get("member_count")),
            stat_var_member_count_exact=_flag(stat_vars.get("member_count_exact")),
            served_count=_count(stat_vars.get("served_count")),
            shape_member_count=_count(shape.get("member_count")),
            shape_member_digest=_text(shape.get("member_digest")),
            entities=tuple(_entity(entity) for entity in _sequence(resolved.get("entities"))),
            groups=_groups(_mapping(resolved.get("entity_groups"))),
            open_entity_axes=open_entity_axes_of(resolved),
            unit_ref=_text(unit.get("ref")),
            unit_label=_text(unit.get("label")),
            unit_per_series=True if unit.get("selection") == "per_series" else None,
            provenance_ref=_text(provenance.get("ref")),
            provenance_label=_text(provenance.get("label")),
            coverage_kind=coverage_kind,
            coverage_window=(
                _window(_mapping(coverage.get("window")))
                if coverage_kind == EXACT_COVERAGE
                else None
            ),
            identity=_identity(_mapping(block.get("definition")).get("identity")),
            open_axes=_open_axes(resolved),
            slot_codes=_slot_codes(resolved),
        )
    except ValueError as exc:
        # pydantic's ValidationError is a ValueError too, so one clause covers
        # a malformed field and a missing one alike.
        raise ValueError(f"the spec block is not usable as a source: {exc}") from exc


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


def _slot_codes(resolved: Mapping[str, object]) -> dict[str, tuple[str, ...] | None] | None:
    """The codes each slot of `resolved.slots` admits, by constraint property.

    A `value` binding admits its ref and every code its `expansion` added;
    Search states an expansion only when the walk added a code, so a value
    without one admits its ref alone. A `set` binding admits each of its
    values' refs and the codes its expansion added, and maps to `None` when
    it states no expansion, since nothing says its values admit no
    descendants and its codes are unknown. An unbound slot, and one with no
    constraint property (the place, time window or source), is left out.
    `None` when `resolved` lists no slots, or a slot carries a binding kind
    this reader does not know or a value without a ref.
    """
    slots = _sequence(resolved.get("slots"))
    if not slots:
        return None
    codes: dict[str, tuple[str, ...] | None] = {}
    for slot in slots:
        prop = _text(_mapping(_mapping(slot.get("key")).get("property")).get("dcid"))
        binding = _mapping(slot.get("binding"))
        if prop is None or binding.get("kind") == "unbound":
            continue
        match binding.get("kind"):
            case "value":
                values = [binding.get("value")]
            case "set":
                values = _sequence_of_any(binding.get("values"))
            case _:
                return None
        added = _sequence_of_any(_mapping(slot.get("expansion")).get("added"))
        refs = [_text(_mapping(_mapping(value).get("ref")).get("dcid")) for value in values]
        refs += [_text(_mapping(code).get("dcid")) for code in added]
        if not values or None in refs:
            return None
        if binding.get("kind") == "set" and not added:
            codes[prop] = None
            continue
        codes[prop] = tuple(sorted({ref for ref in refs if ref is not None}))
    return codes


# The dcid prefix of a country. A country holds no other country, so two
# country refs that differ name rows that never meet.
_COUNTRY_PREFIX = "country/"


def entity_scopes_disjoint(left: SelectedSource, right: SelectedSource) -> bool:
    """Whether the entities two sources bind keep their rows apart.

    They do when a role both sources bind names only countries on each side,
    with no country in common: ODA to Kenya and ODA to Uganda bind
    `destination` to two countries, so no row of one is a row of the other,
    whatever StatVars they share. A role only one side binds, one a side
    leaves open, and one bound to anything but countries (a region holds
    countries) proves nothing.
    """
    for role in {entity.role for entity in left.entities} & {
        entity.role for entity in right.entities
    }:
        refs = [
            {entity.ref for entity in source.entities if entity.role == role}
            for source in (left, right)
        ]
        countries = all(ref.startswith(_COUNTRY_PREFIX) for ref in refs[0] | refs[1])
        if countries and refs[0].isdisjoint(refs[1]):
            return True
    return False


class FamilyOverlap(StrEnum):
    """What two described StatVar families share, as their identities show it."""

    DISJOINT = "disjoint"
    NESTED = "nested"
    UNDECIDED = "undecided"


def family_overlap(left: SelectedSource, right: SelectedSource) -> FamilyOverlap:
    """Whether two sources' described identities can bind a StatVar in common.

    A StatVar has one population type, measured property, stat type,
    measurement qualifier and denominator, which a described identity's
    `shape` pins, and one code on each constraint property, which a member
    carries from the codes its family's slot admits (`slot_codes`). So two
    families are `DISJOINT` when a shape field differs as both state it, or
    when a property both bind to stated codes admits no code in common.
    Health and education aid bind `crsPurpose` to two subtrees of the code
    index that share no code.

    A shape field differs only when both sides state it as a dcid string, or
    both as `{"ref": dcid}`, and the dcids differ. A field one side leaves out
    proves nothing, since a sparse shape omits what a draft never filled. A
    denominator stated as `{"unresolved": true}` proves nothing either. One
    family is written that way while its denominator node is provisional, and
    as `{"ref": dcid}` once the graph confirms it.

    Two families are `NESTED` when their shapes state every field concretely
    (a dcid string or `{"ref": dcid}`) and identically, and one binds every
    property the other binds, each to codes the other admits. A family holds
    every member of a family narrower than it, as all sectors holds health.
    Two identities that each pin one StatVar (`pinned_stat_var`) are `NESTED`
    when they pin the same one and `DISJOINT` otherwise.

    Everything else is `UNDECIDED`. That covers a side with no described shape
    or no `slot_codes`, a property bound to codes not stated (a `None` entry)
    where the verdict would rest on it, two families that each bind a property
    the other leaves open, and two that would nest but whose shapes differ in
    a field one side leaves out, leaves unresolved, or spells another way,
    since the open fields may name different nodes.
    """
    pinned = [pinned_stat_var(source.identity) for source in (left, right)]
    if pinned[0] is not None and pinned[1] is not None:
        return FamilyOverlap.NESTED if pinned[0] == pinned[1] else FamilyOverlap.DISJOINT
    return _described_overlap(left, right)


def _described_overlap(left: SelectedSource, right: SelectedSource) -> FamilyOverlap:
    """`family_overlap` for two sources that do not both pin a StatVar."""
    left_shape, right_shape = _described_shape(left), _described_shape(right)
    if left_shape is None or right_shape is None:
        return FamilyOverlap.UNDECIDED
    if any(
        _states_apart(left_shape.get(field), right_shape.get(field))
        for field in left_shape.keys() & right_shape.keys()
    ):
        return FamilyOverlap.DISJOINT
    if left.slot_codes is None or right.slot_codes is None:
        return FamilyOverlap.UNDECIDED
    if any(
        codes is not None
        and (other := right.slot_codes.get(prop)) is not None
        and set(codes).isdisjoint(other)
        for prop, codes in left.slot_codes.items()
    ):
        return FamilyOverlap.DISJOINT
    nested = _holds(left.slot_codes, right.slot_codes) or _holds(right.slot_codes, left.slot_codes)
    if nested and _same_shape(left_shape, right_shape):
        return FamilyOverlap.NESTED
    return FamilyOverlap.UNDECIDED


def _same_shape(left: Mapping[str, object], right: Mapping[str, object]) -> bool:
    """Whether two shapes state every field, and state it identically.

    A field is stated as a dcid string or `{"ref": dcid}`. An absent field or
    an `{"unresolved": true}` denominator leaves the StatVars it admits open.
    """
    return left.keys() == right.keys() and all(
        _stated(value) and value == right[field] for field, value in left.items()
    )


def _stated(value: object) -> bool:
    """Whether a shape field's value names one dcid: a string or `{"ref": dcid}`."""
    return isinstance(value, str) or isinstance(_mapping(value).get("ref"), str)


def _states_apart(left: object, right: object) -> bool:
    """Whether two values of one shape field are both stated and differ.

    Both must be dcid strings, or both `{"ref": dcid}`. An `{"unresolved":
    true}` denominator, an absent field, or a value spelled differently on
    each side gives nothing to compare.
    """
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        left, right = _mapping(left).get("ref"), _mapping(right).get("ref")
    return isinstance(left, str) and isinstance(right, str) and left != right


def _described_shape(source: SelectedSource) -> Mapping[str, object] | None:
    """`source`'s identity `shape` when it describes a family by schema; `None` otherwise."""
    shape = _mapping((source.identity or {}).get("shape"))
    return shape if shape.get("kind") == DESCRIBED_SHAPE_KIND else None


def _holds(
    wider: Mapping[str, tuple[str, ...] | None], narrower: Mapping[str, tuple[str, ...] | None]
) -> bool:
    """Whether every property `wider` binds is bound in `narrower` to codes `wider` admits.

    A property either side binds to codes not stated proves nothing.
    """
    return all(
        codes is not None and (inner := narrower.get(prop)) is not None and set(inner) <= set(codes)
        for prop, codes in wider.items()
    )


def pinned_stat_var(identity: Mapping[str, object] | None) -> str | None:
    """The one StatVar dcid `identity` pins (`shape.kind` `stat_var`), or `None` for any other identity.

    A described identity (`statvar_schema`) names its members by schema and
    slots, so no dcid can be read from it.
    """
    shape = (identity or {}).get("shape")
    if not isinstance(shape, Mapping) or shape.get("kind") != "stat_var":
        return None
    dcid = shape.get("id")
    return dcid if isinstance(dcid, str) and dcid else None


def is_family(source: SelectedSource) -> bool:
    """Whether a retrieval of `source` holds a family of StatVars.

    A source is one StatVar exactly when `search` states it serves one
    (`served_count`), which it states only once the membership settled. Any
    other count, and a membership `search` did not settle, read as a family:
    the mint elects the members, and a family reading of one series serves
    the same frame.
    """
    return source.served_count != 1


def stat_var_count(source: SelectedSource) -> int:
    """How many StatVars `source` holds: `served_count` once settled, else the members `search` confirmed."""
    if source.served_count is not None:
        return source.served_count
    return source.stat_var_member_count or 0


def member_digest(dcids: Iterable[str]) -> str:
    """Lowercase SHA-256 hex of the RFC 8785 canonical JSON of the sorted `dcids`.

    The digest `search` states as `resolved.shape.member_digest` (dfo-apis
    `statvar_members.digest.member_digest`). Callers pass bare dcids, without
    the `dcid:` prefix.
    """
    return hashlib.sha256(rfc8785.dumps(sorted(dcids))).hexdigest()


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


def _groups(groups: Mapping[str, object]) -> tuple[ResolvedGroup, ...]:
    """The named sets `resolved.entity_groups` states, each keyed by the role column its members fill.

    A group stating no name names nothing a requirement can ask for, so it is
    left out rather than failing the block.
    """
    found = []
    for role, raw in groups.items():
        group = _mapping(raw)
        if (label := _text(group.get("label"))) is not None:
            found.append(ResolvedGroup(role=role, label=label, ref=_text(group.get("dcid"))))
    return tuple(found)


def block_groups(block: Mapping[str, object]) -> tuple[ResolvedGroup, ...]:
    """The named sets a spec block's `resolved.entity_groups` states (`SelectedSource.groups`)."""
    return _groups(_mapping(_mapping(block.get("resolved")).get("entity_groups")))


def wire_groups(groups: Sequence[ResolvedGroup]) -> dict[str, dict[str, str]]:
    """`groups` as `resolved.entity_groups` states them, keyed by role column,
    for a resubmit to post beside the identity they stand for (`_groups`)."""
    return {
        group.role: {"label": group.label, **({"dcid": group.ref} if group.ref else {})}
        for group in groups
    }


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

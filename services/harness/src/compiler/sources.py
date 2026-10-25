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
from common.place_type import FLOW_ROLE_COLUMNS, PartyClass
from planner.records import (
    AmbiguousSourceReading,
    Period,
    ResolvedSourceMention,
    StatVarReading,
    StrictModel,
)
from planner.workflow import (
    ElectedMeasure,
    MeasureAlternative,
    OpenAxis,
    OpenEntityAxis,
    ReadingSetBy,
    ResolvedEntity,
    ResolvedGroup,
    SelectedSource,
)
from pydantic import JsonValue, TypeAdapter

# The `coverage.kind` under which `search` reports a real series window.
EXACT_COVERAGE = "exact"

# The `resolved.facets.provenance.account` of a table of the flows a provider
# reports, as the OECD and ONE aid tables are.
PROVIDER_ACCOUNT = "provider"

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


# The `no_data` reasons (`NoDataReason` in apis' `search.models.vocab`)
# that refuse a `source_code` itself, whatever dataset is pinned to the
# requirement stating it.
SOURCE_CODE_REFUSALS = frozenset(
    {"source_code_shares_variable", "source_code_not_held", "source_code_ambiguous"}
)


class DiscoveryOutcome(StrictModel):
    """One `search` call's answer: its status, the sources it names, and why.

    `sources` holds one entry per spec block the part carries: the single
    reading of a `definite` answer, every candidate of a `candidates` one,
    and the nearest real series a `no_data` answer offers instead. `detail`
    is the envelope's own explanation for `no_data` and `unsupported`. `reason`
    is `no_data`'s own closed reason code (`NoDataReason` in apis'
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
    ambiguous_sources: tuple[AmbiguousSourceReading, ...] = ()


def resolved_source_mentions(part: Mapping[str, object]) -> tuple[ResolvedSourceMention, ...]:
    """The source constraints a part applied, excluding explicitly dropped wording."""
    dropped = [
        str(warning.get("message") or "").casefold()
        for warning in _sequence(part.get("warnings"))
        if warning.get("code") == "query_qualifier_dropped"
    ]
    mentions = (
        ResolvedSourceMention.model_validate(mention)
        for mention in _sequence(_mapping(part.get("restatement")).get("sources"))
    )
    return tuple(
        mention
        for mention in mentions
        if not any((mention.text or mention.ref or "").casefold() in message for message in dropped)
    )


def source_constraint_matches(
    mentions: Sequence[ResolvedSourceMention], *, text: str, ref: str | None, exclude: bool
) -> bool:
    """Whether Search confirmed this wording, polarity and optional chosen reference."""
    return any(
        " ".join((mention.text or "").casefold().split()) == " ".join(text.casefold().split())
        and (mention.polarity == "exclude") == exclude
        and (ref is None or mention.ref == ref)
        for mention in mentions
    )


def bound_refs(part: Mapping[str, object]) -> dict[str, tuple[str, ...]]:
    """The refs Search bound for each entity text of `part`, by the text case-folded.

    Search states, beside the part's `restatement`, one `entity_refs` row
    per entity text it restates: the text and the refs it finally bound.
    "G7" binds `GroupOfSeven`, whatever label Search gives the set. A text
    bound to nothing, being ambiguous, not held or not found, states no
    refs and is left out, and so is every text of a part without
    `entity_refs`, from a Search that predates the field. Such a name is
    matched by label or through the graph's resolve door. A text stated
    twice binds every ref either row states.
    """
    refs: dict[str, tuple[str, ...]] = {}
    for row in _sequence(part.get("entity_refs")):
        text, bound = _text(row.get("text")), _strings(row.get("refs"))
        if text is not None and bound:
            key = text.casefold()
            refs[key] = tuple(dict.fromkeys((*refs.get(key, ()), *bound)))
    return refs


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
    mentions = resolved_source_mentions(part)

    bound = bound_refs(part)

    def constrained(source: SelectedSource) -> SelectedSource:
        return source.model_copy(update={"resolved_sources": mentions, "bound_refs": bound})

    if status is DiscoveryStatus.DEFINITE:
        spec = _mapping(part.get("spec"))
        if not spec:
            raise ValueError("a definite search part names no spec")
        return DiscoveryOutcome(
            status=status, sources=(constrained(source_from_spec_block(spec)),), warnings=warnings
        )
    if status is DiscoveryStatus.CANDIDATES:
        blocks = _sequence(_mapping(part.get("candidates")).get("specs"))
        return DiscoveryOutcome(
            status=status,
            sources=tuple(constrained(s) for s in _sources(blocks)),
            warnings=warnings,
        )
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
            ambiguous_sources=tuple(
                AmbiguousSourceReading.model_validate(group)
                for group in _sequence(no_data.get("ambiguous_sources"))
            ),
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
    sets it binds, the measure it elected and the others it offers, the entity
    axes it leaves open, unit (or that it varies per series), provenance and
    its account, the publisher codes it is exactly (`exact_codes`), the
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
            elected_measure=_elected_measure(resolved),
            stat_var_member_count=_count(stat_vars.get("member_count")),
            stat_var_member_count_exact=_flag(stat_vars.get("member_count_exact")),
            served_count=_count(stat_vars.get("served_count")),
            shape_member_count=_count(shape.get("member_count")),
            shape_member_digest=_text(shape.get("member_digest")),
            entities=tuple(_entity(entity) for entity in _sequence(resolved.get("entities"))),
            groups=_groups(_mapping(resolved.get("entity_groups"))),
            bound_refs={
                text: tuple(refs)
                for text, raw in _mapping(block.get("_bound_refs")).items()
                if (refs := _strings(raw))
            },
            open_entity_axes=open_entity_axes_of(resolved),
            unit_ref=_text(unit.get("ref")),
            unit_label=_text(unit.get("label")),
            unit_per_series=True if unit.get("selection") == "per_series" else None,
            resolved_sources=tuple(
                ResolvedSourceMention.model_validate(mention)
                for mention in _sequence(block.get("_resolved_sources"))
            ),
            provenance_ref=_text(provenance.get("ref")),
            provenance_label=_text(provenance.get("label")),
            provenance_account=_text(provenance.get("account")),
            exact_codes=_exact_codes(stat_vars),
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


def _exact_codes(stat_vars: Mapping[str, object]) -> tuple[str, ...]:
    """The publisher codes `resolved.stat_vars.source_codes` states with `exact` true, stripped and upper-cased, in wire order.

    Only a JSON `true` counts: an entry without `exact`, or with any other
    value, proves nothing about the dataset.
    """
    codes = (
        _text(entry.get("code"))
        for entry in _sequence(stat_vars.get("source_codes"))
        if entry.get("exact") is True
    )
    return tuple(dict.fromkeys(code.strip().upper() for code in codes if code and code.strip()))


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
    carries a value apis could not decode and so could not sum.
    """
    inapplicable = set(_strings(resolved.get("inapplicable_slots")))
    unresolved = set(_strings(resolved.get("unresolved_slots")))
    axes: list[OpenAxis] = []
    for slot in _sequence(resolved.get("slots")):
        prop = _text(_mapping(_mapping(slot.get("key")).get("property")).get("ref"))
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
        prop = _text(_mapping(_mapping(slot.get("key")).get("property")).get("ref"))
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
        refs = [_text(_mapping(_mapping(value).get("ref")).get("ref")) for value in values]
        refs += [_text(_mapping(code).get("ref")) for code in added]
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
    index that share no code. They are `DISJOINT` too when one binds a
    property its own identity lists and the other's does not, since a member
    carries a code only on a property its family's schema lists. DAC2a aid
    binds `dac2aAidType`, which no IDS debt family lists, so no StatVar is
    both. A bound slot no identity lists, a where-classed constraint such as
    `comparisonRegion`, proves nothing (`_binds_unlisted`).

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
    the other leaves open or whose identity lists no slots, and two that would
    nest but whose shapes differ in a field one side leaves out, leaves
    unresolved, or spells another way, since the open fields may name
    different nodes.
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
    if (
        any(
            codes is not None
            and (other := right.slot_codes.get(prop)) is not None
            and set(codes).isdisjoint(other)
            for prop, codes in left.slot_codes.items()
        )
        or _binds_unlisted(left, right)
        or _binds_unlisted(right, left)
    ):
        return FamilyOverlap.DISJOINT
    nested = _holds(left.slot_codes, right.slot_codes) or _holds(right.slot_codes, left.slot_codes)
    if nested and _same_shape(left_shape, right_shape):
        return FamilyOverlap.NESTED
    return FamilyOverlap.UNDECIDED


def _binds_unlisted(source: SelectedSource, other: SelectedSource) -> bool:
    """Whether `source` binds a property both identities could list and `other`'s does not.

    An identity lists the what and how slots alone: apis leaves every when,
    where and source slot out, a constraint property classed `where` such as
    `comparisonRegion` included (`search.engine.spec_id._EXCLUDED_SLOT_AXES`).
    So only a property `source`'s own identity lists counts, and a slot
    missing from both identities never reads as a binding difference.
    Proves nothing when either identity states no slot list.
    """
    own, listed = _listed_properties(source), _listed_properties(other)
    if own is None or listed is None:
        return False
    return any(prop in own and prop not in listed for prop in source.slot_codes or {})


def bound_codes(source: SelectedSource, prop: str) -> tuple[str, ...]:
    """The codes `source`'s identity binds `prop` to, in identity order.

    A `value` binding names one code and a `set` binding each of its refs, once;
    an unbound property, one the identity does not list, and a source with
    no identity name none. These are the codes the question named, before
    the code index widened them to their descendants (`slot_codes`).
    """
    slots = (source.identity or {}).get("slots")
    for slot in slots if isinstance(slots, list) else ():
        if not isinstance(slot, Mapping) or slot.get("property") != prop:
            continue
        binding = _mapping(slot.get("binding"))
        match binding.get("state"):
            case "value":
                refs: Sequence[object] = [binding.get("ref")]
            case "set":
                refs = _sequence_of_any(binding.get("refs"))
            case _:
                refs = []
        return tuple(dict.fromkeys(ref for ref in refs if isinstance(ref, str) and ref))
    return ()


def _listed_properties(source: SelectedSource) -> set[object] | None:
    """The properties `source`'s identity lists a slot for, or `None` when it lists none."""
    slots = (source.identity or {}).get("slots")
    if not isinstance(slots, list):
        return None
    return {slot.get("property") for slot in slots if isinstance(slot, Mapping)}


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


def stat_var_count_is_floor(source: SelectedSource) -> bool:
    """Whether `stat_var_count(source)` is a floor: no served count, and `search` states its confirmed members as one."""
    return source.served_count is None and source.stat_var_member_count_exact is False


def member_digest(dcids: Iterable[str]) -> str:
    """Lowercase SHA-256 hex of the RFC 8785 canonical JSON of the sorted `dcids`.

    The digest `search` states as `resolved.shape.member_digest` (apis
    `statvar_members.digest.member_digest`). Callers pass bare dcids, without
    the `dcid:` prefix.
    """
    return hashlib.sha256(rfc8785.dumps(sorted(dcids))).hexdigest()


def default_reading(source: SelectedSource) -> StatVarReading:
    """How a family is read when the plan does not say: summed when every open axis is summable.

    A family with no open axis has nothing to sum over, so it reads
    separately; so does one with any axis apis may not sum over (a sex,
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
    one column, the fact's count wins on a flow side (`FLOW_ROLE_COLUMNS`),
    since `search` counts its parties there and leaves out the group totals
    the scope lists, which a flow retrieval does not return; on the place
    axis the scope's count wins, since it lists the members it counts. A
    column a `resolved.entities` entry binds is not open. Never
    raises: `harness.tool_contracts` reads the search envelope with it
    too, where a malformed field must not fail the digest.
    """
    bound = {_text(entity.get("role")) for entity in _sequence(resolved.get("entities"))}
    counts: dict[str, int | None] = {}
    for fact in _sequence(resolved.get("role_axis_facts")):
        if (role := _text(fact.get("role"))) is not None:
            counts[role] = _count(fact.get("member_count"))
    scope = _mapping(resolved.get("scope"))
    if (axis := _text(scope.get("axis"))) is not None and (
        axis not in FLOW_ROLE_COLUMNS or counts.get(axis) is None
    ):
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
    """The named sets `resolved.entity_groups` states, each keyed by the role column it fills.

    A group stating no name names nothing a requirement can ask for, so it is
    left out rather than failing the block. `total` is the published total
    party the role binds for the set, absent or `null` when it binds members.
    `whole_of` is the classes of party a whole-population total holds, or a
    union of such totals, absent for any other set. `side_classes` is the
    classes of party, totals aside, the dataset publishes on such a set's side,
    absent for any other set and from a Search that predates it. A class outside
    `common.place_type.PartyClass` fails the block.
    """
    found = []
    for role, raw in groups.items():
        group = _mapping(raw)
        if (label := _text(group.get("label"))) is not None:
            found.append(
                ResolvedGroup(
                    role=role,
                    label=label,
                    ref=_text(group.get("ref")),
                    total=_text(group.get("total")),
                    whole_of=_party_classes(group.get("whole_of")),
                    side_classes=_party_classes(group.get("side_classes")),
                )
            )
    return tuple(found)


def _elected_measure(resolved: Mapping[str, object]) -> ElectedMeasure | None:
    """The measure `resolved.measure_label` names, with each of `resolved.alternative_measures`.

    `None` when the block states no measure label. An alternative whose
    `definition` names no spec id, or that states no label or axis, offers
    nothing to switch to and is left out of the source's alternatives.
    """
    label = _text(resolved.get("measure_label"))
    if label is None:
        return None
    alternatives = []
    for raw in _sequence(resolved.get("alternative_measures")):
        spec_id = _text(_mapping(raw.get("definition")).get("spec_id"))
        alternative_label = _text(raw.get("label"))
        axis = _text(raw.get("axis"))
        if spec_id is not None and alternative_label is not None and axis is not None:
            alternatives.append(
                MeasureAlternative(
                    spec_id=spec_id,
                    label=alternative_label,
                    axis=axis,
                    identity=_identity(_mapping(raw.get("definition")).get("identity")),
                )
            )
    return ElectedMeasure(label=label, alternatives=tuple(alternatives))


def measure_named_by_query(block: Mapping[str, object]) -> bool:
    """Whether the query's own words chose the measure a spec block's search elected.

    They did when, for each other count of the flow the block offers
    (`ElectedMeasure.other_counts`, the options
    `submission.service.measure_choices` offers), every measure axis on
    which that count's `measure_axis_values` differ from the block's own
    has its `resolved.trace.slot_filters` entries all of provenance
    `text_recognized`. An axis both counts agree on does not matter, so
    "gross ODA" names gross over net though the engine elected the flow
    type. A block stating no measure, no other count or no axis values, an
    offered count stating no axis values, and a differing axis with no
    filter or another provenance (`measure_elected`, `declared_default`,
    `binder`) mean the words did not name it.
    """
    resolved = _mapping(block.get("resolved"))
    elected = _elected_measure(resolved)
    offered = {alternative.spec_id for alternative in elected.other_counts()} if elected else set()
    ours = _axis_values(resolved.get("measure_axis_values"))
    if not offered or not ours:
        return False
    provenances: dict[str, set[object]] = {}
    for entry in _sequence(_mapping(resolved.get("trace")).get("slot_filters")):
        axis = _text(_mapping(_mapping(entry.get("key")).get("property")).get("ref"))
        if axis is not None:
            provenances.setdefault(axis, set()).add(entry.get("provenance"))
    named = {axis for axis, found in provenances.items() if found == {_TEXT_RECOGNIZED}}
    for alternative in _sequence(resolved.get("alternative_measures")):
        if _text(_mapping(alternative.get("definition")).get("spec_id")) not in offered:
            continue
        theirs = _axis_values(alternative.get("measure_axis_values"))
        differing = {
            axis for axis in ours.keys() | theirs.keys() if ours.get(axis) != theirs.get(axis)
        }
        if not theirs or not differing <= named:
            return False
    return True


# The slot filter provenance of a value the query's own text named.
_TEXT_RECOGNIZED = "text_recognized"


def _axis_values(values: object) -> dict[str, str | None]:
    """A `measure_axis_values` list as the value dcid on each axis property."""
    return {
        axis: _text(_mapping(entry.get("value")).get("ref"))
        for entry in _sequence(values)
        if (axis := _text(entry.get("axis"))) is not None
    }


def block_groups(block: Mapping[str, object]) -> tuple[ResolvedGroup, ...]:
    """The named sets a spec block's `resolved.entity_groups` states (`SelectedSource.groups`)."""
    return _groups(_mapping(_mapping(block.get("resolved")).get("entity_groups")))


def alternative_definitions(block: Mapping[str, object]) -> dict[str, dict[str, JsonValue]]:
    """The identity of each measure a spec block's `resolved.alternative_measures` offers, by spec_id.

    Each is the block's own identity with its measure axes swapped, over the
    same places and years. An entry whose `definition` states no spec_id or
    identity is left out.
    """
    definitions: dict[str, dict[str, JsonValue]] = {}
    for alternative in _sequence(_mapping(block.get("resolved")).get("alternative_measures")):
        definition = _mapping(alternative.get("definition"))
        spec_id = _text(definition.get("spec_id"))
        identity = _identity(definition.get("identity"))
        if spec_id is not None and identity is not None:
            definitions[spec_id] = identity
    return definitions


# What a whole-population total holds, by flow side and class of party.
_WHOLE_OF_HOLDS: dict[tuple[str, PartyClass], str] = {
    ("origin", "official"): "all official donors",
    ("origin", "private"): "private donors",
    ("destination", "official"): "every recipient country",
    ("destination", "residual"): "aid the source allocates to no single country",
}


def _whole_scope(group: ResolvedGroup) -> str:
    """What `group`'s whole-population total holds, and on the donor side the private
    donors it leaves out when the source publishes some there (`side_classes`). A group
    without `side_classes` says nothing it leaves out, since nothing says what the side
    holds."""
    held = " and ".join(_WHOLE_OF_HOLDS.get((group.role, cls), cls) for cls in group.whole_of)
    if (
        group.role == "origin"
        and "private" in group.side_classes
        and "private" not in group.whole_of
    ):
        return f"{held} (private donors are outside it)"
    return held


def _whole_reading(group: ResolvedGroup, labels: Mapping[str, str]) -> str:
    """The sentence saying `group` is read as the source's total for a whole population."""
    scope = _whole_scope(group)
    if group.total is None:
        return (
            f"The figure for {group.label} adds up the totals the source publishes for "
            f"{scope}, not its members' figures."
        )
    party = labels.get(group.total, group.total)
    subject = (
        group.label if party.casefold() == group.label.casefold() else f"{group.label} are {party}"
    )
    return (
        f"The figures for {subject}, the total the source publishes for {scope}, "
        "not its members' figures added up."
    )


def total_reading(group: ResolvedGroup, labels: Mapping[str, str]) -> str | None:
    """The sentence saying `group` is read as its published total, or `None` when it binds members.

    `labels` maps a ref to its label, so the total party is named as the
    source names it ("Africa, Total") where that differs from the set's
    own name. A group with `whole_of` is read as the source's total for a
    whole population ("Official Donors, Total, the total the source publishes
    for all official donors (private donors are outside it)" where the source
    publishes private donors), or as the sum of such totals when it names no
    single party.
    """
    if group.whole_of:
        return _whole_reading(group, labels)
    if group.total is None:
        return None
    party = labels.get(group.total, group.total)
    named = "" if party.casefold() == group.label.casefold() else f" ({party})"
    return (
        f"The figures for {group.label} are the total the source publishes for the group "
        f"as a whole{named}, not its members' figures added up."
    )


class MembersAnswer(StrEnum):
    """How an answer reads a group bound as its members (`members_reading`).

    `EACH` keeps the column the group sits on, so each member keeps its own
    figure. `SUMMED` adds the members up into one figure, and `COMBINED`
    folds them into one by some other operation, or by one the plan does
    not show is a sum.
    """

    EACH = "each"
    SUMMED = "summed"
    COMBINED = "combined"


def members_reading(group: ResolvedGroup, *, answer: MembersAnswer | None) -> str | None:
    """The sentence saying an answer reads `group` as its members, or `None` when it binds the total.

    `answer` is how the answer reads the members, `None` when the column
    the group sits on is not known. An answer combining a flow side's
    members leaves out what the source allocates to the group and to no
    single member ("Africa, regional"), which only the group's total counts.
    """
    if group.reads_total:
        return None
    if answer is None:
        return (
            f"The figures for {group.label} are its members' figures, not a total the source "
            "publishes for the group."
        )
    if answer is MembersAnswer.EACH:
        return f"The figures for {group.label} are its members' figures, each on its own."
    left_out = (
        " and leaves out amounts the data allocates to no single country"
        if group.role in FLOW_ROLE_COLUMNS
        else ""
    )
    if answer is MembersAnswer.SUMMED:
        return f"The figure for {group.label} adds up its members' figures{left_out}."
    return f"The figure for {group.label} is computed from its members' figures{left_out}."


def wire_groups(groups: Sequence[ResolvedGroup]) -> dict[str, dict[str, str]]:
    """`groups` as `resolved.entity_groups` states them, keyed by role column,
    for a resubmit to post beside the identity they stand for (`_groups`).

    `total` is not posted. Search recomputes it from what the role binds,
    and a server that predates the field refuses it.
    """
    return {
        group.role: {"label": group.label, **({"ref": group.ref} if group.ref else {})}
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


_PARTY_CLASSES: TypeAdapter[tuple[PartyClass, ...]] = TypeAdapter(tuple[PartyClass, ...])


def _party_classes(value: object) -> tuple[PartyClass, ...]:
    """The classes of party `value` lists, none when it lists none.

    Raises:
        pydantic.ValidationError: When it lists a class outside `PartyClass`.
    """
    return _PARTY_CLASSES.validate_python(_strings(value))


def _text(value: object) -> str | None:
    return value if isinstance(value, str) and value else None

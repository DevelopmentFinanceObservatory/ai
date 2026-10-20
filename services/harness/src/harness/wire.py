"""The studio v1 wire contract: every frame and request that crosses HTTP.

These models are the enforced half of `docs/studio-contract-v1.md` -- the
prose doc explains the protocol; this module is what the server actually
validates and emits, and the two ship together. Deltas the repo doc
indexes that change a shape are marked on the field they touch.

Four request kinds arrive over POST, discriminated on `kind`
(`STUDIO_REQUEST_ADAPTER`): `turn.request`, `interrupt.response`,
`composition.mutate`, `turn.cancel`. Twelve frame types leave over SSE, discriminated on
`event` (`STUDIO_FRAME_ADAPTER`): message.delta, message.complete,
plan.proposed, plan.versioned, activity.step, component.created,
component.revised, composition.updated, elicitation.request, suggestions,
error, done.

Every model forbids unknown members, mirroring the schemas'
`additionalProperties: false`, so a frame the contract doesn't describe
fails loudly at the boundary. The deliberate openings are objects another
contract owns, referenced by name so the two cannot drift:
`ComponentCreated.payload` / `ComponentRevised.payload` is the platform's
Presentation contract, and `SeedSpec` reads the Search API's Spec block,
keeping the members a seed hands over and dropping the rest. `TurnRequest.seed_spec` holds its items
unparsed: each is read as a `Seed` on its own (`harness.seeds.seeds_slot`),
so one bad seed fails its own step rather than the request.

`conversation_id` is delta 5: the frontend client mints the conversation
identifier and sends `Authorization: Bearer <token>` alongside it. The
`session_id` is accepted as a deprecated parsing alias in v1
(the fixtures still use it); serialisation always writes
`conversation_id`. The bearer token itself travels in the header, never in
these bodies -- see `harness.auth`.
"""

from __future__ import annotations

import json
from enum import StrEnum
from typing import Annotated, Any, Literal, cast, get_args

from common.place_type import MemberClass
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    model_validator,
)


class _WireModel(BaseModel):
    """Base for every wire shape: unknown members are refused, mirroring the
    contract schemas' `additionalProperties: false`."""

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Shared shapes the studio frames use
# ---------------------------------------------------------------------------


class SpecRef(_WireModel):
    """The canonical query reference: `spec_id` names a definition and is
    stable for it; `content_version` pins the materialised numbers. Both are
    needed -- a reference pinned only by `spec_id` silently changes under a
    refresh."""

    spec_id: str
    content_version: str


class Source(_WireModel):
    """One data source. Licence lives per source because a mixed-source card
    carries mixed rights."""

    id: str
    name: str
    licence: str
    publisher: str | None = None
    url: str | None = None
    retrieved: str | None = None


class Freshness(_WireModel):
    data_through: str
    last_refreshed: str
    partial_period: bool
    next_expected: str | None = None


class WarningSeverity(StrEnum):
    INFO = "info"
    WARN = "warn"
    DEGRADED = "degraded"


class WireWarning(_WireModel):
    """A structured qualification passed through from a source, the analysis
    layer or the resolver. `part_index` or `spec_id` is set when it attaches
    to one conjunction part or one returned spec."""

    code: str
    severity: WarningSeverity
    message: str
    part_index: int | None = Field(default=None, ge=0)
    spec_id: str | None = None


class Provenance(_WireModel):
    """Mandatory on every card. Travels inside the payload so displaying it
    never needs a second fetch."""

    inputs: list[SpecRef] = Field(min_length=1)
    sources: list[Source] = Field(min_length=1)
    freshness: Freshness
    warnings: list[WireWarning]
    generated_at: str
    citation: str
    execution_ref: str | None = None
    refreshed_from: int | None = None


class Magnitude(StrEnum):
    UNITS = "units"
    THOUSANDS = "thousands"
    MILLIONS = "millions"
    BILLIONS = "billions"
    TRILLIONS = "trillions"


class PriceBasis(StrEnum):
    CURRENT = "current"
    CONSTANT = "constant"


# A unit, or one side of a composed unit, named in words: the unit's graph
# name, or its ref where the graph has none.
UnitName = Annotated[str, Field(min_length=1, max_length=200)]


class StructuredUnit(_WireModel):
    """A unit stated in parts, as a converted frame's unit is: the components
    catalogue's `structured_unit`. Every member is optional and at least one
    is present."""

    currency: str | None = Field(default=None, pattern=r"^[A-Z]{3}$")
    magnitude: Magnitude | None = None
    prices: PriceBasis | None = None
    base_year: int | None = None

    @model_validator(mode="after")
    def _states_a_member(self) -> StructuredUnit:
        if not self.model_fields_set:
            raise ValueError("a structured unit states at least one member")
        return self


class CountedStatVar(_WireModel):
    """One side of a composed unit whose measure declares no unit: the StatVar
    it counts, with the name to print for it."""

    stat_var: str = Field(pattern=r"^dcid:.", max_length=200)
    label: UnitName


type UnitPart = UnitName | StructuredUnit | CountedStatVar


class ComposedUnit(_WireModel):
    """The unit of a quotient of two measures, read `<numerator> per
    <denominator>`. A side is never null and never itself a composed unit."""

    numerator: UnitPart
    denominator: UnitPart


# The unit slot of a card's payload, as the platform's fact summary repeats
# it: a name, a structured unit, or a composed unit.
type Unit = UnitName | StructuredUnit | ComposedUnit


class Attribution(_WireModel):
    """The always-visible source line, e.g. "OECD DAC · 4 indicators".
    Rendered on every card regardless of source count."""

    label: str
    source_count: int = Field(ge=1)
    indicator_count: int = Field(ge=1)
    source_detail_url: str | None = None


class ErrorCode(StrEnum):
    """The platform's shared error grammar -- the codes the Presentation
    surface refuses with (verified live against the hosted dev deployment)
    -- plus `cancelled`, a studio-side code for a turn aborted by
    POST /api/studio/cancel."""

    SCHEMA_INVALID = "schema_invalid"
    BINDING_INVALID = "binding_invalid"
    CONSTRAINT_VIOLATION = "constraint_violation"
    UNSUPPORTED_COMPONENT_TYPE = "unsupported_component_type"
    UNSUPPORTED_VERSION = "unsupported_version"
    EDIT_CONFLICT = "edit_conflict"
    NOT_FOUND = "not_found"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    TRANSIENT_UNAVAILABLE = "transient_unavailable"
    RATE_LIMITED = "rate_limited"
    CANCELLED = "cancelled"
    CAPABILITY_UNAVAILABLE = "capability_unavailable"


class RetryMode(StrEnum):
    CORRECT_AND_RETRY = "correct_and_retry"
    REFRESH_THEN_RETRY = "refresh_then_retry"
    RETRY_AFTER_DELAY = "retry_after_delay"
    DO_NOT_RETRY = "do_not_retry"


class ToolError(_WireModel):
    """Machine behaviour keys on `error`/`retry`/`constraint_id`/`pointer`;
    `detail` and `fix` are display text."""

    error: ErrorCode
    retry: RetryMode
    detail: str
    constraint_id: str | None = None
    pointer: str | None = None
    fix: str | None = None
    retry_after_seconds: int | None = Field(default=None, ge=0)


class ComponentType(StrEnum):
    """Closed per catalogue release. A type the renderer does not know
    renders as the unsupported state for its own block."""

    STAT_CARD = "stat_card"
    LINE_CHART = "line_chart"
    BAR_CHART = "bar_chart"
    SCATTER_CHART = "scatter_chart"
    TABLE = "table"
    NOTE_USER = "note_user"
    NOTE_NARRATION = "note_narration"
    SOURCE_CARD = "source_card"


# ---------------------------------------------------------------------------
# fact_summary -- the bounded set of facts narration may cite
# ---------------------------------------------------------------------------


class Window(_WireModel):
    start: str
    end: str


class SeriesChange(_WireModel):
    from_period: str
    to_period: str
    absolute: float | None = None
    percent: float | None = None


class FactSeries(_WireModel):
    id: str
    label: str
    point_count: int = Field(ge=0)
    coordinates: dict[str, str] | None = None
    first: float | None = None
    first_period: str | None = None
    last: float | None = None
    last_period: str | None = None
    change: SeriesChange | None = None


class SeriesTruncated(_WireModel):
    showing: int = Field(ge=0)
    total: int = Field(ge=0)


class AppliedDefault(_WireModel):
    """A choice the platform made where the card's intent left one open: a
    bar chart that listed no values draws every value of its one varying
    dimension. `pointer` is the JSON Pointer to the intent field the default
    fills; `detail` says what was chosen and how to narrow it."""

    pointer: str
    detail: str


class FactSummary(_WireModel):
    """What the chat chip and the More Info panel render, and the only facts
    narration may cite. Closed on purpose: anything repeated on every
    response stacks near-identical copies in the model's context, only the
    newest of them true."""

    component_id: str
    revision: int = Field(ge=1)
    series: list[FactSeries]
    attribution: Attribution
    window: Window | None = None
    unit: Unit | None = None
    series_truncated: SeriesTruncated | None = None
    countries: list[str] | None = None
    defaults: list[AppliedDefault] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Composition -- the server-owned canvas state
# ---------------------------------------------------------------------------


class LayoutHintType(StrEnum):
    ROW = "row"
    COMPARISON = "comparison"


class LayoutHint(_WireModel):
    """Flat visual grouping over contiguous member blocks. `comparison`
    additionally aligns scales and units across members and is refused over
    members whose units are not comparable."""

    type: LayoutHintType
    members: list[str] = Field(min_length=2)


class CompositionActor(StrEnum):
    USER = "user"
    MODEL = "model"


class CardMeta(_WireModel):
    """Presentation-only card fields the platform intent cannot hold
    (Edit Card's subtitle/description/axis labels/style). Owned by the
    harness manifest; never sent to the platform."""

    subtitle: str | None = None
    description: str | None = None
    x_label: str | None = None
    y_label: str | None = None
    # Deliberately open (like the component payload): style is a
    # client-defined, unversioned extension bag the harness stores verbatim
    # and never interprets.
    style: dict[str, Any] | None = None


class CompositionBlock(_WireModel):
    component_id: str
    revision: int = Field(ge=1)
    type: ComponentType
    title: str
    pinned: bool | None = None
    meta: CardMeta | None = None


class CompositionCounts(_WireModel):
    """Server-computed so the All / Stats / Tables / Charts filter chips
    cannot disagree with the manifest they filter."""

    all: int = Field(ge=0)
    stats: int = Field(ge=0)
    charts: int = Field(ge=0)
    tables: int = Field(ge=0)
    notes: int = Field(ge=0)


class CompositionManifest(_WireModel):
    """The authoritative canvas state. The front end holds no layout state of
    its own; every structural change mints a revision stamped with its
    actor."""

    composition_id: str
    revision: int = Field(ge=1)
    actor: CompositionActor
    blocks: list[CompositionBlock]
    counts: CompositionCounts
    layout_hints: list[LayoutHint] | None = None


# ---------------------------------------------------------------------------
# Requests -- discriminated on `kind`
# ---------------------------------------------------------------------------

# Delta 5: the client mints `conversation_id`; `session_id` is
# accepted as a deprecated parsing alias in v1. Serialisation always
# writes `conversation_id`.
_CONVERSATION_ID = Field(validation_alias=AliasChoices("conversation_id", "session_id"))


class ComponentRefIntent(StrEnum):
    FOLLOW_UP = "follow_up"
    EDIT = "edit"


class ComponentRef(_WireModel):
    """Card context for "Ask follow up" and agent-driven "Edit Card", so the
    harness can scope the turn to one card."""

    component_id: str
    revision: int = Field(ge=1)
    intent: ComponentRefIntent


class Surface(StrEnum):
    """Where the turn is asked from. On the download surface the orchestrator
    fetches once and plans nothing (`harness.graph.state.DOWNLOAD_BUDGETS`)."""

    STUDIO = "studio"
    DOWNLOAD = "download"
    EMBED = "embed"


# A Search API resolution has at most six parts, and a Studio handoff seeds
# one per definite part.
MAX_SEEDS = 6

# The largest Spec block a seed may carry, in bytes of compact JSON. The dev
# Search API's blocks for CRS flows measure 300 to 420 KB, most of it the 200
# stat_var items the block lists and its trace; a population block, 42 KB.
MAX_SEED_SPEC_BYTES = 1_048_576

# The largest `/api/studio/turn` body the route reads (`harness.studio`'s
# `_BoundedTurnRoute`): six seeds at the cap, and room for the rest of the
# request.
MAX_TURN_REQUEST_BYTES = MAX_SEEDS * MAX_SEED_SPEC_BYTES + 65_536

# The bounds on what the harness reads of a seed and of Search's answer for
# it. A seed comes from a browser, and Search's answer for its identity goes
# on into session state, tool results and the planner's brief, so each member
# is held to what the Search API can serve. A seed past one is refused whole
# (`SeedSpec`, `Seed`), and so is an answer past one (`SpecBlock`).
#
# The operations one part carries, and the length of an operation's phrase.
# The Search API's request model takes at most eight operations on a part,
# each with a phrase of at most 200 characters, and echoes no more.
MAX_SEED_OPERATIONS = 8
MAX_OPERATION_PHRASE_CHARS = 200
# The sentence Search renders for a part. Search states no cap on it. The
# longest recorded is 324 characters, for a 40-character question; 2,000 is
# the most a whole query may hold (apis `search.engine.config`'s
# `max_query_chars`).
MAX_SEED_SENTENCE_CHARS = 2_000
# The entities a block resolves: an identity binds at most 1,024 refs under
# one role (apis `search.models.identity`), which a region's members can
# reach. The most recorded is 2.
MAX_SEED_ENTITIES = 1_024
# Any other string the harness reads: a dcid, a role, a label, a kind. The
# longest StatVar dcid recorded is 133 characters and Search's own longest
# text cap on a part is 200 (`_PARSED_MAX_VARIABLE_TEXT_CHARS`).
MAX_SEED_TEXT_CHARS = 256
# Any other list the harness reads: a block's slots (15 recorded), its
# coverage dimensions (3), its role facts (1), and an item's slot values (3)
# and constraint properties (2).
MAX_SEED_LIST_ITEMS = 64
# The roles an identity binds entities under: apis `search.models.identity`
# caps the `entities` mapping at 16 keys.
MAX_SEED_ROLES = 16
# A count or a year the harness prints: a whole number of at most 12 digits.
MAX_SEED_COUNT = 10**12


def _seed_text(path: str, value: object, cap: int = MAX_SEED_TEXT_CHARS) -> None:
    """Refuse a string member of a seed past `cap` characters; any other type is its reader's to judge."""
    if isinstance(value, str) and len(value) > cap:
        raise ValueError(f"{path} is {len(value):,} characters, past the cap of {cap:,}")


def _seed_items(path: str, value: object, cap: int = MAX_SEED_LIST_ITEMS) -> list[Any]:
    """The items of a list member of a seed, refused past `cap`; empty for any other type."""
    if not isinstance(value, list):
        return []
    if len(value) > cap:
        raise ValueError(f"{path} holds {len(value):,} items, past the cap of {cap:,}")
    return value


def _seed_count(path: str, value: object) -> None:
    """Refuse a whole-number member of a seed past `MAX_SEED_COUNT` either way."""
    if isinstance(value, int) and abs(value) > MAX_SEED_COUNT:
        raise ValueError(f"{path} is a number past the cap of {MAX_SEED_COUNT:,}")


def _seed_mapping(value: object) -> dict[str, Any]:
    return cast(dict[str, Any], value) if isinstance(value, dict) else {}


class SeedDefinition(_WireModel):
    """A Spec's canonical definition, `{schema_version, spec_id, identity}`,
    exactly as the Search API serialises it. `identity` must hash to
    `spec_id`; data-mcp checks that before it stores the definition
    (`data_mcp.definition_writes.caller_definition`)."""

    schema_version: Literal[1]
    spec_id: str = Field(max_length=MAX_SEED_TEXT_CHARS)
    identity: dict[str, Any]

    @model_validator(mode="after")
    def _what_the_harness_reads_is_bounded(self) -> SeedDefinition:
        """Hold the identity members a search reading states to their bounds.

        `harness.tool_contracts` reads `requested_window` and lists the
        `entities` roles and the `shape` kind. The identity as a whole is
        bounded where it is stored
        (`data_mcp.definition_writes.MAX_IDENTITY_BYTES`).
        """
        window = _seed_mapping(self.identity.get("requested_window"))
        _seed_text("definition.identity.requested_window.kind", window.get("kind"))
        for key in ("start_year", "end_year"):
            year = window.get(key)
            if year is not None and (isinstance(year, bool) or not isinstance(year, int)):
                raise ValueError(f"definition.identity.requested_window.{key} is not a year")
            _seed_count(f"definition.identity.requested_window.{key}", year)
        roles = _seed_mapping(self.identity.get("entities"))
        if len(roles) > MAX_SEED_ROLES:
            raise ValueError(
                f"definition.identity.entities names {len(roles):,} roles, "
                f"past the cap of {MAX_SEED_ROLES:,}"
            )
        for role in roles:
            _seed_text("definition.identity.entities", role)
        shape = _seed_mapping(self.identity.get("shape"))
        _seed_text("definition.identity.shape.kind", shape.get("kind"))
        return self


def check_resolved_bounds(resolved: dict[str, Any]) -> None:
    """Hold every member of a Spec block's `resolved` that the harness reads to its bound.

    The members are the ones `compiler.sources.source_from_spec_block`,
    `harness.tool_contracts.graph_envelope` and the search readings
    (`harness.turn_records.SearchReading`) take: the sentence, the entities,
    the unit and provenance, the coverage, the member counts, the slots, the
    open entity axes and the headline. A member of another type than Search
    serves is left to its reader, which refuses or ignores it.

    Raises:
        ValueError: Naming the first member past its bound.
    """
    _seed_text(
        "resolved.rendered_sentence", resolved.get("rendered_sentence"), MAX_SEED_SENTENCE_CHARS
    )
    _seed_text("resolved.confidence", resolved.get("confidence"))
    for index, entity in enumerate(
        _seed_items("resolved.entities", resolved.get("entities"), MAX_SEED_ENTITIES)
    ):
        for key, value in _seed_mapping(entity).items():
            _seed_text(f"resolved.entities[{index}].{key}", value)
    facets = _seed_mapping(resolved.get("facets"))
    for facet in ("unit", "provenance"):
        for key in ("ref", "label"):
            _seed_text(f"resolved.facets.{facet}.{key}", _seed_mapping(facets.get(facet)).get(key))
    coverage = _seed_mapping(resolved.get("coverage"))
    _seed_text("resolved.coverage.kind", coverage.get("kind"))
    for key, value in _seed_mapping(coverage.get("window")).items():
        _seed_count(f"resolved.coverage.window.{key}", value)
    for index, dimension in enumerate(
        _seed_items("resolved.coverage.dimensions", coverage.get("dimensions"))
    ):
        _seed_text(
            f"resolved.coverage.dimensions[{index}].label", _seed_mapping(dimension).get("label")
        )
        _seed_count(
            f"resolved.coverage.dimensions[{index}].count", _seed_mapping(dimension).get("count")
        )
    stat_vars = _seed_mapping(resolved.get("stat_vars"))
    for key in ("member_count", "served_count"):
        _seed_count(f"resolved.stat_vars.{key}", stat_vars.get(key))
    shape = _seed_mapping(resolved.get("shape"))
    _seed_text("resolved.shape.member_digest", shape.get("member_digest"))
    _seed_count("resolved.shape.member_count", shape.get("member_count"))
    for index, slot in enumerate(_seed_items("resolved.slots", resolved.get("slots"))):
        prop = _seed_mapping(_seed_mapping(_seed_mapping(slot).get("key")).get("property"))
        _seed_text(f"resolved.slots[{index}].key.property.ref", prop.get("ref"))
    for key in ("inapplicable_slots", "unresolved_slots"):
        for value in _seed_items(f"resolved.{key}", resolved.get(key)):
            _seed_text(f"resolved.{key}", value)
    for index, fact in enumerate(
        _seed_items("resolved.role_axis_facts", resolved.get("role_axis_facts"))
    ):
        _seed_text(f"resolved.role_axis_facts[{index}].role", _seed_mapping(fact).get("role"))
        _seed_count(
            f"resolved.role_axis_facts[{index}].member_count",
            _seed_mapping(fact).get("member_count"),
        )
    _seed_text("resolved.scope.axis", _seed_mapping(resolved.get("scope")).get("axis"))
    for key, value in _seed_mapping(resolved.get("headline")).items():
        _seed_count(f"resolved.headline.{key}", value)


def _check_names_its_spec(definition: SeedDefinition, spec_id: str) -> None:
    if definition.spec_id != spec_id:
        raise ValueError(f"the definition names {definition.spec_id!r}, the spec {spec_id!r}")


class SpecBlock(BaseModel):
    """One definite part's Spec block as the Search API serves it, open to
    the Spec's other members.

    `harness.graph`'s `seed` node reads Search's own answer for each seed's
    identity as one and stores it as a definite search part
    (`harness.graph.state.store_search`).
    Each member of `resolved` the harness reads is held to its own bound
    (`check_resolved_bounds`).
    """

    model_config = ConfigDict(extra="allow")

    spec_id: str = Field(max_length=MAX_SEED_TEXT_CHARS)
    definition: SeedDefinition
    resolved: dict[str, Any]

    @model_validator(mode="after")
    def _the_definition_names_this_spec(self) -> SpecBlock:
        _check_names_its_spec(self.definition, self.spec_id)
        return self

    @model_validator(mode="after")
    def _what_the_harness_reads_is_bounded(self) -> SpecBlock:
        check_resolved_bounds(self.resolved)
        return self


# The most code points a set's name may span, as apis bounds `EntityGroup.label`.
MAX_SEED_GROUP_LABEL_CHARS = 120


# The most classes a whole total's `whole_of` or its side's `side_classes` names: every
# `common.place_type.MemberClass`.
MAX_SEED_MEMBER_CLASSES = len(get_args(MemberClass))


class SeedEntityGroup(BaseModel):
    """A set a seed's places were asked for as (`resolved.entity_groups`), as Search served it.

    `total` is the published total party the role bound for the set, absent
    when it bound the members (`planner.workflow.ResolvedGroup.total`).
    `whole_of` names the classes of party that total, or the set of totals the
    role bound, holds in full, absent for any other set
    (`planner.workflow.ResolvedGroup.whole_of`). `side_classes` names the classes
    of party, totals aside, the dataset publishes on that set's side, absent for
    any other set and from a Search that predates it
    (`planner.workflow.ResolvedGroup.side_classes`). The
    members are apis' `EntityGroup` fields, and any other key is refused,
    so a block from an older Search wire fails rather than losing its `ref`.
    """

    model_config = ConfigDict(extra="forbid")

    label: str = Field(min_length=1, max_length=MAX_SEED_GROUP_LABEL_CHARS)
    ref: str | None = Field(default=None, max_length=MAX_SEED_TEXT_CHARS)
    entity_type: str | None = Field(
        default=None,
        max_length=MAX_SEED_TEXT_CHARS,
        exclude_if=lambda entity_type: entity_type is None,
    )
    total: str | None = Field(
        default=None, max_length=MAX_SEED_TEXT_CHARS, exclude_if=lambda total: total is None
    )
    whole_of: tuple[MemberClass, ...] = Field(
        default=(), max_length=MAX_SEED_MEMBER_CLASSES, exclude_if=lambda classes: not classes
    )
    side_classes: tuple[MemberClass, ...] = Field(
        default=(), max_length=MAX_SEED_MEMBER_CLASSES, exclude_if=lambda classes: not classes
    )


class SeedSpec(BaseModel):
    """What a seed hands over of one definite part's Spec block: the spec_id,
    the definition, and the sets its places were asked for as.

    The website posts the block as Search served it. `entity_groups` is read
    from its `resolved.entity_groups`, so the seed's resubmit posts the sets
    beside the identity and Search, which keeps only a set the graph
    confirms, names them again. Only a set on a role the identity binds, with
    a name Search would take, is kept; any other is dropped and the seed
    stands without it, as `compiler.sources` drops a nameless one. A set
    carrying a key `SeedEntityGroup` does not know refuses the seed. Every other member, the rest of `resolved`
    among them, is dropped as the block is read, so nothing
    in the harness reads it from the browser: `harness.graph`'s `seed` node
    takes it from Search's own answer for the identity (`SpecBlock`), and
    Search derives the population from the identity. A seed posted before
    Search stopped listing members, whose `resolved.stat_vars.items` names
    them, reads the same. The block is bounded at `MAX_SEED_SPEC_BYTES` as
    posted; the identity's own bound is data-mcp's
    (`data_mcp.definition_writes.MAX_IDENTITY_BYTES`).
    """

    model_config = ConfigDict(extra="ignore")

    spec_id: str = Field(max_length=MAX_SEED_TEXT_CHARS)
    definition: SeedDefinition
    entity_groups: dict[str, SeedEntityGroup] | None = Field(
        default=None, exclude_if=lambda groups: groups is None
    )

    @model_validator(mode="before")
    @classmethod
    def _groups_from_resolved(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        resolved = data.get("resolved")
        raw = data.get(
            "entity_groups", resolved.get("entity_groups") if isinstance(resolved, dict) else None
        )
        definition = data.get("definition")
        identity = definition.get("identity") if isinstance(definition, dict) else None
        entities = identity.get("entities") if isinstance(identity, dict) else None
        bound = set(entities) if isinstance(entities, dict) else set()
        groups: dict[str, SeedEntityGroup] = {}
        for role, group in (raw if isinstance(raw, dict) else {}).items():
            if role not in bound or len(groups) >= MAX_SEED_ROLES:
                continue
            try:
                groups[role] = SeedEntityGroup.model_validate(group)
            except ValidationError as exc:
                unknown = [e["loc"][0] for e in exc.errors() if e["type"] == "extra_forbidden"]
                if unknown:
                    raise ValueError(
                        f"resolved.entity_groups.{role} carries unknown keys {unknown}"
                    ) from exc
                continue
        return {**data, "entity_groups": groups or None}

    @model_validator(mode="before")
    @classmethod
    def _within_the_size_cap(cls, data: Any) -> Any:
        size = len(json.dumps(data, separators=(",", ":"), default=str).encode())
        if size > MAX_SEED_SPEC_BYTES:
            raise ValueError(
                f"the seed's spec block is {size:,} bytes, past the "
                f"{MAX_SEED_SPEC_BYTES:,}-byte cap on one seed"
            )
        return data

    @model_validator(mode="after")
    def _the_definition_names_this_spec(self) -> SeedSpec:
        _check_names_its_spec(self.definition, self.spec_id)
        return self


class OperationVerb(StrEnum):
    """The analysis a Search question asked for that Search itself does not do."""

    RANK = "rank"
    AGGREGATE = "aggregate"
    GROWTH = "growth"
    SHARE_OF = "share_of"
    EXPLAIN = "explain"
    FORECAST = "forecast"
    REPORT = "report"


class OperationAxis(StrEnum):
    """The dimension an operation runs over."""

    PLACE = "place"
    ORIGIN = "origin"
    DESTINATION = "destination"
    CATEGORY = "category"
    YEAR = "year"


# The bounds the Search API sets on a forecast's `target_year`.
MIN_TARGET_YEAR = 1000
MAX_TARGET_YEAR = 2200


class SeedOperation(_WireModel):
    """An operation on a seed's data, the Search API's `Operation`: `phrase`
    is the question's own words for it, `over` its axis when it has one, and
    `target_year` the year a forecast projects to when the question names one."""

    verb: OperationVerb
    phrase: str = Field(min_length=1, max_length=MAX_OPERATION_PHRASE_CHARS)
    over: OperationAxis | None = None
    target_year: int | None = Field(
        default=None, strict=True, ge=MIN_TARGET_YEAR, le=MAX_TARGET_YEAR
    )

    @model_validator(mode="after")
    def _a_target_year_is_a_forecast_s(self) -> SeedOperation:
        """Search sets `target_year` on a forecast alone, and refuses it on any other verb."""
        if self.target_year is not None and self.verb is not OperationVerb.FORECAST:
            raise ValueError("target_year is set only on a forecast")
        return self


class Seed(_WireModel):
    """A Search result handed to Studio: what the harness takes of the Spec
    the user saw (`SeedSpec`), and the operations the question asked of it."""

    spec: SeedSpec
    operations: list[SeedOperation] = Field(default_factory=list, max_length=MAX_SEED_OPERATIONS)


class TurnRequest(_WireModel):
    """POST /api/studio/turn -- start or continue a turn.

    `surface` is optional with default "studio" (delta 6: doc 1 requires it,
    doc 2's fixtures omit it; resolved in favour of the lenient parse).

    `seed_spec` carries the Search results a "... in Studio" action hands
    over, one per definite part. Each item is parsed as a `Seed` before the
    turn's session is created (`harness.seeds.seeds_slot`), and the session
    keeps the `Seed` alone, so one the harness cannot use fails its own step
    in the turn and leaves the request standing. A seeded request
    cannot also edit a card: the edit route runs no graph, so its seeds
    would go unread.

    `seed_spec` also takes one `{spec_id, content_version}` reference, the
    shape the member was reserved with. A reference carries no Spec block,
    so it seeds nothing: the turn runs from the question, as it did while
    the member was reserved, and the `seed` node reports the unused
    reference as a failed step.
    """

    kind: Literal["turn.request"]
    schema_version: Literal[1]
    conversation_id: str = _CONVERSATION_ID
    message: str
    surface: Surface = Surface.STUDIO
    component_ref: ComponentRef | None = None
    seed_spec: (
        Annotated[list[dict[str, Any]], Field(min_length=1, max_length=MAX_SEEDS)] | SpecRef | None
    ) = None
    # False: components are still created (and stream as component.created)
    # but are NOT appended to the composition; the client adds them later
    # with a composition.mutate "add" op (the Auto-add toggle).
    auto_add: bool = True

    @model_validator(mode="after")
    def _seeds_start_a_turn_not_an_edit(self) -> TurnRequest:
        if (
            isinstance(self.seed_spec, list)
            and self.component_ref is not None
            and self.component_ref.intent is ComponentRefIntent.EDIT
        ):
            raise ValueError("seed_spec starts a turn from Search and cannot edit a card")
        return self


class InterruptDecision(StrEnum):
    APPROVE = "approve"
    AMEND = "amend"
    REJECT = "reject"
    ANSWER = "answer"
    # Elicitation only: decline to answer; the run proceeds on its default
    # (the top-ranked candidate).
    SKIP = "skip"


class InterruptResponse(_WireModel):
    """POST /api/studio/interrupt -- resumes a paused turn.

    `interrupt_id` is echoed verbatim from the frame that paused the run; it
    is opaque (delta 7 -- the fixtures' literal `"approve"` is a sample
    value). `payload` carries the amendment text for
    `amend` and the selected option id for `answer`; absent for a plain
    approve or reject. The response to this POST is a NEW SSE stream
    continuing the same turn (delta 1).
    """

    kind: Literal["interrupt.response"]
    schema_version: Literal[1]
    conversation_id: str = _CONVERSATION_ID
    interrupt_id: str
    decision: InterruptDecision
    # Text only: the amendment (amend) or the answer (answer -- an option id,
    # or free text when the elicitation set allow_free_text). Absent for
    # approve / reject / skip.
    payload: str | None = None


class MutationOp(StrEnum):
    ADD = "add"
    REORDER = "reorder"
    REMOVE = "remove"
    PIN = "pin"
    UNPIN = "unpin"
    SET_LAYOUT_HINTS = "set_layout_hints"
    SET_CARD_META = "set_card_meta"


class AddOp(_WireModel):
    """Append a block for a component this conversation already streamed
    (delta 11); the server rebuilds revision/type/title from what it sent."""

    op: Literal[MutationOp.ADD]
    component_id: str


class ReorderOp(_WireModel):
    op: Literal[MutationOp.REORDER]
    order: list[str] = Field(min_length=1)


class RemoveOp(_WireModel):
    op: Literal[MutationOp.REMOVE]
    component_id: str


class PinOp(_WireModel):
    op: Literal[MutationOp.PIN]
    component_id: str


class UnpinOp(_WireModel):
    op: Literal[MutationOp.UNPIN]
    component_id: str


class SetLayoutHintsOp(_WireModel):
    """Replaces the manifest's layout hints; null clears them."""

    op: Literal[MutationOp.SET_LAYOUT_HINTS]
    layout_hints: list[LayoutHint] | None = None


class SetCardMetaOp(_WireModel):
    """Delta 13: presentation-only card fields the platform intent cannot
    hold; harness-owned, client-rendered."""

    op: Literal[MutationOp.SET_CARD_META]
    component_id: str
    meta: CardMeta


# Discriminated on `op`, so each operation carries exactly its own fields --
# an `add` without `component_id`, or a `pin` smuggling `order`, is refused
# at the boundary.
MutationOperation = Annotated[
    AddOp | ReorderOp | RemoveOp | PinOp | UnpinOp | SetLayoutHintsOp | SetCardMetaOp,
    Field(discriminator="op"),
]


class CompositionMutation(_WireModel):
    """POST /api/studio/composition -- Pin to Canvas, Reorder, Remove.
    Returns the new manifest. Removing a block does not delete the component.

    `conversation_id` is a delta-5 addition to the composition's own
    `composition_id`: every studio request names its conversation
    so the server can locate its session without a composition index.
    A stale `expected_revision` is refused with `edit_conflict` and nothing
    applies.
    """

    kind: Literal["composition.mutate"]
    schema_version: Literal[1]
    conversation_id: str = _CONVERSATION_ID
    # Optional (delta 11): omitted targets the conversation's current
    # composition -- and, for `add` ops, bootstraps an empty one when the
    # conversation has no canvas yet (auto_add=false's first component).
    composition_id: str | None = None
    operations: list[MutationOperation] = Field(min_length=1)
    expected_revision: int | None = Field(default=None, ge=1)


class CancelRequest(_WireModel):
    """POST /api/studio/cancel -- abort the conversation's in-flight
    segment (the Stop button). Idempotent: cancelling an idle conversation
    is a no-op 204. The aborted stream, if still open, ends with a
    stream-fatal `error` frame ("cancelled")."""

    kind: Literal["turn.cancel"]
    schema_version: Literal[1]
    conversation_id: str = _CONVERSATION_ID


StudioRequest = Annotated[
    TurnRequest | InterruptResponse | CompositionMutation | CancelRequest,
    Field(discriminator="kind"),
]

STUDIO_REQUEST_ADAPTER: TypeAdapter[StudioRequest] = TypeAdapter(StudioRequest)


# ---------------------------------------------------------------------------
# Frames -- discriminated on `event`
# ---------------------------------------------------------------------------


class MessageDelta(_WireModel):
    """Append to the streaming assistant message."""

    event: Literal["message.delta"]
    seq: int = Field(ge=0)
    text: str


class MessageComplete(_WireModel):
    """Final text for the turn's message."""

    event: Literal["message.complete"]
    seq: int = Field(ge=0)
    text: str


class PlanStep(_WireModel):
    """A human label -- the executable tool calls stay server-side."""

    id: str
    label: str
    description: str | None = None
    # Delta 17: the tool arguments backing this step, for a client that wants
    # to show more than the label. Optional; a step without it renders as it
    # did before v1.2.
    args: dict[str, Any] | None = None


class DecisionResolvedBy(StrEnum):
    DEFAULT = "default"
    USER = "user"


class PlanDecision(_WireModel):
    """One choice made on the user's behalf, with its alternatives and a
    one-line reason -- what makes the plan reviewable."""

    id: str
    question: str
    chosen: str
    reason: str
    resolved_by: DecisionResolvedBy
    alternatives: list[str] | None = None


class PlanAction(StrEnum):
    APPROVE = "approve"
    AMEND = "amend"
    REJECT = "reject"


class PlanRequirement(_WireModel):
    """One piece of evidence the plan needs, in words -- `sentence` is
    composed by `harness.plan_display`, never assembled client-side, so the
    plan card and the model's own narration say the same thing."""

    id: str
    kind: str  # "dataset" | "aggregate" | "growth" | ...
    sentence: str
    decision_ids: list[str] = []


class PlanOperation(_WireModel):
    """One computed step of the plan, in words."""

    id: str
    kind: str
    sentence: str
    requirement_id: str | None = None


class PlanAsk(_WireModel):
    """One output the user asked for and how the plan accounts for it:
    `answered_by` names the `PlanStep` ids whose results answer it, and
    `declined` says in words why the plan does not compute it. An ask
    answered in part carries both, with `not_computed` naming the part
    declined in the user's words."""

    ask: str
    answered_by: list[str] = []
    declined: str | None = None
    not_computed: str | None = None


class PlanProposed(_WireModel):
    """The plan awaiting approval. The run is genuinely paused: nothing is
    retrieved or computed until an interrupt.response arrives, and this
    frame is terminal for its SSE response (delta 1 -- resume arrives as a
    new stream from POST /api/studio/interrupt)."""

    event: Literal["plan.proposed"]
    seq: int = Field(ge=0)
    interrupt_id: str
    plan_version_id: str
    version: int = Field(ge=1)
    title: str
    steps: list[PlanStep]
    actions: list[PlanAction]
    decisions: list[PlanDecision] | None = None
    # Delta 17 (v1.2): additive only -- `steps` and `version` keep their
    # existing shape. `revision` is the plan's approval key, a string
    # distinct from the integer `version`; `requirements` and `operations`
    # are the evidence requirements and computed steps behind the plan.
    revision: str | None = None
    requirements: list[PlanRequirement] | None = None
    operations: list[PlanOperation] | None = None
    # Delta 20 (v1.5): the outputs the user asked for, in the order the
    # brief lists them; absent when the plan was given none.
    asks: list[PlanAsk] | None = None


class PlanVersioned(_WireModel):
    """A new plan version replaces the card `supersedes` names. Re-render the
    plan card from the fresh plan.proposed that follows.

    An amendment raises `version`. A plan drafted again because the one
    awaiting approval could not be read carries `version` 1, which can be
    lower than the version it supersedes.
    """

    event: Literal["plan.versioned"]
    seq: int = Field(ge=0)
    plan_version_id: str
    version: int = Field(ge=1)
    supersedes: str
    changed: list[str] | None = None


class StepStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    SKIPPED = "skipped"


class StepProduced(_WireModel):
    component_id: str
    revision: int = Field(ge=1)


class StepProgress(_WireModel):
    completed: int = Field(ge=0)
    total: int = Field(ge=1)


class ActivityStep(_WireModel):
    """One execution step -- the only progress channel (Activity tab and the
    "2/3 complete" line). A step-scoped failure surfaces here as
    `status: "failed"`, with a `detail` of one sentence for the user: the
    one `harness.sentences` has for the failure's code. The refusal's own
    text is written for the agent and stays in the trace. The run continues
    where the graph allows (delta 4); a stream-fatal problem is an `error`
    frame instead, whose `detail` is such a sentence too."""

    event: Literal["activity.step"]
    seq: int = Field(ge=0)
    step_id: str
    label: str
    status: StepStatus
    detail: str | None = None
    tool: str | None = None
    duration_ms: int | None = None
    produced: StepProduced | None = None
    progress: StepProgress | None = None


class CreatedFrom(_WireModel):
    """The step and message that produced this card, for Activity-trace
    linking and "Ask follow up"."""

    step_id: str
    message_seq: int | None = Field(default=None, ge=0)


class ComponentCreated(_WireModel):
    """A component was created. `payload` is the platform's resolved
    Presentation revision, deliberately opaque here -- pass it to the
    renderer untouched."""

    event: Literal["component.created"]
    seq: int = Field(ge=0)
    component_id: str
    revision: int = Field(ge=1)
    type: ComponentType
    catalog_version: str
    min_catalog_version: str
    payload: dict[str, Any]
    fact_summary: FactSummary
    provenance: Provenance
    created_from: CreatedFrom | None = None


def usable_component(component: dict[str, Any] | None) -> bool:
    """Whether `component` (a `write_component` result's own `component`
    member) would satisfy the `component.created` wire contract above:
    the full model, not just a `fact_summary` presence check. `make_charts`
    (`harness.orchestrator_tools`) calls this to decide which writes land;
    `harness.studio_translate`'s `_component_frames` validates the same
    contract again at drain time, immediately before a write actually lands.

    `event`, `seq` and `created_from` are not the writer's to give: `event`
    and `seq` are wire bookkeeping no caller of this function has a real
    value for yet, and the real `created_from` (the step that produced the
    component) is only known at drain time. Stand-ins fill the three
    fields here so validation can run on everything else the contract
    actually requires.
    """
    if component is None:
        return False
    try:
        ComponentCreated.model_validate(
            {
                **component,
                "event": "component.created",
                "seq": 0,
                "created_from": component.get("created_from") or {"step_id": "_"},
            }
        )
    except ValidationError:
        return False
    return True


class ComponentRevised(_WireModel):
    """Same component_id, incremented revision. `payload` may be omitted when
    only presentation fields changed -- apply `changed` against the revision
    you hold."""

    event: Literal["component.revised"]
    seq: int = Field(ge=0)
    component_id: str
    revision: int = Field(ge=2)
    changed: list[str] = Field(min_length=1)
    fact_summary: FactSummary
    payload: dict[str, Any] | None = None


class CompositionUpdated(_WireModel):
    event: Literal["composition.updated"]
    seq: int = Field(ge=0)
    manifest: CompositionManifest


class ElicitationInput(StrEnum):
    SINGLE_SELECT = "single_select"
    MULTI_SELECT = "multi_select"
    TEXT = "text"


class ElicitationOption(_WireModel):
    id: str
    label: str
    description: str | None = None
    # Set on at most one option: the candidate the planner's ask favours,
    # when it favours one. `rationale`, on the frame rather than repeated per
    # option, is the planner's reason.
    recommended: bool = False


class ElicitationRequest(_WireModel):
    """A clarifying question. Pauses the run exactly as plan.proposed does
    (terminal frame, delta 1); answer with `decision: "answer"` and the
    chosen option id as `payload`."""

    event: Literal["elicitation.request"]
    seq: int = Field(ge=0)
    interrupt_id: str
    question: str
    input: ElicitationInput
    options: list[ElicitationOption] | None = None
    # The "More Details Question" affordances: an open-ended answer
    # alongside options, and a Skip button.
    allow_free_text: bool = False
    skippable: bool = False
    # Why the `recommended` option (if any) is recommended, as the planner
    # wrote it for the person settling the choice. `None` when no option is
    # recommended.
    rationale: str | None = None


class SuggestionItem(_WireModel):
    label: str
    prompt: str


class Suggestions(_WireModel):
    """Follow-up chips. May be empty in v1 (delta 8 -- the frame is
    additive)."""

    event: Literal["suggestions"]
    seq: int = Field(ge=0)
    items: list[SuggestionItem]


class ErrorFrame(_WireModel):
    """Stream-fatal (delta 4): `done` does not follow this frame. A
    recoverable, step-scoped failure is an activity.step with
    `status: "failed"` instead."""

    event: Literal["error"]
    seq: int = Field(ge=0)
    error: ToolError


class Done(_WireModel):
    """Turn complete. `composition` is the conversation's current manifest,
    absent when the conversation has no canvas yet. Emitted exactly once per
    turn, on the segment that finishes it -- an interrupted segment ends
    with its pausing frame instead."""

    event: Literal["done"]
    seq: int = Field(ge=0)
    trace_id: str
    composition: CompositionManifest | None = None


StudioFrame = Annotated[
    MessageDelta
    | MessageComplete
    | PlanProposed
    | PlanVersioned
    | ActivityStep
    | ComponentCreated
    | ComponentRevised
    | CompositionUpdated
    | ElicitationRequest
    | Suggestions
    | ErrorFrame
    | Done,
    Field(discriminator="event"),
]

STUDIO_FRAME_ADAPTER: TypeAdapter[StudioFrame] = TypeAdapter(StudioFrame)

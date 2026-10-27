"""The `search` tool: data-mcp's wrapper over the Search API's parsed door.

A conjunction arrives already split into `parts`, one per variable asked
for. The door's field names are close to the tool's without matching them
(`variable`/`time`/`unit`/`role` versus the wire's
`variable_text`/`time_text`/`unit_text`/`role_text`, and a source's `exclude`
flag versus the wire's `polarity`), so `to_wire_request` translates. A success
result puts the digest in `content`, a small model-facing answer in
`structuredContent`, and the upstream envelope in `_meta` (see
`data_mcp.search_result`). A caller in a conversation sends a `SearchContext`
in the request's `_meta`, which a parsed request carries to Search when
`Settings.search_sends_conversation_context` is on. Rendering the digest and
the answer lives in `data_mcp.digest`; the shared upstream client lives in
`data_mcp.client_holder`.
"""

from __future__ import annotations

import logging
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Annotated, Any, Literal, Self

from fastmcp.server.dependencies import fastmcp_request_ctx
from fastmcp.tools import ToolResult
from opentelemetry import trace
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from data_mcp.client_holder import get_client
from data_mcp.deadline import CallDeadline
from data_mcp.definition_store import IdentityMismatch, StoredDefinition
from data_mcp.definition_writes import put_definitions, stored_definition
from data_mcp.digest import render_answer, render_digest
from data_mcp.errors import UpstreamError
from data_mcp.scope import MAX_USER_MESSAGE_CHARS, ScopeEntry
from data_mcp.search_result import SEARCH_CONTEXT_META_KEY, SEARCH_ENVELOPE_META_KEY
from data_mcp.settings import Settings
from data_mcp.tool_errors import ToolRefusal, from_upstream_error, to_tool_result

logger = logging.getLogger(__name__)

# The deployment's published parsed-door limits.
MAX_PARTS = 6
_MAX_PART_ENTITIES = 16
_MAX_VARIABLE_CHARS = 200
_MAX_ENTITY_TEXT_CHARS = 128
_MAX_ROLE_CHARS = 64
_MAX_TIME_CHARS = 100
_MAX_UNIT_CHARS = 100
_MAX_SOURCE_CODE_CHARS = 64
_MAX_PART_SOURCES = 8
_MAX_SOURCE_TEXT_CHARS = 120
_MAX_SOURCE_REF_CHARS = 200
_MAX_RAW_QUERY_CHARS = 2000
_MAX_CARRIED_ENTRIES = 16
_MAX_PART_OPERATIONS = 8
_MAX_OPERATION_PHRASE_CHARS = 200
_MAX_FORECAST_TARGET_YEAR = 2200

# Control characters (newlines among them) and the Unicode line and paragraph
# separators, which the door refuses in an operation's phrase.
_LINE_BREAKING_CATEGORIES = frozenset({"Cc", "Zl", "Zp"})

# The wall-clock budget for one call to the Search API's resolve route, retries
# included. It sits above `Settings.upstream_read_timeout_s` (50 s, itself above
# apis' 45 s search deadline) and below the harness's 60 s `search` call
# timeout (`compiler.capabilities`), so a slow search ends here with apis' own
# refusal or a typed transport refusal before the harness gives up on it.
SEARCH_CALL_BUDGET_S = 55.0

# The time phrases the parsed door binds. A phrase outside these draws a
# `time_text_not_parsed` warning, so the `time` description lists them.
TIME_FORMS = (
    "YYYY",
    "YYYY to YYYY",
    "since YYYY",
    "between YYYY and the latest",
    "latest",
)


class SearchEntity(BaseModel):
    """One entity mentioned in a part, with an optional relational phrase.

    `name` or `dcid` says which entity `text` means, so two entities whose
    words are alike ("aid from Congo to Congo") stay two rows. Search refuses
    a row carrying both with `request_invalid`, so the model refuses it first.
    """

    model_config = ConfigDict(extra="forbid")

    text: str = Field(
        min_length=1,
        max_length=_MAX_ENTITY_TEXT_CHARS,
        description="the entity's text as spoken, e.g. 'France'",
    )
    role: str | None = Field(
        default=None,
        min_length=1,
        max_length=_MAX_ROLE_CHARS,
        description=("the relational phrase, matched against a fixed lexicon, e.g. 'from', 'to'"),
    )
    members: bool | None = Field(
        default=None,
        description=(
            "true when `text` names a set of places (a continent, a subregion, or a group "
            "such as the European Union) and the ask is for each member on its own: "
            "'African countries' is {text: 'Africa', members: true}. Leave unset for the "
            "set as one place: 'Africa' is {text: 'Africa'}"
        ),
    )
    name: str | None = Field(
        default=None,
        min_length=1,
        max_length=_MAX_ENTITY_TEXT_CHARS,
        description=(
            "the entity's full name, when `text` could name more than one entity: "
            "'aid from Congo to Congo' sends {text: 'Congo', name: 'Republic of the "
            "Congo'}. Leave unset beside `dcid`"
        ),
    )
    dcid: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "the graph dcid of the entity `text` names, e.g. 'country/COG', as a "
            "search refusal's candidate lists it. Leave unset beside `name`"
        ),
    )

    @model_validator(mode="after")
    def _name_or_dcid(self) -> Self:
        if self.name is not None and self.dcid is not None:
            raise ValueError("set name or dcid, not both: a dcid already names the entity")
        return self


class SearchOperation(BaseModel):
    """One computation the question asks for over a part's data. Search does not
    perform it; it opens the axis the operation runs over, so the part returns
    every value the operation needs."""

    model_config = ConfigDict(extra="forbid")

    verb: Literal["rank", "aggregate", "growth", "share_of", "explain", "forecast", "report"] = (
        Field(description="what the question asks to compute")
    )
    phrase: str = Field(
        min_length=1,
        max_length=_MAX_OPERATION_PHRASE_CHARS,
        description="the user's wording for it, on one line, e.g. 'top 5 creditors'",
    )
    over: Literal["place", "origin", "destination", "category", "year"] | None = Field(
        default=None,
        description=(
            "the axis it runs across: 'origin' for givers (donors, creditors), 'destination' "
            "for receivers, 'place' for countries, 'category' for a breakdown such as "
            "sectors, 'year' for time; leave it out when the verb names none"
        ),
    )
    target_year: int | None = Field(
        default=None,
        ge=1000,
        le=_MAX_FORECAST_TARGET_YEAR,
        description="the year a forecast projects to; set only on a forecast",
    )

    @field_validator("phrase")
    @classmethod
    def _phrase_is_one_line(cls, phrase: str) -> str:
        if any(unicodedata.category(char) in _LINE_BREAKING_CATEGORIES for char in phrase):
            raise ValueError(
                "phrase must not contain control characters, newlines or line separators"
            )
        if not phrase.strip():
            raise ValueError("phrase must not be blank")
        return phrase

    @model_validator(mode="after")
    def _target_year_only_on_a_forecast(self) -> Self:
        if self.target_year is not None and self.verb != "forecast":
            raise ValueError("target_year is set only on a forecast")
        return self


class SearchSource(BaseModel):
    """One publisher or dataset the user named for a part, to read from or to leave out."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(
        min_length=1,
        max_length=_MAX_SOURCE_TEXT_CHARS,
        description="the source's name as the user said it, e.g. 'IDS' or 'the OECD'",
    )
    ref: str | None = Field(
        default=None,
        min_length=1,
        max_length=_MAX_SOURCE_REF_CHARS,
        description=(
            "the reference of a source candidate the user chose from a source_ambiguous "
            "response; keep its original text and exclude flag, and set this to select "
            "that exact candidate instead of matching the text again"
        ),
    )
    exclude: bool = Field(
        default=False,
        description=(
            "true when the user asks to leave this source out, as in 'not from CRS' or "
            "'excluding IMF data'"
        ),
    )


class SearchPart(BaseModel):
    """One variable ask within the request, with its own entities, time, unit, and sources."""

    model_config = ConfigDict(extra="forbid")

    variable: str = Field(
        min_length=1,
        max_length=_MAX_VARIABLE_CHARS,
        description="what is measured",
    )
    entities: list[SearchEntity] | None = Field(
        default=None,
        max_length=_MAX_PART_ENTITIES,
        description="who or where, with the relational phrase as spoken",
    )
    time: str | None = Field(
        default=None,
        min_length=1,
        max_length=_MAX_TIME_CHARS,
        description=(
            "the time phrase, in one of these forms only: "
            + ", ".join(f"'{form}'" for form in TIME_FORMS)
        ),
    )
    unit: str | None = Field(
        default=None,
        min_length=1,
        max_length=_MAX_UNIT_CHARS,
        description=(
            "unit and price basis only, e.g. 'constant 2015 US dollars'; a flow's counting "
            "basis goes in `variable` only if the user named it, else search elects it"
        ),
    )
    open_roles: list[Annotated[str, Field(min_length=1, max_length=_MAX_ROLE_CHARS)]] | None = (
        Field(
            default=None,
            min_length=1,
            max_length=2,
            description=(
                "a flow side the ask lists member by member, as a role word, e.g. ['creditor'] "
                "for 'which creditors' or 'by creditor'; use only for an origin or destination "
                "side of a flow, never a side one of the entities names. A total across a side "
                "sends none, runs no operation over it, and names the population the user asked "
                "for as an entity: 'all donors' covers official and private donors, and a donor "
                "side left unnamed on DAC2a covers official donors only. For a named population, "
                "search binds the source's published total. Where the source publishes none, "
                "search departs `whole_total_unpublished` and returns the members, whose sum is "
                "the total. Where the source holds no series of that total, such as 'official "
                "donors' on the CRS, search finds no data"
            ),
        )
    )
    operations: list[SearchOperation] | None = Field(
        default=None,
        min_length=1,
        max_length=_MAX_PART_OPERATIONS,
        description=(
            "what the question asks to compute over this part's data, in the user's order, "
            "e.g. a rank for 'top 5'; a rank over place, category or year has no flow role, "
            "while origin is a giving side and destination is a receiving side; leave it out "
            "when the question asks for none"
        ),
    )
    sources: list[SearchSource] | None = Field(
        default=None,
        min_length=1,
        max_length=_MAX_PART_SOURCES,
        description=(
            "the publishers or datasets the user names for this part, in their own "
            "words, each with `exclude` when the user asks to leave it out; a lender, "
            "donor or recipient the flow runs from or to is an entity, not a source"
        ),
    )
    source_code: str | None = Field(
        default=None,
        min_length=1,
        max_length=_MAX_SOURCE_CODE_CHARS,
        description=(
            "a publisher's own indicator code, exactly as the user or a skill wrote it, "
            "e.g. 'DT.TDS.DEGG.CD'. It names a family of variables, so keep the rest of "
            "the ask (places, time, `open_roles`) in the other fields. `variable` holds "
            "the user's words for the measure, or the code itself when the user gives none"
        ),
    )
    separate: bool | None = Field(
        default=None,
        description=(
            "true on each part of a comparison sent as several parts, such as 'health aid' "
            "and 'education aid' for 'health versus education aid', so search answers each "
            "part on its own; leave unset otherwise, since search may merge parts that "
            "differ only in one category or place into one answer"
        ),
    )


class SearchContext(BaseModel):
    """The conversation a search is made in, sent by the harness in the request's
    `_meta` under `SEARCH_CONTEXT_META_KEY`.

    `user_message` is the user's own words for the turn, amendments and
    answers included, from which Search reads intent and explicit scope.
    `carried_entries` are the scope entries the conversation settled on
    earlier parts, oldest first, each with provenance `carried`, in Search's
    `Entry` shape (`data_mcp.scope.ScopeEntry`). Search attaches each to a
    part by axis and concept.
    """

    model_config = ConfigDict(extra="forbid")

    user_message: str | None = Field(default=None, min_length=1)
    carried_entries: list[ScopeEntry] = Field(default_factory=list)


def to_wire_request(
    parts: Sequence[SearchPart] | None,
    raw_query: str | None,
    *,
    context: SearchContext | None = None,
) -> dict[str, Any]:
    """Translate tool-facing parts into the parsed door's wire request body, or,
    with no parts, `raw_query` into the raw-text door's, where Search reads the
    question itself.

    A parsed body carries `context`'s user message and carried entries as
    the input's `user_message` and `carried_entries`, each when set, within
    the door's limits: the message's last 4,000 characters, since a turn's
    amendments and answers are appended to it, and the 16 newest entries,
    the list being oldest first. The raw-text door takes neither.
    """
    input_block: dict[str, Any]
    if parts:
        input_block = {"kind": "parsed", "parts": [_wire_part(part) for part in parts]}
        if raw_query is not None:
            input_block["raw_query"] = raw_query
        if context is not None and context.user_message is not None:
            input_block["user_message"] = context.user_message[-MAX_USER_MESSAGE_CHARS:]
        if context is not None and context.carried_entries:
            input_block["carried_entries"] = [
                entry.model_dump(mode="json", exclude_none=True)
                for entry in context.carried_entries[-_MAX_CARRIED_ENTRIES:]
            ]
    else:
        if raw_query is None:
            raise ValueError("a search with no parts needs raw_query, the question Search reads")
        input_block = {"kind": "raw_text", "query": raw_query}
    return {
        "schema_version": 1,
        "input": input_block,
        # `include_observations` is left unset because its default is already off.
        # `facet_options` arrives on every definite spec without a flag.
        # `include_skills` names the skills each result calls for, which the
        # digest lists for `get_skill`.
        "response_options": {"include_skills": True},
    }


def _wire_part(part: SearchPart) -> dict[str, Any]:
    wire: dict[str, Any] = {"variable_text": part.variable}
    if part.entities is not None:
        wire["entities"] = [_wire_entity(entity) for entity in part.entities]
    if part.time is not None:
        wire["time_text"] = part.time
    if part.unit is not None:
        wire["unit_text"] = part.unit
    if part.open_roles is not None:
        wire["open_roles"] = part.open_roles
    if part.operations is not None:
        wire["operations"] = [
            operation.model_dump(exclude_none=True) for operation in part.operations
        ]
    if part.sources is not None:
        wire["sources"] = [
            {
                "text": source.text,
                "polarity": "exclude" if source.exclude else "include",
                **({"ref": source.ref} if source.ref is not None else {}),
            }
            for source in part.sources
        ]
    if part.source_code is not None:
        wire["source_code"] = part.source_code
    if part.separate:
        wire["separate"] = True
    return wire


def _wire_entity(entity: SearchEntity) -> dict[str, Any]:
    wire: dict[str, Any] = {"text": entity.text}
    if entity.role is not None:
        wire["role_text"] = entity.role
    if entity.members:
        wire["members"] = True
    if entity.name is not None:
        wire["name"] = entity.name
    if entity.dcid is not None:
        wire["dcid"] = entity.dcid
    return wire


def _spec_nodes(envelope: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Every Spec-shaped node across `envelope["parts"]` that might carry a definition to register.

    Each part is discriminated on its own `status`, so at most one of these
    locations is ever populated per part: `spec` (definite), `candidates.
    specs` (candidates), or `no_data.nearest_real` and `no_data.flagged_match`
    (no_data). `flagged_match` is a single `FlaggedMatchSpec | None`, never a
    list. `unsupported`'s own node omits `spec_id` by construction, so it is
    never walked here.
    """
    nodes: list[Mapping[str, Any]] = []
    for part in envelope.get("parts") or []:
        status = part.get("status")
        if status == "definite":
            spec = part.get("spec")
            if spec is not None:
                nodes.append(spec)
        elif status == "candidates":
            candidates = part.get("candidates") or {}
            nodes.extend(candidates.get("specs") or [])
        elif status == "no_data":
            no_data = part.get("no_data") or {}
            nodes.extend(no_data.get("nearest_real") or [])
            flagged_match = no_data.get("flagged_match")
            if flagged_match is not None:
                nodes.append(flagged_match)
    return nodes


def _verified_siblings(node: Mapping[str, Any]) -> list[StoredDefinition]:
    """The definitions of `node`'s sibling measures that hash to their spec_id.

    A sibling whose identity does not hash is logged with both spec_ids and
    removed from `resolved.alternative_measures`, so the digest never offers
    an id `get_data` cannot resolve. A mismatch in the elected spec refuses
    the call (`_register_definitions`). A `resolved` that is not an object
    offers no siblings, and the spec's own reader refuses it downstream.
    """
    resolved = node.get("resolved")
    if not isinstance(resolved, dict):
        return []
    alternatives = resolved.get("alternative_measures") or []
    kept: list[Mapping[str, Any]] = []
    definitions: list[StoredDefinition] = []
    for alternative in alternatives:
        try:
            stored = stored_definition(alternative)
        except IdentityMismatch:
            logger.warning(
                "skipped sibling measure %s of %s: its identity does not hash to its spec_id",
                alternative["definition"]["spec_id"],
                node.get("spec_id"),
            )
            continue
        kept.append(alternative)
        if stored is not None:
            definitions.append(stored)
    if len(kept) != len(alternatives):
        resolved["alternative_measures"] = kept
    return definitions


async def _register_definitions(envelope: Mapping[str, Any]) -> ToolResult | None:
    """Write every spec_id-carrying definition in `envelope` to the definition store.

    A node with no `definition`, or a `definition` with no `spec_id`, is
    skipped silently. Each spec's `resolved.alternative_measures` entries are
    written too (`_verified_siblings`). A spec's `resolved.scope_choices`
    writes nothing: its compiled identity is the spec's own, its base is the
    identity a switch posts back to Search, and its options carry no
    spec_id. An option is registered only once Search compiles it, which it
    does for an available option alone. Returns a refusal `ToolResult` in
    place of the search's own result when a spec's definition identity doesn't
    reproduce its own spec_id, or when the store itself can't be written to;
    returns `None` otherwise, including when there was nothing to write. Awaited before
    `search` returns, so an id the digest names is always registered by the
    time the call completes. A fast model racing its own ids cannot beat it.
    """
    definitions: list[StoredDefinition] = []
    for node in _spec_nodes(envelope):
        try:
            stored = stored_definition(node)
        except IdentityMismatch as exc:
            return to_tool_result(ToolRefusal.capability_unavailable(detail=str(exc)))
        if stored is not None:
            definitions.append(stored)
        definitions.extend(_verified_siblings(node))
    return await put_definitions(definitions)


def _request_context() -> object | None:
    """The context the current request's `_meta` carries, unread; `None` when it
    carries none or no MCP request is in flight."""
    request_context = fastmcp_request_ctx.get()
    meta = request_context.meta if request_context is not None else None
    return meta.get(SEARCH_CONTEXT_META_KEY) if meta else None


def _unusable_envelope(exc: Exception) -> ToolResult:
    return to_tool_result(
        ToolRefusal.upstream_response_unusable(
            detail=f"the search response envelope did not match the shape this service expects: {exc}"
        )
    )


def _call_deadline(budget_s: float) -> CallDeadline:
    """The deadline for one resolve call: a retry starts only while time remains on it."""
    return CallDeadline.after(budget_s)


async def resolve_and_register(
    wire: Mapping[str, Any],
    *,
    budget_s: float = SEARCH_CALL_BUDGET_S,
    read_timeout_s: float | None = None,
) -> dict[str, Any] | ToolResult:
    """POST `wire` to the Search API's resolve route and register every definition its envelope carries.

    `budget_s` bounds the call, retries included (`_call_deadline`).
    `read_timeout_s` overrides the client's read timeout
    for each attempt (`Settings.upstream_read_timeout_s` when `None`).

    Returns the envelope, or the refusal `ToolResult` a caller returns in its
    place: the upstream refused or could not be reached, the body is not an
    envelope, or a definition was not stored (`_register_definitions`).
    """
    client = get_client()
    try:
        response = await client.request(
            "POST",
            "/api/search/resolve",
            route="/api/search/resolve",
            json=wire,
            timeout=read_timeout_s,
            deadline=_call_deadline(budget_s),
        )
    except UpstreamError as exc:
        return to_tool_result(from_upstream_error(exc))
    try:
        envelope = response.json()
    except ValueError as exc:
        # A non-JSON body would otherwise raise straight out of the tool;
        # fastmcp's own dispatch loop re-raises an unguarded exception here
        # unwrapped rather than folding it into a `ToolError`, bypassing this
        # refusal grammar entirely, the same class of gap `get_data_tool.py`
        # already guards its own mint response against. `httpx2.Response.json()`
        # raises `json.JSONDecodeError` for malformed JSON and the unrelated
        # `UnicodeDecodeError` for a body that isn't valid UTF-8. The two are
        # siblings, both `ValueError` subclasses, neither a subclass of the
        # other, so catching the shared ancestor is what catches both.
        return to_tool_result(
            ToolRefusal.upstream_response_unusable(
                detail=f"the search response body could not be parsed as JSON: {exc}"
            )
        )
    try:
        # `_register_definitions` walks the envelope with `.get()` and no type
        # check on what it finds, so a body that parsed as JSON but isn't an
        # envelope raises one of these rather than a `ToolError`. The
        # refusals it returns are values, not exceptions, and pass through
        # untouched.
        refusal = await _register_definitions(envelope)
    except (AttributeError, KeyError, TypeError) as exc:
        return _unusable_envelope(exc)
    if refusal is not None:
        return refusal
    return envelope


async def search(
    parts: Annotated[list[SearchPart] | None, Field(min_length=1, max_length=MAX_PARTS)] = None,
    raw_query: Annotated[str | None, Field(min_length=1, max_length=_MAX_RAW_QUERY_CHARS)] = None,
) -> ToolResult:
    """Find what data and indicators exist for the user's ask.

    Call this first for a question about development-finance data: it resolves
    their ask into concrete dataset definitions, each identified by a spec_id
    that later data retrieval uses as its handle. Split a compound ask into one
    part per measured variable ("aid to Kenya, and malaria deaths there" is two parts):

    ```json
    {"parts": [
      {"variable": "aid", "entities": [{"text": "Kenya", "role": "to"}]},
      {"variable": "malaria deaths", "entities": [{"text": "Kenya"}]}
    ]}
    ```

    Fill `variable` and each entity's `text` with the user's own words. Do not
    translate, normalise, or guess canonical names, and include only the
    fields the user actually expressed. An entity that names a set of places
    keeps the set's name as `text`, with `members: true` when the ask is for
    each member on its own: "African countries" is `{"text": "Africa",
    "members": true}`, and "Africa" alone is the region as one place. `role` and
    `time` are matched against a fixed grammar, not free text: `role` names
    which side of the flow the entity is on -- a word for where it originates
    (e.g. donor, provider, creditor, exporter) is "from", a word for where it
    is directed (e.g. recipient, debtor, importer) is "to"; `time` takes only
    the forms its own description lists. `open_roles` lists each member of
    a flow side, in the same role words: send it for a "by creditor" or
    "which creditors" ask. A total across a side sends no `open_roles` and
    names the population the user asked for. "Total aid to Ethiopia from all
    donors" is `{"variable": "aid", "entities": [{"text": "Ethiopia", "role":
    "recipient"}, {"text": "all donors", "role": "donor"}]}`, which covers
    official and private donors. A donor side left unnamed on DAC2a covers
    official donors only, and "official donors" stays official. For a named
    population, search binds the source's published total. Where the source
    publishes none, search departs `whole_total_unpublished` and returns the
    members, whose sum is the total. Where the source holds no series of that
    total, such as "official donors" on the CRS, search finds no data. A
    published total stands in only where no operation runs over its side:
    any operation whose `over` is `origin` or `destination`, `aggregate`
    included, opens that flow side and keeps its members, so a
    total leaves `over` off that side. Keep `open_roles` when the user
    explicitly asks to break down that side. Search lists "official donors"
    member by member only where the dataset holds no private donors, and
    otherwise refuses `entity_role_not_admitted`: open the donor side, read it
    with get_data and `place_types` set, and keep the rows whose
    `financingProvider_party_kind` is `party` and
    `financingProvider_party_class` is `official`.
    "Who are Kenya's top 5 creditors?" is `{"variable": "external debt",
    "entities": [{"text": "Kenya", "role": "debtor"}], "open_roles":
    ["creditor"], "operations": [{"verb": "rank",
    "phrase": "top 5 creditors", "over": "origin"}]}`. A rank over place,
    category or year has no flow side. A worldwide country
    ranking is a rank `over` "place" with no entity. A donor or creditor
    ranking uses `origin`; a recipient or debtor ranking uses `destination`.
    A flow role may contain countries and organisations, so opening it does
    not restrict the entity type. A scoped country ranking keeps its scope:
    "top 5 African countries" keeps
    `{"text": "Africa", "members": true}`, and an explicitly named country
    stays an entity, independently of the ranking axis. Do not add "World" as
    a stand-in for an open population; retain it when the user asks for the
    World total.
    A `role_text_not_recognised` or `time_text_not_parsed` warning in the
    response means that field needs rephrasing into that grammar. A
    publisher or dataset the user names, to read from or to leave out, goes
    in the part's `sources` as spoken: "Ghana's debt service from IDS" is
    `{"text": "IDS"}`, and "aid to Kenya, not from CRS" is `{"text": "CRS",
    "exclude": true}`. A lender or recipient the flow runs to or from stays
    an entity with its role: "debt service to the World Bank" names the
    World Bank as an entity. When a source is ambiguous, ask the user which
    candidate they mean, then retry with that candidate's `ref` on the same
    source entry, keeping its `text` and `exclude` flag. A publisher's
    indicator code that the user or a skill gives ("DT.TDS.DEGG.CD") goes in
    the part's `source_code`, and `variable` holds the user's words for the
    measure, or the code itself when the user gives none. When Search reports
    the code ambiguous (`source_code_ambiguous`), retry with a `sources` entry
    naming one of the sources it lists.

    Two entities written alike stay two rows when each carries its `name` or
    `dcid`: "aid from Congo to Congo" sends `{"text": "Congo", "role": "from",
    "name": "Democratic Republic of the Congo"}` and `{"text": "Congo", "role":
    "to", "name": "Republic of the Congo"}`. A row takes `name` or `dcid`,
    never both. When Search refuses `entity_ambiguous`, the digest lists each
    ambiguous entity by number with its candidates' dcids: ask the user which
    one is meant, then resend that row with the candidate's `dcid`, keeping
    its `text` and `role`.

    A comparison of two categories of one measure ("health versus education
    aid") is one part per category, each with `separate: true`, and so is
    any other comparison sent as several parts. Search may otherwise merge
    parts that differ in one category or place into one answer. A total
    over several categories ("health and education aid") stays one part.

    Leave `parts` out to have Search read `raw_query` itself, as its raw-text
    door does: send one question at a time this way, since Search applies the
    places and years it reads to every part of the text.

    Args:
        parts: One entry per variable asked about, 1..6; leave out to search
            `raw_query` as text.
        raw_query: The user's original query; required when `parts` is left out.

    Returns:
        Search's reading of the question. For a question that names data, a
        digest with one block per part, each keyed by that part's own
        status: `definite` (a resolved spec_id), `candidates` (several
        plausible readings, choose one by spec_id), `no_data` (nothing
        available; may suggest nearby alternatives), or `unsupported` (out of
        scope, with the reason). Beneath the blocks, the skills that apply
        to these results, each by id with its description; pass an id to
        `get_skill` to load that skill's guidance. Structured content repeats
        each part's status and spec_ids, and the skill ids, for hosts that
        read it.

        Search can also read the question as not naming data at all, and then
        returns no parts: a `catalogue` answer means it asked what DFO holds
        (the `sources` tool answers that), a `declined` answer means it is out
        of scope or asks for a judgement rather than for data. Each names the
        country the question named, when it named one.
    """
    if not parts and raw_query is None:
        return to_tool_result(
            ToolRefusal.request_invalid(
                pointer="/raw_query",
                detail="search needs `parts`, or `raw_query` for Search to read as text",
            )
        )
    sent = _request_context()
    if sent is not None and not Settings().search_sends_conversation_context:
        logger.debug("search context dropped: search_sends_conversation_context is off")
        trace.get_current_span().set_attribute("dfo.search.context_dropped", True)
        sent = None
    try:
        context = None if sent is None else SearchContext.model_validate(sent)
    except ValidationError as exc:
        return to_tool_result(
            ToolRefusal.request_invalid(
                pointer=f"/_meta/{SEARCH_CONTEXT_META_KEY}",
                detail=f"the search context is not a SearchContext: {exc}",
            )
        )
    wire = to_wire_request(parts, raw_query, context=context)
    envelope = await resolve_and_register(wire)
    if isinstance(envelope, ToolResult):
        return envelope
    try:
        content = render_digest(envelope)
        answer = render_answer(envelope)
    except (AttributeError, KeyError, TypeError) as exc:
        return _unusable_envelope(exc)
    return ToolResult(
        content=content,
        structured_content=answer,
        meta={SEARCH_ENVELOPE_META_KEY: envelope},
        is_error=False,
    )

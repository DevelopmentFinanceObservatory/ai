"""The `search` tool: data-mcp's wrapper over the Search API's parsed door.

A conjunction arrives already split into `parts`, one per variable asked
for. The door's field names are close to the tool's without matching them
(`variable`/`time`/`unit`/`source`/`role`/`dataset` versus the wire's
`variable_text`/`time_text`/`unit_text`/`source_text`/`role_text`/`provenance`),
so `to_wire_request` translates. The datasets a part can name are read from the
door's served request schema at startup (`fetch_declared_datasets`) and listed
in the tool description (`describe_search`). A success result puts the digest in
`content`, a small model-facing answer in `structuredContent`, and the upstream
envelope in `_meta` (see `data_mcp.search_result`). Rendering the digest and the answer
lives in `data_mcp.digest`; the shared upstream client lives in
`data_mcp.client_holder`.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Any

from fastmcp.tools import ToolResult
from pydantic import BaseModel, ConfigDict, Field

from data_mcp.client_holder import get_client
from data_mcp.deadline import CallDeadline
from data_mcp.definition_store import IdentityMismatch, StoredDefinition
from data_mcp.definition_writes import put_definitions, stored_definition
from data_mcp.digest import render_answer, render_digest
from data_mcp.errors import UpstreamError
from data_mcp.search_result import SEARCH_ENVELOPE_META_KEY
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
MAX_SOURCE_CHARS = 200
_MAX_DATASET_CHARS = 200
_MAX_RAW_QUERY_CHARS = 2000

# The wall-clock budget for one call to the Search API's resolve route, retries
# included. It sits above `Settings.upstream_read_timeout_s` (50 s, itself above
# dfo-apis' 45 s search deadline) and below the harness's 60 s `search` call
# timeout (`compiler.capabilities`), so a slow search ends here with dfo-apis' own
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

_SCHEMA_ROUTE = "/api/search/schema"
# Bounds the whole startup read, so an unreachable dfo-apis delays startup by
# at most this long.
_SCHEMA_FETCH_TIMEOUT_S = 5.0


class SearchEntity(BaseModel):
    """One entity mentioned in a part, with an optional relational phrase."""

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


class SearchPart(BaseModel):
    """One variable ask within the request, with its own entities, time, unit, source, and dataset."""

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
        description="unit and price basis, e.g. 'constant 2015 US dollars'",
    )
    source: str | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_SOURCE_CHARS,
        description=(
            "who reported it, refused when no unit is given and ignored when one is, so "
            "choose the source through get_data's facets instead"
        ),
    )
    open_roles: list[Annotated[str, Field(min_length=1, max_length=_MAX_ROLE_CHARS)]] | None = (
        Field(
            default=None,
            min_length=1,
            max_length=2,
            description=(
                "the side the ask covers every member of, as a role word, e.g. ['creditor'] "
                "for 'which creditors'; never a side one of the entities names"
            ),
        )
    )
    dataset: str | None = Field(
        default=None,
        min_length=1,
        max_length=_MAX_DATASET_CHARS,
        description=(
            "the dcid of a declared table the part asks for, from the list in the tool "
            "description; leave it out unless the user asks for that table"
        ),
    )


@dataclass(frozen=True)
class DeclaredDataset:
    """One table the parsed door accepts as a part's `provenance`."""

    dcid: str
    title: str
    description: str
    opt_in: bool


def declared_datasets(schema: Mapping[str, Any]) -> list[DeclaredDataset]:
    """Read the declared datasets, in rank order, from the door's served schema document.

    They are the `oneOf` values of the string branch of `ParsedPart.provenance`
    in the document's `request` schema. A document of another shape raises
    `LookupError`, `StopIteration` or `TypeError`.
    """
    field = schema["request"]["$defs"]["ParsedPart"]["properties"]["provenance"]
    named = next(branch for branch in field["anyOf"] if branch.get("type") == "string")
    return [
        DeclaredDataset(
            dcid=value["const"],
            title=value["title"],
            description=value["description"],
            opt_in=value["x-opt-in"],
        )
        for value in named.get("oneOf", [])
    ]


async def fetch_declared_datasets() -> list[DeclaredDataset] | None:
    """Read the declared datasets from dfo-apis once, within `_SCHEMA_FETCH_TIMEOUT_S`.

    Returns `None`, and logs a warning, when the schema cannot be read or does
    not have the expected shape.
    """
    try:
        async with asyncio.timeout(_SCHEMA_FETCH_TIMEOUT_S):
            response = await get_client().request(
                "GET",
                _SCHEMA_ROUTE,
                route=_SCHEMA_ROUTE,
                timeout=_SCHEMA_FETCH_TIMEOUT_S,
                retry=False,
            )
        return declared_datasets(response.json())
    except (TimeoutError, UpstreamError, ValueError, LookupError, StopIteration, TypeError) as exc:
        logger.warning(
            "could not read the declared datasets from %s; the search tool lists none: %r",
            _SCHEMA_ROUTE,
            exc,
        )
        return None


def describe_search(base: str, datasets: Sequence[DeclaredDataset] | None) -> str:
    """The search tool's description: `base` followed by the datasets a part can name."""
    if not datasets:
        return f"{base}\n\nNo datasets are listed, so leave `dataset` out."
    lines = [
        f"- `{dataset.dcid}`: {dataset.title}. {dataset.description}"
        + (" Served only when a part names it." if dataset.opt_in else "")
        for dataset in datasets
    ]
    return (
        f"{base}\n\nSet a part's `dataset` to one of these dcids only when the user asks "
        "for that table:\n" + "\n".join(lines)
    )


def to_wire_request(parts: Sequence[SearchPart], raw_query: str | None) -> dict[str, Any]:
    """Translate tool-facing parts into the parsed door's wire request body."""
    input_block: dict[str, Any] = {"kind": "parsed", "parts": [_wire_part(part) for part in parts]}
    if raw_query is not None:
        input_block["raw_query"] = raw_query
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
    if part.source is not None:
        wire["source_text"] = part.source
    if part.open_roles is not None:
        wire["open_roles"] = part.open_roles
    if part.dataset is not None:
        wire["provenance"] = part.dataset
    return wire


def _wire_entity(entity: SearchEntity) -> dict[str, Any]:
    wire: dict[str, Any] = {"text": entity.text}
    if entity.role is not None:
        wire["role_text"] = entity.role
    if entity.members:
        wire["members"] = True
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


async def _register_definitions(envelope: Mapping[str, Any]) -> ToolResult | None:
    """Write every spec_id-carrying definition in `envelope` to the definition store.

    A node with no `definition`, or a `definition` with no `spec_id`, is
    skipped silently. Returns a refusal `ToolResult` in place of the search's
    own result when a definition's identity doesn't reproduce its own
    spec_id, or when the store itself can't be written to; returns `None`
    otherwise, including when there was nothing to write. Awaited before
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
    return await put_definitions(definitions)


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
    parts: Annotated[list[SearchPart], Field(min_length=1, max_length=MAX_PARTS)],
    raw_query: Annotated[str | None, Field(min_length=1, max_length=_MAX_RAW_QUERY_CHARS)] = None,
) -> ToolResult:
    """Find what data and indicators exist for the user's ask.

    Call this first when the user asks about data: it resolves their ask into
    concrete dataset definitions, each identified by a spec_id that later data
    retrieval uses as its handle. Split a compound ask into one part per
    measured variable ("aid to Kenya, and malaria deaths there" is two parts):

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
    the forms its own description lists. `open_roles` asks for every member
    of a side, as in "which creditors does Nigeria pay", in the same role words.
    A `role_text_not_recognised` or `time_text_not_parsed` warning in the
    response means that field needs rephrasing into that grammar. A
    `source` value refuses when no unit is given and is ignored when one is,
    so leave it out and choose the source through `get_data`'s facets
    instead.

    Args:
        parts: One entry per variable asked about, 1..6.
        raw_query: The user's original query.

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
    wire = to_wire_request(parts, raw_query)
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

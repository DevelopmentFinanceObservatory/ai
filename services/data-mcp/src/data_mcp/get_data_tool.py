"""The `get_data` tool: retrieves the data behind a spec_id a prior `search` minted.

`search` resolves an ask into a spec_id but never fetches the underlying data;
`get_data` reads the definition for that id (the local store first, falling
back to apis' own permanent registry on a store miss), mints it through
apis' Datasets door, and returns a frame reference plus a digest. Entity
names in the caller's `scope` are resolved to dcids through
`data_mcp.resolution` before the mint call; the mint call itself and the
resulting manifest never carry a name, only dcids and, later, a spec_id.

The mint posts the spec_id and its identity only. apis resolves the
family's members from the graph on every mint, so no caller supplies them.

Rendering the manifest into a digest lives in `data_mcp.get_data_digest`; the
shared upstream client lives in `data_mcp.client_holder`.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import json
import logging
import re
import time
from collections.abc import Callable, Coroutine, Mapping, Sequence
from dataclasses import dataclass
from http import HTTPStatus
from typing import Annotated, Any, Literal, NamedTuple

import httpx2
import pyarrow as pa
from common.caller import current_writer
from common.materialisation import FrameRef, MintRecord
from common.place_type import PARTY_ROLES, PLACE_ROLE, type_column
from fastmcp import Context
from fastmcp.exceptions import ResourceError
from fastmcp.tools import ToolResult
from mcp.types import ResourceLink, TextContent
from opentelemetry import trace
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from data_mcp import node_lookup
from data_mcp.client import UpstreamClient
from data_mcp.client_holder import get_client
from data_mcp.deadline import (
    DEADLINE_RETRY_AFTER_S,
    CallDeadline,
    Clock,
    current_call_budget,
)
from data_mcp.definition_store import (
    DefinitionStoreUnavailable,
    IdentityMismatch,
    StoredDefinition,
    get_store,
    verify_write,
)
from data_mcp.errors import UpstreamAuthUnavailable, UpstreamError, UpstreamTransportError
from data_mcp.frame_materialisation import (
    DEFAULT_MINT_OPTIONS,
    FLOW_ENRICHMENT_COLUMNS,
    KIND_COLUMNS,
    WITHOUT_PARTY_TOTALS,
    materialise_and_register,
    place_type_columns_for,
    replayed_mint_options,
)
from data_mcp.get_data_digest import (
    BreakdownLevel,
    OperationPollNote,
    labelled_warnings,
    render_dcids,
    render_get_data_digest,
    statistics_needs_fetch,
    summed_family,
    warning_refs,
)
from data_mcp.graph_wire import NotHeldEntry
from data_mcp.manifest import (
    ArtifactEntry,
    DefinitionModel,
    Manifest,
    SeriesStatisticsModel,
    render_validation_errors,
    strip_manifest,
)
from data_mcp.mint_admission import (
    BACKGROUND_MINT_SPAN_ATTRIBUTE,
    Admission,
    admission_headers,
    background_outcome,
    background_result,
    declined_in_background,
    requested_admission,
)
from data_mcp.mint_ledger import HeldMint, MintLedger
from data_mcp.operations import (
    OnPoll,
    PollUpdate,
    QueuedOperation,
    ResolvedOperation,
    operation_without_waiting,
    resolve_operation_response,
)
from data_mcp.resolution import (
    Ambiguous,
    InvalidName,
    NotFound,
    NotHeld,
    ResolutionOutcome,
    ResolvedEntity,
    ResolvedRegion,
    resolve_entities,
)
from data_mcp.series_statistics import (
    SeriesStatisticsSchemaError,
    continuing_method,
    fetch_series_statistics,
    find_statistics_artifact,
)
from data_mcp.settings import Settings
from data_mcp.stat_var_list import without_long_stat_var_list
from data_mcp.tool_errors import (
    ENTITY_DCID_MARKER,
    EntityCandidate,
    ToolErrorCode,
    ToolRefusal,
    facet_issues_from,
    from_upstream_error,
    refusal_code,
    to_tool_result,
)
from data_mcp.version_description import fetch_version_description

# Both `CanonicalDefinition.schema_version` and `MintRequest.schema_version`
# are `Literal[1]` upstream, and a model calling this tool has no meaningful
# choice to make here -- one constant stands in for the store lookup and
# every `schema_version` field the mint request body carries.
_SCHEMA_VERSION = 1

# The mixer behind a mint request supports at most 3 entity-role slots per
# observation (`SdmxRoleCeilingError`, apis' `datasets_api/doors/mint.py`
# around lines 464-474); a role count beyond that already reaches a clean
# `request_invalid` refusal from the mint call itself. 8 stays as this
# tool's own bound rather than matching upstream's 3 exactly -- it exists to
# keep one call's resolve fan-out and mint request bounded against a
# pathological role count, not to preempt upstream's own refusal.
_MAX_ROLES = 8

# Same reasoning as `_MAX_ROLES`, applied to names summed across every role
# supplied in one call.
_MAX_ENTITY_NAMES = 50

# Matches the resolve door's own per-text limit
# (`data_mcp.resolution._MAX_NAME_CHARS`): a name longer than this could
# never resolve, so rejecting it locally saves a wasted round trip.
_MAX_ENTITY_NAME_CHARS = 200

# `data_mcp.definition_store.identity.spec_id_for` mints "spec_" + 32
# lowercase hex chars.
_SPEC_ID_PATTERN = r"^spec_[0-9a-f]{32}$"

# An unmarked value other tools' output shows as a dcid (`country/KEN` in a digest, a
# recall entry or the canvas digest). Namespaced graph dcids are `prefix/segment[/...]`
# (`country/KEN`, `geoId/13`, `wikidataId/Q30`), each segment `[A-Za-z0-9_.-]+` as
# apis' `graph_api.routing` defines the node door's `{dcid}` path, and every
# entity-role namespace is lowercase-initial. A slash alone is not enough:
# `Congo/Brazzaville` and `Bosnia/Herzegovina` are spoken names a caller types, so a
# capitalised segment before the slash is a name to resolve. A bare dcid such as
# `GroupOfSeven` has no slash and passes only behind `ENTITY_DCID_MARKER`.
_DCID_PATTERN = re.compile(r"^[a-z][A-Za-z0-9_.\-]*(?:/[A-Za-z0-9_.\-]+)+$")

# What `get_data(stat_vars="sum")` asks apis for: `by` names the columns to keep,
# and apis sums away every summable code axis it does not name, `stat_var` first.
# `time_period` is a column every frame carries and no code axis, so this keeps no
# code axis and apis returns a full collapse: one StatVar ref standing for the
# family, with every entity role column kept (apis never sums across a role).
_SUM_STAT_VARS_GROUPBY: dict[str, Any] = {"by": ["time_period"], "op": "sum"}

StatVarsReading = Literal["sum", "separate"]

# The `GET /api/datasets/capabilities` field that lists the code systems
# `options.groupby.levels` rolls up. A deployment without it serves no `breakdown`.
_GROUPBY_LEVEL_AXES_FIELD = "groupby_level_axes"
_CAPABILITIES_PATH = "/api/datasets/capabilities"
# The route that reads a definition out of apis' registry by `spec_id` and answers its
# identity under `definition`. A GET of a spec rebuilds its stale head, so it may take a
# materialisation's time.
_OPTIONS_ROUTE = "/api/datasets/{id}/options"

# No `UpstreamError` backs a definition-store read failure, so there is no
# call to derive a wait from. Mirrors `search_tool.py`'s constants of the
# same name and value.
_DEFINITION_STORE_RETRY_AFTER_S = 5.0
_DEFINITION_STORE_SERVICE_NAME = "the definition store"

# The current `CallBudget` shares out one call's deadline (`data_mcp.deadline`). Inside
# its post-mint reserve, the statistics fetch runs only with `_STATISTICS_FLOOR_S` left
# and must finish with `_STATISTICS_RESERVE_S` still unspent, for the digest. The
# description read and the warnings' label read run beside it and end with the same
# reserve unspent.
_STATISTICS_FLOOR_S = 8.0
_STATISTICS_RESERVE_S = 3.0

# The clock a call's deadline reads. Tests replace it to move time without sleeping.
_deadline_clock: Clock = time.monotonic

# The mints this process holds apis mint slots for on behalf of outside callers,
# on `_deadline_clock`'s time. Tests replace it with an empty one.
_mint_ledger = MintLedger()

# Each mint POST still open, under its request key (`_post_mint`), and the event
# set when it ends. Per process.
_open_mint_posts: dict[str, asyncio.Event] = {}

# How much more than `post_mint_reserve_s` plus `min_mint_timeout_s` a foreground
# mint waiting on an open POST leaves of its deadline (`_post_mint`), so the time
# the wait overruns its timeout by cannot take the POST under `min_mint_timeout_s`.
_MINT_WAIT_SLACK_S = 1.0

logger = logging.getLogger(__name__)

# The type column of a place-subject frame.
_PLACE_TYPE_COLUMN = type_column(PLACE_ROLE)


class GetDataScope(BaseModel):
    """The scope override to apply before minting, mirroring the mint route's own `scope` block.

    Each supplied top-level member -- `entities`, `window`, `facets` --
    REPLACES that whole member on the stored definition; an omitted member
    leaves the corresponding part of the definition untouched. See
    `get_data`'s own docstring for the replace-whole rule and its worked
    example.
    """

    model_config = ConfigDict(extra="forbid")

    entities: dict[str, list[str]] | None = Field(
        default=None,
        description=(
            "role -> entity names, as spoken, e.g. "
            '{"recipient": ["Togo", "Ghana"]}. A dcid marked dcid: (e.g. '
            '"dcid:country/GEO") is also accepted and skips resolution -- only '
            "to answer an entity_ambiguous refusal's candidate list from a "
            "prior call, never invented. Replaces the whole entities member -- an "
            "addition must restate every role you want kept. Omit to leave "
            "entities untouched; pass {} to clear them."
        ),
    )
    window: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Replaces the whole window member, e.g. "
            '{"kind": "range", "start_year": 2015, "end_year": 2024}. '
            "Omit to leave the window untouched."
        ),
    )
    facets: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Replaces the whole facets member, keyed by axis -- one of "
            '"provenance", "measurementMethod", "unit". Each value is '
            'exactly {"ref": "..."}, e.g. {"provenance": {"ref": '
            '"OECD_DAC2A"}}. Omit to leave facets untouched.'
        ),
    )

    @model_validator(mode="after")
    def _entity_caps(self) -> GetDataScope:
        if self.entities is None:
            return self
        if len(self.entities) > _MAX_ROLES:
            raise ValueError(f"at most {_MAX_ROLES} roles, got {len(self.entities)}")
        total_names = sum(len(names) for names in self.entities.values())
        if total_names > _MAX_ENTITY_NAMES:
            raise ValueError(
                f"at most {_MAX_ENTITY_NAMES} entity names summed across roles, got {total_names}"
            )
        for names in self.entities.values():
            for name in names:
                if not 1 <= len(name) <= _MAX_ENTITY_NAME_CHARS:
                    raise ValueError(
                        f"entity name length must be 1..{_MAX_ENTITY_NAME_CHARS} characters, "
                        f"got {len(name)} for {name!r}"
                    )
                if _entity_dcid(name) == "":
                    raise ValueError(
                        f"{name!r} carries the {ENTITY_DCID_MARKER} marker but no dcid"
                    )
        return self


def _entity_dcid(value: str) -> str | None:
    """The dcid a `scope.entities` value names, or `None` for a name to resolve.

    A value behind the `dcid:` marker is the dcid after it: the marker is how an
    `entity_ambiguous` refusal names its candidates, and no spoken name starts with it, so a
    bare dcid such as `GroupOfSeven` passes too. An unmarked value is a dcid when it matches
    `_DCID_PATTERN`, as the dcids other tools' output shows do.
    """
    stripped = value.strip()
    if stripped.startswith(ENTITY_DCID_MARKER):
        return stripped.removeprefix(ENTITY_DCID_MARKER).strip()
    return stripped if _DCID_PATTERN.fullmatch(stripped) else None


class Breakdown(BaseModel):
    """One hierarchical code axis to keep, rolled up to one level of its hierarchy."""

    model_config = ConfigDict(extra="forbid")

    code_system: str = Field(
        min_length=1,
        description=(
            'The code system to break the total down by, e.g. "crsPurpose" for CRS sectors.'
        ),
    )
    level: int = Field(
        ge=0,
        description=(
            "The hierarchy depth to roll every code up to: 0 is the top level (for "
            "crsPurpose, the ten OECD sector groups), 1 the level below it, and so on."
        ),
    )

    def groupby(self) -> dict[str, Any]:
        """The `options.groupby` apis sums this breakdown with: keep the year and the
        code axis, roll each code up to its ancestor at `level`, and sum away every other
        summable code axis and `stat_var`."""
        return {
            "by": ["time_period", self.code_system],
            "op": "sum",
            "levels": {self.code_system: self.level},
        }


async def _breakdown_refusal(
    client: UpstreamClient, breakdown: Breakdown, *, deadline: CallDeadline
) -> ToolResult | None:
    """Refuse `breakdown` unless the deployed apis advertises a level roll-up of its
    code system, or `None` to proceed.

    The capabilities document lists the code systems `groupby.levels` accepts under
    `_GROUPBY_LEVEL_AXES_FIELD`. A deployment whose document lacks the field refuses
    `capability_unavailable` naming the option, and one that lists other code systems
    refuses `request_invalid` at `/breakdown/code_system`. The read must finish while
    the current `CallBudget.pre_mint_reserve_s` of `deadline` remains.
    """
    reserve_s = current_call_budget().pre_mint_reserve_s
    try:
        response = await client.request(
            "GET",
            _CAPABILITIES_PATH,
            route=_CAPABILITIES_PATH,
            deadline=deadline.reserving(reserve_s),
        )
        document = response.json()
    except UpstreamError as exc:
        return to_tool_result(from_upstream_error(exc))
    except ValueError as exc:
        return to_tool_result(
            ToolRefusal.upstream_response_unusable(
                detail=f"the capabilities response body could not be parsed as JSON: {exc}"
            )
        )
    if not isinstance(document, dict):
        return to_tool_result(
            ToolRefusal.upstream_response_unusable(
                detail="the capabilities response is not a JSON object"
            )
        )
    capability_version = document.get("capability_version")
    capability_version = capability_version if isinstance(capability_version, str) else None
    axes = document.get(_GROUPBY_LEVEL_AXES_FIELD)
    if not isinstance(axes, list):
        return to_tool_result(
            ToolRefusal.capability_unavailable(
                option="breakdown",
                detail="this dfo-apis deployment does not advertise groupby.levels, "
                "so no total can be broken down by a code level",
                capability_version=capability_version,
            )
        )
    if breakdown.code_system not in axes:
        listed = ", ".join(sorted(str(axis) for axis in axes)) or "none"
        return to_tool_result(
            ToolRefusal.request_invalid(
                pointer="/breakdown/code_system",
                detail=f"{breakdown.code_system!r} cannot be rolled up to a level; "
                f"this deployment rolls up {listed}",
                capability_version=capability_version,
            )
        )
    return None


def _refusal_for_unresolved_outcome(
    outcome: ResolutionOutcome, *, role: str, query: str
) -> ToolResult | None:
    """Build the refusal for a resolution outcome that can't proceed to mint.

    Returns `None` for `ResolvedEntity` and `ResolvedRegion`, which `_resolve_scope_entities`
    handles itself by appending their dcids.
    """
    if isinstance(outcome, Ambiguous):
        return to_tool_result(
            ToolRefusal.entity_ambiguous(
                role=role,
                query=query,
                candidates=[
                    EntityCandidate(dcid=c.dcid, label=c.name, type=c.entity_type)
                    for c in outcome.candidates
                ],
            )
        )
    if isinstance(outcome, NotFound):
        return to_tool_result(
            ToolRefusal.entity_not_found(
                role=role, query=query, near_matches=[m.name for m in outcome.near_matches]
            )
        )
    if isinstance(outcome, NotHeld):
        return to_tool_result(
            ToolRefusal.entity_not_found(role=role, query=query, near_matches=[], not_held=True)
        )
    if isinstance(outcome, InvalidName):
        return to_tool_result(
            ToolRefusal.request_invalid(
                pointer=f"/scope/entities/{role}", role=role, detail=outcome.reason
            )
        )
    return None


async def _resolve_scope_entities(
    entities: Mapping[str, Sequence[str]],
) -> tuple[dict[str, list[str]], dict[str, str]] | ToolResult:
    """Resolve every name in `entities` into dcids for the mint request.

    A value `_entity_dcid` reads as a dcid is sent as that dcid and never reaches the resolve
    door: one behind the `dcid:` marker, as this tool's own `entity_ambiguous` refusal names its
    candidates, or an unmarked namespaced dcid copied from another tool's output. Everything
    else is resolved by name.

    Returns `(wire_entities, entity_labels)` on full success: `wire_entities`
    is role -> bare dcids, order-preserving-deduped so the same dcid reached twice under one
    role (a region and one of its own members, both named) mints once rather than twice --
    upstream preserves this list verbatim into the hashed identity document, so an undeduped
    list would mint two different spec_ids for one logical question depending on phrasing;
    and `entity_labels` is dcid -> name for every entity and region member this call resolved
    by name (a passthrough dcid gets no entry, so the digest renders it as itself).

    Returns a refusal `ToolResult` in place of the tuple on the first name that isn't cleanly
    usable. `role` always names the failed name's own role, per the platform contract's
    entity-refusal requirement; every pointer this function builds names this tool's own
    `scope.entities.{role}` argument path -- the frame a caller who called `get_data` itself
    reads pointers in, not apis' `/input`-wrapped wire pointer.
    """
    names_to_resolve = {
        role: [name for name in names if _entity_dcid(name) is None]
        for role, names in entities.items()
    }
    outcomes = await resolve_entities(names_to_resolve)
    wire_entities: dict[str, list[str]] = {}
    entity_labels: dict[str, str] = {}
    for role, names in entities.items():
        dcids: list[str] = []
        for name in names:
            if (dcid := _entity_dcid(name)) is not None:
                dcids.append(dcid)
                continue
            outcome = outcomes[role][name]
            if isinstance(outcome, ResolvedEntity):
                dcids.append(outcome.dcid)
                entity_labels[outcome.dcid] = outcome.name
            elif isinstance(outcome, ResolvedRegion):
                if not outcome.members:
                    # A group or region with no member country (a region of
                    # territories only) would scope this role to zero dcids;
                    # it refuses instead.
                    return to_tool_result(
                        ToolRefusal.entity_not_found(
                            role=role,
                            query=name,
                            near_matches=[],
                            memberless_kind=outcome.entity_type,
                        )
                    )
                for member in outcome.members:
                    dcids.append(member.dcid)
                    entity_labels[member.dcid] = member.name
            else:
                refusal = _refusal_for_unresolved_outcome(outcome, role=role, query=name)
                if refusal is not None:
                    return refusal
        wire_entities[role] = list(dict.fromkeys(dcids))
    return wire_entities, entity_labels


class _RegistryIdentity(NamedTuple):
    """An identity read from apis' registry, and whether it hashes to the requested id."""

    identity: dict[str, Any]
    hashes_to_requested_id: bool


async def _registry_identity_or_refuse(
    spec_id: str, *, deadline: CallDeadline
) -> _RegistryIdentity | ToolResult:
    """The identity of `spec_id` read from apis' registry, for an id the definition
    store does not hold.

    `get_data` chooses the mint's columns from the spec's identity, and a spec it cannot
    read would be asked for the wrong ones: a flow table refuses a place type column, and
    a frame minted without kinds serves every party total. The read is
    `GET /api/datasets/{id}/options`, which answers the identity of the version it serves
    and rebuilds a stale head as any GET of the id does, so it must finish while the
    current `CallBudget.pre_mint_reserve_s` of `deadline` remains, as the write after it.

    The registry may serve the id under another `spec_id`. Its identity still names the
    same table's sides, so it chooses the columns, but it goes in the mint's `definition`
    sidecar, which carries the id it hashes to, and in the definition store, only when
    that id is the requested one (`hashes_to_requested_id`); a stored one is found by the
    next call. A refusal `ToolResult` when the registry does not hold the id either
    (`definition_not_held`), answers with another failure, serves a body this service
    cannot use, or serves an identity that does not hash to its own id.
    """
    reserve_s = current_call_budget().pre_mint_reserve_s
    try:
        response = await get_client().request(
            "GET",
            f"/api/datasets/{spec_id}/options",
            route=_OPTIONS_ROUTE,
            deadline=deadline.reserving(reserve_s),
        )
        definition = DefinitionModel.model_validate(response.json()["definition"])
    except UpstreamError as exc:
        if exc.code == "spec_not_found":
            return to_tool_result(ToolRefusal.definition_not_held(definition_id=spec_id))
        return to_tool_result(from_upstream_error(exc))
    except (ValueError, KeyError, TypeError) as exc:
        return to_tool_result(
            ToolRefusal.upstream_response_unusable(
                detail=f"the definition of {spec_id} could not be read from the registry: {exc}"
            )
        )
    hashes_to_requested_id = definition.spec_id == spec_id
    try:
        if hashes_to_requested_id:
            if not await _persist_definition(definition, deadline=deadline, reserve_s=reserve_s):
                logger.warning(
                    "get_data: the definition store did not confirm the write of %s; the next "
                    "call reads the registry again",
                    spec_id,
                )
        else:
            verify_write(dict(definition.identity), definition.spec_id)
    except IdentityMismatch as exc:
        return to_tool_result(ToolRefusal.capability_unavailable(detail=str(exc)))
    return _RegistryIdentity(dict(definition.identity), hashes_to_requested_id)


def _pre_mint_deadline_refusal(*, service: str, step: str) -> ToolResult:
    """The retryable refusal for pre-mint work that ran past its share of the deadline."""
    return to_tool_result(
        ToolRefusal.service_unavailable(
            service=service,
            detail=f"{step} did not finish within its share of this call's deadline",
            retry_after_s=DEADLINE_RETRY_AFTER_S,
        )
    )


async def _resolve_entities_or_refuse(
    entities: Mapping[str, Sequence[str]], *, deadline: CallDeadline
) -> tuple[dict[str, list[str]], dict[str, str]] | ToolResult:
    """Wrap `_resolve_scope_entities` with the resolve door's own failure modes.

    `resolve_entities` deliberately lets `UpstreamError` propagate (see its own docstring);
    nothing else in the call path catches it, so without this wrapper a routine 503 from a
    degraded graph would escape as a raw exception -- `is_error=True` with no code, retry
    mode, or wait -- instead of this refusal grammar. A 2xx body this service still can't use
    raises `ValueError`, never a `ToolError`: the generated `ResolveResponse` model raises
    `pydantic.ValidationError` (a `ValueError`) for a body off the wire's schema, such as an
    unrecognised status, and `_outcome_from_result` raises `ValueError` for a result missing a
    field its status sets. `KeyError` and `TypeError` cover a body that is not a JSON object.
    Each means the resolve call itself succeeded and only its body couldn't be used, the same
    class of failure `upstream_response_unusable` already names for the mint call.

    Resolution must finish while the current `CallBudget.pre_mint_reserve_s` of
    `deadline` remains; past that it refuses retryably, before anything is minted.
    """
    reserve_s = current_call_budget().pre_mint_reserve_s
    try:
        async with asyncio.timeout(deadline.reserving(reserve_s).remaining()):
            return await _resolve_scope_entities(entities)
    except TimeoutError:
        return _pre_mint_deadline_refusal(service="dfo-apis", step="entity resolution")
    except UpstreamError as exc:
        return to_tool_result(from_upstream_error(exc))
    except (ValueError, KeyError, TypeError) as exc:
        return to_tool_result(
            ToolRefusal.upstream_response_unusable(
                detail=f"the resolve response did not match the shape this service expects: {exc}"
            )
        )


# The prefix an upstream `request_invalid` pointer carries for a scope-entities role, in
# apis' own wire envelope (`{input: {scope: {entities: {role: [...]}}}}`), which differs
# from this tool's own `scope.entities.{role}` argument path, which `InvalidName` above
# already points into directly.
_UPSTREAM_ENTITIES_ROLE_POINTER_PREFIX = "/input/scope/entities/"


def _rebase_entities_role_pointer(refusal: ToolRefusal) -> ToolRefusal:
    """Rewrite an upstream-derived entities-role pointer onto this tool's own argument path.

    `invalid_role_ref`/`role_incompatible` map to `request_invalid` with a pointer straight
    off apis' wire (`/input/scope/entities/{role}`), while a locally-rejected name
    (`InvalidName` above) already points at `/scope/entities/{role}` -- this tool's own
    `scope` argument, with no `/input` wrapper, since that wrapper names apis' request
    envelope, not anything this tool's own caller ever supplied. Same argument, one
    convention: this rebases the upstream-derived pointer onto the local one rather than
    leaving a caller to notice the two never match.
    """
    pointer = refusal.fix.get("pointer")
    if refusal.code is not ToolErrorCode.REQUEST_INVALID or not isinstance(pointer, str):
        return refusal
    if not pointer.startswith(_UPSTREAM_ENTITIES_ROLE_POINTER_PREFIX):
        return refusal
    rebased = "/scope/entities/" + pointer[len(_UPSTREAM_ENTITIES_ROLE_POINTER_PREFIX) :]
    # Rebuilt through `request_invalid` itself: that factory renders `message` with the
    # pointer embedded inline (`f"...at {pointer}..."`), so `message` and `fix` have to come
    # from one call to name the same rebased path. `retry_after_s` and `upstream_code` are
    # not carried through explicitly: `_request_invalid_refusal` never sets either for this
    # code, so both are already at their factory defaults on `refusal` too.
    return ToolRefusal.request_invalid(
        pointer=rebased,
        detail=refusal.fix.get("detail"),
        role=refusal.fix.get("role"),
        trace_id=refusal.trace_id,
        capability_version=refusal.capability_version,
    )


async def _load_identity_or_refuse(
    spec_id: str, *, deadline: CallDeadline
) -> dict[str, Any] | None | ToolResult:
    """Read the store for `spec_id` and resolve what the mint call below should send.

    Returns the stored canonical identity document on a store hit, exactly what the mint
    call sends as `definition.identity`; `None` when the store missed but the id should
    still attempt the registry-fallback mint path in `get_data` (a `None` identity drives
    that call to omit the `definition` sidecar); or a refusal `ToolResult` when the
    store itself is unreachable or its held bytes fail their own hash check.

    The read must finish while the current `CallBudget.pre_mint_reserve_s` of `deadline`
    remains; past that it refuses retryably.
    """
    reserve_s = current_call_budget().pre_mint_reserve_s
    try:
        async with asyncio.timeout(deadline.reserving(reserve_s).remaining()):
            stored = await get_store().get(spec_id, schema_version=_SCHEMA_VERSION)
    except TimeoutError:
        return _pre_mint_deadline_refusal(
            service=_DEFINITION_STORE_SERVICE_NAME, step="reading the stored definition"
        )
    except DefinitionStoreUnavailable:
        return to_tool_result(
            ToolRefusal.service_unavailable(
                service=_DEFINITION_STORE_SERVICE_NAME,
                retry_after_s=_DEFINITION_STORE_RETRY_AFTER_S,
            )
        )
    except IdentityMismatch as exc:
        return to_tool_result(ToolRefusal.capability_unavailable(detail=str(exc)))

    if stored is not None:
        return json.loads(stored.canonical_bytes)
    # No local hit, but apis' own registry may still hold this id: the local store is
    # per-process by default and even its GCS-backed form expires at 30 days, while the
    # upstream registry's rows are never deleted. Only a registry miss too (`spec_not_found`,
    # caught in `get_data`) completes the platform contract's own definition of
    # `definition_not_held`: "held by neither store nor registry".
    return None


async def _persist_definition(
    definition: DefinitionModel, *, deadline: CallDeadline, reserve_s: float
) -> bool:
    """Write a definition apis served to the store, so a later `get_data` call with its id succeeds.

    Called for a mint route's own manifest carrying a `spec_id` different from the one
    this call was made with: a scope override changed the identity, and upstream re-minted
    under a new id. That new id is what the digest hands the model as its frame reference, so
    it has to be resolvable by a later `get_data` call the same way any other id is -- without
    this write, that later call would refuse `definition_not_held` even though apis' own
    registry now holds this id, which is exactly the case the platform contract's
    `definition_not_held` excludes ("held by neither store nor registry"). Called too for
    the definition of an id this store did not hold (`_registry_identity_or_refuse`). `verify_write`
    canonicalises and hash-checks `definition.identity` the same way `search_tool.py`'s own
    registration path does, so a corrupt identity is caught rather than trusted forward.

    The write must finish while `reserve_s` of `deadline` remains, so the work after it
    keeps its share. A write that runs past that is abandoned unconfirmed: its worker thread may still land it.

    Returns:
        `True` once the write is confirmed durable. `False` when the definition store itself
        is unreachable, or the write wasn't confirmed in time -- the caller still returns this
        call's own result rather than discarding it: the mint already ran, perhaps for most
        of the call's deadline, and produced a real, immediately usable frame, and refusing
        the whole call now would throw that away for a failure that has nothing to do with
        this request. The caller marks the digest instead, so the model is told the truth
        about reuse rather than nothing at all.

    Raises:
        IdentityMismatch: the manifest's own identity does not hash to the spec_id it claims.
            Not a storage failure, and not one a later call could route around either, so this
            propagates rather than being folded into the `False` case above.
    """
    canonical_bytes = verify_write(dict(definition.identity), definition.spec_id)
    stored = StoredDefinition(
        spec_id=definition.spec_id,
        schema_version=definition.schema_version,
        canonical_bytes=canonical_bytes,
    )
    try:
        async with asyncio.timeout(deadline.reserving(reserve_s).remaining()):
            await get_store().put_many([stored])
    except (DefinitionStoreUnavailable, TimeoutError):
        return False
    return True


def _artifact_title(artifact: ArtifactEntry) -> str:
    """A link title that names the artifact's content once, from its `rendering`."""
    rendering = artifact.rendering
    if statistics := rendering.get("statistics"):
        return f"{statistics} statistics artifact"
    if file_format := rendering.get("format"):
        return f"{file_format} artifact"
    return "artifact"


def _artifact_resource_links(manifest: Manifest) -> list[ResourceLink]:
    """One `ResourceLink` per artifact on `manifest`, from its own `resource_uri`.

    A model reads the link to pull the artifact's download URL when it needs
    one, so neither the result nor the manifest in structured content carries
    one. One link
    per artifact on the narrowed manifest (`_narrowed_to_rendering`): the
    requested rendering and its statistics.
    """
    links = []
    for artifact in manifest.artifacts:
        if artifact.resource_uri is None:
            # Unreachable via `strip_manifest`: `Manifest._attach_artifact_resource_uris`
            # always derives and sets this field once validation succeeds.
            raise AssertionError("artifact.resource_uri is unset after strip_manifest")
        links.append(
            ResourceLink(
                type="resource_link",
                uri=artifact.resource_uri,
                # Stable and unique within one manifest, unlike `title`.
                name=artifact.artifact_digest,
                title=_artifact_title(artifact),
                description=(
                    f"Read for the download URL ({artifact.bytes} bytes). "
                    "The URL does not expire; a GET on it redirects to the bytes."
                ),
                # The media type of the JSON document this link resolves to,
                # not the artifact's own `media_type`.
                mimeType="application/json",
                # `size` would mean the resource JSON's own byte count, not
                # the artifact's `bytes`, so it stays unset.
            )
        )
    return links


async def _fetch_statistics_for_digest(
    manifest: Manifest, *, deadline: CallDeadline
) -> tuple[list[SeriesStatisticsModel] | None, bool]:
    """The `(series_stats, statistics_fetch_failed)` pair `render_get_data_digest` renders from.

    Only fetches when `statistics_needs_fetch(manifest.series_count)` -- a
    count of `0`, past the digest's own cap, or `None` (an old record)
    renders from `manifest.series_count` alone and never touches the
    network. An artifact fetch or Parquet-decode failure degrades to
    `(None, True)`, logged here rather than raised: the mint itself already
    succeeded, and the frame reference `render_get_data_digest` renders
    stays real and usable either way.

    The fetch runs only while `_STATISTICS_FLOOR_S` of `deadline` remains, and
    signing plus GET must end with `_STATISTICS_RESERVE_S` still left; either
    bound degrades the same way a failed fetch does.
    """
    if not statistics_needs_fetch(manifest.series_count):
        return None, False
    assert manifest.series_count is not None  # guaranteed by `statistics_needs_fetch` above
    artifact = find_statistics_artifact(manifest)
    if artifact is None:
        logger.warning(
            "get_data: manifest for %s/%s names series_count=%d but carries no statistics artifact",
            manifest.definition.spec_id,
            manifest.content_version,
            manifest.series_count,
        )
        return None, True
    if (remaining_s := deadline.remaining()) < _STATISTICS_FLOOR_S:
        logger.warning(
            "get_data: skipping the statistics fetch for %s/%s with %.1fs of the call "
            "deadline left",
            manifest.definition.spec_id,
            manifest.content_version,
            remaining_s,
        )
        return None, True
    try:
        series_stats = await fetch_series_statistics(
            subject_id=manifest.definition.spec_id,
            content_version=manifest.content_version,
            artifact=artifact,
            expected_series=manifest.series_count,
            continuing=continuing_method(manifest),
            deadline=deadline.reserving(_STATISTICS_RESERVE_S),
        )
    except (
        ResourceError,
        SeriesStatisticsSchemaError,
        httpx2.HTTPError,
        pa.ArrowException,
        ValidationError,
        KeyError,
        TimeoutError,
    ) as exc:
        logger.warning(
            "get_data: could not fetch or decode the statistics artifact for %s/%s: %s",
            manifest.definition.spec_id,
            manifest.content_version,
            exc,
        )
        return None, True
    return series_stats, False


async def _fetch_warning_labels(manifest: Manifest, *, deadline: CallDeadline) -> dict[str, str]:
    """ref -> name for the refs `manifest`'s warnings name (`warning_refs`), read in one
    `node_lookup.describe_nodes` call, for `labelled_warnings`.

    The read must end with `_STATISTICS_RESERVE_S` of `deadline` still left. A failed or
    late read is logged and labels nothing, and a ref the graph does not hold or names
    nothing gets no label, so the digest names those refs bare.
    """
    refs = warning_refs(manifest.warnings)
    if not refs:
        return {}
    dcids = {ref: ref.removeprefix(ENTITY_DCID_MARKER) for ref in refs}
    try:
        async with asyncio.timeout(deadline.reserving(_STATISTICS_RESERVE_S).remaining()):
            outcomes = await node_lookup.describe_nodes(list(dcids.values()))
    except (UpstreamError, ValueError, TimeoutError) as exc:
        # `ValueError` covers a body off the wire models (`pydantic.ValidationError`)
        # and one answering other dcids than were asked.
        logger.warning(
            "get_data: could not label the warning refs of %s/%s: %s",
            manifest.definition.spec_id,
            manifest.content_version,
            exc,
        )
        return {}
    labels: dict[str, str] = {}
    for ref, dcid in dcids.items():
        node = outcomes.get(dcid)
        if node is not None and not isinstance(node, NotHeldEntry) and node.label:
            labels[ref] = node.label
    return labels


def _mint_json_or_refusal(response: httpx2.Response) -> dict[str, Any] | ToolResult:
    """The mint response's parsed JSON body, or the refusal for one that does not parse."""
    try:
        return response.json()
    except ValueError as exc:
        # `httpx2.Response.json()` raises `json.JSONDecodeError` for malformed JSON and
        # `UnicodeDecodeError` for a body that is invalid UTF-8. Both are `ValueError`
        # subclasses and neither subclasses the other, so catching `ValueError` covers both.
        return to_tool_result(
            ToolRefusal.upstream_response_unusable(
                detail=f"the mint response body could not be parsed as JSON: {exc}"
            )
        )


async def _mint_payload_or_refusal(
    response: httpx2.Response,
    *,
    client: UpstreamClient,
    settings: Settings,
    deadline: CallDeadline,
    on_poll: OnPoll | None = None,
    raise_upstream_error: bool = False,
) -> tuple[dict[str, Any], OperationPollNote | None] | ToolResult:
    """The mint route's manifest payload, resolving a 202 operation to completion first.

    The mint POST answers 200 with the manifest directly for a request within
    apis' own sync budget, or 202 with an operation envelope once that budget
    is crossed (`datasets_api.operations.submit`). This is the one place that tells
    the two apart: a 200 body is returned as-is; a 202 body is handed to
    `data_mcp.operations.resolve_operation_response`, which polls the named
    operation to `done` (or refuses) within what remains of the call's `deadline`.
    A mint POST that ran long leaves the poll correspondingly less.

    Args:
        response: The mint POST's own response.
        client: The same upstream client the mint POST itself used.
        settings: This service's own settings.
        deadline: The `get_data` call's deadline.
        on_poll: Forwarded to `resolve_operation_response` unchanged; `None` for
            an ordinary 200 mint, which never polls at all.

    Returns:
        `(manifest_payload, operation_poll_note)` on success -- `operation_poll_note`
        is `None` for an ordinary 200 mint, and set for a mint that had to be polled,
        for `_finish_mint_response` to forward into the digest. A refusal
        `ToolResult` for a non-JSON body or an unresolved/failed operation.
    """
    payload = _mint_json_or_refusal(response)
    if isinstance(payload, ToolResult):
        return payload
    if response.status_code != 202:
        return payload, None
    resolved = await resolve_operation_response(
        client=client,
        submit_payload=payload,
        deadline=deadline,
        poll_interval_s=settings.operation_poll_interval_s,
        on_poll=on_poll,
        raise_upstream_error=raise_upstream_error,
    )
    if isinstance(resolved, ToolResult):
        return resolved
    return resolved.manifest_payload, OperationPollNote(
        operation_id=resolved.operation_id,
        poll_attempts=resolved.poll_attempts,
        elapsed_s=resolved.elapsed_s,
    )


async def _finish_mint_response(
    payload: Mapping[str, Any],
    *,
    requested_spec_id: str,
    options: Mapping[str, Any],
    scope: Mapping[str, Any],
    entity_labels: Mapping[str, str],
    dropped_roles: Sequence[str],
    settings: Settings,
    deadline: CallDeadline,
    asked_for_kinds: bool = False,
    operation_poll: OperationPollNote | None = None,
    breakdown: BreakdownLevel | None = None,
) -> ToolResult:
    """Turn a mint route's manifest payload into the tool's result, or a refusal.

    A manifest that no longer matches this service's own model (schema drift) would
    otherwise raise straight out of
    `get_data`. fastmcp's own dispatch loop re-raises a bare `pydantic.ValidationError`
    unwrapped rather than folding it into a `ToolError`, so an unguarded failure here
    escapes as a raw exception carrying none of this module's refusal fields, after a
    non-retryable mint, and perhaps an operation poll, that spent most of the call's
    deadline.

    When the manifest's own `spec_id` differs from `requested_spec_id`, this call re-minted
    under a new identity; see `_persist_definition` for why that new definition gets
    written to the store here, and what happens when the write itself fails.

    Once the manifest itself validates, it is narrowed to the requested rendering
    (`_narrowed_to_rendering`), and `materialise_and_register` registers the frame in
    the shared materialisation store under the frame reference this digest hands the
    model (`frame_materialisation.frame_ref_for`): for a default mint the version's
    canonical frame, read off the snapshot route, and otherwise the artifact whose
    rendering equals `options`. A frame minted with `options.columns` or
    `options.groupby` has a `content_version` of its own, and the manifest keeps
    apis' pair, which its artifacts are read under. A failed fetch or an unconfirmed
    store write degrades this call's own digest rather than refusing it (see
    `render_get_data_digest`'s `degraded_frame`); a genuinely unusable artifact or
    manifest still refuses.

    Args:
        payload: The mint route's own manifest payload -- either an ordinary 200
            response's body, or the manifest `_mint_payload_or_refusal` polled out of
            a 202 operation.
        requested_spec_id: The `spec_id` this call's own request carried, to detect a re-mint.
        options: The mint request's `options`, which name the artifact the frame is
            read from and, through `options.columns` and `options.groupby`, its frame
            reference.
        scope: The mint request's `scope`, kept with `options` in the frame's mint
            record (`common.materialisation.MintRecord`).
        entity_labels: dcid -> name, forwarded to the digest renderer.
        dropped_roles: Roles silently left unbound, forwarded to the digest
            renderer.
        settings: This service's own settings, threaded to `materialise_and_register` for
            its row-count ceiling and fetch wall-clock budget.
        deadline: The `get_data` call's deadline, which cuts the re-mint persist's, the
            frame fetch's, the statistics fetch's and the description read's budgets.
        asked_for_kinds: Whether the call asked for the party kinds, by `place_types` or
            `without_party_totals`. A manifest naming a flow side, minted without them, refuses
            `upstream_response_unusable`: a flow frame without kinds would serve every
            party total beside its parties.
        operation_poll: Set when `payload` was resolved from a 202 operation rather
            than an ordinary 200 mint. Forwarded to the digest renderer.
        breakdown: The code system and level this mint summed within, forwarded to
            the digest renderer.

    Returns:
        A successful `ToolResult` carrying the digest and stripped manifest,
        plus the frame's reference as `frame_ref`, and its `role_keys` and
        `stat_var_count` whenever
        `materialise_and_register` read or parsed the frame, and
        `summed_stat_var_count` and `summed_stat_var_label`
        (`get_data_digest.summed_family`), or a refusal
        `ToolResult` if the response couldn't be turned into either.
    """
    try:
        manifest = strip_manifest(payload)
    except ValidationError as exc:
        return to_tool_result(
            ToolRefusal.upstream_response_unusable(
                detail="the mint response did not match the manifest shape this service "
                f"expects: {render_validation_errors(exc)}"
            )
        )
    manifest = _narrowed_to_rendering(manifest, options=options)
    if (
        asked_for_kinds
        and place_type_columns_for(manifest.definition.identity) == FLOW_ENRICHMENT_COLUMNS
        and not set(KIND_COLUMNS) <= set(options.get("columns") or ())
    ):
        return to_tool_result(
            ToolRefusal.upstream_response_unusable(
                detail=f"{manifest.definition.spec_id} names a flow side, and this call asked for "
                "party kinds, but the mint was sent without them; the frame would serve every "
                "group total beside its parties"
            )
        )

    remint_persist_failed = False
    if manifest.definition.spec_id != requested_spec_id:
        try:
            persisted = await _persist_definition(
                manifest.definition,
                deadline=deadline,
                reserve_s=current_call_budget().post_mint_reserve_s,
            )
        except IdentityMismatch as exc:
            return to_tool_result(ToolRefusal.capability_unavailable(detail=str(exc)))
        remint_persist_failed = not persisted

    outcome = await materialise_and_register(
        manifest=manifest,
        settings=settings,
        writer=current_writer(),
        options=options,
        scope=scope,
        deadline=deadline,
    )
    if outcome.status == "refused":
        assert outcome.refusal is not None
        return to_tool_result(outcome.refusal)
    frame_ref = outcome.frame_ref
    assert frame_ref is not None  # set on every outcome but a refusal

    # apis' pair, not `frame_ref`, keys the description: a frame minted with
    # `options.columns` or `options.groupby` is held under a pair of its own that apis
    # never stored.
    (series_stats, statistics_fetch_failed), description, warning_labels = await asyncio.gather(
        _fetch_statistics_for_digest(manifest, deadline=deadline),
        fetch_version_description(
            spec_id=manifest.definition.spec_id,
            content_version=manifest.content_version,
            deadline=deadline.reserving(_STATISTICS_RESERVE_S),
        ),
        _fetch_warning_labels(manifest, deadline=deadline),
    )
    summed = summed_family(
        manifest,
        frame_summed=outcome.family_summed,
        series_stats=series_stats,
        description=description,
    )

    # The digest and the returned manifest carry the same labelled warning messages.
    manifest = manifest.model_copy(
        update={"warnings": labelled_warnings(manifest.warnings, warning_labels)}
    )
    try:
        content = render_get_data_digest(
            manifest=manifest,
            frame_ref=frame_ref,
            entity_labels=entity_labels,
            dropped_roles=dropped_roles,
            remint_persist_failed=remint_persist_failed,
            degraded_frame=outcome if outcome.status == "degraded" else None,
            series_stats=series_stats,
            statistics_fetch_failed=statistics_fetch_failed,
            operation_poll=operation_poll,
            breakdown=breakdown,
            description=description,
            summed=summed,
        )
    except ValueError as exc:
        return to_tool_result(ToolRefusal.upstream_response_unusable(detail=str(exc)))
    # The harness reads the stripped manifest and the frame's role keys and StatVar
    # count from structured content. The combined server drops structured content from
    # a successful result, so the digest must stand alone for an outside model.
    # A long `trace.resolved_stat_vars` is left out, so the result's size does not
    # grow with the family's (`data_mcp.stat_var_list`).
    structured_content = without_long_stat_var_list(manifest.model_dump(mode="json"))
    # The frame's reference sits beside the manifest: the manifest's own pair names
    # apis' materialisation, which the statistics artifact is read under, and
    # this one names the frame the store holds for this rendering.
    structured_content["frame_ref"] = frame_ref.model_dump(mode="json")
    if outcome.role_keys is not None:
        # The frame's own role columns and distinct StatVar count sit beside the
        # manifest, which is apis' document; these come from the frame this call
        # read. The harness binds a plan's role columns to the first and bounds each
        # later step's declared StatVar count by the second (`compiler.frame_columns`).
        structured_content["role_keys"] = list(outcome.role_keys)
        structured_content["stat_var_count"] = outcome.stat_var_count
    structured_content["summed_stat_var_count"] = None if summed is None else summed.count
    structured_content["summed_stat_var_label"] = None if summed is None else summed.label
    return ToolResult(
        # The digest text stays first, so `text_of(result)` -- which reads
        # `content[0]` -- keeps working; the resource links follow it.
        content=[TextContent(type="text", text=content), *_artifact_resource_links(manifest)],
        structured_content=structured_content,
        is_error=False,
    )


def _narrowed_to_rendering(manifest: Manifest, *, options: Mapping[str, Any]) -> Manifest:
    """`manifest` listing only the artifact whose `rendering` equals `options`, and the
    statistics artifact.

    A manifest lists every rendering apis stores under its pair, so other callers'
    groupby sums and column selections sit beside the requested one, in no set order.
    The resource links and the harness's freshness date
    (`harness.provenance.data_artifact`) read the narrowed list, and so does the frame
    fetch of a mint with columns or a groupby. A manifest without the requested
    rendering keeps only its statistics, and `materialise_and_register` refuses such a
    mint unless the store already holds the frame. A default mint's frame is the
    version's canonical frame, which the snapshot route lists and no mint response
    does.
    """
    statistics = find_statistics_artifact(manifest)
    artifacts = [a for a in manifest.artifacts if a.rendering == options or a is statistics]
    return manifest.model_copy(update={"artifacts": artifacts})


async def _post_mint(
    client: UpstreamClient,
    request_body: Mapping[str, Any],
    *,
    spec_id: str,
    settings: Settings,
    deadline: CallDeadline,
    admission: Admission = None,
    on_sent: Callable[[], None] | None = None,
) -> httpx2.Response | ToolResult:
    """POST one mint request inside what `deadline` leaves it (`_send_mint`), unless
    this process already holds a POST of the same request open. A refusal from
    `_send_mint` comes back as its `ToolResult`.

    apis runs two identical synchronous mints in flight at once as two mints. A
    background POST whose request body matches an open POST's therefore sends
    nothing and answers at once with `background_result` outcome `"joined"`: the open
    POST mints the frame. A foreground POST waits for the open POST to end, while its
    deadline leaves more than `post_mint_reserve_s`, `min_mint_timeout_s` and
    `_MINT_WAIT_SLACK_S`, and then posts, finding the frame the open POST built. The
    harness posts a card's mints in the background before the user approves it
    (`harness.early_mint`), and the run then asks for the same frames in the
    foreground.

    Every POST is held open per process, in `_open_mint_posts`, from before it is
    sent until it ends, answered, failed or cancelled. The dedupe therefore holds
    where the harness reaches data-mcp in-process, as it does over its loopback
    mount, and two processes can still mint the same request twice.
    """
    request_key = json.dumps(request_body, sort_keys=True)
    if (open_post := _open_mint_posts.get(request_key)) is not None:
        if admission is not None:
            return background_result(spec_id=spec_id, outcome="joined")
        budget = current_call_budget()
        waitable_s = deadline.reserving(
            budget.post_mint_reserve_s + budget.min_mint_timeout_s + _MINT_WAIT_SLACK_S
        ).remaining()
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(open_post.wait(), timeout=max(waitable_s, 0.0))
    ended = _open_mint_posts[request_key] = asyncio.Event()
    try:
        sent = await _send_mint(
            client,
            request_body,
            request_key=request_key,
            spec_id=spec_id,
            settings=settings,
            deadline=deadline,
            admission=admission,
            on_sent=on_sent,
        )
    finally:
        ended.set()
        if _open_mint_posts.get(request_key) is ended:
            del _open_mint_posts[request_key]
    return to_tool_result(sent) if isinstance(sent, ToolRefusal) else sent


async def _send_mint(
    client: UpstreamClient,
    request_body: Mapping[str, Any],
    *,
    request_key: str,
    spec_id: str,
    settings: Settings,
    deadline: CallDeadline,
    admission: Admission,
    on_sent: Callable[[], None] | None,
) -> httpx2.Response | ToolRefusal:
    """POST one mint request inside what `deadline` leaves it. `request_key` is
    `request_body` as the ledger keys it.

    A `"background"` `admission` sends `MINT_ADMISSION_HEADER` on the POST
    (`data_mcp.mint_admission`), so apis admits the mint only while it has room.
    `on_sent` is called once the request has been written out
    (`UpstreamClient.request`).

    The current `CallBudget` sizes it: the POST gets `mint_timeout_s` at most and must
    end with `post_mint_reserve_s` of `deadline` left. With less than
    `min_mint_timeout_s` to offer it, this refuses without sending, so nothing starts
    upstream. An apis ID token that cannot be minted refuses `service_unavailable`
    the same way, with nothing sent. Any POST that gets no response (whether
    `mint_timeout_s` elapses first, the deadline cuts it, or the connection fails
    outright, the same `UpstreamTransportError` each way) refuses with
    `mint_unanswered`: apis does not cancel a mint whose client went away, so it
    may still be running it.

    Under a budget that does not wait out apis' synchronous mint, the POST first
    takes a slot in `_mint_ledger`, or returns the ledger's refusal without sending.
    The slot is freed when apis answers, and otherwise held until apis has
    stopped running the mint, including when the call is cancelled while it waits.

    Not retried: a timed-out mint upstream refuses cancellation, so a second POST
    could start the same work twice.

    Raises:
        UpstreamError: any other mint failure, left to the caller's own mapping.
    """
    budget = current_call_budget()
    mint_deadline = deadline.reserving(budget.post_mint_reserve_s)
    timeout_s = min(budget.mint_timeout_s, mint_deadline.remaining())
    if timeout_s < budget.min_mint_timeout_s:
        return ToolRefusal.service_unavailable(
            service="dfo-apis",
            detail=f"only {max(timeout_s, 0.0):.0f}s of this call's deadline remained for "
            "the mint, too little to start it; nothing was sent",
            retry_after_s=DEADLINE_RETRY_AFTER_S,
        )
    started_at = deadline.clock()
    held: HeldMint | None = None
    if not budget.waits_out_sync_mint:
        admitted = _mint_ledger.admit(
            request_key=request_key,
            spec_id=spec_id,
            now=started_at,
            answer_by=mint_deadline.expires_at,
            limit=settings.external_mint_concurrency,
        )
        if isinstance(admitted, ToolRefusal):
            return admitted
        held = admitted
    try:
        response = await client.request(
            "POST",
            "/api/datasets",
            route="/api/datasets",
            json=request_body,
            retry=False,
            timeout=timeout_s,
            deadline=mint_deadline,
            headers=admission_headers(admission),
            on_sent=on_sent,
        )
    except UpstreamAuthUnavailable as exc:
        if held is not None:
            _mint_ledger.answered(held)
        return ToolRefusal.service_unavailable(
            service="dfo-apis",
            detail=f"{exc.detail}; nothing was sent",
            retry_after_s=DEADLINE_RETRY_AFTER_S,
        )
    except UpstreamTransportError:
        now = deadline.clock()
        return ToolRefusal.mint_unanswered(
            spec_id=spec_id,
            elapsed_s=now - started_at,
            retry_after_s=DEADLINE_RETRY_AFTER_S if held is None else held.expires_at - now,
            waits_out_sync_mint=budget.waits_out_sync_mint,
        )
    except UpstreamError:
        if held is not None:
            _mint_ledger.answered(held)
        raise
    if held is not None:
        _mint_ledger.answered(held)
    return response


async def _attempt_mint(
    *,
    client: UpstreamClient,
    spec_id: str,
    identity: Mapping[str, Any] | None,
    sends_definition: bool,
    wire_scope: Mapping[str, Any],
    stat_vars: StatVarsReading,
    breakdown: Breakdown | None,
    place_types: bool,
    without_party_totals: bool,
    entity_labels: Mapping[str, str],
    dropped_roles: Sequence[str],
    settings: Settings,
    deadline: CallDeadline,
    on_poll: OnPoll | None = None,
    admission: Admission = None,
    on_sent: Callable[[], None] | None = None,
) -> ToolResult:
    """POST the `spec_id` mint and turn the result into this tool's response.

    A background `admission` goes to the mint POST alone (`_post_mint`), and the
    mint ends on apis' answer to it, with `background_result`, with no poll and
    no frame read, or at once, outcome `"joined"`, when a POST of the same request
    is already open. A background mint apis declines refuses with the decline named in its
    `fix` (`data_mcp.mint_admission.declined_in_background`). `on_sent` hears each
    mint POST written out.

    `identity` is `None` on the registry-fallback path, where the request omits the
    `definition` sidecar and a `spec_not_found` refusal completes `definition_not_held`.
    An identity read from the registry (`_registry_identity_or_refuse`) chooses the
    columns, and goes in the sidecar only when `sends_definition`.
    `stat_vars="sum"` adds `_SUM_STAT_VARS_GROUPBY` to the request's options. The sum
    sits outside the spec's identity: the manifest keeps `spec_id`, and apis mints
    the summed frame its own `content_version` whenever more than one StatVar
    contributed rows, so the summed and separate frames never share a frame reference.
    With one contributor the two frames are the same bytes under one `content_version`.
    A `breakdown` adds its own `Breakdown.groupby` instead, which sums the StatVars
    within each code at the breakdown's level.
    `place_types` sets `options.columns` to the columns
    `frame_materialisation.place_type_columns_for` picks for `identity`, which adds them to
    the default columns, and gives the frame a `content_version` of its own
    (`frame_materialisation.frame_ref_for`). A `column_unknown` refusal of the place type
    column from a table without one is sent once more with no columns
    (`_without_unserved_place_type`). A refusal of a flow table's `party_kind` is relayed: a
    flow frame without kinds would let a sum count party totals beside their parties.
    `without_party_totals` on a flow spec asks for the kinds alone (`KIND_COLUMNS`) when
    `place_types` is not set, and sends `options.without_party_totals`: apis leaves out the group
    totals of the
    party roles the identity it serves leaves open, and mints the frame that remains a
    version of its own, which the manifest's pair names and its trace lists the totals
    left out of (`party_totals_left_out`). On any other spec it changes nothing.
    """
    mint_input: dict[str, Any] = {
        "kind": "spec_id",
        "spec_id": spec_id,
        "scope": dict(wire_scope),
    }
    if identity is not None and sends_definition:
        # Sent whenever the definition is held or read from the registry under this id;
        # omitted on the registry-fallback path so `mint_spec_id` is free to serve
        # straight from its own registry on a hit.
        mint_input["definition"] = {
            "schema_version": _SCHEMA_VERSION,
            "spec_id": spec_id,
            "identity": identity,
        }

    options: dict[str, Any] = dict(DEFAULT_MINT_OPTIONS)
    if breakdown is not None:
        options["groupby"] = breakdown.groupby()
    elif stat_vars == "sum":
        options["groupby"] = _SUM_STAT_VARS_GROUPBY
    drops_totals = (
        without_party_totals and place_type_columns_for(identity) == FLOW_ENRICHMENT_COLUMNS
    )
    if place_types:
        options["columns"] = list(place_type_columns_for(identity))
    elif drops_totals:
        options["columns"] = list(KIND_COLUMNS)
    if drops_totals:
        options[WITHOUT_PARTY_TOTALS] = True
    request_body = {
        "schema_version": _SCHEMA_VERSION,
        "input": mint_input,
        "options": options,
    }

    def minted(result: ToolResult) -> ToolResult:
        """A `default_facet_undefined` refusal, from the mint or its poll, naming the
        spec_id and places this call was minting. The cause and retry flag stay apis'."""
        refused = result.structured_content or {}
        if refused.get("code") != ToolErrorCode.DEFAULT_FACET_UNDEFINED.value:
            return result
        entities = wire_scope.get("entities", (identity or {}).get("entities") or {})
        return to_tool_result(
            ToolRefusal.default_facet_undefined(
                issues=facet_issues_from(refused["fix"]),
                detail=refused["fix"]["detail"],
                retryable=refused["fix"]["retryable"],
                spec_id=spec_id,
                entities=entities,
                places=_places_named(
                    [ref for refs in entities.values() for ref in refs], entity_labels
                ),
                capability_version=refused.get("capability_version"),
            )
        )

    removable = set(place_type_columns_for(identity)) if place_types else set()
    for attempt in range(2):
        try:
            posted = await _post_mint(
                client,
                request_body,
                spec_id=spec_id,
                settings=settings,
                deadline=deadline,
                admission=admission,
                on_sent=on_sent,
            )
            if isinstance(posted, ToolResult):
                return posted
            if admission is not None:
                # A background caller needs only the mint started, so it neither polls nor reads a frame.
                outcome = "queued" if posted.status_code == HTTPStatus.ACCEPTED else "minted"
                return background_result(spec_id=spec_id, outcome=outcome)
            resolved = await _mint_payload_or_refusal(
                posted,
                client=client,
                settings=settings,
                deadline=deadline,
                on_poll=on_poll,
                raise_upstream_error=True,
            )
        except UpstreamError as exc:
            if attempt == 0 and _negotiate_unserved_columns(
                options=options, refused=exc, removable=removable, spec_id=spec_id
            ):
                request_body["options"] = options
                continue
            if not sends_definition and exc.code == "spec_not_found":
                # The sidecar-free registry-fallback request above just missed
                # apis' own registry too: `mint_spec_id` raises exactly this
                # code in that case, and the platform contract's
                # `definition_not_held` -- "held by neither store nor registry"
                # -- is what this completes.
                return to_tool_result(ToolRefusal.definition_not_held(definition_id=spec_id))
            refusal = _rebase_entities_role_pointer(from_upstream_error(exc))
            if admission is not None:
                refusal = declined_in_background(refusal, exc)
            return minted(to_tool_result(refusal))
        if isinstance(resolved, ToolResult):
            return minted(resolved)
        payload, operation_poll = resolved
        return await _finish_mint_response(
            payload,
            requested_spec_id=spec_id,
            options=options,
            scope=wire_scope,
            entity_labels=entity_labels,
            dropped_roles=dropped_roles,
            settings=settings,
            deadline=deadline,
            asked_for_kinds=place_types or without_party_totals,
            operation_poll=operation_poll,
            breakdown=(
                None
                if breakdown is None
                else BreakdownLevel(code_system=breakdown.code_system, level=breakdown.level)
            ),
        )
    raise AssertionError("two mint attempts must return or raise")


def _without_unserved_place_type(
    options: dict[str, Any],
    refused: UpstreamError,
    *,
    removable: set[str],
    spec_id: str,
) -> bool:
    """Drop `options.columns` when it asks for the place type column alone and a
    `column_unknown` refusal lists that column as unserved, and say whether it did. A
    refusal of any other code or request, or one whose list is truncated, changes
    nothing.
    """
    alternatives = refused.extensions.get("alternatives")
    if (
        refusal_code(refused) != "column_unknown"
        or refused.extensions.get("pointer") != "/options/columns"
        or options.get("columns") != [_PLACE_TYPE_COLUMN]
        or _PLACE_TYPE_COLUMN not in removable
        or refused.extensions.get("truncated")
        or not isinstance(alternatives, list)
        or not all(
            isinstance(item, dict) and isinstance(item.get("ref"), str) for item in alternatives
        )
    ):
        return False
    served = {alternative["ref"] for alternative in alternatives}
    if _PLACE_TYPE_COLUMN in served:
        return False
    logger.info(
        "get_data: %s serves no %s; minting again with the default columns",
        spec_id,
        _PLACE_TYPE_COLUMN,
    )
    del options["columns"]
    return True


def _without_unserved_flow_types(
    options: dict[str, Any], refused: UpstreamError, *, removable: set[str]
) -> bool:
    """Remove policy-added types for absent flow roles after a complete column listing."""
    alternatives = refused.extensions.get("alternatives")
    if (
        refusal_code(refused) != "column_unknown"
        or refused.extensions.get("pointer") != "/options/columns"
        or refused.extensions.get("truncated")
        or not isinstance(alternatives, list)
        or not all(
            isinstance(item, dict) and isinstance(item.get("ref"), str) for item in alternatives
        )
    ):
        return False
    served = {item["ref"] for item in alternatives}
    columns = options.get("columns")
    if not isinstance(columns, list):
        return False
    requested_types = {token for token in removable & set(columns) if token.endswith("_type")}
    if any(
        any(ref.startswith(f"{token.removesuffix('_type')}_") for ref in served)
        and token not in served
        for token in requested_types
    ):
        return False
    absent = {
        token
        for token in removable & set(columns)
        if token.endswith("_type")
        and not any(ref.startswith(f"{token.removesuffix('_type')}_") for ref in served)
    }
    remaining = [column for column in columns if column not in absent]
    remaining_types = {token for token in remaining if token.endswith("_type")}
    if not absent or not remaining_types:
        return False
    for role in PARTY_ROLES:
        token = type_column(role)
        if token in remaining and f"{token.removesuffix('_type')}_party_kind" not in served:
            return False
    options["columns"] = remaining
    return True


def _negotiate_unserved_columns(
    *,
    options: dict[str, Any],
    removable: set[str],
    spec_id: str,
    refused: UpstreamError,
) -> bool:
    """Narrow policy-added metadata after one complete column refusal."""
    return _without_unserved_place_type(
        options, refused, removable=removable, spec_id=spec_id
    ) or _without_unserved_flow_types(options, refused, removable=removable)


@dataclass(frozen=True, kw_only=True)
class RootMint:
    """A root frame `mint_recorded_root` confirmed in the materialisation store.

    `frame` is the pair the store holds it under, `dataset` the pair apis names
    the materialisation by, `options` the mint options sent, and `stat_var_count` the
    distinct StatVars its rows hold.
    """

    frame: FrameRef
    dataset: FrameRef
    options: dict[str, Any]
    stat_var_count: int


async def mint_recorded_root(
    record: MintRecord, *, deadline: CallDeadline, settings: Settings | None = None
) -> RootMint | QueuedOperation | ToolResult:
    """Mint the dataset `record` names again, with the `options` and `scope` it
    records, and register the current frame of that rendering in the store.
    Recorded `options` are first mapped to what `get_data` sends
    (`frame_materialisation.replayed_mint_options`): a flow root recorded with its
    parties' types or with no columns is minted with their kinds, and comes back
    under a new pair.

    This is `get_data`'s mint without its model-facing parts. No entity name is
    resolved, since a recorded scope already holds dcids, and no digest, statistics
    or resource link is built. The request is `get_data`'s own: `record.dataset
    .spec_id` with `record.scope`, the stored definition beside it when the
    definition store holds one, and those `options`. It goes through
    `_post_mint`, so the current `CallBudget` sizes the POST and the mint ledger
    admits it, and `materialise_and_register` registers the frame and writes the
    mint record of its current pair.

    It does not wait on a queued mint. A 202 whose operation is unfinished returns
    the `QueuedOperation` at once (`operations.operation_without_waiting`). The
    caller sends the same mint again later, and apis answers with the same
    operation, or with the manifest once it is done, so a repeat submits nothing new.

    A re-mint under a new identity is not written to the definition store. The
    returned `dataset.spec_id` then differs from `record.dataset.spec_id`, and the
    caller decides what that means.

    Args:
        record: The mint record of the root to mint again.
        deadline: Bounds every step, as a `get_data` call's deadline does.
        settings: This service's settings, read from the environment when `None`.

    Returns:
        A `RootMint` once the store holds the frame. A `QueuedOperation` for a mint
        apis queued and has not finished. A refusal `ToolResult` otherwise,
        which is `service_unavailable` with a wait when the mint succeeded and the
        frame could not be fetched or stored inside `deadline`.
    """
    settings = settings or Settings()
    client = get_client()
    minted_response = await _mint_recorded_response(
        client, record, settings=settings, deadline=deadline
    )
    if isinstance(minted_response, ToolResult):
        return minted_response
    payload, options = minted_response
    if isinstance(payload, QueuedOperation | ToolResult):
        return payload
    try:
        manifest = strip_manifest(payload)
    except ValidationError as exc:
        return to_tool_result(
            ToolRefusal.upstream_response_unusable(
                detail="the mint response did not match the manifest shape this service "
                f"expects: {render_validation_errors(exc)}"
            )
        )

    outcome = await materialise_and_register(
        manifest=_narrowed_to_rendering(manifest, options=options),
        settings=settings,
        writer=current_writer(),
        options=options,
        scope=record.scope,
        deadline=deadline,
    )
    if outcome.status == "refused":
        assert outcome.refusal is not None
        return to_tool_result(outcome.refusal)
    if outcome.status == "degraded":
        return to_tool_result(
            ToolRefusal.service_unavailable(
                service="the frame store",
                detail="the dataset was minted, and its frame could not be fetched and "
                f"stored inside this call's deadline ({outcome.degraded_reason})",
                retry_after_s=DEADLINE_RETRY_AFTER_S,
            )
        )
    assert outcome.frame_ref is not None  # set on every outcome but a refusal
    assert outcome.stat_var_count is not None  # set on every registered outcome
    return RootMint(
        frame=outcome.frame_ref,
        dataset=FrameRef(
            spec_id=manifest.definition.spec_id, content_version=manifest.content_version
        ),
        options=dict(options),
        stat_var_count=outcome.stat_var_count,
    )


async def recorded_mint_request(
    record: MintRecord, *, deadline: CallDeadline
) -> tuple[dict[str, Any] | None, Mapping[str, Any]] | ToolResult:
    """The stored definition identity of `record`'s dataset, `None` when the definition
    store holds none, and the options a mint of `record` again sends: the recorded ones
    mapped for that identity (`replayed_mint_options`). A recorded
    `options.without_party_totals` is sent as recorded.

    `mint_recorded_root` sends these options, and `harness.live_refresh` reads them to
    decide whether a root is named under an enrichment stamp, so the two cannot differ.

    Returns:
        The identity and the options, or the refusal `ToolResult` of a definition read
        that failed (`_load_identity_or_refuse`).
    """
    identity = await _load_identity_or_refuse(record.dataset.spec_id, deadline=deadline)
    if isinstance(identity, ToolResult):
        return identity
    return identity, replayed_mint_options(record.options, identity=identity)


async def _mint_recorded_response(
    client: UpstreamClient,
    record: MintRecord,
    *,
    settings: Settings,
    deadline: CallDeadline,
) -> tuple[dict[str, Any] | QueuedOperation, Mapping[str, Any]] | ToolResult:
    """Mint `record` with the request `recorded_mint_request` builds and resolve one
    already-terminal queued response without waiting.

    As `_attempt_mint` does, a single column correction can span the POST and an
    immediately terminal operation poll. The mint carries the stored definition beside
    the `spec_id` when the definition store holds one."""
    spec_id = record.dataset.spec_id
    request = await recorded_mint_request(record, deadline=deadline)
    if isinstance(request, ToolResult):
        return request
    identity, replayed = request
    options = dict(replayed)
    recorded_columns = {str(column) for column in record.options.get("columns") or ()}
    if "kind" in recorded_columns:
        recorded_columns.remove("kind")
        recorded_columns.add("party_kind")
    removable: set[str] = {
        str(column) for column in options.get("columns") or ()
    } - recorded_columns
    mint_input: dict[str, Any] = {"kind": "spec_id", "spec_id": spec_id, "scope": record.scope}
    if identity is not None:
        mint_input["definition"] = {
            "schema_version": _SCHEMA_VERSION,
            "spec_id": spec_id,
            "identity": identity,
        }
    request_body = {
        "schema_version": _SCHEMA_VERSION,
        "input": mint_input,
        "options": options,
    }
    for attempt in range(2):
        try:
            posted = await _post_mint(
                client, request_body, spec_id=spec_id, settings=settings, deadline=deadline
            )
            if isinstance(posted, ToolResult):
                return posted
            payload = await _manifest_payload_without_waiting(
                posted,
                client=client,
                settings=settings,
                deadline=deadline,
                raise_upstream_error=True,
            )
        except UpstreamError as exc:
            if attempt == 0 and _negotiate_unserved_columns(
                options=options, refused=exc, removable=removable, spec_id=spec_id
            ):
                request_body["options"] = options
                continue
            if identity is None and exc.code == "spec_not_found":
                return to_tool_result(ToolRefusal.definition_not_held(definition_id=spec_id))
            return to_tool_result(from_upstream_error(exc))
        if isinstance(payload, ToolResult):
            return payload
        return payload, options
    raise AssertionError("two recorded mint attempts must return or raise")


async def _manifest_payload_without_waiting(
    response: httpx2.Response,
    *,
    client: UpstreamClient,
    settings: Settings,
    deadline: CallDeadline,
    raise_upstream_error: bool = False,
) -> dict[str, Any] | QueuedOperation | ToolResult:
    """The mint route's manifest payload, or the operation a 202 queued and has not
    finished. `_mint_payload_or_refusal` for a caller that does not wait."""
    payload = _mint_json_or_refusal(response)
    if isinstance(payload, ToolResult) or response.status_code != 202:
        return payload
    resolved = await operation_without_waiting(
        client=client,
        submit_payload=payload,
        deadline=deadline,
        poll_interval_s=settings.operation_poll_interval_s,
        raise_upstream_error=raise_upstream_error,
    )
    return resolved.manifest_payload if isinstance(resolved, ResolvedOperation) else resolved


# An operation's status as a Studio user reads it in the Activity feed.
_POLL_STATUS_WORDS = {
    "pending": "queued",
    "running": "in progress",
    "done": "ready",
    "failed": "failed",
}


def _report_progress(ctx: Context) -> OnPoll:
    """An `OnPoll` that relays each `data_mcp.operations.PollUpdate` as an MCP
    progress notification.

    `progress` is elapsed seconds: an operation's own remaining time isn't known
    upstream, so there's nothing to report as `total`. The message is shown to a
    Studio user as it stands, so it names no service and no raw status.
    `Context.report_progress` is a no-op when this call carried no progress
    token, and `resolve_operation_response`'s own `on_poll` wrapper already logs
    and swallows anything this raises -- a Studio user who can't see progress
    never turns into a failed mint.
    """

    async def on_poll(update: PollUpdate) -> None:
        status = _POLL_STATUS_WORDS[update.status]
        message = f"Preparing the dataset ({status}, {update.elapsed_s:.0f}s)"
        await ctx.report_progress(progress=update.elapsed_s, total=None, message=message)

    return on_poll


def _places_named(refs: Sequence[str], entity_labels: Mapping[str, str]) -> str:
    """`refs` for a refusal message, each named by its label, the rest only counted.

    A model repeats this message to the user, so a place this call holds no label
    for, such as one the spec's identity binds, is counted and never shown as its dcid.
    """
    named = [entity_labels[ref] for ref in refs if ref in entity_labels]
    unnamed = len(refs) - len(named)
    if not unnamed:
        return render_dcids(named, {})
    noun = "place" if unnamed == 1 else "places"
    return f"{render_dcids(named, {})} and {unnamed} more {noun}" if named else f"{unnamed} {noun}"


async def get_data(
    spec_id: Annotated[
        str,
        Field(
            pattern=_SPEC_ID_PATTERN,
            description="The spec_id from a prior `search` digest, or from this tool's own frame reference.",
        ),
    ],
    scope: Annotated[
        GetDataScope | None,
        Field(
            default=None, description="Overrides to apply before minting; see the tool description."
        ),
    ] = None,
    stat_vars: Annotated[
        StatVarsReading,
        Field(
            description=(
                "How to return a spec that covers several StatVars (a family, such as "
                "every CRS purpose and flow-type code under one aid flow). "
                '"separate" returns one series per StatVar. "sum" returns the family\'s '
                "total: dfo-apis adds the StatVars' values for each year and entity, and "
                "the frame holds one StatVar ref, dcid:<spec_id>, standing for the whole "
                "family. Provider and recipient columns are kept, so a yearly total across "
                'all recipients still needs an aggregate by period. Use "sum" when the '
                'question asks for the family\'s total and "separate" when it compares or '
                "breaks down the StatVars. A spec with one StatVar returns the same frame "
                "either way. A family that cannot be added up safely (gross and net "
                "amounts together, for example) refuses with rollup_not_summable; do not "
                "retry it."
            )
        ),
    ] = "separate",
    place_types: Annotated[
        bool,
        Field(
            description=(
                "Add the Data Commons types of each place the frame names as an "
                'observationAbout_type column. Values are comma-joined ("Country", '
                '"AdministrativeArea1,State", "Place" for Earth), and "untyped" for a place '
                "with none. On a flow frame it adds each party side's type and three cells "
                "named for the side instead: financingProvider_type, financingRecipient_type, "
                "financingProvider_party_kind, financingRecipient_party_kind and so on. "
                "party_kind is party, residual, or total for a group total such as "
                '"Africa, Total", which sums and ranks leave out; party_class is official, '
                'private or residual; party_whole is "true" on the published total of a whole population '
                'such as "Official Donors, Total", and "false" elsewhere. Whole totals add up '
                "with each other when they are the only rows. Beside members of a class a whole "
                "total does not hold, it stays: summed with them on a side a sum pools, and as "
                "its own row on a side an operation groups by. A whole total that may hold a "
                "member is left out of a grouped side and refuses on a pooled one. A side an "
                'operation ranks leaves every total out beside its members. A cell reads "untyped" '
                "where the party has no value. Set it for a spec that "
                "names no place: its frame holds every place with data, at every level, so a filter "
                "(a place_level predicate on observationAbout_type, such as Country) can keep one "
                "level before a rank or a total."
            )
        ),
    ] = False,
    without_party_totals: Annotated[
        bool,
        Field(
            description=(
                "On a flow frame, leave out the group totals (kind total, such as DAC Countries "
                "Total or the G7) from each side the spec leaves open, and add the party cells as "
                "place_types does, so the side's members can be compared or ranked without "
                "totals among them. A sum of these members can differ from the source's "
                "published total. For a total across the whole side, search naming the "
                'population ("all donors") with no operation over that side. A side the spec '
                "names, or one holding only totals, keeps every row, and the digest names what "
                "was left out. On a spec with no flow side it changes nothing."
            )
        ),
    ] = False,
    breakdown: Annotated[
        Breakdown | None,
        Field(
            default=None,
            description=(
                "Break the family down by one hierarchical code system, rolled up to one "
                'level, e.g. {"code_system": "crsPurpose", "level": 0} for aid by sector '
                "group. Every StatVar is summed into its code's ancestor at that level, "
                "per year and entity, and the frame's stat_var_<code_system> column names "
                'that ancestor. Use it with stat_vars "separate"; "sum" totals the '
                "whole family and refuses beside it. A deployment that cannot roll codes "
                "up refuses capability_unavailable; do not retry with breakdown then."
            ),
        ),
    ] = None,
    *,
    ctx: Context,
) -> ToolResult:
    """Retrieve the data behind a spec_id a prior `search` call minted.

    Call this after `search` has given you a spec_id: it mints the underlying
    dataset and returns a frame reference -- `{spec_id, content_version}` --
    that identifies this exact retrieval for downstream tools to use. A
    `scope` change re-mints under a new identity, so the returned spec_id
    then differs from the one you passed in; only the returned id names the
    definition actually served.

    `scope` narrows or widens WHO and WHEN, never WHAT is measured -- a
    change to the variable itself belongs to `search`. Each
    top-level member you supply (`entities`, `window`, `facets`) REPLACES
    that whole member on the stored definition; anything you omit stays as
    it was. This means an addition must restate what it keeps: to add Ghana
    to a spec whose recipients are Togo, with France as provider, send
    `entities: {"provider": ["France"], "recipient": ["Togo", "Ghana"]}`.
    Sending `{"recipient": ["Togo", "Ghana"]}` alone silently drops the
    provider binding and mints a broader spec -- "all providers to Togo and
    Ghana" -- instead of adding one country to the spec you had. An empty
    map, `entities: {}`, is a deliberate clear, not a no-op.

    Entities are supplied as names, exactly as spoken ("France", "Sub-Saharan
    Africa"). A dcid is also accepted and skips resolution entirely: a value
    marked `dcid:` (e.g. "dcid:GroupOfSeven", as an `entity_ambiguous`
    refusal's candidates carry it) or an unmarked `namespace/id` (e.g.
    "country/GEO") is a dcid, and anything else is a name. Copy a dcid from
    a refusal or another tool's output; never invent one. Countries,
    subregions, continents, continental unions, places below a country,
    organisations such as the World Bank, DAC aggregates and membership
    groups such as the G7 resolve; a group or region scopes its member
    countries. "World" resolves to the world as one place.
    `entity_ambiguous` fires only when a name matches two or more distinct
    entities' own canonical or alias names.
    "Guinea" resolves silently because it is Guinea's own name, while
    "Congo" is ambiguous because it names both the Democratic Republic of
    the Congo and the Republic of the Congo. Resolution calls out to this
    deployment's graph and can take longer than usual when that graph is
    degraded.

    A spec that names no place covers every place with data, at every
    level: the world, regions, World Bank groups, countries and their
    subdivisions. `place_types=true` adds the `observationAbout_type` column,
    so a `filter` can keep one level before a rank or a total sets a place
    beside the places inside it. On a flow frame it adds each party's type,
    kind, class and whole-population mark (`financingProvider_type`,
    `financingProvider_party_kind`, `financingProvider_party_class`,
    `financingProvider_party_whole` and the recipient's four) instead.

    `breakdown` sums the family within each code of one hierarchical code
    system rolled up to one level, so a question such as "which sectors got
    the most aid" reads one series per sector group and entity. The frame's
    `stat_var` column holds `dcid:<spec_id>` and `stat_var_<code_system>`
    names each row's code at that level.

    Args:
        spec_id: The spec_id from a prior `search` digest, or from this
            tool's own frame reference on a previous call.
        scope: Overrides to apply before minting, replacing whole top-level
            members (`entities`, `window`, `facets`) as described above.
            Omit entirely to retrieve the stored definition unchanged.
        stat_vars: "sum" for a family's total as one series, "separate"
            (the default) for one series per StatVar.
        place_types: Whether to add the places' types (`observationAbout_type`), or
            each party's type and its kind, class and whole cells on a flow frame, for a
            spec that names no place.
        without_party_totals: Whether a flow frame leaves out the group totals of each side
            the spec leaves open, to compare or rank its members, and adds the party cells as
            `place_types` does. On a spec with no flow side it changes nothing.
        breakdown: The code system and hierarchy level to sum the family
            within, as described above.

    Returns:
        A digest: an acknowledgement of what was retrieved, its coverage,
        the frame reference `{spec_id, content_version}` for downstream
        tools, then the citation, and any dropped-role notes. The
        artifact resource is a URI template, listed by
        resources/templates/list and absent from resources/list.
    """
    settings = Settings()
    admission = requested_admission()
    if admission is not None:
        # So a count of a turn's retrievals can leave out the mints no run asked for.
        trace.get_current_span().set_attribute(BACKGROUND_MINT_SPAN_ATTRIBUTE, True)
    budget_deadline_s = current_call_budget().deadline_s
    deadline_s = settings.get_data_deadline_s if budget_deadline_s is None else budget_deadline_s
    deadline = CallDeadline.after(deadline_s, clock=_deadline_clock)
    if breakdown is not None and stat_vars == "sum":
        return to_tool_result(
            ToolRefusal.request_invalid(
                pointer="/stat_vars",
                detail='"sum" totals the whole family into one series, which leaves no '
                'code to break it down by; use "separate" with breakdown',
            )
        )
    if breakdown is not None:
        refused = await _breakdown_refusal(get_client(), breakdown, deadline=deadline)
        if refused is not None:
            return refused
    identity = await _load_identity_or_refuse(spec_id, deadline=deadline)
    if isinstance(identity, ToolResult):
        return identity
    sends_definition = identity is not None
    if identity is None and (place_types or without_party_totals):
        # The columns the mint asks for follow the spec's identity.
        registry = await _registry_identity_or_refuse(spec_id, deadline=deadline)
        if isinstance(registry, ToolResult):
            return registry
        identity = registry.identity
        sends_definition = registry.hashes_to_requested_id

    wire_scope: dict[str, Any] = {}
    entity_labels: dict[str, str] = {}
    dropped_roles: list[str] = []

    if scope is not None and scope.entities is not None:
        resolved = await _resolve_entities_or_refuse(scope.entities, deadline=deadline)
        if isinstance(resolved, ToolResult):
            return resolved
        wire_entities, entity_labels = resolved
        wire_scope["entities"] = wire_entities
        if scope.entities and identity is not None:
            # An explicitly empty map is a deliberate clear, not an omission
            # -- nothing was silently dropped, so no role is named here. On
            # the registry-fallback path (`identity is None`, above) there is
            # no base identity to diff against -- the response the model
            # eventually sees carries the post-scope-surgery identity the
            # registry served, never the base -- so `dropped_roles` stays
            # empty there rather than guessed at. An open role binds no
            # entity and is not reported dropped. apis closes one the scope
            # binds and keeps every other open.
            base_roles = set((identity.get("entities") or {}).keys())
            dropped_roles = sorted(base_roles - set(scope.entities.keys()))

    if scope is not None and scope.window is not None:
        wire_scope["window"] = scope.window
    if scope is not None and scope.facets is not None:
        wire_scope["facets"] = scope.facets

    mint = functools.partial(
        _attempt_mint,
        client=get_client(),
        spec_id=spec_id,
        identity=identity,
        sends_definition=sends_definition,
        wire_scope=wire_scope,
        stat_vars=stat_vars,
        breakdown=breakdown,
        place_types=place_types,
        without_party_totals=without_party_totals,
        entity_labels=entity_labels,
        dropped_roles=dropped_roles,
        settings=settings,
        deadline=deadline,
        # A background mint never polls, and its task outlives the request,
        # so it holds no reference to the request's `ctx`.
        on_poll=_report_progress(ctx) if admission is None else None,
        admission=admission,
    )
    # A background mint answers once its POST is out. The POST stays open in a task.
    return await (mint() if admission is None else _posted_in_background(mint, spec_id=spec_id))


# The background mints whose POST is still open, so none is collected mid-flight
# and `aclose_background_mints` can close them at shutdown.
_background_mints: set[asyncio.Task[ToolResult]] = set()


async def _posted_in_background(
    mint: Callable[..., Coroutine[Any, Any, ToolResult]], *, spec_id: str
) -> ToolResult:
    """Run `mint` as a task of its own and answer once its POST is written out,
    with `background_result` outcome `"posted"`, or once it ends, with its result,
    whichever comes first.

    The task keeps the POST open until apis answers: apis runs the mint on CPU
    that Cloud Run allocates only while the request is open.
    """
    sent = asyncio.Event()
    task = asyncio.create_task(mint(on_sent=sent.set))
    _background_mints.add(task)
    task.add_done_callback(
        functools.partial(
            _background_mint_done,
            spec_id=spec_id,
            caller=current_writer(),
            started=time.monotonic(),
        )
    )
    sent_wait = asyncio.ensure_future(sent.wait())
    try:
        await asyncio.wait({task, sent_wait}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        sent_wait.cancel()
    if task.done():
        return task.result()
    return background_result(spec_id=spec_id, outcome="posted")


def _background_mint_done(
    task: asyncio.Task[ToolResult], *, spec_id: str, caller: str, started: float
) -> None:
    """Forget `task` and log how apis answered its POST (`declined` when apis had
    no room for it, else the outcome or refusal code), with the caller and the
    seconds since the call started, so the line joins the caller's own."""
    _background_mints.discard(task)
    if task.cancelled():
        outcome = "cancelled"
    elif (exc := task.exception()) is not None:
        outcome = f"error:{type(exc).__name__}"
    else:
        outcome = background_outcome(task.result().structured_content)
    logger.info(
        "background mint answered: spec_id=%s caller=%s outcome=%s elapsed_s=%.2f",
        spec_id,
        caller,
        outcome,
        time.monotonic() - started,
    )


async def aclose_background_mints() -> None:
    """Close every background mint POST still open, at shutdown. apis runs on
    any mint it admitted."""
    tasks = list(_background_mints)
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

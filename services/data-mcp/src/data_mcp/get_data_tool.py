"""The `get_data` tool: retrieves the data behind a spec_id a prior `search` minted.

`search` resolves an ask into a spec_id but never fetches the underlying data;
`get_data` reads the definition for that id -- the local store first, falling
back to dfo-apis' own permanent registry on a store miss -- mints it through
dfo-apis' Datasets door, and returns a frame reference plus a digest. Entity
names in the caller's `scope` are resolved to dcids through
`data_mcp.resolution` before the mint call; the mint call itself and the
resulting manifest never carry a name, only dcids and, later, a spec_id.

The mint posts the spec_id and its identity only. dfo-apis resolves the
family's members from the graph on every mint, so no caller supplies them.

Rendering the manifest into a digest lives in `data_mcp.get_data_digest`; the
shared upstream client lives in `data_mcp.client_holder`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Any, Literal

import httpx2
import pyarrow as pa
from common.caller import current_writer
from common.materialisation import FrameRef, MintRecord
from fastmcp import Context
from fastmcp.exceptions import ResourceError
from fastmcp.tools import ToolResult
from mcp.types import ResourceLink, TextContent
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

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
from data_mcp.errors import UpstreamError, UpstreamTransportError
from data_mcp.frame_materialisation import (
    DEFAULT_MINT_OPTIONS,
    PLACE_TYPE_COLUMNS,
    materialise_and_register,
)
from data_mcp.get_data_digest import (
    BreakdownLevel,
    OperationPollNote,
    UnheldRegionSummary,
    render_dcids,
    render_get_data_digest,
    statistics_needs_fetch,
)
from data_mcp.manifest import (
    DefinitionModel,
    Manifest,
    SeriesStatisticsModel,
    WarningModel,
    render_validation_errors,
    strip_manifest,
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
    OutOfScope,
    ResolutionOutcome,
    ResolvedEntity,
    ResolvedRegion,
    resolve_entities,
)
from data_mcp.series_statistics import (
    SeriesStatisticsSchemaError,
    fetch_series_statistics,
    find_statistics_artifact,
)
from data_mcp.settings import Settings
from data_mcp.stat_var_list import without_long_stat_var_list
from data_mcp.tool_errors import (
    EntityCandidate,
    ToolErrorCode,
    ToolRefusal,
    facet_issues_from,
    from_upstream_error,
    to_tool_result,
)

# Both `CanonicalDefinition.schema_version` and `MintRequest.schema_version`
# are `Literal[1]` upstream, and a model calling this tool has no meaningful
# choice to make here -- one constant stands in for the store lookup and
# every `schema_version` field the mint request body carries.
_SCHEMA_VERSION = 1

# The mixer behind a mint request supports at most 3 entity-role slots per
# observation (`SdmxRoleCeilingError`, dfo-apis' `datasets_api/doors/mint.py`
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

# A dcid the resolve door itself just handed back in an `entity_ambiguous`
# refusal's candidate list -- never a name the model invents. Namespaced
# graph dcids are always `prefix/segment[/segment...]` (`country/KEN`,
# `geoId/13`, `wikidataId/Q30`; dfo-apis' `graph_api.routing` module defines
# the node door's own `{dcid}` path grammar as one or more `/`-joined
# segments, each `[A-Za-z0-9_.-]+`), and every entity-role namespace is
# lowercase-initial -- requiring that is what tells a returned dcid apart
# from plain text without guessing at a stricter grammar dfo-apis doesn't
# publish. A slash alone isn't enough: `Congo/Brazzaville` and
# `Bosnia/Herzegovina` are ordinary spoken disambiguation a caller types,
# and both contain a `/`. A capitalized segment before the slash is a name
# to resolve, never a passthrough dcid. The mint route's own
# `Scope._entities_are_valid` (`datasets_api/mint_models.py`) accepts any
# non-empty value that avoids `+`/`,`, so this pattern is the tighter of
# the two bounds.
_DCID_PATTERN = re.compile(r"^[a-z][A-Za-z0-9_.\-]*(?:/[A-Za-z0-9_.\-]+)+$")

# What `get_data(stat_vars="sum")` asks dfo-apis for: `by` names the columns to keep,
# and dfo-apis sums away every summable code axis it does not name, `stat_var` first.
# `time_period` is a column every frame carries and no code axis, so this keeps no
# code axis and dfo-apis returns a full collapse: one StatVar ref standing for the
# family, with every entity role column kept (dfo-apis never sums across a role).
_SUM_STAT_VARS_GROUPBY: dict[str, Any] = {"by": ["time_period"], "op": "sum"}

StatVarsReading = Literal["sum", "separate"]

# The `GET /api/datasets/capabilities` field that lists the code systems
# `options.groupby.levels` rolls up. A deployment without it serves no `breakdown`.
_GROUPBY_LEVEL_AXES_FIELD = "groupby_level_axes"
_CAPABILITIES_PATH = "/api/datasets/capabilities"

# No `UpstreamError` backs a definition-store read failure, so there is no
# call to derive a wait from. Mirrors `search_tool.py`'s constants of the
# same name and value.
_DEFINITION_STORE_RETRY_AFTER_S = 5.0
_DEFINITION_STORE_SERVICE_NAME = "the definition store"

# The current `CallBudget` shares out one call's deadline (`data_mcp.deadline`). Inside
# its post-mint reserve, the statistics fetch runs only with `_STATISTICS_FLOOR_S` left
# and must finish with `_STATISTICS_RESERVE_S` still unspent, for the digest.
_STATISTICS_FLOOR_S = 8.0
_STATISTICS_RESERVE_S = 3.0

# The clock a call's deadline reads. Tests replace it to move time without sleeping.
_deadline_clock: Clock = time.monotonic

# The mints this process holds dfo-apis mint slots for on behalf of outside callers,
# on `_deadline_clock`'s time. Tests replace it with an empty one.
_mint_ledger = MintLedger()

logger = logging.getLogger(__name__)


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
            '{"recipient": ["Togo", "Ghana"]}. A dcid (e.g. "country/GEO") is '
            "also accepted verbatim and skips resolution -- only to answer "
            "an entity_ambiguous refusal's candidate list from a prior "
            "call, never invented. Replaces the whole entities member -- an "
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
        return self


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
        """The `options.groupby` dfo-apis sums this breakdown with: keep the year and the
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
    """Refuse `breakdown` unless the deployed dfo-apis advertises a level roll-up of its
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
                    EntityCandidate(
                        dcid=c.dcid, label=c.name, type=c.entity_type, parent=c.parent or ""
                    )
                    for c in outcome.candidates
                ],
                truncated=outcome.truncated,
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
            ToolRefusal.entity_not_found(
                role=role,
                query=query,
                near_matches=[],
                not_held_entity_type=outcome.entity_type,
            )
        )
    if isinstance(outcome, OutOfScope):
        return to_tool_result(
            ToolRefusal.entity_out_of_scope(
                role=role,
                query=query,
                entity_type=outcome.entity_type,
                dcid=outcome.dcid,
            )
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
) -> tuple[dict[str, list[str]], dict[str, str], list[UnheldRegionSummary]] | ToolResult:
    """Resolve every name in `entities` into dcids for the mint request.

    A value already shaped like a dcid (`_DCID_PATTERN`) is used verbatim and never reaches
    the resolve door -- this is how a caller answers this tool's own `entity_ambiguous`
    refusal, whose candidate list names dcids to pick from; everything else is resolved by
    name.

    Returns `(wire_entities, entity_labels, unheld_regions)` on full success: `wire_entities`
    is role -> bare dcids, order-preserving-deduped so the same dcid reached twice under one
    role (a region and one of its own members, both named) mints once rather than twice --
    upstream preserves this list verbatim into the hashed identity document, so an undeduped
    list would mint two different spec_ids for one logical question depending on phrasing;
    `entity_labels` is dcid -> name for every entity and region member this call resolved by
    name (a passthrough dcid gets no entry, so the digest renders it as itself); and
    `unheld_regions` carries one summary per resolved region that came back with members this
    graph doesn't hold. `ResolvedRegion` outcomes proceed on their confirmed members even when
    `members_not_held` is non-empty -- a partially-held region still answers the question
    asked.

    Returns a refusal `ToolResult` in place of the tuple on the first name that isn't cleanly
    usable. `role` always names the failed name's own role, per the platform contract's
    entity-refusal requirement; every pointer this function builds names this tool's own
    `scope.entities.{role}` argument path -- the frame a caller who called `get_data` itself
    reads pointers in -- never dfo-apis' `/input`-wrapped wire pointer.
    """
    names_to_resolve = {
        role: [name for name in names if not _DCID_PATTERN.fullmatch(name.strip())]
        for role, names in entities.items()
    }
    outcomes = await resolve_entities(names_to_resolve)
    wire_entities: dict[str, list[str]] = {}
    entity_labels: dict[str, str] = {}
    unheld_regions: list[UnheldRegionSummary] = []
    for role, names in entities.items():
        dcids: list[str] = []
        for name in names:
            stripped = name.strip()
            if _DCID_PATTERN.fullmatch(stripped):
                dcids.append(stripped)
                continue
            outcome = outcomes[role][name]
            if isinstance(outcome, ResolvedEntity):
                dcids.append(outcome.dcid)
                entity_labels[outcome.dcid] = outcome.name
            elif isinstance(outcome, ResolvedRegion):
                if not outcome.members:
                    # The resolve door's `resolved_region` status always
                    # carries at least one confirmed member; a region with
                    # none held returns `not_held` instead. Treating an empty
                    # membership list as that same not-held refusal, rather
                    # than trusting it, keeps a role from silently scoping to
                    # zero dcids if that guarantee ever changes upstream.
                    return to_tool_result(
                        ToolRefusal.entity_not_found(
                            role=role,
                            query=name,
                            near_matches=[],
                            not_held_entity_type=outcome.entity_type,
                        )
                    )
                for member in outcome.members:
                    dcids.append(member.dcid)
                    entity_labels[member.dcid] = member.name
                if outcome.members_not_held:
                    unheld_regions.append(
                        UnheldRegionSummary(
                            region_name=outcome.name,
                            unheld_names=[m.name for m in outcome.members_not_held],
                            unheld_count=len(outcome.members_not_held),
                        )
                    )
            else:
                refusal = _refusal_for_unresolved_outcome(outcome, role=role, query=name)
                if refusal is not None:
                    return refusal
        wire_entities[role] = list(dict.fromkeys(dcids))
    return wire_entities, entity_labels, unheld_regions


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
) -> tuple[dict[str, list[str]], dict[str, str], list[UnheldRegionSummary]] | ToolResult:
    """Wrap `_resolve_scope_entities` with the resolve door's own failure modes.

    `resolve_entities` deliberately lets `UpstreamError` propagate (see its own docstring);
    nothing else in the call path catches it, so without this wrapper a routine 503 from a
    degraded graph would escape as a raw exception -- `is_error=True` with no code, retry
    mode, or wait -- instead of this refusal grammar. A 2xx body this service still can't use
    raises one of three exceptions, never a `ToolError`: `_outcome_from_result` raises
    `ValueError` for a wire status this service doesn't recognise; its own dict-indexed reads
    of the body (`result["status"]`, `result["entity"]["dcid"]`, `result["members"]`, ...) raise
    `KeyError` for a missing key and `TypeError` for a wrongly-typed nested value. All three
    mean the same thing -- the resolve call itself succeeded, only its body couldn't be used,
    the same class of failure `upstream_response_unusable` already names for the mint call.

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
# dfo-apis' own wire envelope (`{input: {scope: {entities: {role: [...]}}}}`) -- distinct
# from this tool's own `scope.entities.{role}` argument path, which `InvalidName` above
# already points into directly.
_UPSTREAM_ENTITIES_ROLE_POINTER_PREFIX = "/input/scope/entities/"


def _rebase_entities_role_pointer(refusal: ToolRefusal) -> ToolRefusal:
    """Rewrite an upstream-derived entities-role pointer onto this tool's own argument path.

    `invalid_role_ref`/`role_incompatible` map to `request_invalid` with a pointer straight
    off dfo-apis' wire (`/input/scope/entities/{role}`), while a locally-rejected name
    (`InvalidName` above) already points at `/scope/entities/{role}` -- this tool's own
    `scope` argument, with no `/input` wrapper, since that wrapper names dfo-apis' request
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
    # No local hit, but dfo-apis' own registry may still hold this id: the local store is
    # per-process by default and even its GCS-backed form expires at 30 days, while the
    # upstream registry's rows are never deleted. Only a registry miss too (`spec_not_found`,
    # caught in `get_data`) completes the platform contract's own definition of
    # `definition_not_held`: "held by neither store nor registry".
    return None


async def _persist_reminted_definition(
    definition: DefinitionModel, *, deadline: CallDeadline
) -> bool:
    """Write a re-minted definition to the store, so a later `get_data` call with its id succeeds.

    Called only when the mint route's own manifest carries a `spec_id` different from the one
    this call was made with: a scope override changed the identity, and upstream re-minted
    under a new id. That new id is what the digest hands the model as its frame reference, so
    it has to be resolvable by a later `get_data` call the same way any other id is -- without
    this write, that later call would refuse `definition_not_held` even though dfo-apis' own
    registry now holds this id, which is exactly the case the platform contract's
    `definition_not_held` excludes ("held by neither store nor registry"). `verify_write`
    canonicalises and hash-checks `definition.identity` the same way `search_tool.py`'s own
    registration path does, so a corrupt identity is caught rather than trusted forward.

    The write must finish while the current `CallBudget.post_mint_reserve_s` of `deadline`
    remains, so the frame fetch, statistics and rendering after it keep their share. A
    write that runs past that is abandoned unconfirmed: its worker thread may still land it.

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
        reserve_s = current_call_budget().post_mint_reserve_s
        async with asyncio.timeout(deadline.reserving(reserve_s).remaining()):
            await get_store().put_many([stored])
    except (DefinitionStoreUnavailable, TimeoutError):
        return False
    return True


def _artifact_resource_links(manifest: Manifest) -> list[ResourceLink]:
    """One `ResourceLink` per artifact on `manifest`, from its own `resource_uri`.

    A model reads the link to pull a freshly signed download URL at fetch
    time. A URL embedded in this tool's result would go stale, so neither
    the result nor the manifest in structured content carries one. One link
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
                title=f"{artifact.rendering.get('format', 'artifact')} artifact",
                description=(
                    f"Read for a freshly signed download URL ({artifact.bytes} bytes). "
                    "The URL is minted at read time and expires in minutes; do not cache it."
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
    network. A signed-URL fetch or Parquet-decode failure degrades to
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
            expected_rows=manifest.series_count,
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
) -> tuple[dict[str, Any], OperationPollNote | None] | ToolResult:
    """The mint route's manifest payload, resolving a 202 operation to completion first.

    The mint POST answers 200 with the manifest directly for a request within
    dfo-apis' own sync budget, or 202 with an operation envelope once that budget
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
    unheld_regions: Sequence[UnheldRegionSummary],
    settings: Settings,
    deadline: CallDeadline,
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
    under a new identity; see `_persist_reminted_definition` for why that new definition gets
    written to the store here, and what happens when the write itself fails.

    Once the manifest itself validates, it is narrowed to the requested rendering
    (`_narrowed_to_rendering`), and `materialise_and_register` fetches the bytes of
    the artifact whose rendering equals `options`, parses them into a `Frame`, and
    registers it in the shared materialisation store under the frame reference this
    digest hands the model (`frame_materialisation.frame_ref_for`). A frame minted with
    `options.columns` has a `content_version` of its own, and the manifest keeps dfo-apis' pair, which its
    artifacts are read under. A failed fetch or an unconfirmed store write degrades this
    call's own digest rather than refusing it (see `render_get_data_digest`'s
    `degraded_frame`); a genuinely unusable artifact or manifest still refuses.

    Args:
        payload: The mint route's own manifest payload -- either an ordinary 200
            response's body, or the manifest `_mint_payload_or_refusal` polled out of
            a 202 operation.
        requested_spec_id: The `spec_id` this call's own request carried, to detect a re-mint.
        options: The mint request's `options`, which name the artifact the frame is
            read from and, through `options.columns`, its frame reference.
        scope: The mint request's `scope`, kept with `options` in the frame's mint
            record (`common.materialisation.MintRecord`).
        entity_labels: dcid -> name, forwarded to the digest renderer.
        dropped_roles: Roles silently left unbound, forwarded to the digest
            renderer.
        unheld_regions: Resolved regions with unheld members, forwarded to
            the digest renderer.
        settings: This service's own settings, threaded to `materialise_and_register` for
            its row-count ceiling and fetch wall-clock budget.
        deadline: The `get_data` call's deadline, which cuts the re-mint persist's, the
            frame fetch's, and the statistics fetch's budgets.
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
        (`_stat_vars_summed_warning`), or a refusal
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

    remint_persist_failed = False
    if manifest.definition.spec_id != requested_spec_id:
        try:
            persisted = await _persist_reminted_definition(manifest.definition, deadline=deadline)
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

    series_stats, statistics_fetch_failed = await _fetch_statistics_for_digest(
        manifest, deadline=deadline
    )

    try:
        content = render_get_data_digest(
            manifest=manifest,
            frame_ref=frame_ref,
            entity_labels=entity_labels,
            dropped_roles=dropped_roles,
            unheld_regions=unheld_regions,
            remint_persist_failed=remint_persist_failed,
            degraded_frame=outcome if outcome.status == "degraded" else None,
            series_stats=series_stats,
            statistics_fetch_failed=statistics_fetch_failed,
            operation_poll=operation_poll,
            breakdown=breakdown,
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
    # dfo-apis' materialisation, which the statistics artifact is read under, and
    # this one names the frame the store holds for this rendering.
    structured_content["frame_ref"] = frame_ref.model_dump(mode="json")
    if outcome.role_keys is not None:
        # The frame's own role columns and distinct StatVar count sit beside the
        # manifest, which is dfo-apis' document; these come from the frame this call
        # read. The harness binds a plan's role columns to the first and bounds each
        # later step's declared StatVar count by the second (`compiler.frame_columns`).
        structured_content["role_keys"] = list(outcome.role_keys)
        structured_content["stat_var_count"] = outcome.stat_var_count
    summed = _stat_vars_summed_warning(manifest)
    structured_content["summed_stat_var_count"] = (
        None if summed is None else summed.contributing_member_count
    )
    structured_content["summed_stat_var_label"] = None if summed is None else summed.stat_var_label
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

    A manifest lists every rendering dfo-apis stores under its pair, so other callers'
    groupby sums and column selections sit beside the requested one, in no set order.
    The frame fetch, the resource links, and the harness's freshness date
    (`harness.provenance.data_artifact`) all read the narrowed list. A manifest without
    the requested rendering keeps only its statistics, and `materialise_and_register`
    refuses it unless the store already holds the frame.
    """
    statistics = find_statistics_artifact(manifest)
    artifacts = [a for a in manifest.artifacts if a.rendering == options or a is statistics]
    return manifest.model_copy(update={"artifacts": artifacts})


# The manifest warning dfo-apis attaches to a frame that sums more than one StatVar.
_STAT_VARS_SUMMED_WARNING = "stat_vars_summed"


def _stat_vars_summed_warning(manifest: Manifest) -> WarningModel | None:
    """The warning naming how many StatVars dfo-apis added into this frame's one family
    series, and that family's label.

    `None` when nothing was summed: a separate retrieval, or a summed one where a single
    StatVar had rows, which dfo-apis serves as that StatVar's own frame with no warning.
    """
    return next((w for w in manifest.warnings if w.code == _STAT_VARS_SUMMED_WARNING), None)


async def _post_mint(
    client: UpstreamClient,
    request_body: Mapping[str, Any],
    *,
    spec_id: str,
    settings: Settings,
    deadline: CallDeadline,
) -> httpx2.Response | ToolRefusal:
    """POST one mint request inside what `deadline` leaves it.

    The current `CallBudget` sizes it: the POST gets `mint_timeout_s` at most and must
    end with `post_mint_reserve_s` of `deadline` left. With less than
    `min_mint_timeout_s` to offer it, this refuses without sending, so nothing starts
    upstream. Any POST that gets no response (whether `mint_timeout_s` elapses first,
    the deadline cuts it, or the connection fails outright, the same
    `UpstreamTransportError` each way) refuses with `mint_unanswered`: dfo-apis does
    not cancel a mint whose client went away, so it may still be running it.

    Under a budget that does not wait out dfo-apis' synchronous mint, the POST first
    takes a slot in `_mint_ledger`, or returns the ledger's refusal without sending.
    The slot is freed when dfo-apis answers, and otherwise held until dfo-apis has
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
            request_key=json.dumps(request_body, sort_keys=True),
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
    wire_scope: Mapping[str, Any],
    stat_vars: StatVarsReading,
    breakdown: Breakdown | None,
    place_types: bool,
    entity_labels: Mapping[str, str],
    dropped_roles: Sequence[str],
    unheld_regions: Sequence[UnheldRegionSummary],
    settings: Settings,
    deadline: CallDeadline,
    on_poll: OnPoll | None = None,
) -> ToolResult:
    """POST the `spec_id` mint and turn the result into this tool's response.

    `identity` is `None` on the registry-fallback path, where the request omits the
    `definition` sidecar and a `spec_not_found` refusal completes `definition_not_held`.
    `stat_vars="sum"` adds `_SUM_STAT_VARS_GROUPBY` to the request's options. The sum
    sits outside the spec's identity: the manifest keeps `spec_id`, and dfo-apis mints
    the summed frame its own `content_version` whenever more than one StatVar
    contributed rows, so the summed and separate frames never share a frame reference.
    With one contributor the two frames are the same bytes under one `content_version`.
    A `breakdown` adds its own `Breakdown.groupby` instead, which sums the StatVars
    within each code at the breakdown's level.
    `place_types` sets `options.columns` to `PLACE_TYPE_COLUMNS`, which adds
    `observationAbout_type` to the default columns, and gives the frame a
    `content_version` of its own (`frame_materialisation.frame_ref_for`).
    """
    mint_input: dict[str, Any] = {
        "kind": "spec_id",
        "spec_id": spec_id,
        "scope": dict(wire_scope),
    }
    if identity is not None:
        # Sent whenever the store holds the definition; omitted on the
        # registry-fallback path so `mint_spec_id` is free to serve
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
    if place_types:
        options["columns"] = list(PLACE_TYPE_COLUMNS)
    request_body = {
        "schema_version": _SCHEMA_VERSION,
        "input": mint_input,
        "options": options,
    }

    def minted(result: ToolResult) -> ToolResult:
        """A `default_facet_undefined` refusal, from the mint or its poll, naming the
        spec_id and places this call was minting. The cause and retry flag stay dfo-apis'."""
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
                trace_id=refused.get("trace_id"),
                capability_version=refused.get("capability_version"),
            )
        )

    try:
        posted = await _post_mint(
            client, request_body, spec_id=spec_id, settings=settings, deadline=deadline
        )
    except UpstreamError as exc:
        if identity is None and exc.code == "spec_not_found":
            # The sidecar-free registry-fallback request above just missed
            # dfo-apis' own registry too: `mint_spec_id` raises exactly this
            # code in that case, and the platform contract's
            # `definition_not_held` -- "held by neither store nor registry"
            # -- is what this completes.
            return to_tool_result(ToolRefusal.definition_not_held(definition_id=spec_id))
        return minted(to_tool_result(_rebase_entities_role_pointer(from_upstream_error(exc))))
    if isinstance(posted, ToolRefusal):
        return to_tool_result(posted)

    resolved = await _mint_payload_or_refusal(
        posted, client=client, settings=settings, deadline=deadline, on_poll=on_poll
    )
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
        unheld_regions=unheld_regions,
        settings=settings,
        deadline=deadline,
        operation_poll=operation_poll,
        breakdown=(
            None
            if breakdown is None
            else BreakdownLevel(code_system=breakdown.code_system, level=breakdown.level)
        ),
    )


@dataclass(frozen=True, kw_only=True)
class RootMint:
    """A root frame `mint_recorded_root` confirmed in the materialisation store.

    `frame` is the pair the store holds it under, `dataset` the pair dfo-apis names
    the materialisation by, and `stat_var_count` the distinct StatVars its rows hold.
    """

    frame: FrameRef
    dataset: FrameRef
    stat_var_count: int


async def mint_recorded_root(
    record: MintRecord, *, deadline: CallDeadline, settings: Settings | None = None
) -> RootMint | QueuedOperation | ToolResult:
    """Mint the dataset `record` names again, with the `options` and `scope` it
    records, and register the current frame of that rendering in the store.

    This is `get_data`'s mint without its model-facing parts. No entity name is
    resolved, since a recorded scope already holds dcids, and no digest, statistics
    or resource link is built. The request is `get_data`'s own: `record.dataset
    .spec_id` with `record.scope`, the stored definition beside it when the
    definition store holds one, and `record.options` as they stand. It goes through
    `_post_mint`, so the current `CallBudget` sizes the POST and the mint ledger
    admits it, and `materialise_and_register` registers the frame and writes the
    mint record of its current pair.

    It does not wait on a queued mint. A 202 whose operation is unfinished returns
    the `QueuedOperation` at once (`operations.operation_without_waiting`). The
    caller sends the same mint again later, and dfo-apis answers with the same
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
        dfo-apis queued and has not finished. A refusal `ToolResult` otherwise,
        which is `service_unavailable` with a wait when the mint succeeded and the
        frame could not be fetched or stored inside `deadline`.
    """
    settings = settings or Settings()
    client = get_client()
    posted = await _post_recorded_mint(client, record, settings=settings, deadline=deadline)
    if isinstance(posted, ToolResult):
        return posted
    payload = await _manifest_payload_without_waiting(
        posted, client=client, settings=settings, deadline=deadline
    )
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
        manifest=_narrowed_to_rendering(manifest, options=record.options),
        settings=settings,
        writer=current_writer(),
        options=record.options,
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
        stat_var_count=outcome.stat_var_count,
    )


async def _post_recorded_mint(
    client: UpstreamClient, record: MintRecord, *, settings: Settings, deadline: CallDeadline
) -> httpx2.Response | ToolResult:
    """POST the mint `record` names, as `_attempt_mint` posts `get_data`'s, with the
    stored definition beside the `spec_id` when the definition store holds one."""
    spec_id = record.dataset.spec_id
    identity = await _load_identity_or_refuse(spec_id, deadline=deadline)
    if isinstance(identity, ToolResult):
        return identity
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
        "options": record.options,
    }
    try:
        posted = await _post_mint(
            client, request_body, spec_id=spec_id, settings=settings, deadline=deadline
        )
    except UpstreamError as exc:
        if identity is None and exc.code == "spec_not_found":
            return to_tool_result(ToolRefusal.definition_not_held(definition_id=spec_id))
        return to_tool_result(from_upstream_error(exc))
    return to_tool_result(posted) if isinstance(posted, ToolRefusal) else posted


async def _manifest_payload_without_waiting(
    response: httpx2.Response,
    *,
    client: UpstreamClient,
    settings: Settings,
    deadline: CallDeadline,
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
                "Add each place's Data Commons types as an observationAbout_type column, "
                'comma-joined ("Country", "AdministrativeArea1,State", "Place" for Earth), and '
                '"untyped" for a place with none. Set it for a spec that names no place: its '
                "frame holds every place with data, at every level, so a filter "
                "(a place_level predicate on observationAbout_type, such as Country) can keep one "
                "level before a rank or a total."
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
    Africa"). A dcid (e.g. "country/GEO") is also accepted, verbatim, and
    skips resolution entirely -- but only to answer an `entity_ambiguous`
    refusal, whose candidate list names dcids to pick from; never invent
    one. Only countries, subregions, and continents resolve through this
    tool; an organisation name like "World Bank" can never resolve here, by
    design, so do not retry one. Binding the full set of continents by name
    approximates a world or global scope, since there is no dedicated world
    or global entity. `entity_ambiguous` fires only when a name
    matches two or more distinct countries' own canonical or alias names.
    "Guinea" resolves silently because it is Guinea's own name, while
    "Congo" is ambiguous because it names both the Democratic Republic of
    the Congo and the Republic of the Congo. Resolution calls out to this
    deployment's graph and can take longer than usual when that graph is
    degraded.

    A spec that names no place covers every place with data, at every
    level: the world, regions, World Bank groups, countries and their
    subdivisions. `place_types=true` adds each place's types as
    `observationAbout_type`, so a `filter` can keep one level before a rank
    or a total sets a place beside the places inside it.

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
        place_types: Whether to add each place's types as
            `observationAbout_type`, for a spec that names no place.
        breakdown: The code system and hierarchy level to sum the family
            within, as described above.

    Returns:
        A digest: an acknowledgement of what was retrieved, its coverage,
        the frame reference `{spec_id, content_version}` for downstream
        tools, then the citation, and any dropped-role or unheld-region
        notes. The
        artifact resource is a URI template, listed by
        resources/templates/list and absent from resources/list.
    """
    settings = Settings()
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

    wire_scope: dict[str, Any] = {}
    entity_labels: dict[str, str] = {}
    dropped_roles: list[str] = []
    unheld_regions: list[UnheldRegionSummary] = []

    if scope is not None and scope.entities is not None:
        resolved = await _resolve_entities_or_refuse(scope.entities, deadline=deadline)
        if isinstance(resolved, ToolResult):
            return resolved
        wire_entities, entity_labels, unheld_regions = resolved
        wire_scope["entities"] = wire_entities
        if scope.entities and identity is not None:
            # An explicitly empty map is a deliberate clear, not an omission
            # -- nothing was silently dropped, so no role is named here. On
            # the registry-fallback path (`identity is None`, above) there is
            # no base identity to diff against -- the response the model
            # eventually sees carries the post-scope-surgery identity the
            # registry served, never the base -- so `dropped_roles` stays
            # empty there rather than guessed at. An open role binds no
            # entity and is not reported dropped. dfo-apis closes one the scope
            # binds and keeps every other open.
            base_roles = set((identity.get("entities") or {}).keys())
            dropped_roles = sorted(base_roles - set(scope.entities.keys()))

    if scope is not None and scope.window is not None:
        wire_scope["window"] = scope.window
    if scope is not None and scope.facets is not None:
        wire_scope["facets"] = scope.facets

    client = get_client()
    on_poll = _report_progress(ctx)

    return await _attempt_mint(
        client=client,
        spec_id=spec_id,
        identity=identity,
        wire_scope=wire_scope,
        stat_vars=stat_vars,
        breakdown=breakdown,
        place_types=place_types,
        entity_labels=entity_labels,
        dropped_roles=dropped_roles,
        unheld_regions=unheld_regions,
        settings=settings,
        deadline=deadline,
        on_poll=on_poll,
    )

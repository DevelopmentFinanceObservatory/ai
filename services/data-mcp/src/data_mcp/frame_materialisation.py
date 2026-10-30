"""Fetches a `get_data` mint's artifact bytes, parses them into a `Frame`, and
registers the result in the shared materialisation store under its frame
reference (`frame_ref_for`) -- the step that closes the gap between a mint and
an analysis-mcp verb's own `frame_ref` resolution.

The frame held under an apis service pair `{spec_id, content_version}` is that
version's canonical frame, the one its `content_version` names. A mint with no
`options.columns`, no `options.groupby` and no `options.without_party_totals`
registers it under the mint's own pair:
the bytes are the version's `CANONICAL_FRAME_RENDERING` artifact, which the
snapshot route lists, and the frame is registered only when
`mint_content_version(frame_sha256=frame_sha256(frame))` is that
`content_version`. Every other mint registers the artifact of its own rendering
under a pair `frame_ref_for` derives from its columns, groupby and cut, unchecked
against the token, since a selection, a sum or a cut is a different frame. A mint
with `options.without_party_totals` names apis' version of the frame without group
totals, so the pair it registers under holds the rows apis serves for it.

Each mint that reads or parses the frame also writes its mint record
(`common.materialisation.MintRecord`) to the store: the apis pair, the
`options` and `scope` the mint was sent with, which a later mint of the same
rendering needs and the frame's own trace does not hold, and the frame's
distinct StatVar count.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta, tzinfo
from functools import partial
from itertools import combinations, permutations
from types import MappingProxyType
from typing import Any, Literal, cast
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx2
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import rfc8785
from common.materialisation import (
    FrameExpired,
    FrameIntegrityError,
    FrameRef,
    FrameTooLarge,
    MaterialisationStore,
    MaterialisationTrace,
    MintRecord,
    frame_from_verified_parquet_bytes,
    get_store,
    groupby_axes,
    normal_groupby,
)
from common.place_type import (
    FLOW_ROLE_COLUMNS,
    PARTY_ROLES,
    TYPED_ROLES,
    UNTYPED,
    enrichment_role,
    type_column,
)
from common.row_budget import RowBudgetExhausted, RowClaim, row_budget
from dfo_frame_grammar import (
    FACET_IDENTITY_COLUMNS,
    STAT_VAR_COLUMN,
    STATUS_COLUMN,
    TIME_PERIOD_COLUMN,
    UNIT_COLUMN,
    VALUE_COLUMN,
    ComposedUnit,
    Frame,
    FrameError,
    UnitRequest,
    array_kernel,
    build_frame_from_arrow,
    canonical_frame_schema,
    frame_sha256,
    map_units,
    mint_content_version,
    unit_cell_array,
    unit_cell_from_json,
    unit_codes,
)
from dfo_frame_grammar.grammar import validate_unit_request
from fastmcp.exceptions import ResourceError
from pydantic import ValidationError

from data_mcp.client_holder import get_client
from data_mcp.deadline import CallDeadline
from data_mcp.manifest import ArtifactEntry, Manifest
from data_mcp.settings import Settings
from data_mcp.tool_errors import ToolRefusal

__all__ = [
    "CANONICAL_FRAME_RENDERING",
    "DEFAULT_MINT_OPTIONS",
    "KIND_COLUMNS",
    "TYPE_COLUMNS",
    "TYPE_COLUMN_SELECTIONS",
    "WITHOUT_PARTY_TOTALS",
    "DegradedReason",
    "MaterialisationOutcome",
    "frame_from_parquet_bytes",
    "frame_ref_for",
    "materialise_and_register",
    "named_groupbys",
    "recorded_stamp",
]

logger = logging.getLogger(__name__)

# The bare link GET has its own small retry budget (see `_fetch_artifact_bytes_uncapped`'s
# docstring): this hop is idempotent, so a handful of quick retries costs little and buys
# resilience against a transient GCS blip. `artifact_link` (the link lookup) is never
# wrapped in a second retry loop of its own -- it already retries internally against
# apis' snapshot route.
_ARTIFACT_GET_ATTEMPTS = 3
_ARTIFACT_GET_BACKOFF_S = 0.5  # short: this hop is idempotent and cheap
_ARTIFACT_GET_TIMEOUT_S = 30.0

# The status the download link answers once its artifact has aged out of the working tier
# and cannot render again (`artifact_expired`), or its version was taken down
# (`snapshot_revoked`). Neither changes on a retry.
_ARTIFACT_GONE_STATUS = 410

# Seconds of a caller's deadline the fetch leaves unspent, for the digest check,
# parse and store write in `_verify_parse_and_register` after it.
_STORE_WRITE_RESERVE_S = 10.0

# The least fetch budget worth starting a fetch with. The store read must end with
# this plus `_STORE_WRITE_RESERVE_S` still left, so the fetch after a read that runs
# out still gets at least this much.
_MIN_FETCH_BUDGET_S = 5.0

# Seconds of a caller's deadline the store write leaves unspent, for the caller's
# rendering of its result.
_AFTER_WRITE_RESERVE_S = 3.0

# The `options` of a mint that sums nothing and selects no columns. Its frame is
# read from the version's canonical frame (`CANONICAL_FRAME_RENDERING`), and the
# requested rendering itself is never fetched. A mint cannot ask for the canonical
# frame alone, since apis' `MintOptions.format` is `csv`, `xlsx` or `parquet`
# and every mint renders one of them, so this asks for the cheapest: a bare parquet
# file in the frame layout, written with no projection and no graph name read. A
# mint with `columns` adds them to these options, and its frame is that rendering
# (`_select_requested_artifact`).
DEFAULT_MINT_OPTIONS: Mapping[str, Any] = MappingProxyType(
    {"format": "parquet", "bundle": False, "layout": "frame"}
)

# The `rendering` apis links every version's canonical frame under
# (`datasets_api.render.CANONICAL_FRAME_RENDERING`). The snapshot route lists it,
# and a mint's own response does not.
CANONICAL_FRAME_RENDERING: Mapping[str, object] = MappingProxyType({"format": "frame"})

# Every `{role}_type` column a frame can carry, sorted. A recorded flow frame minted with
# party types carries its parties' types.
TYPE_COLUMNS: tuple[str, ...] = tuple(type_column(role) for role in TYPED_ROLES)

# The `options.columns` a recorded mint may hold from before flow tables were asked for
# kinds: none, or some of `TYPE_COLUMNS` in their order.
TYPE_COLUMN_SELECTIONS: tuple[tuple[str, ...], ...] = tuple(
    selection
    for size in range(len(TYPE_COLUMNS) + 1)
    for selection in combinations(TYPE_COLUMNS, size)
)

# The bare tokens apis expands to a `{column}_{token}` column for every party column a flow
# table has: the party's kind, class and whole-population mark (`common.place_type`). The
# qualified type tokens are necessary: the door does not expand a bare `type` token. Keep the
# two kinds of enrichment together so a country filter can distinguish countries from
# organisations without losing the party-kind proof that excludes published group totals from
# later sums and ranks.
KIND_COLUMNS: tuple[str, ...] = ("party_kind", "party_class", "party_whole")
FLOW_ENRICHMENT_COLUMNS: tuple[str, ...] = (
    *sorted(type_column(role) for role in PARTY_ROLES),
    *KIND_COLUMNS,
)

# The earlier token for a flow party kind that old recorded selections carry. Replays translate
# it to the door's current `KIND_COLUMNS` while preserving the rest of the selection.
_LEGACY_KIND_COLUMN = "kind"

# The mint option that asks apis for a flow frame without the group totals of the
# party roles its identity leaves open (apis' `MintOptions.without_party_totals`). A
# mint sends it only when true, and apis records it in the artifact's `rendering` only
# then. A cut that leaves rows out is a version of its own, which the manifest's pair names.
WITHOUT_PARTY_TOTALS = "without_party_totals"


# The mint option that asks apis to add the enrichment the version's own role columns take
# (apis' `MintOptions.enrichment`). apis answers as a mint naming the columns it resolved to,
# and the manifest's artifact lists them in its `rendering`, so a frame minted with it is
# named by them (`resolved_options`).
ENRICHMENT = "enrichment"
DEFAULT_ENRICHMENT = "default"

# The coverage dimension apis counts a frame's years under, beside one per role column.
_YEARS_DIMENSION = "years"


def coverage_roles(manifest: Manifest) -> tuple[str, ...]:
    """The role columns of the frame `manifest` names: its coverage dimensions, one per
    role the frame declares, less the years."""
    return tuple(
        dimension.label
        for dimension in manifest.coverage.dimensions
        if dimension.label != _YEARS_DIMENSION
    )


def resolved_options(options: Mapping[str, Any], *, manifest: Manifest) -> Mapping[str, Any]:
    """`options` as the rendering apis minted for them. An `enrichment: "default"` request
    is the `rendering` of the one data artifact the mint's manifest lists for it, which
    names the columns apis resolved the enrichment to: the artifact whose rendering, its
    columns aside, is the request's, and whose columns hold every requested one. A mint
    response lists the rendering it served and the series statistics, nothing else. Any
    other `options`, and a manifest listing no such artifact or more than one, are returned
    as sent, so the frame read refuses for want of its rendering.
    """
    if options.get(ENRICHMENT) != DEFAULT_ENRICHMENT:
        return options
    base = {key: value for key, value in options.items() if key not in (ENRICHMENT, "columns")}
    requested = set(options.get("columns") or ())
    served = [
        artifact.rendering
        for artifact in manifest.artifacts
        if {key: value for key, value in artifact.rendering.items() if key != "columns"} == base
        and isinstance(columns := artifact.rendering.get("columns", []), list)
        and requested <= set(columns)
    ]
    if len(served) != 1:
        return options
    return dict(served[0])


def names_flow_role(identity: Mapping[str, Any] | None) -> bool:
    """Whether `identity` names a flow axis, as an entity or an open role, by its role word
    (`origin`) or its column (`financingRecipient`): the spec whose open party roles
    `without_party_totals` cuts the group totals of."""
    roles = {*((identity or {}).get("entities") or {}), *((identity or {}).get("open_roles") or ())}
    return bool(roles & (FLOW_ROLE_COLUMNS.keys() | PARTY_ROLES))


def replayed_mint_options(
    options: Mapping[str, Any], *, identity: Mapping[str, Any] | None
) -> Mapping[str, Any]:
    """`options` as `get_data` sends them for a root of `identity`.

    A flow root recorded with no `columns` gains every flow enrichment. A recorded flow
    selection, one holding a party type or a kind token, keeps its other explicit columns
    in order and then asks for every missing flow enrichment, the kinds as `KIND_COLUMNS`:
    the earlier `kind` token and a `party_kind` from before the class and whole tokens are
    read as `KIND_COLUMNS`, even when the identity is no longer available, and a recorded
    qualified party-type selection proves that the current kind tokens are valid, without
    guessing other role columns. A root recorded with `without_party_totals` and kinds
    alone is the rendering `get_data(without_party_totals=True)` mints, and replays with
    `KIND_COLUMNS`. Every other selection is unchanged. A flow frame without this metadata
    cannot prove that a sum or rank leaves group totals out. A recorded `groupby` is sent
    with its `by` sorted, so the mint record the replay writes holds the options it sent.
    """
    options = _sorted_by(options)
    columns = options.get("columns") or ()
    kind_tokens = {_LEGACY_KIND_COLUMN, *KIND_COLUMNS}
    if options.get(WITHOUT_PARTY_TOTALS) and columns and set(columns) <= kind_tokens:
        return {**options, "columns": list(KIND_COLUMNS)}
    flow_root = names_flow_role(identity)
    flow_selection = any(
        column in kind_tokens or column in FLOW_ENRICHMENT_COLUMNS for column in columns
    )
    if not flow_root and not flow_selection:
        return options

    selected = [column for column in columns if column not in kind_tokens]
    additions = FLOW_ENRICHMENT_COLUMNS if flow_root else KIND_COLUMNS
    selected.extend(column for column in additions if column not in selected)
    return {**options, "columns": selected}


# Every reserved column name `build_frame_from_arrow` recognises -- everything else in a
# fetched artifact's own schema is a role key. Imported from `dfo_frame_grammar.grammar`'s
# own constants rather than hardcoded here, so this set can never drift from what
# `build_frame_from_arrow` enforces.
_RESERVED_COLUMNS = frozenset(
    {STAT_VAR_COLUMN, TIME_PERIOD_COLUMN, VALUE_COLUMN, UNIT_COLUMN, STATUS_COLUMN}
    | set(FACET_IDENTITY_COLUMNS)
)


# Why a `"degraded"` outcome holds no confirmed frame. `"fetch"`: the store answered
# with a miss and the artifact bytes were not fetched. `"store_unchecked"`: the store
# read ran out of deadline, so the frame may be held, and the bytes were not fetched.
# `"store_write"`: the bytes were fetched and parsed but the store write did not
# finish before the deadline.
DegradedReason = Literal["fetch", "store_unchecked", "store_write"]


@dataclass(frozen=True, kw_only=True)
class MaterialisationOutcome:
    """The result of one `materialise_and_register` call.

    `attempts` counts the bare link GET attempts actually made: `0` on the
    already-held no-op path, `1` on an ordinary successful fetch or a refusal reached
    before any GET. A `"degraded"` outcome's own count is however many GET attempts
    actually landed before the fetch gave up -- anywhere from `0` (a link lookup failure or a
    timeout before the first GET completed) up to `_ARTIFACT_GET_ATTEMPTS` (every attempt
    exhausted), never assumed to be the ceiling. `refusal` is set only when
    `status == "refused"`, and `degraded_reason` only when `status == "degraded"`. A
    `"store_write"` degrade's worker thread may still land the write. `role_keys` and
    `stat_var_count` are the frame's own role columns and distinct StatVars
    (`observed_shape`), and `family_summed` whether its `stat_var` column holds the
    family ref `dcid:<spec_id>` (`holds_family_ref`), all three set whenever this call
    read the held frame or parsed the fetched one: on every `"registered"` outcome and
    on a `"store_write"` degrade. `frame_ref` is the reference the frame is held under
    (`frame_ref_for`), set on every outcome but a refusal.
    """

    status: Literal["registered", "degraded", "refused"]
    attempts: int
    refusal: ToolRefusal | None = None
    degraded_reason: DegradedReason | None = None
    role_keys: tuple[str, ...] | None = None
    stat_var_count: int | None = None
    family_summed: bool | None = None
    frame_ref: FrameRef | None = None


async def materialise_and_register(
    *,
    manifest: Manifest,
    settings: Settings,
    writer: str,
    options: Mapping[str, Any],
    scope: Mapping[str, Any] | None = None,
    transport: httpx2.AsyncBaseTransport | None = None,
    deadline: CallDeadline | None = None,
) -> MaterialisationOutcome:
    """Fetch, parse, and register the frame `manifest` names, or report why it couldn't.

    `options` is the mint request's `options`. The frame registers under the
    reference `frame_ref_for` gives the manifest's pair, `options.columns`,
    `options.groupby` and `options.without_party_totals`. With none of them, that is
    the manifest's own pair, and the bytes
    are the version's canonical frame, found on the snapshot route
    (`_locate_canonical_frame`) and checked against the pair's `content_version`
    before the write (`_parse_and_register`). Otherwise the bytes are the artifact
    of the manifest whose `rendering` equals `options` (`_select_requested_artifact`).

    `scope` is the mint request's `scope`, `None` for a mint that sent none. Every
    outcome that read or parsed the frame also writes its mint record
    (`_write_mint_record`), whether this call registered the frame or found it held,
    so a frame stored without a record gains one the next time it is minted. A refusal writes none, and neither does a degraded outcome that saw no
    frame: the record states the frame's StatVar count, the store keeps the first
    record written under a pair, and one written without the count would stand for
    good.

    Called once per successful `get_data` mint, after the manifest itself has already
    validated. Checks the store first: a pair already held, read back under a row claim
    and passing every check that read makes, is a no-op with no fetch, since `get_data`
    may be called again for exactly this reason (see the degraded-fetch note
    `get_data_digest.py` renders). A held pair that fails the store's own integrity or
    size check on that read is refused rather than treated as a miss: the store is
    content-addressed and insert-only, so a held-but-broken object can't be healed by
    falling through to re-register (see `frame_integrity_failed`'s and `over_ceiling`'s
    own factories for what each case tells the caller). Otherwise fetches the
    requested artifact, bounds its row count from Parquet metadata alone
    (before any row is parsed), parses it into a `Frame`, and registers it under
    its frame reference.

    Returns a `"degraded"` outcome (never a refusal) when the artifact bytes could not be
    fetched within budget: the mint itself succeeded and named a real, durable result, and
    throwing that away would waste a call that can already run up to 125 seconds. A
    genuinely unusable manifest or artifact refuses instead -- see each refusal branch
    below for which.

    `deadline` bounds each step. The store read must end with `_MIN_FETCH_BUDGET_S`
    plus `_STORE_WRITE_RESERVE_S` left, and one that runs out is followed by a fetch
    whose failure degrades with `degraded_reason="store_unchecked"`. The
    fetch's wall-clock budget is `settings.artifact_fetch_budget_s`, cut to what
    `deadline` leaves once `_STORE_WRITE_RESERVE_S` is held back for the store write;
    under `_MIN_FETCH_BUDGET_S` the fetch is not started and the outcome is degraded.
    The store write must end with `_AFTER_WRITE_RESERVE_S` left, and one that runs out
    degrades with `degraded_reason="store_write"`.
    """
    stamp = _named_stamp(manifest, options=options)
    ref = _ref_or_refusal(manifest, options=options, stamp=stamp)
    if isinstance(ref, MaterialisationOutcome):
        return ref
    outcome = await _register(
        ref,
        manifest=manifest,
        options=options,
        stamp=stamp,
        settings=settings,
        writer=writer,
        transport=transport,
        deadline=deadline,
    )
    if outcome.status == "refused":
        return outcome
    if outcome.stat_var_count is not None:
        await _write_mint_record(
            MintRecord(
                frame=ref,
                # The pair `_ref_or_refusal` has already checked against `FrameRef`'s shape.
                dataset=FrameRef(
                    spec_id=manifest.definition.spec_id, content_version=manifest.content_version
                ),
                options=_sorted_by(options),
                scope=dict(scope or {}),
                stat_var_count=outcome.stat_var_count,
                enrichment_stamp=stamp,
            ),
            deadline=deadline,
        )
    return replace(outcome, frame_ref=ref)


def _sorted_by(options: Mapping[str, Any]) -> dict[str, Any]:
    """`options` with `groupby.by` sorted, as apis records it, the form a mint sends and
    its record keeps.

    Raises:
        ValueError: When `by` is no list of axis names
            (`common.materialisation.groupby_axes`).
    """
    groupby = options.get("groupby")
    if groupby is None:
        return dict(options)
    return {**options, "groupby": normal_groupby(groupby)}


async def _write_mint_record(record: MintRecord, *, deadline: CallDeadline | None) -> None:
    """Write `record` to the store while `_AFTER_WRITE_RESERVE_S` of `deadline` remains.

    The store keeps the first record written under a pair, so this is safe on every
    mint. A write that runs out of deadline is logged and left, since the frame is
    registered either way, the next mint of the pair writes the record again, and
    `data_mcp.mint_recovery.recover_mint_record` covers a root that has none.
    """
    write_deadline = deadline.reserving(_AFTER_WRITE_RESERVE_S) if deadline is not None else None
    try:
        async with asyncio.timeout(_seconds_left(write_deadline)):
            await get_store().put_mint_record(record)
    except TimeoutError:
        logger.warning(
            "materialise_and_register: the mint record write for %s/%s ran out of deadline",
            record.frame.spec_id,
            record.frame.content_version,
        )


async def _register(
    ref: FrameRef,
    *,
    manifest: Manifest,
    options: Mapping[str, Any],
    stamp: str | None,
    settings: Settings,
    writer: str,
    transport: httpx2.AsyncBaseTransport | None,
    deadline: CallDeadline | None,
) -> MaterialisationOutcome:
    """`materialise_and_register` for the frame held under `ref`, named under `stamp`
    (`_named_stamp`)."""
    store = get_store()
    checked = await _check_store(
        store, ref, max_frame_rows=settings.max_frame_rows, deadline=deadline
    )
    if isinstance(checked, MaterialisationOutcome):
        return checked
    fetch_failure_reason = checked

    # `frame_ref_for` keeps the mint's own pair exactly for the version's canonical
    # frame, and names every other rendering by its options. A manifest without the
    # requested rendering refuses whatever deadline is left.
    canonical = ref == FrameRef(
        spec_id=manifest.definition.spec_id, content_version=manifest.content_version
    )
    requested = None if canonical else _select_requested_artifact(manifest, options=options)
    if not canonical and requested is None:
        return _no_artifact_refusal(listed_by="the mint's manifest", rendering=options)

    budget_s = settings.artifact_fetch_budget_s
    if deadline is not None:
        budget_s = min(budget_s, deadline.remaining() - _STORE_WRITE_RESERVE_S)
        if budget_s < _MIN_FETCH_BUDGET_S:
            logger.warning(
                "materialise_and_register: only %.1fs of deadline left to fetch %s/%s; "
                "not starting the fetch",
                budget_s,
                ref.spec_id,
                ref.content_version,
            )
            return MaterialisationOutcome(
                status="degraded", attempts=0, degraded_reason=fetch_failure_reason
            )

    if canonical:
        started = asyncio.get_running_loop().time()
        located = await _locate_canonical_frame(
            ref, budget_s=budget_s, fetch_failure_reason=fetch_failure_reason
        )
        if isinstance(located, MaterialisationOutcome):
            return located
        artifact_digest, link = located
        budget_s -= asyncio.get_running_loop().time() - started
    else:
        assert requested is not None  # refused above
        artifact_digest, link = requested.artifact_digest, None
    progress = _FetchProgress()
    fetched = await _fetch_artifact_bytes(
        subject_id=ref.spec_id,
        # The artifact sits under apis' own pair, whatever the rendering.
        content_version=manifest.content_version,
        artifact_digest=artifact_digest,
        link=link,
        transport=transport,
        budget_s=budget_s,
        progress=progress,
    )
    if fetched is None:
        return MaterialisationOutcome(
            status="degraded", attempts=progress.attempts, degraded_reason=fetch_failure_reason
        )

    return await _verify_parse_and_register(
        fetched=fetched,
        artifact_digest=artifact_digest,
        canonical=canonical,
        ref=ref,
        manifest=manifest,
        options=options,
        stamp=stamp,
        settings=settings,
        writer=writer,
        store=store,
        write_deadline=deadline.reserving(_AFTER_WRITE_RESERVE_S) if deadline is not None else None,
    )


def _no_artifact_refusal(*, listed_by: str, rendering: Mapping[str, Any]) -> MaterialisationOutcome:
    """The refusal for a document that lists no artifact under `rendering`."""
    return MaterialisationOutcome(
        status="refused",
        attempts=0,
        refusal=ToolRefusal.upstream_response_unusable(
            detail=f"{listed_by} lists no artifact with the rendering "
            f"{json.dumps(dict(rendering), sort_keys=True)}"
        ),
    )


async def _locate_canonical_frame(
    ref: FrameRef, *, budget_s: float, fetch_failure_reason: DegradedReason
) -> tuple[str, str] | MaterialisationOutcome:
    """The digest and stable link of the canonical frame of the version `ref` names,
    read off the snapshot route within `budget_s`, or the outcome when there is none.

    A snapshot that lists no `CANONICAL_FRAME_RENDERING` artifact refuses as a
    manifest without the requested rendering does. A snapshot read that fails or runs
    out of `budget_s` degrades with `fetch_failure_reason`, as a failed fetch does.

    `artifact_link_by_rendering` is imported here for the reason
    `_fetch_artifact_bytes_uncapped` gives.
    """
    from data_mcp.artifact_resource import artifact_link_by_rendering  # noqa: PLC0415

    try:
        async with asyncio.timeout(budget_s):
            located = await artifact_link_by_rendering(
                ref.spec_id, ref.content_version, CANONICAL_FRAME_RENDERING
            )
    except (ResourceError, TimeoutError) as exc:
        logger.warning(
            "materialise_and_register: could not read the snapshot of %s/%s: %s",
            ref.spec_id,
            ref.content_version,
            exc or "out of budget",
        )
        return MaterialisationOutcome(
            status="degraded", attempts=0, degraded_reason=fetch_failure_reason
        )
    if located is None:
        return _no_artifact_refusal(
            listed_by="the version's snapshot", rendering=CANONICAL_FRAME_RENDERING
        )
    return located


async def _check_store(
    store: MaterialisationStore,
    ref: FrameRef,
    *,
    max_frame_rows: int,
    deadline: CallDeadline | None,
) -> MaterialisationOutcome | DegradedReason:
    """The outcome the store read settles on its own, or the reason a failed fetch
    after it reports.

    A held pair that passes every check the store's read makes is a no-op carrying
    the held frame's role keys and StatVar count, and a held pair that fails the
    store's integrity or size check refuses. A miss or an expired pair returns
    `"fetch"`, and a read that runs out of deadline returns `"store_unchecked"`,
    since the pair may be held. A held pair's rows are claimed
    before it is read (`_read_held`), a claim that does not fit refuses
    `service_unavailable`, and a pair over `max_frame_rows` refuses `over_ceiling`.
    """
    read_deadline = (
        deadline.reserving(_STORE_WRITE_RESERVE_S + _MIN_FETCH_BUDGET_S)
        if deadline is not None
        else None
    )
    try:
        async with asyncio.timeout(_seconds_left(read_deadline)):
            held = await _read_held(store, ref, max_frame_rows=max_frame_rows)
    except TimeoutError:
        logger.warning(
            "materialise_and_register: the store read for %s/%s ran out of deadline; "
            "fetching without knowing whether it holds the pair",
            ref.spec_id,
            ref.content_version,
        )
        return "store_unchecked"
    except FrameExpired:
        return "fetch"  # re-register fresh; `put` heals the expiry
    except RowBudgetExhausted as exc:
        return MaterialisationOutcome(
            status="refused",
            attempts=0,
            refusal=ToolRefusal.row_budget_full(
                requested=exc.requested, free=exc.free, max_rows=exc.max_rows
            ),
        )
    except FrameIntegrityError as exc:
        # Content-addressed and insert-only, so falling through to re-register would
        # just hit the identical corrupted object again -- see
        # `materialise_and_register`'s own docstring.
        return MaterialisationOutcome(
            status="refused",
            attempts=0,
            refusal=ToolRefusal.frame_integrity_failed(
                spec_id=ref.spec_id, content_version=ref.content_version, reason=exc.reason
            ),
        )
    except FrameTooLarge as exc:
        # Falls through for the same reason as `FrameIntegrityError` above, though
        # the held object is exactly what was written -- just over the store's own
        # serve-back ceiling or this service's row ceiling, so only a narrower scope
        # gets past it.
        return MaterialisationOutcome(
            status="refused",
            attempts=0,
            refusal=ToolRefusal.over_ceiling(detail=exc.reason, ceiling_kind="bytes"),
        )
    return "fetch" if held is None else held


async def _read_held(
    store: MaterialisationStore, ref: FrameRef, *, max_frame_rows: int
) -> MaterialisationOutcome | None:
    """The no-op outcome of the frame the store holds under `ref` (`_read_outcome`), or
    `None` when the store does not hold `ref`. The frame is read in full under a claim on its rows
    so a cache hit passes every check a read makes, the store's content-version check
    included, and is then discarded.

    The claim is the row count the store recorded, or `max_frame_rows` for an object
    stored without one, and the read refuses an object holding more rows than that
    before decoding it. Raises `FrameTooLarge` for a recorded count over
    `max_frame_rows`, `RowBudgetExhausted` when the rows do not fit, and whatever the
    store's `open` or `HeldFrame.read` raises. The claim stays held until the read's
    worker thread ends, even when the deadline cuts it off."""
    held = await store.open(ref)
    if held is None:
        return None
    if held.num_rows is not None and held.num_rows > max_frame_rows:
        raise FrameTooLarge(
            ref,
            reason=(
                f"stored frame has {held.num_rows} rows, over this service's "
                f"{max_frame_rows}-row ceiling"
            ),
        )
    rows = max_frame_rows if held.num_rows is None else held.num_rows
    with row_budget().claim(rows) as claim:
        frame, _trace = await claim.hold_through(held.read(max_rows=rows))
    return _read_outcome(frame, ref=ref, attempts=0)


def _seconds_left(deadline: CallDeadline | None) -> float | None:
    """What `asyncio.timeout` takes for `deadline`: `None` for no bound, never negative."""
    return max(deadline.remaining(), 0.0) if deadline is not None else None


async def _verify_parse_and_register(
    *,
    fetched: bytes,
    artifact_digest: str,
    canonical: bool,
    ref: FrameRef,
    manifest: Manifest,
    options: Mapping[str, Any],
    stamp: str | None,
    settings: Settings,
    writer: str,
    store: MaterialisationStore,
    write_deadline: CallDeadline | None,
) -> MaterialisationOutcome:
    """Verify `fetched` against `artifact_digest`, bound and parse it into a
    `Frame`, and register the result under `ref` before `write_deadline` -- or the
    outcome for whichever step failed. Split out of `materialise_and_register` to keep
    that function's own return-statement count within this repo's lint bound.

    The artifact's rows are claimed from the process's frame row budget
    (`common.row_budget`) once its footer gives their count, and held through the parse
    and until the store write ends; a claim that does not fit refuses
    `service_unavailable`.
    """
    computed_digest = f"sha256:{hashlib.sha256(fetched).hexdigest()}"
    if computed_digest != artifact_digest:
        logger.warning(
            "materialise_and_register: fetched bytes for %s/%s do not hash to the "
            "advertised artifact_digest (advertised %s, computed %s)",
            ref.spec_id,
            ref.content_version,
            artifact_digest,
            computed_digest,
        )
        return MaterialisationOutcome(
            status="refused",
            attempts=1,
            refusal=ToolRefusal.upstream_response_unusable(
                detail="the fetched artifact bytes do not hash to the manifest's "
                f"advertised artifact_digest {artifact_digest!r}; "
                f"computed {computed_digest!r}"
            ),
        )

    try:
        row_count = _parquet_row_count(fetched)
    except pa.ArrowException as exc:
        return _unparseable(exc)
    if row_count > settings.max_frame_rows:
        return MaterialisationOutcome(
            status="refused",
            attempts=1,
            refusal=ToolRefusal.over_ceiling(
                detail=f"the fetched artifact parsed to {row_count} rows, over this "
                f"service's {settings.max_frame_rows}-row ceiling"
            ),
        )
    try:
        claim = row_budget().claim(row_count)
    except RowBudgetExhausted as exc:
        return MaterialisationOutcome(
            status="refused",
            attempts=1,
            refusal=ToolRefusal.row_budget_full(
                requested=exc.requested, free=exc.free, max_rows=exc.max_rows
            ),
        )
    with claim:
        return await _parse_and_register(
            fetched=fetched,
            canonical=canonical,
            ref=ref,
            manifest=manifest,
            options=options,
            stamp=stamp,
            writer=writer,
            store=store,
            claim=claim,
            write_deadline=write_deadline,
        )


def _unparseable(exc: Exception) -> MaterialisationOutcome:
    """The refusal for fetched bytes that do not parse into a valid frame."""
    return MaterialisationOutcome(
        status="refused",
        attempts=1,
        refusal=ToolRefusal.upstream_response_unusable(
            detail=f"the fetched artifact did not parse into a valid frame: {exc}"
        ),
    )


async def _parse_and_register(
    *,
    fetched: bytes,
    canonical: bool,
    ref: FrameRef,
    manifest: Manifest,
    options: Mapping[str, Any],
    stamp: str | None,
    writer: str,
    store: MaterialisationStore,
    claim: RowClaim,
    write_deadline: CallDeadline | None,
) -> MaterialisationOutcome:
    """Parse `fetched` into a `Frame` and register it under `ref` before
    `write_deadline`, or the outcome for whichever step failed. `claim` holds the
    frame's rows until the store write ends, even when it outlives
    `write_deadline`.

    A `canonical` artifact is the version's canonical frame: it is read in the
    canonical layout (`frame_from_verified_parquet_bytes`, which refuses any other
    schema and trusts the cells), and registered only when its `frame_sha256` mints
    `ref.content_version`, so the frame held under an apis service pair is the one its
    token names. A mismatch refuses `upstream_response_unusable`. Any other artifact
    is a rendering, validated cell by cell (`frame_from_parquet_bytes`) and
    registered under its derived pair unchecked.

    The frame's trace keeps the mint's `options` and `stamp`, the stamp `ref` is named
    under, so `data_mcp.mint_recovery` can rebuild a derived pair from the held frame
    when its mint record is missing.
    """
    # `pyarrow.ArrowException` (a 2xx response whose body isn't a valid/complete Parquet
    # file) and `KeyError` (a Parquet file missing its `time_period` or `value` column,
    # raised while decoding it) are neither one a `FrameError`, but a genuinely unusable
    # artifact either way -- all three are refused as a `FrameError` is. The trusting
    # decoder checks no row order, so a canonical frame's rows out of order or twice
    # over refuse here too, when its hash streams them (`row_key_not_increasing`).
    try:
        frame = (
            frame_from_verified_parquet_bytes(fetched)
            if canonical
            else frame_from_parquet_bytes(fetched)
        )
        minted = mint_content_version(frame_sha256=frame_sha256(frame)) if canonical else None
    except (FrameError, pa.ArrowException, KeyError) as exc:
        return _unparseable(exc)
    if canonical and minted != ref.content_version:
        logger.warning(
            "materialise_and_register: the canonical frame of %s/%s re-mints %s",
            ref.spec_id,
            ref.content_version,
            minted,
        )
        return MaterialisationOutcome(
            status="refused",
            attempts=1,
            refusal=ToolRefusal.upstream_response_unusable(
                detail=f"the canonical frame of {ref.spec_id} hashes to {minted!r}, "
                f"not to its content_version {ref.content_version!r}"
            ),
        )

    trace = MaterialisationTrace(
        upstream_trace=manifest.trace,
        spec_id=ref.spec_id,
        content_version=ref.content_version,
        options=dict(options),
        enrichment_stamp=stamp,
    )
    try:
        async with asyncio.timeout(_seconds_left(write_deadline)):
            await claim.hold_through(store.put(ref, frame=frame, trace=trace, writer=writer))
    except TimeoutError:
        # The store is insert-only and content-addressed, so a write the worker
        # thread lands later is the same object a retry would write. The row claim
        # stays held until that thread's write ends.
        logger.warning(
            "materialise_and_register: the store write for %s/%s ran out of deadline unconfirmed",
            ref.spec_id,
            ref.content_version,
        )
        return _read_outcome(frame, ref=ref, attempts=1, degraded_reason="store_write")
    except FrameTooLarge as exc:
        # `_verify_parse_and_register`'s row_count check bounds against this
        # service's own `settings.max_frame_rows`; the store's own `_max_frame_rows`
        # is a second, independently-owned constant (see `gcs_store.py`'s own
        # module docstring for why the two are never coupled), so a frame
        # that passed that check can still be refused here if that
        # constant is ever set lower. No object landed, so there is nothing
        # to register -- a refusal, not a raise.
        return MaterialisationOutcome(
            status="refused",
            attempts=1,
            refusal=ToolRefusal.over_ceiling(detail=exc.reason, ceiling_kind="rows"),
        )
    return _read_outcome(frame, ref=ref, attempts=1)


def _read_outcome(
    frame: Frame, *, ref: FrameRef, attempts: int, degraded_reason: DegradedReason | None = None
) -> MaterialisationOutcome:
    """The outcome of a call that read or parsed `frame`, registered unless
    `degraded_reason` says otherwise, carrying what the frame itself shows."""
    role_keys, stat_var_count = observed_shape(frame)
    return MaterialisationOutcome(
        status="registered" if degraded_reason is None else "degraded",
        attempts=attempts,
        degraded_reason=degraded_reason,
        role_keys=role_keys,
        stat_var_count=stat_var_count,
        family_summed=holds_family_ref(frame, spec_id=ref.spec_id),
    )


def requests_kinds(columns: Iterable[str]) -> bool:
    """Whether `columns` asks apis for any of a flow party's kind, class or whole cells."""
    return not set(KIND_COLUMNS).isdisjoint(columns)


def recorded_stamp(columns: Iterable[str], stamp: str | None) -> str | None:
    """`stamp` as a frame minted with `columns` is named under (`frame_ref_for`) and its
    mint record keeps it: only a frame that asks for kinds is named by one."""
    return stamp if requests_kinds(columns) else None


def frame_ref_for(
    *,
    spec_id: str,
    content_version: str,
    columns: tuple[str, ...] = (),
    groupby: Mapping[str, Any] | None = None,
    without_party_totals: bool = False,
    enrichment_stamp: str | None = None,
    keep_by_order: bool = False,
) -> FrameRef:
    """The frame reference of the frame a mint names as `{spec_id, content_version}`,
    minted with `options.columns` set to `columns`, `options.groupby` to `groupby` and
    `options.without_party_totals` to `without_party_totals`.

    apis keeps one `content_version` for every rendering of a materialisation,
    so a frame minted with extra columns (`get_data(place_types=True)`), a sum that
    keeps the version, and one minted with the default columns come back under one
    pair. The default rendering, with no columns, no groupby and no cut, keeps the
    mint's own pair and holds the version's canonical frame. Any other names its options in its
    `content_version`: `cv_` and the first 32 hex digits of the SHA-256 of the
    RFC 8785 form of `{"columns": [...], "content_version": ...}`, with `columns`
    sorted, a `"groupby"` member holding `groupby` with its `by` sorted when it is set,
    and `"without_party_totals": true` when the cut is asked for. Two renderings
    of one pair are then two frames, and the store's insert-only entry for one is
    never served for the other. A groupby sum apis mints as a version of its own
    is still named by its groupby, since data-mcp cannot tell it from a sum that
    keeps the version: the versions route that says so needs a first-party
    principal this service does not hold.

    A `columns` that asks for `KIND_COLUMNS` also names `enrichment_stamp`, the digest
    of the code-index facts behind the kind cells of the artifact the frame is read
    from (`ArtifactEntry.enrichment_stamp`, `_named_stamp`), in the preimage as an
    `"enrichment_stamp"` member. apis keeps one
    `content_version` across a change to a party's kind, so without it the store
    would serve the frame holding the old kinds for as long as it keeps the entry.
    With no stamp, or with columns that ask for no kinds, the ref is the unsalted one.

    `options.without_party_totals` names the frame although apis mints a frame without
    party totals a version of its own: a cut that leaves nothing out serves its base's
    frame under its base's version, and a recipe replaying the frame read with the cut
    replays the cut, so the frame is named apart from the one read without it. apis'
    `analysis_roots.frame_ref_for` names root frames the same way.

    `keep_by_order` names `groupby` with `by` in the order it holds, the pair of a frame
    registered before this function sorted `by` (`named_groupbys`).

    Raises:
        ValueError: When `spec_id` and `content_version` do not fit `FrameRef`'s pair
            shape (`pydantic.ValidationError`), or `groupby`'s `by` is no list of axis
            names (`common.materialisation.groupby_axes`).
    """
    ref = FrameRef(spec_id=spec_id, content_version=content_version)
    if not columns and groupby is None and not without_party_totals:
        return ref
    preimage: dict[str, Any] = {"columns": sorted(columns), "content_version": content_version}
    if groupby is not None:
        preimage["groupby"] = (
            {**groupby, "by": groupby_axes(groupby)} if keep_by_order else normal_groupby(groupby)
        )
    if without_party_totals:
        preimage[WITHOUT_PARTY_TOTALS] = True
    if enrichment_stamp is not None and requests_kinds(columns):
        preimage["enrichment_stamp"] = enrichment_stamp
    rendering = rfc8785.dumps(preimage)
    return FrameRef(
        spec_id=spec_id, content_version=f"cv_{hashlib.sha256(rendering).hexdigest()[:32]}"
    )


# The longest `by` whose every order `named_groupbys` lists: 4 axes give 24 orders. get_data
# sums by two axes at most, and a longer `by` is listed in its sorted and recorded orders.
_MAX_PERMUTED_BY = 4


def named_groupbys(groupby: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """`groupby` with `by` in each order a held frame's pair may name it in, for
    `frame_ref_for(keep_by_order=True)`: sorted, as `frame_ref_for` names it, then as
    `groupby` records it, then, for a `by` of at most `_MAX_PERMUTED_BY` axes, every other
    order. A frame registered before `frame_ref_for` sorted `by` is held under the pair
    named with the order get_data sent, which a record or a version may no longer keep.

    Raises:
        ValueError: When `by` is no list of axis names
            (`common.materialisation.groupby_axes`).
    """
    by = groupby_axes(groupby)
    orders = [sorted(by), by]
    if len(by) <= _MAX_PERMUTED_BY:
        orders.extend(list(order) for order in permutations(by))
    unique = dict.fromkeys(tuple(order) for order in orders)
    return tuple({**groupby, "by": list(order)} for order in unique)


def _ref_or_refusal(
    manifest: Manifest, *, options: Mapping[str, Any], stamp: str | None
) -> FrameRef | MaterialisationOutcome:
    """The reference this frame registers under (`frame_ref_for`, named under `stamp`),
    or the refused outcome for a manifest whose `spec_id`/`content_version` combination
    doesn't match `FrameRef`'s own pair shape.
    """
    try:
        return frame_ref_for(
            spec_id=manifest.definition.spec_id,
            content_version=manifest.content_version,
            columns=tuple(options.get("columns") or ()),
            groupby=options.get("groupby"),
            without_party_totals=options.get(WITHOUT_PARTY_TOTALS) is True,
            enrichment_stamp=stamp,
        )
    except ValidationError as exc:
        return MaterialisationOutcome(
            status="refused",
            attempts=0,
            refusal=ToolRefusal.upstream_response_unusable(
                detail="the manifest's spec_id/content_version did not match the pair shape "
                f"this service expects: {exc}"
            ),
        )


def _named_stamp(manifest: Manifest, *, options: Mapping[str, Any]) -> str | None:
    """The stamp a frame minted with `options` is named under (`frame_ref_for`): the
    `enrichment_stamp` of the artifact it is read from (`_select_requested_artifact`),
    for a frame that asks for kinds (`recorded_stamp`). An artifact without one, such
    as a pinned file rendered under an earlier code index, names the frame unsalted.
    The canonical frame asks for no kinds."""
    artifact = _select_requested_artifact(manifest, options=options)
    stamp = None if artifact is None else artifact.enrichment_stamp
    return recorded_stamp(options.get("columns") or (), stamp)


def _select_requested_artifact(
    manifest: Manifest, *, options: Mapping[str, Any]
) -> ArtifactEntry | None:
    """The artifact whose `rendering` equals the mint request's `options`, or `None`.

    apis stores each artifact's `rendering` as the request's options with unset
    fields left out, and a manifest lists every rendering stored under its pair: other
    callers' groupby sums and column selections sit beside the requested one, in no
    set order. A manifest whose own artifact has expired lists none that matches.
    """
    return next((a for a in manifest.artifacts if a.rendering == options), None)


@dataclass
class _FetchProgress:
    """How many bare GETs a fetch has actually made so far.

    Mutable and passed by reference from `materialise_and_register` down through
    `_fetch_artifact_bytes` into `_fetch_artifact_bytes_uncapped`, rather than returned
    alongside the fetched bytes: `_fetch_artifact_bytes`'s own return shape (`bytes | None`)
    is a seam `data_mcp`'s and `harness`'s own test suites monkeypatch directly, and a
    `TimeoutError` cancelling the fetch coroutine mid-flight would lose a plain return
    value along with the cancelled frame anyway.
    """

    attempts: int = 0


async def _fetch_artifact_bytes(
    *,
    subject_id: str,
    content_version: str,
    artifact_digest: str,
    link: str | None = None,
    transport: httpx2.AsyncBaseTransport | None,
    budget_s: float,
    progress: _FetchProgress,
) -> bytes | None:
    """Fetch one artifact's bytes under one wall-clock budget, or return `None`.
    `link` is the artifact's root-relative stable link when the caller has read it
    already, else `None`.

    A crossed deadline mid-sequence returns `None` (degraded), the same outcome as
    exhausting the GET attempts inside `_fetch_artifact_bytes_uncapped` -- never a raised
    `TimeoutError`. `progress.attempts` records the number of bare GET attempts actually
    made, whichever way this returns.
    """
    try:
        async with asyncio.timeout(budget_s):
            return await _fetch_artifact_bytes_uncapped(
                subject_id=subject_id,
                content_version=content_version,
                artifact_digest=artifact_digest,
                link=link,
                transport=transport,
                progress=progress,
            )
    except TimeoutError:
        logger.warning(
            "materialise_and_register: artifact fetch for %s/%s exceeded the %.1fs budget "
            "after %d GET attempt(s)",
            subject_id,
            content_version,
            budget_s,
            progress.attempts,
        )
        return None


async def _fetch_artifact_bytes_uncapped(
    *,
    subject_id: str,
    content_version: str,
    artifact_digest: str,
    link: str | None,
    transport: httpx2.AsyncBaseTransport | None,
    progress: _FetchProgress,
) -> bytes | None:
    """One `artifact_link` call for the artifact's download link, unless `link` is
    given -- that function already retries internally against apis' snapshot
    route, so it is called once here, never wrapped in a second retry loop -- then up
    to `_ARTIFACT_GET_ATTEMPTS` GETs on the link through `UpstreamClient.stream_link`,
    which sends the ID token to apis only and follows its 302 to a freshly signed
    object-store URL. An ID token that cannot be minted fails an attempt as a
    transport error does. A `410` from the link (`_ARTIFACT_GONE_STATUS`) ends the
    fetch at once, since retrying cannot bring the artifact back; any other GET
    failure retries the link, which signs a new URL each time. `progress` is
    incremented once per GET actually attempted, whatever its outcome.

    `artifact_link` is imported here, not at module level: `data_mcp.artifact_resource`
    itself imports `_SPEC_ID_PATTERN` from `data_mcp.get_data_tool`, which imports
    `materialise_and_register` from this module -- a module-level import here would be
    circular. Deferred to first call, well after both modules have finished loading.
    """
    from data_mcp.artifact_resource import artifact_link  # noqa: PLC0415

    if link is None:
        try:
            link = await artifact_link(subject_id, content_version, artifact_digest)
        except ResourceError as exc:
            logger.warning(
                "materialise_and_register: could not read the artifact link for %s/%s: %s",
                subject_id,
                content_version,
                exc,
            )
            return None

    logger.info(
        "materialise_and_register: fetching artifact bytes for %s/%s", subject_id, content_version
    )
    upstream = get_client()
    for attempt in range(1, _ARTIFACT_GET_ATTEMPTS + 1):
        try:
            async with upstream.stream_link(
                link, timeout_s=_ARTIFACT_GET_TIMEOUT_S, transport=transport
            ) as response:
                status = response.status_code
                body = await response.aread() if response.is_success else None
        except httpx2.TransportError as exc:
            progress.attempts += 1
            logger.warning(
                "materialise_and_register: GET attempt %d/%d for %s/%s failed: %s",
                attempt,
                _ARTIFACT_GET_ATTEMPTS,
                subject_id,
                content_version,
                exc,
            )
            if attempt < _ARTIFACT_GET_ATTEMPTS:
                await asyncio.sleep(_ARTIFACT_GET_BACKOFF_S)
            continue
        progress.attempts += 1
        if body is not None:
            return body
        if status == _ARTIFACT_GONE_STATUS:
            logger.warning(
                "materialise_and_register: the link for %s/%s answered %d; not retrying",
                subject_id,
                content_version,
                status,
            )
            return None
        logger.warning(
            "materialise_and_register: GET attempt %d/%d for %s/%s returned status %d",
            attempt,
            _ARTIFACT_GET_ATTEMPTS,
            subject_id,
            content_version,
            status,
        )
        if attempt < _ARTIFACT_GET_ATTEMPTS:
            await asyncio.sleep(_ARTIFACT_GET_BACKOFF_S)
    logger.warning(
        "materialise_and_register: exhausted %d GET attempt(s) for %s/%s",
        _ARTIFACT_GET_ATTEMPTS,
        subject_id,
        content_version,
    )
    return None


def _parquet_row_count(data: bytes) -> int:
    """The artifact's row count, read from its own Parquet footer metadata only --
    never materialising row data, so an oversized artifact is bounded before any row is
    actually parsed. Raises `pyarrow.ArrowException` for bytes that aren't a valid or
    complete Parquet file; the caller (`materialise_and_register`) maps that to a refusal
    rather than letting it escape."""
    return pq.ParquetFile(io.BytesIO(data)).metadata.num_rows


_UTC_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_MICROSECONDS_PER_DAY = 86_400_000_000

# How many of each temporal type's stored integers make up one day.
_TICKS_PER_DAY = {"s": 86_400, "ms": 86_400_000, "us": 86_400_000_000, "ns": 86_400_000_000_000}


def _time_period_text(column: pa.ChunkedArray) -> pa.ChunkedArray | pa.Array:
    """`time_period` as `build_frame_from_arrow` reads it: text. A `date32`/`date64`
    column becomes `YYYY-MM-DD`, and a `timestamp` column its date portion, taken in
    the column's own time zone when it has one. Any other column passes through
    unchanged, so a string column (the common case for a real fetched artifact) is
    untouched and a malformed one is left for `build_frame_from_arrow` to refuse.

    Each distinct value is converted once with Python's `datetime` and `zoneinfo`
    from its stored integer, so a date outside `0001-01-01` to `9999-12-31` refuses
    `invalid_time_period`. Arrow's own casts are not used: a timestamp to `date32`
    cast wraps a day count past 32 bits to a wrong date, and its time zone rules
    stop applying daylight saving after 2037.
    """
    data_type = column.type
    if pa.types.is_timestamp(data_type):
        ticks_per_day = _TICKS_PER_DAY[data_type.unit]
        zone = _zone(data_type.tz) if data_type.tz is not None else None
    elif pa.types.is_date32(data_type):
        ticks_per_day, zone = 1, None
    elif pa.types.is_date64(data_type):
        ticks_per_day, zone = _TICKS_PER_DAY["ms"], None
    else:
        return column
    encoded = array_kernel("dictionary_encode", column.combine_chunks())
    assert isinstance(encoded, pa.DictionaryArray)
    ticks = encoded.dictionary.cast(pa.int32() if pa.types.is_date32(data_type) else pa.int64())
    dates = [
        _date_text(tick * _MICROSECONDS_PER_DAY // ticks_per_day, zone)
        for tick in ticks.to_pylist()
    ]
    return array_kernel("take", pa.array(dates, type=pa.string()), encoded.indices)


def _zone(name: str) -> tzinfo:
    """The time zone an Arrow timestamp type names: a fixed `+HH:MM` offset or an
    IANA zone. One that names neither refuses `invalid_time_period`."""
    try:
        if name[:1] in "+-":
            zone = datetime.strptime(name, "%z").tzinfo
            assert zone is not None
            return zone
        return ZoneInfo(name)
    except (ValueError, ZoneInfoNotFoundError) as exc:
        raise FrameError(
            "invalid_time_period", f"time_period's time zone {name!r} is not known"
        ) from exc


def _date_text(microseconds: int, zone: tzinfo | None) -> str:
    """The `YYYY-MM-DD` date of the instant `microseconds` after the Unix epoch, in
    `zone` or in UTC, or a refusal when it falls outside Python's `date` range."""
    try:
        moment = _UTC_EPOCH + timedelta(microseconds=microseconds)
        if zone is not None:
            moment = moment.astimezone(zone)
    except OverflowError as exc:
        raise FrameError(
            "invalid_time_period",
            f"time_period is outside 0001-01-01 to 9999-12-31: {microseconds} "
            "microseconds from the Unix epoch",
        ) from exc
    return moment.date().isoformat()


def _value_numbers(column: pa.ChunkedArray) -> pa.ChunkedArray | pa.Array:
    """`value` as `build_frame_from_arrow` reads it: an integer or floating column.
    A decimal column becomes `float64` through Python's `float(Decimal)`, applied once
    per distinct value, because Arrow's own decimal-to-double cast is not correctly
    rounded. A null stays null for `build_frame_from_arrow` to refuse. Any other column
    passes through unchanged.
    """
    if not pa.types.is_decimal(column.type):
        return column
    encoded = array_kernel("dictionary_encode", column.combine_chunks())
    assert isinstance(encoded, pa.DictionaryArray)
    floats = pa.array([float(cell) for cell in encoded.dictionary.to_pylist()], type=pa.float64())
    return array_kernel("take", floats, encoded.indices)


def _check_unit_requests(column: pa.ChunkedArray) -> None:
    """Refuse a struct `unit` column whose requests fail `validate_unit_request` when
    every struct field counts as a member, a null one included.

    A struct column carries `{currency, prices, base_year}` with every field on every
    row, so a current-price unit arrives with a null `base_year`; its null fields are
    dropped before the check, as `build_frame_from_arrow` drops them, so it passes. A struct
    in the frame's own tagged form (`ref`, `request`, `composed`), a current-price unit
    with a base year, or one with no currency is refused `invalid_unit`. Each distinct value is
    checked once. A text or null column passes.
    """
    if not pa.types.is_struct(column.type):
        return
    values = column.combine_chunks()
    for request in values.take(unit_codes(values).first_rows).to_pylist():
        if request is not None:
            validate_unit_request(
                {key: value for key, value in request.items() if value is not None}
            )


_UNIT_TYPE = canonical_frame_schema((UNIT_COLUMN,)).field(UNIT_COLUMN).type


def _tagged_unit_from_text(text: object, *, composed: bool) -> object:
    """One text `unit` cell as the value of the frame's tagged unit struct.

    apis writes a text `unit` column (`materialise.formats._cell_text`): a unit ref
    as its own text, and a converted or composed unit as the JCS text of its
    `as_json` object. A unit ref never starts with `{`, so a cell that does is an
    object. It must parse as JSON, which makes it an object, and the grammar's
    `unit_cell_from_json` must read it, so Arrow never drops an unknown member or fails
    on a mistyped one while building the struct. A composed unit (`{numerator,
    denominator}`) is accepted only when `composed` is set, for a frame the analysis
    runner computed. Otherwise it is refused `invalid_unit`, since an artifact apis
    mints holds none.
    """
    if text is None:
        return None
    cell = cast(str, text)
    if cell.startswith("{"):
        try:
            unit = unit_cell_from_json(json.loads(cell))
        except json.JSONDecodeError as exc:
            raise FrameError(
                "schema_violation", f"unit text {cell[:80]!r} is not a JSON unit object"
            ) from exc
        if not isinstance(unit, UnitRequest) and not (composed and isinstance(unit, ComposedUnit)):
            raise FrameError(
                "invalid_unit", f"unit text {cell[:80]!r} is not a converted-unit object"
            )
    else:
        unit = unit_cell_from_json(cell)
    return unit_cell_array(unit, 1)[0].as_py()


def _with_text_units_decoded(table: pa.Table, *, composed: bool) -> pa.Table:
    """`table` with a text `unit` column decoded into the frame's tagged unit struct
    (`_tagged_unit_from_text`, composed units accepted under `composed`), each distinct
    cell once. A table without a `unit` column, or with a struct or null one, is
    returned unchanged."""
    if UNIT_COLUMN not in table.schema.names:
        return table
    index = table.schema.get_field_index(UNIT_COLUMN)
    column_type = table.schema.field(index).type
    if not (pa.types.is_string(column_type) or pa.types.is_large_string(column_type)):
        return table
    decoded = map_units(
        table.column(index),
        partial(_tagged_unit_from_text, composed=composed),
        result_type=_UNIT_TYPE,
    )
    return table.set_column(index, UNIT_COLUMN, decoded)


def observed_shape(frame: Frame) -> tuple[tuple[str, ...], int]:
    """`frame`'s role columns and its distinct `stat_var` count.

    The role columns are every column outside `_RESERVED_COLUMNS`, in the order
    `grammar.derive_columns` gives them, which is `normalize_role_keys`' sorted order.
    An analysis-mcp verb declares the same pair as its result's `role_keys` and
    `stat_var_count` (`analysis_mcp.result.observed_shape`)."""
    role_keys = tuple(column for column in frame.columns if column not in _RESERVED_COLUMNS)
    return role_keys, len(frame.table.column(STAT_VAR_COLUMN).unique())


def holds_family_ref(frame: Frame, *, spec_id: str) -> bool:
    """Whether `frame`'s `stat_var` column holds the family ref `dcid:<spec_id>`.

    apis writes that ref only on the series of a `groupby` that summed the family's
    StatVars (`datasets_api.render`), so it marks a summed frame whatever the version's
    description or warnings say."""
    return pc.index(frame.table.column(STAT_VAR_COLUMN), f"dcid:{spec_id}").as_py() >= 0


def frame_from_parquet_bytes(data: bytes, *, composed_units: bool = False) -> Frame:
    """Parse `data` (a whole Parquet file's bytes) into a `Frame`.

    `role_keys` is every column name in the table's own schema minus the grammar's
    reserved set (`_RESERVED_COLUMNS`) -- whatever entity/geography roles this dataset's
    shape declares. `status` is reserved and `build_frame_from_arrow` refuses it if the
    table carries it at all, so it is dropped here. Raises `FrameError` for a genuinely
    malformed artifact, `pyarrow.ArrowException` for bytes that aren't a valid or
    complete Parquet file, and `KeyError` for a Parquet file missing `time_period` or
    `value`; the caller (`materialise_and_register`) maps all three to a refusal rather
    than letting any escape. A malformed artifact is refused whatever its row count: a
    zero-row file missing `stat_var` or `time_period`, or typing a role, `time_period` or
    `unit` column as an integer, is refused like a populated one.

    A text `unit` column holds unit refs and converted units as JCS object text, each
    decoded into the frame's own unit cell (`_with_text_units_decoded`). A frame the
    analysis runner computed can hold a composed unit too, the JCS text of its
    `{numerator, denominator}` object, which `composed_units` accepts. A minted artifact
    holds none, so it is refused `invalid_unit` by default.

    A `{role}_type` column, or a party's `{role}_party_kind`, `{role}_party_class` or
    `{role}_party_whole` column (`common.place_type.enrichment_columns`), which
    `get_data(place_types=True)` asks for, is a role key too, with each empty cell
    written `UNTYPED` (`_with_untyped_places`).
    """
    table = _with_untyped_places(pq.read_table(io.BytesIO(data)))
    role_keys = [name for name in table.schema.names if name not in _RESERVED_COLUMNS]
    time_period = _time_period_text(table.column(TIME_PERIOD_COLUMN))
    value = _value_numbers(table.column(VALUE_COLUMN))
    table = table.set_column(
        table.schema.get_field_index(TIME_PERIOD_COLUMN), TIME_PERIOD_COLUMN, time_period
    )
    table = table.set_column(table.schema.get_field_index(VALUE_COLUMN), VALUE_COLUMN, value)
    if UNIT_COLUMN in table.schema.names:
        _check_unit_requests(table.column(UNIT_COLUMN))
    table = _with_text_units_decoded(_with_null_optionals_as_text(table), composed=composed_units)
    if STATUS_COLUMN in table.schema.names:
        table = table.drop_columns(STATUS_COLUMN)
    return build_frame_from_arrow(table, role_keys=role_keys)


def _with_untyped_places(table: pa.Table) -> pa.Table:
    """`table` with each empty or null cell of every enrichment column of a role it
    carries (`common.place_type.enrichment_role`) written as `common.place_type.UNTYPED`.

    apis leaves the cell empty for a place its graph holds no type for, for a party it
    declares no kind for, and in the class column of a total holding parties of more than
    one class, and `build_frame_from_arrow` refuses an empty role cell. A column typed null holds no value at all. A column of another type is left
    unchanged, for `build_frame_from_arrow` to refuse.
    """
    roles = [name for name in table.schema.names if name not in _RESERVED_COLUMNS]
    for name in roles:
        if enrichment_role(name, roles=roles) is None:
            continue
        index = table.schema.get_field_index(name)
        column = table.column(index).combine_chunks()
        if pa.types.is_null(column.type):
            table = table.set_column(index, name, pa.array([UNTYPED] * len(column), pa.string()))
            continue
        if not pa.types.is_string(column.type) and not pa.types.is_large_string(column.type):
            continue
        empty = array_kernel(
            "or_kleene",
            array_kernel("is_null", column),
            array_kernel("equal", column, pa.scalar("", type=column.type)),
        )
        filled = array_kernel("if_else", empty, pa.scalar(UNTYPED, type=column.type), column)
        table = table.set_column(index, name, filled)
    return table


_OPTIONAL_COLUMNS = frozenset({UNIT_COLUMN, *FACET_IDENTITY_COLUMNS})


def _with_null_optionals_as_text(table: pa.Table) -> pa.Table:
    """`table` with each optional column (`unit` and the facet identity columns) whose
    rows are all null, in a populated table, retyped as a null string column.

    `build_frame_from_arrow` accepts a text or null-typed optional column but refuses
    any other type, even when every cell is null; a writer that emits such a column
    as `float64`, a list of nulls, or a struct means "absent". Only a struct `unit`
    column keeps its type (it is the converted-unit type), and a zero-row column keeps
    its type too so a malformed type still refuses.
    """
    for index, field in enumerate(table.schema):
        column = table.column(index)
        if (
            field.name in _OPTIONAL_COLUMNS
            and len(column) > 0
            and column.null_count == len(column)
            and not (field.name == UNIT_COLUMN and pa.types.is_struct(field.type))
        ):
            table = table.set_column(index, field.name, pa.nulls(len(column), pa.string()))
    return table

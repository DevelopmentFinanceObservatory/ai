"""Builds the wire body for dfo-apis' publish door
(`POST /api/datasets/frames/publish`) out of an analysis-mcp-minted frame
already sitting in the shared materialisation store, and reads that store
to get it.

`PlatformAssembler`'s own publish hook (`client.py`) calls `publish_body_for`
before a platform call that needs an `ops_`/`ocv_` ref published, and posts
its result verbatim. dfo-apis pins a different `dfo-spec-kit` revision than
this repo, so it re-renders and re-digests frame1 from `role_keys`/`rows` on
its own side (see `publish_models.PublishFrameRequest`'s own docstring on the
dfo-apis side for why). `publish_body` only reproduces `dfo_frame_grammar.Frame
.build`'s own two arguments from an already-built `Frame`, never a
grammar-rev-dependent rendering.
"""

from __future__ import annotations

import io
import json
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Sequence
from contextlib import asynccontextmanager, contextmanager
from typing import NamedTuple

from common.materialisation import (
    FrameExpired,
    FrameIntegrityError,
    FrameRef,
    FrameTooLarge,
    HeldFrame,
    StoredFrameTrace,
    get_store,
)
from common.place_type import type_column
from common.row_budget import RowClaim, row_budget
from dfo_frame_grammar import (
    FACET_IDENTITY_COLUMNS,
    STAT_VAR_COLUMN,
    STATUS_COLUMN,
    TIME_PERIOD_COLUMN,
    UNIT_COLUMN,
    VALUE_COLUMN,
    Frame,
)
from dfo_frame_grammar.columnar import unit_cells, unit_codes
from dfo_frame_grammar.grammar import UnitRequest
from pydantic import ValidationError

# Every reserved column name `Frame.build` itself recognises -- everything
# else in `frame.columns` is a role key. Mirrors
# `common.materialisation.frame_codec`'s own `_RESERVED_COLUMNS` and
# `data_mcp.frame_materialisation`'s own copy of the identical set.
_RESERVED_COLUMNS = frozenset(
    {STAT_VAR_COLUMN, TIME_PERIOD_COLUMN, VALUE_COLUMN, UNIT_COLUMN, STATUS_COLUMN}
    | set(FACET_IDENTITY_COLUMNS)
)


class PublishBodyNotBuildable(Exception):  # noqa: N818 -- named for what it reports, not a generic "Error"
    """This process cannot build a publish body for this exact
    `(spec_id, content_version)` pair.

    `refuse_locally=False` means this call is silent on purpose: the pair's
    own shape doesn't even fit `FrameRef`'s strict id pattern (a caller-
    supplied ref that merely looks like an `ops_`/`ocv_` pair; every real
    analysis-mcp mint fits it), or the shared store holds nothing under it
    right now (never minted in this process, evicted, or expired). Neither
    is a publish FAILURE -- there is nothing yet to publish -- so the
    caller should proceed to its own platform call unmodified and let the
    platform, the one party that knows whether the ref has ever resolved
    anywhere, answer.

    `refuse_locally=True` means the store DOES hold the frame but this
    process still cannot build a valid publish body from it: today, only a
    stored trace naming no input content versions (a `Trace` predating that
    field), or a corrupted/oversized stored object the store's own read-time
    checks caught. dfo-apis' publish door would refuse the identical body on
    every retry, so this refuses now instead of letting that repeat.
    """

    def __init__(self, *, refuse_locally: bool, message: str) -> None:
        self.refuse_locally = refuse_locally
        super().__init__(message)


# Rows converted to Python objects at once. Bounds the body build's
# Python-object peak to one slice of rows plus the encoded body.
_SLICE_ROWS = 8_192

# The encoding httpx applies to a `json=` request body; `publish_body` emits
# the bytes it would for the same body object.
_ENCODER = json.JSONEncoder(ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _wire_row_slices(frame: Frame, *, left_out: Sequence[str]) -> Iterator[list[dict[str, object]]]:
    """`frame`'s rows as `Frame.build` row mappings, `_SLICE_ROWS` at a time.

    `status` is dropped from every row: it is reserved, always `None` at
    this grammar version, and `Frame.build` refuses a row mapping that
    supplies it at all. So is every role column in `left_out`. A `unit` cell
    holding a unit request (a converted frame's own canonical unit) becomes
    `UnitRequest.as_json()`, the one plain-JSON shape `Frame.build`'s own
    `_normalize_unit` accepts back for it; `None` and a bare string unit ref
    pass through unchanged. Each distinct unit converts once, and rows
    sharing it share the result.
    """
    table = frame.table.drop_columns([STATUS_COLUMN, *left_out])
    unit = table.column(UNIT_COLUMN)
    codes = unit_codes(unit)
    units = [
        cell.as_json() if isinstance(cell, UnitRequest) else cell
        for cell in unit_cells(unit, codes)
    ]
    # The code column keeps `unit`'s position, so each row's keys stay in
    # `frame.columns` order when the code is swapped for its cell below.
    table = table.set_column(table.schema.get_field_index(UNIT_COLUMN), UNIT_COLUMN, codes.codes)
    for offset in range(0, table.num_rows, _SLICE_ROWS):
        rows = table.slice(offset, _SLICE_ROWS).to_pylist()
        for row in rows:
            row[UNIT_COLUMN] = units[row[UNIT_COLUMN]]
        yield rows


def publish_body(
    frame: Frame,
    *,
    spec_id: str,
    content_version: str,
    input_content_versions: Sequence[str],
) -> bytes:
    """The publish door's JSON body for `frame`, encoded as httpx encodes a
    `json=` body.

    `role_keys` and `rows` are `dfo_frame_grammar.Frame.build`'s own two
    arguments, the inverse of what `Frame.build` does to construct `frame`
    (see `_wire_row_slices` for the row mappings). Rows are encoded one
    slice at a time, so the whole row list never exists as Python objects.

    A place-type column (`common.place_type`) beside its place column is left
    out. dfo-apis labels each series and table row by joining every role's
    value, so the column would label a continent "Africa / Continent". It is a
    function of its place column, so the published rows stay distinct.
    """
    encode = _ENCODER.encode
    roles = [name for name in frame.columns if name not in _RESERVED_COLUMNS]
    type_columns = {type_column(role) for role in roles}
    place_types = [role for role in roles if role in type_columns]
    role_keys = [role for role in roles if role not in place_types]
    body = io.BytesIO()
    body.write(
        f'{{"spec_id":{encode(spec_id)},"content_version":{encode(content_version)},'
        f'"role_keys":{encode(role_keys)},"rows":['.encode()
    )
    for index, rows in enumerate(_wire_row_slices(frame, left_out=place_types)):
        if index:
            body.write(b",")
        # A slice is never empty, so this strips exactly its own brackets.
        body.write(encode(rows)[1:-1].encode())
    body.write(f'],"input_content_versions":{encode(list(input_content_versions))}}}'.encode())
    return body.getvalue()


def _input_content_versions(trace: StoredFrameTrace) -> tuple[str, ...] | None:
    """Every input frame's own content_version this derived frame was
    computed from, read duck-typed off the concrete trace type analysis-mcp
    actually stores (`analysis_mcp.trace.Trace.input_content_versions`).

    `StoredFrameTrace` itself only guarantees `upstream_trace`/
    `frame_content_version` (`common.materialisation.trace`'s own
    docstring); presentation-mcp carries no dependency on analysis-mcp to
    name that concrete type, so this reads the field structurally instead.
    `None` when it is absent or empty: a `Trace` predating the field, or a
    bare `MaterialisationTrace` (a shape that never carries an `ops_`/
    `ocv_` pair in the first place, so never reaches this function in
    practice).
    """
    values = getattr(trace, "input_content_versions", None)
    if not values:
        return None
    return tuple(values)


class PublishBody(NamedTuple):
    """An encoded publish body, the number of frame rows it carries, and the
    `RowClaim.hold_through` of the claim holding those rows. Work on `data` that
    can outlive the call awaiting it, such as a compression thread, runs through
    `hold_through` so the claim stays held until that work ends."""

    data: bytes
    rows: int
    hold_through: Callable[[Awaitable[bytes]], Awaitable[bytes]]


@asynccontextmanager
async def publish_body_for(
    *, spec_id: str, content_version: str, max_frame_rows: int
) -> AsyncIterator[PublishBody]:
    """The encoded `publish_body` to publish `{spec_id, content_version}`
    with, and its row count, read out of the shared materialisation store
    (`common.materialisation.get_store`) -- the same process-wide store
    analysis-mcp's own lifespan installs and every one of its verbs writes
    results into.

    Raises `PublishBodyNotBuildable` on entry -- see that exception's own
    docstring for what `refuse_locally` means and every case that sets it.

    The frame's rows are claimed from the process's frame row budget
    (`common.row_budget`) once the store has opened the frame
    (`MaterialisationStore.open`) and before it reads it, and held until the
    block this manages exits, so the claim covers the read, the encoding and
    the publish request that sends the body. The claim is the row count the
    store recorded, or `max_frame_rows` for a frame stored without one, and
    the read refuses a frame holding more rows than that before decoding it.
    A recorded count over `max_frame_rows` refuses locally.
    `RowBudgetExhausted` propagates on entry when the rows do not fit.
    """
    try:
        ref = FrameRef(spec_id=spec_id, content_version=content_version)
    except ValidationError as exc:
        raise PublishBodyNotBuildable(refuse_locally=False, message=str(exc)) from exc

    with _store_failures():
        held = await get_store().open(ref)
    if held is None:
        raise PublishBodyNotBuildable(
            refuse_locally=False,
            message=f"no frame is held locally for {spec_id!r}/{content_version!r}",
        )
    if held.num_rows is not None and held.num_rows > max_frame_rows:
        raise PublishBodyNotBuildable(
            refuse_locally=True,
            message=(
                f"the frame held for {spec_id!r}/{content_version!r} has {held.num_rows} "
                f"rows, over this service's {max_frame_rows}-row ceiling"
            ),
        )
    rows = max_frame_rows if held.num_rows is None else held.num_rows
    with row_budget().claim(rows) as claim:
        yield await _read_and_encode(
            held, claim=claim, spec_id=spec_id, content_version=content_version
        )


@contextmanager
def _store_failures() -> Iterator[None]:
    """Turns a store failure on `open` or `HeldFrame.read` into the
    `PublishBodyNotBuildable` it means on this read-only path."""
    try:
        yield
    except FrameExpired as exc:
        # Expired, with nothing to heal it on this read-only path (unlike
        # `data_mcp.frame_materialisation`, which re-registers on this exact
        # exception) -- treated the same as never minted.
        raise PublishBodyNotBuildable(refuse_locally=False, message=str(exc)) from exc
    except (FrameIntegrityError, FrameTooLarge) as exc:
        # The store DOES hold something under this pair, but it is broken
        # or too large to serve back -- a genuine local problem, not "not
        # published yet".
        raise PublishBodyNotBuildable(refuse_locally=True, message=str(exc)) from exc


async def _read_and_encode(
    held: HeldFrame, *, claim: RowClaim, spec_id: str, content_version: str
) -> PublishBody:
    """`publish_body_for`'s read and encoding under `claim`. The decoded frame
    is dropped on return, so only the encoded body lives through the publish
    request."""
    with _store_failures():
        frame, trace = await claim.hold_through(held.read(max_rows=claim.rows))
    input_content_versions = _input_content_versions(trace)
    if input_content_versions is None:
        raise PublishBodyNotBuildable(
            refuse_locally=True,
            message=(
                f"the frame held for {spec_id!r}/{content_version!r} carries no input "
                "content versions in its own derivation trace, so it cannot be published"
            ),
        )
    data = publish_body(
        frame,
        spec_id=spec_id,
        content_version=content_version,
        input_content_versions=input_content_versions,
    )
    return PublishBody(data=data, rows=frame.table.num_rows, hold_through=claim.hold_through)


__all__ = ["PublishBody", "PublishBodyNotBuildable", "publish_body", "publish_body_for"]

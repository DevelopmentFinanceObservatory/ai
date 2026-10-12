"""Builds the wire body for dfo-apis' publish door
(`POST /api/datasets/frames/publish`) out of an analysis-mcp-minted frame
already sitting in the shared materialisation store, and reads that store
to get it.

`PlatformAssembler`'s own publish hook (`client.py`) calls `publish_body_for`
before a platform call that needs an `ops_`/`ocv_` ref published, and posts
its result verbatim. dfo-apis pins its own `dfo-spec-kit` revision, which can
differ from this repo's, so it re-renders and re-digests frame1 from
`role_keys`/`rows` on its own side (see `publish_models.PublishFrameRequest`'s own docstring on the
dfo-apis side for why). `publish_body` only reproduces `dfo_frame_grammar.Frame
.build`'s own two arguments from an already-built `Frame`, never a
grammar-rev-dependent rendering.

The body also carries the frame's `recipe` (`common.recipe.build_recipe`): the
tree of op-specs that computed it, its roots' mint records, and the two hashes
dfo-apis indexes it by. A frame whose recipe cannot be built is published
without the field and the reason is logged at warning with the pair: an
ancestor's trace has left the store, or a root was registered before data-mcp
wrote mint records and has not been minted since. This service has no recovery
routine for such a root (`data_mcp.mint_recovery` needs dfo-apis' dataset
versions), so its recipe is written later through dfo-apis' recipe route.

dfo-apis' publish body forbids unknown fields, so a dfo-apis deployment that
predates `recipe` refuses every body that carries one. dfo-apis must therefore
deploy first. There is no retry without the field: a publish that dropped the
recipe on a refusal would leave the frame without one for good.

A caller that has already built the recipe passes it to `publish_body_for`, which
then sends that recipe or refuses: a frame such a caller publishes is of no use
to it without one.
"""

from __future__ import annotations

import io
import json
import logging
import re
import unicodedata
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Mapping, Sequence
from contextlib import asynccontextmanager, contextmanager
from typing import Any, NamedTuple

from common.materialisation import (
    ComposedStatVar,
    FrameExpired,
    FrameIntegrityError,
    FrameRef,
    FrameTooLarge,
    HeldFrame,
    StoredFrameTrace,
    composed_stat_vars_of,
    get_store,
)
from common.place_type import enrichment_role
from common.recipe import RecipeError, build_recipe
from common.row_budget import RowClaim, row_budget
from dfo_frame_grammar import (
    FACET_IDENTITY_COLUMNS,
    STAT_VAR_COLUMN,
    STATUS_COLUMN,
    TIME_PERIOD_COLUMN,
    UNIT_COLUMN,
    VALUE_COLUMN,
    Frame,
    unit_cell_to_json,
    unit_cells,
    unit_codes,
)
from pydantic import ValidationError

logger = logging.getLogger(__name__)

# dfo-apis' `MAX_RECIPE_BYTES`: the most bytes of compact JSON its publish door
# accepts as a recipe. A larger one is left out of the body (`_recipe_or_none`), so
# the door never refuses a frame for the size of its recipe.
_RECIPE_CEILING_BYTES = 1_000_000

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
    goes out as `unit_cell_to_json` gives it, the plain-JSON form `Frame.build`
    accepts back: `None`, a bare string unit ref, a converted unit's
    `{currency, prices, base_year}` object, or a composed unit's
    `{numerator, denominator}` object. Each distinct unit converts once, and
    rows sharing it share the result.
    """
    table = frame.table.drop_columns([STATUS_COLUMN, *left_out])
    unit = table.column(UNIT_COLUMN)
    codes = unit_codes(unit)
    units = [unit_cell_to_json(cell) for cell in unit_cells(unit, codes)]
    # The code column keeps `unit`'s position, so each row's keys stay in
    # `frame.columns` order when the code is swapped for its cell below.
    table = table.set_column(table.schema.get_field_index(UNIT_COLUMN), UNIT_COLUMN, codes.codes)
    for offset in range(0, table.num_rows, _SLICE_ROWS):
        rows = table.slice(offset, _SLICE_ROWS).to_pylist()
        for row in rows:
            row[UNIT_COLUMN] = units[row[UNIT_COLUMN]]
        yield rows


# The publish door's bounds on `composed_stat_vars`: entries in the list, side
# refs across the whole list, characters in one ref, counted as written and
# again after NFC, and the shape of a ref.
_MAX_COMPOSED_STAT_VARS = 16
_MAX_COMPOSED_SIDE_REFS = 64
_MAX_COMPOSED_REF_CHARS = 200
_COMPOSED_REF = re.compile(r"dcid:.")


def _door_bound_broken(entries: Sequence[ComposedStatVar]) -> str | None:
    """The name of the first bound of the publish door that `entries` breaks
    as `composed_stat_vars`, or `None` when the door accepts the list."""
    sides = [side for entry in entries for side in (entry.first, entry.second)]
    refs = [entry.stat_var for entry in entries] + [ref for side in sides for ref in side]
    position = {entry.stat_var: index for index, entry in enumerate(entries)}
    broken = {
        "entries": len(entries) > _MAX_COMPOSED_STAT_VARS,
        "empty_side": not all(sides),
        "side_refs": sum(map(len, sides)) > _MAX_COMPOSED_SIDE_REFS,
        "ref_nul": any("\x00" in ref for ref in refs),
        "ref_length": any(
            max(len(ref), len(unicodedata.normalize("NFC", ref))) > _MAX_COMPOSED_REF_CHARS
            for ref in refs
        ),
        "ref_shape": not all(_COMPOSED_REF.match(ref) for ref in refs),
        "duplicate_stat_var": len(position) < len(entries),
        # A side ref that names an entry names an earlier one.
        "entry_order": any(
            position.get(ref, -1) >= index
            for index, entry in enumerate(entries)
            for ref in (*entry.first, *entry.second)
        ),
    }
    return next((bound for bound, is_broken in broken.items() if is_broken), None)


def publish_body(
    frame: Frame,
    *,
    spec_id: str,
    content_version: str,
    input_content_versions: Sequence[str],
    recipe: Mapping[str, Any] | None = None,
    composed_stat_vars: Sequence[ComposedStatVar] = (),
) -> bytes:
    """The publish door's JSON body for `frame`, encoded as httpx encodes a
    `json=` body. `recipe` is sent as the body's `recipe` member, and `None`
    leaves the member out.

    `role_keys` and `rows` are `dfo_frame_grammar.Frame.build`'s own two
    arguments, the inverse of what `Frame.build` does to construct `frame`
    (see `_wire_row_slices` for the row mappings). Rows are encoded one
    slice at a time, so the whole row list never exists as Python objects.

    A place-type or party-kind column (`common.place_type.enrichment_role`)
    beside its place column is left out. dfo-apis labels each series and table
    row by joining every role's value, so the column would label a continent
    "Africa / Continent". It is a function of its place column, so the
    published rows stay distinct.

    `composed_stat_vars` is the body's last member, each entry written as
    `{stat_var, relation, first, second}`. The member is left out when the
    list is empty, and when the list breaks one of the door's bounds
    (`_door_bound_broken`): the door would refuse the whole body, so the
    frame publishes without the names those entries give its series, and
    the bound is logged.
    """
    encode = _ENCODER.encode
    roles = [name for name in frame.columns if name not in _RESERVED_COLUMNS]
    place_types = [role for role in roles if enrichment_role(role, roles=roles) is not None]
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
    body.write(f'],"input_content_versions":{encode(list(input_content_versions))}'.encode())
    if recipe is not None:
        body.write(f',"recipe":{encode(recipe)}'.encode())
    bound = _door_bound_broken(composed_stat_vars) if composed_stat_vars else None
    if bound is not None:
        logger.info(
            "publishing %s/%s without composed_stat_vars: the list breaks the door's %s bound",
            spec_id,
            content_version,
            bound,
        )
    elif composed_stat_vars:
        entries = [
            {
                "stat_var": entry.stat_var,
                "relation": entry.relation,
                "first": list(entry.first),
                "second": list(entry.second),
            }
            for entry in composed_stat_vars
        ]
        body.write(f',"composed_stat_vars":{encode(entries)}'.encode())
    body.write(b"}")
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
    *,
    spec_id: str,
    content_version: str,
    max_frame_rows: int,
    recipe: Mapping[str, Any] | None = None,
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

    The body carries the frame's recipe when one can be built (`_recipe_or_none`).
    Building it reads the ancestors' traces and the roots' mint records from the
    store and decodes no rows, so it needs no claim of its own. A `recipe` the
    caller passes is sent in its place, and one that encodes to more than
    `_RECIPE_CEILING_BYTES` refuses locally before the store is read.
    """
    try:
        ref = FrameRef(spec_id=spec_id, content_version=content_version)
    except ValidationError as exc:
        raise PublishBodyNotBuildable(refuse_locally=False, message=str(exc)) from exc

    if recipe is not None and (encoded_bytes := _encoded_bytes(recipe)) > _RECIPE_CEILING_BYTES:
        raise PublishBodyNotBuildable(
            refuse_locally=True,
            message=(
                f"the recipe of {spec_id!r}/{content_version!r} encodes to {encoded_bytes} "
                f"bytes, over the {_RECIPE_CEILING_BYTES}-byte ceiling"
            ),
        )

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
        yield await _read_and_encode(held, claim=claim, ref=ref, recipe=recipe)


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


def _encoded_bytes(recipe: Mapping[str, Any]) -> int:
    """The bytes `recipe` takes in a publish body."""
    return len(_ENCODER.encode(recipe).encode())


async def _recipe_or_none(ref: FrameRef, trace: StoredFrameTrace) -> dict[str, Any] | None:
    """The recipe of the frame held under `ref`, whose stored trace is `trace`, or
    `None` with a warning naming the pair when none can be built or it encodes to
    more than `_RECIPE_CEILING_BYTES`."""
    try:
        recipe = await build_recipe(ref, store=get_store(), trace=trace)
    except RecipeError as exc:
        logger.warning(
            "publishing %s/%s without a recipe: %s", ref.spec_id, ref.content_version, exc
        )
        return None
    encoded_bytes = _encoded_bytes(recipe)
    if encoded_bytes > _RECIPE_CEILING_BYTES:
        logger.warning(
            "publishing %s/%s without a recipe: it encodes to %d bytes, over the %d-byte ceiling",
            ref.spec_id,
            ref.content_version,
            encoded_bytes,
            _RECIPE_CEILING_BYTES,
        )
        return None
    return recipe


async def _read_and_encode(
    held: HeldFrame, *, claim: RowClaim, ref: FrameRef, recipe: Mapping[str, Any] | None
) -> PublishBody:
    """`publish_body_for`'s read and encoding under `claim`. The decoded frame
    is dropped on return, so only the encoded body lives through the publish
    request. `recipe` is the caller's, and `None` has one built from the store."""
    spec_id, content_version = ref.spec_id, ref.content_version
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
        recipe=await _recipe_or_none(ref, trace) if recipe is None else recipe,
        composed_stat_vars=composed_stat_vars_of(trace),
    )
    return PublishBody(data=data, rows=frame.table.num_rows, hold_through=claim.hold_through)


__all__ = ["PublishBody", "PublishBodyNotBuildable", "publish_body", "publish_body_for"]

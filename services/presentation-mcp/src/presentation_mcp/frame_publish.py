"""Builds the wire body for the apis service's publish door
(`POST /api/datasets/frames/publish`) out of an analysis-mcp-minted frame
already sitting in the shared materialisation store, and reads that store
to get it.

`PlatformAssembler`'s own publish hook (`client.py`) calls `publish_body_for`
before a platform call that needs an `ops_`/`ocv_` ref published, and posts
its result verbatim. apis pins its own `dfo-spec-kit` revision, which can
differ from this repo's, so it re-renders and re-digests frame1 from
`role_keys`/`rows` on its own side (see `publish_models.PublishFrameRequest`'s own docstring on the
apis side for why). `publish_body` only reproduces `dfo_frame_grammar.Frame
.build`'s own two arguments from an already-built `Frame`, never a
grammar-rev-dependent rendering.

The body also carries the frame's `recipe` (`common.recipe.build_recipe`): the
tree of op-specs that computed it, its roots' mint records, and the two hashes
apis indexes it by. A frame whose recipe cannot be built is published
without the field and the reason is logged at warning with the pair: an
ancestor's trace has left the store, or a root was registered before data-mcp
wrote mint records and has not been minted since. This service has no recovery
routine for such a root (`data_mcp.mint_recovery` needs apis' dataset
versions), so its recipe is written later through apis' recipe route.

The apis service's publish body forbids unknown fields, so an apis deployment that
predates `recipe` refuses every body that carries one. apis must therefore
deploy first. There is no retry without the field: a publish that dropped the
recipe on a refusal would leave the frame without one for good.

A caller that has already built the recipe passes it to `publish_body_for`, which
then sends that recipe or refuses: a frame such a caller publishes is of no use
to it without one.

The body also carries the frame's `steps` (`common.steps.assemble_steps`): the
step record of each operation along its ancestry, with a `retrieve` step for each
root. A frame with an ancestor whose stored trace holds no step, has left the
store or fails a store check, or with a root that has no mint record, is
published without the field, logged at warning with the pair, and apis
records its steps as unknown. A step list over the door's byte bound sheds
its coverage members, largest first, until it fits (`_shed_coverage`). An
unusable stored step (`common.steps.StepRecordInvalidError`) or a step list
over the door's bounds without coverage to shed refuses locally, and a list
the door refuses fails the publish: the frame is never published without steps
it has.
"""

from __future__ import annotations

import io
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Mapping, Sequence
from contextlib import asynccontextmanager, contextmanager
from typing import Any, NamedTuple

import rfc8785
from common.materialisation import (
    FrameExpired,
    FrameIntegrityError,
    FrameRef,
    FrameTooLarge,
    HeldFrame,
    StoredFrameTrace,
    get_store,
)
from common.place_type import enrichment_role
from common.recipe import AncestryReads, RecipeError, build_recipe
from common.row_budget import RowClaim, row_budget
from common.steps import StepRecordInvalidError, assemble_steps
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

# apis' `MAX_RECIPE_BYTES`: the most bytes of compact JSON its publish door
# accepts as a recipe. A larger one is left out of the body (`_recipe_or_none`), so
# the door never refuses a frame for the size of its recipe.
_RECIPE_CEILING_BYTES = 1_000_000

# dfo-spec-kit's step bounds (`dfo_descriptions.MAX_STEPS` and `MAX_STEPS_BYTES`):
# the most steps a served list holds, and the most bytes a list takes as JCS
# (RFC 8785) UTF-8 JSON (`dfo_descriptions.steps_bytes`).
_MAX_SERVED_STEPS = 64
_MAX_STEPS_BYTES = 65_536

# apis' `description.steps.MAX_CARD_STEPS`: the steps a card drawn from a
# published frame serves after the frame's own, its selection and a stat card's
# change.
_MAX_CARD_STEPS = 2

# The apis service's publish door's step bound (`description.steps.MAX_COMPUTED_STEPS`), so
# that a card drawn from the frame serves at most `_MAX_SERVED_STEPS`.
_MAX_STEPS = _MAX_SERVED_STEPS - _MAX_CARD_STEPS

# The step members a list over `_MAX_STEPS_BYTES` sheds, in the order it sheds
# them: a coverage member restates per entity or per period what `kept` counts.
_COVERAGE_MEMBERS = ("entity_coverage", "period_coverage")

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
    process still cannot build a valid publish body from it: a stored trace
    naming no input content versions (a `Trace` predating that field), a
    corrupted/oversized stored object the store's own read-time checks
    caught, an unusable stored step among its ancestry, or a step list over
    the door's bounds. The apis service's publish door would refuse the identical body
    on every retry, so this refuses now instead of letting that repeat.
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


def publish_body(
    frame: Frame,
    *,
    spec_id: str,
    content_version: str,
    input_content_versions: Sequence[str],
    recipe: Mapping[str, Any] | None = None,
    steps: Sequence[Mapping[str, Any]] | None = None,
) -> bytes:
    """The publish door's JSON body for `frame`, encoded as httpx encodes a
    `json=` body. `recipe` is sent as the body's `recipe` member, and `None`
    leaves the member out.

    `role_keys` and `rows` are `dfo_frame_grammar.Frame.build`'s own two
    arguments, the inverse of what `Frame.build` does to construct `frame`
    (see `_wire_row_slices` for the row mappings). Rows are encoded one
    slice at a time, so the whole row list never exists as Python objects.

    A place-type or party-kind column (`common.place_type.enrichment_role`)
    beside its place column is left out. apis labels each series and table
    row by joining every role's value, so the column would label a continent
    "Africa / Continent". It is a function of its place column, so the
    published rows stay distinct.

    `steps` is the body's last member, and `None` leaves it out.

    A step list over `_MAX_STEPS_BYTES` (`_steps_bytes`) is sent with
    coverage members shed until it fits (`_shed_coverage`).

    Raises:
        PublishBodyNotBuildable: `steps` holds more than `_MAX_STEPS` steps,
            takes more than `_MAX_STEPS_BYTES` with no coverage left, or holds
            a value JCS cannot serialise, so the door would refuse the body.
            `refuse_locally` is set.
    """
    encode = _ENCODER.encode
    encoded_steps = None if steps is None else _checked_steps(steps, spec_id, content_version)
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
    if encoded_steps is not None:
        body.write(f',"steps":{encoded_steps}'.encode())
    body.write(b"}")
    return body.getvalue()


def _checked_steps(steps: Sequence[Mapping[str, Any]], spec_id: str, content_version: str) -> str:
    """`steps` encoded for the body, less the coverage members it sheds to fit
    (`_shed_coverage`), or `PublishBodyNotBuildable` when the list breaks one
    of the door's bounds or holds a value JCS cannot serialise, which the kit
    refuses as `not_i_json`."""
    if len(steps) > _MAX_STEPS:
        broken = f"holds {len(steps)} steps, over the {_MAX_STEPS}-step bound"
    else:
        try:
            size = _steps_bytes(steps)
        except rfc8785.CanonicalizationError as exc:
            broken = f"holds a value JCS cannot serialise: {exc}"
        else:
            fitted: Sequence[Mapping[str, Any]] = steps
            if size > _MAX_STEPS_BYTES:
                fitted, shed = _shed_coverage(steps, size=size)
                size = _steps_bytes(fitted)
                if shed and size <= _MAX_STEPS_BYTES:
                    logger.info(
                        "publishing %s/%s with %d coverage member(s) shed to fit the "
                        "%d-byte step bound",
                        spec_id,
                        content_version,
                        shed,
                        _MAX_STEPS_BYTES,
                    )
            if size <= _MAX_STEPS_BYTES:
                return _ENCODER.encode(list(fitted))
            broken = f"takes {size} bytes as JCS, over the {_MAX_STEPS_BYTES}-byte bound"
    raise PublishBodyNotBuildable(
        refuse_locally=True,
        message=f"the step list of {spec_id!r}/{content_version!r} {broken}",
    )


def _steps_bytes(steps: Sequence[Mapping[str, Any]]) -> int:
    """The bytes `steps` take as JCS (RFC 8785) UTF-8 JSON, the measure the
    kit's byte bound reads (`dfo_descriptions.steps_bytes`). JCS spells a number
    as ECMAScript does, `1` for `1.0` and `10000000000000000` for `1e16`, so
    Python's JSON spelling miscounts a list holding one."""
    return len(rfc8785.dumps(list(steps)))


def _shed_coverage(
    steps: Sequence[Mapping[str, Any]], *, size: int
) -> tuple[list[dict[str, Any]], int]:
    """`steps`, which take `size` bytes (`_steps_bytes`), less the coverage
    members it sheds to fit `_MAX_STEPS_BYTES`, with how many it shed.

    The list sheds each step's `entity_coverage`, then each step's
    `period_coverage`, the largest of a kind first, until it fits. No other
    member is ever shed, so a list still over the bound with no coverage left
    stays over it."""
    fitted = [dict(step) for step in steps]
    shed = 0
    for member in _COVERAGE_MEMBERS:
        # A member's bytes in its step: `"name":value` and the comma before it,
        # since no step holds it alone.
        sizes = {
            index: len(rfc8785.dumps({member: step[member]})) - 1
            for index, step in enumerate(fitted)
            if member in step
        }
        for index in sorted(sizes, key=sizes.__getitem__, reverse=True):
            if size <= _MAX_STEPS_BYTES:
                return fitted, shed
            del fitted[index][member]
            size -= sizes[index]
            shed += 1
    return fitted, shed


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

    The body carries the frame's steps when they can be assembled
    (`_steps_or_none`), read from the store as the recipe is. An unusable stored
    step, or a step list over the door's bounds, refuses locally.
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


async def _recipe_or_none(
    ref: FrameRef, trace: StoredFrameTrace, *, reads: AncestryReads
) -> dict[str, Any] | None:
    """The recipe of the frame held under `ref`, whose stored trace is `trace`, read
    through `reads`, or `None` with a warning naming the pair when none can be built
    or it encodes to more than `_RECIPE_CEILING_BYTES`."""
    try:
        recipe = await build_recipe(ref, store=get_store(), trace=trace, reads=reads)
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


async def _steps_or_none(
    ref: FrameRef, trace: StoredFrameTrace, *, reads: AncestryReads
) -> list[dict[str, Any]] | None:
    """The step list of the frame held under `ref`, whose stored trace is `trace`,
    read through `reads`, or `None` with a warning naming the pair when an ancestor stores no step, an
    ancestor's trace has left the store or fails a store check, or a root has no
    mint record.

    Raises:
        PublishBodyNotBuildable: A stored step on the walk is unusable
            (`StepRecordInvalidError`). `refuse_locally` is set.
    """
    try:
        steps = await assemble_steps(ref, store=get_store(), trace=trace, reads=reads)
    except StepRecordInvalidError as exc:
        raise PublishBodyNotBuildable(refuse_locally=True, message=str(exc)) from exc
    except RecipeError as exc:
        logger.warning("publishing %s/%s without steps: %s", ref.spec_id, ref.content_version, exc)
        return None
    if steps is None:
        logger.warning(
            "publishing %s/%s without steps: a frame it derives from stores no step",
            ref.spec_id,
            ref.content_version,
        )
    return steps


async def _read_and_encode(
    held: HeldFrame, *, claim: RowClaim, ref: FrameRef, recipe: Mapping[str, Any] | None
) -> PublishBody:
    """`publish_body_for`'s read and encoding under `claim`. The decoded frame
    is dropped on return, so only the encoded body lives through the publish
    request. `recipe` is the caller's, and `None` has one built from the store.
    The recipe and the steps share one `AncestryReads`, so each ancestor's trace
    and each root's mint record is read once."""
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
    reads = AncestryReads(store=get_store())
    data = publish_body(
        frame,
        spec_id=spec_id,
        content_version=content_version,
        input_content_versions=input_content_versions,
        recipe=await _recipe_or_none(ref, trace, reads=reads) if recipe is None else recipe,
        steps=await _steps_or_none(ref, trace, reads=reads),
    )
    return PublishBody(data=data, rows=frame.table.num_rows, hold_through=claim.hold_through)


__all__ = ["PublishBody", "PublishBodyNotBuildable", "publish_body", "publish_body_for"]

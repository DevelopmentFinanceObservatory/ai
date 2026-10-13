"""Reads the `series-statistics/1` Parquet artifact a `get_data` mint carries for its
per-series figures.

`find_statistics_artifact` picks the one matching entry off a manifest's
`artifacts` list, by its `rendering` shape alone -- the same way
`frame_materialisation._select_parquet_artifact` already picks the frame
artifact off the same list. `fetch_series_statistics` then reads it: a
link lookup through `data_mcp.artifact_resource.read_artifact` (the same hop
`frame_materialisation` uses), a streamed GET on that link that follows its
redirect to the object store and refuses once the downloaded body exceeds
`artifact.bytes` (the advertised size) rather than buffering an oversized body in full, a check of the fetched
bytes against `artifact.artifact_digest` (the same check
`frame_materialisation._verify_parse_and_register` runs before it trusts a
frame's bytes), and a read that checks the file's own `dfo.schema` metadata
and its series against the caller's own `expected_series` (the manifest's
`series_count`), since a schema match alone does not prove the artifact
carries the series the manifest promised. A manifest that carries the
artifact but no `series_count` has no count to check, so
`expected_series=None` skips that comparison alone.

A file holds a row per series, so its Parquet-metadata row count is checked
before a single column is read, unless the mint's estimates continue its
series (`continuing_method`, off the manifest's `trace.continuations`). A
continued series holds a row for its reported method and a row for the
continuing one and counts once, as dfo-apis' producer counts it
(`materialise.statistics.SeriesStatisticsFile`), so such a file's series are
counted from its rows (`_series_count`).

`fetch_series_statistics_table` returns the verified file as an Arrow table
in `SERIES_STATISTICS_ARROW_SCHEMA`, the producer's own schema
(`materialise.statistics.SERIES_STATISTICS_SCHEMA` in dfo-apis'
`dfo-data-tools`): every column cast to its type, every non-nullable field
free of nulls, and every `facet` a JSON object of the shape
`SeriesStatisticsModel.facet` accepts. `fetch_series_statistics` converts
that table into one `SeriesStatisticsModel` per row. A row costs about
7.4 KiB as models and about 0.3 KiB as Arrow at 25,000 CRS-shaped series, so
a caller that reads every series (`harness.compiler.evidence`, the harness
process) takes the table, and `get_data_digest.py`, which renders at most
`_STATISTICS_SERIES_CAP` series in full, takes the models. Both processes
have data-mcp's `client_holder` live, since the deployed topology mounts
data-mcp in-process in both today, per `client_holder.py`'s own docstring,
so neither parses the Parquet bytes on its own.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import logging
from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

import httpx2
import pyarrow as pa
import pyarrow.parquet as pq
from dfo_frame_grammar.columnar import array_kernel
from pydantic import BaseModel, TypeAdapter, ValidationError

from data_mcp.deadline import CallDeadline
from data_mcp.manifest import (
    ArtifactEntry,
    Manifest,
    SeriesChangeModel,
    SeriesEntityModel,
    SeriesStatisticsModel,
    StatPointModel,
)

__all__ = [
    "SERIES_STATISTICS_ARROW_SCHEMA",
    "SERIES_STATISTICS_SCHEMA",
    "ContinuedSeries",
    "SeriesStatisticsSchemaError",
    "continued_series",
    "continuing_method",
    "fetch_series_statistics",
    "fetch_series_statistics_table",
    "find_statistics_artifact",
    "series_index",
    "series_statistics_table",
]

logger = logging.getLogger(__name__)

# The Parquet file-level metadata key naming the schema this module decodes,
# and the one value it accepts -- see this repo's wire contract for
# `series-statistics/1`'s full column shape.
_SCHEMA_METADATA_KEY = b"dfo.schema"
SERIES_STATISTICS_SCHEMA = "series-statistics/1"

_ENTITY_TYPE = pa.struct(
    [pa.field("role", pa.string(), nullable=False), pa.field("ref", pa.string(), nullable=False)]
)

# The columns, types and nullability `series-statistics/1` carries, as dfo-apis'
# producer writes them. Nullable exactly where `SeriesStatisticsModel` accepts
# `None`: a series with one observation has no change.
SERIES_STATISTICS_ARROW_SCHEMA = pa.schema(
    [
        pa.field("stat_var", pa.string(), nullable=False),
        pa.field(
            "entities", pa.list_(pa.field("item", _ENTITY_TYPE, nullable=False)), nullable=False
        ),
        pa.field("facet", pa.string(), nullable=False),
        pa.field("observations", pa.int64(), nullable=False),
        pa.field("first_value", pa.float64(), nullable=False),
        pa.field("latest_value", pa.float64(), nullable=False),
        pa.field("minimum_value", pa.float64(), nullable=False),
        pa.field("maximum_value", pa.float64(), nullable=False),
        pa.field("first_period", pa.string(), nullable=False),
        pa.field("latest_period", pa.string(), nullable=False),
        pa.field("minimum_period", pa.string(), nullable=False),
        pa.field("maximum_period", pa.string(), nullable=False),
        pa.field("change_absolute", pa.float64(), nullable=True),
        pa.field("change_percent", pa.float64(), nullable=True),
    ]
)

# `SeriesStatisticsModel.facet`'s own type, checked once per distinct facet.
_FACET = TypeAdapter(SeriesStatisticsModel.model_fields["facet"].annotation)

# This hop's own timeout for the artifact GET. Unlike
# `frame_materialisation`'s fetch, this one carries no retry loop of its
# own: a caller here (the digest, or the harness's evidence check) already
# degrades cleanly on any failure, so a single attempt is enough -- see each
# caller's own docstring for what "degrade" means there.
_FETCH_TIMEOUT_S = 30.0


class SeriesStatisticsSchemaError(ValueError):
    """The fetched artifact cannot be trusted or decoded as `series-statistics/1`.

    Covers a downloaded body that exceeds `artifact.bytes` -- by its
    advertised `Content-Length` or by what was actually streamed -- a
    `dfo.schema` metadata mismatch, a computed digest that does not match
    `artifact.artifact_digest`, a row whose `facet`, `entities`, or
    `observations` column does not decode into `SeriesStatisticsModel`'s
    shape, and a series count that does not match the caller's own
    `expected_series` -- every way this module's own reading can fail,
    folded into the one type its callers catch.
    """


class _Continuation(BaseModel):
    """One `trace.continuations` entry: the method whose estimates continue the
    mint's series past their end. Its `axis` and `periods` go unread."""

    method: str


_CONTINUATIONS = TypeAdapter(list[_Continuation])


def continuing_method(manifest: Manifest) -> str | None:
    """The method `manifest.trace.continuations` names, or `None` when it names
    none. A mint's one `measurementMethod` selection has at most one
    continuation, so the first entry is the one.

    Raises:
        pydantic.ValidationError: `continuations` is not a list of entries
            each naming a `method`.
    """
    continuations = _CONTINUATIONS.validate_python(manifest.trace.get("continuations", []))
    return next((entry.method for entry in continuations), None)


def find_statistics_artifact(manifest: Manifest) -> ArtifactEntry | None:
    """The one statistics artifact on a definition manifest, or `None`.

    Matches `rendering == {"statistics": "series"}` -- the shape the wire
    contract fixes for a definition's own statistics entry. `None` when no
    such entry exists, which a caller should expect only when
    `manifest.series_count` is itself `None` (an old record) -- the
    contract guarantees the entry is present, with zero rows if need be,
    whenever `series_count` is not `None`.
    """
    return next((a for a in manifest.artifacts if a.rendering == {"statistics": "series"}), None)


async def fetch_series_statistics(
    *,
    subject_id: str,
    content_version: str,
    artifact: ArtifactEntry,
    expected_series: int | None,
    continuing: str | None,
    transport: httpx2.AsyncBaseTransport | None = None,
    deadline: CallDeadline | None = None,
) -> list[SeriesStatisticsModel]:
    """`fetch_series_statistics_table`'s rows, one `SeriesStatisticsModel` each,
    in the file's own row order -- empty when `artifact` carries zero rows.

    Takes and raises what `fetch_series_statistics_table` does.
    """
    table = await fetch_series_statistics_table(
        subject_id=subject_id,
        content_version=content_version,
        artifact=artifact,
        expected_series=expected_series,
        continuing=continuing,
        transport=transport,
        deadline=deadline,
    )
    try:
        return [_row_to_model(row) for row in table.to_pylist()]
    except (ValueError, TypeError, KeyError, ValidationError) as exc:
        raise SeriesStatisticsSchemaError(
            f"a series-statistics/1 row failed to decode: {exc}"
        ) from exc


async def fetch_series_statistics_table(
    *,
    subject_id: str,
    content_version: str,
    artifact: ArtifactEntry,
    expected_series: int | None,
    continuing: str | None,
    transport: httpx2.AsyncBaseTransport | None = None,
    deadline: CallDeadline | None = None,
) -> pa.Table:
    """Fetch and read `artifact`'s `series-statistics/1` Parquet bytes.

    Args:
        subject_id: The `spec_` id `artifact` was minted under.
        content_version: The `cv_` id naming the exact retrieval.
        artifact: The statistics entry itself, from `find_statistics_artifact`.
        expected_series: The series count the manifest that named `artifact`
            already promised (`manifest.series_count`) -- checked against the
            file's series once the fetch succeeds. `None` when the manifest
            carries the artifact but no `series_count` (dfo-apis mints an
            all-years definition that way): there is no count to check.
        continuing: The method continuing the mint's series
            (`continuing_method(manifest)`), whose rows count no series of
            their own beside a row of the series they continue; `None` when
            nothing continues them.
        transport: Only ever set by a test, exactly like
            `frame_materialisation.materialise_and_register`'s own `transport`
            -- `None` (the default) means a real network GET.
        deadline: When set, the link lookup and the GET together must finish
            before it; `None` bounds each by its own timeout alone.

    Returns:
        The file's rows in `SERIES_STATISTICS_ARROW_SCHEMA`, in its own row
        order -- empty when `artifact` carries zero rows.

    Raises:
        fastmcp.exceptions.ResourceError: `read_artifact` could not read the
            artifact's link off the snapshot route.
        SeriesStatisticsSchemaError: the downloaded body exceeds `artifact
            .bytes` (the advertised size), the fetched bytes do not hash to
            `artifact.artifact_digest`, the fetched file's `dfo.schema`
            metadata is not `series-statistics/1`, its series count does not
            match `expected_series`, or a column does not fit
            `SERIES_STATISTICS_ARROW_SCHEMA` or a facet `SeriesStatisticsModel`.
        httpx2.HTTPStatusError: the link answered with an error status, such
            as 410 `artifact_expired`, or its redirect target did.
        httpx2.TransportError: the GET itself failed.
        TimeoutError: `deadline` passed before the link lookup and the GET
            finished.
    """
    # Imported here, not at module level: `data_mcp.artifact_resource` imports
    # `data_mcp.get_data_tool`, which imports `frame_materialisation`, which
    # would make a module-level import of `artifact_resource` here circular
    # the moment `get_data_tool` (or anything it imports) reaches for this
    # module -- the same deferred-import reasoning
    # `frame_materialisation._fetch_artifact_bytes_uncapped` already gives
    # for its own `read_artifact` import.
    from data_mcp.artifact_resource import read_artifact  # noqa: PLC0415

    async with asyncio.timeout(deadline.remaining() if deadline is not None else None):
        link = await read_artifact(subject_id, content_version, artifact.artifact_digest)
        async with httpx2.AsyncClient(
            transport=transport,
            timeout=httpx2.Timeout(_FETCH_TIMEOUT_S),
            follow_redirects=True,
        ) as client:
            fetched = await _get_bounded(client, link["url"], max_bytes=artifact.bytes)
    computed_digest = f"sha256:{hashlib.sha256(fetched).hexdigest()}"
    if computed_digest != artifact.artifact_digest:
        raise SeriesStatisticsSchemaError(
            "fetched artifact bytes do not hash to the advertised artifact_digest "
            f"{artifact.artifact_digest!r}; computed {computed_digest!r}"
        )
    return _read(fetched, expected_series=expected_series, continuing=continuing)


async def _get_bounded(client: httpx2.AsyncClient, url: str, *, max_bytes: int) -> bytes:
    """Stream `url`'s body, refusing before it is buffered in full once it exceeds `max_bytes`.

    `artifact.bytes` is dfo-apis' own advertised size for the artifact this link
    names; a body that outgrows it is refused chunk by chunk, so an
    artifact whose advertised size understates what dfo-apis actually stored
    never gets fully read into memory here. `Accept-Encoding: identity` keeps
    the response uncompressed, so a chunk `aiter_bytes` yields is never larger
    than the bytes read off the wire for it -- with compression allowed, a
    small compressed read could decode into a chunk that alone exceeds
    `max_bytes`, checked only after it was already appended. A `Content-Length`
    over `max_bytes` is refused before any of the body is read at all.
    """
    async with client.stream("GET", url, headers={"Accept-Encoding": "identity"}) as response:
        response.raise_for_status()
        content_length = response.headers.get("Content-Length")
        if content_length is not None and int(content_length) > max_bytes:
            raise SeriesStatisticsSchemaError(
                f"the response declared Content-Length={content_length}, over the "
                f"artifact's advertised size of {max_bytes} byte(s)"
            )
        fetched = bytearray()
        async for chunk in response.aiter_bytes():
            if len(fetched) + len(chunk) > max_bytes:
                raise SeriesStatisticsSchemaError(
                    f"downloaded body exceeded the artifact's advertised size of "
                    f"{max_bytes} byte(s)"
                )
            fetched += chunk
        return bytes(fetched)


def _read(data: bytes, *, expected_series: int | None, continuing: str | None) -> pa.Table:
    """Read `data` as a `series-statistics/1` Parquet file into a checked table.

    Checked against the file's own `dfo.schema` metadata, then, with nothing
    `continuing` the series, its Parquet-metadata row count against
    `expected_series`, before any column is read -- a caller trusting the
    wrong schema version or row count would otherwise misread every column
    silently, or pay to read rows a mismatched count already disqualifies,
    instead of refusing. `ParquetFile.metadata.num_rows` and `.schema_arrow`'s
    own key-value metadata both read off the file's footer, so both checks run
    without decompressing a single column. With a `continuing` method a
    series may hold two rows, so its series are counted from the rows read
    (`_series_count`) and checked against `expected_series` after them.

    The columns are then cast to `SERIES_STATISTICS_ARROW_SCHEMA`, which
    refuses a missing column, a value its type cannot hold (a fractional
    `observations`, say) and a null in a non-nullable top-level or struct
    field; a null `entities` element reads back from Parquet with null
    struct fields, so it is refused too. A `facet` that is not a JSON object
    of `SeriesStatisticsModel.facet`'s shape is checked here as well, once
    per distinct string. Every one of these is a
    `SeriesStatisticsSchemaError`, the one error type this module's callers
    catch, so a row that would fail to become a `SeriesStatisticsModel` fails
    the table the same way.
    """
    parquet_file = pq.ParquetFile(io.BytesIO(data))
    metadata = parquet_file.schema_arrow.metadata or {}
    raw_schema = metadata.get(_SCHEMA_METADATA_KEY)
    schema_name = raw_schema.decode("utf-8") if raw_schema is not None else None
    if schema_name != SERIES_STATISTICS_SCHEMA:
        raise SeriesStatisticsSchemaError(
            f"expected dfo.schema={SERIES_STATISTICS_SCHEMA!r} metadata, got {schema_name!r}"
        )
    num_rows = parquet_file.metadata.num_rows
    if expected_series is not None and continuing is None and num_rows != expected_series:
        raise SeriesStatisticsSchemaError(
            f"the series-statistics/1 artifact carries {num_rows} row(s), but "
            f"the manifest named {expected_series}"
        )
    try:
        # `pa.ArrowInvalid` and `json.JSONDecodeError` are both `ValueError`s.
        table = parquet_file.read(columns=SERIES_STATISTICS_ARROW_SCHEMA.names).cast(
            SERIES_STATISTICS_ARROW_SCHEMA
        )
        for facet in array_kernel("unique", table.column("facet").combine_chunks()).to_pylist():
            _FACET.validate_python(json.loads(facet))
    except (ValueError, TypeError, ValidationError) as exc:
        raise SeriesStatisticsSchemaError(
            f"the series-statistics/1 artifact does not fit its schema: {exc}"
        ) from exc
    if (
        expected_series is not None
        and continuing is not None
        and (series := _series_count(table, continuing=continuing)) != expected_series
    ):
        raise SeriesStatisticsSchemaError(
            f"the series-statistics/1 artifact carries {num_rows} row(s) holding "
            f"{series} series, but the manifest named {expected_series}"
        )
    return table


def _series_count(table: pa.Table, *, continuing: str) -> int:
    """The series `table`'s rows hold, counted as dfo-apis' producer counts them
    (`materialise.statistics._FiguresFold`, by `series_index`'s rule)."""
    return len(set(series_index(table, continuing=continuing).to_pylist()))


def series_index(table: pa.Table, *, continuing: str | None) -> pa.Array:
    """The series each row of `table` belongs to, as the position of its series'
    head row: a row of the `continuing` method belongs to the first row of
    another method holding the same variable, entities and facet apart from
    `measurementMethod`, as dfo-apis' producer folds them
    (`materialise.statistics._FiguresFold`). Every other row heads its own
    series, so with nothing `continuing` each row is its own.
    """
    keys = [
        _series_identity(stat_var, entities, json.loads(text))
        for stat_var, entities, text in zip(
            table.column("stat_var").to_pylist(),
            table.column("entities").to_pylist(),
            table.column("facet").to_pylist(),
            strict=True,
        )
    ]
    return pa.array(_series_heads(keys, continuing=continuing), pa.int64())


class ContinuedSeries(NamedTuple):
    """One series of a statistics file: its `reported` row, and the row of the
    method continuing it past its end, or `None` when nothing continues it."""

    reported: SeriesStatisticsModel
    estimates: SeriesStatisticsModel | None


def continued_series(
    rows: Sequence[SeriesStatisticsModel], *, continuing: str | None
) -> list[ContinuedSeries]:
    """`rows` as series, by `series_index`'s rule, in the order of their head rows."""
    heads = _series_heads(
        [
            _series_identity(
                row.stat_var,
                [{"role": entity.role, "ref": entity.ref} for entity in row.entities],
                row.facet,
            )
            for row in rows
        ],
        continuing=continuing,
    )
    estimates = {head: rows[index] for index, head in enumerate(heads) if head != index}
    return [
        ContinuedSeries(reported=row, estimates=estimates.get(index))
        for index, row in enumerate(rows)
        if heads[index] == index
    ]


def _series_heads(keys: Sequence[tuple[str, str | None]], *, continuing: str | None) -> list[int]:
    """`series_index` over each row's `_series_identity`."""
    leads: dict[str, int] = {}
    for index, (identity, method) in enumerate(keys):
        if continuing is None or method != continuing:
            leads.setdefault(identity, index)
    return [
        leads.get(identity, index) if continuing is not None and method == continuing else index
        for index, (identity, method) in enumerate(keys)
    ]


def _series_identity(
    stat_var: str, entities: Sequence[Mapping[str, str]], facet: Mapping[str, Any]
) -> tuple[str, str | None]:
    """A statistics row's series apart from its method, as a comparable string, and
    its `measurementMethod`."""
    rest = {key: value for key, value in facet.items() if key != "measurementMethod"}
    identity = json.dumps([stat_var, list(entities), rest], sort_keys=True)
    method = facet.get("measurementMethod")
    return identity, method if isinstance(method, str) else None


def series_statistics_table(series: Sequence[SeriesStatisticsModel]) -> pa.Table:
    """`series` as a `SERIES_STATISTICS_ARROW_SCHEMA` table, one row each: what
    `fetch_series_statistics_table` would read from an artifact holding them,
    for a caller that has the series and no artifact (a scripted run, a test).
    """
    return pa.Table.from_pylist(
        [
            {
                "stat_var": one.stat_var,
                "entities": [{"role": entity.role, "ref": entity.ref} for entity in one.entities],
                "facet": json.dumps(one.facet, sort_keys=True),
                "observations": one.observations,
                "first_value": one.first.value,
                "latest_value": one.latest.value,
                "minimum_value": one.minimum.value,
                "maximum_value": one.maximum.value,
                "first_period": one.first.period,
                "latest_period": one.latest.period,
                "minimum_period": one.minimum.period,
                "maximum_period": one.maximum.period,
                "change_absolute": one.change.absolute if one.change is not None else None,
                "change_percent": one.change.percent if one.change is not None else None,
            }
            for one in series
        ],
        schema=SERIES_STATISTICS_ARROW_SCHEMA,
    )


def _row_to_model(row: dict[str, Any]) -> SeriesStatisticsModel:
    change = None
    if row["change_absolute"] is not None:
        change = SeriesChangeModel(absolute=row["change_absolute"], percent=row["change_percent"])
    return SeriesStatisticsModel(
        stat_var=row["stat_var"],
        entities=[SeriesEntityModel(role=e["role"], ref=e["ref"]) for e in row["entities"]],
        facet=json.loads(row["facet"]),
        observations=row["observations"],
        first=StatPointModel(value=row["first_value"], period=row["first_period"]),
        latest=StatPointModel(value=row["latest_value"], period=row["latest_period"]),
        minimum=StatPointModel(value=row["minimum_value"], period=row["minimum_period"]),
        maximum=StatPointModel(value=row["maximum_value"], period=row["maximum_period"]),
        change=change,
    )

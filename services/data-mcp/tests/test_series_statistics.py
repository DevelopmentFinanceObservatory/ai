"""Tests for `data_mcp.series_statistics`: finding and reading the
`series-statistics/1` artifact.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx2
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from data_mcp import client_holder
from data_mcp.deadline import CallDeadline
from data_mcp.manifest import Manifest, strip_manifest
from data_mcp.series_statistics import (
    SERIES_STATISTICS_ARROW_SCHEMA,
    SERIES_STATISTICS_SCHEMA,
    SeriesStatisticsSchemaError,
    _get_bounded,
    fetch_series_statistics,
    fetch_series_statistics_table,
    find_statistics_artifact,
    series_statistics_table,
)

from .net_guard import no_network  # noqa: F401 -- imported for its autouse fixture
from .test_client import _client

_SPEC_ID = "spec_" + "1" * 32
_MEMBER_SPEC_ID = "spec_" + "4" * 32
_CONTENT_VERSION = "cv_" + "2" * 32
_ARTIFACT_DIGEST = "sha256:" + "3" * 64
_LINK = f"/api/datasets/{_SPEC_ID}/{_CONTENT_VERSION}/files/{_ARTIFACT_DIGEST}"

_ENTITIES_TYPE = pa.list_(pa.struct([("role", pa.string()), ("ref", pa.string())]))


def _row(
    *,
    stat_var: str = "dcid:MalariaGrantDisbursements",
    entities: list[dict[str, str]] | None = None,
    facet: dict[str, Any] | None = None,
    observations: int = 2,
    first: tuple[float, str] = (12_400_000.0, "2015"),
    latest: tuple[float, str] = (9_800_000.0, "2024"),
    minimum: tuple[float, str] = (7_100_000.0, "2020"),
    maximum: tuple[float, str] = (14_200_000.0, "2017"),
    change: tuple[float, float | None] | None = (-2_600_000.0, -21.0),
) -> dict[str, Any]:
    """One `series-statistics/1` row, as a plain dict -- fed to `_parquet_bytes`."""
    change_absolute, change_percent = change if change is not None else (None, None)
    return {
        "stat_var": stat_var,
        "entities": entities
        if entities is not None
        else [{"role": "recipient", "ref": "country/TGO"}],
        "facet": json.dumps(
            facet if facet is not None else {"unit": "dcid:USDollar"}, sort_keys=True
        ),
        "observations": observations,
        "first_value": first[0],
        "first_period": first[1],
        "latest_value": latest[0],
        "latest_period": latest[1],
        "minimum_value": minimum[0],
        "minimum_period": minimum[1],
        "maximum_value": maximum[0],
        "maximum_period": maximum[1],
        "change_absolute": change_absolute,
        "change_percent": change_percent,
    }


def _parquet_bytes(
    rows: list[dict[str, Any]], *, schema_name: str | None = SERIES_STATISTICS_SCHEMA
) -> bytes:
    """A `series-statistics/1`-shaped Parquet file from `rows` (`_row`'s own shape).

    `schema_name=None` omits the `dfo.schema` metadata key entirely, standing
    in for a file some other producer wrote; a string sets it to whatever a
    test wants to check `_decode`'s own refusal against.
    """
    table = pa.table(
        {
            "stat_var": pa.array([r["stat_var"] for r in rows], type=pa.string()),
            "entities": pa.array([r["entities"] for r in rows], type=_ENTITIES_TYPE),
            "facet": pa.array([r["facet"] for r in rows], type=pa.string()),
            "observations": pa.array([r["observations"] for r in rows], type=pa.int64()),
            "first_value": pa.array([r["first_value"] for r in rows], type=pa.float64()),
            "latest_value": pa.array([r["latest_value"] for r in rows], type=pa.float64()),
            "minimum_value": pa.array([r["minimum_value"] for r in rows], type=pa.float64()),
            "maximum_value": pa.array([r["maximum_value"] for r in rows], type=pa.float64()),
            "first_period": pa.array([r["first_period"] for r in rows], type=pa.string()),
            "latest_period": pa.array([r["latest_period"] for r in rows], type=pa.string()),
            "minimum_period": pa.array([r["minimum_period"] for r in rows], type=pa.string()),
            "maximum_period": pa.array([r["maximum_period"] for r in rows], type=pa.string()),
            "change_absolute": pa.array([r["change_absolute"] for r in rows], type=pa.float64()),
            "change_percent": pa.array([r["change_percent"] for r in rows], type=pa.float64()),
        }
    )
    if schema_name is not None:
        table = table.replace_schema_metadata({"dfo.schema": schema_name})
    buf = io.BytesIO()
    pq.write_table(table, buf)
    return buf.getvalue()


class _ChunkedStream(httpx2.AsyncByteStream):
    """Yields exactly the given chunks, in order -- `MockTransport` otherwise hands the
    whole body back as one chunk, which can't exercise a check that runs per chunk.
    """

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk


class _ExplodingStream(httpx2.AsyncByteStream):
    """A body `_get_bounded` must never read once `Content-Length` alone refuses it."""

    async def __aiter__(self) -> AsyncIterator[bytes]:
        raise AssertionError("the body must never be read once Content-Length already refuses it")
        yield b""  # pragma: no cover -- unreachable; keeps this an async generator


def _digest_for(data: bytes) -> str:
    """The `sha256:`-prefixed digest `fetch_series_statistics` computes over `data`.

    Tests that expect a fetch to reach past the digest check pass this
    (rather than the fixed `_ARTIFACT_DIGEST`) as both the manifest's and
    the snapshot payload's `artifact_digest`, matching what a real mint
    would advertise for these exact bytes.
    """
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def _manifest(
    *,
    rendering: dict[str, Any],
    artifact_digest: str = _ARTIFACT_DIGEST,
    spec_id: str = _SPEC_ID,
    content_version: str = _CONTENT_VERSION,
    # Comfortably over any Parquet fixture this module's own tests write
    # (fixed-overhead-dominated even at zero rows), so a test that doesn't
    # care about the size bound never trips it.
    bytes_advertised: int = 10_000,
) -> Manifest:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "definition": {"schema_version": 1, "spec_id": spec_id, "identity": {}},
        "content_version": content_version,
        "coverage": {"rows": 1, "dimensions": []},
        "citation": {"text": "OECD DAC2a", "sources": []},
        "trace": {},
        "warnings": [],
        "series_count": 1,
        "artifacts": [
            {
                "artifact_digest": artifact_digest,
                "rendering": rendering,
                "bytes": bytes_advertised,
                "media_type": "application/vnd.apache.parquet",
                "created_at": "2026-08-11T00:00:00+00:00",
                "tier": "working",
            }
        ],
    }
    return strip_manifest(payload)


def _snapshot_payload(
    *,
    subject_id: str = _SPEC_ID,
    content_version: str = _CONTENT_VERSION,
    artifact_digest: str = _ARTIFACT_DIGEST,
    url: str = _LINK,
    rendering: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The snapshot route's own response shape `read_artifact` re-validates and reads the link from."""
    return {
        "schema_version": 1,
        "definition": {"schema_version": 1, "spec_id": subject_id, "identity": {}},
        "content_version": content_version,
        "coverage": {"rows": 1, "dimensions": []},
        "citation": {"text": "OECD DAC2a", "sources": []},
        "trace": {},
        "warnings": [],
        "series_count": 1,
        "artifacts": [
            {
                "artifact_digest": artifact_digest,
                "rendering": rendering if rendering is not None else {"statistics": "series"},
                "bytes": 100,
                "media_type": "application/vnd.apache.parquet",
                "created_at": "2026-08-11T00:00:00+00:00",
                "tier": "working",
                "url": url,
            }
        ],
    }


# find_statistics_artifact


def test_find_statistics_artifact_matches_the_definition_rendering() -> None:
    manifest = _manifest(rendering={"statistics": "series"})

    artifact = find_statistics_artifact(manifest)

    assert artifact is not None
    assert artifact.artifact_digest == _ARTIFACT_DIGEST


def test_find_statistics_artifact_ignores_a_member_scoped_rendering() -> None:
    manifest = _manifest(rendering={"statistics": "series", "member": _MEMBER_SPEC_ID})

    assert find_statistics_artifact(manifest) is None


def test_find_statistics_artifact_none_when_absent() -> None:
    manifest = _manifest(rendering={"bundle": False, "format": "parquet", "layout": "frame"})

    assert find_statistics_artifact(manifest) is None


# fetch_series_statistics -- success


async def test_fetch_series_statistics_decodes_every_row() -> None:
    rows = [
        _row(stat_var="dcid:VarA", entities=[{"role": "recipient", "ref": "country/AAA"}]),
        _row(
            stat_var="dcid:VarB",
            entities=[{"role": "recipient", "ref": "country/BBB"}],
            change=None,
        ),
    ]
    data = _parquet_bytes(rows)
    digest = _digest_for(data)
    manifest = _manifest(rendering={"statistics": "series"}, artifact_digest=digest)
    artifact = manifest.artifacts[0]

    def snapshot_handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=_snapshot_payload(artifact_digest=digest))

    def get_handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, content=data)

    async with _client(snapshot_handler) as upstream:
        client_holder.set_client(upstream)
        result = await fetch_series_statistics(
            subject_id=_SPEC_ID,
            content_version=_CONTENT_VERSION,
            artifact=artifact,
            expected_rows=2,
            transport=httpx2.MockTransport(get_handler),
        )

    assert [series.stat_var for series in result] == ["dcid:VarA", "dcid:VarB"]
    assert result[0].entities[0].ref == "country/AAA"
    assert result[0].facet == {"unit": "dcid:USDollar"}
    assert result[0].change is not None
    assert result[0].change.absolute == -2_600_000.0
    assert result[0].change.percent == -21.0
    assert result[1].change is None


async def test_fetch_series_statistics_follows_the_links_redirect_without_credentials() -> None:
    data = _parquet_bytes([])
    digest = _digest_for(data)
    manifest = _manifest(rendering={"statistics": "series"}, artifact_digest=digest)
    artifact = manifest.artifacts[0]
    seen: list[httpx2.Request] = []

    def snapshot_handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=_snapshot_payload(artifact_digest=digest))

    def get_handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        if request.url.host == "storage.example":
            return httpx2.Response(200, content=data)
        return httpx2.Response(
            302, headers={"Location": "https://storage.example/signed?X-Goog-Signature=abc"}
        )

    async with _client(snapshot_handler) as upstream:
        client_holder.set_client(upstream)
        result = await fetch_series_statistics(
            subject_id=_SPEC_ID,
            content_version=_CONTENT_VERSION,
            artifact=artifact,
            expected_rows=0,
            transport=httpx2.MockTransport(get_handler),
        )

    assert result == []
    assert [r.url.path for r in seen] == [_LINK, "/signed"]
    credential_headers = {"authorization", "cookie", "x-dfo-principal"}
    for request in seen:
        assert credential_headers.isdisjoint(request.headers.keys())


async def test_fetch_series_statistics_zero_rows() -> None:
    data = _parquet_bytes([])
    digest = _digest_for(data)
    manifest = _manifest(rendering={"statistics": "series"}, artifact_digest=digest)
    artifact = manifest.artifacts[0]

    def snapshot_handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=_snapshot_payload(artifact_digest=digest))

    def get_handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, content=data)

    async with _client(snapshot_handler) as upstream:
        client_holder.set_client(upstream)
        result = await fetch_series_statistics(
            subject_id=_SPEC_ID,
            content_version=_CONTENT_VERSION,
            artifact=artifact,
            expected_rows=0,
            transport=httpx2.MockTransport(get_handler),
        )

    assert result == []


async def test_fetch_series_statistics_with_no_expected_rows_skips_the_count_check() -> None:
    """A manifest that carries the artifact but no `series_count` has no count to
    promise; the digest and schema checks still run."""
    data = _parquet_bytes([_row(stat_var="dcid:VarA"), _row(stat_var="dcid:VarB")])
    digest = _digest_for(data)
    manifest = _manifest(rendering={"statistics": "series"}, artifact_digest=digest)
    artifact = manifest.artifacts[0]

    def snapshot_handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=_snapshot_payload(artifact_digest=digest))

    def get_handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, content=data)

    async with _client(snapshot_handler) as upstream:
        client_holder.set_client(upstream)
        result = await fetch_series_statistics(
            subject_id=_SPEC_ID,
            content_version=_CONTENT_VERSION,
            artifact=artifact,
            expected_rows=None,
            transport=httpx2.MockTransport(get_handler),
        )

    assert [series.stat_var for series in result] == ["dcid:VarA", "dcid:VarB"]


# fetch_series_statistics -- oversized body refuses


async def test_fetch_series_statistics_refuses_a_body_over_the_advertised_size() -> None:
    data = _parquet_bytes([_row(), _row(stat_var="dcid:VarB")])
    digest = _digest_for(data)
    manifest = _manifest(
        rendering={"statistics": "series"},
        artifact_digest=digest,
        bytes_advertised=len(data) - 1,
    )
    artifact = manifest.artifacts[0]

    def snapshot_handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=_snapshot_payload(artifact_digest=digest))

    def get_handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, content=data)

    async with _client(snapshot_handler) as upstream:
        client_holder.set_client(upstream)
        with pytest.raises(SeriesStatisticsSchemaError, match="advertised size"):
            await fetch_series_statistics(
                subject_id=_SPEC_ID,
                content_version=_CONTENT_VERSION,
                artifact=artifact,
                expected_rows=2,
                transport=httpx2.MockTransport(get_handler),
            )


async def test_fetch_series_statistics_sends_accept_encoding_identity() -> None:
    """A compressed response could decode one small wire read into a chunk far bigger than
    what was actually read, before the per-chunk size check ever sees it; refusing
    compression on the request keeps a chunk `aiter_bytes` yields no larger than the bytes
    that came off the wire for it.
    """
    data = _parquet_bytes([_row()])
    digest = _digest_for(data)
    manifest = _manifest(rendering={"statistics": "series"}, artifact_digest=digest)
    artifact = manifest.artifacts[0]
    requests: list[httpx2.Request] = []

    def snapshot_handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=_snapshot_payload(artifact_digest=digest))

    def get_handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(200, content=data)

    async with _client(snapshot_handler) as upstream:
        client_holder.set_client(upstream)
        await fetch_series_statistics(
            subject_id=_SPEC_ID,
            content_version=_CONTENT_VERSION,
            artifact=artifact,
            expected_rows=1,
            transport=httpx2.MockTransport(get_handler),
        )

    assert requests[0].headers.get("accept-encoding") == "identity"


async def test_fetch_series_statistics_refuses_on_content_length_before_reading_the_body() -> None:
    """A `Content-Length` over `artifact.bytes` refuses before a single byte of the body is
    read: `_ExplodingStream` raises if anything ever iterates it, proving the check runs
    first rather than after the stream starts.
    """
    manifest = _manifest(rendering={"statistics": "series"}, bytes_advertised=100)
    artifact = manifest.artifacts[0]

    def snapshot_handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=_snapshot_payload())

    def get_handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200, stream=_ExplodingStream(), headers={"Content-Length": "1000000"}
        )

    async with _client(snapshot_handler) as upstream:
        client_holder.set_client(upstream)
        with pytest.raises(SeriesStatisticsSchemaError, match="Content-Length"):
            await fetch_series_statistics(
                subject_id=_SPEC_ID,
                content_version=_CONTENT_VERSION,
                artifact=artifact,
                expected_rows=1,
                transport=httpx2.MockTransport(get_handler),
            )


# _get_bounded -- the running total is checked before a chunk is appended, not after


async def test_get_bounded_accepts_a_body_landing_exactly_on_max_bytes() -> None:
    data = b"a" * 10

    def handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, content=data)

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as client:
        fetched = await _get_bounded(client, "https://artifact.example/blob", max_bytes=len(data))

    assert fetched == data


async def test_get_bounded_refuses_the_chunk_that_would_cross_max_bytes() -> None:
    """A first chunk landing exactly on `max_bytes` is kept; a second chunk of one more byte
    is refused. Checking `len(fetched) + len(chunk) > max_bytes` before the `+=` means the
    buffer never grows past `max_bytes` even transiently -- unlike checking after appending,
    which would briefly hold `max_bytes + len(chunk)` bytes before raising.
    """
    first, second = b"a" * 10, b"b"

    def handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, stream=_ChunkedStream([first, second]))

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as client:
        with pytest.raises(SeriesStatisticsSchemaError, match="advertised size of 10"):
            await _get_bounded(client, "https://artifact.example/blob", max_bytes=len(first))


# fetch_series_statistics -- artifact-digest mismatch refuses


@pytest.mark.parametrize("expected_rows", [1, None])
async def test_fetch_series_statistics_refuses_a_digest_mismatch(expected_rows: int | None) -> None:
    data = _parquet_bytes([_row()])
    manifest = _manifest(rendering={"statistics": "series"}, artifact_digest=_ARTIFACT_DIGEST)
    artifact = manifest.artifacts[0]

    def snapshot_handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=_snapshot_payload(artifact_digest=_ARTIFACT_DIGEST))

    def get_handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, content=data)

    async with _client(snapshot_handler) as upstream:
        client_holder.set_client(upstream)
        with pytest.raises(SeriesStatisticsSchemaError, match="artifact_digest"):
            await fetch_series_statistics(
                subject_id=_SPEC_ID,
                content_version=_CONTENT_VERSION,
                artifact=artifact,
                expected_rows=expected_rows,
                transport=httpx2.MockTransport(get_handler),
            )


# fetch_series_statistics -- schema-metadata mismatch refuses


async def test_fetch_series_statistics_refuses_a_wrong_schema_tag() -> None:
    data = _parquet_bytes([_row()], schema_name="series-statistics/2")
    digest = _digest_for(data)
    manifest = _manifest(rendering={"statistics": "series"}, artifact_digest=digest)
    artifact = manifest.artifacts[0]

    def snapshot_handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=_snapshot_payload(artifact_digest=digest))

    def get_handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, content=data)

    async with _client(snapshot_handler) as upstream:
        client_holder.set_client(upstream)
        with pytest.raises(SeriesStatisticsSchemaError, match="series-statistics/2"):
            await fetch_series_statistics(
                subject_id=_SPEC_ID,
                content_version=_CONTENT_VERSION,
                artifact=artifact,
                expected_rows=1,
                transport=httpx2.MockTransport(get_handler),
            )


async def test_fetch_series_statistics_refuses_a_missing_schema_tag() -> None:
    data = _parquet_bytes([_row()], schema_name=None)
    digest = _digest_for(data)
    manifest = _manifest(rendering={"statistics": "series"}, artifact_digest=digest)
    artifact = manifest.artifacts[0]

    def snapshot_handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=_snapshot_payload(artifact_digest=digest))

    def get_handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, content=data)

    async with _client(snapshot_handler) as upstream:
        client_holder.set_client(upstream)
        with pytest.raises(SeriesStatisticsSchemaError):
            await fetch_series_statistics(
                subject_id=_SPEC_ID,
                content_version=_CONTENT_VERSION,
                artifact=artifact,
                expected_rows=1,
                transport=httpx2.MockTransport(get_handler),
            )


# fetch_series_statistics -- row decode failure refuses


async def test_fetch_series_statistics_refuses_a_malformed_facet_cell() -> None:
    rows = [_row(), _row(stat_var="dcid:VarB")]
    rows[1]["facet"] = "{not valid json"
    data = _parquet_bytes(rows)
    digest = _digest_for(data)
    manifest = _manifest(rendering={"statistics": "series"}, artifact_digest=digest)
    artifact = manifest.artifacts[0]

    def snapshot_handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=_snapshot_payload(artifact_digest=digest))

    def get_handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, content=data)

    async with _client(snapshot_handler) as upstream:
        client_holder.set_client(upstream)
        with pytest.raises(SeriesStatisticsSchemaError, match="does not fit its schema"):
            await fetch_series_statistics(
                subject_id=_SPEC_ID,
                content_version=_CONTENT_VERSION,
                artifact=artifact,
                expected_rows=2,
                transport=httpx2.MockTransport(get_handler),
            )


# fetch_series_statistics -- row-count mismatch refuses


async def test_fetch_series_statistics_refuses_a_row_count_mismatch() -> None:
    data = _parquet_bytes([_row(), _row(stat_var="dcid:VarB")])
    digest = _digest_for(data)
    manifest = _manifest(rendering={"statistics": "series"}, artifact_digest=digest)
    artifact = manifest.artifacts[0]

    def snapshot_handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=_snapshot_payload(artifact_digest=digest))

    def get_handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, content=data)

    async with _client(snapshot_handler) as upstream:
        client_holder.set_client(upstream)
        with pytest.raises(SeriesStatisticsSchemaError, match="carries 2 row"):
            await fetch_series_statistics(
                subject_id=_SPEC_ID,
                content_version=_CONTENT_VERSION,
                artifact=artifact,
                expected_rows=3,
                transport=httpx2.MockTransport(get_handler),
            )


async def test_fetch_series_statistics_checks_row_count_before_converting_rows() -> None:
    """A row-count mismatch is caught from Parquet metadata, before any row --

    including a malformed one -- is decoded. `rows[1]`'s `facet` cell would fail
    `_row_to_model` if conversion were reached, so a row-count-mismatch error
    (not a row-decode error) proves the count was checked first.
    """
    rows = [_row(), _row(stat_var="dcid:VarB")]
    rows[1]["facet"] = "{not valid json"
    data = _parquet_bytes(rows)
    digest = _digest_for(data)
    manifest = _manifest(rendering={"statistics": "series"}, artifact_digest=digest)
    artifact = manifest.artifacts[0]

    def snapshot_handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=_snapshot_payload(artifact_digest=digest))

    def get_handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, content=data)

    async with _client(snapshot_handler) as upstream:
        client_holder.set_client(upstream)
        with pytest.raises(SeriesStatisticsSchemaError, match="carries 2 row"):
            await fetch_series_statistics(
                subject_id=_SPEC_ID,
                content_version=_CONTENT_VERSION,
                artifact=artifact,
                expected_rows=3,
                transport=httpx2.MockTransport(get_handler),
            )


async def test_fetch_series_statistics_never_decodes_columns_on_a_row_count_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The row count comes off `ParquetFile.metadata` alone. `ParquetFile.read()`, which
    decompresses every column, is never called once that count already disqualifies the
    artifact -- proven here by making `.read()` raise if it is ever reached, the way it
    would for a manifest naming five series over an honest artifact that actually holds
    hundreds of thousands.
    """
    data = _parquet_bytes([_row(), _row(stat_var="dcid:VarB")])
    digest = _digest_for(data)
    manifest = _manifest(rendering={"statistics": "series"}, artifact_digest=digest)
    artifact = manifest.artifacts[0]

    def explode(self: pq.ParquetFile) -> pa.Table:
        raise AssertionError("a row-count mismatch must refuse before any column is decoded")

    monkeypatch.setattr(pq.ParquetFile, "read", explode)

    def snapshot_handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=_snapshot_payload(artifact_digest=digest))

    def get_handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, content=data)

    async with _client(snapshot_handler) as upstream:
        client_holder.set_client(upstream)
        with pytest.raises(SeriesStatisticsSchemaError, match="carries 2 row"):
            await fetch_series_statistics(
                subject_id=_SPEC_ID,
                content_version=_CONTENT_VERSION,
                artifact=artifact,
                expected_rows=3,
                transport=httpx2.MockTransport(get_handler),
            )


# fetch_series_statistics -- fetch failure


async def test_fetch_series_statistics_raises_on_a_get_failure() -> None:
    manifest = _manifest(rendering={"statistics": "series"})
    artifact = manifest.artifacts[0]

    def snapshot_handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=_snapshot_payload())

    def get_handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(500, text="server error")

    async with _client(snapshot_handler) as upstream:
        client_holder.set_client(upstream)
        with pytest.raises(httpx2.HTTPStatusError):
            await fetch_series_statistics(
                subject_id=_SPEC_ID,
                content_version=_CONTENT_VERSION,
                artifact=artifact,
                expected_rows=1,
                transport=httpx2.MockTransport(get_handler),
            )


async def test_fetch_series_statistics_raises_on_malformed_bytes() -> None:
    data = b"not a parquet file"
    digest = _digest_for(data)
    manifest = _manifest(rendering={"statistics": "series"}, artifact_digest=digest)
    artifact = manifest.artifacts[0]

    def snapshot_handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=_snapshot_payload(artifact_digest=digest))

    def get_handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, content=data)

    async with _client(snapshot_handler) as upstream:
        client_holder.set_client(upstream)
        with pytest.raises(pa.ArrowException):
            await fetch_series_statistics(
                subject_id=_SPEC_ID,
                content_version=_CONTENT_VERSION,
                artifact=artifact,
                expected_rows=1,
                transport=httpx2.MockTransport(get_handler),
            )


# fetch_series_statistics -- deadline


async def test_fetch_series_statistics_raises_timeout_when_signing_outruns_its_deadline() -> None:
    artifact = _manifest(rendering={"statistics": "series"}).artifacts[0]

    async def snapshot_handler(_request: httpx2.Request) -> httpx2.Response:
        await asyncio.sleep(1.0)
        return httpx2.Response(200, json=_snapshot_payload())

    def get_handler(_request: httpx2.Request) -> httpx2.Response:
        raise AssertionError("no GET is expected once signing has outrun the deadline")

    async with _client(snapshot_handler) as upstream:
        client_holder.set_client(upstream)
        with pytest.raises(TimeoutError):
            await fetch_series_statistics(
                subject_id=_SPEC_ID,
                content_version=_CONTENT_VERSION,
                artifact=artifact,
                expected_rows=1,
                transport=httpx2.MockTransport(get_handler),
                deadline=CallDeadline.after(0.05),
            )


async def test_fetch_series_statistics_raises_timeout_when_the_get_outruns_its_deadline() -> None:
    artifact = _manifest(rendering={"statistics": "series"}).artifacts[0]

    def snapshot_handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=_snapshot_payload())

    async def get_handler(_request: httpx2.Request) -> httpx2.Response:
        await asyncio.sleep(1.0)
        return httpx2.Response(200, content=b"")

    async with _client(snapshot_handler) as upstream:
        client_holder.set_client(upstream)
        with pytest.raises(TimeoutError):
            await fetch_series_statistics(
                subject_id=_SPEC_ID,
                content_version=_CONTENT_VERSION,
                artifact=artifact,
                expected_rows=1,
                transport=httpx2.MockTransport(get_handler),
                deadline=CallDeadline.after(0.05),
            )


# fetch_series_statistics_table -- the checked Arrow table


async def _fetch_table(data: bytes, *, expected_rows: int) -> pa.Table:
    """`fetch_series_statistics_table` over `data`, served as the artifact's bytes."""
    digest = _digest_for(data)
    manifest = _manifest(rendering={"statistics": "series"}, artifact_digest=digest)
    artifact = manifest.artifacts[0]

    def snapshot_handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=_snapshot_payload(artifact_digest=digest))

    def get_handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, content=data)

    async with _client(snapshot_handler) as upstream:
        client_holder.set_client(upstream)
        return await fetch_series_statistics_table(
            subject_id=_SPEC_ID,
            content_version=_CONTENT_VERSION,
            artifact=artifact,
            expected_rows=expected_rows,
            transport=httpx2.MockTransport(get_handler),
        )


def _with_column(data: bytes, name: str, column: pa.Array) -> bytes:
    """`data` with column `name` replaced by `column`, schema metadata kept."""
    table = pq.read_table(io.BytesIO(data))
    table = table.set_column(table.schema.get_field_index(name), name, column)
    buf = io.BytesIO()
    pq.write_table(table.replace_schema_metadata({"dfo.schema": SERIES_STATISTICS_SCHEMA}), buf)
    return buf.getvalue()


async def test_the_table_carries_every_row_in_the_producers_schema() -> None:
    rows = [_row(stat_var="dcid:VarA"), _row(stat_var="dcid:VarB", change=None)]

    table = await _fetch_table(_parquet_bytes(rows), expected_rows=2)

    assert table.schema.equals(SERIES_STATISTICS_ARROW_SCHEMA)
    assert table["stat_var"].to_pylist() == ["dcid:VarA", "dcid:VarB"]
    assert table["change_absolute"].to_pylist() == [-2_600_000.0, None]


async def test_the_table_accepts_a_column_its_type_can_hold() -> None:
    # A producer writing int32 observations still fits: the cast widens it.
    rows = [_row(observations=7)]
    data = _with_column(_parquet_bytes(rows), "observations", pa.array([7], pa.int32()))

    table = await _fetch_table(data, expected_rows=1)

    assert table["observations"].to_pylist() == [7]


@pytest.mark.parametrize(
    ("column", "values"),
    [
        pytest.param("stat_var", pa.array([None], pa.string()), id="null-stat-var"),
        pytest.param("first_period", pa.array([None], pa.string()), id="null-period"),
        pytest.param("latest_value", pa.array([None], pa.float64()), id="null-value"),
        pytest.param("observations", pa.array([2.5], pa.float64()), id="fractional-count"),
        pytest.param(
            "entities",
            pa.array([[{"role": None, "ref": "country/TGO"}]], _ENTITIES_TYPE),
            id="null-role",
        ),
        pytest.param("entities", pa.array([[None]], _ENTITIES_TYPE), id="null-entity"),
        pytest.param("entities", pa.array([None], _ENTITIES_TYPE), id="null-entities"),
        pytest.param("facet", pa.array(['["unit"]']), id="facet-not-an-object"),
        pytest.param("facet", pa.array(['{"unit": 5}']), id="facet-value-not-text"),
    ],
)
async def test_the_table_refuses_a_row_no_series_model_could_hold(
    column: str, values: pa.Array
) -> None:
    data = _with_column(_parquet_bytes([_row()]), column, values)

    with pytest.raises(SeriesStatisticsSchemaError):
        await _fetch_table(data, expected_rows=1)


async def test_the_table_refuses_a_missing_column() -> None:
    table = pq.read_table(io.BytesIO(_parquet_bytes([_row()]))).drop_columns(["latest_period"])
    buf = io.BytesIO()
    pq.write_table(table.replace_schema_metadata({"dfo.schema": SERIES_STATISTICS_SCHEMA}), buf)

    with pytest.raises(SeriesStatisticsSchemaError):
        await _fetch_table(buf.getvalue(), expected_rows=1)


async def test_a_table_built_from_models_reads_back_as_the_same_models() -> None:
    data = _parquet_bytes([_row(stat_var="dcid:VarA"), _row(stat_var="dcid:VarB", change=None)])
    table = await _fetch_table(data, expected_rows=2)
    digest = _digest_for(data)
    artifact = _manifest(rendering={"statistics": "series"}, artifact_digest=digest).artifacts[0]

    def snapshot_handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=_snapshot_payload(artifact_digest=digest))

    def get_handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, content=data)

    async with _client(snapshot_handler) as upstream:
        client_holder.set_client(upstream)
        models = await fetch_series_statistics(
            subject_id=_SPEC_ID,
            content_version=_CONTENT_VERSION,
            artifact=artifact,
            expected_rows=2,
            transport=httpx2.MockTransport(get_handler),
        )

    assert series_statistics_table(models).equals(table)

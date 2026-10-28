"""Measure the frame path's peak memory and wall time at several row counts.

Every step runs in a fresh subprocess so one step's peak never hides in the next. A
step's input is built first (outside the timed region), then the step runs alone. Inputs
are built the way production builds them: Arrow columns through `build_frame_from_arrow`,
and an analysis input read back out of the store, so a verb sees the table layout a
stored frame decodes to.

Single stages:

- decode: data-mcp's parse of a Parquet artifact shaped like the apis service's.
- store_put, store_get: `GcsMaterialisationStore` over an in-memory fake bucket, so the
  Parquet codec runs exactly as in production.
- frame_sha256: the grammar's content hash of the frame.
- aggregate, filter: the engine entry points over the CRS-shaped frame.
- rank, growth, compare, share_of, join: the engine entry points over a single-StatVar
  frame of the same row count (these verbs refuse more than one StatVar), which adds a
  `sector` role so one StatVar can still reach 250,000 rows.
- rank_limit, rank_per_period, rank_per_period_limit: `rank` with `limit=10`, with
  `per_period` on, and with both. These rank the 169 recipients (1,690 recipient-year
  cells under `per_period`).
- rank_entities, rank_entities_limit: `rank` over every (provider, recipient, sector)
  triple, n / 10 entities, without and with `limit=10`.

join and join_call run under the production row ceiling (analysis-mcp's `max_frame_rows`
default) and join two frames that share half their keys: the right side's sector is
renamed on every other row. Two rows a matched key make the result as many rows as one
input, the worst join the ceiling admits; a full match would build twice that and be
refused. These figures therefore differ from runs that joined two fully matching frames
with the ceiling lifted. Every other step lifts every ceiling.
- wire: presentation-mcp's `publish_body`, the encoded publish request body.

Whole calls, each starting from an empty process with its inputs held only in the fake
bucket, as a deployed instance holds them in GCS:

- get_data: data-mcp's `_verify_parse_and_register` over the fetched artifact bytes
  (digest check, footer row count, decode, store write).
- aggregate_call ... filter_call: the registered analysis tool function (store read,
  engine, identity mint, store write, digest), over the same inputs as the engine steps.
- publish_call: presentation-mcp's `publish_body_for` (store read, then `publish_body`).

Each step reports `peak_rss_mb` (the process high-water mark, `ru_maxrss`, which includes
the input build) and `delta_mb` (the highest sampled RSS during the step minus the RSS
just before it). Compare `delta_mb` across runs; `peak_rss_mb` is what a container sees.
A whole call's `delta_mb` is what one call adds to an idle instance. `wire` and
`publish_call` also report `body_bytes`, the size of the publish request body.

A whole call also reports `claimed_rows`, the most rows it held at once in the process's
frame row budget (`common.row_budget`): a join claims both inputs. After the table the
script checks the budget. It refuses, and exits 1, unless every whole-call step ran at
the largest row count and claimed rows, so a subset run or a call that stopped claiming
cannot pass. Otherwise it prints the worst bytes a claimed row, the largest `delta_mb` /
`claimed_rows` among those calls, and what that implies at `--memory-mib`:
`(memory - idle - headroom) / worst bytes a row` rows in flight, how many
`max_frame_rows` calls that admits at once, and whether the configured
`frame_rows_in_flight_max` exceeds it, in which case it exits 1. Rerun it whenever
`max_frame_rows`, `frame_rows_in_flight_max` or the instance's memory changes.

    uv run python scripts/bench_frames.py --out bench.json
    uv run python scripts/bench_frames.py --rows 50000 --steps decode,aggregate --out bench.json
    uv run python scripts/bench_frames.py --rows 250000 --memory-mib 2048
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import hashlib
import io
import json
import os
import pickle
import platform
import random
import resource
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from analysis_mcp.aggregate_tool import aggregate
from analysis_mcp.compare_tool import compare
from analysis_mcp.engine.aggregate import run_aggregate
from analysis_mcp.engine.compare import run_compare
from analysis_mcp.engine.filter import run_filter
from analysis_mcp.engine.growth import run_growth
from analysis_mcp.engine.join import run_join
from analysis_mcp.engine.rank import run_rank
from analysis_mcp.engine.share_of import run_share_of
from analysis_mcp.execution_identity import current_execution_identity
from analysis_mcp.filter_tool import filter_verb
from analysis_mcp.growth_tool import growth
from analysis_mcp.identity import op_spec_hash_for
from analysis_mcp.join_tool import join
from analysis_mcp.op_spec import AggregateOpSpec, CategoryMatchPredicate, OrderedThresholdPredicate
from analysis_mcp.rank_tool import rank
from analysis_mcp.settings import Settings as AnalysisSettings
from analysis_mcp.share_of_tool import share_of
from analysis_mcp.trace import AppliedPolicy, ExecutionRecord, Trace
from common.materialisation import FrameRef, MaterialisationTrace, StoredFrameTrace, set_store
from common.materialisation.gcs_store import GcsMaterialisationStore
from common.place_type import kind_column
from common.row_budget import RowBudget, row_budget, set_row_budget
from common.settings import BaseServiceSettings
from data_mcp.frame_materialisation import (
    _verify_parse_and_register,
    frame_from_parquet_bytes,
)
from data_mcp.manifest import strip_manifest
from data_mcp.settings import Settings as DataSettings
from dfo_frame_grammar import Frame, build_frame_from_arrow, frame_sha256
from fastmcp.tools import ToolResult
from presentation_mcp.frame_publish import publish_body, publish_body_for

DEFAULT_ROWS = (50_000, 60_000, 250_000)
N_STAT_VARS = 174
N_PROVIDERS = 7
N_RECIPIENTS = 169
YEARS = tuple(str(year) for year in range(2015, 2025))
SAMPLE_INTERVAL_S = 0.01
SEED = 20260930
# The instance memory the budget figures assume unless `--memory-mib` says otherwise, the
# deployed harness's estimated memory at rest (ADK, the harness and three servers), and
# the headroom kept for the freed Arrow pages a long-lived allocator holds on to.
DEFAULT_MEMORY_MIB = 2048
DEFAULT_IDLE_MIB = 400
DEFAULT_HEADROOM_MIB = 300
# Every ceiling a step could meet is lifted, so a step measures its row count, never a refusal.
UNBOUNDED_ROWS = 10_000_000
# join and join_call keep the production row ceiling, so they measure only a join a caller
# can complete.
PRODUCTION_MAX_FRAME_ROWS = AnalysisSettings.model_fields["max_frame_rows"].default
PRODUCTION_CEILING_STEPS = frozenset({"join", "join_call"})
UNBOUNDED_BYTES = 2**33

CRS_ROLES = ("financingProvider", "financingRecipient")
SECTOR_ROLES = (*CRS_ROLES, "sector")
# Each party's kind, as `get_data(place_types=True)` serves it: the verbs that pool or rank
# several parties refuse a frame without it. Role keys of the frame, never ranked or grouped.
KIND_COLUMNS = tuple(kind_column(role) for role in CRS_ROLES)
SECTOR_A = "dcid:bench_sv_a"
SECTOR_B = "dcid:bench_sv_b"

STEPS = (
    "decode",
    "store_put",
    "store_get",
    "frame_sha256",
    "aggregate",
    "rank",
    "rank_limit",
    "rank_per_period",
    "rank_per_period_limit",
    "rank_entities",
    "rank_entities_limit",
    "share_of",
    "compare",
    "growth",
    "join",
    "filter",
    "wire",
    "get_data",
    "aggregate_call",
    "rank_call",
    "share_of_call",
    "compare_call",
    "growth_call",
    "join_call",
    "filter_call",
    "publish_call",
)
# The steps that run a whole call; the budget check needs every one of them.
WHOLE_CALL_STEPS = tuple(step for step in STEPS if step == "get_data" or step.endswith("_call"))


def _labels(prefix: str, count: int) -> pa.Array:
    return pa.array([f"{prefix}{index}" for index in range(count)], type=pa.string())


def _take(labels: pa.Array, codes: list[int]) -> pa.Array:
    return labels.take(pa.array(codes, type=pa.int32()))


def _values(rng: random.Random, n: int) -> pa.Array:
    return pa.array([rng.uniform(0.0, 1_000_000.0) for _ in range(n)], type=pa.float64())


def _crs_table(n: int) -> pa.Table:
    """`n` distinct (StatVar, provider, recipient, year) rows with non-round float values,
    built as Arrow columns from small integer codes, every party of kind `party`."""
    space = N_STAT_VARS * N_PROVIDERS * N_RECIPIENTS * len(YEARS)
    if n > space:
        raise SystemExit(f"{n} rows exceed the {space}-cell CRS grain")
    rng = random.Random(SEED)
    stat_vars, providers, recipients, years = [], [], [], []
    for index in rng.sample(range(space), n):
        rest, year = divmod(index, len(YEARS))
        rest, recipient = divmod(rest, N_RECIPIENTS)
        stat_var, provider = divmod(rest, N_PROVIDERS)
        stat_vars.append(stat_var)
        providers.append(provider)
        recipients.append(recipient)
        years.append(year)
    return pa.table(
        {
            "stat_var": _take(_labels("dcid:crs_sv_", N_STAT_VARS), stat_vars),
            "financingProvider": _take(_labels("dcid:provider_", N_PROVIDERS), providers),
            "financingRecipient": _take(_labels("dcid:recipient_", N_RECIPIENTS), recipients),
            "time_period": _take(pa.array(YEARS, type=pa.string()), years),
            "value": _values(rng, n),
            "unit": pa.repeat("dcid:USD", n),
            **_party_kinds(n),
        }
    )


def _sector_table(n: int, *, stat_var: str) -> pa.Table:
    """`n` rows of one StatVar: every year for each of `n / 10` (provider, recipient, sector)
    entities, in shuffled order, so growth and share_of find no gap years, every party of
    kind `party`."""
    if n % len(YEARS):
        raise SystemExit(f"{n} rows is not a multiple of the {len(YEARS)} years")
    rng = random.Random(SEED)
    cells = list(range(n))
    rng.shuffle(cells)
    sectors, providers, recipients, years = [], [], [], []
    for cell in cells:
        entity, year = divmod(cell, len(YEARS))
        sector, pair = divmod(entity, N_PROVIDERS * N_RECIPIENTS)
        provider, recipient = divmod(pair, N_RECIPIENTS)
        sectors.append(sector)
        providers.append(provider)
        recipients.append(recipient)
        years.append(year)
    return pa.table(
        {
            "stat_var": pa.repeat(stat_var, n),
            "financingProvider": _take(_labels("dcid:provider_", N_PROVIDERS), providers),
            "financingRecipient": _take(_labels("dcid:recipient_", N_RECIPIENTS), recipients),
            "sector": _take(_labels("dcid:sector_", max(sectors) + 1), sectors),
            "time_period": _take(pa.array(YEARS, type=pa.string()), years),
            "value": _values(rng, n),
            "unit": pa.repeat("dcid:USD", n),
            **_party_kinds(n),
        }
    )


def _party_kinds(n: int) -> dict[str, pa.Array]:
    return {column: pa.repeat("party", n) for column in KIND_COLUMNS}


def _crs_frame(n: int) -> Frame:
    return build_frame_from_arrow(_crs_table(n), role_keys=(*CRS_ROLES, *KIND_COLUMNS))


def _sector_frame(n: int, *, stat_var: str) -> Frame:
    return build_frame_from_arrow(
        _sector_table(n, stat_var=stat_var), role_keys=(*SECTOR_ROLES, *KIND_COLUMNS)
    )


def _artifact_bytes(n: int) -> bytes:
    """A Parquet file laid out like the apis service's artifact: string columns, a float64 `value`,
    a string `unit`, and an all-null `status`."""
    table = _crs_table(n)
    table = table.append_column("status", pa.nulls(n, type=pa.string()))
    sink = io.BytesIO()
    pq.write_table(table, sink)
    return sink.getvalue()


class _FakeBlob:
    def __init__(
        self, objects: dict[str, bytes], metadata: dict[str, dict[str, str]], name: str
    ) -> None:
        self._objects = objects
        self._metadata = metadata
        self._name = name
        self.metadata: dict[str, str] | None = None
        self.size: int | None = None

    def upload_from_string(
        self, data: bytes, *, content_type: str, if_generation_match: int
    ) -> None:
        del content_type, if_generation_match
        self._objects[self._name] = data
        self._metadata[self._name] = dict(self.metadata or {})

    def download_as_bytes(self) -> bytes:
        # A copy, as a real download allocates the object's bytes afresh.
        return bytes(memoryview(self._objects[self._name]))

    def reload(self) -> None:
        if self._name not in self._objects:
            from google.api_core.exceptions import NotFound  # noqa: PLC0415

            raise NotFound(self._name)
        self.metadata = dict(self._metadata.get(self._name, {}))
        self.size = len(self._objects[self._name])


class _FakeBucket:
    def __init__(self, objects: dict[str, bytes], metadata: dict[str, dict[str, str]]) -> None:
        self._objects = objects
        self._metadata = metadata

    def blob(self, name: str) -> _FakeBlob:
        return _FakeBlob(self._objects, self._metadata, name)


class _FakeClient:
    def __init__(
        self,
        objects: dict[str, bytes] | None = None,
        metadata: dict[str, dict[str, str]] | None = None,
    ) -> None:
        self.objects: dict[str, bytes] = objects if objects is not None else {}
        self.metadata: dict[str, dict[str, str]] = metadata if metadata is not None else {}

    def bucket(self, name: str) -> _FakeBucket:
        del name
        return _FakeBucket(self.objects, self.metadata)

    def close(self) -> None:
        pass


def _store(client: _FakeClient) -> GcsMaterialisationStore:
    return GcsMaterialisationStore(
        bucket_name="bench",
        client=client,  # ty: ignore[invalid-argument-type]
        max_frame_rows=UNBOUNDED_ROWS,
        max_encoded_bytes=UNBOUNDED_BYTES,
        max_uncompressed_bytes=UNBOUNDED_BYTES,
    )


def _ref(tag: str, *, derived: bool = False) -> FrameRef:
    digest = hashlib.sha256(tag.encode()).hexdigest()[:32]
    if derived:
        return FrameRef(spec_id=f"ops_{digest}", content_version=f"ocv_{digest}")
    return FrameRef(spec_id=f"spec_{digest}", content_version=f"cv_{digest}")


def _handle(ref: FrameRef) -> dict[str, str]:
    return {"spec_id": ref.spec_id, "content_version": ref.content_version}


def _materialisation_trace(ref: FrameRef, frame: Frame) -> MaterialisationTrace:
    """What a `get_data` registration stores beside its frame."""
    return MaterialisationTrace(
        upstream_trace={
            "applied_window": None,
            "graph_revision": "rev_bench",
            "source_revisions": [{"source": "bench", "revision": "1"}],
            "resolved_stat_vars": [],
            "facets": [{"axis": "unit", "ref": "dcid:USD", "basis": "declared_default"}],
            "normalisations": [],
            "conversion": None,
            "materialiser_build": "bench",
            "capability_version": "cap_bench",
        },
        spec_id=ref.spec_id,
        content_version=ref.content_version,
    )


def _verb_trace(ref: FrameRef, frame: Frame) -> Trace:
    """What an analysis verb stores beside its result frame, which is what gets published."""
    del frame
    op_spec = AggregateOpSpec(frame_ref=_ref("bench-input"), group_by=(), aggregation="sum")
    return Trace(
        upstream_trace={
            "graph_revision": "rev_bench",
            "source_revisions": [{"source": "bench", "revision": "1"}],
            "facets": [],
            "conversion": None,
        },
        frame_content_version="cv_" + "d" * 32,
        op_spec_hash=op_spec_hash_for(op_spec),
        execution_record=ExecutionRecord(content="{}"),
        applied_policy=AppliedPolicy(),
        content_version=ref.content_version,
        input_content_versions=[_ref("bench-input").content_version],
        execution_identity=current_execution_identity(),
    )


@dataclass(frozen=True)
class _Inputs:
    """A step's inputs as a deployed instance holds them: objects in the bucket, and for
    a `get_data` step the artifact bytes it fetches."""

    objects: dict[str, bytes]
    metadata: dict[str, dict[str, str]]
    refs: tuple[FrameRef, ...] = ()
    artifact: bytes = b""


def _stored_inputs(
    frames: dict[str, Frame],
    *,
    trace: Callable[[FrameRef, Frame], StoredFrameTrace] = _materialisation_trace,
    derived: bool = False,
) -> _Inputs:
    client = _FakeClient()
    store = _store(client)
    refs = []
    for tag, frame in frames.items():
        ref = _ref(tag, derived=derived)
        asyncio.run(store.put(ref, frame=frame, trace=trace(ref, frame), writer="bench"))
        refs.append(ref)
    return _Inputs(client.objects, client.metadata, tuple(refs))


def _crs_inputs(n: int) -> _Inputs:
    return _stored_inputs({"crs": _crs_frame(n)})


def _sector_inputs(n: int) -> _Inputs:
    return _stored_inputs({"sector_a": _sector_frame(n, stat_var=SECTOR_A)})


def _join_inputs(n: int) -> _Inputs:
    """Two n-row frames sharing n/2 keys, the right side's sector renamed on every other
    row, which keeps its keys unique."""
    left = _sector_table(n, stat_var=SECTOR_A)
    right = _sector_table(n, stat_var=SECTOR_B)
    sectors = right.column("sector").to_pylist()
    renamed = pa.array(
        [f"dcid:other_{label}" if index % 2 else label for index, label in enumerate(sectors)],
        type=pa.string(),
    )
    right = right.set_column(right.schema.get_field_index("sector"), "sector", renamed)
    return _stored_inputs(
        {
            "sector_a": build_frame_from_arrow(left, role_keys=(*SECTOR_ROLES, *KIND_COLUMNS)),
            "sector_b": build_frame_from_arrow(right, role_keys=(*SECTOR_ROLES, *KIND_COLUMNS)),
        }
    )


def _artifact_inputs(n: int) -> _Inputs:
    return _Inputs({}, {}, artifact=_artifact_bytes(n))


def _published_inputs(n: int) -> _Inputs:
    return _stored_inputs({"published": _crs_frame(n)}, trace=_verb_trace, derived=True)


def _read(store: GcsMaterialisationStore, ref: FrameRef) -> Frame:
    held = asyncio.run(store.get(ref))
    assert held is not None
    return held[0]


def _settled(result: ToolResult) -> None:
    """Refuse to time a call that refused: a refusal skips the work being measured."""
    if result.is_error:
        raise RuntimeError(f"the call refused: {result.structured_content}")


_Run = Callable[[], dict[str, int] | None]
_Setup = Callable[[GcsMaterialisationStore, _Inputs], _Run]


def _setup_decode(store: GcsMaterialisationStore, inputs: _Inputs) -> _Run:
    del store

    def run() -> None:
        frame_from_parquet_bytes(inputs.artifact)

    return run


def _setup_store_put(store: GcsMaterialisationStore, inputs: _Inputs) -> _Run:
    frame = _read(store, inputs.refs[0])
    ref = _ref("put")
    trace = _materialisation_trace(ref, frame)
    objects = inputs.objects

    def run() -> dict[str, int]:
        asyncio.run(store.put(ref, frame=frame, trace=trace, writer="bench"))
        stored = next(data for name, data in objects.items() if ref.content_version in name)
        footer = pq.ParquetFile(io.BytesIO(stored)).metadata
        uncompressed = sum(
            footer.row_group(group).column(column).total_uncompressed_size
            for group in range(footer.num_row_groups)
            for column in range(footer.row_group(group).num_columns)
        )
        return {"object_bytes": len(stored), "uncompressed_bytes": uncompressed}

    return run


def _setup_store_get(store: GcsMaterialisationStore, inputs: _Inputs) -> _Run:
    def run() -> None:
        _read(store, inputs.refs[0])

    return run


def _setup_frame_sha256(store: GcsMaterialisationStore, inputs: _Inputs) -> _Run:
    frame = _read(store, inputs.refs[0])

    def run() -> None:
        frame_sha256(frame)

    return run


def _setup_aggregate(store: GcsMaterialisationStore, inputs: _Inputs) -> _Run:
    frame = _read(store, inputs.refs[0])

    def run() -> None:
        run_aggregate(frame, group_by=["stat_var", "financingProvider"], aggregation="sum")

    return run


def _rank_setup(
    *, rank_by: list[str], per_period: bool = False, limit: int | None = None
) -> _Setup:
    def setup(store: GcsMaterialisationStore, inputs: _Inputs) -> _Run:
        frame = _read(store, inputs.refs[0])

        def run() -> None:
            run_rank(
                frame,
                rank_by=rank_by,
                aggregation="sum",
                window_start=None,
                window_end=None,
                ascending=False,
                per_period=per_period,
                limit=limit,
            )

        return run

    return setup


RANK_LIMIT = 10
RANK_BY_RECIPIENT = ["financingRecipient"]
# One entity a (sector, provider, recipient) triple: n / 10 entities, where
# `RANK_BY_RECIPIENT` ranks N_RECIPIENTS.
RANK_BY_TRIPLE = list(SECTOR_ROLES)


SHARE_NUMERATOR = {"financingProvider": "dcid:provider_0"}
SHARE_BY = ["financingRecipient", "sector"]


def _setup_share_of(store: GcsMaterialisationStore, inputs: _Inputs) -> _Run:
    frame = _read(store, inputs.refs[0])

    def run() -> None:
        run_share_of(
            frame,
            frame,
            stat_var=SECTOR_A,
            numerator_entity=SHARE_NUMERATOR,
            denominator_entity={},
            numerator_by=SHARE_BY,
            denominator_by={column: column for column in SHARE_BY},
        )

    return run


COMPARE_LEFT = {
    "financingProvider": "dcid:provider_0",
    "financingRecipient": "dcid:recipient_0",
    "sector": "dcid:sector_0",
}
COMPARE_RIGHT = {**COMPARE_LEFT, "financingRecipient": "dcid:recipient_1"}


def _setup_compare(store: GcsMaterialisationStore, inputs: _Inputs) -> _Run:
    frame = _read(store, inputs.refs[0])

    def run() -> None:
        run_compare(frame, frame, left_entity=COMPARE_LEFT, right_entity=COMPARE_RIGHT)

    return run


def _setup_growth(store: GcsMaterialisationStore, inputs: _Inputs) -> _Run:
    frame = _read(store, inputs.refs[0])

    def run() -> None:
        run_growth(frame, window_start=YEARS[0], window_end=YEARS[-1], entity_by=list(SECTOR_ROLES))

    return run


def _setup_join(store: GcsMaterialisationStore, inputs: _Inputs) -> _Run:
    left = _read(store, inputs.refs[0])
    right = _read(store, inputs.refs[1])

    def run() -> None:
        run_join(left, right, max_output_rows=PRODUCTION_MAX_FRAME_ROWS)

    return run


FILTER_PREDICATES = (
    OrderedThresholdPredicate(column="value", op="gte", value=250_000.0),
    CategoryMatchPredicate(
        column="financingProvider", values=("dcid:provider_0", "dcid:provider_1")
    ),
)


def _setup_filter(store: GcsMaterialisationStore, inputs: _Inputs) -> _Run:
    frame = _read(store, inputs.refs[0])

    def run() -> None:
        run_filter(frame, predicates=list(FILTER_PREDICATES))

    return run


def _setup_wire(store: GcsMaterialisationStore, inputs: _Inputs) -> _Run:
    frame = _read(store, inputs.refs[0])

    def run() -> dict[str, int]:
        body = publish_body(
            frame,
            spec_id=inputs.refs[0].spec_id,
            content_version=inputs.refs[0].content_version,
            input_content_versions=["cv_" + "c" * 32],
        )
        return {"body_bytes": len(body)}

    return run


def _setup_get_data(store: GcsMaterialisationStore, inputs: _Inputs) -> _Run:
    ref = _ref("get_data")
    digest = f"sha256:{hashlib.sha256(inputs.artifact).hexdigest()}"
    rendering = {"bundle": False, "format": "parquet"}
    manifest = strip_manifest(
        {
            "schema_version": 1,
            "content_version": ref.content_version,
            "coverage": {"rows": 1, "dimensions": []},
            "citation": {"text": "bench", "sources": []},
            "trace": {"graph_revision": "rev_bench", "facets": []},
            "warnings": [],
            "artifacts": [
                {
                    "artifact_digest": digest,
                    "rendering": rendering,
                    "bytes": len(inputs.artifact),
                    "media_type": "application/vnd.apache.parquet",
                    "created_at": "2026-09-30T00:00:00+00:00",
                    "tier": "working",
                }
            ],
            "definition": {"schema_version": 1, "spec_id": ref.spec_id, "identity": {}},
        }
    )
    settings = DataSettings(max_frame_rows=UNBOUNDED_ROWS)

    def run() -> dict[str, int]:
        # A copy, as the fetch allocates the artifact's bytes afresh.
        fetched = bytes(memoryview(inputs.artifact))
        # Read as a requested rendering, which validates every cell. The canonical path
        # trusts the cells but needs a content version minted from the frame's hash.
        outcome = asyncio.run(
            _verify_parse_and_register(
                fetched=fetched,
                artifact_digest=digest,
                canonical=False,
                ref=ref,
                manifest=manifest,
                options=rendering,
                settings=settings,
                writer="bench",
                store=store,
                write_deadline=None,
            )
        )
        if outcome.status != "registered":
            raise RuntimeError(f"get_data did not register: {outcome}")
        return {"artifact_bytes": len(fetched)}

    return run


def _tool_setup(call: Callable[..., Coroutine[Any, Any, ToolResult]]) -> _Setup:
    """A step running `call` with the step's frame handles, as the tool server runs it."""

    def setup(store: GcsMaterialisationStore, inputs: _Inputs) -> _Run:
        set_store(store)
        handles = [_handle(ref) for ref in inputs.refs]

        def run() -> None:
            _settled(asyncio.run(call(*handles)))

        return run

    return setup


def _setup_publish_call(store: GcsMaterialisationStore, inputs: _Inputs) -> _Run:
    set_store(store)
    ref = inputs.refs[0]

    async def body_bytes() -> int:
        async with publish_body_for(
            spec_id=ref.spec_id,
            content_version=ref.content_version,
            max_frame_rows=UNBOUNDED_ROWS,
        ) as body:
            return len(body.data)

    def run() -> dict[str, int]:
        return {"body_bytes": asyncio.run(body_bytes())}

    return run


SETUPS: dict[str, tuple[Callable[[int], _Inputs], _Setup]] = {
    "decode": (_artifact_inputs, _setup_decode),
    "store_put": (_crs_inputs, _setup_store_put),
    "store_get": (_crs_inputs, _setup_store_get),
    "frame_sha256": (_crs_inputs, _setup_frame_sha256),
    "aggregate": (_crs_inputs, _setup_aggregate),
    "rank": (_sector_inputs, _rank_setup(rank_by=RANK_BY_RECIPIENT)),
    "rank_limit": (_sector_inputs, _rank_setup(rank_by=RANK_BY_RECIPIENT, limit=RANK_LIMIT)),
    "rank_per_period": (_sector_inputs, _rank_setup(rank_by=RANK_BY_RECIPIENT, per_period=True)),
    "rank_per_period_limit": (
        _sector_inputs,
        _rank_setup(rank_by=RANK_BY_RECIPIENT, per_period=True, limit=RANK_LIMIT),
    ),
    "rank_entities": (_sector_inputs, _rank_setup(rank_by=RANK_BY_TRIPLE)),
    "rank_entities_limit": (
        _sector_inputs,
        _rank_setup(rank_by=RANK_BY_TRIPLE, limit=RANK_LIMIT),
    ),
    "share_of": (_sector_inputs, _setup_share_of),
    "compare": (_sector_inputs, _setup_compare),
    "growth": (_sector_inputs, _setup_growth),
    "join": (_join_inputs, _setup_join),
    "filter": (_crs_inputs, _setup_filter),
    "wire": (_crs_inputs, _setup_wire),
    "get_data": (_artifact_inputs, _setup_get_data),
    "aggregate_call": (
        _crs_inputs,
        _tool_setup(
            lambda ref: aggregate(
                frame_ref=ref, aggregation="sum", group_by=("stat_var", "financingProvider")
            )
        ),
    ),
    "rank_call": (
        _sector_inputs,
        _tool_setup(
            lambda ref: rank(frame_ref=ref, rank_by=("financingRecipient",), aggregation="sum")
        ),
    ),
    "share_of_call": (
        _sector_inputs,
        _tool_setup(
            lambda ref: share_of(
                frame_ref=ref,
                numerator_entity=SHARE_NUMERATOR,
                denominator_entity={},
                numerator_by=SHARE_BY,
                denominator_by={column: column for column in SHARE_BY},
            )
        ),
    ),
    "compare_call": (
        _sector_inputs,
        _tool_setup(
            lambda ref: compare(frame_ref=ref, left_entity=COMPARE_LEFT, right_entity=COMPARE_RIGHT)
        ),
    ),
    "growth_call": (
        _sector_inputs,
        _tool_setup(
            lambda ref: growth(
                frame_ref=ref, window_start=YEARS[0], window_end=YEARS[-1], entity_by=SECTOR_ROLES
            )
        ),
    ),
    "join_call": (
        _join_inputs,
        _tool_setup(lambda left, right: join(frame_ref=left, right_frame_ref=right)),
    ),
    "filter_call": (
        _crs_inputs,
        _tool_setup(lambda ref: filter_verb(frame_ref=ref, predicates=FILTER_PREDICATES)),
    ),
    "publish_call": (_published_inputs, _setup_publish_call),
}


def _maxrss_mb() -> float:
    """`ru_maxrss` in MiB: bytes on macOS, KiB on Linux."""
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / (1024 * 1024) if platform.system() == "Darwin" else raw / 1024


def _current_rss_mb() -> float:
    statm = Path("/proc/self/statm")
    if statm.exists():
        return int(statm.read_text().split()[1]) * resource.getpagesize() / (1024 * 1024)
    out = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True, check=True
    )
    return int(out.stdout.strip()) / 1024


def _child(step: str, rows: int, inputs_path: Path) -> dict[str, Any]:
    # Each analysis tool reads its row ceiling from the environment on every call.
    ceiling = PRODUCTION_MAX_FRAME_ROWS if step in PRODUCTION_CEILING_STEPS else UNBOUNDED_ROWS
    os.environ["MAX_FRAME_ROWS"] = str(ceiling)
    # The settings refuse a budget below two inputs at that ceiling. The step itself runs
    # against the unbounded budget installed below.
    os.environ["FRAME_ROWS_IN_FLIGHT_MAX"] = str(2 * ceiling)
    # This script's parent wrote the file a moment ago.
    inputs: _Inputs = pickle.loads(inputs_path.read_bytes())
    run = SETUPS[step][1](_store(_FakeClient(inputs.objects, inputs.metadata)), inputs)
    # Unbounded, so the step measures its work and never a refusal; its high-water mark
    # is the rows the step claimed.
    set_row_budget(RowBudget(max_rows=UNBOUNDED_ROWS))
    gc.collect()
    base_mb = _current_rss_mb()
    peak_mb = base_mb
    stop = threading.Event()

    def sample() -> None:
        nonlocal peak_mb
        while not stop.is_set():
            peak_mb = max(peak_mb, _current_rss_mb())
            stop.wait(SAMPLE_INTERVAL_S)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    start = time.perf_counter()
    facts = run()
    wall_s = time.perf_counter() - start
    stop.set()
    sampler.join()
    peak_mb = max(peak_mb, _current_rss_mb())
    return {
        "step": step,
        "rows": rows,
        "wall_s": round(wall_s, 3),
        "peak_rss_mb": round(_maxrss_mb(), 1),
        "base_rss_mb": round(base_mb, 1),
        "delta_mb": round(peak_mb - base_mb, 1),
        "claimed_rows": row_budget().high_water,
        **(facts or {}),
    }


def _run_in_subprocess(step: str, rows: int) -> dict[str, Any]:
    """Build the step's inputs here, then measure the step in a process that holds only
    the inputs' stored bytes, so the input build's own peak never reaches the step's."""
    with tempfile.TemporaryDirectory() as scratch:
        inputs_path = Path(scratch) / "inputs.pickle"
        inputs_path.write_bytes(pickle.dumps(SETUPS[step][0](rows)))
        proc = subprocess.run(
            [
                sys.executable,
                __file__,
                "--child",
                step,
                "--rows",
                str(rows),
                "--inputs",
                inputs_path,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    if proc.returncode != 0:
        tail = proc.stderr.strip().splitlines()[-1:] or ["no stderr"]
        return {"step": step, "rows": rows, "error": tail[0]}
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _print_table(results: list[dict[str, Any]]) -> None:
    header = (
        f"{'step':<24}{'rows':>9}{'wall_s':>9}{'peak_mb':>10}{'base_mb':>10}{'delta_mb':>10}"
        f"{'body_mb':>10}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        if "error" in r:
            print(f"{r['step']:<24}{r['rows']:>9}  ERROR: {r['error']}")
            continue
        body = f"{r['body_bytes'] / 2**20:>10.1f}" if "body_bytes" in r else ""
        print(
            f"{r['step']:<24}{r['rows']:>9}{r['wall_s']:>9.2f}"
            f"{r['peak_rss_mb']:>10.0f}{r['base_rss_mb']:>10.0f}{r['delta_mb']:>10.0f}{body}"
        )


def _print_budget(
    results: list[dict[str, Any]], *, memory_mib: int, idle_mib: int, headroom_mib: int
) -> bool:
    """Print the worst bytes a claimed row among the whole calls at the largest row
    count run and what it implies at `memory_mib`. Returns whether the configured
    `frame_rows_in_flight_max` fits under the implied value, and False when any
    whole-call step did not run at that row count or claimed no rows."""
    largest = max(r["rows"] for r in results)
    at_largest = {r["step"]: r for r in results if r["rows"] == largest and "error" not in r}
    unclaimed = [s for s in WHOLE_CALL_STEPS if not at_largest.get(s, {}).get("claimed_rows")]
    if unclaimed:
        print(
            f"Budget check refused: every whole-call step must run at {largest:,} rows and "
            f"claim frame rows, and {', '.join(unclaimed)} did not."
        )
        return False
    worst = max(
        (at_largest[step] for step in WHOLE_CALL_STEPS),
        key=lambda r: r["delta_mb"] / r["claimed_rows"],
    )
    bytes_a_row = worst["delta_mb"] * 2**20 / worst["claimed_rows"]
    usable_mib = memory_mib - idle_mib - headroom_mib
    implied = int(usable_mib * 2**20 / bytes_a_row)
    configured = BaseServiceSettings().frame_rows_in_flight_max
    max_frame_rows = DataSettings().max_frame_rows
    print(
        f"worst bytes a claimed row: {bytes_a_row:,.0f} ({worst['step']}, "
        f"{worst['delta_mb']} MiB over {worst['claimed_rows']:,} claimed rows)"
    )
    print(
        f"implied frame_rows_in_flight_max at {memory_mib} MiB: ({memory_mib} - {idle_mib} idle "
        f"- {headroom_mib} headroom) MiB / {bytes_a_row:,.0f} B = {implied:,} rows, "
        f"{implied / max_frame_rows:.1f} concurrent {max_frame_rows:,}-row calls"
    )
    fits = configured <= implied
    print(
        f"configured frame_rows_in_flight_max: {configured:,} "
        f"({'fits' if fits else 'OVER the implied value'})"
    )
    return fits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--rows", type=int, nargs="+", default=list(DEFAULT_ROWS))
    parser.add_argument("--steps", default=",".join(STEPS), help="comma-separated step names")
    parser.add_argument("--out", type=Path, help="write JSON results here")
    parser.add_argument(
        "--memory-mib",
        type=int,
        default=DEFAULT_MEMORY_MIB,
        help="instance memory the budget figures are computed for",
    )
    parser.add_argument("--idle-mib", type=int, default=DEFAULT_IDLE_MIB)
    parser.add_argument("--headroom-mib", type=int, default=DEFAULT_HEADROOM_MIB)
    parser.add_argument("--child", choices=STEPS, help=argparse.SUPPRESS)
    parser.add_argument("--inputs", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.child:
        print(json.dumps(_child(args.child, args.rows[0], args.inputs)))
        return 0

    steps = args.steps.split(",")
    unknown = [s for s in steps if s not in SETUPS]
    if unknown:
        parser.error(f"unknown steps: {unknown}")
    results = []
    for rows in args.rows:
        for step in steps:
            results.append(_run_in_subprocess(step, rows))
            _print_table(results[-1:])
    print()
    _print_table(results)
    print()
    fits = _print_budget(
        results,
        memory_mib=args.memory_mib,
        idle_mib=args.idle_mib,
        headroom_mib=args.headroom_mib,
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "platform": platform.platform(),
                    "python": platform.python_version(),
                    "results": results,
                },
                indent=2,
            )
            + "\n"
        )
    return 1 if any("error" in r for r in results) or not fits else 0


if __name__ == "__main__":
    sys.exit(main())

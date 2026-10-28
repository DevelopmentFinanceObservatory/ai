"""Frames on either side of dfo-apis' analysis runner, as the harness tests hold
and read them.

The runner's answers are replayed from `fixtures/runner_replies.json` by the
request each verb sends (`conftest.runner`), and a request names a root by the
dataset version dfo-apis holds, which a frame's mint record states. Two tests
that held different rows under one pair would send one request for both, so
`hold_root` names each frame's dataset version by its canonical bytes, as
analysis-mcp's own tests do (`runner_replay.frame_digest`).

`result_table` reads a verb's result back from dfo-apis, which stored it, as
the harness itself does (`harness.derived_frames`).
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

import pyarrow as pa
from common.materialisation import (
    FrameRef,
    MaterialisationStore,
    MintRecord,
    StoredFrameTrace,
)
from dfo_frame_grammar import Frame, canonical_frame_bytes
from harness.derived_frames import derived_file

# The mint options of a frame read with no enrichment columns.
_PLAIN_OPTIONS: Mapping[str, Any] = {"format": "parquet", "bundle": False, "layout": "frame"}


def dataset_version(ref: FrameRef, frame: Frame) -> FrameRef:
    """The dataset version `hold_root` states for `frame` held under `ref`: `ref`'s
    dataset at the version its canonical bytes name."""
    digest = hashlib.sha256(canonical_frame_bytes(frame)).hexdigest()[:32]
    return FrameRef(spec_id=ref.spec_id, content_version=f"cv_{digest}")


async def hold_root(
    store: MaterialisationStore,
    ref: FrameRef,
    *,
    frame: Frame,
    trace: StoredFrameTrace,
    writer: str = "tests",
) -> None:
    """Hold `frame` under `ref` as `get_data` holds a fetched frame, with the mint
    record naming its dataset version (`dataset_version`). The frame is held under
    that version too, which is where a recording of the runner's answers reads it
    (`runner_replay`)."""
    dataset = dataset_version(ref, frame)
    await store.put(ref, frame=frame, trace=trace, writer=writer)
    await store.put(dataset, frame=frame, trace=trace, writer=writer)
    await store.put_mint_record(
        MintRecord(frame=ref, dataset=dataset, options=dict(_PLAIN_OPTIONS))
    )


async def result_table(ref: Mapping[str, Any] | FrameRef) -> pa.Table:
    """The rows of the derived frame `ref` as dfo-apis serves them."""
    frame = (
        ref
        if isinstance(ref, FrameRef)
        else FrameRef(spec_id=ref["spec_id"], content_version=ref["content_version"])
    )
    return (await derived_file(frame)).frame().table

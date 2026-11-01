"""`Frame` <-> Parquet bytes: the on-disk shape a durable store persists the
`Frame` half of a stored `(Frame, StoredFrameTrace)` pair as, and the layout
apis writes a version's canonical frame in.

`frame_to_parquet_bytes` writes `frame.table` as it stands: every column of
`frame.columns` in order, with exactly `canonical_frame_schema(frame.columns)`
-- `value` a double, `unit` the grammar's `{ref, request, composed}` struct
with `composed` a composed unit's JCS text, `status` Arrow's null type, every
other column a string. Parquet keeps the Arrow schema in its footer, so
`pq.read_table` returns that same schema.

`frame_from_verified_parquet_bytes` reads it back for a caller that has
verified the bytes: a durable store after its checksum and envelope checks, or
data-mcp, which checks the frame against the `content_version` that names it.
It wraps the table with `Frame.from_table`, which refuses any schema other than
`canonical_frame_schema(columns)` or a `unit` layout a release stored (it widens
the two-member `struct<ref, request>`), and trusts the cells.

`frame_to_parquet_bytes`'s own `metadata` parameter, when given, becomes
the written file's file-level key-value metadata. Parquet only accepts
this at write time -- `Table.replace_schema_metadata` has to run before
`pq.write_table`, not after -- so a caller that wants its own metadata (a
durable store's own envelope, say) riding inside the object it writes
passes it here rather than layering it on in a second pass. The decoder
drops it from the frame's table.
"""

from __future__ import annotations

import io
from collections.abc import Mapping

import pyarrow.parquet as pq
from dfo_frame_grammar import Frame


def frame_to_parquet_bytes(frame: Frame, *, metadata: Mapping[str, bytes] | None = None) -> bytes:
    """`frame` as a whole Parquet file's bytes. See this module's own
    docstring for the column layout and for `metadata`.
    """
    table = frame.table
    if metadata:
        table = table.replace_schema_metadata(metadata)
    sink = io.BytesIO()
    pq.write_table(table, sink)
    return sink.getvalue()


def frame_from_verified_parquet_bytes(data: bytes) -> Frame:
    """The inverse of `frame_to_parquet_bytes`, for bytes the caller has
    verified.

    Raises:
        FrameError: `schema_violation` when the table's columns are not a
            derived frame column order or its schema is not
            `canonical_frame_schema` of them.
    """
    table = pq.read_table(io.BytesIO(data)).replace_schema_metadata(None)
    return Frame.from_table(columns=tuple(table.schema.names), table=table)

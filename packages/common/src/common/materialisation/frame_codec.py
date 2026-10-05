"""`Frame` <-> Parquet bytes: the on-disk shape a durable store persists the
`Frame` half of a stored `(Frame, StoredFrameTrace)` pair as.

`frame_to_parquet_bytes` writes `frame.table` as it stands: every column of
`frame.columns` in order, with exactly `canonical_frame_schema(frame.columns)`
-- `value` a double, `unit` the grammar's `{ref, request}` struct, `status`
Arrow's null type, every other column a string. Parquet keeps the Arrow
schema in its footer, so `pq.read_table` returns that same schema.

Two decoders read it back:

- `frame_from_verified_parquet_bytes` is for an object this service wrote
  with `frame_to_parquet_bytes` and has verified since (a durable store
  after its checksum and envelope checks). It wraps the table with
  `Frame.from_table`, which refuses any schema other than
  `canonical_frame_schema(columns)` but trusts the cells.
- `frame_from_parquet_bytes` is for anything else. It validates every cell
  through `build_frame_from_arrow`, and also reads the earlier layout: no
  `status` column, and `unit` as JSON text in a string column (`None`, a
  JSON string for a bare unit ref, or `UnitRequest.as_json()`'s object for a
  converted unit). Each distinct JSON text is parsed once and taken back to
  its rows as the canonical unit struct. The canonical layout never stores
  `unit` as a string, so a string `unit` column always means JSON text.

Role columns need no schema of their own to read back: every column name
that isn't one of frame-grammar's reserved columns is a role, the same
inference `data_mcp.frame_materialisation._frame_from_parquet_bytes` applies
to a fetched artifact's schema.

`frame_to_parquet_bytes`'s own `metadata` parameter, when given, becomes
the written file's file-level key-value metadata. Parquet only accepts
this at write time -- `Table.replace_schema_metadata` has to run before
`pq.write_table`, not after -- so a caller that wants its own metadata (a
durable store's own envelope, say) riding inside the object it writes
passes it here rather than layering it on in a second pass. Both decoders
drop it from the frame's table.
"""

from __future__ import annotations

import io
import json
from collections.abc import Mapping
from typing import cast

import pyarrow as pa
import pyarrow.parquet as pq
from dfo_frame_grammar import (
    FACET_IDENTITY_COLUMNS,
    STAT_VAR_COLUMN,
    STATUS_COLUMN,
    TIME_PERIOD_COLUMN,
    UNIT_COLUMN,
    VALUE_COLUMN,
    Frame,
    FrameError,
    build_frame_from_arrow,
    canonical_frame_schema,
    map_units,
)
from dfo_frame_grammar.grammar import validate_unit_request

_RESERVED_COLUMNS = frozenset(
    {STAT_VAR_COLUMN, TIME_PERIOD_COLUMN, VALUE_COLUMN, UNIT_COLUMN, STATUS_COLUMN}
    | set(FACET_IDENTITY_COLUMNS)
)
_UNIT_TYPE = canonical_frame_schema((UNIT_COLUMN,)).field(UNIT_COLUMN).type


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
    """The inverse of `frame_to_parquet_bytes`, for bytes the caller wrote
    with it and has verified since.

    Raises:
        FrameError: `schema_violation` when the table's columns are not a
            derived frame column order or its schema is not
            `canonical_frame_schema` of them.
    """
    table = pq.read_table(io.BytesIO(data)).replace_schema_metadata(None)
    return Frame.from_table(columns=tuple(table.schema.names), table=table)


def _unit_struct_from_json_text(text: object) -> object:
    """One earlier-layout `unit` cell as the canonical unit struct's value.
    A mapping is checked with the grammar's own unit-request rule first, so
    Arrow never drops an unknown member or fails on a mistyped one while
    building the struct.
    """
    if text is None:
        return None
    cell = json.loads(cast(str, text))
    if cell is None:
        return None
    if isinstance(cell, str):
        return {"ref": cell, "request": None}
    if isinstance(cell, dict):
        validate_unit_request(cast("Mapping[str, object]", cell))
        return {"ref": None, "request": cell}
    raise FrameError(
        "schema_violation", f"unit must be a string, mapping, or null, got {type(cell)!r}"
    )


def frame_from_parquet_bytes(data: bytes) -> Frame:
    """Any frame-shaped Parquet file's bytes, in either layout this module's
    docstring describes, validated into a `Frame` by `build_frame_from_arrow`.
    """
    table = pq.read_table(io.BytesIO(data)).replace_schema_metadata(None)
    status = table.schema.get_field_index(STATUS_COLUMN)
    if status >= 0 and pa.types.is_null(table.schema.field(status).type):
        table = table.remove_column(status)
    unit = table.schema.get_field_index(UNIT_COLUMN)
    if unit >= 0 and pa.types.is_string(table.schema.field(unit).type):
        table = table.set_column(
            unit,
            UNIT_COLUMN,
            map_units(table.column(unit), _unit_struct_from_json_text, result_type=_UNIT_TYPE),
        )
    role_keys = [name for name in table.schema.names if name not in _RESERVED_COLUMNS]
    return build_frame_from_arrow(table, role_keys=role_keys)

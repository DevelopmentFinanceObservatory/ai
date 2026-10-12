"""The `{"kind", "kind_version", "trace"}` codec registry a durable store
persists a `StoredFrameTrace` through.

A durable record outlives the deploy that wrote it -- the bucket's own
30-day lifecycle rule is its only expiry (`gcs_store.py`'s own docstring),
so a newer deploy must read last month's records and an older instance
must read what a newer one wrote during a rollout. Two things make that
possible: `kind` names which registered codec a record decodes through,
independent of any particular Python type surviving across a deploy
unchanged, and each codec's own `decode` is written against a lenient
shape (`extra="ignore"`, defaulted fields) rather than the live model it
reconstructs, so an unknown member a newer deploy added is dropped rather
than refused, and a field a newer deploy added is filled from its own
default rather than required.

`encode_trace` dispatches on `type(trace)`; `decode_trace` dispatches on
`record["kind"]`. `MaterialisationTrace` registers here, at import time,
since `common` owns that type; `analysis_mcp.trace` registers its own
`Trace` the same way, at its own import time, passing this module its own
class object -- `common` never imports `analysis_mcp`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict

from common.materialisation.trace import MaterialisationTrace


class TraceCodecUnknown(Exception):  # noqa: N818 -- named for the situation it names, not a generic "Error"
    """No codec is registered for `kind` -- raised by `decode_trace` for a
    record naming a kind nothing in this process has registered, and by
    `encode_trace` for a trace object whose own runtime type nothing has
    registered either.
    """

    def __init__(self, kind: str) -> None:
        self.kind = kind
        super().__init__(f"no trace codec registered for kind {kind!r}")


@dataclass(frozen=True, kw_only=True)
class _TraceCodec:
    kind: str
    version: int
    encode: Callable[[Any], dict[str, Any]]
    decode: Callable[[dict[str, Any]], Any]


_by_type: dict[type, _TraceCodec] = {}
_by_kind: dict[str, _TraceCodec] = {}


def register_trace_codec(
    trace_type: type[Any],
    *,
    kind: str,
    version: int,
    encode: Callable[[Any], dict[str, Any]],
    decode: Callable[[dict[str, Any]], Any],
) -> None:
    """Register `encode`/`decode` for every `trace_type` instance under
    `kind`. Re-registering the same `trace_type` (a module re-import)
    replaces its own entry rather than raising.
    """
    codec = _TraceCodec(kind=kind, version=version, encode=encode, decode=decode)
    _by_type[trace_type] = codec
    _by_kind[kind] = codec


def encode_trace(trace: Any) -> dict[str, Any]:
    """`{"kind", "kind_version", "trace"}` for `trace`, via the codec
    registered for `type(trace)`.
    """
    codec = _by_type.get(type(trace))
    if codec is None:
        raise TraceCodecUnknown(type(trace).__name__)
    return {"kind": codec.kind, "kind_version": codec.version, "trace": codec.encode(trace)}


def decode_trace(record: Mapping[str, Any]) -> Any:
    """The object `record` (a `{"kind", "kind_version", "trace"}` mapping)
    decodes to, via the codec registered for `record["kind"]`.
    """
    kind = record["kind"]
    codec = _by_kind.get(kind)
    if codec is None:
        raise TraceCodecUnknown(kind)
    return codec.decode(dict(record["trace"]))


class _MaterialisationTraceRecord(BaseModel):
    """Lenient decode shape for `MaterialisationTrace`: unknown members are
    ignored rather than refused."""

    model_config = ConfigDict(extra="ignore")

    upstream_trace: dict[str, Any]
    spec_id: str
    content_version: str
    options: dict[str, Any] | None = None


def _encode_materialisation_trace(trace: MaterialisationTrace) -> dict[str, Any]:
    """The trace's members, with `options` only when it is set, so a trace without
    them encodes as one did before the member existed. A reader that predates the
    member ignores it (`_MaterialisationTraceRecord`)."""
    encoded: dict[str, Any] = {
        "upstream_trace": trace.upstream_trace,
        "spec_id": trace.spec_id,
        "content_version": trace.content_version,
    }
    if trace.options is not None:
        encoded["options"] = trace.options
    return encoded


def _decode_materialisation_trace(record: dict[str, Any]) -> MaterialisationTrace:
    parsed = _MaterialisationTraceRecord.model_validate(record)
    return MaterialisationTrace(
        upstream_trace=parsed.upstream_trace,
        spec_id=parsed.spec_id,
        content_version=parsed.content_version,
        options=parsed.options,
    )


register_trace_codec(
    MaterialisationTrace,
    kind="materialisation",
    version=1,
    encode=_encode_materialisation_trace,
    decode=_decode_materialisation_trace,
)

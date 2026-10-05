"""The `{"kind", "kind_version", "trace"}` codec registry: dispatch by the
stored trace's own runtime type on encode, by the record's own `kind` on
decode, and tolerant decoding of a record carrying a member this process
never declared or missing one it did."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from common.materialisation import MaterialisationTrace
from common.materialisation.trace_codec import (
    TraceCodecUnknown,
    decode_trace,
    encode_trace,
    register_trace_codec,
)
from pydantic import BaseModel, ConfigDict


def test_materialisation_trace_round_trips_through_the_registry() -> None:
    trace = MaterialisationTrace(
        upstream_trace={"graph_revision": "rev1"},
        spec_id="spec_" + "a" * 32,
        content_version="cv_" + "b" * 32,
    )

    record = encode_trace(trace)

    assert record["kind"] == "materialisation"
    assert record["kind_version"] == 1
    assert decode_trace(record) == trace


def test_decode_trace_raises_for_an_unregistered_kind() -> None:
    with pytest.raises(TraceCodecUnknown):
        decode_trace({"kind": "no_such_kind", "kind_version": 1, "trace": {}})


def test_encode_trace_raises_for_an_unregistered_type() -> None:
    with pytest.raises(TraceCodecUnknown):
        encode_trace(object())


@dataclass(frozen=True, kw_only=True)
class _FakeTrace:
    upstream_trace: dict[str, Any]
    frame_content_version: str
    note: str = "default note"


class _FakeTraceRecord(BaseModel):
    """The lenient decode shape registered for `_FakeTrace` below --
    `extra="ignore"` and a defaulted `note`, exercising exactly the two
    tolerances a durable record surviving past its own writer's deploy
    needs."""

    model_config = ConfigDict(extra="ignore")

    upstream_trace: dict[str, Any]
    frame_content_version: str
    note: str = "default note"


def _encode_fake(trace: _FakeTrace) -> dict[str, Any]:
    return {
        "upstream_trace": trace.upstream_trace,
        "frame_content_version": trace.frame_content_version,
        "note": trace.note,
    }


def _decode_fake(record: dict[str, Any]) -> _FakeTrace:
    parsed = _FakeTraceRecord.model_validate(record)
    return _FakeTrace(
        upstream_trace=parsed.upstream_trace,
        frame_content_version=parsed.frame_content_version,
        note=parsed.note,
    )


register_trace_codec(
    _FakeTrace, kind="test_fake", version=1, encode=_encode_fake, decode=_decode_fake
)


def test_decode_ignores_a_member_this_process_never_declared() -> None:
    record = {
        "kind": "test_fake",
        "kind_version": 2,
        "trace": {
            "upstream_trace": {"graph_revision": "rev1"},
            "frame_content_version": "cv_" + "c" * 32,
            "note": "hello",
            "field_a_newer_deploy_added": "ignored",
        },
    }

    decoded = decode_trace(record)

    assert decoded == _FakeTrace(
        upstream_trace={"graph_revision": "rev1"},
        frame_content_version="cv_" + "c" * 32,
        note="hello",
    )


def test_decode_fills_a_missing_field_from_its_default() -> None:
    record = {
        "kind": "test_fake",
        "kind_version": 1,
        "trace": {
            "upstream_trace": {"graph_revision": "rev1"},
            "frame_content_version": "cv_" + "d" * 32,
        },
    }

    decoded = decode_trace(record)

    assert decoded.note == "default note"

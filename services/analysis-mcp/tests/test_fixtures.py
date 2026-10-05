"""The worked fixtures: a fully expanded op-spec and the trace it produces,
cross-checked against the identity functions rather than trusted verbatim.
"""

from __future__ import annotations

import json
from pathlib import Path

from analysis_mcp.identity import (
    canonicalize_op_spec,
    op_spec_hash_for,
    recompute_op_spec_hash,
    recompute_toolkit_content_version,
)
from analysis_mcp.op_spec import OpSpecAdapter
from analysis_mcp.trace import Trace

_FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> dict[str, object]:
    return json.loads((_FIXTURES / name).read_text())


def test_op_spec_fixture_parses() -> None:
    op_spec = OpSpecAdapter.validate_python(_load("op_spec.json"))
    assert op_spec.verb == "aggregate"


def test_trace_fixture_parses() -> None:
    trace = Trace.model_validate(_load("trace.json"))
    assert trace.execution_record.kind == "op_spec_canonical"


def test_trace_fixture_op_spec_hash_matches_the_op_spec_fixture() -> None:
    op_spec = OpSpecAdapter.validate_python(_load("op_spec.json"))
    trace = Trace.model_validate(_load("trace.json"))
    assert trace.op_spec_hash == op_spec_hash_for(op_spec)


def test_trace_fixture_execution_record_content_matches_the_canonical_op_spec_bytes() -> None:
    op_spec = OpSpecAdapter.validate_python(_load("op_spec.json"))
    trace = Trace.model_validate(_load("trace.json"))
    assert trace.execution_record.content.encode() == canonicalize_op_spec(op_spec)


def test_trace_fixture_execution_record_recomputes_the_op_spec_hash() -> None:
    trace = Trace.model_validate(_load("trace.json"))
    canonical_bytes = trace.execution_record.content.encode()
    assert recompute_op_spec_hash(canonical_bytes) == trace.op_spec_hash


def test_trace_fixture_recomputes_the_content_version() -> None:
    """The other identity half this module's own docstring promises: not
    just `op_spec_hash` (above), `content_version` too, from the trace's own
    stored `op_spec_hash`, `execution_identity`, and the op-spec fixture's
    own `frame_ref.content_version` as the sole input content_version --
    this is a single-input `aggregate` call, so there is exactly one."""
    op_spec = OpSpecAdapter.validate_python(_load("op_spec.json"))
    trace = Trace.model_validate(_load("trace.json"))
    recomputed = recompute_toolkit_content_version(
        op_spec_hash=trace.op_spec_hash,
        input_content_versions=[op_spec.frame_ref.content_version],
        execution_identity=trace.execution_identity,
    )
    assert recomputed == trace.content_version

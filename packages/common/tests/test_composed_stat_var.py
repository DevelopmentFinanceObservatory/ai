"""`composed_stat_vars_of`: the one read of a stored trace's composed
StatVars. The proof against a real `analysis_mcp.trace.Trace` lives in
analysis-mcp's own `test_trace.py`, since `common` cannot import
`analysis_mcp`.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest
from common.materialisation import ComposedStatVar, MaterialisationTrace, composed_stat_vars_of

_ENTRY = ComposedStatVar(
    stat_var="dcid:ops_" + "a" * 32,
    relation="sum",
    first=("dcid:x", "dcid:y"),
    second=("dcid:z",),
)


@dataclasses.dataclass(frozen=True, kw_only=True)
class _VerbTrace:
    upstream_trace: dict[str, Any]
    frame_content_version: str
    composed_stat_vars: tuple[ComposedStatVar, ...]


@dataclasses.dataclass(frozen=True, kw_only=True)
class _RenamedFieldTrace:
    upstream_trace: dict[str, Any]
    frame_content_version: str
    composed_statvars: tuple[ComposedStatVar, ...]


def test_a_trace_declaring_the_field_reads_as_its_entries() -> None:
    trace = _VerbTrace(
        upstream_trace={}, frame_content_version="cv_" + "b" * 32, composed_stat_vars=(_ENTRY,)
    )

    assert composed_stat_vars_of(trace) == (_ENTRY,)


def test_a_direct_materialisation_carries_none() -> None:
    trace = MaterialisationTrace(
        upstream_trace={}, spec_id="spec_" + "a" * 32, content_version="cv_" + "b" * 32
    )

    assert composed_stat_vars_of(trace) == ()


def test_a_verb_trace_without_the_field_is_refused_rather_than_read_as_empty() -> None:
    trace = _RenamedFieldTrace(
        upstream_trace={}, frame_content_version="cv_" + "b" * 32, composed_statvars=(_ENTRY,)
    )

    with pytest.raises(TypeError):
        composed_stat_vars_of(trace)

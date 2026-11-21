"""An execution identity for the tests that build a trace or render a digest:
the one analysis-mcp's in-process engine stamped on the frames it stored."""

from __future__ import annotations

from analysis_mcp.execution_identity import ExecutionIdentity

EXECUTION_IDENTITY = ExecutionIdentity(
    engine_version="0.9.1", frame_grammar_version="1.2.0", reduction_policy_version="0.2.0"
)

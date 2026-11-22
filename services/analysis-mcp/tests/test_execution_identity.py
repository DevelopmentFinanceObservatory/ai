"""`ExecutionIdentity`: the engine-binding third of a toolkit result's identity."""

from __future__ import annotations

import pytest
from analysis_mcp.execution_identity import ExecutionIdentity


def test_canonical_string_is_pipe_joined_in_field_order() -> None:
    identity = ExecutionIdentity(
        engine_version="0.1.0", frame_grammar_version="1.0.0", reduction_policy_version="0.2.0"
    )
    assert identity.canonical_string() == "0.1.0|1.0.0|0.2.0"


def test_execution_identity_is_frozen() -> None:
    identity = ExecutionIdentity(
        engine_version="0.1.0", frame_grammar_version="1.0.0", reduction_policy_version="0.2.0"
    )
    with pytest.raises(Exception, match="cannot assign"):
        identity.engine_version = "9.9.9"  # type: ignore[misc]

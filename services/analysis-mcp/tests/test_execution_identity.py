"""`ExecutionIdentity`: the engine-binding third of a toolkit result's identity."""

from __future__ import annotations

import re

import pytest
from analysis_mcp.execution_identity import (
    ENGINE_VERSION,
    REDUCTION_POLICY_VERSION,
    ExecutionIdentity,
    current_execution_identity,
)
from dfo_frame_grammar import FRAME_GRAMMAR_VERSION


def test_current_execution_identity_reads_the_frame_grammar_version_live() -> None:
    identity = current_execution_identity()
    assert identity.frame_grammar_version == FRAME_GRAMMAR_VERSION


def test_current_execution_identity_echoes_the_modules_own_version_constants() -> None:
    """Asserts shape (a dotted version string) and agreement with the module's
    own constants, not a hardcoded literal -- so this test never needs
    editing the next time `ENGINE_VERSION`/`REDUCTION_POLICY_VERSION` moves.
    """
    identity = current_execution_identity()
    assert identity.engine_version == ENGINE_VERSION
    assert re.fullmatch(r"\d+\.\d+\.\d+", identity.engine_version)
    assert identity.reduction_policy_version == REDUCTION_POLICY_VERSION
    assert re.fullmatch(r"\d+\.\d+\.\d+", identity.reduction_policy_version)


def test_canonical_string_is_pipe_joined_in_field_order() -> None:
    identity = ExecutionIdentity(
        engine_version="0.1.0", frame_grammar_version="1.0.0", reduction_policy_version="0.2.0"
    )
    assert identity.canonical_string() == "0.1.0|1.0.0|0.2.0"


def test_execution_identity_is_frozen() -> None:
    identity = current_execution_identity()
    with pytest.raises(Exception, match="cannot assign"):
        identity.engine_version = "9.9.9"  # type: ignore[misc]

"""Tests for `executor.outcomes`: the run's record in session state."""

from __future__ import annotations

import pytest
from common.materialisation import FrameRef
from executor.classify import ToolFailure
from executor.outcomes import (
    RUN_STATE_KEY,
    Accepted,
    AnchorResult,
    LeftOutSeries,
    RediscoveryNeeded,
    RefusalKind,
    Refused,
    StepRecord,
    load_outcome,
    save_outcome,
)
from planner.workflow import ToolId
from pydantic import ValidationError


def test_an_outcome_survives_a_round_trip_through_session_state() -> None:
    state: dict[str, str] = {}
    outcome = Refused(revision="r" * 64, refusal=RefusalKind.SURFACE_CHANGED, detail="changed")

    save_outcome(state, outcome)

    assert RUN_STATE_KEY in state
    assert load_outcome(state) == outcome


def test_a_rediscovery_needed_outcome_survives_a_round_trip_through_session_state() -> None:
    state: dict[str, str] = {}
    outcome = RediscoveryNeeded(
        revision="r" * 64,
        step_id="debt",
        tool=ToolId(server="data-mcp", name="get_data"),
        requirement_id="annual_debt_service",
        spec_id="spec_" + "a" * 32,
        refusal=ToolFailure(
            code="upstream_version_changed", message="refused", retry_mode="refresh_then_retry"
        ),
        completed=(),
    )

    save_outcome(state, outcome)

    assert load_outcome(state) == outcome


def test_a_rolling_runs_anchor_survives_a_round_trip() -> None:
    frame = FrameRef(spec_id="ops_" + "2" * 32, content_version="ocv_" + "3" * 32)
    growth = ToolId(server="analysis-mcp", name="growth")
    step = StepRecord(step_id="rolling", tool=growth, attempts=1, frame_ref=frame, digest="d")
    outcome = Accepted(
        revision="r" * 64,
        evidence=(),
        steps=(step,),
        results={"rolling": frame},
        anchors=(
            AnchorResult(
                anchor_key="rak_" + "7" * 32,
                period="2023",
                left_out=(LeftOutSeries(root=0, coordinates={"recipient": "country/UGA"}),),
            ),
        ),
    )
    state: dict[str, str] = {}

    save_outcome(state, outcome)

    assert load_outcome(state) == outcome


def test_a_rediscovery_must_name_what_to_rediscover() -> None:
    # An instruction to rediscover something unnamed is not an outcome.
    with pytest.raises(ValidationError):
        RediscoveryNeeded(
            revision="r" * 64,
            step_id="totals",
            tool=ToolId(server="analysis-mcp", name="aggregate"),
            requirement_id=None,  # type: ignore[arg-type] -- the rejection under test
            spec_id=None,  # type: ignore[arg-type] -- the rejection under test
            refusal=ToolFailure(
                code="upstream_version_changed", message="refused", retry_mode="refresh_then_retry"
            ),
            completed=(),
        )


def test_no_outcome_reads_as_none() -> None:
    assert load_outcome({}) is None

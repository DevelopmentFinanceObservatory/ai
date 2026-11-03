"""Tests for `executor.outcomes`: the run's record in session state."""

from __future__ import annotations

import json

import pytest
from common.materialisation import FrameRef
from executor.classify import ToolFailure
from executor.outcomes import (
    RUN_STATE_KEY,
    STOP_CAUSES,
    Accepted,
    AnchorResult,
    EvidenceRecord,
    LeftOutSeries,
    RediscoveryNeeded,
    RefusalKind,
    Refused,
    StepRecord,
    StopCause,
    failed_step,
    load_outcome,
    save_outcome,
    stop_cause,
)
from harness.sentences import FAILURE_CLASS, FailureClass
from planner.records import EvidenceIssue, EvidenceIssueCode, Period, Severity
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


def test_an_outcome_stored_before_a_gap_named_its_series_still_loads() -> None:
    """A run recorded before `EvidenceIssue.stat_var` and `entities` existed
    holds gaps without them; it loads with the gap naming no series. A gap
    naming none is stored the same way."""
    frame = FrameRef(spec_id="spec_" + "a" * 32, content_version="cv_" + "1" * 32)
    gap = EvidenceIssue(
        code=EvidenceIssueCode.WINDOW_SHORT,
        severity=Severity.WARNING,
        requirement_id="aid",
        detail="series ODA for ['country/DEU'] covers 2015-2021",
        found={"start": 2015, "end": 2021},
        required={"start": 2015, "end": 2023},
    )
    record = EvidenceRecord(
        requirement_id="aid",
        step_id="aid",
        frame_ref=frame,
        spec_id=frame.spec_id,
        stat_vars=("ODA",),
        window=Period(start=2015, end=2023),
        gaps=(gap,),
        citation="OECD",
    )
    outcome = Accepted(revision="r" * 64, evidence=(record,), steps=(), results={"aid": frame})
    state = {RUN_STATE_KEY: outcome.model_dump_json()}

    loaded = load_outcome(state)

    (stored,) = json.loads(state[RUN_STATE_KEY])["evidence"][0]["gaps"]
    assert set(stored) == {"code", "severity", "requirement_id", "detail", "found", "required"}
    assert loaded == outcome
    assert isinstance(loaded, Accepted)
    assert loaded.evidence[0].gaps[0].entities == ()


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


def test_every_kind_of_stop_has_a_cause() -> None:
    """A caller routes a stop by its cause (`stop_cause`), so a new kind must state one."""
    assert STOP_CAUSES.keys() == set(RefusalKind)


@pytest.mark.parametrize(
    ("retryable", "cause", "failure_class"),
    [
        (True, StopCause.TRANSIENT, FailureClass.TRANSIENT),
        (False, StopCause.HARNESS, FailureClass.SYSTEM_FAULT),
    ],
)
def test_a_transport_stop_is_transient_only_when_a_later_attempt_can_succeed(
    retryable: bool, cause: StopCause, failure_class: FailureClass
) -> None:
    stop = Refused(
        revision="r" * 64,
        refusal=RefusalKind.TRANSPORT_FAILED,
        detail="could not call 'aggregate'",
        step_id="totals",
        retryable=retryable,
    )

    assert stop_cause(stop) is cause
    assert FAILURE_CLASS[failed_step(stop)["code"]] is failure_class


def test_a_stop_that_is_not_a_transport_failure_ignores_the_retryable_mark() -> None:
    stop = Refused(
        revision="r" * 64, refusal=RefusalKind.EMPTY_ANSWER, detail="empty", retryable=False
    )

    assert stop_cause(stop) is StopCause.DATA
    assert failed_step(stop)["code"] == RefusalKind.EMPTY_ANSWER.value

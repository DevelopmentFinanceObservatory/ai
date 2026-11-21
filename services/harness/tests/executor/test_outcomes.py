"""Tests for `executor.outcomes`: the run's record in session state."""

from __future__ import annotations

import json
from typing import Any, get_args

import pytest
from analysis_mcp.tool_errors import ToolErrorCode as AnalysisCode
from analysis_mcp.tool_errors import ToolRefusal as AnalysisRefusal
from common.materialisation import FrameRef
from data_mcp.errors import UpstreamError
from data_mcp.tool_errors import CeilingKind, from_upstream_error
from data_mcp.tool_errors import ToolErrorCode as DataCode
from data_mcp.tool_errors import ToolRefusal as DataRefusal
from executor.classify import (
    _DATA_DETAILS,
    Disposition,
    ExecutorCode,
    ToolFailure,
    parse_tool_failure,
)
from executor.outcomes import (
    _DATA_STOP_REMEDY,
    RUN_STATE_KEY,
    STOP_CAUSES,
    Accepted,
    AnchorResult,
    EvidenceRecord,
    LeftOutSeries,
    MethodChoiceNeeded,
    RediscoveryNeeded,
    RefusalKind,
    Refused,
    Remedy,
    StepRecord,
    StopCause,
    failed_step,
    load_outcome,
    remedy,
    save_outcome,
    stop_cause,
    stop_remedy,
)
from harness.graph.plan_done import _user_sentence
from harness.sentences import FAILURE_CLASS, FailureClass, failure_sentence
from planner.records import EvidenceIssue, EvidenceIssueCode, Period, Severity
from planner.workflow import ToolId
from pydantic import ValidationError


def test_an_outcome_survives_a_round_trip_through_session_state() -> None:
    state: dict[str, str] = {}
    outcome = Refused(revision="r" * 64, refusal=RefusalKind.SURFACE_CHANGED, detail="changed")

    save_outcome(state, outcome)

    assert RUN_STATE_KEY in state
    assert load_outcome(state) == outcome
    # A refusal that names no source stores no `spec_id`, so an older release still reads it.
    assert "spec_id" not in outcome.model_dump(mode="json")
    named = outcome.model_copy(update={"spec_id": "spec_" + "a" * 32})
    assert named.model_dump(mode="json")["spec_id"] == "spec_" + "a" * 32


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
    ("retryable", "cause", "failure_class", "past"),
    [
        (True, StopCause.TRANSIENT, FailureClass.TRANSIENT, Remedy.WAIT),
        (False, StopCause.HARNESS, FailureClass.SYSTEM_FAULT, Remedy.SYSTEM),
    ],
)
def test_a_transport_stop_is_transient_only_when_a_later_attempt_can_succeed(
    retryable: bool, cause: StopCause, failure_class: FailureClass, past: Remedy
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
    assert remedy(failed_step(stop)) is past


def test_a_stop_that_is_not_a_transport_failure_ignores_the_retryable_mark() -> None:
    stop = Refused(
        revision="r" * 64, refusal=RefusalKind.EMPTY_ANSWER, detail="empty", retryable=False
    )

    assert stop_cause(stop) is StopCause.DATA
    assert failed_step(stop)["code"] == RefusalKind.EMPTY_ANSWER.value


# The run ledger's failure codes for a run, which no refusal carries
# (`executor.ledger`, `executor.completion`, `harness.run_ledger`), so none
# reaches `failed_step`.
_LEDGER_CODES = {
    ExecutorCode.CANCELLED,
    ExecutorCode.PUBLICATION_ABANDONED,
    ExecutorCode.COMPUTATION_ABANDONED,
    ExecutorCode.PUBLICATION_UNFINISHED,
    ExecutorCode.PUBLICATION_ERROR,
    ExecutorCode.RUN_NOT_STARTED,
    ExecutorCode.LEDGER_UNAVAILABLE,
}


def _tool_stop(failure: ToolFailure) -> Refused:
    return Refused(
        revision="r" * 64,
        refusal=RefusalKind.STEP_REFUSED,
        detail="refused",
        step_id="totals",
        payload=failure,
    )


def _served(
    refusal: DataRefusal | AnalysisRefusal,
) -> ToolFailure:
    """`refusal` as the executor parses it off the wire, with the server's own retry mode."""
    return ToolFailure(
        code=refusal.code.value,
        message=refusal.message,
        retry_mode=refusal.retry_mode.value,
        fix=refusal.fix,
    )


# data-mcp's refusal of a definition the platform would not register, which
# shares `upstream_error` with an upstream service's own failures.
_SPEC_NOT_REGISTERED = DataRefusal.spec_not_registered(
    tool="get_data", spec_id="spec_" + "0" * 32, upstream_code="spec_invalid", detail="bad"
)


def _upstream(status: int, *, retryable: bool = False) -> DataRefusal:
    """data-mcp's refusal for a dfo-apis answer of `status` under a code it does not map."""
    failure = UpstreamError(status=status, code="unmapped_code", detail="d", retryable=retryable)
    return from_upstream_error(failure, tool="get_data")


def _every_stop() -> list[Refused]:
    """A stop for every code a stopped run can fail with: each server code under
    each retry mode its server answers it with, each `ceiling_kind` and each
    data `detail`; each refusal the executor mints; each stop kind, retryable
    or not; and each evidence finding."""
    fixes: list[dict[str, Any]] = [
        {},
        {"retryable": True},
        {"issues": [{"alternatives": ["x"]}]},
        *({"ceiling_kind": kind} for kind in get_args(CeilingKind)),
        *({"detail": detail} for detail in _DATA_DETAILS),
    ]
    stops = [
        _tool_stop(_served(server(code=code, message="m", fix=fix, retry_after_s=wait)))
        for server, codes in ((DataRefusal, DataCode), (AnalysisRefusal, AnalysisCode))
        for code in codes
        for fix in fixes
        for wait in (None, 1.0)
    ]
    stops.append(_tool_stop(_served(_SPEC_NOT_REGISTERED)))
    stops.append(_tool_stop(parse_tool_failure({"retry_mode": "do_not_retry"})))
    stops += [
        _tool_stop(ToolFailure(code=code.value, message="m", retry_mode="do_not_retry"))
        for code in set(ExecutorCode) - _LEDGER_CODES
    ]
    stops += [
        Refused(revision="r" * 64, refusal=kind, detail="d", retryable=retryable)
        for kind in RefusalKind
        for retryable in (True, False)
    ]
    stops += [
        Refused(
            revision="r" * 64,
            refusal=RefusalKind.REQUIREMENT_UNMET,
            detail="d",
            issues=(
                EvidenceIssue(
                    code=code, severity=Severity.ERROR, requirement_id="totals", detail="x"
                ),
            ),
        )
        for code in EvidenceIssueCode
    ]
    return stops


def test_the_data_stop_map_places_exactly_the_codes_no_cause_or_disposition_decides() -> None:
    """A data stop no method choice or rediscovery settles reads its code's
    place, so a new code fails here until it is placed on purpose."""
    reached = {
        failed["code"]
        for failed in map(failed_step, _every_stop())
        if failed["cause"] == StopCause.DATA
        and failed.get("disposition") not in (Disposition.METHOD_CHOICE, Disposition.REDISCOVER)
    }

    assert reached == _DATA_STOP_REMEDY.keys(), {
        "unplaced": sorted(reached - _DATA_STOP_REMEDY.keys()),
        "stale": sorted(_DATA_STOP_REMEDY.keys() - reached),
    }


def _issue_stop(code: EvidenceIssueCode) -> Refused:
    return Refused(
        revision="r" * 64,
        refusal=RefusalKind.REQUIREMENT_UNMET,
        detail="unmet",
        step_id="totals",
        issues=(
            EvidenceIssue(
                code=code, severity=Severity.ERROR, requirement_id="totals", detail="found"
            ),
        ),
    )


_GET_DATA = ToolId(server="data-mcp", name="get_data")


@pytest.mark.parametrize(
    ("stop", "past"),
    [
        # A refusal that names a wait is waited out.
        pytest.param(
            _tool_stop(
                _served(
                    DataRefusal(
                        code=DataCode.SERVICE_UNAVAILABLE, message="m", fix={}, retry_after_s=1.0
                    )
                )
            ),
            Remedy.WAIT,
            id="transient-refusal",
        ),
        # The harness's and the services' own faults are the system's, but for an
        # approval a fresh plan renews.
        pytest.param(
            _tool_stop(
                _served(AnalysisRefusal(code=AnalysisCode.PLACE_LEVELS_MIXED, message="m", fix={}))
            ),
            Remedy.SYSTEM,
            id="compiler-provable",
        ),
        # An upstream service that failed of itself: a 5xx, or an answer that
        # could not be used.
        pytest.param(_tool_stop(_served(_upstream(502))), Remedy.SYSTEM, id="upstream-5xx"),
        pytest.param(
            _tool_stop(_served(DataRefusal.upstream_response_unusable(detail="bad body"))),
            Remedy.SYSTEM,
            id="upstream-response-unusable",
        ),
        # A refusal that arrived with no code.
        pytest.param(
            _tool_stop(parse_tool_failure({"retry_mode": "do_not_retry", "message": "m"})),
            Remedy.SYSTEM,
            id="codeless-refusal",
        ),
        # A stored frame that failed its integrity check: a narrower scope
        # mints another.
        pytest.param(
            _tool_stop(
                _served(
                    DataRefusal.frame_integrity_failed(
                        spec_id="spec_" + "0" * 32, content_version="v1", reason="digest"
                    )
                )
            ),
            Remedy.REPLAN,
            id="frame-integrity-failed",
        ),
        # A definition the platform refused, or a refusal of the request under
        # a code data-mcp does not map: another spec or request may avoid either.
        pytest.param(
            _tool_stop(_served(_SPEC_NOT_REGISTERED)), Remedy.REPLAN, id="spec-not-registered"
        ),
        pytest.param(_tool_stop(_served(_upstream(422))), Remedy.REPLAN, id="upstream-4xx"),
        # A service failure that names a wait.
        pytest.param(
            _tool_stop(_served(_upstream(503, retryable=True))),
            Remedy.WAIT,
            id="upstream-5xx-wait",
        ),
        pytest.param(
            Refused(revision="r" * 64, refusal=RefusalKind.BINDING_INVALID, detail="d"),
            Remedy.SYSTEM,
            id="binding-invalid",
        ),
        pytest.param(
            Refused(revision="r" * 64, refusal=RefusalKind.STALE_REVISION, detail="d"),
            Remedy.REPLAN,
            id="stale-revision",
        ),
        # A method choice or a source to discover again is re-planned.
        pytest.param(
            MethodChoiceNeeded(
                revision="r" * 64,
                step_id="totals",
                tool=_GET_DATA,
                code=DataCode.OVER_CEILING.value,
                candidates=None,
                refusal=_served(
                    DataRefusal(
                        code=DataCode.OVER_CEILING, message="m", fix={"ceiling_kind": "rows"}
                    )
                ),
                completed=(),
            ),
            Remedy.REPLAN,
            id="rows-ceiling",
        ),
        pytest.param(
            RediscoveryNeeded(
                revision="r" * 64,
                step_id="totals",
                tool=_GET_DATA,
                requirement_id="totals",
                spec_id="spec_" + "0" * 32,
                refusal=_served(
                    DataRefusal(code=DataCode.UPSTREAM_VERSION_CHANGED, message="m", fix={})
                ),
                completed=(),
            ),
            Remedy.REPLAN,
            id="rediscovery",
        ),
        # Any other data stop reads its code's place.
        pytest.param(
            _tool_stop(
                _served(
                    DataRefusal(
                        code=DataCode.OVER_CEILING,
                        message="m",
                        fix={"ceiling_kind": "member_cardinality"},
                    )
                )
            ),
            Remedy.USER,
            id="member-cardinality-ceiling",
        ),
        pytest.param(
            _tool_stop(
                _served(DataRefusal(code=DataCode.FACET_NOT_AVAILABLE, message="m", fix={}))
            ),
            Remedy.REPLAN,
            id="facet-not-available",
        ),
        pytest.param(
            _tool_stop(
                ToolFailure(
                    code=ExecutorCode.ROLLING_ANCHOR_BEHIND_PLAN,
                    message="m",
                    retry_mode="do_not_retry",
                    fix={"anchor": "2022", "planned_anchor": "2023"},
                )
            ),
            Remedy.USER,
            id="anchor-behind-plan",
        ),
        pytest.param(_issue_stop(EvidenceIssueCode.UNIT_MISMATCH), Remedy.REPLAN, id="unit"),
        pytest.param(_issue_stop(EvidenceIssueCode.WINDOW_SHORT), Remedy.USER, id="window"),
    ],
)
def test_a_stopped_runs_remedy_follows_its_cause_its_disposition_and_its_code(
    stop: MethodChoiceNeeded | RediscoveryNeeded | Refused, past: Remedy
) -> None:
    assert remedy(failed_step(stop)) is past


def test_a_failed_step_with_no_wave_remedy_is_the_systems() -> None:
    """A failed step with no wave remedy is the system's, and so is a stop with
    no cause: neither gives a second plan or tells the user the question was
    at fault."""
    stored = failed_step(_issue_stop(EvidenceIssueCode.UNIT_MISMATCH))
    assert remedy(stored) is Remedy.REPLAN
    del stored["wave_remedy"]
    assert remedy(stored) is Remedy.SYSTEM
    del stored["cause"]
    assert stop_remedy(stored) is Remedy.SYSTEM


@pytest.mark.parametrize(
    ("primary", "sibling"),
    [
        (EvidenceIssueCode.WINDOW_SHORT, EvidenceIssueCode.UNIT_MISMATCH),
        (EvidenceIssueCode.UNIT_MISMATCH, EvidenceIssueCode.WINDOW_SHORT),
    ],
    ids=["user-primary", "replan-primary"],
)
def test_a_waves_remedy_is_its_most_severe_stops(
    primary: EvidenceIssueCode, sibling: EvidenceIssueCode
) -> None:
    """A new plan gets past a wave only when it gets past every stop in it, so a
    stop the user has to accept outweighs one another plan avoids, whichever
    the run raised, and the user is told of that stop."""
    wave = _issue_stop(primary).model_copy(update={"also_stopped": (_issue_stop(sibling),)})
    failed = failed_step(wave)

    assert {stop_remedy(failed_step(stop)) for stop in (wave, *wave.also_stopped)} == {
        Remedy.USER,
        Remedy.REPLAN,
    }
    assert remedy(failed) is Remedy.USER
    assert failed["code"] == primary.value
    assert _user_sentence(failed) == failure_sentence(EvidenceIssueCode.WINDOW_SHORT.value)

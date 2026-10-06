"""CI's share of the orchestrator engine's eval set (`evals/studio`, `--mode orchestrator`).

The cases run the real graph and planner seam under a real ADK runner, with
every model scripted, and check what each model was shown. Three things are
held here: the cases report as marked (a pass, or an expected failure that
still fails), each check fails when the property it states is broken, and the
case file can fail, as `test_studio_conformance.py` holds the classic one.

`evals/` sits outside the workspace, so its modules are loaded by file path.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

STUDIO_EVAL_DIR = Path(__file__).resolve().parents[3] / "evals" / "studio"
CASES_PATH = STUDIO_EVAL_DIR / "harness-studio-orchestrator-v1.cases.json"


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, STUDIO_EVAL_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)
    return module


checks = _load("checks")


def _suite() -> dict[str, Any]:
    return json.loads(CASES_PATH.read_text())


async def test_orchestrator_cases_report_as_marked() -> None:
    _load("orchestrator_scripted")
    run = _load("run")

    results, passed = await run.run_suite(mode="orchestrator", url="", judge=False)

    assert results, "no orchestrator case ran"
    assert [r["id"] for r in results] == [c["id"] for c in run.load_cases("orchestrator")]
    # An XPASS is a gap that closed: the marker is stale, so CI fails on it.
    failures = {
        r["id"]: (r["status"], r["contract_violations"] + r["expectation_violations"])
        for r in results
        if r["status"] not in ("PASS", "XFAIL")
    }
    assert passed and failures == {}


def _request(agent: str, *texts: str, system: str = "") -> dict[str, Any]:
    parts = [{"text": text} for text in texts]
    return {
        "kind": "request",
        "agent": agent,
        "system": system,
        "contents": [{"role": "user", "parts": parts}],
    }


def _step(spec_id: str | None = None) -> dict[str, Any]:
    arguments = {"spec_id": spec_id} if spec_id else {}
    return {
        "kind": "call",
        "layer": "step",
        "tool": "get_data",
        "outcome": "ok",
        "arguments": arguments,
    }


def _search_answer(**reading: Any) -> dict[str, Any]:
    response = {"name": "search", "response": {"kind": "resolution", "parts": [reading]}}
    return {
        "kind": "request",
        "agent": "orchestrator",
        "system": "",
        "contents": [{"role": "user", "parts": [{"function_response": response}]}],
    }


def _closing(text: str) -> list[dict[str, Any]]:
    return [{"event": "message.complete", "seq": 0, "text": text}, {"event": "done", "seq": 1}]


def _plan_call(*asks: str) -> dict[str, Any]:
    call = {"function_call": {"name": "plan", "args": {"brief": "b", "asks": list(asks)}}}
    return {
        "kind": "request",
        "agent": "orchestrator",
        "system": "",
        "contents": [{"role": "model", "parts": [call]}],
    }


def _card(*asks: dict[str, Any]) -> list[list[dict[str, Any]]]:
    return [[{"event": "plan.proposed", "steps": [{"id": "s1"}], "asks": list(asks)}]]


def _hand_back(*asks: dict[str, Any]) -> dict[str, Any]:
    return _request("orchestrator", "The approved plan ran.\n" + json.dumps({"asks": list(asks)}))


ANSWERED = {"ask": "A", "answered_by": ["s1"]}
DECLINED = {"ask": "B", "declined": "the data found cannot answer this as asked"}
RAN = {"ask": "A", "answered_by": [{"step": "s1", "status": "data"}]}
SPEC = "spec_" + "1" * 32
READING = {"spec_id": SPEC, "window": {"start": 1990, "end": 2020}, "warnings": [{"code": "w_one"}]}
FENCED = f"[dfo_analyst_graph] said:\n{checks.RELAY_MARKER}\nbody"
CLOSING = 'The series rose. Would you like "Uganda" too?'
TURN_ONE = [{"kind": "turn", "index": 1}, _request("orchestrator", "first")]


@pytest.mark.parametrize(
    ("name", "turn_frames", "log", "violated"),
    [
        (
            "approval_payload_absent_after_approval",
            [],
            [_step(), _request("orchestrator", "ok")],
            False,
        ),
        (
            "approval_payload_absent_after_approval",
            [],
            [_step(), _request("orchestrator", f"called tool `{checks.REQUEST_INPUT_NAME}`")],
            True,
        ),
        # A pause before the first step is the approval itself, and is not held against it.
        (
            "approval_payload_absent_after_approval",
            [],
            [_request("planner", checks.REQUEST_INPUT_NAME), _step()],
            False,
        ),
        # No step ran, so nothing was approved: a case that proves nothing fails.
        ("approval_payload_absent_after_approval", [], [_request("orchestrator", "ok")], True),
        ("no_relay_marker", [], [_request("orchestrator", "plain")], False),
        ("no_relay_marker", [], [_request("orchestrator", FENCED)], True),
        ("no_relay_marker", [], [_request("planner", "x", system=checks.RELAY_MARKER)], False),
        ("planner_sees_no_relay_marker", [], [_request("orchestrator", FENCED)], False),
        ("planner_sees_no_relay_marker", [], [_request("planner", FENCED)], True),
        (
            "planner_first_message_carries_search_evidence",
            [],
            [_search_answer(**READING), _request("planner", f"{SPEC} w_one 1990 2020")],
            False,
        ),
        (
            "planner_first_message_carries_search_evidence",
            [],
            [_search_answer(**READING), _request("planner", f"{SPEC} w_one 1990")],
            True,
        ),
        # Only the first planner request is held to it.
        (
            "planner_first_message_carries_search_evidence",
            [],
            [_search_answer(**READING), _request("planner", "nothing"), _request("planner", SPEC)],
            True,
        ),
        ("planner_first_message_carries_search_evidence", [], [_request("planner", "x")], True),
        (
            "previous_final_message_in_view",
            [_closing(CLOSING), _closing("again")],
            [*TURN_ONE, {"kind": "turn", "index": 2}, _request("orchestrator", "x", CLOSING)],
            False,
        ),
        # The digest is JSON in the system instruction, so the quotes arrive escaped.
        (
            "previous_final_message_in_view",
            [_closing(CLOSING), _closing("again")],
            [
                *TURN_ONE,
                {"kind": "turn", "index": 2},
                _request("orchestrator", "x", system=json.dumps({"assistant": CLOSING})),
            ],
            False,
        ),
        (
            "previous_final_message_in_view",
            [_closing(CLOSING), _closing("again")],
            [*TURN_ONE, {"kind": "turn", "index": 2}, _request("orchestrator", CLOSING[:20])],
            True,
        ),
        (
            "earlier_plan_in_view",
            [],
            [*TURN_ONE, _step(SPEC), {"kind": "turn", "index": 2}, _request("orchestrator", SPEC)],
            False,
        ),
        (
            "earlier_plan_in_view",
            [],
            [*TURN_ONE, _step(SPEC), {"kind": "turn", "index": 2}, _request("orchestrator", "x")],
            True,
        ),
        # A source retrieved in the last turn is not held against that turn.
        (
            "earlier_plan_in_view",
            [],
            [*TURN_ONE, {"kind": "turn", "index": 2}, _request("orchestrator", "x"), _step(SPEC)],
            True,
        ),
        (
            "every_ask_accounted_for",
            _card(ANSWERED, DECLINED),
            [_plan_call("A", "B"), _hand_back(RAN, DECLINED)],
            False,
        ),
        # An ask answered in part names the part it does not compute.
        (
            "every_ask_accounted_for",
            _card({**ANSWERED, **DECLINED, "ask": "A", "not_computed": "its rank"}),
            [_plan_call("A"), _hand_back(RAN)],
            False,
        ),
        (
            "every_ask_accounted_for",
            _card({**ANSWERED, **DECLINED, "ask": "A"}),
            [_plan_call("A"), _hand_back(RAN)],
            True,
        ),
        # An ask the card leaves out, or answers with a step it does not show.
        (
            "every_ask_accounted_for",
            _card(ANSWERED),
            [_plan_call("A", "B"), _hand_back(RAN, DECLINED)],
            True,
        ),
        (
            "every_ask_accounted_for",
            _card({"ask": "A", "answered_by": ["s9"]}),
            [_plan_call("A"), _hand_back(RAN)],
            True,
        ),
        # An ask the hand-back reports without the status of its output.
        (
            "every_ask_accounted_for",
            _card(ANSWERED),
            [_plan_call("A"), _hand_back({"ask": "A", "answered_by": [{"step": "s1"}]})],
            True,
        ),
        ("every_ask_accounted_for", _card(), [_plan_call(), _hand_back()], True),
    ],
)
def test_a_request_check_fails_exactly_when_its_property_is_broken(
    name: str, turn_frames: list[list[dict]], log: list[dict], violated: bool
) -> None:
    evidence = checks.Evidence(turn_frames=turn_frames, log=log)

    assert bool(checks.CASE_CHECKS[name](evidence)) is violated


def test_a_request_check_without_the_request_log_is_a_violation_not_a_pass() -> None:
    for name in sorted(checks.REQUEST_CHECKS):
        violations = checks.CASE_CHECKS[name](checks.Evidence(turn_frames=[]))

        assert len(violations) == 1, name
        assert "request log" in violations[0]


def _step_frame(status: str, detail: str) -> dict[str, Any]:
    return {
        "event": "activity.step",
        "label": "Fetch",
        "tool": "get_data",
        "status": status,
        "detail": detail,
    }


def test_a_failed_step_that_fails_again_with_the_same_outcome_is_a_repeat() -> None:
    failed = _step_frame("failed", "refused")

    def evidence(*turns: list[dict]) -> Any:
        return checks.Evidence(turn_frames=list(turns))

    check = checks.CASE_CHECKS["no_repeated_failed_step"]
    assert check(evidence([failed], [failed])) != []
    assert check(evidence([failed, _step_frame("failed", "refused differently")])) == []
    assert check(evidence([_step_frame("complete", "ok"), _step_frame("complete", "ok")])) == []


def test_every_orchestrator_case_can_fail() -> None:
    run = _load("run")
    suite = _suite()
    cases = suite["cases"]

    ids = [case["id"] for case in cases]
    assert len(ids) == len(set(ids))
    rubric_ids = [r["id"] for case in cases for r in case.get("rubrics", [])]
    assert len(rubric_ids) == len(set(rubric_ids))

    for case in cases:
        label = case["id"]
        assert len(case.get("xfail", "x" * 21)) > 20, f"{label}: xfail without a reason"
        assert not checks.check_names(case.get("checks", ["no_relay_marker"]), label=label)
        assert (
            case.get("rubrics") or case.get("checks") or any(t.get("expect") for t in case["turns"])
        )
        assert [
            v
            for index, turn in enumerate(case["turns"], start=1)
            for v in run.check_expect_keys(turn, label=f"{label} turn {index}")
        ] == []
        scenario = case.get("scenario")
        if case["modes"] == ["orchestrator"]:
            assert scenario in suite["scenarios"] or isinstance(scenario, dict), label
        else:
            assert case["modes"] == ["orchestrator-live"], label
            assert not set(case.get("checks", [])) & checks.REQUEST_CHECKS, (
                f"{label}: a live run records no model requests to check"
            )


def _judged_cases(count: int) -> list[str]:
    return [c["id"] for c in _suite()["cases"] if c.get("rubrics")][:count]


async def test_a_judge_that_stays_unavailable_marks_the_case_and_the_run_goes_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _load("run")
    first, second = _judged_cases(2)

    async def outcome(client: Any, case: dict[str, Any]) -> dict[str, Any]:
        return {"contract": [], "expectation": [], "turn_frames": []}

    async def judge(case: dict[str, Any], turn_frames: Any, config: Any) -> list[dict[str, Any]]:
        if case["id"] == first:
            raise run.JudgeError("503 UNAVAILABLE")
        return [{"rubric_id": r["id"], "score": 1.0, "rationales": []} for r in case["rubrics"]]

    monkeypatch.setattr(run, "run_case", outcome)
    monkeypatch.setattr(run, "_frame_checks", lambda case, outcome: [])
    monkeypatch.setattr(run, "judge_case", judge)

    results, passed = await run.run_suite(
        mode="orchestrator-live", url="http://unused", judge=True, only=[first, second]
    )

    by_id = {r["id"]: r for r in results}
    assert by_id[first]["status"] == "JUDGE_ERROR"
    assert by_id[first]["judge_error"] == "503 UNAVAILABLE"
    assert "rubric_scores" in by_id[second]
    assert not passed


def _failing_judge(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> tuple[ModuleType, list[Any], dict[str, Any], list[list[Any]]]:
    """`run` with a judge client whose every call raises `error`, the
    `http_options` each client was built with, and a judged case to score."""
    from google import genai  # noqa: PLC0415

    run = _load("run")
    built: list[Any] = []

    class Models:
        async def generate_content(self, **kwargs: Any) -> Any:
            raise error

    class FakeClient:
        def __init__(self, *, http_options: Any) -> None:
            built.append(http_options)
            self.aio = type("Aio", (), {"models": Models()})()

    monkeypatch.setattr(genai, "Client", FakeClient)
    case = next(c for c in _suite()["cases"] if c.get("rubrics"))
    turns = [[] for turn in case["turns"] if "message" in turn]
    return run, built, case, turns


@pytest.mark.parametrize("code", [503, 429])
async def test_the_judge_retries_a_transient_error_and_raises_judge_error_when_it_persists(
    code: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    from google.genai import _api_client, errors  # noqa: PLC0415

    kind = errors.ServerError if code >= 500 else errors.ClientError
    run, built, case, turns = _failing_judge(monkeypatch, kind(code, {"error": {"code": code}}))

    with pytest.raises(run.JudgeError, match=str(code)):
        await run.judge_case(case, turns, {"judge_model": "m", "num_samples": 1})

    retry = built[0].retry_options
    assert code in (retry.http_status_codes or _api_client._RETRY_HTTP_STATUS_CODES)
    assert retry.attempts > 1


async def test_a_judge_error_that_is_not_transient_still_stops_the_suite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from google.genai import errors  # noqa: PLC0415

    bad_request = errors.ClientError(400, {"error": {"code": 400, "status": "INVALID_ARGUMENT"}})
    run, _, case, turns = _failing_judge(monkeypatch, bad_request)

    with pytest.raises(errors.ClientError):
        await run.judge_case(case, turns, {"judge_model": "m", "num_samples": 1})

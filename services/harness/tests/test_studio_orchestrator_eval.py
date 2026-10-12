"""CI's share of the studio eval set (`evals/studio`, `--mode scripted`).

The cases run the real workflow and planner seam under a real ADK runner, with
every model scripted, and check what each model was shown. Three things are
held here: the cases report as marked (a pass, or an expected failure that
still fails), each check and each stream invariant fails when the property it
states is broken, and the case file can fail.

`evals/` sits outside the workspace, so its modules are loaded by file path.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from google.genai import types

from .orchestrator_fakes import (
    CannedSearch,
    chart,
    definite_envelope,
    fetch_call,
    frame_for,
    make_charts_call,
    orchestrator_app,
    search_call,
    stream,
    studio_client,
    turn_body,
)

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


async def test_scripted_cases_report_as_marked() -> None:
    _load("orchestrator_scripted")
    run = _load("run")

    results, passed = await run.run_suite(mode="scripted", url="", judge=False)

    assert results, "no scripted case ran"
    assert [r["id"] for r in results] == [c["id"] for c in run.load_cases("scripted")]
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


def test_every_case_can_fail() -> None:
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
        if case["modes"] == ["scripted"]:
            assert scenario in suite["scenarios"] or isinstance(scenario, dict), label
        else:
            assert case["modes"] == ["live"], label
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

    async def judge(
        case: dict[str, Any], turn_frames: Any, config: Any, *, answers: Any = None
    ) -> list[dict[str, Any]]:
        if case["id"] == first:
            raise run.JudgeError("503 UNAVAILABLE")
        return [{"rubric_id": r["id"], "score": 1.0, "rationales": []} for r in case["rubrics"]]

    monkeypatch.setattr(run, "run_case", outcome)
    monkeypatch.setattr(run, "_frame_checks", lambda case, outcome: [])
    monkeypatch.setattr(run, "judge_case", judge)

    results, passed = await run.run_suite(
        mode="live", url="http://unused", judge=True, only=[first, second]
    )

    by_id = {r["id"]: r for r in results}
    assert by_id[first]["status"] == "JUDGE_ERROR"
    assert by_id[first]["judge_error"] == "503 UNAVAILABLE"
    assert "rubric_scores" in by_id[second]
    assert not passed


class _ScriptedStudio:
    """A client whose turn request pauses on one question and whose resume ends the turn."""

    def __init__(self) -> None:
        self.frames = {
            "/api/studio/turn": [
                {
                    "event": "elicitation.request",
                    "interrupt_id": "int_1",
                    "question": "How should aid and debt be combined?",
                    "options": [
                        {"id": "opt_net", "label": "Net ODA plus total debt", "description": "d"},
                        {"id": "opt_ppg", "label": "Net ODA plus PPG debt"},
                    ],
                }
            ],
            "/api/studio/interrupt": [{"event": "done"}],
        }

    def stream(self, method: str, path: str, **request: Any) -> Any:
        lines = [f"data: {json.dumps(frame)}" for frame in self.frames[path]]

        class Response:
            status_code = 200

            async def aiter_lines(self) -> Any:
                for line in lines:
                    yield line

        @asynccontextmanager
        async def opened() -> Any:
            yield Response()

        return opened()


@pytest.mark.parametrize(
    ("turn_answer", "shown"),
    [
        ({}, "Net ODA plus total debt"),
        ({"elicitation_text": "Keep them separate"}, "Keep them separate"),
    ],
)
async def test_the_judge_digest_carries_the_question_asked_and_the_answer_sent(
    turn_answer: dict[str, str], shown: str
) -> None:
    run = _load("run")
    turn = {"message": "Help me combine aid and debt", **turn_answer}

    frames, _, _, answers = await run._run_message_turn(_ScriptedStudio(), "conv", turn, [])

    (asked,) = run._turn_digest(frames, answers)["elicitations"]
    assert asked["question"] == "How should aid and debt be combined?"
    assert [o["label"] for o in asked["options"]] == [
        "Net ODA plus total debt",
        "Net ODA plus PPG debt",
    ]
    assert asked["answer"] == shown
    assert run._turn_digest(frames)["elicitations"][0]["answer"] is None


def test_the_judge_digest_orders_entries_across_lists() -> None:
    """A figure drawn before the question reads as drawn before it, though
    components and questions sit in separate lists."""
    run = _load("run")
    frames = [
        {"event": "component.created", "type": "line_chart", "fact_summary": {}},
        {"event": "elicitation.request", "interrupt_id": "int_1", "question": "Combine how?"},
        {"event": "message.complete", "text": "Done."},
    ]

    digest = run._turn_digest(frames)

    (component,) = digest["components"]
    (asked,) = digest["elicitations"]
    (message,) = digest["messages"]
    assert component["at"] < asked["at"] < message["at"]


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


# --- the contract checks and the runner's expectations ------------------------

_REFERENCE: list[dict[str, Any]] = []


async def _reference_frames() -> list[dict[str, Any]]:
    """One studio turn's frames, conformant: a search, a fetch and a card,
    then an answer citing the card's figures. Built once per test run."""
    if not _REFERENCE:
        app, _model = orchestrator_app(
            search_call("life expectancy in kenya"),
            fetch_call(),
            make_charts_call(chart(frame_for())),
            types.Part.from_text(text="Life expectancy in Kenya rose from 48.5 to 63.6."),
            resolver=CannedSearch(definite_envelope()),
        )
        async with studio_client(app) as client:
            _REFERENCE.extend(await stream(client, "/api/studio/turn", turn_body("q")))
    assert checks.check_transcript(_REFERENCE) == []
    return copy.deepcopy(_REFERENCE)


def _index(frames: list[dict[str, Any]], event: str) -> int:
    return next(index for index, frame in enumerate(frames) if frame["event"] == event)


def _shifted(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(frame, seq=frame["seq"] + 1) for frame in frames]


def _mutated(name: str, frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """`frames` broken the way `name` says."""
    created = _index(frames, "component.created")
    updated = _index(frames, "composition.updated")
    complete = _index(frames, "message.complete")
    done = _index(frames, "done")
    if name == "seq-disorder":
        frames[1]["seq"], frames[2]["seq"] = frames[2]["seq"], frames[1]["seq"]
    elif name == "frames-after-done":
        frames.append(dict(frames[0], seq=99))
    elif name == "double-done":
        frames.append(dict(frames[done], seq=99))
    elif name == "done-after-error":
        error = {
            "event": "error",
            "seq": frames[created]["seq"],
            "error": {"error": "transient_unavailable", "retry": "do_not_retry", "detail": "x"},
        }
        frames = [*frames[:created], error, *_shifted(frames[created:])]
    elif name == "invalid-frame":
        del frames[created]["component_id"]
    elif name == "ungrounded-number":
        frames[complete]["text"] += " GDP reached 987.65 billion."
    elif name == "id-laundered-number":
        # A number that exists only inside an identifier grounds nothing.
        frames[created]["component_id"] = "cmp_0000000000004321"
        frames[created]["fact_summary"]["component_id"] = "cmp_0000000000004321"
        frames[complete]["text"] += " The value hit 4321."
    elif name == "seq-not-from-zero":
        frames = _shifted(frames)
    elif name == "turn-without-terminal":
        frames = frames[:-1]
    elif name == "composition-id-changes":
        frames[done]["composition"]["composition_id"] = "cps_other"
    elif name == "composition-revision-repeats":
        frames[updated]["manifest"]["revision"] = 2
        again = dict(copy.deepcopy(frames[updated]), seq=frames[updated]["seq"] + 1)
        frames = [*frames[: updated + 1], again, *_shifted(frames[updated + 1 :])]
    elif name == "done-older-than-last-update":
        frames[updated]["manifest"]["revision"] = 3
        frames[done]["composition"]["revision"] = 2
    return frames


@pytest.mark.parametrize(
    ("name", "fragment"),
    [
        ("seq-disorder", "strictly increasing"),
        ("frames-after-done", "followed by more frames"),
        ("double-done", "followed by more frames"),
        ("done-after-error", "followed by more frames"),
        ("invalid-frame", "fails contract validation"),
        ("ungrounded-number", "987.65"),
        ("id-laundered-number", "4321"),
        ("seq-not-from-zero", "contract says 0"),
        ("turn-without-terminal", "incomplete turn"),
        ("composition-id-changes", "composition_id changed"),
        ("composition-revision-repeats", "revision not strictly increasing"),
        ("done-older-than-last-update", "older than the last composition.updated"),
    ],
)
async def test_each_invariant_rejects_a_stream_that_breaks_it(name: str, fragment: str) -> None:
    violations = checks.check_transcript(_mutated(name, await _reference_frames()))

    assert any(fragment in violation for violation in violations), violations


async def test_a_revision_of_a_component_this_conversation_never_created_is_rejected() -> None:
    # Only sound over a complete conversation: a recording may hold a later
    # slice, so the runner opts in.
    frames = await _reference_frames()
    created = _index(frames, "component.created")
    orphan = {
        "event": "component.revised",
        "seq": frames[created]["seq"] + 1,
        "component_id": "cmp_never_created_0000000000",
        "revision": 2,
        "changed": ["title"],
        "fact_summary": frames[created]["fact_summary"],
    }
    revised = [*frames[: created + 1], orphan, *_shifted(frames[created + 1 :])]

    strict = checks.check_transcript(revised, complete_conversation=True)
    assert any("never streamed as component.created" in v for v in strict)
    assert not any("never streamed" in v for v in checks.check_transcript(revised))


async def test_a_segment_must_end_on_a_terminal_frame() -> None:
    frames = await _reference_frames()

    assert any("not a terminal frame" in v for v in checks.check_segment(frames[:-1]))
    pause = {"event": "plan.proposed", "seq": 2}
    interleaved = [*frames[:2], pause, *frames[2:]]
    assert any("followed by more" in v for v in checks.check_segment(interleaved))
    assert checks.check_segment(frames) == []


@pytest.mark.parametrize(
    ("text", "ungrounded"),
    [
        ("Population rose from 7.6 million.", []),
        ("Population rose from 7.64 million.", []),
        ("Population rose from 8 million.", []),
        ("Population rose from 7,640,000.", []),
        ("Population rose from 7.7 million.", ["7.7 million"]),
        ("Population rose from 3 million.", ["3 million"]),
    ],
)
def test_a_figure_with_a_scale_word_grounds_at_the_precision_written(
    text: str, ungrounded: list[str]
) -> None:
    """A fact of 7,640,000 grounds "7.6 million" as it grounds "7,640,000".
    A small number with a scale word is a claim, not a structural count."""
    turn = [
        {"event": "component.created", "fact_summary": {"series": [{"first": 7640000.0}]}},
        {"event": "message.complete", "text": text},
    ]

    assert checks.check_grounding([turn]) == (
        [
            f"turn 1: message.complete cites number(s) {ungrounded} "
            "found in no fact frame of this conversation"
        ]
        if ungrounded
        else []
    )


def _drawn_component(points: list[tuple[str, int]], *, summary_last: float) -> list[dict]:
    """A created line chart whose fact summary carries only a rounded last value."""
    return [
        {
            "event": "component.created",
            "fact_summary": {"series": [{"last": summary_last}]},
            "payload": {
                "component": {
                    "type": "line_chart",
                    "series": [
                        {"points": [{"period": period, "value": value} for period, value in points]}
                    ],
                }
            },
        }
    ]


def _ungrounded(frames: list[dict], text: str) -> list[str]:
    return checks.check_grounding([[*frames, {"event": "message.complete", "text": text}]])


@pytest.mark.parametrize(
    ("drawn", "text", "grounded"),
    [
        (817549561057, "Total 817,549,561,057 dollars.", True),
        (817549561057, "Total 817,549,561,058 dollars.", False),
        (817549561057, "Total 817.55 billion dollars.", True),
        (817549561057, "Total 817.56 billion dollars.", False),
        (1259600000, "Total 1.260 billion dollars.", True),
        (1233600000, "Total 1.234 billion dollars.", True),
        (1233600000, "Total 1.2340 billion dollars.", False),
    ],
)
def test_a_drawn_value_grounds_the_exact_figure_the_summary_rounds_away(
    drawn: int, text: str, grounded: bool
) -> None:
    """A card draws an exact value while its fact summary carries a rounding of
    it. The citation's precision is the one written: "1.260" carries four
    figures, and "1.2340" claims a fourth decimal the value does not have."""
    frames = _drawn_component([("2024", drawn)], summary_last=round(drawn, -9))

    assert (_ungrounded(frames, text) == []) is grounded


def test_narration_of_an_outflow_grounds_against_the_negative_figure() -> None:
    frames = [{"event": "component.created", "fact_summary": {"series": [{"last": -223000000.0}]}}]

    assert _ungrounded(frames, "A net outflow of $223 million.") == []
    assert _ungrounded(frames, "A net outflow of $224 million.") != []


def test_drawn_periods_ground_the_years_the_answer_cites() -> None:
    frames = _drawn_component([("2016", 110409325), ("2017/2018", 126882699)], summary_last=1.0)

    assert _ungrounded(frames, "Between 2016 and 2018 the total rose.") == []
    assert _ungrounded(frames, "Between 2016 and 2019 the total rose.") != []


@pytest.mark.parametrize("period", ["2024-12-31", "2024-01-01/2025-12-31"])
def test_drawn_dates_do_not_ground_invented_month_or_day_magnitudes(period: str) -> None:
    frames = _drawn_component([(period, 110409325)], summary_last=1.0)

    assert _ungrounded(frames, "In 2024 the total rose.") == []
    assert _ungrounded(frames, "The rate was 31%.") != []
    assert _ungrounded(frames, "The rate was 12%.") != []


def test_chart_points_ground_only_citations_of_four_or_more_figures() -> None:
    """Each point cited to four figures grounds, and moved one unit in its last
    digit does not. Cited to three figures it sits within half a unit of some
    point, which would let a chart ground any number near it."""
    billions = (1.0134, 1.0521, 1.0873, 1.1246, 1.1618, 1.2337, 1.2726, 1.3091, 1.3468, 1.3832)
    frames = _drawn_component(
        [(str(2000 + i), round(value * 1e9)) for i, value in enumerate(billions)],
        summary_last=1.0,
    )

    for value in billions:
        cited = round(value, 3)
        assert _ungrounded(frames, f"It reached {cited:.3f} billion.") == []
        for moved in (cited - 0.001, cited + 0.001):
            assert _ungrounded(frames, f"It reached {moved:.3f} billion.") != []
        assert _ungrounded(frames, f"It reached {value:.2f} billion.") != []


def test_an_unknown_expect_key_is_reported_as_a_violation() -> None:
    run = _load("run")

    unknown = "terminl"  # codespell:ignore terminl
    typo = {"message": "hi", "expect": {unknown: "done", "plan": {"min_step": 2}}}
    violations = run.check_expect_keys(typo, label="t")

    assert len(violations) == 2
    assert repr(unknown) in violations[0]
    assert "'min_step'" in violations[1]
    assert run.check_expect_keys({"message": "hi", "expect": {"terminal": "done"}}, label="t") == []


def test_a_plan_past_max_version_fails_the_expectation() -> None:
    run = _load("run")
    turn = {"message": "q", "plan_decision": "approve", "expect": {"plan": {"max_version": 1}}}
    frames = [
        {"event": "plan.proposed", "seq": 0, "version": 1, "steps": [{}]},
        {"event": "plan.proposed", "seq": 1, "version": 2, "steps": [{}]},
        {"event": "done", "seq": 2},
    ]

    violations = run._check_turn_expectations(turn, frames, label="t")

    assert any("version 2, expected <= 1" in v for v in violations)
    assert run._check_turn_expectations(turn, frames[:1] + frames[2:], label="t") == []


def test_a_proposed_plan_fails_a_no_plan_expectation() -> None:
    run = _load("run")
    turn = {"message": "q", "expect": {"no_plan": True}}
    frames = [
        {"event": "plan.proposed", "seq": 0, "version": 1, "steps": [{}]},
        {"event": "done", "seq": 1},
    ]

    violations = run._check_turn_expectations(turn, frames, label="t")

    assert violations == ["t: 1 plan.proposed frame(s), expected none"]
    assert run._check_turn_expectations(turn, frames[1:], label="t") == []


async def test_a_question_with_no_options_and_no_answer_text_is_a_miss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _load("run")
    question = {"event": "elicitation.request", "seq": 0, "interrupt_id": "int_1", "options": []}
    calls: list[str] = []

    async def fake_stream(client: object, path: str, body: dict) -> list[dict]:
        calls.append(path)
        return [question]

    monkeypatch.setattr(run, "_stream_frames", fake_stream)
    _, _, missed, _ = await run._run_message_turn(None, "c", {"message": "q"}, [])

    assert calls == ["/api/studio/turn"]  # no resume was sent
    assert missed == [
        "elicitation.request offered no options and the turn names no "
        "elicitation_text to answer it with"
    ]


def test_message_expectations_match_the_closing_text_case_insensitively() -> None:
    run = _load("run")
    turn = {
        "message": "q",
        "expect": {"message_contains": ["Withheld"], "message_excludes": ["declined"]},
    }

    assert run._check_turn_expectations(turn, _closing("The card was withheld."), label="t") == []
    missing = run._check_turn_expectations(turn, _closing("Nothing to show."), label="t")
    assert any("does not mention 'Withheld'" in v for v in missing)
    excluded = run._check_turn_expectations(
        turn, _closing("The card was withheld; the canvas DECLINED it."), label="t"
    )
    assert any("mentions 'declined'" in v for v in excluded)


def test_a_message_expectation_with_no_message_frame_fails() -> None:
    run = _load("run")
    turn = {"message": "q", "expect": {"message_excludes": ["declined"]}}

    violations = run._check_turn_expectations(turn, [{"event": "done", "seq": 0}], label="t")

    assert any("expected a message.complete frame" in v for v in violations)


@pytest.mark.parametrize("value", ["withheld", [], [""], [3], None])
def test_a_malformed_message_expectation_is_reported(value: object) -> None:
    run = _load("run")

    violations = run.check_expect_keys(
        {"message": "q", "expect": {"message_contains": value}}, label="t"
    )

    assert violations == [
        "t: expect.message_contains must be a non-empty list of non-empty strings"
    ]


def test_a_mutate_that_returns_no_manifest_fails_a_block_expectation() -> None:
    # A refused mutate has no manifest to count, so the expectation counts
    # as missed.
    run = _load("run")
    turn = {"operations": [], "expect": {"status": 200, "blocks": 1}}
    refused = {"status": 500, "manifest": None, "error_body": {"error": "boom"}}

    violations = run._check_mutation(turn, refused, label="t")

    assert any("no manifest" in v for v in violations)

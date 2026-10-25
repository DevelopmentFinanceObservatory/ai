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
from collections import Counter
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from google.adk.events import Event
from google.adk.sessions import InMemorySessionService
from google.genai import types
from harness.reply_figures import card_result
from harness.studio_state import ConversationState, StudioStore
from harness.studio_translate import _RecordingExecutor, _Translator

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
FIXTURES = Path(__file__).parent / "fixtures"
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


def _proposal(**translation: Any) -> dict[str, Any]:
    """A `propose_plan` call whose accepted proposal carries `translation`."""
    return {
        "kind": "call",
        "layer": "planner",
        "tool": "propose_plan",
        "response": {"outcome": "ok", "translation": {"outcome": "invalid_input", **translation}},
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
        *(
            (
                "translation_stops_on_a_repeated_failure",
                [],
                [_proposal(), _proposal(repeated=True), *after],
                bool(after),
            )
            for after in ([], [_proposal()])
        ),
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
        return {"contract": [], "expectation": [], "turn_frames": [], **run._reply_record([])}

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
    """A client whose turn request pauses on two questions and whose resume ends the turn."""

    def __init__(self) -> None:
        self.frames = {
            "/api/studio/turn": [
                {
                    "event": "elicitation.request",
                    "interrupt_id": "int_1",
                    "questions": [
                        {
                            "id": "q1",
                            "question": "How should the aid be counted?",
                            "options": [
                                {
                                    "id": "spec_net",
                                    "label": "Net disbursements",
                                    "description": "d",
                                },
                                {"id": "spec_gross", "label": "Gross disbursements"},
                            ],
                        },
                        {
                            "id": "q2",
                            "question": "Current or constant prices?",
                            "options": [
                                {"id": "o1", "label": "Constant 2023 US dollars"},
                                {"id": "o2", "label": "Current US dollars"},
                            ],
                        },
                    ],
                }
            ],
            "/api/studio/interrupt": [{"event": "done"}],
        }
        self.sent: list[dict[str, Any]] = []

    def stream(self, method: str, path: str, **request: Any) -> Any:
        self.sent.append(request["json"])
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
    ("scripted", "sent", "shown"),
    [
        (
            [],
            {
                "q1": {"kind": "option", "option_id": "spec_net"},
                "q2": {"kind": "option", "option_id": "o1"},
            },
            ["Net disbursements", "Constant 2023 US dollars"],
        ),
        (
            [{"option": 2}, {"skipped": True}],
            {"q1": {"kind": "option", "option_id": "spec_gross"}, "q2": {"kind": "skipped"}},
            ["Gross disbursements", "left to the agent"],
        ),
        (
            [{"text": "Count commitments"}],
            {
                "q1": {"kind": "text", "text": "Count commitments"},
                "q2": {"kind": "option", "option_id": "o1"},
            },
            ["Count commitments", "Constant 2023 US dollars"],
        ),
    ],
)
async def test_every_question_of_a_pause_is_answered_and_shown_to_the_judge(
    scripted: list[dict[str, Any]], sent: dict[str, Any], shown: list[str]
) -> None:
    run = _load("run")
    studio = _ScriptedStudio()
    turn = {"message": "How much aid did France give?", "elicitation_answers": scripted}

    frames, _, missed, answers = await run._run_message_turn(studio, "conv", turn, [])

    assert missed == []
    assert studio.sent[-1]["answers"] == sent
    assert "payload" not in studio.sent[-1]
    (asked,) = run._turn_digest(frames, answers)["elicitations"]
    assert [q["question"] for q in asked["questions"]] == [
        "How should the aid be counted?",
        "Current or constant prices?",
    ]
    assert [o["label"] for o in asked["questions"][0]["options"]] == [
        "Net disbursements",
        "Gross disbursements",
    ]
    assert [q["answer"] for q in asked["questions"]] == shown
    unanswered = run._turn_digest(frames)["elicitations"][0]["questions"]
    assert [q["answer"] for q in unanswered] == [None, None]


def test_the_judge_digest_orders_entries_across_lists() -> None:
    """A figure drawn before the question reads as drawn before it, though
    components and questions sit in separate lists."""
    run = _load("run")
    frames = [
        {"event": "component.created", "type": "line_chart", "fact_summary": {}},
        {
            "event": "elicitation.request",
            "interrupt_id": "int_1",
            "questions": [{"id": "q1", "question": "Combine how?"}],
        },
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
    point, which would let a chart ground any number near it, so only the
    series' first and last value ground at three, the two a reply is given."""
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
        given = value in (billions[0], billions[-1])
        assert (_ungrounded(frames, f"It reached {value:.2f} billion.") == []) is given
        assert _ungrounded(frames, f"It reached {value:.1f} billion.") != []


async def _no_executor(*, tool: str, args: dict[str, object]) -> dict[str, object]:
    raise AssertionError(f"no tool call expected, got {tool}")


def test_a_count_explain_dfo_read_grounds_the_reply_that_quotes_it() -> None:
    """The counts reach the reply only through the model, so the translator's
    `explain_dfo` step carries them for the grounding check to read."""
    counts = "DFO holds 108,905 indicators in 92 datasets from 42 publishers."
    translator = _Translator(
        conversation=ConversationState(conversation_id="c1"),
        recorder=_RecordingExecutor(_no_executor, store=StudioStore(InMemorySessionService())),
    )
    call = types.FunctionCall(id="call_1", name="explain_dfo", args={})
    read = types.FunctionResponse(
        id="call_1", name="explain_dfo", response={"explainer": "DFO is", "counts": counts}
    )
    frames = [
        frame.model_dump(mode="json")
        for part in (types.Part(function_call=call), types.Part(function_response=read))
        for frame in translator.translate(
            Event(author="dfo_orchestrator", content=types.Content(parts=[part]))
        )
    ]

    assert _ungrounded(frames, "DFO holds 108,905 indicators in 92 datasets.") == []
    assert _ungrounded(frames, "DFO holds 108,906 indicators.") != []


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
    question = {
        "event": "elicitation.request",
        "seq": 0,
        "interrupt_id": "int_1",
        "questions": [{"id": "q1", "question": "Which?", "options": []}],
    }
    calls: list[str] = []

    async def fake_stream(client: object, path: str, body: dict) -> list[dict]:
        calls.append(path)
        return [question]

    monkeypatch.setattr(run, "_stream_frames", fake_stream)
    _, _, missed, _ = await run._run_message_turn(None, "c", {"message": "q"}, [])

    assert calls == ["/api/studio/turn"]  # no resume was sent
    assert missed == [
        "elicitation.request question 1 offers 0 option(s) and the turn answers it with "
        "{'option': 1}"
    ]


@pytest.mark.parametrize(
    ("rules", "violations"),
    [
        ({"min_questions": 2, "min_options": 2}, []),
        ({"min_questions": 3}, ["t: elicitation asks 2 question(s), expected >= 3"]),
        (
            {"min_options": 3},
            ["t: elicitation question 2 has 2 options, expected >= 3"],
        ),
        (
            {"allow_free_text": True},
            ["t: elicitation question 1 allow_free_text=False, expected True"],
        ),
        ({"skippable": True}, ["t: elicitation skippable=False, expected True"]),
    ],
)
def test_an_elicitation_expectation_reads_every_question(
    rules: dict[str, Any], violations: list[str]
) -> None:
    run = _load("run")
    option = {"id": "o", "label": "x"}
    pause = {
        "event": "elicitation.request",
        "interrupt_id": "int_1",
        "skippable": False,
        "questions": [
            {"id": "q1", "question": "a", "options": [option] * 3, "allow_free_text": False},
            {"id": "q2", "question": "b", "options": [option] * 2, "allow_free_text": True},
        ],
    }
    turn = {"message": "q", "expect": {"elicitation": rules}}

    assert run._check_turn_expectations(turn, [pause], label="t") == violations


def test_a_figure_in_a_later_question_grounds_the_answer() -> None:
    pause = {
        "event": "elicitation.request",
        "questions": [
            {"id": "q1", "question": "Which count?", "options": [{"id": "o1", "label": "Net"}]},
            {
                "id": "q2",
                "question": "Which prices?",
                "options": [{"id": "o1", "label": "Constant 2023 US dollars, 12,345 a year"}],
            },
        ],
    }

    assert _ungrounded([pause], "It came to 12,345.") == []
    assert _ungrounded([pause], "It came to 12,346.") != []


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
@pytest.mark.parametrize("key", ["message_contains", "plan.operations_contain"])
def test_a_malformed_text_list_expectation_is_reported(value: object, key: str) -> None:
    run = _load("run")
    expect: dict[str, Any] = {"message_contains": value}
    if key == "plan.operations_contain":
        expect = {"plan": {"operations_contain": value}}

    violations = run.check_expect_keys({"message": "q", "expect": expect}, label="t")

    assert violations == [f"t: expect.{key} must be a non-empty list of non-empty strings"]


def test_a_mutate_that_returns_no_manifest_fails_a_block_expectation() -> None:
    # A refused mutate has no manifest to count, so the expectation counts
    # as missed.
    run = _load("run")
    turn = {"operations": [], "expect": {"status": 200, "blocks": 1}}
    refused = {"status": 500, "manifest": None, "error_body": {"error": "boom"}}

    violations = run._check_mutation(turn, refused, label="t")

    assert any("no manifest" in v for v in violations)


def _card_frame(
    series: dict[str, dict[str, float]], *, title: str = "A card", unit: Any = "Years"
) -> dict[str, Any]:
    """A `component.created` frame of a chart drawing `series`, each label's
    values by period, its fact summary naming each series' place."""
    drawn = [
        {
            "id": f"s_{label.lower().replace(' ', '_')}",
            "label": label,
            "unit": unit,
            "points": [{"period": period, "value": value} for period, value in points.items()],
        }
        for label, points in series.items()
    ]
    listed = [
        {"id": one["id"], "coordinates": {"observationAbout": f"country/{one['label']}"}}
        for one in drawn
    ]
    return {
        "event": "component.created",
        "component_id": "cmp_" + "_".join(one["id"].removeprefix("s_") for one in drawn),
        "payload": {"component": {"type": "bar_chart", "title": {"text": title}, "series": drawn}},
        "fact_summary": {"series": listed},
    }


def _returned(frame: dict[str, Any], **named: Any) -> dict[str, Any]:
    """The frame's card as `make_charts` returns it for the arguments `named`."""
    return card_result(checks.reply_card(frame).series, **named)


def _rules(text: str, cards: list[Any], **given: Any) -> set[str]:
    return {fault.rule for fault in checks.check_reply(text, cards, **given)}


def test_the_recorded_ranking_replies_fail_where_they_list_the_ranking() -> None:
    """Twenty-three replies live runs gave to a ranking of twelve countries,
    each held to the figures its card returned: the two marks, and the four
    values where the user asked for four. The sixteen that name a country
    they state no figure of break the label rule. Twenty-one, the seven
    brief ones among them, write the largest and the smallest in one
    sentence ("Senegal highest at 68.7 years, while Nigeria lowest at 54.5
    years"), which leaves a value beside both names."""
    recorded = json.loads((FIXTURES / "recorded_ranking_replies.json").read_text())
    frame = _card_frame(
        {label: {recorded["period"]: value} for label, value in recorded["series"].items()},
        title=recorded["title"],
    )

    verdicts = {
        reply["id"]: sorted(
            _rules(
                reply["text"],
                [
                    checks.reply_card(
                        frame, returned=_returned(frame, state_values_for=reply["asked"])
                    )
                ],
            )
        )
        for reply in recorded["replies"]
    }

    assert verdicts == {reply["id"]: reply["rules"] for reply in recorded["replies"]}
    assert Counter(reply["verdict"] for reply in recorded["replies"]) == {
        "fail": 16,
        "undecided": 7,
    }
    assert all(
        reply["verdict"]
        == checks.reply_verdict(checks.ReplyFault(rule, "") for rule in verdicts[reply["id"]])
        for reply in recorded["replies"]
    )
    listing = [rules for rules in verdicts.values() if "series_outside_reply_figures" in rules]
    undecided = [rules for rules in verdicts.values() if "pairing_undecided" in rules]
    assert (len(listing), len(undecided)) == (16, 21)
    assert [rules for rules in verdicts.values() if rules not in listing] == [
        ["pairing_undecided"]
    ] * 7


PAIR = _card_frame({"Uganda": {"2024": 68.52}, "Kenya": {"2024": 63.81}})
BOTH = [{"series": "Kenya", "which": "last"}, {"series": "country/Uganda", "which": "last"}]
SPAN = _card_frame(
    {
        "Kenya": {"2000": 56.1, "2012": 60.9, "2023": 63.6},
        "South Sudan": {"2000": 47.6, "2012": 55.2, "2023": 57.6},
    }
)


@pytest.mark.parametrize(
    ("text", "rules"),
    [
        ("In 2024, life expectancy in Uganda and Kenya was 68.5 and 63.8 years.", set()),
        ("The values in 2024:\n- **Uganda**: 68.5 years\n- **Kenya**: 63.8 years", set()),
        # The year of a sentence carries to the next ones that name none.
        ("In 2024 Uganda led. Uganda stood at 68.5 years. Kenya stood at 63.8 years.", set()),
        # A value with a name on each side of it is beside both, and
        # nothing says which it is of: 63.8 here, 68.5 in the next.
        (
            "In 2024, life expectancy was 68.5 years in Uganda and 63.8 years in Kenya.",
            {"pairing_undecided"},
        ),
        (
            "Uganda (68.5 years) is ahead of Kenya (63.8 years) in 2024.",
            {"pairing_undecided"},
        ),
        (
            "In 2024, life expectancy was 63.8 years in Uganda and 68.5 years in Kenya.",
            {"value_outside_reply_figures", "pairing_undecided"},
        ),
        (
            "The values in 2024:\n- **Uganda**: 63.8 years\n- **Kenya**: 68.5 years",
            {"value_outside_reply_figures", "figure_missing"},
        ),
        (
            "In 2023, life expectancy in Uganda and Kenya was 68.5 and 63.8 years.",
            {"figure_without_period"},
        ),
        ("Life expectancy in Uganda and Kenya was 68.5 and 63.8 years.", {"figure_without_period"}),
        ("In 2024, life expectancy was 68.5 years in Uganda.", {"figure_missing"}),
        # Two figures state no value a reply rounds to three.
        ("In 2024 Uganda stood at about 69 years. Kenya stood at 63.8 years.", {"figure_missing"}),
    ],
)
def test_a_reply_states_each_required_figure_beside_its_series_and_its_year(
    text: str, rules: set[str]
) -> None:
    assert _rules(text, [checks.reply_card(PAIR)], states=BOTH) == rules


@pytest.mark.parametrize(
    ("text", "rules"),
    [
        ("In 2000 Kenya stood at 56.1 years. In 2023 Kenya stood at 63.6 years.", set()),
        ("Between 2000 and 2023 Kenya rose from 56.1 to 63.6 years.", set()),
        (
            "In 2023 Kenya stood at 56.1 years. In 2000 Kenya stood at 63.6 years.",
            {"figure_without_period"},
        ),
        # A value with a year on each side of it is beside both.
        ("Kenya: 56.1 years in 2000, rising to 63.6 years in 2023.", {"pairing_undecided"}),
        (
            "Kenya: 56.1 years in 2023, up from 63.6 years in 2000.",
            {"figure_without_period", "pairing_undecided"},
        ),
        ("South Sudan: 56.1 years in 2000.", {"value_outside_reply_figures"}),
        # Two names and then two values that are both the second name's.
        ("In 2000 and 2023, compared with South Sudan, Kenya stood at 56.1 and 63.6.", set()),
        # Two years and then four values, and the years repeat for each series.
        (
            "In 2000 and 2023, Kenya and South Sudan stood at 56.1 and 47.6 and at 63.6 and 57.6.",
            set(),
        ),
        ("Life expectancy, 2000 to 2023:\n- Kenya: 56.1 and 63.6 years", set()),
        # The card draws no point at 2024, so the list is read in 2023.
        (
            "The data is for 2023, while 2024 covers fewer countries so far. The values:\n"
            "- Kenya: 63.6 years\n- South Sudan: 57.6 years",
            set(),
        ),
        ("Kenya and South Sudan stood at 56.1, 63.6 and 47.6 years.", {"pairing_undecided"}),
        # South Sudan's 57.6 has only Kenya named near it, past another value.
        ("In 2023 the two ranged from 57.6 years to 63.6 years in Kenya.", {"pairing_undecided"}),
        # A value in a sentence that names neither series of the card.
        ("In 2023 the higher of the two stood at 63.6 years.", {"pairing_undecided"}),
        # Two carried years the card draws, and one value that is at neither's turn.
        ("Life expectancy, 2000 to 2023:\n- Kenya: 56.1 years", {"pairing_undecided"}),
        # 2100 is no year or value the card draws, and no word places it.
        ("In 2023 Kenya stood at 63.6 years, short of the 2100 target.", {"pairing_undecided"}),
        # 58 rounds South Sudan's 57.6, and the sentence names only Kenya.
        ("In 2023 Kenya stood at 63.6 years, among 58 countries.", {"pairing_undecided"}),
        # Two names and then each name's two values in turn.
        (
            "Kenya and South Sudan both rose between 2000 and 2023: from 56.1 to 63.6 and "
            "from 47.6 to 57.6.",
            set(),
        ),
        (
            "Kenya and South Sudan both rose between 2000 and 2023: from 47.6 to 57.6 and "
            "from 56.1 to 63.6.",
            {"value_outside_reply_figures"},
        ),
        # A bound that rounds Kenya's 56.1 is no third value.
        ("In 2023 Kenya and South Sudan were both above 56: 63.6 and 57.6 years.", set()),
        (
            "In 2023 Kenya and South Sudan were both above 56: 57.6 and 63.6 years.",
            {"value_outside_reply_figures"},
        ),
        # A value of the card the result did not list.
        ("Kenya stood at 60.9 years in 2012.", {"value_outside_reply_figures"}),
    ],
)
def test_a_reply_states_only_the_figures_its_card_returned(text: str, rules: set[str]) -> None:
    returned = _returned(SPAN, state_values_for=["Kenya", "South Sudan"], state_span=True)

    assert _rules(text, [checks.reply_card(SPAN, returned=returned)]) == rules


NAMED = {
    "PAIR": ["Kenya", "Uganda"],
    "SPAN": ["Kenya", "South Sudan"],
    "NEWEST_FIRST": ["Kenya"],
    "AMOUNTS": ["Exports"],
}
# Annual values that are written as four bare digits, as a year is.
AMOUNTS = _card_frame({"Exports": {"2020": 1500, "2023": 2500}}, unit="US Dollar")
NEWEST_FIRST = _card_frame({"Kenya": {"2023": 63.6, "2012": 60.9, "2000": 56.1}})
UNDATED_CARRY = pytest.mark.xfail(
    strict=True,
    reason="a value under a year an earlier paragraph names reads as that year's, which "
    "is how a correct reply dates a closing sentence; nothing but 'today' says otherwise",
)


@pytest.mark.parametrize(
    ("card", "states", "wrong", "rules", "right", "right_rules"),
    [
        (
            "PAIR",
            BOTH,
            "In 2024, life expectancy in Uganda and Kenya was 63.8 and 68.5 years respectively.",
            {"value_outside_reply_figures", "figure_missing"},
            "In 2024, life expectancy in Uganda and Kenya was 68.5 and 63.8 years respectively.",
            set(),
        ),
        (
            "PAIR",
            BOTH,
            "In 2024 Uganda trailed Kenya, at 63.8 years against 68.5 years.",
            {"value_outside_reply_figures", "figure_missing"},
            "In 2024 Uganda trailed Kenya, at 68.5 years against 63.8 years.",
            set(),
        ),
        (
            "PAIR",
            BOTH,
            "Uganda leads in 2024. It stood at 63.8 years. The other country stood at 68.5 years.",
            {"value_outside_reply_figures", "pairing_undecided", "figure_missing"},
            "Uganda leads in 2024. It stood at 68.5 years. The other country stood at 63.8 years.",
            {"pairing_undecided"},
        ),
        (
            "PAIR",
            [],
            "In 2024 life expectancy was 64 years in Uganda and 69 years in Kenya.",
            {"value_outside_reply_figures", "pairing_undecided"},
            "In 2024 life expectancy was 69 years in Uganda and 64 years in Kenya.",
            {"pairing_undecided"},
        ),
        (
            "SPAN",
            [],
            "Between 2000 and 2023 Kenya fell from 63.6 to 56.1 years.",
            {"figure_without_period"},
            "Between 2000 and 2023 Kenya rose from 56.1 to 63.6 years.",
            set(),
        ),
        # A series drawn newest first ends at its latest period.
        (
            "NEWEST_FIRST",
            [{"series": "Kenya", "which": "last"}],
            "In 2000 Kenya stood at 56.1 years.",
            {"value_outside_reply_figures", "figure_missing"},
            "In 2023 Kenya stood at 63.6 years.",
            set(),
        ),
        (
            "AMOUNTS",
            [],
            "In 2023 Exports were 1500. In 2020 Exports were 2500.",
            {"figure_without_period"},
            "In 2020 Exports were 1500. In 2023 Exports were 2500.",
            set(),
        ),
        (
            "AMOUNTS",
            [],
            "In 2020 Exports were $2500. In 2023 Exports were $1500.",
            {"figure_without_period"},
            "In 2020 Exports were $1500. In 2023 Exports were $2500.",
            set(),
        ),
        pytest.param(
            "SPAN",
            [],
            "In 2000 both were lower.\n\nToday Kenya stands at 56.1 years.",
            {"figure_without_period"},
            "In 2000 both were lower.\n\nIn 2023 Kenya stands at 63.6 years.",
            set(),
            marks=UNDATED_CARRY,
        ),
        (
            "SPAN",
            [],
            "Life expectancy, 2000 to 2023. Latest values:\n- Kenya: 56.1 years\n"
            "- South Sudan: 47.6 years",
            {"pairing_undecided"},
            "Life expectancy, 2000 to 2023. Latest values:\n- Kenya: 63.6 years\n"
            "- South Sudan: 57.6 years",
            set(),
        ),
    ],
)
def test_values_exchanged_between_series_or_years_fail_and_pass_the_right_way_round(
    card: str,
    states: list[dict[str, str]],
    wrong: str,
    rules: set[str],
    right: str,
    right_rules: set[str],
) -> None:
    """Each card returned both ends of the series it draws, as a
    `make_charts` call naming them gets. The right reply passes unless it
    too leaves a value's series open."""
    frame = {"PAIR": PAIR, "SPAN": SPAN, "NEWEST_FIRST": NEWEST_FIRST, "AMOUNTS": AMOUNTS}[card]
    returned = _returned(
        frame, state_values_for=NAMED[card], state_span=card in ("SPAN", "AMOUNTS")
    )
    cards = [checks.reply_card(frame, returned=returned)]

    assert _rules(right, cards, states=states) == right_rules
    assert _rules(wrong, cards, states=states) == rules


@pytest.mark.parametrize(
    ("text", "verdict"),
    [
        ("In 2024, life expectancy in Uganda and Kenya was 68.5 and 63.8 years.", "pass"),
        # 63.8 sits beside both names.
        ("In 2024, life expectancy was 68.5 years in Uganda and 63.8 years in Kenya.", "undecided"),
        ("In 2024, life expectancy in Uganda and Kenya was 63.8 and 68.5 years.", "fail"),
        ("In 2024, life expectancy was 68.5 years in Uganda.", "fail"),
        # 63.8 is held to Uganda and is Kenya's, and 68.5 sits beside both names.
        ("In 2024, life expectancy was 63.8 years in Uganda and 68.5 years in Kenya.", "fail"),
    ],
)
def test_a_reply_passes_fails_or_is_undecided(text: str, verdict: str) -> None:
    """A fault fails the reply whatever else it holds. With no fault, a
    statement the check cannot decide makes it undecided, and a required
    figure that statement writes is not reported missing."""
    findings = checks.check_reply(text, [checks.reply_card(PAIR)], states=BOTH)

    assert checks.reply_verdict(findings) == verdict


def test_an_undecided_reply_fails_no_case_and_the_run_exits_clean(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run = _load("run")
    case = next(
        case["id"]
        for case in _suite()["cases"]
        if "live" in case["modes"] and not case.get("xfail")
    )
    undecided = "In 2024, life expectancy was 68.5 years in Uganda and 63.8 years in Kenya."
    found = checks.check_reply(undecided, [checks.reply_card(PAIR)], states=BOTH)

    async def outcome(client: Any, case: dict[str, Any]) -> dict[str, Any]:
        return {
            "contract": [],
            "expectation": run._reply_misses(1, found),
            "turn_frames": [],
            **run._reply_record([(1, finding) for finding in found]),
        }

    @asynccontextmanager
    async def no_client(url: str) -> AsyncIterator[object]:
        yield object()

    monkeypatch.setattr(run, "run_case", outcome)
    monkeypatch.setattr(run, "_frame_checks", lambda case, outcome: [])
    monkeypatch.setattr(run, "_live_client", no_client)
    monkeypatch.setattr(run, "_RESULTS_DIR", tmp_path)
    monkeypatch.setattr("sys.argv", ["run.py", "--mode", "live", "--case", case])

    assert run.main() == 0
    (written,) = [json.loads(path.read_text()) for path in tmp_path.iterdir()]
    assert [(record["status"], record["reply_verdict"]) for record in written] == [
        ("PASS", "undecided")
    ]
    assert [undecided in statement for statement in written[0]["reply_undecided"]] == [True]
    assert written[0]["reply_rules"] == []
    assert "reply check: 0 pass, 0 fail, 1 undecided" in capsys.readouterr().out


def test_a_figure_a_card_lists_and_a_case_requires_draws_one_fault() -> None:
    """The card marks Uganda its largest and Kenya its smallest, and the
    case requires both."""
    undated = "Uganda stood at 68.5 years. Kenya stood at 63.8 years."

    faults = checks.check_reply(undated, [checks.reply_card(PAIR)], states=BOTH)

    assert [fault.rule for fault in faults] == ["figure_without_period"] * 2


def test_a_redraw_gives_its_reply_no_value_to_state() -> None:
    kenya = _card_frame({"Kenya": {"1960": 48.5, "2024": 63.8}})
    recap = "The table replaces the line chart. In 2024, life expectancy stood at 63.8 years."

    redrawn = [checks.reply_card(kenya, on_canvas={"s_kenya"})]

    assert _rules("The table replaces the line chart.", redrawn) == set()
    assert _rules(recap, redrawn) == {"value_outside_reply_figures"}
    assert _rules(recap, [checks.reply_card(kenya)]) == set()


def test_a_number_no_frame_holds_is_a_fault_and_a_figure_the_card_gave_is_not() -> None:
    """The fact summary lists no value here, as for a series past its twelfth,
    so the card's own figure is all that grounds 63.8."""
    card = checks.reply_card(PAIR)

    def rules(text: str) -> set[str]:
        turns = [[PAIR, {"event": "message.complete", "text": text}]]
        return _rules(text, [card], turns=turns)

    assert rules("In 2024 Kenya stood at 63.8 years.") == set()
    assert rules("In 2024 Kenya stood at 63.8 years, 4.7 below the leader.") == {
        "number_in_no_frame"
    }


def test_a_figure_at_a_named_past_year_grounds_once_the_case_lists_it() -> None:
    """A value between a series' first and last is in no fact summary, and a
    drawn point grounds only at four figures."""
    text = "Kenya stood at 60.9 years in 2012."
    turns = [[SPAN, {"event": "message.complete", "text": text}]]
    listed = [{"series": "Kenya", "which": "at", "period": "2012"}]

    assert _rules(text, [checks.reply_card(SPAN)], states=listed, turns=turns) == set()
    assert _rules(text, [checks.reply_card(SPAN)], turns=turns) == {
        "value_outside_reply_figures",
        "number_in_no_frame",
    }


def test_a_card_of_24_series_gives_its_smallest_and_a_series_past_its_twelfth() -> None:
    """A recorded bar chart of the 24 largest donors to Malawi, whose fact
    summary lists twelve. Iceland is fifteenth."""
    drawn = json.loads((FIXTURES / "drawn_bar_of_24_donors.json").read_text())
    frame = {
        "event": "component.created",
        "payload": {"component": {key: drawn[key] for key in ("type", "title", "series")}},
        "fact_summary": {"series": drawn["summary_series"]},
    }
    states = [{"which": "smallest"}, {"series": "Iceland", "which": "last"}]
    asked = (
        "In 2024 the smallest donor drawn was the Global Environment Facility at "
        "$4.87 million. Iceland gave $11.9 million."
    )

    def faults(text: str) -> set[str]:
        turns = [[frame, {"event": "message.complete", "text": text}]]
        return _rules(text, [checks.reply_card(frame)], states=states, turns=turns)

    assert len(checks.reply_card(frame).series) == 24
    assert faults(asked) == set()
    assert faults(asked.replace("11.9", "11.4")) == {
        "figure_missing",
        "value_outside_reply_figures",
    }
    assert faults(asked + " Japan gave $21.6 million.") == {
        "value_outside_reply_figures",
        "series_outside_reply_figures",
    }
    assert faults("In 2024 Iceland gave $11.9 million.") == {"figure_missing"}


def test_a_cards_counts_ground_the_reply_that_cites_them() -> None:
    """The card draws 24 series and its title says they are the largest of
    70. No step frame is here to repeat the title."""
    drawn = json.loads((FIXTURES / "drawn_bar_of_24_donors.json").read_text())
    title = {"text": drawn["title"], "authorship": "model"}
    frame = {
        "event": "component.created",
        "payload": {
            "component": {"type": drawn["type"], "title": title, "series": drawn["series"]}
        },
        "fact_summary": {"series": drawn["summary_series"]},
    }

    assert "(24 largest of 70 in 2024)" in drawn["title"]
    assert _ungrounded([frame], "The chart shows the 24 largest of 70 donors.") == []
    assert _ungrounded([frame], "The chart shows the 24 largest of 71 donors.") != []
    assert _ungrounded([frame], "The chart shows 25 donors.") != []


@pytest.mark.parametrize(
    ("listed", "rules"),
    [
        (
            "- United States of America: $399 million\n- Global Fund: $231 million\n"
            "- Iceland: $11.9 million",
            set(),
        ),
        (
            "- United States of America: $231 million\n- Global Fund: $399 million\n"
            "- Iceland: $11.9 million",
            {"value_outside_reply_figures"},
        ),
        (
            "- United States of America: $420 million\n- Global Fund: $231 million\n"
            "- Iceland: $11.9 million",
            {"number_in_no_frame"},
        ),
    ],
)
def test_a_reply_listing_a_ranking_under_the_period_note_is_held_value_by_value(
    listed: str, rules: set[str]
) -> None:
    """The case lists no figure and the card's own figures are its largest
    and smallest, so each listed value is held to the point drawn for its
    series in 2024. Iceland is past the twelve series the summary lists."""
    run = _load("run")
    drawn = json.loads((FIXTURES / "drawn_bar_of_24_donors.json").read_text())
    frame = {
        "event": "component.created",
        "payload": {"component": {key: drawn[key] for key in ("type", "title", "series")}},
        "fact_summary": {"series": drawn["summary_series"]},
    }
    note = "The data is for 2024, which covers 85 donors, while 2025 covers 9 donors so far:\n"
    steps = {"event": "activity.step", "label": "85 series in 2024, 9 in 2025", "detail": None}
    turns = [[steps, frame, {"event": "message.complete", "text": note + listed}]]

    faults, _, stated = run._reply_faults({"message": "q", "expect": {}}, turns)

    assert {fault.rule for fault in faults} == rules
    assert (checks.check_grounding(turns, stated=[stated]) == []) == (
        "number_in_no_frame" not in rules
    )


def test_a_value_of_a_series_called_by_another_name_is_undecided() -> None:
    """The reply shortens "United States of America", which no label matches,
    so its value sits beside a label that is not its own. Once the message
    spells the label out elsewhere, the value is held to the label beside it."""
    drawn = json.loads((FIXTURES / "drawn_bar_of_24_donors.json").read_text())
    frame = {
        "event": "component.created",
        "payload": {"component": {key: drawn[key] for key in ("type", "title", "series")}},
        "fact_summary": {"series": drawn["summary_series"]},
    }
    card = checks.reply_card(frame, any_point=True)
    trailed = "In 2024 Germany trailed the United States ($399 million)."

    assert _rules(trailed, [card]) == {"pairing_undecided"}
    assert _rules(f"{trailed} The United States of America led.", [card]) == {
        "value_outside_reply_figures"
    }


def test_a_case_that_lists_no_figure_allows_any_point_beside_its_own_series_and_year() -> None:
    card = checks.reply_card(SPAN, any_point=True)

    assert _rules("Kenya stood at 60.9 years in 2012.", [card]) == set()
    assert _rules("Kenya rose. South Sudan stood at 60.9 years in 2012.", [card]) == {
        "value_outside_reply_figures"
    }
    assert _rules("Kenya stood at 60.9 years in 2023.", [card]) == {"figure_without_period"}
    assert checks.stated_values([card]) == set()


@pytest.mark.parametrize(
    ("text", "rules"),
    [
        ("Kenya stood at 60.9 years in 2012.", set()),
        (
            "Kenya rose. South Sudan stood at 60.9 years in 2012.",
            {"value_outside_reply_figures", "number_in_no_frame"},
        ),
        ("Kenya stood at 60.9 years in 2023.", {"figure_without_period", "number_in_no_frame"}),
        ("Kenya stood at 60.9 years.", {"figure_without_period", "number_in_no_frame"}),
        ("Kenya stood at 61.4 years in 2012.", {"number_in_no_frame"}),
        (
            "In 2023 Kenya and South Sudan were both above 56: 63.6 and 61.0 years.",
            {"number_in_no_frame", "pairing_undecided"},
        ),
    ],
)
def test_a_point_between_a_series_ends_grounds_beside_its_own_series_and_year(
    text: str, rules: set[str]
) -> None:
    """A value between a series' first and last is in no fact summary, and
    the case lists no figure, so the card's figures are every point."""
    run = _load("run")
    turns = [[SPAN, {"event": "message.complete", "text": text}]]

    faults, _, stated = run._reply_faults({"message": "q", "expect": {}}, turns)

    assert {fault.rule for fault in faults} == rules
    assert (checks.check_grounding(turns, stated=[stated]) == []) == (
        "number_in_no_frame" not in rules
    )


@pytest.mark.parametrize(
    "expect",
    [
        None,
        [],
        {},
        {"states": None},
        {"states": [{"series": "Kenya"}]},
        {"states": [{"series": "Kenya", "which": "middle"}]},
        {"states": [{"series": "Kenya", "which": "last", "period": "2023"}]},
        {"states": [{"series": "Kenya", "which": "at"}]},
        {"states": [{"series": "", "which": "last"}]},
        {"states": [{"series": "Kenya", "which": "largest"}]},
        {"states": [{"series": "Kenya", "which": "last", "year": "2023"}]},
        {"states": [], "no_other_values": True},
    ],
)
def test_a_malformed_reply_expectation_is_reported(expect: object) -> None:
    run = _load("run")

    assert run.check_expect_keys({"message": "q", "expect": {"reply": expect}}, label="t") != []


def _composed(*frames: dict[str, Any]) -> dict[str, Any]:
    """A `composition.updated` frame whose canvas holds the cards `frames` drew."""
    blocks = [{"component_id": frame["component_id"]} for frame in frames]
    return {"event": "composition.updated", "manifest": {"blocks": blocks}}


# Six places, Kenya among them, so a card of them is past the four series a
# message may name without a figure.
SIX_COUNTRIES = ("Kenya", "Ghana", "Uganda", "Senegal", "Malawi", "Zambia")


def test_the_runner_holds_a_turn_to_its_cards_as_the_case_lists_figures() -> None:
    """With no `reply` expectation any drawn point may be stated. With one,
    a card gives what it gives unasked, and a card whose every series was on
    the canvas as its turn began gives nothing. A card on the canvas that a
    later turn reads again, drawing nothing, gives the figures the case
    lists, and naming a series of a held card is no fault. A card revised
    in the turn is read at its new revision. A card removed in between is
    off the canvas, so drawing its series again is a new card."""
    run = _load("run")
    kenya = _card_frame({"Kenya": {"1960": 48.5, "2024": 63.8}})
    again = kenya | {"component_id": "cmp_kenya_again"}
    said = {"event": "message.complete", "text": "Kenya rose from 48.5 years in 1960."}
    recap = {"event": "message.complete", "text": "In 2024 Kenya stood at 63.8 years."}
    removed = [_composed(), {"event": "message.complete", "text": "The card is removed."}]

    def rules(expect: dict[str, Any], *turns: list[dict[str, Any]]) -> list[str]:
        turn = {"message": "q", "expect": expect}
        before = [frame for earlier in turns[:-1] for frame in earlier]
        return [fault.rule for fault in run._reply_faults(turn, list(turns), before=before)[0]]

    listed = {"reply": {"states": []}}
    drawn = [kenya, _composed(kenya), recap]
    assert rules({}, [kenya, said]) == []
    assert rules(listed, [kenya, said]) == ["value_outside_reply_figures"]
    assert rules(listed, drawn) == []
    assert rules(listed, drawn, [again, _composed(kenya, again), recap]) == [
        "value_outside_reply_figures"
    ]
    assert rules(listed, drawn, removed, [again, _composed(again), recap]) == []
    last = {"reply": {"states": [{"series": "Kenya", "which": "last"}]}}
    assert rules(last, drawn, [recap]) == []
    assert rules(last, drawn, removed, [recap]) == ["figure_missing"]

    # A held card of six series names Kenya too, and the new Kenya card's
    # own figure may be stated beside its name.
    six = _card_frame({name: {"2023": 70.0 + n} for n, name in enumerate(SIX_COUNTRIES)})
    line = _card_frame({"Kenya": {"2000": 52.0, "2023": 60.0}}, title="Kenya")
    line["component_id"] = "cmp_kenya_line"
    for entries in (line["payload"]["component"]["series"], line["fact_summary"]["series"]):
        entries[0]["id"] = "s_kenya_line"
    stood = {"event": "message.complete", "text": "In 2023 Kenya stood at 60.0 years."}
    assert rules(listed, [six, _composed(six), said], [line, _composed(six, line), stood]) == []
    # A held card the turn did not touch is exempt from the series rule.
    remains = {"event": "message.complete", "text": "Kenya remains on the canvas."}
    assert rules(listed, [six, _composed(six), said], [remains]) == []

    # A card revised in the turn is read at its new revision.
    revised = _card_frame({"Kenya": {"1960": 48.5, "2024": 66.2}}) | {"event": "component.revised"}
    assert revised["component_id"] == kenya["component_id"]
    now = {"event": "message.complete", "text": "In 2024 Kenya stood at 66.2 years."}
    assert rules(last, drawn, [revised, now]) == []
    assert "figure_missing" in rules(last, drawn, [revised, recap])


def test_a_cut_card_is_held_to_its_title_and_its_series_count() -> None:
    run = _load("run")
    cut = _card_frame(
        {name: {"2024": 9.0 - place} for place, name in enumerate("ABC")},
        title="Aid to Malawi by donor (3 largest of 70 in 2024)",
    )
    expect = {"card_title_contains": ["largest of"], "max_card_series": 3}

    def violations(**changed: Any) -> list[str]:
        turn = {"message": "q", "expect": expect | changed}
        return run._check_turn_expectations(turn, [cut], label="t")

    assert violations() == []
    assert any("draws 3 series" in v for v in violations(max_card_series=2))
    assert any("'smallest of'" in v for v in violations(card_title_contains=["smallest of"]))


def test_the_donor_routing_case_fails_a_card_drawn_without_a_cut() -> None:
    run = _load("run")
    case = next(
        case
        for case in json.loads(CASES_PATH.read_text())["cases"]
        if case["id"] == "orchestrator-live-aid-by-donor-draws-one-cut-card-without-a-plan"
    )
    donors = {f"Donor {place}": {"2025": 99.0 - place} for place in range(9)}

    def title_faults(title: str) -> list[str]:
        frames = [_card_frame(donors, title=title)]
        found = run._check_turn_expectations(case["turns"][0], frames, label="t")
        return [v for v in found if "title" in v]

    assert title_faults("Aid to Malawi by donor (8 largest of 70 in 2024)") == []
    assert len(title_faults("Aid to Malawi by donor (2025)")) == 1


# The source sentence of each plan recorded for "general government debt
# service by creditor for 2031" on the deployment: version 2 pinned the
# recipient-only financial flow, version 3 the aggregate. Neither plan kept
# the creditors.
_PROD_CREDITOR_SENTENCES = [
    "Financial flow to 134 recipients for general government in 2031. Current US dollars "
    "\u00b7 World Bank International Debt Statistics (IDS).",
    "External debt service of 134 debtors for public or publicly guaranteed for general "
    "government in 2031. Current US dollars \u00b7 World Bank International Debt Statistics "
    "(IDS).",
]


_BY_CREDITOR_SOURCE = (
    "External debt service of 134 debtors owed to 97 creditors for general government in 2031."
)
_RANKS_CREDITORS = (
    "Ranks by the total of general government debt service by creditor: ranks the 97 "
    "creditors the source holds in any year; the window is collapsed into one ranking."
)
# An aggregate's sentence restates the planner's measure, which carries the user's words.
_SUMS_CREDITORS_AWAY = (
    "The total of general government debt service by creditor, grouped by period."
)


def _creditor_plan_frame(
    version: int, *operations: tuple[str, str], declined: str | None = None
) -> dict[str, Any]:
    """A plan card for the by-creditor question whose operations are `(kind, sentence)`
    in order, its one ask answered by the last. The step label repeats the user's
    "by creditor", as the planner's own words do."""
    ops = [
        {"id": f"op{number}", "kind": kind, "sentence": sentence}
        for number, (kind, sentence) in enumerate(operations)
    ]
    return {
        "event": "plan.proposed",
        "seq": version - 1,
        "version": version,
        "steps": [{"id": "op0", "label": "Fetch general government debt service by creditor"}],
        "requirements": [{"id": "debt_service_by_creditor", "sentence": operations[0][1]}],
        "operations": ops,
        "asks": [
            {
                "ask": "Debt service by creditor",
                "answered_by": [ops[-1]["id"]],
                "declined": declined,
            }
        ],
    }


@pytest.mark.parametrize(
    ("frames", "kept"),
    [
        pytest.param(
            [
                _creditor_plan_frame(number, ("retrieve", sentence))
                for number, sentence in enumerate(_PROD_CREDITOR_SENTENCES, start=2)
            ],
            False,
            id="prod",
        ),
        pytest.param(
            [
                _creditor_plan_frame(
                    2,
                    ("retrieve", _PROD_CREDITOR_SENTENCES[0]),
                    ("aggregate", _SUMS_CREDITORS_AWAY),
                )
            ],
            False,
            id="prod-plus-an-aggregate-of-the-measure",
        ),
        pytest.param(
            [
                _creditor_plan_frame(
                    2, ("retrieve", _BY_CREDITOR_SOURCE), ("aggregate", _SUMS_CREDITORS_AWAY)
                )
            ],
            False,
            id="creditor-source-summed-by-period",
        ),
        pytest.param(
            [
                _creditor_plan_frame(
                    2,
                    ("retrieve", _BY_CREDITOR_SOURCE),
                    declined="the data found does not break this down by the party asked",
                )
            ],
            False,
            id="declined",
        ),
        pytest.param(
            [
                _creditor_plan_frame(
                    2, ("retrieve", _BY_CREDITOR_SOURCE), ("rank", _RANKS_CREDITORS)
                )
            ],
            True,
            id="ranked-creditors",
        ),
        pytest.param(
            [_creditor_plan_frame(2, ("retrieve", _BY_CREDITOR_SOURCE))],
            True,
            id="creditor-source-as-the-answer",
        ),
    ],
)
def test_a_plan_whose_answer_never_names_the_creditors_fails_the_expectation(
    frames: list[dict[str, Any]], kept: bool
) -> None:
    run = _load("run")
    turn = {"message": "q", "expect": {"plan": {"operations_contain": ["creditor"]}}}

    violations = run._check_turn_expectations(turn, frames, label="t")

    assert violations == (
        [] if kept else ["t: the operations answering the plan's asks never say 'creditor'"]
    )

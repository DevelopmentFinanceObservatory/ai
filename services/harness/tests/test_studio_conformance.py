"""CI's model-free share of the harness-studio-v1 eval set.

The eval set proper lives in `evals/studio/` and its live/judged layer never
runs in CI (evals/README.md's two-lane rule). What CI can prove without a
model is everything deterministic about it:

- the recorded transcripts in `docs/transcripts/` — the frontend's reference
  streams — satisfy every wire-contract invariant in `evals/studio/checks.py`
- the scripted-mode cases pass end-to-end against the in-process scripted
  app (the same battery `just eval-studio-contract` runs)
- the case file can actually fail: every case carries a mechanical
  expectation or a rubric, ids are unique, and xfail markers give a reason

`evals/` sits outside the workspace (see `evals/agents/harness_planner_v2/
agent.py`), so its modules are loaded by file path.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
STUDIO_EVAL_DIR = REPO_ROOT / "evals" / "studio"
TRANSCRIPTS_DIR = REPO_ROOT / "docs" / "transcripts"
CASES_PATH = STUDIO_EVAL_DIR / "harness-studio-v1.cases.json"


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, STUDIO_EVAL_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # `run` imports its sibling `checks` as a plain module.
    sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)
    return module


checks = _load("checks")


@pytest.mark.parametrize(
    "transcript", sorted(TRANSCRIPTS_DIR.glob("*.jsonl")), ids=lambda path: path.stem
)
def test_recorded_transcripts_are_contract_conformant(transcript: Path) -> None:
    frames = [json.loads(line) for line in transcript.read_text().splitlines() if line.strip()]

    violations = checks.check_transcript(frames)

    assert violations == []


async def test_scripted_cases_pass_against_the_scripted_app() -> None:
    run = _load("run")

    results, passed = await run.run_suite(mode="scripted", url="", judge=False)

    # An empty battery passes vacuously; every scripted case has to have run.
    assert results, "no scripted case ran"
    assert [r["id"] for r in results] == [c["id"] for c in run.load_cases("scripted")]
    # An xfail case records its misses and reports XFAIL; an XPASS is a gap
    # that closed and a marker to retire, so CI fails on it too.
    failures = {
        r["id"]: (r["status"], r["contract_violations"] + r["expectation_violations"])
        for r in results
        if r["status"] not in ("PASS", "XFAIL")
    }
    assert passed and failures == {}


async def test_the_scripted_client_presents_the_token_the_app_enforces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # `run` loads with STUDIO_API_TOKEN unset (this suite's autouse fixture),
    # the way the real recipe starts before .env is in the process
    # environment; the token then changes before the client and the app it
    # talks to are built. The scripted app and the runner's client both have
    # to read STUDIO_API_TOKEN off the same Settings instance built at that
    # later point, or this produces an HTTP 401.
    run = _load("run")
    monkeypatch.setenv("STUDIO_API_TOKEN", "s3cret")
    case = next(c for c in run.load_cases("scripted") if c["id"] == "scripted-direct-fetch")

    async with run._scripted_client() as client:
        outcome = await run.run_case(client, case)

    assert outcome["contract"] == []
    assert outcome["expectation"] == []


def _reference_frames() -> list[dict]:
    transcript = TRANSCRIPTS_DIR / "studio_transcript.jsonl"
    return [json.loads(line) for line in transcript.read_text().splitlines() if line.strip()]


def _mutations() -> list[tuple[str, list[dict], str]]:
    """(name, corrupted frames, expected violation fragment) triples — each
    invariant must reject a stream that breaks it, or the conformance tests
    above prove nothing (an eval's default state is to pass)."""
    frames = _reference_frames()
    error_frame = {
        "event": "error",
        "seq": 5,
        "error": {"error": "transient_unavailable", "retry": "do_not_retry", "detail": "boom"},
    }

    swapped = copy.deepcopy(frames)
    swapped[3]["seq"], swapped[4]["seq"] = swapped[4]["seq"], swapped[3]["seq"]

    trailing = [*copy.deepcopy(frames), dict(copy.deepcopy(frames[0]), seq=99)]

    double_done = [*copy.deepcopy(frames), dict(copy.deepcopy(frames[-1]), seq=99)]

    done_after_error = [
        *copy.deepcopy(frames[:5]),
        error_frame,
        *(dict(f, seq=f["seq"] + 1) for f in copy.deepcopy(frames[5:])),
    ]

    invalid = copy.deepcopy(frames)
    del invalid[5]["component_id"]

    invented_number = copy.deepcopy(frames)
    for frame in invented_number:
        if frame["event"] == "message.complete":
            frame["text"] += " GDP reached 987.65 billion."

    # A number that exists ONLY inside an identifier must not ground a
    # narrated figure (cmp_...4321 is structure).
    id_laundering = copy.deepcopy(frames)
    for frame in id_laundering:
        if frame["event"] == "component.created":
            frame["component_id"] = "cmp_0000000000004321"
            frame["fact_summary"]["component_id"] = "cmp_0000000000004321"
        if frame["event"] == "message.complete":
            frame["text"] += " The value hit 4321."

    late_start = [dict(f, seq=f["seq"] + 1) for f in copy.deepcopy(frames)]

    unfinished = copy.deepcopy(frames[:-1])

    # `composition.updated` (frame 6) and `done` (frame 10) both carry the
    # manifest; the conversation invariants compare the two.
    renamed_composition = copy.deepcopy(frames)
    renamed_composition[10]["composition"]["composition_id"] = "cps_other"

    stale_update = copy.deepcopy(frames)
    stale_update[6]["manifest"]["revision"] = 2
    second_update = dict(copy.deepcopy(stale_update[6]), seq=7)
    stale_update = [
        *stale_update[:7],
        second_update,
        *(dict(f, seq=f["seq"] + 1) for f in stale_update[7:]),
    ]

    stale_done = copy.deepcopy(frames)
    stale_done[6]["manifest"]["revision"] = 3
    stale_done[10]["composition"]["revision"] = 2

    return [
        ("seq-disorder", swapped, "strictly increasing"),
        ("frames-after-done", trailing, "followed by more frames"),
        ("double-done", double_done, "followed by more frames"),
        ("done-after-error", done_after_error, "followed by more frames"),
        ("invalid-frame", invalid, "fails contract validation"),
        ("ungrounded-number", invented_number, "987.65"),
        ("id-laundered-number", id_laundering, "4321"),
        ("seq-not-from-zero", late_start, "contract says 0"),
        ("turn-without-terminal", unfinished, "incomplete turn"),
        ("composition-id-changes", renamed_composition, "composition_id changed"),
        ("composition-revision-repeats", stale_update, "revision not strictly increasing"),
        ("done-older-than-last-update", stale_done, "older than the last composition.updated"),
    ]


_MUTATIONS = _mutations()


@pytest.mark.parametrize(
    "name,frames,fragment", _MUTATIONS, ids=[name for name, _, _ in _MUTATIONS]
)
def test_each_invariant_rejects_a_stream_that_breaks_it(
    name: str, frames: list[dict], fragment: str
) -> None:
    violations = checks.check_transcript(frames)

    assert any(fragment in violation for violation in violations), (
        f"{name}: expected a violation mentioning {fragment!r}, got {violations}"
    )


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

    violations = checks.check_grounding([turn])

    assert violations == (
        [
            f"turn 1: message.complete cites number(s) {ungrounded} "
            "found in no fact frame of this conversation"
        ]
        if ungrounded
        else []
    )


def test_a_revision_of_a_component_this_conversation_never_created_is_rejected() -> None:
    # Only sound over a complete conversation: a transcript may record a
    # later slice, so the runner opts in and transcripts do not.
    frames = _reference_frames()
    orphan = {
        "event": "component.revised",
        "seq": 6,
        "component_id": "cmp_never_created_0000000000",
        "revision": 2,
        "changed": ["title"],
        "fact_summary": frames[5]["fact_summary"],
    }
    revised = [*frames[:6], orphan, *(dict(f, seq=f["seq"] + 1) for f in frames[6:])]

    strict = checks.check_transcript(revised, complete_conversation=True)
    assert any("never streamed as component.created" in v for v in strict)
    assert not any("never streamed" in v for v in checks.check_transcript(revised))


def test_a_segment_must_end_on_a_terminal_frame() -> None:
    frames = _reference_frames()

    # Cut before `done`: the stream ended without a terminal frame.
    truncated = frames[:-1]
    assert any("not a terminal frame" in v for v in checks.check_segment(truncated))

    # A pause frame mid-stream: terminal frames end their SSE response.
    plan_pause = {"event": "plan.proposed", "seq": 2}
    interleaved = [*frames[:2], plan_pause, *frames[2:]]
    assert any("followed by more" in v for v in checks.check_segment(interleaved))

    assert checks.check_segment(frames) == []


def test_every_case_can_fail() -> None:
    # An eval case's default state is to pass (evals/README.md): a case with
    # neither a mechanical expectation nor a rubric scores green whatever the
    # agent does, so no such case may exist. The same holds one level down:
    # an `expect` key the runner never reads (a typo, a retired check) is an
    # expectation that can only pass.
    run = _load("run")
    suite = json.loads(CASES_PATH.read_text())

    unscoreable = [
        case["id"]
        for case in suite["cases"]
        if not case.get("rubrics") and not any(turn.get("expect") for turn in case["turns"])
    ]
    unchecked = [
        violation
        for case in suite["cases"]
        for index, turn in enumerate(case["turns"], start=1)
        for violation in run.check_expect_keys(turn, label=f"{case['id']} turn {index}")
    ]

    assert unscoreable == []
    assert unchecked == []


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
    _, _, missed = await run._run_message_turn(None, "c", {"message": "q"}, [])

    assert calls == ["/api/studio/turn"]  # no resume was sent
    assert missed == [
        "elicitation.request offered no options and the turn names no "
        "elicitation_text to answer it with"
    ]


def _message_frames(text: str) -> list[dict]:
    return [{"event": "message.complete", "seq": 0, "text": text}, {"event": "done", "seq": 1}]


def test_message_expectations_match_the_closing_text_case_insensitively() -> None:
    run = _load("run")
    turn = {
        "message": "q",
        "expect": {"message_contains": ["Withheld"], "message_excludes": ["declined"]},
    }

    assert (
        run._check_turn_expectations(turn, _message_frames("The card was withheld."), label="t")
        == []
    )

    missing = run._check_turn_expectations(turn, _message_frames("Nothing to show."), label="t")
    assert any("does not mention 'Withheld'" in v for v in missing)

    excluded = run._check_turn_expectations(
        turn, _message_frames("The card was withheld; the canvas DECLINED it."), label="t"
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
    # A refused mutate has no manifest to count; the expectation is missed,
    # never skipped.
    run = _load("run")
    turn = {"operations": [], "expect": {"status": 200, "blocks": 1}}
    refused = {"status": 500, "manifest": None, "error_body": {"error": "boom"}}

    violations = run._check_mutation(turn, refused, label="t")

    assert any("no manifest" in v for v in violations)


def test_case_and_rubric_ids_are_unique_and_xfails_are_reasoned() -> None:
    suite = json.loads(CASES_PATH.read_text())

    case_ids = [case["id"] for case in suite["cases"]]
    assert len(case_ids) == len(set(case_ids))

    rubric_ids = [rubric["id"] for case in suite["cases"] for rubric in case.get("rubrics", [])]
    assert len(rubric_ids) == len(set(rubric_ids))

    for case in suite["cases"]:
        if "xfail" in case:
            # The marker doubles as the retirement note: it says which known
            # gap it tracks.
            assert len(case["xfail"]) > 20, f"{case['id']}: xfail without a reason"

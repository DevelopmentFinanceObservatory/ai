"""Regression coverage for `scripts/ask.py`'s env-var defaults, its
segment reader and its answers to a paused question.

`scripts/ask.py` sits outside the uv workspace (`pyproject.toml`'s
`[tool.uv.workspace]` lists only `packages/*` and `services/*`), so it can't
be imported as a package. It's loaded by file path instead, the same way
`test_studio_orchestrator_eval.py` loads `evals/studio/checks.py`.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from types import ModuleType

import httpx2
import pytest
from harness.wire import ActivityStep, ElicitationRequest, InterruptDecision, StepStatus

REPO_ROOT = Path(__file__).resolve().parents[3]
ASK_PATH = REPO_ROOT / "scripts" / "ask.py"
WIRE_FIXTURES = REPO_ROOT / "services" / "harness" / "tests" / "wire_fixtures" / "studio"


def _load_ask() -> ModuleType:
    spec = importlib.util.spec_from_file_location("ask", ASK_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("ask", module)
    spec.loader.exec_module(module)
    return module


ask = _load_ask()


def test_blank_studio_api_token_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """`just env` carries `STUDIO_API_TOKEN=` from `.env.example` into `.env`
    verbatim, even blank, so the key is always present -- never absent. A
    present-but-empty value must fall back to `_DEFAULT_TOKEN` the same way
    an absent one does; otherwise `--token` becomes `""` and the CLI builds
    the literal header `Bearer `, which httpx2 refuses to send.
    """
    monkeypatch.setenv("STUDIO_API_TOKEN", "")
    args = ask.parse_args(["question"])
    assert args.token == ask._DEFAULT_TOKEN
    assert args.token != ""


def test_unset_studio_api_token_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("STUDIO_API_TOKEN", raising=False)
    args = ask.parse_args(["question"])
    assert args.token == ask._DEFAULT_TOKEN


def test_real_studio_api_token_is_used_verbatim(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STUDIO_API_TOKEN", "a-real-token")
    args = ask.parse_args(["question"])
    assert args.token == "a-real-token"


def test_blank_studio_url_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STUDIO_URL", "")
    args = ask.parse_args(["question"])
    assert args.url == ask._DEFAULT_URL


def test_unset_studio_url_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("STUDIO_URL", raising=False)
    args = ask.parse_args(["question"])
    assert args.url == ask._DEFAULT_URL


@pytest.mark.parametrize(
    ("fixture", "event"),
    [
        ("event-03-plan-proposed.json", "plan.proposed"),
        ("event-09-elicitation-request.json", "elicitation.request"),
        ("event-14-elicitation-questions.json", "elicitation.request"),
        ("event-12-done.json", "done"),
    ],
)
async def test_stream_segment_reads_to_the_end_before_returning_its_ending_frame(
    fixture: str, event: str
) -> None:
    """The server releases the conversation after the pausing frame, and its
    stream ends once it has; `ask` posts `/interrupt` as soon as
    `stream_segment` returns, so the stream must already be at its end."""
    frame = json.loads((WIRE_FIXTURES / fixture).read_text())
    drained = False

    class Body(httpx2.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            nonlocal drained
            yield f"data: {json.dumps(frame)}\n\n".encode()
            yield b": ka\n\n"
            drained = True

    transport = httpx2.MockTransport(
        lambda request: httpx2.Response(
            200, headers={"content-type": "text/event-stream"}, stream=Body()
        )
    )
    async with httpx2.AsyncClient(transport=transport, base_url="http://studio.test") as client:
        ending = await ask.stream_segment(
            client,
            "/api/studio/turn",
            {},
            ask.Console(tty=False, color=False),
            ask.Waiter(start=0.0),
        )

    assert drained
    assert ending is not None
    assert ending.event == event


def _pause(fixture: str) -> ElicitationRequest:
    return ElicitationRequest.model_validate_json((WIRE_FIXTURES / fixture).read_text())


async def _answer(
    monkeypatch: pytest.MonkeyPatch, fixture: str, typed: list[str]
) -> tuple[InterruptDecision, dict[str, dict] | None]:
    """The decision and answers `prompt_elicitation_decision` sends for
    `fixture` when the user types `typed`, one line per prompt."""
    lines = iter(typed)

    async def fake_input(prompt: str = "") -> str:
        return next(lines)

    monkeypatch.setattr(ask, "_ainput", fake_input)
    decision, answers = await ask.prompt_elicitation_decision(
        _pause(fixture), ask.Console(tty=False, color=False), auto_yes=False
    )
    return decision, (
        None if answers is None else {key: a.model_dump() for key, a in answers.items()}
    )


@pytest.mark.parametrize(
    ("typed", "answers"),
    [
        (
            ["2", "1"],
            {
                "q1": {"kind": "option", "option_id": "spec_4d5e6f7a33333333"},
                "q2": {"kind": "option", "option_id": "o1"},
            },
        ),
        (
            ["s", "only grants"],
            {"q1": {"kind": "skipped"}, "q2": {"kind": "text", "text": "only grants"}},
        ),
        (
            ["", "1", "s"],
            {
                "q1": {"kind": "option", "option_id": "spec_3c4d5e6f22222222"},
                "q2": {"kind": "skipped"},
            },
        ),
    ],
)
async def test_each_question_of_a_pause_takes_its_own_answer(
    monkeypatch: pytest.MonkeyPatch, typed: list[str], answers: dict[str, dict]
) -> None:
    sent = await _answer(monkeypatch, "event-14-elicitation-questions.json", typed)

    assert sent == (InterruptDecision.ANSWER, answers)


async def test_skip_on_a_one_question_pause_dismisses_the_whole_pause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent = await _answer(monkeypatch, "event-13-elicitation-skippable.json", ["s"])

    assert sent == (InterruptDecision.SKIP, None)


async def test_yes_answers_every_question_with_its_recommended_or_first_option() -> None:
    pause = _pause("event-14-elicitation-questions.json")
    pause.questions[1].options[1].recommended = True

    decision, answers = await ask.prompt_elicitation_decision(
        pause, ask.Console(tty=False, color=False), auto_yes=True
    )

    assert decision is InterruptDecision.ANSWER
    assert {key: a.model_dump() for key, a in (answers or {}).items()} == {
        "q1": {"kind": "option", "option_id": "spec_3c4d5e6f22222222"},
        "q2": {"kind": "option", "option_id": "o2"},
    }


def test_a_complete_frame_on_a_failed_step_prints_its_detail_under_the_failure(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI cannot redraw the failed line it printed, so a later try's
    `complete` frame on the same step prints its own line and detail after it."""
    console = ask.Console(color=False)
    state = ask.RenderState()
    for seq, status, detail in [
        (1, StepStatus.FAILED, "This step could not be completed."),
        (2, StepStatus.COMPLETE, "This worked on a later try."),
    ]:
        frame = ActivityStep(
            event="activity.step",
            seq=seq,
            step_id="s1",
            label="Searching for data",
            status=status,
            detail=detail,
        )
        ask.render_frame(frame, console, state)

    assert capsys.readouterr().out.splitlines()[-2:] == [
        "  ✓ Searching for data",
        "      This worked on a later try.",
    ]

"""The turn budget on the Pydantic AI orchestrator, measured against dev's.

Both arms run a Studio turn under `Settings` with one bound made small. The
`dev` arm's scripted model keeps to the budget as `BoundedGemini` does, through
`harness.turn_budget.bounded`; the `pai` arm's is wrapped in `budget.BoundedModel`.
"""

from collections.abc import AsyncGenerator
from typing import Any

import pytest
from budget import BoundedModel
from google.adk.models.llm_response import LlmResponse
from harness.sentences import SeamCode, refusal_sentence
from harness.settings import Settings
from harness.turn_budget import bounded
from script import Step, StreamingScriptLlm
from tests.orchestrator_fakes import stream, studio_client, turn_body
from wiring import Arm, build_app, differences, events, stable


class BoundedScriptLlm(StreamingScriptLlm):
    """The scripted ADK model, each call kept to the budget as `BoundedGemini` keeps one."""

    async def generate_content_async(
        self, llm_request: Any, stream: bool = False
    ) -> AsyncGenerator[LlmResponse]:
        async for response in bounded(
            lambda: StreamingScriptLlm.generate_content_async(self, llm_request, stream),
            stream=stream,
        ):
            yield response


async def _turn(
    monkeypatch: pytest.MonkeyPatch, arm: Arm, steps: list[Step], settings: Settings
) -> tuple[list[dict[str, Any]], list[str]]:
    with monkeypatch.context() as patch:
        app, played = build_app(
            patch,
            steps,
            arm=arm,
            settings=settings,
            pai_wrap=BoundedModel,
            dev_model=BoundedScriptLlm,
        )
        async with studio_client(app) as client:
            frames = await stream(client, "/api/studio/turn", turn_body("What is DFO?"))
    return frames, played.log


def _reply(frames: list[dict[str, Any]]) -> str | None:
    return next((f["text"] for f in frames if f["event"] == "message.complete"), None)


async def test_the_call_cap_answers_the_third_call_with_its_sentence_on_both_arms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    steps = [Step(calls=[("explain_dfo", {})])] * 3 + [Step(text=["DFO is an observatory."])]
    settings = Settings(turn_model_calls_max=2)
    pai, pai_log = await _turn(monkeypatch, "pai", steps, settings)
    dev, dev_log = await _turn(monkeypatch, "dev", steps, settings)

    sentence = refusal_sentence({"code": SeamCode.TURN_MODEL_CALLS_SPENT})
    print(f"pai: {pai_log} -> {_reply(pai)!r}")
    print(f"dev: {dev_log} -> {_reply(dev)!r}")
    assert len(pai_log) == len(dev_log) == 2
    assert _reply(pai) == _reply(dev) == sentence
    assert events(pai)[-1] == events(dev)[-1] == "done"
    assert differences(stable(pai), stable(dev)) == []


async def test_a_cut_mid_stream_fails_the_pai_turn_where_dev_ends_on_the_sentence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Measured, not a fix: `BoundedModel` cuts a stalled stream on time, but the
    cut is an exception, so the turn ends on the error frame."""
    steps = [Step(text=["DFO is ", "an observatory."], stall=2.0)]
    settings = Settings(turn_model_time_s=0.3)
    pai, _ = await _turn(monkeypatch, "pai", steps, settings)
    dev, _ = await _turn(monkeypatch, "dev", steps, settings)

    sentence = refusal_sentence({"code": SeamCode.TURN_MODEL_TIME_SPENT})
    print(f"pai: {events(pai)}; dev: {events(dev)} -> {_reply(dev)!r}")
    assert _reply(dev) == f"DFO is \n\n{sentence}"
    assert events(dev)[-1] == "done"
    assert events(pai)[-1] == "error"
    assert [f["text"] for f in pai if f["event"] == "message.delta"] == ["DFO is "]

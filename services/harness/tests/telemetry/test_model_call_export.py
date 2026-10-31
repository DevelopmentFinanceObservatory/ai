"""What one streamed model call through `BoundedGemini` puts on the wire.

ADK's real runner, `Gemini` and inference span run against a stand-in genai
client; only the Langfuse server is fake (`fake_langfuse`). ADK ends the
`generate_content` span when the response that carries a finish reason reaches
it, before the stream ends, which a hand-built span in `test_turn_budget.py`
cannot show.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from types import SimpleNamespace
from typing import Any

from common.telemetry import flush_telemetry, init_telemetry
from google.adk.agents import LlmAgent
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import Client, types
from harness.turn_budget import BoundedGemini, TurnBudget, current_turn_budget
from opentelemetry.proto.trace.v1.trace_pb2 import Span

MODEL = "gemini-3.8-flash"
CHUNKS = ["Life ", "expectancy ", "rose."]


class _Models:
    """genai's `aio.models`: one stream of `CHUNKS`, the last with a finish reason."""

    async def generate_content_stream(self, *, model: str, contents: Any, config: Any) -> Any:
        async def chunks() -> AsyncGenerator[types.GenerateContentResponse]:
            for index, text in enumerate(CHUNKS):
                last = index == len(CHUNKS) - 1
                yield types.GenerateContentResponse(
                    candidates=[
                        types.Candidate(
                            content=types.Content(role="model", parts=[types.Part(text=text)]),
                            finish_reason=types.FinishReason.STOP if last else None,
                        )
                    ]
                )

        return chunks()


class _FakeClient(Client):
    def __init__(self) -> None:
        self.models_calls = _Models()

    @property
    def aio(self) -> Any:
        return SimpleNamespace(models=self.models_calls)

    @property
    def vertexai(self) -> bool:
        return False

    def close(self) -> None:
        pass


def _attribute(span: Span, key: str) -> Any:
    for attribute in span.attributes:
        if attribute.key == key:
            return getattr(attribute.value, attribute.value.WhichOneof("value"))
    return None


async def test_a_finished_stream_keeps_its_chunk_gaps_on_the_model_call_span(
    fake_langfuse: Any,
) -> None:
    init_telemetry(service_name="harness")
    agent = LlmAgent(
        name="answerer", model=BoundedGemini(model=MODEL, client=_FakeClient()), instruction="hi"
    )
    sessions = InMemorySessionService()
    await sessions.create_session(app_name="app", user_id="u", session_id="s")
    runner = Runner(app_name="app", agent=agent, session_service=sessions)
    token = current_turn_budget.set(
        TurnBudget(model_time_s=60.0, model_calls_max=30, wall_s=600.0, first_chunk_s=12.0)
    )
    try:
        async for _ in runner.run_async(
            user_id="u",
            session_id="s",
            new_message=types.Content(role="user", parts=[types.Part(text="q")]),
            run_config=RunConfig(streaming_mode=StreamingMode.SSE),
        ):
            pass
    finally:
        current_turn_budget.reset(token)
    await flush_telemetry()

    (call,) = [span for span in fake_langfuse.spans() if span.name == f"generate_content {MODEL}"]
    assert _attribute(call, "gen_ai.response.finish_reasons") is not None
    gaps = _attribute(call, "dfo.model.chunk_gaps_s")
    assert gaps is not None
    assert len(gaps.values) == len(CHUNKS) - 1

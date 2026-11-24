"""What one streamed model call of the orchestrator puts on the wire.

A Studio turn runs the orchestrator's agent on its real model
(`orchestrator.model.orchestrator_model`) against a scripted genai client; only
the Langfuse server is fake (`fake_langfuse`). The agent's `chat` span carries
the stream's timing, which a hand-built span in `test_turn_budget.py` cannot
show.
"""

from __future__ import annotations

from typing import Any

from common.telemetry import flush_telemetry, init_telemetry
from google.genai import types
from opentelemetry.proto.trace.v1.trace_pb2 import Span
from sqlalchemy.ext.asyncio import AsyncEngine

from ..scripted_genai import Reply
from ..studio_routes import orchestrator_app, stream, studio_client, turn_body
from ..turn_driver import MODEL

CHUNKS = ["Life ", "expectancy ", "rose."]


def _attribute(span: Span, key: str) -> Any:
    for attribute in span.attributes:
        if attribute.key == key:
            return getattr(attribute.value, attribute.value.WhichOneof("value"))
    return None


async def test_a_finished_stream_keeps_its_chunk_gaps_on_the_model_call_span(
    fake_langfuse: Any, pg_engine: AsyncEngine
) -> None:
    init_telemetry(service_name="harness")
    app, _model = orchestrator_app(
        Reply(chunks=[[types.Part.from_text(text=text)] for text in CHUNKS]), engine=pg_engine
    )
    async with studio_client(app) as client:
        frames = await stream(client, "/api/studio/turn", turn_body("q"))
    await flush_telemetry()

    assert frames[-1]["event"] == "done"
    (call,) = [span for span in fake_langfuse.spans() if span.name == f"chat {MODEL}"]
    assert _attribute(call, "gen_ai.response.finish_reasons") is not None
    gaps = _attribute(call, "dfo.model.chunk_gaps_s")
    assert gaps is not None
    assert len(gaps.values) == len(CHUNKS) - 1

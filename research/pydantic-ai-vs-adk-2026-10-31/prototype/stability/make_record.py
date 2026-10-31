"""Write a paused planner `Conversation` as JSON under the installed pydantic-ai.

Run: uv run --no-project --with 'pydantic-ai-slim==<version>' python make_record.py OUT.json

The run is real: a `FunctionModel` returns the parts, the agent executes one tool, and the
second tool raises `CallDeferred`, so the stored conversation carries a genuine
`deferred_tool_requests`. Parts are built by hand inside the `FunctionModel` because it cannot
produce Gemini's thought-signature fields itself; they mirror what `GoogleModel` stores
(`provider_name='google-gla'`, `provider_details={'thought_signature': <base64>}` on the
thinking part and on the function-call part, `provider_details` with `finish_reason` and
`timestamp` on the response).
"""

import base64
import datetime as dt
import json
import sys
from importlib.metadata import version
from pathlib import Path

from pydantic_ai import (
    Agent,
    CallDeferred,
    ConversationTypeAdapter,
    DeferredToolRequests,
    ModelResponse,
    TextPart,
    ThinkingPart,
    ToolCallPart,
)
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RequestUsage

PROVIDER = "google-gla"
CREATED = dt.datetime(2026, 10, 30, 9, 0, tzinfo=dt.UTC)


def signature(label: str) -> str:
    """A thought signature as `GoogleModel` stores it: base64 of the raw bytes."""
    return base64.b64encode(f"opaque-signature-bytes-{label}".encode()).decode()


def response(
    parts: list, *, response_id: str, input_tokens: int, output_tokens: int
) -> ModelResponse:
    return ModelResponse(
        parts=parts,
        usage=RequestUsage(input_tokens=input_tokens, output_tokens=output_tokens),
        model_name="gemini-3-flash-preview",
        provider_name=PROVIDER,
        provider_url="https://generativelanguage.googleapis.com/",
        provider_response_id=response_id,
        provider_details={"finish_reason": "STOP", "timestamp": CREATED},
        finish_reason="stop",
    )


def planner_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    model_responses = [m for m in messages if isinstance(m, ModelResponse)]
    if not model_responses:
        return response(
            [
                ThinkingPart(
                    content="Find ODA series for Ghana, then ask the user to confirm.",
                    provider_name=PROVIDER,
                    provider_details={"thought_signature": signature("think-1")},
                ),
                ToolCallPart(
                    tool_name="search",
                    args={"query": "ODA to Ghana"},
                    tool_call_id="call-search",
                    provider_name=PROVIDER,
                    provider_details={"thought_signature": signature("call-1")},
                ),
            ],
            response_id="resp-1",
            input_tokens=120,
            output_tokens=45,
        )
    return response(
        [
            ThinkingPart(
                content="One series fits. Propose it and wait for confirmation.",
                provider_name=PROVIDER,
                provider_details={"thought_signature": signature("think-2")},
            ),
            TextPart(content="I found one series; confirming before I fetch."),
            ToolCallPart(
                tool_name="confirm_plan",
                args={"spec_id": "series-oda-ghana"},
                tool_call_id="call-confirm",
                provider_name=PROVIDER,
                provider_details={"thought_signature": signature("call-2")},
            ),
        ],
        response_id="resp-2",
        input_tokens=260,
        output_tokens=70,
    )


def build_agent() -> Agent:
    agent = Agent(
        FunctionModel(planner_model),
        instructions="You are the planner. Propose a plan, then wait for approval.",
        output_type=[str, DeferredToolRequests],
    )

    @agent.tool_plain
    def search(query: str) -> dict:
        return {"hits": [{"spec_id": "series-oda-ghana", "title": query}]}

    @agent.tool_plain
    def confirm_plan(spec_id: str) -> str:
        raise CallDeferred(metadata={"spec_id": spec_id, "reason": "needs user"})

    return agent


def main(out: Path) -> None:
    result = build_agent().run_sync("How much ODA did Ghana receive in 2023?")
    conversation = result.conversation
    assert isinstance(result.output, DeferredToolRequests), result.output
    assert conversation.deferred_tool_requests is not None
    out.write_bytes(ConversationTypeAdapter.dump_json(conversation, indent=2))
    record = json.loads(out.read_text())
    print(
        json.dumps(
            {
                "writer": version("pydantic-ai-slim"),
                "file": str(out),
                "top_level_keys": sorted(record),
            }
        )
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]))

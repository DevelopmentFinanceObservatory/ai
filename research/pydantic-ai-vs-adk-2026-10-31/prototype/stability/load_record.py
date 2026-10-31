"""Load stored conversation records under the installed pydantic-ai and resume each one.

Run: uv run --no-project --with 'pydantic-ai-slim==<version>' python load_record.py RECORD...

Prints one JSON object: the reader version and, per record, whether it loaded, whether its
pause resumed, and the field-level diff between the file and the conversation dumped back
after loading. "Resumed" means the agent ran on a `FunctionModel` with `DeferredToolResults`
for the pending call, the model received the stored history intact (every thought signature
in the file, plus the deferred result as a tool return), and the run finished with text and
no pending request.
"""

import json
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any

from pydantic_ai import (
    Agent,
    CallDeferred,
    ConversationTypeAdapter,
    DeferredToolRequests,
    DeferredToolResults,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolReturnPart,
)
from pydantic_ai.conversation import Conversation
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import AgentInfo, FunctionModel

FINAL_TEXT = "Plan confirmed; fetching the series."
ANSWER = "approved by the user"
PENDING_CALL_ID = "call-confirm"


def signatures(messages: list[ModelMessage]) -> list[str]:
    """Every thought signature in the history, in order."""
    return [
        part.provider_details["thought_signature"]
        for message in messages
        if isinstance(message, ModelResponse)
        for part in message.parts
        if getattr(part, "provider_details", None) and "thought_signature" in part.provider_details
    ]


def stored_signatures(record: dict) -> list[str]:
    return [
        part["provider_details"]["thought_signature"]
        for message in record["messages"]
        if message["kind"] == "response"
        for part in message["parts"]
        if (part.get("provider_details") or {}).get("thought_signature")
    ]


def diff(before: Any, after: Any, path: str = "") -> list[dict]:
    """Field-level differences between two JSON values."""
    if isinstance(before, dict) and isinstance(after, dict):
        out = []
        for key in sorted(before.keys() | after.keys()):
            here = f"{path}.{key}" if path else key
            if key not in after:
                out.append({"path": here, "kind": "removed", "before": before[key]})
            elif key not in before:
                out.append({"path": here, "kind": "added", "after": after[key]})
            else:
                out += diff(before[key], after[key], here)
        return out
    if isinstance(before, list) and isinstance(after, list):
        if len(before) != len(after):
            return [{"path": path, "kind": "length", "before": len(before), "after": len(after)}]
        return [
            d
            for i, (b, a) in enumerate(zip(before, after, strict=True))
            for d in diff(b, a, f"{path}[{i}]")
        ]
    if before != after:
        return [{"path": path, "kind": "changed", "before": before, "after": after}]
    return []


def resume(conversation: Conversation) -> dict:
    """Resume the pause on a `FunctionModel` that answers with final text."""
    seen: dict[str, Any] = {}

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen["signatures"] = signatures(messages)
        last = messages[-1]
        seen["returns"] = (
            [
                p.content
                for p in last.parts
                if isinstance(p, ToolReturnPart) and p.tool_call_id == PENDING_CALL_ID
            ]
            if isinstance(last, ModelRequest)
            else []
        )
        return ModelResponse(parts=[TextPart(content=FINAL_TEXT)])

    agent = Agent(FunctionModel(model), output_type=[str, DeferredToolRequests])

    @agent.tool_plain
    def search(query: str) -> dict:
        return {}

    @agent.tool_plain
    def confirm_plan(spec_id: str) -> str:
        raise CallDeferred

    requests = conversation.deferred_tool_requests
    assert requests is not None, "record carries no deferred_tool_requests"
    assert [c.tool_call_id for c in requests.calls] == [PENDING_CALL_ID]
    result = agent.run_sync(
        conversation=conversation,
        deferred_tool_results=DeferredToolResults(calls={PENDING_CALL_ID: ANSWER}),
    )
    assert result.output == FINAL_TEXT, result.output
    assert result.conversation.deferred_tool_requests is None
    assert seen["returns"] == [ANSWER], seen["returns"]
    return seen


def check(path: Path) -> dict:
    raw = path.read_text()
    record = json.loads(raw)
    out: dict[str, Any] = {
        "record": path.name,
        "loaded": False,
        "load_error": None,
        "resumed": False,
        "resume_error": None,
        "round_trip_equal": None,
        "diff": [],
    }
    try:
        conversation = ConversationTypeAdapter.validate_json(raw)
    except Exception as exc:
        out["load_error"] = f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}"
        return out
    out["loaded"] = True

    dumped = json.loads(ConversationTypeAdapter.dump_json(conversation, indent=2))
    out["diff"] = diff(record, dumped)
    out["round_trip_equal"] = not out["diff"]

    try:
        seen = resume(conversation)
        if seen["signatures"] != stored_signatures(record):
            raise AssertionError(
                f"model saw signatures {seen['signatures']}, file has {stored_signatures(record)}"
            )
        out["resumed"] = True
    except Exception as exc:
        out["resume_error"] = f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}"
    return out


if __name__ == "__main__":
    print(
        json.dumps(
            {
                "reader": version("pydantic-ai-slim"),
                "results": [check(Path(p)) for p in sys.argv[1:]],
            }
        )
    )

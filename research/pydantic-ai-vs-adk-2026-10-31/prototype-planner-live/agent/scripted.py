"""Both planner arms against one scripted google-genai client, and what is compared.

The client is replaced at `client.aio.models.generate_content_stream`, the call each
framework's model makes, so ADK's `Gemini` and Pydantic AI's `GoogleModel` both run their
own request building and response parsing, and the request each sends is recorded as it
would go on the wire.
"""

from __future__ import annotations

import base64
import contextlib
import copy
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import entry
import pai_entry
import validate
from entry import Task, load_recording
from google.genai import Client, types
from harness.prefix_cache import PrefixCache

Step = list[tuple[str, dict[str, Any]]]


@dataclass
class Scripted:
    """A google-genai client that answers each streamed call with the next scripted turn,
    past its end with the text `done`, and records what it was sent."""

    script: list[list[types.Part]] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)
    creates: list[dict[str, Any]] = field(default_factory=list)
    client: Client = field(init=False)

    def __post_init__(self) -> None:
        self.client = Client(api_key="scripted")
        self.client.aio.models.generate_content_stream = self._stream  # type: ignore[method-assign]
        self.client.aio.caches.create = self._create  # type: ignore[method-assign]

    async def _stream(self, *, model: str, contents: Any, config: Any) -> Any:
        index = len(self.calls)
        self.calls.append(
            {"model": model, "contents": copy.deepcopy(contents), "config": copy.deepcopy(config)}
        )
        # A copy per call: ADK writes its own call ids into the parts it is handed.
        parts = (
            copy.deepcopy(self.script[index])
            if index < len(self.script)
            else [types.Part(text="done")]
        )
        chunk = types.GenerateContentResponse(
            candidates=[
                types.Candidate(
                    content=types.Content(role="model", parts=parts),
                    finish_reason=types.FinishReason.STOP,
                )
            ],
            usage_metadata=validate.SCRIPTED_USAGE,
        )

        async def chunks() -> Any:
            yield chunk

        return chunks()

    async def _create(self, *, model: str, config: Any) -> types.CachedContent:
        self.creates.append({"model": model, "config": copy.deepcopy(config)})
        return types.CachedContent(name=f"cachedContents/scripted-{len(self.creates)}")


# Gemini signs the first function call of each turn; a replay without it is rejected.
SIGNATURE = b"scripted-thought-signature"


def scripted(steps: Sequence[Step]) -> Scripted:
    """A client whose nth answer is the function calls of `steps[n]`, the first signed."""
    return Scripted(
        [
            [
                types.Part(
                    function_call=types.FunctionCall(name=name, args=args),
                    thought_signature=SIGNATURE if index == 0 else None,
                )
                for index, (name, args) in enumerate(step)
            ]
            for step in steps
        ]
    )


@contextlib.contextmanager
def adk_client(client: Client) -> Iterator[None]:
    """`entry.run_entry`'s seam with the planner's own model talking to `client`."""
    original = entry.build_seam

    def build_seam(**kwargs: Any) -> Any:
        seam = original(**kwargs)
        seam._agent.model.client = client
        return seam

    entry.build_seam = build_seam
    try:
        yield
    finally:
        entry.build_seam = original


async def run_adk(
    task: Task,
    client: Scripted,
    *,
    answers: Sequence[Mapping[str, Any]] = (),
    prefix_cache: PrefixCache | None = None,
    translation: entry.TranslationCalls | None = None,
) -> entry.EntryRun:
    with adk_client(client.client):
        return await entry.run_entry(
            task,
            load_recording(task.id),
            prefix_cache=prefix_cache,
            timeout_s=60.0,
            answers=answers,
            translation=translation,
        )


async def run_pai(
    task: Task,
    client: Scripted,
    *,
    answers: Sequence[Mapping[str, Any]] = (),
    prefix_cache: PrefixCache | None = None,
    keep_thoughts: bool = False,
    translation: entry.TranslationCalls | None = None,
) -> pai_entry.PaiEntry:
    return await pai_entry.run_pai_entry(
        task,
        load_recording(task.id),
        prefix_cache=prefix_cache,
        timeout_s=60.0,
        client=client.client,
        answers=answers,
        keep_thoughts=keep_thoughts,
        translation=translation,
    )


# --- What a run ended as --------------------------------------------------------------

_NO_TASK: Any = type("_Task", (), {"id": ""})()


def outcome(run: entry.EntryRun) -> dict[str, Any]:
    """What a run ended as, in the record's own terms (`entry.entry_record`), without the
    ids and times that differ between any two runs."""
    record = entry.entry_record(
        task=_NO_TASK, run=run, arm="", rep=0, model="", revision="", experiment=""
    )
    return {
        "stop_reason": record["stop_reason"],
        "accepted": record["accepted"],
        "plan": record["plan"],
        "tool_calls": record["tool_calls"],
        "proposals": [
            {key: proposal[key] for key in ("outcome", "code", "kinds", "args")}
            for proposal in record["proposals"]
        ],
        "responses": [call["response"] for call in entry.tool_calls(run.events)],
        "model_calls": len(record["calls"]),
        "budget_closed": record["budget_closed"],
        "surface_misses": record["surface_misses"],
        "error": None if record["error"] is None else record["error"].splitlines()[0],
    }


# --- Requests, as sent ----------------------------------------------------------------


def jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, bytes):
        return base64.b64encode(value).decode()
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [jsonable(item) for item in value]
    return value


def config_dict(config: Any) -> dict[str, Any]:
    """A request config as JSON, `None` fields dropped, whichever form a framework passed."""
    return json.loads(
        json.dumps(
            types.GenerateContentConfig.model_validate(jsonable(config)).model_dump(
                mode="json", exclude_none=True
            )
        )
    )


def contents_list(contents: Any, *, ids: bool = True) -> list[dict[str, Any]]:
    """The request's contents as JSON. Without `ids`, the tool call ids Pydantic AI replays
    are left out, since ADK sends none."""
    dumped = [
        types.Content.model_validate(jsonable(content)).model_dump(mode="json", exclude_none=True)
        for content in contents
    ]
    if not ids:
        for content in dumped:
            for part in content.get("parts", []):
                for key in ("function_call", "function_response"):
                    part.get(key, {}).pop("id", None)
    return dumped


def instruction_text(config: Mapping[str, Any]) -> str | None:
    instruction = config.get("system_instruction")
    if instruction is None or isinstance(instruction, str):
        return instruction
    return "".join(part.get("text", "") for part in instruction.get("parts", []))


def declarations(config: Mapping[str, Any]) -> dict[str, Any]:
    """Each function declaration of the request by name, however the tools are grouped."""
    return {
        declaration["name"]: declaration
        for tool in config.get("tools") or []
        for declaration in tool.get("function_declarations") or []
    }


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def diff(left: Any, right: Any, path: str = "") -> list[dict[str, Any]]:
    """Field-level differences between two JSON values, ADK's on the left."""
    if isinstance(left, dict) and isinstance(right, dict):
        found = []
        for key in sorted(set(left) | set(right), key=str):
            where = f"{path}.{key}" if path else str(key)
            if key not in right:
                found.append({"path": where, "adk": left[key], "pai": "<absent>"})
            elif key not in left:
                found.append({"path": where, "adk": "<absent>", "pai": right[key]})
            else:
                found += diff(left[key], right[key], where)
        return found
    if isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
        return [
            d
            for i, (a, b) in enumerate(zip(left, right, strict=True))
            for d in diff(a, b, f"{path}[{i}]")
        ]
    if canonical(left) != canonical(right):
        return [{"path": path, "adk": left, "pai": right}]
    return []


def request_differences(adk: Mapping[str, Any], pai: Mapping[str, Any]) -> dict[str, Any]:
    """Every difference between two requests sent at the client, ADK's first, by field:
    the system instruction as text, each function declaration, how the declarations are
    grouped into tools, the tool config, the contents without call ids, and every other
    config field."""
    adk_config, pai_config = config_dict(adk["config"]), config_dict(pai["config"])
    compared = ("system_instruction", "tools", "tool_config")
    adk_declarations, pai_declarations = declarations(adk_config), declarations(pai_config)
    fields = {
        "system_instruction": diff(instruction_text(adk_config), instruction_text(pai_config)),
        "declarations": diff(adk_declarations, pai_declarations),
        "declaration_order": diff(list(adk_declarations), list(pai_declarations)),
        "tools_grouping": diff(
            [len(t.get("function_declarations") or []) for t in adk_config.get("tools") or []],
            [len(t.get("function_declarations") or []) for t in pai_config.get("tools") or []],
        ),
        "tool_config": diff(adk_config.get("tool_config"), pai_config.get("tool_config")),
        "contents": diff(
            contents_list(adk["contents"], ids=False), contents_list(pai["contents"], ids=False)
        ),
        "other_config": diff(
            {k: v for k, v in adk_config.items() if k not in compared},
            {k: v for k, v in pai_config.items() if k not in compared},
        ),
    }
    return {name: found for name, found in fields.items() if found}

"""O6: the first Gemini request each framework sends for the same turn, offline.

Both arms run the real root workflow under a bare ADK `Runner` with
`StreamingMode.SSE`, as Studio runs it, on one session holding today's canvas
and conversation digests. Each orchestrator's model talks to its own scripted
google-genai client, which records what it was sent and answers "done", so no
request leaves the process.

- `adk`: dev's `LlmAgent` from `create_orchestrator_agent`, on its own
  `BoundedGemini`, given the scripted client.
- `pai`: `pai_orchestrate_node` on Pydantic AI's `GoogleModel` over the scripted
  client, with Gemini's profile less its JSON schema transformer (which would
  rewrite the declarations ADK sends as they are), with inline system prompts on
  (off, Pydantic AI adds a sentence about `<system>` tags to the instruction) and
  strict tool definitions off (on, it sends `VALIDATED` where ADK means `AUTO`).

The check compares the system instruction (ADK's identity line already stripped
by `_without_adk_identity`), every declaration as canonical JSON, their order and
grouping, the tool config, the thinking config, the contents and every other
config field. Each difference must match one entry of `PAI_BEHAVIOUR`, which says
why Pydantic AI sends it so.
"""

import copy
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pytest
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.genai import Client, types
from harness.graph import workflow as workflow_module
from orchestrator.agent import _retrying
from orchestrator_node import pai_orchestrate_node
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.profiles.google import google_model_profile
from pydantic_ai.providers.google import GoogleProvider
from tests import orchestrator_fakes
from tests.orchestrator_fakes import open_session, orchestrator_runner

MODEL = "gemini-2.5-flash"
QUESTION = "How much aid did Kenya get in 2020?"
DIGESTS = {
    "canvas_digest": json.dumps({"cards": 2, "hidden": 0}),
    "conversation_digest": json.dumps([{"turn": 1, "asked": "Kenya's GDP"}]),
}
_GOOGLE = "pydantic_ai/models/google.py"


@dataclass
class Scripted:
    """A google-genai client that records each call and answers "done"."""

    calls: list[dict[str, Any]] = field(default_factory=list)
    client: Client = field(init=False)

    def __post_init__(self) -> None:
        self.client = Client(api_key="offline-parity")
        self.client.aio.models.generate_content_stream = self._stream  # ty: ignore[invalid-assignment]

    async def _stream(self, *, model: str, contents: Any, config: Any) -> Any:
        self.calls.append(
            {"model": model, "contents": copy.deepcopy(contents), "config": copy.deepcopy(config)}
        )
        chunk = types.GenerateContentResponse(
            candidates=[
                types.Candidate(
                    content=types.Content(role="model", parts=[types.Part(text="done")]),
                    finish_reason=types.FinishReason.STOP,
                )
            ],
            usage_metadata=types.GenerateContentResponseUsageMetadata(
                prompt_token_count=10, candidates_token_count=1, total_token_count=11
            ),
        )

        async def chunks() -> Any:
            yield chunk

        return chunks()


def google_model(client: Client) -> GoogleModel:
    profile = dict(google_model_profile(MODEL) or {})
    profile["json_schema_transformer"] = None
    profile["supports_inline_system_prompts"] = True
    profile["google_supports_strict_tool_definition"] = False
    return GoogleModel(MODEL, provider=GoogleProvider(client=client), profile=profile)  # ty: ignore[invalid-argument-type]


async def first_request(
    monkeypatch: pytest.MonkeyPatch, arm: str, *, break_declaration: str | None = None
) -> dict[str, Any]:
    """Run one turn with `arm`'s orchestrator; the first request its client saw.
    `break_declaration` names a tool whose description the `pai` arm changes."""
    scripted = Scripted()
    with monkeypatch.context() as patch:
        if arm == "adk":

            def adk_model(script: Any) -> Any:
                model = _retrying(MODEL)
                model.client = scripted.client  # ty: ignore[invalid-assignment]
                return model

            patch.setattr(orchestrator_fakes, "ScriptLlm", adk_model)
        else:

            def factory(*, orchestrator: Any) -> Any:
                if break_declaration is not None:
                    tool = next(t for t in orchestrator.tools if t.name == break_declaration)
                    original = tool._get_declaration

                    def broken() -> types.FunctionDeclaration:
                        declaration = original()
                        assert declaration is not None
                        declaration.description = f"{declaration.description} Changed."
                        return declaration

                    patch.setattr(tool, "_get_declaration", broken)
                return pai_orchestrate_node(
                    orchestrator=orchestrator, model=google_model(scripted.client)
                )

            patch.setattr(workflow_module, "orchestrate_node", factory)
        runner, _ = orchestrator_runner()
        await open_session(runner, **DIGESTS)
        async for _ in runner.run_async(
            user_id="u",
            session_id="s",
            new_message=types.UserContent(QUESTION),
            run_config=RunConfig(streaming_mode=StreamingMode.SSE),
        ):
            pass
    assert scripted.calls, f"{arm} made no model call"
    return scripted.calls[0]


# ---------------------------------------------------------------------------
# Comparing two requests
# ---------------------------------------------------------------------------


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    return value


def config_dict(config: Any) -> dict[str, Any]:
    """A request config as JSON, `None` fields dropped, whichever form was passed."""
    validated = types.GenerateContentConfig.model_validate(_jsonable(config))
    return json.loads(json.dumps(validated.model_dump(mode="json", exclude_none=True)))


def contents_list(contents: Any) -> list[dict[str, Any]]:
    return [
        types.Content.model_validate(_jsonable(content)).model_dump(mode="json", exclude_none=True)
        for content in contents
    ]


def instruction_text(config: Mapping[str, Any]) -> str | None:
    instruction = config.get("system_instruction")
    if instruction is None or isinstance(instruction, str):
        return instruction
    return "".join(part.get("text", "") for part in instruction.get("parts", []))


def declarations(config: Mapping[str, Any]) -> dict[str, Any]:
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


_COMPARED = ("system_instruction", "tools", "tool_config", "thinking_config")


def compare(adk: Mapping[str, Any], pai: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    adk_config, pai_config = config_dict(adk["config"]), config_dict(pai["config"])
    adk_decls, pai_decls = declarations(adk_config), declarations(pai_config)
    fields = {
        "model": diff(adk["model"], pai["model"]),
        "system_instruction": diff(instruction_text(adk_config), instruction_text(pai_config)),
        "declarations": diff(
            {name: canonical(d) for name, d in adk_decls.items()},
            {name: canonical(d) for name, d in pai_decls.items()},
        ),
        "declaration_order": diff(list(adk_decls), list(pai_decls)),
        "tools_grouping": diff(
            [len(t.get("function_declarations") or []) for t in adk_config.get("tools") or []],
            [len(t.get("function_declarations") or []) for t in pai_config.get("tools") or []],
        ),
        "tool_config": diff(adk_config.get("tool_config"), pai_config.get("tool_config")),
        "thinking_config": diff(
            adk_config.get("thinking_config"), pai_config.get("thinking_config")
        ),
        "contents": diff(contents_list(adk["contents"]), contents_list(pai["contents"])),
        "other_config": diff(
            {k: v for k, v in adk_config.items() if k not in _COMPARED},
            {k: v for k, v in pai_config.items() if k not in _COMPARED},
        ),
    }
    return {name: found for name, found in fields.items() if found}


# ---------------------------------------------------------------------------
# What Pydantic AI sends differently, and why
# ---------------------------------------------------------------------------

PAI_BEHAVIOUR: dict[str, str] = {
    "instruction_stripped": (
        "pydantic_ai/_instructions.py:256 strips each instruction string, so the instruction "
        "goes without the trailing newline `INSTRUCTION` ends with"
    ),
    "empty_parameters": (
        f"{_GOOGLE}:2519-2523 always sends `parameters_json_schema`, so `explain_dfo`, which ADK "
        "declares with no parameters, goes with an empty object schema"
    ),
    "tools_grouping": f"{_GOOGLE} `_get_tools` puts each function declaration in a `Tool` of its own",
    "tool_config": (
        f"{_GOOGLE} `_get_tool_config` always sends a function calling config with function "
        "tools, AUTO with strict definitions off; ADK sends none, which Gemini reads as AUTO"
    ),
    "headers": f"{_GOOGLE} sends its own `http_options` headers; ADK sends its tracking headers",
    "response_modalities": f"{_GOOGLE} always sends `response_modalities` ['TEXT']",
    "automatic_continuation": (
        "dev sets `automatic_continuation=True` on the agent's config; `GoogleModelSettings` has "
        "no field for it, so Pydantic AI sends none and the genai default (true) applies"
    ),
    "one_user_content": (
        "Pydantic AI merges consecutive requests (`messages._merge_consecutive_messages`) and "
        "maps one request to one `Content`, so the digests and the question go as two parts "
        "of one user content; `prepend_digests` inserts the digests as a content of their own"
    ),
}

_HEADERS = ("http_options.headers", "http_options")


def explain(
    name: str, found: Sequence[Mapping[str, Any]], *, pai_contents: Any
) -> list[str] | None:
    """The `PAI_BEHAVIOUR` entries that account for every difference in `found`,
    or `None` when any one of them is unexplained."""
    reasons: set[str] = set()
    for entry in found:
        path, adk, pai = entry["path"], entry["adk"], entry["pai"]
        if name == "system_instruction" and isinstance(adk, str) and adk.rstrip() == pai:
            reasons.add("instruction_stripped")
        elif name == "declarations" and _empty_parameters(adk, pai):
            reasons.add("empty_parameters")
        elif name == "tools_grouping" and isinstance(adk, list) and pai == [1] * sum(adk):
            reasons.add("tools_grouping")
        elif (
            name == "tool_config"
            and adk is None
            and pai == {"function_calling_config": {"mode": "AUTO"}}
        ):
            reasons.add("tool_config")
        elif name == "other_config" and path.startswith(_HEADERS):
            reasons.add("headers")
        elif name == "other_config" and path == "response_modalities" and pai == ["TEXT"]:
            reasons.add("response_modalities")
        elif name == "other_config" and path == "automatic_continuation" and pai == "<absent>":
            reasons.add("automatic_continuation")
        elif name == "contents" and _one_user_content(entry, pai_contents):
            reasons.add("one_user_content")
        else:
            return None
    return sorted(reasons)


def _empty_parameters(adk: Any, pai: Any) -> bool:
    left, right = json.loads(adk), json.loads(pai)
    return "parameters_json_schema" not in left and right == {
        **left,
        "parameters_json_schema": {"properties": {}, "type": "object"},
    }


def _one_user_content(entry: Mapping[str, Any], pai_contents: Any) -> bool:
    """ADK's user contents, one after another, are the parts of Pydantic AI's one."""
    adk = entry["adk"]
    if entry["path"] != "" or not isinstance(adk, list) or len(pai_contents) != 1:
        return False
    merged = [part for content in adk for part in content["parts"]]
    return all(c["role"] == "user" for c in adk) and pai_contents[0] == {
        "role": "user",
        "parts": merged,
    }


def classify(
    fields: Mapping[str, Any], *, pai_contents: Any
) -> tuple[dict[str, list[str]], list[str]]:
    explained: dict[str, list[str]] = {}
    unexplained: list[str] = []
    for name, found in fields.items():
        reasons = explain(name, found, pai_contents=pai_contents)
        if reasons is None:
            unexplained.append(name)
        else:
            explained[name] = reasons
    return explained, unexplained


async def _check(monkeypatch: pytest.MonkeyPatch, **pai_args: Any) -> Any:
    adk = await first_request(monkeypatch, "adk")
    pai = await first_request(monkeypatch, "pai", **pai_args)
    fields = compare(adk, pai)
    explained, unexplained = classify(fields, pai_contents=contents_list(pai["contents"]))
    return adk, pai, fields, explained, unexplained


async def test_o6_the_first_request_differs_only_where_pydantic_ai_sends_otherwise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adk, _, fields, explained, unexplained = await _check(monkeypatch)

    for name, found in fields.items():
        for entry in found:
            print(f"{name} {entry['path']}: adk={_short(entry['adk'])} pai={_short(entry['pai'])}")
    print(f"explained: {explained}")
    assert unexplained == [], {name: fields[name] for name in unexplained}
    assert explained == {
        "contents": ["one_user_content"],
        "declarations": ["empty_parameters"],
        "other_config": ["automatic_continuation", "headers", "response_modalities"],
        "system_instruction": ["instruction_stripped"],
        "tool_config": ["tool_config"],
        "tools_grouping": ["tools_grouping"],
    }
    # Every declaration but `explain_dfo`'s, the thinking config and the
    # declaration order are byte for byte the same.
    assert [e["path"] for e in fields["declarations"]] == ["explain_dfo"]
    assert "thinking_config" not in fields and "declaration_order" not in fields
    adk_config = config_dict(adk["config"])
    assert 'Your internal name is "dfo_orchestrator"' not in str(instruction_text(adk_config))
    print(f"declarations: {list(declarations(adk_config))}")
    print(f"thinking: {adk_config.get('thinking_config')}")


async def test_o6_control_a_changed_declaration_is_unexplained(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, fields, _, unexplained = await _check(monkeypatch, break_declaration="search")

    print(f"broken declaration: {[e['path'] for e in fields['declarations']]}")
    assert unexplained == ["declarations"]
    assert [e["path"] for e in fields["declarations"]] == ["explain_dfo", "search"]


def _short(value: Any) -> str:
    text = canonical(value)
    return text if len(text) <= 160 else text[:160] + "..."

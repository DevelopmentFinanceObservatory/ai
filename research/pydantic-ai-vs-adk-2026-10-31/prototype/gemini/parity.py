"""Offline parity of the Pydantic AI planner entry with the ADK one: pass line W1.

No model call and no network. Both arms run against the same scripted google-genai
client, so each request is captured at the call the planner's model makes,
`client.aio.models.generate_content_stream`, and nothing is sent. Per task:

- outcomes: each of the task's plans (the reference and every alternative) proposed
  through both entries, as `validate.py` proposes them through the ADK one; the PAI
  entry must raise none of `validate.py`'s faults and end as the ADK entry does;
- first request: ADK's and Pydantic AI's first model request, uncached, compared on
  the system instruction, every function declaration (canonical JSON), the tool
  config, the contents, every other config field, and the session state at the call;
- prefix: both `like-adk` prefixes sent to `PrefixCache`, compared by
  `harness.prefix_cache.prefix_key` and, with each side's wire form normalised, by
  content; and the cached request's config;
- and once, thought replay: a scripted turn carrying an unsigned thought summary and a
  signed function call, and the next request each side sends.

    uv run --with 'pydantic-ai-slim[google]==2.64.0' --package harness \\
        python research/pai-prototype/gemini/parity.py [--only ID,...] [--break NAME]

Writes `parity.json` beside this file (or `--out`). `--break` runs a deliberately broken
variant, to show each check can fail.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import copy
import json
import logging
import sys
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[2] / "evals" / "planner_entry")]

import entry  # noqa: E402
import pai_entry  # noqa: E402
import pai_planner  # noqa: E402
import validate  # noqa: E402
from entry import Task, load_recording, load_tasks, quiet_logs  # noqa: E402
from google.adk.workflow.utils._workflow_hitl_utils import (  # noqa: E402
    REQUEST_INPUT_FUNCTION_CALL_NAME,
)
from google.genai import Client, types  # noqa: E402
from harness.prefix_cache import PrefixCache  # noqa: E402
from submission.service import COMBINATION_OPTIONS, COMBINE_OPTION  # noqa: E402

BREAKS = ("instruction", "declaration", "contents", "state", "outcome", "thoughts", "prefix")

Step = list[tuple[str, dict[str, Any]]]

# A thought summary the scripted model writes, which no later request may carry.
THOUGHT_TEXT = "Scripted thought summary that must not be replayed."
THOUGHT_SIGNATURE = b"scripted-signature"


# --- The scripted google-genai client -------------------------------------------------


@dataclass
class Scripted:
    """A google-genai client that answers each streamed call with the next scripted
    turn and records what it was sent, and each cache it was asked to create."""

    script: list[list[types.Part]] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)
    creates: list[dict[str, Any]] = field(default_factory=list)
    client: Client = field(init=False)

    def __post_init__(self) -> None:
        self.client = Client(api_key="offline-parity")
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
        return types.CachedContent(name=f"cachedContents/offline-{len(self.creates)}")


def calls_of(steps: Sequence[Step]) -> list[list[types.Part]]:
    return [
        [types.Part(function_call=types.FunctionCall(name=name, args=args)) for name, args in step]
        for step in steps
    ]


class RecordingCache(PrefixCache):
    """`PrefixCache`, recording each key looked up."""

    def __init__(self) -> None:
        super().__init__()
        self.keys: list[str] = []

    async def cache_name(self, key: str, **kwargs: Any) -> str | None:
        self.keys.append(key)
        return await super().cache_name(key, **kwargs)


# --- Running each arm -----------------------------------------------------------------


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


@contextlib.contextmanager
def state_at_first_call(snapshots: list[dict[str, Any]]) -> Iterator[None]:
    """Record the session state each arm's planner holds as its first model call goes out,
    through the probe's before-model callback, which both arms call there."""
    original = entry.Probe.before_model

    def before_model(self: entry.Probe, callback_context: Any, llm_request: Any) -> Any:
        if not self.calls and callback_context is not None:
            state = callback_context.state
            snapshots.append(json.loads(json.dumps(_plain(state), default=str)))
        return original(self, callback_context, llm_request)

    entry.Probe.before_model = before_model  # type: ignore[method-assign]
    try:
        yield
    finally:
        entry.Probe.before_model = original  # type: ignore[method-assign]


def _timeless(value: Any) -> Any:
    """`value` without `observed_at`, the wall-clock time the replayed search stamps on
    each reading, which differs between any two runs."""
    if isinstance(value, dict):
        return {key: _timeless(item) for key, item in value.items() if key != "observed_at"}
    if isinstance(value, list):
        return [_timeless(item) for item in value]
    return value


def _plain(state: Any) -> dict[str, Any]:
    return dict(state.to_dict() if hasattr(state, "to_dict") else state)


async def run_adk(
    task: Task,
    *,
    scripted: Scripted,
    prefix_cache: PrefixCache | None = None,
    answers: Sequence[Mapping[str, Any]] = (),
) -> entry.EntryRun:
    with adk_client(scripted.client):
        return await entry.run_entry(
            task,
            load_recording(task.id),
            prefix_cache=prefix_cache,
            timeout_s=60.0,
            answers=answers,
        )


async def run_pai(
    task: Task,
    *,
    scripted: Scripted,
    mode: pai_planner.Mode = "like-adk",
    prefix_cache: PrefixCache | None = None,
    answers: Sequence[Mapping[str, Any]] = (),
    keep_thoughts: bool = False,
) -> pai_entry.PaiEntry:
    return await pai_entry.run_pai_entry(
        task,
        load_recording(task.id),
        mode=mode,
        prefix_cache=prefix_cache,
        timeout_s=60.0,
        client=scripted.client,
        answers=answers,
        keep_thoughts=keep_thoughts,
    )


# --- Outcomes -------------------------------------------------------------------------


def _proposal_script(
    payload: Mapping[str, Any], refusal: str | None
) -> tuple[list[Step], list[dict[str, Any]]]:
    """`validate.propose`'s script and answers for one plan."""
    propose = ("propose_plan", dict(payload))
    if refusal is None:
        return [[propose]], []
    ask = (
        REQUEST_INPUT_FUNCTION_CALL_NAME,
        {
            "message": "Do these two datasets add up without counting a flow twice?",
            "options": [dict(option) for option in COMBINATION_OPTIONS],
        },
    )
    return [[propose], [ask], [propose]], [{"result": {"spec_id": COMBINE_OPTION}}]


def outcome(run: entry.EntryRun) -> dict[str, Any]:
    """What a run ended as, in the record's own terms (`entry.entry_record`)."""
    record = entry.entry_record(
        task=_TASK_PLACEHOLDER, run=run, arm="", rep=0, model="", revision="", experiment=""
    )
    return {
        "stop_reason": record["stop_reason"],
        "accepted": record["accepted"],
        "plan_kinds": record["plan_kinds"],
        "plan": record["plan"],
        "tool_calls": record["tool_calls"],
        "proposals": [
            {key: proposal[key] for key in ("outcome", "code", "kinds", "args")}
            for proposal in record["proposals"]
        ],
        "model_calls": len(record["calls"]),
        "budget_closed": record["budget_closed"],
        "surface_misses": record["surface_misses"],
        "error": None if record["error"] is None else record["error"].splitlines()[0],
    }


# `entry_record` reads only the task's id.
_TASK_PLACEHOLDER: Any = type("_T", (), {"id": ""})()


async def check_outcomes(task: Task) -> dict[str, Any]:
    plans = [("reference", task.reference_payload, task.first_proposal_refusal)]
    plans += [
        (f"alternative {index}", alt.payload, alt.first_proposal_refusal)
        for index, alt in enumerate(task.alternative_payloads)
    ]
    results: dict[str, Any] = {}
    for label, payload, refusal in plans:
        script, answers = _proposal_script(payload, refusal)
        adk = await run_adk(task, scripted=Scripted(calls_of(script)), answers=answers)
        pai = await run_pai(task, scripted=Scripted(calls_of(script)), answers=answers)
        adk_outcome, pai_outcome = outcome(adk), outcome(pai.run)
        faults = validate._faults(pai.run, payload, refusal)
        results[label] = {
            "equal": adk_outcome == pai_outcome and not faults,
            "pai_faults": faults,
            "adk_faults": validate._faults(adk, payload, refusal),
            **({} if adk_outcome == pai_outcome else {"diff": diff(adk_outcome, pai_outcome)}),
        }
    return results


# --- Requests -------------------------------------------------------------------------


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, bytes):
        return base64.b64encode(value).decode()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    return value


def config_dict(config: Any) -> dict[str, Any]:
    """A request config as JSON, `None` fields dropped, whichever form a framework passed."""
    return json.loads(
        json.dumps(
            types.GenerateContentConfig.model_validate(_jsonable(config)).model_dump(
                mode="json", exclude_none=True
            )
        )
    )


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
                found.append({"path": where, "adk": _short(left[key]), "pai": "<absent>"})
            elif key not in left:
                found.append({"path": where, "adk": "<absent>", "pai": _short(right[key])})
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
        return [{"path": path, "adk": _short(left), "pai": _short(right)}]
    return []


def _short(value: Any) -> Any:
    text = canonical(value)
    return value if len(text) <= 300 else text[:300] + "..."


# The request config fields the first-request check compares one by one; every other
# field is compared together under `other_config`.
_COMPARED = ("system_instruction", "tools", "tool_config")


async def check_first_request(task: Task) -> dict[str, Any]:
    adk_scripted, pai_scripted = Scripted(), Scripted()
    states: list[dict[str, Any]] = []
    with state_at_first_call(states):
        adk_run = await run_adk(task, scripted=adk_scripted)
        pai_run = await run_pai(task, scripted=pai_scripted)
    if adk_run.error or pai_run.run.error:
        return {"equal": False, "error": {"adk": adk_run.error, "pai": pai_run.run.error}}
    adk_call, pai_call = adk_scripted.calls[0], pai_scripted.calls[0]
    adk_config, pai_config = config_dict(adk_call["config"]), config_dict(pai_call["config"])
    adk_decls, pai_decls = declarations(adk_config), declarations(pai_config)
    fields: dict[str, Any] = {
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
        "contents": diff(contents_list(adk_call["contents"]), contents_list(pai_call["contents"])),
        "other_config": diff(
            {k: v for k, v in adk_config.items() if k not in _COMPARED},
            {k: v for k, v in pai_config.items() if k not in _COMPARED},
        ),
        "state": diff(*(_timeless(state) for state in states))
        if len(states) == 2
        else [{"path": "", "adk": "?", "pai": "?"}],
    }
    found = {name: found for name, found in fields.items() if found}
    return {"fields": found, "equal": not found, **classify(found)}


def _prefix_content(create: Mapping[str, Any]) -> dict[str, Any]:
    """A cache create's prefix with each framework's wire form normalised: the system
    instruction as text, the declarations as one list, the tool config as sent."""
    raw = _jsonable(create["config"])
    config = config_dict({k: raw.get(k) for k in ("system_instruction", "tools", "tool_config")})
    return {
        "system_instruction": instruction_text(config),
        "declarations": [canonical(d) for d in declarations(config).values()],
        "tool_config": config.get("tool_config"),
    }


async def check_prefix(task: Task) -> dict[str, Any]:
    adk_scripted, pai_scripted = Scripted(), Scripted()
    adk_cache, pai_cache = RecordingCache(), RecordingCache()
    await run_adk(task, scripted=adk_scripted, prefix_cache=adk_cache)
    await run_pai(task, scripted=pai_scripted, mode="like-adk", prefix_cache=pai_cache)
    if not adk_scripted.creates or not pai_scripted.creates:
        return {"equal": False, "error": "a side created no cache"}
    adk_request = config_dict(adk_scripted.calls[0]["config"])
    pai_request = config_dict(pai_scripted.calls[0]["config"])
    fields = {
        "prefix_key": diff(adk_cache.keys[0], pai_cache.keys[0]),
        "prefix_content": diff(
            _prefix_content(adk_scripted.creates[0]), _prefix_content(pai_scripted.creates[0])
        ),
        "cached_request": diff(
            {k: v for k, v in adk_request.items() if k != "cached_content"},
            {k: v for k, v in pai_request.items() if k != "cached_content"},
        ),
        "cached_request_carries_cache": diff(
            "cached_content" in adk_request, "cached_content" in pai_request
        ),
    }
    found = {name: found for name, found in fields.items() if found}
    return {"fields": found, "equal": not found, **classify(found)}


async def check_thoughts(task: Task, *, keep_thoughts: bool) -> dict[str, Any]:
    """The request after a turn that carries an unsigned thought summary and a signed
    function call, from each arm: no summary text, the signature kept."""
    turn = [
        types.Part(text=THOUGHT_TEXT, thought=True),
        types.Part(
            function_call=types.FunctionCall(name="reject_plan", args={}),
            thought_signature=THOUGHT_SIGNATURE,
        ),
    ]
    adk_scripted, pai_scripted = Scripted([turn]), Scripted([turn])
    await run_adk(task, scripted=adk_scripted)
    await run_pai(task, scripted=pai_scripted, keep_thoughts=keep_thoughts)
    adk_second = contents_list(adk_scripted.calls[1]["contents"])
    pai_second = contents_list(pai_scripted.calls[1]["contents"])
    signature = _jsonable(THOUGHT_SIGNATURE)

    def summary(contents: list[dict[str, Any]]) -> dict[str, bool]:
        parts = [part for content in contents for part in content.get("parts", [])]
        return {
            "thought_text_replayed": any(part.get("text") == THOUGHT_TEXT for part in parts),
            "signature_kept": any(
                part.get("function_call") and part.get("thought_signature") == signature
                for part in parts
            ),
        }

    adk_summary, pai_summary = summary(adk_second), summary(pai_second)
    return {
        "adk": adk_summary,
        "pai": pai_summary,
        "equal": adk_summary == pai_summary
        and adk_summary == {"thought_text_replayed": False, "signature_kept": True},
        "second_request_contents": diff(adk_second, pai_second),
        **classify({"second_request_contents": diff(adk_second, pai_second)}),
    }


# --- Deliberately broken variants -----------------------------------------------------


@contextlib.contextmanager
def broken(name: str | None) -> Iterator[dict[str, Any]]:
    """Run with `name` broken on the PAI side, to show the check that covers it fails."""
    options: dict[str, Any] = {"keep_thoughts": False}
    patches: list[tuple[Any, str, Any]] = []

    def patch(owner: Any, attribute: str, value: Any) -> None:
        patches.append((owner, attribute, getattr(owner, attribute)))
        setattr(owner, attribute, value)

    if name == "instruction":
        of = pai_planner.PlannerParts.of

        def instruction_of(seam: Any) -> Any:
            parts = of(seam)
            return pai_planner.PlannerParts(
                parts.instruction + " X", parts.tools, parts.declarations
            )

        patch(pai_planner.PlannerParts, "of", staticmethod(instruction_of))
    elif name == "declaration":
        of = pai_planner.PlannerParts.of

        def declaration_of(seam: Any) -> Any:
            parts = of(seam)
            mutated = [
                d.model_copy(update={"description": (d.description or "") + " "})
                if d.name == "propose_plan"
                else d
                for d in parts.declarations
            ]
            return pai_planner.PlannerParts(parts.instruction, parts.tools, tuple(mutated))

        patch(pai_planner.PlannerParts, "of", staticmethod(declaration_of))
    elif name == "contents":
        patch(pai_entry, "QUESTION_INSTRUCTIONS", None)
    elif name == "state":
        original = pai_entry.orchestrator_state

        async def orchestrator_state(task: Task, resolver: Any) -> dict[str, Any]:
            state = await original(task, resolver)
            state.pop(next(iter(state)))
            return state

        patch(pai_entry, "orchestrator_state", orchestrator_state)
    elif name == "outcome":
        patch(pai_planner, "adk_events", lambda messages: [])
    elif name == "thoughts":
        options["keep_thoughts"] = True
    elif name == "prefix":
        original_prefix = pai_planner.PlannerModel.prefix

        def prefix(self: Any, config: Any) -> Any:
            built = original_prefix(self, config)
            built.tool_config = types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode=types.FunctionCallingConfigMode.AUTO
                )
            )
            return built

        patch(pai_planner.PlannerModel, "prefix", prefix)
    try:
        yield options
    finally:
        for owner, attribute, value in reversed(patches):
            setattr(owner, attribute, value)


# --- Differences that are Pydantic AI's own ------------------------------------------

_GOOGLE = "pydantic_ai_slim/pydantic_ai/models/google.py"
_HEADERS = frozenset(
    {
        "http_options.headers.Content-Type",
        "http_options.headers.User-Agent",
        "http_options.headers.user-agent",
        "http_options.headers.x-goog-api-client",
    }
)

# Each difference `GoogleModel` makes on the wire that no public setting turns off, by
# the check and field it shows in, with the source line that makes it.
PAI_BEHAVIOUR: dict[str, str] = {
    "reject_plan_parameters": (
        f"{_GOOGLE}:2523 always sends `parameters_json_schema`, so `reject_plan`, which ADK "
        "declares with no parameters, goes with an empty object schema"
    ),
    "tools_grouping": f"{_GOOGLE}:1094 puts each function declaration in a `Tool` of its own",
    "tool_config": (
        f"{_GOOGLE}:1053-1074 always sends a function calling config with function tools "
        "(AUTO in like-adk mode, VALIDATED in validated mode); ADK sends none"
    ),
    "headers": f"{_GOOGLE}:1256 sends its own headers; ADK sends its tracking headers",
    "response_modalities": f"{_GOOGLE}:1250 always sends `response_modalities` ['TEXT']",
    "prefix_key": (
        "follows from the three above and from the system instruction going as a `Content` "
        f"({_GOOGLE}:1510) where ADK's goes as a string: `prefix_key` hashes each "
        "framework's own form, so the two arms never share a cache"
    ),
    "call_ids": (
        f"{_GOOGLE}:2141 and :1557 send Pydantic AI's tool call id on each replayed function "
        "call and response; ADK strips its own client ids"
    ),
}


def explain(name: str, found: Sequence[Mapping[str, Any]]) -> list[str] | None:
    """The `PAI_BEHAVIOUR` entries that account for every difference in `found`, or
    `None` when any one of them is unexplained."""
    reasons: set[str] = set()
    for entry_ in found:
        path, adk, pai = entry_["path"], entry_["adk"], entry_["pai"]
        if name in ("declarations", "prefix_content") and _reject_plan_schema(path, adk, pai):
            reasons.add("reject_plan_parameters")
        elif name == "tools_grouping" and adk == [5] and pai == [1, 1, 1, 1, 1]:
            reasons.add("tools_grouping")
        elif (
            name == "tool_config"
            and adk is None
            and pai
            in (
                {"function_calling_config": {"mode": "AUTO"}},
                {"function_calling_config": {"mode": "VALIDATED"}},
            )
        ):
            reasons.add("tool_config")
        elif name in ("other_config", "cached_request") and path in _HEADERS:
            reasons.add("headers")
        elif name in ("other_config", "cached_request") and path == "response_modalities":
            reasons.add("response_modalities")
        elif name == "prefix_key":
            reasons.add("prefix_key")
        elif (
            name == "second_request_contents"
            and path.endswith((".function_call.id", ".function_response.id"))
            and adk == "<absent>"
        ):
            reasons.add("call_ids")
        else:
            return None
    return sorted(reasons)


def _reject_plan_schema(path: str, adk: Any, pai: Any) -> bool:
    if path not in ("reject_plan", "declarations[4]"):
        return False
    left, right = json.loads(adk), json.loads(pai)
    return (
        left.get("name") == "reject_plan"
        and "parameters_json_schema" not in left
        and right == {**left, "parameters_json_schema": {"properties": {}, "type": "object"}}
    )


def classify(fields: Mapping[str, Any]) -> dict[str, Any]:
    """Each differing field as explained by Pydantic AI's own behaviour, or unexplained."""
    explained: dict[str, list[str]] = {}
    unexplained: list[str] = []
    for name, found in fields.items():
        reasons = explain(name, found)
        if reasons is None:
            unexplained.append(name)
        else:
            explained[name] = reasons
    return {"explained": explained, "unexplained": unexplained}


# --- Main -----------------------------------------------------------------------------


async def check(tasks: Sequence[Task], *, keep_thoughts: bool) -> dict[str, Any]:
    report: dict[str, Any] = {"tasks": {}}
    for task in tasks:
        first = await check_first_request(task)
        prefix = await check_prefix(task)
        outcomes = await check_outcomes(task)
        equal = first["equal"] and prefix["equal"] and all(o["equal"] for o in outcomes.values())
        explained = (
            not first.get("unexplained", ["error"])
            and not prefix.get("unexplained", ["error"])
            and all(o["equal"] for o in outcomes.values())
        )
        report["tasks"][task.id] = {
            "equal": equal,
            "explained": explained,
            "first_request": first,
            "prefix": prefix,
            "outcomes": outcomes,
        }
        mark = "same" if equal else "pai " if explained else "DIFF"
        print(f"{mark}  {task.id}", file=sys.stderr)
    report["thoughts"] = await check_thoughts(tasks[0], keep_thoughts=keep_thoughts)
    return report


def summarise(report: Mapping[str, Any]) -> dict[str, Any]:
    """Counts per check, and every field that differs with how many tasks it differs on."""
    tasks = report["tasks"].values()
    differing: dict[str, int] = {}
    for result in tasks:
        for check_name in ("first_request", "prefix"):
            for name in result[check_name].get("fields", {}):
                key = f"{check_name}.{name}"
                differing[key] = differing.get(key, 0) + 1
    plans = [o for result in tasks for o in result["outcomes"].values()]
    return {
        "tasks": len(report["tasks"]),
        "plans_proposed": len(plans),
        "plans_same_outcome": sum(o["equal"] for o in plans),
        "first_request_equal": sum(r["first_request"]["equal"] for r in tasks),
        "prefix_equal": sum(r["prefix"]["equal"] for r in tasks),
        "fields_differing": dict(sorted(differing.items())),
        "tasks_every_difference_explained": sum(r["explained"] for r in tasks),
        "unexplained": sorted(
            {
                f"{check_name}.{name}"
                for r in tasks
                for check_name in ("first_request", "prefix")
                for name in r[check_name].get("unexplained", ["error"])
            }
        ),
        "thoughts_equal": report["thoughts"]["equal"],
        "thoughts_unexplained": report["thoughts"]["unexplained"],
        "pai_behaviour": PAI_BEHAVIOUR,
        "pass": all(r["explained"] for r in tasks)
        and report["thoughts"]["equal"]
        and not report["thoughts"]["unexplained"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", help="comma-separated task ids")
    parser.add_argument("--break", dest="broken", choices=BREAKS)
    parser.add_argument("--out", type=Path, default=HERE / "parity.json")
    args = parser.parse_args(argv)
    quiet_logs()
    logging.getLogger("planner.skills").setLevel(logging.ERROR)
    tasks = load_tasks()
    if args.only:
        wanted = set(args.only.split(","))
        tasks = [task for task in tasks if task.id in wanted]
    with broken(args.broken) as options:
        report = asyncio.run(check(tasks, keep_thoughts=options["keep_thoughts"]))
    report["summary"] = summarise(report)
    report["broken"] = args.broken
    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = {k: v for k, v in report["summary"].items() if k != "pai_behaviour"}
    print(json.dumps(summary, indent=1))
    return 0 if report["summary"]["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())

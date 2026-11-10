"""Offline parity of the planner on Pydantic AI with the planner on ADK.

Every `evals/planner_entry` task runs through `entry.run_entry` (ADK) and
`pai_entry.run_pai_entry` (Pydantic AI), both under the real `PlannerSeam`, both against
one scripted google-genai client each. No model call and no network.

Per task, in each of its cases:

- every plan the task states (the reference and each alternative) is proposed, translated as
  production translates it over the recording, and so declared ready; a plan the task says
  is refused first is refused, the user is asked and answers, and it is proposed again;
- the entry asks the user a question, the user answers, and the plan is proposed in the
  request that answer resumes;
- both arms end the same way (`scripted.outcome`) and leave the same session state, apart
  from wall-clock fields and the `temp:` keys ADK does not persist;
- both arms send the same number of requests, and every one of them differs only by `WIRE`
  (what `GoogleModel` writes for the same content);
- the cached prefix both arms create holds the same instruction and declarations.

Once, on the first task: the argument and response gaps (`GAPS`), and a thought summary.

    uv run --with 'pydantic-ai-slim[google]==2.66.0' --package harness \\
        python research/pai-planner-live/parity/parity.py [--only ID,...] [--break NAME]

Writes `parity.json` beside this file (or `--out`) and exits 1 unless every task, gap and
control-free check agrees. `--break NAME` runs one deliberately broken Pydantic AI arm and
must exit 1 (`BREAKS`).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
import sys
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "services" / "harness").is_dir())
sys.path[:0] = [str(HERE.parent / "agent"), str(ROOT / "evals" / "planner_entry")]

import entry  # noqa: E402
import pai_call  # noqa: E402
import pai_entry  # noqa: E402
import pai_planner  # noqa: E402
import planner.agent as adk_agent  # noqa: E402
import scripted as sc  # noqa: E402
import validate  # noqa: E402
from entry import Task, TranslationCalls, load_recording, load_tasks, quiet_logs  # noqa: E402
from google.genai import types  # noqa: E402
from harness.prefix_cache import PrefixCache  # noqa: E402
from harness.turn_events import PLANNER_ASK  # noqa: E402
from planner.plan import load_plan  # noqa: E402
from submission.service import COMBINATION_OPTIONS, COMBINE_OPTION  # noqa: E402

BREAKS = (
    "instruction",
    "declaration",
    "contents",
    "tool-result",
    "end-run",
    "state",
    "thoughts",
    "prefix",
    "arguments",
    "mandatory",
    "no-parts",
)

# What Pydantic AI's `GoogleModel` sends differently from ADK's `Gemini` for the same request,
# by the field `sc.request_differences` reports it under, and why. `other_config` is listed
# by path, since a header's value carries each library's version. The set is compared exactly:
# a difference that goes away fails the check as a new one does.
WIRE: dict[str, Any] = {
    "declarations": [
        {
            "path": "reject_plan.parameters_json_schema",
            "adk": "<absent>",
            "pai": {"type": "object", "properties": {}},
        }
    ],
    "tools_grouping": [{"path": "", "adk": [4], "pai": [1, 1, 1, 1]}],
    "tool_config": [
        {"path": "", "adk": None, "pai": {"function_calling_config": {"mode": "AUTO"}}}
    ],
    "other_config": [
        "automatic_continuation",
        "http_options.headers.Content-Type",
        "http_options.headers.User-Agent",
        "http_options.headers.user-agent",
        "http_options.headers.x-goog-api-client",
        "response_modalities",
    ],
}

# Where the two arms' session states differ for a reason of their own.
STATE_PREFIX_NOT_PERSISTED = "temp:"
WALL_CLOCK = frozenset({"observed_at"})
# The translation mints a lock token per run, keys its discovery records by an id minted per
# run, and stamps the time it resolved each requirement.
RUN_UNIQUE = frozenset({"translator:discovery_lock_token"})
DISCOVERY = "translator:discovery"
RESOLVED_AT = "translator:resolved_at:"

THOUGHT_TEXT = "A thought summary the model wrote, which no later request may carry."

# The user's reply to the planner's ask, as the studio's interrupt route sends it.
ANSWER = {"result": {"answers": {"q1": {"kind": "option", "spec_id": COMBINE_OPTION}}}}

Step = sc.Step


# --- Cases ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Case:
    """One script and the user's answers to it, run through both arms."""

    label: str
    steps: Sequence[Step]
    answers: Sequence[Mapping[str, Any]] = ()
    translated: bool = False
    # The model's turns as parts, for a script no function call states (a response with no parts).
    turns: Sequence[Sequence[types.Part]] | None = None


def propose(payload: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    return ("propose_plan", dict(payload))


def propose_part(payload: Mapping[str, Any]) -> types.Part:
    return types.Part(function_call=types.FunctionCall(name="propose_plan", args=dict(payload)))


def ask(**arguments: Any) -> tuple[str, dict[str, Any]]:
    return (PLANNER_ASK, arguments)


def combination_ask() -> tuple[str, dict[str, Any]]:
    return ask(
        message="Do these two datasets add up without counting a flow twice?",
        options=[dict(option) for option in COMBINATION_OPTIONS],
    )


def plan_case(label: str, payload: Mapping[str, Any], refusal: str | None) -> Case:
    """`validate.accepted_run`'s script: the plan proposed, or refused first, then asked
    about, answered and proposed again."""
    if refusal is None:
        return Case(label, [[propose(payload)]], translated=True)
    steps = [[propose(payload)], [combination_ask()], [propose(payload)]]
    return Case(label, steps, [ANSWER], translated=True)


def task_cases(task: Task) -> list[Case]:
    """Every plan the task states, and one ask answered before the reference plan."""
    cases = [
        plan_case(payload.label, payload.body, payload.refusal)
        for payload in validate.task_payloads(task)
    ]
    reference = task.reference_payload
    cases.append(Case("ask-then-propose", [[combination_ask()], [propose(reference)]], [ANSWER]))
    return cases


# --- Running both arms ----------------------------------------------------------------


@contextlib.contextmanager
def translating() -> Iterator[None]:
    """The probe's translate step answers the translation's verdict, as production's does,
    in place of the eval's stop at the first recorded plan."""
    original = entry.Probe.translate

    async def translate(self: entry.Probe, call: Any) -> dict[str, object]:
        self.plan = load_plan(call.state)
        self.plan_call = call.call_id
        assert self.translator is not None
        self.translation = await self.translator(call)
        return self.translation

    entry.Probe.translate = translate  # type: ignore[method-assign]
    try:
        yield
    finally:
        entry.Probe.translate = original  # type: ignore[method-assign]


# What each arm's `propose_plan` body was handed, by arm: the class name of each item of each
# list argument, which shows whether ADK's conversion to `BaseModel` ran.
LIST_ARGUMENTS = ("decisions", "evidence_requirements", "declined", "answered")
seen: dict[str, list[dict[str, list[str]]]] = {"adk": [], "pai": []}


@contextlib.contextmanager
def spying(arm: str) -> Iterator[None]:
    """Note the types of the lists each `propose` of `arm` receives."""

    def wrap(original: Any) -> Any:
        def spy(call: Any, proposal: Any, **kwargs: Any) -> Any:
            seen[arm].append(
                {
                    name: [type(item).__name__ for item in getattr(proposal, name)]
                    for name in LIST_ARGUMENTS
                }
            )
            return original(call, proposal, **kwargs)

        return spy

    originals = (adk_agent.propose, pai_planner.propose)
    adk_agent.propose, pai_planner.propose = wrap(originals[0]), wrap(originals[1])
    try:
        yield
    finally:
        adk_agent.propose, pai_planner.propose = originals


@dataclass
class Arms:
    """One case run through both arms."""

    task: Task
    case: Case
    adk_client: sc.Scripted = field(init=False)
    pai_client: sc.Scripted = field(init=False)
    adk: entry.EntryRun = field(init=False)
    pai: pai_entry.PaiEntry = field(init=False)

    async def translation(self) -> TranslationCalls | None:
        if not self.case.translated:
            return None
        held = load_recording(self.task.id).get("translation") or {}
        return TranslationCalls(held, schemas=await validate.tool_schemas())

    async def run(self, *, keep_thoughts: bool = False, prefix: bool = False) -> Arms:
        seen["adk"].clear()
        seen["pai"].clear()
        if self.case.turns is None:
            self.adk_client = sc.scripted(self.case.steps)
            self.pai_client = sc.scripted(self.case.steps)
        else:
            self.adk_client = sc.Scripted([list(turn) for turn in self.case.turns])
            self.pai_client = sc.Scripted([list(turn) for turn in self.case.turns])
        with translating() if self.case.translated else contextlib.nullcontext():
            with spying("adk"):
                self.adk = await sc.run_adk(
                    self.task,
                    self.adk_client,
                    answers=self.case.answers,
                    prefix_cache=PrefixCache() if prefix else None,
                    translation=await self.translation(),
                )
            with spying("pai"):
                self.pai = await sc.run_pai(
                    self.task,
                    self.pai_client,
                    answers=self.case.answers,
                    prefix_cache=PrefixCache() if prefix else None,
                    keep_thoughts=keep_thoughts,
                    translation=await self.translation(),
                )
        return self


# --- What is compared -----------------------------------------------------------------


def shape(differences: Mapping[str, Any]) -> dict[str, Any]:
    """`differences` with `other_config` reduced to the paths that differ."""
    shaped = dict(differences)
    if "other_config" in shaped:
        shaped["other_config"] = [found["path"] for found in shaped["other_config"]]
    return shaped


def _plain(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def _session_state(state: Mapping[str, Any], *, persisted_only: bool) -> Any:
    """`state` as JSON without the wall-clock fields and, for the Pydantic AI arm, the
    `temp:` keys ADK keeps out of its stored session."""

    def clean(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items() if k not in WALL_CLOCK}
        if isinstance(value, list):
            return [clean(item) for item in value]
        return value

    kept: dict[str, Any] = {}
    for key, value in state.items():
        if persisted_only and str(key).startswith(STATE_PREFIX_NOT_PERSISTED):
            continue
        if key in RUN_UNIQUE:
            continue
        if str(key).startswith(RESOLVED_AT):
            # The key names what was resolved; its value is the time it was.
            kept[key] = "<time>"
        elif key == DISCOVERY:
            kept[key] = _discovery(value)
        else:
            kept[key] = value
    return clean(_plain(kept))


def _discovery(discovery: Mapping[str, Any]) -> dict[str, Any]:
    """The translation's discovery state with its records, keyed by an id minted for the
    run, as a list of their content in a fixed order."""
    records = [
        {k: v for k, v in record.items() if k != "id"}
        for record in _plain(discovery["records"]).values()
    ]
    return {
        **discovery,
        "records": sorted(records, key=sc.canonical),
    }


def _short(value: Any, limit: int = 300) -> Any:
    text = sc.canonical(value)
    return value if len(text) <= limit else text[:limit] + "..."


def _shorten(found: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"path": d["path"], "adk": _short(d["adk"]), "pai": _short(d["pai"])} for d in found[:8]
    ]


def compare(arms: Arms) -> dict[str, Any]:
    """How the two arms ended and what each sent, and every difference between them that
    `WIRE` and the state rules do not explain. Empty `unexplained` means parity."""
    unexplained: dict[str, Any] = {}
    errors = {"adk": arms.adk.error, "pai": arms.pai.run.error}
    if errors["adk"] or errors["pai"]:
        unexplained["error"] = {k: v and v.splitlines()[0] for k, v in errors.items()}
    adk_out, pai_out = sc.outcome(arms.adk), sc.outcome(arms.pai.run)
    if found := sc.diff(adk_out, pai_out):
        unexplained["outcome"] = _shorten(found)
    adk_state = _session_state(arms.adk.state, persisted_only=False)
    pai_state = _session_state(arms.pai.run.state, persisted_only=True)
    if found := sc.diff(adk_state, pai_state):
        unexplained["state"] = _shorten(found)
    if found := sc.diff(seen["adk"], seen["pai"]):
        unexplained["argument_types"] = _shorten(found)
    adk_calls, pai_calls = arms.adk_client.calls, arms.pai_client.calls
    if len(adk_calls) != len(pai_calls):
        unexplained["request_count"] = {"adk": len(adk_calls), "pai": len(pai_calls)}
    requests: dict[int, Any] = {}
    for index, (adk, pai) in enumerate(zip(adk_calls, pai_calls, strict=False)):
        differences = shape(sc.request_differences(adk, pai))
        if differences != WIRE:
            requests[index] = {
                name: _shorten(found) if name != "other_config" else found
                for name, found in differences.items()
                if found != WIRE.get(name)
            } or {"wire_closed": sorted(set(WIRE) - set(differences))}
    if requests:
        unexplained["requests"] = requests
    return {
        "stop_reason": adk_out["stop_reason"],
        "requests": len(adk_calls),
        "answers": len(arms.case.answers),
        "tool_calls": adk_out["tool_calls"],
        "unexplained": unexplained,
    }


# --- Per task -------------------------------------------------------------------------


def prefix_of(create: Mapping[str, Any]) -> dict[str, Any]:
    """A cache create's prefix, the instruction as text and each declaration by name."""
    raw = sc.jsonable(create["config"])
    config = sc.config_dict(
        {key: raw.get(key) for key in ("system_instruction", "tools", "tool_config")}
    )
    return {
        "system_instruction": sc.instruction_text(config),
        "declarations": sc.declarations(config),
        "tool_config": config.get("tool_config"),
    }


PREFIX_WIRE = [
    {
        "path": "declarations.reject_plan.parameters_json_schema",
        "adk": "<absent>",
        "pai": {"type": "object", "properties": {}},
    }
]


async def check_prefix(task: Task) -> dict[str, Any]:
    """Both arms create one cache of the same instruction and declarations and send their
    request against it, with no prefix of its own."""
    case = Case("prefix", [[propose(task.reference_payload)]])
    arms = await Arms(task, case).run(prefix=True)
    unexplained: dict[str, Any] = {}
    if len(arms.adk_client.creates) != 1 or len(arms.pai_client.creates) != 1:
        return {
            "unexplained": {"creates": [len(arms.adk_client.creates), len(arms.pai_client.creates)]}
        }
    found = sc.diff(prefix_of(arms.adk_client.creates[0]), prefix_of(arms.pai_client.creates[0]))
    if found != PREFIX_WIRE:
        unexplained["prefix"] = _shorten(found)
    for arm, client in (("adk", arms.adk_client), ("pai", arms.pai_client)):
        request = sc.config_dict(client.calls[0]["config"])
        if request.get("cached_content") != "cachedContents/scripted-1" or (
            "system_instruction" in request or "tools" in request
        ):
            unexplained[f"{arm}_request"] = "does not go against the cache alone"
    return {"unexplained": unexplained}


async def check_task(task: Task) -> dict[str, Any]:
    cases = {}
    for case in task_cases(task):
        arms = await Arms(task, case).run()
        cases[case.label] = compare(arms)
    prefix = await check_prefix(task)
    unexplained = {
        label: result["unexplained"] for label, result in cases.items() if result["unexplained"]
    }
    if prefix["unexplained"]:
        unexplained["prefix"] = prefix["unexplained"]
    return {"cases": cases, "unexplained": unexplained}


# --- The gaps -------------------------------------------------------------------------


def gap_cases(task: Task) -> dict[str, Case]:
    """Scripts for what `agent/NOTES.md` lists as not reproduced and offline parity can
    settle: ADK's conversion of `BaseModel`-typed arguments, its answer to a missing
    mandatory argument, and a response with no parts."""
    reference = dict(task.reference_payload)
    coerced = json.loads(json.dumps(reference))
    for requirement in coerced["evidence_requirements"]:
        if isinstance(requirement.get("period"), dict):
            requirement["period"] = {k: str(v) for k, v in requirement["period"].items()}
    wrong = json.loads(json.dumps(reference))
    wrong["evidence_requirements"][0]["period"] = {"start": 2015, "end": 2022, "stray": 1}
    return {
        "models-valid": Case("models-valid", [[propose(reference)]]),
        "models-coercible": Case("models-coercible", [[propose(coerced)], [propose(reference)]]),
        "models-invalid-item": Case(
            "models-invalid-item", [[propose(wrong)], [propose(reference)]]
        ),
        "models-bad-answered": Case(
            "models-bad-answered",
            [[propose({**reference, "answered": [{"ask": "x"}]})], [propose(reference)]],
        ),
        "missing-ask-message": Case(
            "missing-ask-message", [[ask(options=[])], [propose(reference)]]
        ),
        "missing-skill-id": Case("missing-skill-id", [[("get_skill", {})], [propose(reference)]]),
        "skill-file-page": Case(
            "skill-file-page",
            [[("get_skill", {"skill_id": "oda-dac2a", "path": "SKILL.md"})], [propose(reference)]],
        ),
        "no-parts": Case("no-parts", [], turns=[[]]),
        "no-parts-then-plan": Case("no-parts-then-plan", [], turns=[[], [propose_part(reference)]]),
        "text-only": Case("text-only", [], turns=[[types.Part(text="I have no plan.")]]),
        "thought-only": Case(
            "thought-only", [], turns=[[types.Part(text=THOUGHT_TEXT, thought=True)]]
        ),
    }


def thought_script() -> list[list[types.Part]]:
    return [
        [
            types.Part(text=THOUGHT_TEXT, thought=True),
            types.Part(
                function_call=types.FunctionCall(name="reject_plan", args={}),
                thought_signature=sc.SIGNATURE,
            ),
        ]
    ]


async def check_thoughts(task: Task, *, keep_thoughts: bool) -> dict[str, Any]:
    """The request after a turn that carries an unsigned thought summary and a signed call:
    no summary text, the signature kept, the same contents from each arm."""
    adk, pai = sc.Scripted(thought_script()), sc.Scripted(thought_script())
    await sc.run_adk(task, adk)
    await sc.run_pai(task, pai, keep_thoughts=keep_thoughts)
    adk_second = sc.contents_list(adk.calls[1]["contents"], ids=False)
    pai_second = sc.contents_list(pai.calls[1]["contents"], ids=False)
    unexplained: dict[str, Any] = {}
    if found := sc.diff(adk_second, pai_second):
        unexplained["contents"] = _shorten(found)
    parts = [part for content in pai_second for part in content["parts"]]
    if any(part.get("text") == THOUGHT_TEXT for part in parts):
        unexplained["thought_replayed"] = True
    if not any("function_call" in part and "thought_signature" in part for part in parts):
        unexplained["signature_lost"] = True
    return {"unexplained": unexplained}


async def check_gaps(task: Task) -> dict[str, Any]:
    results = {}
    for label, case in gap_cases(task).items():
        results[label] = compare(await Arms(task, case).run())
    return {
        "cases": results,
        "unexplained": {k: v["unexplained"] for k, v in results.items() if v["unexplained"]},
    }


# --- Deliberately broken variants -----------------------------------------------------


@contextlib.contextmanager
def broken(name: str | None) -> Iterator[dict[str, Any]]:
    """Run with `name` broken on the Pydantic AI arm, to show a check that covers it fails."""
    options: dict[str, Any] = {"keep_thoughts": False}
    patches: list[tuple[Any, str, Any]] = []

    def patch(owner: Any, attribute: str, value: Any) -> None:
        patches.append((owner, attribute, getattr(owner, attribute)))
        setattr(owner, attribute, value)

    if name in ("instruction", "declaration"):
        of = pai_planner.PlannerParts.of

        def changed(seam: Any) -> pai_planner.PlannerParts:
            parts = of(seam)
            if name == "instruction":
                return pai_planner.PlannerParts(
                    parts.instruction + " X", parts.tools, parts.declarations
                )
            declarations = tuple(
                d.model_copy(update={"description": (d.description or "") + " "})
                if d.name == "propose_plan"
                else d
                for d in parts.declarations
            )
            return pai_planner.PlannerParts(parts.instruction, parts.tools, declarations)

        patch(pai_planner.PlannerParts, "of", staticmethod(changed))
    elif name == "contents":
        patch(pai_call, "QUESTION_INSTRUCTIONS", None)
        patch(
            pai_call,
            "user_message",
            lambda text: pai_call.ModelRequest(
                parts=[pai_call.UserPromptPart(content=text)],
                metadata={pai_call.USER_MESSAGE: True},
            ),
        )
        patch(pai_entry, "user_message", pai_call.user_message)
    elif name == "tool-result":
        original = pai_planner.propose_and_translate

        async def with_extra_key(*args: Any, **kwargs: Any) -> dict[str, object]:
            return {**await original(*args, **kwargs), "extra": 1}

        patch(pai_planner, "propose_and_translate", with_extra_key)
    elif name == "end-run":
        patch(pai_planner.PlannerDeps, "end_run", lambda self: None)
    elif name == "state":
        original_state = pai_entry.orchestrator_state

        async def orchestrator_state(task: Task, resolver: Any) -> dict[str, Any]:
            state = await original_state(task, resolver)
            state.pop(next(iter(state)))
            return state

        patch(pai_entry, "orchestrator_state", orchestrator_state)
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
    elif name == "arguments":
        patch(pai_planner, "convert_arguments", lambda tool, args: dict(args))
    elif name == "mandatory":
        patch(pai_planner, "missing_arguments", lambda tool, args: None)
    elif name == "no-parts":
        patch(pai_planner, "OUTPUT_TYPE", str)
    try:
        yield options
    finally:
        for owner, attribute, value in reversed(patches):
            setattr(owner, attribute, value)


# --- Main -----------------------------------------------------------------------------


async def check(tasks: Sequence[Task], *, keep_thoughts: bool) -> dict[str, Any]:
    report: dict[str, Any] = {"tasks": {}}
    for task in tasks:
        report["tasks"][task.id] = result = await check_task(task)
        print(f"{'DIFF' if result['unexplained'] else 'same'}  {task.id}", file=sys.stderr)
    first = tasks[0]
    report["gaps"] = await check_gaps(first)
    report["thoughts"] = await check_thoughts(first, keep_thoughts=keep_thoughts)
    return report


def summarise(report: Mapping[str, Any], tasks: Sequence[Task]) -> dict[str, Any]:
    cases = [c for result in report["tasks"].values() for c in result["cases"].values()]
    plans = sum(len(validate.task_payloads(task)) for task in tasks)
    differing = {
        task_id: result["unexplained"]
        for task_id, result in report["tasks"].items()
        if result["unexplained"]
    }
    unexplained = {
        "tasks": differing,
        "gaps": report["gaps"]["unexplained"],
        "thoughts": report["thoughts"]["unexplained"],
    }
    return {
        "tasks": len(tasks),
        "plans": plans,
        "cases": len(cases),
        "requests_compared": sum(c["requests"] for c in cases),
        # Each answer resumes the entry in a request of its own.
        "requests_after_an_answer": sum(c["answers"] for c in cases),
        "wire": dict(WIRE),
        "unexplained": unexplained,
        "pass": not any(unexplained.values()),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", help="comma-separated task ids")
    parser.add_argument("--break", dest="broken", choices=BREAKS)
    parser.add_argument("--out", type=Path, default=HERE / "parity.json")
    args = parser.parse_args(argv)
    quiet_logs()
    logging.getLogger("planner.skills").setLevel(logging.ERROR)
    os.environ["STUDIO_API_TOKEN"] = ""
    tasks = load_tasks()
    if args.only:
        wanted = set(args.only.split(","))
        tasks = [task for task in tasks if task.id in wanted]
    out = args.out.resolve()
    # Settings reads a `.env` from the working directory; run where there is none.
    with tempfile.TemporaryDirectory() as empty, broken(args.broken) as options:
        os.chdir(empty)
        report = asyncio.run(check(tasks, keep_thoughts=options["keep_thoughts"]))
    report["summary"] = summarise(report, tasks)
    report["broken"] = args.broken
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=1, default=str))
    return 0 if report["summary"]["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())

"""Check every task with no model call: its plans hold, and the set covers what it must.

A scripted model proposes each task's reference plan, and each alternative, through
`entry.run_entry`, the same entry a live run takes, so this proves the recording
replays, the orchestrator's tools take the searches and the hand-off, and the real
`propose_plan` accepts the plan in that state. A task with `first_proposal_refusal`
has to be refused with that code first, ask the user, and be accepted once the user
picks adding them. The reference plan, once accepted, has to do with its years what
the task's `period_reduction` says (`none` or `across_years`; `propose_plan` accepts
no plan that reads `unstated`), and an alternative what its own says when it states
one. The kinds a task's plans use together have to be exactly its
`acceptable_kinds`, and each has to use every one of its `required_kinds`.
`unit_asked` has to agree with whether the question or brief names a currency or price
basis (`UNIT_WORDS`).

The set has to hold at least `MINIMUM_TASKS` tasks, a task requiring each kind, and,
filed under each kind, a task naming a neighbour: the kind it is most easily mistaken
for, which no correct plan for it uses.

    just eval-planner-entry-validate
    just eval-planner-entry-validate --only share-health-oda-kenya
    just eval-planner-entry-validate --show share-health-oda-kenya   # the planner's first message
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from contextlib import aclosing
from typing import Any

from entry import (
    KINDS,
    TASKS_DIR,
    EntryRun,
    PeriodReduction,
    Task,
    first_year_gaps,
    load_recording,
    load_tasks,
    period_reductions,
    plan_period_reduction,
    quiet_logs,
    run_entry,
    stray_recordings,
    tool_calls,
)
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.workflow.utils._workflow_hitl_utils import REQUEST_INPUT_FUNCTION_CALL_NAME
from google.genai import types
from harness.turn_budget import bounded
from pydantic import Field
from submission.service import COMBINATION_OPTIONS, COMBINE_OPTION

MINIMUM_TASKS = 60

# The words a question or brief names a unit in: a currency, or a price basis.
UNIT_WORDS = re.compile(
    r"US dollars?|\bUSD\b|\bdollars?\b|\beuros?\b|current prices|constant prices|constant \d{4}",
    re.IGNORECASE,
)
MINIMUM_REFUSAL_TASKS = 2

Step = list[tuple[str, dict[str, Any]]]


# The usage every scripted generation reports, chunk and whole answer alike.
SCRIPTED_USAGE = types.GenerateContentResponseUsageMetadata(
    prompt_token_count=1000,
    cached_content_token_count=600,
    candidates_token_count=40,
    thoughts_token_count=7,
)


class ScriptedLlm(BaseLlm):
    """Answers each generation with the next step of `script`; past its end, with text.
    Keeps every request it was sent. A streamed generation answers first as a partial
    chunk and then whole, as `Gemini` does under SSE, each with `SCRIPTED_USAGE`. Each
    generation keeps to the current turn budget as the planner's model does
    (`harness.turn_budget.bounded`)."""

    model: str = "scripted-planner"
    script: list[Step] = Field(default_factory=list)
    requests: list[LlmRequest] = Field(default_factory=list)

    async def generate_content_async(self, llm_request: LlmRequest, stream: bool = False) -> Any:
        answer = bounded(lambda: self._answer(llm_request, stream), stream=stream)
        async with aclosing(answer) as responses:
            async for response in responses:
                yield response

    async def _answer(self, llm_request: LlmRequest, stream: bool) -> Any:
        index = len(self.requests)
        self.requests.append(llm_request)
        if index >= len(self.script):
            parts = [types.Part.from_text(text="done")]
        else:
            parts = [
                types.Part.from_function_call(name=name, args=args)
                for name, args in self.script[index]
            ]
        content = types.Content(role="model", parts=parts)
        if stream:
            yield LlmResponse(content=content, partial=True, usage_metadata=SCRIPTED_USAGE)
        yield LlmResponse(content=content, usage_metadata=SCRIPTED_USAGE)


def _kinds(payload: Mapping[str, Any]) -> set[str]:
    return {requirement["kind"] for requirement in payload["evidence_requirements"]}


async def propose(
    task: Task,
    payload: Mapping[str, Any],
    refusal: str | None,
    years: PeriodReduction,
) -> list[str]:
    """Every fault in proposing `payload` for `task` through the eval's entry. `years`
    is what the accepted plan has to do with its years."""
    propose_call = ("propose_plan", dict(payload))
    if refusal is None:
        script: list[Step] = [[propose_call]]
        answers: list[Mapping[str, Any]] = []
    else:
        ask = (
            REQUEST_INPUT_FUNCTION_CALL_NAME,
            {
                "message": "Do these two datasets add up without counting a flow twice?",
                "options": [dict(option) for option in COMBINATION_OPTIONS],
            },
        )
        script = [[propose_call], [ask], [propose_call]]
        answers = [{"result": {"spec_id": COMBINE_OPTION}}]
    run = await run_entry(
        task,
        load_recording(task.id),
        prefix_cache=None,
        model=ScriptedLlm(script=script),
        timeout_s=60.0,
        answers=answers,
    )
    return _faults(run, payload, refusal, years)


def _faults(
    run: EntryRun,
    payload: Mapping[str, Any],
    refusal: str | None,
    years: PeriodReduction,
) -> list[str]:
    if run.error is not None:
        return [run.error.splitlines()[0]]
    calls = tool_calls(run.events)
    responses = [call["response"] or {} for call in calls if call["name"] == "propose_plan"]
    faults = []
    if refusal is not None:
        first = responses[0] if responses else {}
        if first.get("code") != refusal:
            faults.append(
                f"first proposal answered {first.get('outcome')}/{first.get('code')}: "
                f"{str(first.get('message'))[:300]}, not {refusal}"
            )
        if REQUEST_INPUT_FUNCTION_CALL_NAME not in [call["name"] for call in calls]:
            faults.append("the entry never asked the user")
    plan = run.probe.plan
    if plan is None:
        last = responses[-1] if responses else {}
        faults.append(
            f"not accepted: {last.get('outcome')}/{last.get('code')}: "
            f"{str(last.get('message'))[:400]}"
        )
    elif {r.kind for r in plan.evidence_requirements} != _kinds(payload):
        faults.append("the recorded plan's kinds differ from the payload's")
    elif (found := plan_period_reduction(period_reductions(plan))) != years:
        faults.append(f"period_reduction is {years.value}, but the plan's is {found.value}")
    if run.surface_misses:
        faults.append(f"the recording lacks {run.surface_misses}")
    return faults


async def check_task(task: Task) -> list[str]:
    """Every fault in `task`: its fields, its kinds, and each plan it states."""
    faults = []
    required, acceptable = set(task.required_kinds), set(task.acceptable_kinds)
    if unknown := sorted((required | acceptable) - set(KINDS)):
        faults.append(f"unknown kinds {unknown}")
    if not required <= acceptable:
        faults.append(f"required kinds {sorted(required - acceptable)} are not acceptable")
    if stray := sorted(set(task.required_one_of) - acceptable):
        faults.append(f"required_one_of kinds {stray} are not acceptable")
    if task.kind not in acceptable:
        faults.append(f"filed under {task.kind}, which no correct plan for it uses")
    if task.neighbour is not None and task.neighbour.kind in acceptable | {task.kind}:
        faults.append(f"neighbour {task.neighbour.kind} is a kind a correct plan may use")
    named = bool(UNIT_WORDS.search(task.question) or UNIT_WORDS.search(task.brief))
    if named != task.unit_asked:
        faults.append(
            f"unit_asked is {task.unit_asked}, but the question or brief names {'a' if named else 'no'} unit"
        )
    if task.reference_payload.get("question") != task.brief:
        faults.append("the reference payload's question is not the brief")
    if not (TASKS_DIR / f"{task.id}.json").exists():
        faults.append("the task file is not named for its id")
    witnessed: set[str] = set()
    plans = [
        ("reference", task.reference_payload, task.first_proposal_refusal, task.period_reduction)
    ]
    plans += [
        (
            f"alternative {index}",
            alt.payload,
            alt.first_proposal_refusal,
            alt.period_reduction or task.period_reduction,
        )
        for index, alt in enumerate(task.alternative_payloads)
    ]
    for label, payload, refusal, years in plans:
        kinds = _kinds(payload)
        witnessed |= kinds
        if not required <= kinds:
            faults.append(f"{label} lacks required kinds {sorted(required - kinds)}")
        if task.required_one_of and not kinds & set(task.required_one_of):
            faults.append(f"{label} uses none of {sorted(task.required_one_of)}")
        try:
            faults += [
                f"{label}: {fault}" for fault in await propose(task, payload, refusal, years)
            ]
        except Exception as exc:
            faults.append(f"{label}: {type(exc).__name__}: {exc}")
    try:
        recording = load_recording(task.id)
        shown = await run_entry(task, recording, prefix_cache=None, model=ScriptedLlm())
        if gaps := await first_year_gaps(shown.state, recording):
            faults.append(
                f"the recording holds no answer for {len(gaps)} first-year reads, so record "
                f"it again: {gaps[0][:160]}"
            )
    except Exception as exc:
        faults.append(f"first-year reads: {type(exc).__name__}: {exc}")
    if witnessed != acceptable:
        faults.append(
            f"acceptable kinds {sorted(acceptable)} differ from those its plans use, "
            f"{sorted(witnessed)}"
        )
    return faults


def coverage_faults(tasks: Sequence[Task]) -> list[str]:
    """What the set lacks or leaves lying: tasks, a kind nothing requires, a kind with no
    neighbour task, refusal tasks, and recorded files no task uses."""
    faults = []
    if len(tasks) < MINIMUM_TASKS:
        faults.append(f"{len(tasks)} tasks, at least {MINIMUM_TASKS} wanted")
    ids = Counter(task.id for task in tasks)
    faults += [f"id {name!r} is used {count} times" for name, count in ids.items() if count > 1]
    required = Counter(kind for task in tasks for kind in set(task.required_kinds))
    faults += [f"no task requires {kind}" for kind in KINDS if not required[kind]]
    neighboured = {task.kind for task in tasks if task.neighbour is not None}
    faults += [f"no {kind} task names a neighbour" for kind in KINDS if kind not in neighboured]
    refusals = sum(task.first_proposal_refusal is not None for task in tasks)
    if refusals < MINIMUM_REFUSAL_TASKS:
        faults.append(f"{refusals} refusal tasks, at least {MINIMUM_REFUSAL_TASKS} wanted")
    stray = stray_recordings({task.id for task in tasks})
    faults += [f"{path.parent.name}/{path.name} is recorded for no task" for path in stray]
    return faults


async def check_all(
    tasks: Sequence[Task],
    *,
    report: Callable[[Task, list[str]], None] | None = None,
    coverage: bool = True,
) -> dict[str, list[str]]:
    """Each task's faults, by id, with the set's coverage faults under `""`."""
    faults: dict[str, list[str]] = {}
    for task in tasks:
        found = await check_task(task)
        if report is not None:
            report(task, found)
        if found:
            faults[task.id] = found
    if coverage and (gaps := coverage_faults(tasks)):
        faults[""] = gaps
    return faults


def _print(task: Task, faults: list[str]) -> None:
    kinds = ",".join(sorted(_kinds(task.reference_payload)))
    neighbour = f" vs {task.neighbour.kind}" if task.neighbour else ""
    print(f"{'FAIL' if faults else 'ok  '}  {task.kind:<11} {task.id:<48} {kinds}{neighbour}")
    for fault in faults:
        print(f"      {fault}")


async def first_message(task: Task) -> str:
    """The user content of the planner's first request for `task`, as a live run sends it."""
    model = ScriptedLlm()
    await run_entry(task, load_recording(task.id), prefix_cache=None, model=model, timeout_s=60.0)
    return "\n\n".join(
        part.text
        for content in model.requests[0].contents
        if content.role == "user"
        for part in content.parts or []
        if part.text
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", help="comma-separated task ids; skips the coverage check")
    parser.add_argument("--show", metavar="ID", help="print the planner's first message for a task")
    args = parser.parse_args(argv)
    quiet_logs()
    tasks = load_tasks()
    if args.show:
        task = next((task for task in tasks if task.id == args.show), None)
        if task is None:
            raise SystemExit(f"unknown task id {args.show!r}")
        print(asyncio.run(first_message(task)))
        return 0
    if args.only:
        wanted = set(args.only.split(","))
        tasks = [task for task in tasks if task.id in wanted]
        if missing := wanted - {task.id for task in tasks}:
            raise SystemExit(f"unknown task ids {sorted(missing)}")
    faults = asyncio.run(check_all(tasks, report=_print, coverage=not args.only))
    for gap in faults.get("", []):
        print(f"GAP   {gap}")
    by_kind = Counter(task.kind for task in tasks)
    print("filed per kind: " + ", ".join(f"{kind}={by_kind[kind]}" for kind in KINDS))
    failed = len([key for key in faults if key])
    print(f"{len(tasks)} tasks: {len(tasks) - failed} pass, {failed} fail")
    return 1 if faults else 0


if __name__ == "__main__":
    sys.exit(main())

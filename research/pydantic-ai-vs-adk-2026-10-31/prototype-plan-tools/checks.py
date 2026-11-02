"""The pass lines' checks: plan-mcp beside dev's planner entry and dev's root workflow.

Each `*_faults` function answers the faults it found, empty when the line holds, so
a test asserts on it and its control asserts it is not empty.

- P1 and P2 seed a plan session from a planner-entry task's recorded searches
  (`entry.RecordedResolver`), translate over the task's recorded translation calls
  (`entry.TranslationCalls`), and call the tools through a FastMCP client in
  process. Dev's side is `validate.accepted_run` or `entry.run_entry`, the
  planner entry as the served harness runs it, with a scripted model.
- P3 proposes dev's canned plan (`scripted.PROPOSAL`) and runs it on
  `harness.scripted`'s surface, and dev's side runs the same accepted plan
  through dev's root workflow (`tests.orchestrator_fakes.orchestrator_runner`).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from entry import (
    RecordedResolver,
    RecordedSurface,
    Task,
    TranslationCalls,
    _no_anchor_read,
    load_recording,
    period_reductions,
    plan_period_reduction,
    run_entry,
    tool_calls,
)
from execution import dev_run_graph
from executor.ledger import InMemoryRunLedger
from fastmcp import Client
from google.genai import types
from harness.conversation import current_conversation_id
from harness.scripted import ScriptedExecutor, ScriptedSurface, scripted_publish_result
from harness.scripted_plan import scripted_fetch_series_stats
from harness.tool_contracts import resolver_digest
from plan_tools import PlanningService
from planner.plan import Plan, load_plan, plan_revision
from scripted import BRIEF, SEARCH, FixedPlanner, scripted_translate
from server import PlanMcp, SessionKey
from submission.service import COMBINATION_OPTIONS, COMBINE_OPTION
from tests.orchestrator_fakes import (
    CannedSearch,
    approve_answer,
    create_request_input_response,
    definite_envelope,
    handed_back,
    open_session,
    orchestrator_runner,
    plan_call,
    run_turn,
    search_call,
    single_pause,
)
from translator.tool import build_translate
from validate import Payload, ScriptedLlm, accepted_run, task_payloads, tool_schemas

# What validate.py's scripted model asks the user after a `combination_unasked`
# refusal, and what the user answers.
COMBINATION_ASK = {
    "message": "Do these two datasets add up without counting a flow twice?",
    "options": [dict(option) for option in COMBINATION_OPTIONS],
}
COMBINE_ANSWER = {"kind": "option", "spec_id": COMBINE_OPTION}

# Keys whose values differ between two runs of the same thing.
VOLATILE = {"run_id", "observed_at", "approved_at", "at", "created_at", "updated_at"}


def stable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: stable(v) for k, v in value.items() if k not in VOLATILE}
    if isinstance(value, list):
        return [stable(v) for v in value]
    return value


def differences(a: Any, b: Any, path: str = "") -> list[str]:
    """The paths at which `a` and `b` differ."""
    if isinstance(a, dict) and isinstance(b, dict):
        return [
            d
            for key in sorted(set(a) | set(b), key=str)
            for d in differences(a.get(key), b.get(key), f"{path}.{key}")
        ]
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [
            d
            for i, (x, y) in enumerate(zip(a, b, strict=True))
            for d in differences(x, y, f"{path}[{i}]")
        ]
    return [] if a == b else [f"{path}: {str(a)[:200]!r} != {str(b)[:200]!r}"]


async def _no_run(*args: Any) -> dict[str, Any]:
    raise AssertionError("a planning check runs no plan")


@dataclass
class Recorded:
    """A plan-mcp server over one task's recording."""

    server: PlanMcp
    translation: TranslationCalls


async def recorded_server(
    task: Task, *, key: SessionKey = SessionKey.RETURNED, recording: Mapping[str, Any] | None = None
) -> Recorded:
    """plan-mcp answering `task`'s searches, skill reads and translation calls from its
    recording, as `entry.run_entry` answers them."""
    recording = recording if recording is not None else load_recording(task.id)
    translation = TranslationCalls(recording.get("translation") or {}, schemas=await tool_schemas())
    service = PlanningService(
        resolver=RecordedResolver(recording["searches"]),
        surface=RecordedSurface(recording.get("tools") or {}),
        translate=build_translate(
            surface=translation,
            resolve_entities=translation.resolve_entities,
            anchor_read=_no_anchor_read,
            describe_nodes=translation.describe_nodes,
        ),
        run_graph=_no_run,
    )
    return Recorded(PlanMcp(service, key=key), translation)


def opening(task: Task) -> dict[str, Any]:
    return {
        "brief": task.brief,
        "asks": [ask if isinstance(ask, str) else ask.model_dump() for ask in task.asks],
        "searches": [search.model_dump() for search in task.searches],
    }


async def call(client: Client, tool: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
    result = await client.call_tool(tool, dict(arguments), raise_on_error=False)
    assert result.structured_content is not None, result
    return dict(result.structured_content)


@dataclass
class McpProposal:
    """What plan-mcp answered proposing one payload: each `propose_plan` answer in
    order, the plan on record after the last, and the planner's first message."""

    answers: list[dict[str, Any]]
    plan: Plan | None
    first_message: str


async def mcp_propose(
    task: Task, payload: Payload, *, body: Mapping[str, Any] | None = None
) -> McpProposal:
    """`payload` (or `body` in its place) through plan-mcp, as validate.py's script
    sends it: once, and again after asking the user when it is refused for a
    combination."""
    recorded = await recorded_server(task)
    async with Client(recorded.server.mcp) as client:
        opened = await call(client, "start_plan", opening(task))
        assert opened["outcome"] == "ok", opened
        session_id = opened["plan_session"]
        sent = {"plan_session": session_id, **(body if body is not None else payload.body)}
        answers = [await call(client, "propose_plan", sent)]
        if payload.refusal is not None and answers[0].get("code") == payload.refusal:
            recorded_answer = await call(
                client,
                "record_user_answer",
                {"plan_session": session_id, **COMBINATION_ASK, "answer": COMBINE_ANSWER},
            )
            assert recorded_answer["outcome"] == "ok", recorded_answer
            answers.append(await call(client, "propose_plan", sent))
    session = recorded.server.service.sessions[session_id]
    return McpProposal(answers, load_plan(session.state), opened["planner_message"])


async def dev_first_message(task: Task) -> str:
    """The planner's first message on dev: the user content of its first request."""
    model = ScriptedLlm()
    await run_entry(task, load_recording(task.id), prefix_cache=None, model=model, timeout_s=60.0)
    return "\n\n".join(
        part.text
        for content in model.requests[0].contents
        if content.role == "user"
        for part in content.parts or []
        if part.text
    )


async def p1_faults(task: Task, *, body: Mapping[str, Any] | None = None) -> list[str]:
    """Every way plan-mcp's answer to `task`'s reference payload (or `body`) differs
    from dev's planner entry, or fails what validate.py holds the plan to."""
    payload = task_payloads(task)[0]
    recording = load_recording(task.id)
    dev_translation = TranslationCalls(
        recording.get("translation") or {}, schemas=await tool_schemas()
    )
    dev_payload = payload if body is None else Payload(payload.label, body, None, payload.years)
    dev = await accepted_run(task, recording, dev_payload, dev_translation)
    mine = await mcp_propose(task, payload, body=body)
    faults: list[str] = []
    dev_answers = [
        c["response"] or {} for c in tool_calls(dev.events) if c["name"] == "propose_plan"
    ]
    if payload.refusal is not None and body is None and mine.answers[0] != dev_answers[0]:
        faults += [f"first refusal {d}" for d in differences(mine.answers[0], dev_answers[0])]
    last = mine.answers[-1]
    if last.get("outcome") != "ok" or "summary" not in last:
        faults.append(f"not accepted: {last.get('outcome')}/{last.get('code')}")
        return faults
    if dev.probe.translation is None:
        faults.append("dev accepted nothing")
        return faults
    faults += [f"translation {d}" for d in differences(last["translation"], dev.probe.translation)]
    dev_plan = load_plan(dev.state)
    assert mine.plan is not None and dev_plan is not None
    if plan_revision(mine.plan) != plan_revision(dev_plan):
        faults += [
            f"plan {d}"
            for d in differences(
                mine.plan.model_dump(mode="json"), dev_plan.model_dump(mode="json")
            )
        ]
    if mine.plan.status.value != "pending_approval":
        faults.append(f"plan is {mine.plan.status.value}")
    if (found := plan_period_reduction(period_reductions(mine.plan))) != payload.years:
        faults.append(f"period_reduction is {found.value}, not {payload.years.value}")
    kinds = {r.kind for r in mine.plan.evidence_requirements}
    if kinds != {r["kind"] for r in payload.body["evidence_requirements"]}:
        faults.append("the recorded plan's kinds differ from the payload's")
    if mine.first_message not in await dev_first_message(task):
        faults.append("the planner's first message differs")
    return faults


async def dev_first_run(
    task: Task, body: Mapping[str, Any], *, recording: Mapping[str, Any] | None = None
) -> tuple[dict[str, Any], dict[str, object] | None]:
    """The planner's own first `propose_plan` answer to `body` on dev's planner entry,
    and the translation of the plan it recorded, if it recorded one."""
    recording = recording if recording is not None else load_recording(task.id)
    translation = TranslationCalls(recording.get("translation") or {}, schemas=await tool_schemas())
    run = await run_entry(
        task,
        recording,
        prefix_cache=None,
        model=ScriptedLlm(script=[[("propose_plan", dict(body))]]),
        timeout_s=60.0,
        translation=translation,
    )
    answers = [c["response"] for c in tool_calls(run.events) if c["name"] == "propose_plan"]
    assert answers and answers[0] is not None, run.error
    return dict(answers[0]), run.probe.translation


async def mcp_first_answer(
    task: Task, body: Mapping[str, Any], *, recording: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """plan-mcp's `propose_plan` answer to `body` on a session seeded from `task`."""
    recorded = await recorded_server(task, recording=recording)
    async with Client(recorded.server.mcp) as client:
        opened = await call(client, "start_plan", opening(task))
        return await call(client, "propose_plan", {"plan_session": opened["plan_session"], **body})


REFUSAL_KEYS = ("outcome", "code", "action")


def refusal_faults(mine: Mapping[str, Any], dev: Mapping[str, Any]) -> list[str]:
    """Where plan-mcp's refusal differs from dev's in `outcome`, `code` or `action`."""
    return [d for key in REFUSAL_KEYS for d in differences(mine.get(key), dev.get(key), f".{key}")]


# --- P3 ---------------------------------------------------------------------------


@dataclass
class Scripted:
    server: PlanMcp
    ledger: InMemoryRunLedger


class KeyedSearch:
    """A `Resolver` answering each search by its question, from `envelopes`, digested
    as `harness.mcp_client.McpResolver` digests them."""

    def __init__(self, envelopes: Mapping[str, Mapping[str, Any]]) -> None:
        self._envelopes = dict(envelopes)

    async def __call__(self, *, question: str, **kwargs: Any) -> dict[str, Any]:
        del kwargs
        return resolver_digest(self._envelopes[question])


def scripted_server(
    *,
    check_version: bool = True,
    resolver: Any = None,
    key: SessionKey = SessionKey.RETURNED,
) -> Scripted:
    """plan-mcp over canned searches (`resolver`, one Kenya search by default),
    `scripted_translate`, and dev's `execute` and `plan_done` nodes on
    `harness.scripted`'s surface and executor."""
    ledger = InMemoryRunLedger(scope=current_conversation_id.get)
    service = PlanningService(
        resolver=resolver or KeyedSearch({BRIEF: definite_envelope()}),
        surface=RecordedSurface({}),
        translate=scripted_translate,
        run_graph=dev_run_graph(
            executor=ScriptedExecutor(),
            tool_surface=ScriptedSurface(),
            run_ledger=ledger,
            publisher=scripted_publish_result,
            fetch_series_stats=scripted_fetch_series_stats,
        ),
        check_version=check_version,
    )
    return Scripted(PlanMcp(service, key=key), ledger)


SCRIPTED_OPENING = {"brief": BRIEF, "asks": [BRIEF], "searches": [SEARCH]}


async def dev_hand_back(plan: Plan) -> dict[str, Any]:
    """What dev's orchestrator receives for its `plan` call when dev's root workflow
    plans `plan`, the user approves it, and it runs on the same scripted surface."""
    runner, model = orchestrator_runner(
        search_call(SEARCH["query"], *(part["variable"] for part in SEARCH["parts"])),
        plan_call(BRIEF),
        planner=FixedPlanner(plan),
        resolver=CannedSearch(definite_envelope()),
        executor=ScriptedExecutor(),
        tool_surface=ScriptedSurface(),
    )
    await open_session(runner)
    approval = single_pause(await run_turn(runner, types.UserContent(BRIEF)))
    revision = approval.args["payload"]["revision"]
    approve = create_request_input_response(approval.id, {"result": approve_answer(revision)})
    await run_turn(runner, types.Content(role="user", parts=[approve]))
    _, hand_back = handed_back(model)
    return hand_back


@asynccontextmanager
async def client_of(server: PlanMcp) -> Any:
    async with Client(server.mcp) as client:
        yield client


def compact(value: Any) -> int:
    """The characters of `value` as compact JSON, or of `value` itself when text."""
    if isinstance(value, str):
        return len(value)
    return len(json.dumps(value, separators=(",", ":"), ensure_ascii=False))


def tokens(chars: int) -> int:
    """A token estimate at four characters a token."""
    return round(chars / 4)

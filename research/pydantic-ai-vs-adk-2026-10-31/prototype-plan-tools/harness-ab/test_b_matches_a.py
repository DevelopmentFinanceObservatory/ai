"""Arm B drafts as arm A does, with a scripted model, over both transports.

On each of the ten sample tasks the scripted model proposes the task's reference
plan. Arm B's planner must be sent the same first request text as arm A's and
record the same plan; on `sum-oda-oof-ethiopia`, whose reference plan is
refused `combination_unasked` first, the model asks the user, the turn pauses,
resumes with the user's answer, and the second proposal lands, in both arms.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from typing import Any

import ab
import pytest
from ab import Current, build_server, plan_client, run_b
from checks import COMBINATION_ASK, COMBINE_ANSWER, differences, stable
from entry import EntryRun, load_recording, load_tasks, quiet_logs, run_entry, tool_calls
from mcp_planner import BY_HARNESS, PLAN_SESSION_KEY, HarnessPlanMcp, McpPlannerSeam, PlanClient
from outward import ASK_TOOL
from planner.plan import plan_revision
from tasks import SAMPLE
from validate import ScriptedLlm, accepted_run, task_payloads

TASKS = {task.id: task for task in load_tasks() if task.id in SAMPLE}
REFUSAL_TASK = "sum-oda-oof-ethiopia"
ANSWER = {"result": {"answers": {"q1": COMBINE_ANSWER}}}


@pytest.fixture(autouse=True, scope="module")
def _quiet() -> None:
    quiet_logs()


type Side = tuple[PlanClient, Current, HarnessPlanMcp]


@pytest.fixture(params=["inproc", "http"])
async def b_side(request: pytest.FixtureRequest) -> AsyncIterator[Side]:
    current = Current()
    server = build_server(current)
    async with plan_client(server, request.param) as client:
        yield client, current, server


def _first_request_text(model: ScriptedLlm) -> str:
    return "\n\n".join(
        part.text
        for content in model.requests[0].contents
        if content.role == "user"
        for part in content.parts or []
        if part.text
    )


async def _b_run(
    task_id: str, client: PlanClient, current: Current, *, served_declaration: bool = False
) -> tuple[EntryRun, ScriptedLlm]:
    task = TASKS[task_id]
    payload = task_payloads(task)[0]
    propose = ("propose_plan", dict(payload.body))
    script = [[propose]]
    answers: list[Mapping[str, Any]] = []
    if payload.refusal is not None:
        script = [[propose], [(ASK_TOOL, dict(COMBINATION_ASK))], [propose]]
        answers = [ANSWER]
    model = ScriptedLlm(script=script)
    run = await run_b(
        task,
        load_recording(task_id),
        client=client,
        current=current,
        prefix_cache=None,
        model=model,
        answers=answers,
        timeout_s=60.0,
        served_declaration=served_declaration,
    )
    return run, model


def _faults(a: EntryRun, b: EntryRun, a_text: str, b_text: str) -> list[str]:
    faults = []
    if b.error or b.probe.plan is None:
        return [f"B accepted nothing: {b.error or b.probe.refused_code}"]
    assert a.probe.plan is not None, a.error
    if plan_revision(a.probe.plan) != plan_revision(b.probe.plan):
        faults += differences(
            stable(b.probe.plan.model_dump(mode="json")),
            stable(a.probe.plan.model_dump(mode="json")),
        )
    if a_text != b_text:
        faults.append("the planner's first request differs")
    return faults


@pytest.mark.parametrize("served_declaration", [False, True], ids=["listed", "served"])
@pytest.mark.parametrize("task_id", SAMPLE)
async def test_b_records_the_plan_a_records(
    task_id: str, served_declaration: bool, b_side: Side
) -> None:
    client, current, _ = b_side
    task = TASKS[task_id]
    a = await accepted_run(task, load_recording(task_id), task_payloads(task)[0])
    a_model = ScriptedLlm()
    await run_entry(task, load_recording(task_id), prefix_cache=None, model=a_model, timeout_s=60)
    b, b_model = await _b_run(task_id, client, current, served_declaration=served_declaration)
    assert _faults(a, b, _first_request_text(a_model), _first_request_text(b_model)) == []


async def test_b_records_the_users_answer_by_the_harness(
    b_side: Side,
) -> None:
    client, current, server = b_side
    b, _ = await _b_run(REFUSAL_TASK, client, current)
    names = [call["name"] for call in tool_calls(b.events)]
    assert names == ["propose_plan", ASK_TOOL, "propose_plan"], names
    assert [c.tool for c in client.calls] == [
        "start_plan",
        "propose_plan",
        ASK_TOOL,
        "propose_plan",
    ]
    assert b.probe.plan is not None
    (answer,) = server.service.sessions[b.state[PLAN_SESSION_KEY]].answers
    assert answer["by"] == BY_HARNESS
    assert answer["answer"] == COMBINE_ANSWER


async def test_b_control_a_resume_that_records_no_answer_lands_no_plan(
    b_side: Side, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def skip(self: McpPlannerSeam, ctx: Any) -> None:
        del self, ctx

    monkeypatch.setattr(McpPlannerSeam, "_record_answers", skip)
    client, current, _ = b_side
    b, _ = await _b_run(REFUSAL_TASK, client, current)
    refusals = [
        (call["response"] or {}).get("code")
        for call in tool_calls(b.events)
        if call["name"] == "propose_plan"
    ]
    print("control, no answer recorded:", refusals, b.probe.refused_code)
    assert b.probe.plan is None
    assert refusals[-1] == "combination_unasked"


async def test_b_control_a_session_opened_without_the_searches_shows_another_first_request(
    b_side: Side, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_searches(task: Any) -> Any:
        async def hand_off() -> tuple[list[Any], list[dict[str, Any]]]:
            return [ask if isinstance(ask, str) else ask.model_dump() for ask in task.asks], []

        return hand_off

    monkeypatch.setattr(ab, "_hand_off", no_searches)
    client, current, _ = b_side
    task_id = "series-gdp-ghana"
    a_model = ScriptedLlm()
    await run_entry(
        TASKS[task_id], load_recording(task_id), prefix_cache=None, model=a_model, timeout_s=60
    )
    b, b_model = await _b_run(task_id, client, current)
    print("control, no searches:", b.probe.refused_code, len(_first_request_text(b_model)))
    assert _first_request_text(a_model) != _first_request_text(b_model)

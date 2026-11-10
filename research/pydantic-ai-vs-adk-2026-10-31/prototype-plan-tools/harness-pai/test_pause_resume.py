"""Pass line 4: the ask pauses the run, the pause survives a serialised history, and the
resume records the user's answer on plan-mcp, `by: harness`, so the next proposal lands.

On `sum-oda-oof-ethiopia` the scripted model proposes, is refused `combination_unasked`,
and calls the ask. The run returns `DeferredToolRequests`. The history and the requests
go through JSON; a fresh agent over a fresh MCP connection records the answer and resumes
with `DeferredToolResults`. The plan that lands has the revision arm B lands in
`harness-ab/test_b_matches_a.py::test_b_records_the_users_answer_by_the_harness`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from checks import COMBINATION_ASK, COMBINE_ANSWER
from mcp_planner import BY_HARNESS
from outward import ASK_TOOL
from pai_plan_mcp import PlanDeps, open_plan, plan_agent, plan_toolsets, record_answers
from planner.plan import load_plan, plan_revision
from pydantic import TypeAdapter
from pydantic_ai.messages import ModelMessagesTypeAdapter
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults
from rig import Rig, Script, b_client, b_run, call, hand_off_of, say
from test_errors import tool_returns
from validate import task_payloads

TASK = "sum-oda-oof-ethiopia"
ANSWER = {"result": {"answers": {"q1": COMBINE_ANSWER}}}
REQUESTS = TypeAdapter(DeferredToolRequests)


@dataclass
class Paused:
    """What a harness keeps while the user answers: the run's history and requests as
    JSON, and the plan session they draft in."""

    history: bytes
    requests: bytes
    plan_session: str
    first_answer: dict[str, Any]


def proposal(rig: Rig) -> dict[str, Any]:
    return dict(task_payloads(rig.use(TASK))[0].body)


async def pause(rig: Rig) -> tuple[Paused, DeferredToolRequests]:
    """Run to the ask: propose, be refused `combination_unasked`, ask the user."""
    body = proposal(rig)
    task = rig.use(TASK)
    asks, searches = hand_off_of(task)
    mcp, view = plan_toolsets(rig.source)
    script = Script(
        [call("propose_plan", body, id="p1"), call(ASK_TOOL, COMBINATION_ASK, id="ask1")]
    )
    async with mcp:
        opened = await open_plan(mcp, brief=task.question, asks=asks, searches=searches)
        deps = PlanDeps(opened["plan_session"])
        result = await plan_agent(script.model, view, instruction="").run(
            opened["planner_message"], deps=deps
        )
    assert isinstance(result.output, DeferredToolRequests)
    (refused,) = tool_returns(result.all_messages())
    paused = Paused(
        history=ModelMessagesTypeAdapter.dump_json(result.all_messages()),
        requests=REQUESTS.dump_json(result.output),
        plan_session=deps.plan_session,
        first_answer=refused.content,
    )
    return paused, result.output


async def resume(rig: Rig, paused: Paused, *, record: bool) -> tuple[Script, Any]:
    """A fresh agent over a fresh connection finishes the run with the user's answer."""
    requests = REQUESTS.validate_json(paused.requests)
    history = ModelMessagesTypeAdapter.validate_json(paused.history)
    mcp, view = plan_toolsets(rig.source)
    script = Script([call("propose_plan", proposal(rig), id="p2"), say("done")])
    async with mcp:
        if record:
            await record_answers(mcp, paused.plan_session, requests, ANSWER)
        (ask,) = requests.calls
        result = await plan_agent(script.model, view, instruction="").run(
            message_history=history,
            deferred_tool_results=DeferredToolResults(calls={ask.tool_call_id: ANSWER}),
            deps=PlanDeps(paused.plan_session),
        )
    return script, result


async def b_revision(rig: Rig) -> str:
    """The plan revision arm B lands on this task, as `test_b_records_the_users_answer`."""
    body = proposal(rig)
    propose, ask = ("propose_plan", body), (ASK_TOOL, COMBINATION_ASK)
    async with b_client(rig) as client:
        run = await b_run(rig, client, TASK, [[propose], [ask], [propose]], answers=[ANSWER])
    assert run.probe.plan is not None
    return plan_revision(run.probe.plan)


async def test_the_ask_pauses_and_the_resume_lands_the_plan_b_lands(rig: Rig) -> None:
    expected = await b_revision(rig)
    paused, requests = await pause(rig)

    assert paused.first_answer["code"] == "combination_unasked"
    (ask,) = requests.calls
    assert ask.tool_name == ASK_TOOL and ask.args_as_dict() == COMBINATION_ASK
    session = rig.server.service.sessions[paused.plan_session]
    assert session.answers == [] and load_plan(session.state) is None
    assert (
        ModelMessagesTypeAdapter.dump_json(ModelMessagesTypeAdapter.validate_json(paused.history))
        == paused.history
    )

    script, result = await resume(rig, paused, record=True)

    (answer,) = session.answers
    assert answer["by"] == BY_HARNESS and answer["answer"] == COMBINE_ANSWER
    # The model's first request after the resume carries the user's answer as the ask's return.
    ask_return = next(p for p in tool_returns(script.requests[0]) if p.tool_name == ASK_TOOL)
    assert ask_return.outcome == "success" and ask_return.content == ANSWER
    landed = [p for p in tool_returns(result.all_messages()) if p.tool_name == "propose_plan"][-1]
    assert landed.outcome == "success" and landed.content["outcome"] == "ok"
    plan = load_plan(session.state)
    assert plan is not None and plan_revision(plan) == expected
    print("revision", expected, "| session answers", len(session.answers), "by", answer["by"])


async def test_control_a_resume_that_records_nothing_is_refused_again_and_lands_no_plan(
    rig: Rig,
) -> None:
    paused, _ = await pause(rig)
    _, result = await resume(rig, paused, record=False)

    session = rig.server.service.sessions[paused.plan_session]
    last = [p for p in tool_returns(result.all_messages()) if p.tool_name == "propose_plan"][-1]
    print("control, no answer recorded:", last.content["code"])
    assert last.content["code"] == "combination_unasked"
    assert session.answers == [] and load_plan(session.state) is None

"""I6: refusals name `record_user_answer`, and a referral answers `referred`.

The planner's refusals name its own ask tool, `adk_request_input`, in their
`action` and message; plan-mcp's name `record_user_answer`, the tool a client
holds. A proposal translation refers back answers `ok` with no plan ready from
the planner, and `referred` under the referral's kind from plan-mcp.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from checks import call, opening, recorded_server
from entry import Task, load_tasks, quiet_logs
from fastmcp import Client
from test_p2_refusals import DRC, _drc_task
from validate import task_payloads

TASKS = load_tasks()
REFUSED = [task for task in TASKS if task_payloads(task)[0].refusal is not None]


@pytest.fixture(autouse=True, scope="module")
def _quiet() -> None:
    quiet_logs()


async def _both(task: Task, body: dict[str, Any], recording: Any = None) -> tuple[Any, Any]:
    """The planner's own answer to `body` and plan-mcp's, each on a fresh session."""
    answers = []
    for raw in (True, False):
        recorded = await recorded_server(task, recording=recording)
        async with Client(recorded.server.mcp) as client:
            opened = await call(client, "start_plan", opening(task))
            if raw:
                service = recorded.server.service
                session = service.sessions[opened["plan_session"]]
                answers.append(await service.propose(session, body))
            else:
                sent = {"plan_session": opened["plan_session"], **body}
                answers.append(await call(client, "propose_plan", sent))
    return answers[0], answers[1]


@pytest.mark.parametrize("task", REFUSED, ids=[task.id for task in REFUSED])
async def test_i6_a_refusals_action_names_record_user_answer(task: Task) -> None:
    planner, mine = await _both(task, task_payloads(task)[0].body)
    print("I6 action:", planner["action"]["call"], "->", mine["action"]["call"])
    assert planner["action"]["call"] == "adk_request_input"
    assert mine["action"]["call"] == "record_user_answer"
    assert "adk_request_input" not in json.dumps(mine)


async def test_i6_a_referral_answers_referred() -> None:
    recording = {"searches": [], "tools": {}, "translation": None}
    planner, mine = await _both(_drc_task(), DRC["first_attempt"], recording)
    print("I6 referral:", planner["outcome"], "->", mine["outcome"], mine["code"])
    assert planner["outcome"] == "ok" and "summary" not in planner
    assert mine["outcome"] == "referred" and mine["code"] == planner["translation"]["kind"]
    assert "card" not in mine

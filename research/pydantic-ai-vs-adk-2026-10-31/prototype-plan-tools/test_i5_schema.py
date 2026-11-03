"""I5: a schema the client can learn.

An `invalid_plan` refusal names, under `expected`, the schema of each field its
errors point at, without prose and with the bounds the lean declaration leaves
out. The planner's guidance is served as the resource `plan-mcp://guidance` and
the prompt `plan_guidance`, and in `start_plan`'s answer. P5 measures what an
outside model reads before its first proposal.
"""

from __future__ import annotations

import copy

import pytest
from checks import call, mcp_first_answer, opening, recorded_server
from entry import load_tasks, quiet_logs
from fastmcp import Client
from outward import outward
from server import GUIDANCE_URI

TASKS = {task.id: task for task in load_tasks()}
TASK = TASKS["growth-total-oda-ghana"]


@pytest.fixture(autouse=True, scope="module")
def _quiet() -> None:
    quiet_logs()


async def test_i5_a_refusal_names_the_expected_shape_of_the_field_that_failed() -> None:
    body = copy.deepcopy(TASK.reference_payload)
    body["evidence_requirements"][0]["period"] = {"start": "twenty twelve", "end": 2021}
    del body["evidence_requirements"][0]["measure"]
    refused, _ = await mcp_first_answer(TASK, body)
    print("I5 expected:", refused["expected"])
    assert refused["code"] == "invalid_plan"
    start = refused["expected"]["evidence_requirements[0].period.start"]
    assert start["type"] == "integer" and start["minimum"] == 1900
    assert refused["expected"]["evidence_requirements[0].measure"]["type"] == "string"


async def test_i5_control_the_planners_own_refusal_names_no_shape() -> None:
    body = copy.deepcopy(TASK.reference_payload)
    body["evidence_requirements"][0]["period"] = {"start": "twenty twelve", "end": 2021}
    refused, _ = await mcp_first_answer(TASK, body)
    planner_own = {key: value for key, value in refused.items() if key != "expected"}
    assert "expected" not in outward(planner_own)
    assert "expected" in refused


async def test_i5_the_guidance_is_a_resource_a_prompt_and_in_the_answer() -> None:
    recorded = await recorded_server(TASK)
    async with Client(recorded.server.mcp) as client:
        opened = await call(client, "start_plan", opening(TASK))
        (resource,) = await client.read_resource(GUIDANCE_URI)
        prompt = await client.get_prompt("plan_guidance")
    text = getattr(resource, "text", "")
    print("I5 guidance:", len(text), "characters")
    assert text == opened["guidance"] == recorded.server.guidance
    assert prompt.messages[0].content.text == text  # type: ignore[union-attr]
    assert "adk_request_input" not in text and "`record_user_answer`" in text


@pytest.mark.parametrize("missing", ["asks", "brief"])
async def test_i5_a_missing_argument_is_refused_with_its_shape(missing: str) -> None:
    recorded = await recorded_server(TASK)
    arguments = {key: value for key, value in opening(TASK).items() if key != missing}
    async with Client(recorded.server.mcp) as client:
        refused = await call(client, "start_plan", arguments)
        opened = await call(client, "start_plan", opening(TASK))
    assert refused["code"] == "invalid_arguments"
    assert set(refused["expected"]) == {missing}
    assert refused["expected"][missing]["type"] in ("array", "string")
    # Control: the same call with every argument opens a session.
    assert opened["outcome"] == "ok"

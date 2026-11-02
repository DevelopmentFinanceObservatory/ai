"""P1: every planner-entry task's reference plan passes through plan-mcp as it passes
through dev's planner entry: the same first refusal where there is one, the same
translation, the same plan on record, the period reduction and kinds validate.py
holds it to, and the same first message to the planner."""

from __future__ import annotations

import copy

import pytest
from checks import p1_faults
from entry import Task, load_tasks, quiet_logs

TASKS = load_tasks()


@pytest.fixture(autouse=True, scope="module")
def _quiet() -> None:
    quiet_logs()


def test_the_set_is_the_seventy_tasks() -> None:
    assert len(TASKS) == 70


@pytest.mark.parametrize("task", TASKS, ids=[task.id for task in TASKS])
async def test_p1_the_reference_plan_passes_through_the_tools(task: Task) -> None:
    assert await p1_faults(task) == []


async def test_p1_control_a_payload_with_one_field_broken_is_refused() -> None:
    task = next(task for task in TASKS if task.id == "growth-total-oda-ghana")
    broken = copy.deepcopy(task.reference_payload)
    broken["evidence_requirements"][0]["period"] = {"start": "twenty twelve", "end": 2021}
    faults = await p1_faults(task, body=broken)
    print("P1 control:", faults)
    assert faults and faults[0].startswith("not accepted: invalid_input/")

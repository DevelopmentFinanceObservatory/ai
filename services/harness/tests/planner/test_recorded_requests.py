"""Each request of each recorded entry equals its stored expectation.

`evals/planner_entry` holds a set of tasks, each with the plans a correct planner
proposes for it (a reference plan and, for some tasks, alternatives) and a recording
of the data-mcp answers its entry reads. Each case here runs one such plan through
the eval's own entry (`entry.run_entry`: the orchestrator's tools, the graph nodes and
`PlannerSeam`) over the recording, with a scripted google-genai client playing the
model. A plan the real `propose_plan` refuses first, as the recording of a task with a
`first_proposal_refusal` says, is followed by the planner's ask and the user's answer,
which resumes the run.

Every request the model receives, and the plan the entry lands, equal the case's
stored expectation (`tests.planner_expectations`). That covers the first request of
each entry, built from the recorded readings, skills and asks, and, for an entry with
an ask, the request after the ask is answered. The stored requests hold the seam's
brief as a marker (`planner_expectations.BRIEF`). No model or network is involved,
since the recorded answers replay from files.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest
from harness.turn_events import PLANNER_ASK
from planner.settings import AgentSettings
from submission.service import COMBINATION_OPTIONS, COMBINE_OPTION

from ..planner_expectations import expect, expectation_name, shared
from ..scripted_genai import Reply, Request, ScriptedGenai, call_part, reply
from ..test_planner_entry_eval import entry, validate
from ..test_planner_node import _briefs, _numbered_call_ids, _request_dumped

_MODEL = AgentSettings().planner_model_name

_CASES = [
    pytest.param(task, payload, id=f"{task.id}:{payload.label}")
    for task in entry.load_tasks()
    for payload in validate.task_payloads(task)
]


def _script(payload: Any) -> tuple[list[Reply], list[Mapping[str, Any]]]:
    """The model's replies for `payload`, and the user's answers to the asks among them."""
    proposal = reply(call_part("propose_plan", dict(payload.body)))
    if payload.refusal is None:
        return [proposal], []
    ask = reply(
        call_part(
            PLANNER_ASK,
            {
                "message": "Do these two datasets add up without counting a flow twice?",
                "options": [dict(option) for option in COMBINATION_OPTIONS],
            },
        )
    )
    # The reply the studio's interrupt route sends for an answered pause.
    answer = {"result": {"answers": {"q1": {"kind": "option", "spec_id": COMBINE_OPTION}}}}
    return [proposal, ask, proposal], [answer]


async def _requests(task: Any, payload: Any) -> tuple[list[Request], Any]:
    """The requests the model receives in the entry of `payload`, and the run."""
    replies, answers = _script(payload)
    scripted = ScriptedGenai(replies)
    with _numbered_call_ids():
        run = await entry.run_entry(
            task,
            entry.load_recording(task.id),
            prefix_cache=None,
            client=scripted.client,
            answers=answers,
        )
    assert run.error is None, run.error
    assert not run.surface_misses
    return scripted.requests, run


@pytest.mark.parametrize(("task", "payload"), _CASES)
async def test_each_request_equals_its_stored_expectation(
    task: Any, payload: Any, request: pytest.FixtureRequest
) -> None:
    entry.quiet_logs()
    sent, run = await _requests(task, payload)

    assert len(sent) == (3 if payload.refusal is not None else 1)
    assert all(each.model == _MODEL for each in sent)
    assert run.probe.plan is not None
    assert all(len(_briefs(each)) == 1 for each in sent), "one brief a request"
    plan = run.probe.plan.model_dump(mode="json")
    dumped = [_request_dumped(each) for each in sent]
    for each in dumped:
        each["config"] = shared(each["config"], "request_config_recorded_entry")
    expect(expectation_name(request), {"requests": dumped, "plan": plan})

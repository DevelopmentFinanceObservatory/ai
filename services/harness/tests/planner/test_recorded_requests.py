"""The first request of each recorded planner entry is the same on both planners.

`evals/planner_entry` holds a set of tasks, each with the plans a correct planner
proposes for it (a reference plan and, for some tasks, alternatives) and a recording
of the data-mcp answers its entry reads. Each case here runs one such plan through
the eval's own entry (`entry.run_entry`: the orchestrator's tools, the graph nodes and
`PlannerSeam`) over the recording, with a scripted google-genai client playing the
model, once on the ADK planner and once on the Pydantic AI planner. A plan the real
`propose_plan` refuses first, as the recording of a task with a
`first_proposal_refusal` says, is followed by the planner's ask and the user's answer,
which resumes the run.

Every request the model receives has to equal the ADK planner's apart from the wire
differences `assert_pinned_wire_differences` lists. That covers the first request of
each entry, built from the recorded readings, skills and asks, and, for an entry with
an ask, the request after the ask is answered. No model or network is involved, since
the recorded answers replay from files.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest
from harness.turn_events import PLANNER_ASK
from planner.settings import AgentSettings
from submission.service import COMBINATION_OPTIONS, COMBINE_OPTION

from ..scripted_genai import Reply, Request, ScriptedGenai, call_part, reply
from ..test_planner_entry_eval import entry, validate
from ..test_planner_node import Runner, _numbered_call_ids
from .planner_pairs import assert_pinned_wire_differences

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


async def _requests(
    runner: Runner, task: Any, payload: Any, monkeypatch: pytest.MonkeyPatch
) -> tuple[list[Request], Any]:
    """The requests the model receives in the entry of `payload` on `runner`, and the run."""
    replies, answers = _script(payload)
    scripted = ScriptedGenai(replies)
    monkeypatch.setenv("PLANNER_RUNNER", runner)
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
async def test_each_request_equals_the_adk_planners_apart_from_the_pinned_differences(
    task: Any, payload: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry.quiet_logs()
    adk, adk_run = await _requests("adk", task, payload, monkeypatch)
    pydantic, pydantic_run = await _requests("pydantic_ai", task, payload, monkeypatch)

    expected = 3 if payload.refusal is not None else 1
    assert len(adk) == len(pydantic) == expected
    for sent_adk, sent_pydantic in zip(adk, pydantic, strict=True):
        assert sent_adk.model == sent_pydantic.model == _MODEL
        assert_pinned_wire_differences(sent_adk, sent_pydantic)
    assert adk_run.probe.plan == pydantic_run.probe.plan
    assert adk_run.probe.plan is not None

"""P2: refused drafts answer through plan-mcp as the planner's own `propose_plan`
answers them on dev's planner entry, as `outward.outward` words them for an outside
model: the ask tool named `record_user_answer`, a referral answered `referred`, and
`expected` added to an `invalid_plan` refusal.

The recorded refused drafts are the planner-entry payloads marked
`first_proposal_refusal`. Each mutated draft breaks one task's reference payload
one way, so the comparison also covers refusals that carry no action and one whose
action names `propose_plan`. The DRC drafts in
`services/harness/tests/fixtures/propose_plan_drc_aid.json`, proposed on a session
with no search, are compared too: the chain admits `first_attempt` and `plan_v1`,
which were refused when recorded, and refuses `plan_v3`, a patch with no
`question`, as a first draft.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from checks import dev_first_run, mcp_first_answer, refusal_faults
from entry import Task, load_tasks, quiet_logs
from outward import outward
from validate import task_payloads

TASKS = {task.id: task for task in load_tasks()}
ROOT = next(p for p in Path(__file__).resolve().parents if (p / "services" / "harness").is_dir())
DRC = json.loads(
    (ROOT / "services/harness/tests/fixtures/propose_plan_drc_aid.json").read_text("utf-8")
)


@pytest.fixture(autouse=True, scope="module")
def _quiet() -> None:
    quiet_logs()


RECORDED = [
    (task.id, payload.label, payload.body)
    for task in TASKS.values()
    for payload in task_payloads(task)
    if payload.refusal is not None
]


def _without(key: str) -> Callable[[dict[str, Any]], None]:
    def edit(body: dict[str, Any]) -> None:
        body.pop(key, None)

    return edit


def _unknown_output(body: dict[str, Any]) -> None:
    body["answered"] = [{"ask": 0, "outputs": ["no_such_output"]}]


def _across_years_unasked(body: dict[str, Any]) -> None:
    body["across_years"] = [body["evidence_requirements"][-1]["id"]]


def _period_in_words(body: dict[str, Any]) -> None:
    body["evidence_requirements"][0]["period"] = {"start": "twenty twelve", "end": 2021}


MUTATIONS = {
    "no_answered": _without("answered"),
    "unknown_output": _unknown_output,
    "across_years_unasked": _across_years_unasked,
    "period_in_words": _period_in_words,
}
MUTATED_TASKS = (
    "growth-total-oda-ghana",
    "sum-oda-oof-ethiopia",
    "series-population-three",
    "aggregate-total-education-oda-kenya-all-providers",
)


def _mutated(task_id: str, mutation: str) -> dict[str, Any]:
    body = copy.deepcopy(TASKS[task_id].reference_payload)
    MUTATIONS[mutation](body)
    return body


DRAFTS = [(task_id, label, body) for task_id, label, body in RECORDED] + [
    (task_id, mutation, _mutated(task_id, mutation))
    for task_id in MUTATED_TASKS
    for mutation in MUTATIONS
]


def test_the_recorded_set_holds_every_refusal_task() -> None:
    assert len(RECORDED) == 4


@pytest.mark.parametrize(
    ("task_id", "label", "body"), DRAFTS, ids=[f"{t}:{lb}" for t, lb, _ in DRAFTS]
)
async def test_p2_a_refused_draft_answers_as_the_planner_answers_it(
    task_id: str, label: str, body: dict[str, Any]
) -> None:
    task = TASKS[task_id]
    dev, _ = await dev_first_run(task, body)
    mine, schema = await mcp_first_answer(task, body)
    print(
        f"P2 {task_id} {label}: {dev.get('outcome')}/{dev.get('code')} action={dev.get('action')}"
    )
    assert dev.get("outcome") != "ok"
    assert refusal_faults(mine, outward(dev, schema=schema)) == []
    assert mine == outward(dev, schema=schema)


def test_p2_control_the_comparison_catches_a_changed_code() -> None:
    dev = {"outcome": "refused", "code": "combination_unasked", "action": {"call": "x"}}
    changed = {**dev, "code": "asks_unaccounted"}
    assert refusal_faults(changed, dev) == [".code: 'asks_unaccounted' != 'combination_unasked'"]


def _drc_task() -> Task:
    question = DRC["first_attempt"]["question"]
    return Task.model_construct(
        id="drc-aid",
        kind="share_of",
        origin="traced",
        question=question,
        searches=[],
        brief=question,
        asks=list(DRC["asks"]),
        reference_payload=DRC["first_attempt"],
    )


@pytest.mark.parametrize("version", ["first_attempt", "plan_v1", "plan_v3"])
async def test_p2_the_drc_drafts_answer_as_the_planner_answers_them(version: str) -> None:
    recording = {"searches": [], "tools": {}, "translation": None}
    task = _drc_task()
    dev, dev_translation = await dev_first_run(task, DRC[version], recording=recording)
    mine, schema = await mcp_first_answer(task, DRC[version], recording=recording)
    print(f"P2 DRC {version}: {mine.get('outcome')}/{mine.get('code')}")
    if dev_translation is None:
        assert dev.get("outcome") != "ok"
        assert mine == outward(dev, schema=schema)
    else:
        assert mine["outcome"] in ("ok", "referred")
        assert mine["translation"] == dev_translation

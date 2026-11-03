"""I1: the client's own searches reach the plan session.

`start_plan` takes the spec_ids the client's `search` calls returned. Each is read
as `get_data` reads a bare id (the registry's identity, which must hash to it) and
asked of Search again through `resolve_definition`, which must answer it as that
one dataset. A plan pinning the searched dataset then lands as it lands when the
server runs the same searches itself. With neither, the translator searches on its
own and refers the requirement back, which is how the live clients' pins failed.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest
from checks import call, opening, recorded_server, searched_parts
from entry import Task, load_recording, load_tasks, quiet_logs
from fastmcp import Client
from plan_tools import SPEC_NOT_HELD, SPEC_NOT_RESOLVED

TASKS = {task.id: task for task in load_tasks()}
# Tasks whose translator, without the session's sources, searches for the pinned
# dataset itself, a search their recordings do not hold.
PINNED = ("aggregate-total-oda-kenya", "rank-top-donors-malawi")


@pytest.fixture(autouse=True, scope="module")
def _quiet() -> None:
    quiet_logs()


def _pinned(task: Task, spec_id: str) -> dict[str, Any]:
    body = copy.deepcopy(task.reference_payload)
    for requirement in body["evidence_requirements"]:
        if requirement["kind"] == "dataset":
            requirement["source_spec_id"] = spec_id
    return body


async def _landed(task: Task, sources: dict[str, Any]) -> tuple[dict[str, Any], str, list[str]]:
    """The answer to the reference plan pinned to the searched dataset, on a session
    opened with `sources`, the planner's first message, and the spec_ids Search was
    asked to resolve."""
    (spec_id, *_) = searched_parts(load_recording(task.id))
    recorded = await recorded_server(task)
    async with Client(recorded.server.mcp) as client:
        opened = await call(client, "start_plan", {**opening(task), **sources})
        assert opened["outcome"] == "ok", opened
        answer = await call(
            client,
            "propose_plan",
            {"plan_session": opened["plan_session"], **_pinned(task, spec_id)},
        )
    surface = recorded.server.service._surface
    return answer, opened["planner_message"], surface.asked  # type: ignore[attr-defined]


@pytest.mark.parametrize("task_id", PINNED)
async def test_i1_a_plan_pinning_a_searched_spec_id_lands_as_with_the_searches(
    task_id: str,
) -> None:
    task = TASKS[task_id]
    (spec_id, *_) = searched_parts(load_recording(task.id))
    searched, _, _ = await _landed(task, {})
    by_id, message, asked = await _landed(task, {"searches": [], "spec_ids": [spec_id]})
    print(f"I1 {task_id}: {by_id['outcome']} v{by_id.get('version')}, asked {asked}")
    assert "summary" in searched, searched
    assert by_id["summary"]["revision"] == searched["summary"]["revision"]
    assert by_id["translation"] == searched["translation"]
    assert spec_id in message
    assert asked == [spec_id]


@pytest.mark.parametrize("task_id", PINNED)
async def test_i1_control_a_session_without_the_searches_refers_the_pin_back(
    task_id: str,
) -> None:
    answer, _, _ = await _landed(TASKS[task_id], {"searches": []})
    print(f"I1 control {task_id}: {answer['outcome']}/{answer.get('code')}")
    assert answer["outcome"] == "referred"


async def test_i1_an_unknown_or_forged_spec_id_is_refused() -> None:
    task = TASKS[PINNED[0]]
    (spec_id, *_) = searched_parts(load_recording(task.id))
    forged = "spec_" + "0" * 32
    recorded = await recorded_server(task)
    lookup = recorded.server.service._definitions
    assert lookup is not None

    async def definitions(asked: str) -> Any:
        # The registry answers the forged id with another dataset's identity.
        return await lookup(spec_id if asked == forged else asked)

    recorded.server.service._definitions = definitions
    async with Client(recorded.server.mcp) as client:
        unknown = await call(client, "start_plan", {**opening(task), "spec_ids": ["spec_x"]})
        mismatch = await call(client, "start_plan", {**opening(task), "spec_ids": [forged]})
    print("I1 refused:", unknown["code"], "|", mismatch["message"][:100])
    assert unknown["code"] == SPEC_NOT_HELD
    assert mismatch["code"] == SPEC_NOT_HELD


async def test_i1_a_spec_id_search_answers_as_another_dataset_is_refused() -> None:
    task = TASKS[PINNED[0]]
    (spec_id, *_) = searched_parts(load_recording(task.id))
    recorded = await recorded_server(task)
    other = next(iter(searched_parts(load_recording("series-gdp-ghana")).values()))
    recorded.server.service._surface._parts[spec_id] = other  # type: ignore[attr-defined]
    async with Client(recorded.server.mcp) as client:
        answer = await call(client, "start_plan", {**opening(task), "spec_ids": [spec_id]})
    print("I1 another dataset:", answer)
    assert answer["code"] == SPEC_NOT_RESOLVED

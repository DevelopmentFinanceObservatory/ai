"""I4: the user answers and approves directly.

A client that declares MCP elicitation has plan-mcp ask the user itself: the
planner's question through `record_user_answer`, and the plan card through
`run_plan`, each as an MCP 2026-07-28 input request the client's elicitation
handler answers; a handshake-era connection is asked through `ctx.elicit`, and
both eras are driven. What the model sends as `answer` or `approved` is not read then.
A client without elicitation sends the user's answer and `approved: true`, and
every answer and approval records who gave it, `user` or `client`. `run_plan`
refuses a version with no approval on record.
"""

from __future__ import annotations

from typing import Any

import pytest
from checks import (
    COMBINATION_ASK,
    SCRIPTED_OPENING,
    call,
    client_of,
    opening,
    recorded_server,
    scripted_server,
)
from entry import load_tasks, quiet_logs
from fastmcp import Client
from plan_tools import NOT_APPROVED
from scripted import PROPOSAL
from validate import task_payloads

TASKS = load_tasks()
# A task whose reference payload is first refused `combination_unasked`.
COMBINED = next(task for task in TASKS if task_payloads(task)[0].refusal is not None)
SHOW_APART = {"kind": "option", "spec_id": "show them separately"}


@pytest.fixture(autouse=True, scope="module")
def _quiet() -> None:
    quiet_logs()


class User:
    """An elicitation handler: answers each request from `replies` in turn and keeps
    the message it was shown."""

    def __init__(self, *replies: Any) -> None:
        self._replies = list(replies)
        self.shown: list[str] = []

    async def __call__(self, message: str, response_type: Any, params: Any, context: Any) -> Any:
        del response_type, params, context
        self.shown.append(message)
        return self._replies.pop(0)


MODES = ("auto", "legacy")


async def _ran(
    user: User | None, arguments: dict[str, Any], *, elicitation: bool = True, mode: str = "auto"
) -> Any:
    scripted = scripted_server(elicitation=elicitation)
    async with Client(scripted.server.mcp, elicitation_handler=user, mode=mode) as client:
        opened = await call(client, "start_plan", SCRIPTED_OPENING)
        session = opened["plan_session"]
        accepted = await call(client, "propose_plan", {"plan_session": session, **PROPOSAL})
        ran = await call(client, "run_plan", {"plan_session": session, "version": 1, **arguments})
    return ran, accepted["card"], scripted.server.service.sessions[session]


@pytest.mark.parametrize("mode", MODES)
async def test_i4_the_user_approves_the_card_through_elicitation(mode: str) -> None:
    user = User({"approve": True, "comment": ""})
    ran, card, session = await _ran(user, {"approved": False}, mode=mode)
    print("I4 approved:", ran["status"], ran["approved_by"], "| shown:", user.shown[0][:80])
    assert ran["status"] == "finished" and ran["approved_by"] == "user"
    assert card["title"] in user.shown[0]
    assert session.approvals[1]["by"] == "user"


@pytest.mark.parametrize("mode", MODES)
async def test_i4_a_user_who_declines_stops_the_run_whatever_the_model_says(mode: str) -> None:
    user = User({"approve": False, "comment": "Start in 1990"})
    ran, _, session = await _ran(user, {"approved": True}, mode=mode)
    print("I4 declined:", ran)
    assert ran["code"] == NOT_APPROVED
    assert ran["detail"] == {"by": "user", "comment": "Start in 1990"}
    assert session.runs == {}


async def test_i4_a_client_without_elicitation_approves_on_its_word() -> None:
    unapproved, _, _ = await _ran(None, {})
    approved, _, session = await _ran(None, {"approved": True})
    print("I4 client:", unapproved["code"], approved["status"], approved["approved_by"])
    assert unapproved["code"] == NOT_APPROVED
    assert approved["status"] == "finished" and approved["approved_by"] == "client"
    assert session.approvals[1]["by"] == "client"


async def test_i4_control_without_elicitation_the_models_approval_runs_the_plan() -> None:
    user = User({"approve": False, "comment": "no"})
    ran, _, _ = await _ran(user, {"approved": True}, elicitation=False)
    print("I4 control:", ran["status"], ran["approved_by"], "user asked:", user.shown)
    assert ran["status"] == "finished" and user.shown == []


async def _answered(
    user: User | None, *, elicitation: bool = True, mode: str = "auto"
) -> tuple[Any, Any, Any]:
    recorded = await recorded_server(COMBINED, elicitation=elicitation)
    body = task_payloads(COMBINED)[0].body
    async with Client(recorded.server.mcp, elicitation_handler=user, mode=mode) as client:
        opened = await call(client, "start_plan", opening(COMBINED))
        session = opened["plan_session"]
        refused = await call(client, "propose_plan", {"plan_session": session, **body})
        answered = await call(
            client,
            "record_user_answer",
            {"plan_session": session, **COMBINATION_ASK, "answer": SHOW_APART},
        )
        again = await call(client, "propose_plan", {"plan_session": session, **body})
    record = recorded.server.service.sessions[session].answers
    return refused, answered, (again, record)


@pytest.mark.parametrize("mode", MODES)
async def test_i4_the_user_answers_the_planners_question_through_elicitation(mode: str) -> None:
    user = User({"value": "add them"})
    refused, answered, (again, record) = await _answered(user, mode=mode)
    print("I4 answer:", refused["action"]["call"], answered, again["outcome"], record)
    assert refused["action"]["call"] == "record_user_answer"
    assert answered["by"] == "user"
    assert record[-1]["answer"] == {"kind": "option", "spec_id": "add them"}
    assert "summary" in again


async def test_i4_control_without_elicitation_the_models_answer_is_recorded() -> None:
    user = User({"value": "add them"})
    _, answered, (again, record) = await _answered(user, elicitation=False)
    print("I4 answer control:", answered, again.get("code"), user.shown)
    assert answered["by"] == "client" and user.shown == []
    assert record[-1]["answer"] == SHOW_APART
    assert "summary" not in again


async def test_i4_an_unanswered_client_ask_is_refused() -> None:
    scripted = scripted_server()
    async with client_of(scripted.server) as client:
        opened = await call(client, "start_plan", SCRIPTED_OPENING)
        refused = await call(
            client,
            "record_user_answer",
            {"plan_session": opened["plan_session"], "message": "Which years?"},
        )
    assert refused["code"] == "invalid_arguments"

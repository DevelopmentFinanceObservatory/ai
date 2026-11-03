"""P3: an accepted plan runs end to end and hands back what dev's orchestrator receives
for the same plan; `run_plan` refuses an unknown session, a version never accepted
and a superseded version."""

from __future__ import annotations

from typing import Any

from checks import (
    SCRIPTED_OPENING,
    call,
    client_of,
    dev_hand_back,
    differences,
    scripted_server,
    stable,
)
from fastmcp import Client
from plan_tools import SESSION_UNKNOWN, VERSION_NOT_READY, VERSION_SUPERSEDED
from scripted import PROPOSAL, amended


async def _opened(client: Client) -> str:
    opened = await call(client, "start_plan", SCRIPTED_OPENING)
    assert opened["outcome"] == "ok", opened
    return opened["plan_session"]


async def _accepted(client: Client, session: str, body: dict[str, Any]) -> dict[str, Any]:
    answer = await call(client, "propose_plan", {"plan_session": session, **body})
    assert answer["outcome"] == "ok" and "summary" in answer, answer
    return answer


async def test_p3_an_accepted_plan_hands_back_what_dev_hands_the_orchestrator() -> None:
    scripted = scripted_server()
    async with client_of(scripted.server) as client:
        session = await _opened(client)
        accepted = await _accepted(client, session, PROPOSAL)
        assert accepted["card"]["steps"], accepted["card"]
        ran = await call(
            client, "run_plan", {"plan_session": session, "version": 1, "approved": True}
        )
    assert ran["status"] == "finished", ran
    plan = scripted.server.service.sessions[session].ready[1]
    dev = await dev_hand_back(plan)
    print("P3 hand-back keys:", sorted(ran["hand_back"]), "run:", ran["hand_back"]["run"])
    assert differences(stable(ran["hand_back"]), stable(dev)) == []
    assert ran["hand_back"]["run"]["status"] == "completed"


async def test_p3_a_run_past_its_wait_answers_running_and_the_same_call_waits_on_it() -> None:
    scripted = scripted_server()
    async with client_of(scripted.server) as client:
        session = await _opened(client)
        await _accepted(client, session, PROPOSAL)
        first = await call(
            client,
            "run_plan",
            {"plan_session": session, "version": 1, "approved": True, "wait_s": 0},
        )
        second = await call(
            client, "run_plan", {"plan_session": session, "version": 1, "approved": True}
        )
    assert first["status"] == "running", first
    assert second["status"] == "finished" and second["run_id"] == first["run_id"], second
    assert len(scripted.server.service.sessions[session].runs) == 1


async def test_p3_run_plan_refuses_an_unknown_session_and_a_version_never_accepted() -> None:
    scripted = scripted_server()
    async with client_of(scripted.server) as client:
        session = await _opened(client)
        unknown = await call(
            client, "run_plan", {"plan_session": "ps_unknown", "version": 1, "approved": True}
        )
        never = await call(
            client, "run_plan", {"plan_session": session, "version": 1, "approved": True}
        )
        await _accepted(client, session, PROPOSAL)
        later = await call(
            client, "run_plan", {"plan_session": session, "version": 2, "approved": True}
        )
    assert unknown["code"] == SESSION_UNKNOWN
    assert never["code"] == VERSION_NOT_READY
    assert later["code"] == VERSION_NOT_READY


async def _superseded_run(*, check_version: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    scripted = scripted_server(check_version=check_version)
    async with client_of(scripted.server) as client:
        session = await _opened(client)
        await _accepted(client, session, PROPOSAL)
        second = await _accepted(client, session, amended(1))
        assert second["version"] == 2
        old = await call(
            client, "run_plan", {"plan_session": session, "version": 1, "approved": True}
        )
        current = await call(
            client, "run_plan", {"plan_session": session, "version": 2, "approved": True}
        )
    return old, current


async def test_p3_run_plan_refuses_a_superseded_version_and_runs_the_current_one() -> None:
    old, current = await _superseded_run(check_version=True)
    assert old["code"] == VERSION_SUPERSEDED, old
    assert current["status"] == "finished", current
    assert "period 1990-2023" in current["hand_back"]["plan_record"]


async def test_p3_control_a_run_plan_without_the_version_check_runs_a_superseded_plan() -> None:
    old, _ = await _superseded_run(check_version=False)
    print("P3 control:", old.get("status"), old["hand_back"]["plan_record"].splitlines()[:2])
    assert old["status"] == "finished"
    assert "period 1960-2023" in old["hand_back"]["plan_record"]

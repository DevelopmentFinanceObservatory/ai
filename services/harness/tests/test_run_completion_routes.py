"""A planned turn's run left pending, carried on through the Studio routes over
the conversation's stored run ledger (`harness.run_ledger.StudioRunLedger`)."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pytest
from executor.completion import PUBLISH_ATTEMPTS
from executor.ledger import RunStatus, idempotency_key
from google.genai import types
from harness.conversation import current_conversation_key
from harness.graph import answer as answer_module
from harness.run_ledger import StudioRunLedger
from harness.scripted import ScriptedSurface
from harness.studio_store import StudioStore
from sqlalchemy.ext.asyncio import AsyncEngine

from .graph_turns import agent_read
from .scripted_tools import plan_call
from .studio_routes import orchestrator_app, stream, studio_client, turn_body
from .test_run_completion import _FlakyPublisher
from .turn_driver import CONVERSATION, NO_ADK, no_turn_specs

# The principal `tests.studio_routes.AUTH`'s token names.
PRINCIPAL = "u_" + hashlib.sha256(b"test-token").hexdigest()[:16]


def _frame(frames: list[dict[str, Any]], event: str) -> dict[str, Any]:
    return next(frame for frame in frames if frame["event"] == event)


async def test_a_studio_run_left_pending_is_carried_on_from_the_conversation_ledger(
    monkeypatch: pytest.MonkeyPatch, pg_engine: AsyncEngine
) -> None:
    """Through the studio, over its stored run ledger: a run whose
    publication is unfinished at the end of its turn is kept under a key that
    names the conversation, not the turn, and the next turn, in a session of
    its own, completes it and hands the agent its frames and the notes read
    from the plan stored with the ledger."""
    total = "The figures are the sum of 3 indicators."
    monkeypatch.setattr(answer_module, "_stat_var_totals", lambda *a, **k: [total])
    store = StudioStore(pg_engine, adk=NO_ADK, turn_specs=no_turn_specs)
    ledger = StudioRunLedger(store)
    publisher = _FlakyPublisher()
    app, model = orchestrator_app(
        plan_call("Kenya's debt service since 2015"),
        types.Part.from_text(text="Handed to the planner."),
        types.Part.from_text(text="It is pending."),
        types.Part.from_text(text="It is done."),
        tool_surface=ScriptedSurface(),
        store=store,
        run_ledger=ledger,
        publisher=publisher,
    )
    async with studio_client(app) as client:
        first = await stream(client, "/api/studio/turn", turn_body("Kenya's debt service?"))
        plan = _frame(first, "plan.proposed")
        await stream(
            client,
            "/api/studio/interrupt",
            {
                "kind": "interrupt.response",
                "schema_version": 1,
                "conversation_id": CONVERSATION,
                "interrupt_id": plan["interrupt_id"],
                "decision": "approve",
            },
        )
        after_plan = agent_read(model.requests[-1])
        body = json.loads(
            next(line for line in after_plan.split("\n") if line.startswith('{"frames"'))
        )
        assert (body["run"]["status"], body["frames"]) == ("pending", [])

        publisher.fixed = True
        await stream(client, "/api/studio/turn", turn_body("Is it ready?"))

    next_turn = agent_read(model.requests[-1])
    runs = json.loads(next(line for line in next_turn.split("\n") if line.startswith('{"runs"')))
    (resumed,) = runs["runs"]
    assert (resumed["run_id"], resumed["status"]) == (body["run"]["run_id"], "completed")
    assert [frame["frame"] for frame in resumed["frames"]]
    assert resumed["totals"] == [total]
    token = current_conversation_key.set((PRINCIPAL, CONVERSATION))
    try:
        run = await ledger.get(resumed["run_id"])
    finally:
        current_conversation_key.reset(token)
    assert run is not None and run.status is RunStatus.COMPLETED
    assert run.idempotency_key == idempotency_key(
        caller=PRINCIPAL,
        conversation_id=f"{PRINCIPAL}/{CONVERSATION}",
        task=f"plan:{plan['revision']}",
    )


async def test_the_default_ledger_keeps_one_conversations_runs_from_another(
    pg_engine: AsyncEngine,
) -> None:
    publisher = _FlakyPublisher()
    app, model = orchestrator_app(
        plan_call("Kenya's debt service since 2015"),
        types.Part.from_text(text="Handed to the planner."),
        types.Part.from_text(text="It is pending."),
        types.Part.from_text(text="Another conversation."),
        tool_surface=ScriptedSurface(),
        publisher=publisher,
        engine=pg_engine,
    )
    async with studio_client(app) as client:
        first = await stream(client, "/api/studio/turn", turn_body("Kenya's debt service?"))
        await stream(
            client,
            "/api/studio/interrupt",
            {
                "kind": "interrupt.response",
                "schema_version": 1,
                "conversation_id": CONVERSATION,
                "interrupt_id": _frame(first, "plan.proposed")["interrupt_id"],
                "decision": "approve",
            },
        )
        assert '"status": "pending"' in agent_read(model.requests[-1])

        publisher.fixed = True
        other = turn_body("Hello", conversation_id="conv_01other000")
        await stream(client, "/api/studio/turn", other)

    assert '{"runs"' not in agent_read(model.requests[-1])
    assert len(publisher.frames) == PUBLISH_ATTEMPTS

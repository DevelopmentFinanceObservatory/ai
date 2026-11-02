"""A Studio app on dev's root workflow with dev's scripted orchestrator, dev's
scripted planner and a `SlowSurface`, its `execute` node dev's or the background one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from durable_execute import BMode, DurableSurface, install
from executor.ledger import InMemoryRunLedger, RunLedger
from fakes import RecordingPublisher, SlowSurface
from google.adk.sessions import BaseSessionService, InMemorySessionService
from harness.conversation import current_conversation_id
from harness.graph.state import AGENT_NAME
from harness.studio_state import StudioStore
from tests.orchestrator_fakes import CONVERSATION, orchestrator_app, plan_call

BRIEF = "Life expectancy in Kenya since 1960"
type Arm = Literal["durable", "dev"]


@dataclass
class StudioApp:
    app: Any
    model: Any
    surface: SlowSurface
    ledger: RunLedger
    publisher: RecordingPublisher
    store: StudioStore
    session_service: BaseSessionService


def build_app(
    patch: Any,
    *,
    arm: Arm,
    mode: BMode = BMode.DURABLE,
    surface: SlowSurface | None = None,
    plans: int = 2,
    session_service: BaseSessionService | None = None,
    ledger: RunLedger | None = None,
    store: StudioStore | None = None,
) -> StudioApp:
    """The app. `arm` `"dev"` leaves dev's `execute` node in place and gives it
    the bare `SlowSurface`; `"durable"` installs the background node in `mode`
    and wraps the surface in `DurableSurface`."""
    surface = surface or SlowSurface()
    session_service = session_service or InMemorySessionService()
    store = store or StudioStore(session_service, app_name=AGENT_NAME)
    ledger = ledger or InMemoryRunLedger(scope=current_conversation_id.get)
    publisher = RecordingPublisher()
    tool_surface: Any = surface
    if arm == "durable":
        install(patch, store=store, mode=mode)
        tool_surface = DurableSurface(surface)
    app, model = orchestrator_app(
        *[plan_call(BRIEF)] * plans,
        tool_surface=tool_surface,
        session_service=session_service,
        store=store,
        run_ledger=ledger,
        publisher=publisher,
    )
    return StudioApp(app, model, surface, ledger, publisher, store, session_service)


def interrupt_body(interrupt_id: str, decision: str, **extra: Any) -> dict[str, Any]:
    return {
        "kind": "interrupt.response",
        "schema_version": 1,
        "conversation_id": CONVERSATION,
        "interrupt_id": interrupt_id,
        "decision": decision,
        **extra,
    }


CANCEL = {"kind": "turn.cancel", "schema_version": 1, "conversation_id": CONVERSATION}

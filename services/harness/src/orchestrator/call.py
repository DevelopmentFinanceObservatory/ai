"""One orchestrator tool call, as the tools read it, and the mark of a hand-off."""

from __future__ import annotations

from dataclasses import dataclass

from planner.plan import SessionState

# The key of the answer of an `ask_user` or `plan` call that recorded its
# action (`harness.orchestrator_tools`). The orchestrator defers such a call:
# its run ends on it, and the user's answer or the planner's result answers it
# when the run resumes (`orchestrator.run.run_orchestrator`).
HANDED_OFF = "handed_off"


@dataclass(slots=True)
class OrchestratorCall:
    """What an orchestrator tool reads and writes besides its arguments.

    `state` is the turn's state: the tools read the turn's searches, frames,
    specs and call counts from it and write theirs to it. `turn_id` names the
    turn and is the same on every run of the turn, before a hand-off and
    after it, so a card drawn before the user answered counts as drawn this
    turn (`harness.orchestrator_tools`). `call_id` is the model's id for this
    call, the id its answer carries, `None` outside a model's call.
    """

    state: SessionState
    turn_id: str
    call_id: str | None = None


__all__ = ["HANDED_OFF", "OrchestratorCall"]

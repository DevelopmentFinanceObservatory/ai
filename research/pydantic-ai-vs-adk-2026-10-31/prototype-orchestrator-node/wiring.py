"""Test-only wiring: one Studio app per arm, and what a turn stored.

`build_app` serves dev's Studio routes on the real root workflow
(`tests.orchestrator_fakes.orchestrator_app`) with the orchestrator script given
as `Step`s. The two arms differ only in the `orchestrate` node:

- `dev`: dev's node and `LlmAgent`, its model `script.StreamingScriptLlm`, put
  where `orchestrator_app` builds its `ScriptLlm`;
- `pai`: `orchestrator_node.pai_orchestrate_node` on `script.pai_model`, put where
  `create_root_workflow` builds dev's `orchestrate_node`. The ADK agent
  `orchestrator_app` still builds is where the node reads the instruction and
  tools from.
"""

import json
from collections.abc import Callable
from typing import Any, Literal

import pytest
from google.adk.sessions import InMemorySessionService
from harness.graph import workflow as workflow_module
from harness.graph.state import AGENT_NAME
from orchestrator_node import Variant, pai_orchestrate_node
from pydantic_ai.models import Model
from script import Played, Step, StreamingScriptLlm, pai_model
from tests import orchestrator_fakes
from tests.orchestrator_fakes import CONVERSATION, orchestrator_app

AUTH = {"Authorization": "Bearer test-token"}
type Arm = Literal["pai", "dev"]


def build_app(
    monkeypatch: pytest.MonkeyPatch,
    steps: list[Step],
    *,
    arm: Arm = "pai",
    variant: Variant = Variant.LIVE,
    answer_failures: bool = True,
    session_service: InMemorySessionService | None = None,
    pai_wrap: Callable[[Model], Model] = lambda model: model,
    dev_model: type[StreamingScriptLlm] = StreamingScriptLlm,
    **app_args: Any,
) -> tuple[Any, Played]:
    """`pai_wrap` wraps the `pai` arm's model; `dev_model` is the `dev` arm's."""
    played = Played(steps=list(steps))
    if arm == "pai":

        def factory(*, orchestrator: Any) -> Any:
            return pai_orchestrate_node(
                orchestrator=orchestrator,
                model=pai_wrap(pai_model(played)),
                variant=variant,
                answer_failures=answer_failures,
            )

        monkeypatch.setattr(workflow_module, "orchestrate_node", factory)
    else:
        monkeypatch.setattr(
            orchestrator_fakes, "ScriptLlm", lambda script: dev_model(played=played)
        )
    app, _ = orchestrator_app(session_service=session_service, **app_args)
    return app, played


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


def events(frames: list[dict[str, Any]]) -> list[str]:
    return [f["event"] for f in frames]


def frame(frames: list[dict[str, Any]], event: str) -> dict[str, Any]:
    (found,) = [f for f in frames if f["event"] == event]
    return found


def turn_sessions(session_service: InMemorySessionService) -> list[Any]:
    """Every per-turn ADK session Studio created, whoever the principal is."""
    return [
        session
        for by_user in session_service.sessions[AGENT_NAME].values()
        for session in by_user.values()
        if session.id.startswith(CONVERSATION + "#")
    ]


# Fields that differ between any two runs: times, and ids minted per run.
# JSON held as a string is compared parsed.
VOLATILE = {
    "observed_at",
    "run_id",
    "plan_version_id",
    "approved_at",
    "at",
    "seq",
    "interrupt_id",
    "step_id",
    "trace_id",
    "duration_ms",
}


def stable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: stable(v) for k, v in value.items() if k not in VOLATILE}
    if isinstance(value, list):
        return [stable(v) for v in value]
    if isinstance(value, str) and value.startswith(("{", "[")):
        try:
            return stable(json.loads(value))
        except ValueError:
            return value
    return value


def differences(a: Any, b: Any, path: str = "") -> list[str]:
    """The paths at which `a` and `b` differ."""
    if isinstance(a, dict) and isinstance(b, dict):
        return [
            d
            for key in sorted(set(a) | set(b), key=str)
            for d in differences(a.get(key), b.get(key), f"{path}.{key}")
        ]
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [
            d
            for i, (x, y) in enumerate(zip(a, b, strict=True))
            for d in differences(x, y, f"{path}[{i}]")
        ]
    if isinstance(a, list) and isinstance(b, list):
        return [f"{path}: {len(a)} items != {len(b)}: {_summary(a)} != {_summary(b)}"]
    return [] if a == b else [f"{path}: {a!r} != {b!r}"]


def _summary(items: list[Any]) -> list[Any]:
    """Frames by event and label, anything else as it is."""
    return [
        f"{i.get('event')}:{i.get('label') or i.get('status') or ''}"
        if isinstance(i, dict) and "event" in i
        else i
        for i in items
    ]

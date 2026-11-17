"""The events the turn graph's nodes emit beside the loop's own
(`harness.turn.events`), which the Studio translator turns into wire frames.

`orchestrate` emits its status line (`AgentStatus`) and the events of the
agent's run as they happen (`orchestrator.run`): answer text as it streams,
each whole model response, and each response's tool answers. `GraphEvent` is
their union.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from orchestrator.run import ModelResponded, TextDelta, ToolsAnswered


@dataclass(frozen=True)
class AgentStatus:
    """The status line Studio shows as an `orchestrate` entry starts:
    `text`, and `failure`, the refusal a plan failed with, when the
    hand-back it reads carries one."""

    text: str
    failure: dict[str, Any] | None = None


type GraphEvent = AgentStatus | TextDelta | ModelResponded | ToolsAnswered

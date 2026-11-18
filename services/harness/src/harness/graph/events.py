"""The events the turn graph's nodes emit beside the loop's own
(`harness.turn.events`), which the Studio translator turns into wire frames.

`orchestrate` emits its status line (`AgentStatus`) and the events of the
agent's run as they happen (`orchestrator.run`): answer text as it streams,
each whole model response, and each response's tool answers. `planner`
emits `PlannerStarted` as each entry starts, then each model response
(`PlannerResponse`) and each response's tool answers (`PlannerAnswers`) of
its run as they happen (`planner.pydantic_agent`). `execute` emits each
progress report of a plan step as the executor sends it (`StepProgress`).
`GraphEvent` is their union.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from orchestrator.run import ModelResponded, TextDelta, ToolsAnswered

if TYPE_CHECKING:
    # `planner.pydantic_agent` reaches this module through `planner.call` and
    # `harness.graph.orchestrate`, so its names are read for typing only.
    from planner.pydantic_agent import PlannerAnswers, PlannerResponse


@dataclass(frozen=True)
class PlannerStarted:
    """A `planner` entry starting: a revision of a plan on record
    (`revising`), and whether a pick on the plan card opened it (`pick`)."""

    revising: bool
    pick: bool = False


@dataclass(frozen=True)
class AgentStatus:
    """The status line Studio shows as an `orchestrate` entry starts:
    `text`, and `failure`, the refusal a plan failed with, when the
    hand-back it reads carries one."""

    text: str
    failure: dict[str, Any] | None = None


@dataclass(frozen=True)
class StepProgress:
    """A progress report of the approved plan's step `step_id`, which calls
    `tool`: the report's `message`, if it has one, and the step's row on the
    plan card the user approved, its 1-based `position` among `count` rows,
    the row's `description` and whether its result answers the question
    (`gives_answer`). The four row fields are `None` for a step the card
    does not show."""

    step_id: str
    tool: str
    message: str | None
    position: int | None
    count: int | None
    description: str | None
    gives_answer: bool | None


type GraphEvent = (
    AgentStatus
    | TextDelta
    | ModelResponded
    | ToolsAnswered
    | PlannerStarted
    | PlannerResponse
    | PlannerAnswers
    | StepProgress
)

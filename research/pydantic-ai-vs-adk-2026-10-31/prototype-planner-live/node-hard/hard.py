"""Shared set-up for the hard cases: Studio turns on either arm with a `Chunked` client and
request bounds of the test's choosing, and what a turn leaves behind."""

from __future__ import annotations

import functools
import inspect
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import hardened
import pai_planner
import planner.agent as adk_agent
import pytest
import studio_arms
from chunked import Call, Chunked
from google.adk.sessions import InMemorySessionService
from harness.graph.state import AGENT_NAME as GRAPH_NAME
from harness.settings import Settings
from scenes import INTERRUPT, TURN, frame, interrupt_body
from tests.orchestrator_fakes import CONVERSATION, stream, studio_client, turn_body


@contextmanager
def bounds(monkeypatch: pytest.MonkeyPatch, **settings: Any) -> Iterator[None]:
    """Studio apps built through `studio_arms.build_arm` take a programme of `Call`s in
    place of the script and the request bounds in `settings` (`Settings` field names)."""
    with monkeypatch.context() as scope:
        scope.setattr(
            studio_arms,
            "sc",
            SimpleNamespace(scripted=lambda programme: Chunked(programme=programme)),
        )
        scope.setattr(
            studio_arms,
            "orchestrator_app",
            functools.partial(studio_arms.orchestrator_app, settings=Settings(**settings)),
        )
        yield


def session_of(built: studio_arms.Built) -> Any:
    (session,) = [
        s
        for by_user in built.sessions.sessions[GRAPH_NAME].values()
        for s in by_user.values()
        if s.id.startswith(CONVERSATION + "#")
    ]
    return session


@contextmanager
def hard_arm(monkeypatch: pytest.MonkeyPatch, **fixes: Any) -> Iterator[None]:
    """The `pai` arm built through `hardened.hard_planner`, with `fixes` as its switches."""
    with monkeypatch.context() as scope:
        scope.setattr(studio_arms, "pai_planner", functools.partial(hardened.hard_planner, **fixes))
        yield


def trace_tools(monkeypatch: pytest.MonkeyPatch, log: list[str]) -> None:
    """Log each planner tool body as `name:start` and `name:end`, with the state keys it
    wrote between the two in the order it wrote them, on whichever arm calls it
    (`planner.agent` for ADK, `pai_planner` for Pydantic AI). A write is a key that is new
    or whose value changed; two writes to one key in a body count as one."""

    def snapshot(call: Any) -> dict[str, Any]:
        state = call.state
        return dict(state.to_dict() if hasattr(state, "to_dict") else state)

    def traced(name: str, original: Any) -> Any:
        def wrote(before: dict[str, Any], call: Any) -> list[str]:
            after = snapshot(call)
            return [k for k, v in after.items() if k not in before or before[k] != v]

        if inspect.iscoroutinefunction(original):

            async def awaited(call: Any, *args: Any, **kwargs: Any) -> Any:
                before = snapshot(call)
                log.append(f"{name}:start")
                try:
                    return await original(call, *args, **kwargs)
                finally:
                    log.append(f"{name}:end wrote {wrote(before, call)}")

            return awaited

        def plain(call: Any, *args: Any, **kwargs: Any) -> Any:
            before = snapshot(call)
            log.append(f"{name}:start")
            try:
                return original(call, *args, **kwargs)
            finally:
                log.append(f"{name}:end wrote {wrote(before, call)}")

        return plain

    for module in (adk_agent, pai_planner):
        for name in ("ask_user", "propose_and_translate", "reject"):
            monkeypatch.setattr(module, name, traced(name, getattr(module, name)))


async def phases(
    monkeypatch: pytest.MonkeyPatch,
    task: Any,
    plan: Sequence[tuple[str, Sequence[Call]]],
    answers: Sequence[Any],
    *,
    settings: dict[str, Any] | None = None,
    fixes: dict[str, Any] | None = None,
    final: str = "approve",
) -> tuple[list[studio_arms.Built], list[list[dict[str, Any]]]]:
    """One conversation turn run in phases, each on a new app built over the same session
    service, as after a restart or a deploy. Phase `n` runs under the planner `plan[n][0]`,
    whose model answers with `plan[n][1]`. The first posts the turn; each later one answers
    the pause the one before ended on with the next of `answers`. Returns every app and
    every phase's frames."""
    del final
    sessions = InMemorySessionService()
    built: list[studio_arms.Built] = []
    segments: list[list[dict[str, Any]]] = []
    with (
        monkeypatch.context() as scope,
        bounds(monkeypatch, **(settings or {})),
        hard_arm(monkeypatch, **(fixes or {})),
    ):
        scope.setattr(studio_arms, "InMemorySessionService", lambda: sessions)
        for index, (arm, calls) in enumerate(plan):
            app = await studio_arms.build_arm(arm, task, tuple(calls))  # type: ignore[arg-type]
            built.append(app)
            async with studio_client(app.app) as client:
                if index == 0:
                    segments.append(await stream(client, TURN, turn_body(task.question)))
                    continue
                pause = frame(segments[-1], "elicitation.request")
                body = interrupt_body(pause["interrupt_id"], "answer", answers=answers[index - 1])
                segments.append(await stream(client, INTERRUPT, body))
    return built, segments

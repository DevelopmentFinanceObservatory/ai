"""Running a turn of the workflow: sessions, segments and the events a client reads."""

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import aclosing
from pathlib import Path
from typing import Any

from google.adk import Event
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService
from google.adk.workflow.utils._workflow_hitl_utils import create_request_input_response
from google.genai import types

USER = "u1"


def session_service(db: Path) -> DatabaseSessionService:
    return DatabaseSessionService(db_url=f"sqlite+aiosqlite:///{db}")


async def new_session(runner: Runner, session_id: str) -> None:
    await runner.session_service.create_session(
        app_name=runner.app_name, user_id=USER, session_id=session_id
    )


def user_text(text: str) -> types.Content:
    return types.Content(role="user", parts=[types.Part(text=text)])


def answer(interrupt_id: str, response: dict[str, Any]) -> types.Content:
    """The resume message answering `interrupt_id`."""
    return types.Content(role="user", parts=[create_request_input_response(interrupt_id, response)])


def progress_of(event: Event) -> dict[str, int] | None:
    return (event.custom_metadata or {}).get("progress")


def interrupt_of(event: Event) -> dict[str, Any] | None:
    """The `RequestInput` arguments of an interrupt event, or `None`."""
    if not event.long_running_tool_ids:
        return None
    (call,) = event.get_function_calls()
    return dict(call.args or {})


async def events(runner: Runner, session_id: str, message: types.Content) -> AsyncIterator[Event]:
    async with aclosing(
        runner.run_async(user_id=USER, session_id=session_id, new_message=message)
    ) as stream:
        async for event in stream:
            yield event


async def segment(
    runner: Runner,
    session_id: str,
    message: types.Content,
    *,
    on_event: Callable[[Event], Awaitable[None] | None] | None = None,
) -> list[Event]:
    """Every event one `run_async` call yields, in order."""
    seen: list[Event] = []
    async for event in events(runner, session_id, message):
        seen.append(event)
        if on_event is not None:
            maybe = on_event(event)
            if maybe is not None:
                await maybe
    return seen


def progress_list(seen: list[Event]) -> list[tuple[int, int]]:
    return [(p["call"], p["done"]) for e in seen if (p := progress_of(e)) is not None]


def last_interrupt(seen: list[Event]) -> dict[str, Any]:
    return next(i for e in reversed(seen) if (i := interrupt_of(e)) is not None)

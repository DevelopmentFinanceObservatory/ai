"""A wire tap on google-genai that names the agent behind each model call.

`installed` patches `AsyncModels.generate_content` and `generate_content_stream`, which every
agent of a Studio turn calls, on either planner. `collecting` gives the calls made inside its
block. A call's agent comes from the tools it declares and its system instruction (`agent_of`).
The tap passes arguments and answers through and re-raises every error unchanged.
"""

from __future__ import annotations

import contextvars
import functools
import sys
import time
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import contextmanager
from typing import Any

from google.genai import models

Call = dict[str, Any]

_SINK: contextvars.ContextVar[list[Call] | None] = contextvars.ContextVar("tap_sink", default=None)


def _field(config: Any, name: str) -> Any:
    if config is None:
        return None
    return config.get(name) if isinstance(config, dict) else getattr(config, name, None)


def _tool_names(config: Any) -> set[str]:
    names: set[str] = set()
    for tool in _field(config, "tools") or ():
        for declaration in _field(tool, "function_declarations") or ():
            if name := _field(declaration, "name"):
                names.add(name)
    return names


def _instruction(config: Any) -> str:
    system = _field(config, "system_instruction")
    if isinstance(system, str):
        return system
    parts = _field(system, "parts") or ()
    return " ".join(_field(part, "text") or "" for part in parts)


def agent_of(config: Any) -> str:
    """The agent a request belongs to: the planner declares `propose_plan`, the orchestrator
    `search` and `plan`; any other call is named by the head of its instruction."""
    names = _tool_names(config)
    if "propose_plan" in names:
        return "planner"
    if {"search", "plan"} <= names:
        return "orchestrator"
    return "other: " + " ".join(_instruction(config).split())[:50]


def callers() -> list[str]:
    """The module names on the call stack, innermost first, without repeats."""
    names: list[str] = []
    frame = sys._getframe(2)
    while frame is not None:
        name = frame.f_globals.get("__name__", "")
        if name not in names:
            names.append(name)
        frame = frame.f_back
    return names


def running_agent() -> str | None:
    """The name of the ADK agent whose invocation context is on the call stack, or
    `pydantic_ai` for a call made by the Pydantic AI planner."""
    frame = sys._getframe(2)
    while frame is not None:
        if frame.f_globals.get("__name__") == "hardened":
            return "planner (pydantic ai)"
        context = frame.f_locals.get("invocation_context") or frame.f_locals.get("ctx")
        name = getattr(getattr(context, "agent", None), "name", None)
        if isinstance(name, str):
            return name
        frame = frame.f_back
    return None


class _Call:
    def __init__(self, sink: list[Call] | None, kwargs: dict[str, Any]) -> None:
        self.sink = sink
        self.agent = running_agent() or agent_of(kwargs.get("config"))
        self.callers = callers()
        self.cached = bool(_field(kwargs.get("config"), "cached_content"))
        self.model = str(kwargs.get("model"))
        self.started = time.perf_counter()
        self.usage: Any = None
        self.done = False

    def observe(self, response: Any) -> None:
        if (usage := getattr(response, "usage_metadata", None)) is not None:
            self.usage = usage

    def finish(self, error: BaseException | None = None) -> None:
        if self.done or self.sink is None:
            return
        self.done = True
        usage = self.usage
        self.sink.append(
            {
                "agent": self.agent,
                "callers": self.callers[:14],
                "cached_content": self.cached,
                "model": self.model,
                "seconds": round(time.perf_counter() - self.started, 3),
                "prompt_tokens": None if usage is None else usage.prompt_token_count or 0,
                "cached_tokens": None if usage is None else usage.cached_content_token_count or 0,
                "output_tokens": None if usage is None else usage.candidates_token_count or 0,
                "thinking_tokens": None if usage is None else usage.thoughts_token_count or 0,
                "error": None if error is None else f"{type(error).__name__}: {error}"[:300],
            }
        )


@contextmanager
def collecting() -> Iterator[list[Call]]:
    """The calls made inside the block, in the order they finished."""
    calls: list[Call] = []
    token = _SINK.set(calls)
    try:
        yield calls
    finally:
        _SINK.reset(token)


def _wrap_generate(original: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(original)
    async def generate_content(self: Any, *args: Any, **kwargs: Any) -> Any:
        call = _Call(_SINK.get(), kwargs)
        try:
            response = await original(self, *args, **kwargs)
        except BaseException as exc:
            call.finish(exc)
            raise
        call.observe(response)
        call.finish()
        return response

    return generate_content


async def _tapped(stream: AsyncIterator[Any], call: _Call) -> AsyncIterator[Any]:
    try:
        async for chunk in stream:
            call.observe(chunk)
            yield chunk
    except BaseException as exc:
        call.finish(exc)
        raise
    else:
        call.finish()
    finally:
        if aclose := getattr(stream, "aclose", None):
            await aclose()


def _wrap_stream(original: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(original)
    async def generate_content_stream(self: Any, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        call = _Call(_SINK.get(), kwargs)
        try:
            stream = await original(self, *args, **kwargs)
        except BaseException as exc:
            call.finish(exc)
            raise
        return _tapped(stream, call)

    return generate_content_stream


@contextmanager
def installed() -> Iterator[None]:
    """Patch the two methods on `AsyncModels`, and restore them on exit."""
    patches = (("generate_content", _wrap_generate), ("generate_content_stream", _wrap_stream))
    originals = {name: models.AsyncModels.__dict__[name] for name, _ in patches}
    for name, wrap in patches:
        setattr(models.AsyncModels, name, wrap(originals[name]))
    try:
        yield
    finally:
        for name, original in originals.items():
            setattr(models.AsyncModels, name, original)

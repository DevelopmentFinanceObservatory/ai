"""A wire tap on google-genai: one entry per model call, collected per planner entry.

Both entries under comparison (ADK's and Pydantic AI's) call the same google-genai
methods, `AsyncModels.generate_content` and `generate_content_stream`, whose answers carry
the usage the figures need and whose timing no framework exposes. `installed` patches them
at class level; `collecting` gives the calls made inside its block, and the tasks that
block starts, a list of their own. The tap passes arguments and answers through untouched
and re-raises every error unchanged; a bookkeeping failure goes into the call's `tap_error`
and never reaches the caller.
"""

from __future__ import annotations

import contextvars
import functools
import time
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import contextmanager
from typing import Any

from google.genai import models

Call = dict[str, Any]

_SINK: contextvars.ContextVar[list[Call] | None] = contextvars.ContextVar("tap_sink", default=None)


@contextmanager
def collecting() -> Iterator[list[Call]]:
    """The calls made inside the block, in the order they finished."""
    calls: list[Call] = []
    token = _SINK.set(calls)
    try:
        yield calls
    finally:
        _SINK.reset(token)


def _error(exc: BaseException) -> dict[str, Any]:
    return {
        "class": type(exc).__name__,
        "code": getattr(exc, "code", None),
        "status": getattr(exc, "status", None),
        "message": str(exc)[:300],
    }


class _Call:
    """One call in flight."""

    def __init__(self, sink: list[Call] | None, kwargs: dict[str, Any], *, stream: bool) -> None:
        self.sink = sink
        self.stream = stream
        self.started = time.perf_counter()
        self.first_chunk_s: float | None = None
        self.first_chunk_thought: bool | None = None
        self.usage: Any = None
        self.cached_content = bool(_config_field(kwargs.get("config"), "cached_content"))
        self.done = False

    def observe(self, response: Any) -> None:
        if self.first_chunk_s is None:
            self.first_chunk_s = time.perf_counter() - self.started
            self.first_chunk_thought = _opens_with_thought(response)
        if (usage := getattr(response, "usage_metadata", None)) is not None:
            self.usage = usage

    def finish(self, error: BaseException | None = None) -> None:
        if self.done or self.sink is None:
            return
        self.done = True
        usage = self.usage
        record: Call = {
            "stream": self.stream,
            "seconds": round(time.perf_counter() - self.started, 3),
            "first_chunk_s": None if self.first_chunk_s is None else round(self.first_chunk_s, 3),
            "first_chunk_thought": self.first_chunk_thought,
            "cached_content": self.cached_content,
            "prompt_tokens": None if usage is None else usage.prompt_token_count or 0,
            "cached_tokens": None if usage is None else usage.cached_content_token_count or 0,
            "output_tokens": None if usage is None else usage.candidates_token_count or 0,
            "thinking_tokens": None if usage is None else usage.thoughts_token_count or 0,
            "error": None if error is None else _error(error),
        }
        self.sink.append(record)


def _opens_with_thought(response: Any) -> bool | None:
    """Whether the chunk's first part is a thought summary, or `None` when it has no part."""
    for candidate in getattr(response, "candidates", None) or []:
        parts = getattr(getattr(candidate, "content", None), "parts", None) or []
        if parts:
            return bool(getattr(parts[0], "thought", False))
    return None


def _config_field(config: Any, name: str) -> Any:
    if config is None:
        return None
    return config.get(name) if isinstance(config, dict) else getattr(config, name, None)


def _wrap_generate(original: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(original)
    async def generate_content(self: Any, *args: Any, **kwargs: Any) -> Any:
        call = _Call(_SINK.get(), kwargs, stream=False)
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
        call = _Call(_SINK.get(), kwargs, stream=True)
        try:
            stream = await original(self, *args, **kwargs)
        except BaseException as exc:
            call.finish(exc)
            raise
        return _tapped(stream, call)

    return generate_content_stream


@contextmanager
def installed(*, models_cls: type = models.AsyncModels) -> Iterator[None]:
    """Patch the two methods on `models_cls` (google-genai's by default), and restore them
    on exit."""
    patches = (
        ("generate_content", _wrap_generate),
        ("generate_content_stream", _wrap_stream),
    )
    originals = {name: models_cls.__dict__[name] for name, _ in patches}
    if any(getattr(original, "tapped", False) for original in originals.values()):
        raise RuntimeError("the tap is already installed")
    for name, wrap in patches:
        wrapper = wrap(originals[name])
        wrapper.tapped = True  # ty: ignore[unresolved-attribute] -- a marker on a function
        setattr(models_cls, name, wrapper)
    try:
        yield
    finally:
        for name, original in originals.items():
            setattr(models_cls, name, original)

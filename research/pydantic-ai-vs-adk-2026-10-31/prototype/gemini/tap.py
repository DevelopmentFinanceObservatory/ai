"""A wire tap on google-genai: one JSON line per model call and per cache create.

Both frameworks under comparison (ADK and Pydantic AI) call the same three google-genai
methods, `AsyncModels.generate_content`, `AsyncModels.generate_content_stream` and
`AsyncCaches.create`. `installed` patches them at class level, in process, so every call
from either framework passes through and is measured the same way. The driver sets the
labels (`arm`, `task`, `rep`, `entry_id`) around each entry with `labelled`; a call
reads them when it starts, so work the entry spawns inherits them.

The tap is transparent: it passes arguments through untouched, returns what the original
returns (a stream is re-yielded chunk by chunk, closed when the consumer closes it) and
re-raises every error unchanged. A failure inside the tap's own bookkeeping is written
into the record as `tap_errors` and never reaches the caller.
"""

from __future__ import annotations

import base64
import contextvars
import functools
import hashlib
import inspect
import json
import threading
import time
from collections.abc import AsyncIterator, Callable, Iterator, Mapping
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from typing import Any

from google.genai import caches, models
from pydantic import BaseModel

_LABELS: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "tap_labels", default=None
)


@contextmanager
def labelled(**labels: Any) -> Iterator[None]:
    """Label every call made inside the block (and the tasks it starts)."""
    token = _LABELS.set(labels)
    try:
        yield
    finally:
        _LABELS.reset(token)


# --- Normal form -----------------------------------------------------------------


def jsonable(value: Any) -> Any:
    """`value` as plain JSON data, the same whether google-genai built it as a model or
    the caller passed a dict: `None` fields dropped, bytes as urlsafe base64."""
    if isinstance(value, Enum):
        value = value.value
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items() if v is not None}
    if isinstance(value, list | tuple):
        return [jsonable(v) for v in value]
    if isinstance(value, bytes):
        return base64.urlsafe_b64encode(value).decode("ascii")
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return repr(value)


def canonical(value: Any) -> str:
    """Canonical JSON of `value`: sorted keys, no spaces, `jsonable` form."""
    return json.dumps(jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _field(config: Any, name: str) -> Any:
    if config is None:
        return None
    return config.get(name) if isinstance(config, Mapping) else getattr(config, name, None)


def function_call_info(part: Mapping[str, Any]) -> dict[str, Any] | None:
    """A normalised part's function call: name, id, a digest of its arguments, and the
    thought signature it carries (presence and digest), or None for any other part."""
    call = part.get("function_call")
    if not call:
        return None
    signature = part.get("thought_signature")
    return {
        "name": call.get("name"),
        "id": call.get("id"),
        "args_sha": sha(canonical(call.get("args") or {})),
        "has_sig": bool(signature),
        "sig_sha": sha(signature)[:16] if signature else None,
    }


def scan_parts(parts: list[Any]) -> tuple[list[dict[str, Any]], int]:
    """The function calls in normalised `parts`, in order, and how many are thought
    summaries (`thought=True`)."""
    calls, thoughts = [], 0
    for part in parts:
        if not isinstance(part, Mapping):
            continue
        thoughts += bool(part.get("thought"))
        if info := function_call_info(part):
            calls.append(info)
    return calls, thoughts


def scan_contents(contents: list[Any]) -> tuple[list[dict[str, Any]], int]:
    """`scan_parts` over every content of a normalised request, each call tagged with the
    index of the content that carries it."""
    calls, thoughts = [], 0
    for index, content in enumerate(contents):
        if not isinstance(content, Mapping):
            continue
        found, n = scan_parts(content.get("parts") or [])
        calls += [{**info, "content_index": index} for info in found]
        thoughts += n
    return calls, thoughts


# --- The tap -----------------------------------------------------------------------


class Sink:
    """Appends one JSON line per record to a file, from any thread."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._seq = 0

    def next_seq(self) -> int:
        with self._lock:
            self._seq += 1
            return self._seq

    def write(self, record: Mapping[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(line)


def _error(exc: BaseException) -> dict[str, Any]:
    return {
        "class": type(exc).__name__,
        "code": getattr(exc, "code", None),
        "message": str(exc)[:300],
    }


class _Call:
    """One call in flight: what it was sent, and what has come back so far."""

    def __init__(self, sink: Sink, kind: str, bound: Mapping[str, Any], *, stream: bool) -> None:
        self.sink = sink
        self.started = time.perf_counter()
        self.done = False
        self.tap_errors: list[str] = []
        self.record: dict[str, Any] = {
            "kind": kind,
            "seq": sink.next_seq(),
            "t": time.time(),
            **{k: (_LABELS.get() or {}).get(k) for k in ("arm", "task", "rep", "entry_id")},
            "stream": stream,
            "model": bound.get("model"),
        }
        self.first_chunk_s: float | None = None
        self.last_chunk_s: float | None = None
        self.chunks = 0
        self.function_calls: list[dict[str, Any]] = []
        self.thought_parts = 0
        self.usage: dict[str, int | None] | None = None
        self.finish_reason: str | None = None
        try:
            self._describe_request(kind, bound)
        except Exception as exc:
            self.tap_errors.append(f"request: {exc!r}")

    def _describe_request(self, kind: str, bound: Mapping[str, Any]) -> None:
        config = bound.get("config")
        if kind == "cache_create":
            parts = {
                name: _field(config, name)
                for name in ("system_instruction", "tools", "tool_config")
            }
            self.record["ttl"] = _field(config, "ttl")
            for name, value in parts.items():
                text = None if value in (None, [], {}) else canonical(value)
                self.record[name] = text
                self.record[f"{name}_sha"] = None if text is None else sha(text)
            return
        contents = jsonable(bound.get("contents"))
        contents = contents if isinstance(contents, list) else [contents]
        calls, thoughts = scan_contents(contents)
        self.record.update(
            cached_content=_field(config, "cached_content"),
            has_system_instruction=bool(_field(config, "system_instruction")),
            has_tools=bool(_field(config, "tools")),
            has_tool_config=bool(_field(config, "tool_config")),
            thinking_config=jsonable(_field(config, "thinking_config")),
            contents=contents,
            request_function_calls=calls,
            request_thought_parts=thoughts,
        )

    def observe(self, response: Any) -> None:
        """Note one response, or one chunk of a streamed one."""
        now = time.perf_counter() - self.started
        self.chunks += 1
        if self.first_chunk_s is None:
            self.first_chunk_s = now
        self.last_chunk_s = now
        try:
            self._read(response)
        except Exception as exc:
            self.tap_errors.append(f"response: {exc!r}")

    def _read(self, response: Any) -> None:
        if self.record["kind"] == "cache_create":
            self.record["name"] = getattr(response, "name", None)
            return
        for candidate in getattr(response, "candidates", None) or []:
            if candidate.finish_reason is not None:
                self.finish_reason = str(
                    getattr(candidate.finish_reason, "name", candidate.finish_reason)
                )
            content = candidate.content
            if content is None:
                continue
            calls, thoughts = scan_parts(jsonable(content).get("parts") or [])
            self.function_calls += calls
            self.thought_parts += thoughts
            break
        if (usage := getattr(response, "usage_metadata", None)) is not None:
            self.usage = {
                "prompt": usage.prompt_token_count,
                "cached": usage.cached_content_token_count,
                "candidates": usage.candidates_token_count,
                "thoughts": usage.thoughts_token_count,
            }

    def finish(self, error: BaseException | None = None) -> None:
        if self.done:
            return
        self.done = True
        try:
            elapsed = time.perf_counter() - self.started
            self.record["seconds"] = elapsed
            self.record["error"] = None if error is None else _error(error)
            if self.record["kind"] == "generate":
                self.record.update(
                    chunks=self.chunks,
                    first_chunk_s=self.first_chunk_s if self.record["stream"] else None,
                    last_chunk_s=self.last_chunk_s if self.record["stream"] else None,
                    response_function_calls=self.function_calls,
                    response_thought_parts=self.thought_parts,
                    usage=self.usage,
                    finish_reason=self.finish_reason,
                )
            else:
                self.record.setdefault("name", None)
            self.record["tap_errors"] = self.tap_errors
            self.sink.write(self.record)
        except Exception as exc:
            self.sink.write(
                {**self.record, "kind": "tap_failure", "tap_errors": [*self.tap_errors, repr(exc)]}
            )


def _bind(
    original: Callable[..., Any], owner: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> dict[str, Any]:
    try:
        bound = inspect.signature(original).bind(owner, *args, **kwargs)
    except TypeError:
        return dict(kwargs)
    bound.arguments.pop(next(iter(bound.arguments)), None)
    return dict(bound.arguments)


def _wrap_generate(original: Callable[..., Any], sink: Sink) -> Callable[..., Any]:
    @functools.wraps(original)
    async def generate_content(self: Any, *args: Any, **kwargs: Any) -> Any:
        call = _Call(sink, "generate", _bind(original, self, args, kwargs), stream=False)
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


def _wrap_stream(original: Callable[..., Any], sink: Sink) -> Callable[..., Any]:
    @functools.wraps(original)
    async def generate_content_stream(self: Any, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        call = _Call(sink, "generate", _bind(original, self, args, kwargs), stream=True)
        try:
            stream = await original(self, *args, **kwargs)
        except BaseException as exc:
            call.finish(exc)
            raise
        return _tapped(stream, call)

    return generate_content_stream


def _wrap_cache_create(original: Callable[..., Any], sink: Sink) -> Callable[..., Any]:
    @functools.wraps(original)
    async def create(self: Any, *args: Any, **kwargs: Any) -> Any:
        call = _Call(sink, "cache_create", _bind(original, self, args, kwargs), stream=False)
        try:
            response = await original(self, *args, **kwargs)
        except BaseException as exc:
            call.finish(exc)
            raise
        call.observe(response)
        call.finish()
        return response

    return create


@contextmanager
def installed(
    path: Path,
    *,
    models_cls: type = models.AsyncModels,
    caches_cls: type = caches.AsyncCaches,
) -> Iterator[Sink]:
    """Patch the three methods on `models_cls` and `caches_cls` (google-genai's by
    default) to write to `path`, and restore them on exit."""
    sink = Sink(path)
    patches = (
        (models_cls, "generate_content", _wrap_generate),
        (models_cls, "generate_content_stream", _wrap_stream),
        (caches_cls, "create", _wrap_cache_create),
    )
    originals = [(cls, name, cls.__dict__[name]) for cls, name, _ in patches]
    if any(getattr(original, "tapped", False) for _, _, original in originals):
        raise RuntimeError("the tap is already installed")
    for (cls, name, wrap), (_, _, original) in zip(patches, originals, strict=True):
        wrapper = wrap(original, sink)
        wrapper.tapped = True  # ty: ignore[unresolved-attribute] -- a marker on a function
        setattr(cls, name, wrapper)
    try:
        yield sink
    finally:
        for cls, name, original in originals:
            setattr(cls, name, original)

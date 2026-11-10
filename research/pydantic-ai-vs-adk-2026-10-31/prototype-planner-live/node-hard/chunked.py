"""A scripted google-genai client whose calls can stall, be cut and be refused.

`scripted.Scripted` answers every call with one finished chunk. `Chunked` answers each
call from a `Call`: chunks with waits between them, a wait that never ends, or the
refusal google-genai raises when a request is opened against a cache that is gone.
Both arms talk to it at `client.aio.models.generate_content_stream`.
"""

from __future__ import annotations

import asyncio
import copy
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import scripted as sc
import validate
from google.genai import errors, types

NEVER = 3600.0


def function_call(name: str, args: dict[str, Any], *, signed: bool = True) -> types.Part:
    return types.Part(
        function_call=types.FunctionCall(name=name, args=args),
        thought_signature=sc.SIGNATURE if signed else None,
    )


@dataclass
class Call:
    """What one model call does.

    `wait_s` is how long it takes to answer the open; `refuse` is a status code answered
    in place of a stream (google-genai raises it from `generate_content_stream`);
    `chunks` are the chunks, each a list of parts; `gap_s` is the wait before each chunk
    after the first; `hang_after` waits `NEVER` seconds after the last chunk, a stream
    that goes quiet.
    """

    chunks: list[list[types.Part]] = field(default_factory=list)
    wait_s: float = 0.0
    gap_s: float = 0.0
    refuse: int | None = None
    hang_after: bool = False
    first_chunk_wait_s: float = 0.0
    finish: types.FinishReason = types.FinishReason.STOP
    bare: bool = False


def say(*parts: types.Part, **kwargs: Any) -> Call:
    """A call answering with `parts` as one chunk."""
    return Call(chunks=[list(parts)], **kwargs)


@dataclass
class Chunked(sc.Scripted):
    """A client that answers call `n` from `programme[n]`, past its end with `done`."""

    programme: Sequence[Call] = ()
    opened: list[float] = field(default_factory=list)

    async def _stream(self, *, model: str, contents: Any, config: Any) -> Any:
        index = len(self.calls)
        self.calls.append(
            {"model": model, "contents": copy.deepcopy(contents), "config": copy.deepcopy(config)}
        )
        step = (
            self.programme[index] if index < len(self.programme) else say(types.Part(text="done"))
        )
        if step.wait_s:
            await asyncio.sleep(step.wait_s)
        if step.refuse is not None:
            raise errors.ClientError(
                step.refuse,
                {"error": {"code": step.refuse, "message": "CachedContent not found"}},
            )

        async def stream() -> Any:
            if step.first_chunk_wait_s:
                await asyncio.sleep(step.first_chunk_wait_s)
            for position, parts in enumerate(step.chunks):
                if position and step.gap_s:
                    await asyncio.sleep(step.gap_s)
                yield types.GenerateContentResponse(
                    candidates=[
                        types.Candidate(
                            content=(
                                None
                                if step.bare
                                else types.Content(role="model", parts=copy.deepcopy(parts))
                            ),
                            finish_reason=(
                                step.finish
                                if position == len(step.chunks) - 1 and not step.hang_after
                                else None
                            ),
                        )
                    ],
                    usage_metadata=validate.SCRIPTED_USAGE,
                )
            if step.hang_after:
                await asyncio.sleep(NEVER)

        return stream()


def chunked(*programme: Call) -> Chunked:
    return Chunked(programme=programme)

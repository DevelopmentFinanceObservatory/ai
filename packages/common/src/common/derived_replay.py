"""Computing a derived (`ops_`) frame again from its stored recipe.

dfo-apis' analysis runner (`POST /api/datasets/analysis`) replays the recipe of a
frame it no longer holds when the request names that frame's pair with
`latest: true`: it reads the recipe's roots at their datasets' current versions
and stores the frame it computes under a new pair. analysis-mcp's verbs and
presentation-mcp's describe_frame and write_component replay a lost input through
`replay_derived`, each over its own HTTP client.

An answer is one of:

- the pair the runner computed, from a 200 naming one frame;
- `ReplayPending`, a 202: a root dataset is still building;
- `ReplayBusy`, a retryable `service_unavailable` problem document;
- `ReplayFailed`, any other answer, or a 200 that names no single frame, whose
  `code` is then `RESPONSE_INVALID`.

`ReplayPending` and `ReplayBusy` carry the delay the runner named, and a
`ReplayFailed` is logged with the replayed pair.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

import httpx2
from pydantic import ValidationError

from common.materialisation.frame_ref import FrameRef

ANALYSIS_PATH = "/api/datasets/analysis"

# The wait a retryable answer names when dfo-apis sent none.
DEFAULT_RETRY_AFTER_S = 5.0

# The code of a 200 that names no single frame.
RESPONSE_INVALID = "response_invalid"

_BUSY_CODE = "service_unavailable"

# A client's POST of a JSON body to a dfo-apis path. It returns the response for any
# answered call, or the client's own refusal for a call that got none.
type Post[U] = Callable[[str, Mapping[str, Any]], Awaitable[httpx2.Response | U]]


@dataclass(frozen=True, kw_only=True)
class ReplayPending:
    """The runner answered 202 because a root dataset is still building. `operations`
    are the rebuilds it waits on."""

    retry_after_s: float
    operations: tuple[str, ...]


@dataclass(frozen=True, kw_only=True)
class ReplayBusy:
    """The runner is busy and named a wait. `detail` is its own text."""

    retry_after_s: float
    detail: str


@dataclass(frozen=True, kw_only=True)
class ReplayFailed:
    """Any other answer. `code` is the problem document's code, `None` without
    one, and `RESPONSE_INVALID` for a 200 that names no single frame."""

    response: httpx2.Response
    code: str | None
    detail: str


def problem_json(response: httpx2.Response) -> Mapping[str, Any] | None:
    """`response`'s JSON object body, `None` when it has none."""
    try:
        body = response.json()
    except ValueError:
        return None
    return body if isinstance(body, Mapping) else None


def retry_after_s(response: httpx2.Response) -> float:
    """The wait `response`'s `Retry-After` header names, `DEFAULT_RETRY_AFTER_S`
    without a readable one."""
    try:
        return max(float(response.headers.get("retry-after", "")), 0.0)
    except ValueError:
        return DEFAULT_RETRY_AFTER_S


def pending(response: httpx2.Response) -> ReplayPending:
    """The wait and the rebuilds a runner's 202 names."""
    body = problem_json(response) or {}
    wait = body.get("retry_after_s")
    operations = body.get("operations")
    return ReplayPending(
        retry_after_s=float(wait) if isinstance(wait, int | float) else retry_after_s(response),
        operations=tuple(str(op) for op in operations) if isinstance(operations, list) else (),
    )


def busy(response: httpx2.Response) -> ReplayBusy | None:
    """The runner's busy refusal, `None` when `response` is another answer."""
    problem = problem_json(response)
    if problem is None or problem.get("code") != _BUSY_CODE or problem.get("retryable") is not True:
        return None
    detail = str(problem.get("detail") or problem.get("title") or _BUSY_CODE)
    return ReplayBusy(retry_after_s=retry_after_s(response), detail=detail)


def _replayed(response: httpx2.Response) -> FrameRef | ReplayPending | ReplayBusy | ReplayFailed:
    if response.status_code == httpx2.codes.ACCEPTED:
        return pending(response)
    if response.status_code != httpx2.codes.OK:
        if (refused := busy(response)) is not None:
            return refused
        problem = problem_json(response) or {}
        code = problem.get("code")
        return ReplayFailed(
            response=response,
            code=code if isinstance(code, str) else None,
            detail=str(problem.get("detail") or problem.get("title") or response.status_code),
        )
    body = problem_json(response) or {}
    ref_body = body.get("ref")
    members = ref_body.get("members") if isinstance(ref_body, Mapping) else None
    member = members[0] if isinstance(members, list) and len(members) == 1 else None
    spec = member.get("spec") if isinstance(member, Mapping) else None
    try:
        return FrameRef.model_validate(
            {
                "spec_id": spec.get("spec_id") if isinstance(spec, Mapping) else None,
                "content_version": member.get("content_version")
                if isinstance(member, Mapping)
                else None,
            }
        )
    except ValidationError:
        return ReplayFailed(
            response=response,
            code=RESPONSE_INVALID,
            detail="the replayed analysis answer names no single frame",
        )


async def replay_derived[U](
    ref: FrameRef, post: Post[U], *, logger: logging.Logger
) -> FrameRef | ReplayPending | ReplayBusy | ReplayFailed | U:
    """The pair of the derived frame `ref` computed again from its stored recipe on
    its datasets' current versions, or why the runner computed none. `post` is the
    caller's client, whose own refusal for an unanswered call passes back as it is.
    A `ReplayFailed` and an unanswered call are logged on `logger`, the caller's
    own, with `ref`."""
    answer = await post(ANALYSIS_PATH, {"tree": ref.model_dump(), "latest": True})
    replayed = _replayed(answer) if isinstance(answer, httpx2.Response) else answer
    if not isinstance(replayed, FrameRef | ReplayPending | ReplayBusy):
        logger.warning(
            "replay of derived frame %s, %s refused: status %s, %s",
            ref.spec_id,
            ref.content_version,
            answer.status_code if isinstance(answer, httpx2.Response) else None,
            replayed.code if isinstance(replayed, ReplayFailed) else None,
        )
    return replayed

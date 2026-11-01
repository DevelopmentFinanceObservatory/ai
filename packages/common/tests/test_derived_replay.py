"""`common.derived_replay`: the replay request and what each runner answer maps to."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import httpx2
import pytest
from common.derived_replay import (
    ANALYSIS_PATH,
    DEFAULT_RETRY_AFTER_S,
    RESPONSE_INVALID,
    ReplayBusy,
    ReplayFailed,
    ReplayPending,
    replay_derived,
)
from common.materialisation import FrameRef

_LOST = FrameRef(spec_id="ops_" + "1" * 32, content_version="ocv_" + "1" * 32)
_REPLAYED = {"spec_id": _LOST.spec_id, "content_version": "ocv_" + "2" * 32}
_LOGGER = logging.getLogger("tests.derived_replay")


def _ready(member: Mapping[str, Any]) -> httpx2.Response:
    return httpx2.Response(200, json={"ref": {"members": [member]}})


def _problem(status: int, code: str, **extra: Any) -> httpx2.Response:
    return httpx2.Response(status, json={"code": code, "detail": "because", **extra})


@pytest.mark.parametrize(
    ("answer", "outcome"),
    [
        pytest.param(
            _ready(
                {
                    "spec": {"spec_id": _REPLAYED["spec_id"]},
                    "content_version": _REPLAYED["content_version"],
                }
            ),
            FrameRef(**_REPLAYED),
            id="ready",
        ),
        pytest.param(
            httpx2.Response(202, json={"retry_after_s": 4, "operations": ["op_1"]}),
            ReplayPending(retry_after_s=4.0, operations=("op_1",)),
            id="pending",
        ),
        pytest.param(
            httpx2.Response(202, headers={"retry-after": "7"}),
            ReplayPending(retry_after_s=7.0, operations=()),
            id="pending with the header's wait",
        ),
        pytest.param(
            _problem(503, "service_unavailable", retryable=True),
            ReplayBusy(retry_after_s=DEFAULT_RETRY_AFTER_S, detail="because"),
            id="busy",
        ),
    ],
)
async def test_a_replay_posts_the_pair_with_latest_and_maps_the_answer(
    answer: httpx2.Response, outcome: object, caplog: pytest.LogCaptureFixture
) -> None:
    posted: list[tuple[str, Mapping[str, Any]]] = []

    async def post(path: str, body: Mapping[str, Any]) -> httpx2.Response:
        posted.append((path, body))
        return answer

    with caplog.at_level(logging.WARNING, logger=_LOGGER.name):
        assert await replay_derived(_LOST, post, logger=_LOGGER) == outcome

    assert posted == [(ANALYSIS_PATH, {"tree": _LOST.model_dump(), "latest": True})]
    assert caplog.records == []


@pytest.mark.parametrize(
    ("answer", "code"),
    [
        pytest.param(_problem(500, "internal_error"), "internal_error", id="refused"),
        pytest.param(
            _problem(503, "service_unavailable"), "service_unavailable", id="not retryable"
        ),
        pytest.param(_ready({"spec": {}}), RESPONSE_INVALID, id="no single frame"),
        pytest.param("unanswered", None, id="unanswered"),
    ],
)
async def test_a_replay_that_computes_nothing_is_logged_with_the_pair(
    answer: httpx2.Response | str, code: str | None, caplog: pytest.LogCaptureFixture
) -> None:
    async def post(path: str, body: Mapping[str, Any]) -> httpx2.Response | str:
        del path, body
        return answer

    with caplog.at_level(logging.WARNING, logger=_LOGGER.name):
        outcome = await replay_derived(_LOST, post, logger=_LOGGER)

    if isinstance(answer, str):
        assert outcome == answer
    else:
        assert isinstance(outcome, ReplayFailed)
        assert (outcome.response, outcome.code) == (answer, code)
    (record,) = caplog.records
    assert _LOST.spec_id in record.getMessage()
    assert _LOST.content_version in record.getMessage()

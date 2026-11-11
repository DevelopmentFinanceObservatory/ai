"""Tests for `harness.wiring`: which implementation stands behind the planner seam.

`build_planner` is a thin assembly function; its own behaviour worth pinning
is what it does with `surface` -- pass it straight to `PlannerSeam` so the
planner's translator compiles against the one set of schemas
`harness.server.create_app` already built for the `execute` node, rather
than fetching its own copy through a separate `McpSurface`.
"""

from __future__ import annotations

from typing import Any, override

import pytest
from compiler.surface import FixtureSurface
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.genai import Client, types
from harness import planner_seam
from harness.prefix_cache import PrefixCache
from harness.settings import Settings
from harness.wiring import build_planner

from .scripted_genai import ScriptedGenai, reply, text_part
from .test_planner_seam import _SeamHarness


def test_threads_the_given_surface_into_the_planner_seam(monkeypatch: Any) -> None:
    """The stub stands in for `PlannerSeam`, so no planner is built."""
    captured: dict[str, Any] = {}

    class _RecordingSeam:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setattr("harness.wiring.PlannerSeam", _RecordingSeam)
    surface = FixtureSurface(schemas=[])

    build_planner(settings=Settings(), surface=surface)

    assert captured["surface"] is surface


class _LookupSpy(PrefixCache):
    """A prefix cache that records each lookup and finds no cache, so the call
    goes out uncached."""

    def __init__(self) -> None:
        super().__init__()
        self.lookups: list[tuple[Client, str, types.GenerateContentConfig]] = []

    @override
    async def lookup(
        self, *, client: Client, model: str, config: types.GenerateContentConfig
    ) -> tuple[str, str] | None:
        self.lookups.append((client, model, config))
        return None


async def test_the_planner_looks_up_each_request_in_the_prefix_cache_it_is_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The served planner sends each model call through the process's one prefix cache."""
    scripted = ScriptedGenai([reply(text_part("No plan."))])
    # The seam builds its google-genai client from the environment.
    monkeypatch.setattr(planner_seam, "Client", lambda: scripted.client)
    prefix_cache = _LookupSpy()

    planner = build_planner(
        settings=Settings(), surface=FixtureSurface(schemas=[]), prefix_cache=prefix_cache
    )
    await _SeamHarness().run(
        planner,
        question="kenya debt",
        readings=[],
        run_config=RunConfig(streaming_mode=StreamingMode.SSE),
    )

    (request,) = scripted.requests
    assert prefix_cache.lookups == [(scripted.client, request.model, request.config)]

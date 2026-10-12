"""Tests for `harness.wiring`: which implementation stands behind the planner seam.

`build_planner` is a thin assembly function; its own behaviour worth pinning
is what it does with `surface` -- pass it straight to `PlannerSeam` so the
planner's translator compiles against the one set of schemas
`harness.server.create_app` already built for the `execute` node, rather
than fetching its own copy through a separate `McpSurface`.
"""

from __future__ import annotations

from typing import Any

from compiler.surface import FixtureSurface
from harness.prefix_cache import PrefixCache, PrefixCachedGemini
from harness.settings import Settings
from harness.wiring import build_planner


def test_threads_the_given_surface_into_the_planner_seam(monkeypatch: Any) -> None:
    """The stub stands in for `PlannerSeam`, so no real ADK agent is built."""
    captured: dict[str, Any] = {}

    class _RecordingSeam:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setattr("harness.wiring.PlannerSeam", _RecordingSeam)
    surface = FixtureSurface(schemas=[])

    build_planner(settings=Settings(), surface=surface)

    assert captured["surface"] is surface


def test_the_planner_model_reads_the_prefix_cache_it_is_given() -> None:
    """The served planner's model is built on the process's one prefix cache."""
    prefix_cache = PrefixCache()

    planner = build_planner(
        settings=Settings(), surface=FixtureSurface(schemas=[]), prefix_cache=prefix_cache
    )

    model = planner._agent.model
    assert isinstance(model, PrefixCachedGemini)
    assert model.prefix_cache is prefix_cache

"""Tests for `harness.wiring`: which implementation stands behind each generative seam.

`build_generative_nodes` is a thin assembly function; its own behaviour worth
pinning is what it does with `surface` -- pass it straight to `PlannerSeam`
so the planner's translator compiles against the one set of schemas
`harness.server.create_app` already built for the `execute` node, rather
than fetching its own copy through a separate `McpSurface`. These stub
`PlannerSeam` out and check what it was constructed with, rather than
building a real ADK agent.
"""

from __future__ import annotations

from typing import Any

from compiler.surface import FixtureSurface
from harness.prefix_cache import PrefixCache, PrefixCachedGemini
from harness.settings import Settings
from harness.wiring import build_generative_nodes


def test_threads_the_given_surface_into_the_planner_seam_and_converse(monkeypatch: Any) -> None:
    """Converse reads dfo-faq through the same surface the planner reads skills through."""
    captured: dict[str, Any] = {}

    class _RecordingSeam:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setattr("harness.wiring.PlannerSeam", _RecordingSeam)
    surface = FixtureSurface(schemas=[])

    nodes = build_generative_nodes(
        settings=Settings(),
        model_name="gemini-test",
        surface=surface,
    )

    assert captured["surface"] is surface
    assert nodes.converse._surface is surface


def test_defaults_the_surface_to_none_so_the_seam_builds_its_own(monkeypatch: Any) -> None:
    """Callers with no surface to share -- the scripted-mode wiring, tests -- get the
    old behaviour: `PlannerSeam` builds its own from `settings` when `surface` is `None`.
    """
    captured: dict[str, Any] = {}

    class _RecordingSeam:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setattr("harness.wiring.PlannerSeam", _RecordingSeam)

    build_generative_nodes(settings=Settings(), model_name="gemini-test")

    assert captured["surface"] is None


def test_the_planner_model_reads_the_prefix_cache_it_is_given() -> None:
    """The served planner's model is built on the process's one prefix cache."""
    prefix_cache = PrefixCache()

    nodes = build_generative_nodes(
        settings=Settings(),
        model_name="gemini-test",
        surface=FixtureSurface(schemas=[]),
        prefix_cache=prefix_cache,
    )

    model = nodes.planner._agent.model
    assert isinstance(model, PrefixCachedGemini)
    assert model.prefix_cache is prefix_cache

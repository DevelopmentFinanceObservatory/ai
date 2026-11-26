"""The model surface Studio's agents run with (`harness.surface`).

The surface each model maps to and the startup refusals, the minimal tool
descriptions, the tool texts the `hydrogen` instruction set words its own way,
and the surface fingerprint on each model call's span. The Flash surface is
pinned byte for byte by the recorded requests in
`fixtures/orchestrator_requests` (`orchestrator/test_declarations.py`).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import os
import subprocess
import sys
from typing import Any

import pytest
from google.genai import types
from harness import orchestrator_tools
from harness import prefix_cache as prefix_cache_module
from harness import server as server_module
from harness import surface as surface_module
from harness.orchestrator_tools import TOOL_NAMES, build_orchestrator_tools
from harness.server import create_app
from harness.settings import Settings
from harness.surface import (
    FLASH_SURFACE,
    HYDROGEN_SURFACE,
    ModelNameRefusedError,
    Surface,
    SurfaceRefusedError,
    resolve_surface,
    set_studio_surface,
    studio_surface,
    surface_prefix_cache,
)
from opentelemetry import trace
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from orchestrator import declarations, minimal_surface
from orchestrator.agent import orchestrator_agent
from orchestrator.declarations import described_paths, tool_declaration
from orchestrator.model import orchestrator_model
from orchestrator.run import Opening, run_orchestrator

from .orchestrator.pai_runs import DIGESTS, TODAY
from .scripted_genai import Request, ScriptedGenai, reply, text_part
from .test_planner_entry_eval import run as planner_entry_run

_FLASH = "gemini-3.8-flash"
_HYDROGEN = "gemini-4-hydrogen"
# Startup refuses before it opens the database.
_NO_DATABASE = "postgresql+asyncpg://refused@127.0.0.1:1/never"


async def _unused(**kwargs: Any) -> Any:
    raise AssertionError("no call expected")


def _tools() -> dict[str, Any]:
    tools = build_orchestrator_tools(resolver=_unused, executor=_unused)
    return {tool.__name__: tool for tool in tools}


async def _orchestrator_request(surface: Surface) -> Request:
    """The first request the orchestrator sends under `surface`."""
    scripted = ScriptedGenai([reply(text_part("Hello."))])
    tools = build_orchestrator_tools(resolver=_unused, executor=_unused)
    orchestrator = orchestrator_agent(tools, surface=surface)
    async for _ in run_orchestrator(
        orchestrator,
        history=None,
        state={"surface": "studio", **DIGESTS},
        turn_id="turn-1",
        model=orchestrator_model("gemini-test", client=scripted.client, budget=None),
        opening=Opening(question="Hello", today=TODAY, **DIGESTS),
    ):
        pass
    return scripted.requests[0]


# --- the surface each model runs -------------------------------------------------


@pytest.mark.parametrize(
    ("model", "planner_model", "surface"),
    [
        (_FLASH, _FLASH, FLASH_SURFACE),
        (_HYDROGEN, _HYDROGEN, HYDROGEN_SURFACE),
        (f"models/{_HYDROGEN}", f"Gemini:{_HYDROGEN}", HYDROGEN_SURFACE),
    ],
)
def test_each_model_runs_the_surface_it_was_evaluated_with(model, planner_model, surface) -> None:
    assert resolve_surface(model_name=model, planner_model_name=planner_model) == surface


@pytest.mark.parametrize(
    ("model", "planner_model"),
    [("gemini-4-flash", _HYDROGEN), (_FLASH, "gemini-3.8-flash-lite")],
)
def test_a_model_with_no_surface_is_refused(model, planner_model) -> None:
    with pytest.raises(SurfaceRefusedError, match="has no surface"):
        resolve_surface(model_name=model, planner_model_name=planner_model)


@pytest.mark.parametrize(("model", "planner_model"), [(_HYDROGEN, _FLASH), (_FLASH, _HYDROGEN)])
def test_models_on_different_surfaces_are_refused(model, planner_model) -> None:
    with pytest.raises(SurfaceRefusedError, match="different surfaces"):
        resolve_surface(model_name=model, planner_model_name=planner_model)


def test_a_planner_model_name_the_developer_api_cannot_serve_is_refused() -> None:
    with pytest.raises(ModelNameRefusedError, match="PLANNER_MODEL_NAME"):
        resolve_surface(model_name=_FLASH, planner_model_name="claude-sonnet-5-5")


def test_the_process_surface_is_resolved_from_the_model_settings_once(monkeypatch) -> None:
    monkeypatch.setenv("MODEL_NAME", _HYDROGEN)
    monkeypatch.setenv("PLANNER_MODEL_NAME", _HYDROGEN)
    set_studio_surface(None)

    assert studio_surface() == HYDROGEN_SURFACE
    monkeypatch.setenv("MODEL_NAME", _FLASH)
    monkeypatch.setenv("PLANNER_MODEL_NAME", _FLASH)
    assert studio_surface() == HYDROGEN_SURFACE
    set_studio_surface(None)
    assert studio_surface() == FLASH_SURFACE


_IMPORT_UNDER_AN_UNMAPPED_MODEL = """
import harness.server, orchestrator.agent, orchestrator.run, planner.instruction
from harness.surface import SurfaceRefusedError, studio_surface
try:
    studio_surface()
except SurfaceRefusedError:
    print("refused")
"""


def test_importing_the_studio_reads_no_surface(tmp_path) -> None:
    """The modules that build Studio's agents import under a model the map does
    not name, so test collection never resolves the surface; the first read
    refuses it."""
    env = {**os.environ, "MODEL_NAME": "gemini-unmapped", "PLANNER_MODEL_NAME": "gemini-unmapped"}
    done = subprocess.run(
        [sys.executable, "-c", _IMPORT_UNDER_AN_UNMAPPED_MODEL],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "refused"


@pytest.mark.parametrize(
    ("model", "planner_model", "refusal"),
    [
        ("gemini-4-flash", "gemini-4-flash", SurfaceRefusedError),
        (_HYDROGEN, _FLASH, SurfaceRefusedError),
        (_FLASH, _HYDROGEN, SurfaceRefusedError),
        (_FLASH, "claude-sonnet-5-5", ModelNameRefusedError),
    ],
)
def test_startup_refuses_a_model_with_no_surface_or_a_mixed_pair(
    monkeypatch, model, planner_model, refusal
) -> None:
    monkeypatch.setenv("PLANNER_MODEL_NAME", planner_model)

    with pytest.raises(refusal):
        create_app(settings=Settings(database_url=_NO_DATABASE, model_name=model))


class _CountedPrefixCache:
    built = 0

    def __init__(self) -> None:
        type(self).built += 1


@pytest.fixture
def counted_prefix_cache(monkeypatch) -> type[_CountedPrefixCache]:
    """Counts every `PrefixCache` built, through `surface_prefix_cache` or
    around it: the class is replaced where it is defined, where the surface
    module reads it, and in the server, should the server name it."""
    monkeypatch.setattr(_CountedPrefixCache, "built", 0)
    monkeypatch.setattr(surface_module, "PrefixCache", _CountedPrefixCache)
    monkeypatch.setattr(prefix_cache_module, "PrefixCache", _CountedPrefixCache)
    monkeypatch.setattr(server_module, "PrefixCache", _CountedPrefixCache, raising=False)
    return _CountedPrefixCache


@pytest.mark.parametrize(
    ("surface", "enabled", "built"),
    [(FLASH_SURFACE, True, 1), (FLASH_SURFACE, False, 0), (HYDROGEN_SURFACE, True, 0)],
)
def test_an_explicit_prefix_cache_is_built_only_for_a_model_that_takes_one(
    counted_prefix_cache, surface, enabled, built
) -> None:
    cache = surface_prefix_cache(surface, enabled=enabled)

    assert (cache is not None) == bool(built)
    assert counted_prefix_cache.built == built


@pytest.mark.parametrize(("model", "built"), [(_FLASH, 1), (_HYDROGEN, 0)])
def test_the_server_builds_a_prefix_cache_only_for_a_model_that_takes_one(
    monkeypatch, counted_prefix_cache, pg_database_url, model, built
) -> None:
    monkeypatch.setenv("PLANNER_MODEL_NAME", model)
    asked: list[Surface] = []

    def gated(surface: Surface, *, enabled: bool) -> Any:
        asked.append(surface)
        return surface_prefix_cache(surface, enabled=enabled)

    monkeypatch.setattr(server_module, "surface_prefix_cache", gated)

    create_app(settings=Settings(database_url=pg_database_url, model_name=model))

    surface = resolve_surface(model_name=model, planner_model_name=model)
    assert studio_surface() == surface
    # The server asks the surface's gate, and nothing builds a cache around it.
    assert asked == [surface]
    assert counted_prefix_cache.built == built


@pytest.mark.parametrize(
    ("model", "arms", "built"),
    [
        (_FLASH, [], 1),
        (_HYDROGEN, [], 0),
        # An arm with edits keeps a cache of its own beside the shared one.
        (_FLASH, ["baseline", "edited"], 2),
        (_HYDROGEN, ["baseline", "edited"], 0),
    ],
)
async def test_the_planner_entry_eval_builds_a_prefix_cache_only_for_a_model_that_takes_one(
    monkeypatch, tmp_path, counted_prefix_cache, model, arms, built
) -> None:
    monkeypatch.setenv("MODEL_NAME", model)
    monkeypatch.setenv("PLANNER_MODEL_NAME", model)
    edited = dataclasses.replace(
        planner_entry_run.ARMS["baseline"], name="edited", edits=(lambda request: None,)
    )
    monkeypatch.setitem(planner_entry_run.ARMS, "edited", edited)
    # The eval names `PrefixCache` itself, so a cache it builds around the gate counts too.
    monkeypatch.setattr(planner_entry_run, "PrefixCache", _CountedPrefixCache)
    asked: list[Surface] = []

    def gated(surface: Surface, *, enabled: bool) -> Any:
        asked.append(surface)
        return surface_prefix_cache(surface, enabled=enabled)

    monkeypatch.setattr(planner_entry_run, "surface_prefix_cache", gated)
    args = argparse.Namespace(out=tmp_path, run_timeout=1.0, concurrency=1, passes=1, reps=1)

    assert await planner_entry_run.run_all(args, [], arms) == 0

    surface = resolve_surface(model_name=model, planner_model_name=model)
    assert studio_surface() == surface
    assert asked == [surface]
    assert counted_prefix_cache.built == built


# --- minimal tool descriptions -------------------------------------------------


def _descriptions(node: Any) -> list[str]:
    if isinstance(node, dict):
        found = [node["description"]] if isinstance(node.get("description"), str) else []
        return found + [text for value in node.values() for text in _descriptions(value)]
    if isinstance(node, list):
        return [text for value in node for text in _descriptions(value)]
    return []


def _texts_at(declaration: types.FunctionDeclaration, path: str) -> list[str]:
    """Each description `declaration` carries at the dotted `path`."""
    schema = declaration.parameters_json_schema
    assert isinstance(schema, dict)
    found: list[str] = []

    def walk(node: Any, at: str, seen: set[tuple[str, str]]) -> None:
        if not isinstance(node, dict):
            return
        if at == path and isinstance(node.get("description"), str):
            found.append(node["description"])
        for key, value in node.items():
            if key == "properties" and isinstance(value, dict):
                for child, sub in value.items():
                    walk(sub, f"{at}.{child}" if at else child, seen)
            elif key == "items":
                walk(value, at, seen)
            elif key in ("anyOf", "oneOf", "allOf") and isinstance(value, list):
                for sub in value:
                    walk(sub, at, seen)
            elif key == "$ref" and isinstance(value, str):
                target = value.rsplit("/", 1)[-1]
                if (target, at) not in seen:
                    seen.add((target, at))
                    walk(schema.get("$defs", {}).get(target), at, seen)

    walk(schema, "", set())
    return found


def _without_descriptions(node: Any) -> Any:
    if isinstance(node, dict):
        return {k: _without_descriptions(v) for k, v in node.items() if k != "description"}
    if isinstance(node, list):
        return [_without_descriptions(value) for value in node]
    return node


@pytest.mark.parametrize("variant", ["default", "hydrogen"])
@pytest.mark.parametrize("name", TOOL_NAMES)
def test_minimal_descriptions_replace_every_text_and_keep_the_schema(name, variant) -> None:
    tool = _tools()[name]
    full = tool_declaration(tool)
    minimal = tool_declaration(tool, variant=variant, descriptions="minimal")
    # A kept path keeps the variant's full text.
    kept = {
        text
        for path in minimal_surface.MINIMAL_FULL_PATHS.get(name, ())
        for text in _texts_at(tool_declaration(tool, variant=variant), path)
    }
    texts = set(minimal_surface.MINIMAL_PARAM_DESCRIPTIONS.get(name, {}).values()) | kept

    assert minimal.name == full.name
    assert minimal.description == minimal_surface.MINIMAL_TOOL_DESCRIPTIONS[name]
    assert _without_descriptions(minimal.parameters_json_schema) == _without_descriptions(
        full.parameters_json_schema
    )
    assert set(_descriptions(minimal.parameters_json_schema)) <= texts


@pytest.mark.parametrize("name", TOOL_NAMES)
def test_every_minimal_key_names_a_description_the_declaration_carries(name: str) -> None:
    keys = set(minimal_surface.MINIMAL_PARAM_DESCRIPTIONS.get(name, {}))
    kept = minimal_surface.MINIMAL_FULL_PATHS.get(name, frozenset())

    assert not keys & kept
    keys |= kept

    assert keys == described_paths(tool_declaration(_tools()[name]))


@pytest.mark.parametrize("name", TOOL_NAMES)
def test_minimal_texts_keep_the_token_caps(name: str) -> None:
    # A token is counted as four characters: 30 per tool, 12 per parameter,
    # and 20 for a parameter that states a rule the answer depends on.
    rules = minimal_surface.MINIMAL_RULE_PATHS.get(name, frozenset())
    assert len(minimal_surface.MINIMAL_TOOL_DESCRIPTIONS[name]) <= 30 * 4
    for path, text in minimal_surface.MINIMAL_PARAM_DESCRIPTIONS.get(name, {}).items():
        assert len(text) <= (20 if path in rules else 12) * 4, path


@pytest.mark.parametrize(
    ("name", "path", "rule"),
    [
        ("search", None, "With share_of, variable names the numerator alone."),
        ("search", "parts.operations.phrase", "With share_of, variable names the numerator"),
        ("search", "parts.operations.over", "origin: donors, creditors"),
        ("search", "parts.operations.over", "destination: recipients, debtors"),
        ("search", "parts.operations.over", "else no flow side"),
        ("search", "parts.entities.role", "'creditor', 'debtor'"),
        ("search", "parts.entities.role", "'from' donor, 'to' recipient"),
        ("search", "parts.time", "'between YYYY and the latest'"),
        ("search", "parts.open_roles", "none for a total or a named side"),
        ("plan", "asks.across_years", "false per year"),
    ],
)
def test_minimal_texts_keep_the_rules_the_answer_depends_on(name, path, rule) -> None:
    declaration = tool_declaration(_tools()[name], descriptions="minimal")

    texts = [declaration.description or ""] if path is None else _texts_at(declaration, path)
    assert texts
    assert all(rule in text for text in texts)


@pytest.mark.parametrize("variant", ["default", "hydrogen"])
@pytest.mark.parametrize(
    ("path", "rule"),
    [
        ("charts.type", "has no run of periods to draw as a line"),
        ("charts.title", "with no period or year"),
        ("charts.select_json", "A line_chart draws at most 8 series"),
        ("charts.select_json", "a bar_chart or a table at most 24"),
    ],
)
def test_minimal_make_charts_keeps_the_full_chart_texts(variant, path, rule) -> None:
    tool = _tools()["make_charts"]
    minimal = tool_declaration(tool, variant=variant, descriptions="minimal")

    # The variant's full text: HYDROGEN_SELECT_JSON_DESCRIPTION under hydrogen.
    assert _texts_at(minimal, path) == _texts_at(tool_declaration(tool, variant=variant), path)
    assert all(rule in text for text in _texts_at(minimal, path))


def test_a_text_the_minimal_surface_lacks_keeps_the_full_one_logged_once(
    monkeypatch, caplog
) -> None:
    monkeypatch.setattr(declarations, "_KEPT_FULL", set())
    tool_texts = dict(minimal_surface.MINIMAL_TOOL_DESCRIPTIONS)
    del tool_texts["plan"]
    param_texts = {
        **minimal_surface.MINIMAL_PARAM_DESCRIPTIONS,
        "plan": {
            k: v
            for k, v in minimal_surface.MINIMAL_PARAM_DESCRIPTIONS["plan"].items()
            if k != "asks.text"
        },
    }
    monkeypatch.setattr(minimal_surface, "MINIMAL_TOOL_DESCRIPTIONS", tool_texts)
    monkeypatch.setattr(minimal_surface, "MINIMAL_PARAM_DESCRIPTIONS", param_texts)
    plan = _tools()["plan"]
    full = tool_declaration(plan)

    with caplog.at_level(logging.WARNING, logger=declarations.__name__):
        minimal = tool_declaration(plan, descriptions="minimal")
        tool_declaration(plan, descriptions="minimal")

    assert minimal.description == full.description
    assert full.parameters_json_schema is not None
    assert minimal.parameters_json_schema is not None
    full_text = full.parameters_json_schema["$defs"]["AskedOutput"]["properties"]["text"]
    sent_text = minimal.parameters_json_schema["$defs"]["AskedOutput"]["properties"]["text"]
    assert sent_text["description"] == full_text["description"]
    assert [r.getMessage().split(" for ")[1] for r in caplog.records] == [
        "plan, so the full text stays",
        "plan:asks.text, so the full text stays",
    ]


async def test_minimal_descriptions_are_the_declarations_sent() -> None:
    request = await _orchestrator_request(HYDROGEN_SURFACE)
    (tool,) = request.generate_config.tools or []
    assert isinstance(tool, types.Tool)

    assert [d.description for d in tool.function_declarations or []] == [
        minimal_surface.MINIMAL_TOOL_DESCRIPTIONS[name] for name in TOOL_NAMES
    ]


# --- hydrogen texts, full descriptions ---------------------------------------------


@pytest.mark.parametrize("name", TOOL_NAMES)
def test_hydrogen_changes_only_the_texts_it_names(name: str) -> None:
    tool = _tools()[name]
    default = tool_declaration(tool)
    hydrogen = tool_declaration(tool, variant="hydrogen")
    params = orchestrator_tools.HYDROGEN_PARAM_DESCRIPTIONS.get(name, {})

    assert _without_descriptions(hydrogen.parameters_json_schema) == _without_descriptions(
        default.parameters_json_schema
    )
    assert hydrogen.description == default.description
    changed = set(_descriptions(hydrogen.parameters_json_schema)) - set(
        _descriptions(default.parameters_json_schema)
    )
    assert changed == set(params.values())


# --- fingerprint ---------------------------------------------------------------------


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


@pytest.mark.parametrize("surface", [FLASH_SURFACE, HYDROGEN_SURFACE])
async def test_each_model_call_s_span_names_the_surface_it_was_sent(
    surface, exported_spans: InMemorySpanExporter
) -> None:
    set_studio_surface(surface)
    with trace.get_tracer(__name__).start_as_current_span("chat"):
        request = await _orchestrator_request(surface)

    config = request.generate_config
    sent = [
        json.loads(declaration.model_dump_json(exclude_none=True))
        for tool in config.tools or []
        for declaration in getattr(tool, "function_declarations", None) or []
    ]
    chat = [s for s in exported_spans.get_finished_spans() if s.name.startswith("chat ")]
    attributes = dict(chat[0].attributes or {})
    assert {k: v for k, v in attributes.items() if k.startswith("studio.surface.")} == {
        "studio.surface.prompt_variant": surface.variant,
        "studio.surface.tool_desc": surface.tool_descriptions,
        "studio.surface.instruction_hash": _sha(str(config.system_instruction)),
        "studio.surface.declarations_hash": _sha(json.dumps(sent, ensure_ascii=False)),
    }

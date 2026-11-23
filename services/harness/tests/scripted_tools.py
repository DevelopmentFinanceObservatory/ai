"""The orchestrator's tools over a scripted surface, and the model calls that use them.

`scripted_orchestrator` builds the real orchestrator agent
(`orchestrator.agent.orchestrator_agent`) on the real tools
(`harness.orchestrator_tools.build_orchestrator_tools`), bound to stand-ins:
`CannedSearch` answers `search` from data-mcp envelopes `definite_envelope`
builds, and `no_resolver` and `no_executor` fail the test on any call they
receive. Pass `harness.scripted.ScriptedExecutor()` as the executor for a
turn that fetches or draws.

`search_call`, `fetch_call`, `make_charts_call` with `chart`, `plan_call` and
`ask_user_call` build the function call parts a scripted model reply sends
(`tests.scripted_genai.reply`).
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from compiler.surface import ToolSurface
from google.genai import types
from harness.graph.interfaces import Executor
from harness.orchestrator_tools import Resolver, build_orchestrator_tools
from harness.scripted import SCRIPTED_SPEC
from harness.tool_contracts import ResolverError, resolver_digest
from orchestrator.agent import Orchestrator, orchestrator_agent

from .scripted_genai import call_part


async def no_resolver(**kwargs: Any) -> dict[str, Any]:
    """A resolver for a turn that must not search."""
    raise AssertionError(f"no search expected, got {kwargs}")


async def no_executor(*, tool: str, args: dict[str, Any], on_progress: Any = None) -> Any:
    """An executor for a turn that must not call one."""
    raise AssertionError(f"no tool call expected, got {tool}")


def definite_envelope(
    *, spec_id: str = SCRIPTED_SPEC, sentence: str = "Life expectancy in Kenya"
) -> dict[str, Any]:
    """A data-mcp search envelope with one definite part reading `spec_id`."""
    spec = {
        "spec_id": spec_id,
        "resolved": {
            "rendered_sentence": sentence,
            "confidence": "high",
            "coverage": {
                "kind": "exact",
                "has_data": True,
                "window": {"start_year": 1960, "end_year": 2023},
            },
            "stat_vars": {
                "member_count": 1,
                "member_count_exact": True,
                "served_count": 1,
                "data_date_range": {"start": "1960", "end": "2023"},
            },
            "entities": [{"role": "observationAbout", "label": "Kenya", "ref": "country/KEN"}],
            "facets": {},
        },
        "definition": {
            "schema_version": 1,
            "spec_id": spec_id,
            "identity": {"stat_var": "LifeExpectancy_Person"},
        },
    }
    part = {"status": "definite", "variable_text": sentence, "warnings": [], "spec": spec}
    return {"schema_version": 1, "kind": "resolution", "status": "definite", "parts": [part]}


class CannedSearch:
    """A resolver answering each search with the next envelope, digested as
    `harness.mcp_client.McpResolver` digests it. `calls` records each call's
    arguments. A search past the last envelope raises `ResolverError`."""

    def __init__(self, *envelopes: Mapping[str, Any]) -> None:
        self._envelopes = [dict(envelope) for envelope in envelopes]
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        if not self._envelopes:
            raise ResolverError("no further search result was scripted")
        return resolver_digest(self._envelopes.pop(0))


def scripted_orchestrator(
    *,
    resolver: Resolver = no_resolver,
    executor: Executor = no_executor,
    surface: ToolSurface | None = None,
    definition_surface: ToolSurface | None = None,
    recall_part: Callable[..., Awaitable[dict[str, Any]]] | None = None,
) -> Orchestrator:
    """The orchestrator agent on its real tools, bound to `resolver` (search),
    `executor` (fetch, describe, draw), `surface` (an approved plan's steps),
    `definition_surface` (a definition on record) and `recall_part`."""
    tools = build_orchestrator_tools(
        resolver=resolver,
        executor=executor,
        surface=surface,
        definition_surface=definition_surface,
        recall_part=recall_part,
    )
    return orchestrator_agent(tools)


def search_call(*variables: str, text: str | None = None) -> types.Part:
    """A `search` call with one part per variable; `text` makes it a retry."""
    args: dict[str, Any] = {"parts": [{"variable": variable} for variable in variables]}
    if text is not None:
        args["text"] = text
    return call_part("search", args)


def fetch_call(spec_id: str = SCRIPTED_SPEC, **args: Any) -> types.Part:
    """A `fetch` call; `args` are its `scope`, `stat_vars` or `breakdown`."""
    return call_part("fetch", {"spec_id": spec_id, **args})


def chart(
    frame: str,
    *,
    type_: str = "line_chart",
    title: str = "Life expectancy in Kenya",
    select: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """One `make_charts` chart on `frame`, its `select` sent as `select_json`.
    Left unset, the select names the one series of a scripted frame
    (`harness.scripted.ScriptedExecutor`)."""
    if select is None:
        select = {"stat_var": "dcid:LifeExpectancy_Person"}
    return {"frame": frame, "type": type_, "title": title, "select_json": json.dumps(select)}


def make_charts_call(*charts: Mapping[str, Any]) -> types.Part:
    """A `make_charts` call drawing `charts`, each built with `chart`."""
    return call_part("make_charts", {"charts": list(charts)})


def plan_call(brief: str, *asks: str) -> types.Part:
    """A `plan` call handing `brief` and its `asks` to the planner. Given no
    asks, it lists `brief` as the one output asked for, since the `plan` tool
    refuses a call that lists none."""
    return call_part("plan", {"brief": brief, "asks": list(asks or (brief,))})


def ask_user_call(question: str, *options: str) -> types.Part:
    """An `ask_user` call putting `question` to the user with `options`."""
    return call_part(
        "ask_user",
        {"questions": [{"question": question, "options": [{"label": o} for o in options]}]},
    )

"""A chip under an answer, applied by the `seed` node with no search, on the
turn loop over Postgres.

The turn opens with the chip (`harness.graph.state.APPLY_CHOICE_STATE_KEY`)
and the reading an earlier turn stored, and the real `seed`, `orchestrate`
and `ask` nodes run in a test-only graph (`tests.graph_turns`). The chip's
resubmit reaches data-mcp in process over a `SearchStub`, and its fetch is
the orchestrator's own `fetch` tool (`harness.graph.seed.chip_fetch`).
"""

from __future__ import annotations

from typing import Any

import pytest
from compiler.surface import ProgressCallback
from data_mcp.definition_store import InMemoryDefinitionStore
from harness import scripted
from harness.graph.seed import chip_fetch
from harness.graph.state import (
    APPLY_CHOICE_STATE_KEY,
    ORCHESTRATOR_APPLIED_KEY,
    ORCHESTRATOR_PRIOR_LEDGER_KEY,
    ORCHESTRATOR_SEARCHES_KEY,
    ORCHESTRATOR_SPECS_KEY,
    frame_id_for,
)
from harness.graph.switch import SCOPE_CHOICE_NOT_APPLIED
from harness.orchestrator_tools import build_orchestrator_tools
from harness.scripted import SCRIPTED_CV
from harness.tool_contracts import HarnessCode, reading_facts
from harness.turn_records import CHIP_STALE_NOTE, SearchReading
from orchestrator.agent import orchestrator_agent
from sqlalchemy.ext.asyncio import AsyncEngine

from .data_mcp_definitions import DataMcpDefinitions, SearchStub
from .graph_turns import agent_read, opening_graph, opening_nodes
from .scope_scenarios import (
    _AFRICA_TOTAL,
    _AFRICA_UNKNOWN_CLASS,
    _AFRICA_UNPOSTABLE,
    _hashed_scope,
    _switched_answer,
)
from .scripted_genai import ScriptedGenai, reply, text_part
from .scripted_tools import no_resolver, search_call
from .turn_driver import TurnDriver, scripted_models


class _RecordingExecutor(scripted.ScriptedExecutor):
    """`harness.scripted.ScriptedExecutor`, recording each call and each step a node records."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.steps: list[dict[str, Any]] = []

    async def __call__(
        self, *, tool: str, args: dict[str, Any], on_progress: ProgressCallback | None = None
    ) -> dict[str, Any]:
        self.calls.append({"tool": tool, "args": args})
        return await super().__call__(tool=tool, args=args, on_progress=on_progress)

    def record_step(self, **step: Any) -> None:
        self.steps.append(step)


class _BrokenFetchExecutor(_RecordingExecutor):
    """`_RecordingExecutor` whose `get_data` raises, as a fault inside DFO would."""

    async def __call__(
        self, *, tool: str, args: dict[str, Any], on_progress: ProgressCallback | None = None
    ) -> dict[str, Any]:
        if tool == "get_data":
            raise RuntimeError("get_data broke")
        return await super().__call__(tool=tool, args=args, on_progress=on_progress)


def _frame_for(spec_id: str) -> str:
    """The frame id a `fetch` of `spec_id` registers when
    `harness.scripted.ScriptedExecutor` answers its `get_data`."""
    return frame_id_for({"spec_id": spec_id, "content_version": SCRIPTED_CV})


async def _chip_turn(
    engine: AsyncEngine,
    search: SearchStub,
    *,
    option_id: str = "sector:120+130",
    executor: _RecordingExecutor | None = None,
    spec_id: str | None = None,
    entity_groups: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], ScriptedGenai, _RecordingExecutor, dict[str, Any]]:
    """A turn applying a chip that moves the US health aid answer's
    `option_id` axis to `option_id`, offered on the reading `spec_id`, by
    default the answer's own, its reading stored by an earlier turn from a
    block naming `entity_groups` when given, resubmitted through `search`.
    Its model tries to search, then answers. Returns the turn's state as its
    nodes read it, the turn's spec blocks among it, the agent's model, the
    executor and the scope."""
    scope = _hashed_scope()
    pin = scope["compiled"]["spec_id"]
    block = {"spec_id": pin, "resolved": {"scope_choices": scope}}
    if entity_groups is not None:
        block["resolved"]["entity_groups"] = entity_groups
    reading = SearchReading.from_part(
        {"variable": "health aid", "status": "definite", "spec_id": pin, **reading_facts(block)}
    ).model_dump(mode="json")
    executor = executor or _RecordingExecutor()
    models = scripted_models(
        orchestrator=[reply(search_call("health aid")), reply(text_part("done"))]
    )
    tools = build_orchestrator_tools(resolver=no_resolver, executor=executor)
    nodes = opening_nodes(
        orchestrator_agent(tools),
        models,
        executor=executor,
        definition_surface=DataMcpDefinitions(InMemoryDefinitionStore(), search=search),
        fetch=chip_fetch(tools),
    )
    driver = TurnDriver(engine, graph=opening_graph(nodes))
    axis = option_id.split(":", maxsplit=1)[0]
    run = await driver.turn(
        "Count health and population.",
        state={
            "surface": "studio",
            APPLY_CHOICE_STATE_KEY: {
                "choice_id": f"{scope['id']}:{axis}",
                "option_id": option_id,
                "spec_id": spec_id or pin,
            },
            ORCHESTRATOR_PRIOR_LEDGER_KEY: {"readings": [reading], "warnings": {}, "spec_ids": []},
        },
    )
    state = dict(run.cursor.state)
    if specs := await driver.turns.specs(run.cursor):
        state[ORCHESTRATOR_SPECS_KEY] = specs
    return state, models.orchestrator, executor, scope


async def test_a_chip_compiles_through_search_fetches_what_it_compiled_and_never_searches(
    pg_engine: AsyncEngine,
) -> None:
    """The `seed` node posts the stored reading's base identity with the
    switched entries, stores the reading Search compiles, fetches its spec_id
    and hands the agent the result; the agent's own search is refused unsent."""
    scope = _hashed_scope()
    new_pin, envelope = _switched_answer(scope)
    search = SearchStub(answer={scope["id"]: envelope})

    state, model, executor, scope = await _chip_turn(pg_engine, search)

    (posted,) = search.posted
    (entry,) = posted["input"]["entries"]
    assert entry["identity"] == scope["base"]["identity"]
    assert entry["scope_entries"][-1]["option_id"] == "sector:120+130"
    assert [c["args"]["spec_id"] for c in executor.calls if c["tool"] == "get_data"] == [new_pin]
    stored = SearchReading.model_validate(state[ORCHESTRATOR_SEARCHES_KEY][-1])
    assert (stored.text, stored.spec_id) == ("health aid", new_pin)
    assert stored.fetch is not None and stored.fetch.refusal_code is None
    assert state[ORCHESTRATOR_APPLIED_KEY] == {
        "choice_id": f"{scope['id']}:sector",
        "option_id": "sector:120+130",
        "from_option_id": "sector:120",
        "spec_id": new_pin,
    }
    first = agent_read(model.requests[0])
    assert f"counted it again as {new_pin}" in first
    assert _frame_for(new_pin) in first
    searched = model.requests[1].content_list[-1].parts[0].function_response
    assert searched is not None and searched.response is not None
    assert searched.response["code"] == "tool_budget_spent"
    assert [step["result"]["outcome"] for step in executor.steps] == ["ok"]


@pytest.mark.parametrize(
    "group", [_AFRICA_TOTAL, _AFRICA_UNKNOWN_CLASS], ids=["total", "unknown-class"]
)
async def test_a_chip_posts_the_groups_its_reading_was_resolved_with(
    pg_engine: AsyncEngine, group: dict[str, Any]
) -> None:
    """The answer's reading came from a search whose block named Africa as the
    set its recipients were asked for, and no plan kept that block. The chip's
    resubmit posts the set as every resubmit does (`compiler.sources.wire_groups`),
    one naming a class of party this release does not know too, since the reading
    stored what the resubmit posts of it."""
    scope = _hashed_scope()
    _, envelope = _switched_answer(scope)
    search = SearchStub(answer={scope["id"]: envelope})

    await _chip_turn(pg_engine, search, entity_groups={"destination": group})

    (posted,) = search.posted
    (entry,) = posted["input"]["entries"]
    assert entry["entity_groups"] == {"destination": {"label": "Africa", "ref": "africa"}}


async def test_a_chip_whose_readings_sets_could_not_be_read_posts_nothing(
    pg_engine: AsyncEngine,
) -> None:
    """The answer's reading came from a block naming a set on its recipients with a
    registry id that is not text, and no block of it is kept, so nothing names the set. Posting the part
    without it would let Search count another population, so the chip posts nothing
    and its step fails as a plan card's switch over such a block does."""
    scope = _hashed_scope()
    _, envelope = _switched_answer(scope)
    search = SearchStub(answer={scope["id"]: envelope})

    state, _, executor, _ = await _chip_turn(
        pg_engine, search, entity_groups={"destination": _AFRICA_UNPOSTABLE}
    )

    assert search.posted == []
    (step,) = executor.steps
    refusal = step["result"]["refusal"]
    assert (refusal["code"], refusal["retry_mode"]) == (SCOPE_CHOICE_NOT_APPLIED, "do_not_retry")
    assert not any(c["tool"] == "get_data" for c in executor.calls)
    assert ORCHESTRATOR_APPLIED_KEY not in state


async def test_a_chip_offered_on_another_reading_is_refused_stale_and_compiles_nothing(
    pg_engine: AsyncEngine,
) -> None:
    """The chip names a reading other than the newest one holding its choice, as
    after a later switch of that choice: it is refused as the studio refuses a
    stale chip, and nothing is posted or fetched."""
    scope = _hashed_scope()
    _, envelope = _switched_answer(scope)
    search = SearchStub(answer={scope["id"]: envelope})

    state, model, executor, _ = await _chip_turn(pg_engine, search, spec_id=f"spec_{'9' * 32}")

    assert search.posted == []
    assert CHIP_STALE_NOTE in agent_read(model.requests[0])
    (step,) = executor.steps
    assert step["result"]["refusal"]["code"] == SCOPE_CHOICE_NOT_APPLIED
    assert not any(c["tool"] == "get_data" for c in executor.calls)
    assert ORCHESTRATOR_APPLIED_KEY not in state


async def test_a_chip_whose_fetch_raises_answers_as_the_agents_own_fetch_would(
    pg_engine: AsyncEngine,
) -> None:
    """The chip's fetch raises inside DFO: the agent is handed the failure the
    orchestrator's own tool calls answer with, and the turn goes on."""
    scope = _hashed_scope()
    _, envelope = _switched_answer(scope)
    search = SearchStub(answer={scope["id"]: envelope})

    _, model, _, _ = await _chip_turn(pg_engine, search, executor=_BrokenFetchExecutor())

    first = agent_read(model.requests[0])
    assert "fetch failed inside DFO (RuntimeError)" in first
    assert HarnessCode.RAISED.value in first


@pytest.mark.parametrize(
    ("code", "retryable", "step_code", "retry_mode"),
    [
        ("requested_option_unavailable", False, "requested_option_unavailable", "do_not_retry"),
        # A Search that did not answer states no code of its own.
        ("transient_unavailable", True, SCOPE_CHOICE_NOT_APPLIED, "retry_after_delay"),
    ],
    ids=["refused", "transient"],
)
async def test_a_chip_search_refuses_reads_as_a_plan_cards_switch_and_fetches_nothing(
    pg_engine: AsyncEngine, code: str, retryable: bool, step_code: str, retry_mode: str
) -> None:
    """The chip's step fails with the code and retryability the plan card's
    switch reports for the same refusal."""
    scope = _hashed_scope()
    search = SearchStub(
        refuse={
            scope["id"]: {
                "code": code,
                "title": "Requested Option Unavailable",
                "detail": "Scope entry 1 cannot apply to this part: option_unknown.",
                "pointer": "/input/entries/0/scope_entries/1",
                "retryable": retryable,
            }
        }
    )

    state, model, executor, _ = await _chip_turn(pg_engine, search)

    note = (
        f"Search {'could not' if retryable else 'cannot'} count health (CRS 120) and "
        "population and reproductive health (CRS 130) "
        f"{'just now' if retryable else 'for this source'}."
    )
    assert note in agent_read(model.requests[0])
    (step,) = executor.steps
    assert step["result"]["refusal"] == {
        "code": step_code,
        "retry_mode": retry_mode,
        "fix": None,
        "message": note,
    }
    assert not any(c["tool"] == "get_data" for c in executor.calls)
    assert ORCHESTRATOR_APPLIED_KEY not in state

"""A seeded turn reads every fact about a seed from Search's own answer for its identity.

The browser hands over a Spec block. `harness.graph`'s `seed` node keeps its
spec_id, identity and operations, resolves the identity again through
data-mcp's `resolve_definition`, and stores Search's answer as a definite
search part the orchestrator's agent fetches or plans from. These tests run
data-mcp in process over a `SearchStub` that answers from recorded Spec
blocks.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx2
import pytest
from common.hashed_fields import revision_content
from compiler.sources import source_from_spec_block
from data_mcp import client_holder
from data_mcp.client import UpstreamClient
from data_mcp.definition_store import (
    DefinitionStoreUnavailable,
    InMemoryDefinitionStore,
    canonicalize,
    spec_id_for,
)
from data_mcp.definition_store import set_store as set_definition_store
from data_mcp.server import mcp as data_mcp
from data_mcp.settings import Settings as DataMcpSettings
from fastmcp import Client
from google.genai import types
from harness.graph.state import APPROVE_INTERRUPT, ORCHESTRATOR_SEARCHES_KEY, ORCHESTRATOR_SPECS_KEY
from harness.mcp_client import _refusal_outcome
from harness.scripted import ScriptedPlannerModel, ScriptedSurface
from harness.seeds import SEED_STEP_TOOL, resolve_seed, take_seeds
from harness.studio_translate import _node_name
from planner.plan import SEEDS_STATE_KEY
from translator.discovery import resolved_spec

from .data_mcp_definitions import DataMcpDefinitions, GroupsSearchStub, SearchStub
from .orchestrator_fakes import (
    CannedSearch,
    RefusingPlanner,
    ScriptLlm,
    approve_answer,
    call,
    create_request_input_response,
    fetch_call,
    open_session,
    orchestrator_app,
    orchestrator_runner,
    plan_call,
    request_text,
    run_turn,
    single_pause,
    stream,
    studio_client,
    turn_body,
    visited_nodes,
)

_FIXTURES = Path(__file__).parent / "fixtures"

# The Spec block the dev Search API served for "top 5 donors of education aid
# to Kenya": a CRS flow with Kenya bound under `destination`.
_KENYA = json.loads((_FIXTURES / "seed_education_aid_kenya.json").read_text())["spec"]

# Each recorded parsed-door answer's own resolve time, in milliseconds
# (`diagnostics.timing_ms.total`).
_RECORDED_MS = {
    name: json.loads(
        (Path(__file__).parents[2] / "data-mcp" / "tests" / "fixtures" / f"{name}.json").read_text()
    )["diagnostics"]["timing_ms"]["total"]
    for name in (
        "resolve_gdp_per_capita_guatemala",
        "resolve_health_spending_broad",
        "resolve_population_guatemala",
    )
}

_QUESTION = "top 5 donors of education aid"


@pytest.fixture
def definition_store() -> Any:
    store = InMemoryDefinitionStore()
    set_definition_store(store)
    yield store
    set_definition_store(None)


def _forged_kenya() -> dict[str, Any]:
    """Kenya's block with every fact the browser could restate changed: the entity's
    label and the sentence name Uganda, the coverage and headline counts differ, and
    an open axis claims it sums. The identity, spec_id and StatVars are Kenya's."""
    forged = copy.deepcopy(_KENYA)
    resolved = forged["resolved"]
    resolved["entities"][0]["label"] = "Uganda"
    resolved["rendered_sentence"] = resolved["rendered_sentence"].replace("Kenya", "Uganda")
    resolved["coverage"]["dimensions"] = [{"label": "recipients", "count": 1}]
    resolved["headline"] = {**resolved["headline"], "year": 2019, "observations": 7}
    for slot in resolved["slots"]:
        slot["summable"] = True
    return forged


def _block(name: str) -> dict[str, Any]:
    """A Spec block for a dataset named `name`, whose identity hashes to its spec_id."""
    identity = {"identity_version": "spec1", "seed": name, "requested_window": {"kind": "all"}}
    spec_id = spec_id_for(canonicalize(identity))
    return {
        "spec_id": spec_id,
        "definition": {"schema_version": 1, "spec_id": spec_id, "identity": identity},
        "resolved": {
            "rendered_sentence": f"the reading {name}",
            "stat_vars": {"member_count": 1, "member_count_exact": True, "served_count": 1},
        },
    }


_RANK = {"verb": "rank", "phrase": "top 5 donors", "over": "origin"}


def _seeds(
    blocks: Sequence[Mapping[str, Any]], operations: Sequence[Mapping[str, Any]] = (_RANK,)
) -> list[dict[str, Any]]:
    """`seed_spec` handing over `blocks`, each with `operations`."""
    return [{"spec": block, "operations": list(operations)} for block in blocks]


class _Steps:
    """The executor: `record_step` collects each step `seed` records, and any
    tool call fails the test. `recorded` holds every step's tool and id, in
    the order recorded."""

    def __init__(self) -> None:
        self.steps: dict[str, dict[str, Any]] = {}
        self.recorded: list[tuple[str, str]] = []

    def record_step(
        self, *, tool: str, args: dict[str, Any], result: dict[str, Any], step_id: str
    ) -> None:
        del args
        self.steps[step_id] = result
        self.recorded.append((tool, step_id))

    async def __call__(self, *, tool: str, args: dict[str, Any], **_: Any) -> Any:
        raise AssertionError(f"no tool call expected, got {tool}")


@dataclass(frozen=True)
class _Turn:
    """What one turn left: its session state, the steps `seed` recorded, the
    nodes visited in order, and the agent's model."""

    state: dict[str, Any]
    steps: dict[str, dict[str, Any]]
    visited: list[str]
    model: ScriptLlm

    @property
    def read(self) -> str:
        """The text of the agent's first model request."""
        return request_text(self.model.requests[0])

    @property
    def parts(self) -> list[dict[str, Any]]:
        """The `parts` the hand-back gave the agent."""
        line = next(line for line in self.read.split("\n") if line.startswith('{"parts"'))
        return json.loads(line)["parts"]


async def _run(slot: object, surface: DataMcpDefinitions, *, on: str = "studio") -> _Turn:
    """Run one turn on the orchestrator topology with `slot` in the seeds slot,
    as `harness.studio` writes it, resolving seeds through `surface`, on the
    `on` surface."""
    steps = _Steps()
    runner, model = orchestrator_runner(executor=steps, definition_surface=surface)
    seeded = {} if slot is None else {SEEDS_STATE_KEY: json.dumps(slot)}
    await open_session(runner, surface=on, **seeded)
    events = await run_turn(runner, types.UserContent(_QUESTION))
    held = await runner.session_service.get_session(app_name="t", user_id="u", session_id="s")
    assert held is not None
    return _Turn(dict(held.state), steps.steps, visited_nodes(events), model)


# The hand-back's opening, ahead of the parts.
_SEEDED = (
    "The user brought these datasets from Search: fetch or plan each with its spec_id. A "
    "part's operations are what the user's question asked of it; ask for them with plan."
)


async def test_a_seeded_turn_hands_the_agent_each_seed_as_a_definite_search_part(
    definition_store: InMemoryDefinitionStore,
) -> None:
    """Both seeds reach the agent as definite parts with their operations, ahead of
    its first call, and are stored as a search stores its parts, read from `seed`."""
    del definition_store
    water = _block("water")

    turn = await _run(_seeds([_KENYA, water]), DataMcpDefinitions(served=[_KENYA, water]))

    assert turn.visited[:2] == ["seed", "orchestrate"]
    assert _SEEDED in turn.read
    assert [(p["status"], p["spec_id"], p["operations"]) for p in turn.parts] == [
        ("definite", _KENYA["spec_id"], [_RANK]),
        ("definite", water["spec_id"], [_RANK]),
    ]
    assert turn.steps["seeds"]["summary"] == "received 2 of 2 dataset(s) from Search"
    assert turn.state[SEEDS_STATE_KEY] is None
    assert set(turn.state[ORCHESTRATOR_SPECS_KEY]) == {_KENYA["spec_id"], water["spec_id"]}
    assert [(r["spec_id"], r["origin"]) for r in turn.state[ORCHESTRATOR_SEARCHES_KEY]] == [
        (_KENYA["spec_id"], "seed"),
        (water["spec_id"], "seed"),
    ]


# The hand-back's opening on the download surface, which fetches one dataset
# and never plans (`harness.graph.state.DOWNLOAD_BUDGETS`).
_SEEDED_ON_DOWNLOAD = (
    "The user brought these datasets from Search: fetch each with its spec_id. A part's "
    "operations are what the user's question asked of it; they cannot run here, so tell the "
    "user the data comes as served. fetch can run once this turn, so fetch the one dataset "
    "the question most needs."
)


async def test_a_seeded_download_turn_hands_back_one_fetch_and_no_plan(
    definition_store: InMemoryDefinitionStore,
) -> None:
    """On the download surface the hand-back follows the turn's budgets: it offers
    fetch alone, once, for two seeds, and says their operations cannot run."""
    del definition_store
    water = _block("water")

    turn = await _run(
        _seeds([_KENYA, water]), DataMcpDefinitions(served=[_KENYA, water]), on="download"
    )

    assert _SEEDED_ON_DOWNLOAD in turn.read
    assert "plan" not in turn.read.split('{"parts"')[0]
    assert [p["spec_id"] for p in turn.parts] == [_KENYA["spec_id"], water["spec_id"]]


async def test_a_turn_with_no_seeds_reaches_the_agent_with_the_question_alone() -> None:
    turn = await _run(None, DataMcpDefinitions())

    assert turn.visited[:2] == ["seed", "orchestrate"]
    assert turn.read.endswith(f"\n{_QUESTION}")
    assert turn.steps == {}
    assert ORCHESTRATOR_SEARCHES_KEY not in turn.state


async def test_a_single_reference_seed_spec_fails_its_step_and_the_agent_reads_the_question() -> (
    None
):
    reference = {"spec_id": _KENYA["spec_id"], "content_version": "cv_" + "0" * 32}

    turn = await _run(reference, DataMcpDefinitions(served=[_KENYA]))

    assert turn.steps["seeds"]["refusal"]["code"] == "seed_reference_unused"
    assert turn.visited[:2] == ["seed", "orchestrate"]
    assert turn.read.endswith(f"\n{_QUESTION}")


async def test_a_label_the_browser_changed_is_never_read_and_search_names_the_entity(
    definition_store: InMemoryDefinitionStore,
) -> None:
    """The browser relabels Kenya's identity "Uganda". Search's answer for the identity
    names Kenya, so the stored block and the part the agent reads do, and nothing in
    session state or the agent's request says Uganda."""
    del definition_store
    surface = DataMcpDefinitions(served=[_KENYA])

    turn = await _run(_seeds([_forged_kenya()]), surface)

    assert turn.steps["seeds"]["outcome"] == "ok"
    stored = turn.state[ORCHESTRATOR_SPECS_KEY][_KENYA["spec_id"]]
    assert source_from_spec_block(stored).entities[0].label == "Kenya"
    (part,) = turn.parts
    assert part["resolved"] == _KENYA["resolved"]["rendered_sentence"]
    assert "Uganda" not in json.dumps(turn.state)
    assert "Uganda" not in turn.read
    (posted,) = surface.search.posted
    (entry,) = posted["input"]["entries"]
    # The resubmit posts the identity alone; Search derives the population from it.
    assert entry == {
        "entry_kind": "identity",
        "variable_text": _KENYA["spec_id"],
        "identity": _KENYA["definition"]["identity"],
    }


def _elsewhere() -> dict[str, Any]:
    """Kenya's block under another identity: the same dataset over 2024 alone."""
    identity = {
        **_KENYA["definition"]["identity"],
        "requested_window": {"kind": "range", "start_year": 2024, "end_year": 2024},
    }
    spec_id = spec_id_for(canonicalize(identity))
    return {
        **_KENYA,
        "spec_id": spec_id,
        "definition": {"schema_version": 1, "spec_id": spec_id, "identity": identity},
    }


_STALE_ANSWERS: dict[str, tuple[dict[str, Any] | None, str]] = {
    "another-spec-id": (
        {"status": "definite", "parts": [{"status": "definite", "spec": _elsewhere()}]},
        f"Search now resolves it to another dataset ({_elsewhere()['spec_id']})",
    ),
    "no-data": (
        {"status": "no_data", "parts": [{"status": "no_data", "no_data": {"reason": "no_series"}}]},
        "Search finds no data for it (no_series)",
    ),
    "no-definition": (
        {
            "status": "definite",
            "parts": [
                {
                    "status": "definite",
                    "spec": {key: value for key, value in _KENYA.items() if key != "definition"},
                }
            ],
        },
        "Search's answer for it carries no definition to retrieve",
    ),
    "unreadable": (
        {
            "status": "definite",
            "parts": [{"status": "definite", "spec": {**_KENYA, "resolved": "unreadable"}}],
        },
        "Search's answer for it cannot be read (1 errors)",
    ),
    "refused": (
        None,
        "Search no longer resolves it (request_invalid)",
    ),
}


@pytest.mark.parametrize("answer", sorted(_STALE_ANSWERS))
async def test_a_seed_search_no_longer_resolves_as_it_did_is_dropped_with_its_reason(
    definition_store: InMemoryDefinitionStore, answer: str
) -> None:
    """Search answers Kenya's identity with another spec_id, with no data, with a block
    that carries no definition, or with a refusal. The seed fails its step and the agent
    is told why it was left out; the other seed reaches it as a definite part."""
    del definition_store
    envelope, reason = _STALE_ANSWERS[answer]
    other = _block("water")
    search = SearchStub(
        [other], answer={_KENYA["spec_id"]: envelope} if envelope is not None else None
    )

    turn = await _run(_seeds([_KENYA, other]), DataMcpDefinitions(search=search))

    refusal = turn.steps["seed-0"]["refusal"]
    assert refusal["code"] == "seed_dropped"
    assert refusal["retry_mode"] == "do_not_retry"
    assert refusal["message"] == f"seed 0 is left out: {reason}. Search for it again to add it."
    assert turn.steps["seeds"]["summary"] == "received 1 of 2 dataset(s) from Search"
    assert [part["spec_id"] for part in turn.parts] == [other["spec_id"]]
    assert (
        f'"dataset 1 from Search" was left out: {reason}. Tell the user, and say that '
        "searching for it again may find it." in turn.read
    )
    assert list(turn.state[ORCHESTRATOR_SPECS_KEY]) == [other["spec_id"]]


class _UnreachableStore:
    """A definition store whose backend cannot be reached."""

    async def put_many(self, definitions: Any) -> None:
        raise DefinitionStoreUnavailable("backend unreachable")

    async def get(self, spec_id: str, *, schema_version: int) -> None:
        raise DefinitionStoreUnavailable("backend unreachable")

    async def aclose(self) -> None:
        pass


async def test_a_seed_data_mcp_cannot_resolve_fails_unavailable_and_the_agent_answers() -> None:
    """The definition store is unreachable, so no seed can be resolved. The seed's
    step and the count fail `service_unavailable`, and the agent is told to answer
    from the question, which it does."""
    set_definition_store(_UnreachableStore())
    try:
        turn = await _run(_seeds([_KENYA]), DataMcpDefinitions(served=[_KENYA]))
    finally:
        set_definition_store(None)

    assert turn.steps["seed-0"]["refusal"]["code"] == "service_unavailable"
    assert turn.steps["seeds"]["refusal"]["code"] == "service_unavailable"
    assert turn.read.endswith(
        f"\n{_QUESTION}\n"
        '"dataset 1 from Search" was left out: it could not be read just now. Tell the '
        "user, and say that searching for it again may find it.\n"
        "No dataset from Search can be used, so answer from the question."
    )
    assert ORCHESTRATOR_SEARCHES_KEY not in turn.state
    assert len(turn.model.requests) == 1


async def test_seeds_resolve_together_and_the_turn_waits_for_the_slowest(
    definition_store: InMemoryDefinitionStore,
) -> None:
    """Each answer is held for a tenth of a recorded resolve's own time. The three
    requests are in flight at once, so the seeds take as long as the slowest answer,
    well short of their sum."""
    del definition_store
    blocks = [_block(name) for name in _RECORDED_MS]
    delay_s = {
        block["spec_id"]: ms / 10_000
        for block, ms in zip(blocks, _RECORDED_MS.values(), strict=True)
    }
    search = SearchStub(blocks, delay_s=delay_s)

    turn = await _run(_seeds(blocks), DataMcpDefinitions(search=search))

    assert [part["spec_id"] for part in turn.parts] == [block["spec_id"] for block in blocks]
    assert search.peak == len(blocks)
    added = max(end for _, end in search.spans) - min(start for start, _ in search.spans)
    assert max(delay_s.values()) <= added < 0.7 * sum(delay_s.values())


async def test_a_seed_is_fetched_without_a_search_and_a_later_turns_planner_reads_it(
    definition_store: InMemoryDefinitionStore,
) -> None:
    """Over the studio routes: the agent fetches the seed's spec_id with no search,
    and the next turn's planner reads the seed among the earlier readings, with the
    fetch it served."""
    del definition_store
    search = CannedSearch()
    planner = RefusingPlanner()
    app, model = orchestrator_app(
        fetch_call(_KENYA["spec_id"]),
        types.Part.from_text(text="Here are the donors."),
        plan_call("rank donors of education aid to Kenya"),
        types.Part.from_text(text="I could not plan that."),
        resolver=search,
        planner=planner,
        served=[_KENYA],
    )
    body = {**turn_body(_QUESTION), "seed_spec": _seeds([_KENYA])}

    async with studio_client(app) as client:
        first = await stream(client, "/api/studio/turn", body)
        await stream(client, "/api/studio/turn", turn_body("rank them"))

    steps = [f for f in first if f["event"] == "activity.step" and f.get("tool")]
    assert [(f["tool"], f["status"]) for f in steps if f["status"] != "running"] == [
        ("receive_seeds", "complete"),
        ("fetch", "complete"),
    ]
    fetched = model.requests[1].contents[-1].parts[0].function_response
    assert fetched.name == "fetch"
    assert fetched.response["frames"]
    assert search.calls == []
    (readings,) = planner.readings
    seeded = [r for r in readings if r["spec_id"] == _KENYA["spec_id"]]
    assert [(r["origin"], r["status"], r["fetch"] is not None) for r in seeded] == [
        ("seed", "definite", True)
    ]


async def test_a_seed_carries_a_sets_published_total_and_posts_it_to_no_one() -> None:
    # Search recomputes `total` from what the role binds, and a server that
    # predates the field refuses it, so the resubmit posts the set without it.
    fixture = Path(__file__).parent / "fixtures" / "search_spec_region_members.json"
    block = json.loads(fixture.read_text())["spec"]
    block["resolved"]["entity_groups"]["observationAbout"]["total"] = "DACAfrica"
    (taken,) = take_seeds([{"spec": block, "operations": []}], record_step=None)
    search = GroupsSearchStub([block])

    part = await resolve_seed(DataMcpDefinitions(InMemoryDefinitionStore(), search=search), taken)

    assert taken.seed.spec.entity_groups is not None
    assert taken.seed.spec.entity_groups["observationAbout"].total == "DACAfrica"
    (posted,) = search.posted
    assert posted["input"]["entries"][0]["entity_groups"] == {
        "observationAbout": {"label": "Africa", "ref": "africa"}
    }
    source = source_from_spec_block(resolved_spec(part, spec_id=block["spec_id"]))
    assert [(group.label, group.total) for group in source.groups] == [("Africa", "DACAfrica")]


@pytest.mark.parametrize(
    ("whole_of", "hashed"),
    [
        (
            ["party"],
            {"role": "observationAbout", "label": "Africa", "ref": "africa", "whole_of": ["party"]},
        ),
        (None, {"role": "observationAbout", "label": "Africa", "ref": "africa"}),
    ],
    ids=["with-the-marker", "without-the-field"],
)
async def test_a_seed_resolves_its_set_with_the_kinds_its_whole_total_holds(
    monkeypatch: pytest.MonkeyPatch, whole_of: list[str] | None, hashed: dict[str, Any]
) -> None:
    # A seed without the field resolves to the group it always did, so a plan hash over
    # the source does not move.
    fixture = Path(__file__).parent / "fixtures" / "search_spec_region_members.json"
    block = json.loads(fixture.read_text())["spec"]
    if whole_of is not None:
        block["resolved"]["entity_groups"]["observationAbout"]["whole_of"] = whole_of
    (taken,) = take_seeds([{"spec": block, "operations": []}], record_step=None)
    resolved: list[Any] = []

    async def capture(surface: object, **kwargs: Any) -> dict[str, Any]:
        resolved.extend(kwargs["groups"])
        return {}

    monkeypatch.setattr("harness.seeds.resolve_definition", capture)

    await resolve_seed(ScriptedSurface(), taken)

    (group,) = resolved
    assert revision_content(group) == hashed


async def test_a_seed_over_a_sets_places_keeps_the_set_when_resolved_again() -> None:
    # The browser hands over "African countries" as Search served it; the seed
    # node's resubmit posts Africa beside the identity, so Search's answer, the
    # one the turn reads, names Africa again.
    fixture = Path(__file__).parent / "fixtures" / "search_spec_region_members.json"
    block = json.loads(fixture.read_text())["spec"]
    (taken,) = take_seeds([{"spec": block, "operations": []}], record_step=None)
    resolving = DataMcpDefinitions(InMemoryDefinitionStore(), search=GroupsSearchStub([block]))

    part = await resolve_seed(resolving, taken)

    source = source_from_spec_block(resolved_spec(part, spec_id=block["spec_id"]))
    assert [(group.role, group.label, group.ref) for group in source.groups] == [
        ("observationAbout", "Africa", "africa")
    ]


def _forged(block: Mapping[str, Any]) -> dict[str, Any]:
    """`block` with its identity changed, so it does not hash to its spec_id."""
    forged = copy.deepcopy(dict(block))
    forged["definition"]["identity"]["seed"] = "forged"
    return forged


def _misnamed(block: Mapping[str, Any], spec_id: str) -> dict[str, Any]:
    """`block` whose definition names `spec_id`, another block's."""
    misnamed = copy.deepcopy(dict(block))
    misnamed["definition"]["spec_id"] = spec_id
    return misnamed


def _undefined(block: Mapping[str, Any]) -> dict[str, Any]:
    """`block` with no definition."""
    return {key: value for key, value in block.items() if key != "definition"}


@pytest.mark.parametrize(
    ("bad", "code"),
    [
        pytest.param(lambda good: _forged(_block("kenya")), "identity_mismatch", id="forged"),
        pytest.param(
            lambda good: _misnamed(_block("kenya"), good["spec_id"]), "seed_invalid", id="misnamed"
        ),
        pytest.param(lambda good: _undefined(_block("kenya")), "seed_invalid", id="undefined"),
    ],
)
async def test_a_seed_that_cannot_be_read_fails_its_own_step_and_the_other_reaches_the_agent(
    definition_store: InMemoryDefinitionStore, bad: Any, code: str
) -> None:
    """A seed whose identity does not hash to its spec_id, whose definition names
    another spec_id, or that carries no definition fails its own step before
    anything is resolved; the seed beside it reaches the agent."""
    good = _block("education")

    turn = await _run(_seeds([good, bad(good)]), DataMcpDefinitions(served=[good]))

    assert turn.steps["seed-1"]["refusal"]["code"] == code
    assert turn.steps["seeds"]["summary"] == "received 1 of 2 dataset(s) from Search"
    assert [part["spec_id"] for part in turn.parts] == [good["spec_id"]]
    assert list(turn.state[ORCHESTRATOR_SPECS_KEY]) == [good["spec_id"]]
    assert await definition_store.get(bad(good)["spec_id"], schema_version=1) is None


async def test_a_turn_with_no_seed_it_can_read_reaches_the_agent_with_the_question_alone(
    definition_store: InMemoryDefinitionStore,
) -> None:
    del definition_store
    turn = await _run(_seeds([_undefined(_block("education"))]), DataMcpDefinitions())

    assert turn.steps["seed-0"]["refusal"]["code"] == "seed_invalid"
    assert turn.visited[:2] == ["seed", "orchestrate"]
    assert turn.read.endswith(f"\n{_QUESTION}")
    assert ORCHESTRATOR_SEARCHES_KEY not in turn.state


class _RefusesOneSpec(InMemoryDefinitionStore):
    """An in-memory store whose backend cannot take one spec_id's write."""

    def __init__(self, refused: str) -> None:
        super().__init__()
        self._refused = refused

    async def put_many(self, definitions: Any) -> None:
        if any(definition.spec_id == self._refused for definition in definitions):
            raise DefinitionStoreUnavailable("1 of 1 definition write(s) failed")
        await super().put_many(definitions)


async def test_a_seed_the_store_cannot_take_fails_unavailable_and_the_other_reaches_the_agent() -> (
    None
):
    education, water = _block("education"), _block("water")
    store = _RefusesOneSpec(education["spec_id"])
    set_definition_store(store)
    try:
        turn = await _run(_seeds([education, water]), DataMcpDefinitions(served=[education, water]))
    finally:
        set_definition_store(None)

    assert turn.steps["seed-0"]["refusal"]["code"] == "service_unavailable"
    assert turn.steps["seeds"]["outcome"] == "ok"
    assert [part["spec_id"] for part in turn.parts] == [water["spec_id"]]
    assert await store.get(education["spec_id"], schema_version=1) is None
    assert await store.get(water["spec_id"], schema_version=1) is not None


@pytest.mark.parametrize("contradicts", [False, True], ids=["a-copy", "another-block"])
async def test_a_seed_naming_an_earlier_seed_s_spec_id_fails_its_own_step_and_the_first_stands(
    definition_store: InMemoryDefinitionStore, contradicts: bool
) -> None:
    """One spec_id is one seed: a repeat is refused whether it copies the first
    or carries another `resolved` block and other operations under the same id."""
    del definition_store
    first = _block("education")
    repeat = copy.deepcopy(first)
    operations: Sequence[Mapping[str, Any]] = ()
    if contradicts:
        repeat["resolved"]["rendered_sentence"] = "another reading"
        operations = (_RANK,)
    slot = [*_seeds([first], operations=()), *_seeds([repeat], operations=operations)]

    turn = await _run(slot, DataMcpDefinitions(served=[first]))

    refusal = turn.steps["seed-1"]["refusal"]
    assert refusal["code"] == "seed_duplicate"
    assert refusal["message"] == (
        f"seed 1 names the dataset seed 0 names ({first['spec_id']}); a dataset is handed "
        "over once, and the first stands"
    )
    ((spec_id, operations_read),) = [(p["spec_id"], p["operations"]) for p in turn.parts]
    assert (spec_id, operations_read) == (first["spec_id"], [])


def _refused_mint(request: httpx2.Request) -> httpx2.Response:
    """The apis service's 409 `spec_not_exportable`, answering a `spec_id` mint."""
    assert request.url.path == "/api/datasets"
    return httpx2.Response(
        409,
        json={
            "type": "/codes/spec_not_exportable",
            "title": "Spec Not Exportable",
            "status": 409,
            "detail": "the identity enumerates no member",
            "code": "spec_not_exportable",
            "retryable": False,
        },
    )


class _DataMcpGetData(_Steps):
    """`get_data` through data-mcp in process over its own definition `store`,
    its upstream mint answered by `respond`, read as
    `harness.mcp_client.McpExecutor` reads a refusal."""

    def __init__(
        self, store: InMemoryDefinitionStore, respond: Callable[[httpx2.Request], httpx2.Response]
    ) -> None:
        super().__init__()
        self._store = store
        self._respond = respond

    async def __call__(self, *, tool: str, args: dict[str, Any], **_: Any) -> Any:
        upstream = UpstreamClient(
            settings=DataMcpSettings(), transport=httpx2.MockTransport(self._respond)
        )
        client_holder.set_client(upstream)
        try:
            async with Client(data_mcp) as client:
                # data-mcp's lifespan installs a store of its own on entry.
                set_definition_store(self._store)
                result = await client.call_tool(tool, args, raise_on_error=False)
        finally:
            client_holder.set_client(None)
        assert result.is_error, result
        assert result.structured_content is not None
        return _refusal_outcome(result.structured_content)


async def test_a_seed_is_stored_in_the_data_mcp_whose_get_data_retrieves_it(
    definition_store: InMemoryDefinitionStore,
) -> None:
    """data-mcp holds a definition store apart from the one the harness process
    holds, as when it runs as its own service. The seed is stored in data-mcp's,
    and the agent's `fetch` of it posts its definition to the mint from there."""
    service = InMemoryDefinitionStore()
    block = _block("population")
    posted: list[dict[str, Any]] = []

    def respond(request: httpx2.Request) -> httpx2.Response:
        posted.append(json.loads(request.content))
        return _refused_mint(request)

    surface = DataMcpDefinitions(service, served=[block])
    runner, _ = orchestrator_runner(
        fetch_call(block["spec_id"]),
        executor=_DataMcpGetData(service, respond),
        definition_surface=surface,
    )
    await open_session(runner, **{SEEDS_STATE_KEY: json.dumps(_seeds([block]))})
    await run_turn(runner, types.UserContent("population of Kenya"))

    assert await service.get(block["spec_id"], schema_version=1) is not None
    assert await definition_store.get(block["spec_id"], schema_version=1) is None
    (mint,) = posted
    assert mint["input"]["definition"]["identity"] == block["definition"]["identity"]
    assert [call["spec_id"] for call in surface.calls] == [block["spec_id"]]


class _Describes(_Steps):
    """`_Steps` that answers the `describe_frame` calls an approved plan's run makes."""

    async def __call__(self, *, tool: str, args: dict[str, Any], **_: Any) -> Any:
        assert tool == "describe_frame", tool
        return {"outcome": "ok", "view": {"dimensions": [{"name": "observationAbout"}]}}


def _ask_user_answer(pause: types.FunctionCall) -> dict[str, Any]:
    return {"result": {"spec_id": "o1"}}


def _approval(pause: types.FunctionCall) -> dict[str, Any]:
    assert pause.id == APPROVE_INTERRUPT
    return {"result": approve_answer(pause.args["payload"]["revision"])}


_PAUSES = {
    "ask_user": (
        lambda: call(
            "ask_user",
            question="Which donors?",
            options=[{"label": "DAC members"}, {"label": "All"}],
        ),
        _ask_user_answer,
    ),
    "plan": (
        lambda: call("plan", brief="rank donors of education aid to Kenya", asks=[]),
        _approval,
    ),
}


@pytest.mark.parametrize("pausing", sorted(_PAUSES))
async def test_a_seeded_turn_takes_its_seeds_once_across_a_pause(
    definition_store: InMemoryDefinitionStore, pausing: str
) -> None:
    """A seeded turn that pauses on `ask_user` or on a plan's approval resumes
    without running `seed` again: one hand-back and one set of its steps
    across the turn, which then completes."""
    del definition_store
    pausing_call, answer = _PAUSES[pausing]
    steps = _Describes()
    runner, model = orchestrator_runner(
        pausing_call(),
        types.Part.from_text(text="Here are the donors."),
        planner=ScriptedPlannerModel(),
        tool_surface=ScriptedSurface(),
        executor=steps,
        definition_surface=DataMcpDefinitions(served=[_KENYA]),
    )
    await open_session(runner, **{SEEDS_STATE_KEY: json.dumps(_seeds([_KENYA]))})

    pause = single_pause(await run_turn(runner, types.UserContent(_QUESTION)))
    reply = create_request_input_response(pause.id, answer(pause))
    resumed = await run_turn(runner, types.Content(role="user", parts=[reply]))

    assert "seed" not in visited_nodes(resumed)
    assert [step for tool, step in steps.recorded if tool == SEED_STEP_TOOL] == ["seeds"]
    assert request_text(model.requests[-1]).count(_SEEDED) == 1
    assert not [e for e in resumed if e.error_code]
    assert not [c for e in resumed for c in e.get_function_calls() if c.name == "adk_request_input"]
    assert model.script == []


async def test_a_turn_paused_without_a_seed_node_run_resumes_without_taking_its_seeds(
    definition_store: InMemoryDefinitionStore,
) -> None:
    """A turn paused before the workflow had a `seed` node: its events carry no
    `seed` run and its seeds slot still holds the seeds the host wrote. The
    resume clears the slot, takes none of them, hands nothing back, and
    completes."""
    del definition_store
    steps = _Steps()
    runner, model = orchestrator_runner(
        call(
            "ask_user",
            question="Which donors?",
            options=[{"label": "DAC members"}, {"label": "All"}],
        ),
        types.Part.from_text(text="Here are the donors."),
        executor=steps,
        definition_surface=DataMcpDefinitions(served=[_KENYA]),
    )
    await open_session(runner)
    pause = single_pause(await run_turn(runner, types.UserContent(_QUESTION)))
    held = runner.session_service.sessions["t"]["u"]["s"]
    held.events = [e for e in held.events if _node_name(e) != "seed"]
    held.state[SEEDS_STATE_KEY] = json.dumps(_seeds([_KENYA]))

    reply = create_request_input_response(pause.id, _ask_user_answer(pause))
    resumed = await run_turn(runner, types.Content(role="user", parts=[reply]))

    # ADK runs `seed` live on the resume, since the turn's events hold no run of it.
    assert "seed" in visited_nodes(resumed)
    assert steps.recorded == []
    assert _KENYA["spec_id"] not in request_text(model.requests[-1])
    after = await runner.session_service.get_session(app_name="t", user_id="u", session_id="s")
    assert after is not None
    assert after.state[SEEDS_STATE_KEY] is None
    assert ORCHESTRATOR_SEARCHES_KEY not in after.state
    assert not [e for e in resumed if e.error_code]
    assert model.script == []

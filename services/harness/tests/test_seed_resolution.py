"""A seeded turn reads every fact about a seed from Search's own answer for its identity.

The browser hands over a Spec block. `harness.graph`'s `seed` node keeps its
spec_id, identity, StatVar dcids and operations, resolves the identity again
through data-mcp's `resolve_definition`, and reads labels, the sentence, the
population facts and summability from what Search answers. These tests run
data-mcp in process over a `SearchStub` that answers from recorded Spec
blocks.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from compiler.evidence import RetrievalEvidence, check_evidence, errors
from compiler.sources import source_from_spec_block
from data_mcp.definition_store import (
    InMemoryDefinitionStore,
    canonicalize,
    spec_id_for,
)
from data_mcp.definition_store import set_store as set_definition_store
from data_mcp.manifest import SeriesStatisticsModel
from data_mcp.series_statistics import series_statistics_table
from google.adk.runners import InMemoryRunner
from google.adk.workflow._errors import NodeInterruptedError
from google.adk.workflow.utils._workflow_hitl_utils import REQUEST_INPUT_FUNCTION_CALL_NAME
from google.genai import types
from harness.graph import PLANNER_STARTED_KEY, GenerativeNodes, create_root_workflow
from harness.scripted import ScriptedPlannerModel, ScriptedResolver
from harness.seeds import resolve_seed, take_seeds
from planner.plan import (
    SEEDS_STATE_KEY,
    DatasetRequirement,
    Entity,
    dropped_seeds,
    names_dropped_seed,
    seeded_sources,
)
from planner.records import Period
from planner.workflow import ResolvedEntity
from translator.discovery import resolved_spec

from .data_mcp_definitions import DataMcpDefinitions, GroupsSearchStub, SearchStub

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


class _Steps:
    """The executor: `record_step` collects each step `seed` records, and a
    `get_data` answers with one frame of the spec_id it names, recorded in
    `fetched`."""

    def __init__(self) -> None:
        self.steps: dict[str, dict[str, Any]] = {}
        self.fetched: list[str] = []

    def record_step(
        self, *, tool: str, args: dict[str, Any], result: dict[str, Any], step_id: str
    ) -> None:
        del tool, args
        self.steps[step_id] = result

    async def __call__(self, *, tool: str, args: dict[str, Any], **_: Any) -> dict[str, Any]:
        assert tool == "get_data", tool
        self.fetched.append(args["spec_id"])
        ref = {"spec_id": args["spec_id"], "content_version": "cv_" + "9" * 32}
        return {"outcome": "ok", "frame_refs": [ref], "summary": "minted"}


@dataclass(frozen=True)
class _Turn:
    """What one seeded turn left: its session state, the steps `seed` recorded,
    the spec_ids `get_data` was called for, the nodes visited in order, each
    node's last output, the planner node's start event, and the question the
    turn paused on (the `RequestInput` call's args), if it paused on one."""

    state: dict[str, Any]
    steps: dict[str, dict[str, Any]]
    fetched: list[str]
    visited: list[str]
    outputs: dict[str, dict[str, Any]]
    planner_started: dict[str, Any] | None
    asked: dict[str, Any] | None


_RANK = {"verb": "rank", "phrase": "top 5 donors", "over": "origin"}


async def _run(
    blocks: Sequence[Mapping[str, Any]],
    surface: DataMcpDefinitions,
    *,
    operations: Sequence[Mapping[str, Any]] = (_RANK,),
    planner: Any = None,
    resolver: Any = None,
    surface_name: str = "studio",
) -> _Turn:
    """Run one seeded turn on `surface_name` from `blocks`, each with `operations`,
    until it pauses or closes. `resolver` answers the question's own search."""
    steps = _Steps()
    workflow = create_root_workflow(
        resolver=resolver or ScriptedResolver(),
        executor=steps,
        generative=GenerativeNodes(planner=planner or ScriptedPlannerModel()),
        tool_surface=None,
        definition_surface=surface,
    )
    seeds = [{"spec": block, "operations": list(operations)} for block in blocks]
    runner = InMemoryRunner(node=workflow)
    session = await runner.session_service.create_session(
        app_name=runner.app_name,
        user_id="u",
        state={SEEDS_STATE_KEY: json.dumps(seeds), "surface": surface_name},
    )
    events = [
        event
        async for event in runner.run_async(
            user_id="u",
            session_id=session.id,
            new_message=types.Content(
                role="user", parts=[types.Part.from_text(text="top 5 donors of education aid")]
            ),
        )
    ]
    held = await runner.session_service.get_session(
        app_name=runner.app_name, user_id="u", session_id=session.id
    )
    assert held is not None
    visited: list[str] = []
    outputs: dict[str, dict[str, Any]] = {}
    planner_started = asked = None
    for event in events:
        node = (event.node_info.path if event.node_info else "").split("/")[-1].split("@")[0]
        visited.append(node)
        if isinstance(event.output, dict):
            outputs[node] = event.output
        if event.custom_metadata and PLANNER_STARTED_KEY in event.custom_metadata:
            planner_started = event.custom_metadata[PLANNER_STARTED_KEY]
        for call in event.get_function_calls():
            if call.name == REQUEST_INPUT_FUNCTION_CALL_NAME:
                asked = dict(call.args or {})
    return _Turn(
        dict(held.state), steps.steps, steps.fetched, visited, outputs, planner_started, asked
    )


def _kenya_evidence(approved: Any) -> RetrievalEvidence:
    """What a retrieval of the seed returns: Kenya's one series, as the manifest names it.

    The served members are three of the family's CRS purpose codes; the family's
    digest is the one Search states."""
    served = tuple(
        f"Amount_FinancialFlow_Gross_Disbursement_CRSTypeOfFlow10_CRSTypeOfFinance110_"
        f"CRSPurposeCode{code}_Nominal"
        for code in (11110, 11120, 11130)
    )
    stats = series_statistics_table(
        [
            SeriesStatisticsModel.model_validate(
                {
                    "stat_var": f"dcid:{served[0]}",
                    "entities": [{"role": "destination", "ref": "country/KEN"}],
                    "facet": {"unit": "USD"},
                    "observations": 1,
                    "first": {"value": 1.0, "period": "2024"},
                    "latest": {"value": 1.0, "period": "2024"},
                    "minimum": {"value": 1.0, "period": "2024"},
                    "maximum": {"value": 1.0, "period": "2024"},
                }
            )
        ]
    )
    return RetrievalEvidence(
        origin="manifest",
        spec_id=approved.spec_id,
        has_data=True,
        stat_vars=(served[0],),
        resolved_stat_vars=served,
        family_member_digest=approved.shape_member_digest,
        family_member_count=approved.shape_member_count,
        entities=(ResolvedEntity(role="destination", ref="country/KEN", label="country/KEN"),),
        axis_by_column={},
        window=Period(start=2024, end=2024),
        rows=1,
        series=stats,
        identity=approved.identity,
    )


async def test_a_label_the_browser_changed_is_never_read_and_search_names_the_entity(
    definition_store: InMemoryDefinitionStore,
) -> None:
    """The browser relabels Kenya's identity "Uganda". Search's answer for the identity
    names Kenya, so the source the plan pins does, nothing in session state says Uganda,
    and a requirement for Uganda fails against Kenya's evidence."""
    del definition_store
    surface = DataMcpDefinitions(served=[_KENYA])

    turn = await _run([_forged_kenya()], surface)
    state, steps = turn.state, turn.steps

    assert steps["seeds"]["outcome"] == "ok"
    (seeded,) = seeded_sources(state)
    assert seeded.source.entities == (
        ResolvedEntity(role="destination", ref="country/KEN", label="Kenya"),
    )
    assert all(axis.summable is False for axis in seeded.source.open_axes)
    assert seeded.seen_year == 2024
    assert "Uganda" not in json.dumps(state)
    (planned,) = json.loads(state["resolve_envelope"])["seeds"]
    assert planned["sentence"] == _KENYA["resolved"]["rendered_sentence"]
    assert (planned["showed"], planned["coverage"]) == (
        "2024, 242 rows",
        "1 sources, 282 observations, 113 variables",
    )
    (posted,) = surface.search.posted
    (entry,) = posted["input"]["entries"]
    # The resubmit posts the identity alone; Search derives the population from it.
    assert entry == {
        "entry_kind": "identity",
        "variable_text": _KENYA["spec_id"],
        "identity": _KENYA["definition"]["identity"],
    }

    uganda = DatasetRequirement(
        id="aid",
        decision_ids=("source",),
        measure="education aid",
        entities=(Entity(name="Uganda", role="recipient"),),
        period=Period(start=2024, end=2024),
        source_spec_id=seeded.source.spec_id,
    )
    issues = check_evidence(uganda, _kenya_evidence(seeded.source), approved=seeded.source)
    assert errors(issues)


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
    "refused": (
        None,
        "Search no longer resolves it (request_invalid)",
    ),
}


@pytest.mark.parametrize("answer", sorted(_STALE_ANSWERS))
async def test_a_seed_search_no_longer_resolves_as_it_did_is_dropped_and_the_other_plans(
    definition_store: InMemoryDefinitionStore, answer: str
) -> None:
    """Search answers Kenya's identity with another spec_id, with no data, with a block
    that carries no definition, or with a refusal. The seed is dropped with the reason,
    and the turn plans from the other."""
    del definition_store
    envelope, reason = _STALE_ANSWERS[answer]
    other = _block("water")
    search = SearchStub(
        [other], answer={_KENYA["spec_id"]: envelope} if envelope is not None else None
    )

    turn = await _run([_KENYA, other], DataMcpDefinitions(search=search))

    refusal = turn.steps["seed-0"]["refusal"]
    assert refusal["code"] == "seed_dropped"
    assert refusal["retry_mode"] == "do_not_retry"
    assert refusal["message"] == f"seed 0 is left out: {reason}. Search for it again to add it."
    assert turn.steps["seeds"]["outcome"] == "ok"
    assert [item.source.spec_id for item in seeded_sources(turn.state)] == [other["spec_id"]]
    assert turn.state["parts"] == 1
    assert "probe" not in turn.visited
    # The drop a retrieval makes: the dataset is recorded as dropped, so no plan
    # in the turn takes it, and the plan card opens with why it was left out.
    assert [item.source.spec_id for item in dropped_seeds(turn.state)] == [_KENYA["spec_id"]]
    assert names_dropped_seed(turn.state, spec_id=None, identity=_KENYA["definition"]["identity"])
    assert turn.outputs["planner"]["refusal_note"] == (
        f'This plan leaves out "dataset 1 from Search": {reason}. Search for it again to add it.'
    )
    assert turn.state["dropped_untold"] is None


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

    turn = await _run(blocks, DataMcpDefinitions(search=search))

    assert [item.source.spec_id for item in seeded_sources(turn.state)] == [
        block["spec_id"] for block in blocks
    ]
    assert search.peak == len(blocks)
    added = max(end for _, end in search.spans) - min(start for start, _ in search.spans)
    assert max(delay_s.values()) <= added < 0.7 * sum(delay_s.values())


class _Refuses:
    """A planner that drafts no plan, with its reason."""

    async def __call__(self, **_: Any) -> dict[str, Any]:
        return {"refused": {"text": "I could not draft a plan."}}


_NO_DATA = {"status": "no_data", "parts": [{"status": "no_data", "no_data": {"reason": "gone"}}]}


async def test_a_seed_dropped_as_it_was_taken_is_named_when_no_plan_is_drafted(
    definition_store: InMemoryDefinitionStore,
) -> None:
    """The planner drafts nothing from the seed left, so the closing text names the
    dropped seed and why, then the seed left unplanned."""
    del definition_store
    other = _block("water")
    search = SearchStub([other], answer={_KENYA["spec_id"]: _NO_DATA})

    turn = await _run([_KENYA, other], DataMcpDefinitions(search=search), planner=_Refuses())

    assert turn.outputs["deadend"]["text"] == (
        '"dataset 1 from Search" could not be retrieved: Search finds no data for it (gone). '
        'Search for it again. No plan was drafted for "the reading water", so nothing is '
        "shown for it. I could not draft a plan."
    )


async def test_a_seed_dropped_as_it_was_taken_is_named_when_the_turn_fetches_without_a_plan(
    definition_store: InMemoryDefinitionStore,
) -> None:
    """The seed left is one series with no operation, so the turn fetches it with no
    plan card, and the closing text opens with the dropped seed and why."""
    del definition_store
    other = _block("water")
    search = SearchStub([other], answer={_KENYA["spec_id"]: _NO_DATA})

    turn = await _run([_KENYA, other], DataMcpDefinitions(search=search), operations=())

    assert "planner" not in turn.visited
    assert turn.outputs["narrate"]["text"].startswith(
        '"dataset 1 from Search" could not be retrieved: Search finds no data for it (gone). '
        "Search for it again. Prepared 1 frame(s)"
    )
    assert turn.state["dropped_untold"] is None


# What the turn says of Kenya's seed once Search refuses its identity
# (`SearchStub` answers an identity it serves no block for with `request_invalid`).
_KENYA_DROPPED = (
    '"dataset 1 from Search" could not be retrieved: Search no longer resolves it '
    "(request_invalid). Search for it again."
)


def _definite(block: Mapping[str, Any]) -> dict[str, Any]:
    """The question's own search, resolved to `block` as `harness.mcp_client.McpResolver` reads it."""
    return {
        "status": "definite",
        "spec_id": block["spec_id"],
        "choice_points": 0,
        "parts": 1,
        "warnings": [],
        "candidates": [],
        "wide": False,
        "spec_block": block,
    }


def _offering(*blocks: Mapping[str, Any]) -> dict[str, Any]:
    """The question's own search, offering a candidate for each of `blocks`."""
    return {
        "status": "candidates",
        "spec_id": None,
        "choice_points": len(blocks),
        "parts": 1,
        "warnings": [],
        "candidates": [
            {"spec_id": block["spec_id"], "label": block["resolved"]["rendered_sentence"]}
            for block in blocks
        ],
    }


class _Answers:
    """A resolver that answers the question's search with `envelope`."""

    def __init__(self, envelope: Mapping[str, Any]) -> None:
        self._envelope = envelope

    async def __call__(self, **_: Any) -> dict[str, Any]:
        return dict(self._envelope)


_SEARCH_FINDS_THE_DROPPED_DATASET = {
    "the same spec_id": _definite(_KENYA),
    "the same dataset over other years": _definite(_elsewhere()),
    "a candidate for it alone": _offering(_KENYA),
}


@pytest.mark.parametrize("surface_name", ["studio", "download"])
@pytest.mark.parametrize("found", sorted(_SEARCH_FINDS_THE_DROPPED_DATASET))
async def test_a_dropped_seed_the_questions_search_finds_again_is_never_retrieved(
    definition_store: InMemoryDefinitionStore, found: str, surface_name: str
) -> None:
    """Search refuses the only seed, so the turn falls back to the question, whose
    search finds the dataset again. The turn retrieves nothing and closes naming the
    drop and why nothing was retrieved; the receive step counts no seed kept."""
    del definition_store
    resolver = _Answers(_SEARCH_FINDS_THE_DROPPED_DATASET[found])

    turn = await _run(
        [_KENYA],
        DataMcpDefinitions(search=SearchStub()),
        operations=(),
        resolver=resolver,
        surface_name=surface_name,
    )

    assert turn.steps["seed-0"]["refusal"]["code"] == "seed_dropped"
    assert turn.steps["seeds"] == {
        "outcome": "ok",
        "frame_refs": [],
        "summary": "received 0 of 1 dataset(s) from Search",
    }
    assert turn.fetched == []
    assert "fetch" not in turn.visited
    assert turn.outputs["deadend"]["text"] == (
        f"{_KENYA_DROPPED} The question resolves to a dataset this turn left out, so "
        "nothing was retrieved."
    )
    assert turn.state["dropped_untold"] is None


async def test_a_clarifying_question_after_a_drop_names_it_and_offers_only_the_others(
    definition_store: InMemoryDefinitionStore,
) -> None:
    """Every seed is dropped and the question's search offers the dropped dataset and
    another. The turn asks which reading was meant, opening with the drop, and offers
    the other alone."""
    del definition_store
    other = _block("water")

    turn = await _run(
        [_KENYA],
        DataMcpDefinitions(search=SearchStub()),
        operations=(),
        resolver=_Answers(_offering(_KENYA, other)),
    )

    assert turn.visited[-1] == "clarify"
    assert turn.asked is not None
    assert turn.asked["message"] == f"{_KENYA_DROPPED} Which reading did you mean?"
    assert [c["spec_id"] for c in turn.asked["payload"]["candidates"]] == [other["spec_id"]]
    assert turn.state["dropped_untold"] is None


class _Asks:
    """A planner that pauses on a question to the user, as `harness.planner_seam.PlannerSeam`
    does when the planner calls `adk_request_input`."""

    async def __call__(self, **_: Any) -> dict[str, Any]:
        raise NodeInterruptedError()


async def test_a_question_the_planner_asks_after_a_drop_opens_with_it(
    definition_store: InMemoryDefinitionStore,
) -> None:
    """One seed is dropped and the planner, drafting from the other, asks the user
    something. The planner node hands Studio the drop to open the question with
    (`harness.studio_translate`), and counts it told, so the plan card drafted after
    the answer does not name it again."""
    del definition_store
    other = _block("water")
    search = SearchStub([other], answer={_KENYA["spec_id"]: _NO_DATA})

    turn = await _run([_KENYA, other], DataMcpDefinitions(search=search), planner=_Asks())

    assert turn.planner_started == {
        "revising": False,
        "untold_drops": (
            '"dataset 1 from Search" could not be retrieved: Search finds no data for it '
            "(gone). Search for it again."
        ),
    }
    assert turn.state["dropped_untold"] is None


class _Raises:
    """A planner seam that fails, as one does on a timeout or a bug."""

    async def __call__(self, **_: Any) -> dict[str, Any]:
        raise RuntimeError("the planner seam timed out")


async def test_a_planner_that_fails_after_a_seed_is_dropped_fails_the_turn(
    definition_store: InMemoryDefinitionStore,
) -> None:
    """One seed is dropped as it is taken and the first draft from the other raises.
    The turn fails with the planner's error; it does not close as a plan the planner
    chose not to draft."""
    del definition_store
    other = _block("water")
    search = SearchStub([other], answer={_KENYA["spec_id"]: _NO_DATA})

    with pytest.raises(RuntimeError, match="the planner seam timed out"):
        await _run([_KENYA, other], DataMcpDefinitions(search=search), planner=_Raises())


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

"""`harness.recipe_replay`: a stored recipe computed again on newer data.

The verbs are analysis-mcp's own functions behind a double that counts each call,
so the frames, traces and pairs are the ones a deployment stores. The mint is
`Dataset`, which registers a dataset's current frame as data-mcp's mint does.
apis is `FakeApis` behind `httpx2.MockTransport`, and the publisher is
presentation-mcp's `PlatformAssembler.publish_frame` over it. The mounted
services and data-mcp's real mint are in `test_recipe_replay_integration.py`.
"""

from __future__ import annotations

import copy
import subprocess
import sys
from collections.abc import AsyncIterator, Mapping, Sequence
from types import SimpleNamespace
from typing import Any

import pytest
from analysis_mcp.execution_identity import current_execution_identity
from analysis_mcp.filter_tool import filter_verb
from analysis_mcp.growth_tool import growth
from analysis_mcp.join_tool import join
from analysis_mcp.rolling_tool import resolve_anchor
from common.live_apis import DatasetVersion
from common.materialisation import (
    FrameRef,
    InMemoryMaterialisationStore,
    MaterialisationTrace,
    MintRecord,
    set_store,
)
from common.recipe import build_recipe, replay_key
from common.row_budget import RowBudget
from data_mcp.deadline import CallDeadline
from data_mcp.frame_materialisation import frame_ref_for, replayed_mint_options
from data_mcp.get_data_tool import RootMint
from data_mcp.mint_recovery import recover_mint_record
from data_mcp.operations import QueuedOperation
from dfo_frame_grammar import Frame
from fastmcp.tools import ToolResult
from harness.recipe_replay import Pending, Refused, Replayed, replay_recipe

from .replay_support import (
    FIRST,
    FIRST_VALUES,
    OPTIONS,
    REPORTERS,
    SECOND,
    SECOND_VALUES,
    SPEC,
    FakeApis,
    pair,
    problem,
    reporter_frame,
    reporter_rows,
    yearly,
)

_TOOLS = {"filter": filter_verb, "growth": growth, "join": join, "resolve_anchor": resolve_anchor}
# A second dataset, minted from the same frame as `SPEC`.
_BESIDE = "spec_" + "b" * 32
# Keeps the last two years to the latest.
_SINCE = {
    "kind": "ordered_threshold",
    "column": "time_period",
    "op": "gte",
    "rolling": {"anchor": "latest", "offset": -1, "rule": "half1"},
}
_KEPT = ["dcid:FRA", "dcid:GBR"]
_GROUPBY = {"by": ["reporter"], "op": "sum"}
# A definition's identity naming a flow's origin.
_FLOW_IDENTITY = {"entities": {"origin": ["country/FRA"]}}
_BUSY = SimpleNamespace(
    is_error=True,
    structured_content={
        "code": "service_unavailable",
        "retry_mode": "retry_after_delay",
        "retry_after_s": 4.0,
        "message": "busy",
    },
)


class Tools:
    """The replayer's tool seam over analysis-mcp's functions, recording each call.
    `answers` replaces a tool's answer, and `after` runs once a call has returned."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.answers: dict[str, Any] = {}
        self.after: Any = None

    async def __call__(self, *, name: str, arguments: Mapping[str, object]) -> Any:
        self.calls.append((name, copy.deepcopy(dict(arguments))))
        answer = self.answers.get(name) or await _TOOLS[name](**arguments)
        if self.after is not None:
            self.after(name)
        return answer

    @property
    def names(self) -> list[str]:
        return [name for name, _ in self.calls]

    def arguments(self, name: str) -> dict[str, Any]:
        (arguments,) = [arguments for called, arguments in self.calls if called == name]
        return arguments


class Dataset:
    """data-mcp's root mint for one dataset: registers the current frame under the
    current version's pair, or a groupby mint's under the pair `frame_ref_for` names
    its groupby by, and writes its mint record. `answers` are returned first, one for each call. `others`
    holds the frame of any other dataset, by spec id. With `names_groupby` off, a
    groupby mint registers under the version's own pair, as data-mcp did before
    `frame_ref_for` named a groupby. With an `enrichment_stamp`, a mint also names its
    columns and the stamp in its pair and keeps the stamp in its record, as data-mcp does
    for a flow root minted with kinds."""

    def __init__(self, store: InMemoryMaterialisationStore) -> None:
        self.names_groupby = True
        self.enrichment_stamp: str | None = None
        self._store = store
        self.content_version = FIRST
        self.frame: Frame = reporter_frame(FIRST_VALUES)
        self.minted: list[MintRecord] = []
        self.answers: list[QueuedOperation | ToolResult] = []
        self.others: dict[str, Frame] = {}

    async def __call__(
        self, record: MintRecord, *, deadline: CallDeadline
    ) -> RootMint | QueuedOperation | ToolResult:
        del deadline
        self.minted.append(record)
        if self.answers:
            return self.answers.pop(0)
        dataset = FrameRef(spec_id=record.dataset.spec_id, content_version=self.content_version)
        ref = frame_ref_for(
            spec_id=dataset.spec_id,
            content_version=dataset.content_version,
            columns=(
                tuple(record.options.get("columns") or ())
                if self.enrichment_stamp is not None
                else ()
            ),
            groupby=record.options.get("groupby") if self.names_groupby else None,
            enrichment_stamp=self.enrichment_stamp,
        )
        trace = MaterialisationTrace(
            upstream_trace={
                "graph_revision": "rev_test",
                "source_revisions": [{"source": "test", "revision": ref.content_version}],
                "facets": [],
                "conversion": None,
            },
            spec_id=ref.spec_id,
            content_version=ref.content_version,
        )
        frame = self.others.get(ref.spec_id, self.frame)
        await self._store.put(ref, frame=frame, trace=trace, writer="test")
        count = len(set(frame.table.column("stat_var").to_pylist()))
        await self._store.put_mint_record(
            MintRecord(
                frame=ref,
                dataset=dataset,
                options=record.options,
                scope=record.scope,
                stat_var_count=count,
                enrichment_stamp=self.enrichment_stamp,
            )
        )
        return RootMint(frame=ref, dataset=dataset, options=record.options, stat_var_count=count)

    def move(self, values: Mapping[str, Mapping[str, float]], *, to: str = SECOND) -> None:
        """Serve `values` as the dataset's next version."""
        self.content_version = to
        self.frame = reporter_frame(values)


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
async def store() -> AsyncIterator[InMemoryMaterialisationStore]:
    held = InMemoryMaterialisationStore()
    set_store(held)
    yield held
    set_store(None)


def _ref(result: ToolResult) -> FrameRef:
    assert result.is_error is False, result.content
    content = result.structured_content
    assert content is not None
    return FrameRef(spec_id=content["spec_id"], content_version=content["content_version"])


class Scene:
    """One dataset, the tools, apis, and a recipe published from the first version."""

    def __init__(self, store: InMemoryMaterialisationStore) -> None:
        self.store = store
        self.dataset = Dataset(store)
        self.tools = Tools()
        self.apis = FakeApis()
        self.assembler = self.apis.assembler()
        self.budget = RowBudget(max_rows=100_000)

    async def root(self, options: Mapping[str, Any] = OPTIONS, *, spec_id: str = SPEC) -> FrameRef:
        placeholder = FrameRef(spec_id=spec_id, content_version=FIRST)
        minted = await self.dataset(
            MintRecord(frame=placeholder, dataset=placeholder, options=dict(options)),
            deadline=CallDeadline.after(60.0),
        )
        assert isinstance(minted, RootMint)
        self.dataset.minted.clear()
        return minted.frame

    async def fixed(self, options: Mapping[str, Any] = OPTIONS) -> tuple[FrameRef, dict[str, Any]]:
        """A filter to two reporters, then their growth from 2019 to 2022, published."""
        kept = _ref(
            await filter_verb(
                frame_ref=(await self.root(options)).model_dump(),
                predicates=[{"kind": "category_match", "column": "reporter", "values": _KEPT}],
            )
        )
        return await self._published(
            _ref(
                await growth(
                    frame_ref=kept.model_dump(),
                    window_start="2019",
                    window_end="2022",
                    entity_by=("reporter",),
                )
            )
        )

    async def rolling(self) -> tuple[FrameRef, dict[str, Any]]:
        """The growth of every reporter over the three years to the latest, published."""
        root = await self.root()
        populations = [{"spec_id": SPEC, "bindings": {}}]
        anchor = await resolve_anchor(populations=populations, roots=[root.model_dump()])
        assert anchor.structured_content is not None
        assert anchor.structured_content["period"] == "2022"
        return await self._published(
            _ref(
                await growth(
                    frame_ref=root.model_dump(),
                    window_start="2019",
                    window_end="2022",
                    entity_by=("reporter",),
                    rolling={
                        "window_start": {"anchor": "latest", "offset": -3, "rule": "half1"},
                        "window_end": {"anchor": "latest", "offset": 0, "rule": "half1"},
                        "population": populations,
                    },
                    anchor_key=anchor.structured_content["anchor_key"],
                )
            )
        )

    async def since(self, root: FrameRef, *, spec_ids: Sequence[str]) -> FrameRef:
        """The rows of `root` from the year before the latest that the datasets
        `spec_ids` hold, as a rolling filter."""
        populations = [{"spec_id": spec_id, "bindings": {}} for spec_id in spec_ids]
        roots = [(await self.root(spec_id=spec_id)).model_dump() for spec_id in spec_ids]
        anchor = (await resolve_anchor(populations=populations, roots=roots)).structured_content
        assert anchor is not None
        return _ref(
            await filter_verb(
                frame_ref=root.model_dump(),
                predicates=[{**_SINCE, "value": str(int(anchor["period"]) - 1)}],
                rolling={"population": populations},
                anchor_key=anchor["anchor_key"],
            )
        )

    async def joined(self, left: FrameRef, right: FrameRef) -> FrameRef:
        return _ref(await join(frame_ref=left.model_dump(), right_frame_ref=right.model_dump()))

    async def rolling_beside_fixed(self) -> tuple[FrameRef, dict[str, Any]]:
        """A rolling filter on `SPEC` joined with the whole of `_BESIDE`, published."""
        self.dataset.others[_BESIDE] = Frame.build(
            role_keys=["reporter"], rows=reporter_rows(FIRST_VALUES, stat_var="dcid:Population")
        )
        kept = await self.since(await self.root(), spec_ids=[SPEC])
        beside = await self.root(spec_id=_BESIDE)
        return await self._published(await self.joined(kept, beside))

    async def _published(self, ref: FrameRef) -> tuple[FrameRef, dict[str, Any]]:
        recipe = await build_recipe(ref, store=self.store)
        published = await self.assembler.publish_frame(
            spec_id=ref.spec_id, content_version=ref.content_version, recipe=recipe
        )
        assert published is True
        return ref, recipe

    async def replay(
        self, recipe: Mapping[str, Any], **arguments: Any
    ) -> Replayed | Pending | Refused:
        async with self.apis.client() as client:
            return await replay_recipe(
                recipe,
                **{
                    "deadline": CallDeadline.after(60.0),
                    "call_tool": self.tools,
                    "store": self.store,
                    "apis": client,
                    "publish": self.assembler.publish_frame,
                    "row_budget": self.budget,
                    "mint_root": self.dataset,
                    **arguments,
                },
            )

    def growth_of(self, frame: FrameRef) -> dict[str, float]:
        """Each reporter's value in the rows apis holds for `frame`."""
        return {row["reporter"]: row["value"] for row in self.apis.frames[pair(frame)]}


def _minting_kinds(dataset: Dataset) -> Any:
    """`dataset`'s mint behind data-mcp's mapping of recorded options for a flow
    root, as `mint_recorded_root` applies it."""

    async def mint(
        record: MintRecord, *, deadline: CallDeadline
    ) -> RootMint | QueuedOperation | ToolResult:
        options = replayed_mint_options(record.options, identity=_FLOW_IDENTITY)
        return await dataset(record.model_copy(update={"options": options}), deadline=deadline)

    return mint


class _Unrecorded:
    """`store` as it reads for a root registered before mint records: it holds no
    mint record."""

    def __init__(self, store: InMemoryMaterialisationStore) -> None:
        self._store = store

    def __getattr__(self, name: str) -> Any:
        return getattr(self._store, name)

    async def get_mint_record(self, ref: FrameRef) -> MintRecord | None:
        del ref
        return None


@pytest.fixture
async def scene(store: InMemoryMaterialisationStore) -> AsyncIterator[Scene]:
    made = Scene(store)
    yield made
    await made.assembler.aclose()


async def test_a_replay_on_moved_data_publishes_the_new_frame_under_the_same_normal_form(
    scene: Scene,
) -> None:
    original, recipe = await scene.fixed()
    scene.dataset.move(SECOND_VALUES)

    replayed = await scene.replay(recipe)

    assert isinstance(replayed, Replayed), replayed
    assert replayed.frame != original
    assert replayed.newly_published is True
    assert replayed.normal_form_changed is False
    assert replayed.recipe["normal_form_hash"] == recipe["normal_form_hash"]
    assert replayed.recipe["replay_key"] != recipe["replay_key"]
    assert replayed.recipe["roots"][0]["dataset"] == {"spec_id": SPEC, "content_version": SECOND}
    assert scene.apis.recipes[pair(replayed.frame)] == replayed.recipe
    # 2019 to 2022 is 100 to 130 in the first version and 150 to 180 in the second.
    assert scene.growth_of(original) == pytest.approx({"dcid:FRA": 0.3, "dcid:GBR": 0.15})
    assert scene.growth_of(replayed.frame) == pytest.approx({"dcid:FRA": 0.2, "dcid:GBR": 0.1})
    # The root is minted as recorded, and a tree of fixed periods resolves no anchor.
    assert scene.dataset.minted == [MintRecord.model_validate(recipe["roots"][0])]
    assert scene.tools.names == ["filter", "growth"]


@pytest.mark.parametrize(
    "recorded",
    [
        OPTIONS | {"columns": ["financingProvider_type", "financingRecipient_type"]},
        OPTIONS | {"columns": ["kind"]},
        OPTIONS,
    ],
    ids=["flow-party-types", "flow-retired-kind", "flow-no-columns"],
)
async def test_a_flow_root_minted_with_kinds_keeps_the_stored_normal_form(
    scene: Scene, recorded: dict[str, Any]
) -> None:
    _, recipe = await scene.fixed(recorded)
    scene.dataset.move(SECOND_VALUES)

    replayed = await scene.replay(recipe, mint_root=_minting_kinds(scene.dataset))

    assert isinstance(replayed, Replayed), replayed
    (minted,) = scene.dataset.minted
    assert minted.options == OPTIONS | {"columns": ["party_kind"]}
    current = FrameRef(spec_id=SPEC, content_version=SECOND)
    held = await scene.store.get_mint_record(current)
    assert held is not None
    assert held.options == minted.options
    assert replayed.normal_form_changed is False
    assert replayed.recipe["normal_form_hash"] == recipe["normal_form_hash"]
    assert replayed.recipe["roots"][0]["options"] == recorded
    # The key a live refresh looks the replay up by, from the stored recipe.
    assert replayed.recipe["replay_key"] == replay_key(
        normal_form_hash=recipe["normal_form_hash"],
        root_datasets=[current.model_dump()],
        execution_identities=[current_execution_identity().canonical_string()],
    )


@pytest.mark.parametrize(
    "recorded",
    [
        OPTIONS | {"without_party_totals": True},
        OPTIONS | {"groupby": _GROUPBY, "without_party_totals": True},
    ],
    ids=["plain", "groupby"],
)
async def test_a_root_minted_without_party_totals_is_minted_with_the_option_again(
    scene: Scene, recorded: dict[str, Any]
) -> None:
    """The option is apis', sent as recorded and stated again in the replayed recipe,
    whose normal form stays."""
    original, recipe = await scene.fixed(recorded)
    scene.dataset.move(SECOND_VALUES)

    replayed = await scene.replay(recipe)

    assert isinstance(replayed, Replayed), replayed
    (minted,) = scene.dataset.minted
    assert minted.options == recorded
    (root,) = replayed.recipe["roots"]
    assert (
        root["frame"]
        == frame_ref_for(
            spec_id=SPEC, content_version=SECOND, groupby=recorded.get("groupby")
        ).model_dump()
    )
    assert root["options"] == recorded
    assert replayed.normal_form_changed is False
    assert replayed.recipe["normal_form_hash"] == recipe["normal_form_hash"]
    assert replayed.frame != original


async def test_a_second_replay_calls_no_verb_and_publishes_nothing(scene: Scene) -> None:
    _, recipe = await scene.fixed()
    scene.dataset.move(SECOND_VALUES)
    first = await scene.replay(recipe)
    calls, posts = len(scene.tools.calls), len(scene.apis.published())

    second = await scene.replay(recipe)

    assert isinstance(first, Replayed)
    assert isinstance(second, Replayed)
    assert (second.frame, second.recipe) == (first.frame, first.recipe)
    assert second.newly_published is False
    assert len(scene.tools.calls) == calls
    assert len(scene.apis.published()) == posts
    # Every root is minted on every replay, moved or not.
    assert len(scene.dataset.minted) == 2


async def test_a_replay_of_unmoved_data_names_the_frame_already_published(scene: Scene) -> None:
    original, recipe = await scene.fixed()
    posts = len(scene.apis.published())

    replayed = await scene.replay(recipe)

    assert isinstance(replayed, Replayed)
    assert replayed.frame == original
    assert replayed.newly_published is False
    assert scene.tools.calls == []
    assert len(scene.apis.published()) == posts


async def test_a_replay_past_its_deadline_is_pending_and_the_next_call_skips_what_finished(
    scene: Scene,
) -> None:
    _, recipe = await scene.fixed()
    scene.dataset.move(SECOND_VALUES)
    posts = len(scene.apis.published())
    clock = Clock()

    def run_out(name: str) -> None:
        clock.now = 100.0

    scene.tools.after = run_out
    stopped = await scene.replay(recipe, deadline=CallDeadline(expires_at=50.0, clock=clock))

    assert stopped == Pending(retry_after_s=1.0, reason="deadline")
    assert scene.tools.names == ["filter"]
    assert len(scene.apis.published()) == posts

    scene.tools.after = None
    resumed = await scene.replay(recipe, deadline=CallDeadline(expires_at=500.0, clock=clock))

    assert isinstance(resumed, Replayed)
    # The filter finished before the deadline and is not called again.
    assert scene.tools.names == ["filter", "growth"]


async def test_a_queued_root_mint_is_pending_and_no_verb_runs(scene: Scene) -> None:
    _, recipe = await scene.fixed()
    scene.dataset.move(SECOND_VALUES)
    scene.dataset.answers = [
        QueuedOperation(operation_id="op_" + "0" * 64, status="pending", retry_after_s=1.5)
    ]

    queued = await scene.replay(recipe)
    done = await scene.replay(recipe)

    assert queued == Pending(retry_after_s=1.5, reason="mint_queued")
    assert isinstance(done, Replayed)
    # Both calls sent the recorded mint, and the verbs ran once, on the second.
    assert scene.dataset.minted == [MintRecord.model_validate(recipe["roots"][0])] * 2
    assert scene.tools.names == ["filter", "growth"]


async def test_a_verbs_refusal_on_current_data_is_returned_under_its_code(scene: Scene) -> None:
    _, recipe = await scene.fixed()
    # No reporter holds 2019 any more, so the recorded window has no start.
    scene.dataset.move({reporter: yearly(range(2020, 2024), start=1.0) for reporter in REPORTERS})
    posts = len(scene.apis.published())

    refused = await scene.replay(recipe)

    assert isinstance(refused, Refused)
    assert (refused.code, refused.stage) == ("growth_gap_year", "verb")
    assert scene.tools.names == ["filter", "growth"]
    assert len(scene.apis.published()) == posts


async def test_a_refusal_that_names_a_wait_is_pending(scene: Scene) -> None:
    _, recipe = await scene.fixed()
    scene.dataset.move(SECOND_VALUES)
    scene.tools.answers["growth"] = _BUSY

    assert await scene.replay(recipe) == Pending(retry_after_s=4.0, reason="busy")


async def test_rows_the_refresh_budget_has_no_room_for_are_pending_and_no_verb_is_called(
    scene: Scene,
) -> None:
    _, recipe = await scene.fixed()
    scene.dataset.move(SECOND_VALUES)

    # Another replay holds the whole budget, and gives it back.
    with scene.budget.claim(scene.budget.max_rows):
        assert await scene.replay(recipe) == Pending(retry_after_s=2.0, reason="row_budget")
        assert scene.tools.calls == []
    assert isinstance(await scene.replay(recipe), Replayed)


@pytest.mark.parametrize(
    ("recipe_of", "stage"),
    [
        # A rolling recipe claims its roots' rows together for the anchor call.
        pytest.param(Scene.rolling, "anchor", id="roots-of-a-rolling-recipe"),
        pytest.param(Scene.fixed, "verb", id="inputs-of-a-verb"),
    ],
)
async def test_rows_past_the_whole_refresh_budget_are_refused_for_good(
    scene: Scene, recipe_of: Any, stage: str
) -> None:
    """No later call could claim them, so the replay must not answer `Pending`."""
    _, recipe = await recipe_of(scene)
    scene.dataset.move(SECOND_VALUES)
    scene.budget = RowBudget(max_rows=1)

    refused = await scene.replay(recipe)

    assert isinstance(refused, Refused), refused
    assert (refused.code, refused.stage) == ("rows_over_refresh_budget", stage)
    assert scene.tools.calls == []


async def test_a_dataset_joined_beside_a_rolling_step_is_not_read_for_its_period(
    scene: Scene,
) -> None:
    _, recipe = await scene.rolling_beside_fixed()
    scene.dataset.move(SECOND_VALUES)

    replayed = await scene.replay(recipe)

    assert isinstance(replayed, Replayed), replayed
    assert [root["spec_id"] for root in scene.tools.arguments("resolve_anchor")["roots"]] == [SPEC]
    assert scene.tools.names == ["resolve_anchor", "filter", "join"]


async def test_a_root_recorded_with_one_stat_var_that_now_holds_two_is_refused(
    scene: Scene,
) -> None:
    _, recipe = await scene.fixed()
    scene.dataset.content_version = SECOND
    scene.dataset.frame = Frame.build(
        role_keys=["reporter"],
        rows=reporter_rows(SECOND_VALUES) + reporter_rows(SECOND_VALUES, stat_var="dcid:Other"),
    )
    unknown = copy.deepcopy(recipe)
    unknown["roots"][0]["stat_var_count"] = None

    refused = await scene.replay(recipe)
    calls = len(scene.tools.calls)
    uncounted = await scene.replay(unknown)

    assert recipe["roots"][0]["stat_var_count"] == 1
    assert isinstance(refused, Refused)
    assert (refused.code, refused.stage) == ("root_rendering_ambiguous", "mint")
    assert calls == 0
    # A record with no count is replayed as recorded, so the verbs are reached.
    assert getattr(uncounted, "code", None) != "root_rendering_ambiguous"
    assert len(scene.tools.calls) > 0


@pytest.mark.parametrize("stamp_kept", [True, False], ids=["stamp-kept", "stamp-dropped"])
async def test_a_kind_column_sum_registered_under_a_salted_pair_is_recognised_as_the_sum(
    scene: Scene, stamp_kept: bool
) -> None:
    """Its pair names the groupby, the columns and the enrichment stamp, so the record
    must hand the stamp back to `frame_ref_for` to be seen as the sum. Without it the
    pair is not reproduced and the root reads as unsummed once a second StatVar
    contributes."""
    scene.dataset.enrichment_stamp = "stamp-a"
    options = OPTIONS | {"groupby": _GROUPBY, "columns": ["party_kind"]}
    _, recipe = await scene.fixed(options)
    scene.dataset.content_version = SECOND
    scene.dataset.frame = Frame.build(
        role_keys=["reporter"],
        rows=reporter_rows(SECOND_VALUES) + reporter_rows(SECOND_VALUES, stat_var="dcid:Other"),
    )
    (root,) = recipe["roots"]
    assert root["enrichment_stamp"] == "stamp-a"
    if not stamp_kept:
        del root["enrichment_stamp"]

    outcome = await scene.replay(recipe)

    # The fake frame holds two StatVars, so a verb refuses it later; the mint stage is
    # what this test reads.
    assert isinstance(outcome, Refused)
    assert (outcome.stage == "mint") is (not stamp_kept)
    assert (outcome.code == "root_rendering_ambiguous") is (not stamp_kept)


async def test_a_recovered_one_stat_var_sum_that_now_holds_two_is_refused(scene: Scene) -> None:
    """A `groupby` sum over one StatVar registered under the raw version's own pair,
    as data-mcp registered one before `frame_ref_for` named a groupby, cannot be told
    from the unsummed frame by recovery. It keeps the held frame's count, which the
    replay refuses on once a second StatVar contributes."""
    scene.dataset.names_groupby = False
    root = await scene.root(OPTIONS | {"groupby": _GROUPBY})
    kept = _ref(
        await filter_verb(
            frame_ref=root.model_dump(),
            predicates=[{"kind": "category_match", "column": "reporter", "values": _KEPT}],
        )
    )

    async def versions(*, spec_id: str) -> list[DatasetVersion]:
        assert spec_id == SPEC
        return [DatasetVersion(root.content_version, None, "plain", None)]

    async def recover(ref: FrameRef) -> MintRecord:
        return await recover_mint_record(ref, dataset_versions=versions, store=scene.store)

    recipe = await build_recipe(kept, store=_Unrecorded(scene.store), recover=recover)
    assert await scene.assembler.publish_frame(
        spec_id=kept.spec_id, content_version=kept.content_version, recipe=recipe
    )
    scene.dataset.content_version = SECOND
    scene.dataset.frame = Frame.build(
        role_keys=["reporter"],
        rows=reporter_rows(SECOND_VALUES) + reporter_rows(SECOND_VALUES, stat_var="dcid:Other"),
    )

    refused = await scene.replay(recipe)

    assert (recipe["roots"][0]["options"], recipe["roots"][0]["stat_var_count"]) == (OPTIONS, 1)
    assert isinstance(refused, Refused), refused
    assert (refused.code, refused.stage) == ("root_rendering_ambiguous", "mint")
    assert scene.tools.calls == []


@pytest.mark.parametrize(
    ("recorded_base", "names_groupby", "ambiguous"),
    [(None, False, True), ("cv_" + "9" * 32, False, False), (None, True, False)],
    ids=[
        "recorded-under-a-raw-version",
        "recorded-as-a-collapse",
        "recorded-under-the-pair-naming-its-groupby",
    ],
)
async def test_a_summed_root_that_comes_back_as_a_collapse_is_refused_unless_recorded_as_one(
    scene: Scene, recorded_base: str | None, names_groupby: bool, ambiguous: bool
) -> None:
    """A sum registered under its raw version's own pair shared it with the unsummed
    frame, so its record cannot say which it was. One registered under the pair
    naming its groupby was the sum, whatever version it comes back under."""
    scene.dataset.names_groupby = names_groupby
    _, recipe = await scene.fixed(OPTIONS | {"groupby": _GROUPBY})
    scene.dataset.move(SECOND_VALUES)
    scene.apis.versions[SPEC] = [
        {
            "content_version": SECOND,
            "base_content_version": "cv_" + "8" * 32,
            "rendering": "groupby",
            "groupby": _GROUPBY,
        },
        {
            "content_version": FIRST,
            "base_content_version": recorded_base,
            "rendering": "plain" if recorded_base is None else "groupby",
            "groupby": None if recorded_base is None else _GROUPBY,
        },
    ]

    outcome = await scene.replay(recipe)

    if ambiguous:
        assert isinstance(outcome, Refused)
        assert (outcome.code, outcome.stage) == ("root_rendering_ambiguous", "mint")
        assert scene.tools.calls == []
    else:
        assert isinstance(outcome, Replayed)


async def test_a_replay_whose_normal_form_moved_is_refused_and_nothing_is_published(
    scene: Scene,
) -> None:
    _, recipe = await scene.fixed()
    scene.dataset.move(SECOND_VALUES)
    # Another mint reached the new version first, with other options, and the
    # store keeps the first record written under a pair.
    current = FrameRef(spec_id=SPEC, content_version=SECOND)
    await scene.store.put_mint_record(
        MintRecord(frame=current, dataset=current, options=OPTIONS | {"groupby": _GROUPBY})
    )
    posts = len(scene.apis.published())

    refused = await scene.replay(recipe)

    assert isinstance(refused, Refused)
    assert (refused.code, refused.stage) == ("recipe_normal_form_changed", "publish")
    assert len(scene.apis.published()) == posts


async def test_a_frame_another_card_published_under_another_normal_form_is_refused(
    scene: Scene,
) -> None:
    _, recipe = await scene.fixed(
        OPTIONS | {"columns": ["financingProvider_type", "financingRecipient_type"]}
    )
    scene.dataset.move(SECOND_VALUES)

    async def unpublished(**_: Any) -> bool:
        return True

    # A card made on the new version with kinds computes the same frame, and
    # publishes it first under the recipe its own mint records give.
    computed = await scene.replay(
        recipe, mint_root=_minting_kinds(scene.dataset), publish=unpublished
    )
    assert isinstance(computed, Replayed), computed
    _, fresh = await scene._published(computed.frame)
    assert fresh["normal_form_hash"] != recipe["normal_form_hash"]

    refused = await scene.replay(recipe, mint_root=_minting_kinds(scene.dataset))

    assert isinstance(refused, Refused), refused
    assert (refused.code, refused.stage) == ("recipe_normal_form_changed", "publish")
    assert fresh["normal_form_hash"] in refused.detail
    assert recipe["normal_form_hash"] in refused.detail
    assert scene.apis.recipes[pair(computed.frame)] == fresh


async def test_a_publish_refusal_is_returned_under_the_doors_code(scene: Scene) -> None:
    _, recipe = await scene.fixed()
    scene.dataset.move(SECOND_VALUES)
    scene.apis.publish_answer = problem(400, "request_invalid")

    refused = await scene.replay(recipe)

    assert isinstance(refused, Refused)
    assert (refused.code, refused.stage) == ("request_invalid", "publish")


async def test_replacement_populations_go_on_every_rolling_call_and_change_the_normal_form(
    scene: Scene,
) -> None:
    _, recipe = await scene.rolling()
    scene.dataset.move(SECOND_VALUES)
    replacement = [{"spec_id": SPEC, "bindings": {"reporter": ["dcid:USA", "dcid:FRA"]}}]
    canonical = [{"spec_id": SPEC, "bindings": {"reporter": ["dcid:FRA", "dcid:USA"]}}]

    replayed = await scene.replay(recipe, populations=replacement)

    assert isinstance(replayed, Replayed), replayed
    assert replayed.normal_form_changed is True
    assert replayed.recipe["normal_form_hash"] != recipe["normal_form_hash"]
    assert replayed.recipe["anchor"]["period"] == "2023"
    anchor = scene.tools.arguments("resolve_anchor")
    assert anchor == {
        "populations": canonical,
        "roots": [{"spec_id": SPEC, "content_version": SECOND}],
    }
    called = scene.tools.arguments("growth")
    assert called["rolling"]["population"] == canonical
    assert (called["window_start"], called["window_end"]) == ("2020", "2023")
    assert called["anchor_key"].startswith("rak_")


async def test_replacement_populations_for_a_recipe_of_fixed_periods_are_refused(
    scene: Scene,
) -> None:
    _, recipe = await scene.fixed()

    refused = await scene.replay(recipe, populations=[{"spec_id": SPEC, "bindings": {}}])

    assert isinstance(refused, Refused)
    assert (refused.code, refused.stage) == ("populations_not_applicable", "anchor")
    assert scene.tools.calls == []


async def test_rolling_op_specs_that_carry_different_populations_are_refused(
    scene: Scene,
) -> None:
    _, recipe = await scene.rolling()
    disagreeing = copy.deepcopy(recipe)
    inner = copy.deepcopy(disagreeing["tree"])
    inner["frame"] = {"spec_id": "ops_" + "b" * 32, "content_version": "ocv_" + "b" * 32}
    inner["op_spec"]["parameters"]["rolling"]["population"] = [
        {"spec_id": SPEC, "bindings": {"reporter": ["dcid:FRA"]}}
    ]
    disagreeing["tree"]["inputs"]["frame_ref"] = inner

    refused = await scene.replay(disagreeing)

    assert isinstance(refused, Refused)
    assert (refused.code, refused.stage) == ("rolling_populations_differ", "anchor")
    assert scene.tools.calls == []


@pytest.mark.parametrize(
    "recipe",
    [
        {"schema_version": 2, "tree": {"op_spec": {}}, "roots": []},
        {"schema_version": 1, "tree": {"frame": {}, "root": 0}, "roots": []},
        {"schema_version": 1, "tree": {"op_spec": {}}, "roots": [{"frame": "nope"}]},
    ],
    ids=["another-schema-version", "a-root-frame", "a-root-that-is-no-mint-record"],
)
async def test_a_recipe_this_build_cannot_read_is_refused(
    scene: Scene, recipe: dict[str, Any]
) -> None:
    refused = await scene.replay(recipe)

    assert isinstance(refused, Refused)
    assert (refused.code, refused.stage) == ("recipe_invalid", "recipe")
    assert scene.dataset.minted == []


def test_the_replayer_imports_no_model_planner_or_session_code() -> None:
    """A refresh has no conversation: the module loads none of the planner, the
    compiler, the executor, the model runtime, or the rest of the harness package."""
    probe = (
        "import sys, harness.recipe_replay\n"
        "roots = ('planner', 'compiler', 'executor', 'translator', 'orchestrator', "
        "'google.adk', 'google.genai', 'sqlalchemy', 'asyncpg')\n"
        "loaded = sorted(m for m in sys.modules if m.split('.')[0] == 'harness' "
        "or any(m == r or m.startswith(r + '.') for r in roots))\n"
        "print(loaded)"
    )

    ran = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)

    assert ran.stdout.strip() == "['harness', 'harness.recipe_replay']"

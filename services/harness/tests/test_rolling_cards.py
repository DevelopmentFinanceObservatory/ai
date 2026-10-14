"""`harness.rolling_cards`: the rules every `write_component` passes, as
`RollingCards.admit` applies them, and the test for a year in a title.

The frames, traces, anchor records and replays are real: `Scene`
(`test_recipe_replay.py`) runs analysis-mcp's own functions over an in-memory
store, with dfo-apis as `FakeApis`. One reporter of three holds 2023, so the
latest year over all three is 2022 and the latest year over that reporter alone
is 2023.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any

import pytest
from analysis_mcp.growth_tool import growth
from analysis_mcp.rolling_tool import resolve_anchor
from common.materialisation import FrameRef, InMemoryMaterialisationStore, set_store
from common.recipe import RecipeAnchorError, build_recipe
from data_mcp.deadline import CallDeadline
from data_mcp.operations import QueuedOperation
from dfo_frame_grammar import Frame
from harness import rolling_cards
from harness.recipe_replay import RESUME_RETRY_AFTER_S, Pending, Refused, Replayed
from harness.rolling_cards import (
    WITHHELD_FOR_S,
    CardWithheld,
    RollingCards,
    card_selection,
    frame_periods,
    title_refusal,
    written_title_refusal,
)
from harness.tool_contracts import HarnessCode

from .replay_support import REPORTERS, SPEC, STAT_VAR, reporter_frame, reporter_rows, yearly
from .test_recipe_replay import Scene

FRANCE, BRITAIN, AMERICA = REPORTERS
# The growth frame's view: one row for each reporter.
VIEW = {
    "dimensions": [
        {"name": "stat_var", "plural": False, "values": [STAT_VAR], "total_count": 1},
        {"name": "reporter", "plural": True, "values": list(REPORTERS), "total_count": 3},
    ]
}
EVERY_SERIES = [{"spec_id": SPEC, "bindings": {}}]


@pytest.mark.parametrize(
    ("select", "bound", "selection"),
    [
        pytest.param(
            {"reporter": [BRITAIN, FRANCE], "stat_var": STAT_VAR},
            frozenset(),
            {"reporter": sorted([BRITAIN, FRANCE])},
            id="fewer-values-than-the-view-holds",
        ),
        pytest.param(
            {"reporter": list(REPORTERS), "stat_var": STAT_VAR},
            frozenset(),
            {},
            id="every-value-of-the-view",
        ),
        pytest.param(
            {"reporter": list(REPORTERS)},
            frozenset({"reporter"}),
            {"reporter": sorted(REPORTERS)},
            id="every-value-of-a-dimension-the-frame-binds",
        ),
        pytest.param(
            {"sector": "dcid:Health"},
            frozenset(),
            {"sector": ["dcid:Health"]},
            id="a-dimension-the-view-does-not-count",
        ),
    ],
)
def test_a_select_dimension_counts_when_the_frame_binds_it_or_it_narrows_the_view(
    select: dict[str, Any], bound: frozenset[str], selection: dict[str, list[str]]
) -> None:
    assert card_selection(select, view=VIEW, bound=bound) == selection


@asynccontextmanager
async def one_reporter_ahead() -> AsyncIterator[Scene]:
    """A scene over a store of its own whose dataset has France alone at 2023."""
    store = InMemoryMaterialisationStore()
    set_store(store)
    made = Scene(store)
    made.dataset.frame = reporter_frame(
        {
            FRANCE: yearly(range(2019, 2024), start=100.0),
            BRITAIN: yearly(range(2019, 2023), start=200.0),
            AMERICA: yearly(range(2019, 2023), start=300.0),
        }
    )
    try:
        yield made
    finally:
        await made.assembler.aclose()
        set_store(None)


@pytest.fixture
async def scene() -> AsyncIterator[Scene]:
    async with one_reporter_ahead() as made:
        yield made


class Replays:
    """`RollingCards`' replay seam over the scene, keeping each call's populations."""

    def __init__(self, scene: Scene) -> None:
        self._scene = scene
        self.populations: list[Sequence[Mapping[str, Any]]] = []

    async def __call__(
        self,
        recipe: Mapping[str, Any],
        *,
        deadline: CallDeadline,
        populations: Sequence[Mapping[str, Any]],
    ) -> Replayed | Pending | Refused:
        self.populations.append(populations)
        return await self._scene.replay(recipe, deadline=deadline, populations=populations)


class Platform:
    """What `admit` reads from the presentation service: every frame's view, and
    the stored intent of each component."""

    def __init__(self, stored: Mapping[str, Mapping[str, Any]] | None = None) -> None:
        self._stored = stored or {}
        self.stored_reads: list[str] = []
        self.requested_window: Mapping[str, Any] | None = None

    def window(self, frame_ref: Mapping[str, Any]) -> Mapping[str, Any] | None:
        del frame_ref
        return self.requested_window

    async def view(self, frame_ref: Mapping[str, Any]) -> Mapping[str, Any]:
        del frame_ref
        return VIEW

    async def stored(self, component_id: str) -> Mapping[str, Any] | None:
        self.stored_reads.append(component_id)
        return self._stored.get(component_id)


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


async def _admit(
    cards: RollingCards, intent: dict[str, Any], *, platform: Platform | None = None, **args: Any
) -> dict[str, Any]:
    """The intent `admit` sends for a write of `intent`."""
    platform = platform or Platform()
    admitted = await cards.admit(
        {"intent": intent, **args},
        read_view=platform.view,
        read_stored=platform.stored,
        read_window=platform.window,
    )
    return admitted.args["intent"]


def _card(frame_ref: Mapping[str, Any], **intent: Any) -> dict[str, Any]:
    return {
        "type": "bar_chart",
        "frame_ref": dict(frame_ref),
        "select": {"reporter": list(REPORTERS)},
        "title": "Growth by reporter",
        **intent,
    }


async def test_whether_a_frame_rolls_is_read_from_the_store_alone(scene: Scene) -> None:
    """Nothing of the turn that made a frame is needed to tell that it rolls, so
    a later turn editing its card reads the same answer."""
    rolling, recipe = await scene.rolling()
    fixed, _ = await scene.fixed()
    cards = RollingCards(replay=None)

    frame = await cards.frame(rolling.model_dump())

    assert frame is not None
    assert (frame.ref, frame.populations) == (rolling, EVERY_SERIES)
    assert frame.recipe["anchor"] == recipe["anchor"]
    assert frame.recipe["anchor"]["period"] == "2022"
    assert await cards.frame(fixed.model_dump()) is None
    assert await cards.frame(recipe["roots"][0]["frame"]) is None
    # A pair the store never held has no recipe, so no card on it can roll.
    unknown = {"spec_id": "ops_" + "0" * 32, "content_version": "ocv_" + "0" * 32}
    assert await cards.frame(unknown) is None


async def test_a_card_drawing_the_series_the_frame_counted_is_written_as_asked(
    scene: Scene,
) -> None:
    ref, _ = await scene.rolling()
    replays = Replays(scene)
    card = _card(ref.model_dump(), select={"reporter": list(REPORTERS), "stat_var": STAT_VAR})

    assert await _admit(RollingCards(replay=replays), card) == card
    assert replays.populations == []
    assert scene.dataset.minted == []


async def test_cards_drawing_the_same_other_series_share_one_replay(scene: Scene) -> None:
    ref, _ = await scene.rolling()
    replays = Replays(scene)
    cards = RollingCards(replay=replays)
    platform = Platform()
    scene.tools.calls.clear()

    first = await cards.admit(
        {"intent": _card(ref.model_dump(), select={"reporter": FRANCE, "stat_var": STAT_VAR})},
        read_view=platform.view,
        read_stored=platform.stored,
        read_window=platform.window,
    )
    second = await _admit(
        cards, _card(ref.model_dump(), type="table", select={"reporter": [FRANCE]})
    )

    moved = first.args["intent"]["frame_ref"]
    assert first.replayed_from == ref.model_dump()
    assert moved == second["frame_ref"] != ref.model_dump()
    # Only the frame changes: the card keeps its select, title and type.
    assert {**first.args["intent"], "frame_ref": ref.model_dump()} == _card(
        ref.model_dump(), select={"reporter": FRANCE, "stat_var": STAT_VAR}
    )
    assert replays.populations == [[{"spec_id": SPEC, "bindings": {"reporter": [FRANCE]}}]]
    # One replay: one mint of the root, one resolution, one run of the rolling step.
    assert len(scene.dataset.minted) == 1
    assert scene.tools.names == ["resolve_anchor", "growth"]
    replayed = await cards.frame(moved)
    assert replayed is not None
    assert replayed.populations == replays.populations[0]
    # France alone holds 2023, so its card's window ends a year after the frame's.
    assert replayed.recipe["anchor"]["period"] == "2023"
    growth = scene.tools.arguments("growth")
    assert (growth["window_start"], growth["window_end"]) == ("2020", "2023")
    # A card already on the replayed frame, drawing its series, stays there.
    assert await _admit(cards, _card(moved, select={"reporter": FRANCE})) == _card(
        moved, select={"reporter": FRANCE}
    )
    assert len(replays.populations) == 1


_LATEST = {"anchor": "latest", "offset": 0}
_WINDOW = {"start": {"anchor": "latest", "offset": -4}, "end": _LATEST}


@pytest.mark.parametrize(
    ("title", "named"),
    [
        ("Growth to FY2022", "2022"),
        ("Growth, 2022Q4", "2022"),
        ("Growth to 2022", "2022"),
        ("Growth, 2019-2022", "2019"),
        ("Agenda 2030 progress", None),
        ("Growth since 2015", None),
        ("Growth of 12022 projects", None),
        ("Growth over the last three years", None),
    ],
)
def test_a_title_is_refused_for_naming_a_period_the_card_resolved(
    title: str, named: str | None
) -> None:
    """2019 and 2022 were worked out from the latest data; 2015 and 2030 were not."""
    refusal = title_refusal(title, {"2019", "2022"})

    if named is None:
        assert refusal is None
    else:
        assert refusal is not None
        assert (refusal["code"], refusal["fix"]) == (
            HarnessCode.TITLE_QUOTES_ROLLING_PERIOD,
            {"period": named},
        )
        assert named in refusal["message"]


async def _growth_to_the_latest_year_from_2019(scene: Scene) -> FrameRef:
    """Growth from the fixed year 2019 to the latest year, which is 2022."""
    root = await scene.root()
    populations = [{"spec_id": SPEC, "bindings": {}}]
    anchor = await resolve_anchor(populations=populations, roots=[root.model_dump()])
    assert anchor.structured_content is not None
    result = await growth(
        frame_ref=root.model_dump(),
        window_start="2019",
        window_end="2022",
        entity_by=("reporter",),
        rolling={
            "window_end": {"anchor": "latest", "offset": 0, "rule": "half1"},
            "population": populations,
        },
        anchor_key=anchor.structured_content["anchor_key"],
    )
    assert result.structured_content is not None
    return FrameRef(
        spec_id=result.structured_content["spec_id"],
        content_version=result.structured_content["content_version"],
    )


async def test_a_rolling_frames_resolved_periods_leave_out_the_years_the_plan_fixed(
    scene: Scene,
) -> None:
    both_roll, _ = await scene.rolling()
    end_rolls = await _growth_to_the_latest_year_from_2019(scene)
    cards = RollingCards(replay=None)

    moving = await cards.frame(both_roll.model_dump())
    fixed_start = await cards.frame(end_rolls.model_dump())

    assert moving is not None
    assert fixed_start is not None
    assert frame_periods(moving.recipe) == {"2019", "2022"}
    assert frame_periods(fixed_start.recipe) == {"2022"}
    # "Since 2019" names the fixed start of one frame and a moving bound of the other.
    since = _card(end_rolls.model_dump(), title="Growth since 2019")
    assert await _admit(cards, since) == since
    with pytest.raises(CardWithheld) as withheld:
        await _admit(cards, _card(both_roll.model_dump(), title="Growth since 2019"))
    assert withheld.value.refusal["fix"] == {"period": "2019"}


async def test_a_title_on_a_fixed_or_dataset_frame_may_name_any_year(scene: Scene) -> None:
    fixed, recipe = await scene.fixed()
    cards = RollingCards(replay=None)

    for ref in (fixed.model_dump(), recipe["roots"][0]["frame"]):
        card = _card(ref, title="Growth, 2019-2022")
        assert await _admit(cards, card) == card


@pytest.mark.parametrize(
    ("intent", "named"),
    [
        pytest.param({"window": _WINDOW}, "2020", id="relative-window"),
        pytest.param({"window": {"start": "2020", "end": _LATEST}}, "2024", id="relative-end"),
        pytest.param(
            {"change": {"basis": [{**_LATEST, "offset": -1}, _LATEST]}},
            "2023",
            id="relative-change",
        ),
        pytest.param({"window": {"start": "2020", "end": "2024"}}, None, id="named-years"),
        pytest.param({}, None, id="no-period"),
    ],
)
def test_a_stored_cards_title_is_held_to_the_periods_its_relative_ones_resolved_to(
    intent: dict[str, Any], named: str | None
) -> None:
    refusal = written_title_refusal(
        intent,
        title="Aid 2020 to 2024, against 2023",
        window={"start": "2020", "end": "2024"},
        basis=["2023", "2024"],
    )

    assert (refusal["fix"]["period"] if refusal else None) == named


async def test_a_card_stating_a_relative_period_is_marked_for_the_check_after_its_write(
    scene: Scene,
) -> None:
    _, recipe = await scene.rolling()
    dataset = recipe["roots"][0]["frame"]
    cards = RollingCards(replay=None)
    platform = Platform()

    async def marked(**intent: Any) -> bool:
        admitted = await cards.admit(
            {"intent": _card(dataset, **intent)},
            read_view=platform.view,
            read_stored=platform.stored,
            read_window=platform.window,
        )
        return admitted.stored_card is not None

    assert await marked(window=_WINDOW, title="Aid, 2020-2024") is True
    assert await marked(window={"start": "2015", "end": "2020"}) is False


@pytest.mark.parametrize(
    ("requested", "refused"),
    [
        pytest.param({"kind": "range", "start_year": 2015, "end_year": 2024}, True, id="closed"),
        pytest.param({"kind": "range", "start_year": 2015}, False, id="open-end"),
        pytest.param({"kind": "all"}, False, id="all"),
        pytest.param(None, False, id="no-definition-held"),
    ],
)
async def test_a_relative_period_is_refused_on_data_defined_to_end_at_a_stated_year(
    scene: Scene, requested: dict[str, Any] | None, refused: bool
) -> None:
    _, recipe = await scene.rolling()
    dataset = recipe["roots"][0]["frame"]
    platform = Platform()
    platform.requested_window = requested
    cards = RollingCards(replay=None)
    moving = _card(dataset, window=_WINDOW)

    # A revise adding a relative window to a card of named years meets the same rule.
    platform = Platform({"cmp_1": {"stored_intent": _card(dataset)}})
    platform.requested_window = requested
    for write, args in ((moving, {}), ({"window": _WINDOW}, {"component_id": "cmp_1"})):
        if refused:
            with pytest.raises(CardWithheld) as withheld:
                await _admit(cards, write, platform=platform, **args)
            assert withheld.value.refusal["code"] == HarnessCode.RELATIVE_PERIOD_ON_CLOSED_WINDOW
        else:
            assert await _admit(cards, write, platform=platform, **args) == write
    # Named years are drawn on any definition.
    named = _card(dataset, window={"start": "2015", "end": "2020"})
    assert await _admit(cards, named, platform=platform) == named


async def _withheld_by_a_queued_mint(scene: Scene) -> Replays | None:
    scene.dataset.answers = [
        QueuedOperation(operation_id="op_" + "0" * 64, status="pending", retry_after_s=1.5)
    ]
    return Replays(scene)


async def _withheld_by_an_unresolved_anchor(scene: Scene) -> Replays | None:
    scene.tools.answers["resolve_anchor"] = SimpleNamespace(
        is_error=True,
        structured_content={
            "code": "rolling_anchor_unresolved",
            "retry_mode": "correct_and_retry",
            "message": "no period is held by half of the series",
        },
    )
    return Replays(scene)


async def _withheld_with_no_replayer(scene: Scene) -> Replays | None:
    del scene
    return None


@pytest.mark.parametrize(
    ("arrange", "code"),
    [
        pytest.param(_withheld_by_a_queued_mint, HarnessCode.ROLLING_REPLAY_PENDING, id="pending"),
        pytest.param(
            _withheld_by_an_unresolved_anchor, HarnessCode.ROLLING_REPLAY_REFUSED, id="refused"
        ),
        pytest.param(
            _withheld_with_no_replayer, HarnessCode.ROLLING_REPLAY_NOT_CONFIGURED, id="no-replayer"
        ),
    ],
)
async def test_a_replay_that_gives_no_frame_withholds_every_card_of_its_series(
    scene: Scene, arrange: Any, code: str
) -> None:
    """The card is sent on no frame: the run's frame took its latest year over
    other series. A second card of the same series does not replay again until
    the wait has passed: the 1.5 s the queued mint named for a replay that ended
    `Pending`, and `WITHHELD_FOR_S` for one that was refused."""
    ref, _ = await scene.rolling()
    replays = await arrange(scene)
    clock = Clock()
    cards = RollingCards(replay=replays, clock=clock)
    card = _card(ref.model_dump(), select={"reporter": FRANCE})

    codes = []
    for _ in range(2):
        with pytest.raises(CardWithheld) as withheld:
            await _admit(cards, card)
        codes.append(withheld.value.refusal["code"])

    assert codes == [code, code]
    assert replays is None or len(replays.populations) == 1
    assert len(scene.apis.published()) == 1  # the scene's own frame, and no other
    if replays is None:
        return
    # The dataset answers and the tool works again; the wait has to pass first.
    scene.tools.answers.clear()
    wait = 1.5 if code == HarnessCode.ROLLING_REPLAY_PENDING else WITHHELD_FOR_S
    clock.now = wait - 0.1
    with pytest.raises(CardWithheld):
        await _admit(cards, card)
    assert len(replays.populations) == 1
    clock.now = wait
    assert (await _admit(cards, card))["frame_ref"] != ref.model_dump()
    assert len(replays.populations) == 2


async def test_a_replay_cut_at_its_deadline_is_tried_again_from_the_steps_in_the_store(
    scene: Scene,
) -> None:
    """A replay that ended `Pending` withholds its cards for the wait it named,
    not `WITHHELD_FOR_S`, and the attempt after it calls no verb a second time."""
    ref, _ = await scene.rolling()
    ran_out = Clock()

    class CutAfterTheVerb(Replays):
        async def __call__(self, recipe: Any, *, deadline: Any, populations: Any) -> Any:
            del deadline
            return await super().__call__(
                recipe,
                deadline=CallDeadline(expires_at=50.0, clock=ran_out),
                populations=populations,
            )

    def run_out(name: str) -> None:
        ran_out.now = 100.0 if name == "growth" else ran_out.now

    scene.tools.after = run_out
    replays = CutAfterTheVerb(scene)
    clock = Clock()
    cards = RollingCards(replay=replays, clock=clock)
    card = _card(ref.model_dump(), select={"reporter": FRANCE})

    for _ in range(2):
        with pytest.raises(CardWithheld) as withheld:
            await _admit(cards, card)
        assert withheld.value.refusal["code"] == HarnessCode.ROLLING_REPLAY_PENDING
    assert len(replays.populations) == 1

    ran_out.now = 0.0
    scene.tools.after = None
    clock.now = RESUME_RETRY_AFTER_S
    assert (await _admit(cards, card))["frame_ref"] != ref.model_dump()
    assert len(replays.populations) == 2
    # The growth of the first attempt is in the store and is not computed again.
    assert scene.tools.names.count("growth") == 1


async def test_a_revise_is_judged_as_the_stored_card_with_the_patch_merged_in(
    scene: Scene,
) -> None:
    ref, recipe = await scene.rolling()
    stored = _card(ref.model_dump())
    # A stat card on the dataset frame, comparing the latest year with the one before.
    compared = {
        "type": "stat_card",
        "frame_ref": recipe["roots"][0]["frame"],
        "title": "Aid against the year before",
        "change": {"basis": [{**_LATEST, "offset": -1}, _LATEST]},
    }
    platform = Platform(
        {
            "cmp_1": {"stored_intent": stored},
            "cmp_2": {
                "stored_intent": compared,
                "resolved_payload": {"component": {"change": {"basis": ["2021", "2022"]}}},
            },
        }
    )
    replays = Replays(scene)
    cards = RollingCards(replay=replays)

    async def revise(patch: dict[str, Any], component_id: str = "cmp_1") -> str | None:
        """The code the revise `patch` is refused with, `None` when it is sent."""
        try:
            sent = await _admit(cards, patch, platform=platform, component_id=component_id)
        except CardWithheld as withheld:
            return withheld.refusal["code"]
        assert sent == patch
        return None

    # A title naming a period the stored card's `change.basis` resolved is refused.
    assert await revise({"title": "Aid in 2022 against 2021"}, "cmp_2") == (
        HarnessCode.TITLE_QUOTES_ROLLING_PERIOD
    )
    assert await revise({"title": "Aid in 2015 against 2010"}, "cmp_2") is None
    # A patch stating new relative periods is judged on what the new revision resolves.
    for patch, judged_after in (
        ({"change": {"basis": [{**_LATEST, "offset": -2}, _LATEST]}}, True),
        ({"change": {"basis": ["2015", "2020"]}}, False),
    ):
        admitted = await cards.admit(
            {"intent": patch, "component_id": "cmp_2"},
            read_view=platform.view,
            read_stored=platform.stored,
            read_window=platform.window,
        )
        assert (admitted.stored_card is not None) is judged_after
    # A patch that moves the periods a card resolves is judged after the write
    # alone, as the stored card with the patch merged in, even with a title naming
    # a period the stored revision resolved.
    moved = await cards.admit(
        {
            "intent": {"select": {"reporter": [FRANCE]}, "title": "Aid in 2022"},
            "component_id": "cmp_2",
        },
        read_view=platform.view,
        read_stored=platform.stored,
        read_window=platform.window,
    )
    assert moved.stored_card == {
        **compared,
        "select": {"reporter": [FRANCE]},
        "title": "Aid in 2022",
    }
    platform.stored_reads.clear()

    # A patch that can break no rule is sent without reading the stored card.
    assert await revise({"title": "Growth of each reporter"}) is None
    assert platform.stored_reads == []
    assert await revise({"title": "Growth, 2019-2022"}) == HarnessCode.TITLE_QUOTES_ROLLING_PERIOD
    assert await revise({"title": "Growth of 2030 targets"}) is None
    assert await revise({"select": {"reporter": sorted(REPORTERS)}}) is None
    assert replays.populations == []
    # Other series need another frame, which a revise cannot name.
    assert (
        await revise({"select": {"reporter": [FRANCE]}})
        == HarnessCode.ROLLING_SELECT_NEEDS_NEW_FRAME
    )
    # The same card written as a create is moved to that frame, with no second replay.
    created = await _admit(cards, {**stored, "select": {"reporter": [FRANCE]}})
    assert created["frame_ref"] != ref.model_dump()
    assert len(replays.populations) == 1


async def test_a_card_on_a_frame_whose_populations_reach_past_its_tree_is_replayed_over_its_own(
    scene: Scene,
) -> None:
    """S1 on A, S2 on B and S3 on C are filters on one latest year over all
    three, because T3 joins S2 with S3 as T2 joins S1 with S2. T2 reads A and B
    alone, so it has no anchor of its own, and a card on it is written on a
    frame replayed over A and B."""
    gni, debt = "spec_" + "b" * 32, "spec_" + "c" * 32
    for spec_id, stat_var in ((gni, "dcid:GNI"), (debt, "dcid:Debt")):
        scene.dataset.others[spec_id] = Frame.build(
            role_keys=["reporter"],
            rows=reporter_rows(
                {reporter: yearly(range(2019, 2022), start=1.0) for reporter in REPORTERS},
                stat_var=stat_var,
            ),
        )
    every = [SPEC, gni, debt]
    s1, s2, s3 = [
        await scene.since(await scene.root(spec_id=spec_id), spec_ids=every) for spec_id in every
    ]
    t2, t3 = [await scene.joined(left, right) for left, right in ((s1, s2), (s2, s3))]
    del t3  # it puts C in the population, and is not drawn
    with pytest.raises(RecipeAnchorError):
        await build_recipe(t2, store=scene.store)
    replays = Replays(scene)
    cards = RollingCards(replay=replays)

    drawn = FrameRef.model_validate((await _admit(cards, _card(t2.model_dump())))["frame_ref"])

    assert drawn != t2
    assert [[population["spec_id"] for population in call] for call in replays.populations] == [
        [SPEC, gni]
    ]
    replayed = await cards.frame(drawn.model_dump())
    assert replayed is not None
    assert replayed.recipe["anchor"] is not None
    assert [population["spec_id"] for population in replayed.populations] == [SPEC, gni]


async def test_only_the_most_recently_used_replays_are_remembered(
    scene: Scene, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Past the bound the oldest replay is forgotten, and its card replays again."""
    monkeypatch.setattr(rolling_cards, "MAX_REPLAYS_KEPT", 1)
    ref, _ = await scene.rolling()
    replays = Replays(scene)
    cards = RollingCards(replay=replays)

    async def card_of(reporter: str) -> dict[str, Any]:
        return await _admit(cards, _card(ref.model_dump(), select={"reporter": reporter}))

    france = await card_of(FRANCE)
    await card_of(FRANCE)
    assert len(replays.populations) == 1
    await card_of(BRITAIN)
    assert len(replays.populations) == 2

    assert await card_of(FRANCE) == france
    assert len(replays.populations) == 3

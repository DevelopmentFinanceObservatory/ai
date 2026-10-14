"""The rules for a card on a frame whose period follows the latest data.

A rolling frame is a derived frame whose recipe states a period against the
latest one (`common.recipe.build_recipe` gives it an `anchor`). The latest
period is the newest one at least half of a population of series hold, and the
population is hashed into the frame's rolling op-specs, so a frame stands for
one population. A card must stand on the frame of the series it draws, and a
card whose period moves must not name, in its title, a period it will move
past: one that was worked out from the latest data.

`RollingCards.admit` applies both rules to one `write_component` call.
`harness.mcp_client.McpExecutor` runs it on every write, which is the one call
every card passes through: the orchestrator's `make_charts` and the Edit Card
route.

A create (an intent with a `frame_ref` and no `component_id`):

1. On a rolling frame, `title_refusal` refuses a title naming one of the
   frame's resolved periods (`frame_periods`): the recipe's latest period and
   each window bound or filter year a rolling reference set. A year the plan
   fixed, and any other number, may be named.
2. On a rolling frame, `card_selection` keeps the dimensions of the card's
   select that say which series count: one the frame's population already
   binds, and one that lists fewer values than the frame's view holds for it.
   A dimension pinned to the only value the frame holds narrows nothing.
3. `common.rolling.anchor_populations` builds the card's populations from the
   recipe's tree, that selection, and each root frame's column names.
4. Populations equal to the frame's own leave the write as it is. Otherwise
   the recipe is replayed with them, which resolves the latest period again,
   runs the rolling steps on it, and publishes a new frame with its recipe,
   and the write is sent with that frame in place of the one it named, once
   the title has passed the new frame's resolved periods too.
5. An intent that itself states a relative period has its periods resolved by
   the platform as it stores the card. `written_title_refusal` reads them from
   the stored card and refuses a title naming one, after the write: the
   platform then holds a revision that no canvas shows.

A revise (a patch with a `component_id`) is judged as the stored intent with
the patch merged in, under the rules a create meets. The stored card is read
only for a patch that could break a rule: one that changes `select`, `window`
or `change` (the members that can move the periods a card resolves), or whose
title holds four digits in a row. Its title is held to the frame's resolved
periods. A merged card that states a relative period is marked like a create
that does (step 5): its title is judged after the write, against the periods
the new revision resolved. A patch that changes none of those three members
leaves those periods as the stored revision resolved them, so its title is
also judged before the write, against the window and the `change.basis` the
stored card resolved, and no revision is stored for a title that would be
refused. A revise cannot move a card to another frame, so a select that counts
other series is refused `rolling_select_needs_new_frame`, and the caller
writes the merged intent as a new card.

One replay costs a mint request to dfo-apis for each root, one `resolve_anchor`
call, one call of each rolling verb and of every verb above it (a population is
part of an op-spec's identity, so each mints a new frame), one replay-key read,
and one publish. The frame a replay gave is kept, for the `MAX_REPLAYS_KEPT`
most recently used, under the frame it was replayed from and the populations,
so a later card of the same series on that frame is written on it with no call.
A replay that gave no frame is kept, so the cards drawn beside it on the same
series are withheld without another attempt: for the wait the replay itself
named when it ended `Pending` (a second or two, after which the next attempt
starts from the steps the store already holds), and for `WITHHELD_FOR_S` when
it was refused, ran past its hard stop, or had no replayer. Each replay has
`CARD_REPLAY_DEADLINE_S` and is cancelled `_HARD_STOP_GRACE_S` after it.

Whether a frame rolls is read from the materialisation store alone
(`RollingCards.frame`), so it is the same answer whenever it is asked.

A card, created or revised, that states a relative period on a frame that
does not roll is refused `relative_period_on_closed_window` when the dataset's definition ends at a
stated year: such a frame is rebuilt over the same years for good, so the card
could never move. A definition with no end year, or of another kind, is drawn.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Protocol

import rfc8785
from common.live_apis import LiveApisError, LiveApisNotConfigured
from common.materialisation import (
    FrameExpired,
    FrameIntegrityError,
    FrameRef,
    FrameTooLarge,
    get_store,
)
from common.recipe import RecipeAnchorError, RecipeError, build_recipe
from common.rolling import (
    PopulationEmptyError,
    PopulationLevelColumnMissingError,
    anchor_populations,
    canonical_populations,
)
from common.row_budget import RowBudgetExhausted, row_budget
from compiler.surface import ToolSurfaceUnavailableError
from data_mcp.deadline import CallDeadline
from pydantic import ValidationError

from harness.recipe_replay import Pending, Refused, Replayed
from harness.tool_contracts import HarnessCode, harness_refusal

__all__ = [
    "CARD_REPLAY_DEADLINE_S",
    "MAX_FRAMES_KEPT",
    "MAX_REPLAYS_KEPT",
    "WITHHELD_FOR_S",
    "AdmittedWrite",
    "CardReplay",
    "CardWithheld",
    "RollingCards",
    "RollingFrame",
    "card_selection",
    "frame_periods",
    "holds_relative_period",
    "resolved_basis",
    "title_refusal",
    "written_title_refusal",
]

logger = logging.getLogger(__name__)

# The `message` of each refusal a card whose period moves can meet before it is
# written, for the person or model that asked for the card.
_TITLE_MESSAGE = (
    "The title names {period}, which this card worked out from the latest data and will "
    "move past when newer data arrives. Give it a title that leaves that period out."
)
_PENDING_MESSAGE = (
    "The latest period of the series this card shows could not be worked out in time. "
    "Asking again shortly may work."
)
_REFUSED_MESSAGE = (
    "The latest period of the series this card shows could not be worked out, so the "
    "card was not drawn."
)
_NOT_CONFIGURED_MESSAGE = (
    "This deployment cannot work out the latest period of a card's own series, so the "
    "card was not drawn."
)
_CLOSED_WINDOW_MESSAGE = (
    "This data is defined to end at {end_year}, so a period counted back from the latest "
    "data would never move on it. Fetch it again with scope window "
    '{{"kind": "range", "start_year": Y}} and no end_year, then draw the card on that frame.'
)
_NEW_FRAME_MESSAGE = (
    "The series now selected have another latest period than the card's data was "
    "computed for, so the card cannot be changed in place."
)

# Seconds one replay has before it starts no new step. A root mint starts only
# with 25 s of it left (`data_mcp.deadline.EXTERNAL_CALL_BUDGET`).
CARD_REPLAY_DEADLINE_S: Final = 45.0
# How long after its deadline a replay is cancelled. The deadline stops new
# steps; this bounds the one already in flight.
_HARD_STOP_GRACE_S: Final = 8.0
# Seconds a replay that was refused, ran past its hard stop, or had no replayer
# withholds the cards of its series without another attempt. That is longer than
# one replay, so cards drawn beside it share it. A replay that ended `Pending` is
# kept for its own `retry_after_s` instead.
WITHHELD_FOR_S: Final = 90.0
# Frame answers kept, least recently used out first. A rolling frame's entry
# holds its whole recipe, which presentation-mcp caps at 1,000,000 encoded bytes
# and which is a few thousand for a plan of a handful of steps.
MAX_FRAMES_KEPT: Final = 128
# Replayed frames kept, least recently used out first. An entry is two frame
# pairs and one population list, under a kilobyte.
MAX_REPLAYS_KEPT: Final = 1024

_DERIVED_PREFIX = "ops_"
_ROLLING = "rolling"

_YEAR = re.compile(r"\d{4}")
# The members of a revise patch that can move the periods a card resolves.
_PERIOD_MEMBERS: Final = frozenset({"select", "window", "change"})


def _relative(period: object) -> bool:
    return isinstance(period, Mapping)


def holds_relative_period(intent: Mapping[str, Any]) -> bool:
    """Whether `intent` states a period against the latest one: a side of its
    `window`, or an entry of its `change.basis`, that is an object."""
    periods: list[Any] = []
    window = intent.get("window")
    if isinstance(window, Mapping):
        periods += [window.get("start"), window.get("end")]
    change = intent.get("change")
    if isinstance(change, Mapping) and isinstance(change.get("basis"), list):
        periods += change["basis"]
    return any(_relative(period) for period in periods)


def frame_periods(recipe: Mapping[str, Any]) -> frozenset[str]:
    """The periods of a rolling recipe that were worked out from the latest data:
    its anchor's period, and each window bound and filter value a rolling
    reference governs. A bound the plan stated as a fixed year is not one. A
    recipe with a null anchor (see `RollingCards.frame`) gives the latter alone."""
    anchor = recipe["anchor"]
    periods = {str(anchor["period"])} if anchor is not None else set()

    def walk(node: Mapping[str, Any]) -> None:
        if "root" in node:
            return
        parameters = node["op_spec"]["parameters"]
        overlay = parameters.get(_ROLLING)
        if isinstance(overlay, Mapping):
            periods.update(
                str(parameters[name]) for name, reference in overlay.items() if _relative(reference)
            )
        periods.update(
            str(predicate["value"])
            for predicate in parameters.get("predicates", ())
            if _relative(predicate.get(_ROLLING))
        )
        for child in node["inputs"].values():
            walk(child)

    walk(recipe["tree"])
    return frozenset(periods)


def title_refusal(title: object, periods: Iterable[object]) -> dict[str, Any] | None:
    """The `title_quotes_rolling_period` refusal of a title that names one of
    `periods`, `None` for any other title.

    A period is named when the digits of one of its years stand in the title
    with no digit on either side, so "FY2024" and "2024Q4" name 2024 and
    "12024" does not. The refusal's `message` and `fix.period` name the first
    such year. A year outside `periods` may be named: "Agenda 2030", or the
    fixed first year of a window whose end moves.
    """
    text = str(title or "")
    years = sorted({year for period in periods for year in _YEAR.findall(str(period))})
    for year in years:
        if re.search(rf"(?<!\d){year}(?!\d)", text):
            refusal = harness_refusal(
                code=HarnessCode.TITLE_QUOTES_ROLLING_PERIOD,
                message=_TITLE_MESSAGE.format(period=year),
            )
            return {**refusal, "fix": {"period": year}}
    return None


def written_title_refusal(
    intent: Mapping[str, Any],
    *,
    title: object,
    window: Mapping[str, Any] | None,
    basis: Sequence[object] | None,
) -> dict[str, Any] | None:
    """`title_refusal` against the periods the platform resolved for the relative
    periods `intent` states.

    Args:
        intent: The stored intent.
        title: The card's title.
        window: The stored card's resolved window, `{start, end}` as labels.
        basis: The stored card's resolved `change.basis`, as labels.
    """
    resolved: list[object] = []
    stated = intent.get("window")
    if isinstance(stated, Mapping) and window is not None:
        resolved += [window.get(side) for side in ("start", "end") if _relative(stated.get(side))]
    change = intent.get("change")
    stated_basis = change.get("basis") if isinstance(change, Mapping) else None
    if isinstance(stated_basis, list) and basis is not None:
        resolved += [
            label for entry, label in zip(stated_basis, basis, strict=False) if _relative(entry)
        ]
    return title_refusal(title, [label for label in resolved if label is not None])


def resolved_basis(envelope: object) -> Sequence[object] | None:
    """The `change.basis` a stored revision's resolved payload `envelope` drew,
    as labels; `None` for an envelope whose component states no change."""
    component = envelope.get("component") if isinstance(envelope, Mapping) else None
    change = component.get("change") if isinstance(component, Mapping) else None
    basis = change.get("basis") if isinstance(change, Mapping) else None
    return basis if isinstance(basis, list) else None


@dataclass(frozen=True, kw_only=True)
class RollingFrame:
    """A derived frame whose recipe rolls: its pair, its recipe, and the
    populations its latest period was taken over, in canonical form."""

    ref: FrameRef
    recipe: Mapping[str, Any]
    populations: list[dict[str, Any]]


def _rolling_populations(tree: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The populations the rolling op-specs of `tree` carry. `build_recipe` gives
    an anchor only to a tree whose rolling op-specs all carry one list."""
    if "root" in tree:
        return []
    overlay = tree["op_spec"]["parameters"].get(_ROLLING)
    if isinstance(overlay, Mapping):
        return canonical_populations(overlay["population"])
    for child in tree["inputs"].values():
        if found := _rolling_populations(child):
            return found
    return []


def card_selection(
    select: Mapping[str, Any], *, view: Mapping[str, Any], bound: frozenset[str]
) -> dict[str, list[str]]:
    """The dimensions of `select` that say which series a card's latest period
    is taken over, each with the values it keeps.

    Args:
        select: The card's select, each dimension to one value or a list.
        view: The frame's view, as `describe_frame` gives it.
        bound: The columns the frame's own populations bind.

    Returns:
        A dimension in `bound`, and a dimension that lists fewer values than
        the view counts for it (`total_count`, or one for a dimension the view
        marks not plural). A dimension the view gives no count for is kept.
    """
    counts: dict[str, int | None] = {}
    for dimension in view.get("dimensions") or []:
        if not isinstance(dimension, Mapping) or not dimension.get("name"):
            continue
        count = dimension.get("total_count")
        if not isinstance(count, int):
            count = 1 if dimension.get("plural") is False else None
        counts[str(dimension["name"])] = count
    selection: dict[str, list[str]] = {}
    for name, value in select.items():
        values = sorted({str(one) for one in (value if isinstance(value, list) else [value])})
        count = counts.get(name)
        if name in bound or count is None or len(values) < count:
            selection[name] = values
    return selection


class CardReplay(Protocol):
    """Replays a rolling recipe over other populations, as
    `harness.recipe_replay.replay_recipe` does with `populations` given.
    `harness.live_refresh.LiveRefresh.replay` is one."""

    async def __call__(
        self,
        recipe: Mapping[str, Any],
        *,
        deadline: CallDeadline,
        populations: Sequence[Mapping[str, Any]],
    ) -> Replayed | Pending | Refused: ...


class CardWithheld(Exception):  # noqa: N818 -- a control-flow signal carrying a refusal
    """A write is refused before it is sent. `refusal` is a harness refusal
    whose `message` says why in words for the person who asked. `held_for_s`
    is how long a replay that ended this way is not attempted again."""

    def __init__(self, refusal: dict[str, Any], *, held_for_s: float = WITHHELD_FOR_S) -> None:
        self.refusal = refusal
        self.held_for_s = held_for_s
        super().__init__(refusal["message"])


def _withheld(
    code: HarnessCode, message: str, cause: object, *, held_for_s: float = WITHHELD_FOR_S
) -> CardWithheld:
    """The refusal `code` with `message`, the cause logged beside it."""
    logger.warning("a card on a rolling frame was withheld as %s: %s", code, cause)
    return CardWithheld(harness_refusal(code=code, message=message), held_for_s=held_for_s)


@dataclass(frozen=True, kw_only=True)
class AdmittedWrite:
    """A `write_component` call's arguments as they are to be sent.
    `replayed_from` is the frame the intent named when the write was moved to a
    frame replayed from it, whose provenance the new frame shares.
    `stored_card` is the card as the platform will store it (a revise's stored
    intent with the patch merged in), given when it states a relative period:
    its title is still to pass `written_title_refusal` against the periods the
    platform resolves as it stores it."""

    args: dict[str, Any]
    replayed_from: dict[str, Any] | None = None
    stored_card: dict[str, Any] | None = None


# Reads a frame's view, and a component's head revision as `describe_component`
# gives it (`stored_intent`, `fact_summary`) with the revision's resolved
# payload beside them under `resolved_payload`; `None` when unreadable.
ReadView = Callable[[Mapping[str, Any]], Awaitable[Mapping[str, Any] | None]]
ReadStored = Callable[[str], Awaitable[Mapping[str, Any] | None]]
# Reads the `requested_window` of the definition a dataset frame was retrieved
# for; `None` when the caller holds no definition for the frame.
ReadWindow = Callable[[Mapping[str, Any]], Mapping[str, Any] | None]

_FrameKey = tuple[str, str]
_ReplayKey = tuple[str, str, bytes]


def _log_no_recipe(frame: FrameRef, exc: Exception) -> None:
    logger.warning(
        "no recipe for %s at %s, so its cards are written as fixed: %s",
        frame.spec_id,
        frame.content_version,
        exc,
    )


def _refuse_title(title: object, frame: RollingFrame) -> None:
    """Raise `CardWithheld` for a title naming one of `frame`'s resolved periods."""
    if (refusal := title_refusal(title, frame_periods(frame.recipe))) is not None:
        raise CardWithheld(refusal)


def _resolved_later(card: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """`card`, when it states a relative period the platform resolves as it stores it."""
    return dict(card) if card is not None and holds_relative_period(card) else None


def _refuse_closed_window(card: Mapping[str, Any], read_window: ReadWindow) -> None:
    """Raise `CardWithheld` for a `card` stating a relative period on data
    whose definition window ends at a stated year. A card stating none, a
    definition the caller does not hold, one with no end year, and one of
    another kind (`all`, `latest`) pass."""
    ref = card.get("frame_ref")
    if not holds_relative_period(card) or not isinstance(ref, Mapping):
        return
    window = read_window(ref)
    if window is None or window.get("kind") != "range" or window.get("end_year") is None:
        return
    raise CardWithheld(
        harness_refusal(
            code=HarnessCode.RELATIVE_PERIOD_ON_CLOSED_WINDOW,
            message=_CLOSED_WINDOW_MESSAGE.format(end_year=window["end_year"]),
        )
    )


class RollingCards:
    """The rolling-card rules for every write of one process. See the module docstring.

    It remembers three things, each bounded. What a frame pair is, rolling or
    not, for `MAX_FRAMES_KEPT` pairs. The frame a recipe replayed to for given
    populations when it was last replayed in this process, for
    `MAX_REPLAYS_KEPT` replays: a hit writes the card on that frame with no
    call, and a miss replays. And each replay that gave no frame, for the wait
    its `CardWithheld` carries.
    """

    def __init__(
        self,
        *,
        replay: CardReplay | None,
        deadline_s: float = CARD_REPLAY_DEADLINE_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Build it over its dependencies.

        Args:
            replay: Replays a recipe over other populations. `None` withholds
                every card that needs a replay.
            deadline_s: Seconds one replay has.
            clock: Times how long a replay that gave no frame is remembered.
        """
        self._replay = replay
        self._deadline_s = deadline_s
        self._clock = clock
        # What a pair was found to be. A pair names one frame for good, so
        # neither answer goes stale.
        self._frames: OrderedDict[_FrameKey, RollingFrame | None] = OrderedDict()
        self._root_columns: dict[_FrameKey, list[frozenset[str]]] = {}
        self._replayed: OrderedDict[_ReplayKey, FrameRef] = OrderedDict()
        # Each replay that gave no frame, with its refusal and when it may be tried again.
        self._withheld: dict[_ReplayKey, tuple[float, dict[str, Any]]] = {}
        # Replays in flight, so cards written side by side share one.
        self._running: dict[_ReplayKey, asyncio.Task[FrameRef]] = {}

    async def frame(self, ref: Mapping[str, Any]) -> RollingFrame | None:
        """The rolling frame held under `ref`, or `None` for a frame that cannot roll.

        A dataset frame, a derived frame of fixed periods, and a derived frame
        the store can build no recipe for all answer `None`: none of them is
        replayed when data moves, so a card on one shows the periods it was
        written with. A rolling frame whose populations name a dataset no
        rolling period of its tree reads has no anchor of its own; it answers
        with the recipe `build_recipe` puts on its `RecipeAnchorError`, whose
        `anchor` is null, and a card on it is always replayed.
        """
        if not str(ref.get("spec_id", "")).startswith(_DERIVED_PREFIX):
            return None
        try:
            frame = FrameRef.model_validate(ref)
        except ValidationError:
            # No store holds a frame under a pair of another form.
            return None
        key = (frame.spec_id, frame.content_version)
        if key in self._frames:
            self._frames.move_to_end(key)
            return self._frames[key]
        try:
            recipe = await build_recipe(frame, store=get_store())
        except RecipeAnchorError as exc:
            if exc.recipe is None:
                _log_no_recipe(frame, exc)
                return None
            # Its populations name a dataset no rolling period of its tree reads,
            # so it has no anchor of its own. Its cards are replayed over the
            # populations of its own tree (`_populations`), which differ.
            recipe = exc.recipe
        except (RecipeError, FrameExpired, FrameIntegrityError, FrameTooLarge) as exc:
            # Not kept, because the store may hold the missing part on a later read.
            _log_no_recipe(frame, exc)
            return None
        # A tree with no rolling op-spec carries no population.
        populations = _rolling_populations(recipe["tree"])
        found = (
            RollingFrame(ref=frame, recipe=recipe, populations=populations) if populations else None
        )
        self._frames[key] = found
        while len(self._frames) > MAX_FRAMES_KEPT:
            dropped, _ = self._frames.popitem(last=False)
            self._root_columns.pop(dropped, None)
        return found

    async def admit(
        self,
        args: Mapping[str, Any],
        *,
        read_view: ReadView,
        read_stored: ReadStored,
        read_window: ReadWindow,
    ) -> AdmittedWrite:
        """The arguments to send for the `write_component` call `args`.

        Args:
            args: The call's arguments: `intent`, and `component_id` on a revise.
            read_view: Reads a frame's view (`describe_frame`).
            read_stored: Reads a component's stored intent (`describe_component`).
            read_window: Reads the window a dataset frame's definition asks for.

        Raises:
            CardWithheld: A rule refuses the write. Nothing is to be sent.
        """
        intent = args.get("intent")
        if not isinstance(intent, Mapping):
            return AdmittedWrite(args=dict(args))
        component_id = args.get("component_id")
        if component_id is not None:
            card = await self._check_revise(
                str(component_id),
                intent,
                read_view=read_view,
                read_stored=read_stored,
                read_window=read_window,
            )
            return AdmittedWrite(args=dict(args), stored_card=_resolved_later(card))
        ref = intent.get("frame_ref")
        frame = await self.frame(ref) if isinstance(ref, Mapping) else None
        if frame is None:
            _refuse_closed_window(intent, read_window)
            return AdmittedWrite(args=dict(args), stored_card=_resolved_later(intent))
        _refuse_title(intent.get("title"), frame)
        target = await self._frame_for(
            frame, select=intent.get("select") or {}, view=await read_view(frame.ref.model_dump())
        )
        if target == frame.ref:
            return AdmittedWrite(args=dict(args), stored_card=_resolved_later(intent))
        # The card is drawn on the replayed frame, whose periods may be others.
        replayed = await self.frame(target.model_dump())
        if replayed is not None:
            _refuse_title(intent.get("title"), replayed)
        return AdmittedWrite(
            args={**args, "intent": {**intent, "frame_ref": target.model_dump()}},
            replayed_from=frame.ref.model_dump(),
            stored_card=_resolved_later(intent),
        )

    async def _check_revise(
        self,
        component_id: str,
        patch: Mapping[str, Any],
        *,
        read_view: ReadView,
        read_stored: ReadStored,
        read_window: ReadWindow,
    ) -> Mapping[str, Any] | None:
        """Refuse the revise `patch` of `component_id` when the card it leaves
        breaks a rule, and return that card: the stored intent with `patch`
        merged in, or `patch` alone when the stored card cannot be read. A
        patch that can break no rule is not read against the stored intent,
        and gives `None`."""
        moves_periods = not _PERIOD_MEMBERS.isdisjoint(patch)
        names_digits = _YEAR.search(str(patch.get("title") or "")) is not None
        if not (moves_periods or names_digits):
            return None
        body = await read_stored(component_id)
        stored = body.get("stored_intent") if body is not None else None
        if body is None or not isinstance(stored, Mapping):
            # The platform answers a revise of a component it cannot read.
            return patch
        # Top-level members replace whole and `null` deletes, as the platform merges a revise.
        merged = {key: value for key, value in {**stored, **patch}.items() if value is not None}
        ref = merged.get("frame_ref")
        frame = await self.frame(ref) if isinstance(ref, Mapping) else None
        if frame is None:
            _refuse_closed_window(merged, read_window)
        else:
            _refuse_title(merged.get("title"), frame)
        if not moves_periods:
            facts = body.get("fact_summary")
            window = facts.get("window") if isinstance(facts, Mapping) else None
            refusal = written_title_refusal(
                merged,
                title=merged.get("title"),
                window=window if isinstance(window, Mapping) else None,
                basis=resolved_basis(body.get("resolved_payload")),
            )
            if refusal is not None:
                raise CardWithheld(refusal)
        if frame is None or merged.get("select") == stored.get("select"):
            return merged
        target = await self._frame_for(
            frame, select=merged.get("select") or {}, view=await read_view(frame.ref.model_dump())
        )
        if target != frame.ref:
            raise CardWithheld(
                harness_refusal(
                    code=HarnessCode.ROLLING_SELECT_NEEDS_NEW_FRAME, message=_NEW_FRAME_MESSAGE
                )
            )
        return merged

    async def _frame_for(
        self, frame: RollingFrame, *, select: Mapping[str, Any], view: Mapping[str, Any] | None
    ) -> FrameRef:
        """The pair a card with `select` is written on: `frame`'s own, or that of
        its recipe replayed over the series the card draws.

        Raises:
            CardWithheld: The card's populations cannot be built, a root frame
                cannot be read for its columns, or the replay gave no frame.
        """
        populations = await self._populations(frame, select=select, view=view or {})
        if populations == frame.populations:
            return frame.ref
        key = (frame.ref.spec_id, frame.ref.content_version, rfc8785.dumps(populations))
        if key in self._replayed:
            self._replayed.move_to_end(key)
            return self._replayed[key]
        if (held := self._withheld.get(key)) is not None:
            if self._clock() < held[0]:
                raise CardWithheld(held[1])
            del self._withheld[key]
        running = self._running.get(key)
        if running is None:
            running = asyncio.create_task(self._replay_once(frame, populations))
            self._running[key] = running
            running.add_done_callback(lambda _: self._running.pop(key, None))
        try:
            # Shielded, so a caller that goes away leaves the replay to the
            # others waiting on it; it ends itself at its hard stop.
            replayed = await asyncio.shield(running)
        except CardWithheld as withheld:
            now = self._clock()
            # Refusals whose wait has passed go, so the map holds only live ones.
            self._withheld = {kept: held for kept, held in self._withheld.items() if now < held[0]}
            self._withheld[key] = (now + withheld.held_for_s, withheld.refusal)
            raise
        self._replayed[key] = replayed
        while len(self._replayed) > MAX_REPLAYS_KEPT:
            self._replayed.popitem(last=False)
        return replayed

    async def _populations(
        self, frame: RollingFrame, *, select: Mapping[str, Any], view: Mapping[str, Any]
    ) -> list[dict[str, Any]]:
        bound = frozenset(
            column for population in frame.populations for column in population["bindings"]
        )
        try:
            return anchor_populations(
                frame.recipe["tree"],
                frame.recipe["roots"],
                root_columns=await self._columns(frame),
                selection=card_selection(select, view=view, bound=bound),
            )
        except (PopulationEmptyError, PopulationLevelColumnMissingError) as exc:
            raise _withheld(HarnessCode.ROLLING_REPLAY_REFUSED, _REFUSED_MESSAGE, exc) from exc

    async def _columns(self, frame: RollingFrame) -> list[frozenset[str]]:
        """The column names of each root frame of `frame`'s recipe, in its
        `roots` order, read once for each rolling frame."""
        key = (frame.ref.spec_id, frame.ref.content_version)
        if key not in self._root_columns:
            self._root_columns[key] = [
                await self._columns_of(FrameRef.model_validate(root["frame"]))
                for root in frame.recipe["roots"]
            ]
        return self._root_columns[key]

    async def _columns_of(self, root: FrameRef) -> frozenset[str]:
        try:
            held = await get_store().open(root)
            if held is None or held.num_rows is None:
                raise _withheld(
                    HarnessCode.ROLLING_REPLAY_REFUSED,
                    _REFUSED_MESSAGE,
                    f"the store holds no countable root frame {root.spec_id}",
                )
            with row_budget().claim(held.num_rows):
                read, _ = await held.read(max_rows=held.num_rows)
        except RowBudgetExhausted as exc:
            raise _withheld(HarnessCode.ROLLING_REPLAY_PENDING, _PENDING_MESSAGE, exc) from exc
        except (FrameExpired, FrameIntegrityError, FrameTooLarge) as exc:
            raise _withheld(HarnessCode.ROLLING_REPLAY_REFUSED, _REFUSED_MESSAGE, exc) from exc
        return frozenset(read.table.column_names)

    async def _replay_once(
        self, frame: RollingFrame, populations: list[dict[str, Any]]
    ) -> FrameRef:
        """The pair of `frame`'s recipe replayed over `populations`.

        Raises:
            CardWithheld: No replayer is configured, or the replay ended
                `Pending`, `Refused`, past its hard stop, or on a failed call.
        """
        if self._replay is None:
            raise _withheld(
                HarnessCode.ROLLING_REPLAY_NOT_CONFIGURED,
                _NOT_CONFIGURED_MESSAGE,
                "no replayer was given",
            )
        try:
            async with asyncio.timeout(self._deadline_s + _HARD_STOP_GRACE_S):
                outcome = await self._replay(
                    frame.recipe,
                    deadline=CallDeadline.after(self._deadline_s),
                    populations=populations,
                )
        except TimeoutError as exc:
            raise _withheld(
                HarnessCode.ROLLING_REPLAY_PENDING,
                _PENDING_MESSAGE,
                "the replay ran past its hard stop",
            ) from exc
        except LiveApisNotConfigured as exc:
            raise _withheld(
                HarnessCode.ROLLING_REPLAY_NOT_CONFIGURED, _NOT_CONFIGURED_MESSAGE, exc
            ) from exc
        except (LiveApisError, ToolSurfaceUnavailableError) as exc:
            raise _withheld(HarnessCode.ROLLING_REPLAY_REFUSED, _REFUSED_MESSAGE, exc) from exc
        if isinstance(outcome, Pending):
            raise _withheld(
                HarnessCode.ROLLING_REPLAY_PENDING,
                _PENDING_MESSAGE,
                outcome,
                held_for_s=outcome.retry_after_s,
            )
        if isinstance(outcome, Refused):
            raise _withheld(HarnessCode.ROLLING_REPLAY_REFUSED, _REFUSED_MESSAGE, outcome)
        return outcome.frame

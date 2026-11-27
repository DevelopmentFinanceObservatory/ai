"""Whether a card's unit, years or places can change, and the pieces that change it.

A card can change when it draws the frame of one `get_data` retrieval: its
frame is that retrieval's own, not a frame an analysis step derived from it.
The card's `provenance.inputs[0]` names the spec the retrieval served. A
retrieval that named a `scope` is served under a spec of its own, and the
card reads the spec Search answered (its Search block's) with that scope
(`UnitCard.scope`), read back from the served definition (`unit_card`).

The unit can change when Search converts the spec (`resolved.unit_conversion`,
read by `harness.unit_offer.unit_offer`). The years can change when the card
draws more than one and the spec holds data over more than one
(`card_changes`). The places of a role can change when the role binds places;
the places it can change to are read from Search only when asked
(`harness.place_options`).

A unit change resubmits the spec in the new unit
(`harness.unit_switch.resubmit_unit`). A change of years or places names the
new `scope`. Either fetches the spec with the arguments that shaped the card's
frame and its scope, and redraws the card on the new frame in its slot
(`harness.studio_edit.change_events`).
Those arguments are read off the retrieval's manifest, whose data artifact
names the rendering apis minted (`frame_arguments`). A rendering this module
cannot map back to `get_data`'s arguments leaves the card without a change.

A card on a derived frame can change its years and places when the frame is
one sum of one retrieval by year: its lineage is that one retrieval, and the
recipe dfo-apis' analysis runner stored with the frame is one `aggregate` node
with `aggregation` `sum` over that retrieval's frame, no rolling anchor, the
period among its `group_by` columns, and no parameter outside the four the
verb takes (`summed_frame`, read when the card is drawn by
`read_summed_frame`). Recognition reads only that lineage and recipe. A change
fetches the retrieval over the new scope, calls `aggregate` again with the
parameters the recipe recorded and the new frame, and redraws the card on the
sum (`harness.studio_edit.change_events`). Its unit cannot change. A role whose
frame column the sum keeps (`group_by`) changes as a plain card's does. A
role whose column the sum adds up changes only while the card shows one place
for it, one place at a time, so the sum still runs over that one place; with
several places shown it offers no change, since other places would give
another sum under the same title. Every other derived frame offers no change.

`UnitCard` is what the conversation keeps with such a card, under
`CARD_UNIT_KEY` on its stored entry: the spec_id it reads, those arguments and
its scope. Each change carries the scope on, so a change of one axis keeps
what an earlier change set on another. No model reads it. A card on a sum is
stored with its `get_data` arguments under `summed`, beside the sum it
replays, so a release that reads no sum finds no arguments and offers the card
no change.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from analysis_mcp.identity import OP_SPEC_HASH_PREFIX
from analysis_mcp.runner_client import DerivedUnread
from common.materialisation import FrameRef
from common.materialisation.frame_name import KIND_COLUMNS, WITHOUT_PARTY_TOTALS
from common.materialisation.rendering import DEFAULT_MINT_OPTIONS
from common.place_type import FLOW_ROLE_COLUMNS, IDENTITY_KEY_AXES, PLACE_ROLE
from data_mcp.frame_materialisation import complete_rendering
from dfo_frame_grammar.grammar import TIME_PERIOD_COLUMN

from harness.chart_binding import chart_draws
from harness.derived_frames import derived_recipe
from harness.place_options import (
    PlaceOptionsReader,
    PlacesNotServedError,
    PlacesUnavailableError,
    RolePlaces,
    read_role_places,
)
from harness.provenance import CardLineage, ProvenanceInput, data_artifact
from harness.turn_records import SearchReading
from harness.unit_offer import offered_unit_refusal, served_unit, unit_offer, unit_words
from harness.wire import (
    ApplyChange,
    ApplyUnit,
    CardChanges,
    PlaceOption,
    PlacesChange,
    PlacesOffer,
    RoleLabel,
    UnitChange,
    UnitChoice,
    UnitOffer,
    UnitPrices,
    WindowChange,
    WindowOffer,
)

if TYPE_CHECKING:
    from harness.studio_state import ConversationState

# The key of a stored component entry holding its `UnitCard` as JSON. No
# digest, listing or recall states it (`harness.studio_state._model_entry`).
CARD_UNIT_KEY = "unit_card"

logger = logging.getLogger(__name__)

# The `options.groupby` `get_data` sends for `stat_vars: "sum"`
# (`data_mcp.get_data_tool._SUM_STAT_VARS_GROUPBY`).
_SUM_GROUPBY = {"by": ["time_period"], "op": "sum"}


# The parameters `aggregate` takes besides its frame
# (`analysis_mcp.op_spec.AggregateOpSpec`). A recorded step naming another is
# one this module cannot call again as recorded.
_AGGREGATE_PARAMETERS = frozenset({"aggregation", "group_by", "unit", "allow_sparse_breakdowns"})

# The members of a root's recorded mint scope a change's `get_data` scope
# states again (`data_mcp.get_data_tool.GetDataScope`).
_ROOT_SCOPE_MEMBERS = frozenset({"entities", "window", "facets"})


@dataclass(frozen=True)
class SummedFrame:
    """The one `aggregate` call a card's derived frame was computed by, as
    dfo-apis' analysis runner recorded it in the frame's recipe: the frame it
    gave (`frame`), the retrieval frame it read, as the runner recorded it
    (`root`), and the pair the data API serves that retrieval under
    (`served`), each as `(spec_id, content_version)`, and the parameters it
    was called with besides the frame, as recorded (`summed_frame`). For a
    frame minted with extra columns, `root`, the served pair and the frame
    key the card's lineage holds the retrieval under all differ. Only the
    served pair names the retrieval on both sides, and only it can be
    described (`ProvenanceInput.served_ref`)."""

    frame: tuple[str, str]
    root: tuple[str, str]
    served: tuple[str, str]
    parameters: dict[str, Any]

    @property
    def group_by(self) -> tuple[str, ...]:
        """The frame columns the sum keeps apart."""
        return tuple(self.parameters.get("group_by") or ())

    def as_json(self) -> dict[str, Any]:
        return {
            "frame": list(self.frame),
            "root": list(self.root),
            "served": list(self.served),
            "parameters": dict(self.parameters),
        }

    @classmethod
    def from_json(cls, stored: Any) -> SummedFrame | None:
        """The `SummedFrame` `as_json` wrote, or `None` for anything else."""
        if not isinstance(stored, Mapping):
            return None
        frame, root = _pair(stored.get("frame")), _pair(stored.get("root"))
        served, parameters = _pair(stored.get("served")), stored.get("parameters")
        if frame is None or root is None or served is None or not isinstance(parameters, Mapping):
            return None
        return cls(frame=frame, root=root, served=served, parameters=dict(parameters))


def _pair(value: Any) -> tuple[str, str] | None:
    """`value` as a `(spec_id, content_version)` pair: a two-item list or a
    `{spec_id, content_version}` mapping of non-empty strings."""
    if isinstance(value, Mapping):
        value = [value.get("spec_id"), value.get("content_version")]
    if isinstance(value, list | tuple) and len(value) == 2:
        spec_id, content_version = value
        if isinstance(spec_id, str) and isinstance(content_version, str):
            return (spec_id, content_version) if spec_id and content_version else None
    return None


def summed_frame(  # noqa: PLR0911 -- one `None` per way a recipe is not one sum by year
    recipe: Mapping[str, Any], frame: tuple[str, str]
) -> SummedFrame | None:
    """The sum `recipe`, the recipe dfo-apis' analysis runner stored for the
    derived frame `frame` (`harness.derived_frames.derived_recipe`), records:
    `None` unless its tree is one `aggregate` node with `aggregation` `sum`
    whose one input is the root node of its one root (`{"frame", "root"}` in
    `common.recipe`), its `group_by` keeps the period, it names no parameter
    `aggregate` does not take, it pins no unit but by ref (a change binds the
    pinned unit on its refetch, `root_facets`), it has no rolling anchor, and
    its root was minted with no scope member a `get_data` scope does not
    state, under a stored pair (`frame`) and a served one (`dataset`)."""
    tree, roots = recipe.get("tree"), recipe.get("roots")
    if recipe.get("anchor") is not None or not isinstance(tree, Mapping):
        return None
    if not isinstance(roots, list) or len(roots) != 1 or not isinstance(roots[0], Mapping):
        return None
    op_spec, inputs = tree.get("op_spec"), tree.get("inputs")
    if not isinstance(op_spec, Mapping) or op_spec.get("verb") != "aggregate":
        return None
    if not isinstance(inputs, Mapping) or set(inputs) != {"frame_ref"}:
        return None
    read = inputs["frame_ref"]
    if not isinstance(read, Mapping) or set(read) - {"frame"} != {"root"} or read["root"] != 0:
        return None
    parameters = op_spec.get("parameters")
    if not isinstance(parameters, Mapping) or set(parameters) - _AGGREGATE_PARAMETERS:
        return None
    group_by = parameters.get("group_by")
    if (
        parameters.get("aggregation") != "sum"
        or not isinstance(parameters.get("unit"), str | None)
        or not isinstance(group_by, list)
        or not all(isinstance(column, str) for column in group_by)
        or TIME_PERIOD_COLUMN not in group_by
    ):
        return None
    (root,) = roots
    scope = root.get("scope") or {}
    pair, served = _pair(root.get("frame")), _pair(root.get("dataset"))
    if pair is None or served is None or ("frame" in read and _pair(read["frame"]) != pair):
        return None
    if not isinstance(scope, Mapping) or set(scope) - _ROOT_SCOPE_MEMBERS:
        return None
    return SummedFrame(frame=frame, root=pair, served=served, parameters=dict(parameters))


async def read_summed_frame(lineage: CardLineage | None) -> SummedFrame | None:
    """The sum a card drawn from `lineage` shows (`summed_frame`), read from
    the recipe dfo-apis' analysis runner stored with its frame: one read, made
    only for a derived (`ops_`) frame resting on one retrieval, which must be
    the sum's root. The two are matched by the pair the data API serves the
    retrieval under (`SummedFrame.served`), since the runner records the
    frame it read under a pair of its own, which differs from the lineage's
    frame key and from the store's pair. `None` for any other card, and when the
    recipe cannot be read, so such a card offers no change."""
    if lineage is None or len(lineage.lineage) != 1:
        return None
    frame = _pair(list(lineage.frame))
    if frame is None or not frame[0].startswith(OP_SPEC_HASH_PREFIX):
        return None
    try:
        recipe = await derived_recipe(FrameRef(spec_id=frame[0], content_version=frame[1]))
    except DerivedUnread as exc:
        logger.info("the recipe of %s was not read, so its card offers no change: %s", frame, exc)
        return None
    if recipe is None:
        return None
    summed = summed_frame(recipe, frame)
    (member,) = lineage.lineage
    if summed is None or summed.served != _pair(member.served_ref):
        return None
    return summed


def role_column(role: str) -> str:
    """The frame column the places of an identity's `role` are served in: a
    flow axis's party column, the place column, or the role's own name."""
    axis = IDENTITY_KEY_AXES.get(role)
    if axis == "place":
        return PLACE_ROLE
    return FLOW_ROLE_COLUMNS[axis] if axis is not None else role


@dataclass(frozen=True)
class UnitCard:
    """A card that can change: the spec_id its Search block answers, the
    `get_data` arguments, `spec_id` and `scope` aside, that shaped its frame,
    and the `scope` its retrieval named over that spec (`get_data`'s own
    shape, `entities` naming each place `dcid:`-marked), `None` for none,
    with the label of each place a change of places set (`labels`, by ref),
    which the spec's Search block does not name.

    `single`, `drawn_year`, `roles`, `shown`, `facets` and `cut` are the
    card's encoding, fixed when the card is drawn (`encoded`): whether it
    shows one value per series, so its years change one at a time, and the
    one year such a card draws, as the platform's drawn payload states it;
    the roles whose places it can change, each with whether it takes
    several, and the most places each can draw (`most`, the chart's own
    limit, `harness.chart_binding.chart_draws`); the places each role
    draws, by ref, which can be fewer than its frame holds (`drawn_places`); the facet each of `FACET_AXES` the drawn
    series share, by axis, which a change of years or places binds so the
    data API elects none again (`drawn_facets`); and whether its members are
    the largest ones `make_charts` ranked at one period, so neither its years
    nor its places can change, since another year or place has other largest
    members. A card stored before they were kept has `roles` `None`, and
    reads its roles off the places it shows, `shown` and `facets` `None`, and
    `cut` false.

    `years` are the first and last year, and the years between with no
    figures, that the card's spec holds over the places its scope names,
    read from Search after a change of places (`ScopeYears`). The Search
    block states the years of its own places only, so a card whose scope
    names other places offers years only from `years`, and only while they
    describe its spec and places.

    `nouns` is what the card's served data calls each role's places, by
    role: `singular`, `plural` or both, as its retrieval's coverage states
    them (`role_nouns`); `None` where it states none.

    `total` is the sum the card's frame is, for a card on a sum of its one
    retrieval (`SummedFrame`), `None` for a card on the retrieval's own
    frame.

    `facets_unread` marks a card whose fact summary listed only some of its
    series when it was drawn, so the facets it was counted in are not known
    (`drawn_facets`). A card on the retrieval's own frame then offers no
    change of years or places, since nothing would keep a change on those
    facets. A card on a sum reads them from its retrieval's frame when it
    changes (`harness.studio_edit`)."""

    spec_id: str
    args: dict[str, Any]
    scope: dict[str, Any] | None = None
    labels: dict[str, str] | None = None
    single: bool = False
    drawn_year: int | None = None
    roles: dict[str, bool] | None = None
    cut: bool = False
    shown: dict[str, list[str]] | None = None
    facets: dict[str, str] | None = None
    years: ScopeYears | None = None
    most: dict[str, int] | None = None
    nouns: dict[str, dict[str, str]] | None = None
    total: SummedFrame | None = None
    facets_unread: bool = False

    def as_json(self) -> dict[str, Any]:
        stored: dict[str, Any] = {"spec_id": self.spec_id, "args": dict(self.args)}
        if self.scope:
            stored["scope"] = dict(self.scope)
        if self.labels:
            stored["labels"] = dict(self.labels)
        if self.single:
            stored["single"] = True
        if self.drawn_year is not None:
            stored["drawn_year"] = self.drawn_year
        if self.roles is not None:
            stored["roles"] = dict(self.roles)
        if self.cut:
            stored["cut"] = True
        if self.shown is not None:
            stored["shown"] = {role: list(refs) for role, refs in self.shown.items()}
        if self.facets:
            stored["facets"] = dict(self.facets)
        if self.years is not None:
            stored["years"] = self.years.as_json()
        if self.most is not None:
            stored["most"] = dict(self.most)
        if self.nouns:
            stored["nouns"] = {role: dict(forms) for role, forms in self.nouns.items()}
        if self.facets_unread:
            stored["facets_unread"] = True
        if self.total is not None:
            stored["summed"] = {"args": stored.pop("args"), **self.total.as_json()}
        return stored

    @classmethod
    def from_json(cls, stored: Any) -> UnitCard | None:
        """The `UnitCard` `as_json` wrote, or `None` for anything else."""
        if not isinstance(stored, Mapping):
            return None
        spec_id, args = stored.get("spec_id"), stored.get("args")
        summed = stored.get("summed")
        total = SummedFrame.from_json(summed)
        if summed is not None:
            args = summed.get("args") if total is not None else None
        if not isinstance(spec_id, str) or not spec_id or not isinstance(args, Mapping):
            return None
        scope, labels, roles = stored.get("scope"), stored.get("labels"), stored.get("roles")
        shown, facets, most = stored.get("shown"), stored.get("facets"), stored.get("most")
        return cls(
            spec_id=spec_id,
            args=dict(args),
            scope=dict(scope) if isinstance(scope, Mapping) and scope else None,
            labels={str(k): str(v) for k, v in labels.items()}
            if isinstance(labels, Mapping) and labels
            else None,
            single=stored.get("single") is True,
            drawn_year=stored["drawn_year"] if type(stored.get("drawn_year")) is int else None,
            cut=stored.get("cut") is True,
            roles={str(k): v is True for k, v in roles.items()}
            if isinstance(roles, Mapping)
            else None,
            shown={
                str(role): [str(ref) for ref in refs]
                for role, refs in shown.items()
                if isinstance(refs, list)
            }
            if isinstance(shown, Mapping)
            else None,
            facets={str(k): v for k, v in facets.items() if isinstance(v, str)}
            if isinstance(facets, Mapping) and facets
            else None,
            years=ScopeYears.from_json(stored.get("years")),
            most={str(k): v for k, v in most.items() if type(v) is int and v >= 1}
            if isinstance(most, Mapping)
            else None,
            nouns=_read_nouns(stored.get("nouns")),
            total=total,
            facets_unread=stored.get("facets_unread") is True,
        )

    def fetch_args(
        self,
        spec_id: str,
        scope: Mapping[str, Any] | None,
        *,
        facets: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """The `get_data` arguments that fetch `spec_id` as this card's frame
        was fetched, over `scope`, binding `facets` too when given
        (`bound_facets`)."""
        if facets:
            scope = {**(scope or {}), "facets": dict(facets)}
        return {"spec_id": spec_id, **self.args, **({"scope": dict(scope)} if scope else {})}


@dataclass(frozen=True)
class ScopeYears:
    """The years a spec holds over the places a card's scope names, as Search
    stated them (`harness.unit_switch.scope_years`): the first and last year
    with data, the years between with none, and the spec and places they
    describe (`entities`, by role, refs without the `dcid:` mark)."""

    spec_id: str
    entities: dict[str, list[str]]
    first: int
    last: int
    gaps: tuple[int, ...] = ()

    def as_json(self) -> dict[str, Any]:
        return {
            "spec_id": self.spec_id,
            "entities": {role: list(refs) for role, refs in self.entities.items()},
            "first": self.first,
            "last": self.last,
            **({"gaps": list(self.gaps)} if self.gaps else {}),
        }

    @classmethod
    def from_json(cls, stored: Any) -> ScopeYears | None:
        """The `ScopeYears` `as_json` wrote, or `None` for anything else."""
        if not isinstance(stored, Mapping):
            return None
        spec_id, entities = stored.get("spec_id"), stored.get("entities")
        first, last = stored.get("first"), stored.get("last")
        if (
            not isinstance(spec_id, str)
            or not isinstance(entities, Mapping)
            or type(first) is not int
            or type(last) is not int
        ):
            return None
        return cls(
            spec_id=spec_id,
            entities={
                str(role): [str(ref) for ref in refs]
                for role, refs in entities.items()
                if isinstance(refs, list)
            },
            first=first,
            last=last,
            gaps=tuple(year for year in stored.get("gaps") or () if type(year) is int),
        )


# The facet axes a frame carries a column for and a `get_data` scope binds
# (`data_mcp.get_data_tool.GetDataScope.facets`).
FACET_AXES = ("provenance", "measurementMethod", "unit")


def bound_facets(card: UnitCard, block: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """The `facets` a fetch of `card`'s spec over a new scope binds, so the
    data API serves the facets the card was drawn with (`UnitCard.facets`)
    and elects none again: each axis the spec's identity binds by ref, as it
    binds it, and each other drawn facet by its ref. A scope's `facets`
    replaces the identity's whole, and holds refs only, so a spec whose
    identity binds a facet otherwise, as a unit request for constant prices
    does, binds none here, and its redraw is checked instead
    (`strayed_facets`). `None` when nothing is bound."""
    identity = ((block or {}).get("definition") or {}).get("identity") or {}
    own: dict[str, str] = {}
    for axis, binding in (identity.get("facets") or {}).items():
        ref = binding.get("ref") if isinstance(binding, Mapping) else None
        if not isinstance(ref, str) or set(binding) - {"state", "ref"}:
            return None
        own[str(axis)] = ref
    refs = {**(card.facets or {}), **own}
    return {axis: {"ref": ref} for axis, ref in refs.items()} or None


def drawn_facets(fact_summary: Mapping[str, Any] | None) -> dict[str, str]:
    """The facet each of `FACET_AXES` every series `fact_summary` lists shares,
    as the series' coordinates name it, by axis: the facets the data API
    elected for the card. An axis the series name more than one value of, as
    a series continuing across methods does, and one they name none of, is
    left out, since binding it would drop part of what the card draws. A
    summary that lists only some series (`series_truncated`) gives none,
    since a series it leaves out may name another facet."""
    if (fact_summary or {}).get("series_truncated"):
        return {}
    series = [s for s in (fact_summary or {}).get("series") or () if isinstance(s, Mapping)]
    found: dict[str, str] = {}
    for axis in FACET_AXES:
        values = {(s.get("coordinates") or {}).get(axis) for s in series if isinstance(s, Mapping)}
        if len(values) == 1 and isinstance(ref := next(iter(values)), str) and ref:
            found[axis] = ref
    return found


def facet_values(view: Mapping[str, Any]) -> dict[str, list[str]]:
    """The values a frame view (`describe_frame`) lists on each axis of
    `FACET_AXES` it states, by axis, refs without the `dcid:` mark, with an
    empty value for each one its `total_count` counts beyond those listed."""
    held: dict[str, list[str]] = {}
    for dimension in view.get("dimensions") or ():
        if isinstance(dimension, Mapping) and dimension.get("name") in FACET_AXES:
            values = [str(v).removeprefix("dcid:") for v in dimension.get("values") or ()]
            count = dimension.get("total_count")
            if type(count) is int and count > len(values):
                values += [""] * (count - len(values))
            held[str(dimension["name"])] = values
    return held


def _pinned_unit(card: UnitCard) -> str | None:
    """The unit ref the sum under `card` pins, without the `dcid:` mark;
    `None` for no pin, and for a converted or composed unit, which binds by
    no ref."""
    pinned = card.total.parameters.get("unit") if card.total is not None else None
    return pinned.removeprefix("dcid:") if isinstance(pinned, str) else None


def root_facets(card: UnitCard, view: Mapping[str, Any]) -> dict[str, str] | None:
    """The facets `card`, a card on a sum, binds on a change: the one value
    each facet axis of its retrieval's frame view (`view`) holds, the unit
    the sum pins (`aggregate` sums that unit's rows alone, so its retrieval
    may hold others), and the facets the card drew. It returns `None` when
    two of these name different values on an axis. `aggregate` refuses a
    retrieval holding several values of any other axis, so a recorded sum's
    retrieval holds one on each."""
    found = {
        axis: values[0]
        for axis, values in facet_values(view).items()
        if len(set(values)) == 1 and values[0]
    }
    if (pinned := _pinned_unit(card)) is not None:
        if found.get("unit", pinned) != pinned:
            return None
        found["unit"] = pinned
    drawn = {axis: ref.removeprefix("dcid:") for axis, ref in (card.facets or {}).items()}
    if any(axis in found and found[axis] != ref for axis, ref in drawn.items()):
        return None
    return {**found, **(card.facets or {})}


def unlike_series(card: UnitCard, view: Mapping[str, Any]) -> str | None:
    """Why the retrieval whose frame view is `view` cannot be summed as the
    card on a sum `card` was: `"mixed"` when an axis of `FACET_AXES` holds
    more than one value, so the sum would add unlike figures; `"other"` when
    it holds one value other than the one `card` was drawn with on that axis
    (`UnitCard.facets`), or than the unit the recorded sum pins, so the sum
    would carry another label. `None` when every axis holds at most one value
    and each is the card's. An axis the view lists no value of passes, and
    `aggregate` still refuses several units it was given no pin for."""
    held = facet_values(view)
    if any(len(set(values)) > 1 for values in held.values()):
        return "mixed"
    expected = {axis: ref.removeprefix("dcid:") for axis, ref in (card.facets or {}).items()}
    if (pinned := _pinned_unit(card)) is not None:
        expected.setdefault("unit", pinned)
    for axis, values in held.items():
        if values and axis in expected and values[0] != expected[axis]:
            return "other"
    return None


def strayed_facets(card: UnitCard, fact_summary: Mapping[str, Any] | None) -> list[str]:
    """The axes of `card`'s bound facets on which a series `fact_summary`
    lists names another facet: what a redraw fetched with the facets bound
    must never show."""
    series = [s for s in (fact_summary or {}).get("series") or () if isinstance(s, Mapping)]
    return [
        axis
        for axis, ref in (card.facets or {}).items()
        if any(
            (named := (s.get("coordinates") or {}).get(axis)) is not None and named != ref
            for s in series
        )
    ]


def one_value(intent: Mapping[str, Any] | None) -> bool:
    """Whether a card drawn from `intent` shows one value per series: a stat
    card, or a bar chart whose select lists one of its dimensions, which
    `make_charts` draws at one period (`orchestrator_tools._card_period`)."""
    if not isinstance(intent, Mapping):
        return False
    select = intent.get("select")
    return intent.get("type") == "stat_card" or (
        intent.get("type") == "bar_chart"
        and isinstance(select, Mapping)
        and any(isinstance(value, list) for value in select.values())
    )


def _naming(select: Mapping[str, Any], refs: Sequence[str]) -> set[str]:
    """The `select` entries naming any of `refs`, `dcid:`-marked or not."""
    shown = set(refs)
    return {
        name
        for name, value in select.items()
        if any(
            isinstance(v, str) and v.removeprefix("dcid:") in shown
            for v in (value if isinstance(value, list) else [value])
        )
    }


def select_entry(select: Mapping[str, Any], role: str) -> str | None:
    """The key of `select`, a chart's select, that names `role`'s places. A
    select is keyed by the frame's dimensions, so the key is the frame column
    the role is served in (`role_column`), else the role's own name. It
    returns `None` when the select names neither."""
    for key in dict.fromkeys((role_column(role), role)):
        if key in select:
            return key
    return None


def _role_entries(select: Mapping[str, Any], role: str | None, refs: Sequence[str]) -> set[str]:
    """The `select` entries naming a role's places: the role's own entry
    (`select_entry`), none where the select leaves its column unbound, and,
    for a role not named, every entry naming any of `refs`."""
    if role is None:
        return _naming(select, refs)
    key = select_entry(select, role)
    return {key} if key is not None else set()


def takes_several(
    intent: Mapping[str, Any] | None,
    refs: Sequence[str],
    others: Sequence[Sequence[str]] = (),
    *,
    role: str | None = None,
    other_roles: Sequence[str] = (),
) -> bool:
    """Whether a role showing places `refs` on the card drawn from `intent`
    can show several, each drawn as its own series, as dfo-apis' assembler
    draws a select (dfo-data-tools' `assembler/resolve.py` and
    `pipeline/candidate.py`). A select entry naming a role's places as a list
    draws one series per place, and so does a place column the select leaves
    unbound. A stat card draws one value, and a table draws any select. A line
    chart takes several unless its select lists another dimension (beside a
    role it leaves unbound, a list of two or more values), since the
    platform refuses a line or bar chart listing two, or another place role
    (`others`, the places each shows) already draws several series, being
    unbound or listed over several places. A bar chart takes several only
    where its listed dimension is the role's, one bar per place. A bar chart
    of one series draws its periods, and a second place would turn it into
    one bar per place at one period."""
    if not isinstance(intent, Mapping):
        return True
    kind = intent.get("type")
    if kind == "stat_card":
        return False
    if kind == "table":
        return True
    select = intent.get("select")
    select = select if isinstance(select, Mapping) else {}
    naming = _role_entries(select, role, refs)
    # The platform binds a list of one value as a listed dimension
    # (`harness.chart_binding.select_binding_ok`), which refuses a second
    # list beside a role's own entry, written as a list once it names several
    # places. A role the select leaves unbound is written no entry
    # (`harness.studio_edit._redrawn_intent`), so beside it a list of one
    # value draws one series per place.
    listed = {
        name
        for name, value in select.items()
        if isinstance(value, list) and (naming or len(value) > 1)
    }
    for index, other in enumerate(others):
        named = _role_entries(
            select, other_roles[index] if index < len(other_roles) else None, other
        )
        if len(other) > 1 and (not named or named & listed):
            return False
    if kind == "bar_chart":
        return bool(listed) and listed <= naming
    return listed <= naming


def drawn_year(component: Mapping[str, Any] | None) -> int | None:
    """The one year a card showing one value per series draws, as the
    platform's drawn payload states it: a stat card's `period`, and the
    `period` of a bar chart drawn one bar per member (`category_axis.mode`
    `entity`). `None` for any other card, or a period that is no year. The
    card's fact summary states no window for such a card."""
    drawn = ((component or {}).get("payload") or {}).get("component") or {}
    if not isinstance(drawn, Mapping):
        return None
    if drawn.get("type") == "stat_card":
        return _year(drawn.get("period"))
    axis = drawn.get("category_axis")
    if (
        drawn.get("type") == "bar_chart"
        and isinstance(axis, Mapping)
        and axis.get("mode") == "entity"
    ):
        return _year(axis.get("period"))
    return None


def encoded(
    card: UnitCard,
    block: Mapping[str, Any] | None,
    *,
    intent: Mapping[str, Any] | None,
    cut: bool | None = None,
    component: Mapping[str, Any] | None = None,
) -> UnitCard:
    """`card` with its encoding as the card drawn from `intent` fixes it:
    `single` when it shows one value per series (`one_value`), and as `roles`
    each role whose places it shows, the ones `card` already kept, else
    those `block` and its scope bind, each marked with whether it can show
    several places (`takes_several`). `cut` says whether `make_charts` cut
    the card to its largest members; `None` keeps the card's own. A card
    showing one value per series reads the year it draws off `component`,
    the written card (`drawn_year`), else keeps its own. The places each role
    draws are the ones `intent`'s `select` names of its frame's places, else
    all of them (`drawn_places`), and the facets are the ones `component`'s
    fact summary shows (`drawn_facets`), else the card's own."""
    held = chosen_places(card, block) if block is not None else {}
    if block is None:
        shown = dict(card.shown or {})
    elif intent is not None:
        shown = _selected_places(held, intent)
    else:
        shown = drawn_places(card, block)
    if card.roles is not None:
        bound: list[str] = list(card.roles)
    else:
        bound = [role for role, refs in shown.items() if refs]
    # A role the card's sum adds up draws no series of its own.
    drawing = {
        other: refs
        for other, refs in shown.items()
        if refs and (card.total is None or role_column(other) in card.total.group_by)
    }
    roles = {
        role: takes_several(
            intent,
            shown.get(role) or (),
            [refs for other, refs in drawing.items() if other != role],
            role=role,
            other_roles=[other for other in drawing if other != role],
        )
        for role in bound
    }
    single = one_value(intent)
    year = (drawn_year(component) if component is not None else card.drawn_year) if single else None
    summary = (component or {}).get("fact_summary") or {}
    facets = {
        **(card.facets or {}),
        **(drawn_facets(summary) if component is not None else {}),
    }
    unread = bool(summary.get("series_truncated")) if component is not None else card.facets_unread
    most = card.most
    if intent is not None:
        type_ = str(intent.get("type") or "")
        view = _drawn_view(component)
        # A table draws rows at each period, so one whose periods cannot be
        # counted states no limit for a role that takes several places.
        draws = (
            None if view is None and type_ == "table" else chart_draws(type_=type_, view=view or {})
        )
        most = {
            role: draws if several else 1
            for role, several in roles.items()
            if draws is not None or not several
        }
    return replace(
        card,
        single=single,
        drawn_year=year,
        roles=roles,
        cut=card.cut if cut is None else cut,
        shown=shown if block is not None else card.shown,
        facets=facets or None,
        most=most,
        facets_unread=unread,
    )


# A period `harness.chart_binding.periods_covered` counts: a year, a month or a day.
_COUNTED_PERIOD = re.compile(r"\d{4}(?:-\d{2}(?:-\d{2})?)?")


def _drawn_view(component: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """The part of a frame view `chart_draws` reads, for the card `component`
    draws: the first and last period its fact summary states, whole, which a
    table draws rows at, months and days counted as such. It is empty when
    the summary states none, and `None` when it states periods
    `periods_covered` cannot count, such as quarters, or two of different
    grains."""
    window = ((component or {}).get("fact_summary") or {}).get("window") or {}
    start, end = window.get("start"), window.get("end")
    if not isinstance(start, str) or not isinstance(end, str):
        return {}
    if (
        _COUNTED_PERIOD.fullmatch(start)
        and _COUNTED_PERIOD.fullmatch(end)
        and len(start) == len(end)
    ):
        return {"coverage": [start, end]}
    return None


def _selected_places(
    held: Mapping[str, list[str]], intent: Mapping[str, Any] | None
) -> dict[str, list[str]]:
    """The places each role of `held`, the card's frame places by role, draws
    on the card drawn from `intent`: the ones its `select` entry for the role
    names, in the entry's order, else every place the frame holds."""
    select = (intent or {}).get("select")
    select = select if isinstance(select, Mapping) else {}
    shown: dict[str, list[str]] = {}
    for role, refs in held.items():
        key = select_entry(select, role)
        value = select.get(key) if key is not None else None
        named = [
            v.removeprefix("dcid:")
            for v in (value if isinstance(value, list) else [value])
            if isinstance(v, str)
        ]
        picked = [ref for ref in dict.fromkeys(named) if ref in refs]
        shown[role] = picked or list(refs)
    return shown


def frame_arguments(rendering: Mapping[str, Any]) -> dict[str, Any] | None:
    """The `get_data` arguments, `spec_id` and `scope` aside, that mint
    `rendering`, a retrieval's data artifact rendering; `None` for one no call
    of `get_data` sends.

    `get_data` sends the default options (`DEFAULT_MINT_OPTIONS`), a groupby
    for `stat_vars: "sum"` or for a `breakdown`, the kind columns for
    `without_party_totals` and that option itself, and an enrichment for
    `place_types`, which the rendering names by the columns apis resolved it
    to: any column beyond the kind columns. A rendering apis linked after a
    failed metadata read (`metadata: "incomplete"`) maps as its complete one."""
    rest = complete_rendering(rendering)
    for key, value in DEFAULT_MINT_OPTIONS.items():
        if rest.pop(key, None) != value:
            return None
    args: dict[str, Any] = {}
    groupby = rest.pop("groupby", None)
    if groupby == _SUM_GROUPBY:
        args["stat_vars"] = "sum"
    elif groupby is not None:
        breakdown = _breakdown(groupby)
        if breakdown is None:
            return None
        args["breakdown"] = breakdown
    without_totals = rest.pop(WITHOUT_PARTY_TOTALS, False)
    if without_totals not in (True, False):
        return None
    if without_totals:
        args["without_party_totals"] = True
    columns = rest.pop("columns", None)
    if columns is not None:
        if not isinstance(columns, list) or not all(isinstance(c, str) for c in columns):
            return None
        if not set(columns) <= set(KIND_COLUMNS):
            args["place_types"] = True
        elif not without_totals:
            return None
    return None if rest else args


def _breakdown(groupby: Any) -> dict[str, Any] | None:
    """The `breakdown` argument whose `options.groupby` is `groupby`
    (`data_mcp.get_data_tool.Breakdown.groupby`), or `None`."""
    if not isinstance(groupby, Mapping) or set(groupby) != {"by", "op", "levels"}:
        return None
    levels = groupby["levels"]
    if groupby["op"] != "sum" or not isinstance(levels, Mapping) or len(levels) != 1:
        return None
    ((code_system, level),) = levels.items()
    if not isinstance(level, int) or groupby["by"] != sorted(["time_period", code_system]):
        return None
    return {"code_system": code_system, "level": level}


def unit_card(
    component: Mapping[str, Any],
    lineage: CardLineage | None,
    blocks: Mapping[str, Mapping[str, Any]] | None = None,
    *,
    total: SummedFrame | None = None,
) -> UnitCard | None:
    """The `UnitCard` of `component`, a written card, drawn from `lineage`;
    `None` when the card does not draw one retrieval's own frame, or the sum
    `total` of it (`read_summed_frame`), or its arguments cannot be read. A
    card on the sum reads the retrieval under it as a card on the
    retrieval's own frame does, and keeps the sum (`UnitCard.total`).

    The card reads the served spec when `blocks`, the Search blocks the
    conversation keeps by spec_id (`ConversationBlocks`), hold one for it.
    Otherwise it reads the spec the retrieval asked for
    (`ProvenanceInput.requested_spec_id`) when the served definition's
    identity matches that spec's block outside its places and window: with
    the places and window the served definition names as its scope
    (`_scope_over`), the places as Search resolved them, under the block's
    own roles, labelled by the names the retrieval resolved them by
    (`get_data`'s `entity_labels`). The served identity is the one source,
    since the model's `get_data` arguments name places as spoken, under its
    own role names, and may name facets the scope cannot carry. A retrieval
    recorded before the requested spec was kept reads the one block that
    matches so; with several, or none, it reads its own spec and offers no
    change. A served identity that differs in anything else, its facets
    among them, offers no change."""
    if lineage is None or len(lineage.lineage) != 1:
        return None
    (member,) = lineage.lineage
    served = member.served_ref
    own = lineage.frame in {
        (member.spec_id, member.content_version),
        (served["spec_id"], served["content_version"]),
    }
    summed = (
        total is not None
        and total.frame == lineage.frame
        and total.served == (served["spec_id"], served["content_version"])
    )
    if not own and not summed:
        return None
    inputs = (component.get("provenance") or {}).get("inputs") or []
    spec_id = served["spec_id"]
    drawn = spec_id if own else lineage.frame[0]
    if len(inputs) != 1 or not spec_id or inputs[0].get("spec_id") != drawn:
        return None
    card = _retrieval_card(member, spec_id, blocks)
    return replace(card, total=total) if card is not None and not own else card


def _retrieval_card(
    member: ProvenanceInput,
    spec_id: str,
    blocks: Mapping[str, Mapping[str, Any]] | None,
) -> UnitCard | None:
    """The `UnitCard` of a card reading retrieval `member`, served under
    `spec_id`, as `unit_card` reads it."""
    rendering = data_artifact(member.manifest.get("artifacts") or []).get("rendering")
    args = frame_arguments(rendering) if isinstance(rendering, Mapping) else None
    if args is None:
        return None
    nouns = role_nouns(member.manifest)
    if blocks is None or blocks.get(spec_id) is not None:
        return UnitCard(spec_id=spec_id, args=args, nouns=nouns)
    served_identity = (member.manifest.get("definition") or {}).get("identity")

    def scope_of(block: Mapping[str, Any] | None) -> dict[str, Any] | None:
        if block is None:
            return None
        return _scope_over((block.get("definition") or {}).get("identity"), served_identity)

    requested = member.requested_spec_id
    if requested is not None:
        matched = [(requested, scope)] if (scope := scope_of(blocks.get(requested))) else []
    else:
        matched = [
            (base_id, scope) for base_id, block in blocks.items() if (scope := scope_of(block))
        ]
    if len(matched) != 1:
        return UnitCard(spec_id=spec_id, args=args, nouns=nouns)
    ((base_id, scope),) = matched
    named = member.manifest.get("entity_labels")
    named = named if isinstance(named, Mapping) else {}
    scoped = [
        str(ref).removeprefix("dcid:")
        for refs in (scope.get("entities") or {}).values()
        for ref in refs
    ]
    labels = {
        ref: str(named[ref]) for ref in scoped if isinstance(named.get(ref), str) and named[ref]
    }
    return UnitCard(spec_id=base_id, args=args, scope=scope, labels=labels or None, nouns=nouns)


def role_nouns(manifest: Mapping[str, Any]) -> dict[str, dict[str, str]] | None:
    """What a retrieval's served data calls each role's places, by role, from
    its manifest's `coverage.dimensions`: each entry's `noun`, keyed by the
    flow axis it sits on (`axis`, `origin` or `destination`, the role a
    block names) and by its own `label`, keeping the forms it states as
    non-empty strings. `None` where no entry states one."""
    nouns: dict[str, dict[str, str]] = {}
    for dimension in (manifest.get("coverage") or {}).get("dimensions") or ():
        if not isinstance(dimension, Mapping):
            continue
        forms = _read_forms(dimension.get("noun"))
        if forms is None:
            continue
        for key in (dimension.get("label"), dimension.get("axis")):
            if isinstance(key, str) and key:
                nouns[key] = forms
    return nouns or None


def _read_forms(noun: Any) -> dict[str, str] | None:
    """The `singular` and `plural` of `noun` that are non-empty strings, `None`
    for neither."""
    if not isinstance(noun, Mapping):
        return None
    forms = {
        form: noun[form]
        for form in ("singular", "plural")
        if isinstance(noun.get(form), str) and noun[form].strip()
    }
    return forms or None


def _read_nouns(stored: Any) -> dict[str, dict[str, str]] | None:
    """The `UnitCard.nouns` `as_json` wrote, `None` for anything else."""
    if not isinstance(stored, Mapping):
        return None
    nouns = {str(role): forms for role, noun in stored.items() if (forms := _read_forms(noun))}
    return nouns or None


def _scope_over(base: Any, served: Any) -> dict[str, Any] | None:
    """The `get_data` scope that serves identity `served` over identity
    `base`: their places (`entities`) and window (`requested_window`) where
    they differ, `None` where the two differ elsewhere or either is unread.
    Slots compare in any order."""
    if not isinstance(base, Mapping) or not isinstance(served, Mapping):
        return None

    def rest(identity: Mapping[str, Any]) -> dict[str, Any]:
        kept = {k: v for k, v in identity.items() if k not in {"entities", "requested_window"}}
        slots = kept.get("slots")
        if isinstance(slots, list):
            kept["slots"] = sorted(
                slots,
                key=lambda slot: (
                    str(slot.get("property")) if isinstance(slot, Mapping) else str(slot)
                ),
            )
        return kept

    if rest(base) != rest(served):
        return None
    scope: dict[str, Any] = {}
    entities = served.get("entities")
    if entities != base.get("entities") and isinstance(entities, Mapping):
        scope["entities"] = {
            str(role): [f"dcid:{ref}" for ref in refs] for role, refs in entities.items()
        }
    window = served.get("requested_window")
    if window != base.get("requested_window") and isinstance(window, Mapping):
        scope["window"] = dict(window)
    return scope


class ConversationBlocks(Mapping[str, Mapping[str, Any]]):
    """Every Search block the conversation keeps, by spec_id, with `extra`
    over them: the open turn's (`turn_specs`), else the newest an earlier
    turn's record kept, as `stored_block` reads one. A lookup reads the one
    block asked for; only iterating reads them all."""

    def __init__(
        self,
        turn_specs: Mapping[str, Any],
        turns: list[dict[str, Any]],
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        self._turn_specs, self._turns, self._extra = turn_specs, turns, extra or {}

    def __getitem__(self, spec_id: str) -> Mapping[str, Any]:
        block = self._extra.get(spec_id)
        if not isinstance(block, Mapping):
            block = stored_block(self._turn_specs, self._turns, spec_id)
        if block is None:
            raise KeyError(spec_id)
        return block

    def _all(self) -> dict[str, Mapping[str, Any]]:
        blocks: dict[str, Mapping[str, Any]] = {}
        for turn in self._turns:
            blocks.update(
                {k: v for k, v in (turn.get("specs") or {}).items() if isinstance(v, Mapping)}
            )
        blocks.update({k: v for k, v in self._turn_specs.items() if isinstance(v, Mapping)})
        blocks.update({k: v for k, v in self._extra.items() if isinstance(v, Mapping)})
        return blocks

    def __iter__(self) -> Iterator[str]:
        return iter(self._all())

    def __len__(self) -> int:
        return len(self._all())


def card_unit_change(card: UnitCard, block: Mapping[str, Any] | None) -> UnitChange | None:
    """The `unit_change` of a card reading `card`'s spec, whose Search block is
    `block`: the units the block converts into, starting on the one it is
    served in; `None` when it converts into none or its served unit lies
    outside them, and for a card on a sum (`UnitCard.total`), whose unit
    change would redraw the card on the retrieval and not on its sum."""
    if block is None or card.total is not None:
        return None
    offer = unit_offer([block])
    served = served_unit(block)
    if offer is None or served is None or offered_unit_refusal(offer, served) is not None:
        return None
    return UnitChange(spec_id=card.spec_id, offer=offer.model_copy(update={"start": served}))


def stored_unit_change(
    entry: Mapping[str, Any], turn_specs: Mapping[str, Any], turns: list[dict[str, Any]]
) -> UnitChange | None:
    """The `unit_change` of the card whose stored entry is `entry`: its
    `UnitCard` with the Search block a turn kept for its spec, read as
    `unit_move` reads them. `None` for a card stored with no `UnitCard`, or
    whose spec's block is gone or converts into none."""
    card = UnitCard.from_json(entry.get(CARD_UNIT_KEY))
    if card is None:
        return None
    return card_unit_change(card, stored_block(turn_specs, turns, card.spec_id))


def card_changes(
    card: UnitCard,
    block: Mapping[str, Any] | None,
    *,
    fact_summary: Mapping[str, Any] | None,
    places: bool = False,
) -> CardChanges | None:
    """The `card_changes` of a card reading `card`'s spec, whose Search block
    is `block`, that draws the years its `fact_summary` states; `None` when it
    offers no change.

    `window` is offered when the card draws one year or a range and its data
    holds more than one: Search's `years_with_data`, else the window its
    answer serves (`download.window`, then `coverage.window`), widened to the
    years the card draws. A card whose scope names other places than the
    block offers the years Search stated for them (`UnitCard.years`), and
    none while it states none for its spec and places. With `places`
    (Search's places door is on), `places` holds each role whose places the
    card draws. A card cut to its largest members at one period
    (`UnitCard.cut`) offers neither, since other years or places have other
    largest members."""
    if block is None or (card.facets_unread and card.total is None):
        return None
    held = _years_of(card, block)
    window = (
        None
        if card.cut or held is None
        else _window_offer(held, fact_summary, single=card.single, drawn=card.drawn_year)
    )
    roles = _places_offers(card, block) if places and not card.cut else []
    if window is None and not roles:
        return None
    return CardChanges(spec_id=card.spec_id, window=window, places=roles or None)


def _year(value: Any) -> int | None:
    """The calendar year `value`, a fact summary's period, names: its first
    four digits when it opens on them, else `None`."""
    text = str(value) if isinstance(value, int | str) else ""
    return int(text[:4]) if len(text) >= 4 and text[:4].isdigit() else None


def years_held(block: Mapping[str, Any]) -> tuple[int, int, list[int]] | None:
    """The first and last year `block`'s spec holds data, with the years
    between them it holds none of. They come from `years_with_data` where
    Search states it, else from the window its answer serves, with none
    missing."""
    resolved = block.get("resolved") or {}
    held = resolved.get("years_with_data")
    if isinstance(held, Mapping):
        start, end = held.get("start_year"), held.get("end_year")
        if isinstance(start, int) and isinstance(end, int):
            gaps = [year for year in held.get("gap_years") or () if isinstance(year, int)]
            return start, end, gaps
    for window in (
        (resolved.get("download") or {}).get("window"),
        (resolved.get("coverage") or {}).get("window"),
    ):
        if isinstance(window, Mapping):
            start, end = window.get("start_year"), window.get("end_year")
            if isinstance(start, int) and isinstance(end, int):
                return start, end, []
    return None


def _scoped_entities(card: UnitCard) -> dict[str, list[str]] | None:
    """The places `card`'s scope names, by role, refs without the `dcid:`
    mark; `None` when it names none."""
    scoped = (card.scope or {}).get("entities")
    if not isinstance(scoped, Mapping):
        return None
    return {
        str(role): [str(ref).removeprefix("dcid:") for ref in refs]
        for role, refs in scoped.items()
        if isinstance(refs, list)
    }


def scope_needs_years(card: UnitCard, block: Mapping[str, Any]) -> dict[str, list[str]] | None:
    """The places `card`'s scope names when they are not `block`'s own, so
    the block's years do not describe the card and Search must be read for
    them (`UnitCard.years`); `None` when the block's years do."""
    scoped = _scoped_entities(card)
    identity = (block.get("definition") or {}).get("identity") or {}
    own = {
        str(role): [str(ref) for ref in refs]
        for role, refs in (identity.get("entities") or {}).items()
        if isinstance(refs, list)
    }
    if scoped is None or {r: sorted(v) for r, v in scoped.items()} == {
        r: sorted(v) for r, v in own.items()
    }:
        return None
    return scoped


def _years_of(card: UnitCard, block: Mapping[str, Any]) -> tuple[int, int, list[int]] | None:
    """The years `card`'s spec holds over the places it shows: the block's
    (`years_held`), else, for a scope naming other places, the ones Search
    stated for those places and that spec (`UnitCard.years`)."""
    scoped = scope_needs_years(card, block)
    if scoped is None:
        return years_held(block)
    years = card.years
    if (
        years is None
        or years.spec_id != card.spec_id
        or {r: sorted(v) for r, v in years.entities.items()}
        != {r: sorted(v) for r, v in scoped.items()}
    ):
        return None
    return years.first, years.last, list(years.gaps)


def effective_identity(card: UnitCard, block: Mapping[str, Any]) -> dict[str, Any]:
    """The identity `card` reads: its block's, with the places and window its
    scope names in place of the block's own (`_SCOPE_REPLACES`' members), as
    the data API reads `get_data`'s scope over it."""
    identity = dict((block.get("definition") or {}).get("identity") or {})
    scoped = _scoped_entities(card)
    if scoped is not None:
        identity["entities"] = scoped
    window = (card.scope or {}).get("window")
    if isinstance(window, Mapping):
        identity["requested_window"] = dict(window)
    return identity


def _window_offer(
    held: tuple[int, int, list[int]],
    fact_summary: Mapping[str, Any] | None,
    *,
    single: bool,
    drawn: int | None = None,
) -> WindowOffer | None:
    """The years a card can change to. A card showing one value per series
    (`single`) offers the year it draws (`drawn`) as both bounds, and no
    window when that year is unknown; any other card offers the years its
    `fact_summary` states it draws."""
    if single:
        start = end = drawn
    else:
        window = (fact_summary or {}).get("window") or {}
        start, end = _year(window.get("start")), _year(window.get("end"))
    if start is None or end is None or end < start:
        return None
    first, last, gaps = min(held[0], start), max(held[1], end), held[2]
    if first == last:
        return None
    missing = [year for year in gaps if first < year < last]
    return WindowOffer(
        start_year=start,
        end_year=end,
        first_year=first,
        last_year=last,
        missing_years=missing or None,
        single=True if single else None,
    )


def _places_offers(card: UnitCard, block: Mapping[str, Any]) -> list[PlacesOffer]:
    """One offer per role of the card's encoding (`UnitCard.roles`), taking
    several places where it says so, whatever number it shows now; a card
    stored with no encoding offers each role it shows places for, taking
    several where it shows several. A role showing a place neither the
    block nor the card's labels name offers none."""
    labels = {**place_labels(block), **(card.labels or {})}
    shown = drawn_places(card, block)
    roles = card.roles if card.roles is not None else {r: len(v) > 1 for r, v in shown.items()}
    # In the order Search lists the block's places, so every redraw lists
    # the roles alike.
    listed = [
        str(entity.get("role"))
        for entity in (block.get("resolved") or {}).get("entities") or ()
        if isinstance(entity, Mapping)
    ]
    roles = dict(
        sorted(
            roles.items(), key=lambda kv: listed.index(kv[0]) if kv[0] in listed else len(listed)
        )
    )
    most = dict(card.most or {})
    if card.total is not None:
        roles, most = _summed_roles(card.total, roles, shown, most)
    # A place with no name is not shown by its ref, so its role offers no
    # change, and nor does a role drawing more places than it can take.
    return [
        PlacesOffer(
            role=role,
            multiple=several,
            chosen=[PlaceOption(id=ref, label=labels[ref]) for ref in shown[role]],
            max=most.get(role),
            label=RoleLabel(**forms) if (forms := (card.nouns or {}).get(role)) else None,
        )
        for role, several in roles.items()
        if shown.get(role)
        and all(labels.get(ref) for ref in shown[role])
        and (most.get(role) is None or len(shown[role]) <= most[role])
    ]


def _summed_roles(
    total: SummedFrame,
    roles: Mapping[str, bool],
    shown: Mapping[str, list[str]],
    most: Mapping[str, int],
) -> tuple[dict[str, bool], dict[str, int]]:
    """The roles a card on the sum `total` offers places for, each with
    whether it takes several, and the most places each can draw: a role
    whose column the sum keeps as `roles` and `most` have it, and a role
    whose column the sum adds up only while the card shows one place for it,
    one place at a time, so the sum still runs over one place."""
    kept: dict[str, bool] = {}
    limits = dict(most)
    for role, several in roles.items():
        if role_column(role) in total.group_by:
            kept[role] = several
        elif len(shown.get(role) or ()) == 1:
            kept[role] = False
            limits[role] = 1
    return kept, limits


def drawn_places(card: UnitCard, block: Mapping[str, Any]) -> dict[str, list[str]]:
    """The places each role of `card` draws, by ref: the ones kept when it was
    drawn (`UnitCard.shown`), else every place its frame holds
    (`chosen_places`)."""
    held = chosen_places(card, block)
    if card.shown is None:
        return held
    return {role: list(card.shown.get(role) or refs) for role, refs in held.items()}


def chosen_places(card: UnitCard, block: Mapping[str, Any]) -> dict[str, list[str]]:
    """The places each role of `card`'s frame holds, by ref: its scope's where
    the scope names places, else its spec's identity's."""
    scoped = (card.scope or {}).get("entities")
    if isinstance(scoped, Mapping):
        return {
            str(role): [str(name).removeprefix("dcid:") for name in names]
            for role, names in scoped.items()
        }
    identity = (block.get("definition") or {}).get("identity") or {}
    entities = identity.get("entities") or {}
    return {str(role): [str(ref) for ref in refs] for role, refs in entities.items()}


def place_labels(block: Mapping[str, Any]) -> dict[str, str]:
    """The label Search gives each place `block` binds, by ref."""
    return {
        str(entity["ref"]): str(entity.get("label") or entity["ref"])
        for entity in (block.get("resolved") or {}).get("entities") or ()
        if isinstance(entity, Mapping) and entity.get("ref")
    }


def stored_card_changes(
    entry: Mapping[str, Any],
    turn_specs: Mapping[str, Any],
    turns: list[dict[str, Any]],
    *,
    places: bool = False,
) -> CardChanges | None:
    """The `card_changes` of the card whose stored entry is `entry`, read as
    `change_move` reads them. `None` for a card stored with no `UnitCard`,
    or whose spec's block is gone or offers no change."""
    card = UnitCard.from_json(entry.get(CARD_UNIT_KEY))
    if card is None:
        return None
    return card_changes(
        card,
        stored_block(turn_specs, turns, card.spec_id),
        fact_summary=entry.get("fact_summary"),
        places=places,
    )


@dataclass(frozen=True)
class UnitMove:
    """A unit change the route admitted: the card (`component_id`, titled
    `title`), what it reads (`card`), the unit it is served in (`served`) and
    the one it moves to (`unit`), within `offer`, with the Search block of its
    spec and the newest reading of that spec, `None` when no reading names it."""

    component_id: str
    title: str
    card: UnitCard
    served: UnitChoice
    unit: UnitChoice
    offer: UnitOffer
    block: dict[str, Any]
    reading: SearchReading | None


# The `detail` of a unit change refused because the card moved on: the client
# shows it in the reply.
UNIT_STALE_NOTE = "This card has changed since its unit menu opened, so nothing changed."
UNIT_GONE_NOTE = "This card is no longer on the canvas, so nothing changed."
# The refusal of a unit change of a card on the canvas that offers none.
UNIT_UNOFFERED = ("unit_not_offered", "This card's unit cannot change.")


def _admitted(
    conversation: ConversationState,
    component_id: str,
    spec_id: str | None,
    *,
    stale: str,
    unoffered: tuple[str, str],
) -> tuple[UnitCard, dict[str, Any], Mapping[str, Any], str] | tuple[str, str]:
    """Card `component_id` of `conversation` as a unit change or a change of
    years or places reads it: what it reads, its spec's Search block, its
    digest entry and its title; or its refusal's constraint and detail:
    `scope_choice_stale` with `UNIT_GONE_NOTE` when the card is off the
    canvas and with `stale` when it reads a spec other than `spec_id` (any,
    with `None`), and `unoffered` for a card on the canvas that can change
    nothing, because it kept nothing a change reads or no turn kept its
    spec's block."""
    composition = conversation.composition
    blocks = composition.blocks if composition is not None else []
    on_canvas = next((b for b in blocks if b.component_id == component_id), None)
    if on_canvas is None:
        return "scope_choice_stale", UNIT_GONE_NOTE
    entry = conversation.components.get(component_id) or {}
    card = UnitCard.from_json(entry.get(CARD_UNIT_KEY))
    if card is None:
        return unoffered
    if spec_id is not None and card.spec_id != spec_id:
        return "scope_choice_stale", stale
    block = stored_block(conversation.turn_specs, conversation.turns, card.spec_id)
    if block is None:
        return unoffered
    return card, block, entry, str(entry.get("title") or on_canvas.title or "The card")


def unit_move(conversation: ConversationState, pick: ApplyUnit) -> UnitMove | tuple[str, str]:
    """The change `pick` asks of `conversation`, or its refusal's constraint
    and detail: `scope_choice_stale` when the card is off the canvas or its
    `provenance.inputs[0]` names a spec other than `pick.spec_id`;
    `unit_not_offered` for a card that offers no unit change, a unit the
    card's offer does not hold or the one it already shows. `pick.turn` plays
    no part."""
    held = _admitted(
        conversation,
        pick.component_id,
        pick.spec_id,
        stale=UNIT_STALE_NOTE,
        unoffered=UNIT_UNOFFERED,
    )
    if len(held) == 2:
        return held
    card, block, _, title = held
    change = card_unit_change(card, block)
    if change is None:
        return UNIT_UNOFFERED
    offer, served = change.offer, change.offer.start
    if (refused := offered_unit_refusal(offer, pick.unit)) is not None:
        return "unit_not_offered", f"That unit is not offered for this card: {refused}."
    if pick.unit == served:
        return "unit_not_offered", f"The card already shows {unit_words(served, offer)}."
    return UnitMove(
        component_id=pick.component_id,
        title=title,
        card=card,
        served=served,
        unit=pick.unit,
        offer=offer,
        block=block,
        reading=stored_reading(conversation.turn_readings, conversation.turns, card.spec_id),
    )


@dataclass(frozen=True)
class ChangeMove:
    """A change of a card's years or places the route admitted: the card
    (`component_id`, titled `title`), what it reads (`card`), the change, the
    `scope` its new retrieval names (the card's own with the change's member
    replaced), its words (`words`: "2010 to 2024", "Uganda"), the offer it
    was checked against (`offer`) and the Search block of its spec. A change
    of places carries the labels of the places it leaves and shows (`old` and
    `new`) and the places Search listed for its role (`listed`). `drawn` are
    the first and last year the card draws, as its fact summary states them,
    `None` when it states none."""

    component_id: str
    title: str
    card: UnitCard
    change: WindowChange | PlacesChange
    scope: dict[str, Any]
    words: str
    offer: CardChanges
    block: dict[str, Any]
    old: tuple[str, ...] = ()
    new: tuple[str, ...] = ()
    listed: RolePlaces | None = None
    drawn: tuple[int, int] | None = None


CHANGE_STALE_NOTE = "This card has changed since its menu opened, so nothing changed."
# The refusal of a change of years or places of a card on the canvas that
# offers none.
CHANGE_UNOFFERED = ("change_not_offered", "This card's years and places cannot change.")
# How `/places` refuses a card on the canvas that offers no change.
PLACES_UNOFFERED = ("scope_choice_stale", CHANGE_STALE_NOTE)


# The refusal of a change of places on a deployment whose apis serves no
# places door (`harness.place_options.PlacesNotServedError`), which the route
# answers as `/places` does, 503 `capability_unavailable`.
PLACES_NOT_SERVED = "places_not_served"


# The `detail` of a change refused because Search's places door could not be read.
PLACES_UNREAD_NOTE = "The places this card can show cannot be read now."


def changeable(
    conversation: ConversationState,
    component_id: str,
    spec_id: str | None,
    *,
    places: bool,
    unoffered: tuple[str, str] = PLACES_UNOFFERED,
) -> tuple[UnitCard, dict[str, Any], CardChanges, str] | tuple[str, str]:
    """The card `component_id` of `conversation` as a change reads it: what it
    reads, its spec's Search block, its offer and its title; or its refusal's
    constraint and detail: `scope_choice_stale` when the card is off the
    canvas or reads a spec other than `spec_id` (any, with `None`), and
    `unoffered` when it offers no change (`/places` answers that
    `scope_choice_stale` too, as its contract states, and `change_move` answers
    `CHANGE_UNOFFERED`). `places` says whether Search's places door is on."""
    held = _admitted(
        conversation, component_id, spec_id, stale=CHANGE_STALE_NOTE, unoffered=unoffered
    )
    if len(held) == 2:
        return held
    card, block, entry, title = held
    offer = card_changes(card, block, fact_summary=entry.get("fact_summary"), places=places)
    if offer is None:
        return unoffered
    return card, block, offer, title


async def change_move(  # noqa: PLR0911 -- one refusal per way a change cannot apply
    conversation: ConversationState,
    pick: ApplyChange,
    *,
    reader: PlaceOptionsReader | None,
    listed: RolePlaces | None = None,
) -> ChangeMove | tuple[str, str]:
    """The change `pick` asks of `conversation`, or its refusal's constraint
    and detail: `scope_choice_stale` as `changeable` refuses;
    `change_not_offered` for a card that offers no change, years or places
    outside the card's offer, or the ones it shows; `change_no_data` for years
    its data holds none of. A change
    of places reads the places its role can change to from Search through
    `reader` (the places door, `None` when off), or takes them from `listed`
    when the route read them already, and is refused
    `place_options_unavailable` when they cannot be read."""
    held = changeable(
        conversation,
        pick.component_id,
        pick.spec_id,
        places=reader is not None,
        unoffered=CHANGE_UNOFFERED,
    )
    if len(held) == 2:
        return held
    card, block, offer, title = held
    window = ((conversation.components.get(pick.component_id) or {}).get("fact_summary") or {}).get(
        "window"
    ) or {}
    first, last = _year(window.get("start")), _year(window.get("end"))
    drawn = (first, last) if first is not None and last is not None else None
    change = pick.change
    if isinstance(change, WindowChange):
        scope, words = _window_move(card, offer, change)
        if isinstance(scope, str):
            return scope, words
        return ChangeMove(
            pick.component_id, title, card, change, scope, words, offer, block, drawn=drawn
        )
    refused = _places_refusal(offer, change)
    if refused is not None:
        return refused
    assert reader is not None, "a places offer is made only with the door on"
    if listed is None:
        try:
            listed = await read_role_places(
                reader,
                identity=effective_identity(card, block),
                role=change.role,
                conversation_id=conversation.conversation_id,
                component_id=pick.component_id,
            )
        except PlacesNotServedError:
            return PLACES_NOT_SERVED, PLACES_UNREAD_NOTE
        except PlacesUnavailableError:
            return "place_options_unavailable", PLACES_UNREAD_NOTE
    moved = _places_move(pick.component_id, title, card, block, offer, change, listed)
    return replace(moved, drawn=drawn) if isinstance(moved, ChangeMove) else moved


def _window_move(
    card: UnitCard, offer: CardChanges, change: WindowChange
) -> tuple[dict[str, Any], str] | tuple[str, str]:
    window = offer.window
    if window is None:
        return "change_not_offered", "This card's years cannot change."
    start, end = change.start_year, change.end_year
    if window.single and start != end:
        return "change_not_offered", "This card shows one year at a time."
    if start < window.first_year or end > window.last_year:
        return (
            "change_not_offered",
            f"The data runs from {window.first_year} to {window.last_year}.",
        )
    if (start, end) == (window.start_year, window.end_year):
        return "change_not_offered", f"The card already shows {start} to {end}."
    missing = set(window.missing_years or ())
    if all(year in missing for year in range(start, end + 1)):
        return "change_no_data", f"The data holds no figures from {start} to {end}."
    held = (card.scope or {}).get("window") or {}
    kept = {"include": held["include"]} if isinstance(held, Mapping) and "include" in held else {}
    scope = {
        **(card.scope or {}),
        "window": {"kind": "range", "start_year": start, "end_year": end, **kept},
    }
    return scope, f"{start} to {end}"


def _places_refusal(offer: CardChanges, change: PlacesChange) -> tuple[str, str] | None:
    """Why `change` is refused before Search is read: a role the card does not
    offer, a count its `multiple` or `max` does not allow, or the places it
    shows."""
    places = next((p for p in offer.places or () if p.role == change.role), None)
    if places is None:
        return "change_not_offered", "The places of this card cannot change there."
    picked = list(dict.fromkeys(change.entities))
    if not places.multiple and len(picked) != 1:
        return "change_not_offered", "This card shows one place at a time."
    if places.max is not None and len(picked) > places.max:
        return "change_not_offered", f"This card can show at most {places.max} places there."
    if set(picked) == {o.id for o in places.chosen}:
        return "change_not_offered", "The card already shows those places."
    return None


def _places_move(
    component_id: str,
    title: str,
    card: UnitCard,
    block: Mapping[str, Any],
    offer: CardChanges,
    change: PlacesChange,
    listed: RolePlaces,
) -> ChangeMove | tuple[str, str]:
    places = next(p for p in offer.places or () if p.role == change.role)
    labels = {**dict(listed.places), **{o.id: o.label for o in places.chosen}}
    picked = list(dict.fromkeys(change.entities))
    if any(ref not in labels for ref in picked):
        return "change_not_offered", "That place is not offered for this card."
    kept = {**(card.labels or {}), **{ref: labels[ref] for ref in picked}}
    entities = {**chosen_places(card, block), change.role: picked}
    scope = {
        **(card.scope or {}),
        "entities": {role: [f"dcid:{ref}" for ref in refs] for role, refs in entities.items()},
    }
    new = tuple(labels[ref] for ref in picked)
    return ChangeMove(
        component_id,
        title,
        replace(card, labels=kept, shown={**drawn_places(card, block), change.role: picked}),
        change,
        scope,
        joined_words(list(new)),
        offer,
        dict(block),
        old=tuple(o.label for o in places.chosen),
        new=new,
        listed=listed,
    )


def joined_words(words: list[str]) -> str:
    """`words` as a reader lists them: "A", "A and B", "A, B and C"."""
    if len(words) <= 2:
        return " and ".join(words)
    return f"{', '.join(words[:-1])} and {words[-1]}"


def stored_block(
    turn_specs: Mapping[str, Any], turns: list[dict[str, Any]], spec_id: str
) -> dict[str, Any] | None:
    """The Search block of `spec_id`: the open turn's (`turn_specs`), else the
    newest one an earlier turn's record kept; `None` when none did."""
    block = turn_specs.get(spec_id)
    if isinstance(block, dict):
        return block
    for turn in reversed(turns):
        block = (turn.get("specs") or {}).get(spec_id)
        if isinstance(block, dict):
            return block
    return None


def stored_reading(
    turn_readings: list[dict[str, Any]], turns: list[dict[str, Any]], spec_id: str
) -> SearchReading | None:
    """The newest reading whose own spec_id is `spec_id`, the open turn's first."""
    newest_first = [
        *reversed(turn_readings),
        *(r for turn in reversed(turns) for r in reversed(turn.get("readings") or ())),
    ]
    for stored in newest_first:
        if isinstance(stored, Mapping) and stored.get("spec_id") == spec_id:
            return SearchReading.model_validate(stored)
    return None


def unit_pattern(unit: UnitChoice, offer: UnitOffer | None) -> str:
    """The pattern, read case-insensitively, of each phrase that names `unit`
    (`retitled` lists them): the currency by its code or by the name `offer`
    gives it (by its code alone with no offer), after an optional price
    basis; a price basis followed by "prices"; or, for a constant unit,
    "constant <base year>" alone."""
    names = {currency.code: currency.name for currency in offer.currencies} if offer else {}
    name = names.get(unit.currency)
    if unit.prices is UnitPrices.CONSTANT:
        year = re.escape(str(unit.base_year))
        basis = rf"constant(?:\s+{year})?"
        priced = rf"{basis}|{year}"
        alone = rf"|(?P<alone>constant\s+{year})"
    else:
        basis = priced = "current"
        alone = ""
    currencies = [rf"(?P<code>(?-i:{re.escape(unit.currency)}))"]
    if name:
        currencies.append(rf"(?P<name>{re.escape(name)})")
    return (
        rf"(?<!\w)(?:(?:(?P<basis>{basis})\s+)?(?:{'|'.join(currencies)})"
        rf"|(?P<prices>{priced})\s+prices{alone})(?!\w)"
    )


def retitled(title: str, *, old: UnitChoice, new: UnitChoice, offer: UnitOffer) -> str:
    """`title` with each phrase that names `old` naming `new` in the same shape.

    A phrase is the currency `old` is served in, by its code or by the name
    `offer` gives it, after an optional price basis; a price basis followed by
    "prices"; or, when `old` is constant, "constant <base year>" alone. The
    basis is the one `old` is served in: "constant" or "constant <base
    year>" when `old` is constant, "current" when it is current, and "<base
    year>" before "prices" when it is constant. The whole phrase is written
    again from `new`, its basis in the words `unit_words` gives it
    ("constant 2023", "current"): the basis and the new currency, by code or
    by name as the phrase named the old one, the basis and "prices", or the
    basis alone. A currency named with no basis is named alone, and a year
    before "prices" stays a bare year when `new` is constant. Any other words
    stand, another basis and a year alone among them.
    """
    names = {currency.code: currency.name for currency in offer.currencies}
    old_name = names.get(old.currency)
    new_name = names.get(new.currency, new.currency)
    words = unit_words(new, offer)
    new_basis = words[: len(words) - len(new_name) - 1].lower()
    pattern = unit_pattern(old, offer)

    def swap(match: re.Match[str]) -> str:
        said = match.group(0)
        if match.groupdict().get("alone") is not None:
            phrase = new_basis
        elif match.group("prices") is not None:
            bare_year = match.group("prices")[:1].isdigit() and new.base_year is not None
            phrase = f"{new.base_year if bare_year else new_basis} prices"
        elif match.group("basis") is not None:
            currency = new.currency if match.group("code") is not None else new_name
            phrase = f"{new_basis} {currency}"
        elif match.group("code") is not None:
            return new.currency
        else:
            # A name opens on a capital only where it opens the title, or
            # where the title capitalises a name written lower-case.
            assert old_name is not None
            if match.start() == 0 or (said[:1].isupper() and old_name[:1].islower()):
                return new_name[:1].upper() + new_name[1:]
            return new_name
        opener = phrase[:1].upper() if said[:1].isupper() else phrase[:1]
        return opener + phrase[1:]

    return re.sub(pattern, swap, title, flags=re.IGNORECASE)

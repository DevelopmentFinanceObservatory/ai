"""Seeded rolling plans over synthetic series, run as admitted and as translation restates them.

Each case draws a world, donor series that each hold every year from a first
year to a last, and a plan over it as the planner states one: a dataset
requirement read with no end year over the years Search states for the world
(the first year any series holds to the newest), and a rolling growth, filter,
or filter feeding a growth or a total, whose rolling years count from Search's
newest year. `Plan` itself admits the plan or refuses it, so every rule a
proposal meets decides, the shapes included. The newest year a run counts from
is `dfo_frame_grammar.latest_anchor` over the world's series, the half rule
analysis-mcp's `resolve_anchor` applies.

A run writes each rolling year as its newest year moved by the year's offset.
`run_admitted` runs the plan as admitted, from the run world's newest year,
with no restatement and no check, as the harness ran plans before translation
read the newest year. `run_translated` runs the plan as translation records
it: where the newest year
read at translation is earlier than the plan's, restated from that year
(`planner.plan.restated_to_anchors`), and absent where a plan rule refuses the
restatement. Its run then refuses a newest year earlier than the plan's, as
`executor.run` does with `rolling_anchor_behind_plan`.

`outcome` is the one judgement made here rather than read from the code. A
step runs empty when the years it reads are none: a growth window that ends
where it starts, an endpoint the filter feeding it drops, an end year after the
newest any series holds, or a filter keeping no year up to that newest. Those
are what the plan's rolling years decide. A step misses when it reads years
inside that span that no one series holds, as when a growth's fixed start
predates every series still reporting. A series starting late or stopping
early causes that in any world, so both sides count it alike.

Each case runs in three worlds: the one it was planned over, one where newer
years reach some series, and one where a lagging series joins or a series
holding the newest year stops.
"""

from __future__ import annotations

import operator
import random
from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import StrEnum

from dfo_frame_grammar import latest_anchor
from planner.plan import (
    AggregateRequirement,
    DatasetRequirement,
    Decision,
    Entity,
    EvidenceRequirement,
    FilterRequirement,
    FirstYear,
    GrowthRequirement,
    Period,
    Plan,
    ResolvedBy,
    _stated_newest,
    restated_to_anchors,
    rolls,
)
from planner.records import GroupBy, PeriodThreshold, Reduction, RelativePeriod, RollingWindow

# Search's newest year in every world drawn: the newest year any series holds.
NEWEST = 2024
DATASET = "oda"
_DECISION = Decision(
    slot="measure",
    description="Which aid flow to read.",
    options=["bilateral ODA", "total ODA"],
    chosen="bilateral ODA",
    resolved_by=ResolvedBy.PLANNER_DEFAULT,
    rationale="Bilateral ODA is what a donor reports by recipient.",
)
# How many years before `NEWEST` a series stops, with its weight.
_LAGS = ((0, 6), (1, 3), (2, 2), (3, 1))
_OPS = {
    "gt": operator.gt,
    "gte": operator.ge,
    "lt": operator.lt,
    "lte": operator.le,
    "eq": operator.eq,
    "ne": operator.ne,
}


class Outcome(StrEnum):
    RAN = "ran"
    EMPTY = "empty"
    # The years read fall within the newest year a series holds, but no one
    # series holds them: a series starting late or stopping early.
    MISSED = "missed"
    # Translation restated the plan and a plan rule refused it.
    RESTATEMENT_REFUSED = "restatement_refused"
    # The run's newest year is before the plan's (`rolling_anchor_behind_plan`).
    ANCHOR_BEHIND = "anchor_behind"


class WorldKind(StrEnum):
    SAME = "same"
    NEWER = "newer"
    LAGGING = "lagging"


@dataclass(frozen=True)
class World:
    """Each donor series' first and last year."""

    series: Mapping[str, tuple[int, int]]

    @property
    def first(self) -> int:
        return min(first for first, _ in self.series.values())

    @property
    def anchor(self) -> int:
        """The newest year at least half the series hold."""
        population = {
            key: [str(year) for year in range(first, last + 1)]
            for key, (first, last) in self.series.items()
        }
        return int(latest_anchor([population]).period)


@dataclass(frozen=True)
class Case:
    """One admitted plan and the worlds it runs in, the first the one it was planned over."""

    plan: Plan
    worlds: Mapping[WorldKind, World]


def _world(rng: random.Random) -> World:
    lags, weights = zip(*_LAGS, strict=True)
    series = {
        f"donor_{n}": (rng.randint(2000, 2012), NEWEST - rng.choices(lags, weights)[0])
        for n in range(rng.randint(2, 8))
    }
    if all(last < NEWEST for _, last in series.values()):
        key = rng.choice(sorted(series))
        series[key] = (series[key][0], NEWEST)
    return World(series=series)


def _newer(world: World, rng: random.Random) -> World:
    """`world` with one to two newer years reaching some series, one at least."""
    moved = {key: rng.randint(0, 2) for key in world.series}
    moved[rng.choice(sorted(moved))] = rng.randint(1, 2)
    return World(
        series={key: (first, last + moved[key]) for key, (first, last) in world.series.items()}
    )


def _lagging(world: World, rng: random.Random) -> World:
    """`world` with a series stopping two to four years early added, or a series holding the newest year removed."""
    series = dict(world.series)
    holding = sorted(key for key, (_, last) in series.items() if last == NEWEST)
    if len(series) > 2 and rng.random() < 0.5:
        del series[rng.choice(holding)]
    else:
        series[f"donor_{len(series)}"] = (rng.randint(2000, 2012), NEWEST - rng.randint(2, 4))
    return World(series=series)


def _offset(offset: int) -> RelativePeriod:
    return RelativePeriod(offset=offset)


def _growth(input_id: str, start: int, *, rolling_start: bool) -> GrowthRequirement:
    return GrowthRequirement(
        id="growth",
        input_id=input_id,
        window_start=start,
        window_end=NEWEST,
        rolling=RollingWindow(
            window_start=_offset(start - NEWEST) if rolling_start else None,
            window_end=_offset(0),
        ),
    )


def _steps(rng: random.Random, first: int) -> list[EvidenceRequirement]:
    """One draw of the steps over the dataset, their rolling years stated from `NEWEST`."""
    start = NEWEST - rng.randint(1, 8)
    fixed = rng.randint(first, NEWEST)
    match rng.randrange(7):
        case 0:
            return [_growth(DATASET, start, rolling_start=True)]
        case 1:
            # "Growth from 2023 to the newest year."
            return [_growth(DATASET, min(fixed, NEWEST - 1), rolling_start=False)]
        case 2:
            kept = [PeriodThreshold(op="gte", year=start, rolling=_offset(start - NEWEST))]
        case 3:
            back = rng.randint(0, 3)
            kept = [PeriodThreshold(op="eq", year=NEWEST - back, rolling=_offset(-back))]
        case 4:
            kept = [
                PeriodThreshold(op="gte", year=fixed),
                PeriodThreshold(op="lte", year=NEWEST, rolling=_offset(0)),
            ]
        case 5:
            kept = [PeriodThreshold(op="gte", year=start, rolling=_offset(start - NEWEST))]
            recent = FilterRequirement(id="recent", input_id=DATASET, predicates=tuple(kept))
            return [recent, _growth("recent", start, rolling_start=True)]
        case _:
            kept = [PeriodThreshold(op="gte", year=start, rolling=_offset(start - NEWEST))]
            recent = FilterRequirement(id="recent", input_id=DATASET, predicates=tuple(kept))
            total = AggregateRequirement(
                id="total", input_id="recent", reduction=Reduction.SUM, group_by=(GroupBy.PERIOD,)
            )
            return [recent, total]
    return [FilterRequirement(id="recent", input_id=DATASET, predicates=tuple(kept))]


def cases(seed: int, count: int) -> Iterator[Case]:
    """`count` admitted plans drawn from `seed`, each with its three worlds."""
    rng = random.Random(seed)
    made = 0
    while made < count:
        world = _world(rng)
        dataset = DatasetRequirement(
            id=DATASET,
            decision_ids=("measure",),
            measure="bilateral ODA",
            entities=[Entity(name="Kenya", role="recipient")],
            across=("donor",),
            period=Period(start=world.first, end=NEWEST),
            period_end_open=True,
        )
        steps = _steps(rng, world.first)
        worlds = {
            WorldKind.SAME: world,
            WorldKind.NEWER: _newer(world, rng),
            WorldKind.LAGGING: _lagging(world, rng),
        }
        try:
            plan = Plan(
                question="How has aid to Kenya moved in recent years?",
                decisions=[_DECISION],
                evidence_requirements=[dataset, *steps],
                first_years=[FirstYear(requirement_id=DATASET, year=world.first)],
            )
        except ValueError:
            continue
        made += 1
        yield Case(plan=plan, worlds=worlds)


def planned_anchor(plan: Plan) -> int:
    """The newest year `plan`'s rolling years are stated from."""
    return max(_stated_newest(r) for r in plan.evidence_requirements if rolls(r))


def outcome(plan: Plan, world: World, anchor: int) -> Outcome:
    """How a run of `plan` over `world` ends with its rolling years written from `anchor`."""
    by_id = {r.id: r for r in plan.evidence_requirements}
    recent = by_id.get("recent")
    newest = max(last for _, last in world.series.values())

    def year(stated: int, rolling: RelativePeriod | None) -> int:
        return anchor + rolling.offset if rolling else stated

    def kept(at: int) -> bool:
        if not isinstance(recent, FilterRequirement):
            return True
        return all(
            _OPS[p.op](at, year(p.year, p.rolling))
            for p in recent.predicates
            if isinstance(p, PeriodThreshold)
        )

    def holds(key: str, at: int) -> bool:
        first, last = world.series[key]
        return first <= at <= last

    growth = by_id.get("growth")
    if isinstance(growth, GrowthRequirement):
        assert growth.rolling is not None
        assert growth.window_start is not None and growth.window_end is not None
        start = year(growth.window_start, growth.rolling.window_start)
        end = year(growth.window_end, growth.rolling.window_end)
        if end <= start or end > newest or not (kept(start) and kept(end)):
            return Outcome.EMPTY
        if not any(holds(key, start) and holds(key, end) for key in world.series):
            return Outcome.MISSED
        return Outcome.RAN
    read = [at for at in range(world.first, newest + 1) if kept(at)]
    if not read:
        return Outcome.EMPTY
    if not any(holds(key, at) for key in world.series for at in read):
        return Outcome.MISSED
    return Outcome.RAN


def translation(case: Case) -> Plan | None:
    """The plan translation records, or `None` where a plan rule refuses the restatement."""
    anchor = case.worlds[WorldKind.SAME].anchor
    if anchor >= planned_anchor(case.plan):
        return case.plan
    anchors = {r.id: anchor for r in case.plan.evidence_requirements if rolls(r)}
    try:
        return restated_to_anchors(case.plan, anchors=anchors)
    except ValueError:
        return None


def run_admitted(case: Case, kind: WorldKind) -> Outcome:
    world = case.worlds[kind]
    return outcome(case.plan, world, world.anchor)


def run_translated(plan: Plan | None, case: Case, kind: WorldKind) -> Outcome:
    """A run of `plan`, the translation of `case`, in one of its worlds."""
    if plan is None:
        return Outcome.RESTATEMENT_REFUSED
    world = case.worlds[kind]
    if world.anchor < planned_anchor(plan):
        return Outcome.ANCHOR_BEHIND
    return outcome(plan, world, world.anchor)


@dataclass(frozen=True)
class Tally:
    """How every run of a seed's cases ended, by side and world."""

    admitted: Mapping[WorldKind, Counter[Outcome]]
    translated: Mapping[WorldKind, Counter[Outcome]]
    # Plans translation states from an earlier newest year.
    restated: int
    # The same-world runs of admitted plans that run empty, by how their translation ends.
    admitted_empties: Counter[Outcome]
    # Lagging-world runs of translated plans whose newest year drops below the plan's.
    dropped: Counter[Outcome]


def tally(seed: int, count: int) -> Tally:
    admitted = {kind: Counter[Outcome]() for kind in WorldKind}
    translated = {kind: Counter[Outcome]() for kind in WorldKind}
    admitted_empties, dropped = Counter[Outcome](), Counter[Outcome]()
    restated = 0
    for case in cases(seed, count):
        plan = translation(case)
        restated += plan is not None and plan is not case.plan
        for kind in WorldKind:
            admitted[kind][run_admitted(case, kind)] += 1
            translated[kind][run_translated(plan, case, kind)] += 1
        if run_admitted(case, WorldKind.SAME) is Outcome.EMPTY:
            admitted_empties[run_translated(plan, case, WorldKind.SAME)] += 1
        lagging = case.worlds[WorldKind.LAGGING]
        if plan is not None and lagging.anchor < planned_anchor(plan):
            dropped[run_translated(plan, case, WorldKind.LAGGING)] += 1
    return Tally(
        admitted=admitted,
        translated=translated,
        restated=restated,
        admitted_empties=admitted_empties,
        dropped=dropped,
    )

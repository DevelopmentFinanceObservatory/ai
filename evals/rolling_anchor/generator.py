"""Seeded rolling plans over synthetic series, run as admitted and as translation restates them.

Each case draws a world, series that each hold every year from a first year to
a last, and a plan over it as the planner states one: a dataset requirement read
with no end year, and a rolling growth, filter, or filter feeding a growth or a
total, whose rolling years count from Search's newest year. The dataset's stated
years start at the first year any series holds, or, in a third of the draws, at
the first year its steps read, so a step moved earlier reads the dataset's whole
history (`planner.plan.whole_history_split`). `Plan` itself admits the plan or
refuses it, so every rule a proposal meets decides, the shapes included. The
plan is recorded as `propose_plan` records it, split
(`planner.plan.with_authored_requirements`), and compiled against the live
servers' schemas (`compiler.compile.compile_plan`). The newest year a run counts
from is `dfo_frame_grammar.latest_anchor` over the world's series, the half rule
analysis-mcp's `resolve_anchor` applies.

A run writes each rolling year as its newest year moved by the year's offset.
`run_admitted` runs the plan as admitted, from the run world's newest year,
with no restatement and no check, as the harness ran plans before translation
read the newest year. `run_translated` runs the plan as translation records it.
Each group of the compiled workflow is read with the newest year it states
(`executor.run.planned_anchor`) and the world's, as `executor.run.read_anchors`
answers it (`executor.run.AnchorRead`). A group behind has the plan and the
translation restated (`translator.tool.restated_to_the_data`) and compiled
again, and a plan rule or the compiler can refuse that. The run then refuses a
newest year before the plan's, as `executor.run._Run._check_anchor_held` does,
through `AnchorRead.behind`.

`outcome` is the one judgement made here. The harness's code gives the rest. A
step runs empty when it reads no years, as with a growth window that ends
where it starts, an endpoint the filter feeding it drops, an end year after the
newest any series holds, or a filter keeping no year up to that newest. The
plan's rolling years decide these. A step misses when it reads years
inside that span that no one series holds, as when a growth's fixed start
predates every series still reporting, or the dataset's stated start cuts them
off. A series starting late or stopping early causes that in any world, so both
sides count it alike.

Each case runs in three worlds: the one it was planned over, one where newer
years reach some series, and one where a lagging series joins or a series
holding the newest year stops.
"""

from __future__ import annotations

import asyncio
import functools
import operator
import random
from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import StrEnum

from analysis_mcp.server import mcp as analysis_mcp
from compiler.compile import CompileResult, compile_plan
from compiler.rolling import anchor_groups
from compiler.sources import member_digest
from compiler.surface import McpSurface, ToolSchema
from data_mcp.server import mcp as data_mcp
from dfo_frame_grammar import latest_anchor
from executor.outcomes import AnchorResult
from executor.run import AnchorRead, planned_anchor
from planner.operations import Translation
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
    PlanStatus,
    ResolvedBy,
    Unit,
    frame_shapes,
    inputs_of,
    with_authored_requirements,
)
from planner.records import GroupBy, PeriodThreshold, Reduction, RelativePeriod, RollingWindow
from planner.workflow import ResolvedEntity, SelectedSource, Workflow
from translator.mechanical import build_translation
from translator.tool import restated_to_the_data

# Search's newest year in every world drawn, which is the newest year any series holds.
NEWEST = 2024
DATASET = "debt"
_SPEC = "spec_" + "a" * 32
_ANCHOR_KEY = "rak_" + "7" * 32
_DECISION = Decision(
    slot="measure",
    description="Which flow to read.",
    options=["external debt service paid", "total debt stock"],
    chosen="external debt service paid",
    resolved_by=ResolvedBy.PLANNER_DEFAULT,
    rationale="Debt service is what a debtor pays each year.",
)
# The source every dataset requirement, and every copy of one, is retrieved from.
_SOURCE = SelectedSource(
    spec_id=_SPEC,
    sentence="external debt service paid by Kenya",
    confidence="high",
    has_data=True,
    served_count=1,
    shape_member_count=1,
    shape_member_digest=member_digest(["DebtServicePaid"]),
    entities=(ResolvedEntity(role="destination", ref="country/KEN", label="Kenya"),),
    unit_ref="usd",
    unit_label="US dollars",
    coverage_window=Period(start=2000, end=NEWEST),
    identity={"stat_var": "DebtServicePaid"},
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
    # series holds them, as with a series starting late or stopping early.
    MISSED = "missed"
    # Translation restated the plan and a plan rule refused it.
    RESTATEMENT_REFUSED = "restatement_refused"
    # Translation restated the plan and the compiler refused it.
    RESTATEMENT_UNCOMPILED = "restatement_uncompiled"
    # The run's newest year is before the plan's (`rolling_anchor_behind_plan`).
    ANCHOR_BEHIND = "anchor_behind"


class WorldKind(StrEnum):
    SAME = "same"
    NEWER = "newer"
    LAGGING = "lagging"


@dataclass(frozen=True)
class World:
    """Each series' first and last year."""

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
    """One admitted plan as recorded, its translation, the workflow it compiles to, and its worlds.

    `workflow` is `None` for a plan admitted that does not compile.
    """

    plan: Plan
    translation: Translation
    workflow: Workflow | None
    worlds: Mapping[WorldKind, World]


@dataclass(frozen=True)
class Translated:
    """A case's plan and workflow as translation records them, or the outcome refusing them."""

    plan: Plan | None
    workflow: Workflow | None
    refused: Outcome | None = None


@functools.cache
def _schemas() -> dict[str, ToolSchema]:
    """The live servers' tool schemas, read in process once."""
    surface = McpSurface(sources={"data-mcp": data_mcp, "analysis-mcp": analysis_mcp})
    return {schema.name: schema for schema in asyncio.run(surface.list_tools())}


def _datasets(plan: Plan) -> list[str]:
    return [r.id for r in plan.evidence_requirements if isinstance(r, DatasetRequirement)]


def _compiled(plan: Plan, translation: Translation) -> CompileResult:
    return compile_plan(
        plan=plan,
        translation=translation,
        schemas=_schemas(),
        discovered=dict.fromkeys(_datasets(plan), _SOURCE),
    )


def _world(rng: random.Random) -> World:
    lags, weights = zip(*_LAGS, strict=True)
    series = {
        f"series_{n}": (rng.randint(2000, 2012), NEWEST - rng.choices(lags, weights)[0])
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
        series[f"series_{len(series)}"] = (rng.randint(2000, 2012), NEWEST - rng.randint(2, 4))
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


def _steps(rng: random.Random, first: int) -> tuple[list[EvidenceRequirement], int]:
    """One draw of the steps over the dataset, their rolling years stated from `NEWEST`,
    with the first year they read."""
    start = NEWEST - rng.randint(1, 8)
    fixed = rng.randint(first, NEWEST)
    match rng.randrange(7):
        case 0:
            return [_growth(DATASET, start, rolling_start=True)], start
        case 1:
            # "Growth from 2023 to the newest year."
            fixed = min(fixed, NEWEST - 1)
            return [_growth(DATASET, fixed, rolling_start=False)], fixed
        case 2:
            kept = [PeriodThreshold(op="gte", year=start, rolling=_offset(start - NEWEST))]
        case 3:
            back = rng.randint(0, 3)
            start = NEWEST - back
            kept = [PeriodThreshold(op="eq", year=start, rolling=_offset(-back))]
        case 4:
            start = fixed
            kept = [
                PeriodThreshold(op="gte", year=fixed),
                PeriodThreshold(op="lte", year=NEWEST, rolling=_offset(0)),
            ]
        case 5:
            kept = [PeriodThreshold(op="gte", year=start, rolling=_offset(start - NEWEST))]
            recent = FilterRequirement(id="recent", input_id=DATASET, predicates=tuple(kept))
            return [recent, _growth("recent", start, rolling_start=True)], start
        case _:
            kept = [PeriodThreshold(op="gte", year=start, rolling=_offset(start - NEWEST))]
            recent = FilterRequirement(id="recent", input_id=DATASET, predicates=tuple(kept))
            total = AggregateRequirement(
                id="total", input_id="recent", reduction=Reduction.SUM, group_by=(GroupBy.PERIOD,)
            )
            return [recent, total], start
    return [FilterRequirement(id="recent", input_id=DATASET, predicates=tuple(kept))], start


def cases(seed: int, count: int) -> Iterator[Case]:
    """`count` admitted plans drawn from `seed`, each recorded, compiled and with its three worlds."""
    rng = random.Random(seed)
    made = 0
    while made < count:
        world = _world(rng)
        steps, reads_from = _steps(rng, world.first)
        stated = reads_from if rng.random() < 1 / 3 and reads_from > world.first else world.first
        dataset = DatasetRequirement(
            id=DATASET,
            decision_ids=("measure",),
            measure="external debt service paid",
            entities=[Entity(name="Kenya", role="debtor")],
            period=Period(start=stated, end=NEWEST),
            period_end_open=True,
            unit=Unit(currency="USD", prices="current"),
        )
        worlds = {
            WorldKind.SAME: world,
            WorldKind.NEWER: _newer(world, rng),
            WorldKind.LAGGING: _lagging(world, rng),
        }
        try:
            plan = Plan(
                question="How has Kenya's debt service moved in recent years?",
                decisions=[_DECISION],
                evidence_requirements=[dataset, *steps],
                first_years=[FirstYear(requirement_id=DATASET, year=world.first)],
                status=PlanStatus.DRAFT,
            )
        except ValueError:
            continue
        made += 1
        recorded = with_authored_requirements(plan, plan.evidence_requirements)
        translation = build_translation(recorded, dict.fromkeys(_datasets(recorded), _SPEC))
        yield Case(
            plan=recorded,
            translation=translation,
            workflow=_compiled(recorded, translation).workflow,
            worlds=worlds,
        )


def read(workflow: Workflow, anchor: int) -> tuple[AnchorRead, ...]:
    """Each group of `workflow`'s rolling steps, read over a world whose newest year is `anchor`."""
    steps = {step.id: step for step in workflow.steps}
    return tuple(
        AnchorRead(
            steps=group.steps,
            planned=planned_anchor(group, steps),
            result=AnchorResult(anchor_key=_ANCHOR_KEY, period=str(anchor)),
        )
        for group in anchor_groups(workflow)
    )


def outcome(plan: Plan, world: World, anchor: int) -> Outcome:
    """How a run of `plan` over `world` ends with its rolling years written from `anchor`."""
    by_id = {r.id: r for r in plan.evidence_requirements}
    recent = by_id.get("recent")
    growth = by_id.get("growth")
    newest = max(last for _, last in world.series.values())
    reader = growth if isinstance(growth, GrowthRequirement) else recent
    assert reader is not None
    (retrieved,) = inputs_of(reader)
    while not isinstance(by_id[retrieved], DatasetRequirement):
        (retrieved,) = inputs_of(by_id[retrieved])
    held = frame_shapes(plan)[retrieved].period
    assert held is not None

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
        return max(first, held.start) <= at <= last

    if isinstance(growth, GrowthRequirement):
        assert growth.rolling is not None
        start = year(growth.window_start, growth.rolling.window_start)
        end = year(growth.window_end, growth.rolling.window_end)
        if end <= start or end > newest or not (kept(start) and kept(end)):
            return Outcome.EMPTY
        if not any(holds(key, start) and holds(key, end) for key in world.series):
            return Outcome.MISSED
        return Outcome.RAN
    years = [at for at in range(world.first, newest + 1) if kept(at)]
    if not years:
        return Outcome.EMPTY
    if not any(holds(key, at) for key in world.series for at in years):
        return Outcome.MISSED
    return Outcome.RAN


def translation(case: Case) -> Translated:
    """The plan and workflow translation records for `case`, read over the world it was planned over."""
    assert case.workflow is not None
    groups = read(case.workflow, case.worlds[WorldKind.SAME].anchor)
    if not any(group.behind for group in groups):
        return Translated(plan=case.plan, workflow=case.workflow)
    restated = restated_to_the_data(case.plan, case.translation, case.workflow, groups)
    if isinstance(restated, CompileResult):
        return Translated(plan=None, workflow=None, refused=Outcome.RESTATEMENT_REFUSED)
    plan, moved = restated
    workflow = _compiled(plan, moved).workflow
    if workflow is None:
        return Translated(plan=None, workflow=None, refused=Outcome.RESTATEMENT_UNCOMPILED)
    return Translated(plan=plan, workflow=workflow)


def run_admitted(case: Case, kind: WorldKind) -> Outcome:
    world = case.worlds[kind]
    return outcome(case.plan, world, world.anchor)


def run_translated(translated: Translated, case: Case, kind: WorldKind) -> Outcome:
    """A run of `translated`, the translation of `case`, in one of its worlds."""
    if translated.refused is not None:
        return translated.refused
    assert translated.plan is not None and translated.workflow is not None
    world = case.worlds[kind]
    if any(group.behind for group in read(translated.workflow, world.anchor)):
        return Outcome.ANCHOR_BEHIND
    return outcome(translated.plan, world, world.anchor)


@dataclass(frozen=True)
class Tally:
    """How every run of a seed's cases ended, by side and world."""

    admitted: Mapping[WorldKind, Counter[Outcome]]
    translated: Mapping[WorldKind, Counter[Outcome]]
    # Plans admitted that do not compile, which neither side runs.
    uncompiled: int
    # Plans with a retrieval starting before its dataset's stated start.
    whole_history: int
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
    uncompiled = whole_history = restated = 0
    for case in cases(seed, count):
        if case.workflow is None:
            uncompiled += 1
            continue
        whole_history += _reads_whole_history(case.plan)
        recorded = translation(case)
        restated += recorded.plan is not None and recorded.plan is not case.plan
        for kind in WorldKind:
            admitted[kind][run_admitted(case, kind)] += 1
            translated[kind][run_translated(recorded, case, kind)] += 1
        if run_admitted(case, WorldKind.SAME) is Outcome.EMPTY:
            admitted_empties[run_translated(recorded, case, WorldKind.SAME)] += 1
        if recorded.workflow is not None:
            lagging = case.worlds[WorldKind.LAGGING]
            if any(group.behind for group in read(recorded.workflow, lagging.anchor)):
                dropped[run_translated(recorded, case, WorldKind.LAGGING)] += 1
    return Tally(
        admitted=admitted,
        translated=translated,
        uncompiled=uncompiled,
        whole_history=whole_history,
        restated=restated,
        admitted_empties=admitted_empties,
        dropped=dropped,
    )


def _reads_whole_history(plan: Plan) -> bool:
    """Whether a retrieval of `plan` starts before its dataset's stated start."""
    shapes = frame_shapes(plan)
    return any(
        (period := shapes[r.id].period) is not None and period.start < r.period.start
        for r in plan.evidence_requirements
        if isinstance(r, DatasetRequirement)
    )

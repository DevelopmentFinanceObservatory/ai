"""The sentences that say a read drew, or stopped short of, partial or projected years.

A source's partly reported years (CRS 2025) and the projected years continuing
a reported series (IDS 2026 to 2030) are left out of a read unless its window
opts in. A get_data manifest states every such run its source holds on
`coverage.held_back` (`common.held_back.HeldBack`), one per kind and source,
each marked `served` where the read drew it; Search states the same runs on
`resolved.coverage`. Every sentence and line here is built from those runs
alone, so one run gives one fact. The closing line of a Studio reply and a
card's label state the sentences (`held_back_note`, `served_note`). What a
model reads, data-mcp's digests and the harness's answer notes, states each
run as a terse data line (`held_fact_line`) that a reply cannot lift whole,
so the reply says the run once, in the code's closing line.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from common.held_back import (
    HeldBack,
    ReportedShare,
    held_back_of,
    reached_held_back,
    reaches_left_out,
)

# What a sentence calls a source whose display name is not stated.
_UNNAMED_SOURCE = "the source"

# The closing sentence of a reply whose turn asked about the future and whose
# cards hold no projections.
NO_PROJECTIONS_NOTE = "The data holds no projections."


def held_back_note(run: HeldBack) -> str:
    """The sentence saying a read left `run`'s years out.

    A partial run names the year the figures end in, the one before it, and
    the share of the prior year it holds where that is known. A projected run
    names its source as `source_name` gives it, "The source" when unstated.
    """
    years = _years(run.start_year, run.end_year)
    if run.kind == "projected":
        return (
            f"{_capitalised(_source_name(run.source_name))} also publishes projections"
            f" for {years}, which these figures leave out."
        )
    share = run.reported_share
    shared = f" ({_share_words(share)})" if share is not None else ""
    verb = "is" if run.start_year == run.end_year else "are"
    return (
        f"{years} {verb} partly reported{shared} and {verb} left out of these figures,"
        f" which end in {run.start_year - 1}."
    )


def served_note(run: HeldBack) -> str:
    """The sentence saying a frame holds `run`'s years.

    A partial run names its share of the prior year where known and the
    newest complete year, the one before it starts. A projected run names its
    source as `source_name` gives it and the latest reported year.
    """
    years = _years(run.start_year, run.end_year)
    if run.kind == "projected":
        return (
            f"Figures for {years} are {_projections_of(run.source_name)}."
            f" The latest reported year is {run.start_year - 1}."
        )
    share = run.reported_share
    shared = f": {_share_words(share)}" if share is not None else ""
    verb = "is" if run.start_year == run.end_year else "are"
    return (
        f"{years} {verb} partly reported{shared}. The newest complete year is {run.start_year - 1}."
    )


def held_fact_line(run: HeldBack) -> str:
    """The data line a model reads for `run`, such as
    "held: 2025 partial, 0.5% of 2024 value reported, left out" or
    "held: 2026-2030 projected (World Bank IDS), served".

    A partial run names the share of the prior year it holds where known, on
    its basis ("value" or "series"). A projected run names its source where
    stated. The line ends "served" where the read drew the run, else "left
    out".
    """
    years = (
        str(run.start_year)
        if run.start_year == run.end_year
        else (f"{run.start_year}-{run.end_year}")
    )
    if run.kind == "projected":
        what = f"{years} projected" + (f" ({run.source_name})" if run.source_name else "")
    elif (share := run.reported_share) is not None:
        what = f"{years} partial, {_percent(share.share)} of {share.against} {share.basis} reported"
    else:
        what = f"{years} partial"
    return f"held: {what}, {'served' if run.served else 'left out'}"


class HeldFact(BaseModel):
    """One partial or projected run of years a read drew (`served`) or left out,
    with the sentence that says so (`note`) and the data line a model reads
    of it (`line`, `held_fact_line`).

    `start` and `end` are the run's first and last years as period labels;
    `source` is the provenance of the source that marks or continues the run,
    and `source_name` its catalogue name, where stated. `share`, `basis` and
    `against` are a partial run's `ReportedShare`, where known. A run is one
    per kind and source (`merge_held`).
    """

    model_config = ConfigDict(frozen=True)
    kind: Literal["partial", "projected"]
    start: str
    end: str
    source: str
    served: bool
    note: str
    line: str
    source_name: str | None = Field(default=None, exclude_if=lambda name: name is None)
    share: float | None = Field(default=None, exclude_if=lambda share: share is None)
    basis: Literal["value", "series"] | None = Field(
        default=None, exclude_if=lambda basis: basis is None
    )
    against: int | None = Field(default=None, exclude_if=lambda against: against is None)


def held_back_fact(run: HeldBack) -> HeldFact:
    """The fact a stated `run` gives: drawn (`served_note`) when the read's
    `include` served it, else left out (`held_back_note`)."""
    share = run.reported_share
    return HeldFact(
        kind=run.kind,
        start=str(run.start_year),
        end=str(run.end_year),
        source=run.source,
        served=run.served,
        note=served_note(run) if run.served else held_back_note(run),
        line=held_fact_line(run),
        source_name=run.source_name,
        share=share.share if share else None,
        basis=share.basis if share else None,
        against=share.against if share else None,
    )


def held_facts(manifest: Mapping[str, Any], *, every_run: bool = False) -> list[HeldFact]:
    """One fact per run of a get_data `manifest`'s `coverage.held_back` that
    its window reaches (`reached_held_back`, `held_back_fact`). With
    `every_run`, each run whatever the window, for a card that decides reach
    against the window it draws (`reaches`). dfo-apis states one run per kind
    and source, so each run is one fact.

    dfo-apis states a run whole whatever the window, so a served run ends at
    the window's `end_year`, the last year the read holds, where the window
    ends inside it (`_read_end`). A read of 2020 to 2027 serving IDS's 2026
    to 2030 projections holds 2026 to 2027. Its first year stays whole, since
    the year before it is the latest reported one."""
    coverage = manifest.get("coverage")
    coverage = coverage if isinstance(coverage, Mapping) else {}
    definition = manifest.get("definition")
    label = definition.get("spec_id") if isinstance(definition, Mapping) else None
    read = held_back_of if every_run else reached_held_back
    window = coverage.get("window")
    end = window.get("end_year") if isinstance(window, Mapping) else None
    return [
        held_back_fact(_read_end(run, end=end if isinstance(end, int) else None))
        for run in read(coverage, label=str(label) if label else None)
    ]


def reached_runs(coverage: Mapping[str, Any]) -> tuple[HeldBack, ...]:
    """The runs of a Search part's or a read's `coverage` its window reaches
    (`reached_held_back`), each served run ending where the window ends inside
    it (`_read_end`), as `held_facts` words them."""
    window = coverage.get("window")
    end = window.get("end_year") if isinstance(window, Mapping) else None
    end = end if isinstance(end, int) else None
    return tuple(_read_end(run, end=end) for run in reached_held_back(coverage))


def _read_end(run: HeldBack, *, end: int | None) -> HeldBack:
    """`run` with a served run's last year cut to `end`, the last year its
    read holds, where `end` falls inside it."""
    if not run.served or end is None or not run.start_year <= end < run.end_year:
        return run
    return run.model_copy(update={"end_year": end})


def merge_held(facts: Iterable[HeldFact]) -> list[HeldFact]:
    """The runs of several reads' `facts`, one per kind and source: drawn
    (`served`) where any fact of it is drawn, since a frame holds it, from
    the earliest first year to the newest last year of those facts, with
    the share and name of the one ending newest. Drawn runs come first, each
    in order of its first fact."""
    groups: dict[tuple[str, str], list[HeldFact]] = {}
    for fact in facts:
        groups.setdefault((fact.kind, fact.source), []).append(fact)
    merged: list[HeldFact] = []
    for group in groups.values():
        kept = [fact for fact in group if fact.served] or group
        newest = max(kept, key=lambda fact: int(fact.end))
        share = (
            ReportedShare(basis=newest.basis, share=newest.share, against=newest.against)
            if newest.basis is not None and newest.share is not None and newest.against is not None
            else None
        )
        run = HeldBack(
            kind=newest.kind,
            served=newest.served,
            source=newest.source,
            source_name=newest.source_name,
            start_year=min(int(fact.start) for fact in kept),
            end_year=int(newest.end),
            reported_share=share,
        )
        merged.append(held_back_fact(run))
    return sorted(merged, key=lambda fact: not fact.served)


def held_runs(manifests: Iterable[Mapping[str, Any]]) -> list[HeldFact]:
    """Every partial or projected run the get_data `manifests` state, each
    once (`held_facts` with `every_run`, `merge_held`). A run keeps its own
    first year whatever window a card draws from it, a served run ends where
    its read ends, and a card decides which it reaches (`reaches`,
    `drawn_window`)."""
    return merge_held(
        fact for manifest in manifests for fact in held_facts(manifest, every_run=True)
    )


def drawn_window(
    drawn: Mapping[str, Any], manifests: Iterable[Mapping[str, Any]]
) -> Mapping[str, Any]:
    """The `{start, end}` window a card reaches runs against (`reaches`):
    `drawn`, the window the card states, where it states an end, else the
    span of the `coverage.window` of the `manifests` it is drawn from, so a
    table stating no window reaches only the runs its reads reach."""
    if period_year(drawn.get("end")) is not None:
        return drawn
    windows = [
        window
        for manifest in manifests
        if isinstance(coverage := manifest.get("coverage"), Mapping)
        if isinstance(window := coverage.get("window"), Mapping)
    ]
    starts = [year for window in windows if isinstance(year := window.get("start_year"), int)]
    ends = [year for window in windows if isinstance(year := window.get("end_year"), int)]
    if not ends or len(ends) < len(windows):
        return drawn
    return {
        "start": str(min(starts)) if starts and len(starts) == len(windows) else None,
        "end": str(max(ends)),
    }


def held_period(fact: HeldFact) -> dict[str, Any]:
    """`fact` as a wire `HeldPeriod` dict (`harness.wire.Freshness.held`)."""
    return {
        "status": fact.kind,
        "start": fact.start,
        "end": fact.end,
        "served": fact.served,
        "source_name": fact.source_name,
        "share": fact.share,
        "basis": fact.basis,
        "against": str(fact.against) if fact.against is not None else None,
    }


def reaches(period: Mapping[str, Any], *, drawn: Mapping[str, Any]) -> bool:
    """Whether a card drawing `drawn`, a `{start, end}` window of period
    labels, reaches the run `period` (a wire `HeldPeriod` dict).

    A drawn (`served`) run is reached where it overlaps `drawn`. A run left
    out is reached where `drawn` ends at or after the year before it starts,
    so the figures would run on into it. A card stating no window end, or a
    run naming no year, reaches it.
    """
    first, last = period_year(drawn.get("start")), period_year(drawn.get("end"))
    start, end = period_year(period.get("start")), period_year(period.get("end"))
    if start is None or end is None or last is None:
        return True
    if not period.get("served"):
        return reaches_left_out(start, last_year=last)
    return (first is None or end >= first) and start <= last


def held_period_notes(runs: Iterable[Mapping[str, Any]]) -> list[str]:
    """The sentences the partial and projected runs of cards state, each run
    once (`held_period_facts`). The runs a card drew (`served_note`) come
    first, then those a card left out (`held_back_note`)."""
    return list(dict.fromkeys(fact.note for fact in held_period_facts(runs)))


def held_period_facts(runs: Iterable[Mapping[str, Any]]) -> list[HeldFact]:
    """The partial and projected runs of cards, each once (`merge_held`).
    `runs` are the wire `HeldPeriod` entries a turn card records with the
    `source` of each (`harness.studio_state.card_runs`). The runs a card drew
    come first. A run left out is dropped where a card drew the same run,
    since a figure on the canvas holds it. An entry whose `start` names no
    year is skipped.
    """
    facts: list[HeldFact] = []
    for run in runs:
        kind = run.get("status")
        start, end = period_year(run.get("start")), period_year(run.get("end"))
        if kind not in ("partial", "projected") or start is None:
            continue
        held = HeldBack(
            kind=kind,
            served=bool(run.get("served")),
            source=str(run.get("source") or ""),
            source_name=run.get("source_name"),
            start_year=start,
            end_year=end if end is not None and end >= start else start,
            reported_share=_period_share(run) if kind == "partial" else None,
        )
        facts.append(held_back_fact(held))
    return merge_held(facts)


def _period_share(period: Mapping[str, Any]) -> ReportedShare | None:
    """A wire `HeldPeriod`'s share of its prior period, where it states one."""
    against = period_year(period.get("against"))
    if period.get("share") is None or against is None:
        return None
    try:
        return ReportedShare.model_validate(
            {"share": period["share"], "basis": period.get("basis"), "against": against}
        )
    except ValidationError:
        return None


def held_lines(manifests: Iterable[Mapping[str, Any]]) -> list[str]:
    """One data line (`held_fact_line`) per run the get_data `manifests`
    state that their windows reach (`held_facts`), each run once across them
    (`merge_held`)."""
    facts = merge_held(fact for manifest in manifests for fact in held_facts(manifest))
    return list(dict.fromkeys(fact.line for fact in facts))


def holds_projections(manifest: Mapping[str, Any]) -> bool:
    """Whether a get_data `manifest` states a projected run, served or left
    out (`coverage.held_back`)."""
    coverage = manifest.get("coverage")
    return isinstance(coverage, Mapping) and any(
        run.kind == "projected" for run in held_back_of(coverage)
    )


def include_hint(runs: Iterable[HeldBack]) -> str | None:
    """The sentence telling a get_data caller how to be served the runs it was
    not served, when the user's words license them. It asks for a
    `scope.window` that is a range naming those years, with the `include`
    member added, since a `latest` or open window cannot serve a partial or
    projected year as the newest figure. `None` when every run was served or
    there are none."""
    kinds = sorted({run.kind for run in runs if not run.served})
    if not kinds:
        return None
    listed = ", ".join(f'"{kind}"' for kind in kinds)
    return (
        "When the user names these years or asks about projections, get_data serves"
        " them if its scope.window is a range naming those years and adds"
        f' "include": [{listed}].'
    )


def _years(start: int, end: int) -> str:
    return str(start) if start == end else f"{start} to {end}"


def _share_words(share: ReportedShare) -> str:
    """How much of `share.against` a partial period holds so far: "0.1% of
    2024's value so far", or of its "series" on the series basis."""
    return f"{_percent(share.share)} of {share.against}'s {share.basis} so far"


def _percent(share: float) -> str:
    """`share` as a percentage the way the card label words it: one decimal
    below 10%, whole above, and "less than 0.1%" for a share under 0.05%."""
    value = share * 100
    if 0 < value < 0.05:
        return "less than 0.1%"
    if value < 10:
        return f"{round(value, 1):g}%"
    return f"{round(value)}%"


def _source_name(name: str | None) -> str:
    return name if name else _UNNAMED_SOURCE


def _capitalised(name: str) -> str:
    """A source name at the head of a sentence: the name as given, such as
    "World Bank IDS", or "The source" for an unnamed one."""
    return "The source" if name == _UNNAMED_SOURCE else name


def _projections_of(name: str | None) -> str:
    return f"{name} projections" if name else f"{_UNNAMED_SOURCE}'s projections"


def period_year(label: object) -> int | None:
    """The year a period label falls in: `2025` for `2025`, `"2025"` or a
    finer-grained `"2025-06"`; `None` for a label naming no year. A partial
    mark at its source's own grain and a run of whole years are compared on
    this year."""
    if isinstance(label, int):
        return label
    text = str(label) if label is not None else ""
    return int(text[:4]) if len(text) >= 4 and text[:4].isdigit() else None

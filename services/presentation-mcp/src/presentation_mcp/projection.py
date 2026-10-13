"""The text projection: a resolved revision as bounded Markdown.

A write_component or describe_component result leads its `content` with
this, beside the model vocabulary, where `server.py`'s `include_projection`
rule calls for it. It is built from the
envelope alone: a one-line description derived from the component's
structured fields, stamped with the component id and revision; the
model-authored title, announced as the model's; a table of the visible data;
and the footer (freshness, warnings, citation, as `card_facts` reads them). A projection
without its data table keeps every line but the line_chart, bar_chart, and
table bodies, and says where the table is (`ROWLESS_NOTICE`, or
`UNKNOWN_HOST_ROWLESS_NOTICE` for a client whose capabilities the server
cannot see).

The projection is bounded whatever the payload. Every free-text field but a
unit is clipped to a character limit, and table rows are added only while the
table stays within `MAX_TABLE_BYTES`. A unit is printed whole, so two units
never read as one: a table's unit cells count toward `MAX_TABLE_BYTES`, and
the paragraph naming each series' unit adds series only while it stays within
`MAX_UNITS_BYTES`. So the largest payload the catalogue admits (50 series of
500 points, or a 2,000-row table of 32 columns) projects to at most
`MAX_PROJECTION_BYTES`. What the bounds cut, the projection says, and it
names the frame that holds the rest. A final cut holds the same bound for an
envelope outside the catalogue's limits.

A revise that leaves the visible data as it was gets `delta_projection`, one
line naming what the patch changed, in place of the table.

`app_view` reuses `describe` and the `*_text` helpers, so the text and
interactive views word labels, sources, and warnings alike.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

MAX_ROWS = 20
MAX_VALUE_COLUMNS = 5
MAX_CELL_CHARS = 32
MAX_TABLE_BYTES = 4096
MAX_UNITS_BYTES = 2048
MAX_TEXT_CHARS = 120
MAX_NAMED_ENTITIES = 3
MAX_WARNINGS = 3
MAX_SOURCES = 5
MAX_URL_CHARS = 300
# A narration is its text, so the projection keeps all of it. The catalogue
# caps it at 2,000 characters.
MAX_NOTE_CHARS = 2000
MAX_PROJECTION_BYTES = 12 * 1024
# What closes a projection cut at `MAX_PROJECTION_BYTES`.
PROJECTION_CUT_NOTICE = "The rest of this projection is cut at its 12 KiB bound."
# What stands in for a line_chart, bar_chart, or table body in a projection
# without its data table, for a client that declared it draws views.
ROWLESS_NOTICE = (
    "The data table is left out, since this client draws the component. "
    "`describe_component` with `include_projection` set to `full` returns it."
)
# The same, for a client whose capabilities the server cannot see.
UNKNOWN_HOST_ROWLESS_NOTICE = (
    "The data table is left out. This server cannot tell whether this client "
    "draws the component as a view. `describe_component` with "
    "`include_projection` set to `full` returns the table."
)

_MARKDOWN_ACTIVE = re.compile(r"([\\`*_\[\]<>|~#])")
_HTTPS_URL = re.compile(r"https://[^\s<>]+")


def text_projection(envelope: Mapping[str, Any], *, rowless_notice: str | None = None) -> str:
    """The envelope as bounded Markdown.

    Args:
        envelope: The resolved revision, `{schema_version, catalog_version,
            component_id, revision, provenance, component}`, with a
            `description` on a described card.
        rowless_notice: Text put in place of a line_chart, bar_chart, or
            table body, or `None` to keep the body. Other types project
            whole either way.

    Returns:
        At most `MAX_PROJECTION_BYTES` of UTF-8 Markdown.
    """
    component = envelope["component"]
    provenance = envelope["provenance"]
    facts = card_facts(envelope)
    stamp = f"`{envelope['component_id']}`, revision {envelope['revision']}"
    lines = [f"{_md(describe(component))} ({stamp})."]
    title = title_text(component)
    if title is not None:
        lines.append(f"Title, written by the model: {_md(title)}")

    body = _BODIES.get(component["type"])
    if rowless_notice is not None and component["type"] in _DATA_TABLE_TYPES:
        lines.extend(["", rowless_notice])
    elif body is not None:
        body_lines, cuts = body(component, facts)
        lines.extend(["", *body_lines])
        if cuts:
            lines.extend(["", _cut_line(cuts, provenance)])

    footer = _footer(facts)
    if footer:
        lines.extend(["", *footer])
    return _bounded("\n".join(lines))


def delta_projection(envelope: Mapping[str, Any], changed: Sequence[Mapping[str, Any]]) -> str:
    """A revise that left the visible data unchanged, as one stamped line.

    Args:
        envelope: The resolved revision the revise minted.
        changed: The write reply's `changed` entries, `{"field", "prior",
            "new"}`, or `{"field"}` alone where the value is not disclosed.

    Returns:
        The changes, e.g. "scale linear to log, values unchanged
        (`cmp_1`, revision 3)."
    """
    stamp = f"`{envelope['component_id']}`, revision {envelope['revision']}"
    if not changed:
        return f"Nothing changed ({stamp})."
    changes = "; ".join(_md(_change_text(entry)) for entry in changed)
    return f"{changes}, values unchanged ({stamp})."


def describe(component: Mapping[str, Any]) -> str:
    """The component's one-line description, derived from its structured fields.

    It names the component type, what is measured, the entities or
    categories shown, and the period covered. The model-authored title never
    stands in for it.
    """
    kind = component["type"]
    match kind:
        case "line_chart":
            window = component["x_axis"]["window"]
            text = (
                f"Line chart of {_clip(component['y_axis']['label'], MAX_TEXT_CHARS)} for "
                f"{_entities(component['series'])}, {window['start']} to {window['end']}"
            )
        case "bar_chart":
            axis = component["category_axis"]
            by = f" by {_clip(axis['category_dimension'])}" if axis["mode"] == "entity" else ""
            span = (
                axis["period"]
                if axis["mode"] == "entity"
                else f"{axis['window']['start']} to {axis['window']['end']}"
            )
            text = (
                f"Bar chart of {_clip(component['value_axis']['label'], MAX_TEXT_CHARS)}{by} "
                f"for {_entities(component['series'])}, {span}"
            )
        case "table":
            rows = component["rows"]
            dimensions = [
                _clip(column["label"])
                for column in component["columns"]
                if column["key"] not in ("period", "value")
            ]
            by = f" by {_named(dimensions)}" if dimensions else ""
            text = f"Table of {len(rows):,} rows{by}, {_span(sorted({r['period'] for r in rows}))}"
        case "stat_card":
            text = (
                f"Stat card of {_clip(component['label'], MAX_TEXT_CHARS)}, {component['period']}"
            )
        case "note_narration":
            text = "Narration in the model's voice"
        case "source_card":
            count = len(component["source_refs"])
            text = f"Source card listing {count} source{'' if count == 1 else 's'}"
        case _:
            text = f"A {kind} component, a type this server does not draw"
    return text


def title_text(component: Mapping[str, Any]) -> str | None:
    """The model-authored title, clipped, or `None` for a member without one
    and for a type this server does not draw."""
    if component["type"] not in _BODIES:
        return None
    title = component.get("title")
    return None if title is None else _clip(title["text"], MAX_TEXT_CHARS)


def unit_text(unit: Any) -> str:
    """A unit slot as words on one line, whole: `""` for null, a string as
    given, a structured unit as currency, magnitude, and price basis, and a
    composed unit as `<numerator> per <denominator>`, each side worded the
    same way and a StatVar side by its label."""
    if unit is None:
        return ""
    if isinstance(unit, str):
        return _one_line(unit)
    if "numerator" in unit:
        return f"{_unit_part_text(unit['numerator'])} per {_unit_part_text(unit['denominator'])}"
    parts = [unit.get("currency", "")]
    if unit.get("magnitude") not in (None, "units"):
        parts.append(unit["magnitude"])
    words = " ".join(part for part in parts if part)
    match unit.get("prices"), unit.get("base_year"):
        case "constant", int(year):
            prices = f"constant {year} prices"
        case "constant", _:
            prices = "constant prices"
        case "current", _:
            prices = "current prices"
        case _:
            prices = ""
    return ", ".join(part for part in (words, prices) if part)


def _unit_part_text(part: Any) -> str:
    """One side of a composed unit: a StatVar part's label, else `unit_text`."""
    if not isinstance(part, str) and "stat_var" in part:
        return _one_line(part["label"])
    return unit_text(part)


def number_text(value: float) -> str:
    """A value as digits with thousands separators, integral values without a decimal."""
    if isinstance(value, float) and value.is_integer() and abs(value) < 1e15:
        value = int(value)
    return f"{value:,}"


def signed_text(value: float, *, number: Callable[[float], str] = number_text) -> str:
    """`value` written by `number`, with a leading `+` on a positive value."""
    return f"{'+' if value > 0 else ''}{number(value)}"


def cell_text(entry: Mapping[str, Any]) -> str:
    """A point or row's value, or its status where it carries no value."""
    if "value" in entry:
        return number_text(entry["value"])
    return entry.get("status", "")


def change_text(change: Mapping[str, Any], *, number: Callable[[float], str] = number_text) -> str:
    """A stat card's change: its signed value, written by `number`, its
    derivation, and its basis periods."""
    value = signed_text(change["value"], number=number)
    start, end = change["basis"]
    if change["derivation"] == "percent_change":
        return f"{value}% (percent change, {start} to {end})"
    return f"{value} (difference, {start} to {end})"


def shared_unit(units: Sequence[Any]) -> str | None:
    """The one unit every entry shares, as words, or `None` when they differ
    or there is no entry. Units are compared by their words, so two the
    payload spells differently that read alike are one unit."""
    texts = {unit_text(unit) for unit in units}
    return texts.pop() if len(texts) == 1 else None


@dataclass(frozen=True)
class Source:
    """A source a card names. `key` is what a component's `source_ref` or
    `source_refs` entry points at: a `provenance.sources` id, or a description
    source's `ref`, which is `None` for an input whose id is unknown."""

    key: str | None
    name: str
    licence: str | None
    retrieved: str | None
    url: str | None


@dataclass(frozen=True)
class Freshness:
    """How current a card's data is, each date `None` where the card states none.

    `newer_download` is a source download after `last_refreshed` that the card's data
    does not hold.
    """

    data_through: str | None
    partial_period: bool
    last_refreshed: str | None
    next_expected: str | None
    newer_download: str | None


@dataclass(frozen=True)
class Caveat:
    """A warning or caveat: its kind, and the sentence saying what it covers."""

    kind: str
    detail: str


@dataclass(frozen=True)
class CardFacts:
    """What a card says about its data: its sources, freshness, caveats, and
    citation. A narration note, which draws no data, states none of them."""

    sources: tuple[Source, ...]
    freshness: Freshness | None
    caveats: tuple[Caveat, ...]
    citation: str | None


def card_facts(envelope: Mapping[str, Any]) -> CardFacts:
    """The card's sources, freshness, caveats, and citation.

    A card with a `description` keeps lineage alone in `provenance`, so they
    come from the description: its `sources`, each retrieved on the `cite`
    date and linked to no URL; its `freshness`, whose latest period is the
    data-through period, whose source download at mint is the last refresh,
    whose later download is a newer download the card does not hold, and whose
    next release is the next update; its `caveats` with their
    `text`; and `cite.text`. A card without one reads them from `provenance`,
    where a narration note carries none.
    """
    description = envelope.get("description")
    if description is not None:
        return _described_facts(description)
    provenance = envelope["provenance"]
    freshness = provenance.get("freshness")
    return CardFacts(
        sources=tuple(
            Source(
                key=source["id"],
                name=source["name"],
                licence=source["licence"],
                retrieved=source["retrieved"],
                url=source["url"],
            )
            for source in provenance.get("sources", ())
        ),
        freshness=None
        if freshness is None
        else Freshness(
            data_through=freshness["data_through"],
            partial_period=freshness["partial_period"],
            last_refreshed=freshness["last_refreshed"],
            next_expected=freshness["next_expected"],
            newer_download=None,
        ),
        caveats=tuple(
            Caveat(kind=warning["kind"], detail=warning["detail"])
            for warning in provenance.get("warnings", ())
        ),
        citation=provenance.get("citation"),
    )


def _described_facts(description: Mapping[str, Any]) -> CardFacts:
    cite = description["cite"]
    freshness = description["freshness"]
    return CardFacts(
        sources=tuple(
            Source(
                key=source["ref"],
                name=source["label"],
                licence=source.get("licence"),
                retrieved=cite.get("retrieved"),
                url=None,
            )
            for source in description["sources"]
        ),
        freshness=Freshness(
            data_through=freshness.get("latest_period"),
            partial_period=False,
            last_refreshed=freshness.get("source_downloaded_at_mint"),
            next_expected=freshness.get("next_release"),
            newer_download=freshness.get("source_downloaded_since"),
        ),
        caveats=tuple(
            Caveat(kind=caveat["kind"], detail=caveat["text"]) for caveat in description["caveats"]
        ),
        citation=cite["text"],
    )


def sources_for(facts: CardFacts, refs: Sequence[str] | None = None) -> list[Source]:
    """The card's sources, in order, restricted to `refs` when given."""
    if refs is None:
        return list(facts.sources)
    by_key = {source.key: source for source in facts.sources}
    return [by_key[ref] for ref in refs if ref in by_key]


def source_url(source: Source) -> str | None:
    """A source's URL where it is https and short enough to show whole."""
    url = source.url
    if url is None or len(url) > MAX_URL_CHARS or _HTTPS_URL.fullmatch(url) is None:
        return None
    return url


def freshness_text(facts: CardFacts) -> str | None:
    """The freshness line, or `None` when the card states none."""
    freshness = facts.freshness
    if freshness is None:
        return None
    parts = []
    if freshness.data_through:
        partial = ", the last period partial" if freshness.partial_period else ""
        parts.append(f"Data through {freshness.data_through}{partial}")
    if freshness.last_refreshed:
        parts.append(f"last refreshed {freshness.last_refreshed}")
    if freshness.newer_download:
        parts.append(f"a newer source download on {freshness.newer_download} is not in this card")
    if freshness.next_expected:
        parts.append(f"next update expected {freshness.next_expected}")
    if not parts:
        return None
    text = "; ".join(parts)
    return text[0].upper() + text[1:] + "."


def warning_texts(
    facts: CardFacts,
    *,
    limit: int | None = MAX_TEXT_CHARS,
    count: int | None = MAX_WARNINGS,
) -> list[str]:
    """Up to `count` warnings as `kind: detail`, then a count of the rest, or
    every warning for `None`.

    Each detail is clipped to `limit` characters, or kept whole for `None`.
    """
    warnings = facts.caveats
    shown = warnings if count is None else warnings[:count]
    texts = [
        f"{warning.kind.replace('_', ' ')}: "
        f"{warning.detail if limit is None else _clip(warning.detail, limit)}"
        for warning in shown
    ]
    if len(warnings) > len(shown):
        texts.append(f"{len(warnings) - len(shown)} more warnings")
    return texts


def citation_text(facts: CardFacts) -> str | None:
    """The citation, clipped, or `None` for a card that cites nothing."""
    return None if facts.citation is None else _clip(facts.citation, MAX_TEXT_CHARS)


# --- bodies -------------------------------------------------------------------

type _Body = tuple[list[str], list[str]]


def _series_body(component: Mapping[str, Any], facts: CardFacts) -> _Body:
    """A period-by-series table, latest periods first to be kept."""
    del facts
    series = component["series"]
    shown = series[:MAX_VALUE_COLUMNS]
    cells = [{point["period"]: cell_text(point) for point in one["points"]} for one in shown]
    periods = sorted({period for by_period in cells for period in by_period})
    header = ["Period", *(one["label"] for one in shown)]
    rows = [[period, *(by_period.get(period, "") for by_period in cells)] for period in periods]
    table, kept = _table(header, rows, keep="last")

    unit_lines, unit_cuts = _unit_lines(shown, noun="series")
    cuts = []
    if kept < len(periods):
        cuts.append(f"the latest {kept} of {len(periods):,} periods")
    if len(shown) < len(series):
        cuts.append(f"{len(shown)} of {len(series)} series")
    return [*table, *unit_lines], [*cuts, *unit_cuts]


def _bar_body(component: Mapping[str, Any], facts: CardFacts) -> _Body:
    """Entity mode as one row per bar, units named for the bars kept; period
    mode as a series table."""
    axis = component["category_axis"]
    if axis["mode"] != "entity":
        return _series_body(component, facts)
    series = component["series"]
    header = [axis["category_dimension"], component["value_axis"]["label"]]
    rows = [[one["label"], cell_text(one["points"][0])] for one in series]
    table, kept = _table(header, rows, keep="first")
    cuts = [f"{kept} of {len(series)} bars"] if kept < len(series) else []
    unit_lines, unit_cuts = _unit_lines(series[:kept], noun="bars")
    return [*table, *unit_lines], [*cuts, *unit_cuts]


def _table_body(component: Mapping[str, Any], facts: CardFacts) -> _Body:
    """The table's own columns and rows, dimension columns capped first."""
    del facts
    columns = component["columns"]
    dimensions = [column for column in columns if column["key"] not in ("period", "value")]
    shown = [
        *dimensions[:MAX_VALUE_COLUMNS],
        *(column for column in columns if column["key"] in ("period", "value")),
    ]
    rows = component["rows"]
    unit = shared_unit([row["unit"] for row in rows])
    header = [column["label"] for column in shown]
    if unit is None:
        header.append("Unit")
    body = []
    for row in rows:
        cells = [_table_cell(row, column["key"]) for column in shown]
        if unit is None:
            cells.append(unit_text(row["unit"]))
        body.append(cells)
    table, kept = _table(
        header, body, keep="first", whole=() if unit is not None else (len(shown),)
    )

    lines = [*table, *(["", f"Unit: {_md(unit)}."] if unit else [])]
    cuts = []
    if kept < len(rows):
        cuts.append(f"the first {kept} of {len(rows):,} rows")
    if len(shown) < len(columns):
        cuts.append(f"{len(shown)} of {len(columns)} columns")
    return lines, cuts


def _stat_body(component: Mapping[str, Any], facts: CardFacts) -> _Body:
    """The headline value with its unit and period, then its change."""
    del facts
    unit = unit_text(component["unit"])
    value = number_text(component["value"])
    lines = [f"Value: **{value}**{f' {_md(unit)}' if unit else ''}, {component['period']}."]
    change = component.get("change")
    if change is not None:
        lines.append(f"Change: {change_text(change)}.")
    return lines, []


def _note_body(component: Mapping[str, Any], facts: CardFacts) -> _Body:
    """The narration's text, labelled as the model's."""
    del facts
    return [f"Model narration: {_md(_clip(component['text'], MAX_NOTE_CHARS))}"], []


def _source_body(component: Mapping[str, Any], facts: CardFacts) -> _Body:
    """One bullet per listed source: name, licence, retrieval date, and link,
    each but the name where the card states it."""
    sources = sources_for(facts, component["source_refs"])
    lines = []
    for source in sources[:MAX_SOURCES]:
        parts = [_md(_clip(source.name, MAX_TEXT_CHARS))]
        if source.licence:
            parts.append(f"licence {_md(_clip(source.licence, MAX_TEXT_CHARS))}")
        if source.retrieved:
            parts.append(f"retrieved {source.retrieved}")
        url = source_url(source)
        link = f": <{url}>" if url else ""
        lines.append(f"- {', '.join(parts)}{link}")
    cuts = [f"{MAX_SOURCES} of {len(sources)} sources"] if len(sources) > MAX_SOURCES else []
    return lines, cuts


_BODIES: dict[str, Callable[[Mapping[str, Any], CardFacts], _Body]] = {
    "line_chart": _series_body,
    "bar_chart": _bar_body,
    "table": _table_body,
    "stat_card": _stat_body,
    "note_narration": _note_body,
    "source_card": _source_body,
}

# The types whose body is a data table.
_DATA_TABLE_TYPES = frozenset({"line_chart", "bar_chart", "table"})


# --- helpers ------------------------------------------------------------------


def _table(
    header: Sequence[str],
    rows: Sequence[Sequence[str]],
    *,
    keep: Literal["first", "last"],
    whole: Collection[int] = (),
) -> tuple[list[str], int]:
    """A Markdown table of as many rows as fit `MAX_ROWS` and `MAX_TABLE_BYTES`.

    `keep="first"` keeps rows from the top. `keep="last"` keeps them from the
    bottom, still printed in order. Every cell is clipped to `MAX_CELL_CHARS`
    but those of the columns at the `whole` positions, which hold one-line text
    already; `MAX_TABLE_BYTES` bounds the table either way.

    Returns:
        The table's lines and how many of `rows` it kept.
    """
    lines = [_table_row(header), "| " + " | ".join("---" for _ in header) + " |"]
    size = sum(len(line.encode()) + 1 for line in lines)
    ordered = rows if keep == "first" else list(reversed(rows))
    kept: list[str] = []
    for row in ordered[:MAX_ROWS]:
        line = _table_row(row, whole=whole)
        size += len(line.encode()) + 1
        if size > MAX_TABLE_BYTES:
            break
        kept.append(line)
    if keep == "last":
        kept.reverse()
    return lines + kept, len(kept)


def _table_row(cells: Sequence[str], *, whole: Collection[int] = ()) -> str:
    return (
        "| "
        + " | ".join(
            _md(cell if position in whole else _clip(cell)) for position, cell in enumerate(cells)
        )
        + " |"
    )


def _table_cell(row: Mapping[str, Any], key: str) -> str:
    if key == "period":
        return row["period"]
    if key == "value":
        return cell_text(row)
    return row["dimensions"].get(key, "")


def _unit_lines(series: Sequence[Mapping[str, Any]], *, noun: str) -> _Body:
    """A `Unit:` paragraph when the series share a unit, else one naming each
    series' unit, for as many series as fit `MAX_UNITS_BYTES`; nothing when
    they share no unit at all.

    Returns:
        The paragraph's lines and what it cut, counted in `noun`.
    """
    unit = shared_unit([one["unit"] for one in series])
    if unit is not None:
        return (["", f"Unit: {_md(unit)}."] if unit else []), []
    kept: list[str] = []
    size = len("Units: .")
    for one in series:
        entry = f"{_md(_clip(one['label']))}: {_md(unit_text(one['unit']) or 'none')}"
        size += len(f"; {entry}".encode())
        if size > MAX_UNITS_BYTES:
            break
        kept.append(entry)
    lines = ["", f"Units: {'; '.join(kept)}."] if kept else []
    cuts = [f"the units of {len(kept)} of {len(series)} {noun}"] if len(kept) < len(series) else []
    return lines, cuts


def _change_text(entry: Mapping[str, Any]) -> str:
    """One changed field: `field prior to new`, or what is known of it."""
    field = _clip(str(entry["field"]))
    if "new" not in entry:
        return f"{field} changed"
    prior, new = entry.get("prior"), entry["new"]
    if new is None:
        return f"{field} cleared"
    if prior is None:
        return f"{field} set to {_scalar_text(new)}"
    return f"{field} {_scalar_text(prior)} to {_scalar_text(new)}"


def _scalar_text(value: Any) -> str:
    return _clip(value if isinstance(value, str) else json.dumps(value))


def _cut_line(cuts: Sequence[str], provenance: Mapping[str, Any]) -> str:
    """The truncation disclosure, naming the frame that holds the rest."""
    where = ""
    if provenance["inputs"]:
        frame = provenance["inputs"][0]
        where = f"; the full data is in frame `{frame['spec_id']}` at `{frame['content_version']}`"
    return f"Showing {' and '.join(cuts)}{where}."


def _footer(facts: CardFacts) -> list[str]:
    lines = []
    freshness = freshness_text(facts)
    if freshness is not None:
        lines.append(freshness)
    lines.extend(f"Warning, {_md(text)}" for text in warning_texts(facts))
    citation = citation_text(facts)
    if citation is not None:
        lines.append(f"Source: {_md(citation)}")
    return lines


def _entities(series: Sequence[Mapping[str, Any]]) -> str:
    return _named([_clip(one["label"]) for one in series])


def _named(names: Sequence[str]) -> str:
    """Up to `MAX_NAMED_ENTITIES` names, then a count of the rest."""
    shown = list(names[:MAX_NAMED_ENTITIES])
    if len(names) > MAX_NAMED_ENTITIES:
        shown.append(f"{len(names) - MAX_NAMED_ENTITIES:,} more")
    if len(shown) == 1:
        return shown[0]
    return f"{', '.join(shown[:-1])} and {shown[-1]}"


def _span(periods: Sequence[str]) -> str:
    if not periods:
        return "no periods"
    if periods[0] == periods[-1]:
        return periods[0]
    return f"{periods[0]} to {periods[-1]}"


def _bounded(text: str) -> str:
    """`text` whole when it fits `MAX_PROJECTION_BYTES`, else cut after its
    last whole line that fits and closed with `PROJECTION_CUT_NOTICE`."""
    if len(text.encode()) <= MAX_PROJECTION_BYTES:
        return text
    budget = MAX_PROJECTION_BYTES - len(f"\n\n{PROJECTION_CUT_NOTICE}".encode())
    head = text.encode()[:budget].decode(errors="ignore")
    head = head.rpartition("\n")[0] or head
    return f"{head.rstrip()}\n\n{PROJECTION_CUT_NOTICE}"


def _one_line(text: str) -> str:
    """`text` with every run of whitespace, line breaks included, as one space."""
    return " ".join(text.split())


def _clip(text: str, limit: int = MAX_CELL_CHARS) -> str:
    """`text` on one line, cut to `limit` characters with an ellipsis."""
    flat = _one_line(text)
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _md(text: str) -> str:
    """`text` with every Markdown-active character escaped."""
    return _MARKDOWN_ACTIVE.sub(r"\\\1", text)

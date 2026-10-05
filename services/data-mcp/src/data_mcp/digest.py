"""Renders the Search API's resolve-door response envelope into the model-facing digest.

The envelope carries one part per variable asked, in query order. Each part
is shaped by its own status (`definite`/`candidates`/`no_data`/`unsupported`),
rendered as its own block. A part's own warnings render inside its block, an
`unsupported` part limited to warnings about the caller's own input. Beneath
every block come the skills the envelope attaches to its specs, then the
envelope's diagnostics warnings as a shared trailer. `render_digest(envelope)`
is the public seam: the digest corpus imports it to exercise rendering with no
network and no tool. `render_answer(envelope)` is its data-shaped twin: the
identifiers, labels, statuses, skill ids and warnings the digest names, as the
small `structuredContent` a host may hand a model.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from common.skills import ordered_skill_ids

# Up to two nearest-real suggestions are ever rendered on a no_data part,
# matching the engine's own cap on how many it returns.
_NEAREST_REAL_CAP = 2

# Up to three elected statvars are ever named on a no_data part, or beside a
# spec_id. A longer list collapses to a trailing count so the digest stays
# bounded.
_ELECTED_STAT_VAR_CAP = 3

# Up to six candidates are ever rendered on a candidates part, matching the
# door's own ceiling on how many it returns. The full list stays in the
# result's `_meta` envelope; only the model-facing surface is bounded.
_CANDIDATE_CAP = 6

# Up to three facet-axis option values are ever named on a definite part's
# facet options line, matching the bounded-list style of the caps above. A
# longer list collapses to a trailing count.
_FACET_OPTIONS_CAP = 3

# Up to three facet axes ever get their own options line on a definite part,
# in wire order. A longer list collapses to a trailing count of the rest.
_FACET_OPTIONS_AXIS_CAP = 3

# Up to five skills are ever described beneath the part blocks. A longer list
# collapses to a trailing count.
_SKILL_CAP = 5

# A skill's description is written to be matched against, so it renders on
# one line, cut at this many characters.
_SKILL_DESCRIPTION_CAP = 160

# `provenance` is the facet axis name; the digest renders it as `source`,
# matching the tool's own field name for the same concept. Reused for a
# definite part's facets line, its facet options line, and for naming the
# field behind an unsupported refusal.
_AXIS_DISPLAY_NAMES: dict[str, str] = {"provenance": "source"}

# An `unsupported` part's own warnings render only these codes. Each
# describes something about the caller's own input (a phrase the lexicon
# didn't recognise, a facet it can't bind, a default it fell back to), so it
# bears on why this part refused. Every other warning on the same part
# belongs to an earlier step of the same request's pipeline and has no
# bearing on this part's own refusal. Every other status keeps every warning
# it carries.
_CALLER_INPUT_WARNING_CODES: frozenset[str] = frozenset(
    {
        "role_text_not_recognised",
        "time_text_not_parsed",
        "facet_binding_unsupported",
        "default_not_declared",
    }
)

_BASIS_DISPLAY: dict[str, str] = {
    "declared_default": "(default)",
    "bound": "(bound)",
    "fallback": "(fallback)",
}


def render_digest(envelope: Mapping[str, Any]) -> str:
    """Render the resolve door's envelope into the model-facing digest.

    One block per entry in `envelope["parts"]`, in query order, each shaped
    by that part's own status: `definite`, `candidates`, `no_data`, and
    `unsupported` each get their own rendering. A definite block also names
    each facet axis's other observed values, capped and counted, when the
    request asked for option coverage. Every spec_id the digest offers is
    followed by the StatVars Search elected for it, capped and counted, which
    a caller passes back as `get_data`'s `resolved_stat_var_dcids`. A part
    with an unrecognised status renders too, naming itself, rather than
    discarding it behind a refusal. A single-part envelope renders that one
    block bare; an envelope of two or more parts labels each block by its
    `variable_text`.
    `envelope["status"] == "mixed"` (the parts disagree) is stated as a
    leading line. Each part's own warnings render inside its block, though an
    `unsupported` part limits its warnings to those about the caller's own
    input. The skills the envelope's specs name render next, each with its
    id and description, most specific first and capped at `_SKILL_CAP`, so a
    model can load one with `get_skill`. The envelope's diagnostics warnings
    render as a shared trailer beneath everything. The digest surfaces
    summaries and metadata only; `resolved.observations` values never reach
    it, regardless of what an envelope happens to carry.
    """
    kind = _answer_kind(envelope)
    if kind != "resolution":
        trailer = [_warning_line(warning) for warning in _diagnostics_warnings(envelope)]
        return "\n\n".join([_render_answer_kind(envelope, kind), *trailer])
    parts = envelope.get("parts") or []
    if not parts:
        return "No parts were returned; the envelope could not be summarised."
    multi = len(parts) > 1
    blocks = [_render_part(index, part, multi=multi) for index, part in enumerate(parts)]
    sections = list(blocks)
    if envelope.get("status") == "mixed":
        sections.insert(0, "Mixed outcome: the parts below differ in status.")
    skills = _skills_section(envelope)
    if skills is not None:
        sections.append(skills)
    trailer = [_warning_line(warning) for warning in _diagnostics_warnings(envelope)]
    return "\n\n".join([*sections, *trailer]) if trailer else "\n\n".join(sections)


def render_answer(envelope: Mapping[str, Any]) -> dict[str, Any]:
    """The small model-facing answer, what `render_digest` names as data.

    `structuredContent` carries this and nothing else of the envelope, since
    a host may hand it to the model whole. Per part it holds the part's
    `label` (its `variable_text`), its `status`, the `specs` a model can pass
    to `get_data` (a definite part's spec, the candidates shown, or the
    nearest real specs a `no_data` part suggests, capped exactly as the digest
    caps them), and the part's own `warnings` when it has any. The envelope's
    `status`, the ids of the skills the digest lists (under its cap), and
    its diagnostics warnings sit beside `parts`. A spec is its
    `spec_id`, its rendered sentence as `label`, and, when the digest names
    any, the StatVar dcids it names as `resolved_stat_var_dcids`, the
    `get_data` argument they are passed back as. A `no_data` part's flagged
    match is left out, since the digest names it as a reason for the refusal
    and it is no spec to retrieve.
    """
    kind = _answer_kind(envelope)
    if kind != "resolution":
        answer: dict[str, Any] = {"kind": kind, "status": None, "parts": []}
        scope = _scope_place(envelope)
        if scope is not None:
            answer["scope"] = {"place": scope}
        if kind == "declined":
            answer["reason"] = envelope.get("reason")
        warnings = _diagnostics_warnings(envelope)
        if warnings:
            answer["warnings"] = [_answer_warning(warning) for warning in warnings]
        return answer
    answer = {
        "status": envelope.get("status"),
        "parts": [_answer_part(part) for part in envelope.get("parts") or []],
    }
    if "kind" in envelope:
        answer["kind"] = kind
    skill_ids = ordered_skill_ids(envelope)[:_SKILL_CAP]
    if skill_ids:
        answer["skills"] = skill_ids
    warnings = _diagnostics_warnings(envelope)
    if warnings:
        answer["warnings"] = [_answer_warning(warning) for warning in warnings]
    return answer


def _answer_part(part: Mapping[str, Any]) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "label": part.get("variable_text"),
        "status": part.get("status"),
        "specs": [_answer_spec(spec) for spec in _answer_specs(part)],
    }
    warnings = _part_warnings(part)
    if warnings:
        entry["warnings"] = [_answer_warning(warning) for warning in warnings]
    return entry


def _answer_specs(part: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """The specs the digest offers for `part`, in the digest's order and under its caps."""
    status = part.get("status")
    if status == "definite":
        spec = part.get("spec")
        return [spec] if spec else []
    if status == "candidates":
        return list((part.get("candidates") or {}).get("specs") or [])[:_CANDIDATE_CAP]
    if status == "no_data":
        return list((part.get("no_data") or {}).get("nearest_real") or [])[:_NEAREST_REAL_CAP]
    return []


def _answer_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "spec_id": spec.get("spec_id"),
        "label": (spec.get("resolved") or {}).get("rendered_sentence"),
    }
    dcids = [dcid for dcid, _ in _spec_stat_vars(spec)[:_ELECTED_STAT_VAR_CAP]]
    if dcids:
        entry["resolved_stat_var_dcids"] = dcids
    return entry


def _answer_warning(warning: Mapping[str, Any]) -> dict[str, Any]:
    entry = {"code": warning.get("code"), "message": warning.get("message", "")}
    if warning.get("spec_id") is not None:
        entry["spec_id"] = warning["spec_id"]
    return entry


def _render_part(index: int, part: Mapping[str, Any], *, multi: bool) -> str:
    status = part.get("status")
    renderer = _PART_RENDERERS.get(status)
    body = renderer(part) if renderer is not None else _render_unknown_part(status)
    if multi:
        body = f"{_part_label(index, part)}\n{body}"
    warnings = [_warning_line(warning) for warning in _part_warnings(part)]
    return "\n\n".join([body, *warnings]) if warnings else body


def _part_label(index: int, part: Mapping[str, Any]) -> str:
    text = part.get("variable_text")
    return f'Part {index + 1}: "{text}"' if text else f"Part {index + 1}:"


def _render_unknown_part(status: Any) -> str:
    return f"Unrecognised part status {status!r}; this part could not be summarised."


def _render_definite_part(part: Mapping[str, Any]) -> str:
    return _spec_block(part.get("spec") or {})


def _render_candidates_part(part: Mapping[str, Any]) -> str:
    candidates = part.get("candidates") or {}
    specs = candidates.get("specs") or []
    shown = specs[:_CANDIDATE_CAP]
    header = (
        f"{len(shown)} candidate interpretations — the chosen spec_id is the "
        "handle for retrieving the data."
    )
    blocks = [_spec_summary(index + 1, spec) for index, spec in enumerate(shown)]
    sections = [header, *blocks]
    if candidates.get("truncated") or len(specs) > len(shown):
        sections.append("More candidates exist than are shown here.")
    return "\n\n".join(sections)


def _render_no_data_part(part: Mapping[str, Any]) -> str:
    no_data = part.get("no_data") or {}
    reason = no_data.get("reason")
    if no_data.get("category") == "unresolved":
        header = f"The ask could not be resolved (reason: {reason})."
    else:
        header = f"Resolved, but no data is available (reason: {reason})."
    sections = [header]
    elected_stat_vars = no_data.get("elected_stat_vars") or []
    if elected_stat_vars:
        elected_line = _elected_stat_vars_line(elected_stat_vars)
        if elected_line is not None:
            sections.append(elected_line)
    nearest_real = (no_data.get("nearest_real") or [])[:_NEAREST_REAL_CAP]
    sections.extend(_nearest_real_line(spec) for spec in nearest_real)
    flagged_match = no_data.get("flagged_match")
    if flagged_match:
        gate = no_data.get("gate")
        sections.append(_flagged_match_line(flagged_match, reason=reason, gate=gate))
        if reason == "qualifier_not_bound":
            sections.extend(_unbound_qualifier_lines(no_data.get("unbound_qualifiers") or []))
    return "\n\n".join(sections)


def _render_unsupported_part(part: Mapping[str, Any]) -> str:
    unsupported = part.get("unsupported") or {}
    code = unsupported.get("code")
    detail = _normalise_detail(unsupported.get("detail") or "")
    field = _unsupported_field_name(unsupported.get("pointer") or "")
    if field is None:
        return f"Not supported: {detail}".strip()
    step = (
        f"Drop `{field}` and retry."
        if code == "facet_binding_unsupported"
        else f"Drop the field named `{field}` and retry."
    )
    return f"Not supported: the {field} field. {detail} {step}".strip()


def _normalise_detail(detail: str) -> str:
    """Turn an upstream `detail` fragment into its own sentence, capitalised and full-stopped.

    The upstream engine writes `detail` as a lower-case fragment meant to
    continue an existing sentence. The digest gives it its own sentence
    instead, so the first character is upper-cased and a trailing full stop
    is added when the fragment doesn't already end in one.
    """
    if not detail:
        return detail
    text = detail[0].upper() + detail[1:]
    return text if text.endswith((".", "!", "?")) else f"{text}."


def _unsupported_field_name(pointer: str) -> str | None:
    """The tool-facing field name behind an unsupported refusal's pointer, if any.

    The pointer into a refused facet contains a `facets/<axis>` segment,
    whichever section of the door's request shape it points into. The axis
    token is unescaped per RFC 6901 and mapped to the tool's own field name
    through `_AXIS_DISPLAY_NAMES`. A pointer with no `facets` segment, or
    whose axis token is empty (a trailing slash), names no field, so the raw
    pointer is never interpolated into the sentence.
    """
    if not pointer:
        return None
    tokens = [
        token.replace("~1", "/").replace("~0", "~") for token in pointer.lstrip("/").split("/")
    ]
    if "facets" not in tokens:
        return None
    axis_index = tokens.index("facets") + 1
    if axis_index >= len(tokens):
        return None
    axis = tokens[axis_index]
    if not axis:
        return None
    return _AXIS_DISPLAY_NAMES.get(axis, axis)


_PART_RENDERERS: dict[str, Callable[[Mapping[str, Any]], str]] = {
    "definite": _render_definite_part,
    "candidates": _render_candidates_part,
    "no_data": _render_no_data_part,
    "unsupported": _render_unsupported_part,
}


def _spec_lines(spec: Mapping[str, Any]) -> tuple[str, str]:
    resolved = spec.get("resolved") or {}
    spec_id = spec.get("spec_id", "")
    sentence = resolved.get("rendered_sentence") or ""
    confidence = resolved.get("confidence") or "unknown"
    coverage = _coverage_line(resolved.get("coverage") or {})
    return f'"{sentence}"  [{spec_id}]', f"confidence {confidence} · {coverage}"


def _spec_block(spec: Mapping[str, Any]) -> str:
    """A definite part's full block: sentence, spec_id, confidence, coverage, StatVars, entities, facets, facet options."""
    headline, confidence_line = _spec_lines(spec)
    lines = [headline, confidence_line]
    stat_vars = _stat_vars_line(spec)
    if stat_vars:
        lines.append(stat_vars)
    resolved = spec.get("resolved") or {}
    entities = resolved.get("entities") or []
    if entities:
        lines.append(_entities_line(entities))
    facets = resolved.get("facets") or {}
    if facets:
        lines.append(_facets_line(facets))
    lines.extend(_facet_options_lines(resolved.get("facet_options") or []))
    return "\n".join(lines)


def _spec_summary(number: int, spec: Mapping[str, Any]) -> str:
    """A candidate's numbered block: sentence, spec_id, confidence, coverage, StatVars. Omits grounding refs."""
    headline, confidence_line = _spec_lines(spec)
    lines = [f"{number}. {headline}", f"   {confidence_line}"]
    stat_vars = _stat_vars_line(spec)
    if stat_vars:
        lines.append(f"   {stat_vars}")
    return "\n".join(lines)


def _spec_stat_vars(spec: Mapping[str, Any]) -> list[tuple[str, str | None]]:
    """`(dcid, label)` for each StatVar Search elected for `spec`, from `resolved.stat_vars.items`.

    An item whose `ref` is not a non-empty string names no dcid a caller could
    pass back, so it is skipped.
    """
    items = ((spec.get("resolved") or {}).get("stat_vars") or {}).get("items") or []
    return [
        (item["ref"], item.get("label") if isinstance(item.get("label"), str) else None)
        for item in items
        if isinstance(item, Mapping) and isinstance(item.get("ref"), str) and item["ref"]
    ]


def _stat_vars_line(spec: Mapping[str, Any]) -> str | None:
    """One line naming the StatVars Search elected for `spec`, for `get_data`'s `resolved_stat_var_dcids`.

    Up to `_ELECTED_STAT_VAR_CAP` dcids, each with its label, then a trailing
    count for the rest. A shortened list is safe to pass back: dfo-apis checks
    each posted StatVar is a member of the family and serves the whole family
    either way. No line when Search elected none.
    """
    stat_vars = _spec_stat_vars(spec)
    if not stat_vars:
        return None
    shown = [
        f"{dcid} ({label})" if label else dcid for dcid, label in stat_vars[:_ELECTED_STAT_VAR_CAP]
    ]
    remaining = len(stat_vars) - len(shown)
    tail = f", and {remaining} more" if remaining > 0 else ""
    return f"StatVars for get_data's resolved_stat_var_dcids: {', '.join(shown)}{tail}."


def _entities_line(entities: Sequence[Mapping[str, Any]]) -> str:
    pieces = [
        f"{entity.get('role')} {entity.get('label') or entity.get('ref')} ({entity.get('ref')})"
        for entity in entities
    ]
    return " · ".join(pieces)


def _facets_line(facets: Mapping[str, Any]) -> str:
    return " · ".join(_facet_piece(axis, facet) for axis, facet in facets.items())


def _facet_piece(axis: str, facet: Mapping[str, Any]) -> str:
    display_axis = _AXIS_DISPLAY_NAMES.get(axis, axis)
    if facet.get("selection") == "per_series":
        return f"{display_axis} varies per series"
    label = facet.get("label") or facet.get("ref") or ""
    piece = f"{display_axis} {label}".strip()
    basis = _BASIS_DISPLAY.get(facet.get("basis", ""), "")
    return f"{piece} {basis}".strip() if basis else piece


def _facet_options_lines(facet_options: Sequence[Mapping[str, Any]]) -> list[str]:
    """Up to `_FACET_OPTIONS_AXIS_CAP` axis lines from a definite spec's `facet_options`, in wire order.

    Populated only when the search request turned on option coverage. An axis
    with no observed value is already omitted upstream, so every entry here
    names at least one value. More axes than the cap collapse to one trailing
    line naming how many were left out.
    """
    lines = [_facet_axis_line(axis_options) for axis_options in facet_options]
    lines = [line for line in lines if line is not None]
    shown = lines[:_FACET_OPTIONS_AXIS_CAP]
    remaining = len(lines) - len(shown)
    if remaining > 0:
        shown.append(f"and {remaining} more axes with options.")
    return shown


def _facet_axis_line(axis_options: Mapping[str, Any]) -> str | None:
    """One line naming an axis's other observed values, by the ref `get_data`'s own `facets` scope takes.

    Named by the axis's raw wire name (`provenance`, not `source`), since
    that is the key `get_data`'s `facets` scope is keyed by. A display name
    here would name a field the caller cannot pass. Each value renders as its
    dcid, the handle `get_data` accepts, with its label alongside for
    readability.
    """
    axis = axis_options.get("axis")
    values_all = [_facet_option_label(option) for option in axis_options.get("options") or []]
    values_all = [value for value in values_all if value]
    if not values_all:
        return None
    shown = values_all[:_FACET_OPTIONS_CAP]
    remaining = len(values_all) - len(shown)
    if remaining > 0:
        tail = f", and {remaining} more"
    elif axis_options.get("truncated"):
        # The wire truncated its own list before this cap ever applied, so
        # there is more beyond what's shown even though the local remainder
        # is zero.
        tail = ", and more"
    else:
        tail = ""
    return f"{axis} options for get_data: {', '.join(shown)}{tail}."


def _facet_option_label(option: Mapping[str, Any]) -> str | None:
    ref = (option.get("value") or {}).get("ref") or {}
    dcid = ref.get("dcid")
    label = ref.get("label")
    if dcid and label:
        return f"{dcid} ({label})"
    return dcid or label


def _coverage_line(coverage: Mapping[str, Any]) -> str:
    kind = coverage.get("kind")
    data_word = "has data" if coverage.get("has_data") else "no data"
    if kind == "exact":
        return _coverage_exact_line(coverage, data_word)
    if kind == "breadth":
        return _coverage_breadth_line(coverage, data_word)
    return data_word


def _coverage_exact_line(coverage: Mapping[str, Any], data_word: str) -> str:
    count = coverage.get("observation_count")
    window = _window_text(coverage.get("window"))
    lead = f"{count} observations" if count is not None else "observations"
    prefix = f"{lead}, {window}" if window else lead
    return f"{prefix} · {data_word}"


def _coverage_breadth_line(coverage: Mapping[str, Any], data_word: str) -> str:
    dims = coverage.get("dimensions") or []
    dims_text = ", ".join(f"{dim.get('count')} {dim.get('label')}" for dim in dims)
    return f"{dims_text} · {data_word}" if dims_text else data_word


def _window_text(window: Mapping[str, Any] | None) -> str:
    if not window:
        return ""
    start, end = window.get("start_year"), window.get("end_year")
    if start is not None and end is not None:
        return f"{start}-{end}"
    if start is not None:
        return f"since {start}"
    if end is not None:
        return f"through {end}"
    return ""


def _elected_stat_vars_line(elected_stat_vars: Sequence[Mapping[str, Any]]) -> str | None:
    """One line naming the statvar family a no_data check confirmed and found empty.

    Up to three names, then a trailing count for the rest, matching the cap on
    every other bounded list in this module. An entry with neither `label`
    nor `dcid` is skipped, since it names nothing. If none survive, no line
    is rendered.
    """
    names_all = [ref.get("label") or ref.get("dcid") for ref in elected_stat_vars]
    names_all = [name for name in names_all if name]
    if not names_all:
        return None
    shown = names_all[:_ELECTED_STAT_VAR_CAP]
    names = ", ".join(shown)
    remaining = len(names_all) - len(shown)
    tail = f", and {remaining} more" if remaining > 0 else ""
    return f"Checked: {names}{tail}."


def _nearest_real_line(spec: Mapping[str, Any]) -> str:
    resolved = spec.get("resolved") or {}
    spec_id = spec.get("spec_id", "")
    sentence = resolved.get("rendered_sentence") or ""
    line = f'Nearby: "{sentence}"  [{spec_id}]'
    relaxation = spec.get("relaxation") or {}
    axis = relaxation.get("axis")
    line = f"{line} — relaxed {axis}" if axis else line
    stat_vars = _stat_vars_line(spec)
    return f"{line}\n{stat_vars}" if stat_vars else line


def _flagged_match_line(spec: Mapping[str, Any], *, reason: Any, gate: Any) -> str:
    """One line naming the elected variable a no_data outcome flagged as its nearest match.

    `gate` and `reason` are read from the outer `no_data` object, never from
    `spec` itself: `flagged_match` is a plain Spec with no `gate` key of its
    own. `role_incompatible` and `qualifier_not_bound` each carry their own
    wording and no gate (the refusal fires before an availability verdict);
    every other reason that flags a match fires because a quality gate
    rejected it, so the gate sentence only renders when `gate` is set.
    """
    spec_id = spec.get("spec_id", "")
    prefix = f"A same-variable match exists [{spec_id}]"
    if reason == "role_incompatible":
        return f"{prefix} but its declared roles cannot support this request."
    if reason == "qualifier_not_bound":
        return f"{prefix} but does not satisfy every requested qualifier."
    if gate:
        return f"{prefix} but failed quality gate {gate}."
    return f"{prefix}."


def _unbound_qualifier_lines(unbound_qualifiers: Sequence[Mapping[str, Any]]) -> list[str]:
    """Explain each unapplied qualifier using the disposition from the wire.

    Each entry's `text` is the verbatim span that named the term; an entry
    with no `text` is skipped. Keep query order when dispositions differ.
    """
    lines = []
    for qualifier in unbound_qualifiers:
        text = qualifier.get("text")
        if not text:
            continue
        if qualifier.get("disposition") == "no_vocabulary":
            lines.append(f'"{text}" is not a distinction this data records.')
        else:
            lines.append(f'"{text}" names a category the search could not apply.')
    return lines


def _skills_section(envelope: Mapping[str, Any]) -> str | None:
    """The skills the envelope attaches to its specs: one line each, up to `_SKILL_CAP`.

    Ordered by `common.skills.ordered_skill_ids`, so the most specific skill
    of the first result leads. A `None` or empty map renders nothing.
    """
    skills = envelope.get("skills") or {}
    if not skills:
        return None
    ordered = ordered_skill_ids(envelope)
    shown = ordered[:_SKILL_CAP]
    lines = ["Skills for these results; get_skill loads one by id:"]
    lines.extend(_skill_line(skill_id, skills[skill_id]) for skill_id in shown)
    remaining = len(ordered) - len(shown)
    if remaining > 0:
        lines.append(f"and {remaining} more skill{'s' if remaining > 1 else ''}.")
    return "\n".join(lines)


def _skill_line(skill_id: str, skill: Mapping[str, Any]) -> str:
    """`- id: description`. A skill's `name` equals its id, so the id alone names it."""
    description = " ".join(str(skill.get("description") or "").split())
    if len(description) > _SKILL_DESCRIPTION_CAP:
        description = description[: _SKILL_DESCRIPTION_CAP - 1].rstrip() + "…"
    return f"- {skill_id}: {description}" if description else f"- {skill_id}"


def _answer_kind(envelope: Mapping[str, Any]) -> str:
    """The resolve answer's `kind`. A body with no `kind` predates the
    discriminator and is a resolution: the parts, as every body was."""
    return str(envelope.get("kind") or "resolution")


def _scope_place(envelope: Mapping[str, Any]) -> dict[str, Any] | None:
    """A catalogue or declined answer's confirmed country, `{dcid, label}`, or `None`."""
    place = (envelope.get("scope") or {}).get("place")
    if not isinstance(place, Mapping) or not place.get("dcid"):
        return None
    return {"dcid": place["dcid"], "label": place.get("label") or place["dcid"]}


def _render_answer_kind(envelope: Mapping[str, Any], kind: str) -> str:
    """What a `catalogue` or `declined` answer means, stated as search's own
    reading of the question: it names no part to retrieve, so a model
    chooses what to do with that. An unknown kind names itself."""
    place = _scope_place(envelope)
    named = f" It named {place['label']}." if place is not None else ""
    if kind == "catalogue":
        return (
            "Search read this question as asking what data DFO holds, not as naming data "
            f"to retrieve, so it resolved no parts.{named} The `sources` tool lists what "
            "DFO holds and documents each dataset."
        )
    if kind == "declined":
        reason = envelope.get("reason")
        if reason == "out_of_scope":
            why = "it is not about data DFO holds"
        elif reason == "requires_analysis":
            why = "it asks for a judgement and names no data to show"
        else:
            why = f"reason {reason!r}"
        return f"Search declined this question: {why}; it resolved no parts.{named}"
    return f"Search answered with kind {kind!r}, which this service does not know; it carries no parts."


def _diagnostics_warnings(envelope: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    diagnostics = envelope.get("diagnostics") or {}
    return list(diagnostics.get("warnings") or [])


def _part_warnings(part: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    warnings = list(part.get("warnings") or [])
    if part.get("status") == "unsupported":
        # A caller-input warning explains the refusal beside it; anything
        # else is a note from an earlier step of the same request's
        # pipeline and does not belong to this part's own refusal.
        warnings = [
            warning for warning in warnings if warning.get("code") in _CALLER_INPUT_WARNING_CODES
        ]
    return warnings


def _warning_line(warning: Mapping[str, Any]) -> str:
    message = warning.get("message", "")
    spec_id = warning.get("spec_id")
    return f"[{spec_id}] {message}" if spec_id is not None else message

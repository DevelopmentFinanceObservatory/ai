"""Renders the Search API's resolve-door response envelope into the model-facing digest.

The envelope carries one part per variable asked, in query order. Each part
is shaped by its own status (`definite`/`candidates`/`no_data`/`unsupported`),
rendered as its own block. A part's own warnings render inside its block, a
refused part (`unsupported`, or `no_data` that never resolved the ask) limited
to warnings about the caller's own input. Beneath
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

# The own warnings of a part that refused (`unsupported`, or `no_data` in the
# `unresolved` category) render only these codes. Each describes something
# about the caller's own input (a phrase the lexicon didn't recognise, a facet
# it can't bind, a default it fell back to), so it bears on why this part
# refused. Every other warning on the same part belongs to an earlier step of
# the same request's pipeline, or describes an answer the part never reached
# (a defaulted time window), and has no bearing on the refusal. Every other
# status keeps every warning it carries.
_CALLER_INPUT_WARNING_CODES: frozenset[str] = frozenset(
    {
        "role_text_not_recognised",
        "time_text_not_parsed",
        "facet_binding_unsupported",
        "default_not_declared",
    }
)

# A sibling measure's `axis` and the words that introduce it on a definite
# block's measure choice line, in the order the line names them.
_CHOICE_AXIS_LEADS: dict[str, str] = {"what": "other counts", "unit": "other prices"}

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
    followed by what it measures (`_measure_line`), which names no member
    StatVar. A part with an unrecognised status renders too, naming itself,
    rather than discarding it behind a refusal. A single-part envelope renders that one
    block bare; an envelope of two or more parts labels each block by its
    `variable_text`. A definite block also names the measure Search elected
    and each sibling measure's own spec_id (`_measure_choice_line`).
    A mixed envelope leads with Search's "Answered 2 of 3 parts.", the text of
    its `sentence_plan` tokens of segment `coverage` (`_answered_line`); an
    envelope with none has no leading line. A part whose ask named
    sources ends its block with one line saying which it asked for and
    which it excluded (`_sources_line`). Each
    part's own warnings render inside its block, though a refused part
    (`_is_refusal`) limits its warnings to those about the caller's own
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
    if answered := _answered_line(envelope):
        sections.insert(0, answered)
    skills = _skills_section(envelope)
    if skills is not None:
        sections.append(skills)
    trailer = [_warning_line(warning) for warning in _diagnostics_warnings(envelope)]
    return "\n\n".join([*sections, *trailer]) if trailer else "\n\n".join(sections)


def _answered_line(envelope: Mapping[str, Any]) -> str | None:
    """The text of the envelope sentence's `coverage` tokens, the "Answered N of M parts."
    the Search API opens a mixed resolution with, or None when it sends none."""
    plan = envelope.get("sentence_plan") or []
    text = "".join(token.get("text", "") for token in plan if token.get("segment") == "coverage")
    return text.strip() or None


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
    `spec_id`, its rendered sentence as `label`, and the facts the digest's
    measure line states (`_measure`); a definite part's spec adds the
    elected measure and its siblings (`_measure_choice`). A `no_data` part's flagged
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
        "specs": [
            _answer_spec(spec, definite=part.get("status") == "definite")
            for spec in _answer_specs(part)
        ],
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


def _answer_spec(spec: Mapping[str, Any], *, definite: bool) -> dict[str, Any]:
    entry = {
        "spec_id": spec.get("spec_id"),
        "label": (spec.get("resolved") or {}).get("rendered_sentence"),
        **_measure(spec),
    }
    return {**entry, **_measure_choice(spec)} if definite else entry


def _answer_warning(warning: Mapping[str, Any]) -> dict[str, Any]:
    entry = {"code": warning.get("code"), "message": warning.get("message", "")}
    if warning.get("spec_id") is not None:
        entry["spec_id"] = warning["spec_id"]
    return entry


def _render_part(index: int, part: Mapping[str, Any], *, multi: bool) -> str:
    status = part.get("status")
    renderer = _PART_RENDERERS.get(status)
    body = renderer(part) if renderer is not None else _render_unknown_part(status)
    if sources := _sources_line(part):
        body = f"{body}\n{sources}"
    if multi:
        body = f"{_part_label(index, part)}\n{body}"
    warnings = [_warning_line(warning) for warning in _part_warnings(part)]
    return "\n\n".join([body, *warnings]) if warnings else body


def _sources_line(part: Mapping[str, Any]) -> str | None:
    """One line saying which sources the part's ask named or excluded, as Search read them.

    Read from the part's `restatement.sources`, so only sources the caller
    asked about appear. Each reads as the caller's wording, whether it was
    asked for or excluded, and what Search resolved it to: the ref, with the
    label when the part's spec names one. A wording Search could not resolve
    carries no ref, and renders as the wording alone. For example `Sources:
    asked for "IDS", read as World Bank International Debt Statistics (IDS)
    (provenance/WorldBankInternationalDebtStatistics) · excluded "CRS", read
    as provenance/OECDCreditorReportingSystem.` No line when the part named
    no source.
    """
    sources = (part.get("restatement") or {}).get("sources") or []
    if not sources:
        return None
    labels = _spec_ref_labels(part.get("spec") or {})
    pieces = [_source_piece(source, labels) for source in sources]
    return f"Sources: {' · '.join(pieces)}."


def _source_piece(source: Mapping[str, Any], labels: Mapping[str, str]) -> str:
    verb = "excluded" if source.get("polarity") == "exclude" else "asked for"
    piece = f'{verb} "{source.get("text")}"'
    ref = source.get("ref")
    if not ref:
        return piece
    label = labels.get(ref)
    return f"{piece}, read as {label} ({ref})" if label else f"{piece}, read as {ref}"


def _spec_ref_labels(spec: Mapping[str, Any]) -> dict[str, str]:
    """The label of each ref the spec's facets and facet options name."""
    resolved = spec.get("resolved") or {}
    labels: dict[str, str] = {}
    for facet in (resolved.get("facets") or {}).values():
        if facet.get("ref") and facet.get("label"):
            labels[facet["ref"]] = facet["label"]
    for axis_options in resolved.get("facet_options") or []:
        for option in axis_options.get("options") or []:
            ref = (option.get("value") or {}).get("ref") or {}
            if ref.get("ref") and ref.get("label"):
                labels[ref["ref"]] = ref["label"]
    return labels


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
    unbound_qualifiers = no_data.get("unbound_qualifiers") or []
    if no_data.get("category") == "unresolved":
        header = f"The ask could not be resolved (reason: {reason})."
    elif reason == "qualifier_not_bound" and no_data.get("flagged_match"):
        header = _unapplied_phrase_header(unbound_qualifiers)
    else:
        header = f"Resolved, but no data is available (reason: {reason})."
    sections = [header]
    if message := no_data.get("message"):
        sections.append(message)
    sections.extend(_ambiguous_source_lines(no_data.get("ambiguous_sources") or []))
    checked = _checked_count(no_data)
    if checked:
        sections.append(f"Checked {checked:,} StatVar{'' if checked == 1 else 's'}.")
    nearest_real = (no_data.get("nearest_real") or [])[:_NEAREST_REAL_CAP]
    sections.extend(_nearest_real_line(spec) for spec in nearest_real)
    flagged_match = no_data.get("flagged_match")
    if flagged_match:
        gate = no_data.get("gate")
        sections.append(_flagged_match_line(flagged_match, reason=reason, gate=gate))
        if reason == "qualifier_not_bound":
            sections.extend(_unbound_qualifier_lines(unbound_qualifiers))
    return "\n\n".join(sections)


def _unapplied_phrase_header(unbound_qualifiers: Sequence[Mapping[str, Any]]) -> str:
    """The heading of a `qualifier_not_bound` part that flagged a same-variable match.

    The data exists without the phrase, so the heading states a refusal to
    apply it: the quoted phrases in query order without repeats, or "every
    phrase" when no entry carries text.
    """
    texts = list(dict.fromkeys(str(q["text"]) for q in unbound_qualifiers if q.get("text")))
    quoted = [f'"{text}"' for text in texts]
    if not quoted:
        named = "every phrase of this ask"
    elif len(quoted) == 1:
        named = quoted[0]
    elif len(quoted) == 2:
        named = f"{quoted[0]} and {quoted[1]}"
    else:
        named = f"{', '.join(quoted[:-1])}, and {quoted[-1]}"
    return (
        f"Search found matching data but could not apply {named} to it, so this part has "
        "no answer (reason: qualifier_not_bound)."
    )


def _ambiguous_source_lines(groups: Sequence[Mapping[str, Any]]) -> list[str]:
    """One line per ambiguous source wording, naming each source it could mean.

    Each candidate reads as its label, then its ref and kind (`provenance`,
    `dataset` or `source`), so a model can ask the user which one is meant.
    Candidates are capped at `_CANDIDATE_CAP`, with a trailing count of the rest.
    """
    lines = []
    for group in groups:
        candidates = group.get("candidates") or []
        shown = [_source_candidate(candidate) for candidate in candidates[:_CANDIDATE_CAP]]
        remaining = len(candidates) - len(shown)
        tail = f", and {remaining} more" if remaining > 0 else ""
        lines.append(f'"{group.get("text")}" could mean: {"; ".join(shown)}{tail}.')
    return lines


def _source_candidate(candidate: Mapping[str, Any]) -> str:
    return f"{candidate.get('label')} ({candidate.get('ref')}, {candidate.get('kind')})"


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
    """A definite part's full block: sentence, spec_id, confidence, coverage, measure, measure choice, entities, facets, facet options."""
    headline, confidence_line = _spec_lines(spec)
    lines = [headline, confidence_line]
    measure = _measure_line(spec)
    if measure:
        lines.append(measure)
    choice = _measure_choice_line(spec)
    if choice:
        lines.append(choice)
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
    """A candidate's numbered block: sentence, spec_id, confidence, coverage, measure. Omits grounding refs."""
    headline, confidence_line = _spec_lines(spec)
    lines = [f"{number}. {headline}", f"   {confidence_line}"]
    measure = _measure_line(spec)
    if measure:
        lines.append(f"   {measure}")
    return "\n".join(lines)


def _measure(spec: Mapping[str, Any]) -> dict[str, Any]:
    """What `spec` measures, as data: Search's `measure` label, the family's `member_count`, whether that count is exact, and the axes its members vary along.

    The count is `resolved.stat_vars.member_count`, exact unless
    `member_count_exact` is false. A count of zero is left out. An axis is a
    slot with a constraint property that `inapplicable_slots` does not list,
    unbound or bound to a set or an expanded value, named by its label. No
    member StatVar is named. A fact Search does not state is left out.
    """
    resolved = spec.get("resolved") or {}
    stat_vars = resolved.get("stat_vars") or {}
    facts: dict[str, Any] = {}
    label = (resolved.get("shape") or {}).get("label")
    if isinstance(label, str) and label:
        facts["measure"] = label
    count = stat_vars.get("member_count")
    if isinstance(count, int) and not isinstance(count, bool) and count > 0:
        facts["member_count"] = count
        facts["member_count_exact"] = stat_vars.get("member_count_exact") is not False
    axes = _varying_axes(resolved)
    if axes:
        facts["varies_by"] = axes
    if codes := source_codes(spec):
        facts["source_codes"] = codes
    return facts


def source_codes(spec: Mapping[str, Any]) -> list[dict[str, str]]:
    """The publisher codes of the family `spec` answers, as `{"code", "source"}` entries in wire order.

    The codes are `resolved.stat_vars.source_codes`, each a code and the
    provenance dcid that issues it. `source` is the label of the provenance
    the spec binds (`resolved.facets.provenance`) when that is the code's
    provenance, and is left out otherwise, so the code stands alone. Empty
    when Search states no codes.
    """
    resolved = spec.get("resolved") or {}
    bound = (resolved.get("facets") or {}).get("provenance") or {}
    ref, label = bound.get("ref"), bound.get("label")
    codes: list[dict[str, str]] = []
    for entry in (resolved.get("stat_vars") or {}).get("source_codes") or []:
        code = entry.get("code") if isinstance(entry, Mapping) else None
        if not isinstance(code, str) or not code:
            continue
        named = isinstance(ref, str) and entry.get("provenance") == ref and label
        codes.append({"code": code, "source": label} if named else {"code": code})
    return codes


def _varying_axes(resolved: Mapping[str, Any]) -> list[str]:
    """The label of each slot `resolved` leaves open along a constraint property, in slot order."""
    inapplicable = set(resolved.get("inapplicable_slots") or [])
    axes: list[str] = []
    for slot in resolved.get("slots") or []:
        if not isinstance(slot, Mapping):
            continue
        key = slot.get("key") or {}
        prop = (key.get("property") or {}).get("ref")
        if not isinstance(prop, str) or prop in inapplicable:
            continue
        binding = slot.get("binding") or {}
        if binding.get("kind") == "value" and not (slot.get("expansion") or {}).get("added"):
            continue
        axes.append(key.get("label") or (key.get("property") or {}).get("label") or prop)
    return axes


def _measure_line(spec: Mapping[str, Any]) -> str | None:
    """One line saying what `spec` measures (`_measure`), naming no member StatVar.

    For example "Measure: gross disbursements · 3,687 StatVars varying by
    sector, instrument." A count Search states as a floor reads "at least".
    The family's publisher codes close the line, the source's label once
    before its codes, as "Source codes: World Bank IDS DT.TDS.DEGG.CD,
    DT.TDS.DEGG.KD". No line when Search states none of these.
    """
    facts = _measure(spec)
    pieces: list[str] = []
    if "measure" in facts:
        pieces.append(f"Measure: {facts['measure']}")
    if "member_count" in facts:
        count = facts["member_count"]
        floor = "" if facts["member_count_exact"] else "at least "
        noun = "StatVar" if count == 1 and not floor else "StatVars"
        counted = f"{floor}{count:,} {noun}"
        if "varies_by" in facts:
            counted += f" varying by {', '.join(facts['varies_by'])}"
        pieces.append(counted)
    if "source_codes" in facts:
        by_source: dict[str, list[str]] = {}
        for code in facts["source_codes"]:
            by_source.setdefault(code.get("source", ""), []).append(code["code"])
        named = (f"{source} {', '.join(codes)}".strip() for source, codes in by_source.items())
        pieces.append(f"Source codes: {', '.join(named)}")
    if not pieces:
        return None
    return f"{' · '.join(pieces)}."


def _measure_choice(spec: Mapping[str, Any]) -> dict[str, Any]:
    """The measure Search elected for `spec` and the sibling measures it could answer with, as data.

    `counted_as` is `resolved.measure_label`. `alternatives` holds each
    `resolved.alternative_measures` entry as `{spec_id, label, axis}`, in wire
    order: axis "what" counts the flow another way ("net disbursements"),
    axis "unit" switches its prices ("constant prices"). An entry missing its
    spec_id or label, or on another axis, is left out. Empty when no
    alternative remains, since the measure line already says what `spec`
    measures.
    """
    resolved = spec.get("resolved") or {}
    label = resolved.get("measure_label")
    alternatives: list[dict[str, Any]] = []
    for alternative in resolved.get("alternative_measures") or []:
        spec_id = (alternative.get("definition") or {}).get("spec_id")
        alternative_label = alternative.get("label")
        axis = alternative.get("axis")
        if spec_id and alternative_label and axis in _CHOICE_AXIS_LEADS:
            alternatives.append({"spec_id": spec_id, "label": alternative_label, "axis": axis})
    if not isinstance(label, str) or not label or not alternatives:
        return {}
    return {"counted_as": label, "alternatives": alternatives}


def _measure_choice_line(spec: Mapping[str, Any]) -> str | None:
    """One line naming the elected measure, then each sibling by its own spec_id (`_measure_choice`).

    For example "Counted as gross disbursements; other counts: net
    disbursements [spec_b]; other prices: constant prices [spec_c]." No line
    when Search names no alternative.
    """
    choice = _measure_choice(spec)
    if not choice:
        return None
    pieces = [f"Counted as {choice['counted_as']}"]
    for axis, lead in _CHOICE_AXIS_LEADS.items():
        named = [
            f"{alternative['label']} [{alternative['spec_id']}]"
            for alternative in choice["alternatives"]
            if alternative["axis"] == axis
        ]
        if named:
            pieces.append(f"{lead}: {', '.join(named)}")
    return f"{'; '.join(pieces)}."


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
    dcid = ref.get("ref")
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


def _checked_count(no_data: Mapping[str, Any]) -> int:
    """How many StatVars a no_data check confirmed and found empty (`elected_count`), naming none."""
    count = no_data.get("elected_count")
    return count if isinstance(count, int) and not isinstance(count, bool) else 0


def _nearest_real_line(spec: Mapping[str, Any]) -> str:
    resolved = spec.get("resolved") or {}
    spec_id = spec.get("spec_id", "")
    sentence = resolved.get("rendered_sentence") or ""
    line = f'Nearby: "{sentence}"  [{spec_id}]'
    relaxation = spec.get("relaxation") or {}
    axis = relaxation.get("axis")
    line = f"{line} — relaxed {axis}" if axis else line
    measure = _measure_line(spec)
    return f"{line}\n{measure}" if measure else line


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
        line = f"{prefix} but does not satisfy every requested qualifier."
        sentence = (spec.get("resolved") or {}).get("rendered_sentence")
        return f'{line} It shows: "{sentence}"' if sentence else line
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
    """A catalogue or declined answer's confirmed country, `{ref, label}`, or `None`."""
    place = (envelope.get("scope") or {}).get("place")
    if not isinstance(place, Mapping) or not place.get("ref"):
        return None
    return {"ref": place["ref"], "label": place.get("label") or place["ref"]}


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
    if _is_refusal(part):
        # A caller-input warning explains the refusal beside it; anything
        # else is a note from an earlier step of the same request's
        # pipeline and does not belong to this part's own refusal.
        warnings = [
            warning for warning in warnings if warning.get("code") in _CALLER_INPUT_WARNING_CODES
        ]
    return warnings


def _is_refusal(part: Mapping[str, Any]) -> bool:
    """Whether `part` refused the ask: `unsupported`, or `no_data` that never resolved it."""
    status = part.get("status")
    if status == "unsupported":
        return True
    return status == "no_data" and (part.get("no_data") or {}).get("category") == "unresolved"


def _warning_line(warning: Mapping[str, Any]) -> str:
    message = warning.get("message", "")
    spec_id = warning.get("spec_id")
    return f"[{spec_id}] {message}" if spec_id is not None else message

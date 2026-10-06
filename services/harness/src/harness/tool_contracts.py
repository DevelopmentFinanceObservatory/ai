"""What the resolver and the MCP clients agree on.

The resolver's refusal type, `graph_envelope` (the digest of a search
envelope that becomes the graph's decision inputs: status / spec_id /
candidates / choice_points / parts / warnings / wide / skills), the
presentation tool names `McpExecutor` routes on, and the
refusal helpers: the `origin` values an executor stamps, `harness_refusal` for a refusal the harness mints,
and `platform_trace_id_of`. The full envelope, with its dozens of unbound
definition slots, never reaches graph state.

`harness.graph`, `harness.mcp_client` and `harness.scripted` import this module.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from common.skills import ordered_skill_ids
from compiler.sources import DESCRIBED_SHAPE_KIND, open_entity_axes_of

logger = logging.getLogger(__name__)

_PRESENTATION_TOOLS = frozenset({"describe_frame", "write_component", "describe_component"})

# Who minted a refusal an executor returns: an MCP service, or the harness
# itself (a decision it took after the service answered, or a routing defect).
# A refusal that states neither is treated as unattributed.
REFUSAL_ORIGIN_SERVICE = "service"
REFUSAL_ORIGIN_HARNESS = "harness"

# The platform states warning severities as info/warning/degraded; the wire
# vocabulary is info/warn/degraded. Anything unrecognised maps to "warn" --
# an unknown severity reads as a caution.
_SEVERITY_TO_WIRE = {"info": "info", "warn": "warn", "warning": "warn", "degraded": "degraded"}


class ResolverError(RuntimeError):
    """A service-level search failure (not a resolve outcome): the resolver
    has no refusal channel, so this surfaces as the turn's stream-fatal
    error. `refusal` carries the service's structured refusal when there is
    one, so the stream error can state the real code (e.g.
    `capability_unavailable` on an identity-reproduction failure)."""

    def __init__(self, message: str, *, refusal: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.refusal = refusal or {}


def harness_refusal(*, code: str, message: str) -> dict[str, Any]:
    """The refusal grammar's body for a refusal the harness mints itself: stamped
    `origin: harness`, never retried, with no fix."""
    return {
        "code": code,
        "retry_mode": "do_not_retry",
        "fix": None,
        "origin": REFUSAL_ORIGIN_HARNESS,
        "message": message,
    }


def platform_trace_id_of(refusal: Mapping[str, Any]) -> str | None:
    """The platform's own trace id on a refusal, when it is a non-empty string."""
    platform_trace_id = refusal.get("platform_trace_id")
    return platform_trace_id if isinstance(platform_trace_id, str) and platform_trace_id else None


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _wire_warnings(warnings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "code": str(warning.get("code", "unspecified")),
            "severity": _SEVERITY_TO_WIRE.get(str(warning.get("severity", "")), "warn"),
            "message": str(warning.get("message", "")),
        }
        for warning in warnings
    ]


def _search_warnings(envelope: dict[str, Any], part: dict[str, Any]) -> list[dict[str, Any]]:
    """The request-level warnings, then the part's own."""
    shared = (envelope.get("diagnostics") or {}).get("warnings") or []
    return _wire_warnings(list(shared) + list(part.get("warnings") or []))


def _spec_candidates(specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One entry per offered spec: its id, its sentence and its width facts (`_wide_facts`)."""
    return [
        {
            "spec_id": spec.get("spec_id"),
            "label": (spec.get("resolved") or {}).get("rendered_sentence") or spec.get("spec_id"),
            **_wide_facts(spec),
        }
        for spec in specs
    ]


def _identity(spec: dict[str, Any]) -> Mapping[str, Any]:
    """`spec.definition.identity` -- the identity the resolution actually
    carries, never `spec.resolved.shape`, a different, unrelated object of
    the same name."""
    return (spec.get("definition") or {}).get("identity") or {}


def _identity_shape_kind(spec: dict[str, Any]) -> str | None:
    return (_identity(spec).get("shape") or {}).get("kind")


def _entity_roles(spec: dict[str, Any]) -> list[str]:
    """The identity's own bound entity roles, sorted (`observationAbout`,
    or a role-bearing shape's `origin`/`destination`)."""
    return sorted((_identity(spec).get("entities") or {}).keys())


def _year(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coverage_window(resolved: dict[str, Any]) -> tuple[int, int] | None:
    """The first and last years the resolver found data across.

    `resolved.download.window`, the first and last years of the points a
    mint of the definition writes, when it states both. Otherwise
    `resolved.stat_vars.data_date_range`, the first and last dates of the
    points the answer serves across the family's served members. `None`
    when neither states a usable range."""
    download = resolved.get("download")
    window = download.get("window") if isinstance(download, Mapping) else None
    if isinstance(window, Mapping):
        start = _year(window.get("start_year"))
        end = _year(window.get("end_year"))
        if start is not None and end is not None:
            return start, end
    stat_vars = resolved.get("stat_vars")
    date_range = stat_vars.get("data_date_range") if isinstance(stat_vars, Mapping) else None
    if not isinstance(date_range, Mapping):
        return None
    start = _year(date_range.get("start"))
    end = _year(date_range.get("end"))
    return (start, end) if start is not None and end is not None else None


def requested_range(identity: Mapping[str, Any]) -> tuple[int, int] | None:
    """The years an identity's `requested_window` names outright: both ends
    of a `range` kind, as whole years. `None` for any other kind, and for a
    range naming one end or neither."""
    requested = identity.get("requested_window") or {}
    if not isinstance(requested, Mapping) or requested.get("kind") != "range":
        return None
    start = _year(requested.get("start_year"))
    end = _year(requested.get("end_year"))
    return (start, end) if start is not None and end is not None else None


def _window(spec: dict[str, Any]) -> dict[str, int | str] | None:
    """The dataset requirement's settled time window, as whole years, and
    which side settled it.

    `spec.definition.identity.requested_window` (the same field
    `compiler.evidence._scope_window_issues` reads) states a complete
    range only under its `range` kind, at `start_year`/`end_year`; that
    pair (`requested_range`) wins when both parse, `source: "requested"`.
    A `range` naming only one end ("since 1990"), or an `all` kind with no
    time text at all, falls back to the years the resolver found data
    across (`_coverage_window`): a bare open end is completed from it
    (`{"start": 1990, "end": <the coverage's latest year>}`), and a
    fully open request takes the whole span, `source: "coverage"`.
    `latest` names a single most-recent observation rather than a range
    either side could state, so it is left out. Returns `None` when
    neither side states a usable range, leaving `window` out of the
    envelope and the period a decision for the caller."""
    identity = _identity(spec)
    if (asked := requested_range(identity)) is not None:
        return {"start": asked[0], "end": asked[1], "source": "requested"}
    requested = identity.get("requested_window") or {}
    kind = requested.get("kind") if isinstance(requested, Mapping) else None
    start = _year(requested.get("start_year")) if kind == "range" else None
    end = _year(requested.get("end_year")) if kind == "range" else None
    if kind not in (None, "range", "all"):
        return None
    coverage = _coverage_window(spec.get("resolved") or {})
    if coverage is None:
        return None
    coverage_start, coverage_end = coverage
    return {
        "start": start if start is not None else coverage_start,
        "end": end if end is not None else coverage_end,
        "source": "coverage",
    }


def _wide_facts(spec: dict[str, Any]) -> dict[str, Any]:
    """The single-series facts and derived `wide` verdict for one resolved
    spec: shape kind, entity roles, member count and whether it is exact or a
    floor, how many StatVars the mint serves once the membership settled, and
    whether the identity mints as one series (`graph_envelope`'s docstring
    has the full predicate). Also
    `open_entity_axes`, each entity axis `search` left open with how many members
    it holds, read by `compiler.sources.open_entity_axes_of` as a selected source
    reads them, so a planner can state the party a rank covers. Used
    for the `definite` branch's chosen spec and for each offered spec in a
    `candidates` branch, so a clarify choice's width is known at a clarify
    re-entry without a second upstream `search` call
    (`harness.mcp_client.McpResolver.__call__`'s `spec_id`-set shortcut
    reads it back off the graph's own state)."""
    resolved = spec.get("resolved") or {}
    stat_vars = resolved.get("stat_vars") or {}
    member_count = _whole(stat_vars.get("member_count"))
    served_count = _whole(stat_vars.get("served_count"))
    shape_kind = _identity_shape_kind(spec)
    single_series = shape_kind == "stat_var" or served_count == 1
    several = shape_kind == DESCRIBED_SHAPE_KIND or (served_count or 0) > 1
    return {
        "wide": not single_series and several,
        "stat_var_shape_kind": shape_kind,
        "stat_var_item_count": member_count,
        "stat_var_member_count_exact": stat_vars.get("member_count_exact") is not False,
        "entity_roles": _entity_roles(spec),
        "stat_var_served_count": served_count,
        "open_entity_axes": [
            {"column": axis.column, "members": axis.member_count}
            for axis in open_entity_axes_of(resolved)
        ],
    }


def _whole(value: Any) -> int | None:
    """`value` when it is a whole number, `None` otherwise; a JSON `true` is no count."""
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def graph_envelope(envelope: dict[str, Any]) -> dict[str, Any]:
    """Map the search resolve envelope to the graph's decision inputs.

    The envelope carries one part per variable asked, each with its own
    status. The graph decides on the first part: v1 runs one spec per turn,
    and the envelope's own `status` is the parts' shared value or `mixed`.
    `parts` reports how many parts search returned so the graph can say so.

    `choice_points` counts the methodological forks the resolution settled
    by default on the user's behalf: facet axes on the resolved spec whose
    basis is a default or a fallback (anything but `bound`), plus per-series
    facet variation. The depth policy's own docstring leaves the source of
    choice_points open; this is the deployment's answer, and it is
    deterministic from the envelope.

    `window`, on the `definite` branch only, and only when either side
    states a usable range (`_window` has the full read): the settled time
    window as whole years plus which side settled it, `{"start": ...,
    "end": ..., "source": "requested" | "coverage"}`. `"requested"` is the
    identity's own `requested_window` when its `range` kind names both
    ends outright. Otherwise, when the question gave no time text at all or
    named only one end of an open range ("since 1990"), the years the
    resolver found data across (`_coverage_window`) fill in what's missing,
    `"coverage"`. The resolver has already found data across some span, so
    that is what the planner states rather than guessing. Left out of the
    digest when the identity names only a single most-recent observation
    (`requested_window.kind == "latest"`, never a range either side could
    state) and when neither side states a usable range at all, in which
    case the period is left to the planner to decide, exactly as when the
    identity carries no window at all.

    `wide`, on the `definite` branch only, is a separate signal from
    `choice_points`: whether the resolved identity mints as more than a
    single series. It is not "every slot unbound" -- that condition is true
    of essentially every ordinary ask (the `when`/source slots are always
    unbound, and a demographic slot is unbound whenever the query did not
    name it), so treating it as wide would route to the planner and the
    approval gate exactly the asks that mint as one series. A pinned-arm
    identity (`shape.kind == "stat_var"`) is one series. Any other identity
    is one series when `resolved.stat_vars.served_count` is one: `search`
    states how many StatVars the mint serves once the membership settled,
    so "population of Guatemala" (the family's root, `Count_Person`, serves
    alone) is one series. Every other described-arm identity is wide, a
    membership `search` did not settle (`served_count` null) included. Any
    other identity is wide when `search` states it serves more than one.

    The raw facts a later change to this predicate would need --
    `stat_var_shape_kind`, `stat_var_item_count` (the members `search`
    confirmed, `resolved.stat_vars.member_count`), `entity_roles`,
    `stat_var_served_count` -- ride alongside `wide` in the digest, so reading
    them never needs another trip back into `mcp_client.py`'s raw envelope.

    A `no_data` part maps by its `category`: `unresolved` (or nearest-real
    suggestions present) keeps `no_data` status so the graph's clarify edge
    can offer the suggestions; an understood-but-absent result maps to
    `unsupported`: absent data gets a plain statement, and the graph's
    clarify loop has no give-up edge of its own.

    `part_digests` keeps every part, the first included, as a short summary
    (`_part_summary`): what was asked, its status, its spec_id or
    candidates, what it resolved to, and for a `no_data` part its `reason`
    code. The decision fields above still read the first part only.

    `skills`, on every branch, reads the whole envelope: the skills the
    envelope's `skills` map attaches to any part's specs, as
    `{"id", "description"}` pairs in `common.skills.ordered_skill_ids`
    order, every one of them. `harness.planner_seam` renders the first five
    and counts the rest. The planner loads one with `get_skill` when its
    description bears on the question. A resolve that names none gives an
    empty list.
    """
    return {**_decision_digest(envelope), "skills": _named_skills(envelope)}


def _named_skills(envelope: Mapping[str, Any]) -> list[dict[str, str]]:
    """The envelope's skills as `{"id", "description"}` pairs, most specific first."""
    skills = envelope.get("skills") or {}
    return [
        {
            "id": skill_id,
            "description": " ".join(str(skills[skill_id].get("description") or "").split()),
        }
        for skill_id in ordered_skill_ids(envelope)
    ]


def _decision_digest(envelope: dict[str, Any]) -> dict[str, Any]:
    """`graph_envelope`'s fields: every one but `skills`."""
    kind = str(envelope.get("kind") or "resolution")
    if kind != "resolution":
        return _answer_kind_digest(envelope, kind)
    parts = list(envelope.get("parts") or [])
    count = len(parts) or 1
    digest = _part_digest(envelope, parts[0] if parts else {}, count)
    digest["part_digests"] = [_part_summary(envelope, part) for part in parts]
    return digest


def resolver_digest(envelope: dict[str, Any]) -> dict[str, Any]:
    """The digest a `harness.graph.Resolver` returns for a search envelope.

    `graph_envelope` reduces the raw envelope to `spec_id` alone.
    `harness.planner_seam` needs the `definite` part's own `spec` block
    whole, to build a pinned requirement's source from it with no second
    `search` call (`translator.discovery.discover_source`), so `spec_block`
    holds it, `None` on every other status, matching `graph_envelope`'s own
    `spec_id`. Each part digest carries its own block the same way, for a
    caller that fetches each part (`harness.orchestrator_tools`), and
    `candidate_blocks`, each candidate's block keyed by its spec_id, so a
    candidate the caller picks is fetched as a family or a single series
    the same way a definite part is. `kind` is what the envelope says of
    itself: a resolution, a catalogue, or a declined question, `resolution`
    when it states none. The skills it names for `get_skill` are
    `graph_envelope`'s own `skills`.
    """
    digest = graph_envelope(envelope)
    raw_parts = envelope.get("parts") or []
    raw_part = raw_parts[0] if raw_parts else {}
    digest["spec_block"] = raw_part.get("spec") if digest.get("status") == "definite" else None
    for part, summary in zip(raw_parts, digest.get("part_digests") or [], strict=False):
        summary["spec_block"] = part.get("spec") if summary.get("status") == "definite" else None
        summary["candidate_blocks"] = {
            spec["spec_id"]: spec
            for spec in (part.get("candidates") or {}).get("specs") or []
            if spec.get("spec_id")
        }
    digest["kind"] = envelope.get("kind") or "resolution"
    return digest


def reading_facts(spec: Mapping[str, Any]) -> dict[str, Any]:
    """What a stored search reading holds of a `definite` part's spec block beyond its summary.

    `coverage_kind` is `resolved.coverage.kind` (`exact`, `breadth`,
    `bare`); `requested_window_kind` the identity's `requested_window.kind`
    (`range`, `all`, `latest`); `entity_roles` and `open_entity_axes` as
    `_wide_facts` reads them; `recent_years` the headline's per-year counts,
    newest first, as `{"year", "observations", "series"}`, empty when the
    block carries no headline or the headline no `recent_years`. These are
    keys `harness.turn_records.SearchReading.from_part` reads. The
    orchestrator's search result shows the model none of them.
    """
    spec = dict(spec)
    resolved = spec.get("resolved") or {}
    requested = _identity(spec).get("requested_window")
    headline = resolved.get("headline")
    years = headline.get("recent_years") if isinstance(headline, Mapping) else None
    facts = _wide_facts(spec)
    return {
        "coverage_kind": (resolved.get("coverage") or {}).get("kind"),
        "requested_window_kind": requested.get("kind") if isinstance(requested, Mapping) else None,
        "entity_roles": facts["entity_roles"],
        "open_entity_axes": facts["open_entity_axes"],
        "recent_years": [
            {key: year.get(key) for key in ("year", "observations", "series")}
            for year in years or ()
            if isinstance(year, Mapping)
            and isinstance(year.get("year"), int)
            and isinstance(year.get("observations"), int)
        ],
    }


def chosen_part(spec: Mapping[str, Any], *, variable: str | None) -> dict[str, Any]:
    """A candidate's spec block, chosen and fetched, as a `_part_summary` entry with `reading_facts`.

    `status` is `chosen`, and `variable` is the text of the part that offered
    it. Window, `wide` and the resolved sentence are read from the block
    the way a `definite` part's are.
    """
    spec = dict(spec)
    summary: dict[str, Any] = {
        "variable": variable,
        "status": "chosen",
        "spec_id": spec.get("spec_id"),
        "resolved": (spec.get("resolved") or {}).get("rendered_sentence"),
        "wide": _wide_facts(spec)["wide"],
        **reading_facts(spec),
    }
    window = _window(spec)
    if window is not None:
        summary["window"] = window
    return summary


def _part_summary(envelope: dict[str, Any], part: dict[str, Any]) -> dict[str, Any]:
    """One search part, as `graph_envelope`'s `part_digests` names it."""
    digest = _part_digest(envelope, part, 1)
    resolved = ((part.get("spec") or {}).get("resolved") or {}).get("rendered_sentence")
    summary = {
        "variable": part.get("variable_text"),
        "status": digest["status"],
        "spec_id": digest["spec_id"],
        "resolved": resolved if digest["status"] == "definite" else None,
        "warnings": digest["warnings"],
        "candidates": [
            {key: candidate[key] for key in ("spec_id", "label", "wide")}
            for candidate in digest["candidates"]
        ],
    }
    for key in ("window", "wide"):
        if key in digest:
            summary[key] = digest[key]
    no_data = part.get("no_data") or {}
    for key in ("reason", "message"):
        if value := no_data.get(key):
            summary[key] = value
    return summary


def _answer_kind_digest(envelope: dict[str, Any], kind: str) -> dict[str, Any]:
    """The decision inputs for a `catalogue` or `declined` resolve answer.

    A `catalogue` answer (the question asked what DFO holds) and a
    `declined` answer (out of scope, or a judgement with no data to show)
    carry no parts, so nothing routes to a fetch or a plan: the status is
    `unsupported`, with one warning stating search's reading in plain
    words, which is what the dead end tells the user. `kind`, `reason` and
    the confirmed country ride along for a caller that answers differently.
    """
    place = (envelope.get("scope") or {}).get("place") or {}
    label = place.get("label") or place.get("dcid")
    named = f" It named {label}." if label else ""
    reason = envelope.get("reason")
    if kind == "catalogue":
        message = f"Search read this question as asking what data DFO holds, not as naming data to retrieve.{named}"
    elif reason == "out_of_scope":
        message = f"Search read this question as not about data DFO holds.{named}"
    elif reason == "requires_analysis":
        message = f"Search read this question as asking for a judgement and naming no data to show.{named}"
    else:
        message = f"Search answered with kind {kind!r} and resolved no data.{named}"
    return {
        "status": "unsupported",
        "spec_id": None,
        "choice_points": 0,
        "parts": 1,
        "warnings": [
            *_search_warnings(envelope, {}),
            {"code": reason or kind, "severity": "warn", "message": message},
        ],
        "candidates": [],
        "kind": kind,
        "reason": reason,
        "scope_place": label,
        "part_digests": [],
    }


def _part_digest(envelope: dict[str, Any], part: dict[str, Any], count: int) -> dict[str, Any]:
    """`graph_envelope`'s decision fields for one part of `envelope`."""
    status = str(part.get("status") or envelope.get("status") or "unsupported")
    warnings = _search_warnings(envelope, part)
    if status == "definite":
        spec = part.get("spec") or {}
        facets = (spec.get("resolved") or {}).get("facets") or {}
        choice_points = sum(
            1
            for facet in facets.values()
            if facet.get("selection") == "per_series" or facet.get("basis") != "bound"
        )
        digest = {
            "status": "definite",
            "spec_id": spec.get("spec_id"),
            "choice_points": choice_points,
            "parts": count,
            "warnings": warnings,
            "candidates": [],
            **_wide_facts(spec),
        }
        window = _window(spec)
        if window is not None:
            digest["window"] = window
        return digest
    if status == "candidates":
        candidates = _spec_candidates((part.get("candidates") or {}).get("specs") or [])
        return {
            "status": "candidates",
            "spec_id": None,
            "choice_points": len(candidates),
            "parts": count,
            "warnings": warnings,
            "candidates": candidates,
        }
    if status == "no_data":
        no_data = part.get("no_data") or {}
        nearest = no_data.get("nearest_real") or []
        if no_data.get("category") == "unresolved" or nearest:
            return {
                "status": "no_data",
                "spec_id": None,
                "choice_points": 0,
                "parts": count,
                "warnings": warnings,
                "candidates": _spec_candidates(nearest),
            }
        warnings.append(
            {
                "code": "no_data",
                "severity": "warn",
                "message": str(no_data.get("reason") or "the data is not available"),
            }
        )
        return {
            "status": "unsupported",
            "spec_id": None,
            "choice_points": 0,
            "parts": count,
            "warnings": warnings,
            "candidates": [],
        }
    unsupported = part.get("unsupported") or {}
    if unsupported.get("detail"):
        warnings.append(
            {"code": "unsupported", "severity": "warn", "message": str(unsupported["detail"])}
        )
    return {
        "status": "unsupported",
        "spec_id": None,
        "choice_points": 0,
        "parts": count,
        "warnings": warnings,
        "candidates": [],
    }


__all__ = ["ResolverError", "chosen_part", "graph_envelope", "reading_facts"]

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
from collections.abc import Mapping, Sequence
from typing import Any

from common.skills import ordered_skill_ids
from compiler.sources import (
    DESCRIBED_SHAPE_KIND,
    elected_stat_vars,
    open_entity_axes_of,
    pinned_stat_var,
)

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
    """One entry per offered spec: its id, its sentence, its width facts
    (`_wide_facts`) and `stat_vars`, the StatVars `search` elected for it
    (`_elected_dcids`), which a `get_data` call for the chosen candidate
    posts as `resolved_stat_var_dcids`."""
    return [
        {
            "spec_id": spec.get("spec_id"),
            "label": (spec.get("resolved") or {}).get("rendered_sentence") or spec.get("spec_id"),
            "stat_vars": _elected_dcids(spec.get("resolved") or {}),
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


def _elected_items(resolved: dict[str, Any]) -> Sequence[Mapping[str, Any]]:
    return (resolved.get("stat_vars") or {}).get("items") or []


def _elected_dcids(resolved: dict[str, Any]) -> list[str]:
    """The dcid of each elected item, in order, as `get_data` takes them.

    Read by the reader a source's are (`compiler.sources.elected_stat_vars`),
    so a chosen candidate, a `definite` reading and a plan's retrieval post
    the same list for the same block. A list that reader refuses offers none
    of its entries: one cut to the readable ones would differ from the list a
    plan's retrieval posts. The refusal is logged and the candidate stays on
    offer, since this reads a candidate that may never be chosen and must
    not fail the digest.
    """
    try:
        return list(elected_stat_vars(resolved))
    except ValueError:
        logger.warning(
            "a candidate elects StatVars get_data does not take; it offers none to post",
            exc_info=True,
        )
        return []


def _year(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coverage_window(resolved: dict[str, Any]) -> tuple[int, int] | None:
    """The earliest `start` and latest `end` year across every elected
    item's own `data_date_range`, the same per-item field a served
    `get_data` mint reads to build a frame's window. This is the span
    the resolver found data across, regardless of whether the question
    named a time range. `None` when no elected item states a usable
    range."""
    starts: list[int] = []
    ends: list[int] = []
    for item in _elected_items(resolved):
        date_range = item.get("data_date_range") if isinstance(item, Mapping) else None
        if not isinstance(date_range, Mapping):
            continue
        start = _year(date_range.get("start"))
        end = _year(date_range.get("end"))
        if start is not None and end is not None:
            starts.append(start)
            ends.append(end)
    if not starts:
        return None
    return min(starts), max(ends)


def _window(spec: dict[str, Any]) -> dict[str, int | str] | None:
    """The dataset requirement's settled time window, as whole years, and
    which side settled it.

    `spec.definition.identity.requested_window` (the same field
    `compiler.evidence._scope_window_issues` reads) states a complete
    range only under its `range` kind, at `start_year`/`end_year`; that
    pair wins when both parse, `source: "requested"`. A `range` naming
    only one end ("since 1990"), or an `all` kind with no time text at
    all, falls back to the resolved stat_var items' own
    `data_date_range` (`_coverage_window`), the coverage the resolver
    actually found data across: a bare open end is completed from it
    (`{"start": 1990, "end": <the coverage's latest year>}`), and a
    fully open request takes the whole span, `source: "coverage"`.
    `latest` names a single most-recent observation rather than a range
    either side could state, so it is left out. Returns `None` when
    neither side states a usable range, leaving `window` out of the
    envelope and the period a decision for the caller."""
    identity = _identity(spec)
    requested = identity.get("requested_window") or {}
    kind = requested.get("kind") if isinstance(requested, Mapping) else None
    start = end = None
    if kind == "range":
        start = _year(requested.get("start_year"))
        end = _year(requested.get("end_year"))
    if start is not None and end is not None:
        return {"start": start, "end": end, "source": "requested"}
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
    spec: shape kind, entity roles, elected item count, truncation, and
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
    identity = _identity(spec)
    items = _elected_items(resolved)
    shape_kind = _identity_shape_kind(spec)
    entity_roles = _entity_roles(spec)
    item_count = len(items)
    truncated = bool((resolved.get("stat_vars") or {}).get("truncated"))
    described_arm = shape_kind == DESCRIBED_SHAPE_KIND
    pinned_arm = shape_kind == "stat_var"
    single_series = pinned_stat_var(identity, items, slots=resolved.get("slots")) is not None
    # One series -- a pinned arm, or a described family whose elected item
    # is its root -- always wins: see `graph_envelope`'s docstring for why
    # `truncated` must not be allowed to override it.
    wide = False if pinned_arm or single_series else (described_arm or truncated)
    return {
        "wide": wide,
        "stat_var_shape_kind": shape_kind,
        "stat_var_item_count": item_count,
        "entity_roles": entity_roles,
        "stat_var_truncated": truncated,
        "open_entity_axes": [
            {"column": axis.column, "members": axis.member_count}
            for axis in open_entity_axes_of(resolved)
        ],
    }


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
    ends outright. Otherwise -- the question gave no time text at all, or
    it named only one end of an open range ("since 1990") -- the resolved
    stat_var items' own `data_date_range` fills in what's missing,
    `"coverage"`: the resolver has already found data across some span, so
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
    identity (`shape.kind == "stat_var"`) is one series. A described-arm
    identity is one series when `search` elected exactly one item and that
    item is the family's root (`compiler.sources.pinned_stat_var`): entity
    roles exactly `{"observationAbout"}`, no `set`-bound slot, the item's
    `constraint_properties` equal the identity's `value`- and `exact`-bound
    slot properties, its `slot_values` equal those slots' refs, and no
    `value`-bound slot's entry in `resolved.slots` carries an `expansion`.
    A root dominates every member with extra constraints, which is the test
    dfo-apis runs to prove its member walk complete, so "population of
    Guatemala" (nothing bound, `Count_Person` with
    `constraint_properties: []`) and "female population in Mexico"
    (`gender` bound, `Count_Person_Female` with `["gender"]`) both mint as
    one series (`data_mcp.substitution.root_member` names the case the
    block cannot show in which a root still mints as several series). Every
    other described-arm identity is wide: nothing elected, two or more
    elected items, an item constrained beyond the bound slots, an item
    whose `constraint_properties` is missing or `null` (Search does not
    know its constraints), an item constrained to another code than a
    slot binds, a `value` binding that widens to descendant codes (a
    hierarchical family such as `crsPurpose`, whose subtree the mint
    serves), a `set`-bound slot, or a role-bearing identity
    (`origin`/`destination`, the CRS `FinancialFlow` case) that
    legitimately enumerates many members. Any other identity is wide when
    `resolved.stat_vars.truncated` is `true`.

    Neither `resolved.stat_vars.member_count` nor `truncated` may drive this
    directly: `member_count` is capped at 1 for a non-set-bound standard
    shape today by an early upstream `break`, and both fields change
    meaning once dfo-apis #893 lands. `truncated` is computed against the
    200-item display cap over that same pre-collapsed count, so it is
    structurally `false` for every ordinary standard ask until #893 ships --
    reading it as the primary signal would route every standard identity as
    narrow, exactly the bug this field exists to fix.

    One series -- a pinned arm, or a described identity whose elected item
    is its root -- always wins over `truncated`, as it does in dfo-apis,
    where a confirmed root proves a truncated member walk complete.
    Post-#893, `member_count` stops being `len(items)` and becomes the
    confirmed-at-place population, and `truncated` becomes `len(items) <
    member_count`; `items` itself is unchanged, so an ordinary standard
    identity elects exactly one winner while `member_count` grows past one
    and `truncated` flips true. If `truncated` could override a root, every
    one of those identities -- the case the root test exists to keep off
    the slow path -- would flip wide the day #893 ships, with nothing in
    this repo's fixtures able to catch it, since they all carry today's
    values. So `truncated` sharpens a verdict only for an identity the root
    test does not cover; it never inverts a single-series call.

    The raw facts a later change to this predicate would need --
    `stat_var_shape_kind`, `stat_var_item_count`, `entity_roles`,
    `stat_var_truncated` -- ride alongside `wide` in the digest, so reading
    them never needs another trip back into `mcp_client.py`'s raw envelope.

    A `no_data` part maps by its `category`: `unresolved` (or nearest-real
    suggestions present) keeps `no_data` status so the graph's clarify edge
    can offer the suggestions; an understood-but-absent result maps to
    `unsupported`: absent data gets a plain statement, and the graph's
    clarify loop has no give-up edge of its own.

    `part_digests` keeps every part, the first included, as a short summary
    (`_part_summary`): what was asked, its status, its spec_id or
    candidates, and what it resolved to. The decision fields above still
    read the first part only.

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


__all__ = ["ResolverError", "graph_envelope"]

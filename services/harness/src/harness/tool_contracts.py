"""What the resolver and the MCP clients agree on.

The resolver's refusal type, `resolver_digest` (the digest of a search
envelope the orchestrator's `search` tool stores and reads: status / spec_id /
candidates / parts / warnings / wide / skills, and each part's spec block),
the presentation tool names `McpExecutor` routes on, and the refusal helpers:
the `origin` values an executor stamps, `HarnessCode` and `harness_refusal`
for a refusal the harness mints. The full envelope, with its dozens of unbound definition slots,
never reaches session state.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from enum import StrEnum
from typing import Any

from common.skills import ordered_skill_ids
from compiler.sources import (
    DESCRIBED_SHAPE_KIND,
    block_groups,
    open_entity_axes_of,
    resolved_source_mentions,
    total_reading,
)
from data_mcp.digest import source_codes

from harness.wire import RetryMode

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


class HarnessCode(StrEnum):
    """The `code` of a failure the harness mints itself."""

    # `harness.chart_binding`: a chart select that draws more series than the
    # cap, and one that is malformed.
    SERIES_LIMIT_EXCEEDED = "series_limit_exceeded"
    SELECT_INVALID = "select_invalid"
    # A chart cut to its largest series finds no period that at least half of
    # its comparable members hold.
    NO_SHARED_PERIOD = "no_shared_period"
    # The kind of a frame's members, which a chart cut ranks by, could not be
    # read.
    MEMBER_KIND_UNREAD = "member_kind_unread"
    # `harness.rolling_cards`: what a card whose period moves can meet before
    # it is written.
    TITLE_QUOTES_ROLLING_PERIOD = "title_quotes_rolling_period"
    ROLLING_REPLAY_PENDING = "rolling_replay_pending"
    ROLLING_REPLAY_REFUSED = "rolling_replay_refused"
    ROLLING_REPLAY_NOT_CONFIGURED = "rolling_replay_not_configured"
    ROLLING_SELECT_NEEDS_NEW_FRAME = "rolling_select_needs_new_frame"
    RELATIVE_PERIOD_ON_CLOSED_WINDOW = "relative_period_on_closed_window"
    # `harness.mcp_client`: a tool the executor does not route, a card the
    # platform stored with no revision to draw or no source to state, a search
    # with no parts, and a refusal that arrived with no code.
    UNKNOWN_TOOL = "unknown_tool"
    RESOLVED_REVISION_UNAVAILABLE = "resolved_revision_unavailable"
    PROVENANCE_UNAVAILABLE = "provenance_unavailable"
    NO_SEARCH_PARTS = "no_search_parts"
    UNSPECIFIED = "unspecified"
    # `harness.orchestrator_tools`: a `fetch` that served no observations, one
    # whose call did not complete, and one that ran past its deadline.
    EMPTY_RESULT = "empty_result"
    TRANSIENT_UNAVAILABLE = "transient_unavailable"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    # A chart the `make_charts` tool was asked for and did not draw.
    CHART_NOT_DRAWN = "chart_not_drawn"
    # A tool called after its budget of calls for the turn is spent.
    TOOL_BUDGET_SPENT = "tool_budget_spent"
    # A tool call whose arguments do not fit the tool's schema.
    TOOL_ARGUMENTS_INVALID = "tool_arguments_invalid"
    # A card edit that changes a field set only when the card is created.
    FIELD_CREATE_ONLY = "field_create_only"
    # A card write that was sent and got no answer, so the platform may hold
    # the card.
    WRITE_UNCONFIRMED = "write_unconfirmed"
    # A card that was drawn, whose payload the figures of a reply could not
    # be read from.
    FIGURES_UNREAD = "figures_unread"
    # A turn whose state could not be saved.
    STATE_NOT_SAVED = "state_not_saved"
    # `harness.graph.execute`: a step whose source has to be found again.
    REDISCOVERY_NEEDED = "rediscovery_needed"
    # A chart that names a frame no `fetch` of the turn returned.
    FRAME_UNKNOWN = "frame_unknown"
    # A hand-off called while the turn already waits on one.
    HAND_OFF_PENDING = "hand_off_pending"
    # An exception that states no refusal: a tool call that raised, recorded
    # as its failed step, or one that ended the turn's stream.
    RAISED = "raised"
    # A tool this deployment has not wired.
    TOOL_NOT_CONFIGURED = "tool_not_configured"
    # A card removal that names no card of the conversation.
    CARD_NOT_FOUND = "card_not_found"
    # `harness.graph.resume`: a run whose stored submission cannot be read,
    # and one that raised while it was carried on.
    RUN_RECORD_UNREADABLE = "run_record_unreadable"
    RESUME_FAILED = "resume_failed"
    # `harness.mcp_client.McpExecutor.publish_result`: why a derived frame was
    # not published. The frame store could not be reached or refused the read,
    # the dataset's versions could not be read, the replay recipe could not be
    # built, the store no longer holds the frame, no publisher is wired, and
    # apis holds the frame under a recipe of another normal form.
    STORE_UNAVAILABLE = "store_unavailable"
    STORE_REFUSED = "store_refused"
    DATASET_VERSIONS_UNAVAILABLE = "dataset_versions_unavailable"
    RECIPE_UNBUILDABLE = "recipe_unbuildable"
    REPLAY_UNAVAILABLE = "replay_unavailable"
    PUBLISH_UNAVAILABLE = "publish_unavailable"
    RECIPE_NORMAL_FORM_CONFLICT = "recipe_normal_form_conflict"


# The retry mode of a harness code whose refusal can be acted on. Every other
# code is `do_not_retry`, since the same call meets the same refusal.
_RETRY_MODES: Mapping[HarnessCode, RetryMode] = {
    HarnessCode.TOOL_ARGUMENTS_INVALID: RetryMode.CORRECT_AND_RETRY,
    HarnessCode.SELECT_INVALID: RetryMode.CORRECT_AND_RETRY,
    HarnessCode.FIELD_CREATE_ONLY: RetryMode.CORRECT_AND_RETRY,
    HarnessCode.STATE_NOT_SAVED: RetryMode.RETRY_AFTER_DELAY,
    HarnessCode.MEMBER_KIND_UNREAD: RetryMode.RETRY_AFTER_DELAY,
    HarnessCode.FRAME_UNKNOWN: RetryMode.CORRECT_AND_RETRY,
    HarnessCode.CARD_NOT_FOUND: RetryMode.CORRECT_AND_RETRY,
}


def harness_retry_mode(code: HarnessCode) -> RetryMode:
    """The retry mode a failure the harness mints under `code` states."""
    return _RETRY_MODES.get(code, RetryMode.DO_NOT_RETRY)


def harness_refusal(
    *,
    code: HarnessCode,
    message: str,
    fix: Mapping[str, Any] | None = None,
    retry_mode: RetryMode | None = None,
) -> dict[str, Any]:
    """The refusal grammar's body for a refusal the harness mints itself: stamped
    `origin: harness`, with the `fix` given and the retry mode of its code.
    `retry_mode` states another mode than the code's own, for a refusal whose
    message names a corrected call the harness accepts, which is
    `correct_and_retry`."""
    return {
        "code": code.value,
        "retry_mode": (retry_mode or harness_retry_mode(code)).value,
        "fix": dict(fix) if fix is not None else None,
        "origin": REFUSAL_ORIGIN_HARNESS,
        "message": message,
    }


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


def _elected_year(resolved: Mapping[str, Any]) -> int | None:
    """The year Search's answer states it serves: `resolved.headline.year`,
    which for a `latest` identity is the year Search elected. `None` when
    the answer carries no headline or its headline states no whole year, as
    for data with no flow roles or with periods shorter than a year."""
    headline = resolved.get("headline")
    return _whole(headline.get("year")) if isinstance(headline, Mapping) else None


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
    `latest` names one year, the year Search elected to serve
    (`_elected_year`): that year is both ends, `source: "elected"`, and
    the coverage is not read, since it spans every year the members hold.
    Returns `None` when no side states a usable range or year, leaving
    `window` out of the envelope and the period a decision for the
    caller."""
    identity = _identity(spec)
    if (asked := requested_range(identity)) is not None:
        return {"start": asked[0], "end": asked[1], "source": "requested"}
    requested = identity.get("requested_window") or {}
    kind = requested.get("kind") if isinstance(requested, Mapping) else None
    start = _year(requested.get("start_year")) if kind == "range" else None
    end = _year(requested.get("end_year")) if kind == "range" else None
    if kind == "latest":
        elected = _elected_year(spec.get("resolved") or {})
        if elected is None:
            return None
        return {"start": elected, "end": elected, "source": "elected"}
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
    `candidates` branch."""
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
    """Map the search resolve envelope to its digest.

    The envelope carries one part per variable asked, each with its own
    status. The top-level fields read the first part, and the envelope's
    own `status` is the parts' shared value or `mixed`. `parts` reports how
    many parts search returned.

    `choice_points` counts the methodological forks the resolution settled
    by default on the user's behalf: facet axes on the resolved spec whose
    basis is a default or a fallback (anything but `bound`), plus per-series
    facet variation.

    `window`, on the `definite` branch only, and only when either side
    states a usable range or year (`_window` has the full read): the settled
    time window as whole years plus which side settled it, `{"start": ...,
    "end": ..., "source": "requested" | "coverage" | "elected"}`.
    `"requested"` is the
    identity's own `requested_window` when its `range` kind names both
    ends outright. Otherwise, when the question gave no time text at all or
    named only one end of an open range ("since 1990"), the years the
    resolver found data across (`_coverage_window`) fill in what's missing,
    `"coverage"`. The resolver has already found data across some span, so
    that is what the planner states rather than guessing. An identity
    naming the most recent observation (`requested_window.kind ==
    "latest"`) carries the one year Search elected to serve
    (`resolved.headline.year`) as both ends, `"elected"`. Left out of the
    digest when a `latest` answer's headline states no year and when
    neither side states a usable range at all, in which case the period is
    left to the planner to decide, exactly as when the identity carries no
    window at all.

    `wide`, on the `definite` branch only, is a separate signal from
    `choice_points`: whether the resolved identity mints as more than a
    single series. It is not "every slot unbound" -- that condition is true
    of essentially every ordinary ask (the `when`/source slots are always
    unbound, and a demographic slot is unbound whenever the query did not
    name it), so treating it as wide would mark wide exactly the asks that
    mint as one series. A pinned-arm identity (`shape.kind == "stat_var"`)
    is one series. Any other identity
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
    suggestions present) keeps `no_data` status, with the suggestions as its
    `candidates`. An understood-but-absent result maps to `unsupported`,
    with search's reason as a `no_data` warning.

    `part_digests` keeps every part, the first included, as a short summary
    (`_part_summary`): what was asked, its status, its spec_id or
    candidates, what it resolved to, for a `no_data` part its `reason`
    code (with, for `qualifier_not_bound`, the phrases search could not
    apply and the sentence of the match it found without them), and, when
    search states any, its `departures`, each a `code` and
    the `text` saying how the answer differs from what was asked. A
    `definite` part binding a named set as its published total carries
    `group_readings`, a sentence each saying so (`_total_readings`). A
    `definite` part whose count Search elected beside others carries
    `counted_as`, with `alternatives` (other counts) and `other_prices`, each
    by its own spec_id (`measure_choice`), and `source_codes`, the publisher
    codes of the family it found, each with its source's label when the part
    binds that source (`data_mcp.digest.source_codes`). The top-level fields
    above read the first part only.

    `skills`, on every branch, reads the whole envelope: the skills the
    envelope's `skills` map attaches to any part's specs, as
    `{"id", "description"}` pairs in `common.skills.ordered_skill_ids`
    order, every one of them. The orchestrator's `search` result shows the
    first five and counts the rest, and each stored reading keeps the ids.
    A resolve that names none gives an empty list.
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
    """The digest a `harness.orchestrator_tools.Resolver` returns for a search envelope.

    `graph_envelope` reduces each spec to its `spec_id`. Each part digest
    also carries a `definite` part's own `spec` block whole as `spec_block`,
    `None` on every other status, and `candidate_blocks`, the block of each
    candidate its digest offers keyed by its spec_id (`_offered_specs`).
    `harness.graph.state.store_search` stores them, so `fetch` and the
    planner's pins read a part's or a picked candidate's block with no
    second `search` call. The top-level `spec_block` holds
    the first part's block the same way. `kind` is what the envelope says
    of itself: a resolution, a catalogue, or a declined question, `resolution`
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
            spec["spec_id"]: spec for spec in _offered_specs(part) if spec.get("spec_id")
        }
    digest["kind"] = envelope.get("kind") or "resolution"
    return digest


def _offered_specs(part: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """The spec blocks `part` offers as its digest's `candidates` (`_part_digest`).

    A `candidates` part offers its `candidates.specs`; a `no_data` part its
    nearest real series (`no_data.nearest_real`), which the orchestrator may
    fetch and the planner pin as it would a candidate. Every other part
    offers none.
    """
    status = part.get("status")
    if status == "candidates":
        return list((part.get("candidates") or {}).get("specs") or [])
    if status == "no_data":
        return list((part.get("no_data") or {}).get("nearest_real") or [])
    return []


def reading_facts(spec: Mapping[str, Any]) -> dict[str, Any]:
    """What a stored search reading holds of a `definite` part's spec block beyond its summary.

    `coverage_kind` is `resolved.coverage.kind` (`exact`, `breadth`,
    `bare`); `requested_window_kind` the identity's `requested_window.kind`
    (`range`, `all`, `latest`); `entity_roles` and `open_entity_axes` as
    `_wide_facts` reads them. These are keys
    `harness.turn_records.SearchReading.from_part` reads, and the
    orchestrator's search result shows the model none of them. The block's
    per-year counts (`resolved.headline.recent_years`) are left out, since a
    frame's own manifest states what its year holds
    (`common.period_note.period_note`).
    """
    spec = dict(spec)
    resolved = spec.get("resolved") or {}
    requested = _identity(spec).get("requested_window")
    facts = _wide_facts(spec)
    return {
        "coverage_kind": (resolved.get("coverage") or {}).get("kind"),
        "requested_window_kind": requested.get("kind") if isinstance(requested, Mapping) else None,
        "entity_roles": facts["entity_roles"],
        "open_entity_axes": facts["open_entity_axes"],
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
    if part.get("separate") is True:
        summary["separate"] = True
    for key in ("window", "wide"):
        if key in digest:
            summary[key] = digest[key]
    if digest["status"] in ("definite", "candidates") and (
        mentions := resolved_source_mentions(part)
    ):
        summary["resolved_sources"] = [mention.model_dump(mode="json") for mention in mentions]
    no_data = part.get("no_data") or {}
    for key in ("reason", "message"):
        if value := no_data.get(key):
            summary[key] = value
    if no_data.get("reason") == "qualifier_not_bound":
        summary.update(_unapplied_phrases(no_data))
    if ambiguous := no_data.get("ambiguous_sources"):
        summary["ambiguous_sources"] = ambiguous
    departures = [
        {"code": str(departure["code"]), "text": str(departure["text"])}
        for departure in part.get("departures") or ()
        if isinstance(departure, Mapping) and departure.get("code") and departure.get("text")
    ]
    if departures:
        summary["departures"] = departures
    if digest["status"] == "definite" and (readings := _total_readings(part.get("spec") or {})):
        summary["group_readings"] = readings
    if digest["status"] == "definite":
        summary.update(measure_choice(part.get("spec") or {}))
        if codes := source_codes(part.get("spec") or {}):
            summary["source_codes"] = codes
    return summary


def measure_choice(spec: Mapping[str, Any]) -> dict[str, Any]:
    """The count Search elected for `spec` and the other counts and prices it offers, as data.

    `counted_as` is `resolved.measure_label`. `alternatives` holds each
    `resolved.alternative_measures` entry on axis "what" (another count,
    "net disbursements") and `other_prices` each on axis "unit" (the same
    count in other prices, "constant prices"), both as `{spec_id, label}`
    in wire order. An entry missing its spec_id or label is left out, as
    data-mcp's `digest._measure_choice` leaves it out, and so is a list
    left empty. Empty when no alternative remains.
    """
    resolved = spec.get("resolved") or {}
    label = resolved.get("measure_label")
    choice: dict[str, Any] = {}
    for axis, key in (("what", "alternatives"), ("unit", "other_prices")):
        offered = [
            {"spec_id": spec_id, "label": alternative["label"]}
            for alternative in resolved.get("alternative_measures") or ()
            if alternative.get("axis") == axis
            and alternative.get("label")
            and (spec_id := (alternative.get("definition") or {}).get("spec_id"))
        ]
        if offered:
            choice[key] = offered
    if not isinstance(label, str) or not label or not choice:
        return {}
    return {"counted_as": label, **choice}


def _total_readings(spec: Mapping[str, Any]) -> list[str]:
    """A sentence per named set `spec` binds as its published total (`compiler.sources.total_reading`).

    Search states no departure for a set read whole, so this is where the
    reading is said. A set read as its members has search's own
    `group_read_as_members` departure. A block whose sets are not readable
    has none.
    """
    try:
        groups = block_groups(spec)
    except ValueError:
        logger.warning("search: named sets of %s are not readable", spec.get("spec_id"))
        return []
    entities = (spec.get("resolved") or {}).get("entities") or ()
    labels = {
        str(entity["ref"]): str(entity["label"])
        for entity in entities
        if isinstance(entity, Mapping) and entity.get("ref") and entity.get("label")
    }
    return [text for group in groups if (text := total_reading(group, labels)) is not None]


def _unapplied_phrases(no_data: Mapping[str, Any]) -> dict[str, Any]:
    """A `qualifier_not_bound` refusal's phrases and the match it found without them.

    `unbound_qualifiers` is each phrase's text, in query order, without
    repeats. `flagged_match` is the rendered sentence of the same-variable
    spec search found but would not answer with. Either key is left out
    when the wire carries nothing for it.
    """
    facts: dict[str, Any] = {}
    texts = [
        str(qualifier["text"])
        for qualifier in no_data.get("unbound_qualifiers") or ()
        if isinstance(qualifier, Mapping) and qualifier.get("text")
    ]
    if texts:
        facts["unbound_qualifiers"] = list(dict.fromkeys(texts))
    flagged = no_data.get("flagged_match") or {}
    if sentence := (flagged.get("resolved") or {}).get("rendered_sentence"):
        facts["flagged_match"] = sentence
    return facts


def series_description(spec: Mapping[str, Any]) -> tuple[str, tuple[str, ...]] | None:
    """A spec block's heading and the terms of its byline, from `resolved.sentence_plan`.

    The heading is the text of every `heading` token, without its closing
    full stop: "Total climate-specific financial support provided worldwide,
    2011 to 2022". The byline's terms are its slot tokens' texts, in order:
    ("US dollars", "SDG Global Database"). `None` when the block carries no
    heading tokens.
    """
    plan = (spec.get("resolved") or {}).get("sentence_plan") or ()
    tokens = [token for token in plan if isinstance(token, Mapping)]
    heading = "".join(
        str(token.get("text") or "") for token in tokens if token.get("segment") == "heading"
    ).strip()
    if not heading:
        return None
    byline = tuple(
        text
        for token in tokens
        if token.get("segment") == "byline"
        and token.get("kind") == "slot"
        and (text := str(token.get("text") or "").strip())
    )
    return heading.removesuffix("."), byline


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
    label = place.get("label") or place.get("ref")
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


__all__ = [
    "ResolverError",
    "chosen_part",
    "graph_envelope",
    "reading_facts",
    "series_description",
]

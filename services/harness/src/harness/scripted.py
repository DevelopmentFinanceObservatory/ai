"""Deterministic seams for the studio's scripted test mode.

`STUDIO_SCRIPTED=true` serves the full wire contract with no LLM, no MCP
call, and no upstream: every seam here -- resolver, executor, planner,
compose, edit -- answers from fixed data instead of generating. The QUESTION
picks the path, so a frontend test can reach every stream shape on demand:

    contains "comprehensive"  -> plan pause (approve/amend/reject)
    contains "ambiguous"      -> elicitation pause (answer/free text/skip)
    contains "nothing"        -> no-data close
    contains "fail"           -> stream-fatal capability_unavailable error
    contains "canvas"         -> `ScriptedConverse` sets `about: "canvas"`,
                                 which routes straight to `canvas_answer`:
                                 the reply comes from the digest alone, no
                                 data fetch at all
    contains "compare"        -> direct fetch of a two-country frame -> one
                                 line_chart with both entities selected
    contains "guatemala"      -> a direct fetch of whichever two-entity frame
                                 matches the entities converse actually
                                 resolved (Guatemala+Nigeria alone; Kenya+
                                 Guatemala once a Kenya card is on the canvas
                                 and the wording adds to it) -> one line_chart
    contains "to the chart", "to that chart", or "same chart"
                              -> `ScriptedConverse` sets `revises` to the
                                 newest block in the digest it was given, so
                                 the same fetch path lands as a change to
                                 that card instead of a new one (a digest is
                                 required)
    contains "misnamed"       -> plan pause; after approval the analysis step
                                 refuses `role_ambiguous` (a column the frame
                                 lacks) and the retry's forwarded fix refuses
                                 as a schema error, so the graph replans
    contains "withheld"       -> direct fetch -> the card is refused with the
                                 harness's `provenance_unavailable` (the
                                 frame has no source manifest) -> done
    contains "rejected"       -> direct fetch -> the card is refused by the
                                 presentation service with `constraint_violation`
                                 (a stamped `origin: service` refusal) -> done
    contains "unstamped"      -> direct fetch -> the card is refused with a code
                                 and no `origin` at all -> done
    anything else             -> direct fetch -> one component -> done

Frame CONTENT is fixed. With `deterministic_ids` (which scripted mode turns
on), wire ids are sequential (`int_0000000000000001`, ...), so two runs of
the same requests produce identical streams (`duration_ms` excepted).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from common.mcp_telemetry import WRITE_TOKEN_KEY
from compiler.surface import (
    ProgressCallback,
    ProgressReport,
    ToolCallResult,
    ToolSchema,
    ToolSurfaceUnavailableError,
    deliver_progress,
)
from data_mcp.search_result import SEARCH_ENVELOPE_META_KEY
from google.adk import Context
from planner.plan import plan_revision
from planner.workflow import ToolId

from harness.scripted_plan import (
    SCRIPTED_TOOL_SCHEMAS,
    scripted_aggregate_result,
    scripted_manifest,
    scripted_plan,
)
from harness.tool_contracts import (
    REFUSAL_ORIGIN_SERVICE,
    ResolverError,
    harness_refusal,
)

SCRIPTED_SPEC = "spec_" + "f" * 32
# The frame behind a "compare" question: one measure, two countries, so its
# view holds two series and its card draws one line each.
SCRIPTED_TWO_ENTITY_SPEC = "spec_" + "e" * 32
# The frame behind "add Guatemala to the chart" when the canvas already
# holds a Kenya card: the same two-series shape as `SCRIPTED_TWO_ENTITY_SPEC`,
# over the revised card's own entity plus the newly named one.
SCRIPTED_ADD_ENTITY_SPEC = "spec_" + "9" * 32
# The frame behind a "misnamed" question: its analysis steps refuse.
SCRIPTED_MISNAMED_SPEC = "spec_" + "d" * 32
# The frames behind the "withheld", "rejected" and "unstamped" questions: each
# one's `write_component` is refused (`_WRITE_REFUSALS`).
SCRIPTED_WITHHELD_SPEC = "spec_" + "c" * 32
SCRIPTED_REJECTED_SPEC = "spec_" + "b" * 32
SCRIPTED_UNSTAMPED_SPEC = "spec_" + "a" * 32
# The frame (and its choice points) a question word selects, checked in
# order. "guatemala" is handled separately, below the ambiguous/nothing/fail
# checks in `ScriptedResolver`, since which frame it names depends on the
# entities converse resolved, not the word alone.
_FRAME_BY_QUESTION_WORD: dict[str, tuple[str, int]] = {
    "compare": (SCRIPTED_TWO_ENTITY_SPEC, 0),
    "misnamed": (SCRIPTED_MISNAMED_SPEC, 2),
    "withheld": (SCRIPTED_WITHHELD_SPEC, 0),
    "rejected": (SCRIPTED_REJECTED_SPEC, 0),
    "unstamped": (SCRIPTED_UNSTAMPED_SPEC, 0),
}
# The two entities behind `SCRIPTED_TWO_ENTITY_SPEC`, in the order its frame
# holds them.
SCRIPTED_TWO_ENTITY_CODES = ("country/GTM", "country/NGA")
SCRIPTED_TWO_ENTITY_LABELS = ("Guatemala", "Nigeria")
# The two entities behind `SCRIPTED_ADD_ENTITY_SPEC`: the revised Kenya
# card's own entity, then the newly named Guatemala.
SCRIPTED_ADD_ENTITY_CODES = ("country/KEN", "country/GTM")
SCRIPTED_ADD_ENTITY_LABELS = ("Kenya", "Guatemala")
# Every multi-entity frame this module knows, keyed by spec_id: its entity
# codes, labels, and each series' first/last values, in matching order.
# `_component`, `ScriptedExecutor.describe_frame` and `ScriptedCompose` all
# read this one table rather than special-casing `SCRIPTED_TWO_ENTITY_SPEC`,
# so a second two-entity frame (like this one) is a data row, not a branch.
_MULTI_ENTITY_FRAMES: dict[
    str, tuple[tuple[str, ...], tuple[str, ...], tuple[float, ...], tuple[float, ...]]
] = {
    SCRIPTED_TWO_ENTITY_SPEC: (
        SCRIPTED_TWO_ENTITY_CODES,
        SCRIPTED_TWO_ENTITY_LABELS,
        (45.8, 37.2),
        (72.6, 54.5),
    ),
    SCRIPTED_ADD_ENTITY_SPEC: (
        SCRIPTED_ADD_ENTITY_CODES,
        SCRIPTED_ADD_ENTITY_LABELS,
        (48.5, 45.8),
        (63.6, 72.6),
    ),
}
# Which multi-entity spec answers a given entity-label set, so the resolver
# can pick a frame from what converse actually resolved (the revised card's
# entity plus the newly named one) instead of re-guessing from the question
# text.
_MULTI_ENTITY_SPEC_BY_LABELS: dict[frozenset[str], str] = {
    frozenset(labels): spec_id
    for spec_id, (_codes, labels, _firsts, _lasts) in _MULTI_ENTITY_FRAMES.items()
}
_FRAME_COLUMNS = ["stat_var", "observationAbout", "time_period", "value", "unit"]
SCRIPTED_CV = "cv_" + "f" * 32
_FRAME_REF = {"spec_id": SCRIPTED_SPEC, "content_version": SCRIPTED_CV}

# The one-series fact summary a Kenyan card carries, and the two-series one
# the two-country frame's card carries.
_FACT_SUMMARY = {
    "component_id": "cmp_scripted0001",
    "revision": 1,
    "series": [
        {
            "id": "ken",
            "label": "Kenya",
            "point_count": 64,
            "coordinates": {
                "unit": "Year",
                "stat_var": "dcid:LifeExpectancy_Person",
                "observationAbout": "country/KEN",
            },
            "first": 48.5,
            "last": 63.6,
        }
    ],
    "indicators": [
        {
            "id": "dcid:LifeExpectancy_Person",
            "name": "Life Expectancy",
            "description": "mp=lifeExpectancy,pt=Person",
        }
    ],
    "attribution": {
        "label": "WorldDevelopmentIndicators · 1 indicator(s)",
        "source_count": 1,
        "indicator_count": 1,
    },
    "window": {"start": "1960", "end": "2023"},
}


def _multi_entity_fact_summary(
    codes: tuple[str, ...],
    labels: tuple[str, ...],
    firsts: tuple[float, ...],
    lasts: tuple[float, ...],
) -> dict[str, Any]:
    """The fact summary a multi-entity card carries: one series per entity,
    in `codes`/`labels` order."""
    return {
        **_FACT_SUMMARY,
        "series": [
            {
                "id": code.rsplit("/", 1)[-1].lower(),
                "label": label,
                "point_count": 64,
                "coordinates": {
                    "unit": "Year",
                    "stat_var": "dcid:LifeExpectancy_Person",
                    "observationAbout": code,
                },
                "first": first,
                "last": last,
            }
            for code, label, first, last in zip(codes, labels, firsts, lasts, strict=True)
        ],
        "attribution": {
            "label": "WorldDevelopmentIndicators · 1 indicator(s)",
            "source_count": 1,
            "indicator_count": 1,
        },
    }


# Every multi-entity frame's fact summary, precomputed from `_MULTI_ENTITY_FRAMES`.
_MULTI_ENTITY_FACT_SUMMARIES: dict[str, dict[str, Any]] = {
    spec_id: _multi_entity_fact_summary(*entry) for spec_id, entry in _MULTI_ENTITY_FRAMES.items()
}

_PROVENANCE = {
    "inputs": [_FRAME_REF],
    "sources": [
        {
            "id": "dc/base/WorldDevelopmentIndicators",
            "name": "World Development Indicators",
            "licence": "CC BY 4.0",
        }
    ],
    "freshness": {
        "data_through": "2023",
        "last_refreshed": "2026-06-30",
        "partial_period": False,
    },
    "warnings": [],
    "generated_at": "2026-01-01T00:00:00Z",
    "citation": "World Development Indicators (2026).",
}


# The envelope's own provenance block, in the platform's resolved-revision
# shape (`inputs` is set per card).
_ENVELOPE_SOURCE_REF = "src_wdi"
_ENVELOPE_PROVENANCE = {
    "execution_ref": None,
    "sources": [
        {
            "id": _ENVELOPE_SOURCE_REF,
            "name": "World Development Indicators",
            "url": "https://databank.worldbank.org/source/world-development-indicators",
            "licence": "CC BY 4.0",
            "retrieved": "2026-06-30",
        }
    ],
    "freshness": {
        "data_through": "2023",
        "last_refreshed": "2026-06-30",
        "next_expected": None,
        "partial_period": False,
    },
    "warnings": [],
    "generated_at": "2026-01-01T00:00:00Z",
    "citation": "World Development Indicators (2026).",
    "refreshed_from": None,
}
_UNIT = "Year"
_Y_LABEL = "Life expectancy"


def _points(first: float, last: float, *, start: int, end: int) -> list[dict[str, Any]]:
    """One point a year from `start` to `end`, on a straight line from
    `first` to `last`, matching the fact summary's `point_count`."""
    span = end - start
    return [
        {"period": str(start + step), "value": round(first + (last - first) * step / span, 1)}
        for step in range(span + 1)
    ]


def _resolved_component(
    component_type: str, *, title: str, fact_summary: dict[str, Any]
) -> dict[str, Any]:
    """The envelope's `component` for one of the four types a scripted card
    takes, drawn from `fact_summary`'s series and window, in the shape of
    dfo-spec-kit's catalogue fixtures."""
    window = fact_summary["window"]
    series: list[dict[str, Any]] = [
        {
            "id": entry["id"],
            "label": entry["label"],
            "unit": _UNIT,
            "source_ref": _ENVELOPE_SOURCE_REF,
            "points": _points(
                entry["first"], entry["last"], start=int(window["start"]), end=int(window["end"])
            ),
        }
        for entry in fact_summary["series"]
    ]
    heading = {"text": title, "authorship": "model"}
    if component_type == "bar_chart":
        return {
            "type": "bar_chart",
            "title": heading,
            "series": [{**entry, "points": entry["points"][-1:]} for entry in series],
            "category_axis": {
                "mode": "entity",
                "category_dimension": "observationAbout",
                "period": window["end"],
            },
            "value_axis": {
                "label": _Y_LABEL,
                "scale": "linear",
                "range": {"min": None, "max": None},
            },
        }
    if component_type == "stat_card":
        latest = series[0]["points"][-1]
        return {
            "type": "stat_card",
            "title": heading,
            "value": latest["value"],
            "label": f"{_Y_LABEL}, {series[0]['label']}",
            "period": latest["period"],
            "unit": _UNIT,
        }
    if component_type == "table":
        return {
            "type": "table",
            "title": heading,
            "columns": [
                {"key": "observationAbout", "label": "Country"},
                {"key": "period", "label": "Year"},
                {"key": "value", "label": _Y_LABEL},
            ],
            "rows": [
                {
                    "series_id": entry["id"],
                    "dimensions": {"observationAbout": entry["label"]},
                    "period": point["period"],
                    "source_ref": _ENVELOPE_SOURCE_REF,
                    "unit": _UNIT,
                    "value": point["value"],
                }
                for entry in series
                for point in entry["points"]
            ],
        }
    return {
        "type": "line_chart",
        "title": heading,
        "series": series,
        "x_axis": {"window": {"start": window["start"], "end": window["end"]}},
        "y_axis": {"label": _Y_LABEL, "scale": "linear", "range": {"min": None, "max": None}},
    }


# The stored intent `describe_component` answers with for a scripted card.
_STORED_INTENT = {
    "type": "line_chart",
    "title": "Life expectancy in Kenya (1960-2023)",
    "select": {"stat_var": "dcid:LifeExpectancy_Person"},
    "frame_ref": _FRAME_REF,
}


def _component(
    component_id: str = "cmp_scripted0001",
    *,
    component_type: str = "line_chart",
    title: str = "Life expectancy in Kenya (1960-2023)",
    revision: int = 1,
    frame_ref: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """A scripted `component.created` body: the resolved revision in
    `payload`, beside the harness's own fact summary and provenance."""
    resolved_ref = frame_ref or _FRAME_REF
    fact_summary = _MULTI_ENTITY_FACT_SUMMARIES.get(resolved_ref.get("spec_id"), _FACT_SUMMARY)
    catalog_version = "2026.8.0"
    return {
        "component_id": component_id,
        "revision": revision,
        "type": component_type,
        "catalog_version": catalog_version,
        "min_catalog_version": catalog_version,
        "payload": {
            "schema_version": 1,
            "catalog_version": catalog_version,
            "component_id": component_id,
            "revision": revision,
            "provenance": {"inputs": [resolved_ref], **_ENVELOPE_PROVENANCE},
            "component": _resolved_component(
                component_type, title=title, fact_summary=fact_summary
            ),
        },
        "fact_summary": {**fact_summary, "component_id": component_id, "revision": revision},
        "provenance": {**_PROVENANCE, "inputs": [resolved_ref]},
    }


def _scripted_view(*, entity_codes: tuple[str, ...] | None = None) -> dict[str, Any]:
    """The describe_frame result for a scripted frame. A multi-entity
    frame's view marks `observationAbout` plural with its codes and
    `series_count` accordingly; every other frame holds one series."""
    if entity_codes:
        dimensions = [
            {"name": "stat_var", "plural": False, "values": ["dcid:LifeExpectancy_Person"]},
            {
                "name": "observationAbout",
                "plural": True,
                "values": list(entity_codes),
            },
        ]
    else:
        dimensions = [{"name": "stat_var", "values": ["dcid:LifeExpectancy_Person"]}]
    return {
        "outcome": "ok",
        "frame_refs": [],
        "view": {
            "dimensions": dimensions,
            "period_grain": "year",
            "coverage": ["1960", "2023"],
            "series_count": len(entity_codes) if entity_codes else 1,
        },
    }


def _part_entity_labels(parts: list[dict[str, Any]] | None) -> frozenset[str]:
    """The entity labels named across every search part's `entities` --
    `ScriptedConverse`'s own reading of the question against the canvas
    digest, not a second guess from the question text."""
    return frozenset(
        str(entity["text"])
        for part in (parts or [])
        for entity in part.get("entities") or []
        if isinstance(entity, dict) and entity.get("text")
    )


class ScriptedResolver:
    """The question's words pick the resolve outcome; see the module doc.

    "guatemala" is the one exception: which frame it names depends on which
    entities converse actually resolved (`parts`, forwarded from
    `ScriptedConverse`'s `search_parts`), since the same word means "compare
    Guatemala and Nigeria" alone and "Kenya and Guatemala" once a Kenya card
    is on the canvas and the question adds to it.
    """

    async def __call__(
        self,
        *,
        question: str,
        spec_id: str | None = None,
        wide: bool = False,
        parts: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        lowered = question.lower()
        if spec_id is not None:
            # Mirrors `McpResolver.__call__`'s clarify re-entry: the part
            # count comes from the decomposition `graph.probe` carries
            # forward.
            return self._definite(choice_points=0, wide=wide, parts=len(parts) if parts else 1)
        if "fail" in lowered:
            raise ResolverError(
                "search refused: scripted failure",
                refusal={
                    "code": "capability_unavailable",
                    "retry_mode": "do_not_retry",
                    "fix": {"detail": "scripted: identity does not reproduce spec_id"},
                },
            )
        if "nothing" in lowered:
            return {
                "status": "unsupported",
                "spec_id": None,
                "choice_points": 0,
                "parts": 1,
                "warnings": [
                    {"code": "no_data", "severity": "warn", "message": "scripted: no data"}
                ],
                "candidates": [],
            }
        if "ambiguous" in lowered:
            return {
                "status": "candidates",
                "spec_id": None,
                "choice_points": 2,
                "parts": 1,
                "warnings": [],
                "candidates": [
                    {"spec_id": SCRIPTED_SPEC, "label": "Reading A (scripted)"},
                    {"spec_id": "spec_" + "e" * 32, "label": "Reading B (scripted)"},
                ],
            }
        if "guatemala" in lowered:
            spec_id = _MULTI_ENTITY_SPEC_BY_LABELS.get(
                _part_entity_labels(parts), SCRIPTED_TWO_ENTITY_SPEC
            )
            return self._definite(choice_points=0, spec_id=spec_id)
        for word, (frame_spec_id, choice_points) in _FRAME_BY_QUESTION_WORD.items():
            if word in lowered:
                return self._definite(choice_points=choice_points, spec_id=frame_spec_id)
        return self._definite(choice_points=2 if "comprehensive" in lowered else 0)

    @staticmethod
    def _definite(
        *,
        choice_points: int,
        spec_id: str = SCRIPTED_SPEC,
        wide: bool = False,
        parts: int = 1,
    ) -> dict[str, Any]:
        return {
            "status": "definite",
            "spec_id": spec_id,
            "choice_points": choice_points,
            "parts": parts,
            "warnings": [],
            "candidates": [],
            "wide": wide,
        }


class ScriptedExecutor:
    """Fixed results per tool. `get_data` carries no component of its own --
    `ScriptedCompose`, below, is what turns a produced frame (a direct
    fetch's or an approved plan's) into one, the same mechanism a real
    compose model uses, so scripted mode never double-creates one."""

    async def __call__(
        self, *, tool: str, args: dict[str, Any], on_progress: ProgressCallback | None = None
    ) -> dict[str, Any]:
        del on_progress  # a fixed result reports no progress
        if tool == "get_data":
            return {
                "outcome": "ok",
                "frame_refs": [
                    {**_FRAME_REF, "spec_id": str(args.get("spec_id") or SCRIPTED_SPEC)}
                ],
                "summary": "64 observations · 1960 to 2023 · WorldDevelopmentIndicators",
            }
        if tool == "describe_frame":
            frame_ref = args.get("frame_ref") or {}
            entry = _MULTI_ENTITY_FRAMES.get(frame_ref.get("spec_id"))
            return _scripted_view(entity_codes=entry[0] if entry else None)
        if tool == "describe_component":
            return {
                "outcome": "ok",
                "frame_refs": [],
                "component_body": {
                    "component_id": str(args.get("component_id", "cmp_scripted0001")),
                    "revision": 1,
                    "stored_intent": _STORED_INTENT,
                    "fact_summary": _FACT_SUMMARY,
                },
            }
        if tool == "write_component":
            intent = args.get("intent") or {}
            frame_ref = intent.get("frame_ref")
            if "component_id" in args:  # a revise patch
                return {
                    "outcome": "ok",
                    "frame_refs": [],
                    "component": _component(
                        str(args["component_id"]),
                        title=str(intent.get("title") or "Life expectancy in Kenya"),
                        revision=2,
                        frame_ref=frame_ref,
                    ),
                    "summary": "revised",
                }
            return _write_refusal(frame_ref) or {
                "outcome": "ok",
                "frame_refs": [],
                "component": _component(
                    "cmp_scripted0002",
                    component_type=str(intent.get("type", "line_chart")),
                    title=str(intent.get("title") or "Scripted card"),
                    frame_ref=frame_ref,
                ),
                "summary": "created",
                # The grant a later Edit Card patch revises this card with
                # (`common.mcp_telemetry.WRITE_TOKEN_KEY`).
                WRITE_TOKEN_KEY: "cmw_scripted0002",
            }
        # analysis verbs: a fixed derived frame reference, except on the
        # misnamed frame, which plays the platform's two refusals in order.
        return _misnamed_refusal(args) or {
            "outcome": "ok",
            "frame_refs": [_FRAME_REF],
            "summary": f"scripted {tool} result",
        }


class ScriptedPlannerModel:
    """A `PlannerModel` seam for scripted mode.

    Drafts the canned plan `harness.scripted_plan.scripted_plan` builds. A
    re-entry -- a replan after a refusal, or a user amendment -- redrafts the
    same steps as the next version rather than reasoning about which one it
    is, since scripted mode has no model to reason with; that is exactly the
    "the same steps as a new version" behaviour the retired `_draft_plan`
    gave a re-proposal.
    """

    async def __call__(
        self,
        *,
        ctx: Context,
        question: str,
        envelope: dict[str, Any],
        prior_plan: dict[str, Any] | None = None,
        refusal: dict[str, Any] | None = None,
        amendment: str | None = None,
        canvas_digest: dict[str, Any] | None = None,
        conversation_digest: dict[str, Any] | None = None,
        spec_id_elected: bool = False,
        spec_id_elected_label: str | None = None,
        on_progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        # Nothing to report and nothing to run through the graph's own
        # session: this seam has no model round trips to relay and no
        # agent to host via `ctx.run_node`. Scripted mode drafts the same
        # canned plan regardless of who chose the envelope's spec_id.
        del ctx, refusal, amendment, canvas_digest, conversation_digest
        del spec_id_elected, spec_id_elected_label, on_progress
        spec_id = envelope.get("spec_id")
        parts = int(envelope.get("parts") or 1)
        plan = scripted_plan(question, spec_id=spec_id, parts=parts)
        if prior_plan is not None:
            prior_version = int(prior_plan.get("version", 1))
            plan = plan.model_copy(
                update={"version": prior_version + 1, "previous_version": prior_version}
            )
        return {
            "plan": plan.model_dump(mode="json"),
            "summary": {"version": plan.version, "revision": plan_revision(plan)},
        }


class ScriptedConverse:
    """A `ConverseModel` seam for scripted mode: one `life expectancy` part
    naming the countries the question asks about.

    "to the chart", "to that chart", or "same chart" is the wording this
    seam reads as a change to a card already on the canvas: it names the
    newest block in the digest it was given as `revises`, and the part
    carries that card's own entities with the newly asked ones, which is
    what a real converse draft does.

    "canvas" is a separate trigger, checked independently of `revises`: it
    sets `about: "canvas"`, which routes the turn to `canvas_answer`
    instead of a data fetch -- the module docstring's diversion, not this
    seam's own drafting.
    """

    async def __call__(
        self,
        *,
        question: str,
        canvas_digest: dict[str, Any] | None = None,
        conversation_digest: dict[str, Any] | None = None,
        recall_result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        del conversation_digest, recall_result
        lowered = question.lower()
        revises = _scripted_revises(lowered, canvas_digest)
        entities = [
            {"text": label} for label in _scripted_entities(lowered, canvas_digest, revises)
        ]
        part: dict[str, Any] = {"variable": "life expectancy", "entities": entities}
        return {
            "text": "Looking into life expectancy.",
            "search_parts": [part],
            "about": "canvas" if "canvas" in lowered else "data",
            "revises": revises,
            "recall": None,
        }


# The word that makes a scripted question a change to the newest card; the
# entity it adds is named by its own country word.
_SCRIPTED_REVISE_WORDS = ("to the chart", "to that chart", "same chart")
_SCRIPTED_ENTITY_WORDS = {
    "guatemala": "Guatemala",
    "nigeria": "Nigeria",
    "kenya": "Kenya",
}


def _scripted_revises(lowered_question: str, canvas_digest: dict[str, Any] | None) -> str | None:
    """The newest canvas block's id when the question asks to change a chart
    and a digest holds one; None otherwise (a fresh, standalone question)."""
    if not any(word in lowered_question for word in _SCRIPTED_REVISE_WORDS):
        return None
    blocks = (canvas_digest or {}).get("blocks") or []
    newest = blocks[-1] if blocks else {}
    component_id = newest.get("component_id") if isinstance(newest, dict) else None
    return str(component_id) if component_id else None


def _scripted_entities(
    lowered_question: str,
    canvas_digest: dict[str, Any] | None,
    revises: str | None,
) -> list[str]:
    """The entity names the scripted part asks for: the revised card's own
    entities first, then whichever country the question names."""
    named = [label for word, label in _SCRIPTED_ENTITY_WORDS.items() if word in lowered_question]
    if revises is None:
        return named or ["Kenya"]
    blocks = (canvas_digest or {}).get("blocks") or []
    existing = next(
        (
            block.get("entities") or []
            for block in blocks
            if isinstance(block, dict) and block.get("component_id") == revises
        ),
        [],
    )
    return [str(entity) for entity in existing] + [
        label for label in named if label not in existing
    ]


class ScriptedCompose:
    """A `ComposeModel` seam for scripted mode: one `line_chart` intent per
    produced frame that `frame_provenance` says is the plan's final step, so
    a direct fetch's frame and an approved plan's terminal frame both reach
    the canvas through `write_component`, the same mechanism a real compose
    model uses -- never a frame that skips it, and never an intermediate
    step's frame either, since `harness.graph._admissible_intents` is not
    the only thing standing between this seam and a card drawn from the
    wrong step."""

    async def __call__(
        self,
        *,
        question: str,
        frame_refs: list[dict[str, Any]],
        frame_views: list[dict[str, Any]],
        frame_provenance: list[dict[str, Any]],
    ) -> dict[str, Any]:
        del question

        def select_for(ref: dict[str, Any]) -> dict[str, Any]:
            # A multi-entity frame draws one line per entity, so the entity
            # dimension is selected as the array of every code. The Kenya
            # frame holds one series, which its `stat_var` value alone pins.
            entry = _MULTI_ENTITY_FRAMES.get(ref.get("spec_id"))
            if entry is not None:
                codes, _labels, _firsts, _lasts = entry
                return {"observationAbout": list(codes)}
            return {"stat_var": "dcid:LifeExpectancy_Person"}

        def title_for(ref: dict[str, Any]) -> str:
            entry = _MULTI_ENTITY_FRAMES.get(ref.get("spec_id"))
            if entry is not None:
                _codes, labels, _firsts, _lasts = entry
                return f"Life expectancy in {' and '.join(labels)} (1960-2023)"
            return "Life expectancy in Kenya (1960-2023)"

        return {
            "intents": [
                {
                    "type": "line_chart",
                    "title": title_for(ref),
                    "select": select_for(ref),
                    "frame_ref": ref,
                }
                for ref, provenance in zip(frame_refs, frame_provenance, strict=True)
                if provenance.get("is_final_step")
            ]
        }


def _misnamed_refusal(args: dict[str, Any]) -> dict[str, Any] | None:
    """The misnamed frame's refusals: `role_ambiguous` with a diagnostic fix
    first, then a schema refusal once that fix arrives as an argument."""
    frame_ref = args.get("frame_ref")
    spec = frame_ref.get("spec_id") if isinstance(frame_ref, dict) else None
    if spec != SCRIPTED_MISNAMED_SPEC:
        return None
    if "parameter" in args:
        return {
            "outcome": "refusal",
            "refusal": {
                "code": "unspecified",
                "retry_mode": "do_not_retry",
                "fix": {},
                "message": "unexpected keyword argument 'parameter'",
            },
        }
    return {
        "outcome": "refusal",
        "refusal": {
            "code": "role_ambiguous",
            "retry_mode": "correct_and_retry",
            "fix": {"parameter": "entity_by", "columns": ["country"], "available": _FRAME_COLUMNS},
            "message": "entity_by names column(s) this frame has no grain for: ['country']",
        },
    }


# The refusal `write_component` answers for each frame that has one, keyed by
# `spec_id`: the harness withholding a card whose frame has no source manifest
# (as `McpExecutor` does), a presentation service's own `constraint_violation`
# (`origin: service`, with the advice text that carries the constraint id), and
# a refusal that states no `origin`.
_WRITE_REFUSALS: dict[str, dict[str, Any]] = {
    SCRIPTED_WITHHELD_SPEC: harness_refusal(
        code="provenance_unavailable",
        message=(
            "no source manifest is linked to this frame; withholding the card "
            "rather than stating a source it cannot support"
        ),
    ),
    SCRIPTED_REJECTED_SPEC: {
        "code": "constraint_violation",
        "retry_mode": "correct_and_retry",
        "fix": None,
        "advice": "violates constraint 'comparable_units_required'",
        "origin": REFUSAL_ORIGIN_SERVICE,
        "message": "two series do not share a commensurable unit",
    },
    SCRIPTED_UNSTAMPED_SPEC: {
        "code": "internal_error",
        "retry_mode": "do_not_retry",
        "fix": None,
        "message": "Internal error",
    },
}


def _write_refusal(frame_ref: object) -> dict[str, Any] | None:
    """The `write_component` refusal for the frame `frame_ref` names, when it is
    one of `_WRITE_REFUSALS`'s; `None` for any other."""
    if not isinstance(frame_ref, dict):
        return None
    refusal = _WRITE_REFUSALS.get(str(frame_ref.get("spec_id")))
    return {"outcome": "refusal", "refusal": dict(refusal)} if refusal else None


def scripted_spec_block(
    *, spec_id: str, identity: Mapping[str, Any], stat_vars: Sequence[str]
) -> dict[str, Any]:
    """The Spec block scripted Search serves for `identity`: the definition as
    posted, a sentence naming the spec_id, and the StatVars posted, since no
    graph is there to re-read them."""
    return {
        "spec_id": spec_id,
        "definition": {"schema_version": 1, "spec_id": spec_id, "identity": dict(identity)},
        "resolved": {
            "rendered_sentence": f"The scripted reading of {spec_id}",
            "stat_vars": {"items": [{"ref": ref, "slot_values": []} for ref in stat_vars]},
        },
    }


class ScriptedSurface:
    """A `ToolSurface` over `harness.scripted_plan`'s own declared schemas.

    `list_tools()` answers `SCRIPTED_TOOL_SCHEMAS` verbatim -- the same
    objects `scripted_plan`'s workflow is compiled against -- so a canned
    plan's `input_schema_hash`/`output_schema_hash`/`implementation_version`
    pins always match what this surface declares. `call_tool` answers by
    `(tool.name, arguments)`: a fixed manifest for `get_data`, and for
    `aggregate` the fixed result `scripted_plan.scripted_aggregate_result`
    states, unless the frame it reads is the misnamed one, in which case it
    plays the same `role_ambiguous` refusal `ScriptedExecutor` plays on the
    frozen path -- `_misnamed_refusal` is shared between the two, since both
    are one server's answer to one frame. `resolve_definition` stands in for
    Search with `scripted_spec_block`, and stores nothing, since no scripted
    answer reads a definition.

    `get_data_progress` holds the reports a `get_data` call hands its
    caller's `on_progress`, in order, before it answers, standing in for
    data-mcp's reports while it waits on a background operation. It is
    empty unless a caller scripts it. A raising callback fails nothing
    (`compiler.surface.deliver_progress`).
    """

    def __init__(self, *, get_data_progress: Sequence[ProgressReport] = ()) -> None:
        self._get_data_progress = tuple(get_data_progress)

    async def list_tools(self) -> list[ToolSchema]:
        return list(SCRIPTED_TOOL_SCHEMAS)

    async def call_tool(
        self,
        *,
        tool: ToolId,
        arguments: Mapping[str, Any],
        on_progress: ProgressCallback | None = None,
    ) -> ToolCallResult:
        if tool.name == "get_data":
            if on_progress is not None:
                for report in self._get_data_progress:
                    deliver_progress(on_progress, report)
            spec_id = str(arguments.get("spec_id") or "")
            return ToolCallResult(
                is_error=False,
                structured_content=scripted_manifest(spec_id),
                text="scripted get_data result",
            )
        if tool.name == "aggregate":
            refused = _misnamed_refusal(dict(arguments))
            if refused is not None:
                refusal = refused["refusal"]
                return ToolCallResult(
                    is_error=True, structured_content=refusal, text=str(refusal["message"])
                )
            return ToolCallResult(
                is_error=False,
                structured_content=scripted_aggregate_result(),
                text="scripted aggregate result",
            )
        if tool.name == "resolve_definition":
            block = scripted_spec_block(
                spec_id=str(arguments.get("spec_id") or ""),
                identity=dict(arguments.get("identity") or {}),
                stat_vars=list(arguments.get("stat_vars") or []),
            )
            envelope = {
                "schema_version": 1,
                "status": "definite",
                "parts": [{"status": "definite", "spec": block}],
            }
            return ToolCallResult(
                is_error=False,
                structured_content={"spec_id": block["spec_id"], "status": "definite"},
                text=f"Search resolved the identity of {block['spec_id']}: definite.",
                meta={SEARCH_ENVELOPE_META_KEY: envelope},
            )
        raise ToolSurfaceUnavailableError(f"no scripted answer for {tool.name!r}")


class ScriptedEdit:
    """A type word in the instruction recreates; anything else renames."""

    async def __call__(
        self,
        *,
        message: str,
        stored_intent: dict[str, Any],
        frame_view: dict[str, Any],
    ) -> dict[str, Any]:
        del frame_view
        lowered = message.lower()
        for component_type in ("bar_chart", "line_chart", "stat_card", "table"):
            if component_type.replace("_", " ") in lowered or component_type in lowered:
                return {
                    "text": f"Changed the card to a {component_type.replace('_', ' ')}.",
                    "patch": None,
                    "recreate": {
                        "type": component_type,
                        "title": str(stored_intent.get("title", "Scripted card")),
                        "select": stored_intent.get("select")
                        or {"stat_var": "dcid:LifeExpectancy_Person"},
                    },
                }
        return {
            "text": "Renamed the card.",
            "patch": {"title": f"Scripted: {message[:80]}"},
            "recreate": None,
        }

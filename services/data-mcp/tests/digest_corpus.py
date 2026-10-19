"""The digest corpus: authored search envelopes tagged with the semantic status they render.

Each `CorpusCase` pairs a user question and the tool `parts` that ask it with an
upstream envelope authored so every part shares one of the resolve door's four
semantic statuses (`definite`, `candidates`, `no_data`, `unsupported`).
`test_digest_corpus.py` checks the corpus is internally coherent; `test_digest_budget.py`
measures each case's rendered digest against a committed character budget.

This module defines its **own** envelope factories rather than importing `_spec` /
`_envelope` from `test_digest.py`: those defaults are tuned for that test's minimal
needs, and reusing them would mean every budget number moves when someone tunes them
for an unrelated reason. It also imports nothing from `net_guard.py` — the coherence
checks here need no network guard, which keeps this module independent of what
`test_digest_budget.py` loads.

Sentences are authored at the long end of realistic (~85 characters) because digest
size tracks them directly; budgets set against short sentences would flatter the
surface. `FRAME_PAIRS` and `SENTENCE_PAIRS` name cases whose envelopes differ only in
a way that should (frame pairs) or shouldn't (sentence pairs) move the rendered
digest — see the module docstrings of `test_digest_corpus.py` and `test_digest_budget.py`
for what each grouping proves.

`CASES` and `BUDGETS` are built additively from per-status lists and a per-status
mapping, so a second status family (`get_data`'s, once that tool lands) can be added
beside the four here rather than requiring this module to be restructured around it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Sentence pool ---------------------------------------------------------------------
#
# Realistic dev-finance asks, each ~85 characters, reused as both the resolved
# `rendered_sentence` and (unless a case needs something more specific) the tool
# part's `variable` text. Indexed positionally by the cases below.

_SENTENCES: list[str] = [
    "malaria grant disbursements from France to Togo, 2015 to 2024, constant 2015 US dollars",
    "official development assistance for primary education in Kenya, 2018 to 2023 annually",
    "humanitarian aid commitments from Germany to Yemen, 2019 to 2022, current US dollars",
    "climate adaptation grant funding from Japan to Bangladesh, 2016 to 2021, constant prices",
    "food security programme disbursements from Canada to Mali, 2017 to 2023, current dollars",
    "water and sanitation infrastructure loans from the World Bank to Ethiopia, 2015 to 2020",
    "maternal health grant disbursements from Norway to Malawi, 2014 to 2022, constant dollars",
    "renewable energy project financing from the EU to Senegal, 2016 to 2024, current euros",
    "refugee resettlement assistance from Sweden to Jordan, 2018 to 2023, current US dollars",
    "agricultural resilience grants from the Netherlands to Niger, 2015 to 2021, constant prices",
    "vaccine procurement financing from Gavi to Nigeria, 2019 to 2024, current US dollars",
    "debt relief commitments from the Paris Club to Zambia, 2020 to 2023, nominal dollars",
]

_BASE_SENTENCE = _SENTENCES[0]

# The known suffix appended to `_BASE_SENTENCE` for `definite_1_long_sentence`. Its
# own character count is what the sentence-sensitivity regression compares the
# digest's character delta against.
_LONG_SENTENCE_SUFFIX = ", excluding technical assistance and administrative overhead costs"

# Coverage fixtures, exercising every `coverage.kind` the renderer branches on -------

_EXACT_COVERAGE: dict[str, Any] = {
    "kind": "exact",
    "has_data": True,
    "observation_count": 10,
    "window": {"start_year": 2015, "end_year": 2024},
}
_BREADTH_COVERAGE: dict[str, Any] = {
    "kind": "breadth",
    "has_data": True,
    "dimensions": [{"count": 12, "label": "countries"}, {"count": 4, "label": "years"}],
}
_BARE_COVERAGE: dict[str, Any] = {"kind": "bare", "has_data": True}
_BARE_NO_DATA_COVERAGE: dict[str, Any] = {"kind": "bare", "has_data": False}

# Entities and facets, exercising the renderer's roles and facet bases --------------

_FRANCE_TOGO_ENTITIES: list[dict[str, Any]] = [
    {"role": "provider", "label": "France", "ref": "country/FRA"},
    {"role": "recipient", "label": "Togo", "ref": "country/TGO"},
]

_BOUND_AND_DEFAULT_FACETS: dict[str, Any] = {
    "provenance": {"basis": "bound", "ref": "oecd_dac_2a", "label": "OECD DAC table 2a"},
    "unit": {"basis": "declared_default", "ref": "usd", "label": "US dollars"},
}
_FALLBACK_SOURCE_FACET: dict[str, Any] = {
    "provenance": {"basis": "fallback", "ref": "unesco_uis", "label": "UNESCO UIS"}
}
_PER_SERIES_UNIT_FACET: dict[str, Any] = {"unit": {"basis": "bound", "selection": "per_series"}}


def _facet_option(dcid: str, label: str, *, value_kind: str = "enum_value") -> dict[str, Any]:
    """One `SlotOption`-shaped facet option value, keyed by dcid and label."""
    return {"value": {"ref": {"ref": dcid, "label": label}, "value_kind": value_kind}}


# `facet_options`, populated on every definite spec below since apis serves it on
# every definite spec -- one entry per axis a resolved
# candidate carried an observed value for. `provenance` exceeds the three-value
# render cap on purpose, exercising the digest's "and N more" tail.
_FACET_OPTIONS_FIXTURE: list[dict[str, Any]] = [
    {
        "axis": "provenance",
        "options": [
            _facet_option("oecd_dac_2a", "OECD DAC table 2a", value_kind="source"),
            _facet_option("oecd_dac_1", "OECD DAC table 1", value_kind="source"),
            _facet_option("unesco_uis", "UNESCO UIS", value_kind="source"),
            _facet_option("world_bank_wdi", "World Development Indicators", value_kind="source"),
            _facet_option("un_comtrade", "UN Comtrade", value_kind="source"),
        ],
        "truncated": True,
        "source": "observed",
    },
    {
        "axis": "measurementMethod",
        "options": [
            _facet_option("CensusACS5yr", "5-year American Community Survey"),
            _facet_option("CensusACS1yr", "1-year American Community Survey"),
            _facet_option("WorldBankEstimate", "World Bank estimate"),
            _facet_option("UNPopulationDivisionEstimate", "UN Population Division estimate"),
        ],
        "truncated": True,
        "source": "observed",
    },
    {
        "axis": "unit",
        "options": [
            _facet_option("usd", "US dollars"),
            _facet_option("eur", "euros"),
        ],
        "truncated": False,
        "source": "observed",
    },
]

# Five axes, five options each, exercising the digest's axis-count cap: only the
# first three axes get their own options line, the rest collapse to one trailing
# count line.
_FACET_OPTIONS_FIVE_AXES_FIXTURE: list[dict[str, Any]] = [
    {
        "axis": axis,
        "options": [
            _facet_option(f"{axis}_{index}", f"{axis} option {index}") for index in range(5)
        ],
        "truncated": False,
        "source": "observed",
    }
    for axis in ["provenance", "unit", "measurementMethod", "observationPeriod", "scalingFactor"]
]


# Synthetic bulk, for the frame pairs ------------------------------------------------
#
# Material that inflates an envelope's serialized size without `render_digest` ever
# reading it, so the frame pairs can prove the digest tracks only what it renders.


def _bulk_observation_points(count: int) -> list[dict[str, Any]]:
    """`count` synthetic observation points, shaped like the engine's real payload."""
    return [{"date": str(2000 + index), "value": float(index)} for index in range(count)]


def _bulk_stat_vars(count: int) -> list[dict[str, Any]]:
    """`count` synthetic stat_var entries, shaped like `resolved.stat_vars.items`."""
    return [
        {"ref": f"SyntheticStatVar{index}", "label": f"synthetic {index}"} for index in range(count)
    ]


def _bulk_diagnostics(count: int) -> dict[str, Any]:
    """Non-rendered diagnostic bulk. `render_digest` reads only `diagnostics.warnings`."""
    return {
        "engine_debug_spans": [
            {"span": f"span_{index}", "duration_ms": index} for index in range(count)
        ]
    }


def _bulk_definition_tree(*, width: int, depth: int) -> dict[str, Any]:
    """A synthetic tree `depth` levels deep and `width` keys wide at each level.

    Sits beside (never on) the pointer path a `definition` is resolved against, so it
    inflates envelope size without moving what `_render_unsupported_part` reads out.
    """
    node: dict[str, Any] = {"leaf": "synthetic"}
    for level in range(depth):
        node = {f"branch_{level}_{index}": dict(node) for index in range(width)}
    return node


# Envelope factories ------------------------------------------------------------------


def _spec(
    spec_id: str = "spec_1",
    *,
    sentence: str = _BASE_SENTENCE,
    confidence: str | None = "high",
    coverage: dict[str, Any] | None = None,
    stat_vars: list[dict[str, Any]] | None = None,
    entities: list[dict[str, Any]] | None = None,
    facets: dict[str, Any] | None = None,
    facet_options: list[dict[str, Any]] | None = None,
    observations: dict[str, Any] | None = None,
    source_codes: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """One resolved spec block: sentence, confidence, coverage, StatVars, entities, facets, facet options."""
    items = stat_vars if stat_vars is not None else []
    resolved: dict[str, Any] = {
        "rendered_sentence": sentence,
        "confidence": confidence,
        "coverage": coverage if coverage is not None else _BARE_COVERAGE,
        "stat_vars": {
            "items": items,
            "member_count": len(items),
            "truncated": False,
            **({"source_codes": source_codes} if source_codes else {}),
        },
        "entities": entities if entities is not None else [],
        "facets": facets if facets is not None else {},
        "facet_options": facet_options if facet_options is not None else [],
    }
    if observations is not None:
        resolved["observations"] = observations
    return {"spec_id": spec_id, "resolved": resolved}


def _envelope_part(
    status: str,
    *,
    variable_text: str | None,
    warnings: list[dict[str, Any]] | None = None,
    **status_payload: Any,
) -> dict[str, Any]:
    """One `parts` entry: a part's own status, variable_text, warnings, and payload."""
    return {
        "status": status,
        "variable_text": variable_text,
        "warnings": warnings if warnings is not None else [],
        **status_payload,
    }


def _envelope(
    status: str,
    *,
    parts: list[dict[str, Any]],
    diagnostics_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The envelope wrapper: `query_echo`, `diagnostics`, and the parts list."""
    diagnostics: dict[str, Any] = {
        "engine_build": "corpus",
        "trace_id": "trace-corpus",
        "warnings": [],
    }
    if diagnostics_extra is not None:
        diagnostics.update(diagnostics_extra)
    return {
        "schema_version": 1,
        "query_echo": {"entry_path": "parsed", "extract_skipped": True},
        "diagnostics": diagnostics,
        "status": status,
        "parts": parts,
    }


def _single_part_envelope(
    status: str,
    *,
    variable_text: str | None = _BASE_SENTENCE,
    part_warnings: list[dict[str, Any]] | None = None,
    diagnostics_extra: dict[str, Any] | None = None,
    **status_payload: Any,
) -> dict[str, Any]:
    """The common case: an envelope of exactly one part."""
    part = _envelope_part(
        status, variable_text=variable_text, warnings=part_warnings, **status_payload
    )
    return _envelope(status, parts=[part], diagnostics_extra=diagnostics_extra)


def _part(sentence: str) -> dict[str, Any]:
    """The tool argument that asks for `sentence` — one entry of a `parts` list."""
    return {"variable": sentence}


# The dataclasses ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CorpusCase:
    """One question, the tool parts that ask it, and the envelope it should produce.

    `question` and `parts` are what make this a corpus of questions rather than a
    pile of envelopes: they record the tool arguments that ask each question, so a
    test can drive the real tool with a case instead of rendering its envelope
    directly. `status` is the semantic status every entry of `envelope["parts"]`
    is authored to share.
    """

    id: str
    question: str
    parts: list[dict[str, Any]]
    status: str
    envelope: dict[str, Any]


@dataclass(frozen=True, slots=True)
class StatusBudget:
    """A status's character ceiling, and (where the corpus sweeps one spec count) its marginal cost.

    `marginal` is the maximum characters a single additional spec block (interpretation
    or candidate) may add — a more sensitive instrument than the ceiling: a status can
    have room left under its ceiling while one spec block has quietly got fatter.
    """

    ceiling: int
    marginal: int | None = None


@dataclass(frozen=True, slots=True)
class FramePair:
    """Two case ids whose envelopes carry identical digest-relevant fields.

    The fat one carries extra material `render_digest` never reads (a
    `resolved.observations` payload, a deep `unsupported.definition` tree, or extra
    diagnostics), so the pair proves the digest tracks only what it
    renders, not the size of the frame behind it.
    """

    lean_id: str
    fat_id: str


@dataclass(frozen=True, slots=True)
class SentencePair:
    """Two case ids differing only by a known suffix appended to the sentence.

    `suffix` is that literal text; the digest character delta between the two cases
    should equal the suffix's own length exactly, or a grounding line is restating the
    sentence somewhere.
    """

    base_id: str
    long_id: str
    suffix: str


# definite cases ------------------------------------------------------------------
#
# A multi-part case wraps one resolved spec per part of the conjunction that asked
# for it, bounded by the tool's own `MAX_PARTS` (`data_mcp.search_tool`) — the sweep
# below is a sweep over conjunction width, up to that ceiling.


def _definite_parts(sentences: list[str], specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One definite part per (sentence, spec) pair, in order."""
    return [
        _envelope_part("definite", variable_text=sentence, spec=spec)
        for sentence, spec in zip(sentences, specs, strict=True)
    ]


def _definite_1_envelope(sentence: str) -> dict[str, Any]:
    """`definite_1`'s envelope, parameterised by sentence — shared with its sentence-pair partner."""
    return _single_part_envelope(
        "definite",
        variable_text=sentence,
        spec=_spec(
            spec_id="spec_9f2c",
            sentence=sentence,
            coverage=_EXACT_COVERAGE,
            entities=_FRANCE_TOGO_ENTITIES,
            facets=_BOUND_AND_DEFAULT_FACETS,
            facet_options=_FACET_OPTIONS_FIXTURE,
        ),
    )


_DEFINITE_1_QUESTION = (
    "How much malaria grant money did France give Togo between 2015 and 2024, in "
    "constant 2015 US dollars?"
)

# The six per-part warnings authored onto `definite_6_warned`, one per part in order.
_DEFINITE_6_WARNINGS: list[dict[str, Any]] = [
    {
        "code": "date_coverage_partial",
        "severity": "info",
        "message": "coverage starts in 2016",
    },
    {
        "code": "role_not_resolved",
        "severity": "warning",
        "message": "role read as recipient by default",
    },
    {
        "code": "unit_defaulted",
        "severity": "info",
        "message": "unit defaulted to current prices",
    },
    {
        "code": "date_coverage_partial",
        "severity": "info",
        "message": "coverage ends in 2022",
    },
    {
        "code": "source_defaulted",
        "severity": "info",
        "message": "source defaulted to broadest available",
    },
    {
        "code": "role_not_resolved",
        "severity": "warning",
        "message": "role read as provider by default",
    },
]

_DEFINITE_CASES: list[CorpusCase] = [
    CorpusCase(
        id="definite_1",
        question=_DEFINITE_1_QUESTION,
        parts=[_part(_BASE_SENTENCE)],
        status="definite",
        envelope=_definite_1_envelope(_BASE_SENTENCE),
    ),
    CorpusCase(
        id="definite_2",
        question=(
            "What are the ODA disbursements for education in Kenya, and humanitarian aid "
            "to Yemen from Germany, over their respective windows?"
        ),
        parts=[_part(_SENTENCES[1]), _part(_SENTENCES[2])],
        status="definite",
        envelope=_envelope(
            "definite",
            parts=_definite_parts(
                [_SENTENCES[1], _SENTENCES[2]],
                [
                    _spec(
                        spec_id="spec_a1",
                        sentence=_SENTENCES[1],
                        coverage=_BREADTH_COVERAGE,
                        facet_options=_FACET_OPTIONS_FIXTURE,
                    ),
                    _spec(
                        spec_id="spec_a2",
                        sentence=_SENTENCES[2],
                        coverage=_EXACT_COVERAGE,
                        facets=_FALLBACK_SOURCE_FACET,
                        facet_options=_FACET_OPTIONS_FIXTURE,
                    ),
                ],
            ),
        ),
    ),
    CorpusCase(
        id="definite_3",
        question=(
            "Give me climate adaptation grants to Bangladesh, food security funding to "
            "Mali, and water infrastructure loans to Ethiopia."
        ),
        parts=[_part(_SENTENCES[3]), _part(_SENTENCES[4]), _part(_SENTENCES[5])],
        status="definite",
        envelope=_envelope(
            "definite",
            parts=_definite_parts(
                [_SENTENCES[3], _SENTENCES[4], _SENTENCES[5]],
                [
                    _spec(
                        spec_id="spec_b1",
                        sentence=_SENTENCES[3],
                        coverage=_BARE_COVERAGE,
                        facet_options=_FACET_OPTIONS_FIXTURE,
                    ),
                    _spec(
                        spec_id="spec_b2",
                        sentence=_SENTENCES[4],
                        coverage=_EXACT_COVERAGE,
                        facets=_PER_SERIES_UNIT_FACET,
                        facet_options=_FACET_OPTIONS_FIXTURE,
                    ),
                    _spec(
                        spec_id="spec_b3",
                        sentence=_SENTENCES[5],
                        coverage=_BREADTH_COVERAGE,
                        facet_options=_FACET_OPTIONS_FIXTURE,
                    ),
                ],
            ),
        ),
    ),
    CorpusCase(
        id="definite_6",
        question="Give me all six of: " + "; ".join(_SENTENCES[:6]),
        parts=[_part(sentence) for sentence in _SENTENCES[:6]],
        status="definite",
        envelope=_envelope(
            "definite",
            parts=_definite_parts(
                _SENTENCES[:6],
                [
                    _spec(
                        spec_id=f"spec_c{index}",
                        sentence=sentence,
                        coverage=_BARE_COVERAGE,
                        facet_options=_FACET_OPTIONS_FIXTURE,
                    )
                    for index, sentence in enumerate(_SENTENCES[:6])
                ],
            ),
        ),
    ),
    CorpusCase(
        id="definite_6_warned",
        question="Give me all six of: "
        + "; ".join(_SENTENCES[:6])
        + " (with partial coverage on some).",
        parts=[_part(sentence) for sentence in _SENTENCES[:6]],
        status="definite",
        envelope=_envelope(
            "definite",
            parts=[
                _envelope_part(
                    "definite",
                    variable_text=sentence,
                    warnings=[_DEFINITE_6_WARNINGS[index]],
                    spec=_spec(
                        spec_id=f"spec_d{index}",
                        sentence=sentence,
                        coverage=_EXACT_COVERAGE,
                        facet_options=_FACET_OPTIONS_FIXTURE,
                    ),
                )
                for index, sentence in enumerate(_SENTENCES[:6])
            ],
        ),
    ),
    CorpusCase(
        id="definite_stat_vars_50",
        question=_DEFINITE_1_QUESTION,
        parts=[_part(_BASE_SENTENCE)],
        status="definite",
        envelope=_single_part_envelope(
            "definite",
            variable_text=_BASE_SENTENCE,
            spec=_spec(
                spec_id="spec_stat_vars",
                sentence=_BASE_SENTENCE,
                coverage=_EXACT_COVERAGE,
                entities=_FRANCE_TOGO_ENTITIES,
                facets=_BOUND_AND_DEFAULT_FACETS,
                stat_vars=_bulk_stat_vars(50),
            ),
        ),
    ),
    CorpusCase(
        id="definite_1_lean",
        question=_DEFINITE_1_QUESTION,
        parts=[_part(_BASE_SENTENCE)],
        status="definite",
        envelope=_single_part_envelope(
            "definite",
            variable_text=_BASE_SENTENCE,
            spec=_spec(
                spec_id="spec_frame_pair",
                sentence=_BASE_SENTENCE,
                coverage=_EXACT_COVERAGE,
                entities=_FRANCE_TOGO_ENTITIES,
                facets=_BOUND_AND_DEFAULT_FACETS,
            ),
        ),
    ),
    CorpusCase(
        id="definite_1_fat",
        question=_DEFINITE_1_QUESTION,
        parts=[_part(_BASE_SENTENCE)],
        status="definite",
        envelope=_single_part_envelope(
            "definite",
            variable_text=_BASE_SENTENCE,
            diagnostics_extra=_bulk_diagnostics(200),
            spec=_spec(
                spec_id="spec_frame_pair",
                sentence=_BASE_SENTENCE,
                coverage=_EXACT_COVERAGE,
                entities=_FRANCE_TOGO_ENTITIES,
                facets=_BOUND_AND_DEFAULT_FACETS,
                observations={
                    "series": [
                        {
                            "stat_var": {"ref": "MalariaGrantDisbursements"},
                            "entities": [],
                            "points": _bulk_observation_points(100),
                            "total_obs": 100,
                        }
                    ],
                    "total_obs": 100,
                    "truncated": False,
                },
            ),
        ),
    ),
    CorpusCase(
        id="definite_1_long_sentence",
        question=_DEFINITE_1_QUESTION + " Exclude technical assistance and admin overhead.",
        parts=[_part(_BASE_SENTENCE + _LONG_SENTENCE_SUFFIX)],
        status="definite",
        envelope=_definite_1_envelope(_BASE_SENTENCE + _LONG_SENTENCE_SUFFIX),
    ),
    CorpusCase(
        id="definite_many_axes",
        question=_DEFINITE_1_QUESTION,
        parts=[_part(_BASE_SENTENCE)],
        status="definite",
        envelope=_single_part_envelope(
            "definite",
            variable_text=_BASE_SENTENCE,
            spec=_spec(
                spec_id="spec_many_axes",
                sentence=_BASE_SENTENCE,
                coverage=_EXACT_COVERAGE,
                entities=_FRANCE_TOGO_ENTITIES,
                facets=_BOUND_AND_DEFAULT_FACETS,
                facet_options=_FACET_OPTIONS_FIVE_AXES_FIXTURE,
            ),
        ),
    ),
    CorpusCase(
        id="definite_source_codes",
        question=_DEFINITE_1_QUESTION,
        parts=[_part(_BASE_SENTENCE)],
        status="definite",
        envelope=_single_part_envelope(
            "definite",
            variable_text=_BASE_SENTENCE,
            spec=_spec(
                spec_id="spec_source_codes",
                sentence=_BASE_SENTENCE,
                coverage=_EXACT_COVERAGE,
                entities=_FRANCE_TOGO_ENTITIES,
                facets=_BOUND_AND_DEFAULT_FACETS,
                source_codes=[
                    {"provenance": "oecd_dac_2a", "code": "DAC2A.1010"},
                    {"provenance": "oecd_dac_2a", "code": "DAC2A.1020"},
                ],
            ),
        ),
    ),
]


# candidates cases ------------------------------------------------------------------


def _candidates_envelope(count: int, *, truncated: bool) -> dict[str, Any]:
    sentences = [_SENTENCES[index % len(_SENTENCES)] for index in range(count)]
    specs = [
        _spec(
            spec_id=f"spec_cand{index}",
            sentence=sentence,
            coverage=_BARE_COVERAGE if index % 2 == 0 else _BREADTH_COVERAGE,
        )
        for index, sentence in enumerate(sentences)
    ]
    return _single_part_envelope(
        "candidates",
        variable_text=sentences[0] if sentences else None,
        candidates={
            "ordering": "intent_rank",
            "max_candidates": 24,
            "specs": specs,
            "truncated": truncated,
        },
    )


_CANDIDATES_QUESTION = (
    "I asked about development finance without pinning down which reading I meant — "
    "show me the candidate interpretations."
)

_CANDIDATES_CASES: list[CorpusCase] = [
    CorpusCase(
        id=f"candidates_{count}",
        question=_CANDIDATES_QUESTION,
        parts=[_part(_SENTENCES[0])],
        status="candidates",
        envelope=_candidates_envelope(count, truncated=False),
    )
    for count in (1, 2, 3, 4, 5, 6)
]
_CANDIDATES_CASES.extend(
    [
        CorpusCase(
            id="candidates_12",
            question=_CANDIDATES_QUESTION,
            parts=[_part(_SENTENCES[0])],
            status="candidates",
            envelope=_candidates_envelope(12, truncated=False),
        ),
        CorpusCase(
            id="candidates_24",
            question=_CANDIDATES_QUESTION,
            parts=[_part(_SENTENCES[0])],
            status="candidates",
            envelope=_candidates_envelope(24, truncated=False),
        ),
        CorpusCase(
            id="candidates_6_truncated",
            question=_CANDIDATES_QUESTION,
            parts=[_part(_SENTENCES[0])],
            status="candidates",
            envelope=_candidates_envelope(6, truncated=True),
        ),
    ]
)


# skills cases ------------------------------------------------------------------------
#
# `candidates_6` with a growing number of skills attached across its six readings.
# Descriptions are authored at the length a real one runs to, some past the digest's
# one-line cut, since the skills section's size tracks them directly.

_SKILL_POOL: list[tuple[str, str]] = [
    (
        "health-financing",
        "Health financing definitions, the CRS purpose codes that carry health aid, and "
        "the deflation choices a cross-country comparison has to make.",
    ),
    (
        "oda-basics",
        "What counts as official development assistance, grant-equivalent versus cash-flow "
        "reporting, and why commitments and disbursements differ in any given year.",
    ),
    (
        "malaria-programmes",
        "Malaria control spending: which purpose codes and channels carry it, how Global "
        "Fund disbursements are attributed, and the gaps in recipient-side reporting.",
    ),
    (
        "humanitarian-flows",
        "Humanitarian assistance across OCHA FTS and the CRS, where the two disagree, and "
        "how pooled funds and multilateral core contributions are counted.",
    ),
    (
        "climate-finance",
        "Rio markers, principal versus significant objectives, and why adaptation and "
        "mitigation totals overlap and must not be summed.",
    ),
    (
        "debt-relief",
        "Debt relief in the CRS: action on debt, HIPC and MDRI, and how rescheduling "
        "shows up in ODA totals.",
    ),
    (
        "education-aid",
        "Education aid by level, the imputed student costs some donors report as ODA, and "
        "the scholarship lines that inflate totals.",
    ),
    (
        "water-sanitation",
        "WASH spending under purpose codes 140xx, and the split between large systems and "
        "basic drinking water and sanitation.",
    ),
    (
        "deflators",
        "Constant-price series: DAC deflators, base-year choice, and when to convert "
        "currency before or after deflating.",
    ),
    (
        "multilateral-channels",
        "Core versus earmarked multilateral funding, imputed multilateral shares, and "
        "double counting across donor and agency reports.",
    ),
    (
        "loans-and-grants",
        "Grant, loan, and equity instruments, concessionality thresholds, and how "
        "repayments net against gross disbursements.",
    ),
    (
        "private-flows",
        "Private flows at market terms, mobilised private finance, and why neither belongs "
        "in an ODA total.",
    ),
]


def _candidates_with_skills_envelope(skill_count: int) -> dict[str, Any]:
    """`candidates_6`'s envelope with the first `skill_count` pool skills attached.

    Skills are dealt across the six readings in pool order, a contiguous run
    to each, so the digest's first-appearance order is the pool's own and a
    longer pool only appends. The envelope's `skills` map carries each skill's
    front matter once.
    """
    envelope = _candidates_envelope(6, truncated=False)
    specs = envelope["parts"][0]["candidates"]["specs"]
    for spec in specs:
        spec["resolved"]["skills"] = []
    pool = _SKILL_POOL[:skill_count]
    for index, (skill_id, _description) in enumerate(pool):
        specs[index * len(specs) // skill_count]["resolved"]["skills"].append(skill_id)
    envelope["skills"] = {
        skill_id: {
            "id": skill_id,
            "name": skill_id,
            "description": description,
            "content_version": None,
        }
        for skill_id, description in pool
    }
    return envelope


_SKILLS_CASES: list[CorpusCase] = [
    CorpusCase(
        id=f"candidates_6_skills_{count}",
        question=_CANDIDATES_QUESTION,
        parts=[_part(_SENTENCES[0])],
        status="candidates",
        envelope=_candidates_with_skills_envelope(count),
    )
    for count in (1, 5, 6, 12)
]


# no_data cases -----------------------------------------------------------------------


def _no_data_bare_envelope(
    *, diagnostics_extra: dict[str, Any] | None = None, bulk: bool = False
) -> dict[str, Any]:
    no_data: dict[str, Any] = {"reason": "no_observations", "category": "absent"}
    if bulk:
        no_data["candidate_pool_considered"] = _bulk_stat_vars(100)
    return _single_part_envelope(
        "no_data",
        variable_text=_SENTENCES[11],
        diagnostics_extra=diagnostics_extra,
        no_data=no_data,
    )


_NO_DATA_QUESTION = "Is there any data on debt relief commitments from the Paris Club to Zambia?"

_NO_DATA_CASES: list[CorpusCase] = [
    CorpusCase(
        id="no_data_bare",
        question=_NO_DATA_QUESTION,
        parts=[_part(_SENTENCES[11])],
        status="no_data",
        envelope=_no_data_bare_envelope(),
    ),
    CorpusCase(
        id="no_data_unresolved",
        question="Is there data on funding from Elbonia to Freedonia, 2015 to 2020?",
        parts=[_part("funding from Elbonia to Freedonia, 2015 to 2020")],
        status="no_data",
        envelope=_single_part_envelope(
            "no_data",
            variable_text="funding from Elbonia to Freedonia, 2015 to 2020",
            no_data={"reason": "entity_not_resolved", "category": "unresolved"},
        ),
    ),
    CorpusCase(
        id="no_data_nearest_5",
        question=_NO_DATA_QUESTION,
        parts=[_part(_SENTENCES[11])],
        status="no_data",
        envelope=_single_part_envelope(
            "no_data",
            variable_text=_SENTENCES[11],
            no_data={
                "reason": "no_observations",
                "category": "absent",
                "nearest_real": [
                    {
                        **_spec(spec_id=f"spec_near_{axis}", sentence=_SENTENCES[11]),
                        "relaxation": {"axis": axis, "dropped_value_dcids": []},
                    }
                    for axis in ("where", "when", "unit", "source", "provenance")
                ],
            },
        ),
    ),
    CorpusCase(
        id="no_data_flagged",
        question=_NO_DATA_QUESTION,
        parts=[_part(_SENTENCES[11])],
        status="no_data",
        envelope=_single_part_envelope(
            "no_data",
            variable_text=_SENTENCES[11],
            no_data={
                "reason": "no_observations",
                "category": "absent",
                "nearest_real": [
                    {
                        **_spec(spec_id="spec_near_where", sentence=_SENTENCES[11]),
                        "relaxation": {"axis": "where", "dropped_value_dcids": []},
                    },
                    {
                        **_spec(spec_id="spec_near_when", sentence=_SENTENCES[11]),
                        "relaxation": {"axis": "when", "dropped_value_dcids": []},
                    },
                ],
                "flagged_match": _spec(spec_id="spec_flagged", sentence=_SENTENCES[11]),
                "gate": "min_countries_5",
            },
        ),
    ),
    CorpusCase(
        id="no_data_bare_fat",
        question=_NO_DATA_QUESTION,
        parts=[_part(_SENTENCES[11])],
        status="no_data",
        envelope=_no_data_bare_envelope(diagnostics_extra=_bulk_diagnostics(200), bulk=True),
    ),
    CorpusCase(
        id="no_data_elected_stat_vars",
        question=_NO_DATA_QUESTION,
        parts=[_part(_SENTENCES[11])],
        status="no_data",
        envelope=_single_part_envelope(
            "no_data",
            variable_text=_SENTENCES[11],
            no_data={
                "reason": "no_observations",
                "category": "absent",
                "elected_stat_vars": [
                    {"dcid": "DebtReliefCommitments", "label": "debt relief commitments"}
                ],
            },
        ),
    ),
]


# unsupported cases -------------------------------------------------------------------

_UNSUPPORTED_DEFINITION: dict[str, Any] = {
    "identity": {
        "facets": {"provenance": {"state": "value", "ref": "x", "label": "a bound source facet"}}
    }
}
_UNSUPPORTED_LONG_DETAIL = (
    "the provenance axis cannot be bound from free text once a facet has been resolved to a "
    "specific value upstream; ask without naming a source, or drop the source phrase entirely "
    "and let the door pick the broadest available series"
)


def _unsupported_envelope(
    *, variable_text: str, detail: str, definition: dict[str, Any]
) -> dict[str, Any]:
    return _single_part_envelope(
        "unsupported",
        variable_text=variable_text,
        unsupported={
            "code": "facet_binding_unsupported",
            "pointer": "/identity/facets/provenance/label",
            "detail": detail,
            "definition": definition,
        },
    )


_UNSUPPORTED_CASES: list[CorpusCase] = [
    CorpusCase(
        id="unsupported_short",
        question="Can you bind the source to OECD DAC directly in the query text?",
        parts=[_part("malaria grants from France to Togo, source OECD DAC")],
        status="unsupported",
        envelope=_unsupported_envelope(
            variable_text="malaria grants from France to Togo, source OECD DAC",
            detail="not supported",
            definition={"identity": {"facets": {"provenance": {"label": "a bound source facet"}}}},
        ),
    ),
    CorpusCase(
        id="unsupported_long_detail",
        question=(
            "Can you bind the source facet to a specific table directly from free text, once "
            "it's already been resolved upstream?"
        ),
        parts=[_part("malaria grants from France to Togo, source OECD DAC table 2a")],
        status="unsupported",
        envelope=_unsupported_envelope(
            variable_text="malaria grants from France to Togo, source OECD DAC table 2a",
            detail=_UNSUPPORTED_LONG_DETAIL,
            definition=_UNSUPPORTED_DEFINITION,
        ),
    ),
    CorpusCase(
        id="unsupported_fat_definition",
        question=(
            "Can you bind the source facet to a specific table directly from free text, once "
            "it's already been resolved upstream?"
        ),
        parts=[_part("malaria grants from France to Togo, source OECD DAC table 2a")],
        status="unsupported",
        envelope=_unsupported_envelope(
            variable_text="malaria grants from France to Togo, source OECD DAC table 2a",
            detail=_UNSUPPORTED_LONG_DETAIL,
            definition={
                **_UNSUPPORTED_DEFINITION,
                "diagnostics_bulk": _bulk_definition_tree(width=4, depth=4),
            },
        ),
    ),
    CorpusCase(
        id="unsupported_unit_axis",
        question=(
            "Can you give me this in PPP terms even though this deployment doesn't carry "
            "that basis?"
        ),
        parts=[_part("renewable energy project financing from the EU to Senegal, in PPP terms")],
        status="unsupported",
        envelope=_single_part_envelope(
            "unsupported",
            variable_text="renewable energy project financing from the EU to Senegal, in PPP terms",
            unsupported={
                "code": "facet_binding_unsupported",
                "pointer": "/identity/facets/unit",
                "detail": (
                    "the requested unit's currency/prices/base_year combination is not "
                    "expressed by any unit this deployment's data carries"
                ),
                "definition": {"identity": {}},
            },
        ),
    ),
    CorpusCase(
        id="unsupported_warnings_filtered",
        question=(
            "Can you bind the source facet to a specific table directly from free text, once "
            "it's already been resolved upstream?"
        ),
        parts=[_part("malaria grants from France to Togo, source OECD DAC table 2a")],
        status="unsupported",
        envelope=_single_part_envelope(
            "unsupported",
            variable_text="malaria grants from France to Togo, source OECD DAC table 2a",
            part_warnings=[
                {
                    "code": "near_neighbor_ambiguous",
                    "severity": "info",
                    "message": (
                        "an earlier variable-recall step compared two close candidate readings"
                    ),
                },
                {
                    "code": "default_not_declared",
                    "severity": "info",
                    "message": "the unit fell back to a deployment default",
                },
            ],
            unsupported={
                "code": "facet_binding_unsupported",
                "pointer": "/identity/facets/provenance/label",
                "detail": _UNSUPPORTED_LONG_DETAIL,
                "definition": _UNSUPPORTED_DEFINITION,
            },
        ),
    ),
]


# The corpus ----------------------------------------------------------------------

CASES: list[CorpusCase] = [
    *_DEFINITE_CASES,
    *_CANDIDATES_CASES,
    *_SKILLS_CASES,
    *_NO_DATA_CASES,
    *_UNSUPPORTED_CASES,
]

FRAME_PAIRS: list[FramePair] = [
    FramePair(lean_id="definite_1_lean", fat_id="definite_1_fat"),
    FramePair(lean_id="no_data_bare", fat_id="no_data_bare_fat"),
    FramePair(lean_id="unsupported_long_detail", fat_id="unsupported_fat_definition"),
]

SENTENCE_PAIRS: list[SentencePair] = [
    SentencePair(
        base_id="definite_1", long_id="definite_1_long_sentence", suffix=_LONG_SENTENCE_SUFFIX
    )
]

# Per-status character budgets. The rule, not the numbers, is what a future change to
# a budget should be an argument about: for each status, take the largest digest this
# corpus's own factories produce for it (measured with `len(digest)` — see
# `test_digest_budget.py`), multiply by 1.4, and round up to the next 100. Measured
# against this corpus:
#
#   definite:    4017 (definite_6_warned)                         -> 5700
#   candidates:  1883 (candidates_6_skills_12, skills capped)     -> 2700
#   no_data:      407 (no_data_flagged)                           -> 600
#   unsupported:  330 (unsupported_warnings_filtered)             -> 500
#
# Every definite spec above carries a facet_options fixture, since the tool
# always requests option coverage and every definite spec a real response
# returns carries one. Each facet options line also carries every value's
# dcid alongside its label, so `get_data`'s facets scope has a ref it can
# actually bind.
#
# The marginal budgets are a second, more sensitive instrument: the maximum
# characters a single additional spec block may cost on average, catching a spec
# block that has quietly got fatter while a status's ceiling still has headroom.
# Measured worst is 781 chars/interpretation (definite_many_axes, one spec,
# the largest because it carries five facet_options axes) and 225
# chars/candidate (candidates_1, one spec). Multiplied by 1.4 and rounded up
# to the next 50, that's 1100 and 350.
BUDGETS: dict[str, StatusBudget] = {
    "definite": StatusBudget(ceiling=5700, marginal=1100),
    "candidates": StatusBudget(ceiling=2700, marginal=350),
    "no_data": StatusBudget(ceiling=600),
    "unsupported": StatusBudget(ceiling=500),
}

# The skills section's own marginal budget: the characters one more described skill
# may add. Measured at 166 chars/skill across `candidates_6_skills_1` ->
# `candidates_6_skills_5` (descriptions of 126 to 156 characters). Multiplied by 1.4
# and rounded up to the next 50, that's 250. A description past the digest's
# 160-character cut costs no more than one at it.
SKILL_MARGINAL = 250

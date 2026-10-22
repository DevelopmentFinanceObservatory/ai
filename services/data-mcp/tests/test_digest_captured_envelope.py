"""A real `/api/search/resolve` envelope, pinned against a captured fixture.

Every other digest test (`test_digest.py`, `digest_corpus.py`) builds its envelopes by
hand, so none of them would notice apis changing the resolve door's wire shape.
This file reads a genuine response instead.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from data_mcp.digest import render_digest

_FIXTURES = Path(__file__).parent / "fixtures"


def _load_captured_envelope(name: str = "resolve_population_guatemala.json") -> dict[str, Any]:
    return json.loads((_FIXTURES / name).read_text())


def test_captured_envelope_renders_to_the_pinned_digest() -> None:
    """`resolve_population_guatemala.json` is a real `/api/search/resolve` response,
    captured against a checkout of the apis repository at commit `ed164c1e`, the
    commit the dev deployment runs, for the parsed-door query "population for
    Guatemala since 2010" (issue 212's own query). This is the same round trip
    that produced `spec_ed5b608a610cdc5a4bc703cf7d1bc2d0`, the spec_id the issue
    reports an unpinned mint refusing on `over_ceiling`.

    The fixture is converted to the wire without a member list: `resolved.stat_vars`
    carries the counts, `served_count` and the spec-level `data_date_range` the
    recorded item stated.

    `render_digest` states the measure and its StatVar count beside the
    spec_id, naming no member. The rendered-digest assertion below
    pins what the digest produces from this envelope, so a later change to
    sentence, coverage, StatVar or facet-option rendering shows up here as a
    diff against real bytes, not just against a hand-authored corpus case.
    """
    envelope = _load_captured_envelope()

    part = envelope["parts"][0]
    assert part["status"] == "definite"
    spec = part["spec"]
    assert spec["spec_id"] == "spec_ed5b608a610cdc5a4bc703cf7d1bc2d0"

    resolved = spec["resolved"]
    stat_vars = resolved["stat_vars"]
    assert stat_vars == {
        "member_count": 1,
        "member_count_exact": True,
        "served_count": 1,
        "data_date_range": {"start": "1960", "end": "2024"},
    }

    rendered = render_digest(envelope)
    assert rendered == (
        '"Total population in Guatemala, since 2010, according to World Development '
        'Indicators, Wikidata Population, and Subnational Demographics Stats."  '
        "[spec_ed5b608a610cdc5a4bc703cf7d1bc2d0]\n"
        "confidence high · 27 observations, since 2010 · has data\n"
        "Measure: Total population · 1 StatVar.\n"
        "observationAbout Guatemala (country/GTM)\n"
        "provenance options for get_data: dc/base/WorldDevelopmentIndicators "
        "(WorldDevelopmentIndicators), dc/base/Subnational_Demographics_Stats "
        "(Subnational_Demographics_Stats), dc/base/WikidataPopulation "
        "(WikidataPopulation).\n"
        "measurementMethod options for get_data: WorldBankSubnationalPopulationEstimate "
        "(World Bank Subnational Population Estimates.), WikidataPopulation "
        "(WikidataPopulation)."
    )


def test_captured_aid_envelope_names_the_elected_measure_and_its_sibling() -> None:
    """`resolve_aid_france_kenya.json` is a real dev resolve response whose spec carries
    `measure_label` and one `alternative_measures` entry (its `comment` says how it was
    recorded). The digest names the elected measure and the sibling's own spec_id, so a
    model can offer net disbursements and fetch it.
    """
    envelope = _load_captured_envelope("resolve_aid_france_kenya.json")

    assert render_digest(envelope) == (
        "\"France's gross disbursements to Kenya for official development assistance, "
        '1978 to 2024. Current US dollars · OECD DAC2a."  '
        "[spec_85bd3b844a3a080e524ee49dfd344bba]\n"
        "confidence medium · 47 observations · has data\n"
        "Measure: gross disbursements (official development assistance) · 1 StatVar.\n"
        "Counted as gross disbursements; other counts: net disbursements "
        "[spec_c67524ce5da3bfd4d94994a76c560b28].\n"
        "destination Kenya (country/KEN) · origin France (country/FRA)\n"
        "source OECD DAC2A (fallback) · unit US Dollar (fallback)\n"
        "provenance options for get_data: provenance/OECDDAC2A (OECD DAC2A).\n"
        "unit options for get_data: CurrencyUSD (US Dollar).\n\n"
        "No aid type was named, so the answer covers official development assistance.\n\n"
        "no time_text was given; the answer was resolved across every period with data.\n\n"
        "Skills for these results; get_skill loads one by id:\n"
        "- oda-dac2a: Context for working with the OECD DAC2A table, ODA disbursements by "
        "donor and recipient country or region. Covers its measures, multilateral agencies "
        "reporting…"
    )

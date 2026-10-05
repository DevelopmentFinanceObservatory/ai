"""A real `/api/search/resolve` envelope, pinned against a captured fixture.

Every other digest test (`test_digest.py`, `digest_corpus.py`) builds its envelopes by
hand, so none of them would notice dfo-apis changing the resolve door's wire shape.
This file reads a genuine response instead.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from data_mcp.digest import render_digest

_FIXTURES = Path(__file__).parent / "fixtures"


def _load_captured_envelope() -> dict[str, Any]:
    return json.loads((_FIXTURES / "resolve_population_guatemala.json").read_text())


def test_captured_envelope_renders_to_the_pinned_digest() -> None:
    """`resolve_population_guatemala.json` is a real `/api/search/resolve` response,
    captured 2026-09-17 against a checkout of dfo-apis at commit `ed164c1e` -- the
    commit the dev deployment runs -- for the parsed-door query "population for
    Guatemala since 2010" (issue 212's own query). This is the same round trip
    that produced `spec_ed5b608a610cdc5a4bc703cf7d1bc2d0`, the spec_id the issue
    reports an unpinned mint refusing on `over_ceiling`.

    dfo-apis#893 will change what `resolved.stat_vars.member_count` and
    `resolved.stat_vars.truncated` mean for this query: today a non-set-bound
    standard shape caps `member_count` at 1 via an early `break` and `truncated`
    can never fire on that path, so this fixture carries `member_count: 1` and
    `truncated: false`. After #893 lands, the same query returns the number of
    candidate members confirmed to have data at the place, and `truncated`
    reflects a real cap. `resolved.stat_vars.items`, the field names, and the
    nesting are unaffected -- only what the two numbers mean changes. This
    fixture needs re-recording once #893 deploys; until then this test's
    `member_count`/`truncated` assertions pin the pre-#893 shape deliberately.
    The item's `constraint_properties` postdates the capture and is written in by
    hand: `Count_Person` asserts no constraint, so Search serves an empty list.

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
    assert stat_vars["items"] == [
        {
            "ref": "Count_Person",
            "label": "Total population",
            "slot_values": [],
            "constraint_properties": [],
            "data_date_range": {"start": "1960", "end": "2024"},
        }
    ]
    # Pre-#893 shape -- see the docstring above for what changes and what doesn't.
    assert stat_vars["member_count"] == 1
    assert stat_vars["truncated"] is False

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

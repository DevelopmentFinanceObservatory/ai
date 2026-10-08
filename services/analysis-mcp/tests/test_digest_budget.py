"""The digest budget: a committed baseline, a size ceiling, and two named
size-regression checks -- scaled down from `data_mcp.tests.test_digest_budget`'s
own three-gate shape to what `render_aggregate_digest`'s single, undiscriminated
digest shape needs (no status/candidate-sweep machinery, since `aggregate` has
neither).

Failure messages naming `just analysis-digest-baseline` mean a deliberate
digest change, not a regression: the recipe re-measures the corpus and
rewrites `digest_baseline.json`, and the diff of that regeneration is the
review artifact.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
from digest_corpus import (
    CASES,
    CEILING,
    COMPARE_CASES,
    FILTER_CASES,
    FRAME_PAIR_CASES,
    FRAME_PAIRS,
    GROWTH_CASES,
    JOIN_CASES,
    QUOTIENT_OF_CASES,
    RANK_CASES,
    SHARE_OF_CASES,
    SUM_OF_CASES,
)

# The root conftest.py chdirs every test into a tmp_path, so this must be absolute.
_BASELINE_PATH = Path(__file__).resolve().parent / "digest_baseline.json"
_REGENERATE = "just analysis-digest-baseline"

_ALL_CASES = (
    tuple(CASES)
    + tuple(FRAME_PAIR_CASES)
    + tuple(GROWTH_CASES)
    + tuple(COMPARE_CASES)
    + tuple(RANK_CASES)
    + tuple(SHARE_OF_CASES)
    + tuple(SUM_OF_CASES)
    + tuple(QUOTIENT_OF_CASES)
    + tuple(JOIN_CASES)
    + tuple(FILTER_CASES)
)


def _measure(case) -> dict[str, Any]:
    digest = case.render(**case.kwargs)
    return {"id": case.id, "digest_chars": len(digest)}


@pytest.fixture(scope="module")
def measurements() -> dict[str, dict[str, Any]]:
    return {case.id: _measure(case) for case in _ALL_CASES}


@pytest.fixture(scope="module", autouse=True)
def _regenerate_baseline_if_requested(measurements: dict[str, dict[str, Any]]) -> None:
    """Rewrite the committed baseline when `DIGEST_BASELINE_UPDATE=1`, following
    `data_mcp`'s own precedent for a manually-invoked write (the `just
    analysis-digest-baseline` recipe)."""
    if os.environ.get("DIGEST_BASELINE_UPDATE") == "1":
        payload = {"cases": sorted(measurements.values(), key=lambda case: case["id"])}
        _BASELINE_PATH.write_text(json.dumps(payload, indent=2) + "\n")


@pytest.fixture(scope="module")
def baseline(_regenerate_baseline_if_requested: None) -> dict[str, Any]:
    return json.loads(_BASELINE_PATH.read_text())


@pytest.fixture(scope="module")
def baseline_by_id(baseline: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {case["id"]: case for case in baseline["cases"]}


# Gate 1: corpus and baseline ids match exactly, both ways ------------------------


def test_corpus_and_baseline_ids_match_exactly(
    measurements: dict[str, dict[str, Any]], baseline_by_id: dict[str, dict[str, Any]]
) -> None:
    corpus_ids = set(measurements)
    baseline_ids = set(baseline_by_id)
    missing_from_baseline = sorted(corpus_ids - baseline_ids)
    missing_from_corpus = sorted(baseline_ids - corpus_ids)
    assert not missing_from_baseline, (
        f"corpus cases with no baseline row: {missing_from_baseline}. "
        f"Regenerate the baseline: {_REGENERATE}"
    )
    assert not missing_from_corpus, (
        f"baseline rows with no corresponding corpus case: {missing_from_corpus}. "
        f"Regenerate the baseline: {_REGENERATE}"
    )


# Gate 2: every case's digest stays under the ceiling ------------------------------


@pytest.mark.parametrize(
    "case_id", [case.id for case in _ALL_CASES], ids=[case.id for case in _ALL_CASES]
)
def test_digest_chars_stay_within_the_ceiling(
    case_id: str, measurements: dict[str, dict[str, Any]]
) -> None:
    measured = measurements[case_id]
    assert measured["digest_chars"] <= CEILING, (
        f"{case_id}: {measured['digest_chars']} chars exceeds the ceiling of {CEILING}. "
        f"Regenerate the baseline to see the current numbers ({_REGENERATE}); if the growth "
        "is deliberate, raise CEILING in digest_corpus.py, otherwise this is a size regression."
    )


@pytest.mark.parametrize(
    "case_id", [case.id for case in _ALL_CASES], ids=[case.id for case in _ALL_CASES]
)
def test_measurements_match_the_committed_baseline(
    case_id: str, measurements: dict[str, dict[str, Any]], baseline_by_id: dict[str, dict[str, Any]]
) -> None:
    measured = measurements[case_id]
    expected = baseline_by_id[case_id]
    assert measured["digest_chars"] == expected["digest_chars"], (
        f"{case_id}: digest size drifted from the committed baseline "
        f"({expected['digest_chars']} -> {measured['digest_chars']} chars). "
        f"Regenerate the baseline if this is deliberate: {_REGENERATE}"
    )


# Gate 3: the two named regressions ------------------------------------------------


@pytest.mark.parametrize(
    "pair", FRAME_PAIRS, ids=[f"{pair.lean_id}_vs_{pair.fat_id}" for pair in FRAME_PAIRS]
)
def test_frame_pairs_hold_digest_size_steady_while_the_frame_grows(
    pair, measurements: dict[str, dict[str, Any]]
) -> None:
    """A digest whose length tracks the size of `AggregateResult.frame` is a
    leak: `render_aggregate_digest` must never read it."""
    lean, fat = measurements[pair.lean_id], measurements[pair.fat_id]
    assert lean["digest_chars"] == fat["digest_chars"], (
        f"{pair.lean_id!r} and {pair.fat_id!r} carry the same group_summaries behind "
        f"wildly different frames, but rendered {lean['digest_chars']} vs "
        f"{fat['digest_chars']} chars -- the digest is leaking frame material it should "
        f"never read. Regenerate the baseline if you're unsure this is a regression: "
        f"{_REGENERATE}"
    )


def test_content_past_the_group_cap_costs_at_most_the_headers_own_digit_growth() -> None:
    """A special case, not the general property (see the next test for
    that): when the magnitude-picked 20 entries happen to carry the same
    digit width at every count, content past the cap really does cost
    nothing extra, and the two digests differ only by the header's own
    group-count digits -- here, not at all, since 25 and 30 are both two
    digits. `digest_corpus.py`'s own `capped_25_groups`/`capped_30_groups`
    cases use a shared 3-digit value band (`offset=100`) specifically to
    hold the picked entries' own digit width constant and isolate this
    case.
    """
    twenty_five_case = next(case for case in CASES if case.id == "capped_25_groups")
    thirty_case = next(case for case in CASES if case.id == "capped_30_groups")
    twenty_five = twenty_five_case.render(**twenty_five_case.kwargs)
    thirty = thirty_case.render(**thirty_case.kwargs)
    assert len(twenty_five) == len(thirty), (
        "a 30-group result should cost the same digest size as a 25-group result once "
        f"both exceed the cap, when the picked entries' own digit width is held constant -- "
        f"got {len(twenty_five)} vs {len(thirty)} chars. Regenerate the baseline: {_REGENERATE}"
    )


def test_content_past_the_group_cap_grows_by_a_bounded_amount() -> None:
    """The general property `_capped`'s magnitude selection actually gives,
    for `digest_corpus.py`'s own `wide_band_*_groups` cases (no digit-width
    trick -- real ascending values from `offset=0`): as `count` grows from
    25 to 1000 to 100000, the top-20-by-magnitude entries `_capped` picks
    carry more digits each time (5..24, then 980..999, then 99980..99999),
    so the rendered digest is NOT the same size at every count the way the
    previous test's held-constant-width pair is. `_truncation_note`'s own
    fixed text costs nothing extra regardless of count -- this growth comes
    from which entries survive the cap and how wide they are, not from the
    note. The real guarantee is a *bounded* cost, not an *exactly zero*
    one: growth stays small relative to `CEILING` even across a 4000x count
    increase, which is what actually protects the budget.
    """
    small = next(case for case in CASES if case.id == "wide_band_25_groups")
    medium = next(case for case in CASES if case.id == "wide_band_1000_groups")
    large = next(case for case in CASES if case.id == "wide_band_100000_groups")
    small_chars = len(small.render(**small.kwargs))
    medium_chars = len(medium.render(**medium.kwargs))
    large_chars = len(large.render(**large.kwargs))

    assert small_chars < medium_chars < large_chars, (
        "content past the cap should grow, not stay flat, as the candidate pool widens "
        f"-- got {small_chars}, {medium_chars}, {large_chars} chars for 25/1000/100000 "
        f"groups. Regenerate the baseline if this is a deliberate change: {_REGENERATE}"
    )
    # Generous relative to the measured growth (117 chars, 1158 -> 1275,
    # across a 4000x count increase from 25 to 100000 groups): this catches
    # a real leak (an uncapped loop, a truncation note that starts
    # repeating per-entry), not the legitimate few-character-per-digit
    # growth magnitude selection causes.
    bound = 500
    assert large_chars - small_chars <= bound, (
        f"content past the cap grew by {large_chars - small_chars} chars from 25 to 100000 "
        f"groups, over the {bound}-char bound this test allows for legitimate digit-width "
        f"growth -- looks like a real leak, not the expected bounded cost. Regenerate the "
        f"baseline if you're sure this is deliberate: {_REGENERATE}"
    )
    assert large_chars <= CEILING, (
        f"{large_chars} chars exceeds the ceiling of {CEILING} even with the bounded-cost "
        "property holding -- CEILING itself needs revisiting."
    )

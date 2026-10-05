"""Budgets, the committed baseline, and the four named digest-size regressions.

Three fail-closed gates, all against the corpus in `digest_corpus.py`:

1. every corpus case id has a baseline row and every baseline row has a corpus case;
2. each case's `digest_chars` is within its status's ceiling, and (for the two
   statuses that sweep a spec count) the average cost per rendered spec block is
   within the status's marginal budget;
3. four named digest-size regressions: a digest that leaks frame size, a
   candidate list that grows without a cap, a skills list that grows without a
   cap, and a grounding ref that restates its sentence.

Failure messages that name `just digest-baseline` mean a deliberate digest change,
not a regression: the recipe re-measures the corpus and rewrites
`digest_baseline.json`, and the diff of that regeneration is the review artifact.

`no_network` is imported for its autouse effect: every test in this module refuses
outbound network sockets, matching the "no network call" claim the corpus makes for
itself.
"""

from __future__ import annotations

import json
import os
import re
from itertools import pairwise
from pathlib import Path
from typing import Any

import pytest
from data_mcp.digest import render_digest

from .digest_corpus import (
    BUDGETS,
    CASES,
    FRAME_PAIRS,
    SENTENCE_PAIRS,
    SKILL_MARGINAL,
    CorpusCase,
    FramePair,
    SentencePair,
)
from .net_guard import no_network  # noqa: F401 - autouse effect

# The root conftest.py chdirs every test into a tmp_path, so this must be absolute.
_BASELINE_PATH = Path(__file__).resolve().parent / "digest_baseline.json"

_REGENERATE = "just digest-baseline"

# Mirrors the literal truncation line `_render_candidates` appends in digest.py, plus
# the "\n\n" section separator `render_digest` joins with. Not imported from
# digest.py: the point of the candidate-sweep regression below is to notice if that
# literal ever moves, which an import of the same constant would hide.
_CANDIDATE_TRUNCATION_SUFFIX = "\n\nMore candidates exist than are shown here."

# Mirrors the skills section's overflow line in digest.py, with the "\n" its lines
# join with, for the same reason.
_SKILL_OVERFLOW_ONE = "\nand 1 more skill."
_SKILL_OVERFLOW_SEVEN = "\nand 7 more skills."


# Measurement -------------------------------------------------------------------


def _offered_specs(case: CorpusCase) -> list[dict[str, Any]]:
    """The spec blocks `case`'s envelope offers, per the corpus's own accounting rule.

    One spec per definite part, `len(candidates.specs)` for candidates (always one
    part in this corpus), `len(no_data.nearest_real)` for no_data (ditto), and none
    for unsupported.
    """
    parts = case.envelope.get("parts") or []
    if case.status == "definite":
        return [part["spec"] for part in parts if "spec" in part]
    if case.status == "candidates":
        return list((parts[0].get("candidates") or {}).get("specs") or [])
    if case.status == "no_data":
        return list((parts[0].get("no_data") or {}).get("nearest_real") or [])
    return []


def _rendered_specs(case: CorpusCase, digest: str) -> list[dict[str, Any]]:
    """The subset of `_offered_specs(case)` that actually printed, by marker search."""
    return [spec for spec in _offered_specs(case) if f"[{spec.get('spec_id', '')}]" in digest]


def _measure(case: CorpusCase) -> dict[str, Any]:
    digest = render_digest(case.envelope)
    return {
        "id": case.id,
        "status": case.status,
        "spec_count": len(_offered_specs(case)),
        "rendered_spec_count": len(_rendered_specs(case, digest)),
        "digest_chars": len(digest),
        "envelope_chars": len(json.dumps(case.envelope)),
    }


def _write_baseline(measurements: dict[str, dict[str, Any]]) -> None:
    payload = {"cases": sorted(measurements.values(), key=lambda case: case["id"])}
    # `indent=2` plus a trailing newline matches the `pretty-format-json` pre-commit
    # hook's own formatting (`--indent=2 --no-sort-keys`), so it leaves this file alone.
    _BASELINE_PATH.write_text(json.dumps(payload, indent=2) + "\n")


@pytest.fixture(scope="module")
def measurements() -> dict[str, dict[str, Any]]:
    return {case.id: _measure(case) for case in CASES}


@pytest.fixture(scope="module", autouse=True)
def _regenerate_baseline_if_requested(measurements: dict[str, dict[str, Any]]) -> None:
    """Rewrite the committed baseline when `DIGEST_BASELINE_UPDATE=1`.

    Env-gated, following `test_contract_integration.py`'s precedent for a
    manually-invoked write. Wrapped by the `digest-baseline` `just` recipe. Runs
    before `baseline` below reads the file, so gate 1 still exercises the
    freshly-written ids rather than silently trusting them.
    """
    if os.environ.get("DIGEST_BASELINE_UPDATE") == "1":
        _write_baseline(measurements)


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


# Gate 2: status ceilings and marginal-per-block budgets --------------------------


@pytest.mark.parametrize("case_id", [case.id for case in CASES], ids=[case.id for case in CASES])
def test_digest_chars_stay_within_the_status_ceiling(
    case_id: str, measurements: dict[str, dict[str, Any]]
) -> None:
    measured = measurements[case_id]
    budget = BUDGETS[measured["status"]]
    assert measured["digest_chars"] <= budget.ceiling, (
        f"{case_id}: {measured['digest_chars']} chars exceeds the {measured['status']!r} "
        f"ceiling of {budget.ceiling}. Regenerate the baseline to see the current numbers "
        f"({_REGENERATE}); if the growth is deliberate, recompute BUDGETS in "
        f"digest_corpus.py, otherwise this is a size regression."
    )


@pytest.mark.parametrize("case_id", [case.id for case in CASES], ids=[case.id for case in CASES])
def test_average_cost_per_rendered_spec_block_stays_within_the_marginal_budget(
    case_id: str, measurements: dict[str, dict[str, Any]]
) -> None:
    """The more sensitive gate: `digest_chars / rendered_spec_count`.

    A status can have room left under its ceiling while one spec block has quietly
    got fatter; this catches that even when the ceiling wouldn't. The candidate-sweep
    regression below additionally checks the sharper, incremental cost of one more
    candidate. This check is the coarser one applied uniformly to every case.
    """
    measured = measurements[case_id]
    budget = BUDGETS[measured["status"]]
    if budget.marginal is None or measured["rendered_spec_count"] == 0:
        pytest.skip(f"{measured['status']!r} carries no marginal budget, or nothing rendered")
    average_cost = measured["digest_chars"] / measured["rendered_spec_count"]
    assert average_cost <= budget.marginal, (
        f"{case_id}: {average_cost:.1f} chars/spec-block exceeds the "
        f"{measured['status']!r} marginal budget of {budget.marginal}. Regenerate the "
        f"baseline to see the current numbers: {_REGENERATE}"
    )


# Gate 3: the three named regressions ---------------------------------------------


@pytest.mark.parametrize(
    "pair", FRAME_PAIRS, ids=[f"{pair.lean_id}_vs_{pair.fat_id}" for pair in FRAME_PAIRS]
)
def test_frame_pairs_hold_digest_size_steady_while_the_frame_grows(
    pair: FramePair, measurements: dict[str, dict[str, Any]]
) -> None:
    """A digest whose length tracks the size of the frame behind it is a leak.

    Nothing a frame pair's fat envelope adds is ever read by `render_digest`, so
    `digest_chars` must be exactly equal across the pair while `envelope_chars`
    grows at least tenfold.
    """
    lean, fat = measurements[pair.lean_id], measurements[pair.fat_id]
    assert lean["digest_chars"] == fat["digest_chars"], (
        f"{pair.lean_id!r} and {pair.fat_id!r} should render the same digest size, "
        f"but got {lean['digest_chars']} vs {fat['digest_chars']} chars — the "
        f"digest is leaking frame material it should never read. Compare against "
        f"{_REGENERATE}'s fresh numbers if you're unsure this is a regression."
    )
    assert fat["envelope_chars"] >= 10 * lean["envelope_chars"], (
        f"{pair.fat_id!r}'s envelope ({fat['envelope_chars']} chars) should be at "
        f"least tenfold {pair.lean_id!r}'s ({lean['envelope_chars']} chars) for the "
        f"pair to actually exercise this regression."
    )


_CANDIDATE_SWEEP_COUNTS = (1, 2, 3, 4, 5, 6)


def test_candidate_sweep_never_renders_more_than_the_cap() -> None:
    for case in CASES:
        if case.status != "candidates":
            continue
        digest = render_digest(case.envelope)
        rendered = len(_rendered_specs(case, digest))
        assert rendered <= 6, (
            f"{case.id}: rendered {rendered} candidates, cap is 6. Regenerate the "
            f"baseline to see the current numbers: {_REGENERATE}"
        )


def test_candidate_sweep_marginal_cost_stays_under_the_marginal_budget(
    measurements: dict[str, dict[str, Any]],
) -> None:
    budget = BUDGETS["candidates"]
    assert budget.marginal is not None
    sweep = [measurements[f"candidates_{count}"] for count in _CANDIDATE_SWEEP_COUNTS]
    for previous, current in pairwise(sweep):
        marginal_cost = current["digest_chars"] - previous["digest_chars"]
        assert marginal_cost <= budget.marginal, (
            f"candidates_{previous['spec_count']} -> candidates_{current['spec_count']}: "
            f"+{marginal_cost} chars for one more candidate exceeds the marginal "
            f"budget of {budget.marginal}. Regenerate the baseline to see the current "
            f"numbers: {_REGENERATE}"
        )


def test_candidate_sweep_past_the_cap_costs_exactly_the_truncation_line() -> None:
    six = render_digest(next(case for case in CASES if case.id == "candidates_6").envelope)
    twelve = render_digest(next(case for case in CASES if case.id == "candidates_12").envelope)
    expected = len(six) + len(_CANDIDATE_TRUNCATION_SUFFIX)
    assert len(twelve) == expected, (
        f"candidates_12 should cost exactly candidates_6 plus the truncation line "
        f"({expected} chars). Regenerate the baseline: {_REGENERATE}"
    )


def test_candidate_sweep_past_twelve_costs_nothing_more() -> None:
    cases_by_id = {case.id: case for case in CASES}
    twelve = render_digest(cases_by_id["candidates_12"].envelope)
    twenty_four = render_digest(cases_by_id["candidates_24"].envelope)
    assert twelve == twenty_four, (
        "candidates_24 should render byte-for-byte the same as candidates_12 — the "
        "cap already bounds what's shown, so a longer offered list past the cap must "
        f"cost nothing more. Regenerate the baseline: {_REGENERATE}"
    )


def test_skills_marginal_cost_stays_under_the_skill_budget(
    measurements: dict[str, dict[str, Any]],
) -> None:
    one = measurements["candidates_6_skills_1"]["digest_chars"]
    five = measurements["candidates_6_skills_5"]["digest_chars"]
    per_skill = (five - one) / 4
    assert per_skill <= SKILL_MARGINAL, (
        f"candidates_6_skills_1 -> candidates_6_skills_5: {per_skill:.1f} chars per "
        f"described skill exceeds the skill budget of {SKILL_MARGINAL}. Regenerate the "
        f"baseline to see the current numbers: {_REGENERATE}"
    )


def test_skills_past_the_cap_cost_exactly_the_overflow_line(
    measurements: dict[str, dict[str, Any]],
) -> None:
    five = measurements["candidates_6_skills_5"]["digest_chars"]
    six = measurements["candidates_6_skills_6"]["digest_chars"]
    twelve = measurements["candidates_6_skills_12"]["digest_chars"]
    assert six == five + len(_SKILL_OVERFLOW_ONE), (
        f"candidates_6_skills_6 should cost exactly candidates_6_skills_5 plus the "
        f"overflow line ({five + len(_SKILL_OVERFLOW_ONE)} chars), got {six}. "
        f"Regenerate the baseline: {_REGENERATE}"
    )
    assert twelve == five + len(_SKILL_OVERFLOW_SEVEN), (
        f"candidates_6_skills_12 should cost exactly candidates_6_skills_5 plus the "
        f"overflow line ({five + len(_SKILL_OVERFLOW_SEVEN)} chars), got {twelve}. "
        f"Regenerate the baseline: {_REGENERATE}"
    )


def test_grounding_refs_never_restate_a_rendered_specs_sentence() -> None:
    """Grounding refs that repeat the sentence printed above them.

    Checked against the digest with its warnings trailer stripped: the warnings
    trailer legitimately echoes a part's sentence text (it's how a warning names
    which part it's about), which is a different rendering feature from a spec's own
    grounding ref and isn't what this regression is about. Within that body, a
    sentence's occurrence count must equal the number of *distinct* rendered specs
    that legitimately carry it. Some corpus cases (e.g. `no_data_nearest_5`)
    authored several specs sharing one sentence on purpose. Any occurrence beyond
    that tally means a grounding ref restated a sentence somewhere it shouldn't have.
    """
    for case in CASES:
        digest = render_digest(case.envelope)
        body = _digest_without_warnings_or_part_labels(case.envelope)
        expected_counts: dict[str, int] = {}
        for spec in _rendered_specs(case, digest):
            sentence = (spec.get("resolved") or {}).get("rendered_sentence") or ""
            assert sentence, (
                f"{case.id}: a rendered spec has an empty rendered_sentence, which "
                f"this restatement check cannot count meaningfully"
            )
            expected_counts[sentence] = expected_counts.get(sentence, 0) + 1
        for sentence, expected in expected_counts.items():
            actual = body.count(sentence)
            assert actual == expected, (
                f"{case.id}: sentence {sentence!r} occurs {actual} times in the digest "
                f"body, expected {expected} (one per rendered spec that carries it) — "
                f"a grounding ref may be restating a sentence. Regenerate the "
                f"baseline: {_REGENERATE}"
            )


_PART_LABEL_LINE = re.compile(r'Part \d+: ".*"\n')


def _digest_without_warnings_or_part_labels(envelope: dict[str, Any]) -> str:
    """`render_digest(envelope)` with every warning and every part label suppressed.

    A multi-part envelope's `Part N: "<variable_text>"` label echoes a part's own
    ask text, and a warning may name a spec, so both are cleared before the
    grounding-ref restatement count.
    """
    diagnostics = dict(envelope.get("diagnostics") or {})
    diagnostics["warnings"] = []
    parts = [{**part, "warnings": []} for part in envelope.get("parts") or []]
    digest = render_digest({**envelope, "diagnostics": diagnostics, "parts": parts})
    return _PART_LABEL_LINE.sub("", digest)


@pytest.mark.parametrize(
    "pair", SENTENCE_PAIRS, ids=[f"{pair.base_id}_vs_{pair.long_id}" for pair in SENTENCE_PAIRS]
)
def test_sentence_sensitivity_delta_matches_the_suffixs_own_length(
    pair: SentencePair, measurements: dict[str, dict[str, Any]]
) -> None:
    """`definite_1` and its `_long_sentence` partner differ only by a known suffix.

    The digest's character delta between them should equal that suffix's own length
    exactly. If a grounding line ever restated the sentence, the delta would roughly
    double instead.
    """
    base, long = measurements[pair.base_id], measurements[pair.long_id]
    delta = long["digest_chars"] - base["digest_chars"]
    assert delta == len(pair.suffix), (
        f"{pair.base_id!r} -> {pair.long_id!r}: digest grew by {delta} chars, but the "
        f"appended suffix is {len(pair.suffix)} chars — a grounding ref may be "
        f"restating the sentence. Regenerate the baseline: {_REGENERATE}"
    )

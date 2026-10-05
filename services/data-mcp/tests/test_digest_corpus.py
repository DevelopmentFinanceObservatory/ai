"""Corpus coherence: case ids are unique, every envelope renders under its tagged
status, and frame pairs render identically.

`_STATUS_SHAPE_PATTERNS` below asserts only each status's opening-line *shape*, with
any count left unbound as `\\d+`; it leaves counts, truncation lines, and how many
specs appear unchecked, so `_CANDIDATE_CAP` can change the candidates header for
over-cap cases without breaking this check.
Content and size assertions belong to `test_digest_budget.py`; this module
only proves the corpus is self-consistent.
"""

from __future__ import annotations

import re

from data_mcp.digest import render_digest

from .digest_corpus import CASES, FRAME_PAIRS, SENTENCE_PAIRS

# One pattern per status, matched against the digest's start; the count stays
# unbound as `\d+` so an over-cap case (e.g. `candidates_12`) stays covered for
# status tagging without binding this test to how many specs the renderer shows.
_PART_LABEL_PREFIX = r'(Part \d+: ".*"\n)?'

_STATUS_SHAPE_PATTERNS: dict[str, re.Pattern[str]] = {
    "definite": re.compile(_PART_LABEL_PREFIX + r'"'),
    "candidates": re.compile(_PART_LABEL_PREFIX + r"\d+ candidate interpretations"),
    "no_data": re.compile(
        _PART_LABEL_PREFIX
        + r"(Resolved, but no data is available \(reason: |The ask could not be resolved \(reason: )"
    ),
    "unsupported": re.compile(_PART_LABEL_PREFIX + r"Not supported:"),
}


def test_case_ids_are_unique() -> None:
    ids = [case.id for case in CASES]
    assert len(ids) == len(set(ids))


def test_every_case_declares_a_status_with_a_known_shape() -> None:
    known = _STATUS_SHAPE_PATTERNS.keys()
    unknown = [case.id for case in CASES if case.status not in known]
    assert not unknown, f"cases with an unrecognised status: {unknown}"


def test_every_case_envelope_renders_under_its_tagged_status() -> None:
    for case in CASES:
        digest = render_digest(case.envelope)
        pattern = _STATUS_SHAPE_PATTERNS[case.status]
        assert pattern.match(digest), (
            f"{case.id}: digest does not open with the {case.status!r} shape "
            f"{pattern.pattern!r}: {digest[:120]!r}"
        )


def test_every_case_envelope_status_field_matches_its_tag() -> None:
    for case in CASES:
        assert case.envelope.get("status") == case.status, (
            f"{case.id} is tagged {case.status!r} but its envelope's status field disagrees"
        )


def test_frame_pair_ids_reference_real_cases() -> None:
    case_ids = {case.id for case in CASES}
    for pair in FRAME_PAIRS:
        assert pair.lean_id in case_ids, f"unknown frame-pair lean id {pair.lean_id!r}"
        assert pair.fat_id in case_ids, f"unknown frame-pair fat id {pair.fat_id!r}"


def test_frame_pairs_render_identically() -> None:
    cases_by_id = {case.id: case for case in CASES}
    for pair in FRAME_PAIRS:
        lean_digest = render_digest(cases_by_id[pair.lean_id].envelope)
        fat_digest = render_digest(cases_by_id[pair.fat_id].envelope)
        assert lean_digest == fat_digest, (
            f"{pair.lean_id!r} and {pair.fat_id!r} are meant to be digest-identical but differ"
        )


def test_sentence_pair_ids_reference_real_cases() -> None:
    case_ids = {case.id for case in CASES}
    for pair in SENTENCE_PAIRS:
        assert pair.base_id in case_ids, f"unknown sentence-pair base id {pair.base_id!r}"
        assert pair.long_id in case_ids, f"unknown sentence-pair long id {pair.long_id!r}"


def test_sentence_pairs_differ_only_by_the_declared_suffix() -> None:
    cases_by_id = {case.id: case for case in CASES}
    for pair in SENTENCE_PAIRS:
        base_sentence = cases_by_id[pair.base_id].envelope["parts"][0]["spec"]["resolved"][
            "rendered_sentence"
        ]
        long_sentence = cases_by_id[pair.long_id].envelope["parts"][0]["spec"]["resolved"][
            "rendered_sentence"
        ]
        assert long_sentence == base_sentence + pair.suffix

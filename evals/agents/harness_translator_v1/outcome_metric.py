"""`matches_subset`: how a translation case's expected outcome is matched.

A case in `harness-translator-v1.evalset.json` states the outcome it expects
as a `tool_responses` entry of its turn's `intermediate_data`, named for the
response it describes: `{"name": "translation", "response": {"outcome":
"ok", "steps": [...], "sources": [...]}}` for a plan that has to compile, or
`{"name": "translation", "response": {"outcome": "invalid_input", "kind":
..., "requirement_id": ...}}` for one that has to come back referred.
`services/harness/tests/test_eval_outcomes.py` drives each case through the
translator's core functions and checks the response it reaches against that
expectation with `matches_subset`.

The match is a subset: every key the case names has to be present and equal
on the actual response; keys the case leaves out (an operation id the
translation chose, `reason` prose, `sentence`, `searched_as`, ...) are not
compared. A list is compared element-wise, in order, and must be the same
length, so `steps` is checked step by step in order, since the tool
sequence is the thing under test. `sources` alone is compared as an
unordered collection: the compiler emits its sources in a canonical order
of its own, and a case states which `spec_id`s were selected, not the order
the compiler puts them in.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

# The one list a case's expectation names whose order is the compiler's,
# not the case's: see `matches_subset`.
_UNORDERED_KEYS = frozenset({"sources"})


def matches_subset(expected: object, actual: object) -> bool:
    """Whether `actual` agrees with `expected` on every key or item `expected` names.

    A dict matches when every one of `expected`'s keys is present in `actual`
    with a matching value -- a key `expected` leaves out is never checked. A
    list matches when `actual` is the same length and each pair matches by
    position; a shorter or longer `actual` fails, since a case names every
    step it expects, in order. The one exception is a list under a key in
    `_UNORDERED_KEYS` (`sources`), matched as a collection: same length, and
    every expected item matched to a distinct actual one, in any order --
    the compiler orders sources canonically and a case should not have to
    know how. Anything else matches by equality.

    `services/harness/tests/test_eval_outcomes.py` checks each case's real
    translation outcome against the case's expectation with it.
    """
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            return False
        actual_by_key = dict(actual)
        return all(
            key in actual_by_key and _matches_value(key, value, actual_by_key[key])
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(matches_subset(e, a) for e, a in zip(expected, actual, strict=True))
        )
    return expected == actual


def _matches_value(key: object, expected: object, actual: object) -> bool:
    if key in _UNORDERED_KEYS and isinstance(expected, list) and isinstance(actual, list):
        return len(actual) == len(expected) and _matches_unordered(expected, actual)
    return matches_subset(expected, actual)


def _matches_unordered(expected: Sequence[object], actual: Sequence[object]) -> bool:
    """Whether every item of `expected` matches a distinct item of `actual`, in any order.

    Tries each unmatched actual item for the first expected one and recurses
    on the rest, so an expectation loose enough to match several actual items
    still finds an assignment when one exists. The lists are a handful of
    sources long, so the search is cheap.
    """
    if not expected:
        return True
    first, rest = expected[0], expected[1:]
    return any(
        matches_subset(first, candidate)
        and _matches_unordered(rest, [*actual[:index], *actual[index + 1 :]])
        for index, candidate in enumerate(actual)
    )

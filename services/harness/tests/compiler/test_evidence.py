"""Tests for `compiler.evidence`: does a discovered source meet a requirement?"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest
from compiler.evidence import (
    EvidenceIssueCode,
    RetrievalEvidence,
    SeriesEvidence,
    Severity,
    check_evidence,
    entity_matches,
    errors,
    evidence_from_manifest,
    evidence_from_source,
    open_axes_for,
    opens_place_axis,
    resolve_entity,
    unlabelled_series_refs,
    with_series_labels,
)
from compiler.sources import member_digest, source_from_spec_block, stat_var_reading
from data_mcp.manifest import Manifest, SeriesStatisticsModel
from data_mcp.series_statistics import SERIES_STATISTICS_ARROW_SCHEMA, series_statistics_table
from planner.plan import Coverage, DatasetRequirement, Entity, Unit, open_window
from planner.records import (
    EvidenceIssue,
    Period,
    ResolvedSourceMention,
    SeriesEntity,
    StatVarReading,
)
from planner.workflow import OpenEntityAxis, ResolvedEntity, ResolvedGroup, SelectedSource

RequirementFactory = Callable[..., DatasetRequirement]
SourceFactory = Callable[..., SelectedSource]

SPEC_KENYA = "spec_" + "a" * 32
SPEC_OTHER = "spec_" + "b" * 32


def _codes(requirement: DatasetRequirement, source: SelectedSource) -> list[tuple[str, str]]:
    return [
        (issue.code.value, issue.severity.value)
        for issue in check_evidence(requirement, evidence_from_source(source))
    ]


def test_a_source_that_resolves_everything_raises_only_the_unit_notes(
    requirement: RequirementFactory, source: SourceFactory
) -> None:
    # The fixture's unit text names USD but no price basis, so a matching
    # currency is a note and the basis is recorded as unverified. The
    # requirement's `debtor` against the source's `destination` column is the
    # same side of the flow and says nothing.
    assert _codes(requirement(), source()) == [
        ("unit_matches", "info"),
        ("unit_price_basis_unverified", "warning"),
    ]


def test_a_source_with_no_data_is_an_error(
    requirement: RequirementFactory, source: SourceFactory
) -> None:
    issues = check_evidence(requirement(), evidence_from_source(source(has_data=False)))

    assert [issue.code for issue in errors(issues)] == [EvidenceIssueCode.NO_DATA_AT_SOURCE]


def test_an_entity_the_source_does_not_resolve_is_an_error(
    requirement: RequirementFactory, wide_source: SourceFactory
) -> None:
    # The finding names every label the source did resolve, so a two-country
    # source is what shows that the list is the source's and not the ask's.
    issues = check_evidence(
        requirement(entities=[Entity(name="Uganda")]), evidence_from_source(wide_source())
    )

    (issue,) = errors(issues)
    assert issue.code is EvidenceIssueCode.ENTITY_MISSING
    assert issue.required == "Uganda"
    assert issue.found == ["Kenya", "Ghana"]


def test_entity_names_match_case_folded(
    requirement: RequirementFactory, source: SourceFactory
) -> None:
    issues = check_evidence(
        requirement(entities=[Entity(name="kenya")]), evidence_from_source(source())
    )

    assert errors(issues) == []


def test_a_source_resolving_no_entities_leaves_them_unverified(
    requirement: RequirementFactory, source: SourceFactory
) -> None:
    codes = _codes(requirement(), source(entities=()))

    assert ("entity_unverified", "warning") in codes
    assert not any(severity == "error" for _, severity in codes)


def test_a_source_resolving_no_entities_is_an_error_when_the_requirement_fails_on_gaps(
    requirement: RequirementFactory, source: SourceFactory
) -> None:
    issues = check_evidence(
        requirement(coverage=Coverage(on_missing="fail")), evidence_from_source(source(entities=()))
    )

    (issue,) = errors(issues)
    assert issue.code is EvidenceIssueCode.ENTITY_UNVERIFIED


def test_a_source_stating_no_coverage_leaves_its_data_unverified(
    requirement: RequirementFactory, source: SourceFactory
) -> None:
    codes = _codes(requirement(), source(has_data=None))

    assert ("data_unverified", "warning") in codes
    assert "no_data_at_source" not in [code for code, _ in codes]


def test_a_source_stating_no_coverage_is_an_error_when_the_requirement_fails_on_gaps(
    requirement: RequirementFactory, source: SourceFactory
) -> None:
    issues = check_evidence(
        requirement(coverage=Coverage(on_missing="fail")),
        evidence_from_source(source(has_data=None)),
    )

    (issue,) = errors(issues)
    assert issue.code is EvidenceIssueCode.DATA_UNVERIFIED


def test_a_source_without_an_identity_is_an_error(
    requirement: RequirementFactory, source: SourceFactory
) -> None:
    # Drift detection compares an approved identity against the manifest's;
    # a plan cannot be approved against a source whose identity is unknown.
    issues = check_evidence(requirement(), evidence_from_source(source(identity=None)))

    (issue,) = errors(issues)
    assert issue.code is EvidenceIssueCode.IDENTITY_UNVERIFIED


def test_an_empty_identity_is_known_and_says_nothing(
    requirement: RequirementFactory, source: SourceFactory
) -> None:
    assert errors(check_evidence(requirement(), evidence_from_source(source(identity={})))) == []


# --- Roles: sides of the flow ---------------------------------------------------


def test_a_role_phrase_on_the_same_side_as_the_column_is_not_flagged(
    requirement: RequirementFactory, source: SourceFactory
) -> None:
    # `debtor` and `recipient` are two words for the receiving party.
    codes = [
        code
        for code, _ in _codes(requirement(entities=[Entity(name="Kenya", role="debtor")]), source())
    ]

    assert not any(code.startswith("entity_role") for code in codes)


def test_a_role_phrase_on_the_other_side_of_the_flow_from_the_column_is_an_error(
    requirement: RequirementFactory, source: SourceFactory
) -> None:
    kenya_debtor = ResolvedEntity(role="debtor", ref="country/KEN", label="Kenya")
    issues = check_evidence(
        requirement(entities=[Entity(name="Kenya", role="creditor")]),
        evidence_from_source(source(entities=(kenya_debtor,))),
    )

    (issue,) = errors(issues)
    assert issue.code is EvidenceIssueCode.ENTITY_ROLE_MISMATCH
    assert issue.found == "debtor"
    assert issue.required == "creditor"


@pytest.mark.parametrize(("phrase", "column"), [("from", "destination"), ("to", "origin")])
def test_a_directional_phrase_on_the_other_side_from_where_search_places_it_is_an_error(
    requirement: RequirementFactory, source: SourceFactory, phrase: str, column: str
) -> None:
    # `search` places `from` on `origin` and `to` on `destination`, so Kenya
    # under the other column was not bound from the phrase.
    kenya = ResolvedEntity(role=column, ref="country/KEN", label="Kenya")
    issues = check_evidence(
        requirement(entities=[Entity(name="Kenya", role=phrase)]),
        evidence_from_source(source(entities=(kenya,))),
    )

    (issue,) = errors(issues)
    assert issue.code is EvidenceIssueCode.ENTITY_ROLE_MISMATCH
    assert (issue.found, issue.required) == (column, phrase)


def test_a_role_phrase_the_lexicon_cannot_place_is_unverified(
    requirement: RequirementFactory, source: SourceFactory
) -> None:
    issues = check_evidence(
        requirement(entities=[Entity(name="Kenya", role="guarantor")]),
        evidence_from_source(source()),
    )

    assert errors(issues) == []
    (issue,) = [i for i in issues if i.code is EvidenceIssueCode.ENTITY_ROLE_UNVERIFIED]
    assert issue.severity is Severity.WARNING
    assert issue.found == "destination"
    assert issue.required == "guarantor"


def test_a_role_phrase_naming_the_column_itself_is_verified_even_outside_the_lexicon(
    requirement: RequirementFactory, source: SourceFactory
) -> None:
    kenya = ResolvedEntity(role="guarantor", ref="country/KEN", label="Kenya")
    codes = [
        code
        for code, _ in _codes(
            requirement(entities=[Entity(name="Kenya", role="Guarantor")]),
            source(entities=(kenya,)),
        )
    ]

    assert not any(code.startswith("entity_role") for code in codes)


def test_a_role_column_the_lexicon_cannot_place_is_unverified(
    requirement: RequirementFactory, source: SourceFactory
) -> None:
    # IDS-style naming: a `country` column says nothing about which side of
    # the flow it holds, so a `debtor` phrase against it is the wrong-side
    # case the check cannot rule out.
    kenya = ResolvedEntity(role="country", ref="country/KEN", label="Kenya")
    issues = check_evidence(
        requirement(entities=[Entity(name="Kenya", role="debtor")]),
        evidence_from_source(source(entities=(kenya,))),
    )

    assert errors(issues) == []
    (issue,) = [i for i in issues if i.code is EvidenceIssueCode.ENTITY_ROLE_UNVERIFIED]
    assert issue.severity is Severity.WARNING
    assert issue.found == "country"
    assert issue.required == "debtor"


def test_a_requirement_stating_no_role_asks_nothing_of_the_column(
    requirement: RequirementFactory, source: SourceFactory
) -> None:
    codes = [code for code, _ in _codes(requirement(entities=[Entity(name="Kenya")]), source())]

    assert not any(code.startswith("entity_role") for code in codes)


def test_an_entity_resolved_under_two_roles_with_no_stated_role_is_ambiguous(
    requirement: RequirementFactory, source: SourceFactory
) -> None:
    kenya_debtor = ResolvedEntity(role="debtor", ref="country/KEN-D", label="Kenya")
    kenya_creditor = ResolvedEntity(role="creditor", ref="country/KEN-C", label="Kenya")
    issues = check_evidence(
        requirement(entities=[Entity(name="Kenya")]),
        evidence_from_source(source(entities=(kenya_debtor, kenya_creditor))),
    )

    (issue,) = errors(issues)
    assert issue.code is EvidenceIssueCode.ENTITY_ROLE_AMBIGUOUS
    assert issue.found == ["creditor", "debtor"]


def test_a_stated_role_tells_apart_an_entity_resolved_under_two_roles(
    requirement: RequirementFactory, source: SourceFactory
) -> None:
    kenya_debtor = ResolvedEntity(role="debtor", ref="country/KEN-D", label="Kenya")
    kenya_creditor = ResolvedEntity(role="creditor", ref="country/KEN-C", label="Kenya")
    issues = check_evidence(
        requirement(entities=[Entity(name="Kenya", role="debtor")]),
        evidence_from_source(source(entities=(kenya_debtor, kenya_creditor))),
    )

    assert errors(issues) == []


@pytest.mark.parametrize(
    ("phrase", "column"),
    [
        ("recipient", "destination"),
        ("debtor", "destination"),
        ("donor", "origin"),
        ("creditor", "origin"),
    ],
)
def test_a_party_word_answers_the_search_role_column_on_its_side(
    requirement: RequirementFactory, source: SourceFactory, phrase: str, column: str
) -> None:
    # `origin` and `destination` are the columns `search` resolves a flow
    # entity under, so a party word on the same side is verified, not merely
    # left unplaced.
    kenya = ResolvedEntity(role=column, ref="country/KEN", label="Kenya")
    codes = [
        code
        for code, _ in _codes(
            requirement(entities=[Entity(name="Kenya", role=phrase)]), source(entities=(kenya,))
        )
    ]

    assert not any(code.startswith("entity_role") for code in codes)


@pytest.mark.parametrize(("phrase", "column"), [("donor", "destination"), ("recipient", "origin")])
def test_a_party_word_against_the_search_role_column_across_the_flow_is_an_error(
    requirement: RequirementFactory, source: SourceFactory, phrase: str, column: str
) -> None:
    kenya = ResolvedEntity(role=column, ref="country/KEN", label="Kenya")
    issues = check_evidence(
        requirement(entities=[Entity(name="Kenya", role=phrase)]),
        evidence_from_source(source(entities=(kenya,))),
    )

    (issue,) = errors(issues)
    assert issue.code is EvidenceIssueCode.ENTITY_ROLE_MISMATCH
    assert (issue.found, issue.required) == (column, phrase)


def test_a_coverage_window_short_of_the_period_is_an_error_when_the_requirement_fails_on_gaps(
    requirement: RequirementFactory, source: SourceFactory
) -> None:
    issues = check_evidence(
        requirement(period=Period(start=2005, end=2024), coverage=Coverage(on_missing="fail")),
        evidence_from_source(source()),
    )

    (issue,) = errors(issues)
    assert issue.code is EvidenceIssueCode.WINDOW_SHORT
    assert issue.found == {"start": 2010, "end": 2024}
    assert issue.required == {"start": 2005, "end": 2024}


def test_a_coverage_window_short_of_the_period_is_a_warning_when_the_requirement_reports_gaps(
    requirement: RequirementFactory, source: SourceFactory
) -> None:
    issues = check_evidence(
        requirement(period=Period(start=2005, end=2024)),  # default on_missing=report_gaps
        evidence_from_source(source()),
    )

    assert errors(issues) == []
    (issue,) = [i for i in issues if i.code is EvidenceIssueCode.WINDOW_SHORT]
    assert issue.severity is Severity.WARNING


def test_a_source_without_a_window_leaves_the_period_unverified(
    requirement: RequirementFactory, source: SourceFactory
) -> None:
    assert ("period_unverified", "warning") in _codes(requirement(), source(coverage_window=None))


def test_a_source_without_a_window_is_an_error_when_the_requirement_fails_on_gaps(
    requirement: RequirementFactory, source: SourceFactory
) -> None:
    issues = check_evidence(
        requirement(coverage=Coverage(on_missing="fail")),
        evidence_from_source(source(coverage_window=None)),
    )

    (issue,) = errors(issues)
    assert issue.code is EvidenceIssueCode.PERIOD_UNVERIFIED


# --- Scope: what the identity document binds ------------------------------------


def _identity(**members: object) -> dict[str, object]:
    return {"stat_var": "DebtServicePaid", **members}


# The window of the copy a requirement stating `period_end_open` is pinned to:
# a range with no end year, which holds every later year.
_OPEN_WINDOW = {"kind": "range", "start_year": 2015}
_OPEN_COPY = "spec_" + "c" * 32


def _scope_codes(requirement: DatasetRequirement, source: SelectedSource) -> list[tuple[str, str]]:
    return [
        (code, severity)
        for code, severity in _codes(requirement, source)
        if code.startswith("scope")
    ]


def test_an_identity_binding_exactly_the_required_entities_says_nothing(
    requirement: RequirementFactory, source: SourceFactory
) -> None:
    identity = _identity(entities={"recipient": ["country/KEN"]})

    assert _scope_codes(requirement(), source(identity=identity)) == []


def test_an_identity_that_does_not_bind_a_required_entity_is_an_error(
    requirement: RequirementFactory, source: SourceFactory
) -> None:
    identity = _identity(entities={"recipient": ["country/GHA"]})
    issues = check_evidence(requirement(), evidence_from_source(source(identity=identity)))

    (issue,) = errors(issues)
    assert issue.code is EvidenceIssueCode.SCOPE_ENTITIES_UNBOUND
    assert issue.found == ["country/GHA"]
    assert issue.required == ["country/KEN"]


def test_an_identity_bound_wider_than_the_requirement_is_a_warning(
    requirement: RequirementFactory, source: SourceFactory
) -> None:
    # A region name legitimately expands into its members.
    identity = _identity(entities={"recipient": ["country/KEN", "country/GHA"]})
    issues = check_evidence(requirement(), evidence_from_source(source(identity=identity)))

    assert errors(issues) == []
    (issue,) = [i for i in issues if i.code is EvidenceIssueCode.SCOPE_ENTITIES_WIDER]
    assert issue.severity is Severity.WARNING
    assert issue.found == ["country/GHA", "country/KEN"]


def test_identity_refs_are_compared_across_roles_and_never_the_roles_themselves(
    requirement: RequirementFactory, source: SourceFactory
) -> None:
    # The identity's keys are graph properties, not the source's role columns.
    identity = _identity(entities={"debtor_of": ["country/KEN"]})

    assert _scope_codes(requirement(), source(identity=identity)) == []


def test_an_identity_omitting_its_entities_says_nothing_about_them(
    requirement: RequirementFactory, source: SourceFactory
) -> None:
    assert _scope_codes(requirement(entities=[Entity(name="Uganda")]), source()) == []


def test_an_identity_with_an_empty_entity_map_states_none_and_says_nothing(
    requirement: RequirementFactory, source: SourceFactory
) -> None:
    # data-mcp writes `entities: {}` to clear the filter: an unfiltered
    # definition, wider than the requirement, not one that binds nothing.
    identity = _identity(entities={})

    assert _scope_codes(requirement(), source(identity=identity)) == []


def test_an_identity_requesting_only_the_latest_observation_can_never_cover_the_period(
    requirement: RequirementFactory, source: SourceFactory
) -> None:
    identity = _identity(requested_window={"kind": "latest"})
    issues = check_evidence(requirement(), evidence_from_source(source(identity=identity)))

    (issue,) = errors(issues)
    assert issue.code is EvidenceIssueCode.SCOPE_WINDOW_SHORT
    assert issue.found == {"kind": "latest"}


def test_an_identity_requesting_all_years_says_nothing_about_the_window(
    requirement: RequirementFactory, source: SourceFactory
) -> None:
    identity = _identity(requested_window={"kind": "all"})

    assert _scope_codes(requirement(), source(identity=identity)) == []


def test_an_identity_range_covering_the_period_says_nothing(
    requirement: RequirementFactory, source: SourceFactory
) -> None:
    identity = _identity(requested_window={"kind": "range", "start_year": 2010, "end_year": 2024})

    assert _scope_codes(requirement(), source(identity=identity)) == []


def test_an_identity_range_falling_short_of_the_period_takes_the_coverage_severity(
    requirement: RequirementFactory, source: SourceFactory
) -> None:
    identity = _identity(requested_window={"kind": "range", "start_year": 2018, "end_year": 2024})

    assert _scope_codes(requirement(), source(identity=identity)) == [
        ("scope_window_short", "warning")
    ]
    assert _scope_codes(
        requirement(coverage=Coverage(on_missing="fail")), source(identity=identity)
    ) == [("scope_window_short", "error")]


def test_an_identity_range_disjoint_from_the_period_is_an_error_whatever_the_coverage(
    requirement: RequirementFactory, source: SourceFactory
) -> None:
    identity = _identity(requested_window={"kind": "range", "start_year": 1990, "end_year": 2000})
    issues = check_evidence(requirement(), evidence_from_source(source(identity=identity)))

    (issue,) = errors(issues)
    assert issue.code is EvidenceIssueCode.SCOPE_WINDOW_SHORT
    assert issue.required == {"start": 2015, "end": 2024}


def test_an_identity_range_without_its_years_says_nothing(
    requirement: RequirementFactory, source: SourceFactory
) -> None:
    identity = _identity(requested_window={"kind": "range"})

    assert _scope_codes(requirement(), source(identity=identity)) == []


@pytest.mark.parametrize(
    ("start_year", "expected"),
    [
        (2015, []),
        (2010, []),
        (2018, [("scope_window_short", "warning")]),
        (2030, [("scope_window_short", "error")]),
    ],
    ids=["from-the-period-start", "from-before-it", "from-inside-it", "from-after-it"],
)
def test_an_identity_range_with_no_end_year_is_short_only_when_it_starts_late(
    requirement: RequirementFactory,
    source: SourceFactory,
    start_year: int,
    expected: list[tuple[str, str]],
) -> None:
    # The requirement asks for 2015-2024; a range with no end year holds every
    # year from its start on.
    identity = _identity(requested_window={"kind": "range", "start_year": start_year})

    assert _scope_codes(requirement(period_end_open=True), source(identity=identity)) == expected


def test_a_unit_naming_the_required_currency_matches(
    requirement: RequirementFactory, source: SourceFactory
) -> None:
    issues = check_evidence(
        requirement(unit=Unit(currency="USD", prices="current")), evidence_from_source(source())
    )

    (issue,) = [i for i in issues if i.code is EvidenceIssueCode.UNIT_MATCHES]
    assert issue.severity is Severity.INFO


def test_a_us_dollar_symbol_maps_to_usd_and_matches(
    requirement: RequirementFactory, source: SourceFactory
) -> None:
    issues = check_evidence(
        requirement(unit=Unit(currency="USD", prices="current")),
        evidence_from_source(source(unit_ref="US$", unit_label=None)),
    )

    (issue,) = [i for i in issues if i.code is EvidenceIssueCode.UNIT_MATCHES]
    assert issue.severity is Severity.INFO


def test_a_unit_naming_a_different_currency_is_a_mismatch(
    requirement: RequirementFactory, source: SourceFactory
) -> None:
    issues = check_evidence(
        requirement(unit=Unit(currency="USD", prices="current")),
        evidence_from_source(source(unit_ref="eur", unit_label="Euros")),
    )

    (issue,) = errors(issues)
    assert issue.code is EvidenceIssueCode.UNIT_MISMATCH
    assert issue.found == "EUR"
    assert issue.required == "USD"


def test_a_currency_code_inside_an_underscored_unit_ref_is_seen(
    requirement: RequirementFactory, source: SourceFactory
) -> None:
    # `_` is a word character, so a word-boundary match could not see this
    # USD and read a matching unit as unverified.
    issues = check_evidence(
        requirement(unit=Unit(currency="USD", prices="current")),
        evidence_from_source(source(unit_ref="SDG_CU_USD_M", unit_label=None)),
    )

    (issue,) = [i for i in issues if i.code is EvidenceIssueCode.UNIT_MATCHES]
    assert issue.severity is Severity.INFO


def test_a_source_unit_label_naming_ppp_is_not_us_dollars(
    requirement: RequirementFactory, source: SourceFactory
) -> None:
    issues = check_evidence(
        requirement(unit=Unit(currency="USD", prices="current")),
        evidence_from_source(source(unit_ref="usd", unit_label="PPP, current international $")),
    )

    (issue,) = errors(issues)
    assert issue.code is EvidenceIssueCode.UNIT_MISMATCH
    assert issue.found == "PPP"


def test_a_unit_naming_no_recognisable_currency_is_unverified(
    requirement: RequirementFactory, source: SourceFactory
) -> None:
    issues = check_evidence(
        requirement(unit=Unit(currency="USD", prices="current")),
        evidence_from_source(source(unit_ref=None, unit_label="local currency units")),
    )

    (issue,) = [i for i in issues if i.code is EvidenceIssueCode.UNIT_UNVERIFIED]
    assert issue.severity is Severity.WARNING


# --- Price basis ----------------------------------------------------------------


def _basis_codes(requirement: DatasetRequirement, source: SelectedSource) -> list[tuple[str, str]]:
    return [(c, s) for c, s in _codes(requirement, source) if c.startswith("unit_price_basis")]


def test_a_unit_stating_current_prices_meets_a_current_price_requirement(
    requirement: RequirementFactory, source: SourceFactory
) -> None:
    current = requirement(unit=Unit(currency="USD", prices="current"))
    issues = check_evidence(
        current,
        evidence_from_source(source(unit_ref="usd", unit_label="US dollars, current prices")),
    )

    assert [i for i in issues if i.code.value.startswith("unit_price_basis")] == []
    (matches,) = [i for i in issues if i.code is EvidenceIssueCode.UNIT_MATCHES]
    assert matches.detail.endswith("in current prices")


def test_a_unit_stating_constant_prices_fails_a_current_price_requirement(
    requirement: RequirementFactory, source: SourceFactory
) -> None:
    issues = check_evidence(
        requirement(unit=Unit(currency="USD", prices="current")),
        evidence_from_source(source(unit_ref="usd", unit_label="Constant 2015 US dollars")),
    )

    (issue,) = errors(issues)
    assert issue.code is EvidenceIssueCode.UNIT_PRICE_BASIS_MISMATCH
    assert issue.found == "constant"
    assert issue.required == "current"


def test_a_unit_stating_no_price_basis_is_unverified_not_refused(
    requirement: RequirementFactory, source: SourceFactory
) -> None:
    # The base year is missing from the search surface (issue #193); an error
    # here would block every constant-price plan.
    assert _basis_codes(requirement(), source()) == [("unit_price_basis_unverified", "warning")]


def test_a_unit_naming_both_bases_states_neither(
    requirement: RequirementFactory, source: SourceFactory
) -> None:
    assert _basis_codes(
        requirement(), source(unit_label="US dollars, current or constant 2015 prices")
    ) == [("unit_price_basis_unverified", "warning")]


def test_a_constant_price_requirement_is_met_by_the_same_base_year(
    requirement: RequirementFactory, source: SourceFactory
) -> None:
    constant = requirement(unit=Unit(currency="USD", prices="constant", base_year=2015))

    assert (
        _basis_codes(
            constant,
            source(unit_ref="ConstantUSD_2015", unit_label="constant 2015 US dollars"),
        )
        == []
    )


def test_a_constant_price_requirement_fails_on_another_base_year(
    requirement: RequirementFactory, source: SourceFactory
) -> None:
    constant = requirement(unit=Unit(currency="USD", prices="constant", base_year=2015))
    issues = check_evidence(
        constant, evidence_from_source(source(unit_label="constant 2010 US dollars"))
    )

    (issue,) = errors(issues)
    assert issue.code is EvidenceIssueCode.UNIT_PRICE_BASIS_MISMATCH
    assert (issue.found, issue.required) == (2010, 2015)


def test_a_constant_price_requirement_is_refused_without_a_base_year(
    requirement: RequirementFactory, source: SourceFactory
) -> None:
    # An answer over a source of unknown base year would claim a year it
    # does not have.
    constant = requirement(unit=Unit(currency="USD", prices="constant", base_year=2015))

    assert _basis_codes(constant, source(unit_label="real US dollars")) == [
        ("unit_price_basis_unverified", "error")
    ]


def test_a_requirement_without_a_unit_asks_nothing_of_the_sources_unit(
    requirement: RequirementFactory, source: SourceFactory
) -> None:
    codes = [
        code for code, _ in _codes(requirement(unit=None), source(unit_ref=None, unit_label=None))
    ]

    assert not any(code.startswith("unit") for code in codes)


def test_a_pinned_source_spec_id_matching_the_evidence_is_not_flagged(
    requirement: RequirementFactory, source: SourceFactory
) -> None:
    issues = check_evidence(
        requirement(source_spec_id=SPEC_KENYA),
        evidence_from_source(source(spec_id=SPEC_KENYA)),
    )

    assert not any(i.code is EvidenceIssueCode.SOURCE_PIN_MISMATCH for i in issues)


def test_a_pinned_source_spec_id_the_evidence_does_not_match_is_an_error(
    requirement: RequirementFactory, source: SourceFactory
) -> None:
    issues = check_evidence(
        requirement(source_spec_id=SPEC_OTHER),
        evidence_from_source(source(spec_id=SPEC_KENYA)),
    )

    (issue,) = [i for i in errors(issues) if i.code is EvidenceIssueCode.SOURCE_PIN_MISMATCH]
    assert issue.found == SPEC_KENYA
    assert issue.required == SPEC_OTHER


def test_a_stated_source_not_named_by_the_provenance_is_a_mismatch(
    requirement: RequirementFactory, source: SourceFactory
) -> None:
    issues = check_evidence(
        requirement(source="OECD DAC CRS"),
        evidence_from_source(
            source(provenance_ref="worldbank/ids", provenance_label="World Bank IDS")
        ),
    )

    (issue,) = errors(issues)
    assert issue.code is EvidenceIssueCode.SOURCE_MISMATCH
    assert issue.required == "OECD DAC CRS"


def test_a_stated_source_named_by_the_provenance_label_is_not_flagged(
    requirement: RequirementFactory, source: SourceFactory
) -> None:
    issues = check_evidence(
        requirement(source="OECD DAC"),
        evidence_from_source(
            source(provenance_ref="oecd/dac/crs", provenance_label="OECD DAC CRS table 2a")
        ),
    )

    assert errors(issues) == []


def test_a_stated_source_named_only_by_the_provenance_ref_is_not_flagged(
    requirement: RequirementFactory, source: SourceFactory
) -> None:
    # The label names a different table, so the ref alone has to carry the match.
    # `oecd/dac/crs` and "OECD DAC" coincide once punctuation and spacing go.
    issues = check_evidence(
        requirement(source="OECD DAC"),
        evidence_from_source(source(provenance_ref="oecd/dac/crs", provenance_label="Table 2a")),
    )

    assert errors(issues) == []


def test_a_stated_source_matches_a_provenance_label_spelled_without_spaces(
    requirement: RequirementFactory, source: SourceFactory
) -> None:
    # data-mcp serves the unspaced form of the name, which a requirement written
    # by a human spells with spaces.
    issues = check_evidence(
        requirement(source="World Development Indicators"),
        evidence_from_source(
            source(provenance_ref=None, provenance_label="WorldDevelopmentIndicators")
        ),
    )

    assert errors(issues) == []


def test_a_stated_source_matches_a_provenance_ref_carrying_a_path_prefix(
    requirement: RequirementFactory, source: SourceFactory
) -> None:
    issues = check_evidence(
        requirement(source="World Development Indicators"),
        evidence_from_source(
            source(provenance_ref="dc/base/WorldDevelopmentIndicators", provenance_label=None)
        ),
    )

    assert errors(issues) == []


def test_a_stated_source_matches_a_provenance_label_split_at_an_acronym_boundary(
    requirement: RequirementFactory, source: SourceFactory
) -> None:
    # The camel-case rule alone splits on a lower-to-upper transition, which
    # "UNICEFData" never has -- the run stays all-uppercase until "Data"
    # starts. The acronym-to-word rule catches that boundary too.
    issues = check_evidence(
        requirement(source="UNICEF Data"),
        evidence_from_source(source(provenance_ref=None, provenance_label="UNICEFData")),
    )

    assert errors(issues) == []


@pytest.mark.parametrize(
    ("stated", "label"),
    [("OECD DAC2a", "OECD DAC2A"), ("Covid19 Deaths", "Covid19Deaths")],
    ids=["table-letter-in-either-case", "digit-before-a-word"],
)
def test_a_stated_source_matches_a_provenance_label_with_a_digit_before_a_capital(
    requirement: RequirementFactory, source: SourceFactory, stated: str, label: str
) -> None:
    # Search labels the DAC2a provenance "OECD DAC2A", and a requirement copies
    # the search sentence's "OECD DAC2a".
    issues = check_evidence(
        requirement(source=stated),
        evidence_from_source(source(provenance_ref=None, provenance_label=label)),
    )

    assert errors(issues) == []


def test_a_stated_source_matches_a_provenance_label_across_accented_and_plain_spellings(
    requirement: RequirementFactory, source: SourceFactory
) -> None:
    # The requirement is typed with Czech diacritics; the provenance spells
    # the same name in plain ASCII camel case, as data-mcp serves it.
    issues = check_evidence(
        requirement(source="Český statistický úřad"),
        evidence_from_source(source(provenance_ref=None, provenance_label="CeskyStatistickyUrad")),
    )

    assert errors(issues) == []


def test_an_accented_acronym_does_not_match_a_single_ascii_letter_it_folds_near(
    requirement: RequirementFactory, source: SourceFactory
) -> None:
    # "CSU" folds from "ČSÚ" as one token, not the bare "s" an ASCII-only
    # tokenizer would have left after dropping "Č" and "Ú".
    issues = check_evidence(
        requirement(source="ČSÚ"),
        evidence_from_source(source(provenance_ref=None, provenance_label="S Bank")),
    )

    (issue,) = errors(issues)
    assert issue.code is EvidenceIssueCode.SOURCE_MISMATCH
    assert issue.required == "ČSÚ"


def test_a_hiragana_voiced_mark_does_not_fold_to_its_unvoiced_base(
    requirement: RequirementFactory, source: SourceFactory
) -> None:
    # NFKD decomposes the voiced kana "が" into the unvoiced base "か" plus a
    # combining sound mark; folding must leave that mark alone since its
    # base is not Latin, so the two kana stay distinct words.
    issues = check_evidence(
        requirement(source="が"),
        evidence_from_source(source(provenance_ref=None, provenance_label="か")),
    )

    (issue,) = errors(issues)
    assert issue.code is EvidenceIssueCode.SOURCE_MISMATCH


def test_a_katakana_semi_voiced_mark_does_not_fold_to_its_unvoiced_base(
    requirement: RequirementFactory, source: SourceFactory
) -> None:
    # Same shape as the hiragana case, with the semi-voiced mark that
    # separates "パ" from "ハ".
    issues = check_evidence(
        requirement(source="パ"),
        evidence_from_source(source(provenance_ref=None, provenance_label="ハ")),
    )

    (issue,) = errors(issues)
    assert issue.code is EvidenceIssueCode.SOURCE_MISMATCH


def test_a_greek_accent_does_not_fold_to_its_unaccented_base(
    requirement: RequirementFactory, source: SourceFactory
) -> None:
    # "ά" (alpha with tonos) NFKD-decomposes to "α" plus a combining acute  # noqa: RUF003
    # accent with the same non-zero combining class a Latin accent uses;
    # folding must not strip it, since accented and unaccented Greek are
    # different words.
    issues = check_evidence(
        requirement(source="ά"),
        evidence_from_source(source(provenance_ref=None, provenance_label="α")),  # noqa: RUF001
    )

    (issue,) = errors(issues)
    assert issue.code is EvidenceIssueCode.SOURCE_MISMATCH


def test_a_devanagari_nukta_does_not_fold_to_its_base_consonant(
    requirement: RequirementFactory, source: SourceFactory
) -> None:
    # "क़" (qa) NFKD-decomposes to "क" (ka) plus a combining nukta, and the
    # pair is a composition exclusion: NFC never recomposes it back to one
    # code point, so the mark reaches `_tokens` as its own character. A
    # word-character scan alone would drop it (it is not itself a word
    # character) and merge two unrelated consonants into one token.
    issues = check_evidence(
        requirement(source="क़"),
        evidence_from_source(source(provenance_ref=None, provenance_label="क")),
    )

    (issue,) = errors(issues)
    assert issue.code is EvidenceIssueCode.SOURCE_MISMATCH


def test_a_short_acronym_does_not_match_letters_inside_an_unrelated_word(
    requirement: RequirementFactory, source: SourceFactory
) -> None:
    # "ida" falls consecutively inside "guidance" ("gu-ida-nce"), so a blob
    # comparison with punctuation and spacing dropped would match "IDA" to
    # it. Tokenised, "guidance" is one word.
    issues = check_evidence(
        requirement(source="IDA"),
        evidence_from_source(
            source(provenance_ref="dc/base/PolicyGuidanceNotes", provenance_label=None)
        ),
    )

    (issue,) = errors(issues)
    assert issue.code is EvidenceIssueCode.SOURCE_MISMATCH
    assert issue.required == "IDA"


def test_a_source_name_of_punctuation_alone_matches_nothing(
    requirement: RequirementFactory, source: SourceFactory
) -> None:
    # Every normalised side is empty, so neither containment direction holds.
    issues = check_evidence(
        requirement(source="---"),
        evidence_from_source(source(provenance_ref="/", provenance_label=".")),
    )

    (issue,) = errors(issues)
    assert issue.code is EvidenceIssueCode.SOURCE_MISMATCH
    assert issue.required == "---"


def test_a_stated_source_with_no_provenance_known_is_unverified(
    requirement: RequirementFactory, source: SourceFactory
) -> None:
    issues = check_evidence(
        requirement(source="OECD DAC CRS"),
        evidence_from_source(source(provenance_ref=None, provenance_label=None)),
    )

    (issue,) = [i for i in issues if i.code is EvidenceIssueCode.SOURCE_UNVERIFIED]
    assert issue.severity is Severity.WARNING


def test_every_finding_is_reported_in_one_pass(
    requirement: RequirementFactory, source: SourceFactory
) -> None:
    codes = _codes(
        requirement(entities=[Entity(name="Uganda")], period=Period(start=1990, end=2024)),
        source(has_data=False),
    )

    assert [code for code, _ in codes] == [
        "no_data_at_source",
        "entity_missing",
        "window_short",
        "unit_matches",
        "unit_price_basis_unverified",
    ]


def test_resolve_entity_returns_none_for_no_match() -> None:
    assert resolve_entity(Entity(name="Uganda"), ()) is None


def test_resolve_entity_returns_the_one_label_match() -> None:
    kenya = ResolvedEntity(role="recipient", ref="country/KEN", label="Kenya")
    assert resolve_entity(Entity(name="kenya"), (kenya,)) is kenya


@pytest.mark.parametrize(
    ("role", "columns"),
    [("debtor", ("debtor", "creditor")), ("donor", ("origin", "destination"))],
    ids=["column-named-by-role", "column-on-the-role-side"],
)
def test_resolve_entity_narrows_by_role_column(role: str, columns: tuple[str, str]) -> None:
    wanted, other = (
        ResolvedEntity(role=column, ref=f"country/KEN-{column}", label="Kenya")
        for column in columns
    )
    assert resolve_entity(Entity(name="Kenya", role=role), (wanted, other)) is wanted


def test_resolve_entity_returns_none_when_role_cannot_narrow_to_one() -> None:
    debtor = ResolvedEntity(role="debtor", ref="country/KEN-D", label="Kenya")
    creditor = ResolvedEntity(role="creditor", ref="country/KEN-C", label="Kenya")
    assert resolve_entity(Entity(name="Kenya"), (debtor, creditor)) is None


def test_entity_matches_returns_every_label_match_before_role_narrows() -> None:
    debtor = ResolvedEntity(role="debtor", ref="country/KEN-D", label="Kenya")
    creditor = ResolvedEntity(role="creditor", ref="country/KEN-C", label="Kenya")
    assert entity_matches(Entity(name="Kenya", role="debtor"), (debtor, creditor)) == [
        debtor,
        creditor,
    ]


# --- The manifest side ------------------------------------------------------------------

SPEC = "spec_" + "a" * 32


def _trace(*refs: str) -> dict[str, object]:
    """A manifest `trace` binding the spec to `refs`, in the shape `trace.resolved_stat_vars` sends."""
    return {"resolved_stat_vars": [{"ref": f"dcid:{ref}", "roles": ["recipient"]} for ref in refs]}


def _family_trace(family: Iterable[str]) -> dict[str, object]:
    """The trace fields stating the family the mint admitted before reducing it."""
    members = list(family)
    return {"family_member_digest": member_digest(members), "family_member_count": len(members)}


def _stat(stat_var: str, **overrides: object) -> dict[str, object]:
    """One series-statistics row, `SeriesStatisticsModel`-shaped: ten annual Kenya
    observations, 2015-2024, in USD. Fed to `_series_stats`, not to `_manifest` --
    per-series figures travel to `evidence_from_manifest` as its own `series_stats`
    argument, separate from the manifest.
    """
    fields: dict[str, object] = {
        "stat_var": f"dcid:{stat_var}",
        "entities": [{"role": "recipient", "ref": "country/KEN"}],
        "facet": {"unit": "USD"},
        "observations": 10,
        "first": {"value": 1.0, "period": "2015"},
        "latest": {"value": 2.0, "period": "2024"},
        "minimum": {"value": 1.0, "period": "2015"},
        "maximum": {"value": 2.0, "period": "2024"},
    }
    return fields | overrides


def _series_stats(*specs: dict[str, object]) -> pa.Table:
    """`evidence_from_manifest`'s `series_stats` argument, built from `_stat`-shaped dicts."""
    return series_statistics_table([SeriesStatisticsModel.model_validate(spec) for spec in specs])


def _default_series_stats() -> pa.Table:
    """The one series `_manifest`'s own default `trace` (`DebtServicePaid` alone) matches --
    what a test passes as `series_stats` when it does not care about statistics itself."""
    return _series_stats(_stat("DebtServicePaid"))


def _manifest(**overrides: object) -> Manifest:
    """A `get_data` manifest for ten annual Kenya observations, 2015 to 2024, in USD.

    `trace` serves `DebtServicePaid` alone by default and states the digest
    of that one-member family, matching the `source` fixture's approved
    digest, so a test that does not care about membership does not pick up a
    finding from it by accident. Carries no per-series
    statistics of its own: those move separately, through `evidence_from_manifest`'s
    `series_stats` argument (`_default_series_stats`/`_series_stats`), since
    the manifest itself carries only `series_count` and a reference to its
    statistics artifact.
    """
    fields: dict[str, object] = {
        "schema_version": 1,
        "definition": {
            "schema_version": 1,
            "spec_id": SPEC,
            "identity": {"stat_var": "DebtServicePaid"},
        },
        "content_version": "cv_" + "1" * 32,
        "coverage": {"rows": 10, "dimensions": []},
        "citation": {"text": "World Bank", "sources": []},
        "trace": _trace("DebtServicePaid") | _family_trace(["DebtServicePaid"]),
        "warnings": [],
        "artifacts": [],
    }
    return Manifest.model_validate(fields | overrides)


def _sparse(observations: int) -> RetrievalEvidence:
    return evidence_from_manifest(
        _manifest(),
        series_stats=_series_stats(_stat("DebtServicePaid", observations=observations)),
    )


def test_a_manifest_is_read_into_evidence_by_series(
    requirement: RequirementFactory, source: SourceFactory
) -> None:
    evidence = evidence_from_manifest(_manifest(), series_stats=_default_series_stats())

    assert evidence.origin == "manifest"
    assert evidence.spec_id == SPEC
    assert evidence.stat_vars == ("DebtServicePaid",)
    assert evidence.resolved_stat_vars == ("DebtServicePaid",)
    assert [(e.role, e.ref) for e in evidence.entities] == [("recipient", "country/KEN")]
    assert evidence.unit_ref == "USD"
    assert evidence.window is not None and (evidence.window.start, evidence.window.end) == (
        2015,
        2024,
    )
    assert evidence.rows == 10
    assert evidence.identity == {"stat_var": "DebtServicePaid"}


def test_a_retrieval_matching_the_approved_source_meets_the_requirement(
    requirement: RequirementFactory, source: SourceFactory
) -> None:
    issues = check_evidence(
        requirement(),
        evidence_from_manifest(_manifest(), series_stats=_default_series_stats()),
        approved=source(),
    )

    assert errors(issues) == []
    # The unit names its currency but no price basis, which the surface does
    # not carry yet, so the basis rides along as a note.
    assert [issue.code for issue in issues] == [
        EvidenceIssueCode.UNIT_MATCHES,
        EvidenceIssueCode.UNIT_PRICE_BASIS_UNVERIFIED,
    ]


@pytest.mark.parametrize(
    ("unit_ref", "unit_label", "request_unit", "refused_by"),
    [
        pytest.param(
            "undata/UNIT_MEASURE-CR_USD_K",
            "United States dollars at constant prices",
            None,
            EvidenceIssueCode.UNIT_PRICE_BASIS_UNVERIFIED,
            id="constant-prices-of-an-unstated-base-year",
        ),
        pytest.param(
            "undata/UNIT_MEASURE-CR_USD_K",
            "United States dollars at constant 2024 prices",
            None,
            EvidenceIssueCode.UNIT_PRICE_BASIS_UNVERIFIED,
            id="constant-2024-prices-stated-only-in-the-label",
        ),
        pytest.param(
            "ConstantUSD_2024",
            "Constant 2024 US dollars",
            None,
            None,
            id="unit-id-in-constant-2024",
        ),
        pytest.param(
            "CurrencyUSD",
            "US Dollar",
            {"currency": "USD", "prices": "constant", "base_year": 2024},
            None,
            id="bound-by-request-to-constant-2024",
        ),
    ],
)
def test_discovery_and_the_manifest_read_a_price_basis_alike(
    requirement: RequirementFactory,
    source: SourceFactory,
    unit_ref: str,
    unit_label: str,
    request_unit: dict[str, object] | None,
    refused_by: EvidenceIssueCode | None,
) -> None:
    # A retrieved series states its unit as dfo-apis writes it, a request
    # object or a ref with no label, so a plan accepted at approval is
    # accepted once retrieved, and one refused at run time never compiles.
    constant = requirement(unit=Unit(currency="USD", prices="constant", base_year=2024))
    identity: dict[str, Any] = {"stat_var": "DebtServicePaid"}
    if request_unit is not None:
        identity["facets"] = {"unit": {"state": "value", "request": request_unit}}
    discovered = source(unit_ref=unit_ref, unit_label=unit_label, identity=identity)
    retrieved = evidence_from_manifest(
        _manifest(),
        series_stats=_series_stats(
            _stat("DebtServicePaid", facet={"unit": request_unit or unit_ref})
        ),
    )

    def refusals(evidence: RetrievalEvidence) -> list[EvidenceIssueCode]:
        return [
            issue.code
            for issue in errors(check_evidence(constant, evidence))
            if issue.code.value.startswith("unit")
        ]

    at_approval = refusals(evidence_from_source(discovered))
    assert at_approval == refusals(retrieved)
    assert at_approval == ([refused_by] if refused_by is not None else [])


def test_an_entity_is_found_through_the_ref_the_approved_source_resolved(
    requirement: RequirementFactory, source: SourceFactory
) -> None:
    # The manifest names `country/KEN`; the requirement names "Kenya". Without
    # the approved source to join them the entity cannot be confirmed.
    without = check_evidence(
        requirement(), evidence_from_manifest(_manifest(), series_stats=_default_series_stats())
    )

    assert [i.code for i in errors(without)] == [EvidenceIssueCode.ENTITY_MISSING]


_DEBTORS = (("Kenya", "debtor"), ("Ghana", "debtor"))
# The `wide_source` fixture's two debtors, and a creditor beside them.
_WIDE = (
    ResolvedEntity(role="destination", ref="country/KEN", label="Kenya"),
    ResolvedEntity(role="destination", ref="country/GHA", label="Ghana"),
)
_GERMANY = ResolvedEntity(role="origin", ref="country/DEU", label="Germany")


@pytest.mark.parametrize(
    ("named", "on_missing", "approved_entities", "held", "severity"),
    [
        # Ghana's series stopped before the period: Kenya stands for the
        # debtors named, so the run answers for Kenya and leaves Ghana out.
        pytest.param(_DEBTORS, "report_gaps", "wide", "KEN", Severity.WARNING, id="left-out"),
        # Kenya and Ghana sit on one column of the approved source, whatever
        # the phrases: "recipient" and "debtor" place on one flow side, and
        # "country" and "economy" on none.
        pytest.param(
            (("Kenya", "recipient"), ("Ghana", "debtor")),
            "report_gaps",
            "wide",
            "KEN",
            Severity.WARNING,
            id="same-side",
        ),
        pytest.param(
            (("Kenya", "country"), ("Ghana", "economy")),
            "report_gaps",
            "wide",
            "KEN",
            Severity.WARNING,
            id="unplaced-phrases-one-column",
        ),
        pytest.param(_DEBTORS, "fail", "wide", "KEN", Severity.ERROR, id="fails-on-gaps"),
        # No debtor left to answer for.
        pytest.param(
            (("Ghana", "debtor"),), "report_gaps", "wide", "KEN", Severity.ERROR, id="none-left"
        ),
        # Germany remains, on another column.
        pytest.param(
            (("Germany", "creditor"), ("Ghana", "debtor")),
            "report_gaps",
            "with-germany",
            "DEU",
            Severity.ERROR,
            id="other-column",
        ),
        # A name the approved source never resolved is no gap in the data.
        pytest.param(_DEBTORS, "report_gaps", "kenya", "KEN", Severity.ERROR, id="unresolved"),
        pytest.param(_DEBTORS, "report_gaps", None, "KEN", Severity.ERROR, id="unapproved"),
    ],
)
def test_a_retrieval_holding_no_series_for_a_named_entity_leaves_it_out_while_one_remains(
    requirement: RequirementFactory,
    source: SourceFactory,
    wide_source: SourceFactory,
    named: tuple[tuple[str, str], ...],
    on_missing: str,
    approved_entities: str | None,
    held: str,
    severity: Severity,
) -> None:
    approved = {
        "wide": wide_source(),
        "with-germany": wide_source(entities=(*_WIDE, _GERMANY)),
        "kenya": source(),
    }.get(approved_entities or "")
    issues = check_evidence(
        requirement(
            entities=[Entity(name=name, role=role) for name, role in named],
            coverage=Coverage(on_missing=on_missing),
        ),
        evidence_from_manifest(
            _manifest(),
            series_stats=_series_stats(
                _stat("DebtServicePaid", entities=[{"role": "recipient", "ref": f"country/{held}"}])
            ),
        ),
        approved=approved,
    )

    assert [
        issue.severity
        for issue in issues
        if issue.code is EvidenceIssueCode.ENTITY_MISSING and issue.required == "Ghana"
    ] == [severity]


def test_an_identity_other_than_the_approved_one_is_drift(
    requirement: RequirementFactory, source: SourceFactory
) -> None:
    issues = check_evidence(
        requirement(),
        evidence_from_manifest(_manifest(), series_stats=_default_series_stats()),
        approved=source(identity={"stat_var": "SomethingElse"}),
    )

    assert [i.code for i in errors(issues)] == [EvidenceIssueCode.SOURCE_DRIFT]


def _open_manifest() -> Manifest:
    """What `get_data` answers for the copy of the definition over a range with no end year."""
    return _manifest(
        definition={
            "schema_version": 1,
            "spec_id": _OPEN_COPY,
            "identity": _identity(requested_window=_OPEN_WINDOW),
        }
    )


def test_a_retrieval_of_the_approved_open_copy_is_not_drift(
    requirement: RequirementFactory, source: SourceFactory
) -> None:
    approved = source(spec_id=_OPEN_COPY, identity=_identity(requested_window=_OPEN_WINDOW))

    issues = check_evidence(
        requirement(period_end_open=True),
        evidence_from_manifest(_open_manifest(), series_stats=_default_series_stats()),
        approved=approved,
    )

    assert errors(issues) == []


def test_a_retrieval_of_the_open_copy_drifts_from_an_approved_closed_one(
    requirement: RequirementFactory, source: SourceFactory
) -> None:
    closed = {"kind": "range", "start_year": 2015, "end_year": 2024}

    issues = check_evidence(
        requirement(),
        evidence_from_manifest(_open_manifest(), series_stats=_default_series_stats()),
        approved=source(identity=_identity(requested_window=closed)),
    )

    assert [i.code for i in errors(issues)] == [EvidenceIssueCode.SOURCE_DRIFT]


def test_fewer_observations_than_years_are_gaps_and_the_requirement_says_how_to_take_them(
    requirement: RequirementFactory, source: SourceFactory
) -> None:
    lenient = requirement(coverage=Coverage(on_missing="report_gaps"))
    strict = requirement(coverage=Coverage(on_missing="fail"))

    accepted = check_evidence(lenient, _sparse(7), approved=source())
    refused = check_evidence(strict, _sparse(7), approved=source())

    gap = next(i for i in accepted if i.code is EvidenceIssueCode.COVERAGE_GAPS)
    assert gap.severity is Severity.WARNING and (gap.found, gap.required) == (7, 10)
    assert [i.code for i in errors(refused)] == [EvidenceIssueCode.COVERAGE_GAPS]


def test_more_observations_than_years_mean_the_grain_is_not_annual(
    requirement: RequirementFactory, source: SourceFactory
) -> None:
    issues = check_evidence(requirement(), _sparse(40), approved=source())

    assert [i.code for i in errors(issues)] == [EvidenceIssueCode.GRAIN_AMBIGUOUS]


_ESTIMATED = _stat(
    "DebtServicePaid",
    facet={"unit": "USD", "measurementMethod": "Estimated"},
    observations=2,
    first={"value": 2.1, "period": "2025"},
    latest={"value": 2.2, "period": "2026"},
    minimum={"value": 2.1, "period": "2025"},
    maximum={"value": 2.2, "period": "2026"},
)
_CONTINUED = {
    "continuations": [
        {"axis": "measurementMethod", "method": "Estimated", "periods": ["2025", "2026"]}
    ]
}
_PER_SERIES_CODES = {
    EvidenceIssueCode.WINDOW_SHORT,
    EvidenceIssueCode.COVERAGE_GAPS,
    EvidenceIssueCode.GRAIN_AMBIGUOUS,
}


@pytest.mark.parametrize(
    ("reported_observations", "continuations", "found"),
    [
        # Ten reported years and two estimated ones span 2015-2026 as one series.
        pytest.param(10, _CONTINUED, [], id="continued-series-is-whole"),
        # Its observations are summed across both rows: 8 + 2 across 12 years,
        # and its largest value is the estimates' 2.2.
        pytest.param(
            8,
            _CONTINUED,
            [(EvidenceIssueCode.COVERAGE_GAPS, 10, 12, 2.2)],
            id="continued-series-gaps-count-both-rows",
        ),
        # Without a continuation the estimates row is a series of its own, short
        # of 2015-2024.
        pytest.param(
            10,
            {},
            [
                (
                    EvidenceIssueCode.WINDOW_SHORT,
                    {"start": 2025, "end": 2026},
                    {"start": 2015, "end": 2024},
                    2.2,
                )
            ],
            id="uncontinued-estimates-are-their-own-series",
        ),
    ],
)
def test_a_series_and_the_estimates_continuing_it_are_held_to_the_period_as_one(
    requirement: RequirementFactory,
    source: SourceFactory,
    reported_observations: int,
    continuations: dict[str, object],
    found: list[tuple[EvidenceIssueCode, object, object, float]],
) -> None:
    """A frame Datasets continued with its source's estimates (the Zambia IMF case) holds
    one series as a row per `measurementMethod`; the manifest counts it once."""
    manifest = _manifest()
    manifest = manifest.model_copy(update={"trace": {**manifest.trace, **continuations}})
    evidence = evidence_from_manifest(
        manifest,
        series_stats=_series_stats(
            _stat("DebtServicePaid", observations=reported_observations), _ESTIMATED
        ),
    )

    issues = check_evidence(requirement(), evidence, approved=source())

    assert [
        (issue.code, issue.found, issue.required, issue.magnitude)
        for issue in issues
        if issue.code in _PER_SERIES_CODES
    ] == found


@pytest.mark.parametrize(
    ("approved", "kenya"),
    [
        pytest.param(True, "Kenya", id="labelled-by-the-approved-source"),
        pytest.param(False, None, id="unlabelled-with-no-approved-source"),
    ],
)
def test_a_short_series_finding_names_its_entities(
    requirement: RequirementFactory, source: SourceFactory, approved: bool, kenya: str | None
) -> None:
    """A series ending in 2021 with a year missing is short of 2015-2024 on
    both counts; each finding carries its entities, with the label the
    approved source gives a ref and none where it labels none."""
    named = (
        SeriesEntity(role="recipient", ref="country/KEN", label=kenya),
        SeriesEntity(role="donor", ref="donor/X"),
    )
    evidence = evidence_from_manifest(
        _manifest(),
        series_stats=_series_stats(
            _stat(
                "DebtServicePaid",
                entities=[
                    {"role": "recipient", "ref": "country/KEN"},
                    {"role": "donor", "ref": "donor/X"},
                ],
                observations=6,
                latest={"value": 2.0, "period": "2021"},
            )
        ),
    )

    issues = check_evidence(requirement(), evidence, approved=source() if approved else None)

    assert [
        (issue.code, issue.entities)
        for issue in issues
        if issue.code in {EvidenceIssueCode.WINDOW_SHORT, EvidenceIssueCode.COVERAGE_GAPS}
    ] == [(EvidenceIssueCode.WINDOW_SHORT, named), (EvidenceIssueCode.COVERAGE_GAPS, named)]


def _short_of(*refs: str, size: float | None, labelled: str | None = None) -> EvidenceIssue:
    """A `window_short` finding on the series of `refs`, of largest absolute value `size`,
    the ref `labelled` labelled by itself and the rest unlabelled."""
    return EvidenceIssue(
        code=EvidenceIssueCode.WINDOW_SHORT,
        severity=Severity.WARNING,
        requirement_id="debt",
        detail="",
        stat_var="Aid",
        entities=tuple(
            SeriesEntity(role="donor", ref=ref, label=ref if ref == labelled else None)
            for ref in refs
        ),
        magnitude=size,
    )


@pytest.mark.parametrize(
    ("limit", "refs"),
    [
        # Largest first, a size stated before none, ties by ref; a labelled ref
        # and one repeated are asked once or not at all.
        pytest.param(10, ["donor/C", "donor/A", "donor/B", "donor/D"], id="largest-first"),
        pytest.param(2, ["donor/C", "donor/A"], id="bounded"),
    ],
)
def test_the_refs_a_short_series_lookup_names_are_the_largest_series_unlabelled_ones(
    limit: int, refs: list[str]
) -> None:
    findings = [
        _short_of("donor/D", size=None),
        _short_of("donor/B", size=5.0),
        _short_of("donor/A", "donor/K", size=5.0, labelled="donor/K"),
        _short_of("donor/C", "donor/A", size=9.0),
        EvidenceIssue(
            code=EvidenceIssueCode.UNIT_UNVERIFIED,
            severity=Severity.WARNING,
            requirement_id="debt",
            detail="",
        ),
    ]

    assert unlabelled_series_refs(findings, limit=limit) == refs


def test_a_short_series_lookup_labels_only_the_entities_it_named() -> None:
    finding = _short_of("donor/A", "donor/B", "donor/K", size=1.0, labelled="donor/K")

    (labelled,) = with_series_labels([finding], {"donor/A": "Austria", "donor/K": "Kenya"})

    assert [entity.label for entity in labelled.entities] == ["Austria", None, "donor/K"]


def test_a_series_negative_throughout_ranks_by_how_far_below_zero_it_reaches(
    requirement: RequirementFactory, source: SourceFactory
) -> None:
    """Two donors' series to Kenya end in 2021: one of net repayments, -50 at
    its lowest, and one of grants, 1 at its peak. The repayments are the
    larger series, so their donor is asked for first."""

    def ending_2021(donor: str, low: float, high: float) -> dict[str, object]:
        return _stat(
            "DebtServicePaid",
            entities=[
                {"role": "recipient", "ref": "country/KEN"},
                {"role": "donor", "ref": donor},
            ],
            observations=7,
            first={"value": high, "period": "2015"},
            latest={"value": low, "period": "2021"},
            minimum={"value": low, "period": "2021"},
            maximum={"value": high, "period": "2015"},
        )

    evidence = evidence_from_manifest(
        _manifest(),
        series_stats=_series_stats(
            ending_2021("donor/GRANTS", 0.5, 1.0), ending_2021("donor/REPAYS", -50.0, -10.0)
        ),
    )

    issues = check_evidence(requirement(), evidence, approved=source())

    assert {issue.magnitude for issue in issues if issue.entities} == {1.0, 50.0}
    assert unlabelled_series_refs(issues, limit=10) == ["donor/REPAYS", "donor/GRANTS"]


@pytest.mark.parametrize(
    ("end_open", "on_missing", "found"),
    [
        # Kenya's series covers the 2015-2024 asked for, but Ghana's runs to
        # 2025, the newest year retrieved: Kenya's is short of that, and
        # meets the requirement however it treats gaps.
        pytest.param(True, "report_gaps", [("country/KEN", Severity.WARNING)], id="open"),
        pytest.param(True, "fail", [("country/KEN", Severity.WARNING)], id="open-failing-on-gaps"),
        pytest.param(False, "report_gaps", [], id="closed"),
    ],
)
def test_an_open_period_holds_each_series_to_the_newest_year_retrieved(
    requirement: RequirementFactory,
    source: SourceFactory,
    end_open: bool,
    on_missing: str,
    found: list[tuple[str, Severity]],
) -> None:
    evidence = evidence_from_manifest(
        _manifest(),
        series_stats=_series_stats(
            _stat("DebtServicePaid"),
            _stat(
                "DebtServicePaid",
                entities=[{"role": "recipient", "ref": "country/GHA"}],
                observations=11,
                latest={"value": 2.0, "period": "2025"},
            ),
        ),
    )
    asked = requirement(period_end_open=end_open, coverage=Coverage(on_missing=on_missing))

    issues = check_evidence(asked, evidence, approved=source())

    short = [issue for issue in issues if issue.code is EvidenceIssueCode.WINDOW_SHORT]
    assert [(issue.entities[0].ref, issue.severity) for issue in short] == found
    assert all(issue.required == {"start": 2015, "end": 2025} for issue in short)


@pytest.mark.parametrize(
    ("start_year", "asked"),
    [
        pytest.param(
            1990,
            "the requirement asks for 2015-2024, and its retrieval requests every year from 1990",
            id="read-from-the-first-year",
        ),
        pytest.param(2015, "the requirement asks for 2015-2024", id="read-from-the-stated-start"),
    ],
)
def test_a_short_series_states_the_years_its_retrieval_requested(
    requirement: RequirementFactory, start_year: int, asked: str
) -> None:
    """A read of a dataset from its first year on record is held to the stated
    period, and its gap says what it requested as well."""
    manifest = _manifest()
    identity = {"stat_var": "DebtServicePaid", "requested_window": open_window(start_year)}
    manifest = manifest.model_copy(
        update={"definition": manifest.definition.model_copy(update={"identity": identity})}
    )
    evidence = evidence_from_manifest(
        manifest,
        series_stats=_series_stats(
            _stat(
                "DebtServicePaid",
                observations=34,
                first={"value": 1.0, "period": "1990"},
                latest={"value": 2.0, "period": "2023"},
            )
        ),
    )

    (gap,) = [
        issue
        for issue in check_evidence(requirement(), evidence)
        if issue.code is EvidenceIssueCode.WINDOW_SHORT
    ]

    assert gap.detail == (f"series DebtServicePaid for ['country/KEN'] covers 1990-2023, {asked}")
    assert gap.required == {"start": 2015, "end": 2024}


@pytest.mark.parametrize("series_stats", [None, _series_stats()], ids=["none", "empty"])
def test_a_manifest_missing_statistics_is_always_refused_whatever_on_missing_says(
    requirement: RequirementFactory,
    source: SourceFactory,
    series_stats: pa.Table | None,
) -> None:
    # `None` (the caller found no `series_count`, or the fetch failed) and
    # a table with no rows (`series_count == 0`) are the same gap -- mandatory metadata is
    # missing, and a check that cannot run is not a passed one, so this is
    # an error even under `report_gaps`, which only ever waives short
    # coverage, not an inability to check it at all. No unit is asked for,
    # so the only error either way is the missing statistics themselves.
    manifest = _manifest()

    lenient = check_evidence(
        requirement(coverage=Coverage(on_missing="report_gaps"), unit=None),
        evidence_from_manifest(manifest, series_stats=series_stats),
        approved=source(),
    )
    strict = check_evidence(
        requirement(coverage=Coverage(on_missing="fail"), unit=None),
        evidence_from_manifest(manifest, series_stats=series_stats),
        approved=source(),
    )

    assert [i.code for i in errors(lenient)] == [EvidenceIssueCode.STATISTICS_UNAVAILABLE]
    # Under `fail` the entities and the window are unverified errors too: a
    # manifest summarising no series states neither. The point of the case is
    # that the missing statistics are refused whatever `on_missing` says.
    assert EvidenceIssueCode.STATISTICS_UNAVAILABLE in [i.code for i in errors(strict)]


def test_a_differing_spec_id_alone_is_drift(
    requirement: RequirementFactory, source: SourceFactory
) -> None:
    # The identity matches; only the spec_id differs. Still drift: a manifest
    # under a spec_id other than the one approved is not the retrieval the
    # plan was approved to run, whatever its identity says.
    manifest = _manifest(
        definition={
            "schema_version": 1,
            "spec_id": SPEC_OTHER,
            "identity": {"stat_var": "DebtServicePaid"},
        }
    )

    issues = check_evidence(
        requirement(),
        evidence_from_manifest(manifest, series_stats=_default_series_stats()),
        approved=source(),
    )

    assert EvidenceIssueCode.SOURCE_DRIFT in [i.code for i in errors(issues)]


def test_a_manifest_citation_matching_the_stated_source_is_not_flagged(
    requirement: RequirementFactory, source: SourceFactory
) -> None:
    manifest = _manifest(
        citation={
            "text": "World Bank",
            "sources": [{"ref": "worldbank/ids", "label": "World Bank IDS"}],
        }
    )

    issues = check_evidence(
        requirement(source="World Bank IDS"),
        evidence_from_manifest(manifest, series_stats=_default_series_stats()),
        approved=source(),
    )

    assert errors(issues) == []


def test_a_manifest_citation_contradicting_the_stated_source_is_a_mismatch(
    requirement: RequirementFactory, source: SourceFactory
) -> None:
    manifest = _manifest(
        citation={
            "text": "OECD",
            "sources": [{"ref": "oecd/dac/crs", "label": "OECD DAC CRS"}],
        }
    )

    issues = check_evidence(
        requirement(source="World Bank IDS"),
        evidence_from_manifest(manifest, series_stats=_default_series_stats()),
        approved=source(),
    )

    (issue,) = errors(issues)
    assert issue.code is EvidenceIssueCode.SOURCE_MISMATCH
    assert issue.found == "oecd/dac/crs OECD DAC CRS"


def test_a_manifest_citation_spelled_without_spaces_matches_the_stated_source(
    requirement: RequirementFactory, source: SourceFactory
) -> None:
    manifest = _manifest(
        citation={
            "text": "World Development Indicators",
            "sources": [
                {"ref": "dc/base/WorldDevelopmentIndicators", "label": "WorldDevelopmentIndicators"}
            ],
        }
    )

    issues = check_evidence(
        requirement(source="World Development Indicators"),
        evidence_from_manifest(manifest, series_stats=_default_series_stats()),
        approved=source(),
    )

    assert errors(issues) == []


def test_a_manifest_without_a_unit_is_an_error_when_the_requirement_states_one(
    requirement: RequirementFactory, source: SourceFactory
) -> None:
    manifest = _manifest()
    series_stats = _series_stats(_stat("DebtServicePaid", facet={}))

    issues = check_evidence(
        requirement(unit=Unit(currency="USD", prices="current")),
        evidence_from_manifest(manifest, series_stats=series_stats),
        approved=source(),
    )

    (issue,) = errors(issues)
    assert issue.code is EvidenceIssueCode.UNIT_UNVERIFIED
    assert issue.severity is Severity.ERROR


_MATCHES = ("unit_matches", "info")
_BASIS_UNSTATED = ("unit_price_basis_unverified", "warning")


@pytest.mark.parametrize(
    ("unit", "expected"),
    [
        # Data Commons unit dcids, as apis serves the CRS and ONE facets.
        ("CurrencyUSD", [_MATCHES, _BASIS_UNSTATED]),
        ("dcid:USDollar", [_MATCHES, _BASIS_UNSTATED]),
        ("CurrencyEUR", [("unit_mismatch", "error"), _BASIS_UNSTATED]),
        ("ConstantUSD_2021", [_MATCHES, ("unit_price_basis_mismatch", "error")]),
        # International (PPP) dollars are not US dollars, though the UN's id names USD.
        ("undata/UNIT_MEASURE-CR_USD_PPP", [("unit_mismatch", "error"), _BASIS_UNSTATED]),
        ("InternationalDollar", [("unit_mismatch", "error"), _BASIS_UNSTATED]),
        # A unit named only in its label, bare or as a `{label}`-only facet, reads
        # as PPP before its `$` maps to USD.
        (
            "PPP (constant 2017 international $)",
            [("unit_mismatch", "error"), ("unit_price_basis_mismatch", "error")],
        ),
        ({"label": "International dollars"}, [("unit_mismatch", "error"), _BASIS_UNSTATED]),
        (
            {"label": "Purchasing power parity ($)"},
            [("unit_mismatch", "error"), _BASIS_UNSTATED],
        ),
        # The SDG/OECD unit ids state their price basis in `CU` and `CON`.
        ("SDG_CU_USD_M", [_MATCHES]),
        ("SDG_CON_2015_USD", [_MATCHES, ("unit_price_basis_mismatch", "error")]),
        # The field-by-field unit a mint serves when its definition binds the
        # unit by request.
        ({"currency": "USD", "prices": "current"}, [_MATCHES]),
        (
            {"currency": "USD", "prices": "constant", "base_year": 2021},
            [_MATCHES, ("unit_price_basis_mismatch", "error")],
        ),
        (
            {"currency": {"role": "destination"}, "prices": "current"},
            [("unit_unverified", "error"), _BASIS_UNSTATED],
        ),
    ],
)
def test_a_manifest_unit_is_read_by_its_structure(
    requirement: RequirementFactory, source: SourceFactory, unit: object, expected: object
) -> None:
    series_stats = _series_stats(_stat("DebtServicePaid", facet={"unit": unit}))

    issues = check_evidence(
        requirement(unit=Unit(currency="USD", prices="current")),
        evidence_from_manifest(_manifest(), series_stats=series_stats),
        approved=source(),
    )

    assert [(issue.code.value, issue.severity.value) for issue in issues] == expected


def test_a_manifest_with_no_rows_is_an_error(
    requirement: RequirementFactory, source: SourceFactory
) -> None:
    empty = _manifest(coverage={"rows": 0, "dimensions": []})

    issues = check_evidence(
        requirement(),
        evidence_from_manifest(empty, series_stats=_default_series_stats()),
        approved=source(),
    )

    assert EvidenceIssueCode.ROWS_EMPTY in [i.code for i in errors(issues)]


# --- Open axes: a population covered whole ----------------------------------------

DONORS = OpenEntityAxis(column="origin", member_count=55)
RECIPIENTS = OpenEntityAxis(column="destination", member_count=140)
COUNTRIES = OpenEntityAxis(column="observationAbout", member_count=193)
KENYA_DESTINATION = ResolvedEntity(role="destination", ref="country/KEN", label="Kenya")


def test_no_open_axis_answers_a_party_word() -> None:
    assert open_axes_for(("donor",), ()) == (
        "the source leaves no axis open, so it holds no population of every 'donor'"
    )


def test_axes_that_were_never_read_answer_no_party_word() -> None:
    # `None` is a source recorded without its open entity axes: nothing says
    # it leaves none open, so the reason asks for a fresh discovery.
    assert open_axes_for(("donor",), None) == (
        "the source was recorded without the axes it leaves open, so whether it holds every "
        "'donor' is unknown; discover the source again"
    )


@pytest.mark.parametrize(
    ("across", "lone"),
    [
        ("country", COUNTRIES),
        ("economy", COUNTRIES),
        ("donor", DONORS),
        ("Funder", DONORS),
        ("financingProvider", DONORS),
        ("recipient", RECIPIENTS),
        ("target", RECIPIENTS),
    ],
)
def test_a_lone_open_axis_answers_a_word_on_its_own_side(across: str, lone: OpenEntityAxis) -> None:
    # `observationAbout` is on neither side of a flow, so a word the lexicon
    # places on no side ("country", "economy") is the subject it holds.
    assert open_axes_for((across,), (lone,)) == (lone,)


@pytest.mark.parametrize(
    ("across", "lone"),
    [
        ("donors", DONORS),
        ("Donors", DONORS),
        ("providers", DONORS),
        ("creditors", DONORS),
        ("lenders", DONORS),
        ("recipients", RECIPIENTS),
        ("beneficiaries", RECIPIENTS),
        ("countries", COUNTRIES),
        ("economies", COUNTRIES),
    ],
)
def test_a_plural_party_answers_the_axis_its_singular_does(
    across: str, lone: OpenEntityAxis
) -> None:
    # Each token places as written or as its singular. A place word also
    # covers a lone flow axis (`open_axes_for`), so only a flow word is held
    # to its own side here.
    for axis in (DONORS, RECIPIENTS) if lone != COUNTRIES else (COUNTRIES,):
        assert (open_axes_for((across,), (axis,)) == (axis,)) is (axis == lone)


@pytest.mark.parametrize(
    ("across", "party_word", "stated_as"),
    [
        ("donor countries", "donor", "donors"),
        ("Donor countries", "donor", "donors"),
        ("donor country", "donor", "donors"),
        ("recipient country", "recipient", "recipients"),
        ("recipient economies", "recipient", "recipients"),
        ("source country", "source", "sources"),
    ],
)
def test_a_place_noun_beside_a_party_word_answers_no_axis_and_names_the_whole_party(
    across: str, party_word: str, stated_as: str
) -> None:
    # "Donor countries" is narrower than the giving side, as "bilateral
    # donor" is: the open `origin` axis holds agencies too, so the word
    # answers no axis and the reason names the word that covers them all.
    for axes in ((DONORS,), (RECIPIENTS,), (COUNTRIES,), (), (RECIPIENTS, DONORS)):
        assert open_axes_for((across,), axes) == (
            f"{across!r} narrows a party to its countries, and a source's open axis holds every "
            f"{party_word}, agencies included; state `across` as {stated_as!r} to open that "
            "whole side, then add a direct `{kind:place_level, role:<party>, level:Country, "
            "strict:true}` filter bound to that role and redirect the calculation through that "
            "filter, or name the countries in `entities`"
        )


@pytest.mark.parametrize(
    "across",
    [
        "African country",
        "G7 country",
        "low-income country",
        "bilateral donor",
        "DAC donor",
        "multilateral donor",
        "private donor",
        "non-donor",
        "donors to Kenya",
        "donors and recipients",
        "donor recipient",
        "financing_providers",
        "sector",
        "year",
        "region",
        "city",
        "agency",
        "-",
    ],
)
def test_a_word_outside_the_party_vocabulary_answers_no_axis(across: str) -> None:
    # An open axis holds every member of a party. A qualifier ("African",
    # "bilateral") or another noun ("sector", "year") describes a population
    # the axis does not hold, so ranking it would rank the whole axis under
    # a narrower name. A word on both sides of a flow names no one party.
    for axes in ((DONORS,), (RECIPIENTS,), (COUNTRIES,), (), (RECIPIENTS, DONORS)):
        assert open_axes_for((across,), axes) == (
            f"{across!r} is not a party a source's open axis holds: an open axis ranks every "
            "member it holds, and each `across` word states that whole party, such as "
            "'donor', 'recipient' or 'country'; a narrower population (a region, an income "
            "group, a donor type) needs its members named in `entities`, or a filter the plan "
            "can express"
        )


@pytest.mark.parametrize(
    ("across", "lone", "reason"),
    [
        (
            "donor",
            RECIPIENTS,
            "'donor' names the other side of the flow from the source's one open axis, "
            "'destination'",
        ),
        (
            "funder",
            RECIPIENTS,
            "'funder' names the other side of the flow from the source's one open axis, "
            "'destination'",
        ),
        (
            "recipient",
            COUNTRIES,
            "'recipient' names the receiving side of a flow, and the source's one open axis, "
            "'observationAbout', is no side of a flow; state `across` as the place noun, such "
            "as 'country'",
        ),
        (
            "donor",
            COUNTRIES,
            "'donor' names the giving side of a flow, and the source's one open axis, "
            "'observationAbout', is no side of a flow; state `across` as the place noun, such "
            "as 'country'",
        ),
    ],
)
def test_a_lone_open_axis_off_the_side_the_word_names_is_a_reason(
    across: str, lone: OpenEntityAxis, reason: str
) -> None:
    assert open_axes_for((across,), (lone,)) == reason


@pytest.mark.parametrize("across", ["country", "countries", "economy"])
@pytest.mark.parametrize("lone", [DONORS, RECIPIENTS])
def test_a_place_word_covers_the_one_flow_axis_open_over_several_members(
    across: str, lone: OpenEntityAxis
) -> None:
    # A source keyed by debtor leaves its countries open on `destination`,
    # and `observationAbout` is not open, so "country" covers that axis.
    one_other = OpenEntityAxis(
        column="origin" if lone is RECIPIENTS else "destination", member_count=1
    )

    assert open_axes_for((across,), (lone,)) == (lone,)
    assert open_axes_for((across,), (lone, one_other)) == (lone,)


def test_country_covers_the_ids_debtors_on_their_flow_axis() -> None:
    # IDS leaves its 134 debtors open on `destination` and no `observationAbout`
    # (dev, total and by-creditor series alike); the twin also opens 165 creditors.
    debtors = OpenEntityAxis(column="destination", member_count=134)
    creditors = OpenEntityAxis(column="origin", member_count=165)

    assert open_axes_for(("country",), (debtors,)) == (debtors,)
    assert open_axes_for(("creditor", "country"), (debtors, creditors)) == (creditors, debtors)


def test_a_place_word_prefers_the_place_axis_and_leaves_an_open_flow_axis_unstated() -> None:
    assert open_axes_for(("country",), (COUNTRIES,)) == (COUNTRIES,)
    assert open_axes_for(("country",), (DONORS, COUNTRIES)) == (
        "the source also leaves 'origin' open over several members, a party `across` does "
        "not state; add its party word to `across` to cover every member of it too, or name "
        "the members the answer needs in `entities`, so each 'country' is not repeated once "
        "per member of it"
    )
    assert open_axes_for(("donor", "country"), (DONORS, COUNTRIES)) == (DONORS, COUNTRIES)


@pytest.mark.parametrize(
    ("across", "covered"),
    [
        (("donor", "recipient"), (DONORS, RECIPIENTS)),
        (("recipients", "donors"), (RECIPIENTS, DONORS)),
        (("creditor", "country"), (DONORS, RECIPIENTS)),
        (("country", "lender"), (RECIPIENTS, DONORS)),
    ],
)
def test_across_covers_every_open_multi_member_axis_one_word_each(
    across: tuple[str, ...], covered: tuple[OpenEntityAxis, ...]
) -> None:
    # "All donors by all recipients": each word covers its own axis, in
    # `across` order. A place word beside a flow word takes the flow axis the
    # other word leaves.
    assert open_axes_for(across, (RECIPIENTS, DONORS)) == covered


def test_a_place_word_beside_two_open_flow_axes_is_a_reason() -> None:
    assert open_axes_for(("country",), (RECIPIENTS, DONORS)) == (
        "the source leaves 'destination', 'origin' open, each over several members, so "
        "'country' could be any of them; state each party the requirement covers in "
        "`across`, such as 'donor' and 'recipient'"
    )


@pytest.mark.parametrize("unknown", ["reportingAgency", "sector", "observation_about"])
def test_an_open_column_no_party_word_names_answers_no_word(unknown: str) -> None:
    # Only `observationAbout` is the place. Any other column the lexicon
    # cannot place holds a party nothing in the plan can state.
    lone = OpenEntityAxis(column=unknown, member_count=12)

    for across in ("donor", "recipient"):
        assert open_axes_for((across,), (lone,)) == (
            f"the source's one open axis, {unknown!r}, is a column no party word names, so it "
            f"cannot be read as holding every {across!r}"
        )
    assert open_axes_for(("country",), (lone,)) == (
        f"the source leaves {unknown!r} open, and none of them holds several places a "
        "requirement over every 'country' could cover"
    )


@pytest.mark.parametrize("across", ["donor", "recipient"])
@pytest.mark.parametrize("other_count", [140, 2, 0, None])
def test_a_second_open_axis_holding_several_members_or_uncounted_is_a_reason(
    across: str, other_count: int | None
) -> None:
    # A second open axis is a party the requirement neither names nor states:
    # each member of the population would repeat once per member of it. An
    # uncounted axis may hold several.
    column = "destination" if across == "donor" else "origin"
    other = OpenEntityAxis(column=column, member_count=other_count)
    own = RECIPIENTS if across == "recipient" else DONORS
    axes = tuple(sorted((other, own), key=lambda axis: axis.column))

    assert open_axes_for((across,), axes) == (
        f"the source also leaves {column!r} open over several members, a party `across` does "
        "not state; add its party word to `across` to cover every member of it too, or name "
        f"the members the answer needs in `entities`, so each {across!r} is not repeated once "
        "per member of it"
    )


def test_a_second_open_axis_holding_one_member_leaves_the_population_whole() -> None:
    # One recipient on the open `destination` axis repeats nothing: each
    # donor still carries one series.
    one_recipient = OpenEntityAxis(column="destination", member_count=1)

    assert open_axes_for(("donor",), (one_recipient, DONORS)) == (DONORS,)
    assert open_axes_for(("country",), (one_recipient, COUNTRIES)) == (COUNTRIES,)


def test_several_open_axes_none_on_the_words_party_are_a_reason_naming_them() -> None:
    assert open_axes_for(("donor",), (RECIPIENTS, COUNTRIES)) == (
        "the source leaves 'destination', 'observationAbout' open, and exactly one of them "
        "has to hold the giving side of the flow 'donor' names"
    )


@pytest.mark.parametrize(
    ("member_count", "reason"),
    [
        (
            None,
            "the source does not count the members of its open axis 'origin', so it cannot be "
            "shown to hold every 'donor'; name the entities the answer needs in `entities` "
            "instead of stating that party in `across`",
        ),
        (
            1,
            "the source holds 1 member on its open axis 'origin', which is no population of "
            "every 'donor'; name the entities the answer needs in `entities` instead of "
            "stating that party in `across`",
        ),
        (
            0,
            "the source holds 0 members on its open axis 'origin', which is no population of "
            "every 'donor'; name the entities the answer needs in `entities` instead of "
            "stating that party in `across`",
        ),
    ],
)
def test_an_open_axis_search_did_not_count_or_holding_fewer_than_two_answers_no_word(
    member_count: int | None, reason: str
) -> None:
    assert open_axes_for(
        ("donor",), (OpenEntityAxis(column="origin", member_count=member_count),)
    ) == (reason)
    assert open_axes_for(("donor",), (OpenEntityAxis(column="origin", member_count=2),)) == (
        OpenEntityAxis(column="origin", member_count=2),
    )


@pytest.mark.parametrize("member_count", [None, 0, 1])
def test_a_source_whose_open_axis_holds_no_population_is_open_axis_unresolved(
    requirement: RequirementFactory, source: SourceFactory, member_count: int | None
) -> None:
    uncounted = source(
        entities=(KENYA_DESTINATION,),
        open_entity_axes=(OpenEntityAxis(column="origin", member_count=member_count),),
    )

    (issue,) = errors(
        check_evidence(
            requirement(entities=[Entity(name="Kenya", role="recipient")], across=("donor",)),
            evidence_from_source(uncounted),
        )
    )

    assert issue.code is EvidenceIssueCode.OPEN_AXIS_UNRESOLVED
    assert (issue.found, issue.required) == (["origin"], ["donor"])
    assert "instead of stating that party in `across`" in issue.detail


@pytest.mark.parametrize(
    ("axes", "opens"),
    [
        ((COUNTRIES,), True),
        ((OpenEntityAxis(column="observationAbout", member_count=None),), True),
        ((OpenEntityAxis(column="observationAbout", member_count=1),), False),
        ((DONORS, RECIPIENTS), False),
        ((), False),
        (None, False),
    ],
)
def test_a_source_opens_its_place_axis_over_more_than_one_place(
    axes: tuple[OpenEntityAxis, ...] | None, opens: bool
) -> None:
    # One member is a single place, as Earth is for a worldwide total.
    assert opens_place_axis(axes) is opens


def test_a_place_axis_left_open_under_no_across_is_open_axis_unresolved(
    requirement: RequirementFactory, source: SourceFactory
) -> None:
    # Kenya resolves under `destination`, and `observationAbout` holds every
    # place with data at every level: a total would sum them all unless the
    # requirement states the place party, which the plan's level rule reads.
    open_place = source(entities=(KENYA_DESTINATION,), open_entity_axes=(COUNTRIES,))

    (issue,) = errors(
        check_evidence(
            requirement(entities=[Entity(name="Kenya", role="recipient")]),
            evidence_from_source(open_place),
        )
    )

    assert issue.code is EvidenceIssueCode.OPEN_AXIS_UNRESOLVED
    assert issue.found == ["observationAbout"]
    assert "leaves its place axis 'observationAbout' open" in issue.detail
    assert "state the place party, such as 'country'" in issue.detail


def test_a_place_axis_holding_one_place_needs_no_across(
    requirement: RequirementFactory, source: SourceFactory
) -> None:
    earth = source(
        entities=(KENYA_DESTINATION,),
        open_entity_axes=(OpenEntityAxis(column="observationAbout", member_count=1),),
    )

    assert (
        errors(
            check_evidence(
                requirement(entities=[Entity(name="Kenya", role="recipient")]),
                evidence_from_source(earth),
            )
        )
        == []
    )


# apis `search/engine/role_text.py`, `_ROLE_TEXT_TO_AXIS`: every word
# `search` places on a flow axis, and the axis it places it on.
_SEARCH_ROLE_TEXT_TO_AXIS = {
    "from": "origin",
    "donor": "origin",
    "provider": "origin",
    "funder": "origin",
    "source": "origin",
    "creditor": "origin",
    "lender": "origin",
    "exporter": "origin",
    "financingprovider": "origin",
    "to": "destination",
    "recipient": "destination",
    "beneficiary": "destination",
    "destination": "destination",
    "debtor": "destination",
    "borrower": "destination",
    "importer": "destination",
    "target": "destination",
    "financingrecipient": "destination",
}


@pytest.mark.parametrize(("word", "axis"), sorted(_SEARCH_ROLE_TEXT_TO_AXIS.items()))
def test_every_word_search_places_answers_the_axis_search_places_it_on(
    word: str, axis: str
) -> None:
    by_column = {open_axis.column: open_axis for open_axis in (DONORS, RECIPIENTS)}
    other = "destination" if axis == "origin" else "origin"

    assert open_axes_for((word,), (by_column[axis],)) == (by_column[axis],)
    assert isinstance(open_axes_for((word,), (by_column[other],)), str)
    assert isinstance(open_axes_for((word,), (COUNTRIES,)), str)


def test_a_population_the_source_leaves_open_needs_no_named_entity(
    requirement: RequirementFactory, source: SourceFactory
) -> None:
    codes = _codes(
        requirement(entities=[], across=("donor",)), source(entities=(), open_entity_axes=(DONORS,))
    )

    assert not any(severity == "error" for _, severity in codes)
    assert not any(code.startswith(("entity_", "open_axis")) for code, _ in codes)


def test_a_population_no_open_axis_holds_is_an_error(
    requirement: RequirementFactory, source: SourceFactory
) -> None:
    issues = check_evidence(
        requirement(entities=[], across=("donor",)),
        evidence_from_source(source(entities=(), open_entity_axes=())),
    )

    (issue,) = errors(issues)
    assert issue.code is EvidenceIssueCode.OPEN_AXIS_UNRESOLVED
    assert (issue.found, issue.required) == ([], ["donor"])
    assert issue.detail == (
        "the source leaves no axis open, so it holds no population of every 'donor'"
    )
    assert "entity_unverified" not in [i.code.value for i in issues]


def test_a_source_recorded_without_its_open_axes_does_not_read_as_leaving_none_open(
    requirement: RequirementFactory,
) -> None:
    # A source dumped before `open_entity_axes` existed restores with the
    # field unset. The recorded source leaves `origin` open, so "leaves no
    # axis open" would be false of it.
    dumped = _recorded_role_open_source().model_dump(mode="json")
    del dumped["open_entity_axes"]
    restored = SelectedSource.model_validate(dumped)
    asked = requirement(
        entities=[Entity(name="Kenya", role="recipient")],
        across=("donor",),
        period=Period(start=2015, end=2023),
    )

    (issue,) = errors(check_evidence(asked, evidence_from_source(restored)))

    assert restored.open_entity_axes is None
    assert issue.code is EvidenceIssueCode.OPEN_AXIS_UNRESOLVED
    assert "recorded without the axes it leaves open" in issue.detail
    assert "discover the source again" in issue.detail


def test_a_requirement_naming_an_entity_and_a_population_checks_both(
    requirement: RequirementFactory, source: SourceFactory
) -> None:
    # Education aid to Kenya from every donor: Kenya resolves under
    # `destination`, and the donors are the open `origin` axis.
    asked = requirement(entities=[Entity(name="Kenya", role="recipient")], across=("donor",))

    assert (
        errors(
            check_evidence(
                asked,
                evidence_from_source(
                    source(entities=(KENYA_DESTINATION,), open_entity_axes=(DONORS,))
                ),
            )
        )
        == []
    )
    (issue,) = errors(
        check_evidence(
            asked,
            evidence_from_source(source(entities=(KENYA_DESTINATION,), open_entity_axes=())),
        )
    )
    assert issue.code is EvidenceIssueCode.OPEN_AXIS_UNRESOLVED


def _recorded_role_open_source() -> SelectedSource:
    """dev `search`'s own source for education aid to Kenya, 2015 to 2023, every donor left open."""
    fixture = Path(__file__).parent.parent / "fixtures" / "search_spec_role_open.json"
    return source_from_spec_block(json.loads(fixture.read_text())["spec"])


def test_a_recorded_open_donor_side_answers_a_requirement_across_donors(
    requirement: RequirementFactory,
) -> None:
    asked = requirement(
        entities=[Entity(name="Kenya", role="recipient")],
        across=("donor",),
        period=Period(start=2015, end=2023),
    )

    issues = check_evidence(asked, evidence_from_source(_recorded_role_open_source()))

    assert errors(issues) == []
    assert not {issue.code.value for issue in issues} & {
        "entity_unverified",
        "entity_role_unverified",
        "entity_role_mismatch",
        "open_axis_unresolved",
    }


def test_a_recorded_open_donor_side_does_not_answer_a_requirement_across_recipients(
    requirement: RequirementFactory,
) -> None:
    # The lone open axis is `origin`, the giving side, so a population of
    # recipients has no column to rank or total over.
    asked = requirement(entities=[], across=("recipient",), period=Period(start=2015, end=2023))

    issues = check_evidence(asked, evidence_from_source(_recorded_role_open_source()))

    assert EvidenceIssueCode.OPEN_AXIS_UNRESOLVED in [issue.code for issue in errors(issues)]


def _donor_stats(*entities: dict[str, str]) -> pa.Table:
    return _series_stats(_stat("DebtServicePaid", entities=list(entities)))


# The apis service's mint for education aid to Kenya from every donor: the manifest as
# served and the first rows of its series statistics. The series name each
# entity's role by declared property (`financingProvider`), where Search names
# the open donor side `origin`, and the manifest's coverage dimensions state
# each property's flow axis.
_RECORDED_DONOR_MINT: dict[str, Any] = json.loads(
    (Path(__file__).parent.parent / "fixtures" / "mint_manifest_donor_open_axis.json").read_text()
)
_FLOW_DIMENSIONS: list[dict[str, Any]] = _RECORDED_DONOR_MINT["manifest"]["coverage"]["dimensions"]
_KENYA_REF = {"role": "financingRecipient", "ref": "country/KEN"}


def _donor_ref(ref: str) -> dict[str, str]:
    return {"role": "financingProvider", "ref": ref}


def _flow_manifest(**overrides: object) -> Manifest:
    """`_manifest` for a flow dataset, its coverage dimensions the recorded mint's."""
    return _manifest(coverage={"rows": 10, "dimensions": _FLOW_DIMENSIONS}, **overrides)


def _kenya_from(*donors: str) -> pa.Table:
    """One series of aid to Kenya per donor ref in `donors`."""
    return _series_stats(
        *(_stat("DebtServicePaid", entities=[_KENYA_REF, _donor_ref(donor)]) for donor in donors)
    )


def _recorded_donor_retrieval(*, axes: bool = True) -> RetrievalEvidence:
    """The recorded mint read as evidence; `axes=False` drops each dimension's `axis`,
    as an apis service that predates the field serves it."""
    manifest = dict(_RECORDED_DONOR_MINT["manifest"])
    if not axes:
        coverage = manifest["coverage"]
        manifest["coverage"] = coverage | {
            "dimensions": [
                {key: value for key, value in dimension.items() if key != "axis"}
                for dimension in coverage["dimensions"]
            ]
        }
    series = pa.Table.from_pylist(
        _RECORDED_DONOR_MINT["series_statistics"], schema=SERIES_STATISTICS_ARROW_SCHEMA
    )
    return evidence_from_manifest(Manifest.model_validate(manifest), series_stats=series)


def test_a_recorded_retrieval_holds_the_open_donor_axis_under_its_declared_property(
    requirement: RequirementFactory, source: SourceFactory
) -> None:
    # Search states the open donor side as `origin`, and the mint names the
    # donors under `financingProvider`, which its coverage dimensions place
    # on the `origin` axis.
    approved = source(
        entities=(KENYA_DESTINATION,),
        open_entity_axes=_recorded_role_open_source().open_entity_axes,
    )
    evidence = _recorded_donor_retrieval()
    assert approved.open_entity_axes == (DONORS,)
    assert {entity.role for entity in evidence.entities} == {
        "financingProvider",
        "financingRecipient",
    }

    issues = check_evidence(_kenya_from_every_donor(requirement), evidence, approved=approved)

    assert EvidenceIssueCode.OPEN_AXIS_UNRESOLVED not in [issue.code for issue in issues]


def test_a_recorded_retrieval_stating_no_axes_holds_nothing_on_the_open_donor_axis(
    requirement: RequirementFactory, source: SourceFactory
) -> None:
    # Without the dimensions' `axis` nothing ties `financingProvider` to
    # `origin`, so the population cannot be shown to have come back.
    approved = source(entities=(KENYA_DESTINATION,), open_entity_axes=(DONORS,))

    issues = check_evidence(
        _kenya_from_every_donor(requirement),
        _recorded_donor_retrieval(axes=False),
        approved=approved,
    )

    (issue,) = [one for one in issues if one.code is EvidenceIssueCode.OPEN_AXIS_UNRESOLVED]
    assert issue.detail == (
        "the retrieval holds no entity on the 'origin' axis, which the plan was approved to "
        "read as every 'donor'; a population holds at least 2"
    )


def test_a_retrieval_holding_two_entities_under_the_open_column_covers_the_population(
    requirement: RequirementFactory, source: SourceFactory
) -> None:
    # The approved axis counts 55 donors over every year, and the window may
    # hold fewer, so two is the bar, as it is before approval.
    approved = source(entities=(KENYA_DESTINATION,), open_entity_axes=(DONORS,))
    evidence = evidence_from_manifest(
        _flow_manifest(), series_stats=_kenya_from("country/FRA", "country/DEU")
    )

    issues = check_evidence(
        requirement(entities=[Entity(name="Kenya", role="recipient")], across=("donor",)),
        evidence,
        approved=approved,
    )

    assert errors(issues) == []


def test_a_retrieval_holding_one_entity_under_the_open_column_is_an_error(
    requirement: RequirementFactory, source: SourceFactory
) -> None:
    # France alone is a ranking of one donor under a plan approved to rank
    # every donor the source holds.
    approved = source(entities=(KENYA_DESTINATION,), open_entity_axes=(DONORS,))
    evidence = evidence_from_manifest(_flow_manifest(), series_stats=_kenya_from("country/FRA"))

    (issue,) = errors(
        check_evidence(
            requirement(entities=[Entity(name="Kenya", role="recipient")], across=("donor",)),
            evidence,
            approved=approved,
        )
    )

    assert issue.code is EvidenceIssueCode.OPEN_AXIS_UNRESOLVED
    assert (issue.found, issue.required) == (["country/FRA"], "donor")
    assert issue.detail == (
        "the retrieval holds 1 entity under the 'financingProvider' column, which the plan "
        "was approved to read as every 'donor'; a population holds at least 2"
    )


def test_a_retrieval_naming_nothing_under_the_open_column_is_an_error(
    requirement: RequirementFactory, source: SourceFactory
) -> None:
    # The approved source says which axis was open; the manifest states no
    # open axes of its own, only the entities it returned and the column on
    # each flow axis.
    approved = source(entities=(KENYA_DESTINATION,), open_entity_axes=(DONORS,))
    evidence = evidence_from_manifest(
        _flow_manifest(),
        series_stats=_series_stats(_stat("DebtServicePaid", entities=[_KENYA_REF])),
    )

    (issue,) = errors(
        check_evidence(
            requirement(entities=[Entity(name="Kenya", role="recipient")], across=("donor",)),
            evidence,
            approved=approved,
        )
    )
    assert issue.code is EvidenceIssueCode.OPEN_AXIS_UNRESOLVED
    assert (issue.found, issue.required) == ([], "donor")
    assert "holds no entity under the 'financingProvider' column" in issue.detail


def _summed_donor_family() -> tuple[SelectedSource, Manifest]:
    """The recorded education-aid source approved to be read summed, and a manifest for it.

    The manifest's trace serves two StatVars and states the family digest the
    recorded block states, so membership is confirmed.
    """
    approved = stat_var_reading(_recorded_role_open_source(), StatVarReading.SUM)
    refs = [
        "Amount_FinancialFlow_Gross_Disbursement_CRSPurposeCode11110_Nominal",
        "Amount_FinancialFlow_Gross_Disbursement_CRSPurposeCode11120_Nominal",
    ]
    manifest = _flow_manifest(
        definition={
            "schema_version": 1,
            "spec_id": approved.spec_id,
            "identity": approved.identity,
        },
        trace={
            "resolved_stat_vars": [
                {"ref": f"dcid:{ref}", "roles": ["financingProvider", "financingRecipient"]}
                for ref in refs
            ],
            "family_member_digest": approved.shape_member_digest,
            "family_member_count": approved.shape_member_count,
        },
        citation={"text": "OECD CRS", "sources": []},
    )
    return approved, manifest


def _summed_series(*donors: str) -> pa.Table:
    """The series a summed retrieval carries: one summed StatVar ref per donor, 2015 to 2023.

    With no donor, one series naming Kenya alone, the frame a sum that
    collapsed the donor axis would carry.
    """
    entities = [[_KENYA_REF, _donor_ref(donor)] for donor in donors] or [[_KENYA_REF]]
    return _series_stats(
        *(
            _stat(
                "Amount_FinancialFlow_family_sum",
                entities=one,
                observations=9,
                latest={"value": 2.0, "period": "2023"},
                maximum={"value": 2.0, "period": "2023"},
            )
            for one in entities
        )
    )


def _kenya_from_every_donor(requirement: RequirementFactory) -> DatasetRequirement:
    return requirement(
        measure="education aid",
        entities=[Entity(name="Kenya", role="recipient")],
        across=("donor",),
        period=Period(start=2015, end=2023),
        unit=None,
    )


def test_a_summed_retrieval_keeping_its_donors_covers_the_population(
    requirement: RequirementFactory,
) -> None:
    # apis sums the family and keeps every entity role column, so the
    # summed frame carries one series per donor under `financingProvider`.
    approved, manifest = _summed_donor_family()
    donors = ("country/FRA", "country/DEU", "country/USA", "country/JPN")
    evidence = evidence_from_manifest(manifest, series_stats=_summed_series(*donors))

    issues = check_evidence(_kenya_from_every_donor(requirement), evidence, approved=approved)

    assert approved.stat_var_reading is StatVarReading.SUM
    assert errors(issues) == []


def test_a_summed_retrieval_that_dropped_its_donors_is_an_error(
    requirement: RequirementFactory,
) -> None:
    # A sum that collapsed the donor axis leaves one series for Kenya and
    # nothing to rank.
    approved, manifest = _summed_donor_family()
    evidence = evidence_from_manifest(manifest, series_stats=_summed_series())

    issues = check_evidence(_kenya_from_every_donor(requirement), evidence, approved=approved)

    assert [issue.code for issue in errors(issues)] == [EvidenceIssueCode.OPEN_AXIS_UNRESOLVED]


# --- Resolved binding membership: approved versus executed -----------------------


_SCHEMA_IDENTITY = {"shape": {"kind": "statvar_schema", "measured_property": "amount"}}
_POPULATION_FAMILY = ["Count_Person", "Count_Person_Female", "Count_Person_Male"]


def _membership(
    requirement: RequirementFactory,
    approved: SelectedSource,
    *served: str,
    family: Sequence[str] | None = None,
) -> list[EvidenceIssue]:
    """The `STAT_VAR_MEMBERSHIP_DRIFT` findings for a retrieval serving `served`, whose
    trace states the digest of `family`, or no family digest when `family` is `None`."""
    trace: dict[str, object] = _trace(*served)
    if family is not None:
        trace |= _family_trace(family)
    manifest = _manifest(
        definition={"schema_version": 1, "spec_id": SPEC, "identity": approved.identity},
        trace=trace,
    )
    issues = check_evidence(
        requirement(),
        evidence_from_manifest(manifest, series_stats=_series_stats(*map(_stat, served))),
        approved=approved,
    )
    return [i for i in issues if i.code is EvidenceIssueCode.STAT_VAR_MEMBERSHIP_DRIFT]


def _family_source(source: SourceFactory, **overrides: object) -> SelectedSource:
    """A described source `search` approved as the population family, stating its digest."""
    fields: dict[str, object] = {
        "identity": _SCHEMA_IDENTITY,
        "served_count": None,
        "shape_member_count": len(_POPULATION_FAMILY),
        "shape_member_digest": member_digest(_POPULATION_FAMILY),
    }
    return source(**(fields | overrides))


def test_a_family_whose_trace_states_the_approved_digest_has_no_membership_finding(
    requirement: RequirementFactory, source: SourceFactory
) -> None:
    """The mint serves the family reduced to its root; the trace's family digest is
    the approved one, so the reduction is no drift."""
    approved = _family_source(source)

    assert _membership(requirement, approved, "Count_Person", family=_POPULATION_FAMILY) == []


@pytest.mark.parametrize(
    "family",
    [
        ["Count_Person", "Count_Person_Female"],
        ["Count_Person", "Count_Person_Female", "Count_Person_Urban"],
        [*_POPULATION_FAMILY, "Count_Person_Urban"],
    ],
    ids=["lost-a-member", "swapped-a-member", "gained-a-member"],
)
def test_a_family_whose_trace_states_another_digest_is_drift(
    requirement: RequirementFactory, source: SourceFactory, family: list[str]
) -> None:
    approved = _family_source(source)

    (drift,) = _membership(requirement, approved, "Count_Person", family=family)

    assert drift.severity is Severity.ERROR
    assert drift.found == {"member_count": len(family), "member_digest": member_digest(family)}
    assert drift.required == {
        "member_count": 3,
        "member_digest": member_digest(_POPULATION_FAMILY),
    }


@pytest.mark.parametrize(
    ("approved_digest", "family"),
    [(False, _POPULATION_FAMILY), (True, None)],
    ids=["no-approved-digest", "no-trace-digest"],
)
def test_a_side_stating_no_digest_leaves_membership_unverified(
    requirement: RequirementFactory,
    source: SourceFactory,
    approved_digest: bool,
    family: list[str] | None,
) -> None:
    """`search` states no digest for a walk it could not finish, and the mint none for
    a family it could not state whole: membership is unverified, never confirmed."""
    approved = _family_source(source, **({} if approved_digest else {"shape_member_digest": None}))

    (drift,) = _membership(requirement, approved, "Count_Person", family=family)

    assert drift.severity is Severity.WARNING


@pytest.mark.parametrize(
    ("served", "found"),
    [
        (("DebtServicePaid",), []),
        (("DebtServiceReceived",), [EvidenceIssueCode.STAT_VAR_MEMBERSHIP_DRIFT]),
    ],
    ids=["its-own-dcid", "another-dcid"],
)
def test_a_pinned_identity_is_held_to_the_dcid_it_names(
    requirement: RequirementFactory,
    source: SourceFactory,
    served: tuple[str, ...],
    found: list[EvidenceIssueCode],
) -> None:
    """A pinned identity names its one dcid, so no digest is needed to check it."""
    approved = source(
        identity={"shape": {"kind": "stat_var", "id": "dcid:DebtServicePaid"}},
        shape_member_digest=None,
    )

    issues = _membership(requirement, approved, *served)

    assert [issue.code for issue in issues if issue.severity is Severity.ERROR] == found


@pytest.mark.parametrize(
    ("served", "drift"),
    [(("Count_Person",), False), (tuple(_POPULATION_FAMILY), True)],
    ids=["one-served", "three-served"],
)
def test_a_source_approved_as_one_series_is_held_to_one_served_stat_var(
    requirement: RequirementFactory, source: SourceFactory, served: tuple[str, ...], drift: bool
) -> None:
    """`served_count` of one made the plan read the source as one series, so a
    retrieval serving more is drift whatever the family's digest says."""
    approved = _family_source(source, served_count=1)

    issues = _membership(requirement, approved, *served, family=_POPULATION_FAMILY)

    assert [issue.found for issue in issues] == ([len(served)] if drift else [])


def test_an_unapproved_identity_still_gets_the_membership_check(
    requirement: RequirementFactory, source: SourceFactory
) -> None:
    """An unknown identity is the case with the least protection, so it must not
    also get the least checking: a family that moved is still drift."""
    approved = _family_source(source, identity=None)
    manifest = _manifest(
        definition={"schema_version": 1, "spec_id": SPEC, "identity": {"stat_var": "X"}},
        trace=_trace("Count_Person") | _family_trace(["Count_Person_Urban"]),
    )

    issues = check_evidence(
        requirement(),
        evidence_from_manifest(manifest, series_stats=_series_stats(_stat("Count_Person"))),
        approved=approved,
    )

    codes = [i.code for i in issues]
    assert EvidenceIssueCode.IDENTITY_UNVERIFIED in codes
    (drift,) = [i for i in errors(issues) if i.code is EvidenceIssueCode.STAT_VAR_MEMBERSHIP_DRIFT]
    assert drift.severity is Severity.ERROR


_FIXTURES = Path(__file__).parent.parent / "fixtures"


def test_member_digest_matches_the_digest_dfo_apis_computed() -> None:
    """The recorded "education grants from France, 2020-2024" block states a 183-member
    family digest, and the recorded mint's trace serves those 183 members."""
    block = json.loads((_FIXTURES / "family_spec_blocks.json").read_text())[
        "france_education_grants"
    ]["parts"][0]["spec"]
    trace = json.loads((_FIXTURES / "mint_trace_france_education_grants.json").read_text())
    approved = source_from_spec_block(block)
    entries = trace["trace"]["resolved_stat_vars"]

    assert trace["spec_id"] == block["spec_id"]
    assert approved.shape_member_digest == (
        "7f6018987b665147990da0af3bd8a7b41b77d5acbd0c67d06cc81fcddf26ff83"
    )
    assert member_digest(e["ref"].removeprefix("dcid:") for e in entries) == (
        approved.shape_member_digest
    )


def test_a_trace_family_digest_is_read_off_the_manifest() -> None:
    manifest = _manifest(trace=_trace("Count_Person") | _family_trace(_POPULATION_FAMILY))
    stated = evidence_from_manifest(manifest, series_stats=_default_series_stats())
    silent = evidence_from_manifest(
        _manifest(
            trace=_trace("Count_Person")
            | {"family_member_digest": None, "family_member_count": None},
        ),
        series_stats=_default_series_stats(),
    )

    assert (stated.family_member_digest, stated.family_member_count) == (
        member_digest(_POPULATION_FAMILY),
        3,
    )
    assert (silent.family_member_digest, silent.family_member_count) == (None, None)


def test_the_code_of_a_substituted_retrieval_still_loads() -> None:
    # No check reports it; a run saved before the mint kept the spec_id holds it.
    saved = {
        "code": "source_substituted",
        "severity": "warning",
        "requirement_id": "annual_debt_service",
        "detail": "the retrieval is data-mcp's substitution mint",
    }

    assert EvidenceIssue.model_validate(saved).code is EvidenceIssueCode.SOURCE_SUBSTITUTED


def test_an_identity_that_drifted_stacks_no_membership_finding(
    requirement: RequirementFactory, source: SourceFactory
) -> None:
    issues = check_evidence(
        requirement(),
        evidence_from_manifest(_manifest(), series_stats=_default_series_stats()),
        approved=source(identity={"stat_var": "SomethingElse"}),
    )

    assert [i.code for i in errors(issues)] == [EvidenceIssueCode.SOURCE_DRIFT]
    assert EvidenceIssueCode.STAT_VAR_MEMBERSHIP_DRIFT not in [i.code for i in issues]


def test_a_manifest_whose_trace_states_no_resolved_stat_vars_warns(
    requirement: RequirementFactory, source: SourceFactory
) -> None:
    # A manifest minted before the field existed, or any other trace that
    # simply carries no `resolved_stat_vars` key: membership must not fall
    # back to comparing against `statistics`, the coverage-derived set, so
    # it reports the binding as unverifiable instead.
    manifest = _manifest(trace={})

    issues = check_evidence(
        requirement(),
        evidence_from_manifest(manifest, series_stats=_default_series_stats()),
        approved=source(),
    )

    (issue,) = [i for i in issues if i.code is EvidenceIssueCode.STAT_VAR_MEMBERSHIP_DRIFT]
    assert issue.severity is Severity.WARNING


def test_a_manifest_missing_statistics_but_with_an_intact_binding_reports_no_membership_finding(
    requirement: RequirementFactory, source: SourceFactory
) -> None:
    # Missing statistics is a coverage gap (`STATISTICS_UNAVAILABLE`,
    # `_series_issues`'s own concern) and says nothing about the binding,
    # which `trace.resolved_stat_vars` -- `_manifest`'s default -- still
    # states intact here. `series_stats` defaults to `None`, matching a
    # manifest whose `series_count` is itself `None` or whose fetch failed.
    manifest = _manifest()

    issues = check_evidence(
        requirement(coverage=Coverage(on_missing="report_gaps"), unit=None),
        evidence_from_manifest(manifest),
        approved=source(),
    )

    assert [i.code for i in errors(issues)] == [EvidenceIssueCode.STATISTICS_UNAVAILABLE]
    assert [i for i in issues if i.code is EvidenceIssueCode.STAT_VAR_MEMBERSHIP_DRIFT] == []


def test_check_evidence_with_no_approved_source_reports_no_membership_finding(
    requirement: RequirementFactory,
) -> None:
    issues = check_evidence(
        requirement(),
        evidence_from_manifest(_manifest(), series_stats=_default_series_stats()),
    )

    assert [i for i in issues if i.code is EvidenceIssueCode.STAT_VAR_MEMBERSHIP_DRIFT] == []


# --- A large statistics artifact is read column by column ---------------------


def _crs_shaped_series(count: int, *, off_period: frozenset[int] = frozenset()) -> pa.Table:
    """`count` series naming a donor, a recipient and a sector each, the way a CRS
    retrieval does, all covering 2015-2024 annually except the rows in `off_period`,
    which have three years missing."""
    return pa.Table.from_pylist(
        [
            {
                "stat_var": f"dcid:CRS_Disbursement_{i % 40}",
                "entities": [
                    {"role": "donor", "ref": f"country/D{i % 30}"},
                    {"role": "recipient", "ref": f"country/R{i % 150}"},
                    {"role": "sector", "ref": f"sector/{i % 200}"},
                ],
                "facet": '{"unit": "USD"}',
                "observations": 7 if i in off_period else 10,
                "first_value": 1.0,
                "latest_value": 2.0,
                "minimum_value": 1.0,
                "maximum_value": 2.0,
                "first_period": "2015",
                "latest_period": "2024",
                "minimum_period": "2015",
                "maximum_period": "2024",
                "change_absolute": 1.0,
                "change_percent": 100.0,
            }
            for i in range(count)
        ],
        schema=SERIES_STATISTICS_ARROW_SCHEMA,
    )


def test_the_entities_of_a_large_artifact_keep_the_order_each_first_appears() -> None:
    series = _crs_shaped_series(25_000)
    expected: list[tuple[str, str]] = []
    for named in series["entities"].to_pylist():
        for entity in named:
            if (entity["role"], entity["ref"]) not in expected:
                expected.append((entity["role"], entity["ref"]))

    evidence = evidence_from_manifest(_manifest(), series_stats=series)

    assert [(entity.role, entity.ref) for entity in evidence.entities] == expected


def test_only_a_series_off_the_period_is_read_row_by_row(
    monkeypatch: pytest.MonkeyPatch, requirement: RequirementFactory
) -> None:
    built: list[SeriesEvidence] = []

    class _Counted(SeriesEvidence):
        def __init__(self, **fields: object) -> None:
            super().__init__(**fields)
            built.append(self)

    monkeypatch.setattr("compiler.evidence.SeriesEvidence", _Counted)
    series = _crs_shaped_series(5_000, off_period=frozenset({12, 2_400, 4_999}))

    issues = check_evidence(
        requirement(coverage=Coverage(on_missing="report_gaps")),
        evidence_from_manifest(_manifest(), series_stats=series),
    )

    assert [one.observations for one in built] == [7, 7, 7]
    gaps = [issue for issue in issues if issue.code is EvidenceIssueCode.COVERAGE_GAPS]
    assert [issue.detail for issue in gaps] == [
        "series CRS_Disbursement_12 for ['country/D12', 'country/R12', 'sector/12'] has "
        "7 observations across 10 years",
        "series CRS_Disbursement_0 for ['country/D0', 'country/R0', 'sector/0'] has "
        "7 observations across 10 years",
        "series CRS_Disbursement_39 for ['country/D19', 'country/R49', 'sector/199'] has "
        "7 observations across 10 years",
    ]


# --- Named sets ------------------------------------------------------------------------

# The G7 turn's source (spec_4ba8cec3): G7 donors to Africa, as `search` names
# both sets in `resolved.entity_groups` and binds their members. Two African
# members stand in for the 59.
_G7_MEMBERS = {
    "CAN": "Canada",
    "DEU": "Germany",
    "FRA": "France",
    "GBR": "United Kingdom",
    "ITA": "Italy",
    "JPN": "Japan",
    "USA": "United States of America",
}
_G7_GROUPS = (
    ResolvedGroup(role="destination", label="Africa", ref="africa"),
    ResolvedGroup(role="origin", label="G7"),
)


def _g7_source(source: SourceFactory) -> SelectedSource:
    donors = tuple(
        ResolvedEntity(role="origin", ref=f"country/{code}", label=label)
        for code, label in _G7_MEMBERS.items()
    )
    recipients = (
        ResolvedEntity(role="destination", ref="country/GHA", label="Ghana"),
        ResolvedEntity(role="destination", ref="country/KEN", label="Kenya"),
    )
    return source(entities=donors + recipients, groups=_G7_GROUPS, identity=None)


@pytest.mark.parametrize(
    ("entities", "missing"),
    [
        ([Entity(name="G7", role="donor"), Entity(name="Africa", role="recipient")], []),
        ([Entity(name="g7", role="origin"), Entity(name="Africa")], []),
        ([Entity(name="G7", role="recipient"), Entity(name="Africa", role="recipient")], ["G7"]),
        ([Entity(name="G20", role="donor"), Entity(name="Africa", role="recipient")], ["G20"]),
    ],
    ids=["by-flow-side", "by-column-and-unstated", "wrong-side", "unnamed-set"],
)
def test_a_named_set_the_source_binds_meets_an_entity_naming_it(
    requirement: RequirementFactory,
    source: SourceFactory,
    entities: list[Entity],
    missing: list[str],
) -> None:
    issues = check_evidence(
        requirement(entities=entities), evidence_from_source(_g7_source(source))
    )

    assert [
        issue.required for issue in errors(issues) if issue.code is EvidenceIssueCode.ENTITY_MISSING
    ] == missing


def test_an_identity_binding_a_named_sets_members_is_not_wider_than_the_requirement(
    requirement: RequirementFactory, source: SourceFactory
) -> None:
    # France and Africa's members are every ref the identity binds, so none
    # is unnamed.
    identity = _identity(
        entities={"donor": ["country/FRA"], "recipient": ["country/GHA", "country/KEN"]}
    )
    wanted = requirement(
        entities=[Entity(name="France", role="donor"), Entity(name="Africa", role="recipient")]
    )

    assert _scope_codes(wanted, _g7_source(source).model_copy(update={"identity": identity})) == []


def _g7_retrieval(*refs: str) -> RetrievalEvidence:
    return evidence_from_manifest(
        _manifest(),
        series_stats=_series_stats(
            *(
                _stat("DebtServicePaid", entities=[{"role": "financingProvider", "ref": ref}])
                for ref in refs
            )
        ),
    )


@pytest.mark.parametrize(
    ("refs", "present"),
    [(("country/USA", "country/JPN"), True), (("country/CHN",), False)],
    ids=["holds-members", "holds-none"],
)
def test_a_retrieval_meets_a_named_set_when_it_holds_a_member_the_approved_source_bound(
    requirement: RequirementFactory, source: SourceFactory, refs: tuple[str, ...], present: bool
) -> None:
    issues = check_evidence(
        requirement(entities=[Entity(name="G7", role="donor")]),
        _g7_retrieval(*refs),
        approved=_g7_source(source),
    )

    missing = [i for i in errors(issues) if i.code is EvidenceIssueCode.ENTITY_MISSING]
    assert (missing == []) is present


# A set Search bound as the one party its dataset publishes as the set's total
# (apis `entity_groups[role].total`): the G7 as donor, a resolver group with
# no graph node, and Africa as recipient.
_TOTAL_GROUPS = (
    ResolvedGroup(role="origin", label="G7", total="GroupOfSeven"),
    ResolvedGroup(role="destination", label="Africa", ref="africa", total="DACAfrica"),
)


def _entity_codes(issues: list[EvidenceIssue]) -> list[str]:
    # These sources state no identity, which is an error of its own.
    return [
        issue.code.value
        for issue in errors(issues)
        if issue.code is not EvidenceIssueCode.IDENTITY_UNVERIFIED
    ]


def _total_source(source: SourceFactory) -> SelectedSource:
    return source(
        entities=(
            ResolvedEntity(role="origin", ref="GroupOfSeven", label="G7 countries, Total"),
            ResolvedEntity(role="destination", ref="DACAfrica", label="Africa, Total"),
        ),
        groups=_TOTAL_GROUPS,
        identity=None,
    )


@pytest.mark.parametrize(
    ("entities", "codes"),
    [
        ([Entity(name="G7", role="donor"), Entity(name="Africa", role="recipient")], []),
        (
            [Entity(name="G7"), Entity(name="Africa", role="recipient", members=True)],
            ["group_read_as_total"],
        ),
        ([Entity(name="G7", role="donor", members=True)], ["group_read_as_total"]),
    ],
    ids=["bare-names", "africa-members", "g7-members"],
)
def test_a_set_bound_as_its_published_total_meets_a_bare_name_and_refuses_members(
    requirement: RequirementFactory,
    source: SourceFactory,
    entities: list[Entity],
    codes: list[str],
) -> None:
    issues = check_evidence(
        requirement(entities=entities), evidence_from_source(_total_source(source))
    )

    assert _entity_codes(issues) == codes


@pytest.mark.parametrize(("members", "codes"), [(False, []), (True, ["group_read_as_total"])])
def test_a_retrieval_of_a_set_bound_as_its_total_holds_the_total_party(
    requirement: RequirementFactory, source: SourceFactory, members: bool, codes: list[str]
) -> None:
    issues = check_evidence(
        requirement(entities=[Entity(name="G7", role="donor", members=members)]),
        _g7_retrieval("GroupOfSeven"),
        approved=_total_source(source),
    )

    assert _entity_codes(issues) == codes


@pytest.mark.parametrize(("members", "codes"), [(False, []), (True, ["group_read_as_total"])])
def test_a_union_of_whole_totals_meets_a_bare_name_and_refuses_members(
    requirement: RequirementFactory, source: SourceFactory, members: bool, codes: list[str]
) -> None:
    label = "all donors including private"
    union = source(
        entities=(
            ResolvedEntity(role="origin", ref="DACOfficialDonors", label="Official Donors, Total"),
            ResolvedEntity(role="origin", ref="DACPrivateDonors", label="Private Donors, Total"),
        ),
        groups=(ResolvedGroup(role="origin", label=label, whole_of=("official", "private")),),
        identity=None,
    )

    issues = check_evidence(
        requirement(entities=[Entity(name=label, role="donor", members=members)]),
        evidence_from_source(union),
    )

    assert _entity_codes(issues) == codes


def test_a_set_read_as_its_members_meets_an_entity_asking_for_them(
    requirement: RequirementFactory, source: SourceFactory
) -> None:
    issues = check_evidence(
        requirement(entities=[Entity(name="G7", role="donor", members=True)]),
        evidence_from_source(_g7_source(source)),
    )

    assert _entity_codes(issues) == []


_US_REF = {"united states": "country/USA"}


@pytest.mark.parametrize(
    ("entity_refs", "missing"),
    [(_US_REF, []), ({}, ["United States"])],
    ids=["by-ref", "unresolved"],
)
def test_a_member_named_other_than_its_label_is_met_by_the_ref_its_name_resolves_to(
    requirement: RequirementFactory,
    source: SourceFactory,
    entity_refs: dict[str, str],
    missing: list[str],
) -> None:
    # The graph labels country/USA "United States of America".
    asked = requirement(
        entities=[Entity(name="United States", role="donor"), Entity(name="Japan", role="donor")]
    )
    found = _g7_source(source).model_copy(update={"entity_refs": entity_refs})

    issues = check_evidence(asked, evidence_from_source(found))

    assert [
        issue.required for issue in errors(issues) if issue.code is EvidenceIssueCode.ENTITY_MISSING
    ] == missing


@pytest.mark.parametrize(
    ("refs", "missing"),
    [(("country/USA",), []), (("country/JPN",), ["United States"])],
    ids=["holds-it", "holds-another"],
)
def test_a_retrieval_meets_a_member_by_the_ref_its_name_resolved_to_at_approval(
    requirement: RequirementFactory,
    source: SourceFactory,
    refs: tuple[str, ...],
    missing: list[str],
) -> None:
    approved = _g7_source(source).model_copy(update={"entity_refs": _US_REF})

    issues = check_evidence(
        requirement(entities=[Entity(name="United States", role="donor")]),
        _g7_retrieval(*refs),
        approved=approved,
    )

    assert [
        issue.required for issue in errors(issues) if issue.code is EvidenceIssueCode.ENTITY_MISSING
    ] == missing


@pytest.mark.parametrize("exclude", [False, True])
def test_search_confirmed_source_wording_is_authoritative(requirement, source, exclude):
    provenance = "provenance/OECDDAC2A"
    wording = "CRS" if exclude else "DAC"
    mention = {
        "text": wording,
        "ref": "provenance/CRS" if exclude else None,
        "polarity": "exclude" if exclude else "include",
    }
    resolved = source().model_copy(
        update={"provenance_ref": provenance, "provenance_label": "OECD DAC2a"}
    )
    # Preserve the resolved source's other evidence, attaching only the API's authoritative constraint.

    resolved = resolved.model_copy(
        update={"resolved_sources": (ResolvedSourceMention.model_validate(mention),)}
    )
    ask = requirement(source=wording, source_exclude=exclude)
    assert not [
        f
        for f in check_evidence(ask, evidence_from_source(resolved))
        if f.code in (EvidenceIssueCode.SOURCE_MISMATCH, EvidenceIssueCode.SOURCE_UNVERIFIED)
    ]
    manifest_evidence = evidence_from_source(resolved).model_copy(
        update={"origin": "manifest", "resolved_sources": ()}
    )
    assert not [
        f
        for f in check_evidence(ask, manifest_evidence, approved=resolved)
        if f.code in (EvidenceIssueCode.SOURCE_MISMATCH, EvidenceIssueCode.SOURCE_UNVERIFIED)
    ]
    unrelated = resolved.model_copy(update={"resolved_sources": ()})
    assert [
        f
        for f in check_evidence(ask, evidence_from_source(unrelated))
        if f.code in (EvidenceIssueCode.SOURCE_MISMATCH, EvidenceIssueCode.SOURCE_UNVERIFIED)
    ]

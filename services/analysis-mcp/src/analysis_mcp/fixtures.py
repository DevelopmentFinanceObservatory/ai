"""Fixture frames: development-finance shapes the verbs need, built the way
`materialise` builds them -- through `Frame.build` alone, never the bare
dataclass constructor -- so every fixture already carries the grammar's own
guarantees (NFC normalisation, finite values, row-key uniqueness) rather
than a hand-assembled shape that merely looks like a `Frame`.

A "null-bearing frame (status-annotated missing observations)" is not
buildable against `dfo_frame_grammar.Frame.build`'s *actual* current
contract. `grammar.py`/`frame.py` (frame-grammar pinned at
`0325320ebcb08f034bef6f209024771c02732eb0`) require every row's `value` to
be a finite `int`/`float` and refuse `status` outright if a caller supplies
it -- `status` is reserved and always `None` "at this grammar version"
(`Frame.build`'s own docstring). There is no cell-level way to spell "this
observation is missing" today; the closest honest analogue is a frame with
*fewer rows than its full entity-by-time grid*, i.e. missing rows rather
than a missing value in a present row. `sparse_coverage_frame` below is that
analogue, named for what it actually is. It bounds what the policy language
can mean: "aggregate skips nulls and records realised coverage" can only
mean "skips absent grain cells," not "skips a null cell," until
frame-grammar's `status` column actually populates.

Every fixture shares one `stat_var` vocabulary (development-finance content:
ODA disbursements, GNI, sized in the tens of rows) and is registered under a
`common.materialisation.MaterialisationTrace` by `_fixture_trace`, standing
in for the trace a real `get_data` materialisation would carry.

Policy language downstream should say "skips absent grid cells," not "skips
nulls" -- `analysis_mcp.engine.aggregate` already states it that way, and
every later verb's coverage language should follow the same wording rather
than the "null-bearing" framing above.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from common.materialisation import FrameRef, MaterialisationTrace
from dfo_frame_grammar import Frame

if TYPE_CHECKING:
    from common.materialisation import MaterialisationStore

_UNIT_USD = "dcid:USDollar"
_UNIT_PERCENT = "dcid:Percent"

_STAT_VAR_ODA_DISBURSED = "dcid:OfficialDevelopmentAssistanceNet_Disbursement"
_STAT_VAR_ODA_RECEIVED = "dcid:OfficialDevelopmentAssistanceNet_Received"
_STAT_VAR_GNI = "dcid:GrossNationalIncome"
_STAT_VAR_ODA_BILATERAL = "dcid:OdaBilateralGrossDisbursement"
_STAT_VAR_ODA_IMPUTED_MULTILATERAL = "dcid:OdaImputedMultilateralGrossDisbursement"

# Plausible, not authoritative -- these figures exist only to give a fixture
# frame realistic magnitude, never to answer a real development-finance
# question. Units are USD, not USD millions: the fixture would still be
# valid at any scale, this one just avoids implying false precision.
_ODA_DISBURSED_BY_DONOR_YEAR: dict[tuple[str, str], float] = {
    ("dcid:USA", "2021"): 47_805_000_000.0,
    ("dcid:USA", "2022"): 55_303_000_000.0,
    ("dcid:USA", "2023"): 63_575_000_000.0,
    ("dcid:GBR", "2021"): 15_688_000_000.0,
    ("dcid:GBR", "2022"): 15_694_000_000.0,
    ("dcid:GBR", "2023"): 19_073_000_000.0,
    ("dcid:FRA", "2021"): 15_476_000_000.0,
    ("dcid:FRA", "2022"): 15_942_000_000.0,
    ("dcid:FRA", "2023"): 15_366_000_000.0,
    ("dcid:DEU", "2021"): 33_269_000_000.0,
    ("dcid:DEU", "2022"): 35_037_000_000.0,
    ("dcid:DEU", "2023"): 36_680_000_000.0,
    ("dcid:JPN", "2021"): 17_631_000_000.0,
    ("dcid:JPN", "2022"): 17_463_000_000.0,
    ("dcid:JPN", "2023"): 19_640_000_000.0,
}

_ODA_SHARE_OF_GNI_BY_DONOR_YEAR: dict[tuple[str, str], float] = {
    ("dcid:USA", "2022"): 0.0022,
    ("dcid:USA", "2023"): 0.0024,
    ("dcid:GBR", "2022"): 0.0051,
    ("dcid:GBR", "2023"): 0.0058,
    ("dcid:FRA", "2022"): 0.0056,
    ("dcid:FRA", "2023"): 0.0050,
}

_ODA_RECEIVED_BY_RECIPIENT_YEAR: dict[tuple[str, str], float] = {
    ("dcid:KEN", "2021"): 3_115_000_000.0,
    ("dcid:KEN", "2022"): 2_939_000_000.0,
    ("dcid:KEN", "2023"): 3_302_000_000.0,
    ("dcid:ETH", "2021"): 5_223_000_000.0,
    ("dcid:ETH", "2023"): 4_871_000_000.0,
    ("dcid:BGD", "2022"): 3_886_000_000.0,
    ("dcid:BGD", "2023"): 4_015_000_000.0,
    ("dcid:HTI", "2021"): 1_207_000_000.0,
    ("dcid:HTI", "2022"): 1_344_000_000.0,
}

_GNI_BY_DONOR_YEAR: dict[tuple[str, str], float] = {
    ("dcid:USA", "2022"): 25_130_000_000_000.0,
    ("dcid:USA", "2023"): 26_486_000_000_000.0,
    ("dcid:GBR", "2022"): 3_076_000_000_000.0,
    ("dcid:GBR", "2023"): 3_290_000_000_000.0,
    ("dcid:FRA", "2022"): 2_846_000_000_000.0,
    ("dcid:FRA", "2023"): 3_030_000_000_000.0,
}

_ODA_SINGLE_REPORTER_TIME_SERIES: dict[str, float] = {
    "2017": 34_990_000_000.0,
    "2018": 34_728_000_000.0,
    "2019": 34_620_000_000.0,
    "2020": 35_575_000_000.0,
    # 2021 is deliberately absent, an interior gap `growth` computes across and reports.
    "2022": 55_303_000_000.0,
    "2023": 63_575_000_000.0,
    "2024": 65_000_000_000.0,
}


def plain_single_unit_frame() -> Frame:
    """Five donors, three years, one `stat_var`, one unit -- the baseline
    shape every verb's happy path runs against."""
    return Frame.build(
        role_keys=["reporter"],
        rows=[
            {
                "stat_var": _STAT_VAR_ODA_DISBURSED,
                "reporter": donor,
                "time_period": year,
                "value": value,
                "unit": _UNIT_USD,
            }
            for (donor, year), value in _ODA_DISBURSED_BY_DONOR_YEAR.items()
        ],
    )


def multi_unit_frame() -> Frame:
    """The same donor/year grain reported in two heterogeneous units --
    ODA in USD alongside ODA as a share of GNI -- so a verb that resolves
    unit before computing has real heterogeneity to refuse or disambiguate
    over -- every verb refuses on unit/facet heterogeneity."""
    rows = [
        {
            "stat_var": _STAT_VAR_ODA_DISBURSED,
            "reporter": donor,
            "time_period": year,
            "value": value,
            "unit": _UNIT_USD,
        }
        for (donor, year), value in _ODA_DISBURSED_BY_DONOR_YEAR.items()
        if year in ("2022", "2023") and donor in ("dcid:USA", "dcid:GBR", "dcid:FRA")
    ]
    rows += [
        {
            "stat_var": _STAT_VAR_ODA_DISBURSED,
            "reporter": donor,
            "time_period": year,
            "value": value,
            "unit": _UNIT_PERCENT,
        }
        for (donor, year), value in _ODA_SHARE_OF_GNI_BY_DONOR_YEAR.items()
    ]
    return Frame.build(role_keys=["reporter"], rows=rows)


def sparse_coverage_frame() -> Frame:
    """Four recipients across three years with some (recipient, year) cells
    simply absent as rows -- the closest fixture the current frame-grammar
    contract allows to a "missing observation" shape (see the module
    docstring: there is no null-value cell to build instead)."""
    return Frame.build(
        role_keys=["recipient"],
        rows=[
            {
                "stat_var": _STAT_VAR_ODA_RECEIVED,
                "recipient": recipient,
                "time_period": year,
                "value": value,
                "unit": _UNIT_USD,
            }
            for (recipient, year), value in _ODA_RECEIVED_BY_RECIPIENT_YEAR.items()
        ],
    )


def unit_partially_declared_frame() -> Frame:
    """Three donor/2022 ODA rows, two declaring `unit=dcid:USDollar` and one
    leaving `unit` unbound entirely -- a shape no other fixture carries.
    Models a row still awaiting currency normalisation upstream: the
    observation itself exists, but its own `unit` cell was never populated.
    Exercises `filter`'s own `category_match` narrowing exemption (see
    `engine/filter.py`'s module docstring) against a real `None` cell,
    rather than a frame where every row happens to declare a unit."""
    return Frame.build(
        role_keys=["reporter"],
        rows=[
            {
                "stat_var": _STAT_VAR_ODA_DISBURSED,
                "reporter": donor,
                "time_period": "2022",
                "value": _ODA_DISBURSED_BY_DONOR_YEAR[donor, "2022"],
                **({"unit": _UNIT_USD} if donor != "dcid:FRA" else {}),
            }
            for donor in ("dcid:USA", "dcid:GBR", "dcid:FRA")
        ],
    )


def join_compatible_frame_pair() -> tuple[Frame, Frame]:
    """Two frames sharing a `(reporter, time_period)` grain over different
    `stat_var`s -- ODA disbursed and GNI -- joinable on that shared grain,
    for `join`'s refusal and happy-path tests alike."""
    donors_years = tuple(_GNI_BY_DONOR_YEAR)
    oda = Frame.build(
        role_keys=["reporter"],
        rows=[
            {
                "stat_var": _STAT_VAR_ODA_DISBURSED,
                "reporter": donor,
                "time_period": year,
                "value": _ODA_DISBURSED_BY_DONOR_YEAR[donor, year],
                "unit": _UNIT_USD,
            }
            for donor, year in donors_years
        ],
    )
    gni = Frame.build(
        role_keys=["reporter"],
        rows=[
            {
                "stat_var": _STAT_VAR_GNI,
                "reporter": donor,
                "time_period": year,
                "value": value,
                "unit": _UNIT_USD,
            }
            for (donor, year), value in _GNI_BY_DONOR_YEAR.items()
        ],
    )
    return oda, gni


# One donor's aid to one recipient in USD, from two datasets. Reported
# bilateral flows run over 2010-2024 and modelled imputed multilateral shares
# over 2011-2023.
_ODA_BILATERAL_FRA_GTM_BY_YEAR: dict[str, float] = {
    str(year): 4_000_000.0 + 250_000.0 * (year - 2010) for year in range(2010, 2025)
}
_ODA_IMPUTED_MULTILATERAL_FRA_GTM_BY_YEAR: dict[str, float] = {
    str(year): 1_500_000.0 + 100_000.0 * (year - 2011) for year in range(2011, 2024)
}


def two_source_sum_frame_pair() -> tuple[Frame, Frame]:
    """France's ODA to Guatemala from two datasets, for `sum_of`. The first
    holds reported bilateral flows (2010-2024) and the second modelled
    imputed multilateral shares (2011-2023), both in USD, each with its own
    `provenance` and `measurementMethod`. The periods overlap on 2011-2023."""
    bilateral = Frame.build(
        role_keys=["donor", "recipient"],
        rows=[
            {
                "stat_var": _STAT_VAR_ODA_BILATERAL,
                "donor": "dcid:FRA",
                "recipient": "dcid:GTM",
                "time_period": year,
                "value": value,
                "unit": _UNIT_USD,
                "provenance": "dcid:OECD_CRS",
                "measurementMethod": "dcid:Reported",
            }
            for year, value in _ODA_BILATERAL_FRA_GTM_BY_YEAR.items()
        ],
    )
    imputed = Frame.build(
        role_keys=["donor", "recipient"],
        rows=[
            {
                "stat_var": _STAT_VAR_ODA_IMPUTED_MULTILATERAL,
                "donor": "dcid:FRA",
                "recipient": "dcid:GTM",
                "time_period": year,
                "value": value,
                "unit": _UNIT_USD,
                "provenance": "dcid:ONE_ImputedMultilateral",
                "measurementMethod": "dcid:Modelled",
            }
            for year, value in _ODA_IMPUTED_MULTILATERAL_FRA_GTM_BY_YEAR.items()
        ],
    )
    return bilateral, imputed


def time_series_with_gap_year_frame() -> Frame:
    """One reporter's own series across seven years with 2021 missing --
    a genuine interior gap, which `growth` computes across and reports."""
    return Frame.build(
        role_keys=["reporter"],
        rows=[
            {
                "stat_var": _STAT_VAR_ODA_DISBURSED,
                "reporter": "dcid:USA",
                "time_period": year,
                "value": value,
                "unit": _UNIT_USD,
            }
            for year, value in _ODA_SINGLE_REPORTER_TIME_SERIES.items()
        ],
    )


def unit_disjoint_by_reporter_frame() -> Frame:
    """USA reporting in USD across 2020-2021, GBR reporting in Percent over
    the same two years -- each reporter internally unit-homogeneous, unlike
    `multi_unit_frame` (where USA itself carries both units). A caller
    narrowed to one reporter's own scope (comparing USA's 2020 against its
    own 2021, say) resolves cleanly, though the frame at large is
    heterogeneous."""
    return Frame.build(
        role_keys=["reporter"],
        rows=[
            {
                "stat_var": _STAT_VAR_ODA_DISBURSED,
                "reporter": "dcid:USA",
                "time_period": year,
                "value": value,
                "unit": _UNIT_USD,
            }
            for year, value in (("2020", 44_170_000_000.0), ("2021", 47_805_000_000.0))
        ]
        + [
            {
                "stat_var": _STAT_VAR_ODA_DISBURSED,
                "reporter": "dcid:GBR",
                "time_period": year,
                "value": value,
                "unit": _UNIT_PERCENT,
            }
            for year, value in (("2020", 0.0070), ("2021", 0.0050))
        ],
    )


def unit_disjoint_facet_leak_frame() -> Frame:
    """USA in USD across 2021-2022 declaring `measurementMethod=dcid:M1`;
    GBR in Percent over the same two years declaring `measurementMethod=dcid:M2`.
    The same reporter-disjoint unit split as `unit_disjoint_by_reporter_frame`,
    with a facet axis genuinely heterogeneous at the frame level (M1 vs M2).
    When pinned to USA's unit, a resolver should exclude GBR's rows and thus
    never expose M2 as a facet selection or raise a facet heterogeneity error."""
    return Frame.build(
        role_keys=["reporter"],
        rows=[
            {
                "stat_var": _STAT_VAR_ODA_DISBURSED,
                "reporter": "dcid:USA",
                "time_period": year,
                "value": value,
                "unit": _UNIT_USD,
                "measurementMethod": "dcid:M1",
            }
            for year, value in (("2021", 47_805_000_000.0), ("2022", 55_303_000_000.0))
        ]
        + [
            {
                "stat_var": _STAT_VAR_ODA_DISBURSED,
                "reporter": "dcid:GBR",
                "time_period": year,
                "value": value,
                "unit": _UNIT_PERCENT,
                "measurementMethod": "dcid:M2",
            }
            for year, value in (("2021", 0.0055), ("2022", 0.0051))
        ],
    )


def facet_disjoint_by_reporter_frame() -> Frame:
    """USA and GBR, both in USD across 2021-2022, each declaring a different
    `provenance` -- unlike the unit-disjoint fixtures, they agree on `unit`
    entirely, so only a facet axis distinguishes them. Each reporter's own
    slice resolves cleanly on its own (one `provenance` value apiece). A
    two-sided verb pairing one reporter's slice against the other's must
    refuse the pairing, since resolving either side alone never exposes the
    other side's provenance."""
    return Frame.build(
        role_keys=["reporter"],
        rows=[
            {
                "stat_var": _STAT_VAR_ODA_DISBURSED,
                "reporter": "dcid:USA",
                "time_period": year,
                "value": value,
                "unit": _UNIT_USD,
                "provenance": "dcid:P1",
            }
            for year, value in (("2021", 47_805_000_000.0), ("2022", 55_303_000_000.0))
        ]
        + [
            {
                "stat_var": _STAT_VAR_ODA_DISBURSED,
                "reporter": "dcid:GBR",
                "time_period": year,
                "value": value,
                "unit": _UNIT_USD,
                "provenance": "dcid:P2",
            }
            for year, value in (("2021", 15_688_000_000.0), ("2022", 15_694_000_000.0))
        ],
    )


def unit_and_unitless_row_frame() -> Frame:
    """One recipient declaring `unit=dcid:USDollar`, the other leaving
    `unit` unbound entirely. Tests the unitless-row exclusion policy: an
    unpinned computation excludes the unitless row from its arithmetic,
    consistent with what pinning already does."""
    return Frame.build(
        role_keys=["recipient"],
        rows=[
            {
                "stat_var": _STAT_VAR_ODA_RECEIVED,
                "recipient": "dcid:KEN",
                "time_period": "2023",
                "value": 100.0,
                "unit": _UNIT_USD,
            },
            {
                "stat_var": _STAT_VAR_ODA_RECEIVED,
                "recipient": "dcid:ETH",
                "time_period": "2023",
                "value": 5.0,
            },
        ],
    )


def _fixture_ref(name: str) -> FrameRef:
    """A deterministic, validly-shaped `FrameRef` standing in for "the pair
    a real `get_data` materialisation would register this fixture's frame
    under" -- real spec_id/content_version minting happens upstream
    (apis), out of reach for a fixture; this only needs to satisfy
    `FrameRef`'s own patterns."""
    digest = hashlib.sha256(name.encode()).hexdigest()[:32]
    return FrameRef(spec_id=f"spec_{digest}", content_version=f"cv_{digest}")


def _fixture_trace(name: str, frame: Frame) -> MaterialisationTrace:
    """The `MaterialisationTrace` for `frame`, registered under `_fixture_ref(name)`
    -- standing in for what a real `get_data` fetch-and-parse would register,
    a bare materialisation rather than a toolkit verb result.
    """
    ref = _fixture_ref(name)
    source_revisions: tuple[Mapping[str, object], ...] = (
        {"source": "fixture_finance_ledger", "revision": "2026-08-01"},
    )
    facets: tuple[Mapping[str, object], ...] = (
        {"axis": "unit", "ref": _UNIT_USD, "basis": "declared_default"},
    )
    return MaterialisationTrace(
        upstream_trace={
            "applied_window": None,
            "graph_revision": "rev_fixture_2026_08_01",
            "source_revisions": list(source_revisions),
            "resolved_stat_vars": [
                {
                    "ref": _STAT_VAR_ODA_DISBURSED,
                    "roles": ["reporter"],
                    "observation_count": frame.num_rows,
                }
            ],
            "facets": list(facets),
            "normalisations": [],
            "conversion": None,
            "materialiser_build": "fixture@#36",
            "capability_version": "cap_fixture_1",
        },
        spec_id=ref.spec_id,
        content_version=ref.content_version,
    )


def live_shaped_trace(name: str, frame: Frame) -> MaterialisationTrace:
    """A `MaterialisationTrace` for `frame`, shaped like a real apis
    manifest's own `trace` field rather than `_fixture_trace`'s already-closed
    shape: each `facets` entry carries bookkeeping members
    (`default_policy`, `default_policy_version`, `dispatch_facet_id`,
    `selection_reason`) beyond `{axis, ref, basis}`, its `basis` is
    `"deployment_fallback"`, and a second entry carries `ref: None`, an
    axis with no facet selected. For checking that the toolkit stores and
    carries the wire shape a live materialisation actually produces.
    """
    ref = _fixture_ref(name)
    facets: tuple[Mapping[str, object], ...] = (
        {
            "axis": "unit",
            "ref": _UNIT_USD,
            "basis": "deployment_fallback",
            "default_policy": "latest",
            "default_policy_version": "v3",
            "dispatch_facet_id": "df_live_1",
            "selection_reason": "no explicit pin",
        },
        {
            "axis": "measurementMethod",
            "ref": None,
            "basis": "declared_default",
            "selection_reason": "absent_on_selected_facet",
        },
    )
    return MaterialisationTrace(
        upstream_trace={
            "applied_window": None,
            "graph_revision": "rev_live_2026_08_01",
            "source_revisions": [{"source": "live_finance_ledger", "revision": "2026-08-01"}],
            "resolved_stat_vars": [
                {
                    "ref": _STAT_VAR_ODA_DISBURSED,
                    "roles": ["reporter"],
                    "observation_count": frame.num_rows,
                }
            ],
            "facets": list(facets),
            "normalisations": [],
            "conversion": None,
            "materialiser_build": "live@#36",
            "capability_version": "cap_live_1",
        },
        spec_id=ref.spec_id,
        content_version=ref.content_version,
    )


@dataclass(frozen=True, kw_only=True)
class SeededFixtures:
    """The `{spec_id, content_version}` pairs `seed_fixture_store` registered,
    one field per fixture shape."""

    plain_single_unit: dict[str, str]
    multi_unit: dict[str, str]
    sparse_coverage: dict[str, str]
    join_pair: tuple[dict[str, str], dict[str, str]]
    time_series_with_gap: dict[str, str]
    sum_pair: tuple[dict[str, str], dict[str, str]]


async def seed_fixture_store(store: MaterialisationStore) -> SeededFixtures:
    """Register every fixture frame into `store` under a deterministic,
    session-independent pair, via `store.put` -- the store's one write path,
    idempotent, so calling it once per fixture at startup is safe.
    """
    plain_frame = plain_single_unit_frame()
    multi_frame = multi_unit_frame()
    sparse_frame = sparse_coverage_frame()
    oda_frame, gni_frame = join_compatible_frame_pair()
    time_series_frame = time_series_with_gap_year_frame()
    bilateral_frame, imputed_frame = two_source_sum_frame_pair()

    plain_ref = _fixture_ref("plain_single_unit")
    await store.put(
        plain_ref,
        frame=plain_frame,
        trace=_fixture_trace("plain_single_unit", plain_frame),
        writer="fixtures",
    )

    multi_unit_ref = _fixture_ref("multi_unit")
    await store.put(
        multi_unit_ref,
        frame=multi_frame,
        trace=_fixture_trace("multi_unit", multi_frame),
        writer="fixtures",
    )

    sparse_ref = _fixture_ref("sparse_coverage")
    await store.put(
        sparse_ref,
        frame=sparse_frame,
        trace=_fixture_trace("sparse_coverage", sparse_frame),
        writer="fixtures",
    )

    oda_ref = _fixture_ref("join_pair_oda")
    await store.put(
        oda_ref,
        frame=oda_frame,
        trace=_fixture_trace("join_pair_oda", oda_frame),
        writer="fixtures",
    )
    gni_ref = _fixture_ref("join_pair_gni")
    await store.put(
        gni_ref,
        frame=gni_frame,
        trace=_fixture_trace("join_pair_gni", gni_frame),
        writer="fixtures",
    )

    time_series_ref = _fixture_ref("time_series_with_gap")
    await store.put(
        time_series_ref,
        frame=time_series_frame,
        trace=_fixture_trace("time_series_with_gap", time_series_frame),
        writer="fixtures",
    )

    bilateral_ref = _fixture_ref("sum_pair_bilateral")
    await store.put(
        bilateral_ref,
        frame=bilateral_frame,
        trace=_fixture_trace("sum_pair_bilateral", bilateral_frame),
        writer="fixtures",
    )
    imputed_ref = _fixture_ref("sum_pair_imputed")
    await store.put(
        imputed_ref,
        frame=imputed_frame,
        trace=_fixture_trace("sum_pair_imputed", imputed_frame),
        writer="fixtures",
    )

    return SeededFixtures(
        plain_single_unit={
            "spec_id": plain_ref.spec_id,
            "content_version": plain_ref.content_version,
        },
        multi_unit={
            "spec_id": multi_unit_ref.spec_id,
            "content_version": multi_unit_ref.content_version,
        },
        sparse_coverage={
            "spec_id": sparse_ref.spec_id,
            "content_version": sparse_ref.content_version,
        },
        join_pair=(
            {"spec_id": oda_ref.spec_id, "content_version": oda_ref.content_version},
            {"spec_id": gni_ref.spec_id, "content_version": gni_ref.content_version},
        ),
        time_series_with_gap={
            "spec_id": time_series_ref.spec_id,
            "content_version": time_series_ref.content_version,
        },
        sum_pair=(
            {"spec_id": bilateral_ref.spec_id, "content_version": bilateral_ref.content_version},
            {"spec_id": imputed_ref.spec_id, "content_version": imputed_ref.content_version},
        ),
    )

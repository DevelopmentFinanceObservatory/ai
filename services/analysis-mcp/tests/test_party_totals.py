"""Party totals: `aggregate`, `rank`, `share_of` and `quotient_of` leave out the flow parties
that total other parties, and record them as `party_total` exclusions.

The frames carry `financingRecipient_party_kind` as `get_data(place_types=true)` registers it for a
flow table: the kind apis serves for each party, `party`, `residual` or `total`, and `untyped`
where it serves none. A donor-side frame (`party_frames.provider_frame`) carries
`financingProvider_party_kind` likewise, beside the sector (`official` or `private`) and whole
(`true` or `false`) cells, and the tests that sum whole totals or count private donors read it.
"""

from __future__ import annotations

import secrets

import pyarrow as pa
import pytest
from analysis_mcp.aggregate_tool import aggregate
from analysis_mcp.engine.aggregate import run_aggregate
from analysis_mcp.engine.common import PlaceLevelsMixedError
from analysis_mcp.engine.growth import run_growth
from analysis_mcp.engine.join import run_join
from analysis_mcp.engine.quotient_of import run_quotient_of
from analysis_mcp.engine.rank import run_rank
from analysis_mcp.engine.share_of import run_share_of
from analysis_mcp.execution_identity import current_execution_identity
from analysis_mcp.identity import op_spec_hash_for
from analysis_mcp.op_spec import AggregateOpSpec
from analysis_mcp.trace import AppliedPolicy, ExecutionRecord, Trace
from common.materialisation import FrameRef, InMemoryMaterialisationStore, set_store
from dfo_frame_grammar import Frame
from party_frames import (
    GROUP_TOTAL,
    OFFICIAL,
    OFFICIAL_WHOLE,
    PRIVATE,
    PRIVATE_WHOLE,
    PROVIDER,
    PROVIDER_KINDS,
    PROVIDER_SECTORS,
    PROVIDER_WHOLE,
    RESIDUAL,
    provider_frame,
)

_ODA = "dcid:Amount_FinancialFlow_Gross_Disbursement_DAC2aAidType206_Nominal"
_RECIPIENT = "financingRecipient"
_KINDS = f"{_RECIPIENT}_party_kind"
_TYPES = f"{_RECIPIENT}_type"

# France's 2022 ODA to a few of DAC2A's recipients, in USD millions: two countries, a
# residual and an organisation, and three totals holding them again.
_RECIPIENTS = {
    "country/MAR": ("party", 600.0),
    "country/IND": ("party", 400.0),
    "DACAfricaUnallocated": ("residual", 100.0),
    "EuropeanCommission": ("party", 2600.0),
    "DACAfrica": ("total", 5400.0),
    "WorldBankLowerMiddleIncome": ("total", 4700.0),
    "HeavilyIndebtedPoorCountries": ("total", 2800.0),
}
_TOTALS = ("DACAfrica", "HeavilyIndebtedPoorCountries", "WorldBankLowerMiddleIncome")


def _frame(
    recipients: dict[str, tuple[str, float]] = _RECIPIENTS,
    *,
    years: tuple[str, ...] = ("2022",),
    stat_var: str = _ODA,
    kinds: bool = True,
    types: bool = False,
) -> Frame:
    return Frame.build(
        role_keys=[
            "financingProvider",
            _RECIPIENT,
            *([_KINDS] if kinds else []),
            *([_TYPES] if types else []),
        ],
        rows=[
            {
                "stat_var": stat_var,
                "financingProvider": "country/FRA",
                _RECIPIENT: recipient,
                **({_KINDS: kind} if kinds else {}),
                **({_TYPES: "untyped"} if types else {}),
                "time_period": year,
                "value": value * (1 + offset),
                "unit": "USD",
            }
            for offset, year in enumerate(years)
            for recipient, (kind, value) in recipients.items()
        ],
    )


def _sum(frame: Frame):
    return run_aggregate(frame, group_by=("time_period",), aggregation="sum")


@pytest.mark.parametrize(
    "group_by",
    [("time_period",), (_KINDS, "time_period")],
    ids=["across-recipients", "by-recipient-kind"],
)
def test_a_sum_across_recipients_leaves_out_the_totals_and_names_them(
    group_by: tuple[str, ...],
) -> None:
    # Grouped by kind, the `total` group would add Africa to the income groups that
    # overlap it (5400 + 4700 + 2800); a shared kind does not make totals additive.
    result = run_aggregate(_frame(), group_by=group_by, aggregation="sum")

    assert sum(summary.value for summary in result.group_summaries) == 3700.0
    (excluded,) = result.exclusions.records
    assert excluded.reason == "party_total"
    assert excluded.row_count == 3
    assert excluded.entities == {_RECIPIENT: _TOTALS}


@pytest.mark.parametrize(
    "recipients",
    [
        {"country/MAR": ("party", 600.0), "geoId/06": ("untyped", 1.0)},
        {"country/MAR": ("party", 600.0), "geoId/06": ("whole_total", 1.0)},
        {"DACAfrica": ("total", 5400.0), "DACAsia": ("total", 2000.0)},
    ],
    ids=["no-kind", "unknown-cell", "every-one-a-total"],
)
def test_a_sum_across_parties_of_no_kind_or_all_totals_is_refused(
    recipients: dict[str, tuple[str, float]],
) -> None:
    with pytest.raises(PlaceLevelsMixedError) as raised:
        _sum(_frame(recipients))

    assert raised.value.column == _KINDS


@pytest.mark.parametrize("verb", ["sum", "rank", "sum-by-recipient-type"])
def test_a_sum_or_rank_across_parties_with_no_kind_column_is_refused(verb: str) -> None:
    # A flow frame minted without kinds cannot tell totals from parties, so a sum fails
    # closed rather than counting "Africa, Total" beside the countries it holds, and a
    # rank rather than ranking it above them.
    # A frame minted before kinds carries `financingRecipient_type`, and grouping by it
    # would sum the totals sharing a type cell on top of each other.
    run = {
        "sum": lambda: _sum(_frame(kinds=False)),
        "rank": lambda: _rank(_frame(kinds=False)),
        "sum-by-recipient-type": lambda: run_aggregate(
            _frame(kinds=False, types=True), group_by=(_TYPES, "time_period"), aggregation="sum"
        ),
    }[verb]
    with pytest.raises(PlaceLevelsMixedError) as raised:
        run()

    assert (raised.value.column, raised.value.types) == (_KINDS, ())


def test_a_sum_over_recipients_grouped_first_still_leaves_out_the_totals() -> None:
    # Summing per recipient first keeps the recipient's kind, so the later sum across
    # recipients finds the totals the direct sum leaves out: 3700, not 16600.
    by_recipient = run_aggregate(
        _frame(), group_by=(_RECIPIENT, "time_period"), aggregation="sum"
    ).frame

    result = _sum(by_recipient)

    assert _KINDS in by_recipient.columns
    (summary,) = result.group_summaries
    assert summary.value == _sum(_frame()).group_summaries[0].value == 3700.0
    (excluded,) = result.exclusions.records
    assert excluded.reason == "party_total"
    assert excluded.entities[_RECIPIENT] == _TOTALS


def test_a_rank_of_recipient_growth_leaves_out_the_totals() -> None:
    growth = run_growth(
        _frame(years=("2021", "2022")),
        window_start="2021",
        window_end="2022",
        entity_by=(_RECIPIENT,),
        change="absolute",
    ).frame

    result = _rank(growth)

    ranked = [dict(rank.entity_key)[_RECIPIENT] for rank in result.entity_ranks]
    assert ranked == ["EuropeanCommission", "country/MAR", "country/IND"]
    assert [record.reason for record in result.exclusions.records] == [
        "party_total",
        "party_residual",
    ]
    assert result.exclusions.records[1].entities == {_RECIPIENT: ("DACAfricaUnallocated",)}


_KEYED_BY_RECIPIENT = {
    "aggregate": lambda: run_aggregate(
        _frame(), group_by=(_RECIPIENT, "time_period"), aggregation="sum"
    ),
    "growth": lambda: run_growth(
        _frame(years=("2021", "2022")),
        window_start="2021",
        window_end="2022",
        entity_by=(_RECIPIENT,),
    ),
    "rank": lambda: _rank(_frame(), rank_by=(_RECIPIENT, "financingProvider")),
    "join": lambda: run_join(
        _frame(), _frame(stat_var="dcid:Other"), join_keys=(_RECIPIENT, "financingProvider")
    ),
    "share_of": lambda: run_share_of(
        _frame(),
        _frame(),
        stat_var="dcid:share",
        numerator_entity={},
        denominator_entity={},
        numerator_by=(_RECIPIENT,),
    ),
    "quotient_of": lambda: run_quotient_of(
        _frame(),
        _frame(),
        stat_var="dcid:ratio",
        numerator_entity={},
        denominator_entity={},
        numerator_by=(_RECIPIENT,),
        denominator_by={_RECIPIENT: _RECIPIENT},
    ),
}


@pytest.mark.parametrize("verb", list(_KEYED_BY_RECIPIENT))
def test_a_verb_keyed_by_recipient_keeps_each_recipients_kind(verb: str) -> None:
    result = _KEYED_BY_RECIPIENT[verb]()
    output = result.frame

    assert _KINDS in output.columns
    kinds = dict(
        zip(
            output.table.column(_RECIPIENT).to_pylist(),
            output.table.column(_KINDS).to_pylist(),
            strict=True,
        )
    )
    assert kinds["EuropeanCommission"] == "party"
    if verb == "rank":
        # A ranking compares members, so the residual is left out as a residual.
        assert "DACAfricaUnallocated" not in kinds
        residual = [r for r in result.exclusions.records if r.reason == "party_residual"]
        assert [record.entities for record in residual] == [{_RECIPIENT: ("DACAfricaUnallocated",)}]
    else:
        assert kinds["DACAfricaUnallocated"] == "residual"


@pytest.mark.parametrize(
    "recipients",
    [{"DACAfrica": ("total", 5400.0)}, {"geoId/06": ("untyped", 1.0)}],
    ids=["one-total", "one-of-no-kind"],
)
def test_one_party_alone_sums_whatever_its_kind(
    recipients: dict[str, tuple[str, float]],
) -> None:
    result = _sum(_frame(recipients))

    assert result.exclusions.records == ()


def test_a_rank_of_recipients_leaves_out_the_totals() -> None:
    result = run_rank(
        _frame(),
        rank_by=(_RECIPIENT,),
        aggregation=None,
        window_start=None,
        window_end=None,
        ascending=False,
    )

    ranked = [dict(rank.entity_key)[_RECIPIENT] for rank in result.entity_ranks]
    assert ranked == ["EuropeanCommission", "country/MAR", "country/IND"]
    assert [record.reason for record in result.exclusions.records] == [
        "party_total",
        "party_residual",
    ]
    assert result.exclusions.records[1].entities == {_RECIPIENT: ("DACAfricaUnallocated",)}


def _rank(frame: Frame, rank_by: tuple[str, ...] = (_RECIPIENT,)):
    return run_rank(
        frame,
        rank_by=rank_by,
        aggregation="sum",
        window_start=None,
        window_end=None,
        ascending=False,
    )


@pytest.mark.parametrize(
    ("recipients", "ranked", "excluded"),
    [
        (
            {"DACAfrica": ("total", 5400.0), "DACAsia": ("total", 2000.0)},
            ["DACAfrica", "DACAsia"],
            [],
        ),
        # A role holding no party keeps its residuals: they are all there is to rank.
        (
            {
                "DACAfricaUnallocated": ("residual", 100.0),
                "DACAsiaUnallocated": ("residual", 50.0),
                "DACAfrica": ("total", 5400.0),
            },
            ["DACAfricaUnallocated", "DACAsiaUnallocated"],
            ["party_total"],
        ),
    ],
    ids=["all-totals", "residuals-and-a-total"],
)
def test_a_rank_of_recipients_holding_no_party_ranks_what_it_holds(
    recipients: dict[str, tuple[str, float]], ranked: list[str], excluded: list[str]
) -> None:
    # "Which regions receive the most French aid?" ranks DACGroup rows side by side.
    result = _rank(_frame(recipients))

    assert [dict(rank.entity_key)[_RECIPIENT] for rank in result.entity_ranks] == ranked
    assert [record.reason for record in result.exclusions.records] == excluded


def test_a_rank_pooling_recipients_that_are_all_totals_is_refused() -> None:
    # Ranking providers sums each provider's recipients, and a sum of totals counts twice.
    regions = {"DACAfrica": ("total", 5400.0), "DACAsia": ("total", 2000.0)}

    with pytest.raises(PlaceLevelsMixedError) as raised:
        _rank(_frame(regions), rank_by=("financingProvider",))

    assert raised.value.column == _KINDS


def test_a_share_of_a_named_total_divides_by_the_parties_alone() -> None:
    # Africa's share of everything France gave: the numerator binds the DACAfrica row,
    # and the denominator sums the parties with the totals left out.
    result = run_share_of(
        _frame(),
        _frame(),
        stat_var="dcid:share",
        numerator_entity={_RECIPIENT: "DACAfrica"},
        denominator_entity={},
    )

    (share,) = result.period_shares
    assert share.denominator_value == 3700.0
    totals = [record for record in result.exclusions.records if record.reason == "party_total"]
    assert [(record.side, record.row_count) for record in totals] == [("denominator", 3)]
    assert result.warnings[-1].startswith("share_exceeds_whole: 1 share(s) above 100%")


def test_a_share_within_its_whole_carries_no_warning() -> None:
    result = run_share_of(
        _frame(),
        _frame(),
        stat_var="dcid:share",
        numerator_entity={_RECIPIENT: "country/MAR"},
        denominator_entity={},
    )

    assert not any(warning.startswith("share_exceeds_whole") for warning in result.warnings)


@pytest.fixture
def store():
    held = InMemoryMaterialisationStore()
    set_store(held)
    yield held
    set_store(None)


def _trace() -> Trace:
    op_spec = AggregateOpSpec(
        frame_ref=FrameRef(spec_id="spec_" + "a" * 32, content_version="cv_" + "b" * 32),
        group_by=(),
        aggregation="sum",
    )
    return Trace(
        upstream_trace={
            "graph_revision": "rev_test",
            "source_revisions": [{"source": "test", "revision": "1"}],
            "facets": [],
            "conversion": None,
        },
        frame_content_version="cv_" + "d" * 32,
        op_spec_hash=op_spec_hash_for(op_spec),
        execution_record=ExecutionRecord(content="{}"),
        applied_policy=AppliedPolicy(),
        content_version="ocv_" + "c" * 32,
        input_content_versions=["cv_" + "b" * 32],
        execution_identity=current_execution_identity(),
    )


async def test_the_aggregate_tool_reports_the_totals_it_left_out(store) -> None:
    digest = secrets.token_hex(16)
    ref = FrameRef(spec_id=f"spec_{digest}", content_version=f"cv_{digest}")
    await store.put(ref, frame=_frame(), trace=_trace(), writer="test")

    result = await aggregate(
        frame_ref={"spec_id": ref.spec_id, "content_version": ref.content_version},
        aggregation="sum",
        group_by=("time_period",),
    )

    assert result.is_error is False, result.structured_content
    (excluded,) = result.structured_content["exclusions"]
    assert excluded["reason"] == "party_total"
    assert excluded["entities"] == {_RECIPIENT: list(_TOTALS)}
    assert "party_total (totals of other parties" in result.content[0].text


_FILTER_RETRY = f"category_match on {_RECIPIENT!r}"
# The kinds a filter keeps to compare or rank the members, leaving out the totals and the
# unallocated residuals.
_MEMBER_KINDS = {"member_kinds": ["party"]}


@pytest.mark.parametrize(
    ("frame", "fix", "retry"),
    [
        (
            lambda: _frame({"country/MAR": ("party", 600.0), "geoId/06": ("untyped", 1.0)}),
            {"types": ["party", "untyped"], **_MEMBER_KINDS},
            _FILTER_RETRY,
        ),
        (
            lambda: _frame({"DACAfrica": ("total", 5400.0), "DACAsia": ("total", 2000.0)}),
            {"types": ["total"], **_MEMBER_KINDS},
            _FILTER_RETRY,
        ),
        # A filter adds no kind column, so only a retrieval with kinds is offered.
        (lambda: _frame(kinds=False), {"types": []}, "place_types=true"),
    ],
    ids=["no-kind", "every-one-a-total", "no-kind-column"],
)
async def test_the_aggregate_tool_refuses_unpoolable_parties_with_the_party_column(
    store, frame, fix: dict[str, list[str]], retry: str
) -> None:
    digest = secrets.token_hex(16)
    ref = FrameRef(spec_id=f"spec_{digest}", content_version=f"cv_{digest}")
    await store.put(ref, frame=frame(), trace=_trace(), writer="test")

    result = await aggregate(
        frame_ref={"spec_id": ref.spec_id, "content_version": ref.content_version},
        aggregation="sum",
        group_by=("time_period",),
    )

    assert result.is_error is True
    refusal = result.structured_content
    assert refusal["code"] == "place_levels_mixed"
    assert refusal["fix"] == {"column": _KINDS, **fix}
    # The retry names the party column, not a place_level filter on the kind column, or
    # the retrieval that adds the kind column; never a filter that keeps several parties.
    assert retry in result.content[0].text
    assert (_FILTER_RETRY in result.content[0].text) == (retry == _FILTER_RETRY)


_OFFICIAL = "DACOfficialDonors"
_PRIVATE_TOTAL = "DACPrivateDonors"

# Ethiopia's 2021 ODA in USD millions: France and Germany, the Gates Foundation, and the two
# whole-population totals DAC2A publishes beside them.
_DONORS = {
    "country/FRA": (OFFICIAL, 120.0),
    "country/DEU": (OFFICIAL, 200.0),
    "GatesFoundation": (PRIVATE, 30.0),
    _OFFICIAL: (OFFICIAL_WHOLE, 4267.6),
    _PRIVATE_TOTAL: (PRIVATE_WHOLE, 310.8),
}
_WHOLE_TOTALS = {name: _DONORS[name] for name in (_OFFICIAL, _PRIVATE_TOTAL)}


@pytest.mark.parametrize("official", [GROUP_TOTAL, OFFICIAL_WHOLE], ids=["total", "whole-total"])
def test_a_total_beside_members_is_left_out_whole_or_not(official: tuple[str, str, str]) -> None:
    donors = {**_DONORS, _OFFICIAL: (official, 4267.6)}

    result = run_aggregate(provider_frame(donors), group_by=("time_period",), aggregation="sum")

    assert result.group_summaries[0].value == pytest.approx(350.0)
    (excluded,) = result.exclusions.records
    assert excluded.reason == "party_total"
    assert excluded.entities == {PROVIDER: (_OFFICIAL, _PRIVATE_TOTAL)}


def test_a_private_donor_is_summed_and_counted_with_its_value() -> None:
    result = run_aggregate(provider_frame(_DONORS), group_by=("time_period",), aggregation="sum")

    (pooled,) = result.pooled_parties
    assert (pooled.role, pooled.count, pooled.whole_totals) == (PROVIDER, 3, False)
    assert (pooled.private_count, pooled.private_value) == (1, 30.0)
    assert result.group_summaries[0].value == pytest.approx(350.0)


def test_a_mean_reports_no_pooled_parties() -> None:
    result = run_aggregate(provider_frame(_DONORS), group_by=("time_period",), aggregation="mean")

    assert result.pooled_parties == ()


def test_private_parties_whose_pooled_value_overflows_are_counted_without_a_value() -> None:
    # Each year's sum is finite, and the private donor's value across both years is not.
    frame = provider_frame(
        {"country/FRA": (OFFICIAL, 1.0), "GatesFoundation": (PRIVATE, 1e308)},
        years=("2020", "2021"),
    )

    result = run_aggregate(frame, group_by=("time_period",), aggregation="sum")
    share = run_share_of(
        frame,
        frame,
        stat_var="dcid:share",
        numerator_entity={PROVIDER: "country/FRA"},
        denominator_entity={},
    )

    assert [summary.value for summary in result.group_summaries] == [1e308 + 1.0] * 2
    (pooled,) = result.pooled_parties
    assert (pooled.private_count, pooled.private_value) == (1, None)
    assert [(entry.private_count, entry.private_value) for entry in share.denominator_pooled] == [
        (1, None)
    ]


def test_a_pool_of_whole_totals_alone_is_summed_with_the_private_total_counted() -> None:
    # "All donors including private": the official and the private total hold disjoint
    # populations, so Ethiopia's 2021 sum is 4,267.6m + 310.8m, of which the private
    # donors' total is 310.8m.
    result = run_aggregate(
        provider_frame(_WHOLE_TOTALS), group_by=("time_period",), aggregation="sum"
    )

    assert result.group_summaries[0].value == pytest.approx(4578.4)
    assert result.exclusions.records == ()
    (pooled,) = result.pooled_parties
    assert (pooled.count, pooled.whole_totals) == (2, True)
    assert (pooled.private_count, pooled.private_value) == (1, 310.8)


@pytest.mark.parametrize("donors", [_DONORS, _WHOLE_TOTALS], ids=["members", "whole-totals"])
def test_a_frame_with_no_sector_column_reports_nothing_about_private_parties(
    donors: dict[str, tuple[tuple[str, str, str], float]],
) -> None:
    # A frame minted before the sector column cannot tell a private party apart, so it
    # reports no private count at all rather than a count of zero.
    frame = provider_frame(donors, sectors=False)

    summed = run_aggregate(frame, group_by=("time_period",), aggregation="sum")
    ranked = _rank(frame, rank_by=(PROVIDER,))

    (pooled,) = summed.pooled_parties
    assert (pooled.private_count, pooled.private_value) == (None, None)
    assert ranked.private_parties == ()


def _years(frame_by_year: dict[str, Frame]) -> Frame:
    """One frame holding each year's rows of `frame_by_year`."""
    first = next(iter(frame_by_year.values()))
    table = pa.concat_tables(frame.table for frame in frame_by_year.values())
    return Frame.from_table(columns=first.columns, table=table)


# DAC2A's official total is whole from 2021 on in this frame, and a group total in 2020.
_WHOLE_FROM_2021 = {
    "2020": provider_frame({**_WHOLE_TOTALS, _OFFICIAL: (GROUP_TOTAL, 4000.0)}, years=("2020",)),
    "2021": provider_frame(_WHOLE_TOTALS, years=("2021",)),
}
# The same, with DAC2A's official total carrying no whole cell in 2020.
_UNTYPED_IN_2020 = {
    **_WHOLE_FROM_2021,
    "2020": provider_frame(
        {**_WHOLE_TOTALS, _OFFICIAL: (("total", "official", "untyped"), 4000.0)}, years=("2020",)
    ),
}


@pytest.mark.parametrize(
    "frame",
    [
        lambda: provider_frame({**_WHOLE_TOTALS, _OFFICIAL: (GROUP_TOTAL, 4267.6)}),
        lambda: _years(_WHOLE_FROM_2021),
    ],
    ids=["one-not-whole", "whole-in-some-years"],
)
def test_a_sum_of_totals_not_all_whole_is_refused(frame) -> None:
    with pytest.raises(PlaceLevelsMixedError) as raised:
        run_aggregate(frame(), group_by=("time_period",), aggregation="sum")

    assert (raised.value.column, raised.value.types) == (PROVIDER_KINDS, ("total",))


@pytest.mark.parametrize(
    "frame_by_year", [_WHOLE_FROM_2021, _UNTYPED_IN_2020], ids=["not-whole", "untyped"]
)
def test_a_total_whole_in_some_years_reads_not_whole_after_a_verb_keyed_by_it(
    frame_by_year: dict[str, Frame],
) -> None:
    # Summed per donor first, each donor keeps one sector and whole cell. A total whole in
    # some years only takes `false`, so the later sum across donors stays refused.
    by_donor = run_aggregate(_years(frame_by_year), group_by=(PROVIDER,), aggregation="sum").frame
    cells = {
        donor: (sector, whole)
        for donor, sector, whole in zip(
            *(
                by_donor.table.column(column).to_pylist()
                for column in (PROVIDER, PROVIDER_SECTORS, PROVIDER_WHOLE)
            ),
            strict=True,
        )
    }

    assert cells == {_OFFICIAL: ("official", "false"), _PRIVATE_TOTAL: ("private", "true")}
    with pytest.raises(PlaceLevelsMixedError):
        run_aggregate(by_donor, group_by=(), aggregation="sum")


def test_a_rank_of_donors_tags_private_ones_and_leaves_out_totals_and_residuals() -> None:
    donors = {**_DONORS, "DACUnallocated": (RESIDUAL, 15.0)}

    result = _rank(provider_frame(donors), rank_by=(PROVIDER,))

    ranked = [dict(rank.entity_key)[PROVIDER] for rank in result.entity_ranks]
    assert ranked == ["country/DEU", "country/FRA", "GatesFoundation"]
    assert result.private_parties == ((PROVIDER, "GatesFoundation"),)
    assert [(record.reason, record.entities) for record in result.exclusions.records] == [
        ("party_total", {PROVIDER: (_OFFICIAL, _PRIVATE_TOTAL)}),
        ("party_residual", {PROVIDER: ("DACUnallocated",)}),
    ]


def test_a_rank_of_whole_totals_alone_ranks_them_and_tags_the_private_one() -> None:
    result = _rank(provider_frame(_WHOLE_TOTALS), rank_by=(PROVIDER,))

    assert [dict(rank.entity_key)[PROVIDER] for rank in result.entity_ranks] == [
        _OFFICIAL,
        _PRIVATE_TOTAL,
    ]
    assert result.private_parties == ((PROVIDER, _PRIVATE_TOTAL),)


def test_a_share_of_whole_totals_pools_them_in_its_denominator() -> None:
    # The private share of "all donors including private": the numerator sits inside its
    # denominator.
    result = run_share_of(
        provider_frame(_DONORS),
        provider_frame(_WHOLE_TOTALS),
        stat_var="dcid:share",
        numerator_entity={PROVIDER: _PRIVATE_TOTAL},
        denominator_entity={},
    )

    (share,) = result.period_shares
    assert share.denominator_value == pytest.approx(4578.4)
    assert share.percent == pytest.approx(310.8 / 4578.4 * 100)

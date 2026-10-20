"""Party totals: `aggregate`, `rank`, `share_of` and `quotient_of` leave out the flow parties
that total other parties, and record them as `party_total` exclusions.

The frames carry `financingRecipient_party_kind` as `get_data(place_types=true)` registers it for a
flow table: the kind apis serves for each party, `party`, `private`, `residual`, `total` or
`whole_total`, and `untyped` where it serves none. A donor-side frame
(`party_frames.provider_frame`) carries `financingProvider_party_kind` likewise.
"""

from __future__ import annotations

import secrets

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
from party_frames import PROVIDER, PROVIDER_KINDS, provider_frame

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
        {"country/MAR": ("party", 600.0), "geoId/06": ("philanthropy", 1.0)},
        {"DACAfrica": ("total", 5400.0), "DACAsia": ("total", 2000.0)},
        {"DACAfrica": ("total", 5400.0), "DACAllRecipients": ("whole_total", 9000.0)},
    ],
    ids=["no-kind", "unknown-cell", "every-one-a-total", "total-beside-whole-total"],
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


# The kinds a filter keeps to compare or rank the members: parties and private parties,
# leaving out the totals and the unallocated residuals.
_MEMBER_KINDS = {"member_kinds": ["party", "private"]}


_RECIPIENT_PHRASES = ("category_match on 'financingRecipient'", '("all recipients")')


@pytest.mark.parametrize(
    ("frame", "column", "fix", "phrases"),
    [
        (
            lambda: _frame({"country/MAR": ("party", 600.0), "geoId/06": ("untyped", 1.0)}),
            _KINDS,
            {"types": ["party", "untyped"], **_MEMBER_KINDS},
            _RECIPIENT_PHRASES,
        ),
        (
            lambda: _frame({"DACAfrica": ("total", 5400.0), "DACAsia": ("total", 2000.0)}),
            _KINDS,
            {"types": ["total"], **_MEMBER_KINDS},
            _RECIPIENT_PHRASES,
        ),
        # The donor side names donors, not recipients, as the population to search for.
        (
            lambda: provider_frame({"country/FRA": ("party", 600.0), "geoId/06": ("untyped", 1.0)}),
            PROVIDER_KINDS,
            {"types": ["party", "untyped"], **_MEMBER_KINDS},
            ("category_match on 'financingProvider'", '("all donors")'),
        ),
        # A frame with no kind column has no kinds to filter on until a retrieval adds them.
        (lambda: _frame(kinds=False), _KINDS, {"types": []}, ("single financingRecipient party",)),
    ],
    ids=["no-kind", "every-one-a-total", "no-kind-donor-side", "no-kind-column"],
)
async def test_the_aggregate_tool_refuses_unpoolable_parties_with_the_party_column(
    store, frame, column: str, fix: dict[str, list[str]], phrases: tuple[str, ...]
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
    assert refusal is not None
    assert refusal["code"] == "place_levels_mixed"
    assert refusal["retry_mode"] == "correct_and_retry"
    assert refusal["fix"] == {"column": column, **fix}
    # The role and the side's population are written out, with no placeholder left in the text.
    assert all(phrase in refusal["message"] for phrase in phrases)
    assert "{" not in refusal["message"]


_OFFICIAL = "DACOfficialDonors"
_PRIVATE_TOTAL = "DACPrivateDonors"

# Ethiopia's 2021 ODA in USD millions: France and Germany, the Gates Foundation, and the two
# whole-population totals DAC2A publishes beside them.
_DONORS = {
    "country/FRA": ("party", 120.0),
    "country/DEU": ("party", 200.0),
    "GatesFoundation": ("private", 30.0),
    _OFFICIAL: ("whole_total", 4267.6),
    _PRIVATE_TOTAL: ("whole_total", 310.8),
}


@pytest.mark.parametrize("total_kind", ["total", "whole_total"])
def test_a_whole_total_beside_members_is_left_out_like_a_total(total_kind: str) -> None:
    donors = {**_DONORS, _OFFICIAL: (total_kind, 4267.6)}

    result = run_aggregate(provider_frame(donors), group_by=("time_period",), aggregation="sum")

    assert result.group_summaries[0].value == pytest.approx(350.0)
    (excluded,) = result.exclusions.records
    assert excluded.reason == "party_total"
    assert excluded.entities == {PROVIDER: (_OFFICIAL, _PRIVATE_TOTAL)}


def test_a_private_donor_is_summed_and_counted_with_its_value() -> None:
    result = run_aggregate(provider_frame(_DONORS), group_by=("time_period",), aggregation="sum")

    (pooled,) = result.pooled_parties
    assert (pooled.role, pooled.count, pooled.private_count) == (PROVIDER, 3, 1)
    assert pooled.private_value == 30.0
    assert result.group_summaries[0].value == pytest.approx(350.0)


def test_a_mean_reports_no_pooled_parties() -> None:
    result = run_aggregate(provider_frame(_DONORS), group_by=("time_period",), aggregation="mean")

    assert result.pooled_parties == ()


def test_private_parties_whose_pooled_value_overflows_are_counted_without_a_value() -> None:
    # Each year's sum is finite, and the private donor's value across both years is not.
    frame = provider_frame(
        {"country/FRA": ("party", 1.0), "GatesFoundation": ("private", 1e308)},
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


def test_a_pool_of_whole_totals_alone_is_summed_and_kept() -> None:
    # "All donors including private": the official and the private total hold disjoint
    # populations, so Ethiopia's 2021 sum is 4,267.6m + 310.8m.
    totals = {name: _DONORS[name] for name in (_OFFICIAL, _PRIVATE_TOTAL)}

    result = run_aggregate(provider_frame(totals), group_by=("time_period",), aggregation="sum")

    assert result.group_summaries[0].value == pytest.approx(4578.4)
    assert result.exclusions.records == ()
    assert [pooled.private_count for pooled in result.pooled_parties] == [0]


def test_a_rank_of_donors_tags_private_ones_and_leaves_out_whole_totals() -> None:
    result = _rank(provider_frame(_DONORS), rank_by=(PROVIDER,))

    ranked = [dict(rank.entity_key)[PROVIDER] for rank in result.entity_ranks]
    assert ranked == ["country/DEU", "country/FRA", "GatesFoundation"]
    assert result.private_parties == ((PROVIDER, "GatesFoundation"),)
    assert [record.reason for record in result.exclusions.records] == ["party_total"]


def test_a_rank_of_whole_totals_alone_ranks_them() -> None:
    totals = {name: _DONORS[name] for name in (_OFFICIAL, _PRIVATE_TOTAL)}

    result = _rank(provider_frame(totals), rank_by=(PROVIDER,))

    assert [dict(rank.entity_key)[PROVIDER] for rank in result.entity_ranks] == [
        _OFFICIAL,
        _PRIVATE_TOTAL,
    ]


def test_a_share_of_whole_totals_pools_them_in_its_denominator() -> None:
    # The private share of "all donors including private": the numerator sits inside its
    # denominator.
    frame = provider_frame(_DONORS)

    result = run_share_of(
        frame,
        provider_frame({k: _DONORS[k] for k in (_OFFICIAL, _PRIVATE_TOTAL)}),
        stat_var="dcid:share",
        numerator_entity={PROVIDER: _PRIVATE_TOTAL},
        denominator_entity={},
    )

    (share,) = result.period_shares
    assert share.denominator_value == pytest.approx(4578.4)
    assert share.percent == pytest.approx(310.8 / 4578.4 * 100)

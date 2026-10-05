"""Loads `verb_coverage_corpus.json` and keeps it honest against the
registered tool set: every verb named in a covered entry's chain must be a
real, registered tool name, and the corpus must stay large enough, and
covered enough, to mean something.

`test_every_covered_entrys_chain_actually_executes` goes one step further,
for every covered entry: it builds a real frame, registers it, and runs the
entry's own `verb_chain` through the real tool functions, feeding each
step's own `{spec_id, content_version}` pair into the next -- so a chain
naming a registered tool (which the first test above already checks) is
also proven runnable, not just spellable. Some chains legitimately refuse
partway through (see `_ENTRY_PLANS`'s own per-entry `expect_refusal`); this
file records that expectation explicitly rather than accepting whatever
happens, which would prove nothing at all.

Reading a step's own next `frame_ref` off its own `result.structured_content`
(rather than a minted handle) is legitimate only because this corpus test
calls the verb functions directly, not through the harness. On the real wire
the harness reads `structured_content` in code (`executor.run` takes a step's
declared shape and row count, `harness.graph` carries only the digest text
into `narrate`'s `step_summaries`), so the digest *text* is the only path a
model has to a verb's result. That's what makes the digest's own
identity-block relabel (`digest.py`) load-bearing rather than cosmetic. A
model never sees `structured_content`, but this test legitimately can, since
it drives the bare tool functions itself.
"""

from __future__ import annotations

import json
import secrets
from collections.abc import Awaitable, Callable
from pathlib import Path

import pytest
from analysis_mcp.aggregate_tool import aggregate
from analysis_mcp.compare_tool import compare
from analysis_mcp.execution_identity import current_execution_identity
from analysis_mcp.filter_tool import filter_verb
from analysis_mcp.growth_tool import growth
from analysis_mcp.identity import op_spec_hash_for
from analysis_mcp.join_tool import join
from analysis_mcp.op_spec import AggregateOpSpec, CategoryMatchPredicate, OrderedThresholdPredicate
from analysis_mcp.rank_tool import rank
from analysis_mcp.server import mcp
from analysis_mcp.share_of_tool import share_of
from analysis_mcp.sum_of_tool import sum_of
from analysis_mcp.tool_errors import ToolResult
from analysis_mcp.trace import AppliedPolicy, ExecutionRecord, Trace
from common.materialisation import FrameRef, InMemoryMaterialisationStore, set_store
from dfo_frame_grammar import Frame

_CORPUS_PATH = Path(__file__).parent / "verb_coverage_corpus.json"
_MIN_COVERED_ENTRIES = 15


def _load_entries() -> list[dict[str, object]]:
    return json.loads(_CORPUS_PATH.read_text())["entries"]


async def _registered_tool_names() -> set[str]:
    tools = await mcp.list_tools()
    return {tool.name for tool in tools}


async def test_every_covered_entrys_verb_chain_names_only_registered_tools() -> None:
    registered = await _registered_tool_names()
    entries = _load_entries()

    for entry in entries:
        if entry.get("uncovered"):
            continue
        for verb in entry["verb_chain"]:
            assert verb in registered, (
                f"corpus entry {entry['question']!r} names verb {verb!r}, which is not a "
                f"registered tool ({sorted(registered)!r})"
            )


def test_every_entry_is_either_covered_or_explains_the_gap() -> None:
    for entry in _load_entries():
        if entry.get("uncovered"):
            assert entry.get("uncovered_reason"), (
                f"uncovered entry {entry['question']!r} has no uncovered_reason"
            )
            assert "verb_chain" not in entry
        else:
            assert entry.get("verb_chain"), (
                f"covered entry {entry['question']!r} has an empty chain"
            )


def test_the_corpus_stays_large_and_mostly_covered() -> None:
    entries = _load_entries()
    covered = [entry for entry in entries if not entry.get("uncovered")]

    assert len(entries) >= 20, f"corpus has only {len(entries)} entries; #38 asks for ~20"
    assert len(covered) >= _MIN_COVERED_ENTRIES, (
        f"only {len(covered)} of {len(entries)} entries are covered by a verb chain, "
        f"below the {_MIN_COVERED_ENTRIES}-entry floor"
    )


def test_every_verb_in_the_closed_set_answers_at_least_one_question() -> None:
    entries = _load_entries()
    verbs_used = {
        verb for entry in entries if not entry.get("uncovered") for verb in entry["verb_chain"]
    }

    assert verbs_used == {
        "aggregate",
        "compare",
        "growth",
        "rank",
        "share_of",
        "join",
        "filter",
        "sum_of",
    }


# Chain execution -----------------------------------------------------------------
#
# Every covered entry gets its own small frame(s) and its own runner below --
# not a generic "run the chain with default params" harness, since each
# question implies its own group_by/window/entity/predicates, and a generic
# harness that ignored that would just be re-proving `Frame.build` works,
# the same vacuity this toolkit's own perturbation tests avoid.


def _row(
    *,
    stat_var: str = "dcid:X",
    reporter: str = "dcid:USA",
    year: str = "2020",
    value: float = 1.0,
    **extra: object,
) -> dict[str, object]:
    return {
        "stat_var": stat_var,
        "reporter": reporter,
        "time_period": year,
        "value": value,
        **extra,
    }


def _frame(rows: list[dict[str, object]], *, role_keys: list[str] | None = None) -> Frame:
    return Frame.build(role_keys=role_keys or ["reporter"], rows=rows)


def _minimal_trace() -> Trace:
    """A real, internally-consistent `Trace` for a hand-built frame -- the
    same pattern `test_aggregate_tool.py`'s own `_minimal_trace` uses:
    real `op_spec_hash`/`execution_identity` minting, a placeholder
    `upstream_trace`/`content_version` since nothing here reads them back
    against a real upstream `materialise` run."""
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


async def _seed(store: InMemoryMaterialisationStore, frame: Frame) -> dict[str, str]:
    digest = secrets.token_hex(16)
    ref = FrameRef(spec_id=f"spec_{digest}", content_version=f"cv_{digest}")
    await store.put(ref, frame=frame, trace=_minimal_trace(), writer="test")
    return {"spec_id": ref.spec_id, "content_version": ref.content_version}


def _pair(result: ToolResult) -> dict[str, str]:
    assert result.is_error is False, f"expected success, got: {result.content[0].text}"
    return {
        "spec_id": result.structured_content["spec_id"],
        "content_version": result.structured_content["content_version"],
    }


def _digest(result: ToolResult) -> str:
    return result.content[0].text


def _assert_all_in(text: str, *substrings: str) -> None:
    for substring in substrings:
        assert substring in text, f"{substring!r} not found in digest:\n{text}"


def _assert_rows(result: ToolResult, rows: int) -> None:
    assert result.structured_content["rows"] == rows, result.structured_content


def _assert_none_in(text: str, *substrings: str) -> None:
    for substring in substrings:
        assert substring not in text, f"{substring!r} unexpectedly found in digest:\n{text}"


class _Entry:
    """One covered corpus entry's own execution plan: `run` builds and
    registers whatever frame(s) it needs and drives `question`'s own
    `verb_chain` through the real tool functions, returning the chain's own
    final `ToolResult`. `expect_refusal`, when set, is the refusal `code`
    the final step must return instead of succeeding; `check`, when set,
    asserts on the successful result's own digest -- a computed number this
    entry's own frame data makes specific, not just "didn't crash"."""

    def __init__(
        self,
        question: str,
        run: Callable[[InMemoryMaterialisationStore], Awaitable[ToolResult]],
        *,
        expect_refusal: str | None = None,
        check: Callable[[ToolResult], None] | None = None,
    ) -> None:
        self.question = question
        self.run = run
        self.expect_refusal = expect_refusal
        self.check = check


async def _run_nigeria_gdp(store: InMemoryMaterialisationStore) -> ToolResult:
    frame = _frame(
        [
            _row(stat_var="dcid:GrossDomesticProduction", reporter="dcid:NGA", year=year, value=v)
            for year, v in (
                ("2021", 440_000_000_000.0),
                ("2022", 477_000_000_000.0),
                ("2023", 363_000_000_000.0),
            )
        ]
    )
    ref = await _seed(store, frame)
    return await aggregate(frame_ref=ref, aggregation="sum", group_by=("time_period",))


async def _run_kenya_oda_received_2022(store: InMemoryMaterialisationStore) -> ToolResult:
    frame = _frame(
        [
            _row(
                stat_var="dcid:OfficialDevelopmentAssistanceNet_Receipt",
                reporter="dcid:KEN",
                year="2022",
                value=2_800_000_000.0,
            )
        ]
    )
    ref = await _seed(store, frame)
    return await aggregate(frame_ref=ref, aggregation="sum", group_by=())


async def _run_nigeria_oda_by_year(store: InMemoryMaterialisationStore) -> ToolResult:
    frame = _frame(
        [
            _row(
                stat_var="dcid:OfficialDevelopmentAssistanceNet_Receipt",
                reporter="dcid:NGA",
                year=year,
                value=v,
            )
            for year, v in (
                ("2019", 3_300_000_000.0),
                ("2020", 3_600_000_000.0),
                ("2021", 3_100_000_000.0),
                ("2022", 2_400_000_000.0),
                ("2023", 2_900_000_000.0),
            )
        ]
    )
    ref = await _seed(store, frame)
    return await aggregate(frame_ref=ref, aggregation="sum", group_by=("time_period",))


async def _run_uganda_debt_2024(store: InMemoryMaterialisationStore) -> ToolResult:
    frame = _frame(
        [
            _row(
                stat_var="dcid:ExternalDebtStock",
                reporter="dcid:UGA",
                year="2024",
                value=21_500_000_000.0,
            )
        ]
    )
    ref = await _seed(store, frame)
    return await aggregate(frame_ref=ref, aggregation="sum", group_by=())


async def _run_kenya_all_flow_types_2022(store: InMemoryMaterialisationStore) -> ToolResult:
    frame = _frame(
        [
            _row(reporter="dcid:KEN", year="2022", value=v, measurementMethod=method)
            for method, v in (
                ("dcid:Grant", 1_500_000_000.0),
                ("dcid:Loan", 900_000_000.0),
                ("dcid:TechnicalCooperation", 400_000_000.0),
            )
        ]
    )
    ref = await _seed(store, frame)
    return await aggregate(frame_ref=ref, aggregation="sum", group_by=())


async def _run_rwanda_growth(store: InMemoryMaterialisationStore) -> ToolResult:
    frame = _frame(
        [
            _row(reporter="dcid:RWA", year=year, value=v)
            for year, v in (
                ("2020", 1_100_000_000.0),
                ("2021", 1_250_000_000.0),
                ("2022", 1_300_000_000.0),
            )
        ]
    )
    ref = await _seed(store, frame)
    return await growth(frame_ref=ref, window_start="2020", window_end="2022")


async def _run_kenya_current_account_growth(store: InMemoryMaterialisationStore) -> ToolResult:
    frame = _frame(
        [
            _row(
                stat_var="dcid:CurrentAccountBalanceAsFractionOfGDP",
                reporter="dcid:KEN",
                year=year,
                value=v,
                unit="dcid:Percent",
            )
            for year, v in (("2022", -5.1), ("2023", -4.3))
        ]
    )
    ref = await _seed(store, frame)
    return await growth(frame_ref=ref, window_start="2022", window_end="2023")


async def _run_uganda_loans_2021(store: InMemoryMaterialisationStore) -> ToolResult:
    frame = _frame(
        [
            _row(reporter="dcid:UGA", year="2021", value=v, measurementMethod=method)
            for method, v in (("dcid:Loan", 700_000_000.0), ("dcid:Grant", 1_100_000_000.0))
        ]
    )
    seeded = await _seed(store, frame)
    filter_ref = _pair(
        await filter_verb(
            frame_ref=seeded,
            predicates=(CategoryMatchPredicate(column="measurementMethod", values=("dcid:Loan",)),),
        )
    )
    return await aggregate(frame_ref=filter_ref, aggregation="sum", group_by=())


async def _run_ethiopia_private_grants_2022(store: InMemoryMaterialisationStore) -> ToolResult:
    frame = _frame(
        [
            _row(
                reporter="dcid:ETH",
                year="2022",
                value=v,
                measurementMethod=method,
                provenance=provenance,
            )
            for method, provenance, v in (
                ("dcid:Grant", "dcid:PrivateSource", 250_000_000.0),
                ("dcid:Grant", "dcid:OfficialSource", 900_000_000.0),
                ("dcid:Loan", "dcid:PrivateSource", 300_000_000.0),
            )
        ]
    )
    seeded = await _seed(store, frame)
    filter_ref = _pair(
        await filter_verb(
            frame_ref=seeded,
            predicates=(
                CategoryMatchPredicate(column="measurementMethod", values=("dcid:Grant",)),
                CategoryMatchPredicate(column="provenance", values=("dcid:PrivateSource",)),
            ),
        )
    )
    return await aggregate(frame_ref=filter_ref, aggregation="sum", group_by=())


async def _run_ghana_technical_cooperation(store: InMemoryMaterialisationStore) -> ToolResult:
    frame = _frame(
        [
            _row(reporter="dcid:GHA", year=year, value=v, measurementMethod=method)
            for year in ("2021", "2022", "2023")
            for method, v in (
                ("dcid:TechnicalCooperation", 150_000_000.0 + int(year) - 2021),
                ("dcid:Grant", 900_000_000.0),
            )
        ]
    )
    seeded = await _seed(store, frame)
    filter_ref = _pair(
        await filter_verb(
            frame_ref=seeded,
            predicates=(
                CategoryMatchPredicate(
                    column="measurementMethod", values=("dcid:TechnicalCooperation",)
                ),
            ),
        )
    )
    return await aggregate(frame_ref=filter_ref, aggregation="sum", group_by=("time_period",))


async def _run_kenya_net_ppg_flows_2024(store: InMemoryMaterialisationStore) -> ToolResult:
    disbursements = _frame(
        [
            _row(
                stat_var="dcid:PPG_Disbursement",
                reporter="dcid:KEN",
                year="2024",
                value=1_800_000_000.0,
            )
        ]
    )
    repayments = _frame(
        [_row(stat_var="dcid:PPG_Repayment", reporter="dcid:KEN", year="2024", value=700_000_000.0)]
    )
    left = await _seed(store, disbursements)
    right = await _seed(store, repayments)
    return await compare(frame_ref=left, right_frame_ref=right, left_entity={}, right_entity={})


async def _run_usa_share_of_dac_oda_2022(store: InMemoryMaterialisationStore) -> ToolResult:
    frame = _frame(
        [
            _row(reporter=reporter, year="2022", value=v)
            for reporter, v in (
                ("dcid:USA", 47_805_000_000.0),
                ("dcid:GBR", 15_688_000_000.0),
                ("dcid:FRA", 15_476_000_000.0),
                ("dcid:JPN", 17_507_000_000.0),
                ("dcid:DEU", 35_000_000_000.0),
            )
        ]
    )
    ref = await _seed(store, frame)
    return await share_of(
        frame_ref=ref,
        numerator_entity={"reporter": "dcid:USA"},
        denominator_entity={},
    )


async def _run_ethiopia_share_of_ssa_receipts_2023(
    store: InMemoryMaterialisationStore,
) -> ToolResult:
    frame = _frame(
        [
            _row(
                stat_var="dcid:OfficialDevelopmentAssistanceNet_Receipt",
                reporter=reporter,
                year="2023",
                value=v,
            )
            for reporter, v in (
                ("dcid:ETH", 3_200_000_000.0),
                ("dcid:KEN", 2_900_000_000.0),
                ("dcid:TZA", 2_500_000_000.0),
            )
        ]
    )
    ref = await _seed(store, frame)
    return await share_of(
        frame_ref=ref,
        numerator_entity={"reporter": "dcid:ETH"},
        denominator_entity={},
    )


async def _run_usa_oda_alongside_gni(store: InMemoryMaterialisationStore) -> ToolResult:
    oda = _frame(
        [
            _row(
                stat_var="dcid:OfficialDevelopmentAssistanceNet_Disbursement",
                reporter="dcid:USA",
                year=year,
                value=v,
            )
            for year, v in (("2022", 55_303_000_000.0), ("2023", 63_575_000_000.0))
        ]
    )
    gni = _frame(
        [
            _row(stat_var="dcid:GrossNationalIncome", reporter="dcid:USA", year=year, value=v)
            for year, v in (("2022", 25_700_000_000_000.0), ("2023", 27_400_000_000_000.0))
        ]
    )
    left = await _seed(store, oda)
    right = await _seed(store, gni)
    return await join(frame_ref=left, right_frame_ref=right, join_keys=("reporter",))


async def _run_rank_france_germany_japan_2023(store: InMemoryMaterialisationStore) -> ToolResult:
    frame = _frame(
        [
            _row(reporter=reporter, year="2023", value=v)
            for reporter, v in (
                ("dcid:FRA", 15_476_000_000.0),
                ("dcid:DEU", 35_000_000_000.0),
                ("dcid:JPN", 17_507_000_000.0),
            )
        ]
    )
    ref = await _seed(store, frame)
    return await rank(
        frame_ref=ref,
        rank_by=("reporter",),
        window_start="2021",
        window_end="2023",
        ascending=False,
    )


async def _run_top_two_donors_2023(store: InMemoryMaterialisationStore) -> ToolResult:
    frame = _frame(
        [
            _row(reporter=reporter, year="2023", value=v)
            for reporter, v in (
                ("dcid:USA", 63_575_000_000.0),
                ("dcid:DEU", 35_000_000_000.0),
                ("dcid:JPN", 17_507_000_000.0),
                ("dcid:FRA", 15_476_000_000.0),
            )
        ]
    )
    ref = await _seed(store, frame)
    return await rank(frame_ref=ref, rank_by=("reporter",), ascending=False, limit=2)


async def _run_rank_donors_over_20b_2023(store: InMemoryMaterialisationStore) -> ToolResult:
    frame = _frame(
        [
            _row(reporter=reporter, year="2023", value=v)
            for reporter, v in (
                ("dcid:USA", 63_575_000_000.0),
                ("dcid:DEU", 35_000_000_000.0),
                ("dcid:JPN", 17_507_000_000.0),
                ("dcid:FRA", 15_476_000_000.0),
            )
        ]
    )
    seeded = await _seed(store, frame)
    filter_ref = _pair(
        await filter_verb(
            frame_ref=seeded,
            predicates=(
                OrderedThresholdPredicate(column="value", op="gt", value=20_000_000_000.0),
            ),
        )
    )
    return await rank(
        frame_ref=filter_ref,
        rank_by=("reporter",),
        window_start="2021",
        window_end="2023",
        ascending=False,
    )


async def _run_usa_observations_over_50b(store: InMemoryMaterialisationStore) -> ToolResult:
    frame = _frame(
        [
            _row(reporter="dcid:USA", year=year, value=v)
            for year, v in (
                ("2021", 47_805_000_000.0),
                ("2022", 55_303_000_000.0),
                ("2023", 63_575_000_000.0),
            )
        ]
    )
    ref = await _seed(store, frame)
    return await filter_verb(
        frame_ref=ref,
        predicates=(
            CategoryMatchPredicate(column="reporter", values=("dcid:USA",)),
            OrderedThresholdPredicate(column="value", op="gt", value=50_000_000_000.0),
        ),
    )


async def _run_donors_share_of_health_oda(store: InMemoryMaterialisationStore) -> ToolResult:
    """Keyed share_of feeding rank: the chain PR 2 exists for. Each donor's own
    health ODA divides by that same donor's own total ODA (denominator_scope
    per_axis, the default when `denominator_by` is given), then the shares
    are ranked -- proving the rank runs over the *share*, not the raw values
    (France's health spend is smaller than Germany's, in absolute terms, but
    its share is larger)."""
    health = _frame(
        [
            _row(reporter=reporter, year="2023", value=v)
            for reporter, v in (
                ("dcid:USA", 6_000_000_000.0),
                ("dcid:GBR", 3_000_000_000.0),
                ("dcid:FRA", 9_000_000_000.0),
            )
        ]
    )
    total = _frame(
        [
            _row(reporter=reporter, year="2023", value=v)
            for reporter, v in (
                ("dcid:USA", 20_000_000_000.0),
                ("dcid:GBR", 15_000_000_000.0),
                ("dcid:FRA", 12_000_000_000.0),
            )
        ]
    )
    health_ref = await _seed(store, health)
    total_ref = await _seed(store, total)
    share_ref = _pair(
        await share_of(
            frame_ref=health_ref,
            denominator_frame_ref=total_ref,
            numerator_by=["reporter"],
            denominator_by={"reporter": "reporter"},
        )
    )
    return await rank(
        frame_ref=share_ref,
        rank_by=("reporter",),
        window_start="2022",
        window_end="2023",
        ascending=False,
    )


async def _run_africa_share_of_chinese_lending_since_2010(
    store: InMemoryMaterialisationStore,
) -> ToolResult:
    """Un-keyed share_of feeding an absolute growth: the share itself is
    dimensionless, so the change reads in percentage points, not in a
    percentage of a percentage."""
    africa = _frame(
        [
            _row(reporter="dcid:AFRICA_AGGREGATE", year=year, value=v)
            for year, v in (("2010", 20.0), ("2011", 45.0))
        ]
    )
    china_total = _frame(
        [
            _row(reporter="dcid:CHINA_TOTAL", year=year, value=v)
            for year, v in (("2010", 100.0), ("2011", 100.0))
        ]
    )
    africa_ref = await _seed(store, africa)
    china_ref = await _seed(store, china_total)
    share_ref = _pair(await share_of(frame_ref=africa_ref, denominator_frame_ref=china_ref))
    return await growth(
        frame_ref=share_ref, window_start="2010", window_end="2011", change="absolute"
    )


async def _run_kenya_rank_change_since_2010(store: InMemoryMaterialisationStore) -> ToolResult:
    """The motivating per-period-rank chain, over a fixture with a genuine hole:
    Uganda has no row in 2015. A growth straight off the population-wide rank
    series refuses on Uganda's own gap year -- the blind spot a fixture with
    every entity in every period would hide -- so this checks that refusal
    directly before building the real, filtered chain the question needs."""
    frame = _frame(
        [
            _row(stat_var="dcid:AidReceived", reporter="dcid:KEN", year="2010", value=50.0),
            _row(stat_var="dcid:AidReceived", reporter="dcid:UGA", year="2010", value=30.0),
            _row(stat_var="dcid:AidReceived", reporter="dcid:TZA", year="2010", value=70.0),
            _row(stat_var="dcid:AidReceived", reporter="dcid:KEN", year="2011", value=60.0),
            # Uganda has no row in 2011 -- the population's series has a hole.
            _row(stat_var="dcid:AidReceived", reporter="dcid:TZA", year="2011", value=40.0),
            _row(stat_var="dcid:AidReceived", reporter="dcid:KEN", year="2012", value=90.0),
            _row(stat_var="dcid:AidReceived", reporter="dcid:UGA", year="2012", value=80.0),
            _row(stat_var="dcid:AidReceived", reporter="dcid:TZA", year="2012", value=20.0),
        ]
    )
    seeded = await _seed(store, frame)
    ranked_ref = _pair(
        await rank(
            frame_ref=seeded,
            rank_by=("reporter",),
            window_start="2010",
            window_end="2012",
            ascending=False,
            per_period=True,
        )
    )

    unfiltered = await growth(
        frame_ref=ranked_ref,
        window_start="2010",
        window_end="2012",
        entity_by=("reporter",),
        change="absolute",
    )
    assert unfiltered.is_error is True, (
        "a growth straight off the population-wide rank series was expected to refuse on "
        "Uganda's own missing 2011 -- a fixture with no hole would hide exactly this"
    )
    assert unfiltered.structured_content["code"] == "growth_gap_year"

    kenya_ref = _pair(
        await filter_verb(
            frame_ref=ranked_ref,
            predicates=(CategoryMatchPredicate(column="reporter", values=("dcid:KEN",)),),
        )
    )
    return await growth(
        frame_ref=kenya_ref,
        window_start="2010",
        window_end="2012",
        entity_by=("reporter",),
        change="absolute",
    )


async def _run_france_guatemala_total_aid(store: InMemoryMaterialisationStore) -> ToolResult:
    bilateral = _frame(
        [
            _row(
                stat_var="dcid:OdaBilateral",
                reporter="dcid:FRA",
                year=year,
                value=value,
                unit="dcid:USDollar",
                provenance="dcid:OECD_CRS",
            )
            for year, value in (("2010", 4_000_000.0), ("2011", 4_250_000.0), ("2012", 4_500_000.0))
        ]
    )
    imputed = _frame(
        [
            _row(
                stat_var="dcid:OdaImputedMultilateral",
                reporter="dcid:FRA",
                year=year,
                value=value,
                unit="dcid:USDollar",
                provenance="dcid:ONE_ImputedMultilateral",
            )
            for year, value in (("2011", 1_500_000.0), ("2012", 1_600_000.0))
        ]
    )
    left = await _seed(store, bilateral)
    right = await _seed(store, imputed)
    return await sum_of(frame_ref=left, right_frame_ref=right)


_ENTRY_PLANS: tuple[_Entry, ...] = (
    _Entry(
        "What was Nigeria's GDP from 2021 to 2023?",
        _run_nigeria_gdp,
        check=lambda r: _assert_all_in(_digest(r), "440000000000", "477000000000", "363000000000"),
    ),
    _Entry(
        "How much total ODA did Kenya receive in 2022?",
        _run_kenya_oda_received_2022,
        check=lambda r: (
            (_ for _ in ()).throw(AssertionError()) if "2800000000" not in _digest(r) else None
        ),
    ),
    _Entry(
        "What were total ODA disbursements to Nigeria from 2019 to 2023 by year?",
        _run_nigeria_oda_by_year,
        check=lambda r: _assert_all_in(_digest(r), "3300000000", "2400000000", "2900000000"),
    ),
    _Entry(
        "What is Uganda's total external debt in 2024?",
        _run_uganda_debt_2024,
        check=lambda r: _assert_all_in(_digest(r), "21500000000"),
    ),
    _Entry(
        "What were total disbursements to Kenya in 2022 across all flow types?",
        _run_kenya_all_flow_types_2022,
        # Three flow types can only be told apart in this grammar by
        # `measurementMethod`, a facet-identity column -- and `aggregate`
        # has no parameter to pin or fold a heterogeneous facet the way
        # `pinned_unit` does for `unit` (`_DISALLOWED_GROUP_BY_COLUMNS`
        # excludes every facet column from `group_by` outright), so summing
        # "across all flow types" in one call genuinely refuses.
        expect_refusal="facet_heterogeneous",
    ),
    _Entry(
        "How did ODA to Rwanda change from 2020 to 2022?",
        _run_rwanda_growth,
        check=lambda r: _assert_all_in(_digest(r), "1100000000", "1300000000"),
    ),
    _Entry(
        "How did Kenya's current account balance as % of GDP change from 2022 to 2023?",
        _run_kenya_current_account_growth,
        check=lambda r: _assert_all_in(_digest(r), "-5.1", "-4.3"),
    ),
    _Entry(
        "How much ODA in loans did Uganda receive in 2021?",
        _run_uganda_loans_2021,
        check=lambda r: _assert_all_in(_digest(r), "700000000"),
    ),
    _Entry(
        "How much private grant funding did Ethiopia receive in 2022?",
        _run_ethiopia_private_grants_2022,
        check=lambda r: _assert_all_in(_digest(r), "250000000"),
    ),
    _Entry(
        "What was technical cooperation ODA to Ghana per year from 2021 to 2023?",
        _run_ghana_technical_cooperation,
        check=lambda r: _assert_all_in(_digest(r), "150000000", "150000002"),
    ),
    _Entry(
        "What were Kenya's net PPG financial flows in 2024? (disbursements minus repayments)",
        _run_kenya_net_ppg_flows_2024,
        # `compare`'s `delta` is `right - left` (see `compare_tool.compare`'s
        # own docstring); `_run_kenya_net_ppg_flows_2024` seeds disbursements
        # as `left` and repayments as `right`, so the digest's own delta is
        # repayments minus disbursements, `-1100000000`, not the question's
        # "disbursements minus repayments" framing negated.
        check=lambda r: _assert_all_in(_digest(r), "1800000000 -> 700000000", "delta -1100000000"),
    ),
    _Entry(
        "What share of total DAC-donor ODA disbursed in 2022 did the United States account for?",
        _run_usa_share_of_dac_oda_2022,
        check=lambda r: _assert_all_in(_digest(r), "47805000000 / 131476000000"),
    ),
    _Entry(
        "What share of Sub-Saharan Africa's total ODA receipts did Ethiopia account for in 2023?",
        _run_ethiopia_share_of_ssa_receipts_2023,
        check=lambda r: _assert_all_in(_digest(r), "3200000000 / 8600000000"),
    ),
    _Entry(
        "Show the USA's ODA disbursements alongside its GNI for 2022 and 2023 in one table.",
        _run_usa_oda_alongside_gni,
        check=lambda r: _assert_all_in(
            _digest(r),
            "left=55303000000, right=25700000000000",
            "left=63575000000, right=27400000000000",
        ),
    ),
    _Entry(
        "Which of France, Germany, and Japan disbursed the most ODA in 2023?",
        _run_rank_france_germany_japan_2023,
        check=lambda r: _assert_all_in(_digest(r), "#1 reporter=dcid:DEU", "35000000000"),
    ),
    _Entry(
        "Which were the top 2 ODA donors in 2023?",
        _run_top_two_donors_2023,
        check=lambda r: (
            _assert_all_in(
                _digest(r),
                "limit 2 (kept 2 of 4 ranked entities)",
                "#2 reporter=dcid:DEU",
                "beyond_limit (ranked beyond the limit): 2 row(s) "
                "[reporter=dcid:FRA, reporter=dcid:JPN]",
            ),
            _assert_rows(r, 2),
        ),
    ),
    _Entry(
        "Among donors whose 2023 ODA disbursements exceeded $20 billion, rank them from highest to lowest.",
        _run_rank_donors_over_20b_2023,
        check=lambda r: (
            _assert_all_in(_digest(r), "#1 reporter=dcid:USA", "#2 reporter=dcid:DEU"),
            _assert_none_in(_digest(r), "dcid:JPN", "dcid:FRA"),
        ),
    ),
    _Entry(
        "List every observation in the USA's ODA disbursement series worth more than $50 billion.",
        _run_usa_observations_over_50b,
        check=lambda r: (
            _assert_all_in(_digest(r), "2 kept row(s)", "55303000000", "63575000000"),
            _assert_none_in(_digest(r), "47805000000"),
        ),
    ),
    _Entry(
        "Rank donors by their share of official development assistance to health.",
        _run_donors_share_of_health_oda,
        check=lambda r: (
            _assert_all_in(_digest(r), "#1 reporter=dcid:FRA", "0.75"),
            _assert_none_in(_digest(r), "#1 reporter=dcid:USA", "#1 reporter=dcid:GBR"),
        ),
    ),
    _Entry(
        "How has Africa's share of Chinese lending changed since 2010, in percentage points?",
        _run_africa_share_of_chinese_lending_since_2010,
        check=lambda r: _assert_all_in(_digest(r), "+0.25", "percentage points", "0.2 -> 0.45"),
    ),
    _Entry(
        "How has Kenya's rank among aid recipients changed since 2010?",
        _run_kenya_rank_change_since_2010,
        check=lambda r: _assert_all_in(_digest(r), "1 place better", "2 -> 1"),
    ),
    _Entry(
        "Give me the total (bilateral and imputed multilateral) aid France gave Guatemala "
        "between 2010 and the latest",
        _run_france_guatemala_total_aid,
        # 2010 is bilateral only, so the sum covers 2011-2012 and names it.
        check=lambda r: _assert_all_in(
            _digest(r),
            "2011: 4250000 + 1500000 = 5750000",
            "6100000",
            "2010: not in the sum; only the left side",
        ),
    ),
)


@pytest.fixture
def _corpus_store():
    store = InMemoryMaterialisationStore()
    set_store(store)
    yield store
    set_store(None)


async def test_every_covered_entrys_chain_actually_executes(_corpus_store) -> None:
    """Runs every covered corpus entry's own `verb_chain` against a real,
    registered frame and checks the final step's own outcome -- success or a
    specific named refusal, per `_ENTRY_PLANS`' own `expect_refusal`, never
    "whatever happens": an assertion that accepted any outcome would prove
    only that the call didn't crash, not that the chain does what its
    question claims."""
    entries = {entry["question"]: entry for entry in _load_entries() if not entry.get("uncovered")}
    plans = {plan.question: plan for plan in _ENTRY_PLANS}

    assert plans.keys() == entries.keys(), (
        f"covered corpus entries with no execution plan: {entries.keys() - plans.keys()}; "
        f"execution plans naming no covered corpus entry: {plans.keys() - entries.keys()}"
    )

    for question, plan in plans.items():
        result = await plan.run(_corpus_store)
        if plan.expect_refusal is None:
            assert result.is_error is False, (
                f"{question!r} (chain {entries[question]['verb_chain']!r}) expected success, "
                f"refused instead: {result.content[0].text}"
            )
            if plan.check is not None:
                plan.check(result)
        else:
            assert result.is_error is True, (
                f"{question!r} (chain {entries[question]['verb_chain']!r}) expected refusal "
                f"{plan.expect_refusal!r}, succeeded instead"
            )
            assert result.structured_content["code"] == plan.expect_refusal, (
                f"{question!r} expected refusal {plan.expect_refusal!r}, got "
                f"{result.structured_content['code']!r}"
            )

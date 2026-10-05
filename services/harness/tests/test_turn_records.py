"""Tests for `harness.turn_records`: plan records and search readings, and their text.

The fixture holds two compiled plans from recorded Studio sessions (their
StatVar member lists cut to three, their `stat_var_member_count` as
recorded) and the `search` and `fetch` tool outputs of two recorded turns.
"""

from __future__ import annotations

import copy
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from harness.plan_display import requirement_rows
from harness.turn_records import (
    PlanAsk,
    PlanRecord,
    SearchReading,
    WindowReading,
    render_readings,
)
from planner.plan import (
    CUT_MARKER,
    AskAnswer,
    Decision,
    DecisionFixes,
    Decline,
    Plan,
    ResolvedBy,
    UnsupportedOperation,
    plan_revision,
    search_pin_slot,
)

_FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "turn_records_recorded.json").read_text()
)

_SEARCHED_SPEC = "spec_6d04429386509f8f28af0a969ffa478b"
_BILATERAL_SPEC = "spec_4b23d29a5d384574e1a40d2ffb22e952"
_IMPUTED_SPEC = "spec_d483f9b9d15fc52a1e1652ca32546c96"


def _plan(name: str) -> Plan:
    return Plan.model_validate(_FIXTURE["plans"][name])


def _readings(name: str) -> list[SearchReading]:
    """The turn's readings in call order, each with the outcome of a later fetch of its spec_id."""
    readings: list[SearchReading] = []
    for call in _FIXTURE["searches"][name]:
        output = call["output"]
        if call["tool"] == "search":
            skills = [skill["id"] for skill in output.get("skills") or []]
            readings += [SearchReading.from_part(part, skills=skills) for part in output["parts"]]
        else:
            readings = [
                reading.with_fetch(output) if reading.spec_id == output["spec_id"] else reading
                for reading in readings
            ]
    return readings


def _member_dcids(plan: Plan) -> set[str]:
    assert plan.workflow is not None
    members = {dcid for source in plan.workflow.sources for dcid in source.stat_vars}
    for step in plan.workflow.steps:
        dcids = step.args.get("resolved_stat_var_dcids")
        if isinstance(dcids, list):
            members.update(str(dcid) for dcid in dcids)
    return members


_TOP_DONORS = """\
Plan {revision} (approved), status approved: "top donors to guatemala since 2020"
period 2020-2023, assumed by the planner (decision time_period)
decisions:
- time_period = 2020-2023 [planner_default]: Captures the most recent complete reporting years for official development finance data.
- measure_choice = official development assistance [planner_default]: Official development assistance (ODA) is the standard international metric for aid flows.
- ranking_scope = all reporting donors in the population [planner_default]: Evaluating all reporting donors gives a complete view of aid to Guatemala.
- ranking_limit = top 10 donors [planner_default]: A top 10 ranking provides a detailed yet concise overview of the leading donors.
- aggregation_method = cumulative sum across years [planner_default]: Summing across the 2020-2023 window reflects the total volume of assistance provided since 2020.
sources:
- spec_fd8b7635d17fe5f552ae2f31b6c78b89 OECD Creditor Reporting System (CRS) for official development assistance; data 2020-2023 (exact) within the years asked; family of 489 StatVars; from translator
outputs: top_donors
steps:
- guatemala_oda: Fetch official development assistance for Guatemala and every donor (spec_fd8b7635d17fe5f552ae2f31b6c78b89, sum)
- top_donors: Rank the top 10 donors by the total of official development assistance [failed]"""

_JOIN = """\
Plan {revision} (not approved), status pending_approval: "Total aid (sum of bilateral ODA and imputed multilateral ODA) from France to Guatemala from 2010 to 2023"
period 2010-2023, as the requirements state it, no period decision
decisions:
- bilateral_source = OECD Creditor Reporting System (CRS) [planner_default]: Use OECD Creditor Reporting System (CRS) bilateral ODA dataset from earlier context.
- imputed_source = ONE Imputed Multilateral Sectors [planner_default]: Use ONE Imputed Multilateral Sectors dataset from earlier context.
- combining_method = join [planner_default]: Join the bilateral ODA and imputed multilateral ODA series into one combined frame across common years.
warnings:
- window_short on imputed_multilateral_oda: the source covers 2011-2023, the requirement asks for 2010-2023
sources:
- spec_4b23d29a5d384574e1a40d2ffb22e952 OECD Creditor Reporting System (CRS) for official development assistance; data 2010-2023 (exact) within the years asked; family of 57 StatVars; from search
- spec_d483f9b9d15fc52a1e1652ca32546c96 ONE Imputed Multilateral Sectors for official development assistance, imputed multilateral ODA; data 2011-2023 (exact) within the years asked; family of 133 StatVars; from seed
outputs: joined_oda
steps:
- bilateral_oda: Fetch official development assistance for France and Guatemala (spec_4b23d29a5d384574e1a40d2ffb22e952, sum)
- imputed_multilateral_oda: Fetch official development assistance, imputed multilateral ODA for France and Guatemala (spec_d483f9b9d15fc52a1e1652ca32546c96, sum)
- joined_oda: Join official development assistance with official development assistance, imputed multilateral ODA"""


def _top_donors() -> tuple[Plan, PlanRecord]:
    plan = _plan("guatemala_top_donors").approve()
    record = PlanRecord.from_plan(plan, searched={_SEARCHED_SPEC}, failed_step="top_donors")
    return plan, record


def _join() -> tuple[Plan, PlanRecord]:
    plan = _plan("france_guatemala_join")
    record = PlanRecord.from_plan(plan, searched={_BILATERAL_SPEC}, seeded={_IMPUTED_SPEC})
    return plan, record


@pytest.mark.parametrize(("build", "golden"), [(_top_donors, _TOP_DONORS), (_join, _JOIN)])
def test_plan_record_renders_a_recorded_plan_whole(build: Any, golden: str) -> None:
    plan, record = build()
    assert record.revision == plan_revision(plan)
    assert record.render(budget=10_000) == golden.format(revision=record.revision[:12])


def test_plan_record_states_period_basis_failed_step_and_family_count() -> None:
    _, top_donors = _top_donors()
    _, join = _join()

    assert [(p.start, p.end, p.assumed, p.decision) for p in top_donors.periods] == [
        (2020, 2023, True, "time_period")
    ]
    assert [(p.start, p.end, p.assumed, p.decision) for p in join.periods] == [
        (2010, 2023, False, None)
    ]
    assert [step.id for step in top_donors.steps if step.failed] == ["top_donors"]
    assert top_donors.approved_revision == top_donors.revision
    # The fixture lists three members; the family is counted as search confirmed it.
    assert [(s.spec_id, s.members, s.origin) for s in join.sources] == [
        (_BILATERAL_SPEC, 57, "search"),
        (_IMPUTED_SPEC, 133, "seed"),
    ]
    assert [w.code for w in join.warnings] == ["window_short"]


@pytest.mark.parametrize(
    ("requested_window", "coverage_source", "fact"),
    [
        (
            {"kind": "range", "start_year": 2020, "end_year": 2023},
            "requested",
            "data 2020-2023 (exact) within the years asked",
        ),
        ({"kind": "all"}, "coverage", "coverage 2020-2023 (exact)"),
        # One end named: the end year is the source's own, as `_window` reads it.
        ({"kind": "range", "start_year": 2020}, "coverage", "coverage 2020-2023 (exact)"),
        # A range the coverage spills past does not bound it.
        (
            {"kind": "range", "start_year": 2021, "end_year": 2023},
            "coverage",
            "coverage 2020-2023 (exact)",
        ),
    ],
)
def test_a_source_window_inside_the_years_asked_never_reads_as_its_coverage(
    requested_window: dict[str, Any], coverage_source: str, fact: str
) -> None:
    raw = copy.deepcopy(_FIXTURE["plans"]["guatemala_top_donors"])
    (source,) = raw["workflow"]["sources"]
    source["identity"]["requested_window"] = requested_window

    record = PlanRecord.from_plan(Plan.model_validate(raw))

    assert [s.coverage_source for s in record.sources] == [coverage_source]
    assert f"; {fact};" in record.render(budget=10_000)


@pytest.mark.parametrize("name", ["guatemala_top_donors", "france_guatemala_join"])
@pytest.mark.parametrize("budget", [10_000, 900, 300])
def test_plan_record_never_holds_or_renders_a_member_list(name: str, budget: int) -> None:
    plan = _plan(name)
    members = _member_dcids(plan)
    assert len(members) >= 3
    record = PlanRecord.from_plan(plan)
    stored = record.model_dump_json()
    rendered = record.render(budget=budget)
    assert not [dcid for dcid in members if dcid in stored or dcid in rendered]


@pytest.mark.parametrize("budget", range(150, 1800, 50))
def test_plan_record_render_fits_its_budget_and_repeats(budget: int) -> None:
    _, record = _top_donors()
    text = record.render(budget=budget)
    assert len(text) <= budget
    assert text == record.render(budget=budget)


def test_plan_record_folds_steps_but_keeps_outputs_and_the_failed_step() -> None:
    plan, _ = _join()
    record = PlanRecord.from_plan(plan, failed_step="bilateral_oda")
    full = record.render(budget=10_000)
    # One character short of the whole render: only the first level applies.
    text = record.render(budget=len(full) - 1)

    step_lines = text.split("steps:\n")[1].splitlines()
    assert step_lines[0].startswith("- bilateral_oda:")
    assert step_lines[0].endswith("[failed]")
    assert step_lines[1].startswith("- joined_oda:")
    assert step_lines[2] == "- and 1 more step"
    assert "window_short on imputed_multilateral_oda" in text


_DECLINE = Decline(
    ask="the total of bilateral and imputed multilateral aid",
    code=UnsupportedOperation.CALCULATION_OVER_COMBINED,
)
_DECLINE_LINE = "- the total of bilateral and imputed multilateral aid [calculation_over_combined]"


def test_plan_record_carries_what_the_plan_declined() -> None:
    plan = _plan("france_guatemala_join")
    plan = plan.model_copy(update={"declined": (_DECLINE,)})

    record = PlanRecord.from_plan(plan)

    assert record.declined == (_DECLINE,)
    assert _DECLINE_LINE in record.render(budget=10_000).splitlines()


def test_plan_record_digest_line_states_period_assumptions_sources_and_declines() -> None:
    _, top_donors = _top_donors()
    _, joined = _join()
    declined = joined.model_copy(update={"declined": (_DECLINE,)})

    top_line = top_donors.digest_line().split(" | ")
    join_line = declined.digest_line().split(" | ")

    assert top_line[0] == "period 2020-2023, assumed by the planner (decision time_period)"
    assert top_line[1].startswith("assumed: time_period = 2020-2023: Captures the most recent")
    assert top_line[-1] == (
        "sources: spec_fd8b7635d17fe5f552ae2f31b6c78b89 OECD Creditor Reporting System (CRS); "
        "data 2020-2023 (exact) within the years asked; family of 489 StatVars; from translator"
    )
    assert join_line[0] == "period 2010-2023, as the requirements state it, no period decision"
    assert _BILATERAL_SPEC in join_line[2] and _IMPUTED_SPEC in join_line[2]
    assert join_line[-1] == (
        "not computed: the total of bilateral and imputed multilateral aid "
        "[calculation_over_combined]"
    )


def test_plan_record_reads_each_source_no_search_returned() -> None:
    _, top_donors = _top_donors()
    _, joined = _join()
    observed_at = datetime(2026, 10, 5, tzinfo=UTC)

    (translated,) = top_donors.source_readings(observed_at=observed_at)
    (seeded,) = joined.source_readings(observed_at=observed_at)

    assert (translated.spec_id, translated.origin) == (
        "spec_fd8b7635d17fe5f552ae2f31b6c78b89",
        "translator",
    )
    assert translated.window == WindowReading(start=2020, end=2023, source="requested")
    assert (translated.coverage_kind, translated.observed_at) == ("exact", observed_at)
    assert (seeded.spec_id, seeded.origin, seeded.text) == (
        _IMPUTED_SPEC,
        "seed",
        "official development assistance, imputed multilateral ODA",
    )


def test_a_search_pin_decision_is_left_out_of_the_record_and_the_card() -> None:
    joined = _plan("france_guatemala_join")
    slot = search_pin_slot(_BILATERAL_SPEC)
    pin = Decision(
        slot=slot,
        description="Whether to pin the source a search settled on.",
        options=(f"pin {_BILATERAL_SPEC}", "leave the source unpinned"),
        chosen=f"pin {_BILATERAL_SPEC}",
        resolved_by=ResolvedBy.PLANNER_DEFAULT,
        rationale="A search this turn settled on this source for the measure.",
        fixes=DecisionFixes(source_spec_id=_BILATERAL_SPEC),
    )
    plan = joined.model_copy(
        update={
            "decisions": (*joined.decisions, pin),
            "evidence_requirements": tuple(
                r.model_copy(update={"decision_ids": (*r.decision_ids, slot)})
                if r.id == "bilateral_oda"
                else r
                for r in joined.evidence_requirements
            ),
        }
    )

    record = PlanRecord.from_plan(plan)

    assert [d.slot for d in record.decisions] == [d.slot for d in joined.decisions]
    decisions = record.render(budget=10_000).split("decisions:")[1].split("warnings:")[0]
    assert "spec_" not in decisions
    assert all(slot not in row["decision_ids"] for row in requirement_rows(plan))


def test_plan_record_maps_each_ask_to_its_output_steps_or_its_decline() -> None:
    plan = _plan("france_guatemala_join").model_copy(
        update={
            "asks": ("bilateral and multilateral aid side by side", _DECLINE.ask),
            "answered": (AskAnswer(ask=0, outputs=("joined_oda",)),),
            "declined": (_DECLINE.model_copy(update={"ask_ref": 1}),),
        }
    )

    record = PlanRecord.from_plan(plan)

    assert record.asks == (
        PlanAsk(ask="bilateral and multilateral aid side by side", outputs=("joined_oda",)),
        PlanAsk(
            ask=_DECLINE.ask,
            declined=UnsupportedOperation.CALCULATION_OVER_COMBINED,
            not_computed=_DECLINE.ask,
        ),
    )
    assert record.declined == ()
    lines = record.render(budget=10_000).splitlines()
    assert '- 0 "bilateral and multilateral aid side by side": answered by joined_oda' in lines
    assert f'- 1 "{_DECLINE.ask}": not computed [calculation_over_combined]' in lines
    assert f"not computed: {_DECLINE.ask} [calculation_over_combined]" in record.digest_line()


def test_plan_record_states_what_an_ask_answered_in_part_leaves_out() -> None:
    part = "a total over the two series combined"
    plan = _plan("france_guatemala_join").model_copy(
        update={
            "asks": (_DECLINE.ask,),
            "answered": (AskAnswer(ask=0, outputs=("joined_oda",)),),
            "declined": (_DECLINE.model_copy(update={"ask": part, "ask_ref": 0}),),
        }
    )

    record = PlanRecord.from_plan(plan)

    assert record.asks == (
        PlanAsk(
            ask=_DECLINE.ask,
            outputs=("joined_oda",),
            declined=UnsupportedOperation.CALCULATION_OVER_COMBINED,
            not_computed=part,
        ),
    )
    lines = record.render(budget=10_000).splitlines()
    assert (
        f'- 0 "{_DECLINE.ask}": answered by joined_oda; not computed "{part}" '
        "[calculation_over_combined]"
    ) in lines
    assert f"not computed: {part} [calculation_over_combined]" in record.digest_line()


def test_plan_record_loses_steps_then_measures_then_sources_before_any_decision() -> None:
    _, joined = _join()
    record = joined.model_copy(
        update={
            "steps": joined.steps * 15,
            "sources": joined.sources * 6,
            "declined": (_DECLINE,),
        }
    )
    full = record.render(budget=100_000)
    reached_the_cut = False
    for budget in range(300, len(full), 20):
        text = record.render(budget=budget)
        assert len(text) <= budget
        steps_listed = any(f"- {step.id}: " in text for step in record.steps)
        sources_listed = any(f"- {source.spec_id} " in text for source in record.sources)
        measures_listed = " for official development assistance;" in text
        kept = all(f"- {d.slot} = {d.chosen} [" in text for d in record.decisions) and all(
            f"- {w.code} on {w.requirement_id}" in text for w in record.warnings
        )
        assert not kept or _DECLINE_LINE in text
        assert not measures_listed or sources_listed
        if sources_listed and not measures_listed:
            assert not steps_listed
        whole = all(
            f"[{d.resolved_by.value}]: {d.rationale}\n" in text + "\n" for d in record.decisions
        )
        if not (kept and whole):
            assert not (steps_listed or sources_listed)
        reached_the_cut |= kept and text.endswith(CUT_MARKER)
    assert reached_the_cut


_KENYA = """\
Search readings (oldest first):
1. "aid by sector" no_data
   warnings: retrieval_score_weak, !time_text_not_parsed, !role_text_not_recognised, skills_not_matchable
2. "official development assistance by sector" definite spec_ab788eeca3bf5fc6bddb31cef15936e3
   Gross disbursements (nominal; US dollar) for official development assistance to Kenya, latest values, 1990 to 2024, according to OECD Creditor Reporting System (CRS). In 2024, the latest year with data, there are 1,852 observations across 1,852 series, covering 75 donors and 1 recipient. No year was named.
   wide
   warnings: retrieval_score_weak, role_scope_defaulted
   fetched: 2982 observations, 1990\u20132024 · OECD Creditor Reporting System (CRS) · 2982 series
3. "aid by sector" definite spec_171f6e6af278db843fb67e29ec117d20
   Gross disbursements (nominal; US dollar) for social infrastructure and services, economic infrastructure and services, productive sector, commodity aid and general programme assistance, multisector, humanitarian aid, administrative costs of donors, unallocated and unspecified, action relating to debt, and support for refugees and asylum seekers in donor countries for official development assistance to Kenya, latest values, 1990 to 2024, according to OECD Creditor Reporting System (CRS). In 2024, the latest year with data, there are 1,852 observations across 1,852 series, covering 75 donors and 1 recipient. No year was named.
   wide
   warnings: retrieval_score_weak, role_scope_defaulted
   fetched: 2982 observations, 1990\u20132024 · OECD Creditor Reporting System (CRS) · 2982 series
4. "official development assistance" definite spec_ab788eeca3bf5fc6bddb31cef15936e3
   Gross disbursements (nominal; US dollar) for official development assistance to Kenya, latest values, 1990 to 2024, according to OECD Creditor Reporting System (CRS). In 2024, the latest year with data, there are 1,852 observations across 1,852 series, covering 75 donors and 1 recipient. No year was named.
   wide
   warnings: retrieval_score_weak, role_scope_defaulted
Skills named (* = already read): oda-crs*, oda-concepts, oda-dataset-selection"""

_FRANCE = """\
Search readings (oldest first):
1. "bilateral official development assistance" candidates
   candidate spec_75ddf108608eadff0b7918ae26e529c7 Outbound official development assistance (ODA) from OECD-DAC countries on grant equivalent basis (united states dollars at constant prices) in France and Guatemala, 2018 to 2023, according to SDG Global Database. This series starts in 2018.
   candidate spec_8517f68e739983ff59327b2becb2d0f4 Net outbound official development assistance (ODA) from OECD-DAC countries (united states dollars at constant prices) in France and Guatemala, 2010 to 2023, according to SDG Global Database.
   warnings: retrieval_score_weak, direction_discarded, query_qualifier_dropped
2. "official development assistance" definite spec_4b23d29a5d384574e1a40d2ffb22e952
   Gross disbursements (nominal; US dollar) for official development assistance to Guatemala from France, 2010 to 2023, according to OECD Creditor Reporting System (CRS). In 2023, the latest year with data in the years asked, there are 15 observations across 15 series, covering 1 donor and 1 recipient. The question names 2010 to 2023. These counts are for 2023 alone.
   window 2010-2023 (requested); wide
   warnings: retrieval_score_weak
   fetched: 14 observations, 2010\u20132023 · OECD Creditor Reporting System (CRS)
Skills named (* = already read): oda-crs*"""


@pytest.mark.parametrize(
    ("name", "golden"), [("kenya_sectors", _KENYA), ("france_bilateral", _FRANCE)]
)
def test_readings_render_recorded_search_parts_whole(name: str, golden: str) -> None:
    text = render_readings(_readings(name), budget=10_000, skills_read={"oda-crs"}).text
    assert text == golden


def test_reading_keeps_the_part_facts_as_fields() -> None:
    no_data, definite, *_ = _readings("kenya_sectors")
    candidates, requested = _readings("france_bilateral")

    assert (no_data.status, no_data.spec_id) == ("no_data", None)
    assert [w.code for w in no_data.warnings][1:3] == [
        "time_text_not_parsed",
        "role_text_not_recognised",
    ]
    assert definite.skills == ("oda-crs",)
    assert definite.fetch is not None
    assert definite.fetch.summary is not None
    assert [c.spec_id for c in candidates.candidates] == [
        "spec_75ddf108608eadff0b7918ae26e529c7",
        "spec_8517f68e739983ff59327b2becb2d0f4",
    ]
    assert requested.window is not None
    assert (requested.window.start, requested.window.end, requested.window.source) == (
        2010,
        2023,
        "requested",
    )


def test_reading_reads_the_structured_fields_a_part_adds() -> None:
    part = {
        **_FIXTURE["searches"]["france_bilateral"][1]["output"]["parts"][0],
        "coverage_kind": "exact",
        "requested_window_kind": "range",
        "entity_roles": ["origin", "destination"],
        "open_entity_axes": [{"column": "origin", "members": 75}],
        "recent_years": [
            {"year": 2024, "observations": 0, "series": 0},
            {"year": 2023, "observations": 15, "series": 15},
        ],
    }
    reading = SearchReading.from_part(part, origin="translator").with_fetch(
        {"spec_id": part["spec_id"], "refusal": "too large", "code": "frame_too_large"}
    )

    assert reading.coverage_kind == "exact"
    assert reading.roles == ("origin", "destination")
    assert [(a.column, a.members) for a in reading.open_axes] == [("origin", 75)]
    assert [(y.year, y.observations) for y in reading.recent_years] == [(2024, 0), (2023, 15)]
    assert reading.fetch is not None
    assert reading.fetch.refusal_code == "frame_too_large"
    lines = render_readings([reading], budget=10_000).text.splitlines()
    assert lines[1].endswith("(from translator)")
    assert lines[3] == (
        "   window 2010-2023 (requested); coverage exact; asked range; recent years 2024: 0 obs "
        "in 0 series, 2023: 15 obs in 15 series; wide; roles donor/creditor/exporter, "
        "recipient/debtor/importer; open donor/creditor/exporter (75 members)"
    )
    assert lines[-1] == "   fetch refused: frame_too_large"


@pytest.mark.parametrize("name", ["kenya_sectors", "france_bilateral"])
@pytest.mark.parametrize("budget", range(100, 2400, 50))
def test_readings_render_fits_its_budget_repeats_and_lists_the_ids_it_names(
    name: str, budget: int
) -> None:
    readings = _readings(name)
    rendered = render_readings(readings, budget=budget)
    assert len(rendered.text) <= budget
    assert rendered == render_readings(readings, budget=budget)
    every_id = {
        spec_id
        for reading in readings
        for spec_id in (reading.spec_id, *(c.spec_id for c in reading.candidates))
        if spec_id
    }
    assert set(rendered.spec_ids) == {
        spec_id for spec_id in every_id if re.search(rf"{spec_id}\b", rendered.text)
    }


def test_readings_fold_the_oldest_first_and_keep_ids_and_blocking_codes() -> None:
    readings = _readings("kenya_sectors")
    text = render_readings(readings, budget=700).text

    lines = text.splitlines()
    assert lines[1] == (
        '1. "aid by sector" no_data; warnings: retrieval_score_weak, '
        "!time_text_not_parsed, !role_text_not_recognised, skills_not_matchable"
    )
    assert lines[2].startswith(
        '2. "official development assistance by sector" definite '
        "spec_ab788eeca3bf5fc6bddb31cef15936e3; fetched;"
    )
    assert lines[-1].startswith("Skills named")


def test_readings_drop_the_oldest_with_a_count_when_folding_is_not_enough() -> None:
    text = render_readings(_readings("kenya_sectors"), budget=420).text

    lines = text.splitlines()
    assert lines[1] == "(2 earlier readings not shown)"
    assert lines[2].startswith('3. "aid by sector" definite spec_171f6e6af278db843fb67e29ec117d20')


def test_a_no_data_reading_names_its_reason_in_place_of_its_status() -> None:
    part = {"variable": "aid", "status": "no_data", "reason": "role_not_resolved"}

    text = render_readings([SearchReading.from_part(part)], budget=10_000).text

    assert text.splitlines()[1] == '1. "aid" no data: role_not_resolved'

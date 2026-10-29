"""An input alignment moved onto a compiled spec_id keeps its own identity.

Alignment compiles an input of a calculation onto the options its other inputs are
counted on, and Search answers the compile with a spec_id, canonical over the
identity, that another requirement may pin as its own. Each requirement keeps its
own provenance whatever spec_id it shares: who settled each axis, which pin it was
moved from, and which row of the plan card governs it.

Two calculations carry these tests. "How does US gross ODA compare with its
average?" compares a total the user stated gross with an average Search left on
net, so alignment compiles the average's input on gross, onto the spec_id of the
user's own gross pin. "How does US ODA compare with France's multilateral ODA?"
moves the US total onto DAC1's multilateral row, which Search answers with the
spec_id it gives US multilateral ODA, so a later share of that ODA over the total
pins the same spec_id directly.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any, cast

from analysis_mcp.server import mcp as analysis_mcp
from compiler.compile import compile_plan
from compiler.surface import McpSurface, ToolCallResult, ToolSurface
from data_mcp.definition_store import canonicalize, spec_id_for
from data_mcp.search_result import SEARCH_ENVELOPE_META_KEY
from data_mcp.server import mcp as data_mcp
from harness.plan_display import choice_id, shown_decisions
from harness.scope_choices import planner_axes
from planner.plan import (
    ASKS_STATE_KEY,
    AnswerMode,
    AskAnswer,
    DatasetRequirement,
    Decision,
    Entity,
    Period,
    PinSource,
    Plan,
    Reduction,
    ResolvedBy,
    load_plan,
    pin_sources,
    save_pin_sources,
)
from planner.tool_results import Outcome, ProblemCode
from planner.workflow import ToolId, Workflow
from submission.service import COMBINE_OPTION
from translator.discovery import build_discovery_tools, load_log
from translator.mechanical import build_translation

from ..basis_alignment import us_2023_block
from ..decision_drafts import drafts
from .test_agent import (
    _COMPARE,
    _GHANA_DEBT_IDENTITY,
    _US_MULTILATERAL,
    _US_TOTAL,
    DAC2A,
    IDS,
    DecisionFactory,
    RequirementFactory,
    _another_part,
    _combination_asked,
    _dac_countries,
    _difference,
    _FakeToolContext,
    _pins,
    _reduced,
    _SearchRefused,
    _share,
    _show_blocks,
    _sum_of_two_sources,
    _tools_by_name,
    _tools_reading,
)

# --- a gross total compared with a net average ------------------------------------


def _stated_gross() -> dict[str, Any]:
    """US DAC1 total ODA in 2023 counted gross, as dev Search answered it, its gross
    entry read from the asked words ("gross ODA") in place of the choice it was
    recorded from."""
    block = json.loads(json.dumps(us_2023_block("dac1_total_gross")))
    (entry,) = block["resolved"]["scope_choices"]["entries"]
    entry["origin"] = {"kind": "text", "text": "gross", "start": 3, "end": 8}
    return block


_GROSS = _stated_gross()
_NET = f"spec_{'7' * 32}"


def _left_on_net() -> dict[str, Any]:
    """`_GROSS`'s part counted net by default, under the spec_id `_NET`: the measure
    basis no one stated, as Search answers an ask that names none."""
    text = json.dumps(_GROSS)
    for old, new in ((_GROSS["spec_id"], _NET), ('"ref": "Gross"', '"ref": "Net"')):
        assert old in text, old
        text = text.replace(old, new)
    block = json.loads(text)
    scope = block["resolved"]["scope_choices"]
    (basis,) = [c for c in scope["choices"] if c["axis"] == "measure_basis"]
    basis["active"], basis["provenance"] = "measure_basis:net", "default"
    scope["entries"] = [e for e in scope["entries"] if e["axis"] != "measure_basis"]
    return block


class _Compiles:
    """A tool surface whose `resolve_definition` answers `block` to a resubmit of the
    part `base` names on the options `option_ids`, as dev Search answered that post,
    recording each call's arguments in `calls`. Search refuses every other call."""

    def __init__(
        self, block: Mapping[str, Any], *, base: Mapping[str, Any], option_ids: list[str]
    ) -> None:
        self._block = block
        self._base = base
        self._option_ids = option_ids
        self.calls: list[Mapping[str, Any]] = []

    async def list_tools(self) -> list[Any]:
        return []

    async def call_tool(
        self, *, tool: ToolId, arguments: Mapping[str, Any], on_progress: Any = None
    ) -> ToolCallResult:
        del on_progress
        if tool.name != "resolve_definition":
            return ToolCallResult(is_error=True, structured_content={"code": "refused"})
        self.calls.append(arguments)
        assert arguments["spec_id"] == self._base["spec_id"]
        assert arguments["identity"] == self._base["identity"]
        assert [e["option_id"] for e in arguments["scope_entries"]][-1:] == self._option_ids
        part = {"status": "definite", "spec": json.loads(json.dumps(self._block))}
        return ToolCallResult(is_error=False, meta={SEARCH_ENVELOPE_META_KEY: {"parts": [part]}})


def _gross_compiles() -> _Compiles:
    """`_GROSS` answering a resubmit of its part on gross disbursements."""
    scope = _GROSS["resolved"]["scope_choices"]
    return _Compiles(_GROSS, base=scope["base"], option_ids=["measure_basis:gross"])


async def _gross_compared_with_net_average(
    ctx: _FakeToolContext, decision: DecisionFactory
) -> Plan:
    """Propose US total ODA in 2023, its total stated gross, compared with its average
    left on net, both on one DAC1 part. The sides differ by their operation, a sum
    against a mean, so the net input is aligned on gross, which Search compiles to
    the spec_id of the user's gross pin. Returns the plan on record."""
    save_pin_sources(
        ctx.state,
        {
            _GROSS["spec_id"]: PinSource(block=_GROSS, settled=True),
            _NET: PinSource(block=_left_on_net(), settled=True),
        },
    )
    question = "How does US gross ODA in 2023 compare with its average?"
    ctx.state[ASKS_STATE_KEY] = [question]
    surface = _gross_compiles()

    def dataset(requirement_id: str, pin: str) -> DatasetRequirement:
        return DatasetRequirement(
            id=requirement_id,
            decision_ids=["measure"],
            measure="US total ODA",
            entities=[Entity(name="United States of America", role="donor")],
            period=Period(start=2023, end=2023),
            source_spec_id=pin,
        )

    result = await _tools_reading(cast(ToolSurface, surface))["propose_plan"].func(
        question=question,
        answer_mode=AnswerMode.COMPUTED,
        decisions=drafts([decision()]),
        answered=[AskAnswer(ask=0, outputs=(_COMPARE,))],
        evidence_requirements=[
            dataset("first", _GROSS["spec_id"]),
            dataset("second", _NET),
            _reduced("one", "first", Reduction.SUM),
            _reduced("two", "second", Reduction.MEAN),
            _difference("one", "two"),
        ],
        tool_context=ctx,
    )
    assert result["outcome"] == Outcome.OK.value, result
    assert len(surface.calls) == 1
    plan = load_plan(ctx.state)
    assert plan is not None
    return plan


async def _compiled(ctx: _FakeToolContext) -> Workflow:
    """The plan `ctx` holds, each input discovered from its pin with no call to Search,
    translated and compiled against data-mcp's and analysis-mcp's schemas from the
    source discovery recorded for each requirement."""
    plan = load_plan(ctx.state)
    assert plan is not None
    refused = _SearchRefused()
    (discover_source,) = [
        t
        for t in build_discovery_tools(surface=cast(ToolSurface, refused))
        if t.name == "discover_source"
    ]
    pins = {rid: pin for rid, pin in _pins(ctx) if pin is not None}
    for requirement_id in pins:
        found = await discover_source.func(requirement_id=requirement_id, tool_context=ctx)
        assert found["status"] == "definite", (requirement_id, found)
    assert refused.calls == []
    recorded = load_log(ctx.state).sources
    discovered = {rid: recorded[rid][pin] for rid, pin in pins.items()}
    tools = await McpSurface(
        sources={"data-mcp": data_mcp, "analysis-mcp": analysis_mcp}
    ).list_tools()
    compiled = compile_plan(
        plan=plan,
        translation=build_translation(plan, {rid: s.spec_id for rid, s in discovered.items()}),
        schemas={tool.name: tool for tool in tools},
        discovered=discovered,
    )
    assert compiled.workflow is not None, compiled.issues
    return compiled.workflow


def _basis_rows(plan: Plan) -> dict[str, tuple[ResolvedBy, frozenset[str]]]:
    """The plan card's measure-basis rows, by row id, each with who settled it and the
    dataset requirements citing it."""
    datasets = [r for r in plan.evidence_requirements if isinstance(r, DatasetRequirement)]
    return {
        choice_id(plan, d): (
            d.resolved_by,
            frozenset(r.id for r in datasets if d.slot in r.decision_ids),
        )
        for d in shown_decisions(plan)
        if d.scope is not None and d.scope.axis == "measure_basis"
    }


async def test_a_gross_total_compared_with_a_net_average_compiles_both_on_gross(
    decision: DecisionFactory,
) -> None:
    """The proposal compiles the average's input on gross through the resubmit a
    switch posts, and the workflow reads both inputs from the gross pin. The user's
    gross pin keeps the source they were shown, its basis theirs."""
    ctx = _FakeToolContext()
    plan = await _gross_compared_with_net_average(ctx, decision)

    assert [
        (r.id, r.source_spec_id)
        for r in plan.evidence_requirements
        if isinstance(r, DatasetRequirement)
    ] == [("first", _GROSS["spec_id"]), ("second", _GROSS["spec_id"])]
    assert pin_sources(ctx.state)[_GROSS["spec_id"]] == PinSource(block=_GROSS, settled=True)
    assert planner_axes(pin_sources(ctx.state)[_GROSS["spec_id"]].block) == frozenset()

    workflow = await _compiled(ctx)

    assert {b.requirement_id: b.spec_id for b in workflow.evidence_bindings} == {
        "first": _GROSS["spec_id"],
        "second": _GROSS["spec_id"],
    }
    assert [f"{step.tool.server}.{step.tool.name}" for step in workflow.steps] == [
        "data-mcp.get_data",
        "analysis-mcp.aggregate",
        "data-mcp.get_data",
        "analysis-mcp.aggregate",
        "analysis-mcp.compare",
    ]
    assert workflow.terminal_steps == (_COMPARE,)


async def test_the_card_shows_the_averages_basis_as_the_planners_and_the_totals_as_the_users(
    decision: DecisionFactory,
) -> None:
    """The total's gross basis is the user's, read off the search their words settled,
    and the average's is the planner's, which alignment moved off its net pin. The
    card lists two basis rows, each governing its own input."""
    ctx = _FakeToolContext()
    plan = await _gross_compared_with_net_average(ctx, decision)

    cited = {
        rid: [
            d.resolved_by
            for d in plan.decisions
            if d.slot in plan.requirement(rid).decision_ids
            and d.scope is not None
            and d.scope.axis == "measure_basis"
        ]
        for rid in ("first", "second")
    }

    assert cited == {"first": [ResolvedBy.USER], "second": [ResolvedBy.PLANNER_DEFAULT]}
    assert sorted(_basis_rows(plan).values(), key=lambda row: sorted(row[1])) == [
        (ResolvedBy.USER, frozenset({"first"})),
        (ResolvedBy.PLANNER_DEFAULT, frozenset({"second"})),
    ]


# --- US ODA compared with France's multilateral ODA ---------------------------------


def _multilateral_compiles() -> _Compiles:
    """`_US_MULTILATERAL` answering a resubmit of `_US_TOTAL`'s part on the
    multilateral row: spec_ids are canonical over the identity, so the compile is
    US multilateral ODA's own spec_id."""
    scope = _US_TOTAL["resolved"]["scope_choices"]
    return _Compiles(_US_MULTILATERAL, base=scope["base"], option_ids=["provider:multilateral"])


def _proposing() -> Callable[..., Any]:
    """`propose_plan` over a Search that compiles the US total onto the multilateral
    row (`_multilateral_compiles`)."""
    return _tools_reading(cast(ToolSurface, _multilateral_compiles()))["propose_plan"].func


async def test_a_direct_pin_sharing_an_aligned_compiles_spec_id_keeps_its_own_row(
    decision: DecisionFactory,
) -> None:
    """An earlier plan compared the US total with France's multilateral ODA, which moved
    the total onto DAC1's multilateral row, and Search answered that compile with the
    spec_id it gives US multilateral ODA. The amended plan reads US multilateral ODA,
    pinned to that row directly, as a share of the total, restated on the compile it
    was moved to. Read as the total's compile, the direct pin would be counted as the
    total and the share divide one series by itself. Each side keeps its own row:
    the total goes back to its own pin."""
    ctx = _FakeToolContext()
    france = _another_part(_US_MULTILATERAL)
    _show_blocks(ctx, _US_MULTILATERAL, _US_TOTAL, france)
    propose = _proposing()
    compared = await propose(
        question="How does US ODA compare with France's multilateral ODA?",
        answer_mode=AnswerMode.COMPUTED,
        decisions=drafts([decision()]),
        evidence_requirements=[
            _dac_countries("second", _US_TOTAL),
            _dac_countries("other", france),
            _difference("second", "other"),
        ],
        tool_context=ctx,
    )
    assert compared["outcome"] == Outcome.OK.value, compared
    assert dict(_pins(ctx))["second"] == _US_MULTILATERAL["spec_id"]

    result = await propose(
        question="What share of US ODA goes to multilaterals?",
        answer_mode=AnswerMode.COMPUTED,
        decisions=drafts([decision()]),
        evidence_requirements=[
            _dac_countries("first", _US_MULTILATERAL),
            _dac_countries("second", _US_MULTILATERAL),
            _share("first", "second"),
        ],
        removed_evidence_requirements=["other", _COMPARE],
        amends_version=1,
        tool_context=ctx,
    )

    assert result["outcome"] == Outcome.OK.value, result
    assert _pins(ctx) == [
        ("second", _US_TOTAL["spec_id"]),
        ("first", _US_MULTILATERAL["spec_id"]),
    ]


async def test_a_compile_leaves_the_shown_pins_as_the_planner_was_shown_them(
    decision: DecisionFactory,
) -> None:
    """The planner was shown the US total and France's multilateral ODA, not US
    multilateral ODA. Comparing them compiles the total onto DAC1's multilateral row,
    under US multilateral ODA's spec_id. The compile is no source the planner was
    shown, so the shown pins keep their keys: a later direct pin of that spec_id
    finds no shown source to read the compile's provenance from."""
    ctx = _FakeToolContext()
    france = _another_part(_US_MULTILATERAL)
    _show_blocks(ctx, _US_TOTAL, france)
    shown = set(pin_sources(ctx.state))

    result = await _proposing()(
        question="How does US ODA compare with France's multilateral ODA?",
        answer_mode=AnswerMode.COMPUTED,
        decisions=drafts([decision()]),
        evidence_requirements=[
            _dac_countries("second", _US_TOTAL),
            _dac_countries("other", france),
            _difference("second", "other"),
        ],
        tool_context=ctx,
    )

    assert result["outcome"] == Outcome.OK.value, result
    assert dict(_pins(ctx))["second"] == _US_MULTILATERAL["spec_id"]
    assert set(pin_sources(ctx.state)) == shown


# --- a carried sum re-pinned ----------------------------------------------------------


def test_propose_plan_reads_a_carried_sum_s_earlier_datasets_off_the_plan_it_amends(
    decision: DecisionFactory,
    pinning_decision: Callable[..., Decision],
    requirement: RequirementFactory,
) -> None:
    """The user agreed to add aid and debt. The amendment re-pins `debt`, under the
    same id, to a third dataset: the earlier plan's sum read IDS, so the sum now adds
    a pair the user was never asked about."""
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    ctx.session.events.extend(_combination_asked({"spec_id": COMBINE_OPTION}))
    summed = _sum_of_two_sources(
        ctx,
        datasets=(DAC2A, IDS),
        resolved_by=ResolvedBy.USER,
        decision=decision,
        pinning_decision=pinning_decision,
        requirement=requirement,
    )
    first = tools["propose_plan"].func(
        question=summed.question,
        answer_mode=summed.answer_mode,
        decisions=drafts(summed.decisions),
        evidence_requirements=list(summed.evidence_requirements),
        tool_context=ctx,
    )
    assert first["outcome"] == Outcome.OK.value, first
    other = "provenance/IMFWorldEconomicOutlook"
    other_spec = spec_id_for(canonicalize({**_GHANA_DEBT_IDENTITY, "dataset": "Ghana's flows"}))
    save_pin_sources(
        ctx.state,
        {
            **pin_sources(ctx.state),
            other_spec: PinSource(
                block={
                    "spec_id": other_spec,
                    "resolved": {
                        "rendered_sentence": other_spec,
                        "facets": {"provenance": {"ref": other, "account": None}},
                    },
                },
                settled=True,
            ),
        },
    )
    debt = summed.requirement("debt").model_copy(update={"source_spec_id": other_spec})

    result = tools["propose_plan"].func(
        answer_mode=summed.answer_mode,
        decisions=[],
        evidence_requirements=[debt],
        amends_version=1,
        tool_context=ctx,
    )

    assert result["code"] == ProblemCode.COMBINATION_UNASKED.value, result
    assert result["detail"] == {"requirement_id": "net_flows", "datasets": sorted([DAC2A, other])}

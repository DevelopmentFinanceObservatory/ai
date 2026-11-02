"""A recorded planner turn on the prices the user chose, replayed on dev Search's answers.

The user asked for UK ODA trends, in-donor refugee costs and the share going
to Africa, and answered the orchestrator's prices question on the UK ODA part
(`spec_20bdd…`) with constant prices. The planner pinned the constant-price
series a later search found, and its first draft was recorded. Translation
then handed `uk_total_oda` back: the pin it was aligned to was not found, and
discovery offered `spec_bf915…`, the same series over 1960 to 2025. Each of
the planner's six re-pins to that series is replayed here, in the order and
with the arguments the planner sent.
"""

from __future__ import annotations

import asyncio
import copy
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import pytest
from compiler.sources import outcome_from_envelope
from compiler.surface import ToolCallResult
from data_mcp.search_result import SEARCH_ENVELOPE_META_KEY
from harness.graph.state import ORCHESTRATOR_SPECS_KEY, store_search, stored_mapping
from harness.tool_contracts import resolver_digest
from planner import agent as planner_agent
from planner.agent import build_planner_tools
from planner.plan import (
    ASKS_STATE_KEY,
    SCOPE_UNREAD_BLOCK_KEY,
    DatasetRequirement,
    DecisionFixes,
    PinSource,
    ResolvedBy,
    alignments,
    found_sources,
    load_plan,
    pin_sources,
    plan_revision,
    save_found_sources,
    save_pin_sources,
)
from planner.workflow import ToolId
from translator.discovery import (
    DiscoveryLog,
    DiscoveryRecord,
    found_blocks,
    read_found_scope,
    requirement_revision,
    save_log,
)
from translator.referral import Referral, ReferralKind, save_referral

from ..session_events import response_event
from .test_agent import _clarify_answer_event, _clarify_ask_event, _FakeToolContext, _option

_FIXTURE = Path(__file__).parents[1] / "fixtures" / "prices_decision_uk_oda.json"
RECORDED: dict[str, Any] = json.loads(_FIXTURE.read_text())

PART = "spec_20bdd20a52e5f19f51a96c54bb40c2de"
FOUND = "spec_bf915cf71ebfc1b5772ca64b393e50ec"
_TOTAL = "spec_8877bd41b91e6a87835be50f04331cf6"
_AFRICA = "spec_8e4d7752bce78dd20af5e103a074db50"


class RecordedSearch:
    """A tool surface answering each `resolve_definition` call with the envelope
    dev Search answered it with in `resolves`, matched on its arguments, and
    answering every call as `answer` says when given: `refused`, as data-mcp
    refuses one Search rejects, or `candidates`, an answer that is not
    definite. A read of a spec_id in `definite`, posting no scope entries,
    is answered with the definite part it maps to. `calls` holds each call's
    arguments."""

    def __init__(
        self,
        resolves: list[Mapping[str, Any]],
        *,
        answer: Literal["refused", "candidates"] | None = None,
        definite: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        self.resolves = resolves
        self.answer = answer
        self.definite = definite or {}
        self.calls: list[Mapping[str, Any]] = []

    async def list_tools(self) -> list[Any]:
        return []

    async def call_tool(
        self, *, tool: ToolId, arguments: Mapping[str, Any], on_progress: Any = None
    ) -> ToolCallResult:
        del on_progress
        assert tool.name == "resolve_definition", tool
        self.calls.append(arguments)
        if self.answer == "refused":
            return ToolCallResult(
                is_error=True,
                structured_content={"code": "upstream_error", "upstream_code": "spec_not_found"},
            )
        if self.answer == "candidates":
            envelope = as_candidates()
        elif arguments["spec_id"] in self.definite and not arguments.get("scope_entries"):
            envelope = {"parts": [self.definite[arguments["spec_id"]]]}
        else:
            (envelope,) = [
                one["envelope"] for one in self.resolves if one["arguments"] == arguments
            ]
        return ToolCallResult(
            is_error=False,
            structured_content={"spec_id": arguments["spec_id"], "status": envelope.get("status")},
            meta={SEARCH_ENVELOPE_META_KEY: envelope},
        )


def _turn_so_far(ctx: _FakeToolContext) -> None:
    """The planner's entry as it opened: the asks the `plan` call recorded, both
    searches' parts shown, and the user's answer of constant prices to the
    question on `PART`."""
    ctx.state[ASKS_STATE_KEY] = RECORDED["asks"]
    save_pin_sources(
        ctx.state,
        {
            part["spec"]["spec_id"]: PinSource(
                block=part["spec"],
                settled=True,
                warnings=tuple(w["code"] for w in part.get("warnings", ())),
            )
            for search in ("current", "constant")
            for part in RECORDED["searches"][search]
        },
    )
    candidates = [
        {"spec_id": f"o{n}", "label": label, "part": PART, "prices": prices}
        for n, (label, prices) in enumerate(
            (("current prices", "current"), ("constant prices", "constant")), start=1
        )
    ]
    ctx.session.events = [_clarify_ask_event(candidates), _clarify_answer_event(_option("o2"))]


def _searched(spec_id: str) -> dict[str, Any]:
    """The part the orchestrator's constant-price search answered for `spec_id`."""
    (part,) = (
        part for part in RECORDED["searches"]["constant"] if part["spec"]["spec_id"] == spec_id
    )
    return part


def _served_as_candidate(spec: Mapping[str, Any]) -> dict[str, Any]:
    """`spec` as Search serves it in a candidates part: with no `resolved.scope_choices`."""
    served = copy.deepcopy(dict(spec))
    del served["resolved"]["scope_choices"]
    return served


def as_candidates() -> dict[str, Any]:
    """Translation's search answered as a `candidates` part offering `FOUND` and
    `8877`, each spec as Search serves a candidate: with no
    `resolved.scope_choices`, which dfo-apis states on a definite part alone
    (`search.engine.assemble.build_candidates_part` there)."""
    (found,) = (part["spec"] for part in RECORDED["translator"]["parts"])
    specs = [_served_as_candidate(spec) for spec in (found, _searched(_TOTAL)["spec"])]
    candidates = {"ordering": "relevance", "max_candidates": 6, "specs": specs, "truncated": False}
    return {"parts": [{"variable_text": "ODA", "status": "candidates", "candidates": candidates}]}


def _handed_back(ctx: _FakeToolContext, *, held: bool, envelope: Mapping[str, Any]) -> None:
    """Translation's verdict on the first draft: `uk_total_oda`'s pin not found,
    `FOUND` offered in its place, as discovery records what its search
    answered (`envelope`), and the blocks it found when `held`
    (`translator.discovery.found_blocks`)."""
    outcome = outcome_from_envelope(envelope)
    (candidate,) = (source for source in outcome.sources if source.spec_id == FOUND)
    referral = Referral(
        kind=ReferralKind.PINNED_SOURCE_NOT_FOUND,
        reason=(
            "'uk_total_oda' pins 'spec_09663f8f6ae191d0c9929803aea4938d', which this "
            "revision's discovery did not find. This search found 1 source(s) instead."
        ),
        requirement_id="uk_total_oda",
        candidates=(candidate,),
    )
    save_referral(ctx.state, referral)
    plan = load_plan(ctx.state)
    assert plan is not None
    requirement = plan.requirement("uk_total_oda")
    assert isinstance(requirement, DatasetRequirement)
    record = DiscoveryRecord(
        revision=plan_revision(plan),
        requirement_id="uk_total_oda",
        requirement_key=requirement_revision(requirement),
        tool="search",
        variable_sent="ODA",
        status=outcome.status,
        spec_ids=tuple(source.spec_id for source in outcome.sources),
    )
    save_log(ctx.state, DiscoveryLog().with_record(record, outcome.sources))
    if held:
        save_found_sources(ctx.state, found_blocks(outcome))


def after_hand_back(
    *,
    held: bool = True,
    envelope: Mapping[str, Any] | None = None,
    search: RecordedSearch | None = None,
) -> tuple[dict[str, Any], _FakeToolContext]:
    """The planner's tools and its context once the first draft was recorded
    and handed back, its two proposals answered as the recording has them:
    the first refused for an unread requirement, the second recorded with its
    total aligned to net disbursements by dev Search's compile. `held` says
    the offered series' block is on record, and `envelope` is what
    translation's search answered, the recorded one by default (`_handed_back`).
    `search` answers the re-pins' calls, the recorded answers by default."""
    tools = {
        tool.name: tool
        for tool in build_planner_tools(surface=RecordedSearch(RECORDED["resolves"]))
    }
    ctx = _FakeToolContext()
    _turn_so_far(ctx)
    for call in RECORDED["calls"][:2]:
        answer = _propose(tools, ctx, call["arguments"])
        assert (answer["outcome"], answer.get("code")) == (
            call["prod"]["outcome"],
            call["prod"]["code"],
        ), answer
        if answer["outcome"] == "ok":
            answer = {**answer, "translation": {"kind": "pinned_source_not_found"}}
        ctx.session.events = [*ctx.session.events, response_event("propose_plan", answer)]
    _handed_back(ctx, held=held, envelope=envelope or RECORDED["translator"])
    if search is not None:
        tools = {tool.name: tool for tool in build_planner_tools(surface=search)}
    return tools, ctx


def _propose(
    tools: dict[str, Any], ctx: _FakeToolContext, arguments: Mapping[str, Any]
) -> dict[str, Any]:
    """The answer of one `propose_plan` call stating `arguments`."""
    return asyncio.run(tools["propose_plan"].func(**arguments, tool_context=ctx))


_PRICES_SLOT = f"user_prices_{PART}"
_UNIT = {"currency": "USD", "prices": "constant", "base_year": 2024}
_PRICES = {
    "slot": _PRICES_SLOT,
    "chosen": "constant prices",
    "part": PART,
    "requirement_id": "uk_total_oda",
    "unit": _UNIT,
}


@pytest.mark.parametrize(
    ("at", "outcome", "code"),
    [
        ("11:22:56.327", "ok", None),
        ("11:23:03.161", "ok", None),
        ("11:23:07.318", "ok", None),
        ("11:23:11.466", "refused", "user_decision_dropped"),
        ("11:23:15.603", "invalid_input", "invalid_plan"),
        ("11:23:18.828", "ok", None),
    ],
)
def test_a_re_pin_to_the_series_discovery_found_keeps_the_users_prices_and_the_net_basis(
    at: str, outcome: str, code: str | None
) -> None:
    """Each re-pin of `uk_total_oda` to `FOUND`, made on the handed-back
    draft. With discovery's block of `FOUND` held, `FOUND` reads as the part's series
    in constant 2024 US dollars, and as a series counted on the OECD headline
    basis that admission puts on the net basis of the share's numerator. A
    re-pin stating that unit is recorded with the user's decision cited,
    whether it restates the decision or not, and with `uk_total_oda` pinned
    to `FOUND`'s compile on net disbursements under its measure-basis
    decision, as the first draft was. One stating no unit, or no base year,
    is refused naming the unit to state."""
    tools, ctx = after_hand_back()
    (call,) = [call for call in RECORDED["calls"] if call["at"] == at]

    answer = _propose(tools, ctx, call["arguments"])

    assert (answer["outcome"], answer.get("code")) == (outcome, code), answer
    if outcome != "ok":
        assert answer["detail"] == _PRICES
        return
    plan = load_plan(ctx.state)
    assert plan is not None
    assert plan.version == 2
    aligned = alignments(ctx.state)["uk_total_oda"]
    assert (aligned.origin, aligned.option_ids) == (FOUND, ("measure_basis:net",))
    requirement = plan.requirement("uk_total_oda")
    assert isinstance(requirement, DatasetRequirement)
    assert requirement.source_spec_id == aligned.spec_id
    chosen = {d.slot: d for d in plan.decisions}
    basis = chosen["scope_measure_basis_of_uk_total_oda"]
    assert basis.chosen == "net disbursements, constant 2024 USD"
    assert {basis.slot, _PRICES_SLOT} <= set(requirement.decision_ids)
    prices = chosen[_PRICES_SLOT]
    assert (prices.chosen, prices.resolved_by) == ("constant prices", ResolvedBy.USER)
    assert prices.fixes == DecisionFixes(prices="constant")


@pytest.mark.parametrize(
    ("at", "outcome", "code"),
    [
        ("11:22:56.327", "refused", "user_decision_dropped"),
        ("11:23:03.161", "ok", None),
        ("11:23:07.318", "ok", None),
        ("11:23:11.466", "refused", "user_decision_contradicted"),
        ("11:23:15.603", "invalid_input", "invalid_plan"),
        ("11:23:18.828", "ok", None),
    ],
)
def test_a_restated_prices_decision_leaves_the_users_decision_standing(
    at: str, outcome: str, code: str | None
) -> None:
    """With no block of `FOUND` on record, no requirement reads the part
    after the re-pin, so code writes no prices decision. A re-pin that cites
    the user's decision and states a unit in constant prices still lands,
    whatever decision it drafts under the user's slot, since code alone
    writes that slot: the decision the plan carries stands as the user made
    it. A re-pin that leaves it uncited, or states no unit, is refused."""
    tools, ctx = after_hand_back(held=False)
    first = load_plan(ctx.state)
    assert first is not None
    (call,) = [call for call in RECORDED["calls"] if call["at"] == at]

    answer = _propose(tools, ctx, call["arguments"])

    assert (answer["outcome"], answer.get("code")) == (outcome, code), answer
    if outcome != "ok":
        return
    plan = load_plan(ctx.state)
    assert plan is not None
    assert plan.version == 2
    (decided,) = [d for d in first.decisions if d.slot == _PRICES_SLOT]
    assert [d for d in plan.decisions if d.slot == _PRICES_SLOT] == [decided]
    assert _PRICES_SLOT in plan.requirement("uk_total_oda").decision_ids


def test_a_candidate_found_with_no_scope_choices_is_held_as_unread() -> None:
    """Search states no scope choices on a candidate, so discovery holds a
    candidate's block marked unread, and a definite answer's block unmarked."""
    candidates = found_blocks(outcome_from_envelope(as_candidates()))
    definite = found_blocks(outcome_from_envelope(RECORDED["translator"]))

    assert [block.get(SCOPE_UNREAD_BLOCK_KEY) for block in candidates.values()] == [True, True]
    assert SCOPE_UNREAD_BLOCK_KEY not in definite[FOUND]


def test_a_candidate_leaves_a_definite_block_of_its_series_standing() -> None:
    """A later search answering `FOUND` as a candidate keeps the scope choices
    an earlier definite answer stated for it, and holds the candidate it
    found nothing for marked unread."""
    state = _FakeToolContext().state
    save_found_sources(state, found_blocks(outcome_from_envelope(RECORDED["translator"])))

    save_found_sources(state, found_blocks(outcome_from_envelope(as_candidates())))

    held = found_sources(state)
    assert held[FOUND] == found_blocks(outcome_from_envelope(RECORDED["translator"]))[FOUND]
    assert held[_TOTAL][SCOPE_UNREAD_BLOCK_KEY] is True


def test_recording_a_re_pin_drops_the_blocks_it_can_no_longer_pin() -> None:
    """The recorded re-pin pins `FOUND`'s compile, so `FOUND`'s block stays on
    record and a block offered earlier that no requirement pins is dropped."""
    tools, ctx = after_hand_back()
    save_found_sources(
        ctx.state, {_TOTAL: found_blocks(outcome_from_envelope(as_candidates()))[_TOTAL]}
    )
    (call,) = [call for call in RECORDED["calls"] if call["at"] == "11:22:56.327"]

    answer = _propose(tools, ctx, call["arguments"])

    assert answer["outcome"] == "ok", answer
    assert list(found_sources(ctx.state)) == [FOUND]


def test_reading_a_candidates_scope_resolves_its_identity_once() -> None:
    """A candidate's scope choices are read by resolving its identity with no
    scope entries posted, which Search answers with the choices it states on
    a definite part: `FOUND` is counted on the OECD headline basis, with a
    net choice the share can compile it on."""
    search = RecordedSearch(RECORDED["resolves"])
    candidate = found_blocks(outcome_from_envelope(as_candidates()))[FOUND]

    read = asyncio.run(read_found_scope(search, FOUND, candidate))

    (call,) = search.calls
    assert call["scope_entries"] == []
    block = read[FOUND]
    assert SCOPE_UNREAD_BLOCK_KEY not in block
    choices = {c["axis"]: c for c in block["resolved"]["scope_choices"]["choices"]}
    assert choices["measure_basis"]["active"] == "measure_basis:oecd_headline"


def test_a_re_pin_to_a_candidate_lands_on_net_once_its_scope_is_read() -> None:
    """Search offered `FOUND` as a candidate, whose block states no scope
    choices. Admission has its scope read before it aligns the share, by one
    resolve of `FOUND`'s identity, then compiles it on net disbursements, as
    the share's numerator is counted: two Search calls, and the plan pins the
    compile under the measure-basis decision, with the user's prices cited."""
    search = RecordedSearch(RECORDED["resolves"])
    tools, ctx = after_hand_back(envelope=as_candidates(), search=search)
    (call,) = [call for call in RECORDED["calls"] if call["at"] == "11:22:56.327"]

    answer = _propose(tools, ctx, call["arguments"])

    assert answer["outcome"] == "ok", answer
    assert [
        (one["spec_id"], [entry["option_id"] for entry in one["scope_entries"]])
        for one in search.calls
    ] == [(FOUND, []), (FOUND, ["measure_basis:net"])]
    plan = load_plan(ctx.state)
    assert plan is not None
    aligned = alignments(ctx.state)["uk_total_oda"]
    assert (aligned.origin, aligned.option_ids) == (FOUND, ("measure_basis:net",))
    requirement = plan.requirement("uk_total_oda")
    assert requirement.source_spec_id == aligned.spec_id
    assert {"scope_measure_basis_of_uk_total_oda", _PRICES_SLOT} <= set(requirement.decision_ids)


@pytest.mark.parametrize("answer", ["refused", "candidates"])
def test_a_re_pin_to_a_candidate_whose_basis_cannot_be_read_is_refused(
    answer: Literal["refused", "candidates"],
) -> None:
    """Search offered `FOUND` as a candidate, whose block states no scope
    choices, and will not answer its identity as one definite dataset.
    Admitted as it stands, the share would divide net DAC2a disbursements by
    `FOUND`'s OECD headline total. The re-pin is refused, and no plan is
    recorded in its place."""
    search = RecordedSearch(RECORDED["resolves"], answer=answer)
    tools, ctx = after_hand_back(envelope=as_candidates(), search=search)
    (call,) = [call for call in RECORDED["calls"] if call["at"] == "11:22:56.327"]

    answer_ = _propose(tools, ctx, call["arguments"])

    assert (answer_["outcome"], answer_["code"]) == ("unsupported", "operation_unsupported")
    assert answer_["detail"]["basis"]["requirement_id"] == "uk_total_oda"
    assert [one["spec_id"] for one in search.calls] == [FOUND]
    plan = load_plan(ctx.state)
    assert plan is not None
    assert plan.version == 1


def test_a_read_that_leaves_a_candidate_unread_is_made_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A read whose answer still states no scope choices is not made again in
    the call: the proposal answers the refusal asking for it."""

    reads: list[str] = []

    async def unread(
        surface: Any, spec_id: str, block: Mapping[str, Any]
    ) -> dict[str, dict[str, Any]]:
        del surface
        reads.append(spec_id)
        assert len(reads) == 1, reads
        return {spec_id: dict(block)}

    monkeypatch.setattr(planner_agent, "read_found_scope", unread)
    tools, ctx = after_hand_back(envelope=as_candidates())
    (call,) = [call for call in RECORDED["calls"] if call["at"] == "11:22:56.327"]

    answer = _propose(tools, ctx, call["arguments"])

    assert answer["code"] == "scope_unread", answer
    assert [item["spec_id"] for item in answer["detail"]["read"]] == [FOUND]
    assert reads == [FOUND]


def test_a_share_whose_every_input_is_an_unread_candidate_has_each_read_first() -> None:
    """Both of the share's inputs are pinned to candidates Search offered with
    no scope choices: `FOUND`, the re-pin of `uk_total_oda`, and the net
    DAC2a series of `uk_oda_africa`, which a referral offered in place of the
    one the planner was shown. Neither carries a choice until it is read, and
    admitted unread the share would divide net disbursements by `FOUND`'s
    OECD headline total. Each is read before the share aligns, and
    `uk_total_oda` lands on `FOUND`'s compile on net disbursements."""
    search = RecordedSearch(RECORDED["resolves"], definite={_AFRICA: _searched(_AFRICA)})
    tools, ctx = after_hand_back(envelope=as_candidates(), search=search)
    save_pin_sources(ctx.state, {k: v for k, v in pin_sources(ctx.state).items() if k != _AFRICA})
    africa = {
        "variable_text": "ODA to Africa",
        "status": "candidates",
        "candidates": {
            "ordering": "relevance",
            "max_candidates": 6,
            "specs": [_served_as_candidate(_searched(_AFRICA)["spec"])],
            "truncated": False,
        },
    }
    save_found_sources(ctx.state, found_blocks(outcome_from_envelope({"parts": [africa]})))
    (call,) = [call for call in RECORDED["calls"] if call["at"] == "11:22:56.327"]

    answer = _propose(tools, ctx, call["arguments"])

    assert answer["outcome"] == "ok", answer
    assert sorted(one["spec_id"] for one in search.calls if not one["scope_entries"]) == sorted(
        [FOUND, _AFRICA]
    )
    aligned = alignments(ctx.state)["uk_total_oda"]
    assert (aligned.origin, aligned.option_ids) == (FOUND, ("measure_basis:net",))
    plan = load_plan(ctx.state)
    assert plan is not None
    assert plan.requirement("uk_total_oda").source_spec_id == aligned.spec_id


def _shown_as_candidate(ctx: _FakeToolContext) -> None:
    """Show the planner `FOUND` with the block the orchestrator stores for it
    when its search answers it as a candidate (`store_search`)."""
    store = _FakeToolContext().state
    store_search(store, resolver_digest(as_candidates()))
    block = stored_mapping(store, ORCHESTRATOR_SPECS_KEY)[FOUND]
    save_pin_sources(ctx.state, {**pin_sources(ctx.state), FOUND: PinSource(block=dict(block))})


@pytest.mark.parametrize("found", ["definite", "candidate", None])
def test_a_candidate_the_planner_was_shown_leaves_the_share_aligned(
    found: Literal["definite", "candidate"] | None,
) -> None:
    """The planner was shown `FOUND` as a candidate, whose block states no
    scope choices, and translation's discovery holds a block of it too:
    `definite`, the recorded answer stating `FOUND`'s choices, `candidate`,
    one marked unread, or none. The block that says how `FOUND` is counted
    is the one read, or `FOUND` is read first, so `uk_total_oda` lands on
    `FOUND`'s compile on net disbursements, as the share's numerator is
    counted."""
    search = RecordedSearch(RECORDED["resolves"])
    tools, ctx = after_hand_back(
        held=found is not None,
        envelope=RECORDED["translator"] if found == "definite" else as_candidates(),
        search=search,
    )
    _shown_as_candidate(ctx)
    (call,) = [call for call in RECORDED["calls"] if call["at"] == "11:22:56.327"]

    answer = _propose(tools, ctx, call["arguments"])

    assert answer["outcome"] == "ok", answer
    aligned = alignments(ctx.state)["uk_total_oda"]
    assert (aligned.origin, aligned.option_ids) == (FOUND, ("measure_basis:net",))
    plan = load_plan(ctx.state)
    assert plan is not None
    assert plan.requirement("uk_total_oda").source_spec_id == aligned.spec_id


def test_a_candidate_read_as_stating_no_choices_is_admitted_on_that_answer() -> None:
    """The planner was shown `FOUND` as a candidate, and Search's answer to
    the read of its scope states no choices. That answer is the block on
    record from then on, so `FOUND` is read once and the re-pin lands on
    `FOUND` as Search counts it, with nothing to align it on."""
    (part,) = RECORDED["translator"]["parts"]
    search = RecordedSearch(
        RECORDED["resolves"], definite={FOUND: {**part, "spec": _served_as_candidate(part["spec"])}}
    )
    tools, ctx = after_hand_back(held=False, envelope=as_candidates(), search=search)
    _shown_as_candidate(ctx)
    (call,) = [call for call in RECORDED["calls"] if call["at"] == "11:22:56.327"]

    answer = _propose(tools, ctx, call["arguments"])

    assert answer["outcome"] == "ok", answer
    assert [one["spec_id"] for one in search.calls] == [FOUND]
    assert "uk_total_oda" not in alignments(ctx.state)
    plan = load_plan(ctx.state)
    assert plan is not None
    assert plan.requirement("uk_total_oda").source_spec_id == FOUND

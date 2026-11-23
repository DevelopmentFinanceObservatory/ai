"""`propose_plan` against the provider scope the user's words state.

Each requirement carries the user's scope words in its `measure`; Search's reader
reads them there and in the turn's words (`planner.stated_scope`), and admission
counts a requirement on the option both name (`submission.service._stated_claims`),
refuses a pin that cannot count it (`_stated_unserved`), refuses once a plan stating
the turn's one option nowhere (`_stated_unattributed`), and refuses a sum that adds
such a requirement to imputed multilateral aid (`_stated_summed`).

The reader answers each text as dfo-apis' route recorded it
(`fixtures/stated_scope_readings.json`), through `LiveApisClient`. The G7 blocks are
the ones dev Search answered for "Show G7 aid to Africa as a share of G7 aid to all
developing countries each year from 1970 to 2024, including imputed multilateral aid"
(`fixtures/g7_inclusive_share_spec_blocks.json`), whose first plan divided by DAC1
total ODA; the US blocks are `basis_alignment`'s.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import httpx2
import pytest
from common.live_apis import LiveApisClient
from compiler.surface import ToolCallResult
from data_mcp.search_result import SEARCH_ENVELOPE_META_KEY
from harness import studio
from harness.conversation import current_user_message
from harness.planner_adapter import PlannerAdapter
from harness.settings import Settings
from harness.studio_state import ConversationState
from harness.studio_store import keep_turn_records
from harness.turn.store import AgentRecord
from harness.turn_events import PLANNER_ASK
from harness.wire import TurnRequest
from planner.plan import (
    AlignedPin,
    DatasetRequirement,
    PinSource,
    aligned_pins,
    load_plan,
    save_pin_sources,
    windowed_pins,
)
from planner.specs import planner_tool_specs
from planner.stated_scope import (
    STATED_SCOPE_KEPT_STATE_KEY,
    STATED_SCOPE_STATE_KEY,
    ApisStatedScopeReader,
    StatedScopeReader,
    stated_scope_kept,
    stated_uncompiled,
)
from planner.tool_results import Outcome, ProblemCode
from planner.workflow import ToolId
from submission.service import KEEP_PUBLISHED_OPTION, STATED_SCOPE_OPTIONS, save_aligned_pin

from ..basis_alignment import (
    HEALTH,
    TOTAL,
    basis_block,
    dac1_multilateral,
    dac1_with_contributions,
    provider_block,
)
from ..planner_host import node_ctx
from ..planner_tools import call_planner_tool
from ..session_events import model_response, one_answer, tool_answers, tool_call, user_answer

_FIXTURES = Path(__file__).parents[1] / "fixtures"
_RECORDED: dict[str, Any] = json.loads(
    (_FIXTURES / "g7_inclusive_share_spec_blocks.json").read_text()
)
_READINGS: dict[str, Any] = json.loads((_FIXTURES / "stated_scope_readings.json").read_text())[
    "readings"
]

_G7 = _RECORDED["proposal"]["question"]
_G7_AMENDED = f"{_G7}\n(then: Start in 1980.)"
_G7_BILATERAL = "What share of G7 ODA is bilateral?"
_US_TOTAL = "US bilateral aid to Kenya as a share of total US ODA"
_US_NO_SCOPE = "US aid to Kenya as a share of total US ODA"
_US_NO_TOTAL = "US bilateral aid to Kenya as a share of US ODA"
_FRANCE = "How much bilateral aid does France give to Mali, and what is France's total ODA?"
_MULTILATERALS = "What share of US ODA goes to multilaterals?"
_GNI = "Germany's bilateral ODA as a share of its GNI"
_TWO_SCOPES = (
    "Show G7 bilateral aid to Africa as a share of G7 total aid to Africa including imputed "
    "multilateral aid, 2010 to 2024"
)
_SEPARATELY = (
    "Show G7 bilateral aid and imputed multilateral aid to Africa separately, 2010 to 2024"
)
_DEBT = "How much do African countries owe to bilateral creditors?"
_LATER = "Extend it to 2025."

_INCLUSIVE = "provider:inclusive"
_BILATERAL = "provider:bilateral"
_STATED_INCLUSIVE = "net ODA including imputed multilateral aid"


def _block(name: str) -> dict[str, Any]:
    """A copy of the recorded G7 spec block `name`."""
    return copy.deepcopy(_RECORDED[name])


def _spec(name: str) -> str:
    return _RECORDED[name]["spec_id"]


DAC1 = _spec("dac1")


class _Context:
    """The context a planner tool call runs on (`..planner_tools`): the turn's
    state, its transcript and the served call's id."""

    def __init__(self, *records: AgentRecord) -> None:
        self.state: dict[str, Any] = {}
        self.records = list(records)
        self.run_ended = False
        self.function_call_id = "call-1"


class _Reader:
    """Search's reader as dfo-apis' route answers it for the recorded texts
    (`_READINGS`), through `LiveApisClient`, recording each call's texts in `calls`.
    With `failing`, the route answers 503 to every call."""

    def __init__(self, *, failing: bool = False) -> None:
        self.calls: list[list[str]] = []
        self.failing = failing

    def handler(self, request: httpx2.Request) -> httpx2.Response:
        assert (request.method, request.url.path) == ("POST", "/api/search/stated-scope")
        texts = json.loads(request.content)["texts"]
        self.calls.append(texts)
        if self.failing:
            return httpx2.Response(503, json={"code": "unavailable", "retryable": True})
        return httpx2.Response(200, json={"readings": [_READINGS[text] for text in texts]})

    def reader(self) -> StatedScopeReader:
        async def no_sleep(_: float) -> None:
            return None

        return ApisStatedScopeReader(
            LiveApisClient(
                base_url="http://apis.test",
                transport=httpx2.MockTransport(self.handler),
                max_attempts=1,
                sleep=no_sleep,
            )
        )


# Search's compile of a recorded DAC2a bilateral part on provider:inclusive, by the
# part's name.
_RECORDED_COMPILES = {"africa_bilateral": "africa_inclusive", "g7_bilateral": "g7_inclusive"}


def _switched(origin: Mapping[str, Any], option_id: str, entries: Any) -> dict[str, Any]:
    """`origin` compiled on `option_id`, as Search answers a resubmit posting
    `entries`: a spec_id of its own, its provider choice explicit on the option and
    its compiled identity binding it, so it reads another series."""
    block = copy.deepcopy(dict(origin))
    digest = hashlib.sha256(f"{origin['spec_id']}:{option_id}".encode()).hexdigest()
    spec_id = f"spec_{digest[:32]}"
    block["spec_id"] = spec_id
    resolved = block["resolved"]
    resolved["rendered_sentence"] = f"{resolved.get('rendered_sentence', '')} ({option_id})"
    scope = resolved["scope_choices"]
    scope["compiled"]["spec_id"] = spec_id
    identity = scope["compiled"]["identity"]
    identity["slots"] = [
        *identity.get("slots", []),
        {"property": "providerScope", "binding": {"state": "value", "ref": option_id}},
    ]
    scope["entries"] = entries
    for choice in scope["choices"]:
        if choice["axis"] == "provider":
            choice["active"], choice["provenance"] = option_id, "explicit"
    return block


class _Search:
    """A tool surface whose `resolve_definition` compiles a pin of `blocks` on the
    provider option posted: a recorded DAC2a part on provider:inclusive as dev Search
    did (`_RECORDED_COMPILES`), any other as `_switched` makes it. With `refusal`, it
    refuses every compile. Each call's arguments are recorded in `calls`."""

    def __init__(self, *blocks: Mapping[str, Any], refusal: Mapping[str, Any] | None = None):
        self.blocks = blocks
        self.refusal = refusal
        self.calls: list[Mapping[str, Any]] = []

    async def list_tools(self) -> list[Any]:
        return []

    async def call_tool(
        self, *, tool: ToolId, arguments: Mapping[str, Any], on_progress: Any = None
    ) -> ToolCallResult:
        del on_progress
        assert tool.name == "resolve_definition", tool
        self.calls.append(arguments)
        if self.refusal is not None:
            return ToolCallResult(is_error=True, structured_content=dict(self.refusal))
        (option_id,) = [
            e["option_id"] for e in arguments["scope_entries"] if e["axis"] == "provider"
        ]
        (origin,) = [
            block
            for block in self.blocks
            if (block["resolved"].get("scope_choices") or {}).get("base", {}).get("spec_id")
            == arguments["spec_id"]
        ]
        name = next((n for n in _RECORDED_COMPILES if _spec(n) == origin["spec_id"]), None)
        compiled = (
            _block(_RECORDED_COMPILES[name])
            if name is not None and option_id == _INCLUSIVE
            else _switched(origin, option_id, arguments["scope_entries"])
        )
        part = {"status": "definite", "spec": compiled}
        return ToolCallResult(is_error=False, meta={SEARCH_ENVELOPE_META_KEY: {"parts": [part]}})


def _show(ctx: _Context, *blocks: Mapping[str, Any]) -> None:
    """Record `blocks` as the settled answers the planner was shown this entry."""
    save_pin_sources(
        ctx.state,
        {block["spec_id"]: PinSource(block=dict(block), settled=True) for block in blocks},
    )


async def _propose(
    ctx: _Context,
    requirements: list[dict[str, Any]],
    decisions: Sequence[dict[str, Any]] = (),
    *,
    words: str | None,
    surface: _Search | None = None,
    reader: _Reader | None = None,
    answer_mode: str = "series",
    amends_version: int | None = None,
    question: str | None = None,
) -> dict[str, Any]:
    """`propose_plan` on `ctx` with the turn's words `words`, Search's compiles
    through `surface` and its reader of stated scope `reader`."""
    surface = surface if surface is not None else _Search()
    reader = reader if reader is not None else _Reader()
    tools = {
        spec.name: spec
        for spec in planner_tool_specs(surface=surface, stated_scope=reader.reader())
    }
    amends = {} if amends_version is None else {"amends_version": amends_version}
    token = current_user_message.set(words)
    try:
        return await call_planner_tool(
            tools["propose_plan"],
            question=question or words or _G7,
            answer_mode=answer_mode,
            decisions=list(decisions),
            evidence_requirements=requirements,
            tool_context=ctx,
            **amends,
        )
    finally:
        current_user_message.reset(token)


def _dataset(
    requirement_id: str,
    pin: str,
    measure: str,
    *,
    donor: str = "Group of Seven",
    recipient: str | None = None,
    period: tuple[int, int] = (1970, 2024),
    **fields: Any,
) -> dict[str, Any]:
    """A dataset requirement pinning `pin`, from `donor`, to `recipient` where given."""
    entities = [{"name": donor, "role": "donor"}]
    if recipient is not None:
        entities.append({"name": recipient, "role": "recipient"})
    return {
        "kind": "dataset",
        "id": requirement_id,
        "measure": measure,
        "entities": entities,
        "period": {"start": period[0], "end": period[1]},
        "source_spec_id": pin,
        **fields,
    }


def _share(
    numerator: str, denominator: str, share_id: str = "share", **fields: Any
) -> dict[str, Any]:
    return {
        "kind": "share_of",
        "id": share_id,
        "numerator": {"input_id": numerator},
        "denominator": {"input_id": denominator},
        **fields,
    }


def _pins(ctx: _Context) -> dict[str, str | None]:
    """The pin each dataset requirement of the plan on record reads, by id, a pin
    put on the requirement's own years read as the one it was copied from."""
    plan = load_plan(ctx.state)
    assert plan is not None
    origins = {copy.spec_id: copy.windowed_from for copy in windowed_pins(ctx.state).values()}
    return {
        requirement.id: origins.get(pin, pin)
        for requirement in plan.evidence_requirements
        if isinstance(requirement, DatasetRequirement)
        for pin in [requirement.source_spec_id]
    }


def _held(ctx: _Context, origin: str, option_id: str) -> str:
    """The spec_id of the compile of `origin` on `option_id` the state holds."""
    (pin,) = [
        held
        for held in aligned_pins(ctx.state)
        if held.origin == origin and held.option_ids == (option_id,)
    ]
    return pin.spec_id


# --- the reader ------------------------------------------------------------


async def test_a_route_that_does_not_answer_refuses_the_proposal_and_records_nothing() -> None:
    """The stated-scope read is made before admission; where the route does not
    answer, `propose_plan` answers `SURFACE_UNAVAILABLE` naming the read, admits
    nothing and guesses no scope."""
    ctx = _Context()
    _show(ctx, _block("africa_bilateral"))
    reader = _Reader(failing=True)

    result = await _propose(
        ctx,
        [_dataset("aid", _spec("africa_bilateral"), "net ODA", recipient="Africa")],
        words=_G7,
        reader=reader,
    )

    assert result["code"] == ProblemCode.SURFACE_UNAVAILABLE.value, result
    assert result["detail"] == {"read": "stated_scope"}
    assert load_plan(ctx.state) is None


async def test_each_text_is_read_once_a_turn() -> None:
    """The turn's words and each `measure` go to the route once: a second proposal
    stating the same texts reads none, and an amendment's new words are read alone."""
    ctx = _Context()
    _show(ctx, _block("separate_bilateral"), _block("separate_imputed"))
    reader = _Reader()
    requirements = [
        _dataset("bilateral", _spec("separate_bilateral"), "bilateral aid", recipient="Africa"),
        _dataset("imputed", _spec("separate_imputed"), "imputed multilateral aid"),
    ]

    await _propose(ctx, requirements, words=_SEPARATELY, reader=reader)
    await _propose(ctx, requirements, words=_SEPARATELY, reader=reader)

    assert reader.calls == [[_SEPARATELY, "bilateral aid", "imputed multilateral aid"]]


async def test_a_measure_with_no_reading_held_refuses_the_proposal() -> None:
    """Admission reads only the readings held: a proposal whose tools read nothing
    (`planner_tool_specs` with no reader), stating a `measure` the turn has not read,
    answers `SURFACE_UNAVAILABLE` naming the read and records nothing."""
    ctx = _Context()
    bilateral = _block("separate_bilateral")
    _show(ctx, bilateral)
    pin = bilateral["spec_id"]
    first = await _propose(
        ctx, [_dataset("aid", pin, "bilateral aid", recipient="Africa")], words=_G7_BILATERAL
    )
    assert first["outcome"] == Outcome.OK.value, first
    tools = {spec.name: spec for spec in planner_tool_specs(surface=_Search(bilateral))}
    token = current_user_message.set(_G7_BILATERAL)
    try:
        result = await call_planner_tool(
            tools["propose_plan"],
            question=_G7_BILATERAL,
            answer_mode="series",
            decisions=[],
            evidence_requirements=[_dataset("aid", pin, "bilateral ODA", recipient="Africa")],
            amends_version=1,
            tool_context=ctx,
        )
    finally:
        current_user_message.reset(token)

    assert result["code"] == ProblemCode.SURFACE_UNAVAILABLE.value, result
    assert result["detail"] == {"read": "stated_scope"}
    plan = load_plan(ctx.state)
    assert plan is not None
    assert plan.version == 1


# --- false moves -----------------------------------------------------------


@pytest.mark.parametrize(
    ("words", "moved"),
    [
        pytest.param(_G7_BILATERAL, True, id="words-name-bilateral"),
        pytest.param(_G7, False, id="words-name-another-scope"),
    ],
)
async def test_a_numerator_stating_bilateral_moves_and_its_total_stays(
    words: str, moved: bool
) -> None:
    """ "What share of G7 ODA is bilateral?": the numerator's `measure` says bilateral,
    as the user's words do, so it is counted on DAC1's bilateral row; the total states
    nothing and stays on DAC1 total ODA, since moving it too would divide the row by
    itself. A `measure` saying bilateral under words that name another scope states
    nothing: nothing moves, and both sides read DAC1 total ODA as one series."""
    ctx = _Context()
    dac1 = _block("dac1")
    _show(ctx, dac1)
    surface = _Search(dac1)
    requirements = [
        _dataset("bilateral", DAC1, "bilateral ODA"),
        _dataset("total", DAC1, "official development assistance"),
        _share("bilateral", "total"),
    ]

    result = await _propose(ctx, requirements, words=words, surface=surface)

    if not moved:
        # Both sides read DAC1 total ODA as it stands.
        assert result["code"] == ProblemCode.SHARE_OF_ITSELF.value, result
        assert surface.calls == []
        return
    assert result["outcome"] == Outcome.OK.value, result
    assert _pins(ctx) == {"bilateral": _held(ctx, DAC1, _BILATERAL), "total": DAC1}
    assert len(surface.calls) == 1


def _us_bilateral() -> dict[str, Any]:
    """US aid to Kenya counted as bilateral flows, as the words "bilateral aid" ask."""
    return provider_block(HEALTH, narrow=True, active=_BILATERAL, provenance="explicit")


@pytest.mark.parametrize(
    ("words", "measure", "moved"),
    [
        pytest.param(_US_TOTAL, "total ODA", False, id="a-total"),
        pytest.param(_US_TOTAL, "ODA", True, id="no-total"),
        pytest.param(_US_NO_SCOPE, "total ODA", True, id="words-naming-no-scope"),
        pytest.param(_US_NO_TOTAL, "total ODA", True, id="words-asking-no-total"),
    ],
)
async def test_a_total_the_user_asks_for_stays_on_its_own_scope(
    words: str, measure: str, moved: bool
) -> None:
    """ "US bilateral aid to Kenya as a share of total US ODA": the denominator's
    `measure` asks for a total, as the user's words do, so it is held on what DAC1
    total ODA counts and the share states the difference. A denominator whose
    `measure` asks for no total is counted as bilateral flows, as the numerator is,
    and so is one asking for a total where the user's words name no provider scope,
    as alignment counts it with no stated scope, or ask for no total."""
    ctx = _Context()
    total = dac1_with_contributions()
    _show(ctx, _us_bilateral(), total)
    surface = _Search(total)
    requirements = [
        _dataset("bilateral", HEALTH, "bilateral aid", donor="United States", recipient="Kenya"),
        _dataset("total", total["spec_id"], measure, donor="United States"),
        _share("bilateral", "total"),
    ]

    result = await _propose(ctx, requirements, words=words, surface=surface)

    assert result["outcome"] == Outcome.OK.value, result
    plan = load_plan(ctx.state)
    assert plan is not None
    differ = [d for d in plan.decisions if d.caveat == "providers_differ"]
    if moved:
        assert _pins(ctx)["total"] == _held(ctx, total["spec_id"], _BILATERAL)
        assert differ == []
    else:
        assert _pins(ctx)["total"] == total["spec_id"]
        assert surface.calls == []
        assert len(differ) == 1


async def test_a_second_figure_of_its_own_is_left_as_it_stands() -> None:
    """ "How much bilateral aid does France give to Mali, and what is France's total
    ODA?": two figures and no calculation; the total is read as DAC1 counts it."""
    ctx = _Context()
    total = dac1_with_contributions()
    _show(ctx, _us_bilateral(), total)
    surface = _Search(total)
    requirements = [
        _dataset("bilateral", HEALTH, "bilateral aid", donor="France", recipient="Mali"),
        _dataset("total", total["spec_id"], "total ODA", donor="France"),
    ]

    result = await _propose(ctx, requirements, words=_FRANCE, surface=surface)

    assert result["outcome"] == Outcome.OK.value, result
    assert _pins(ctx) == {"bilateral": HEALTH, "total": total["spec_id"]}
    assert surface.calls == []


def _dac1_row(spec_id: str, active: str) -> dict[str, Any]:
    """The US DAC1 total (`dac1_multilateral`) under `spec_id`, counted as `active`."""
    block = dac1_multilateral()
    block["spec_id"] = spec_id
    scope = block["resolved"]["scope_choices"]
    scope["compiled"]["spec_id"] = spec_id
    (choice,) = [c for c in scope["choices"] if c["axis"] == "provider"]
    choice["active"] = active
    (option,) = [o for o in choice["options"] if o["option_id"] == active]
    if option["binding"].get("kind") == "slot":
        (aid_type,) = option["binding"]["refs"]
        for slot in scope["compiled"]["identity"]["slots"]:
            if slot["property"] == "dac1AidType":
                slot["binding"] = {"state": "value", "ref": aid_type}
    return block


async def test_core_contributions_over_their_total_stay_two_rows() -> None:
    """ "What share of US ODA goes to multilaterals?": the numerator's `measure` names
    DAC1's multilateral row, as the words do; the total stays on total ODA."""
    ctx = _Context()
    multilateral = _dac1_row(f"spec_{'8' * 32}", "provider:multilateral")
    total = _dac1_row(f"spec_{'9' * 32}", "provider:contributions")
    _show(ctx, multilateral, total)
    surface = _Search(multilateral, total)
    requirements = [
        _dataset("multilateral", multilateral["spec_id"], "multilateral ODA", donor="US"),
        _dataset("total", total["spec_id"], "ODA", donor="US"),
        _share("multilateral", "total"),
    ]

    result = await _propose(ctx, requirements, words=_MULTILATERALS, surface=surface)

    assert result["outcome"] == Outcome.OK.value, result
    assert _pins(ctx) == {"multilateral": multilateral["spec_id"], "total": total["spec_id"]}
    assert surface.calls == []


async def test_a_series_with_no_provider_choice_is_left_alone() -> None:
    """ "Germany's bilateral ODA as a share of its GNI": the ODA is counted as
    bilateral flows, and GNI, which has no provider choice, is read as it stands."""
    ctx = _Context()
    oda = dac1_with_contributions()
    gni = basis_block(TOTAL, narrow=False, active="measure_basis:gross")
    _show(ctx, oda, gni)
    surface = _Search(oda)
    requirements = [
        _dataset("oda", oda["spec_id"], "bilateral ODA", donor="Germany"),
        _dataset("gni", TOTAL, "GNI", donor="Germany"),
        _share("oda", "gni"),
    ]

    result = await _propose(ctx, requirements, words=_GNI, surface=surface)

    assert result["outcome"] == Outcome.OK.value, result
    assert _pins(ctx) == {"oda": _held(ctx, oda["spec_id"], _BILATERAL), "gni": TOTAL}


async def test_a_later_turn_whose_words_state_nothing_moves_nothing() -> None:
    """A requirement whose `measure` still says "including imputed multilateral aid",
    in a turn whose words name no provider scope: nothing is stated, nothing moves."""
    ctx = _Context()
    africa = _block("africa_bilateral")
    _show(ctx, africa)
    surface = _Search(africa)
    requirement = _dataset("aid", africa["spec_id"], _STATED_INCLUSIVE, recipient="Africa")

    result = await _propose(ctx, [requirement], words=_LATER, surface=surface, question=_G7)

    assert result["outcome"] == Outcome.OK.value, result
    assert _pins(ctx) == {"aid": africa["spec_id"]}
    assert surface.calls == []


# --- false refusals --------------------------------------------------------


async def test_two_scopes_in_the_words_are_served_each_on_its_side() -> None:
    """ "G7 bilateral aid to Africa as a share of G7 total aid to Africa including
    imputed multilateral aid": the words state two scopes, so neither is the
    question's; each side's `measure` states its own, both named in the words, and
    the share states the difference with no ask and no refusal."""
    ctx = _Context()
    bilateral, inclusive = _block("bilateral_share_bilateral"), _block("bilateral_share_inclusive")
    _show(ctx, bilateral, inclusive)
    surface = _Search(bilateral, inclusive)
    requirements = [
        _dataset("bilateral", bilateral["spec_id"], "bilateral aid", recipient="Africa"),
        _dataset(
            "total",
            inclusive["spec_id"],
            "total aid including imputed multilateral aid",
            recipient="Africa",
        ),
        _share("bilateral", "total"),
    ]

    result = await _propose(
        ctx, requirements, words=_TWO_SCOPES, surface=surface, answer_mode="computed"
    )

    assert result["outcome"] == Outcome.OK.value, result
    assert surface.calls == []
    plan = load_plan(ctx.state)
    assert plan is not None
    assert [d.caveat for d in plan.decisions if d.caveat == "providers_differ"] == [
        "providers_differ"
    ]


async def test_bilateral_and_imputed_aid_shown_separately_are_served() -> None:
    """ "G7 bilateral aid and imputed multilateral aid to Africa separately": a bare
    "multilateral" names no option, so the words state nothing and the omission
    check is silent."""
    ctx = _Context()
    _show(ctx, _block("separate_bilateral"), _block("separate_imputed"))
    requirements = [
        _dataset("bilateral", _spec("separate_bilateral"), "bilateral aid", recipient="Africa"),
        _dataset("imputed", _spec("separate_imputed"), "imputed multilateral aid"),
    ]

    result = await _propose(ctx, requirements, words=_SEPARATELY)

    assert result["outcome"] == Outcome.OK.value, result


async def test_bilateral_creditors_with_no_provider_choice_are_served() -> None:
    """A debt question about bilateral creditors states bilateral, and no pin has a
    provider choice that could count it, so nothing is refused."""
    ctx = _Context()
    debt = basis_block(TOTAL, narrow=False, active="measure_basis:gross")
    _show(ctx, debt)

    result = await _propose(
        ctx, [_dataset("debt", TOTAL, "external debt", donor="Africa")], words=_DEBT
    )

    assert result["outcome"] == Outcome.OK.value, result


async def test_a_switch_on_the_plan_card_stands_through_an_amendment() -> None:
    """The user switched G7 aid to Africa to bilateral flows on the plan card, though
    its `measure` still says "including imputed multilateral aid" as their words do.
    The switch is their later word: nothing moves, and an amendment of the period
    keeps it."""
    ctx = _Context()
    switched = _block("africa_switched_bilateral")
    _show(ctx, switched)
    surface = _Search(switched)
    requirement = _dataset("aid", switched["spec_id"], _STATED_INCLUSIVE, recipient="Africa")

    first = await _propose(ctx, [requirement], words=_G7, surface=surface)
    amended = await _propose(
        ctx,
        [{**requirement, "period": {"start": 1980, "end": 2024}}],
        words=_G7_AMENDED,
        surface=surface,
        amends_version=1,
        question=_G7,
    )

    assert first["outcome"] == Outcome.OK.value, first
    assert amended["outcome"] == Outcome.OK.value, amended
    assert _pins(ctx) == {"aid": switched["spec_id"]}
    assert surface.calls == []


# --- positives ---------------------------------------------------------------


def _g7_share(*, numerator: str, denominator: str, **share: Any) -> list[dict[str, Any]]:
    """G7 aid to Africa over G7 aid to developing countries, each a DAC2a bilateral
    part as dev Search answered it, with the `measure` given for each side."""
    return [
        _dataset("africa", _spec("africa_bilateral"), numerator, recipient="Africa"),
        _dataset(
            "developing", _spec("g7_bilateral"), denominator, recipient="developing countries"
        ),
        _share("africa", "developing", **share),
    ]


def _g7_blocks() -> list[dict[str, Any]]:
    return [_block(name) for name in ("africa_bilateral", "africa_imputed", "g7_bilateral")]


@pytest.mark.parametrize(
    ("numerator", "denominator", "accept"),
    [
        pytest.param(_STATED_INCLUSIVE, _STATED_INCLUSIVE, False, id="both-sides-state-it"),
        pytest.param(_STATED_INCLUSIVE, "net ODA", False, id="the-numerator-states-it"),
        pytest.param(_STATED_INCLUSIVE, _STATED_INCLUSIVE, True, id="accepting-a-difference"),
    ],
)
async def test_sides_stating_the_scope_are_counted_on_it_with_no_refusal(
    numerator: str, denominator: str, accept: bool
) -> None:
    """G7 aid to Africa over G7 aid to developing countries, the user's words asking
    for imputed multilateral aid. Each side whose `measure` says so is moved to its
    DAC2a part's inclusive compile with no refusal; a side that does not is aligned
    with the side that does. `accept_provider_difference` keeps no stated side off
    the scope."""
    ctx = _Context()
    blocks = _g7_blocks()
    _show(ctx, *blocks)
    surface = _Search(*blocks)
    share = {"accept_provider_difference": True} if accept else {}

    result = await _propose(
        ctx,
        _g7_share(numerator=numerator, denominator=denominator, **share),
        words=_G7,
        surface=surface,
        answer_mode="computed",
    )

    assert result["outcome"] == Outcome.OK.value, result
    assert _pins(ctx) == {
        "africa": _spec("africa_inclusive"),
        "developing": _spec("g7_inclusive"),
    }
    assert sorted(call["spec_id"] for call in surface.calls) == sorted(
        [_spec("africa_bilateral"), _spec("g7_bilateral")]
    )
    plan = load_plan(ctx.state)
    assert plan is not None
    (rationale,) = [d.rationale for d in plan.decisions if d.slot == "scope_provider_of_africa"]
    assert rationale.startswith("The user's question asked for")


async def test_a_plan_stating_the_scope_nowhere_is_refused_once() -> None:
    """The plan for the G7 question that states imputed multilateral aid on no side
    is refused once, naming the part that can count it; proposed again unchanged, it
    is the planner's word that the scope qualifies none of its figures, and is
    admitted as written."""
    ctx = _Context()
    blocks = _g7_blocks()
    _show(ctx, *blocks)
    surface = _Search(*blocks)
    requirements = _g7_share(numerator="net ODA", denominator="net ODA")

    first = await _propose(ctx, requirements, words=_G7, surface=surface, answer_mode="computed")
    again = await _propose(ctx, requirements, words=_G7, surface=surface, answer_mode="computed")

    assert first["code"] == ProblemCode.STATED_SCOPE_UNATTRIBUTED.value, first
    assert first["detail"] == {
        "requirement_id": "africa",
        "option_id": _INCLUSIVE,
        "offering": ["africa", "developing"],
    }
    assert again["outcome"] == Outcome.OK.value, again
    assert _pins(ctx) == {"africa": _spec("africa_bilateral"), "developing": _spec("g7_bilateral")}
    assert surface.calls == []


def _recorded() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The first plan admitted for the G7 question: Africa's bilateral part plus the
    imputed series, over DAC1 total ODA, accepting a provider difference."""
    proposal = copy.deepcopy(_RECORDED["proposal"])
    return proposal["evidence_requirements"], proposal["decisions"]


async def test_the_recorded_plan_is_refused_for_stating_the_scope_nowhere() -> None:
    """The plan first admitted for the G7 question names imputed multilateral aid in
    no `measure`, so it is refused once, though it accepts a provider difference."""
    ctx = _Context()
    _show(ctx, *_g7_blocks(), _block("g7_imputed"), _block("dac1"))

    result = await _propose(ctx, *_recorded(), words=_G7, answer_mode="computed")

    assert result["code"] == ProblemCode.STATED_SCOPE_UNATTRIBUTED.value, result
    assert result["detail"]["offering"] == ["africa_bilateral"]


def _africa_sum(bilateral_measure: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """G7 aid to Africa as its bilateral part, under `bilateral_measure`, plus the
    imputed multilateral series, as the first plan for the G7 question summed them."""
    requirements, decisions = _recorded()
    kept = [
        r for r in requirements if r["id"] in {"africa_bilateral", "africa_imputed", "africa_total"}
    ]
    (bilateral,) = [r for r in kept if r["id"] == "africa_bilateral"]
    bilateral["measure"] = bilateral_measure
    return kept, [d for d in decisions if d["slot"] == "africa_total_method"]


@pytest.mark.parametrize(
    ("bilateral_measure", "denominator", "code", "step"),
    [
        pytest.param(
            _STATED_INCLUSIVE,
            None,
            ProblemCode.PROVIDERS_UNALIGNED,
            "africa_total",
            id="the-summed-part-states-it",
        ),
        pytest.param(
            "net ODA",
            _STATED_INCLUSIVE,
            ProblemCode.PROVIDERS_UNALIGNED,
            "share",
            id="the-other-side-states-it",
        ),
        pytest.param(
            "bilateral aid", None, ProblemCode.STATED_SCOPE_UNATTRIBUTED, None, id="neither"
        ),
    ],
)
async def test_bilateral_plus_imputed_aid_is_refused_toward_the_inclusive_dataset(
    bilateral_measure: str, denominator: str | None, code: ProblemCode, step: str | None
) -> None:
    """G7 aid to Africa as bilateral aid plus the imputed multilateral series, where
    the user's words ask for imputed multilateral aid. Counting the bilateral part as
    its `measure` or the other side states would add the imputed series to a figure
    already holding it, so the plan is refused toward one requirement pinning the
    part counted that way; a sum stating the scope nowhere is refused for that. No
    compile is asked for."""
    ctx = _Context()
    blocks = [*_g7_blocks()]
    _show(ctx, *blocks)
    surface = _Search(*blocks)
    requirements, decisions = _africa_sum(bilateral_measure)
    if denominator is not None:
        requirements = [
            *requirements,
            _dataset(
                "developing", _spec("g7_bilateral"), denominator, recipient="developing countries"
            ),
            _share("africa_total", "developing"),
        ]

    result = await _propose(
        ctx, requirements, decisions, words=_G7, surface=surface, answer_mode="computed"
    )

    assert result["code"] == code.value, result
    if step is not None:
        assert result["detail"]["provider"]["step_id"] == step
        assert result["detail"]["provider"]["sum_id"] == "africa_total"
        assert result["detail"]["provider"]["unchosen_id"] == "africa_imputed"
    assert surface.calls == []
    assert load_plan(ctx.state) is None


async def test_two_sides_stating_one_scope_on_one_series_are_refused() -> None:
    """A share both of whose `measure`s say bilateral, over DAC1 total ODA read the
    same way on both sides: counted as stated, both read DAC1's bilateral row, and
    the share would answer 100%."""
    ctx = _Context()
    dac1 = _block("dac1")
    _show(ctx, dac1)
    surface = _Search(dac1)
    requirements = [
        _dataset("bilateral", DAC1, "bilateral ODA"),
        _dataset("total", DAC1, "bilateral ODA"),
        _share("bilateral", "total"),
    ]

    result = await _propose(ctx, requirements, words=_G7_BILATERAL, surface=surface)

    assert result["code"] == ProblemCode.SHARE_OF_ITSELF.value, result


_DAC1_STATED = "official development assistance including imputed multilateral aid"


def _dac1_total(measure: str, *, period: tuple[int, int] = (1970, 2024)) -> dict[str, Any]:
    return _dataset("g7_total", DAC1, measure, period=period)


def _kept(answer: str) -> list[AgentRecord]:
    """The `stated_provider_dropped` refusal of DAC1 for the G7 question, the
    planner's ask carrying its options, and the user's `answer`."""
    refusal = {
        "outcome": Outcome.REFUSED.value,
        "code": ProblemCode.STATED_PROVIDER_DROPPED.value,
        "message": "refused",
        "detail": {
            "requirement_id": "g7_total",
            "unserved": [{"requirement_id": "g7_total", "spec_id": DAC1, "option_id": _INCLUSIVE}],
            "serving": [],
        },
    }
    ask = {"message": "Keep DAC1?", "options": [dict(o) for o in STATED_SCOPE_OPTIONS]}
    return [
        tool_answers(("propose-1", "propose_plan", refusal)),
        model_response(tool_call("ask-1", PLANNER_ASK, ask)),
        user_answer("ask-1", one_answer({"spec_id": answer})),
    ]


@pytest.mark.parametrize(
    ("shown", "numerator", "serving"),
    [
        pytest.param(("g7_bilateral",), False, ["g7_bilateral"], id="a-dataset-serves-it"),
        pytest.param((), False, [], id="none-serves-it"),
        pytest.param(
            ("africa_bilateral", "africa_inclusive", "g7_bilateral"),
            True,
            ["g7_bilateral"],
            id="the-plans-own-pins-and-their-compiles-serve-none",
        ),
    ],
)
async def test_a_pin_that_cannot_count_the_stated_scope_is_refused_with_the_ask(
    shown: tuple[str, ...], numerator: bool, serving: list[str]
) -> None:
    """G7 total ODA on DAC1, whose `measure` says "including imputed multilateral
    aid" as the user's words do: DAC1 cannot count it. The refusal lists each
    dataset on record that can, and carries the ask that keeps DAC1 as published.
    The Africa part the plan pins for another figure, and its inclusive compile,
    measure that figure and are not listed."""
    ctx = _Context()
    _show(ctx, _block("dac1"), *(_block(name) for name in shown))
    requirements = [_dac1_total(_DAC1_STATED)]
    if numerator:
        africa, compiled = _block("africa_bilateral"), _block("africa_inclusive")
        save_aligned_pin(
            ctx.state,
            AlignedPin(
                origin=africa["spec_id"],
                option_ids=(_INCLUSIVE,),
                spec_id=compiled["spec_id"],
                digest={"spec_block": compiled},
            ),
            warnings=(),
        )
        requirements.append(
            _dataset("africa", africa["spec_id"], _STATED_INCLUSIVE, recipient="Africa")
        )

    result = await _propose(ctx, requirements, words=_G7)

    assert result["code"] == ProblemCode.STATED_PROVIDER_DROPPED.value, result
    assert result["detail"]["unserved"] == [
        {"requirement_id": "g7_total", "spec_id": DAC1, "option_id": _INCLUSIVE}
    ]
    assert result["detail"]["serving"] == [_spec(name) for name in serving]
    assert result["action"] == {
        "call": PLANNER_ASK,
        "arguments": {"options": [dict(o) for o in STATED_SCOPE_OPTIONS]},
        "needs": ["message"],
    }
    assert load_plan(ctx.state) is None


@pytest.mark.parametrize(
    ("answer", "admitted"),
    [
        pytest.param(KEEP_PUBLISHED_OPTION, True, id="kept"),
        pytest.param("count it as asked", False, id="not-kept"),
    ],
)
async def test_the_users_keep_holds_through_an_amendment(answer: str, admitted: bool) -> None:
    """The user answered the ask: keeping DAC1 as published admits it, and an
    amendment changing the period, made by a planner call whose transcript holds no
    ask, keeps it, the keep held in the planner's state
    (`planner.stated_scope.STATED_SCOPE_KEPT_STATE_KEY`). Any other answer keeps
    nothing."""
    ctx = _Context(*_kept(answer))
    _show(ctx, _block("dac1"))
    measure = "official development assistance including imputed multilateral aid"

    first = await _propose(ctx, [_dac1_total(measure)], words=_G7)

    if not admitted:
        assert first["code"] == ProblemCode.STATED_PROVIDER_DROPPED.value, first
        return
    assert first["outcome"] == Outcome.OK.value, first
    assert stated_scope_kept(ctx.state) == {(DAC1, _INCLUSIVE)}
    ctx.records = []
    amended = _dac1_total(measure, period=(1980, 2024))
    second = await _propose(
        ctx,
        [amended],
        words=_G7_AMENDED,
        amends_version=1,
        question=_G7,
    )
    assert second["outcome"] == Outcome.OK.value, second
    assert _pins(ctx) == {"g7_total": DAC1}


def _dac1_without_bilateral() -> dict[str, Any]:
    """The recorded DAC1 total with its bilateral option unavailable, as one that
    cannot count bilateral flows either."""
    block = _block("dac1")
    (choice,) = [
        c for c in block["resolved"]["scope_choices"]["choices"] if c["axis"] == "provider"
    ]
    for option in choice["options"]:
        if option["option_id"] == _BILATERAL:
            option["availability"] = {"status": "unavailable", "code": "not_published"}
    return block


def _next_turn(ctx: _Context) -> _Context:
    """The planner context of the conversation's next turn, after the turn `ctx`
    ran: the conversation keeps the turn's records (`keep_turn_records`) and the next
    turn opens on what Studio seeds it with (`harness.studio._turn_opening`)."""
    conversation = ConversationState(conversation_id="conv_01keep")
    keep_turn_records(conversation, ctx.state)
    body = TurnRequest.model_validate(
        {
            "kind": "turn.request",
            "schema_version": 1,
            "conversation_id": "conv_01keep",
            "message": _LATER,
        }
    )
    _, state, _ = studio._turn_opening(conversation, body, None)
    after = _Context()
    after.state = state
    return after


@pytest.mark.parametrize(
    ("words", "measure", "admitted"),
    [
        pytest.param(_G7_AMENDED, _DAC1_STATED, True, id="the-option-kept"),
        pytest.param(_G7_BILATERAL, "bilateral ODA", False, id="another-option"),
    ],
)
async def test_a_keep_lasts_the_conversation(words: str, measure: str, admitted: bool) -> None:
    """The user kept DAC1 as published in one turn. A later turn, whose planner
    reads no ask, admits DAC1 again for the option kept, as the conversation holds
    the keep (`ConversationState.stated_scope_kept`); asked for another option DAC1
    cannot count, it is refused with the ask again."""
    first = _Context(*_kept(KEEP_PUBLISHED_OPTION))
    _show(first, _block("dac1"))
    kept = await _propose(first, [_dac1_total(_DAC1_STATED)], words=_G7)
    assert kept["outcome"] == Outcome.OK.value, kept

    ctx = _next_turn(first)
    _show(ctx, _dac1_without_bilateral())
    result = await _propose(ctx, [_dac1_total(measure)], words=words, question=words)

    if admitted:
        assert result["outcome"] == Outcome.OK.value, result
        assert _pins(ctx) == {"g7_total": DAC1}
    else:
        assert result["code"] == ProblemCode.STATED_PROVIDER_DROPPED.value, result
        assert result["detail"]["unserved"] == [
            {"requirement_id": "g7_total", "spec_id": DAC1, "option_id": _BILATERAL}
        ]


@pytest.mark.parametrize(
    ("refusal", "code"),
    [
        pytest.param(
            {"code": "requested_option_unavailable", "retry_mode": "do_not_retry"},
            ProblemCode.STATED_PROVIDER_DROPPED,
            id="refused",
        ),
        pytest.param(
            {"code": "upstream_unavailable", "retry_mode": "retry_after_delay"},
            ProblemCode.SURFACE_UNAVAILABLE,
            id="unanswered",
        ),
    ],
)
async def test_a_stated_compile_search_refuses_cannot_count_the_scope(
    refusal: dict[str, str], code: ProblemCode
) -> None:
    """G7 aid to Africa stating imputed multilateral aid, whose compile counted that
    way Search refuses: the pin cannot count it, so the plan is refused with the
    ask; where Search can answer later, it is asked to propose again."""
    ctx = _Context()
    africa = _block("africa_bilateral")
    _show(ctx, africa)
    surface = _Search(africa, refusal=refusal)
    requirement = _dataset("aid", africa["spec_id"], _STATED_INCLUSIVE, recipient="Africa")

    result = await _propose(ctx, [requirement], words=_G7, surface=surface)

    assert result["code"] == code.value, result
    assert len(surface.calls) == 1
    if code is ProblemCode.STATED_PROVIDER_DROPPED:
        assert result["detail"]["unserved"] == [
            {"requirement_id": "aid", "spec_id": africa["spec_id"], "option_id": _INCLUSIVE}
        ]


async def test_a_compile_held_from_an_earlier_proposal_is_reused() -> None:
    """A stated requirement whose compile the turn already holds is pinned to it
    with no new compile."""
    ctx = _Context()
    africa = _block("africa_bilateral")
    _show(ctx, africa)
    compiled = _block("africa_inclusive")
    save_aligned_pin(
        ctx.state,
        AlignedPin(
            origin=africa["spec_id"],
            option_ids=(_INCLUSIVE,),
            spec_id=compiled["spec_id"],
            digest={"spec_block": compiled},
        ),
        warnings=(),
    )
    surface = _Search(africa)
    requirement = _dataset("aid", africa["spec_id"], _STATED_INCLUSIVE, recipient="Africa")

    result = await _propose(ctx, [requirement], words=_G7, surface=surface)

    assert result["outcome"] == Outcome.OK.value, result
    assert _pins(ctx) == {"aid": compiled["spec_id"]}
    assert surface.calls == []


@pytest.mark.parametrize(
    ("failing", "code"),
    [
        pytest.param(True, ProblemCode.SURFACE_UNAVAILABLE, id="route-unavailable"),
        pytest.param(False, ProblemCode.STATED_PROVIDER_DROPPED, id="read-and-applied"),
    ],
)
async def test_a_revision_reads_the_stated_scope_before_admission(
    failing: bool, code: ProblemCode
) -> None:
    """A plan revised with no model call (`PlannerAdapter.revise`) reads the turn's
    words and every `measure` it and the plan on record state before admission, as
    `propose_plan` does: where the route does not answer, the revision is refused
    for it; read, DAC1 under a `measure` stating imputed multilateral aid is refused
    as one that cannot count it."""
    planned = _Context()
    _show(planned, _block("dac1"))
    first = await _propose(planned, [_dac1_total("official development assistance")], words=None)
    assert first["outcome"] == Outcome.OK.value, first
    plan = load_plan(planned.state)
    assert plan is not None
    planned.state.pop(STATED_SCOPE_STATE_KEY, None)
    reader = _Reader(failing=failing)

    async def translate(call: Any) -> dict[str, object]:
        raise AssertionError("a refused revision is not translated")

    adapter = PlannerAdapter(
        settings=Settings(), surface=_Search(), translate=translate, stated_scope=reader.reader()
    )
    token = current_user_message.set(_G7)
    try:
        result = await adapter.revise(
            ctx=node_ctx(state=planned.state, message=_G7),
            prior_plan=plan.model_dump(mode="json"),
            evidence_requirements=[_dac1_total(_DAC1_STATED)],
            inherits={},
        )
    finally:
        current_user_message.reset(token)

    assert result["refused"]["code"] == code.value, result
    assert reader.calls == [[_G7, _DAC1_STATED, "official development assistance"]]


_SUMMED = "US bilateral plus multilateral ODA"


async def test_bilateral_plus_multilateral_words_name_dac1s_own_total() -> None:
    """ "US bilateral plus multilateral ODA" on DAC1 total ODA: Search reads words
    summing the two lines as the dataset's own summing option, which on DAC1 is the
    total it is counted on, so the pin stands with no ask and no compile."""
    ctx = _Context()
    total = dac1_with_contributions()
    _show(ctx, total)
    surface = _Search(total)

    result = await _propose(
        ctx,
        [_dataset("oda", total["spec_id"], _SUMMED, donor="United States")],
        words=_SUMMED,
        surface=surface,
    )

    assert result["outcome"] == Outcome.OK.value, result
    assert _pins(ctx) == {"oda": total["spec_id"]}
    assert surface.calls == []


def _released() -> list[AgentRecord]:
    """The ask for DAC1 counting imputed multilateral aid, answered "count it as
    asked" (`_kept`)."""
    return _kept("count it as asked")


async def test_an_answer_not_to_keep_is_not_asked_again_in_the_turn() -> None:
    """The user answered the ask for DAC1 with "count it as asked". The planner
    proposing DAC1 again is refused with no ask: the user has answered, so the
    refusal asks the planner to remove the figure or repin it."""
    ctx = _Context(*_released())
    _show(ctx, _block("dac1"))

    result = await _propose(ctx, [_dac1_total(_DAC1_STATED)], words=_G7)

    assert result["code"] == ProblemCode.STATED_PROVIDER_DROPPED.value, result
    assert result["action"]["call"] == "propose_plan"
    assert result["detail"]["unserved"] == [
        {"requirement_id": "g7_total", "spec_id": DAC1, "option_id": _INCLUSIVE}
    ]


async def test_an_answer_not_to_keep_releases_a_keep_the_conversation_held() -> None:
    """A turn opening with DAC1 kept for imputed multilateral aid, whose transcript
    holds the user's answer not to keep it: DAC1 is refused again, and the plan
    repinned to the DAC2a total counted that way is admitted, holding no keep in the
    turn's state or the conversation's."""
    seeded = _Context()
    seeded.state[STATED_SCOPE_KEPT_STATE_KEY] = [[DAC1, _INCLUSIVE]]
    ctx = _next_turn(seeded)
    ctx.records = _released()
    g7 = _block("g7_bilateral")
    _show(ctx, _block("dac1"), g7)
    surface = _Search(g7)

    refused = await _propose(ctx, [_dac1_total(_DAC1_STATED)], words=_G7)
    repinned = await _propose(
        ctx,
        [_dataset("g7_total", g7["spec_id"], _DAC1_STATED, recipient="developing countries")],
        words=_G7,
        surface=surface,
    )

    assert refused["code"] == ProblemCode.STATED_PROVIDER_DROPPED.value, refused
    assert repinned["outcome"] == Outcome.OK.value, repinned
    assert _pins(ctx) == {"g7_total": _spec("g7_inclusive")}
    assert stated_scope_kept(ctx.state) == frozenset()
    conversation = ConversationState(
        conversation_id="conv_01keep", stated_scope_kept=frozenset({(DAC1, _INCLUSIVE)})
    )
    keep_turn_records(conversation, ctx.state)
    assert conversation.stated_scope_kept == frozenset()


_G7_TOTAL_INCLUSIVE = (
    "What share of G7 total ODA, including imputed multilateral aid, goes to Africa?"
)


async def test_a_total_under_imputed_multilateral_aid_is_aligned() -> None:
    """ "What share of G7 total ODA, including imputed multilateral aid, goes to
    Africa?" with the numerator stating imputed multilateral aid and the denominator
    "G7 total ODA" on DAC2a: bilateral plus imputed multilateral aid is itself a
    count of total ODA, so the denominator holds nothing of its own and is aligned
    with the numerator."""
    ctx = _Context()
    blocks = _g7_blocks()
    _show(ctx, *blocks)
    surface = _Search(*blocks)
    requirements = _g7_share(
        numerator="ODA to Africa including imputed multilateral aid", denominator="G7 total ODA"
    )

    result = await _propose(
        ctx, requirements, words=_G7_TOTAL_INCLUSIVE, surface=surface, answer_mode="computed"
    )

    assert result["outcome"] == Outcome.OK.value, result
    assert _pins(ctx) == {"africa": _spec("africa_inclusive"), "developing": _spec("g7_inclusive")}


async def test_a_kept_total_holds_its_own_scope_in_a_share() -> None:
    """G7 aid to Africa stating imputed multilateral aid over DAC1 total ODA, which
    the user kept as published: the numerator is counted as stated, DAC1 holds what
    it counts, and the share states the difference."""
    ctx = _Context(*_kept(KEEP_PUBLISHED_OPTION))
    blocks = [_block("africa_bilateral"), _block("dac1")]
    _show(ctx, *blocks)
    surface = _Search(*blocks)
    requirements = [
        _dataset("africa", _spec("africa_bilateral"), _STATED_INCLUSIVE, recipient="Africa"),
        _dac1_total(_DAC1_STATED),
        _share("africa", "g7_total"),
    ]

    result = await _propose(ctx, requirements, words=_G7, surface=surface, answer_mode="computed")

    assert result["outcome"] == Outcome.OK.value, result
    assert _pins(ctx) == {"africa": _spec("africa_inclusive"), "g7_total": DAC1}
    plan = load_plan(ctx.state)
    assert plan is not None
    assert [d.caveat for d in plan.decisions if d.caveat == "providers_differ"] == [
        "providers_differ"
    ]


async def test_a_pin_switched_on_the_plan_card_offers_no_omitted_scope() -> None:
    """G7 aid to Africa switched to bilateral flows on the plan card, its `measure`
    naming no scope, under words asking for imputed multilateral aid: the switch is
    the user's later word, so the plan is not refused for leaving the scope out."""
    ctx = _Context()
    switched = _block("africa_switched_bilateral")
    _show(ctx, switched)

    result = await _propose(
        ctx, [_dataset("aid", switched["spec_id"], "net ODA", recipient="Africa")], words=_G7
    )

    assert result["outcome"] == Outcome.OK.value, result


async def test_a_compile_moving_more_than_the_stated_scope_is_refused_for_its_calculation() -> None:
    """US aid to Kenya stating imputed multilateral aid, over a net total its share
    counts it on: one compile moves it on both axes, and where Search refuses it the
    refusal names the share and records nothing against the stated option, since
    the measure basis may be what Search refused."""
    ctx = _Context()
    aid = provider_block(HEALTH, narrow=True, active=_BILATERAL)
    total = basis_block(TOTAL, narrow=False, active="measure_basis:net", provenance="explicit")
    _show(ctx, aid, total)
    surface = _Search(
        aid, refusal={"code": "requested_option_unavailable", "retry_mode": "do_not_retry"}
    )
    requirements = [
        _dataset("aid", HEALTH, _STATED_INCLUSIVE, donor="United States", recipient="Kenya"),
        _dataset("total", TOTAL, "net ODA", donor="United States"),
        _share("aid", "total"),
    ]

    result = await _propose(ctx, requirements, words=_G7, surface=surface)

    assert result["code"] == ProblemCode.OPERATION_UNSUPPORTED.value, result
    (call,) = surface.calls
    assert {e["option_id"] for e in call["scope_entries"]} >= {_INCLUSIVE, "measure_basis:net"}
    assert stated_uncompiled(ctx.state) == {}


async def test_a_sum_stating_the_scope_nowhere_is_refused_once_a_turn() -> None:
    """Bilateral aid to Africa plus the imputed multilateral series, under words
    asking for imputed multilateral aid, stating it in no `measure`: refused once,
    and admitted proposed again unchanged, the planner's word that the user's words
    qualify none of its figures."""
    ctx = _Context()
    _show(ctx, *_g7_blocks())
    requirements, decisions = _africa_sum("bilateral aid")

    first = await _propose(ctx, requirements, decisions, words=_G7, answer_mode="computed")
    again = await _propose(ctx, requirements, decisions, words=_G7, answer_mode="computed")

    assert first["code"] == ProblemCode.STATED_SCOPE_UNATTRIBUTED.value, first
    assert again["outcome"] == Outcome.OK.value, again

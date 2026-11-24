"""`propose_plan` against the provider option each dataset requirement declares.

The planner declares the option the user's words ask a figure to count
(`DatasetRequirement.scope_options`). Admission counts the pin on it
(`submission.service._declared_claims`), refuses a pin that cannot count it, refuses a
share, quotient or comparison only some of whose inputs declare one
(`_half_declared`), refuses a sum adding such a requirement to imputed multilateral
aid (`_stated_summed`), and writes a line on the plan card's provider row where the
figure and the user's words part (`_scope_notes`). Search's reader reads the turn's
words once (`planner.stated_scope`).

The reader answers each text as dfo-apis' route recorded it
(`fixtures/stated_scope_readings.json`), through `LiveApisClient`. The G7 blocks are
the ones dev Search answered for "Show G7 aid to Africa as a share of G7 aid to all
developing countries each year from 1970 to 2024, including imputed multilateral aid"
(`fixtures/g7_inclusive_share_spec_blocks.json`); the US blocks are `basis_alignment`'s.
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
from compiler.surface import ToolCallResult, ToolSurfaceUnavailableError
from data_mcp.search_result import SEARCH_ENVELOPE_META_KEY
from harness.conversation import current_user_message
from harness.planner_adapter import PlannerAdapter
from harness.sentences import TERMINAL_PROPOSE_CODES
from harness.settings import Settings
from harness.turn.store import AgentRecord
from planner.plan import (
    AlignedPin,
    DatasetRequirement,
    PinSource,
    aligned_pins,
    load_plan,
    save_pin_sources,
    windowed_pins,
)
from planner.source_reads import _NO_ONE_DATASET, _UNANSWERED
from planner.specs import planner_tool_specs
from planner.stated_scope import (
    STATED_SCOPE_STATE_KEY,
    ApisStatedScopeReader,
    StatedScopeReader,
    stated_uncompiled,
)
from planner.tool_results import Outcome, ProblemCode
from planner.tools import load_pending_ask
from planner.workflow import ToolId
from submission.service import _moved_axis_slot, _stated_label, save_aligned_pin

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
_CONTRIBUTIONS = "provider:contributions"


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
    With `failing`, the route answers 503 to every call, and with `refusing` the status
    and body given, which is no retry's cure when the body says `retryable: false`."""

    def __init__(
        self, *, failing: bool = False, refusing: tuple[int, dict[str, Any]] | None = None
    ) -> None:
        self.calls: list[list[str]] = []
        self.failing = failing
        self.refusing = refusing

    def handler(self, request: httpx2.Request) -> httpx2.Response:
        assert (request.method, request.url.path) == ("POST", "/api/search/stated-scope")
        texts = json.loads(request.content)["texts"]
        self.calls.append(texts)
        if self.refusing is not None:
            return httpx2.Response(self.refusing[0], json=self.refusing[1])
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
    refuses every compile, or, with `refusing` too, each compile posting that option.
    With `then`, every compile after the first answers that refusal instead, with
    `empty`, every compile answers no part, and with `unreachable_after`, every compile
    after that many cannot reach Search, for good. Each call's arguments are recorded
    in `calls`."""

    def __init__(
        self,
        *blocks: Mapping[str, Any],
        refusal: Mapping[str, Any] | None = None,
        refusing: str | None = None,
        then: Mapping[str, Any] | None = None,
        empty: bool = False,
        unreachable_after: int | None = None,
    ):
        self.blocks = blocks
        self.refusal = refusal
        self.refusing = refusing
        self.then = then
        self.empty = empty
        self.unreachable_after = unreachable_after
        self.calls: list[Mapping[str, Any]] = []

    async def list_tools(self) -> list[Any]:
        return []

    async def call_tool(
        self, *, tool: ToolId, arguments: Mapping[str, Any], on_progress: Any = None
    ) -> ToolCallResult:
        del on_progress
        assert tool.name == "resolve_definition", tool
        self.calls.append(arguments)
        if self.then is not None and len(self.calls) > 1:
            return ToolCallResult(is_error=True, structured_content=dict(self.then))
        if self.unreachable_after is not None and len(self.calls) > self.unreachable_after:
            raise ToolSurfaceUnavailableError("data-mcp is not covered", retryable=False)
        if self.empty:
            return ToolCallResult(is_error=False, meta={SEARCH_ENVELOPE_META_KEY: {"parts": []}})
        posted = {entry["option_id"] for entry in arguments["scope_entries"]}
        if self.refusal is not None and (self.refusing is None or self.refusing in posted):
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
    declares: str | None = None,
    **fields: Any,
) -> dict[str, Any]:
    """A dataset requirement pinning `pin`, from `donor`, to `recipient` where given,
    declaring the provider option `declares` where given."""
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
        **({"scope_options": [declares]} if declares is not None else {}),
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


def _dac1_total(
    measure: str, declares: str | None = None, *, period: tuple[int, int] = (1970, 2024)
) -> dict[str, Any]:
    """G7 total ODA on DAC1, under `measure`, declaring `declares` where given."""
    return _dataset("g7_total", DAC1, measure, period=period, declares=declares)


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


async def test_a_retryable_failure_asks_to_propose_again_without_vouching_for_the_plan() -> None:
    """Admission has not run when the read fails, so the refusal does not call the plan
    sound, and carries none of Search's codes."""
    ctx = _Context()
    _show(ctx, _block("africa_bilateral"))

    result = await _propose(
        ctx,
        [_dataset("aid", _spec("africa_bilateral"), "net ODA", recipient="Africa")],
        words=_G7,
        reader=_Reader(failing=True),
    )

    assert result["code"] == ProblemCode.SURFACE_UNAVAILABLE.value, result
    assert "propose" in result["message"]
    assert "sound" not in result["message"]
    assert "unavailable" not in result["message"]


@pytest.mark.parametrize(
    "refusing",
    [
        pytest.param((404, {"detail": "not found"}), id="route-not-deployed"),
        pytest.param((422, {"code": "request_invalid", "retryable": False}), id="request-refused"),
    ],
)
async def test_a_permanent_failure_stops_the_entry_naming_the_read(
    refusing: tuple[int, dict[str, Any]],
) -> None:
    """A route no retry changes ends the planner's entry on a terminal refusal, so the
    turn stops at once and the planner does not loop; the refusal names the read in
    `detail` and keeps Search's codes and statuses out of its text."""
    ctx = _Context()
    _show(ctx, _block("africa_bilateral"))

    result = await _propose(
        ctx,
        [_dataset("aid", _spec("africa_bilateral"), "net ODA", recipient="Africa")],
        words=_G7,
        reader=_Reader(refusing=refusing),
    )

    assert result["code"] == ProblemCode.STATED_SCOPE_UNREADABLE.value, result
    assert result["code"] in {code.value for code in TERMINAL_PROPOSE_CODES}
    assert result["detail"]["read"] == "stated_scope"
    assert "request_invalid" not in result["message"]
    assert str(refusing[0]) not in result["message"]
    assert "retry" not in result["message"].lower() or "fails the same way" in result["message"]
    assert load_plan(ctx.state) is None


async def test_the_words_are_cut_to_the_routes_cap_from_their_start() -> None:
    """Studio bounds a message by the request's bytes alone, so the read cuts the words
    to the route's per-text cap, keeping their end as Search reads a message."""
    ctx = _Context()
    _show(ctx, _block("africa_bilateral"))
    reader = _Reader()
    words = "x" * 5000 + _G7
    reader.failing = True
    await _propose(
        ctx,
        [_dataset("aid", _spec("africa_bilateral"), "net ODA", recipient="Africa")],
        words=words,
        reader=reader,
    )

    sent = reader.calls[0][0]
    assert words.endswith(sent)
    assert len(sent) < len(words)


def _us_bilateral() -> dict[str, Any]:
    """US aid to Kenya counted as bilateral flows, as the words "bilateral aid" ask."""
    return provider_block(HEALTH, narrow=True, active=_BILATERAL, provenance="explicit")


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
    """ "What share of US ODA goes to multilaterals?": declaring nothing, the
    numerator on DAC1's multilateral row, as the words name it, and the total on total
    ODA stay two rows, since alignment would divide one row by itself, and neither
    row reads a line."""
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
    assert _notes(ctx) == {}


async def test_bilateral_and_imputed_aid_shown_separately_are_served() -> None:
    """ "G7 bilateral aid and imputed multilateral aid to Africa separately": a bare
    "multilateral" names no option, so the words name nothing and no row reads a
    line."""
    ctx = _Context()
    _show(ctx, _block("separate_bilateral"), _block("separate_imputed"))
    requirements = [
        _dataset("bilateral", _spec("separate_bilateral"), "bilateral aid", recipient="Africa"),
        _dataset("imputed", _spec("separate_imputed"), "imputed multilateral aid"),
    ]

    result = await _propose(ctx, requirements, words=_SEPARATELY)

    assert result["outcome"] == Outcome.OK.value, result
    assert _notes(ctx) == {}


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
    the requirement still declares imputed multilateral aid as their words ask. The
    switch is their later word: nothing moves, and an amendment of the period keeps
    it."""
    ctx = _Context()
    switched = _block("africa_switched_bilateral")
    _show(ctx, switched)
    surface = _Search(switched)
    requirement = _dataset(
        "aid", switched["spec_id"], "net ODA", recipient="Africa", declares=_INCLUSIVE
    )

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


def _recorded() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The first plan admitted for the G7 question: Africa's bilateral part plus the
    imputed series, over DAC1 total ODA, less `accept_provider_difference`, which no
    calculation takes."""
    proposal = copy.deepcopy(_RECORDED["proposal"])
    for requirement in proposal["evidence_requirements"]:
        requirement.pop("accept_provider_difference", None)
    return proposal["evidence_requirements"], proposal["decisions"]


def _africa_sum(declares: str | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """G7 aid to Africa as its bilateral part, declaring `declares` where given, plus
    the imputed multilateral series, as the first plan for the G7 question summed
    them."""
    requirements, decisions = _recorded()
    kept = [
        r for r in requirements if r["id"] in {"africa_bilateral", "africa_imputed", "africa_total"}
    ]
    (bilateral,) = [r for r in kept if r["id"] == "africa_bilateral"]
    if declares is not None:
        bilateral["scope_options"] = [declares]
    return kept, [d for d in decisions if d["slot"] == "africa_total_method"]


@pytest.mark.parametrize(
    ("declares", "denominator", "step"),
    [
        pytest.param(_INCLUSIVE, None, "africa_total", id="the-summed-part-declares-it"),
        pytest.param(None, _INCLUSIVE, "share", id="the-other-side-declares-it"),
    ],
)
async def test_bilateral_plus_imputed_aid_is_refused_toward_the_inclusive_dataset(
    declares: str | None, denominator: str | None, step: str
) -> None:
    """G7 aid to Africa as bilateral aid plus the imputed multilateral series, where
    the user's words ask for imputed multilateral aid. Counting the bilateral part on
    the inclusive option it or the other side declares would add the imputed series
    to a figure already holding it, so the plan is refused toward one requirement
    pinning the part counted that way. No compile is asked for."""
    ctx = _Context()
    blocks = [*_g7_blocks()]
    _show(ctx, *blocks)
    surface = _Search(*blocks)
    requirements, decisions = _africa_sum(declares)
    if denominator is not None:
        requirements = [
            *requirements,
            _dataset(
                "developing",
                _spec("g7_bilateral"),
                "net ODA",
                recipient="developing countries",
                declares=denominator,
            ),
            _share("africa_total", "developing"),
        ]

    result = await _propose(
        ctx, requirements, decisions, words=_G7, surface=surface, answer_mode="computed"
    )

    assert result["code"] == ProblemCode.PROVIDERS_UNALIGNED.value, result
    assert result["detail"]["provider"]["step_id"] == step
    assert result["detail"]["provider"]["sum_id"] == "africa_total"
    assert result["detail"]["provider"]["unchosen_id"] == "africa_imputed"
    assert surface.calls == []
    assert load_plan(ctx.state) is None


async def test_two_sides_stating_one_scope_on_one_series_are_refused() -> None:
    """A share both of whose inputs declare bilateral, over DAC1 total ODA read the
    same way on both sides: counted as declared, both read DAC1's bilateral row, and
    the share would answer 100%."""
    ctx = _Context()
    dac1 = _block("dac1")
    _show(ctx, dac1)
    surface = _Search(dac1)
    requirements = [
        _dataset("bilateral", DAC1, "bilateral ODA", declares=_BILATERAL),
        _dataset("total", DAC1, "bilateral ODA", declares=_BILATERAL),
        _share("bilateral", "total"),
    ]

    result = await _propose(ctx, requirements, words=_G7_BILATERAL, surface=surface)

    assert result["code"] == ProblemCode.SHARE_OF_ITSELF.value, result


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
    """G7 aid to Africa declaring imputed multilateral aid, whose compile counted that
    way Search refuses: the pin cannot count it, so the plan is refused naming
    Search's code in `detail`; where Search can answer later, it is asked to propose
    again."""
    ctx = _Context()
    africa = _block("africa_bilateral")
    _show(ctx, africa)
    surface = _Search(africa, refusal=refusal)
    requirement = _dataset(
        "aid", africa["spec_id"], "net ODA", recipient="Africa", declares=_INCLUSIVE
    )

    result = await _propose(ctx, [requirement], words=_G7, surface=surface)

    assert result["code"] == code.value, result
    assert len(surface.calls) == 1
    assert refusal["code"] not in result["message"]
    if code is ProblemCode.STATED_PROVIDER_DROPPED:
        assert result["detail"]["unserved"] == [
            {"requirement_id": "aid", "spec_id": africa["spec_id"], "option_id": _INCLUSIVE}
        ]
        assert result["detail"]["causes"] == {"aid": refusal["code"]}
    else:
        assert result["detail"]["cause"] == _UNANSWERED


async def test_a_stated_compile_with_no_part_records_no_one_dataset() -> None:
    """G7 aid to Africa declaring imputed multilateral aid, whose compile counted that
    way Search answers with no part: no refusal of Search's, so the pin is recorded as
    holding no one dataset counted that way, and refused."""
    ctx = _Context()
    africa = _block("africa_bilateral")
    _show(ctx, africa)
    surface = _Search(africa, empty=True)
    requirement = _dataset(
        "aid", africa["spec_id"], "net ODA", recipient="Africa", declares=_INCLUSIVE
    )

    result = await _propose(ctx, [requirement], words=_G7, surface=surface)

    assert result["code"] == ProblemCode.STATED_PROVIDER_DROPPED.value, result
    assert result["detail"]["causes"] == {"aid": _NO_ONE_DATASET}
    assert stated_uncompiled(ctx.state) == {(africa["spec_id"], _INCLUSIVE): _NO_ONE_DATASET}


async def test_a_stated_compile_that_cannot_reach_search_ends_the_entry() -> None:
    """G7 aid to Africa declaring imputed multilateral aid, whose compile counted that
    way cannot reach Search for good: a fault of the system, no answer about the pin.
    The entry ends (`STATED_COMPILE_UNREACHABLE`, a terminal code) naming the compile,
    nothing is recorded against the option, and the user is asked nothing."""
    ctx = _Context()
    africa = _block("africa_bilateral")
    _show(ctx, africa)
    surface = _Search(africa, unreachable_after=0)
    requirement = _dataset(
        "aid", africa["spec_id"], "net ODA", recipient="Africa", declares=_INCLUSIVE
    )

    result = await _propose(ctx, [requirement], words=_G7, surface=surface)

    assert result["code"] == ProblemCode.STATED_COMPILE_UNREACHABLE.value, result
    assert ProblemCode.STATED_COMPILE_UNREACHABLE in TERMINAL_PROPOSE_CODES
    assert result["detail"] == {
        "compile": "stated_scope",
        "spec_id": africa["spec_id"],
        "option_ids": [_INCLUSIVE],
    }
    assert stated_uncompiled(ctx.state) == {}
    assert load_pending_ask(ctx.state) is None


async def test_a_compile_held_from_an_earlier_proposal_is_reused() -> None:
    """A declared requirement whose compile the turn already holds is pinned to it
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
    requirement = _dataset(
        "aid", africa["spec_id"], "net ODA", recipient="Africa", declares=_INCLUSIVE
    )

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
    words before admission, as `propose_plan` does: where the route does not answer,
    the revision is refused for it; read, DAC1 declaring imputed multilateral aid is
    refused as one that cannot count it."""
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
            evidence_requirements=[_dac1_total("official development assistance", _INCLUSIVE)],
            inherits={},
        )
    finally:
        current_user_message.reset(token)

    assert result["refused"]["code"] == code.value, result
    assert reader.calls == [[_G7]]


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


_UNANSWERED_REFUSAL = {"code": "upstream_unavailable", "retry_mode": "retry_after_delay"}


_REFUSED_FOR_GOOD = {"code": "requested_option_unavailable", "retry_mode": "do_not_retry"}


@pytest.mark.parametrize(
    ("refusing", "then", "code"),
    [
        pytest.param(
            None, None, ProblemCode.STATED_PROVIDER_DROPPED, id="the-stated-scope-refused"
        ),
        pytest.param(
            "measure_basis:net",
            None,
            ProblemCode.OPERATION_UNSUPPORTED,
            id="the-calculations-refused",
        ),
        pytest.param(
            None,
            _UNANSWERED_REFUSAL,
            ProblemCode.SURFACE_UNAVAILABLE,
            id="the-stated-scope-unanswered",
        ),
        pytest.param(
            None,
            "unreachable",
            ProblemCode.STATED_COMPILE_UNREACHABLE,
            id="the-stated-scope-unreachable",
        ),
    ],
)
async def test_a_compile_moving_more_than_the_stated_scope_is_refused_for_what_search_refuses(
    refusing: str | None, then: dict[str, str] | str | None, code: ProblemCode
) -> None:
    """US aid to Kenya declaring imputed multilateral aid, over a net total its share
    counts it on: one compile moves it on both axes. Where Search refuses it, the
    pin is compiled on the declared option alone to find which move Search refuses.
    That refused too, the pin cannot count what it declares and is refused; served,
    the combined refusal stands, naming the share; not answered, the
    planner is asked to propose again; unreachable for good, the entry ends naming
    that compile. None of those records anything against the declared option."""
    ctx = _Context()
    aid = provider_block(HEALTH, narrow=True, active=_BILATERAL)
    total = basis_block(TOTAL, narrow=False, active="measure_basis:net", provenance="explicit")
    _show(ctx, aid, total)
    surface = _Search(
        aid,
        refusal=_REFUSED_FOR_GOOD,
        refusing=refusing,
        then=then if isinstance(then, dict) else None,
        unreachable_after=1 if then == "unreachable" else None,
    )
    requirements = [
        _dataset(
            "aid",
            HEALTH,
            "net ODA",
            donor="United States",
            recipient="Kenya",
            declares=_INCLUSIVE,
        ),
        _dataset("total", TOTAL, "net ODA", donor="United States"),
        _share("aid", "total"),
    ]

    result = await _propose(ctx, requirements, words=_G7, surface=surface)

    assert result["code"] == code.value, result
    both, alone = surface.calls
    assert {e["option_id"] for e in both["scope_entries"]} >= {_INCLUSIVE, "measure_basis:net"}
    assert "measure_basis:net" not in {e["option_id"] for e in alone["scope_entries"]}
    if code is ProblemCode.STATED_COMPILE_UNREACHABLE:
        assert result["detail"] == {
            "compile": "stated_scope",
            "spec_id": HEALTH,
            "option_ids": [_INCLUSIVE],
        }
        assert stated_uncompiled(ctx.state) == {}
        assert load_pending_ask(ctx.state) is None
    elif code is ProblemCode.STATED_PROVIDER_DROPPED:
        assert stated_uncompiled(ctx.state) == {
            (HEALTH, _INCLUSIVE): "requested_option_unavailable"
        }
        assert result["detail"]["causes"] == {"aid": "requested_option_unavailable"}
    else:
        assert (
            result["detail"]["basis"]["requirement_id"],
            result["detail"]["basis"]["step_id"],
        ) == (
            "aid",
            "share",
        )
        assert stated_uncompiled(ctx.state) == {}
    if code is ProblemCode.SURFACE_UNAVAILABLE:
        assert result["detail"]["cause"] == _UNANSWERED
        assert result["detail"]["basis"]["option_ids"] == [_INCLUSIVE]


def _aid_over_net_total() -> tuple[_Context, list[dict[str, Any]], Mapping[str, Any]]:
    """US aid to Kenya declaring imputed multilateral aid as a share of a net total:
    the context, the requirements and the aid block."""
    ctx = _Context()
    aid = provider_block(HEALTH, narrow=True, active=_BILATERAL)
    total = basis_block(TOTAL, narrow=False, active="measure_basis:net", provenance="explicit")
    _show(ctx, aid, total)
    requirements = [
        _dataset(
            "aid",
            HEALTH,
            "net ODA",
            donor="United States",
            recipient="Kenya",
            declares=_INCLUSIVE,
        ),
        _dataset("total", TOTAL, "net ODA", donor="United States"),
        _share("aid", "total"),
    ]
    return ctx, requirements, aid


async def test_a_combined_compile_refused_for_good_is_not_posted_again_in_the_turn() -> None:
    """Search refuses for good the compile moving the aid on the stated scope and the
    net basis, and does not answer the stated scope's alone: the planner is asked to
    propose again. Proposing again posts only the stated scope's compile, once, and
    Search not answering it a second time, the combined refusal stands."""
    ctx, requirements, aid = _aid_over_net_total()
    first_surface = _Search(aid, refusal=_REFUSED_FOR_GOOD, then=_UNANSWERED_REFUSAL)

    first = await _propose(ctx, requirements, words=_G7, surface=first_surface)
    again_surface = _Search(aid, refusal=_UNANSWERED_REFUSAL)
    again = await _propose(ctx, requirements, words=_G7, surface=again_surface)

    assert first["code"] == ProblemCode.SURFACE_UNAVAILABLE.value, first
    assert len(first_surface.calls) == 2
    assert again["code"] == ProblemCode.OPERATION_UNSUPPORTED.value, again
    assert (again["detail"]["basis"]["requirement_id"], again["detail"]["basis"]["step_id"]) == (
        "aid",
        "share",
    )
    (alone,) = again_surface.calls
    assert {e["option_id"] for e in alone["scope_entries"]} >= {_INCLUSIVE}
    assert "measure_basis:net" not in {e["option_id"] for e in alone["scope_entries"]}


def test_an_option_with_no_label_and_no_words_is_named_by_its_scope() -> None:
    """An option no table names, no pin labels and no words state is still named, by
    its provider scope in words, and never by its id."""
    label = _stated_label("provider:pooled_total", words="")

    assert label.strip()
    assert "provider:" not in label
    assert "_" not in label


async def test_the_words_are_read_once_a_turn_and_no_measure_is_read() -> None:
    """The turn's words go to the route once: a second proposal reads nothing, and no
    requirement's `measure` is sent."""
    ctx = _Context()
    _show(ctx, _block("separate_bilateral"), _block("separate_imputed"))
    reader = _Reader()
    requirements = [
        _dataset("bilateral", _spec("separate_bilateral"), "bilateral aid", recipient="Africa"),
        _dataset("imputed", _spec("separate_imputed"), "imputed multilateral aid"),
    ]

    await _propose(ctx, requirements, words=_SEPARATELY, reader=reader)
    await _propose(ctx, requirements, words=_SEPARATELY, reader=reader)

    assert reader.calls == [[_SEPARATELY]]


# --- declared options --------------------------------------------------------


def _g7_blocks() -> list[dict[str, Any]]:
    return [_block(name) for name in ("africa_bilateral", "africa_imputed", "g7_bilateral")]


def _g7_share(africa: str | None, developing: str | None) -> list[dict[str, Any]]:
    """G7 aid to Africa over G7 aid to developing countries, each a DAC2a bilateral
    part as dev Search answered it, declaring the option given for each side."""
    return [
        _dataset(
            "africa", _spec("africa_bilateral"), "net ODA", recipient="Africa", declares=africa
        ),
        _dataset(
            "developing",
            _spec("g7_bilateral"),
            "net ODA",
            recipient="developing countries",
            declares=developing,
        ),
        _share("africa", "developing"),
    ]


def _notes(ctx: _Context) -> dict[str, str]:
    """The plan card's line on each requirement's provider row, by requirement id."""
    plan = load_plan(ctx.state)
    assert plan is not None
    return {note.requirement_id: note.line for note in plan.scope_notes}


async def test_the_g7_share_declared_on_both_sides_is_counted_inclusive() -> None:
    """The G7 inclusive question with both sides declaring inclusive: each is moved to
    its DAC2a part's inclusive compile with no refusal and no line on the card."""
    ctx = _Context()
    blocks = _g7_blocks()
    _show(ctx, *blocks)

    result = await _propose(
        ctx,
        _g7_share(_INCLUSIVE, _INCLUSIVE),
        words=_G7,
        surface=_Search(*blocks),
        answer_mode="computed",
    )

    assert result["outcome"] == Outcome.OK.value, result
    assert _pins(ctx) == {
        "africa": _spec("africa_inclusive"),
        "developing": _spec("g7_inclusive"),
    }
    assert _notes(ctx) == {}


@pytest.mark.parametrize(
    ("africa", "developing"),
    [
        pytest.param(_INCLUSIVE, None, id="numerator-only"),
        pytest.param(None, _INCLUSIVE, id="denominator-only"),
    ],
)
async def test_a_share_declared_on_one_side_only_is_refused(
    africa: str | None, developing: str | None
) -> None:
    """A share one of whose inputs declares a provider option and the other none is
    refused once, naming both and the undeclared side's own option, and compiles
    nothing."""
    ctx = _Context()
    blocks = _g7_blocks()
    _show(ctx, *blocks)
    surface = _Search(*blocks)

    result = await _propose(
        ctx, _g7_share(africa, developing), words=_G7, surface=surface, answer_mode="computed"
    )

    assert result["code"] == ProblemCode.STATED_SCOPE_UNATTRIBUTED.value, result
    undeclared = "developing" if africa is not None else "africa"
    assert result["detail"] == {
        "requirement_id": "share",
        "declared": ["africa" if africa is not None else "developing"],
        "undeclared": [undeclared],
    }
    assert f"{undeclared!r} {_BILATERAL!r}" in result["message"]
    assert surface.calls == []
    assert load_plan(ctx.state) is None


async def test_a_plan_declaring_nothing_reads_the_words_on_each_provider_row() -> None:
    """The G7 inclusive question planned with no declaration: admitted on bilateral,
    and each side's provider row on the plan card reads that the words name the
    inclusive count and no figure is counted that way. Each requirement cites the row
    holding its line in place of its pin's own."""
    ctx = _Context()
    blocks = _g7_blocks()
    _show(ctx, *blocks)

    result = await _propose(
        ctx, _g7_share(None, None), words=_G7, surface=_Search(*blocks), answer_mode="computed"
    )

    assert result["outcome"] == Outcome.OK.value, result
    assert _pins(ctx) == {"africa": _spec("africa_bilateral"), "developing": _spec("g7_bilateral")}
    notes = _notes(ctx)
    assert sorted(notes) == ["africa", "developing"]
    assert all(
        line.startswith("Your words name bilateral aid plus imputed") for line in notes.values()
    )
    plan = load_plan(ctx.state)
    assert plan is not None
    by_slot = {decision.slot: decision for decision in plan.decisions}
    for requirement_id, line in notes.items():
        row = by_slot[_moved_axis_slot(requirement_id, "provider")]
        assert row.rationale == line
        assert row.slot in plan.requirement(requirement_id).decision_ids


async def test_a_declared_option_the_words_do_not_name_reads_so_on_its_row() -> None:
    """ "What share of G7 ODA is bilateral?": the numerator declares bilateral and
    moves; the total declares DAC1's own option and stays, its row reading that the
    words do not name it."""
    ctx = _Context()
    dac1 = _block("dac1")
    _show(ctx, dac1)
    requirements = [
        _dataset("bilateral", DAC1, "bilateral ODA", declares=_BILATERAL),
        _dataset("total", DAC1, "official development assistance", declares=_CONTRIBUTIONS),
        _share("bilateral", "total"),
    ]

    result = await _propose(ctx, requirements, words=_G7_BILATERAL, surface=_Search(dac1))

    assert result["outcome"] == Outcome.OK.value, result
    assert _pins(ctx) == {"bilateral": _held(ctx, DAC1, _BILATERAL), "total": DAC1}
    (line,) = _notes(ctx).values()
    assert _notes(ctx)["total"] == line
    assert line.startswith("Counted as bilateral aid plus core contributions")


@pytest.mark.parametrize("total_declared", [True, False], ids=["total-declared", "total-bare"])
async def test_a_total_beside_a_declared_part_declares_its_own_option(
    total_declared: bool,
) -> None:
    """ "US bilateral aid to Kenya as a share of total US ODA": a total declaring
    DAC1's contributions option holds it beside the bilateral part. A bare total is
    the half-declared share, refused naming the total's own option."""
    ctx = _Context()
    total = dac1_with_contributions()
    _show(ctx, _us_bilateral(), total)
    surface = _Search(total)
    requirements = [
        _dataset(
            "bilateral",
            HEALTH,
            "bilateral aid",
            donor="United States",
            recipient="Kenya",
            declares=_BILATERAL,
        ),
        _dataset(
            "total",
            total["spec_id"],
            "total ODA",
            donor="United States",
            declares=_CONTRIBUTIONS if total_declared else None,
        ),
        _share("bilateral", "total"),
    ]

    result = await _propose(ctx, requirements, words=_US_TOTAL, surface=surface)

    if total_declared:
        assert result["outcome"] == Outcome.OK.value, result
        assert _pins(ctx)["total"] == total["spec_id"]
        assert surface.calls == []
    else:
        assert result["code"] == ProblemCode.STATED_SCOPE_UNATTRIBUTED.value, result
        assert f"'total' {_CONTRIBUTIONS!r}" in result["message"]


async def test_inputs_declaring_different_options_keep_them_and_state_the_difference() -> None:
    """G7 aid to Africa declared inclusive over DAC1 total ODA declared on its own
    contributions option, as a user asking for the DAC1 total would: each keeps its
    option, the share states that its inputs count different providers' spending,
    and the DAC1 row reads that the words do not name its option."""
    ctx = _Context()
    blocks = [*_g7_blocks(), _block("dac1")]
    _show(ctx, *blocks)
    requirements = [
        _dataset(
            "africa", _spec("africa_bilateral"), "net ODA", recipient="Africa", declares=_INCLUSIVE
        ),
        _dataset("total", DAC1, "official development assistance", declares=_CONTRIBUTIONS),
        _share("africa", "total"),
    ]

    result = await _propose(
        ctx, requirements, words=_G7, surface=_Search(*blocks), answer_mode="computed"
    )

    assert result["outcome"] == Outcome.OK.value, result
    assert _pins(ctx) == {"africa": _spec("africa_inclusive"), "total": DAC1}
    plan = load_plan(ctx.state)
    assert plan is not None
    assert [d.caveat for d in plan.decisions if d.caveat == "providers_differ"] == [
        "providers_differ"
    ]
    assert list(_notes(ctx)) == ["total"]


async def test_two_scopes_in_the_words_are_served_each_on_its_side() -> None:
    """ "G7 bilateral aid to Africa as a share of G7 total aid to Africa including
    imputed multilateral aid": each side declares its own option, both named in the
    words, so nothing moves, no line is written, and the share states the
    difference."""
    ctx = _Context()
    bilateral, inclusive = _block("bilateral_share_bilateral"), _block("bilateral_share_inclusive")
    _show(ctx, bilateral, inclusive)
    surface = _Search(bilateral, inclusive)
    requirements = [
        _dataset("bilateral", bilateral["spec_id"], "aid", recipient="Africa", declares=_BILATERAL),
        _dataset("total", inclusive["spec_id"], "aid", recipient="Africa", declares=_INCLUSIVE),
        _share("bilateral", "total"),
    ]

    result = await _propose(
        ctx, requirements, words=_TWO_SCOPES, surface=surface, answer_mode="computed"
    )

    assert result["outcome"] == Outcome.OK.value, result
    assert surface.calls == []
    assert _notes(ctx) == {}
    plan = load_plan(ctx.state)
    assert plan is not None
    assert [d.caveat for d in plan.decisions if d.caveat == "providers_differ"] == [
        "providers_differ"
    ]


async def test_an_option_the_pin_does_not_count_is_refused_naming_what_it_offers() -> None:
    """G7 aid to Africa declaring core contributions, which its DAC2a part does not
    publish: refused, naming the options the pin offers and Search's code in
    `detail`, with no compile."""
    ctx = _Context()
    africa = _block("africa_bilateral")
    _show(ctx, africa)
    surface = _Search(africa)
    requirement = _dataset(
        "aid", africa["spec_id"], "net ODA", recipient="Africa", declares=_CONTRIBUTIONS
    )

    result = await _propose(ctx, [requirement], words=_G7, surface=surface)

    assert result["code"] == ProblemCode.STATED_PROVIDER_DROPPED.value, result
    assert result["detail"]["unserved"] == [
        {"requirement_id": "aid", "spec_id": africa["spec_id"], "option_id": _CONTRIBUTIONS}
    ]
    assert f"{_BILATERAL}, {_INCLUSIVE}" in result["message"]
    assert surface.calls == []


async def test_a_line_holds_through_an_amendment_whose_words_name_nothing() -> None:
    """A total declaring an option the words do not name reads so on its row. An
    amendment under words naming nothing carries the declaration and the line, and
    adds none."""
    ctx = _Context()
    dac1 = _block("dac1")
    _show(ctx, dac1)
    surface = _Search(dac1)
    requirements = [
        _dataset("bilateral", DAC1, "bilateral ODA", declares=_BILATERAL),
        _dataset("total", DAC1, "official development assistance", declares=_CONTRIBUTIONS),
        _share("bilateral", "total"),
    ]
    first = await _propose(ctx, requirements, words=_G7_BILATERAL, surface=surface)
    assert first["outcome"] == Outcome.OK.value, first
    set_then = _notes(ctx)

    amended = await _propose(
        ctx,
        [
            *({**r, "period": {"start": 1980, "end": 2024}} for r in requirements[:2]),
            requirements[2],
        ],
        words=_LATER,
        surface=surface,
        amends_version=1,
        question=_G7_BILATERAL,
    )

    assert amended["outcome"] == Outcome.OK.value, amended
    assert list(set_then) == ["total"]
    assert _notes(ctx) == set_then


async def test_a_declaration_is_judged_on_the_words_of_the_turn_that_set_it() -> None:
    """The G7 share declared inclusive on both sides, as the words ask: no line. An
    amendment under words naming nothing that carries the declarations raises none;
    one that changes them to bilateral is judged on its own words, which name
    nothing, so each row reads that its option is not named."""
    ctx = _Context()
    blocks = _g7_blocks()
    _show(ctx, *blocks)
    surface = _Search(*blocks)

    first = await _propose(
        ctx, _g7_share(_INCLUSIVE, _INCLUSIVE), words=_G7, surface=surface, answer_mode="computed"
    )
    carried = await _propose(
        ctx,
        _g7_share(_INCLUSIVE, _INCLUSIVE),
        words=_LATER,
        surface=surface,
        answer_mode="computed",
        amends_version=1,
        question=_G7,
    )
    carried_notes = _notes(ctx)
    changed = await _propose(
        ctx,
        _g7_share(_BILATERAL, _BILATERAL),
        words=_LATER,
        surface=surface,
        answer_mode="computed",
        amends_version=2,
        question=_G7,
    )

    assert [first["outcome"], carried["outcome"], changed["outcome"]] == [Outcome.OK.value] * 3
    assert carried_notes == {}
    notes = _notes(ctx)
    assert sorted(notes) == ["africa", "developing"]
    assert all(line.startswith("Counted as bilateral aid only") for line in notes.values())


async def test_a_declaration_under_words_naming_nothing_reads_so() -> None:
    """A requirement declaring inclusive in a turn whose words name no provider
    option: it is counted that way, and its row reads that the words do not name it."""
    ctx = _Context()
    africa = _block("africa_bilateral")
    _show(ctx, africa)
    requirement = _dataset(
        "aid", africa["spec_id"], "net ODA", recipient="Africa", declares=_INCLUSIVE
    )

    result = await _propose(ctx, [requirement], words=_LATER, surface=_Search(africa), question=_G7)

    assert result["outcome"] == Outcome.OK.value, result
    assert _pins(ctx) == {"aid": _spec("africa_inclusive")}
    assert list(_notes(ctx)) == ["aid"]

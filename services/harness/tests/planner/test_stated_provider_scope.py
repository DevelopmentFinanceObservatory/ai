"""`propose_plan` against the provider option each dataset requirement declares.

The planner declares the option the user's words ask a figure to count
(`DatasetRequirement.scope_options`). Admission counts the pin on it
(`submission.service._declared_claims`), refuses a pin that cannot count it, refuses a
share, quotient or comparison only some of whose inputs declare one
(`_half_declared`), refuses a sum adding such a requirement to imputed multilateral
aid (`_stated_summed`), and writes a line on the plan card's provider row where the
figure and the user's words part (`_provider_line`). Search's reader reads the turn's
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
import logging
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast, get_args

import httpx2
import pytest
from common.live_apis import LiveApisClient
from compiler.sources import open_side_column, source_from_spec_block
from compiler.surface import ToolCallResult, ToolSurfaceUnavailableError
from data_mcp.search_result import SEARCH_ENVELOPE_META_KEY
from harness import studio as studio_module
from harness.conversation import current_user_message
from harness.graph import approve as approve_module
from harness.graph import planning as planning_module
from harness.graph import switch as switch_module
from harness.graph.answer import stored_run_plan
from harness.graph.state import ORCHESTRATOR_BRIEF_KEY, readable_plan
from harness.graph.switch import SwitchFailedError
from harness.plan_display import choice_id
from harness.planner_adapter import PlannerAdapter
from harness.scope_choices import ProviderScope
from harness.sentences import FAILURE_CLASS, TERMINAL_PROPOSE_CODES, FailureClass
from harness.settings import Settings
from harness.turn.store import AgentRecord
from planner.plan import (
    ALIGNED_PINS_STATE_KEY,
    PLAN_STATE_KEY,
    AlignedPin,
    DatasetRequirement,
    PinSource,
    Plan,
    PlanStatus,
    ProviderOptionId,
    ResolvedBy,
    ScopeWords,
    UnsupportedOperation,
    aligned_pins,
    covered,
    load_optional_declines,
    load_plan,
    plan_revision,
    save_pin_sources,
    scope_choice_slot,
    stored_plan,
    whole_on_source,
    windowed_pins,
)
from planner.source_reads import _NO_ONE_DATASET, _UNANSWERED
from planner.specs import planner_tool_specs
from planner.stated_scope import (
    STATED_SCOPE_STATE_KEY,
    ApisStatedScopeReader,
    StatedScopeReader,
    StatedScopeRecord,
    TextReading,
    read_stated_scope,
    stated_scope_record,
    stated_uncompiled,
)
from planner.tool_results import Outcome, ProblemCode
from planner.tools import load_pending_ask
from planner.workflow import ToolId
from submission.service import (
    _ALIGNED_REASONS,
    _scope_words,
    _stated_label,
    save_aligned_pin,
)

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
from ..scripted_genai import ScriptedGenai, call_part, reply, text_part
from ..scripted_planner import SIGNED, scripted_planner

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
    and body given, which is no retry's cure when the body says `retryable: false`.
    Search's capabilities document lists `doors` as live, answers a bare 404, as a
    wrong address does, where `doors` is `None`, and 503 with `capabilities_down`; each
    read is counted in `capability_reads`."""

    def __init__(
        self,
        *,
        failing: bool = False,
        refusing: tuple[int, dict[str, Any]] | None = None,
        doors: Sequence[str] | None = ("options", "stated_scope"),
        capabilities_down: bool = False,
    ) -> None:
        self.calls: list[list[str]] = []
        self.capabilities_down = capabilities_down
        self.failing = failing
        self.refusing = refusing
        self.doors = doors
        self.capability_reads = 0

    def handler(self, request: httpx2.Request) -> httpx2.Response:
        if (request.method, request.url.path) == ("GET", "/api/search/capabilities"):
            self.capability_reads += 1
            if self.capabilities_down:
                return httpx2.Response(503, json={"code": "unavailable", "retryable": True})
            if self.doors is None:
                return httpx2.Response(404, text="<html>Not Found</html>")
            doors = [{"name": door, "live": True} for door in self.doors]
            return httpx2.Response(200, json={"doors": doors})
        assert (request.method, request.url.path) == ("POST", "/api/search/stated-scope")
        texts = json.loads(request.content)["texts"]
        self.calls.append(texts)
        if self.refusing is not None:
            return httpx2.Response(self.refusing[0], json=self.refusing[1])
        if self.failing:
            return httpx2.Response(503, json={"code": "unavailable", "retryable": True})
        return httpx2.Response(200, json={"readings": [_READINGS[text] for text in texts]})

    def reader(self, *, clock: Callable[[], float] | None = None) -> StatedScopeReader:
        async def no_sleep(_: float) -> None:
            return None

        return ApisStatedScopeReader(
            LiveApisClient(
                base_url="http://apis.test",
                transport=httpx2.MockTransport(self.handler),
                max_attempts=1,
                sleep=no_sleep,
            ),
            **({"clock": clock} if clock is not None else {}),
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
    surface: _Search | _Wholes | None = None,
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


async def test_a_route_that_does_not_answer_ends_the_entry_on_a_retryable_refusal() -> None:
    """The stated-scope read is made before admission; where the route does not
    answer after the reader's retries, `propose_plan` ends the entry on
    `STATED_SCOPE_UNAVAILABLE`, which asks the user to try again shortly, admits
    nothing and guesses no scope. The refusal does not call the plan sound."""
    ctx = _Context()
    _show(ctx, _block("africa_bilateral"))

    result = await _propose(
        ctx,
        [_dataset("aid", _spec("africa_bilateral"), "net ODA", recipient="Africa")],
        words=_G7,
        reader=_Reader(failing=True),
    )

    assert result["code"] == ProblemCode.STATED_SCOPE_UNAVAILABLE.value, result
    assert result["code"] in {code.value for code in TERMINAL_PROPOSE_CODES}
    assert FAILURE_CLASS[result["code"]] is FailureClass.TRANSIENT
    assert result["detail"] == {"read": "stated_scope", "status": 503, "code": "unavailable"}
    assert "sound" not in result["message"]
    assert "503" not in result["message"]
    assert "unavailable" not in result["message"]
    assert load_plan(ctx.state) is None


async def test_each_read_retries_on_its_own_so_an_earlier_failure_spends_nothing() -> None:
    """Every read site retries within its own call, so neither the brief's hint read
    nor an earlier proposal's failure in the turn turns a later failure into a fault
    in the system, and a read that answers after them admits the plan."""
    ctx = _Context()
    bilateral = _block("separate_bilateral")
    _show(ctx, bilateral)
    requirements = [_dataset("aid", bilateral["spec_id"], "bilateral aid", recipient="Africa")]
    token = current_user_message.set(_G7_BILATERAL)
    try:
        hint = await read_stated_scope(ctx.state, _Reader(failing=True).reader())
    finally:
        current_user_message.reset(token)

    codes = [hint["code"] if hint is not None else None]
    for reader in (_Reader(failing=True), _Reader(failing=True), _Reader()):
        result = await _propose(ctx, requirements, words=_G7_BILATERAL, reader=reader)
        codes.append(result.get("code") or result["outcome"])

    unavailable = ProblemCode.STATED_SCOPE_UNAVAILABLE.value
    assert codes == [unavailable, unavailable, unavailable, Outcome.OK.value]


_NOT_FOUND = (404, {"code": "not_found", "status": 404, "retryable": False})
_HTML_404 = (404, {"detail": "not found"})


@pytest.mark.parametrize("refusing", [_NOT_FOUND, _HTML_404], ids=["apis-not-found", "bare-404"])
async def test_an_apis_listing_no_stated_scope_door_admits_the_plan_with_no_stated_scope(
    refusing: tuple[int, dict[str, Any]], caplog: pytest.LogCaptureFixture
) -> None:
    """An apis deployed before the stated-scope route, or rolled back past it, answers
    the route 404 and lists no `stated_scope` door in Search's capabilities document.
    That reads as no reading held: each turn's plan is admitted as a turn with no words
    of the user's is, on the pin the planner chose. The process's reader reads the
    document once while its answer holds, and warns once."""
    reader = _Reader(refusing=refusing, doors=("options",))
    tools = {
        spec.name: spec
        for spec in planner_tool_specs(surface=_Search(), stated_scope=reader.reader())
    }
    turns = [_Context(), _Context()]
    token = current_user_message.set(_G7)
    try:
        results = []
        for ctx in turns:
            _show(ctx, _block("africa_bilateral"))
            results.append(
                await call_planner_tool(
                    tools["propose_plan"],
                    question=_G7,
                    answer_mode="series",
                    decisions=[],
                    evidence_requirements=[
                        _dataset("aid", _spec("africa_bilateral"), "net ODA", recipient="Africa")
                    ],
                    tool_context=ctx,
                )
            )
    finally:
        current_user_message.reset(token)

    assert [result["outcome"] for result in results] == [Outcome.OK.value] * 2, results
    for ctx in turns:
        assert _pins(ctx) == {"aid": _spec("africa_bilateral")}
        assert ctx.state[STATED_SCOPE_STATE_KEY] is None
    assert len(reader.calls) == 2
    assert reader.capability_reads == 1
    warned = [
        r
        for r in caplog.records
        if r.name == "planner.stated_scope" and r.levelno == logging.WARNING
    ]
    assert len(warned) == 1


@pytest.mark.parametrize(
    ("refusing", "doors", "down", "code"),
    [
        pytest.param(
            _HTML_404, None, False, ProblemCode.STATED_SCOPE_UNREADABLE, id="wrong-address"
        ),
        pytest.param(
            _NOT_FOUND,
            ("options", "stated_scope"),
            False,
            ProblemCode.STATED_SCOPE_UNREADABLE,
            id="door-listed",
        ),
        pytest.param(
            _NOT_FOUND,
            ("options",),
            True,
            ProblemCode.STATED_SCOPE_UNAVAILABLE,
            id="capabilities-unavailable",
        ),
    ],
)
async def test_a_404_the_capabilities_document_does_not_explain_ends_the_entry(
    refusing: tuple[int, dict[str, Any]],
    doors: Sequence[str] | None,
    down: bool,
    code: ProblemCode,
) -> None:
    """A 404 reads as the route not served only on Search's own word: where the
    capabilities document cannot be read, as at a wrong or retired address, or lists
    the door, the entry ends on `STATED_SCOPE_UNREADABLE`, and where the document's
    read fails in a way that can pass, on `STATED_SCOPE_UNAVAILABLE`. No plan ignores
    the scope the user stated."""
    ctx = _Context()
    _show(ctx, _block("africa_bilateral"))
    reader = _Reader(refusing=refusing, doors=doors, capabilities_down=down)

    result = await _propose(
        ctx,
        [_dataset("aid", _spec("africa_bilateral"), "net ODA", recipient="Africa")],
        words=_G7,
        reader=reader,
    )

    assert result["code"] == code.value, result
    assert result["detail"]["status"] == (503 if down else 404)
    assert reader.capability_reads >= 1 if down else reader.capability_reads == 1
    assert load_plan(ctx.state) is None


async def test_a_route_gone_mid_turn_keeps_the_reading_held_and_admits_the_plan() -> None:
    """An apis rolled back past the route mid-turn keeps the reading of the turn's
    words held earlier, which no proposal reads again, so the plan is admitted on the
    first proposal against the scope the words state."""
    ctx = _Context()
    _show(ctx, _block("africa_bilateral"))
    token = current_user_message.set(_G7)
    try:
        assert await read_stated_scope(ctx.state, _Reader().reader()) is None
    finally:
        current_user_message.reset(token)
    held = stated_scope_record(ctx.state)
    gone = _Reader(refusing=_NOT_FOUND, doors=("options",))

    result = await _propose(
        ctx,
        [_dataset("aid", _spec("africa_bilateral"), "net ODA", recipient="Africa")],
        words=_G7,
        reader=gone,
    )

    assert result["outcome"] == Outcome.OK.value, result
    assert gone.calls == []
    assert held is not None
    assert stated_scope_record(ctx.state) == held


@pytest.mark.parametrize(
    "refusing",
    [
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
    assert FAILURE_CLASS[result["code"]] is FailureClass.SYSTEM_FAULT
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


@pytest.mark.parametrize("declares", [None, _BILATERAL], ids=["bare", "declared"])
async def test_a_row_the_user_switched_reads_no_line(declares: str | None) -> None:
    """The user switched G7 aid to Africa to bilateral flows on the plan card, and the
    words ask for imputed multilateral aid. The row reads the user's choice: no line
    that the words name another option, or that bilateral is not named."""
    ctx = _Context()
    switched = _block("africa_switched_bilateral")
    _show(ctx, switched)
    requirement = _dataset(
        "aid", switched["spec_id"], "net ODA", recipient="Africa", declares=declares
    )

    result = await _propose(ctx, [requirement], words=_G7, surface=_Search(switched))

    assert result["outcome"] == Outcome.OK.value, result
    assert _notes(ctx) == {}


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
    # The repair names the field a declaration is made in.
    assert "`scope_options`" in result["message"]
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
        pytest.param(True, ProblemCode.STATED_SCOPE_UNAVAILABLE, id="route-unavailable"),
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


@pytest.mark.parametrize("declares", [None, _CONTRIBUTIONS], ids=["bare", "declared"])
async def test_bilateral_plus_multilateral_words_name_dac1s_own_total(
    declares: str | None,
) -> None:
    """ "US bilateral plus multilateral ODA" on DAC1 total ODA: Search reads words
    summing the two lines as the dataset's own summing option, which on DAC1 is the
    total it is counted on, so the pin stands with no ask, no compile and no line on
    its row, bare or declaring that option as the brief asks."""
    ctx = _Context()
    total = dac1_with_contributions()
    _show(ctx, total)
    surface = _Search(total)

    result = await _propose(
        ctx,
        [_dataset("oda", total["spec_id"], _SUMMED, donor="United States", declares=declares)],
        words=_SUMMED,
        surface=surface,
    )

    assert result["outcome"] == Outcome.OK.value, result
    assert _pins(ctx) == {"oda": total["spec_id"]}
    assert surface.calls == []
    assert _notes(ctx) == {}


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
    """The plan card's line on each requirement's provider row (`DecisionScope.line`),
    by requirement id."""
    plan = load_plan(ctx.state)
    assert plan is not None
    rows = {decision.slot: decision for decision in plan.decisions}
    return {
        requirement.id: row.rationale
        for requirement in plan.evidence_requirements
        for slot in requirement.decision_ids
        if (row := rows.get(slot)) is not None and row.scope is not None and row.scope.line
    }


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
    inclusive count and no figure is counted that way. The line is the rationale of the
    pin's own provider row, which the requirement cites."""
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
        pin = plan.requirement(requirement_id).source_spec_id  # type: ignore[union-attr]
        row = by_slot[scope_choice_slot(pin, "provider")]
        assert row.rationale == line
        assert row.slot in plan.requirement(requirement_id).decision_ids


class _NoRevision:
    """A `harness.graph.PlannerModel` that offers the plan it was given back."""

    async def revise(self, **kwargs: Any) -> dict[str, Any]:
        return {"plan": kwargs["prior_plan"], "summary": "revised"}


async def test_a_row_reading_a_line_switches_on_the_plan_card() -> None:
    """The G7 share planned with no declaration reads a line on Africa's provider row.
    Switching that row to the inclusive count the words name posts Africa's pin to
    Search on it, as a switch of any pin's own row does."""
    ctx = _Context()
    blocks = _g7_blocks()
    _show(ctx, *blocks)
    proposed = await _propose(
        ctx, _g7_share(None, None), words=_G7, surface=_Search(*blocks), answer_mode="computed"
    )
    assert proposed["outcome"] == Outcome.OK.value, proposed
    plan = load_plan(ctx.state)
    assert plan is not None
    (row,) = [
        d
        for d in plan.decisions
        if d.slot in plan.requirement("africa").decision_ids
        and d.scope is not None
        and d.scope.axis == "provider"
    ]
    assert row.rationale == _notes(ctx)["africa"]
    surface = _Search(*blocks)

    with suppress(SwitchFailedError):
        await switch_module._switched_plan(
            node_ctx(state=ctx.state),
            plan,
            choice_id=choice_id(plan, row),
            option_id=_INCLUSIVE,
            planner=cast(Any, _NoRevision()),
            surface=cast(Any, surface),
        )

    assert [call["spec_id"] for call in surface.calls] == [
        _block("africa_bilateral")["resolved"]["scope_choices"]["base"]["spec_id"]
    ]


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
        # "total" names the denominator's contributions option, and the numerator's
        # bilateral option is named outright, so neither row reads a line.
        assert _notes(ctx) == {}
    else:
        assert result["code"] == ProblemCode.STATED_SCOPE_UNATTRIBUTED.value, result
        assert f"'total' {_CONTRIBUTIONS!r}" in result["message"]


async def test_a_total_in_the_words_names_the_denominator_and_not_the_numerator() -> None:
    """ "Share of total US ODA": the words ask for a total, which names contributions
    for the denominator only. A numerator the planner also declared on contributions
    keeps the line that the words do not name it."""
    ctx = _Context()
    total = dac1_with_contributions()
    dac1 = _block("dac1")
    _show(ctx, dac1, total)
    requirements = [
        _dataset(
            "numerator",
            DAC1,
            "ODA",
            donor="United States",
            recipient="Kenya",
            declares=_CONTRIBUTIONS,
        ),
        _dataset(
            "total", total["spec_id"], "total ODA", donor="United States", declares=_CONTRIBUTIONS
        ),
        _share("numerator", "total"),
    ]

    result = await _propose(ctx, requirements, words=_US_NO_SCOPE, surface=_Search(dac1, total))

    assert result["outcome"] == Outcome.OK.value, result
    assert set(_notes(ctx)) == {"numerator"}
    assert "your words do not name" in _notes(ctx)["numerator"]


def _record(words: str, **reading: Any) -> StatedScopeRecord:
    return StatedScopeRecord(words=words, readings={words: TextReading(**reading)})


@pytest.mark.parametrize(
    ("turn", "held", "expected"),
    [
        pytest.param(
            _record("compare it with total US ODA", total=True),
            ScopeWords(mentions=(_BILATERAL,)),
            ScopeWords(total=True),
            id="a-total-alone-is-recorded",
        ),
        pytest.param(
            _record("only bilateral", mentions=(_BILATERAL,)),
            ScopeWords(total=True),
            ScopeWords(mentions=(_BILATERAL,)),
            id="an-option-alone-drops-a-stale-total",
        ),
        pytest.param(
            _record("start in 1980"),
            ScopeWords(mentions=(_BILATERAL,), total=True),
            ScopeWords(mentions=(_BILATERAL,), total=True),
            id="words-naming-neither-keep-the-held-ones",
        ),
    ],
)
def test_the_scope_words_are_those_of_the_latest_turn_naming_an_option_or_a_total(
    turn: StatedScopeRecord, held: ScopeWords, expected: ScopeWords
) -> None:
    previous = SimpleNamespace(scope_words=held)

    assert _scope_words(turn, previous) == expected  # ty: ignore[invalid-argument-type]


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
    # The planner may remove the figure and decline the part asking for it.
    assert UnsupportedOperation.UNSUPPORTED_ASK in load_optional_declines(ctx.state)


async def test_a_line_holds_through_an_amendment_whose_words_name_nothing() -> None:
    """A total declaring an option the words do not name reads so on its row. An
    amendment under words naming nothing is judged on the words of the last turn that
    named an option, so the total's row reads the same line and no other row reads
    one."""
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
    """The G7 share declared inclusive on both sides, as the words ask: no line.
    Amendments under words naming nothing are judged on the words of the first turn,
    which name inclusive: declarations kept inclusive read no line, and declarations
    changed to bilateral each read that their option is not named."""
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


_COUNT_INCLUSIVE = "Actually, count it including imputed multilateral aid"


async def test_words_naming_a_new_option_judge_an_unchanged_declaration_afresh() -> None:
    """The G7 share declared bilateral on both sides, as the words ask: no line. The
    next turn asks to count it including imputed multilateral aid and the planner
    carries the declarations: each row reads that its option is not named."""
    ctx = _Context()
    blocks = _g7_blocks()
    _show(ctx, *blocks)
    surface = _Search(*blocks)

    first = await _propose(
        ctx,
        _g7_share(_BILATERAL, _BILATERAL),
        words=_G7_BILATERAL,
        surface=surface,
        answer_mode="computed",
    )
    first_notes = _notes(ctx)
    carried = await _propose(
        ctx,
        _g7_share(_BILATERAL, _BILATERAL),
        words=_COUNT_INCLUSIVE,
        surface=surface,
        answer_mode="computed",
        amends_version=1,
        question=_G7_BILATERAL,
    )

    assert [first["outcome"], carried["outcome"]] == [Outcome.OK.value] * 2, carried
    assert first_notes == {}
    notes = _notes(ctx)
    assert sorted(notes) == ["africa", "developing"]
    assert all(line.startswith("Counted as bilateral aid only") for line in notes.values())


async def test_a_line_is_dropped_once_its_input_is_pinned_elsewhere() -> None:
    """The G7 share planned with no declaration reads a line on Africa's row, which is
    counted bilateral. An amendment under words naming nothing pins Africa's inclusive
    part directly: Africa is counted the way the words ask, so its row reads no line."""
    ctx = _Context()
    blocks = [*_g7_blocks(), _block("africa_inclusive")]
    _show(ctx, *blocks)
    surface = _Search(*blocks)

    first = await _propose(
        ctx, _g7_share(None, None), words=_G7, surface=surface, answer_mode="computed"
    )
    assert "africa" in _notes(ctx)
    repinned = [
        {**r, "source_spec_id": _spec("africa_inclusive")} if r["id"] == "africa" else r
        for r in _g7_share(None, None)
    ]
    amended = await _propose(
        ctx,
        repinned,
        words=_LATER,
        surface=surface,
        answer_mode="computed",
        amends_version=1,
        question=_G7,
    )

    assert [first["outcome"], amended["outcome"]] == [Outcome.OK.value] * 2
    assert "africa" not in _notes(ctx)


async def test_a_pin_with_no_provider_row_records_no_line() -> None:
    """G7 aid to Africa declaring bilateral under words naming nothing, on a part whose
    only other provider options Search does not publish: the pin states no provider
    row, so the plan records no line for a row the card does not show."""
    ctx = _Context()
    africa = _block("africa_bilateral")
    for choice in africa["resolved"]["scope_choices"]["choices"]:
        for option in choice["options"] if choice["axis"] == "provider" else ():
            if option["option_id"] != _BILATERAL:
                option["availability"] = {"status": "unavailable", "code": "provider_not_offered"}
    _show(ctx, africa)
    requirement = _dataset(
        "aid", africa["spec_id"], "net ODA", recipient="Africa", declares=_BILATERAL
    )

    result = await _propose(ctx, [requirement], words=_LATER, surface=_Search(africa), question=_G7)

    assert result["outcome"] == Outcome.OK.value, result
    plan = load_plan(ctx.state)
    assert plan is not None
    assert not [d for d in plan.decisions if d.scope is not None and d.scope.axis == "provider"]
    assert _notes(ctx) == {}


async def test_a_stored_plan_carrying_accept_provider_difference_still_loads() -> None:
    """A plan stored with `accept_provider_difference` set on its share, a field the
    share does not have, loads as the plan without it, so its turn amends it in place
    of planning again."""
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
    plan = load_plan(ctx.state)
    stored = json.loads(ctx.state[PLAN_STATE_KEY])
    (share,) = [r for r in stored["evidence_requirements"] if r["kind"] == "share_of"]
    share["accept_provider_difference"] = True

    assert stored_plan(json.dumps(stored), state_key=PLAN_STATE_KEY) == plan


def test_every_provider_option_the_harness_reads_can_be_declared() -> None:
    """A provider choice the harness reads holds options of its provider scopes alone
    (`harness.scope_choices.ProviderScope`), so `scope_options` admits exactly one id
    per scope and every option `_half_declared`'s repair names is declarable."""
    assert set(get_args(ProviderOptionId)) == {
        f"provider:{scope}" for scope in get_args(ProviderScope)
    }


async def test_a_line_naming_an_option_goes_once_another_figure_counts_it() -> None:
    """G7 aid to Africa and to developing countries, two series planned with no
    declaration under words asking for imputed multilateral aid: each row reads that no
    figure is counted that way. An amendment under words naming nothing pins Africa's
    inclusive part: a figure is counted that way, so neither row reads a line."""
    ctx = _Context()
    blocks = [*_g7_blocks(), _block("africa_inclusive")]
    _show(ctx, *blocks)
    surface = _Search(*blocks)
    series = _g7_share(None, None)[:2]

    first = await _propose(ctx, series, words=_G7, surface=surface)
    assert sorted(_notes(ctx)) == ["africa", "developing"]
    amended = await _propose(
        ctx,
        [
            {**r, "source_spec_id": _spec("africa_inclusive")} if r["id"] == "africa" else r
            for r in series
        ],
        words=_LATER,
        surface=surface,
        amends_version=1,
        question=_G7,
    )

    assert [first["outcome"], amended["outcome"]] == [Outcome.OK.value] * 2, amended
    assert _pins(ctx)["developing"] == _spec("g7_bilateral")
    assert _notes(ctx) == {}


def _provider_row(ctx: _Context, requirement_id: str) -> Any:
    """The provider row requirement `requirement_id` of the plan on record cites."""
    plan = load_plan(ctx.state)
    assert plan is not None
    (row,) = [
        d
        for d in plan.decisions
        if d.slot in plan.requirement(requirement_id).decision_ids
        and d.scope is not None
        and d.scope.axis == "provider"
    ]
    return row


async def test_a_carried_requirements_row_follows_a_figure_added_beside_it() -> None:
    """G7 aid to developing countries planned with no declaration under words asking
    for imputed multilateral aid: its row reads that no figure is counted that way. An
    amendment under words naming nothing adds G7 aid to Africa counted inclusive and
    leaves the developing requirement as it was: a figure is now counted that way, so
    the carried requirement's row reads no line."""
    ctx = _Context()
    blocks = [*_g7_blocks(), _block("africa_inclusive")]
    _show(ctx, *blocks)
    surface = _Search(*blocks)
    developing = _g7_share(None, None)[1]

    first = await _propose(ctx, [developing], words=_G7, surface=surface)
    line = _notes(ctx)["developing"]
    amended = await _propose(
        ctx,
        [
            _dataset(
                "africa", _spec("africa_inclusive"), "net ODA including imputed", recipient="Africa"
            )
        ],
        words=_LATER,
        surface=surface,
        amends_version=1,
        question=_G7,
    )

    assert [first["outcome"], amended["outcome"]] == [Outcome.OK.value] * 2, amended
    assert _notes(ctx) == {}
    assert _provider_row(ctx, "developing").rationale != line


async def test_words_naming_the_declared_option_clear_its_line() -> None:
    """The G7 share declared bilateral on both sides under words asking for imputed
    multilateral aid: each row reads that bilateral is not named. The user answers
    with words naming bilateral, and the planner keeps the declarations: the words
    name the option each figure is counted on, so no row reads a line."""
    ctx = _Context()
    blocks = _g7_blocks()
    _show(ctx, *blocks)
    surface = _Search(*blocks)

    first = await _propose(
        ctx, _g7_share(_BILATERAL, _BILATERAL), words=_G7, surface=surface, answer_mode="computed"
    )
    lines = _notes(ctx)
    kept = await _propose(
        ctx,
        _g7_share(_BILATERAL, _BILATERAL),
        words="bilateral aid",
        surface=surface,
        answer_mode="computed",
        amends_version=1,
        question=_G7,
    )

    assert [first["outcome"], kept["outcome"]] == [Outcome.OK.value] * 2, kept
    assert sorted(lines) == ["africa", "developing"]
    assert _notes(ctx) == {}
    assert _provider_row(ctx, "africa").rationale not in lines.values()


def _economic_activity(block: dict[str, Any]) -> dict[str, Any]:
    """A copy of `block` measuring economic activity, as GNI does, its donor still
    bound on the giving side, as DAC1 reads GNI."""
    block = copy.deepcopy(block)
    block["resolved"]["shape"]["population_type"] = {
        "ref": "EconomicActivity",
        "label": "economic activity",
    }
    return block


def _without_scope(block: dict[str, Any]) -> dict[str, Any]:
    """A copy of `block` measuring economic activity with no scope choices at all."""
    block = _economic_activity(block)
    del block["resolved"]["scope_choices"]
    return block


def _creditor_spelled(block: dict[str, Any]) -> dict[str, Any]:
    """A copy of `block`, a flow with no provider choice, its parties spelled as a
    creditor and a debtor."""
    block = copy.deepcopy(block)
    roles = {"origin": "creditor", "destination": "debtor"}
    for entity in block["resolved"]["entities"]:
        entity["role"] = roles.get(entity["role"], entity["role"])
    return block


@pytest.mark.parametrize(
    ("other", "caveats"),
    [
        pytest.param(_economic_activity, [], id="gni-on-dac1"),
        pytest.param(_without_scope, [], id="gni-with-no-scope"),
        pytest.param(_creditor_spelled, ["providers_differ"], id="flow-spelled-by-creditor"),
    ],
)
async def test_a_series_with_no_provider_choice_is_left_alone(
    other: Callable[[dict[str, Any]], dict[str, Any]], caveats: list[str]
) -> None:
    """ "Germany's bilateral ODA as a share of its GNI": the ODA declares bilateral and
    is counted as bilateral flows, and GNI, which has no provider choice, is read as it
    stands. GNI measures economic activity, so it takes no part in the providers the
    share compares, whether DAC1 binds its donor or it states no scope at all. A
    series with no provider choice that measures a flow, however its parties are
    spelled, counts providers nothing records, so the share states the difference."""
    ctx = _Context()
    oda = dac1_with_contributions()
    second = other(basis_block(TOTAL, narrow=False, active="measure_basis:gross"))
    _show(ctx, oda, second)
    surface = _Search(oda)
    requirements = [
        _dataset("oda", oda["spec_id"], "bilateral ODA", donor="Germany", declares=_BILATERAL),
        _dataset("gni", TOTAL, "GNI", donor="Germany"),
        _share("oda", "gni"),
    ]

    result = await _propose(ctx, requirements, words=_GNI, surface=surface)

    assert result["outcome"] == Outcome.OK.value, result
    assert _pins(ctx) == {"oda": _held(ctx, oda["spec_id"], _BILATERAL), "gni": TOTAL}
    plan = load_plan(ctx.state)
    assert plan is not None
    assert [d.caveat for d in plan.decisions if d.caveat is not None] == caveats


async def test_a_declaration_with_no_words_read_keeps_the_planners_sentence() -> None:
    """A requirement declaring inclusive in an entry whose words were never read, as a
    chip turn's, and read by no calculation: it is counted that way, its row
    attributes the option to the plan's wording of the input, and it reads no
    line."""
    ctx = _Context()
    blocks = _g7_blocks()
    _show(ctx, *blocks)
    requirement = _dataset(
        "aid", _spec("africa_bilateral"), "net ODA", recipient="Africa", declares=_INCLUSIVE
    )

    result = await _propose(
        ctx, [requirement], words=None, surface=_Search(*blocks), question="G7 aid to Africa"
    )

    assert result["outcome"] == Outcome.OK.value, result
    assert _pins(ctx) == {"aid": _spec("africa_inclusive")}
    row = _provider_row(ctx, "aid")
    assert row.resolved_by is ResolvedBy.PLANNER_DEFAULT
    assert not row.scope.line
    # No calculation reads it, so no alignment reason applies.
    assert _ALIGNED_REASONS["provider"] not in row.rationale
    assert _notes(ctx) == {}


@pytest.mark.parametrize("declares", [None, _BILATERAL], ids=["bare", "declared"])
async def test_words_naming_an_option_the_dataset_does_not_publish_read_so(
    declares: str | None,
) -> None:
    """G7 aid to Africa under words asking for imputed multilateral aid, on a part that
    publishes it only as unavailable and offers core contributions: the row reads that
    the dataset does not publish it and what the figure is counted as, with no offer
    to change it, and the requirement cites a caveat saying so, which the reply
    states."""
    ctx = _Context()
    africa = _block("africa_bilateral")
    for choice in africa["resolved"]["scope_choices"]["choices"]:
        for option in choice["options"] if choice["axis"] == "provider" else ():
            if option["option_id"] == _INCLUSIVE:
                option["availability"] = {"status": "unavailable", "code": "provider_not_offered"}
            if option["option_id"] == _CONTRIBUTIONS:
                option["availability"] = {"status": "available", "detail": {}}
    _show(ctx, africa)
    requirement = _dataset(
        "aid", africa["spec_id"], "net ODA", recipient="Africa", declares=declares
    )

    result = await _propose(ctx, [requirement], words=_G7, surface=_Search(africa))

    assert result["outcome"] == Outcome.OK.value, result
    assert _provider_row(ctx, "aid").scope.line
    plan = load_plan(ctx.state)
    assert plan is not None
    (caveat,) = [
        d
        for d in plan.decisions
        if d.caveat == "scope_unpublished" and d.slot in plan.requirement("aid").decision_ids
    ]
    assert caveat.chosen != caveat.options[1]


async def test_words_naming_an_option_blocked_by_another_choice_offer_it() -> None:
    """G7 aid to Africa under words asking for imputed multilateral aid, on a part where
    that option is blocked only by another choice the user can change
    (`conflicts_with`): the card can reach it, so the row reads the line offering it
    and the answer states no unpublished caveat."""
    ctx = _Context()
    africa = _block("africa_bilateral")
    for choice in africa["resolved"]["scope_choices"]["choices"]:
        for option in choice["options"] if choice["axis"] == "provider" else ():
            if option["option_id"] == _INCLUSIVE:
                option["availability"] = {"status": "unavailable", "code": "conflicts_with"}
    _show(ctx, africa)
    requirement = _dataset("aid", africa["spec_id"], "net ODA", recipient="Africa")

    result = await _propose(ctx, [requirement], words=_G7, surface=_Search(africa))

    assert result["outcome"] == Outcome.OK.value, result
    assert _provider_row(ctx, "aid").scope.line
    plan = load_plan(ctx.state)
    assert plan is not None
    assert [d.caveat for d in plan.decisions if d.caveat is not None] == []


async def test_a_planner_draft_on_the_unpublished_caveats_slot_gives_way_to_codes() -> None:
    """A proposal whose own decision takes the slot of the caveat code writes where the
    words name an option the dataset does not publish: code's caveat is written there
    and the requirement cites it."""
    ctx = _Context()
    africa = _block("africa_bilateral")
    for choice in africa["resolved"]["scope_choices"]["choices"]:
        for option in choice["options"] if choice["axis"] == "provider" else ():
            if option["option_id"] == _INCLUSIVE:
                option["availability"] = {"status": "unavailable", "code": "provider_not_offered"}
            if option["option_id"] == _CONTRIBUTIONS:
                option["availability"] = {"status": "available", "detail": {}}
    _show(ctx, africa)
    slot = "scope_unpublished_of_aid"
    requirement = _dataset(
        "aid", africa["spec_id"], "net ODA", recipient="Africa", decision_ids=[slot]
    )
    draft = {
        "slot": slot,
        "description": "Whose spending the figure counts.",
        "options": ["bilateral", "imputed"],
        "chosen": "bilateral",
        "resolved_by": "planner_default",
        "rationale": "The planner's own words.",
    }

    result = await _propose(ctx, [requirement], [draft], words=_G7, surface=_Search(africa))

    assert result["outcome"] == Outcome.OK.value, result
    plan = load_plan(ctx.state)
    assert plan is not None
    (written,) = [d for d in plan.decisions if d.slot == slot]
    assert written.caveat == "scope_unpublished"
    assert slot in plan.requirement("aid").decision_ids


# --- a declared whole ------------------------------------------------------

# Search's parts and the wholes it derives from them (`fixtures/share_wholes_through_search.json`).
_WHOLES: dict[str, Any] = json.loads((_FIXTURES / "share_wholes_through_search.json").read_text())


class _Wholes:
    """A tool surface whose `resolve_definition` answers each recorded whole request:
    the case whose `posted` identity, entries and `whole` the call posts, with its
    whole, or its refusal as data-mcp relays it, `fix` carrying its `detail`, the
    `cause` it states and `alternatives` where given. Each call's arguments are
    recorded in `calls`."""

    def __init__(
        self, *cases: Mapping[str, Any], alternatives: Sequence[tuple[str, str]] = ()
    ) -> None:
        self.cases = cases
        self.alternatives = alternatives
        self.calls: list[Mapping[str, Any]] = []

    async def list_tools(self) -> list[Any]:
        return []

    async def call_tool(
        self, *, tool: ToolId, arguments: Mapping[str, Any], on_progress: Any = None
    ) -> ToolCallResult:
        del on_progress
        assert tool.name == "resolve_definition", tool
        self.calls.append(arguments)
        (case,) = [
            case
            for case in self.cases
            if case["posted"]
            == {key: arguments.get(key) for key in ("identity", "scope_entries", "whole")}
        ]
        if "whole" in case:
            part = {"status": "definite", "spec": copy.deepcopy(case["whole"])}
            return ToolCallResult(
                is_error=False, meta={SEARCH_ENVELOPE_META_KEY: {"parts": [part]}}
            )
        refused = case["refused"]
        fix: dict[str, Any] = {"detail": refused["detail"]}
        if "cause" in refused:
            fix["cause"] = refused["cause"]["code"]
        if self.alternatives:
            fix["alternatives"] = [{"ref": r, "label": label} for r, label in self.alternatives]
        return ToolCallResult(
            is_error=True,
            structured_content={
                "code": "upstream_error",
                "retry_mode": "do_not_retry",
                "message": refused["detail"],
                "fix": fix,
                "upstream_code": refused["code"],
            },
        )


def _provider_share(
    case: str, *, donor: str, period: tuple[int, int], measure: str = "bilateral ODA"
) -> list[dict[str, Any]]:
    """The recorded part `case` as a share of its whole over every provider."""
    part = _WHOLES[case]["part"]
    return [
        _dataset("part", part["spec_id"], measure, donor=donor, period=period),
        {
            "kind": "dataset",
            "id": "total",
            "measure": "total ODA",
            "period": {"start": period[0], "end": period[1]},
            "whole_of": {"part": "part", "axis": "provider"},
        },
        _share("part", "total"),
    ]


@pytest.mark.parametrize(
    ("case", "donor", "period", "words"),
    [
        pytest.param("us_bilateral", "United States", (2010, 2023), None, id="us"),
        pytest.param("g7_bilateral", "Group of Seven", (2015, 2023), _G7_BILATERAL, id="g7"),
    ],
)
async def test_a_provider_whole_is_the_whole_search_derives_from_its_part(
    case: str, donor: str, period: tuple[int, int], words: str | None
) -> None:
    """The bilateral share of US and of G7 ODA, its whole declared over every provider:
    admission posts the part's base identity and scope entries as its block records
    them, with `whole: {axis: provider}`, and pins the whole to Search's answer, DAC1
    total ODA for the same donors. The share's caveat names the option the total counts
    beside the part's."""
    ctx = _Context()
    recorded = _WHOLES[case]
    _show(ctx, recorded["part"])
    surface = _Wholes(recorded)

    result = await _propose(
        ctx,
        _provider_share(case, donor=donor, period=period),
        words=words,
        surface=surface,  # type: ignore[arg-type]
        answer_mode="computed",
    )

    assert result["outcome"] == Outcome.OK.value, result
    (call,) = surface.calls
    scope = recorded["part"]["resolved"]["scope_choices"]
    assert call["identity"] == scope["base"]["identity"]
    assert call["scope_entries"] == scope["entries"]
    assert call["whole"] == {"axis": "provider"}
    assert _pins(ctx) == {
        "part": recorded["part"]["spec_id"],
        "total": recorded["whole"]["spec_id"],
    }
    (held,) = [pin for pin in aligned_pins(ctx.state) if pin.whole_axis == "provider"]
    assert held.origin == recorded["part"]["spec_id"]
    plan = load_plan(ctx.state)
    assert plan is not None
    (caveat,) = [d for d in plan.decisions if d.caveat == "derived_whole"]
    assert caveat.fixes is not None
    assert caveat.fixes.source_spec_id == recorded["whole"]["spec_id"]
    assert "The total counts" in caveat.rationale


async def test_a_stored_plan_with_a_sector_whole_is_planned_again() -> None:
    """A plan an earlier release stored with a whole over every sector, an axis Search
    derives no whole on, reads as no plan: the node reading it clears the slot
    (`harness.graph.state.readable_plan`), so the question is planned again rather
    than failing."""
    ctx = _Context()
    recorded = _WHOLES["us_bilateral"]
    _show(ctx, recorded["part"])
    result = await _propose(
        ctx,
        _provider_share("us_bilateral", donor="United States", period=(2010, 2023)),
        words=None,
        surface=_Wholes(recorded),  # type: ignore[arg-type]
        answer_mode="computed",
    )
    assert result["outcome"] == Outcome.OK.value, result
    stored = json.loads(ctx.state[PLAN_STATE_KEY])
    (whole,) = [r for r in stored["evidence_requirements"] if r.get("whole_of")]
    whole["whole_of"]["axis"] = "sector"
    raw = json.dumps(stored)
    node = SimpleNamespace(state={"plan": raw})

    assert stored_plan(raw, state_key=PLAN_STATE_KEY) is None
    assert readable_plan(cast(Any, node), raw) is None
    assert node.state == {"plan": None}


async def test_a_provider_whole_search_moves_to_all_recipients_says_so() -> None:
    """France's bilateral ODA, on DAC2a to developing countries, as a share of its whole
    over every provider: Search's whole counts total ODA to all recipients (DAC)
    (`derived.rebind`), so the caveat names that total and never says both sides
    count the same recipients, and the whole covers all recipients on its source."""
    ctx = _Context()
    recorded = _WHOLES["france_developing"]
    _show(ctx, recorded["part"])

    result = await _propose(
        ctx,
        _provider_share("france_developing", donor="France", period=(2019, 2019)),
        words=None,
        surface=_Wholes(recorded),  # type: ignore[arg-type]
        answer_mode="computed",
    )

    assert result["outcome"] == Outcome.OK.value, result
    plan = load_plan(ctx.state)
    assert plan is not None
    (caveat,) = [d for d in plan.decisions if d.caveat == "derived_whole"]
    assert caveat.chosen.endswith(", for all recipients (DAC)"), caveat.chosen
    assert "so both sides count the same donors and measure basis." in caveat.rationale
    whole, part = plan.requirement("total"), plan.requirement("part")
    assert isinstance(whole, DatasetRequirement)
    assert isinstance(part, DatasetRequirement)
    read = whole_on_source(
        whole,
        part,
        source_from_spec_block(recorded["whole"]),
        summed=False,
        part_source=source_from_spec_block(recorded["part"]),
    )
    assert [(e.role, e.dcid) for e in read.entities] == [
        ("donor", None),
        ("recipient", "DACAllRecipients"),
    ]


async def test_a_donor_whole_holds_the_parts_provider_option_as_search_derives_it() -> None:
    """The US share of all aid to Kenya, on DAC2a, divides by its part's whole over every
    donor: Search's published total for official donors (DAC), on the part's bilateral
    flows. It is admitted with no sum over the donor side, which Search binds to one
    total."""
    ctx = _Context()
    recorded = _WHOLES["us_to_kenya"]
    part = recorded["part"]
    _show(ctx, part)
    surface = _Wholes(recorded)
    requirements = [
        _dataset(
            "us_oda",
            part["spec_id"],
            "ODA to Kenya",
            donor="United States",
            recipient="Kenya",
            period=(2019, 2019),
        ),
        {
            "kind": "dataset",
            "id": "all_oda",
            "measure": "ODA to Kenya from all donors",
            "period": {"start": 2019, "end": 2019},
            "whole_of": {"part": "us_oda", "axis": "donor"},
        },
        _share("us_oda", "all_oda"),
    ]

    result = await _propose(
        ctx,
        requirements,
        words=None,
        surface=surface,  # type: ignore[arg-type]
        answer_mode="computed",
    )

    assert result["outcome"] == Outcome.OK.value, result
    assert surface.calls[0]["whole"] == {"axis": "donor"}
    assert _pins(ctx)["all_oda"] == recorded["whole"]["spec_id"]
    plan = load_plan(ctx.state)
    assert plan is not None
    assert plan.whole_sums == ()


@pytest.mark.parametrize(
    ("case", "axis", "alternatives", "expected"),
    [
        # US total ODA already counts every provider: an older Search refuses, stating
        # no cause, so the refusal carries none and quotes Search's words.
        pytest.param("us_total", "provider", (), None, id="no-cause"),
        # Saudi Arabia is no DAC member, so DAC countries' total cannot hold it: Search
        # states the cause.
        pytest.param("saudi_oda", "donor", (), "part_outside_whole", id="cause"),
        # The same, Search offering every donor's sum in its place.
        pytest.param(
            "saudi_oda",
            "donor",
            (("donor:members", "every donor"),),
            "part_outside_whole",
            id="cause-and-alternatives",
        ),
    ],
)
async def test_a_whole_search_refuses_is_one_underivable_refusal_with_searchs_cause(
    case: str,
    axis: str,
    alternatives: tuple[tuple[str, str], ...],
    expected: str | None,
) -> None:
    """A whole Search will not derive is refused under `SHARE_WHOLE_UNDERIVABLE`,
    carrying Search's cause, and the options it offers in place of the one refused,
    and nothing is held. A refusal stating no cause quotes Search's own words."""
    ctx = _Context()
    recorded = _WHOLES[case]
    part = recorded["part"]
    _show(ctx, part)
    surface = _Wholes(recorded, alternatives=alternatives)
    donor = "United States" if case == "us_total" else "Saudi Arabia"
    requirements = [
        _dataset("part", part["spec_id"], "ODA", donor=donor, period=(2019, 2022)),
        {
            "kind": "dataset",
            "id": "total",
            "measure": "total ODA",
            "period": {"start": 2019, "end": 2022},
            "whole_of": {"part": "part", "axis": axis},
        },
        _share("part", "total"),
    ]

    result = await _propose(
        ctx,
        requirements,
        words=None,
        surface=surface,  # type: ignore[arg-type]
        answer_mode="computed",
    )

    assert result["code"] == ProblemCode.SHARE_WHOLE_UNDERIVABLE.value, result
    assert result["detail"]["whole"] == {
        "whole_id": "total",
        "part_spec_id": part["spec_id"],
        "cause": expected,
        "alternatives": [{"ref": ref, "label": label} for ref, label in alternatives],
    }
    # One repair the planner can make: a total found by search, on the scope an
    # alternative names, or a decline.
    counted = " counted that way" if alternatives else ""
    assert f"leave it unpinned so a search finds that total{counted};" in result["message"]
    assert all(f"as {label} instead" in result["message"] for _, label in alternatives)
    stated = (
        f": {recorded['refused']['detail'].rstrip('.')}." if expected is None else f" ({expected})."
    )
    assert f"over every {axis}{stated}" in result["message"]
    assert not [pin for pin in aligned_pins(ctx.state) if pin.whole_axis == axis]
    assert load_plan(ctx.state) is None


@pytest.mark.parametrize(
    "name", [name for name, case in _WHOLES.items() if isinstance(case, dict) and "whole" in case]
)
def test_how_search_states_it_derived_a_whole_is_what_its_block_serves(name: str) -> None:
    """Every whole Search answered the `whole` request with: where it states it summed
    a donor or recipient side over its members, the block leaves that side open, and
    where it states a published total or an option, the block binds it
    (`compiler.sources.open_side_column`), so the source built from any answer for
    the whole's spec_id reads the side as Search derived it. Where Search moved a side
    beside the widened axis (`rebind`), the block binds that total there."""
    whole = _WHOLES[name]["whole"]
    resolved = whole["resolved"]
    derived = resolved["scope_choices"]["derived"]
    source = source_from_spec_block(_drop_derived(whole))

    if derived["axis"] in ("donor", "recipient"):
        opened = open_side_column(source.open_entity_axes, derived["axis"])
        assert (opened is not None) is (derived["kind"] == "sum_of_members"), name
    if derived.get("rebind"):
        axis, total = derived["rebind"].split(":")
        column = "destination" if axis == "recipient" else "origin"
        assert [e.ref for e in source.entities if e.role == column] == [total]


def _drop_derived(block: dict[str, Any]) -> dict[str, Any]:
    """`block` with no `resolved.scope_choices.derived`."""
    stated = copy.deepcopy(block)
    del stated["resolved"]["scope_choices"]["derived"]
    return stated


# Two approved plans for the US share of ODA to Kenya over every donor, as the release
# before Search's `whole` request stored them: one summing an open donor side, one
# reading official donors' published total.
_OLD_PLANS: dict[str, Any] = json.loads((_FIXTURES / "plans_before_search_wholes.json").read_text())


class _Ledger:
    """A run ledger holding `raw` as the plan of every revision."""

    def __init__(self, raw: str) -> None:
        self.raw = raw

    async def plan(self, revision: str) -> str:
        del revision
        return self.raw


@pytest.mark.parametrize(
    ("name", "across", "dcids"),
    [
        pytest.param("summed", ("donor",), [], id="summed"),
        pytest.param("published", (), ["DACOfficialDonors"], id="published"),
    ],
)
async def test_a_plan_the_previous_release_stored_loads_and_reads_its_whole(
    monkeypatch: pytest.MonkeyPatch, name: str, across: tuple[str, ...], dcids: list[str]
) -> None:
    """An approved plan the release before Search's `whole` request stored, its whole's
    source stating no derived whole: the session's plan, the run's plan and the
    approval the turn took all load it, and its whole reads the donor side as its
    source serves it, open where the plan sums it, else official donors' total."""
    raw = _OLD_PLANS[name]
    plan = stored_plan(raw, state_key=PLAN_STATE_KEY)
    assert plan is not None
    run_plan = await stored_run_plan(cast(Any, _Ledger(raw)), run_id="run", revision="r")
    assert run_plan == plan
    monkeypatch.setattr(studio_module, "plan_approved", lambda card, **_: card)
    card = SimpleNamespace(revision=plan_revision(plan))
    taken = studio_module._taken_approval(
        cast(Any, SimpleNamespace(turn_approved=[], turn_choice_baseline={})),
        cast(Any, SimpleNamespace(state={"plan": raw})),
        cast(Any, card),
    )
    assert taken is card

    whole = covered(plan.requirement("all_oda"), plan)
    assert whole.across == across
    assert [e.dcid for e in whole.entities if e.role == "donor"] == dcids


def _old_plan_awaiting_approval(state: dict[str, Any], *, derived: bool) -> str:
    """The summed plan of `_OLD_PLANS` stored in `state`, a turn's graph state,
    awaiting approval, beside the whole pin its whole is retrieved from, as the
    release before Search's `whole` request held it, or as this release holds it
    with `derived`. Its revision."""
    plan = Plan.model_validate_json(_OLD_PLANS["summed"]).model_copy(
        update={"status": PlanStatus.PENDING_APPROVAL, "approved_at": None}
    )
    state["plan"] = plan.model_dump_json()
    whole = _WHOLES["us_health_kenya"]["whole"]
    block = copy.deepcopy(whole)
    if not derived:
        del block["resolved"]["scope_choices"]["derived"]
    source = covered_source(plan, "all_oda")
    state[ALIGNED_PINS_STATE_KEY] = [
        AlignedPin(
            origin=covered_source(plan, "us"),
            option_ids=(),
            spec_id=source,
            digest={"spec_block": {**block, "spec_id": source}},
            whole_axis="donor",
        ).model_dump(mode="json")
    ]
    return plan_revision(plan)


def covered_source(plan: Plan, requirement_id: str) -> str:
    """The spec_id `plan`'s workflow retrieves `requirement_id` from."""
    assert plan.workflow is not None
    (step_id,) = [
        b.step_id for b in plan.workflow.evidence_bindings if b.requirement_id == requirement_id
    ]
    source = plan.workflow.source_of(step_id)
    assert source is not None
    return source.spec_id


@pytest.mark.parametrize(
    ("derived", "amendment"),
    [(False, None), (False, "only 2020"), (True, None)],
    ids=["old-whole-pin", "old-whole-pin-amended", "current-whole-pin"],
)
async def test_a_plan_over_a_whole_pin_the_previous_release_compiled_is_planned_again_at_approval(
    derived: bool, amendment: str | None
) -> None:
    """A plan the previous release compiled and showed, answered after the deploy,
    whose whole is retrieved from a whole pin that release compiled by editing the
    part's identity: the answer runs nothing, and the question is planned again,
    Search deriving the whole, with the amendment the user typed on the card and a
    note saying why. A plan whose whole pin states Search's derived whole is
    approved as shown."""
    ctx = node_ctx()
    revision = _old_plan_awaiting_approval(ctx.state, derived=derived)
    answer = (
        {"approved": None, "amendment": amendment}
        if amendment
        else {"approved": revision, "at": "2026-11-25T10:00:00+00:00"}
    )

    edge = approve_module.decide(ctx, answer, early_mints=None)

    if derived:
        assert edge.label == "approved"
        return
    assert (edge.label, edge.output) == (
        "amend",
        {"unreadable_plan": True, "stale_whole": True, "answer": answer},
    )
    assert ctx.state["plan"] is None
    # The planner drafts again from the question, the user's amendment kept.
    drafted: dict[str, Any] = {}

    async def planner(**kwargs: Any) -> dict[str, Any]:
        drafted.update(kwargs)
        return {"refused": {"text": "stopped here", "code": "test"}}

    await planning_module._draft(ctx, edge.output, planner=cast(Any, planner), answer=None)

    assert (drafted["prior_plan"], drafted["amendment"]) == (None, amendment)


def _sector_plan_awaiting_approval(state: dict[str, Any]) -> None:
    """The summed plan of `_OLD_PLANS` stored in `state`, a turn's graph state,
    awaiting approval with its whole taken over every sector, as the release
    before Search's `whole` request held a sector share, a shape this release's
    `Plan` rejects."""
    stored = json.loads(_OLD_PLANS["summed"])
    stored.update(status=PlanStatus.PENDING_APPROVAL.value, approved_at=None)
    (whole,) = [r for r in stored["evidence_requirements"] if r.get("whole_of")]
    whole["whole_of"]["axis"] = "sector"
    state["plan"] = json.dumps(stored)


@pytest.mark.parametrize("stored", ["sector-whole", "donor-whole"])
async def test_an_amended_plan_the_previous_release_left_pending_is_drafted_again_with_the_amendment(
    stored: str, decision: Any, requirement: Any, derived_requirement: Any
) -> None:
    """A plan the release before Search's `whole` request left awaiting approval,
    with the planner's own record of it, amended after the deploy. The plan is a
    sector share, whose stored shape this release rejects, or a donor share over a
    whole that release compiled. The real `PlannerAdapter` drafts the question
    again from the start. Its model is shown the user's amendment, and the fresh draft its
    `propose_plan` makes, stating no `amends_version`, is recorded in place of the
    old plan."""
    ctx = node_ctx()
    if stored == "sector-whole":
        _sector_plan_awaiting_approval(ctx.state)
    else:
        _old_plan_awaiting_approval(ctx.state, derived=False)
    ctx.state[PLAN_STATE_KEY] = ctx.state["plan"]
    ctx.state[ORCHESTRATOR_BRIEF_KEY] = "kenya debt"
    amendment = "only 2020"
    proposal = {
        "question": "kenya debt",
        "answer_mode": "computed",
        "decisions": [decision().model_dump(mode="json")],
        "evidence_requirements": [
            requirement().model_dump(mode="json"),
            derived_requirement().model_dump(mode="json"),
        ],
    }
    scripted = ScriptedGenai(
        [
            reply(call_part("propose_plan", proposal, id="call-1", signature=SIGNED)),
            reply(text_part("done")),
        ]
    )

    edge = approve_module.decide(ctx, {"approved": None, "amendment": amendment}, early_mints=None)
    await planning_module._draft(ctx, edge.output, planner=scripted_planner(scripted), answer=None)

    brief = "\n".join(
        part.text or ""
        for content in scripted.requests[0].content_list
        if content.role == "user"
        for part in content.parts or ()
    )
    assert amendment in brief
    drafted = load_plan(ctx.state)
    assert drafted is not None
    assert (drafted.question, drafted.version, drafted.previous_version) == ("kenya debt", 1, None)


class _WholeUnanswered:
    """A tool surface whose `resolve_definition` answers a whole request with
    `answer`, a failure other than Search refusing the whole."""

    def __init__(self, answer: ToolCallResult) -> None:
        self.answer = answer

    async def list_tools(self) -> list[Any]:
        return []

    async def call_tool(
        self, *, tool: ToolId, arguments: Mapping[str, Any], on_progress: Any = None
    ) -> ToolCallResult:
        del arguments, on_progress
        assert tool.name == "resolve_definition", tool
        return self.answer


@pytest.mark.parametrize(
    "answer",
    [
        # data-mcp refuses the call itself, as an older data-mcp without `whole` does.
        pytest.param(
            ToolCallResult(
                is_error=True,
                structured_content={
                    "code": "request_invalid",
                    "retry_mode": "do_not_retry",
                    "message": "unknown field 'whole'",
                    "fix": {"detail": "unknown field 'whole'"},
                },
            ),
            id="data-mcp-request-invalid",
        ),
        # Search answers with no part.
        pytest.param(
            ToolCallResult(is_error=False, meta={SEARCH_ENVELOPE_META_KEY: {"parts": []}}),
            id="no-part",
        ),
    ],
)
async def test_a_whole_request_search_never_refused_is_whole_refused_and_earns_no_decline(
    answer: ToolCallResult,
) -> None:
    """US bilateral ODA's whole over every provider, where data-mcp refuses the call or
    Search answers no part: nothing says Search will not derive the whole, so it is
    refused under `whole_refused`, saying nothing of Search's derivation, and earns
    no decline."""
    ctx = _Context()
    recorded = _WHOLES["us_bilateral"]
    _show(ctx, recorded["part"])

    result = await _propose(
        ctx,
        _provider_share("us_bilateral", donor="United States", period=(2010, 2023)),
        words=None,
        surface=_WholeUnanswered(answer),  # type: ignore[arg-type]
        answer_mode="computed",
    )

    assert result["code"] == ProblemCode.SHARE_WHOLE_UNDERIVABLE.value, result
    assert result["detail"]["whole"]["cause"] == "whole_refused"
    assert "Search does not derive" not in result["message"]
    assert UnsupportedOperation.UNSUPPORTED_ASK not in load_optional_declines(ctx.state)


def _without_derived(derived: dict[str, Any]) -> None:
    derived.clear()


def _of_another_part(derived: dict[str, Any]) -> None:
    derived["part"]["spec_id"] = f"spec_{'f' * 32}"


def _on_another_axis(derived: dict[str, Any]) -> None:
    derived["axis"] = "donor"


@pytest.mark.parametrize(
    ("edit", "cause"),
    [
        # A release before the `whole` request answers with no derived whole.
        pytest.param(_without_derived, "whole_refused", id="no-derived"),
        # Search compiled the part posted to another part, so the whole is not its.
        pytest.param(_of_another_part, "part_moved", id="another-part"),
        # Search derived the whole on an axis other than the one asked.
        pytest.param(_on_another_axis, "part_moved", id="another-axis"),
    ],
)
async def test_a_whole_search_answers_for_no_whole_of_the_pinned_part_is_refused_unheld(
    edit: Callable[[dict[str, Any]], None], cause: str
) -> None:
    """Search's answer for US bilateral ODA's whole over every provider, where it states
    no derived whole, or derives the whole of another part or on another axis, is not
    the whole of the pinned part: refused under `cause`, not held, and earning no
    decline."""
    ctx = _Context()
    recorded = copy.deepcopy(_WHOLES["us_bilateral"])
    scope = recorded["whole"]["resolved"]["scope_choices"]
    edit(scope["derived"])
    if not scope["derived"]:
        del scope["derived"]
    _show(ctx, recorded["part"])

    result = await _propose(
        ctx,
        _provider_share("us_bilateral", donor="United States", period=(2010, 2023)),
        words=None,
        surface=_Wholes(recorded),  # type: ignore[arg-type]
        answer_mode="computed",
    )

    assert result["code"] == ProblemCode.SHARE_WHOLE_UNDERIVABLE.value, result
    assert result["detail"]["whole"]["cause"] == cause
    assert not [pin for pin in aligned_pins(ctx.state) if pin.whole_axis == "provider"]
    # The harness's reading of Search's answer, not Search's refusal: no decline.
    assert UnsupportedOperation.UNSUPPORTED_ASK not in load_optional_declines(ctx.state)

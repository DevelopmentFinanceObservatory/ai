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
import logging
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import httpx2
import pytest
from common.live_apis import LiveApisClient
from compiler.surface import ToolCallResult, ToolSurfaceUnavailableError
from data_mcp.search_result import SEARCH_ENVELOPE_META_KEY
from harness import studio
from harness.conversation import current_user_message
from harness.planner_adapter import PlannerAdapter
from harness.sentences import FAILURE_CLASS, TERMINAL_PROPOSE_CODES, FailureClass
from harness.settings import Settings
from harness.studio_state import ConversationState
from harness.studio_store import keep_turn_records
from harness.turn.store import AgentRecord
from harness.wire import TurnRequest
from planner.plan import (
    MAX_MEASURE_CHARS,
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
    KeepQuestion,
    StatedScopeReader,
    TextReading,
    combined_refused,
    keep_question,
    read_stated_scope,
    record_keep_answer,
    record_keep_asked,
    stated_scope_record,
    stated_uncompiled,
    unattributed_refused,
)
from planner.tool_results import Outcome, ProblemCode
from planner.tools import load_pending_ask
from planner.workflow import ToolId
from submission.service import (
    STATED_SCOPE_OPTIONS,
    SourceReadings,
    _stated_label,
    _stated_unserved,
    _Unserved,
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
# A `measure` naming the narrower bilateral option beside the inclusive one, which
# Search reads as stating neither.
_MIXED = "bilateral net disbursements including imputed multilateral aid"


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
    Search's capabilities document lists `doors` as live, and answers a bare 404, as a
    wrong address does, where `doors` is `None`; each read is counted in
    `capability_reads`."""

    def __init__(
        self,
        *,
        failing: bool = False,
        refusing: tuple[int, dict[str, Any]] | None = None,
        doors: Sequence[str] | None = ("options", "stated_scope"),
    ) -> None:
        self.calls: list[list[str]] = []
        self.failing = failing
        self.refusing = refusing
        self.doors = doors
        self.capability_reads = 0

    def handler(self, request: httpx2.Request) -> httpx2.Response:
        if (request.method, request.url.path) == ("GET", "/api/search/capabilities"):
            self.capability_reads += 1
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
        hint = await read_stated_scope(ctx.state, _Reader(failing=True).reader(), measures=())
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
                        _dataset(
                            "aid", _spec("africa_bilateral"), _STATED_INCLUSIVE, recipient="Africa"
                        )
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
    ("refusing", "doors"),
    [
        pytest.param(_HTML_404, None, id="wrong-address"),
        pytest.param(_NOT_FOUND, ("options", "stated_scope"), id="door-listed"),
    ],
)
async def test_a_404_the_capabilities_document_does_not_explain_ends_the_entry(
    refusing: tuple[int, dict[str, Any]], doors: Sequence[str] | None
) -> None:
    """A 404 reads as the route not served only on Search's own word: where the
    capabilities document cannot be read, as at a wrong or retired address, or lists
    the door, the entry ends on `STATED_SCOPE_UNREADABLE` and no plan ignores the
    scope the user stated."""
    ctx = _Context()
    _show(ctx, _block("africa_bilateral"))
    reader = _Reader(refusing=refusing, doors=doors)

    result = await _propose(
        ctx,
        [_dataset("aid", _spec("africa_bilateral"), _STATED_INCLUSIVE, recipient="Africa")],
        words=_G7,
        reader=reader,
    )

    assert result["code"] == ProblemCode.STATED_SCOPE_UNREADABLE.value, result
    assert result["detail"]["status"] == 404
    assert reader.capability_reads == 1
    assert load_plan(ctx.state) is None


async def test_a_route_gone_mid_turn_keeps_the_readings_held() -> None:
    """An apis rolled back past the route mid-turn keeps the reading of the turn's
    words held earlier: a new text reads as stating nothing, and the turn's words
    still hold the plan to the scope they state."""
    ctx = _Context()
    _show(ctx, _block("africa_bilateral"))
    token = current_user_message.set(_G7)
    try:
        assert await read_stated_scope(ctx.state, _Reader().reader(), measures=()) is None
        gone = _Reader(refusing=_NOT_FOUND, doors=("options",))
        assert await read_stated_scope(ctx.state, gone.reader(), measures=["net ODA"]) is None
    finally:
        current_user_message.reset(token)

    record = stated_scope_record(ctx.state)
    assert record is not None
    assert record.readings[_G7].mentions
    assert record.readings["net ODA"] == TextReading()


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


async def test_a_measure_over_the_plans_cap_gets_the_validation_error_unread() -> None:
    """A `measure` Plan validation refuses (`MAX_MEASURE_CHARS`) is never sent to the
    route, whose own cap it could break, so the planner reads the validation error."""
    ctx = _Context()
    _show(ctx, _block("africa_bilateral"))
    reader = _Reader()
    measure = "net ODA " * 30
    assert len(measure) > MAX_MEASURE_CHARS

    result = await _propose(
        ctx,
        [_dataset("aid", _spec("africa_bilateral"), measure, recipient="Africa")],
        words=_G7,
        reader=reader,
    )

    assert result["code"] == ProblemCode.INVALID_PLAN.value, result
    assert reader.calls == [[_G7]]


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


async def test_a_total_naming_two_narrower_options_states_nothing() -> None:
    """ "What share of G7 ODA is bilateral?" with the total's `measure` "bilateral and
    multilateral ODA": the user's words state the narrower bilateral option, which a
    `measure` naming it beside another option does not state, so the total stays on
    DAC1 total ODA and only the numerator moves."""
    ctx = _Context()
    dac1 = _block("dac1")
    _show(ctx, dac1)
    surface = _Search(dac1)
    requirements = [
        _dataset("bilateral", DAC1, "bilateral ODA"),
        _dataset("total", DAC1, "bilateral and multilateral ODA"),
        _share("bilateral", "total"),
    ]

    result = await _propose(ctx, requirements, words=_G7_BILATERAL, surface=surface)

    assert result["outcome"] == Outcome.OK.value, result
    assert _pins(ctx) == {"bilateral": _held(ctx, DAC1, _BILATERAL), "total": DAC1}


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
        pytest.param(_MIXED, _MIXED, False, id="a-narrower-word-beside-it"),
    ],
)
async def test_sides_stating_the_scope_are_counted_on_it_with_no_refusal(
    numerator: str, denominator: str, accept: bool
) -> None:
    """G7 aid to Africa over G7 aid to developing countries, the user's words asking
    for imputed multilateral aid. Each side whose `measure` says so is moved to its
    DAC2a part's inclusive compile with no refusal; a side that does not is aligned
    with the side that does. `accept_provider_difference` keeps no stated side off
    the scope. A `measure` naming bilateral beside imputed multilateral aid, the
    inclusive count's own label, states the inclusive option the user's words do."""
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


async def test_a_changed_plan_after_the_omission_refusal_is_judged_again() -> None:
    """The plan for the G7 question stating imputed multilateral aid on no side is
    refused; the planner then changes a `measure`, still stating it nowhere. Only the
    refused plan proposed again unchanged is the planner's word that the scope
    qualifies none of its figures, so the changed one is refused again, and then
    admitted unchanged."""
    ctx = _Context()
    blocks = _g7_blocks()
    _show(ctx, *blocks)
    surface = _Search(*blocks)
    first_plan = _g7_share(numerator="net ODA", denominator="net ODA")
    changed = _g7_share(numerator="ODA", denominator="net ODA")

    first = await _propose(ctx, first_plan, words=_G7, surface=surface, answer_mode="computed")
    second = await _propose(ctx, changed, words=_G7, surface=surface, answer_mode="computed")
    third = await _propose(ctx, changed, words=_G7, surface=surface, answer_mode="computed")

    assert first["code"] == ProblemCode.STATED_SCOPE_UNATTRIBUTED.value, first
    assert second["code"] == ProblemCode.STATED_SCOPE_UNATTRIBUTED.value, second
    assert third["outcome"] == Outcome.OK.value, third


async def test_a_change_to_a_requirement_not_offering_the_option_draws_no_refusal_again() -> None:
    """The plan first admitted for the G7 question is refused for stating imputed
    multilateral aid nowhere, naming Africa's bilateral part, the one requirement whose
    pin offers it. The planner proposes it again with the DAC1 total's `measure`
    changed, a requirement no attribution rests on: the omission stands answered."""
    ctx = _Context()
    _show(ctx, *_g7_blocks(), _block("g7_imputed"), _block("dac1"))
    requirements, decisions = _recorded()
    first = await _propose(ctx, requirements, decisions, words=_G7, answer_mode="computed")
    changed = copy.deepcopy(requirements)
    (total,) = [r for r in changed if r["id"] == "g7_total_oda"]
    total["measure"] = "total ODA"

    again = await _propose(ctx, changed, decisions, words=_G7, answer_mode="computed")

    assert first["code"] == ProblemCode.STATED_SCOPE_UNATTRIBUTED.value, first
    assert again["outcome"] == Outcome.OK.value, again


async def test_slots_an_earlier_release_wrote_in_another_shape_are_not_read() -> None:
    """A paused turn's state outlives a deploy. The keep question, the omission
    refusals and the refused combined compiles as earlier releases wrote them, each
    under its own key, are neither misread nor fatal: the G7 proposal on DAC1 is asked
    about, though the old keep slot holds DAC1 as kept."""
    ctx = _Context()
    ctx.state.update(
        {
            "planner:stated_scope_keep": {
                "offered": [],
                "asked_call_id": None,
                "asked": [],
                "kept": [[DAC1, _INCLUSIVE]],
                "released": [],
            },
            "planner:stated_scope_unattributed": {_INCLUSIVE: [["g7_total", DAC1, "ODA"]]},
            "planner:stated_scope_combined_refused": [
                [DAC1, [_INCLUSIVE, "measure_basis:net"], "not_offered", "refused"]
            ],
        }
    )
    _show(ctx, _block("dac1"))

    result = await _propose(ctx, [_dac1_total(_DAC1_STATED)], words=_G7)

    assert result["code"] == ProblemCode.STATED_PROVIDER_DROPPED.value, result
    assert _asked_now(ctx) == ((DAC1, _INCLUSIVE),)
    assert unattributed_refused(ctx.state) == {}
    assert combined_refused(ctx.state) == {}


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


def _answered(
    ctx: _Context, *, keep: bool, pairs: tuple[tuple[str, str], ...] = ((DAC1, _INCLUSIVE),)
) -> None:
    """The keep question about `pairs` asked earlier in the turn, and the user's
    answer to it: kept as published when `keep`, else not
    (`planner.stated_scope.record_keep_answer`)."""
    record_keep_asked(ctx.state, call_id="propose-keep", pairs=pairs, proposal={})
    assert record_keep_answer(ctx.state, call_id="propose-keep", keep=keep)


def _asked_now(ctx: _Context) -> tuple[tuple[str, str], ...]:
    """The pairs the keep question now with the user asks about, after asserting
    that the proposal paused on it as a closed question under its own call's id."""
    pending = load_pending_ask(ctx.state)
    assert pending is not None, "the proposal should have paused on the keep question"
    assert pending.call_id == ctx.function_call_id
    assert pending.args["options"] == [dict(o) for o in STATED_SCOPE_OPTIONS]
    assert pending.args["candidates"] == []
    assert ctx.run_ended
    question = keep_question(ctx.state)
    assert question.asked_call_id == ctx.function_call_id
    return question.asked


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
async def test_a_pin_that_cannot_count_the_stated_scope_is_refused_then_asked_about(
    shown: tuple[str, ...], numerator: bool, serving: list[str]
) -> None:
    """G7 total ODA on DAC1, whose `measure` says "including imputed multilateral
    aid" as the user's words do: DAC1 cannot count it. Where a dataset on record
    can, the refusal lists it, once, and asks the planner to repin; the Africa part
    the plan pins for another figure, and its inclusive compile, measure that figure
    and are not listed. Proposing DAC1 again, or with nothing serving at once,
    `propose_plan` pauses on the question whether to keep it as published."""
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
    if serving:
        assert result["action"]["call"] == "propose_plan", result
        assert load_pending_ask(ctx.state) is None
        assert not ctx.run_ended
        assert keep_question(ctx.state).offered == ((DAC1, _INCLUSIVE),)
        listed = result
        result = await _propose(ctx, requirements, words=_G7)
        assert listed["detail"]["serving"] == [_spec(name) for name in serving]

    assert result["code"] == ProblemCode.STATED_PROVIDER_DROPPED.value, result
    assert result["detail"]["unserved"] == [
        {"requirement_id": "g7_total", "spec_id": DAC1, "option_id": _INCLUSIVE}
    ]
    assert "action" not in result
    assert _asked_now(ctx) == ((DAC1, _INCLUSIVE),)
    assert load_plan(ctx.state) is None


@pytest.mark.parametrize(("keep", "admitted"), [(True, True), (False, False)], ids=["kept", "not"])
async def test_the_users_keep_holds_through_an_amendment(keep: bool, admitted: bool) -> None:
    """The user answered the question: keeping DAC1 as published admits it, and an
    amendment changing the period later in the turn keeps it, the answer held in the
    turn's state (`planner.stated_scope.KEEP_STATE_KEY`). Any other answer keeps
    nothing, and DAC1 is refused with no question."""
    ctx = _Context()
    _answered(ctx, keep=keep)
    _show(ctx, _block("dac1"))
    measure = "official development assistance including imputed multilateral aid"

    first = await _propose(ctx, [_dac1_total(measure)], words=_G7)

    if not admitted:
        assert first["code"] == ProblemCode.STATED_PROVIDER_DROPPED.value, first
        assert first["action"]["call"] == "propose_plan", first
        assert load_pending_ask(ctx.state) is None
        return
    assert first["outcome"] == Outcome.OK.value, first
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
    turn opens on the state Studio builds for it (`harness.studio._turn_opening`)."""
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


_FRANCE_NEXT = "now for France"
_ASKED_AGAIN = "Actually, count it including imputed multilateral aid"


async def test_a_keep_holds_for_its_turn_only() -> None:
    """The user kept DAC1 as published in turn 1. Turn 2, "now for France", names no
    provider scope, so it states nothing and DAC1 is admitted, and the turn holds no
    answer. Turn 3, "Actually, count it including imputed multilateral aid", states
    the option again: DAC1 is asked about again."""
    first = _Context()
    _answered(first, keep=True)
    _show(first, _block("dac1"))
    kept = await _propose(first, [_dac1_total(_DAC1_STATED)], words=_G7)
    assert kept["outcome"] == Outcome.OK.value, kept

    second = _next_turn(first)
    _show(second, _block("dac1"))
    france = await _propose(second, [_dac1_total(_DAC1_STATED)], words=_FRANCE_NEXT, question=_G7)
    assert france["outcome"] == Outcome.OK.value, france
    assert keep_question(second.state) == KeepQuestion()

    third = _next_turn(second)
    _show(third, _block("dac1"))
    again = await _propose(third, [_dac1_total(_DAC1_STATED)], words=_ASKED_AGAIN, question=_G7)
    assert again["code"] == ProblemCode.STATED_PROVIDER_DROPPED.value, again
    assert _asked_now(third) == ((DAC1, _INCLUSIVE),)


async def test_a_keep_holds_for_its_own_option_only() -> None:
    """The user kept DAC1 as published for imputed multilateral aid. An amendment
    in the same turn asking for bilateral flows, which this DAC1 cannot count either,
    is asked about: the keep was for the other option."""
    ctx = _Context()
    _answered(ctx, keep=True)
    _show(ctx, _dac1_without_bilateral())
    kept = await _propose(ctx, [_dac1_total(_DAC1_STATED)], words=_G7)
    assert kept["outcome"] == Outcome.OK.value, kept

    result = await _propose(
        ctx,
        [_dac1_total("bilateral ODA")],
        words=f"{_G7}\n(then: {_G7_BILATERAL})",
        amends_version=1,
        question=_G7,
    )

    assert result["code"] == ProblemCode.STATED_PROVIDER_DROPPED.value, result
    assert _asked_now(ctx) == ((DAC1, _BILATERAL),)


async def test_an_unserved_stated_option_reads_as_words_with_the_code_in_detail() -> None:
    """The refusal's text tells the planner why the pin cannot count the option as a
    phrase; Search's availability code stays in `detail.causes`."""
    ctx = _Context()
    block = _dac1_without_bilateral()
    (choice,) = [
        c for c in block["resolved"]["scope_choices"]["choices"] if c["axis"] == "provider"
    ]
    for option in choice["options"]:
        if option["option_id"] == _BILATERAL:
            option["availability"] = {"status": "unavailable", "code": "provider_not_offered"}
    _show(ctx, block)

    result = await _propose(
        ctx, [_dac1_total("bilateral ODA")], words=_G7_BILATERAL, question=_G7_BILATERAL
    )

    assert result["code"] == ProblemCode.STATED_PROVIDER_DROPPED.value, result
    assert "provider_not_offered" not in result["message"]
    assert result["detail"]["causes"] == {"g7_total": "provider_not_offered"}


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
    assert refusal["code"] not in result["message"]
    if code is ProblemCode.STATED_PROVIDER_DROPPED:
        assert result["detail"]["unserved"] == [
            {"requirement_id": "aid", "spec_id": africa["spec_id"], "option_id": _INCLUSIVE}
        ]
        assert result["detail"]["causes"] == {"aid": refusal["code"]}
    else:
        assert result["detail"]["cause"] == _UNANSWERED


async def test_a_stated_compile_with_no_part_records_no_one_dataset() -> None:
    """G7 aid to Africa stating imputed multilateral aid, whose compile counted that
    way Search answers with no part: no refusal of Search's, so the pin is recorded as
    holding no one dataset counted that way, and refused with the question."""
    ctx = _Context()
    africa = _block("africa_bilateral")
    _show(ctx, africa)
    surface = _Search(africa, empty=True)
    requirement = _dataset("aid", africa["spec_id"], _STATED_INCLUSIVE, recipient="Africa")

    result = await _propose(ctx, [requirement], words=_G7, surface=surface)

    assert result["code"] == ProblemCode.STATED_PROVIDER_DROPPED.value, result
    assert result["detail"]["causes"] == {"aid": _NO_ONE_DATASET}
    assert stated_uncompiled(ctx.state) == {(africa["spec_id"], _INCLUSIVE): _NO_ONE_DATASET}


async def test_a_stated_compile_that_cannot_reach_search_ends_the_entry() -> None:
    """G7 aid to Africa stating imputed multilateral aid, whose compile counted that
    way cannot reach Search for good: a fault of the system, no answer about the pin.
    The entry ends (`STATED_COMPILE_UNREACHABLE`, a terminal code) naming the compile,
    nothing is recorded against the option, and the user is asked nothing."""
    ctx = _Context()
    africa = _block("africa_bilateral")
    _show(ctx, africa)
    surface = _Search(africa, unreachable_after=0)
    requirement = _dataset("aid", africa["spec_id"], _STATED_INCLUSIVE, recipient="Africa")

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
        pytest.param(True, ProblemCode.STATED_SCOPE_UNAVAILABLE, id="route-unavailable"),
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


async def test_an_answer_not_to_keep_is_not_asked_again_in_the_turn() -> None:
    """The user answered the question about DAC1 with anything but keeping it as
    published. The answer is final for the turn: the planner proposing DAC1 again is
    refused with no question, asking it to remove the figure or repin it to the
    DAC2a total on record that counts the option."""
    ctx = _Context()
    _answered(ctx, keep=False)
    _show(ctx, _block("dac1"), _block("g7_bilateral"))

    result = await _propose(ctx, [_dac1_total(_DAC1_STATED)], words=_G7)

    assert result["code"] == ProblemCode.STATED_PROVIDER_DROPPED.value, result
    assert result["action"]["call"] == "propose_plan"
    assert load_pending_ask(ctx.state) is None
    assert f"did not keep {DAC1} counted as published" in result["message"]
    assert result["detail"]["serving"] == [_spec("g7_bilateral")]
    assert result["detail"]["unserved"] == [
        {"requirement_id": "g7_total", "spec_id": DAC1, "option_id": _INCLUSIVE}
    ]


async def test_an_answer_not_to_keep_releases_a_keep_made_earlier_in_the_turn() -> None:
    """A turn holding DAC1 kept for imputed multilateral aid, then asked about it
    again and answered not to keep it: DAC1 is refused, and the plan repinned to the
    DAC2a total counted that way is admitted."""
    ctx = _Context()
    _answered(ctx, keep=True)
    _answered(ctx, keep=False)
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
    assert keep_question(ctx.state).kept == ()


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
    ctx = _Context()
    _answered(ctx, keep=True)
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
    """US aid to Kenya stating imputed multilateral aid, over a net total its share
    counts it on: one compile moves it on both axes. Where Search refuses it, the
    pin is compiled on the stated scope alone to find which move Search refuses. That
    refused too, the pin cannot count what the user asked for and is refused with the
    ask; served, the combined refusal stands, naming the share; not answered, the
    planner is asked to propose again; unreachable for good, the entry ends naming
    that compile. None of those records anything against the stated scope."""
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
        _dataset("aid", HEALTH, _STATED_INCLUSIVE, donor="United States", recipient="Kenya"),
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
        assert _asked_now(ctx) == ((HEALTH, _INCLUSIVE),)
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
    """US aid to Kenya stating imputed multilateral aid as a share of a net total:
    the context, the requirements and the aid block."""
    ctx = _Context()
    aid = provider_block(HEALTH, narrow=True, active=_BILATERAL)
    total = basis_block(TOTAL, narrow=False, active="measure_basis:net", provenance="explicit")
    _show(ctx, aid, total)
    requirements = [
        _dataset("aid", HEALTH, _STATED_INCLUSIVE, donor="United States", recipient="Kenya"),
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


def test_an_option_with_no_label_and_no_words_is_named_by_its_scope() -> None:
    """An option no table names, no pin labels and no words state is still named, by
    its provider scope in words, and never by its id."""
    label = _stated_label("provider:pooled_total", words="")

    assert label.strip()
    assert "provider:" not in label
    assert "_" not in label


def test_the_question_tells_untitled_datasets_apart() -> None:
    """Two pins with no sentence or name of Search's: the question numbers them, and
    names neither by its id."""
    unserved = [
        _Unserved(
            requirement_id=requirement_id,
            pin=pin,
            option_id=_INCLUSIVE,
            label="bilateral aid plus imputed multilateral aid",
            words="including imputed multilateral aid",
            cause="the dataset has no option counting it that way",
            code="not_offered",
        )
        for requirement_id, pin in (("first", "spec_untitled_one"), ("second", "spec_untitled_two"))
    ]

    refused = _stated_unserved(unserved, [], sources=SourceReadings.of_shown({}))

    message = refused["detail"]["ask"]["message"]
    assert "the first dataset chosen" in message
    assert "the second dataset chosen" in message
    assert "spec_untitled" not in message

"""A plan reads a year a source holds back only where the user's words license it.

Search states on a pin's coverage every run of years its sources hold back,
whatever the pin's window (`resolved.coverage.held_back`). They are a source's
partly reported years, such as CRS 2025, and the projected years continuing a
reported series, such as IDS 2026 to 2030. `propose_plan` refuses a step reading a partly reported year the user's
words do not name with `period_partial_unnamed`, and one reading a projected
year they neither name nor ask about the future for with
`period_projected_unasked` (`submission.service._held_back_years_read`).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from types import SimpleNamespace
from typing import Any, cast

import pytest
from compiler.surface import ToolSurface
from data_mcp.definition_store import canonicalize, spec_id_for
from harness.conversation import current_user_message
from harness.studio import _turn_opening
from harness.studio_state import ConversationState
from harness.studio_store import keep_turn_records
from harness.turn_records import RecordedIdentity
from harness.wire import TurnRequest
from planner.plan import (
    ALIGNED_PINS_STATE_KEY,
    ASKS_STATE_KEY,
    AlignedPin,
    AnswerMode,
    AskAnswer,
    DatasetRequirement,
    Decision,
    Entity,
    EvidenceRequirement,
    GrowthRequirement,
    Period,
    PinSource,
    current_recorded_identity,
    load_plan,
    retrieves_period,
    save_pin_sources,
    windowed_pins,
)
from planner.specs import planner_tool_specs
from planner.stated_scope import (
    LICENCE_STATE_KEY,
    STATED_SCOPE_STATE_KEY,
    StatedScopeRecord,
    TextReading,
    read_stated_scope,
)
from planner.tool_results import Outcome, ProblemCode
from translator.discovery import _PinRecords, _recorded_source

from ..decision_drafts import drafts
from ..planner_tools import call_planner_tool

ASK = "how much aid Kenya received"
WORDS = "How much aid did Kenya receive?"

_CRS_PARTIAL = {
    "kind": "partial",
    "source": "dc/oecd_crs",
    "source_name": "OECD CRS",
    "start_year": 2025,
    "end_year": 2025,
    "reported_share": {"basis": "value", "share": 0.001, "against": 2024},
}
_IDS_PROJECTED = {
    "kind": "projected",
    "source": "dc/wb_ids",
    "source_name": "World Bank IDS",
    "start_year": 2026,
    "end_year": 2030,
    "reported_share": None,
}

_FIXED = {"stat_var": "NetODA", "requested_window": {"kind": "all"}}
_FIXED_SPEC = spec_id_for(canonicalize(_FIXED))
_OPEN = {"stat_var": "DebtService", "requested_window": {"kind": "all"}}
_OPEN_SPEC = spec_id_for(canonicalize(_OPEN))
# The orchestrator's search for "2023 to 2025": Search opted into the partial
# year the search time names, so its block states 2025 as served.
_SEARCHED = {
    "stat_var": "NetODA",
    "requested_window": {
        "kind": "range",
        "start_year": 2023,
        "end_year": 2025,
        "include": ["partial"],
    },
}
_SEARCHED_SPEC = spec_id_for(canonicalize(_SEARCHED))
_IDENTITIES = (
    (_FIXED_SPEC, _FIXED),
    (_OPEN_SPEC, _OPEN),
    (_SEARCHED_SPEC, _SEARCHED),
)


def _pins(*held_back: dict[str, Any]) -> dict[str, PinSource]:
    """Search's answer for both datasets, its coverage holding `held_back`."""
    return {
        spec: PinSource(
            block={
                "spec_id": spec,
                "definition": {"identity": identity},
                "resolved": {
                    "download": {"rows": 35, "window": {"start_year": 1990, "end_year": 2030}},
                    "coverage": {"held_back": list(held_back)},
                },
            }
        )
        for spec, identity in _IDENTITIES
    }


@pytest.fixture(autouse=True)
def typed_turn() -> Iterator[None]:
    """A Studio turn whose words are `WORDS`, as `harness.studio` sets them."""
    token = current_user_message.set(WORDS)
    yield
    current_user_message.reset(token)


@pytest.fixture(autouse=True)
def on_record() -> Iterator[None]:
    """Both pins on record, as `harness.studio` sets the lookup for a turn."""
    recorded = {spec: RecordedIdentity(identity=identity) for spec, identity in _IDENTITIES}
    token = current_recorded_identity.set(recorded.get)
    yield
    current_recorded_identity.reset(token)


def _aid(
    period: Period, *, open_end: bool = False, id: str = "aid", spec: str = _FIXED_SPEC
) -> DatasetRequirement:
    return DatasetRequirement(
        id=id,
        decision_ids=("measure",),
        measure="net ODA",
        entities=(Entity(name="Kenya", role="recipient"),),
        period=period,
        period_end_open=open_end,
        source_spec_id=_OPEN_SPEC if open_end else spec,
    )


def _propose(
    requirements: list[EvidenceRequirement],
    decisions: list[Decision],
    *,
    held_back: Sequence[dict[str, Any]],
    named_years: Sequence[int] = (),
    projection: bool = False,
    state: dict[str, Any] | None = None,
    amends_version: int | None = None,
    more_pins: dict[str, PinSource] | None = None,
) -> dict[str, Any]:
    """Propose `requirements` under words Search read as naming `named_years` and
    asking about the future where `projection` says. Dataset requirements alone
    answer the ask as series; otherwise the last requirement answers it."""
    state = {} if state is None else state
    state[ASKS_STATE_KEY] = [ASK]
    state[STATED_SCOPE_STATE_KEY] = StatedScopeRecord(
        words=WORDS,
        readings={WORDS: TextReading(named_years=tuple(named_years), projection=projection)},
    ).model_dump(mode="json")
    save_pin_sources(state, {**_pins(*held_back), **(more_pins or {})})
    ctx = SimpleNamespace(state=state, function_call_id=None, run_ended=False, records=[])
    tools = {tool.name: tool for tool in planner_tool_specs()}
    series = all(isinstance(r, DatasetRequirement) for r in requirements)
    return call_planner_tool(
        tools["propose_plan"],
        question=ASK,
        answer_mode=AnswerMode.SERIES if series else AnswerMode.COMPUTED,
        decisions=drafts(decisions),
        evidence_requirements=requirements,
        removed_decisions=[],
        answered=[
            AskAnswer(
                ask=0,
                outputs=tuple(r.id for r in requirements) if series else (requirements[-1].id,),
            )
        ],
        declined=[],
        across_years=[],
        amends_version=amends_version,
        tool_context=ctx,
    )


def _gap(
    held: str,
    *,
    step: str = "aid",
    period: tuple[int, int],
    newest: int,
    change: dict[str, Any],
) -> dict[str, Any]:
    return {
        "requirement_id": step,
        "held": held,
        "period": {"start": period[0], "end": period[1]},
        "newest_complete": newest,
        "change": change,
    }


def _refused(result: dict[str, Any], code: ProblemCode, *gaps: dict[str, Any]) -> None:
    assert result["outcome"] == Outcome.INVALID_INPUT.value, result
    assert result["code"] == code.value, result
    assert result["detail"] == {"gaps": list(gaps)}


def _growth(start: int, end: int) -> GrowthRequirement:
    return GrowthRequirement(
        id="growth", decision_ids=("measure",), input_id="aid", window_start=start, window_end=end
    )


@pytest.mark.parametrize(
    ("held_back", "projection"),
    [
        pytest.param([_CRS_PARTIAL], False, id="crs-partial-year-unnamed"),
        pytest.param([_CRS_PARTIAL, _IDS_PROJECTED], True, id="projections-asked-but-not-read"),
    ],
)
def test_a_partly_reported_year_the_user_did_not_name_is_refused(
    decision: Callable[..., Decision], held_back: list[dict[str, Any]], projection: bool
) -> None:
    """CRS read 2023 to 2025, 2025 partly reported, no year named. A
    projection ask licenses 2025 only for a read reaching the projections."""
    result = _propose(
        [_aid(Period(start=2023, end=2025))],
        [decision()],
        held_back=held_back,
        projection=projection,
    )

    _refused(
        result,
        ProblemCode.PERIOD_PARTIAL_UNNAMED,
        _gap(
            "partial",
            period=(2023, 2025),
            newest=2024,
            change={"period": {"start": 2023, "end": 2024}},
        ),
    )


def test_a_partial_year_a_model_written_search_served_is_refused_and_re_pinned_without_it(
    decision: Callable[..., Decision],
) -> None:
    """With the orchestrator's search time "2023 to 2025", Search served 2025
    and says so (`served`). The user named no year, so the plan reading 2025 is
    refused, and its change reads 2023 to 2024 from a copy with no `include`."""
    state: dict[str, Any] = {}
    kwargs: dict[str, Any] = {"held_back": [{**_CRS_PARTIAL, "served": True}], "state": state}

    refused = _propose(
        [_aid(Period(start=2023, end=2025), spec=_SEARCHED_SPEC)], [decision()], **kwargs
    )

    _refused(
        refused,
        ProblemCode.PERIOD_PARTIAL_UNNAMED,
        _gap(
            "partial",
            period=(2023, 2025),
            newest=2024,
            change={"period": {"start": 2023, "end": 2024}},
        ),
    )
    changed = _propose(
        [_aid(Period(start=2023, end=2024), spec=_SEARCHED_SPEC)], [decision()], **kwargs
    )
    assert changed["outcome"] == Outcome.OK.value, changed
    copy = next(iter(windowed_pins(state).values()))
    assert copy.identity["requested_window"] == _range(2023, 2024)


@pytest.mark.parametrize(
    ("period", "held_back", "named_years", "projection"),
    [
        pytest.param(
            Period(start=2023, end=2025), [_CRS_PARTIAL], (2025,), False, id="a-named-partial-year"
        ),
        pytest.param(
            Period(start=2023, end=2025),
            [_CRS_PARTIAL],
            (2023, 2026),
            False,
            id="a-partial-year-inside-a-named-range",
        ),
        pytest.param(
            Period(start=2026, end=2026), [_IDS_PROJECTED], (), True, id="a-projection-asked"
        ),
        pytest.param(
            Period(start=2026, end=2028),
            [_IDS_PROJECTED],
            (2026, 2028),
            False,
            id="a-named-range-of-projected-years",
        ),
        pytest.param(
            Period(start=2020, end=2030),
            [_CRS_PARTIAL, _IDS_PROJECTED],
            (),
            True,
            id="a-partial-year-before-asked-projections",
        ),
        pytest.param(Period(start=2023, end=2025), [], (), False, id="nothing-held-back"),
    ],
)
def test_a_held_back_year_the_users_words_license_is_admitted(
    decision: Callable[..., Decision],
    period: Period,
    held_back: list[dict[str, Any]],
    named_years: tuple[int, ...],
    projection: bool,
) -> None:
    result = _propose(
        [_aid(period)],
        [decision()],
        held_back=held_back,
        named_years=named_years,
        projection=projection,
    )

    assert result["outcome"] == Outcome.OK.value, result


def test_a_projected_year_read_with_no_projection_asked_is_refused(
    decision: Callable[..., Decision],
) -> None:
    """IDS read in 2026, projected 2026 to 2030, nothing asked or named."""
    result = _propose(
        [_aid(Period(start=2026, end=2026))], [decision()], held_back=[_IDS_PROJECTED]
    )

    _refused(
        result,
        ProblemCode.PERIOD_PROJECTED_UNASKED,
        _gap(
            "projected",
            period=(2026, 2026),
            newest=2025,
            change={"period": {"start": 2025, "end": 2025}},
        ),
    )


def test_the_newest_complete_year_precedes_every_held_back_run(
    decision: Callable[..., Decision],
) -> None:
    """A named partial 2025 is read and the unasked projected 2026 refused. The
    change ends at 2024, before both runs."""
    result = _propose(
        [_aid(Period(start=2020, end=2026))],
        [decision()],
        held_back=[_CRS_PARTIAL, _IDS_PROJECTED],
        named_years=(2020, 2025),
    )

    _refused(
        result,
        ProblemCode.PERIOD_PROJECTED_UNASKED,
        _gap(
            "projected",
            period=(2020, 2026),
            newest=2024,
            change={"period": {"start": 2020, "end": 2024}},
        ),
    )


def test_one_refusal_names_every_step_reading_an_unlicensed_year(
    decision: Callable[..., Decision],
) -> None:
    result = _propose(
        [
            _aid(Period(start=2023, end=2025)),
            _aid(Period(start=2026, end=2027), id="debt", spec=_OPEN_SPEC),
        ],
        [decision()],
        held_back=[_CRS_PARTIAL, _IDS_PROJECTED],
    )

    _refused(
        result,
        ProblemCode.PERIOD_PARTIAL_UNNAMED,
        _gap(
            "partial",
            period=(2023, 2025),
            newest=2024,
            change={"period": {"start": 2023, "end": 2024}},
        ),
        _gap(
            "projected",
            step="debt",
            period=(2026, 2027),
            newest=2024,
            change={"period": {"start": 2024, "end": 2024}},
        ),
    )


@pytest.mark.parametrize("open_end", [False, True], ids=["a-fixed-read", "an-open-read"])
def test_a_growth_ending_on_a_partial_year_inside_the_named_range_is_refused(
    decision: Callable[..., Decision], open_end: bool
) -> None:
    """With 2024 and 2028 named and the future asked about, the read of 2024 to
    2028 is admitted, since 2025 lies inside the named range, but a growth from
    2024 to 2025 compares at a partly reported year the user did not name."""
    result = _propose(
        [_aid(Period(start=2024, end=2028), open_end=open_end), _growth(2024, 2025)],
        [decision()],
        held_back=[_CRS_PARTIAL],
        named_years=(2024, 2028),
        projection=True,
    )

    _refused(
        result,
        ProblemCode.PERIOD_PARTIAL_UNNAMED,
        _gap(
            "partial",
            step="growth",
            period=(2024, 2025),
            newest=2024,
            change={"window_start": 2023, "window_end": 2024},
        ),
    )


@pytest.mark.parametrize(
    ("named_years", "admitted"),
    [
        pytest.param((2015,), False, id="one-named-year-before-it"),
        pytest.param((2015, 2025), True, id="the-year-ends-the-named-range"),
    ],
)
def test_a_partial_year_is_licensed_only_between_the_named_years(
    decision: Callable[..., Decision], named_years: tuple[int, ...], admitted: bool
) -> None:
    """ "Kenya ODA since 2015" names 2015 alone, so a read of 2015 to 2025 does not
    reach CRS's partly reported 2025 on the user's words; "2015 to 2025" does."""
    result = _propose(
        [_aid(Period(start=2015, end=2025))],
        [decision()],
        held_back=[_CRS_PARTIAL],
        named_years=named_years,
    )

    if admitted:
        assert result["outcome"] == Outcome.OK.value, result
        return
    _refused(
        result,
        ProblemCode.PERIOD_PARTIAL_UNNAMED,
        _gap(
            "partial",
            period=(2015, 2025),
            newest=2024,
            change={"period": {"start": 2015, "end": 2024}},
        ),
    )


@pytest.mark.parametrize(
    ("named_years", "admitted"),
    [
        pytest.param((2024,), False, id="a-named-year-before-the-run"),
        pytest.param((2025,), True, id="the-runs-first-year-named"),
    ],
)
@pytest.mark.parametrize("open_end", [False, True], ids=["a-fixed-read", "an-open-read"])
def test_a_growth_and_its_read_license_a_projected_year_by_one_rule(
    decision: Callable[..., Decision],
    open_end: bool,
    named_years: tuple[int, ...],
    admitted: bool,
) -> None:
    """IDS projects from 2025, a year already past. Naming 2025 licenses its
    projection for the dataset read and the growth ending there alike; naming
    2024 alone licenses it for neither."""
    result = _propose(
        [_aid(Period(start=2020, end=2025), open_end=open_end), _growth(2020, 2025)],
        [decision()],
        held_back=[{**_IDS_PROJECTED, "start_year": 2025}],
        named_years=named_years,
    )

    if admitted:
        assert result["outcome"] == Outcome.OK.value, result
        return
    assert result["code"] == ProblemCode.PERIOD_PROJECTED_UNASKED.value, result
    steps = [(gap["requirement_id"], gap["held"]) for gap in result["detail"]["gaps"]]
    assert steps == [*([] if open_end else [("aid", "projected")]), ("growth", "projected")]


def _retrieved_window(
    requirement: DatasetRequirement,
    decision: Decision,
    *,
    held_back: Sequence[dict[str, Any]],
    named_years: Sequence[int] = (),
    projection: bool = False,
) -> tuple[dict[str, object], str]:
    """The `requested_window` and spec_id the plan's retrieval of `requirement` pins."""
    state: dict[str, Any] = {}
    result = _propose(
        [requirement],
        [decision],
        held_back=held_back,
        named_years=named_years,
        projection=projection,
        state=state,
    )
    assert result["outcome"] == Outcome.OK.value, result
    copy = next(iter(windowed_pins(state).values()))
    window = copy.identity["requested_window"]
    assert spec_id_for(canonicalize(copy.identity)) == copy.spec_id
    return window, copy.spec_id


def _range(start: int, end: int, *include: str) -> dict[str, object]:
    window: dict[str, object] = {"kind": "range", "start_year": start, "end_year": end}
    if include:
        window["include"] = list(include)
    return window


@pytest.mark.parametrize(
    ("period", "held_back", "named_years", "projection", "include"),
    [
        pytest.param(
            Period(start=2025, end=2025),
            [_CRS_PARTIAL],
            (2025,),
            False,
            ("partial",),
            id="a-named-partial-year",
        ),
        pytest.param(
            Period(start=2026, end=2028),
            [_IDS_PROJECTED],
            (),
            True,
            ("projected",),
            id="a-projection-asked",
        ),
        pytest.param(
            Period(start=2024, end=2027),
            [_CRS_PARTIAL, _IDS_PROJECTED],
            (2024, 2027),
            False,
            ("partial", "projected"),
            id="both-classes-named-in-one-range",
        ),
        pytest.param(
            Period(start=2024, end=2027),
            [_CRS_PARTIAL, _IDS_PROJECTED],
            (),
            True,
            ("partial", "projected"),
            id="a-partial-year-before-asked-projections",
        ),
        pytest.param(
            Period(start=2020, end=2024),
            [_CRS_PARTIAL],
            (2025,),
            False,
            (),
            id="a-period-ending-before-the-held-back-year",
        ),
        pytest.param(Period(start=2020, end=2025), [], (2025,), False, (), id="nothing-held-back"),
    ],
)
def test_a_retrieval_over_a_licensed_held_back_year_opts_in(
    decision: Callable[..., Decision],
    period: Period,
    held_back: list[dict[str, Any]],
    named_years: tuple[int, ...],
    projection: bool,
    include: tuple[str, ...],
) -> None:
    window, spec_id = _retrieved_window(
        _aid(period),
        decision(),
        held_back=held_back,
        named_years=named_years,
        projection=projection,
    )

    assert window == _range(period.start, period.end, *include)
    base = {**_FIXED, "requested_window": _range(period.start, period.end)}
    unmarked = spec_id_for(canonicalize(base))
    assert (spec_id == unmarked) is (not include)


def test_the_opted_in_window_hashes_with_its_include_in_the_canonical_bytes() -> None:
    """The canonical window is spec-kit's `window-range-include-partial` form."""
    window = _range(2025, 2025, "partial")

    assert (
        b'"requested_window":{"end_year":2025,"include":["partial"],"kind":"range"'
        in canonicalize({**_FIXED, "requested_window": window})
    )


def test_an_open_window_never_opts_in(decision: Callable[..., Decision]) -> None:
    window, _ = _retrieved_window(
        _aid(Period(start=2020, end=2025), open_end=True),
        decision(),
        held_back=[_CRS_PARTIAL, _IDS_PROJECTED],
        named_years=(2025,),
        projection=True,
    )

    assert window == {"kind": "range", "start_year": 2020}


def test_retrieves_period_compares_include(decision: Callable[..., Decision]) -> None:
    requirement = _aid(Period(start=2025, end=2025))

    assert retrieves_period(
        _range(2025, 2025, "partial"), requirement, first_year=None, include=("partial",)
    )
    assert not retrieves_period(
        _range(2025, 2025), requirement, first_year=None, include=("partial",)
    )
    assert not retrieves_period(_range(2025, 2025, "partial"), requirement, first_year=None)


def test_a_windowed_copy_restated_in_an_amendment_keeps_its_include(
    decision: Callable[..., Decision],
) -> None:
    state: dict[str, Any] = {}
    kwargs: dict[str, Any] = {"held_back": [_CRS_PARTIAL], "named_years": (2025,), "state": state}
    first = _propose([_aid(Period(start=2025, end=2025))], [decision()], **kwargs)
    assert first["outcome"] == Outcome.OK.value, first
    copy = next(iter(windowed_pins(state).values()))

    again = _propose(
        [_aid(Period(start=2025, end=2025), spec=copy.spec_id)],
        [],
        amends_version=1,
        **kwargs,
    )

    assert again["outcome"] == Outcome.OK.value, again
    assert {
        pin.spec_id: pin.identity["requested_window"] for pin in windowed_pins(state).values()
    } == {copy.spec_id: _range(2025, 2025, "partial")}


def test_a_windowed_copy_pinning_an_unnamed_held_back_year_is_still_refused(
    decision: Callable[..., Decision],
) -> None:
    state: dict[str, Any] = {}
    kwargs: dict[str, Any] = {"held_back": [_CRS_PARTIAL], "state": state}
    first = _propose([_aid(Period(start=2023, end=2024))], [decision()], **kwargs)
    assert first["outcome"] == Outcome.OK.value, first
    copy = next(iter(windowed_pins(state).values()))

    again = _propose(
        [_aid(Period(start=2023, end=2025), spec=copy.spec_id)],
        [],
        amends_version=1,
        **kwargs,
    )

    _refused(
        again,
        ProblemCode.PERIOD_PARTIAL_UNNAMED,
        _gap(
            "partial",
            period=(2023, 2025),
            newest=2024,
            change={"period": {"start": 2023, "end": 2024}},
        ),
    )


async def test_an_amendment_in_a_later_turn_keeps_the_licence_an_earlier_turn_set(
    decision: Callable[..., Decision],
) -> None:
    """Turn 1's words ask about projections, and its plan reads 2026 to 2028.
    The conversation keeps the licence at the turn's close (`keep_turn_records`),
    and a later turn opens on it (`harness.studio._turn_opening`), so that turn's
    amendment of turn 1's plan, its words naming no year, is admitted."""
    conversation = ConversationState(conversation_id="conv_licence")

    async def asks_projections(texts: Sequence[str]) -> list[TextReading]:
        return [TextReading(projection=True) for _ in texts]

    _, first, _ = _turn_opening(conversation, _turn_request(), None, model_name="gemini-4-hydrogen")
    assert await read_stated_scope(first, asks_projections) is None
    proposed = _propose(
        [_aid(Period(start=2026, end=2028))],
        [decision()],
        held_back=[_IDS_PROJECTED],
        projection=True,
        state=first,
    )
    assert proposed["outcome"] == Outcome.OK.value, proposed
    keep_turn_records(conversation, first)

    _, opened, _ = _turn_opening(
        conversation, _turn_request(), None, model_name="gemini-4-hydrogen"
    )
    later = {
        **{k: v for k, v in first.items() if k not in {LICENCE_STATE_KEY, STATED_SCOPE_STATE_KEY}},
        **opened,
    }
    amended = _propose(
        [_aid(Period(start=2026, end=2028))],
        [],
        held_back=[_IDS_PROJECTED],
        state=later,
        amends_version=1,
    )

    assert amended["outcome"] == Outcome.OK.value, amended


def _turn_request() -> TurnRequest:
    return TurnRequest.model_validate(
        {
            "kind": "turn.request",
            "schema_version": 1,
            "conversation_id": "conv_licence",
            "message": WORDS,
        }
    )


_COMPILED = {"stat_var": "NetODA_bilateral", "requested_window": {"kind": "all"}}
_COMPILED_SPEC = spec_id_for(canonicalize(_COMPILED))


class _ResolveCalledError(Exception):
    """Raised by `_Surface` once discovery asks Search for the pin's identity."""


class _Surface:
    """A tool surface that stops at the first call, which discovery makes only
    once it has accepted the identity on record as the requirement's window."""

    async def list_tools(self) -> list[Any]:
        return []

    async def call_tool(self, **_: Any) -> Any:
        raise _ResolveCalledError


async def test_admission_and_discovery_read_a_compiled_pins_include_alike(
    decision: Callable[..., Decision],
) -> None:
    """A pin whose block is on record only as a compile (`AlignedPin`), CRS 2025
    partly reported and named: admission opts the copy into the partial year,
    and discovery reads the same `include` against the same block, so it
    accepts the copy and asks Search for it."""
    block = {
        "spec_id": _COMPILED_SPEC,
        "definition": {"identity": _COMPILED},
        "resolved": {
            "download": {"rows": 35, "window": {"start_year": 1990, "end_year": 2030}},
            "coverage": {"held_back": [_CRS_PARTIAL]},
        },
    }
    state: dict[str, Any] = {
        ALIGNED_PINS_STATE_KEY: [
            AlignedPin(
                origin=_FIXED_SPEC,
                option_ids=("bilateral",),
                spec_id=_COMPILED_SPEC,
                digest={"spec_block": block},
            ).model_dump(mode="json")
        ]
    }
    requirement = _aid(Period(start=2023, end=2025), spec=_COMPILED_SPEC)

    result = _propose(
        [requirement], [decision()], held_back=[], named_years=(2023, 2025), state=state
    )

    assert result["outcome"] == Outcome.OK.value, result
    copy = next(iter(windowed_pins(state).values()))
    assert copy.identity["requested_window"] == _range(2023, 2025, "partial")
    plan = load_plan(state)
    assert plan is not None
    pinned = next(r for r in plan.evidence_requirements if isinstance(r, DatasetRequirement))
    with pytest.raises(_ResolveCalledError):
        await _recorded_source(
            _PinRecords(state),
            plan=plan,
            requirement=pinned,
            first_year=None,
            surface=cast(ToolSurface, _Surface()),
        )

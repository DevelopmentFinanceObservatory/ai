"""`harness.wire` against the contract's own fixtures.

The fixtures in `wire_fixtures/studio/` are the contract drafts' studio payload
fixtures verbatim (`dfofrontendpayloads.md`, 2026-08-04) -- the shapes the
frontend team builds against. Three things are pinned here: every event
fixture round-trips bit-for-bit through `STUDIO_FRAME_ADAPTER` (parse, then
`model_dump(mode="json", exclude_unset=True)` reproduces the fixture
exactly); every request fixture parses through `STUDIO_REQUEST_ADAPTER`
with the docs' `session_id` accepted as the deprecated alias for
`conversation_id` (contract delta 5) and round-trips up to that one rename;
and the boundary refuses what the contract says it refuses: unknown
members, unknown discriminators, out-of-vocabulary enum values.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from harness.wire import (
    MAX_OPERATION_PHRASE_CHARS,
    MAX_SEED_COUNT,
    MAX_SEED_ENTITIES,
    MAX_SEED_LIST_ITEMS,
    MAX_SEED_OPERATIONS,
    MAX_SEED_ROLES,
    MAX_SEED_SENTENCE_CHARS,
    MAX_SEED_SPEC_BYTES,
    MAX_SEED_TEXT_CHARS,
    STUDIO_FRAME_ADAPTER,
    STUDIO_REQUEST_ADAPTER,
    ActivityStep,
    CancelRequest,
    ComponentCreated,
    ComponentRevised,
    CompositionMutation,
    CompositionUpdated,
    Done,
    ElicitationRequest,
    ErrorFrame,
    InterruptResponse,
    MessageComplete,
    MessageDelta,
    OperationAxis,
    OperationVerb,
    PlanOperation,
    PlanProposed,
    PlanRequirement,
    PlanStep,
    PlanVersioned,
    Seed,
    SpecBlock,
    SpecRef,
    Suggestions,
    Surface,
    TurnRequest,
    check_resolved_bounds,
    usable_component,
)
from pydantic import ValidationError

FIXTURES = Path(__file__).parent / "wire_fixtures" / "studio"

EVENT_FIXTURES = {
    "event-01-message-delta.json": MessageDelta,
    "event-02-message-complete.json": MessageComplete,
    "event-03-plan-proposed.json": PlanProposed,
    "event-04-plan-versioned.json": PlanVersioned,
    "event-05-activity-step-running.json": ActivityStep,
    "event-06-component-created.json": ComponentCreated,
    "event-07-component-revised.json": ComponentRevised,
    "event-08-composition-updated.json": CompositionUpdated,
    "event-09-elicitation-request.json": ElicitationRequest,
    "event-10-suggestions.json": Suggestions,
    "event-11-error.json": ErrorFrame,
    "event-12-done.json": Done,
    "event-13-elicitation-skippable.json": ElicitationRequest,
}

REQUEST_FIXTURES = {
    "request-01-turn.json": TurnRequest,
    "request-02-turn-card-scoped.json": TurnRequest,
    "request-03-interrupt-approve.json": InterruptResponse,
    "request-04-interrupt-amend.json": InterruptResponse,
    "request-05-composition-mutate.json": CompositionMutation,
    "request-06-turn-no-auto-add.json": TurnRequest,
    "request-07-interrupt-skip.json": InterruptResponse,
    "request-08-composition-add.json": CompositionMutation,
    "request-09-composition-set-card-meta.json": CompositionMutation,
    "request-10-cancel.json": CancelRequest,
    "request-11-turn-seeded.json": TurnRequest,
}


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def test_every_contract_fixture_is_covered() -> None:
    """A fixture file added to the contract without a pin here fails this test."""
    on_disk = {p.name for p in FIXTURES.glob("*.json")}
    assert on_disk == set(EVENT_FIXTURES) | set(REQUEST_FIXTURES)


@pytest.mark.parametrize("name", sorted(EVENT_FIXTURES))
def test_event_fixture_round_trips_exactly(name: str) -> None:
    raw = _load(name)

    frame = STUDIO_FRAME_ADAPTER.validate_python(raw)

    assert type(frame) is EVENT_FIXTURES[name]
    assert frame.model_dump(mode="json", exclude_unset=True) == raw


@pytest.mark.parametrize("name", sorted(REQUEST_FIXTURES))
def test_request_fixture_parses_with_session_id_as_alias(name: str) -> None:
    """The turn/interrupt fixtures carry `session_id`; v1 parses it as
    the deprecated alias of `conversation_id` (delta 5) and serialisation
    writes `conversation_id`, so their round-trip is exact up to that one
    rename. The composition.mutate fixture is delta-adjusted on disk (the
    drafts' shape carried no conversation identifier; delta 5 adds one), so
    it round-trips exactly as stored."""
    raw = _load(name)

    request = STUDIO_REQUEST_ADAPTER.validate_python(raw)

    assert type(request) is REQUEST_FIXTURES[name]
    expected = dict(raw)
    if "session_id" in expected:
        expected["conversation_id"] = expected.pop("session_id")
    assert request.model_dump(mode="json", exclude_unset=True) == expected


def test_fixtures_that_omit_optional_flags_parse_to_the_contract_defaults() -> None:
    """`exclude_unset` round-trips hide defaults, so pin them here: a turn
    adds its card to the canvas unless told otherwise, and an elicitation
    is a closed single choice unless it says it takes free text or a skip."""
    turn = STUDIO_REQUEST_ADAPTER.validate_python(_load("request-01-turn.json"))
    assert isinstance(turn, TurnRequest)
    assert turn.auto_add is True
    no_auto_add = STUDIO_REQUEST_ADAPTER.validate_python(_load("request-06-turn-no-auto-add.json"))
    assert isinstance(no_auto_add, TurnRequest)
    assert no_auto_add.auto_add is False

    closed = STUDIO_FRAME_ADAPTER.validate_python(_load("event-09-elicitation-request.json"))
    assert isinstance(closed, ElicitationRequest)
    assert closed.allow_free_text is False
    assert closed.skippable is False
    open_ended = STUDIO_FRAME_ADAPTER.validate_python(_load("event-13-elicitation-skippable.json"))
    assert isinstance(open_ended, ElicitationRequest)
    assert open_ended.allow_free_text is True
    assert open_ended.skippable is True


def test_turn_request_accepts_conversation_id_directly() -> None:
    raw = _load("request-01-turn.json")
    raw["conversation_id"] = raw.pop("session_id")

    request = STUDIO_REQUEST_ADAPTER.validate_python(raw)

    assert isinstance(request, TurnRequest)
    assert request.conversation_id == "sess_01j9zc0000"


def test_turn_request_surface_defaults_to_studio() -> None:
    """Delta 6: doc 1 requires `surface`, doc 2's fixtures omit it --
    resolved in favour of the lenient parse."""
    request = STUDIO_REQUEST_ADAPTER.validate_python(_load("request-01-turn.json"))

    assert isinstance(request, TurnRequest)
    assert request.surface is Surface.STUDIO


def test_a_request_carrying_both_id_spellings_is_refused() -> None:
    """`session_id` is an alias of `conversation_id`: once `conversation_id`
    claims the alias, a leftover `session_id` is an unknown member."""
    raw = _load("request-01-turn.json")
    raw["conversation_id"] = raw["session_id"]

    with pytest.raises(ValidationError):
        STUDIO_REQUEST_ADAPTER.validate_python(raw)


def test_an_unknown_frame_member_is_refused() -> None:
    raw = _load("event-01-message-delta.json")
    raw["surprise"] = True

    with pytest.raises(ValidationError):
        STUDIO_FRAME_ADAPTER.validate_python(raw)


_PER_HEAD = {"stat_var": "dcid:Count_Person", "label": "Total population"}
_CONSTANT_USD = {"currency": "USD", "prices": "constant", "base_year": 2023}


def _written_component(unit: object) -> dict[str, Any]:
    """A `write_component` result's `component` whose fact summary states `unit`."""
    component = _load("event-06-component-created.json")
    del component["event"], component["seq"]
    component["fact_summary"]["unit"] = unit
    return component


@pytest.mark.parametrize(
    "unit",
    [
        None,
        "US Dollar",
        _CONSTANT_USD,
        {"magnitude": "trillions"},
        {"numerator": "US Dollar", "denominator": _PER_HEAD},
        {"numerator": _CONSTANT_USD, "denominator": "Square Kilometer"},
    ],
    ids=["none", "name", "structured", "one-member", "per-stat-var", "structured-per-name"],
)
def test_a_component_is_usable_with_each_unit_the_platform_states(unit: object) -> None:
    assert usable_component(_written_component(unit))


@pytest.mark.parametrize(
    "unit",
    [
        "",
        {},
        {"currency": "USD", "price_basis": "constant"},
        {"currency": "usd"},
        {"numerator": "US Dollar"},
        {"numerator": "US Dollar", "denominator": None},
        {"numerator": "US Dollar", "denominator": _PER_HEAD, "label": "per head"},
        {"numerator": "US Dollar", "denominator": {"stat_var": "dcid:Count_Person"}},
        {"numerator": "US Dollar", "denominator": {**_PER_HEAD, "stat_var": "Count_Person"}},
        {"numerator": "US Dollar", "denominator": {"numerator": "a", "denominator": "b"}},
        _PER_HEAD,
    ],
    ids=[
        "empty-name",
        "no-member",
        "unknown-member",
        "lower-case-currency",
        "no-denominator",
        "null-side",
        "extra-member",
        "stat-var-without-a-label",
        "stat-var-without-dcid",
        "composed-side",
        "bare-stat-var",
    ],
)
def test_a_component_whose_unit_is_malformed_is_not_usable(unit: object) -> None:
    assert not usable_component(_written_component(unit))


def test_an_unknown_event_discriminator_is_refused() -> None:
    with pytest.raises(ValidationError):
        STUDIO_FRAME_ADAPTER.validate_python({"event": "message.finished", "seq": 0, "text": "x"})


def test_an_unknown_request_kind_is_refused() -> None:
    with pytest.raises(ValidationError):
        STUDIO_REQUEST_ADAPTER.validate_python(
            {"kind": "turn.start", "schema_version": 1, "session_id": "s", "message": "m"}
        )


def test_an_out_of_vocabulary_error_code_is_refused() -> None:
    raw = _load("event-11-error.json")
    raw["error"]["error"] = "catastrophic_failure"

    with pytest.raises(ValidationError):
        STUDIO_FRAME_ADAPTER.validate_python(raw)


def test_a_wrong_schema_version_is_refused() -> None:
    raw = _load("request-01-turn.json")
    raw["schema_version"] = 2

    with pytest.raises(ValidationError):
        STUDIO_REQUEST_ADAPTER.validate_python(raw)


def test_component_revised_requires_at_least_one_changed_field() -> None:
    raw = _load("event-07-component-revised.json")
    raw["changed"] = []

    with pytest.raises(ValidationError):
        STUDIO_FRAME_ADAPTER.validate_python(raw)


def test_a_mutation_op_missing_its_required_field_is_refused() -> None:
    """Delta 11/13: operations are op-discriminated -- an `add` without its
    component_id (or a `set_card_meta` without `meta`) must not parse."""
    base = {
        "kind": "composition.mutate",
        "schema_version": 1,
        "conversation_id": "conv_x",
        "operations": [{"op": "add"}],
    }
    with pytest.raises(ValidationError):
        STUDIO_REQUEST_ADAPTER.validate_python(base)
    with pytest.raises(ValidationError):
        STUDIO_REQUEST_ADAPTER.validate_python(
            {**base, "operations": [{"op": "set_card_meta", "component_id": "cmp_1"}]}
        )


def test_a_mutation_op_carrying_a_foreign_field_is_refused() -> None:
    with pytest.raises(ValidationError):
        STUDIO_REQUEST_ADAPTER.validate_python(
            {
                "kind": "composition.mutate",
                "schema_version": 1,
                "conversation_id": "conv_x",
                "operations": [{"op": "pin", "component_id": "cmp_1", "order": ["cmp_1"]}],
            }
        )


def test_a_plan_proposed_without_the_new_fields_still_validates() -> None:
    """Delta 17's three additions and `PlanStep.args` are optional: a
    plan.proposed shaped like pre-v1.2 code still parses. Backward
    compatibility is the point of an additive delta."""
    raw = _load("event-03-plan-proposed.json")
    for step in raw["steps"]:
        del step["args"]
    del raw["requirements"]
    del raw["operations"]
    del raw["revision"]

    frame = STUDIO_FRAME_ADAPTER.validate_python(raw)

    assert isinstance(frame, PlanProposed)
    assert frame.revision is None
    assert frame.requirements is None
    assert frame.operations is None
    assert all(step.args is None for step in frame.steps)


def test_studio_frame_adapter_still_discriminates_plan_proposed() -> None:
    frame = STUDIO_FRAME_ADAPTER.validate_python(_load("event-03-plan-proposed.json"))

    assert isinstance(frame, PlanProposed)
    assert frame.event == "plan.proposed"
    assert frame.revision == "rev_01j9ze2k7c"
    assert [r.id for r in frame.requirements or []] == ["req1"]
    assert [o.id for o in frame.operations or []] == ["op1"]


def test_a_plan_proposed_carrying_an_unknown_field_is_refused() -> None:
    """`extra="forbid"` on `_WireModel` is what makes "additive" mean
    something: a field the contract doesn't name still fails loudly."""
    raw = _load("event-03-plan-proposed.json")
    raw["surprise"] = True

    with pytest.raises(ValidationError):
        STUDIO_FRAME_ADAPTER.validate_python(raw)


def test_plan_requirement_and_plan_operation_refuse_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        PlanRequirement.model_validate(
            {"id": "r1", "kind": "dataset", "sentence": "x", "surprise": True}
        )
    with pytest.raises(ValidationError):
        PlanOperation.model_validate(
            {"id": "o1", "kind": "growth", "sentence": "x", "surprise": True}
        )


def test_plan_requirement_decision_ids_defaults_to_empty() -> None:
    requirement = PlanRequirement(id="r1", kind="dataset", sentence="x")

    assert requirement.decision_ids == []


def test_plan_step_args_defaults_to_none() -> None:
    step = PlanStep(id="s1", label="Fetch")

    assert step.args is None


def test_an_interrupt_payload_must_be_text() -> None:
    answer = {
        "kind": "interrupt.response",
        "schema_version": 1,
        "conversation_id": "conv_x",
        "interrupt_id": "int_1",
        "decision": "answer",
        "payload": "spec_1",
    }

    accepted = STUDIO_REQUEST_ADAPTER.validate_python(answer)
    assert isinstance(accepted, InterruptResponse)
    assert accepted.payload == "spec_1"

    # The same answer with a structured payload is refused for the payload's
    # shape, the one member that differs.
    with pytest.raises(ValidationError):
        STUDIO_REQUEST_ADAPTER.validate_python({**answer, "payload": {"spec_id": "spec_1"}})


def test_a_seeded_turn_carries_each_search_spec_and_its_operations() -> None:
    request = STUDIO_REQUEST_ADAPTER.validate_python(_load("request-11-turn-seeded.json"))

    assert isinstance(request, TurnRequest)
    assert request.seed_spec is not None
    (raw,) = request.seed_spec
    seed = Seed.model_validate(raw)
    assert seed.spec.spec_id == seed.spec.definition.spec_id
    assert [(op.verb, op.over) for op in seed.operations] == [
        (OperationVerb.RANK, OperationAxis.PLACE)
    ]


def test_a_seed_keeps_only_the_spec_id_the_identity_and_the_operations() -> None:
    """Every other member of the block the browser posts, `resolved` whole among
    them, is dropped as the seed is read."""
    raw = _load("request-11-turn-seeded.json")["seed_spec"][0]
    assert raw["spec"]["resolved"]["rendered_sentence"]

    kept = Seed.model_validate(raw).model_dump(mode="json", exclude_none=True)

    assert kept == {
        "spec": {"spec_id": raw["spec"]["spec_id"], "definition": raw["spec"]["definition"]},
        "operations": raw["operations"],
    }


def test_a_seed_posted_with_the_member_list_search_served_before_reads_the_same() -> None:
    """A tab opened before Search stopped listing members posts its block with
    `resolved.stat_vars.items`, `truncated` and a member digest; the seed reads
    it as it reads one without them."""
    raw = _load("request-11-turn-seeded.json")["seed_spec"][0]
    listed = copy.deepcopy(raw)
    listed["spec"]["resolved"]["stat_vars"] = {
        "items": [{"ref": "Count_Person", "slot_values": []}],
        "member_count": 1,
        "truncated": False,
    }
    listed["spec"]["resolved"]["shape"] = {"member_digest": "0" * 64}

    assert Seed.model_validate(listed) == Seed.model_validate(raw)


def test_a_turn_without_seeds_parses_to_none() -> None:
    request = STUDIO_REQUEST_ADAPTER.validate_python(_load("request-01-turn.json"))

    assert isinstance(request, TurnRequest)
    assert request.seed_spec is None


def _seeded_with(change: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    """The seeded request fixture with `change` applied to its one seed."""
    raw = _load("request-11-turn-seeded.json")
    change(raw["seed_spec"][0])
    return raw


def _rename_definition(seed: dict[str, Any]) -> None:
    seed["spec"]["definition"]["spec_id"] = "spec_" + "0" * 32


def _unknown_verb(seed: dict[str, Any]) -> None:
    seed["operations"][0]["verb"] = "summarise"


def _unknown_member(seed: dict[str, Any]) -> None:
    seed["resolved_stat_var_dcids"] = ["Count_Person"]


def _past_the_cap(seed: dict[str, Any]) -> None:
    seed["spec"]["trace"] = {"padding": "x" * MAX_SEED_SPEC_BYTES}


def _spec_not_an_object(seed: dict[str, Any]) -> None:
    seed["spec"] = "education aid to Kenya"


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        (_rename_definition, "the definition names"),
        (_unknown_verb, "verb"),
        (_unknown_member, "resolved_stat_var_dcids"),
        (_past_the_cap, r"past the 1,048,576-byte cap on one seed"),
        (_spec_not_an_object, "valid dictionary"),
    ],
    ids=[
        "definition-names-another-spec",
        "unknown-verb",
        "unknown-member",
        "past-the-cap",
        "spec-not-an-object",
    ],
)
def test_a_seed_the_harness_cannot_use_is_refused_on_its_own_and_the_request_stands(
    change: Callable[[dict[str, Any]], None], reason: str
) -> None:
    """The request parses; `harness.graph`'s `seed` node refuses the seed."""
    raw = _seeded_with(change)

    request = STUDIO_REQUEST_ADAPTER.validate_python(raw)

    assert isinstance(request, TurnRequest)
    assert request.seed_spec is not None
    with pytest.raises(ValidationError, match=reason):
        Seed.model_validate(request.seed_spec[0])


@pytest.mark.parametrize("count", [0, 7])
def test_seed_spec_holds_one_to_six_seeds(count: int) -> None:
    raw = _load("request-11-turn-seeded.json")
    raw["seed_spec"] = raw["seed_spec"] * count

    with pytest.raises(ValidationError):
        STUDIO_REQUEST_ADAPTER.validate_python(raw)


def test_seeds_cannot_ride_a_card_edit() -> None:
    """The edit route runs no graph, so its seeds would go unread."""
    raw = _load("request-11-turn-seeded.json")
    raw["component_ref"] = {"component_id": "cmp_1", "revision": 1, "intent": "edit"}

    with pytest.raises(ValidationError, match="cannot edit a card"):
        STUDIO_REQUEST_ADAPTER.validate_python(raw)


def test_seed_spec_takes_a_list_of_seeds_or_the_reserved_single_reference() -> None:
    """A request carrying one `{spec_id, content_version}`, the shape the
    member was reserved with, still parses; it carries no block, so
    `harness.graph`'s `seed` node reports it unused. A list is the seeds."""
    raw = _load("request-01-turn.json")
    reference = {"spec_id": "spec_" + "0" * 32, "content_version": "cv_" + "0" * 32}

    referenced = STUDIO_REQUEST_ADAPTER.validate_python({**raw, "seed_spec": reference})
    seeded = STUDIO_REQUEST_ADAPTER.validate_python(_load("request-11-turn-seeded.json"))
    absent = STUDIO_REQUEST_ADAPTER.validate_python({**raw, "seed_spec": None})

    assert isinstance(referenced, TurnRequest)
    assert referenced.seed_spec == SpecRef(**reference)
    assert isinstance(seeded, TurnRequest)
    assert isinstance(seeded.seed_spec, list)
    assert isinstance(absent, TurnRequest)
    assert absent.seed_spec is None


def test_a_reference_may_ride_a_card_edit_as_it_could_while_reserved() -> None:
    raw = _load("request-01-turn.json")
    raw["component_ref"] = {"component_id": "cmp_1", "revision": 1, "intent": "edit"}
    raw["seed_spec"] = {"spec_id": "spec_" + "0" * 32, "content_version": "cv_" + "0" * 32}

    assert isinstance(STUDIO_REQUEST_ADAPTER.validate_python(raw), TurnRequest)


@pytest.mark.parametrize(
    "seed_spec",
    [{"spec_id": "spec_" + "0" * 32}, {"spec": {}}, "spec_" + "0" * 32, ["a seed"], [None]],
    ids=["half-a-reference", "one-seed-outside-a-list", "a-string", "a-text-item", "a-null-item"],
)
def test_any_other_seed_spec_shape_is_refused(seed_spec: object) -> None:
    raw = _load("request-01-turn.json")

    with pytest.raises(ValidationError):
        STUDIO_REQUEST_ADAPTER.validate_python({**raw, "seed_spec": seed_spec})


def _resolved(seed: dict[str, Any]) -> dict[str, Any]:
    return seed["spec"]["resolved"]


def _entity(label: str = "Kenya") -> dict[str, Any]:
    return {"ref": "country/KEN", "label": label, "role": "destination"}


# A member of Search's answer past its bound refuses the answer
# (`SpecBlock`), as the seed node reads it.
_ANSWER_OVER_CAP: dict[str, tuple[Callable[[dict[str, Any]], None], str]] = {
    "sentence": (
        lambda seed: _resolved(seed).update(rendered_sentence="x" * (MAX_SEED_SENTENCE_CHARS + 1)),
        r"resolved\.rendered_sentence is 2,001 characters, past the cap of 2,000",
    ),
    "entities": (
        lambda seed: _resolved(seed).update(entities=[_entity()] * (MAX_SEED_ENTITIES + 1)),
        r"resolved\.entities holds 1,025 items, past the cap of 1,024",
    ),
    "entity-label": (
        lambda seed: _resolved(seed).update(entities=[_entity("x" * (MAX_SEED_TEXT_CHARS + 1))]),
        r"resolved\.entities\[0\]\.label is 257 characters",
    ),
    "unit-label": (
        lambda seed: _resolved(seed).update(
            facets={"unit": {"ref": "USD", "label": "x" * (MAX_SEED_TEXT_CHARS + 1)}}
        ),
        r"resolved\.facets\.unit\.label is 257 characters",
    ),
    "provenance-label": (
        lambda seed: _resolved(seed).update(
            facets={"provenance": {"label": "x" * (MAX_SEED_TEXT_CHARS + 1)}}
        ),
        r"resolved\.facets\.provenance\.label is 257 characters",
    ),
    "coverage-dimensions": (
        lambda seed: _resolved(seed).update(
            coverage={"dimensions": [{"label": "x", "count": 1}] * (MAX_SEED_LIST_ITEMS + 1)}
        ),
        r"resolved\.coverage\.dimensions holds 65 items",
    ),
    "coverage-label": (
        lambda seed: _resolved(seed).update(
            coverage={"dimensions": [{"label": "x" * (MAX_SEED_TEXT_CHARS + 1), "count": 1}]}
        ),
        r"resolved\.coverage\.dimensions\[0\]\.label is 257 characters",
    ),
    "coverage-count": (
        lambda seed: _resolved(seed).update(
            coverage={"dimensions": [{"label": "x", "count": MAX_SEED_COUNT + 1}]}
        ),
        r"resolved\.coverage\.dimensions\[0\]\.count is a number past the cap",
    ),
    "served-count": (
        lambda seed: _resolved(seed).update(stat_vars={"served_count": MAX_SEED_COUNT + 1}),
        r"resolved\.stat_vars\.served_count is a number past the cap",
    ),
    "slots": (
        lambda seed: _resolved(seed).update(slots=[{}] * (MAX_SEED_LIST_ITEMS + 1)),
        r"resolved\.slots holds 65 items",
    ),
    "slot-property": (
        lambda seed: _resolved(seed).update(
            slots=[{"key": {"property": {"ref": "x" * (MAX_SEED_TEXT_CHARS + 1)}}}]
        ),
        r"resolved\.slots\[0\]\.key\.property\.ref is 257 characters",
    ),
    "role-axis-fact": (
        lambda seed: _resolved(seed).update(
            role_axis_facts=[{"role": "x" * (MAX_SEED_TEXT_CHARS + 1), "member_count": 1}]
        ),
        r"resolved\.role_axis_facts\[0\]\.role is 257 characters",
    ),
    "scope-axis": (
        lambda seed: _resolved(seed).update(scope={"axis": "x" * (MAX_SEED_TEXT_CHARS + 1)}),
        r"resolved\.scope\.axis is 257 characters",
    ),
    "headline": (
        lambda seed: _resolved(seed).update(headline={"year": 2024, "observations": 10**30}),
        r"resolved\.headline\.observations is a number past the cap",
    ),
}

# A member of the seed past its bound refuses the seed (`Seed`).
_SEED_OVER_CAP: dict[str, tuple[Callable[[dict[str, Any]], None], str]] = {
    "window-kind": (
        lambda seed: seed["spec"]["definition"]["identity"].update(
            requested_window={"kind": "x" * (MAX_SEED_TEXT_CHARS + 1)}
        ),
        r"definition\.identity\.requested_window\.kind is 257 characters",
    ),
    "window-year": (
        lambda seed: seed["spec"]["definition"]["identity"].update(
            requested_window={"kind": "range", "start_year": "x" * 5000, "end_year": 2024}
        ),
        r"definition\.identity\.requested_window\.start_year is not a year",
    ),
    "roles": (
        lambda seed: seed["spec"]["definition"]["identity"].update(
            entities={f"role{index}": ["x"] for index in range(MAX_SEED_ROLES + 1)}
        ),
        r"definition\.identity\.entities names 17 roles, past the cap of 16",
    ),
    "role": (
        lambda seed: seed["spec"]["definition"]["identity"].update(
            entities={"x" * (MAX_SEED_TEXT_CHARS + 1): ["x"]}
        ),
        r"definition\.identity\.entities is 257 characters",
    ),
    "shape-kind": (
        lambda seed: seed["spec"]["definition"]["identity"].update(
            shape={"kind": "x" * (MAX_SEED_TEXT_CHARS + 1)}
        ),
        r"definition\.identity\.shape\.kind is 257 characters",
    ),
    "target-year-on-a-rank": (
        lambda seed: seed["operations"][0].update(target_year=2050),
        "target_year is set only on a forecast",
    ),
    "spec-id": (
        lambda seed: seed["spec"].update(spec_id="spec_" + "0" * 300),
        "spec_id",
    ),
    "phrase": (
        lambda seed: seed["operations"][0].update(phrase="x" * (MAX_OPERATION_PHRASE_CHARS + 1)),
        "phrase",
    ),
    "blank-phrase": (lambda seed: seed["operations"][0].update(phrase=""), "phrase"),
    "operations": (
        lambda seed: seed.update(operations=seed["operations"] * (MAX_SEED_OPERATIONS + 1)),
        "operations",
    ),
}


@pytest.mark.parametrize("member", sorted(_SEED_OVER_CAP))
def test_a_seed_with_a_member_past_its_bound_is_refused_whole(member: str) -> None:
    """A seed is browser input, and what the harness keeps of it goes on to
    Search and into state. Nothing is cut to fit."""
    change, reason = _SEED_OVER_CAP[member]
    raw = _seeded_with(change)

    with pytest.raises(ValidationError, match=reason):
        Seed.model_validate(raw["seed_spec"][0])


@pytest.mark.parametrize("member", sorted(_ANSWER_OVER_CAP))
def test_an_answer_with_a_member_past_its_bound_is_refused_whole(member: str) -> None:
    """Search's answer for a seed's identity goes on into state, tool results and
    the planner's brief, so each member the harness reads is held to its bound."""
    change, reason = _ANSWER_OVER_CAP[member]
    raw = _seeded_with(change)

    with pytest.raises(ValidationError, match=reason):
        SpecBlock.model_validate(raw["seed_spec"][0]["spec"])


@pytest.mark.parametrize("member", sorted(_ANSWER_OVER_CAP))
def test_a_seed_ignores_a_member_it_does_not_keep_whatever_its_size(member: str) -> None:
    """The browser's own `resolved` is dropped as the seed is read, so the size of
    its members refuses nothing."""
    change, _ = _ANSWER_OVER_CAP[member]
    raw = _seeded_with(change)

    Seed.model_validate(raw["seed_spec"][0])


def test_a_seed_at_every_bound_is_taken() -> None:
    def at_the_caps(seed: dict[str, Any]) -> None:
        _resolved(seed).update(
            rendered_sentence="x" * MAX_SEED_SENTENCE_CHARS,
            entities=[_entity("x" * MAX_SEED_TEXT_CHARS)] * MAX_SEED_ENTITIES,
        )
        seed["operations"] = [
            {"verb": "rank", "phrase": "x" * MAX_OPERATION_PHRASE_CHARS}
        ] * MAX_SEED_OPERATIONS

    raw = _seeded_with(at_the_caps)["seed_spec"][0]
    seed = Seed.model_validate(raw)

    assert len(seed.operations) == MAX_SEED_OPERATIONS
    SpecBlock.model_validate(raw["spec"])


def test_every_recorded_search_block_is_within_the_seed_bounds() -> None:
    """The bounds are for what Search serves: each Spec block recorded from the
    Search API, in either service's fixtures, passes them."""

    def blocks(value: object) -> Iterator[dict[str, Any]]:
        if isinstance(value, dict):
            if isinstance(value.get("resolved"), dict) and "spec_id" in value:
                yield value
            for member in value.values():
                yield from blocks(member)
        elif isinstance(value, list):
            for member in value:
                yield from blocks(member)

    services = Path(__file__).parents[2]
    seen = 0
    for path in sorted(services.glob("*/tests/**/*.json")):
        try:
            recorded = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        for block in blocks(recorded):
            check_resolved_bounds(block["resolved"])
            seen += 1
    assert seen >= 10


def _forecast(target_year: object) -> dict[str, Any]:
    return {"verb": "forecast", "phrase": "forecast aid to 2050", "target_year": target_year}


def test_a_seed_operation_carries_the_year_a_forecast_projects_to() -> None:
    """Search states `target_year` on a forecast; the model forbids unknown members."""
    raw = _seeded_with(lambda seed: seed["operations"].append(_forecast(2050)))

    seed = Seed.model_validate(raw["seed_spec"][0])

    assert [operation.target_year for operation in seed.operations] == [None, 2050]


@pytest.mark.parametrize("target_year", [999, 2201, "soon", "2050", 2050.0, True])
def test_a_target_year_outside_search_s_bounds_refuses_its_seed(target_year: object) -> None:
    raw = _seeded_with(lambda seed: seed["operations"].append(_forecast(target_year)))

    with pytest.raises(ValidationError, match="target_year"):
        Seed.model_validate(raw["seed_spec"][0])


def _region_members_block() -> dict[str, Any]:
    """dfo-apis#1222's spec block for health expenditure per capita over Africa's countries."""
    fixture = Path(__file__).parent / "fixtures" / "search_spec_region_members.json"
    return json.loads(fixture.read_text())["spec"]


def test_a_seed_keeps_the_set_its_places_were_asked_for_as() -> None:
    """A block whose role was asked for as a set ("African countries") hands the set
    over beside the identity, so the seed's resubmit asks Search for it again."""
    seed = Seed.model_validate({"spec": _region_members_block(), "operations": []})

    assert seed.spec.entity_groups is not None
    assert {
        role: group.model_dump(exclude_none=True) for role, group in seed.spec.entity_groups.items()
    } == {"observationAbout": {"ref": "africa", "label": "Africa", "entity_type": "Continent"}}


@pytest.mark.parametrize(
    ("group", "kept"),
    [
        (
            {"label": "all donors", "total": "DACOfficialDonors", "whole_of": ["party"]},
            {"label": "all donors", "total": "DACOfficialDonors", "whole_of": ("party",)},
        ),
        (
            {
                "label": "all donors including private",
                "total": None,
                "whole_of": ["party", "private"],
            },
            {"label": "all donors including private", "whole_of": ("party", "private")},
        ),
        ({"label": "Africa", "ref": "africa"}, {"label": "Africa", "ref": "africa"}),
        ({"label": "Africa", "whole_of": []}, {"label": "Africa"}),
    ],
    ids=["single-total", "set-of-totals", "without-the-field", "empty-field-is-absent"],
)
def test_a_seed_carries_the_kinds_a_whole_total_holds(group: object, kept: dict[str, Any]) -> None:
    block = _region_members_block()
    block["resolved"]["entity_groups"] = {"observationAbout": group}

    seed = Seed.model_validate({"spec": block, "operations": []})

    assert seed.spec.entity_groups is not None
    assert seed.spec.entity_groups["observationAbout"].model_dump(exclude_none=True) == kept


@pytest.mark.parametrize(
    "whole_of", ["party", [""], list(range(2))], ids=["a-string", "blank-kind", "not-text"]
)
def test_a_set_whose_whole_of_is_not_a_list_of_kinds_is_dropped(whole_of: object) -> None:
    block = _region_members_block()
    block["resolved"]["entity_groups"] = {
        "observationAbout": {"label": "Africa", "whole_of": whole_of}
    }

    seed = Seed.model_validate({"spec": block, "operations": []})

    assert seed.spec.entity_groups is None


@pytest.mark.parametrize(
    "group",
    [
        {"dcid": "africa", "label": "Africa"},
        {"ref": "africa", "label": "Africa", "members": ["country/KEN"]},
        {"label": "all donors", "whole_total": ["party"]},
    ],
    ids=["dcid-before-ref", "unknown-member", "near-miss-of-whole-of"],
)
def test_a_set_carrying_a_key_search_does_not_serve_refuses_the_seed(group: object) -> None:
    # A block from the Search wire that named a set's node `dcid` would otherwise
    # hand the set over by its name alone and lose its node.
    block = _region_members_block()
    block["resolved"]["entity_groups"] = {"observationAbout": group}

    with pytest.raises(ValidationError, match="unknown keys"):
        Seed.model_validate({"spec": block, "operations": []})


@pytest.mark.parametrize(
    "group",
    [
        {"label": ""},
        {"label": None, "ref": "africa"},
        {"label": "A" * 121, "ref": "africa"},
        "Africa",
    ],
    ids=["empty-label", "null-label", "label-too-long", "not-an-object"],
)
def test_a_set_search_would_not_take_is_dropped_and_the_seed_stands(group: object) -> None:
    # A set with no usable name is left out, as `compiler.sources` leaves out a
    # nameless one; the seed's spec_id and identity still hand over.
    block = _region_members_block()
    block["resolved"]["entity_groups"] = {"observationAbout": group}

    seed = Seed.model_validate({"spec": block, "operations": []})

    assert seed.spec.entity_groups is None
    assert seed.spec.definition.spec_id == block["spec_id"]


def test_a_set_on_a_role_the_identity_binds_nothing_on_is_dropped() -> None:
    block = _region_members_block()
    block["resolved"]["entity_groups"]["origin"] = {"label": "G7", "ref": "G7"}

    seed = Seed.model_validate({"spec": block, "operations": []})

    assert seed.spec.entity_groups is not None
    assert list(seed.spec.entity_groups) == ["observationAbout"]

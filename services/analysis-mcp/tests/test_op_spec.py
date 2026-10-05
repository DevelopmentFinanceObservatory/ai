"""The op-spec schema: closed verb set, keyword-only params, the pair-shaped
`frame_ref`."""

from __future__ import annotations

import pytest
from analysis_mcp.identity import op_spec_hash_for
from analysis_mcp.op_spec import (
    AggregateOpSpec,
    CategoryMatchPredicate,
    CompareOpSpec,
    FilterOpSpec,
    GrowthOpSpec,
    JoinOpSpec,
    OpSpecAdapter,
    OrderedThresholdPredicate,
    RankOpSpec,
    ShareOfOpSpec,
    Verb,
)
from common.materialisation import FrameRef
from dfo_frame_grammar.grammar import UnitRequest
from pydantic import ValidationError

SNAPSHOT_REF = {
    "spec_id": "spec_" + "a" * 32,
    "content_version": "cv_" + "b" * 32,
}
TOOLKIT_RESULT_REF = {
    "spec_id": "ops_" + "a" * 32,
    "content_version": "ocv_" + "b" * 32,
}


def test_every_verb_round_trips_through_the_adapter() -> None:
    for verb in Verb:
        payload: dict[str, object] = {"verb": verb.value, "frame_ref": SNAPSHOT_REF}
        if verb is Verb.AGGREGATE:
            payload["aggregation"] = "sum"
        elif verb is Verb.GROWTH:
            payload["window_start"] = "2020"
            payload["window_end"] = "2023"
        elif verb is Verb.RANK:
            payload["rank_by"] = ["reporter"]
        elif verb is Verb.JOIN:
            payload["right_frame_ref"] = SNAPSHOT_REF
        op_spec = OpSpecAdapter.validate_python(payload)
        assert op_spec.verb == verb
        # round-trips through JSON, not just the Python object
        dumped = op_spec.model_dump(mode="json")
        assert OpSpecAdapter.validate_python(dumped) == op_spec


def test_aggregate_requires_the_aggregation_field() -> None:
    with pytest.raises(ValidationError):
        AggregateOpSpec(frame_ref=SNAPSHOT_REF)


def test_aggregate_aggregation_is_not_schema_closed() -> None:
    """The vocabulary is closed at the engine, not the schema -- see
    `AggregateOpSpec`'s own docstring for why."""
    op_spec = AggregateOpSpec(frame_ref=SNAPSHOT_REF, aggregation="not_a_real_reduction")
    assert op_spec.aggregation == "not_a_real_reduction"


def test_aggregate_unit_defaults_to_none() -> None:
    op_spec = AggregateOpSpec(frame_ref=SNAPSHOT_REF, aggregation="sum")
    assert op_spec.unit is None


CONSTANT_EUR = {"currency": "EUR", "prices": "constant", "base_year": 2020}


@pytest.mark.parametrize(
    ("pin", "parsed"),
    [
        ("dcid:USDollar", "dcid:USDollar"),
        (CONSTANT_EUR, UnitRequest(currency="EUR", prices="constant", base_year=2020)),
        ({"currency": "USD", "prices": "current"}, UnitRequest(currency="USD", prices="current")),
        (
            UnitRequest(currency="USD", prices="current"),
            UnitRequest(currency="USD", prices="current"),
        ),
    ],
)
def test_a_unit_pin_is_a_ref_or_a_converted_unit(pin: object, parsed: object) -> None:
    """Every verb's unit pin parses a converted-unit object into the
    `UnitRequest` a converted frame's `unit` cell holds, so the engine
    compares the two by equality."""
    specs = [
        AggregateOpSpec(frame_ref=SNAPSHOT_REF, aggregation="sum", unit=pin),
        GrowthOpSpec(frame_ref=SNAPSHOT_REF, window_start="2020", window_end="2021", unit=pin),
        RankOpSpec(frame_ref=SNAPSHOT_REF, rank_by=("donor",), unit=pin),
        CompareOpSpec(frame_ref=SNAPSHOT_REF, left_unit=pin, right_unit=pin),
        ShareOfOpSpec(frame_ref=SNAPSHOT_REF, numerator_unit=pin, denominator_unit=pin),
        JoinOpSpec(
            frame_ref=SNAPSHOT_REF, right_frame_ref=SNAPSHOT_REF, left_unit=pin, right_unit=pin
        ),
    ]
    pins = [
        getattr(spec, field)
        for spec in specs
        for field in ("unit", "left_unit", "right_unit", "numerator_unit", "denominator_unit")
        if field in type(spec).model_fields
    ]
    assert pins == [parsed] * 9


def test_a_converted_pin_dumps_as_its_request_object() -> None:
    """The op-spec's JSON, and so its identity preimage, carries a converted
    pin as the request object, with no `base_year` key at current prices."""
    current = AggregateOpSpec(
        frame_ref=SNAPSHOT_REF, aggregation="sum", unit={"currency": "USD", "prices": "current"}
    )
    constant = AggregateOpSpec(frame_ref=SNAPSHOT_REF, aggregation="sum", unit=CONSTANT_EUR)

    assert current.model_dump(mode="json")["unit"] == {"currency": "USD", "prices": "current"}
    assert constant.model_dump(mode="json")["unit"] == CONSTANT_EUR
    assert OpSpecAdapter.validate_python(constant.model_dump(mode="json")) == constant


def test_a_converted_pin_and_a_ref_pin_mint_different_op_spec_hashes() -> None:
    ref = AggregateOpSpec(frame_ref=SNAPSHOT_REF, aggregation="sum", unit="dcid:Euro")
    converted = AggregateOpSpec(frame_ref=SNAPSHOT_REF, aggregation="sum", unit=CONSTANT_EUR)
    again = AggregateOpSpec(
        frame_ref=SNAPSHOT_REF,
        aggregation="sum",
        unit=UnitRequest(currency="EUR", prices="constant", base_year=2020),
    )

    assert op_spec_hash_for(converted) != op_spec_hash_for(ref)
    assert op_spec_hash_for(converted) == op_spec_hash_for(again)


@pytest.mark.parametrize(
    "pin",
    [
        {"currency": "usd", "prices": "current"},
        {"currency": "USD", "prices": "nominal"},
        {"currency": "EUR", "prices": "constant"},
        {"currency": "USD", "prices": "current", "base_year": 2020},
        {"currency": "EUR", "prices": "constant", "base_year": True},
        {"currency": "USD", "prices": "current", "label": "dollars"},
        {"currency": {"role": "observationAbout"}, "prices": "current"},
        "u" * 201,
        42,
    ],
)
def test_a_malformed_unit_pin_is_refused(pin: object) -> None:
    with pytest.raises(ValidationError):
        AggregateOpSpec(frame_ref=SNAPSHOT_REF, aggregation="sum", unit=pin)


def test_dataset_snapshot_frame_ref_form_is_accepted() -> None:
    op_spec = AggregateOpSpec(frame_ref=FrameRef(**SNAPSHOT_REF), aggregation="sum")
    assert op_spec.frame_ref.spec_id == SNAPSHOT_REF["spec_id"]


def test_toolkit_result_frame_ref_form_is_accepted() -> None:
    op_spec = AggregateOpSpec(frame_ref=FrameRef(**TOOLKIT_RESULT_REF), aggregation="sum")
    assert op_spec.frame_ref.spec_id == TOOLKIT_RESULT_REF["spec_id"]


@pytest.mark.parametrize(
    "bad_ref",
    [
        {"spec_id": "not-a-spec-id", "content_version": "cv_" + "b" * 32},
        {"spec_id": "spec_" + "a" * 32, "content_version": "not-a-cv"},
    ],
)
def test_malformed_frame_ref_is_refused(bad_ref: object) -> None:
    with pytest.raises(ValidationError):
        AggregateOpSpec(frame_ref=bad_ref, aggregation="sum")


def test_unknown_verb_refused_at_parse() -> None:
    with pytest.raises(ValidationError):
        OpSpecAdapter.validate_python({"verb": "delete", "frame_ref": SNAPSHOT_REF})


def test_unknown_field_refused() -> None:
    with pytest.raises(ValidationError):
        AggregateOpSpec(frame_ref=SNAPSHOT_REF, aggregation="sum", not_a_real_field=1)


def test_positional_arguments_refused() -> None:
    with pytest.raises(TypeError):
        AggregateOpSpec(SNAPSHOT_REF)  # type: ignore[call-arg]


def test_filter_accepts_the_closed_predicate_vocabulary() -> None:
    op_spec = FilterOpSpec(
        frame_ref=SNAPSHOT_REF,
        predicates=(
            OrderedThresholdPredicate(column="value", op="gte", value=1_000_000),
            CategoryMatchPredicate(column="reporter", values=("FRA", "DEU")),
        ),
    )
    assert len(op_spec.predicates) == 2


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_ordered_threshold_predicate_refuses_non_finite_numeric_values(value: float) -> None:
    """A non-finite numeric predicate value refuses at parse, not at `op_spec_hash_for`."""
    with pytest.raises(ValidationError):
        OrderedThresholdPredicate(column="value", op="gt", value=value)


@pytest.mark.parametrize(
    "period",
    [
        pytest.param("2015-06", id="month"),
        # The longest period the frame grammar accepts; the filter engine
        # refuses it as an interval operand.
        pytest.param("2015-01-01/2016-12-31", id="day-span"),
    ],
)
def test_ordered_threshold_predicate_accepts_a_period_string_operand(period: str) -> None:
    predicate = OrderedThresholdPredicate(column="time_period", op="gte", value=period)
    assert predicate.value == period


@pytest.mark.parametrize("period", ["2015-13", "15", "2015/06", "2015-06-31-01"])
def test_ordered_threshold_predicate_refuses_a_malformed_period_string(period: str) -> None:
    with pytest.raises(ValidationError):
        OrderedThresholdPredicate(column="time_period", op="gte", value=period)


def test_value_threshold_kind_is_refused_by_the_discriminator() -> None:
    """The `value_threshold` kind has no schema to parse against -- a spec
    naming it fails on `Predicate`'s own discriminator, the same as any
    other unrecognised `kind`."""
    with pytest.raises(ValidationError):
        FilterOpSpec(
            frame_ref=SNAPSHOT_REF,
            predicates=[{"kind": "value_threshold", "column": "value", "op": "gte", "value": 1}],
        )


def test_predicate_outside_the_closed_vocabulary_is_refused() -> None:
    with pytest.raises(ValidationError):
        FilterOpSpec(
            frame_ref=SNAPSHOT_REF,
            predicates=[{"kind": "regex_match", "column": "reporter", "pattern": ".*"}],
        )


def test_filter_predicates_cannot_carry_scope_narrowing_fields() -> None:
    """Filter's grammar has no field for a role, window, or facet binding."""
    with pytest.raises(ValidationError):
        FilterOpSpec(frame_ref=SNAPSHOT_REF, role="reporter")  # type: ignore[call-arg]


def test_compare_has_no_scope_narrowing_field_either() -> None:
    with pytest.raises(ValidationError):
        CompareOpSpec(frame_ref=SNAPSHOT_REF, window="2015-2020")  # type: ignore[call-arg]


@pytest.mark.parametrize("field", ["window_start", "window_end"])
def test_growth_year_window_refuses_non_ascii_digits(field: str) -> None:
    """`\\d` matches any Unicode decimal digit, not just ASCII -- a
    full-width digit string would otherwise pass this pattern. `int()` and
    `range()` handle Unicode decimal digits fine, so the string itself
    wouldn't crash there; the actual failure is a `KeyError` once the
    still-full-width string is used to look up an NFC-ASCII period key
    inside the engine (e.g. `growth.py`'s `bucket[window_start]`), rather
    than refusing here."""
    kwargs = {"frame_ref": SNAPSHOT_REF, "window_start": "2020", "window_end": "2023"}
    kwargs[field] = "\uff12\uff10\uff12\uff10"  # full-width "2020"
    with pytest.raises(ValidationError):
        GrowthOpSpec(**kwargs)


@pytest.mark.parametrize("field", ["window_start", "window_end"])
def test_rank_year_window_refuses_non_ascii_digits(field: str) -> None:
    kwargs = {
        "frame_ref": SNAPSHOT_REF,
        "rank_by": ["reporter"],
        "window_start": "2020",
        "window_end": "2023",
    }
    kwargs[field] = "\uff12\uff10\uff12\uff10"  # full-width "2020"
    with pytest.raises(ValidationError):
        RankOpSpec(**kwargs)


def test_growth_change_defaults_to_relative() -> None:
    op_spec = GrowthOpSpec(frame_ref=SNAPSHOT_REF, window_start="2020", window_end="2023")
    assert op_spec.change == "relative"


def test_growth_change_accepts_absolute() -> None:
    op_spec = GrowthOpSpec(
        frame_ref=SNAPSHOT_REF, window_start="2020", window_end="2023", change="absolute"
    )
    assert op_spec.change == "absolute"


def test_growth_change_outside_the_closed_vocabulary_refuses() -> None:
    with pytest.raises(ValidationError):
        GrowthOpSpec(
            frame_ref=SNAPSHOT_REF, window_start="2020", window_end="2023", change="cumulative"
        )


def test_share_of_numerator_by_and_denominator_by_default_empty() -> None:
    op_spec = ShareOfOpSpec(frame_ref=SNAPSHOT_REF)
    assert op_spec.numerator_by == ()
    assert op_spec.denominator_by == {}


def test_share_of_denominator_by_pairing_by_name_round_trips() -> None:
    op_spec = ShareOfOpSpec(
        frame_ref=SNAPSHOT_REF,
        numerator_by=["donor", "sector"],
        denominator_by={"donor": "provider", "sector": "area"},
    )
    assert op_spec.numerator_by == ("donor", "sector")
    assert op_spec.denominator_by == {"donor": "provider", "sector": "area"}


def test_share_of_denominator_by_key_not_in_numerator_by_refuses() -> None:
    """A `denominator_by` key `numerator_by` doesn't name is refused at
    construction, not deferred to the engine -- see `ShareOfOpSpec`'s own
    `model_validator`."""
    with pytest.raises(ValidationError, match="denominator_by names column"):
        ShareOfOpSpec(
            frame_ref=SNAPSHOT_REF,
            numerator_by=["donor"],
            denominator_by={"sector": "area"},
        )


def test_share_of_denominator_by_may_leave_a_numerator_by_column_out_at_the_op_spec() -> None:
    """A partial mapping (every `denominator_by` key names a real
    `numerator_by` column, but not every `numerator_by` column has one) is
    schema-legal here -- the engine, not this class, refuses it
    (`ShareOfKeyUnknownError`), since "every numerator_by column has an
    entry" only reads naturally once both fields are compared together."""
    op_spec = ShareOfOpSpec(
        frame_ref=SNAPSHOT_REF,
        numerator_by=["donor", "sector"],
        denominator_by={"donor": "provider"},
    )
    assert op_spec.denominator_by == {"donor": "provider"}


def test_category_match_predicate_values_are_nfc_normalised() -> None:
    """An NFD-decomposed value (e.g. 'e' + combining acute) construction-time
    normalises to its NFC form -- the same normalisation `Frame.build`
    applies to every cell, and `analysis_mcp.identity` applies again before
    hashing (idempotently)."""
    nfd = "café"  # "café" spelled with a combining acute accent
    nfc = "café"  # "café" as one precomposed code point
    predicate = CategoryMatchPredicate(column="reporter", values=(nfd,))
    assert predicate.values == (nfc,)


def test_rank_per_period_defaults_to_false() -> None:
    op_spec = RankOpSpec(frame_ref=SNAPSHOT_REF, rank_by=["reporter"])
    assert op_spec.per_period is False


def test_rank_per_period_accepts_true() -> None:
    op_spec = RankOpSpec(frame_ref=SNAPSHOT_REF, rank_by=["reporter"], per_period=True)
    assert op_spec.per_period is True


def test_rank_limit_defaults_to_none() -> None:
    assert RankOpSpec(frame_ref=SNAPSHOT_REF, rank_by=["reporter"]).limit is None


@pytest.mark.parametrize("limit", [1, 1000])
def test_rank_limit_accepts_its_bounds(limit: int) -> None:
    op_spec = RankOpSpec(frame_ref=SNAPSHOT_REF, rank_by=["reporter"], limit=limit)
    assert op_spec.limit == limit


@pytest.mark.parametrize("limit", [0, -1, 1001])
def test_rank_limit_outside_its_bounds_refuses(limit: int) -> None:
    with pytest.raises(ValidationError):
        RankOpSpec(frame_ref=SNAPSHOT_REF, rank_by=["reporter"], limit=limit)

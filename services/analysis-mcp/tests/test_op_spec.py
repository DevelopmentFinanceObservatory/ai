"""The op-spec schema: closed verb set, keyword-only params, the pair-shaped
`frame_ref`."""

from __future__ import annotations

from collections.abc import Callable
from typing import get_args

import pytest
from analysis_mcp.identity import build_op_spec_preimage, op_spec_hash_for
from analysis_mcp.op_spec import (
    MIN_ROLLING_OFFSET,
    AggregateOpSpec,
    CategoryMatchPredicate,
    CompareOpSpec,
    FilterOpSpec,
    GrowthOpSpec,
    JoinOpSpec,
    OpSpecAdapter,
    OrderedThresholdPredicate,
    PlaceLevelPredicate,
    QuotientOfOpSpec,
    RankOpSpec,
    RollingRef,
    ShareOfOpSpec,
    Verb,
    with_anchor_key,
)
from common.materialisation import FrameRef
from common.rolling import canonical_populations
from dfo_frame_grammar import ANCHOR_RULE
from dfo_frame_grammar.grammar import ComposedUnit, StatVarPart, UnitRequest
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
EUR_PER_PERSON = {"numerator": CONSTANT_EUR, "denominator": {"stat_var": "dcid:Count_Person"}}
EUR_PER_PERSON_CELL = ComposedUnit(
    numerator=UnitRequest(currency="EUR", prices="constant", base_year=2020),
    denominator=StatVarPart(stat_var="dcid:Count_Person"),
)


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
        (EUR_PER_PERSON, EUR_PER_PERSON_CELL),
        (
            {"numerator": "dcid:USDollar", "denominator": "dcid:Euro"},
            ComposedUnit(numerator="dcid:USDollar", denominator="dcid:Euro"),
        ),
        (EUR_PER_PERSON_CELL, EUR_PER_PERSON_CELL),
    ],
)
def test_a_unit_pin_is_a_ref_a_converted_unit_or_a_composed_unit(
    pin: object, parsed: object
) -> None:
    """Every verb's unit pin parses a converted-unit object into the
    `UnitRequest`, and a composed-unit object into the `ComposedUnit`, that
    a frame's `unit` cell holds, so the engine compares the two by
    equality."""
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


def test_a_composed_pin_dumps_as_its_object_whatever_order_its_members_came_in() -> None:
    composed = AggregateOpSpec(frame_ref=SNAPSHOT_REF, aggregation="sum", unit=EUR_PER_PERSON)
    reordered = AggregateOpSpec(
        frame_ref=SNAPSHOT_REF,
        aggregation="sum",
        unit={
            "denominator": {"stat_var": "dcid:Count_Person"},
            "numerator": {"base_year": 2020, "prices": "constant", "currency": "EUR"},
        },
    )

    assert composed.model_dump(mode="json")["unit"] == EUR_PER_PERSON
    assert OpSpecAdapter.validate_python(composed.model_dump(mode="json")) == composed
    assert op_spec_hash_for(composed) == op_spec_hash_for(reordered)
    assert op_spec_hash_for(composed) != op_spec_hash_for(
        AggregateOpSpec(frame_ref=SNAPSHOT_REF, aggregation="sum", unit=CONSTANT_EUR)
    )


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
        {"numerator": "dcid:USDollar"},
        {"numerator": "dcid:USDollar", "denominator": None},
        {"numerator": "dcid:USDollar", "denominator": "dcid:Euro", "label": "rate"},
        {"numerator": EUR_PER_PERSON, "denominator": "dcid:Euro"},
        {"numerator": "dcid:USDollar", "denominator": {"stat_var": "Count_Person"}},
        {"numerator": "dcid:USDollar", "denominator": {"stat_var": "dcid:X", "currency": "USD"}},
        {"numerator": "u" * 201, "denominator": "dcid:Euro"},
        {"numerator": "dcid:USDollar", "denominator": {"stat_var": "dcid:" + "u" * 200}},
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


def test_a_strict_country_predicate_changes_the_approved_operation_identity() -> None:
    legacy = FilterOpSpec(
        frame_ref=SNAPSHOT_REF,
        predicates=(PlaceLevelPredicate(column="financingRecipient_type", level="Country"),),
    )
    explicit_legacy = FilterOpSpec(
        frame_ref=SNAPSHOT_REF,
        predicates=(
            PlaceLevelPredicate(column="financingRecipient_type", level="Country", strict=False),
        ),
    )
    strict = FilterOpSpec(
        frame_ref=SNAPSHOT_REF,
        predicates=(
            PlaceLevelPredicate(column="financingRecipient_type", level="Country", strict=True),
        ),
    )

    assert op_spec_hash_for(legacy) == op_spec_hash_for(explicit_legacy)
    assert op_spec_hash_for(strict) != op_spec_hash_for(legacy)
    assert "strict" not in build_op_spec_preimage(legacy)["parameters"]["predicates"][0]
    assert build_op_spec_preimage(strict)["parameters"]["predicates"][0]["strict"] is True


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


def test_rank_window_accepts_one_year_and_refuses_a_reversed_pair() -> None:
    one_year = RankOpSpec(
        frame_ref=SNAPSHOT_REF, rank_by=["reporter"], window_start="2020", window_end="2020"
    )
    assert one_year.window_end == "2020"
    with pytest.raises(ValidationError, match="must not be earlier"):
        RankOpSpec(
            frame_ref=SNAPSHOT_REF, rank_by=["reporter"], window_start="2021", window_end="2020"
        )


def test_growth_window_refuses_one_year() -> None:
    with pytest.raises(ValidationError):
        GrowthOpSpec(frame_ref=SNAPSHOT_REF, window_start="2020", window_end="2020")


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


def test_quotient_of_denominator_by_pairing_by_name_round_trips() -> None:
    op_spec = QuotientOfOpSpec(
        frame_ref=SNAPSHOT_REF,
        numerator_by=["donor", "sector"],
        denominator_by={"donor": "provider", "sector": "area"},
    )
    assert op_spec.numerator_by == ("donor", "sector")
    assert op_spec.denominator_by == {"donor": "provider", "sector": "area"}
    assert OpSpecAdapter.validate_python(op_spec.model_dump(mode="json")) == op_spec


def test_quotient_of_keyed_numerator_with_no_denominator_by_is_a_pooled_denominator() -> None:
    op_spec = QuotientOfOpSpec(frame_ref=SNAPSHOT_REF, numerator_by=["donor"])
    assert op_spec.denominator_by == {}


@pytest.mark.parametrize(
    ("denominator_by", "match"),
    [
        ({"sector": "area"}, "denominator_by names column"),
        ({"donor": "provider"}, "denominator_by leaves out"),
    ],
    ids=["a-key-outside-numerator-by", "a-partial-mapping"],
)
def test_quotient_of_refuses_a_denominator_by_that_does_not_map_numerator_by_exactly(
    denominator_by: dict[str, str], match: str
) -> None:
    """Unlike `share_of`, a partial mapping is refused here, so the engine
    never sees one."""
    numerator_by = ["donor"] if "sector" in denominator_by else ["donor", "sector"]
    with pytest.raises(ValidationError, match=match):
        QuotientOfOpSpec(
            frame_ref=SNAPSHOT_REF, numerator_by=numerator_by, denominator_by=denominator_by
        )


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


# Rolling overlay ---------------------------------------------------------------

LATEST = {"anchor": "latest", "offset": 0, "rule": "half1"}
ROOT = "spec_" + "c" * 32
OTHER_ROOT = "spec_" + "d" * 32
EVERY_SERIES = [{"spec_id": ROOT}]
ANCHOR_KEY = "rak_" + "e" * 32


def _back(offset: int) -> dict[str, object]:
    return {**LATEST, "offset": offset}


def _pairing(
    *,
    left: list[dict[str, object]] | None = None,
    right: list[dict[str, object]] | None = None,
    keys: dict[str, str] | None = None,
) -> dict[str, object]:
    """A pairing population over `left` and `right`, by default every series of
    each root, paired on `keys`, by default `recipient` on both sides."""
    return {
        "pair": [
            [{"spec_id": ROOT}] if left is None else left,
            [{"spec_id": OTHER_ROOT}] if right is None else right,
        ],
        "keys": {"recipient": "recipient"} if keys is None else keys,
    }


def _rolling(**bounds: object) -> dict[str, object]:
    return {"population": EVERY_SERIES, **bounds}


def _growth(**fields: object) -> GrowthOpSpec:
    return GrowthOpSpec(frame_ref=SNAPSHOT_REF, window_start="2019", window_end="2024", **fields)


def _rank(**fields: object) -> RankOpSpec:
    return RankOpSpec(frame_ref=SNAPSHOT_REF, rank_by=("reporter",), **fields)


def _threshold(**fields: object) -> dict[str, object]:
    return {"kind": "ordered_threshold", "column": "time_period", "op": "gte", **fields}


def _filter(*predicates: dict[str, object], **fields: object) -> FilterOpSpec:
    return FilterOpSpec(frame_ref=SNAPSHOT_REF, predicates=predicates, **fields)


def test_the_rolling_rule_is_the_frame_grammars() -> None:
    assert get_args(RollingRef.model_fields["rule"].annotation) == (ANCHOR_RULE,)


def test_a_rolling_window_keeps_its_literals_and_enters_the_preimage_without_unset_members() -> (
    None
):
    op_spec = _growth(rolling=_rolling(window_end=LATEST))

    assert (op_spec.window_start, op_spec.window_end) == ("2019", "2024")
    assert build_op_spec_preimage(op_spec)["parameters"]["rolling"] == {
        "window_end": LATEST,
        "population": [{"spec_id": ROOT, "bindings": {}}],
    }
    assert "rolling" not in build_op_spec_preimage(_growth())["parameters"]


@pytest.mark.parametrize(
    "bounds",
    [
        {"window_start": _back(-5), "window_end": LATEST},
        {"window_start": _back(-6), "window_end": _back(-1)},
        {"window_start": _back(-5)},
    ],
)
def test_a_rolling_window_whose_literals_agree_with_its_offsets_is_accepted(
    bounds: dict[str, object],
) -> None:
    assert _growth(rolling=_rolling(**bounds)).rolling is not None
    assert _rank(window_start="2019", window_end="2024", rolling=_rolling(**bounds)).rolling


@pytest.mark.parametrize(
    "rolling",
    [
        pytest.param(_rolling(window_start=_back(-4), window_end=LATEST), id="literals-disagree"),
        pytest.param(_rolling(window_end=_back(1)), id="offset-above-zero"),
        pytest.param(_rolling(window_end=_back(MIN_ROLLING_OFFSET - 1)), id="offset-out-of-range"),
        pytest.param(_rolling(window_end={**LATEST, "offset": -1.0}), id="offset-not-an-integer"),
        pytest.param(_rolling(window_end={**LATEST, "rule": "third1"}), id="unknown-rule"),
        pytest.param(_rolling(window_end={**LATEST, "anchor": "first"}), id="unknown-anchor"),
        pytest.param(_rolling(), id="no-rolling-bound"),
        pytest.param(_rolling(window_end=LATEST, window=LATEST), id="unknown-member"),
        pytest.param({"window_end": LATEST}, id="no-population"),
        pytest.param({"window_end": LATEST, "population": []}, id="empty-population"),
        pytest.param(
            {"window_end": LATEST, "population": [{"spec_id": "ops_" + "c" * 32}]},
            id="population-of-a-derived-frame",
        ),
        pytest.param(
            {
                "window_end": LATEST,
                "population": [{"spec_id": ROOT, "content_version": "cv_" + "c" * 32}],
            },
            id="population-with-a-content-version",
        ),
        pytest.param(
            {"window_end": LATEST, "population": [{"spec_id": ROOT, "bindings": {"reporter": []}}]},
            id="empty-binding",
        ),
        pytest.param(
            {"window_end": LATEST, "population": [_pairing(keys={})]},
            id="pairing-without-keys",
        ),
        pytest.param(
            {"window_end": LATEST, "population": [_pairing(right=[])]},
            id="pairing-with-an-empty-side",
        ),
    ],
)
def test_a_malformed_or_inconsistent_rolling_window_is_refused(rolling: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        _growth(rolling=rolling)
    with pytest.raises(ValidationError):
        _rank(window_start="2019", window_end="2024", rolling=rolling)


def test_a_rolling_bound_on_a_rank_without_a_window_is_refused() -> None:
    with pytest.raises(ValidationError):
        _rank(rolling=_rolling(window_end=LATEST))


def test_the_ordering_rule_still_reads_a_rolling_windows_literals() -> None:
    fields = {"window_start": "2024", "window_end": "2019"}
    with pytest.raises(ValidationError):
        GrowthOpSpec(frame_ref=SNAPSHOT_REF, rolling=_rolling(window_end=LATEST), **fields)
    with pytest.raises(ValidationError):
        _rank(rolling=_rolling(window_end=LATEST), **fields)


def test_a_growth_with_a_rolling_window_refuses_a_one_year_window() -> None:
    fields = {"window_start": "2024", "window_end": "2024"}
    with pytest.raises(ValidationError):
        GrowthOpSpec(frame_ref=SNAPSHOT_REF, rolling=_rolling(window_end=LATEST), **fields)


def test_populations_have_one_spelling() -> None:
    nfd, nfc = "Cote d'Ivoire\u0301", "Cote d'Ivoir\u00e9"
    usa = {"spec_id": ROOT, "bindings": {"reporter": ["dcid:USA"]}}
    listed = [
        {"spec_id": OTHER_ROOT},
        {"spec_id": ROOT, "bindings": {"reporter": ["dcid:USA", nfd, "dcid:USA"], "flow": ["oda"]}},
        usa,
        usa,
    ]
    respelled = [
        usa,
        {"spec_id": ROOT, "bindings": {"flow": ["oda"], "reporter": [nfc, "dcid:USA"]}},
        {"spec_id": OTHER_ROOT, "bindings": {}},
    ]

    def growth_over(population: list[dict[str, object]]) -> GrowthOpSpec:
        return _growth(rolling={"window_end": LATEST, "population": population})

    assert build_op_spec_preimage(growth_over(listed))["parameters"]["rolling"]["population"] == [
        {"spec_id": ROOT, "bindings": {"flow": ["oda"], "reporter": [nfc, "dcid:USA"]}},
        {"spec_id": ROOT, "bindings": {"reporter": ["dcid:USA"]}},
        {"spec_id": OTHER_ROOT, "bindings": {}},
    ]
    assert op_spec_hash_for(growth_over(listed)) == op_spec_hash_for(growth_over(respelled))
    assert op_spec_hash_for(growth_over(listed)) != op_spec_hash_for(growth_over([usa]))
    # A pairing is spelled as `common.rolling` spells it, after every flat entry.
    paired = [_pairing(left=listed, keys={"recipient": "observationAbout"}), *listed]
    assert build_op_spec_preimage(growth_over(paired))["parameters"]["rolling"]["population"] == (
        canonical_populations(paired)
    )


@pytest.mark.parametrize(
    "predicate",
    [
        pytest.param(_threshold(column="value", value=1, rolling=LATEST), id="value-column"),
        pytest.param(_threshold(column="reporter", value="2020", rolling=LATEST), id="role-column"),
        pytest.param(_threshold(value=2020, rolling=LATEST), id="numeric-operand"),
        pytest.param(_threshold(value="2019/2024", rolling=LATEST), id="span"),
        pytest.param(_threshold(value="2024-03-01", rolling=_back(-1)), id="day-at-an-offset"),
        pytest.param(_threshold(value="2020", rolling=_back(1)), id="offset-above-zero"),
    ],
)
def test_a_rolling_predicate_off_a_period_threshold_is_refused(
    predicate: dict[str, object],
) -> None:
    _filter(_threshold(value="2020", rolling=LATEST), rolling=_rolling())
    with pytest.raises(ValidationError):
        _filter(predicate, rolling=_rolling())


_NINE_BACK = {"anchor": "latest", "offset": -9}
_NOW = {"anchor": "latest", "offset": 0}


def test_a_relative_year_stands_in_for_a_window_bound_or_a_period_threshold() -> None:
    growth = GrowthOpSpec(frame_ref=SNAPSHOT_REF, window_start=_NINE_BACK, window_end=_NOW)
    assert growth.model_dump(mode="json", include={"window_start", "window_end"}) == {
        "window_start": _NINE_BACK,
        "window_end": _NOW,
    }
    GrowthOpSpec(frame_ref=SNAPSHOT_REF, window_start=_NINE_BACK, window_end="2024")
    _rank(window_start="2015", window_end=_NOW)
    _filter(_threshold(value=_NINE_BACK))


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(
            lambda: GrowthOpSpec(frame_ref=SNAPSHOT_REF, window_start=_NOW, window_end=_NOW),
            id="growth-one-year",
        ),
        pytest.param(lambda: _rank(window_start=_NOW, window_end=_NINE_BACK), id="rank-reversed"),
        pytest.param(lambda: _rank(window_end=_NOW), id="rank-one-bound"),
        pytest.param(
            lambda: GrowthOpSpec(
                frame_ref=SNAPSHOT_REF,
                window_start="2015",
                window_end=_NOW,
                rolling=_rolling(window_end=LATEST),
            ),
            id="beside-an-overlay",
        ),
        pytest.param(
            lambda: _rank(window_start="2015", window_end={**_NOW, "offset": 1}),
            id="offset-above-zero",
        ),
        pytest.param(
            lambda: _rank(window_start="2015", window_end={**_NOW, "rule": "half1"}),
            id="unknown-member",
        ),
        pytest.param(lambda: _filter(_threshold(column="value", value=_NOW)), id="value-column"),
        pytest.param(
            lambda: _filter(_threshold(value=_NOW, rolling=LATEST), rolling=_rolling()),
            id="threshold-beside-rolling",
        ),
    ],
)
def test_a_misplaced_relative_year_is_refused(build: Callable[[], object]) -> None:
    with pytest.raises(ValidationError):
        build()


@pytest.mark.parametrize("value", ["2024", "2024-03", "2024-03-01"])
def test_a_rolling_predicate_at_the_latest_period_is_accepted_at_every_grain(value: str) -> None:
    op_spec = _filter(_threshold(value=value, rolling=LATEST), rolling=_rolling())
    assert op_spec.predicates[0].implied_anchor() == value


def test_rolling_predicates_of_one_filter_stand_for_one_latest_period() -> None:
    lower = _threshold(value="2020", rolling=_back(-4))
    _filter(lower, _threshold(op="lte", value="2024", rolling=LATEST), rolling=_rolling())
    with pytest.raises(ValidationError):
        _filter(lower, _threshold(op="lte", value="2023", rolling=LATEST), rolling=_rolling())


def test_a_filter_states_its_populations_exactly_when_a_predicate_is_rolling() -> None:
    rolling_predicate = _threshold(value="2024", rolling=LATEST)

    _filter(rolling_predicate, rolling=_rolling())
    _filter(_threshold(value="2024"))
    with pytest.raises(ValidationError):
        _filter(_threshold(value="2024"), rolling=_rolling())
    with pytest.raises(ValidationError):
        _filter(rolling_predicate)
    with pytest.raises(ValidationError):
        _filter(rolling_predicate, rolling={"population": []})


def test_an_anchor_key_accompanies_exactly_a_rolling_op_spec() -> None:
    rolling = _growth(rolling=_rolling(window_end=LATEST))

    assert with_anchor_key(rolling, ANCHOR_KEY) is rolling
    assert with_anchor_key(_growth(), None).rolling is None
    for op_spec, anchor_key in [
        (rolling, None),
        (_growth(), ANCHOR_KEY),
        (rolling, "rak_" + "E" * 32),
        (rolling, "ops_" + "e" * 32),
    ]:
        with pytest.raises(ValidationError):
            with_anchor_key(op_spec, anchor_key)

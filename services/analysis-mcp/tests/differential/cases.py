"""The seeded case generator behind `engine_cases.jsonl`: frames and op
parameters for every verb engine, built to reach the arithmetic, ordering and
refusal paths an engine rewrite could silently change.

Every case is plain JSON: its input frames in canonical form (`frame_spec`),
and keyword arguments that `harness.IMPLEMENTATIONS` passes straight to a verb
engine. `generate_cases()` is deterministic for `SEED`, so the committed
fixture's inputs can be regenerated and checked
(`test_engine_differential.test_generator_reproduces_fixture_inputs`).

Each verb gets a block of hand-built edge cases (cancellation sums,
overflowing groups, ties, signed zeros, gaps, zero denominators, repeated
keys, converted units, more than 20 excluded entities) followed by a block of
random cases drawn from the same vocabulary.
"""

from __future__ import annotations

import itertools
import json
import random
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from dfo_frame_grammar import STATUS_COLUMN, TIME_PERIOD_COLUMN, Frame, canonical_frame_bytes

SEED = 20260930

type Spec = dict[str, Any]
type Case = dict[str, Any]

SV_A = "dcid:DiffStatVarA"
SV_B = "dcid:DiffStatVarB"
USD = "dcid:USDollar"
PCT = "dcid:Percent"
ABSENT_UNIT = "dcid:Euro"
REQUEST_USD = {"currency": "USD", "prices": "current"}
REQUEST_EUR = {"currency": "EUR", "prices": "constant", "base_year": 2020}
MM_A = "dcid:MethodA"
MM_B = "dcid:MethodB"
PROV_1 = "dcid:Provenance1"
PROV_2 = "dcid:Provenance2"

DONORS = tuple(f"dcid:Donor{index:02d}" for index in range(30))
RECIPIENTS = tuple(f"dcid:Recipient{index:02d}" for index in range(30))
YEARS = tuple(str(year) for year in range(2010, 2026))
MONTHS = ("2020-01", "2020-02", "2020-03", "2021-01", "2021-06")

AGGREGATIONS = ("sum", "mean", "max", "min")

# Exact sums that a naive left-to-right `+` gets wrong: `1e16 + 1` rounds back
# to `1e16`, so only a correctly-rounded sum sees the three ones.
CANCELLATION = (1e16, 1.0, 1.0, 1.0)
CANCELLATION_TO_ONE = (1e16, 1.0, -1e16)
# `2**53 + 1` is exactly halfway between two doubles and rounds to even,
# down to `2**53`; the `1e-30` makes the exact sum round up instead. Python's
# compensated `sum()` (3.12+) loses the `1e-30` in its correction term and
# returns `2**53`; only a correctly-rounded sum returns `2**53 + 2`.
TIE_BREAK = (9007199254740992.0, 1e-30, 1.0)
NEAR_MAX = 1.5e308


def frame_spec(role_keys: Sequence[str], rows: Iterable[Mapping[str, object]]) -> Spec:
    """`Frame.build(role_keys=..., rows=...)` in canonical JSON form: the
    built frame's columns (minus `status`) and its canonically sorted rows,
    with a `UnitRequest` unit written as its JSON object."""
    frame = Frame.build(role_keys=role_keys, rows=rows)
    document = json.loads(canonical_frame_bytes(frame))
    columns = list(frame.columns)
    status = columns.index(STATUS_COLUMN)
    return {
        "role_keys": sorted(role_keys),
        "columns": [column for column in columns if column != STATUS_COLUMN],
        "rows": [row[:status] + row[status + 1 :] for row in document["rows"]],
    }


def frame_from_spec(spec: Spec) -> Frame:
    """The `Frame` a `frame_spec` describes."""
    columns = spec["columns"]
    return Frame.build(
        role_keys=spec["role_keys"],
        rows=[dict(zip(columns, row, strict=True)) for row in spec["rows"]],
    )


def _nonround(rng: random.Random) -> float:
    """A finite double with a full-precision mantissa, sign mostly positive."""
    magnitude = rng.uniform(1.0, 10.0) * 10.0 ** rng.randint(-3, 11)
    return -magnitude if rng.random() < 0.15 else magnitude


def _values_general(rng: random.Random) -> float:
    return _nonround(rng)


def _values_positive(rng: random.Random) -> float:
    return abs(_nonround(rng))


def _values_ties(rng: random.Random) -> float:
    return rng.choice((1234.5678901234567, -0.3333333333333333, 98765.4321, 0.0))


def _values_signed(rng: random.Random) -> float:
    return rng.choice((rng.uniform(-10.0, 10.0), 0.0, rng.uniform(-1e-9, 1e-9)))


def _values_cancelling(rng: random.Random) -> float:
    return rng.choice((*CANCELLATION, *CANCELLATION_TO_ONE, *TIE_BREAK, 0.1, 0.2, 0.3))


VALUE_POLICIES: dict[str, Callable[[random.Random], float]] = {
    "general": _values_general,
    "positive": _values_positive,
    "ties": _values_ties,
    "signed": _values_signed,
    "cancelling": _values_cancelling,
}

UNIT_POLICIES: dict[str, Callable[[random.Random], object]] = {
    "usd": lambda rng: USD,
    "usd_or_null": lambda rng: USD if rng.random() < 0.8 else None,
    "null": lambda rng: None,
    "usd_or_pct": lambda rng: rng.choice((USD, PCT)),
    "mixed": lambda rng: rng.choice((USD, PCT, None)),
    "request": lambda rng: REQUEST_USD,
    "request_eur": lambda rng: REQUEST_EUR,
    "request_or_null": lambda rng: REQUEST_USD if rng.random() < 0.8 else None,
    "two_requests": lambda rng: rng.choice((REQUEST_USD, REQUEST_EUR)),
    "usd_or_request": lambda rng: rng.choice((USD, REQUEST_USD)),
    "pct_only": lambda rng: PCT,
}

# Weighted so most random cases compute rather than refuse.
_UNIT_WEIGHTS = {
    "usd": 6,
    "usd_or_null": 4,
    "null": 2,
    "usd_or_pct": 2,
    "mixed": 1,
    "request": 1,
    "request_or_null": 1,
    "two_requests": 1,
    "usd_or_request": 1,
}

FACET_POLICIES: dict[str, Callable[[random.Random], tuple[str | None, str | None]]] = {
    "none": lambda rng: (None, None),
    "mm": lambda rng: (MM_A, None),
    "mm_b": lambda rng: (MM_B, None),
    "mm_partial": lambda rng: (MM_A if rng.random() < 0.7 else None, None),
    "mm_split": lambda rng: (rng.choice((MM_A, MM_B)), None),
    "prov_partial": lambda rng: (None, PROV_1 if rng.random() < 0.7 else None),
    "both": lambda rng: (MM_A, PROV_1),
}

_FACET_WEIGHTS = {"none": 8, "mm": 2, "mm_partial": 2, "mm_split": 1, "prov_partial": 2, "both": 1}


def _weighted(rng: random.Random, weights: Mapping[str, int]) -> str:
    return rng.choices(list(weights), weights=list(weights.values()))[0]


def grid(
    rng: random.Random,
    *,
    roles: Mapping[str, Sequence[str]],
    periods: Sequence[str],
    stat_vars: Sequence[str] = (SV_A,),
    density: float = 1.0,
    unit: str = "usd",
    facets: str = "none",
    values: str = "general",
) -> Spec:
    """A frame over every (role values, stat_var, period) cell, each kept
    with probability `density`, its unit, facets and value drawn from the
    named policies."""
    rows: list[dict[str, object]] = []
    for combination in itertools.product(*roles.values()):
        for stat_var in stat_vars:
            for period in periods:
                if rng.random() >= density:
                    continue
                measurement_method, provenance = FACET_POLICIES[facets](rng)
                rows.append(
                    {
                        "stat_var": stat_var,
                        **dict(zip(roles, combination, strict=True)),
                        TIME_PERIOD_COLUMN: period,
                        "value": VALUE_POLICIES[values](rng),
                        "unit": UNIT_POLICIES[unit](rng),
                        "measurementMethod": measurement_method,
                        "provenance": provenance,
                    }
                )
    return frame_spec(list(roles), rows)


def rows_frame(role_keys: Sequence[str], rows: Sequence[Mapping[str, object]]) -> Spec:
    """A hand-built frame; every row defaults to `SV_A` and `USD`."""
    return frame_spec(role_keys, [{"stat_var": SV_A, "unit": USD, **row} for row in rows])


def _random_periods(rng: random.Random) -> list[str]:
    if rng.random() < 0.12:
        return rng.sample(MONTHS, rng.randint(2, len(MONTHS)))
    start = rng.randrange(len(YEARS) - 6)
    return list(YEARS[start : start + rng.randint(2, 6)])


def _random_roles(rng: random.Random, *, allow_none: bool = False) -> dict[str, list[str]]:
    shapes: list[tuple[str, ...]] = [("donor",), ("donor",), ("donor", "recipient")]
    if allow_none:
        shapes.append(())
    shape = rng.choice(shapes)
    pools = {"donor": DONORS, "recipient": RECIPIENTS}
    return {role: list(rng.sample(pools[role], rng.randint(1, 4))) for role in shape}


def _random_grid(rng: random.Random, *, allow_no_roles: bool = False, **overrides: Any) -> Spec:
    arguments: dict[str, Any] = {
        "roles": _random_roles(rng, allow_none=allow_no_roles),
        "periods": _random_periods(rng),
        "stat_vars": (SV_A,),
        "density": rng.choice((1.0, 1.0, 1.0, 0.75, 0.5)),
        "unit": _weighted(rng, _UNIT_WEIGHTS),
        "facets": _weighted(rng, _FACET_WEIGHTS),
        "values": rng.choice(("general", "general", "positive", "ties", "signed", "cancelling")),
    }
    arguments.update(overrides)
    return grid(rng, **arguments)


def _string_units(spec: Spec) -> list[str]:
    unit_index = spec["columns"].index("unit")
    return sorted({row[unit_index] for row in spec["rows"] if isinstance(row[unit_index], str)})


def _column_values(spec: Spec, column: str) -> list[Any]:
    """The distinct cells of `column`, `None` last, unit requests left out."""
    index = spec["columns"].index(column)
    distinct = {row[index] for row in spec["rows"] if not isinstance(row[index], dict)}
    return sorted(distinct, key=lambda value: (value is None, value))


def _computing_overrides(rng: random.Random) -> dict[str, Any]:
    """Grid arguments for a two-sided verb's random case that leave it more
    likely to compute than to refuse: one role, one unit, few facets."""
    return {
        "roles": {"donor": list(rng.sample(DONORS, rng.randint(1, 4)))},
        "unit": rng.choice(("usd", "usd", "usd", "usd", "usd_or_null", "null", "usd_or_pct")),
        "facets": rng.choice(("none", "none", "none", "mm", "mm_partial", "prov_partial")),
        "values": rng.choice(("general", "positive", "positive", "cancelling")),
    }


def _random_pin(rng: random.Random, spec: Spec) -> str | None:
    """`None` most of the time; otherwise a unit the frame declares, or one
    it does not."""
    roll = rng.random()
    units = _string_units(spec)
    if roll < 0.7:
        return None
    if roll < 0.92 and units:
        return rng.choice(units)
    return ABSENT_UNIT


def _random_binding(rng: random.Random, spec: Spec) -> dict[str, str]:
    """An entity binding over the frame's own role columns: usually one
    present value, sometimes empty, a value it lacks, or a disallowed column."""
    roles = spec["role_keys"]
    roll = rng.random()
    if roll < 0.08:
        return {"value": "1"}
    if roll < 0.15 or not roles:
        return {}
    column = rng.choice(roles)
    present = [value for value in _column_values(spec, column) if value is not None]
    if roll < 0.2 or not present:
        return {column: "dcid:NotPresent"}
    return {column: rng.choice(present)}


def _cancelling_frame(series: Sequence[Sequence[float]]) -> Spec:
    """One donor per entry of `series`, its values in period order, so a
    group's canonical row order is the order a naive sum would add them in."""
    return rows_frame(
        ["donor"],
        [
            {"donor": DONORS[donor], "time_period": YEARS[index], "value": value}
            for donor, values in enumerate(series)
            for index, value in enumerate(values)
        ],
    )


def _case(verb: str, name: str, frames: Sequence[Spec], **params: Any) -> Case:
    return {"id": f"{verb}-{name}", "verb": verb, "frames": list(frames), "params": params}


def _many_entities_frame(rng: random.Random, *, count: int, unit: str = "usd") -> Spec:
    return grid(
        rng,
        roles={"donor": DONORS[:count]},
        periods=YEARS[5:8],
        unit=unit,
        values="positive",
    )


def aggregate_cases(rng: random.Random) -> list[Case]:
    """`run_aggregate(frame, group_by=, aggregation=, pinned_unit=)`."""
    cases = []

    def aggregate(
        name: str,
        frame: Spec,
        *,
        group_by: Sequence[str],
        aggregation: str,
        pinned_unit: str | None = None,
    ) -> None:
        cases.append(
            _case(
                "aggregate",
                name,
                [frame],
                group_by=list(group_by),
                aggregation=aggregation,
                pinned_unit=pinned_unit,
            )
        )

    cancelling = _cancelling_frame((CANCELLATION, CANCELLATION_TO_ONE, TIE_BREAK))
    for aggregation in AGGREGATIONS:
        aggregate(
            f"cancellation-{aggregation}", cancelling, group_by=["donor"], aggregation=aggregation
        )
    overflowing = rows_frame(
        ["donor"],
        [
            {"donor": donor, "time_period": year, "value": NEAR_MAX}
            for donor in (DONORS[3], DONORS[1])
            for year in YEARS[:2]
        ],
    )
    for aggregation in ("sum", "mean"):
        aggregate(
            f"two-overflowing-groups-{aggregation}",
            overflowing,
            group_by=["donor"],
            aggregation=aggregation,
        )
    tied = rows_frame(
        ["donor"],
        [
            {"donor": DONORS[0], "time_period": "2021", "value": 7.123456789},
            {"donor": DONORS[0], "time_period": "2019", "value": 7.123456789},
            {"donor": DONORS[0], "time_period": "2020", "value": -7.123456789},
            {"donor": DONORS[1], "time_period": "2020", "value": -7.123456789},
            {"donor": DONORS[1], "time_period": "2022", "value": -7.123456789},
        ],
    )
    for aggregation in ("max", "min"):
        aggregate(f"period-ties-{aggregation}", tied, group_by=["donor"], aggregation=aggregation)
    aggregate("empty-frame", frame_spec(["donor"], []), group_by=[], aggregation="sum")
    unitless_many = grid(
        rng,
        roles={"donor": DONORS[:26]},
        periods=YEARS[:2],
        unit="usd_or_null",
        values="positive",
    )
    for pin in (None, USD):
        aggregate(
            f"unitless-over-20-entities-pin-{pin is not None}",
            unitless_many,
            group_by=["time_period"],
            aggregation="sum",
            pinned_unit=pin,
        )
    two_units = _many_entities_frame(rng, count=24, unit="usd_or_pct")
    aggregate(
        "unit-not-selected-over-20-entities",
        two_units,
        group_by=["donor"],
        aggregation="mean",
        pinned_unit=PCT,
    )
    for policy in ("request", "two_requests", "usd_or_request", "request_or_null"):
        frame = grid(rng, roles={"donor": DONORS[:3]}, periods=YEARS[:3], unit=policy)
        for pin in (None, USD):
            aggregate(
                f"converted-{policy}-pin-{pin is not None}",
                frame,
                group_by=["donor"],
                aggregation="sum",
                pinned_unit=pin,
            )
    for facets in ("mm_partial", "mm_split", "prov_partial", "both"):
        frame = grid(rng, roles={"donor": DONORS[:3]}, periods=YEARS[:4], facets=facets)
        aggregate(f"facets-{facets}", frame, group_by=["time_period"], aggregation="mean")
    sparse = grid(
        rng,
        roles={"donor": DONORS[:5], "recipient": RECIPIENTS[:4]},
        periods=YEARS[:3],
        density=0.4,
    )
    aggregate("sparse-grid", sparse, group_by=["donor", "recipient"], aggregation="sum")
    two_stat_vars = grid(
        rng, roles={"donor": DONORS[:2]}, periods=YEARS[:3], stat_vars=(SV_A, SV_B)
    )
    for group_by in (["donor"], ["stat_var", "donor"], ["time_period", "stat_var"]):
        aggregate(
            f"two-stat-vars-{'-'.join(group_by)}",
            two_stat_vars,
            group_by=group_by,
            aggregation="sum",
        )
    for group_by, aggregation in ((["value"], "sum"), (["sector"], "sum"), ([], "median")):
        aggregate(
            f"invalid-{'-'.join(group_by) or aggregation}",
            two_stat_vars,
            group_by=group_by,
            aggregation=aggregation,
        )

    for index in range(36):
        frame = _random_grid(rng, stat_vars=rng.choice(((SV_A,), (SV_A,), (SV_A,), (SV_A, SV_B))))
        candidates = [*frame["role_keys"], "stat_var", "time_period"]
        aggregate(
            f"random-{index:03d}",
            frame,
            group_by=rng.sample(candidates, rng.randint(0, len(candidates))),
            aggregation=rng.choice(AGGREGATIONS),
            pinned_unit=_random_pin(rng, frame),
        )
    return cases


# Unit policy pairs for two-sided verbs where a converted unit meets a ref,
# another converted unit, or an unbound side. Drawn after each verb's random
# block, so the cases before them keep their inputs.
_CROSS_SIDE_CONVERTED_UNITS = (
    ("request", "usd"),
    ("usd", "request"),
    ("request", "request_eur"),
    ("null", "request"),
    ("request", "null"),
)


def compare_cases(rng: random.Random) -> list[Case]:
    """`run_compare(left, right, stat_var=, left_entity=, right_entity=,
    left_pinned_unit=, right_pinned_unit=)`. One frame spec means both sides read it."""
    cases = []
    two_donors = grid(rng, roles={"donor": DONORS[:2]}, periods=YEARS[:5], values="signed")
    left, right = {"donor": DONORS[0]}, {"donor": DONORS[1]}
    cases.append(_case("compare", "two-donors", [two_donors], **_compare_params(left, right)))
    equal = rows_frame(
        ["donor"],
        [
            {"donor": donor, "time_period": year, "value": value}
            for donor in DONORS[:2]
            for year, value in zip(YEARS[:3], (0.1 + 0.2, -2.5e-8, 3.3), strict=True)
        ],
    )
    cases.append(_case("compare", "equal-values", [equal], **_compare_params(left, right)))
    overflow = rows_frame(
        ["donor"],
        [
            {"donor": DONORS[0], "time_period": "2020", "value": -NEAR_MAX},
            {"donor": DONORS[1], "time_period": "2020", "value": NEAR_MAX},
        ],
    )
    cases.append(_case("compare", "overflow", [overflow], **_compare_params(left, right)))
    disjoint = rows_frame(
        ["donor"],
        [
            {"donor": DONORS[0], "time_period": "2019", "value": 1.5},
            {"donor": DONORS[1], "time_period": "2020", "value": 2.5},
        ],
    )
    cases.append(_case("compare", "disjoint", [disjoint], **_compare_params(left, right)))
    cases.append(_case("compare", "repeated-period", [two_donors], **_compare_params({}, right)))
    many = _many_entities_frame(rng, count=25)
    cases.append(
        _case(
            "compare",
            "entity-not-selected-over-20",
            [many],
            **_compare_params({"donor": DONORS[0]}, {"donor": DONORS[24]}),
        )
    )
    partial = grid(rng, roles={"donor": DONORS[:2]}, periods=YEARS[:6], density=0.7)
    cases.append(_case("compare", "partial-overlap", [partial], **_compare_params(left, right)))
    for left_unit, right_unit in (("usd", "pct_only"), ("usd", "null"), ("null", "usd")):
        left_frame = grid(rng, roles={"donor": DONORS[:1]}, periods=YEARS[:3], unit=left_unit)
        right_frame = grid(rng, roles={"donor": DONORS[1:2]}, periods=YEARS[:3], unit=right_unit)
        cases.append(
            _case(
                "compare",
                f"units-{left_unit}-{right_unit}",
                [left_frame, right_frame],
                **_compare_params(left, right),
            )
        )
    left_facet = grid(rng, roles={"donor": DONORS[:1]}, periods=YEARS[:3], facets="mm")
    right_facet = grid(rng, roles={"donor": DONORS[1:2]}, periods=YEARS[:3], facets="mm_split")
    right_one_sided = grid(
        rng, roles={"donor": DONORS[1:2]}, periods=YEARS[:3], facets="prov_partial"
    )
    for name, right_frame in (("split", right_facet), ("one-sided", right_one_sided)):
        cases.append(
            _case(
                "compare",
                f"facets-{name}",
                [left_facet, right_frame],
                **_compare_params(left, right),
            )
        )
    mismatch = grid(rng, roles={"donor": DONORS[1:2]}, periods=YEARS[:3], facets="mm_b")
    cases.append(
        _case(
            "compare",
            "facets-mismatch",
            [left_facet, mismatch],
            **_compare_params(left, right),
        )
    )
    for policy in ("request", "two_requests", "usd_or_request"):
        frame = grid(rng, roles={"donor": DONORS[:2]}, periods=YEARS[:3], unit=policy)
        cases.append(
            _case("compare", f"converted-{policy}", [frame], **_compare_params(left, right))
        )

    for index in range(40):
        left_frame = _random_grid(rng, **_computing_overrides(rng))
        frames = [left_frame]
        right_frame = left_frame
        if rng.random() < 0.4:
            right_frame = _random_grid(
                rng,
                stat_vars=(rng.choice((SV_A, SV_B)),),
                periods=_column_values(left_frame, "time_period") or list(YEARS[:3]),
                **_computing_overrides(rng),
            )
            frames.append(right_frame)
        cases.append(
            _case(
                "compare",
                f"random-{index:03d}",
                frames,
                **_compare_params(
                    _random_binding(rng, left_frame),
                    _random_binding(rng, right_frame),
                    left_pin=_random_pin(rng, left_frame),
                    right_pin=_random_pin(rng, right_frame),
                ),
            )
        )

    for left_unit, right_unit in _CROSS_SIDE_CONVERTED_UNITS:
        left_frame = grid(rng, roles={"donor": DONORS[:1]}, periods=YEARS[:3], unit=left_unit)
        right_frame = grid(rng, roles={"donor": DONORS[1:2]}, periods=YEARS[:3], unit=right_unit)
        cases.append(
            _case(
                "compare",
                f"cross-units-{left_unit}-{right_unit}",
                [left_frame, right_frame],
                **_compare_params(left, right),
            )
        )
    return cases


def _compare_params(
    left: Mapping[str, str],
    right: Mapping[str, str],
    *,
    left_pin: str | None = None,
    right_pin: str | None = None,
) -> dict[str, Any]:
    return {
        "stat_var": "dcid:ops_" + "d" * 32,
        "left_entity": dict(left),
        "right_entity": dict(right),
        "left_pinned_unit": left_pin,
        "right_pinned_unit": right_pin,
    }


def growth_cases(rng: random.Random) -> list[Case]:
    """`run_growth(frame, window_start=, window_end=, entity_by=, change=,
    pinned_unit=)`."""
    cases = []

    def growth(name: str, frame: Spec, **params: Any) -> None:
        arguments = {
            "window_start": YEARS[0],
            "window_end": YEARS[3],
            "entity_by": ["donor"],
            "change": "relative",
            "pinned_unit": None,
        }
        arguments.update(params)
        cases.append(_case("growth", name, [frame], **arguments))

    def series(values_by_donor: Mapping[str, Sequence[float]]) -> Spec:
        return rows_frame(
            ["donor"],
            [
                {"donor": donor, "time_period": YEARS[index], "value": value}
                for donor, values in values_by_donor.items()
                for index, value in enumerate(values)
            ],
        )

    bases = {
        "zero-start": (0.0, 1.5, 2.5, 3.5),
        "zero-end": (1.5, 2.5, 3.5, 0.0),
        "sign-crossing": (-1.25, 2.5, 3.5, 4.75),
        "equal-negative": (-3.141592653589793, 1.0, 2.0, -3.141592653589793),
        "equal-positive": (2.718281828459045, 1.0, 2.0, 2.718281828459045),
        "tiny": (1e-300, 1.0, 2.0, 1e300),
    }
    for name, values in bases.items():
        for change in ("relative", "absolute"):
            growth(f"{name}-{change}", series({DONORS[0]: values}), change=change)
    growth(
        "overflow-absolute", series({DONORS[0]: (-NEAR_MAX, 0.0, 0.0, NEAR_MAX)}), change="absolute"
    )
    gappy = rows_frame(
        ["donor"],
        [
            {"donor": DONORS[0], "time_period": year, "value": 1.0 + index / 7}
            for index, year in enumerate(YEARS[:8])
            if year not in (YEARS[2], YEARS[5])
        ],
    )
    growth("gap-years", gappy, window_end=YEARS[7])
    growth("empty-window", gappy, window_start="1990", window_end="1995")
    growth("empty-window-no-entity", gappy, window_start="1990", window_end="1991", entity_by=[])
    multi = grid(rng, roles={"donor": DONORS[:3], "recipient": RECIPIENTS[:2]}, periods=YEARS[:4])
    growth("ambiguous-entity", multi)
    growth("two-role-entity", multi, entity_by=["recipient", "donor"])
    growth("invalid-entity-by", multi, entity_by=["time_period"])
    growth(
        "two-stat-vars",
        grid(rng, roles={"donor": DONORS[:2]}, periods=YEARS[:4], stat_vars=(SV_A, SV_B)),
    )
    monthly = grid(rng, roles={"donor": DONORS[:2]}, periods=MONTHS)
    growth("monthly-periods", monthly, window_start="2020", window_end="2021")
    many = grid(
        rng,
        roles={"donor": DONORS[:25]},
        periods=YEARS[:6],
        values="positive",
        unit="usd",
    )
    growth("outside-window-over-20", many, window_start=YEARS[1], window_end=YEARS[3])
    for policy in ("usd_or_null", "request", "two_requests"):
        frame = grid(rng, roles={"donor": DONORS[:2]}, periods=YEARS[:4], unit=policy)
        growth(f"units-{policy}", frame, change="absolute")

    for index in range(34):
        frame = _random_grid(
            rng,
            periods=list(YEARS[rng.randint(0, 3) : rng.randint(6, 10)]),
            density=rng.choice((1.0, 1.0, 1.0, 0.9)),
        )
        years = sorted({period[:4] for period in _column_values(frame, "time_period")}) or [
            YEARS[0]
        ]
        start = rng.choice(years)
        end = rng.choice([year for year in years if year >= start])
        roles = frame["role_keys"]
        entity_by = rng.choice((roles, roles[:1], [], roles[::-1]))
        growth(
            f"random-{index:03d}",
            frame,
            window_start=start,
            window_end=end,
            entity_by=list(entity_by),
            change=rng.choice(("relative", "relative", "absolute")),
            pinned_unit=_random_pin(rng, frame),
        )
    return cases


def rank_cases(rng: random.Random) -> list[Case]:
    """`run_rank(frame, rank_by=, aggregation=, window_start=, window_end=,
    ascending=, pinned_unit=, per_period=, limit=)`."""
    cases = []

    def rank(name: str, frame: Spec, **params: Any) -> None:
        arguments = {
            "rank_by": ["donor"],
            "aggregation": None,
            "window_start": None,
            "window_end": None,
            "ascending": False,
            "pinned_unit": None,
            "per_period": False,
            "limit": None,
        }
        arguments.update(params)
        cases.append(_case("rank", name, [frame], **arguments))

    tied = rows_frame(
        ["donor"],
        [
            {"donor": donor, "time_period": "2020", "value": value}
            for donor, value in zip(
                (DONORS[4], DONORS[2], DONORS[0], DONORS[3], DONORS[1], DONORS[5]),
                (0.0, 2.5, 0.0, 2.5, -1.75, 0.0),
                strict=True,
            )
        ],
    )
    for ascending in (False, True):
        rank(f"ties-with-zeros-ascending-{ascending}", tied, ascending=ascending)
        rank(
            f"ties-per-period-ascending-{ascending}",
            tied,
            ascending=ascending,
            per_period=True,
        )
        rank(f"limit-at-a-tie-ascending-{ascending}", tied, ascending=ascending, limit=2)
        rank(
            f"limit-per-period-ascending-{ascending}",
            tied,
            ascending=ascending,
            per_period=True,
            limit=1,
        )
    overflowing = rows_frame(
        ["donor"],
        [
            {"donor": donor, "time_period": year, "value": NEAR_MAX}
            for donor in (DONORS[2], DONORS[1])
            for year in YEARS[:2]
        ],
    )
    rank("two-overflowing-groups", overflowing, aggregation="sum")
    cancelling = _cancelling_frame((CANCELLATION, CANCELLATION_TO_ONE, TIE_BREAK, (2.0, 0.5)))
    for aggregation in AGGREGATIONS:
        rank(f"cancellation-{aggregation}", cancelling, aggregation=aggregation)
    rank("ambiguous-without-aggregation", cancelling)
    monthly = grid(rng, roles={"donor": DONORS[:3]}, periods=MONTHS, values="ties")
    rank("monthly-window", monthly, aggregation="mean", window_start="2020", window_end="2020")
    rank("monthly-per-period", monthly, aggregation="max", per_period=True)
    for ascending in (False, True):
        rank(
            f"limit-per-period-across-periods-ascending-{ascending}",
            monthly,
            aggregation="max",
            ascending=ascending,
            per_period=True,
            limit=1,
        )
    rank("empty-window", monthly, aggregation="sum", window_start="1990", window_end="1991")
    rank("empty-frame", frame_spec(["donor"], []))
    sparse = grid(
        rng, roles={"donor": DONORS[:6]}, periods=YEARS[:4], density=0.6, unit="usd_or_null"
    )
    rank("sparse-unequal-coverage", sparse, aggregation="sum")
    rank("sparse-unequal-coverage-pinned", sparse, aggregation="sum", pinned_unit=USD)
    many = _many_entities_frame(rng, count=26)
    rank(
        "outside-window-over-20",
        many,
        aggregation="min",
        window_start=YEARS[6],
        window_end=YEARS[7],
    )
    rank("limit-drops-over-20", many, aggregation="min", limit=3)
    rank("limit-beyond-the-population", many, aggregation="min", limit=30)
    rank("unsupported-aggregation", many, aggregation="median")
    rank("invalid-rank-by", many, rank_by=["unit"])
    for policy in ("request", "two_requests", "usd_or_request"):
        frame = grid(rng, roles={"donor": DONORS[:3]}, periods=YEARS[:1], unit=policy)
        rank(f"converted-{policy}", frame)

    for index in range(36):
        frame = _random_grid(rng)
        roles = frame["role_keys"]
        window: tuple[str | None, str | None] = (None, None)
        if rng.random() < 0.5:
            years = sorted({period[:4] for period in _column_values(frame, "time_period")})
            start = rng.choice(years)
            window = (start, rng.choice([year for year in years if year >= start]))
        rank(
            f"random-{index:03d}",
            frame,
            rank_by=list(rng.choice((roles, roles[:1], roles[::-1]))),
            aggregation=rng.choice((None, None, *AGGREGATIONS)),
            window_start=window[0],
            window_end=window[1],
            ascending=rng.random() < 0.5,
            pinned_unit=_random_pin(rng, frame),
            per_period=rng.random() < 0.35,
        )
    return cases


def share_of_cases(rng: random.Random) -> list[Case]:
    """`run_share_of(numerator, denominator, stat_var=, numerator_entity=,
    denominator_entity=, numerator_by=, denominator_by=,
    numerator_pinned_unit=, denominator_pinned_unit=)`."""
    cases = []

    def share(name: str, frames: Sequence[Spec], **params: Any) -> None:
        arguments: dict[str, Any] = {
            "stat_var": "dcid:ops_" + "c" * 32,
            "numerator_entity": {"donor": DONORS[0]},
            "denominator_entity": {},
            "numerator_by": [],
            "denominator_by": {},
            "numerator_pinned_unit": None,
            "denominator_pinned_unit": None,
        }
        arguments.update(params)
        cases.append(_case("share_of", name, frames, **arguments))

    pooled = grid(rng, roles={"donor": DONORS[:4]}, periods=YEARS[:3], values="positive")
    share("pooled", [pooled])
    zero_denominator = rows_frame(
        ["donor"],
        [
            {"donor": DONORS[0], "time_period": "2020", "value": 0.7},
            {"donor": DONORS[1], "time_period": "2020", "value": -0.7},
        ],
    )
    share("zero-denominator", [zero_denominator])
    negative_denominator = rows_frame(
        ["donor"],
        [
            {"donor": DONORS[0], "time_period": "2020", "value": 0.0},
            {"donor": DONORS[1], "time_period": "2020", "value": -4.2},
        ],
    )
    share("signed-zero-share", [negative_denominator])
    cancelling = rows_frame(
        ["donor"],
        [
            {"donor": DONORS[index], "time_period": "2020", "value": value}
            for index, value in enumerate(CANCELLATION)
        ],
    )
    share("cancellation-denominator", [cancelling], numerator_entity={"donor": DONORS[1]})
    tie_break = rows_frame(
        ["donor"],
        [
            {"donor": DONORS[index], "time_period": "2020", "value": value}
            for index, value in enumerate(TIE_BREAK)
        ],
    )
    share("tie-break-denominator", [tie_break], numerator_entity={"donor": DONORS[2]})
    overflow = rows_frame(
        ["donor"],
        [{"donor": DONORS[index], "time_period": "2020", "value": NEAR_MAX} for index in range(2)],
    )
    share("overflow-denominator", [overflow])
    tiny = rows_frame(
        ["donor"],
        [
            {"donor": DONORS[0], "time_period": "2020", "value": 1e300},
            {"donor": DONORS[1], "time_period": "2020", "value": -1e300},
            {"donor": DONORS[2], "time_period": "2020", "value": 1e-300},
        ],
    )
    share("overflow-ratio", [tiny], denominator_entity={"donor": DONORS[2]})
    gappy = grid(rng, roles={"donor": DONORS[:3]}, periods=YEARS[:4], density=0.7)
    share("grain-mismatch", [gappy])
    share(
        "no-rows-either-side",
        [gappy],
        numerator_entity={"donor": "dcid:NotPresent"},
        denominator_entity={"donor": "dcid:NotPresent"},
    )
    keyed = grid(
        rng,
        roles={"donor": DONORS[:3], "recipient": RECIPIENTS[:3]},
        periods=YEARS[:3],
        values="positive",
    )
    share("keyed-pooled", [keyed], numerator_by=["recipient"])
    share(
        "keyed-by-recipient",
        [keyed],
        numerator_by=["recipient"],
        denominator_by={"recipient": "recipient"},
    )
    share(
        "keyed-unknown",
        [keyed],
        numerator_by=["recipient", "donor"],
        denominator_by={"recipient": "recipient"},
    )
    denominator = grid(rng, roles={"area": RECIPIENTS[1:3]}, periods=YEARS[:3], values="positive")
    share(
        "keyed-unmatched",
        [keyed, denominator],
        numerator_by=["recipient"],
        denominator_by={"recipient": "area"},
    )
    share(
        "keyed-no-rows",
        [keyed],
        numerator_entity={"donor": "dcid:NotPresent"},
        numerator_by=["recipient"],
    )
    many = _many_entities_frame(rng, count=22, unit="usd_or_null")
    share("unitless-over-20", [many])
    for left_policy, right_policy in (("usd", "pct_only"), ("null", "usd"), ("usd", "null")):
        numerator = grid(rng, roles={"donor": DONORS[:2]}, periods=YEARS[:2], unit=left_policy)
        denominator = grid(rng, roles={"donor": DONORS[:2]}, periods=YEARS[:2], unit=right_policy)
        share(f"units-{left_policy}-{right_policy}", [numerator, denominator])
    numerator_facet = grid(rng, roles={"donor": DONORS[:2]}, periods=YEARS[:2], facets="mm")
    for facets in ("mm_split", "prov_partial", "both"):
        denominator = grid(rng, roles={"donor": DONORS[:2]}, periods=YEARS[:2], facets=facets)
        share(f"facets-{facets}", [numerator_facet, denominator])
    for policy in ("request", "two_requests", "usd_or_request"):
        frame = grid(rng, roles={"donor": DONORS[:3]}, periods=YEARS[:2], unit=policy)
        share(f"converted-{policy}", [frame])

    for index in range(40):
        numerator = _random_grid(rng, **_computing_overrides(rng))
        frames = [numerator]
        denominator = numerator
        if rng.random() < 0.35:
            denominator = _random_grid(
                rng,
                periods=_column_values(numerator, "time_period") or list(YEARS[:3]),
                **_computing_overrides(rng),
            )
            frames.append(denominator)
        numerator_roles = numerator["role_keys"]
        numerator_by = rng.choice(([], [], numerator_roles[-1:]))
        denominator_by: dict[str, str] = {}
        if numerator_by and rng.random() < 0.6:
            candidates = denominator["role_keys"] or ["recipient"]
            denominator_by = {numerator_by[0]: rng.choice(candidates)}
        share(
            f"random-{index:03d}",
            frames,
            numerator_entity=_random_binding(rng, numerator),
            denominator_entity=rng.choice(({}, {}, _random_binding(rng, denominator))),
            numerator_by=list(numerator_by),
            denominator_by=denominator_by,
            numerator_pinned_unit=_random_pin(rng, numerator),
            denominator_pinned_unit=_random_pin(rng, denominator),
        )

    for left_policy, right_policy in _CROSS_SIDE_CONVERTED_UNITS:
        numerator = grid(rng, roles={"donor": DONORS[:2]}, periods=YEARS[:2], unit=left_policy)
        denominator = grid(rng, roles={"donor": DONORS[:2]}, periods=YEARS[:2], unit=right_policy)
        share(f"cross-units-{left_policy}-{right_policy}", [numerator, denominator])
    return cases


def join_cases(rng: random.Random) -> list[Case]:
    """`run_join(left, right, join_keys=, left_pinned_unit=, right_pinned_unit=)`."""
    cases = []

    def join(name: str, frames: Sequence[Spec], **params: Any) -> None:
        arguments: dict[str, Any] = {
            "join_keys": [],
            "left_pinned_unit": None,
            "right_pinned_unit": None,
        }
        arguments.update(params)
        cases.append(_case("join", name, frames, **arguments))

    def pair(**overrides: Any) -> tuple[Spec, Spec]:
        arguments: dict[str, Any] = {"roles": {"donor": DONORS[:4]}, "periods": YEARS[:4]}
        arguments.update(overrides)
        left = grid(rng, stat_vars=(SV_A,), **arguments)
        right = grid(rng, stat_vars=(SV_B,), **arguments)
        return left, right

    join("matched", list(pair()))
    join("partial", list(pair(density=0.6)))
    join("collision", list(pair())[:1])
    join(
        "same-stat-var-different-unit",
        [pair()[0], grid(rng, roles={"donor": DONORS[:4]}, periods=YEARS[:4], unit="pct_only")],
    )
    two_role_left = grid(
        rng, roles={"donor": DONORS[:2], "recipient": RECIPIENTS[:2]}, periods=YEARS[:2]
    )
    two_role_right = grid(
        rng,
        roles={"donor": DONORS[:2], "recipient": RECIPIENTS[:2]},
        periods=YEARS[:2],
        stat_vars=(SV_B,),
    )
    join("repeated-key", [two_role_left, two_role_right], join_keys=["donor"])
    join("two-role-keys", [two_role_left, two_role_right])
    join("unknown-key", [two_role_left, two_role_right], join_keys=["time_period"])
    recipient_only = grid(
        rng, roles={"recipient": RECIPIENTS[:2]}, periods=YEARS[:2], stat_vars=(SV_B,)
    )
    join(
        "no-common-key", [grid(rng, roles={"donor": DONORS[:2]}, periods=YEARS[:2]), recipient_only]
    )
    world = grid(rng, roles={}, periods=YEARS[:3], stat_vars=(SV_B,))
    join("role-less-right", [grid(rng, roles={"donor": DONORS[:1]}, periods=YEARS[:3]), world])
    join(
        "role-less-right-ambiguous",
        [grid(rng, roles={"donor": DONORS[:2]}, periods=YEARS[:3]), world],
    )
    left_many = _many_entities_frame(rng, count=25)
    right_few = grid(rng, roles={"donor": DONORS[:3]}, periods=YEARS[5:8], stat_vars=(SV_B,))
    join("unmatched-over-20", [left_many, right_few])
    join("unitless-excluded", list(pair(unit="usd_or_null")))
    join("pinned", list(pair(unit="usd_or_pct")), left_pinned_unit=USD, right_pinned_unit=PCT)
    join("facets-partial", list(pair(facets="mm_partial")))
    join("facets-split", list(pair(facets="mm_split")))
    for policy in ("request", "two_requests", "usd_or_request"):
        join(f"converted-{policy}", list(pair(unit=policy)))

    for index in range(34):
        left = _random_grid(rng, allow_no_roles=True, stat_vars=(SV_A,))
        right_roles = _random_roles(rng, allow_none=True)
        if rng.random() < 0.6 and left["role_keys"]:
            right_roles = {role: _column_values(left, role) for role in left["role_keys"]}
        right = grid(
            rng,
            roles=right_roles,
            periods=_column_values(left, "time_period") or list(YEARS[:2]),
            stat_vars=(rng.choice((SV_B, SV_B, SV_A)),),
            density=rng.choice((1.0, 0.7)),
            unit=_weighted(rng, _UNIT_WEIGHTS),
        )
        common = sorted(set(left["role_keys"]) & set(right["role_keys"]))
        join_keys = rng.choice(([], [], common, common[:1]))
        join(
            f"random-{index:03d}",
            [left, right],
            join_keys=list(join_keys),
            left_pinned_unit=_random_pin(rng, left),
            right_pinned_unit=_random_pin(rng, right),
        )
    return cases


def filter_cases(rng: random.Random) -> list[Case]:
    """`run_filter(frame, predicates=)`, predicates as `FilterOpSpec` JSON."""
    cases = []

    def filter_case(name: str, frame: Spec, predicates: Sequence[Mapping[str, Any]]) -> None:
        cases.append(_case("filter", name, [frame], predicates=[dict(p) for p in predicates]))

    frame = grid(
        rng,
        roles={"donor": DONORS[:4], "recipient": RECIPIENTS[:2]},
        periods=YEARS[:4],
        unit="mixed",
        facets="mm_partial",
        values="signed",
    )
    filter_case("no-predicates", frame, [])
    filter_case(
        "empty-result",
        frame,
        [{"kind": "ordered_threshold", "column": "value", "op": "gt", "value": 1e300}],
    )
    filter_case(
        "zero-eq",
        frame,
        [{"kind": "ordered_threshold", "column": "value", "op": "eq", "value": 0.0}],
    )
    filter_case(
        "negative-zero-operand",
        frame,
        [{"kind": "ordered_threshold", "column": "value", "op": "lte", "value": -0.0}],
    )
    filter_case(
        "period-threshold",
        frame,
        [{"kind": "ordered_threshold", "column": "time_period", "op": "gte", "value": YEARS[2]}],
    )
    filter_case(
        "unit-category", frame, [{"kind": "category_match", "column": "unit", "values": [USD]}]
    )
    filter_case(
        "facet-category",
        frame,
        [{"kind": "category_match", "column": "measurementMethod", "values": [MM_A]}],
    )
    filter_case(
        "conjunction",
        frame,
        [
            {"kind": "category_match", "column": "donor", "values": list(DONORS[1:3])},
            {"kind": "ordered_threshold", "column": "value", "op": "ne", "value": 0.0},
            {"kind": "ordered_threshold", "column": "time_period", "op": "lt", "value": YEARS[3]},
        ],
    )
    filter_case(
        "unknown-column", frame, [{"kind": "category_match", "column": "sector", "values": ["x"]}]
    )
    filter_case(
        "kind-mismatch",
        frame,
        [{"kind": "ordered_threshold", "column": "donor", "op": "gt", "value": "a"}],
    )
    filter_case(
        "operand-mismatch",
        frame,
        [{"kind": "ordered_threshold", "column": "value", "op": "gt", "value": "2020"}],
    )
    filter_case(
        "status-column", frame, [{"kind": "category_match", "column": "status", "values": ["x"]}]
    )
    many = _many_entities_frame(rng, count=30)
    filter_case(
        "predicate-over-20",
        many,
        [{"kind": "category_match", "column": "donor", "values": [DONORS[29]]}],
    )
    monthly = grid(rng, roles={"donor": DONORS[:2]}, periods=(*MONTHS, "2020", "2021-06-15"))
    filter_case(
        "mixed-grain-period",
        monthly,
        [{"kind": "ordered_threshold", "column": "time_period", "op": "gt", "value": "2020"}],
    )
    for policy in ("request", "usd_or_request", "two_requests"):
        converted = grid(rng, roles={"donor": DONORS[:3]}, periods=YEARS[:2], unit=policy)
        filter_case(
            f"converted-{policy}-unit-category",
            converted,
            [{"kind": "category_match", "column": "unit", "values": [USD]}],
        )
        filter_case(
            f"converted-{policy}-keep",
            converted,
            [{"kind": "ordered_threshold", "column": "value", "op": "gt", "value": -1e308}],
        )
        filter_case(
            f"converted-{policy}-drop-all",
            converted,
            [{"kind": "ordered_threshold", "column": "value", "op": "gt", "value": 1e308}],
        )

    for index in range(36):
        frame = _random_grid(rng, allow_no_roles=True)
        predicates = [_random_predicate(rng, frame) for _ in range(rng.randint(1, 3))]
        filter_case(f"random-{index:03d}", frame, predicates)
    return cases


def _random_predicate(rng: random.Random, spec: Spec) -> dict[str, Any]:
    values = _column_values(spec, "value") or [0.0]
    periods = _column_values(spec, "time_period") or ["2020"]
    roll = rng.random()
    if roll < 0.4:
        operand = rng.choice(values) if rng.random() < 0.5 else _nonround(rng)
        return {
            "kind": "ordered_threshold",
            "column": "value",
            "op": rng.choice(("gt", "gte", "lt", "lte", "eq", "ne")),
            "value": operand,
        }
    if roll < 0.55:
        return {
            "kind": "ordered_threshold",
            "column": "time_period",
            "op": rng.choice(("gt", "gte", "lt", "lte", "eq", "ne")),
            "value": rng.choice(periods),
        }
    columns = [*spec["role_keys"], "stat_var", "unit", "measurementMethod", "time_period"]
    column = rng.choice(columns)
    present = [value for value in _column_values(spec, column) if isinstance(value, str)]
    chosen = rng.sample(present, rng.randint(0, len(present))) if present else []
    if rng.random() < 0.3:
        chosen.append("dcid:NotPresent")
    return {"kind": "category_match", "column": column, "values": chosen or ["dcid:NotPresent"]}


VERB_GENERATORS: dict[str, Callable[[random.Random], list[Case]]] = {
    "aggregate": aggregate_cases,
    "compare": compare_cases,
    "growth": growth_cases,
    "rank": rank_cases,
    "share_of": share_of_cases,
    "join": join_cases,
    "filter": filter_cases,
}


def generate_cases() -> list[Case]:
    """Every case, verb by verb, each verb drawing from its own `SEED`-derived
    stream so that adding a case to one verb leaves the others unchanged."""
    cases: list[Case] = []
    for offset, generator in enumerate(VERB_GENERATORS.values()):
        cases.extend(generator(random.Random(SEED + offset)))
    identifiers = [case["id"] for case in cases]
    duplicated = sorted({i for i in identifiers if identifiers.count(i) > 1})
    if duplicated:
        raise ValueError(f"duplicate case ids: {duplicated}")
    return cases

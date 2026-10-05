"""Tests for `compiler.arguments`: does a set of arguments fit a tool's own schema?"""

from __future__ import annotations

from compiler.arguments import (
    literal_argument_errors,
    undeclared_parameters,
    unusable_schema_reason,
)
from compiler.surface import ToolSchema

Schemas = dict[str, ToolSchema]


def test_a_usable_schema_has_no_reason_to_refuse(schemas: Schemas) -> None:
    assert unusable_schema_reason(schemas["aggregate"].parameters) is None


def test_an_unusable_schema_is_reported_rather_than_raised() -> None:
    # A tool schema is untrusted input from another service; one jsonschema
    # itself rejects would otherwise raise part way through a check.
    reason = unusable_schema_reason({"type": "object", "properties": "nonsense"})

    assert reason is not None
    assert "nonsense" in reason


def test_undeclared_parameters_are_named_and_sorted(schemas: Schemas) -> None:
    assert undeclared_parameters(
        ["spec_id", "per_capita", "frame_ref"], schemas["get_data"].parameters
    ) == [
        "frame_ref",
        "per_capita",
    ]


def test_a_schema_declaring_no_arguments_declares_nothing() -> None:
    assert undeclared_parameters(["frame_ref"], {}) == ["frame_ref"]


def test_complete_literal_arguments_pass(schemas: Schemas) -> None:
    assert (
        literal_argument_errors(
            arguments={"spec_id": "spec-abc"}, schema=schemas["get_data"].parameters
        )
        == []
    )


def test_a_bound_argument_is_not_reported_as_a_missing_required_one(schemas: Schemas) -> None:
    # `frame_ref` is required by the schema and supplied by a binding at run
    # time. Validating it as a literal would fail every real plan, so it has
    # to be held out of `required` as well as out of the payload.
    errors = literal_argument_errors(
        arguments={"aggregation": "sum"},
        schema=schemas["aggregate"].parameters,
        bound=["frame_ref"],
    )

    assert errors == []


def test_a_missing_required_argument_is_reported(schemas: Schemas) -> None:
    errors = literal_argument_errors(
        arguments={}, schema=schemas["aggregate"].parameters, bound=["frame_ref"]
    )

    assert errors == ["'aggregation' is a required property"]


def test_a_wrongly_typed_argument_is_reported_at_its_location(schemas: Schemas) -> None:
    errors = literal_argument_errors(
        arguments={"spec_id": 12345}, schema=schemas["get_data"].parameters
    )

    assert errors == ["spec_id: 12345 is not of type 'string'"]


def test_an_invented_argument_is_reported(schemas: Schemas) -> None:
    # The real schemas carry `additionalProperties: false`, which is what
    # turns a plausible-sounding but undeclared argument into a failure.
    errors = literal_argument_errors(
        arguments={"spec_id": "spec-abc", "per_capita": True}, schema=schemas["get_data"].parameters
    )

    assert errors == ["Additional properties are not allowed ('per_capita' was unexpected)"]


def test_several_errors_report_in_a_stable_order(schemas: Schemas) -> None:
    # jsonschema's own error order is unspecified, and these messages are
    # what a retry gets handed. Sorted by location so a second run reports
    # the same sequence.
    errors = literal_argument_errors(
        arguments={"frame_ref": {"spec_id": "s"}, "aggregation": 123, "per_capita": True},
        schema=schemas["aggregate"].parameters,
    )

    assert errors == [
        "Additional properties are not allowed ('per_capita' was unexpected)",
        "aggregation: 123 is not of type 'string'",
    ]

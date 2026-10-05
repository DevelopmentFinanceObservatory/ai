"""Fields a content hash covers once they are set.

A hash over a serialised model moves for every saved record when the model
gains a field, since a record saved without the field reloads with it at its
default. `HASHED_WHEN_SET` marks such a field in its own annotation, and
`revision_content` is the model's JSON content without the marked fields that
still hold their default, so a saved record hashes as it did. Each hashed
surface pins the fields it hashes unconditionally against recorded data:
`planner.plan.plan_revision` in the harness's
`tests/fixtures/revision_base_fields.json`, and analysis-mcp's `op_spec_hash`
in its `op_spec_hash_vectors.json`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final, override

from pydantic import BaseModel


class _HashedWhenSet:
    """The type of `HASHED_WHEN_SET`; see there."""

    __slots__ = ()

    @override
    def __repr__(self) -> str:
        return "HASHED_WHEN_SET"


# Marks a field added to a hashed model after its base format. Declared in
# the field's own annotation, with a default:
# `limit: Annotated[int | None, HASHED_WHEN_SET] = Field(default=None)`.
HASHED_WHEN_SET: Final = _HashedWhenSet()


def revision_content(
    model: BaseModel, *, include: set[str] | None = None, exclude: set[str] | None = None
) -> dict[str, Any]:
    """Return the JSON content of `model` that a content hash covers.

    `model.model_dump(mode="json")` narrowed to `include` or without
    `exclude`, less every `HASHED_WHEN_SET` field holding its default, at any
    depth.
    """
    content = model.model_dump(mode="json", include=include, exclude=exclude)
    _drop_unset(model, content)
    return content


def _drop_unset(value: object, dumped: Any) -> None:
    """Delete from `dumped`, the JSON dump of `value`, each `HASHED_WHEN_SET` field at its default."""
    if isinstance(value, BaseModel):
        for name, field in type(value).model_fields.items():
            if name not in dumped:
                continue
            held = getattr(value, name)
            added = any(isinstance(item, _HashedWhenSet) for item in field.metadata)
            if added and held == field.get_default(call_default_factory=True):
                del dumped[name]
            else:
                _drop_unset(held, dumped[name])
    elif isinstance(value, Mapping):
        for held, dumped_item in zip(value.values(), dumped.values(), strict=True):
            _drop_unset(held, dumped_item)
    elif isinstance(value, (tuple, list)):
        for held, dumped_item in zip(value, dumped, strict=True):
            _drop_unset(held, dumped_item)

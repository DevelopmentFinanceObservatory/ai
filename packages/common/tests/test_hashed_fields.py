"""Tests for `common.hashed_fields`: fields a content hash covers once they are set."""

from __future__ import annotations

from typing import Annotated

from common.hashed_fields import HASHED_WHEN_SET, revision_content
from pydantic import BaseModel, Field


class _Inner(BaseModel):
    name: str
    limit: Annotated[int | None, HASHED_WHEN_SET] = None


class _Outer(BaseModel):
    id: str
    note: str | None = None
    tags: Annotated[tuple[str, ...], HASHED_WHEN_SET] = ()
    flag: Annotated[bool, HASHED_WHEN_SET] = Field(default=False)
    inner: _Inner
    items: tuple[_Inner, ...] = ()
    by_key: dict[str, _Inner] = Field(default_factory=dict)


def test_a_marked_field_at_its_default_is_left_out_at_any_depth() -> None:
    outer = _Outer(
        id="a",
        inner=_Inner(name="x"),
        items=(_Inner(name="y"), _Inner(name="z", limit=3)),
        by_key={"k": _Inner(name="w")},
    )

    assert revision_content(outer) == {
        "id": "a",
        "note": None,
        "inner": {"name": "x"},
        "items": [{"name": "y"}, {"name": "z", "limit": 3}],
        "by_key": {"k": {"name": "w"}},
    }


def test_a_marked_field_enters_once_it_holds_something_else() -> None:
    outer = _Outer(id="a", tags=("t",), flag=True, inner=_Inner(name="x", limit=1))

    content = revision_content(outer)

    assert content["tags"] == ["t"]
    assert content["flag"] is True
    assert content["inner"] == {"name": "x", "limit": 1}


def test_an_unmarked_field_at_its_default_stays() -> None:
    assert revision_content(_Outer(id="a", inner=_Inner(name="x")))["note"] is None


def test_include_and_exclude_narrow_the_content() -> None:
    outer = _Outer(id="a", inner=_Inner(name="x"))

    assert revision_content(outer, include={"id", "inner"}) == {"id": "a", "inner": {"name": "x"}}
    assert "id" not in revision_content(outer, exclude={"id"})


def test_the_marker_leaves_the_json_schema_as_it_is() -> None:
    class _Plain(BaseModel):
        name: str
        limit: int | None = None

    marked = _Inner.model_json_schema()
    plain = _Plain.model_json_schema()

    assert marked["properties"] == plain["properties"]

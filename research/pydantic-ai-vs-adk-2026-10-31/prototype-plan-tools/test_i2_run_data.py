"""I2: `run_plan` returns the data, and frame references the public tools read.

A finished run answers `data`: per frame the hand-back lists, its id, the
`{spec_id, content_version}` pair, and its rows as `outward.table` lays them out
while every frame's rows stay within `plan_tools.ROW_CAP` rows and
`ROW_CHARS_CAP` characters; past either, its row count alone.

The pair is held to the public tools themselves, in process: `export_data`,
called on data-mcp's own server, reaches dfo-apis with it (a recorded upstream
answers), and `describe_frame`'s own check (`presentation_mcp.client.
_frame_ref_or_refusal`) admits it. The frame id the first version handed back is
refused by both.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

import pytest
from checks import SCRIPTED_OPENING, call, client_of, scripted_server
from data_mcp import client_holder
from data_mcp.errors import UpstreamError
from data_mcp.server import mcp as data_mcp
from fastmcp import Client
from presentation_mcp.client import _frame_ref_or_refusal
from scripted import PROPOSAL

# One row per year of the scripted plan's mean by period, 1960 to 2023.
ROWS = [
    {"stat_var": "dcid:LifeExpectancy_Person", "time_period": str(year), "value": 60.0 + i / 10}
    for i, year in enumerate(range(1960, 2024))
]


async def _rows(ref: Mapping[str, Any]) -> list[dict[str, Any]]:
    assert str(ref["spec_id"]).startswith("ops_"), ref
    return ROWS


async def _finished(**options: Any) -> dict[str, Any]:
    scripted = scripted_server(read_rows=_rows, **options)
    async with client_of(scripted.server) as client:
        opened = await call(client, "start_plan", SCRIPTED_OPENING)
        session = opened["plan_session"]
        await call(client, "propose_plan", {"plan_session": session, **PROPOSAL})
        ran = await call(
            client, "run_plan", {"plan_session": session, "version": 1, "approved": True}
        )
    assert ran["status"] == "finished", ran
    return ran


async def test_i2_a_small_result_carries_its_rows_and_its_pair() -> None:
    ran = await _finished(row_cap=100)
    (frame,) = ran["data"]
    print("I2 data:", {k: v for k, v in frame.items() if k != "rows"}, len(frame["rows"]), "rows")
    assert frame["frame"] == ran["hand_back"]["frames"][0]["frame"]
    assert set(frame["frame_ref"]) == {"spec_id", "content_version"}
    assert frame["fixed"] == {"stat_var": "dcid:LifeExpectancy_Person"}
    assert frame["columns"] == ["time_period", "value"]
    assert frame["rows"][0] == ["1960", 60.0] and len(frame["rows"]) == len(ROWS)


async def test_i2_control_a_result_past_the_cap_carries_its_row_count_alone() -> None:
    ran = await _finished()  # 64 rows against the default cap of 60
    (frame,) = ran["data"]
    print("I2 control:", frame)
    assert "rows" not in frame
    assert frame["row_count"] == len(ROWS)
    assert set(frame["frame_ref"]) == {"spec_id", "content_version"}


class _RecordedUpstream:
    """data-mcp's upstream client, recording each request and answering 404."""

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    async def request(self, method: str, path: str, **kwargs: Any) -> Any:
        self.sent.append({"method": method, "path": path, "json": kwargs.get("json")})
        raise UpstreamError(status=404, code="frame_not_held", detail="recorded upstream")

    def caller_headers(self) -> dict[str, str]:
        return {}

    async def aclose(self) -> None:
        return None


@pytest.fixture
def upstream() -> Iterator[_RecordedUpstream]:
    recorded = _RecordedUpstream()
    client_holder.set_client(recorded)  # type: ignore[arg-type]
    yield recorded
    client_holder.set_client(None)


async def _export(frame_ref: Any) -> dict[str, Any]:
    async with Client(data_mcp) as client:
        result = await client.call_tool(
            "export_data", {"frame_ref": frame_ref}, raise_on_error=False
        )
    return dict(result.structured_content or {"text": result.content[0].text})


async def test_i2_the_public_tools_read_the_pair(upstream: _RecordedUpstream) -> None:
    ran = await _finished(row_cap=100)
    pair = ran["data"][0]["frame_ref"]
    exported = await _export(pair)
    described = _frame_ref_or_refusal(pair)
    print("I2 export:", exported.get("code"), upstream.sent, "describe:", described)
    assert upstream.sent and upstream.sent[0]["json"]["frame"] == pair
    assert described == (pair["spec_id"], pair["content_version"])


async def test_i2_control_the_public_tools_refuse_the_first_versions_frame_id(
    upstream: _RecordedUpstream,
) -> None:
    ran = await _finished(row_cap=100)
    frame_id = ran["data"][0]["frame"]
    by_id = await _export({"spec_id": frame_id})
    bare = await _export({"spec_id": ran["data"][0]["frame_ref"]["spec_id"]})
    described = _frame_ref_or_refusal({"spec_id": frame_id})
    print("I2 control export:", by_id.get("code"), bare.get("code"), "describe:", described)
    assert by_id.get("code") == "frame_ref_incomplete"
    assert bare.get("code") == "frame_ref_incomplete"
    assert not isinstance(described, tuple)
    assert upstream.sent == []

"""I3: a strict-mode client is not refused.

A client whose tools are strict (OpenAI's Responses API makes them strict unless
told otherwise) sends every optional field, filled with an empty value: `""`,
`0`, `[]`, `null`, or an object of such values. `strict` fills a reference plan
that way, by the schema `propose_plan` reads it against. Through plan-mcp it
lands as the plain reference plan lands; with the empty values passed through,
the planner refuses it. Every tool is listed with `strict: false` in its
`_meta`, at version 2.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest
import server as server_module
from checks import call, opening, recorded_server
from entry import load_tasks, quiet_logs
from fastmcp import Client
from outward import drop_unset

TASKS = {task.id: task for task in load_tasks()}
TASK = "growth-total-oda-ghana"


@pytest.fixture(autouse=True, scope="module")
def _quiet() -> None:
    quiet_logs()


def _resolve(node: Any, root: Mapping[str, Any]) -> Any:
    while isinstance(node, Mapping) and "$ref" in node:
        node = root["$defs"][node["$ref"].rsplit("/", 1)[-1]]
    return node


def _empty(node: Mapping[str, Any], root: Mapping[str, Any]) -> Any:
    """What a strict client sends for an optional field it has no value for."""
    node = _resolve(node, root)
    kind = node.get("type")
    if "enum" in node or "const" in node or "oneOf" in node or "anyOf" in node:
        return None
    if kind == "string":
        return ""
    if kind in ("integer", "number"):
        return 0
    if kind == "array":
        return []
    if kind == "object":
        return strict({}, node, root)
    return None


def strict(value: Any, schema: Mapping[str, Any], root: Mapping[str, Any]) -> Any:
    """`value` with every optional field the schema declares and `value` leaves out
    filled with `_empty`, at every depth."""
    node = _resolve(schema, root)
    if isinstance(value, list):
        return [strict(item, node.get("items") or {}, root) for item in value]
    if not isinstance(value, Mapping):
        return value
    branches = [_resolve(b, root) for b in node.get("oneOf") or node.get("anyOf") or []]
    for branch in branches:
        if ((branch.get("properties") or {}).get("kind") or {}).get("const") == value.get("kind"):
            node = branch
    properties = node.get("properties") or {}
    filled = {key: strict(item, properties.get(key) or {}, root) for key, item in value.items()}
    required = node.get("required") or ()
    for key, declared in properties.items():
        prop = _resolve(declared, root)
        if key not in filled and key not in required and prop.get("type") != "boolean":
            filled[key] = _empty(prop, root)
    return filled


async def _proposed(body: Mapping[str, Any]) -> tuple[dict[str, Any], Mapping[str, Any]]:
    task = TASKS[TASK]
    recorded = await recorded_server(task)
    async with Client(recorded.server.mcp) as client:
        opened = await call(client, "start_plan", opening(task))
        answer = await call(
            client, "propose_plan", {"plan_session": opened["plan_session"], **body}
        )
    return answer, recorded.server.schemas["propose_plan"]


async def test_i3_a_strict_client_lands_the_plan_a_plain_one_lands() -> None:
    reference = dict(TASKS[TASK].reference_payload)
    plain, schema = await _proposed(reference)
    filled = strict(reference, schema, schema)
    assert filled != reference
    assert drop_unset(filled, schema) == drop_unset(reference, schema)
    landed, _ = await _proposed(filled)
    print("I3:", sorted(set(filled) - set(reference)), landed["outcome"], landed.get("version"))
    assert "summary" in plain
    assert landed["summary"]["revision"] == plain["summary"]["revision"]


async def test_i3_control_the_empty_values_passed_through_are_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = dict(TASKS[TASK].reference_payload)
    _, schema = await _proposed(reference)
    monkeypatch.setattr(server_module, "drop_unset", lambda value, schema: dict(value))
    refused, _ = await _proposed(strict(reference, schema, schema))
    print("I3 control:", refused["outcome"], refused.get("code"), refused.get("message", "")[:120])
    assert refused["outcome"] != "ok"


def test_i3_a_zero_the_field_admits_is_kept() -> None:
    schema = {
        "type": "object",
        "properties": {
            "ask_ref": {"type": "integer", "minimum": 0},
            "amends_version": {"type": "integer"},
            "wait_s": {"type": "number", "minimum": 0},
            "base_year": {"type": "integer", "minimum": 1900},
        },
    }
    sent = {"ask_ref": 0, "amends_version": 0, "wait_s": 0, "base_year": 0}
    assert drop_unset(sent, schema) == {"ask_ref": 0, "wait_s": 0}


async def test_i3_every_tool_is_listed_not_strict_at_version_two() -> None:
    recorded = await recorded_server(TASKS[TASK])
    async with Client(recorded.server.mcp) as client:
        tools = await client.list_tools()
    for tool in tools:
        print("I3 listing:", tool.name, tool.meta)
        assert (tool.meta or {}).get("strict") is False
        assert (tool.meta or {}).get("fastmcp", {}).get("version") == "2"

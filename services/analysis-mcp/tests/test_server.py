"""Settings + server-scaffold tests: env-driven config and the FastMCP surface."""

from __future__ import annotations

import copy
import json
import logging

import httpx2
import jsonschema
import pytest
from analysis_mcp import (
    aggregate_tool,
    compare_tool,
    filter_tool,
    growth_tool,
    join_tool,
    quotient_of_tool,
    rank_tool,
    rolling_tool,
    share_of_tool,
    sum_of_tool,
    verbs,
)
from analysis_mcp import server as server_module
from analysis_mcp.aggregate_tool import aggregate
from analysis_mcp.compare_tool import compare
from analysis_mcp.fixtures import seed_fixture_store
from analysis_mcp.result import IMPLEMENTATION_VERSIONS, VERB_OUTPUT_SCHEMA
from analysis_mcp.rolling_tool import RESOLVE_ANCHOR_OUTPUT_SCHEMA, RESOLVE_ANCHOR_VERSION
from analysis_mcp.server import _build_store, _lifespan, _with_internal_error_backstop, mcp
from analysis_mcp.settings import Settings
from analysis_mcp.verbs import FOLLOWS_TREE
from common import refusal as refusal_module
from common.declaration import declared_tool
from common.materialisation import InMemoryMaterialisationStore, get_store, set_store
from fastmcp import Client, FastMCP
from fastmcp.tools import Tool

_ENV_VARS = ("PORT", "MODEL_NAME")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate tests from any of these vars set in the ambient environment."""
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def test_defaults_without_env() -> None:
    settings = Settings()

    assert settings.port == 8080
    assert settings.model_name == "gemini-3.8-flash"
    assert settings.frame_store_bucket is None
    assert settings.frame_store_verify_content_version is False


def test_reads_frame_store_settings_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FRAME_STORE_BUCKET", "my-bucket")
    monkeypatch.setenv("FRAME_STORE_VERIFY_CONTENT_VERSION", "true")

    settings = Settings()

    assert settings.frame_store_bucket == "my-bucket"
    assert settings.frame_store_verify_content_version is True


def test_reads_port_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PORT", "9090")

    settings = Settings()

    assert settings.port == 9090


async def test_server_exposes_one_tool_per_verb() -> None:
    async with Client(mcp) as client:
        tools = await client.list_tools()

    assert {tool.name for tool in tools} == {
        "aggregate",
        "compare",
        "growth",
        "rank",
        "share_of",
        "join",
        "filter",
        "sum_of",
        "quotient_of",
        "resolve_anchor",
    }


async def test_every_verb_declares_its_implementation_version_and_output_schema() -> None:
    """Read off the registered `FunctionTool` objects directly
    (`run_middleware=False`), not through a `Client`, since a client-facing
    `list_tools()` dereferences `$ref`/`$defs` at serve time -- the
    registered schema itself is what must equal `VERB_OUTPUT_SCHEMA`.
    """
    tools = {tool.name: tool for tool in await mcp.list_tools(run_middleware=False)}
    # The one tool that is no verb states its own result and version.
    anchor_tool = tools.pop("resolve_anchor")

    assert anchor_tool.version == RESOLVE_ANCHOR_VERSION
    assert anchor_tool.output_schema == RESOLVE_ANCHOR_OUTPUT_SCHEMA
    assert set(tools) == set(IMPLEMENTATION_VERSIONS)
    for tool in tools.values():
        assert tool.version == IMPLEMENTATION_VERSIONS[tool.name]
        assert tool.output_schema == VERB_OUTPUT_SCHEMA


async def test_a_real_successful_calls_structured_content_validates_against_its_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The check the MCP client runs on every successful call
    (`mcp.client.session`'s own `structuredContent`/`outputSchema`
    validation), reproduced directly against a real call's payload.
    """
    store = InMemoryMaterialisationStore()
    handles = await seed_fixture_store(store)

    async def _stub_build_store(_settings: object) -> InMemoryMaterialisationStore:
        return store

    monkeypatch.setattr(server_module, "_build_store", _stub_build_store)

    async with Client(server_module.mcp) as client:
        result = await client.call_tool(
            "aggregate", {"frame_ref": handles.plain_single_unit, "aggregation": "sum"}
        )

    assert result.is_error is False
    jsonschema.validate(result.structured_content, VERB_OUTPUT_SCHEMA)


async def test_role_keys_is_empty_for_a_keyless_aggregate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = InMemoryMaterialisationStore()
    handles = await seed_fixture_store(store)

    async def _stub_build_store(_settings: object) -> InMemoryMaterialisationStore:
        return store

    monkeypatch.setattr(server_module, "_build_store", _stub_build_store)

    async with Client(server_module.mcp) as client:
        result = await client.call_tool(
            "aggregate", {"frame_ref": handles.plain_single_unit, "aggregation": "sum"}
        )

    assert result.structured_content["role_keys"] == []


async def test_role_keys_carries_the_grouped_entity_column_for_aggregate_by_entity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = InMemoryMaterialisationStore()
    handles = await seed_fixture_store(store)

    async def _stub_build_store(_settings: object) -> InMemoryMaterialisationStore:
        return store

    monkeypatch.setattr(server_module, "_build_store", _stub_build_store)

    async with Client(server_module.mcp) as client:
        result = await client.call_tool(
            "aggregate",
            {
                "frame_ref": handles.plain_single_unit,
                "aggregation": "sum",
                "group_by": ["reporter"],
            },
        )

    assert result.structured_content["role_keys"] == ["reporter"]


async def test_role_keys_carries_ranks_own_axis_column(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = InMemoryMaterialisationStore()
    handles = await seed_fixture_store(store)

    async def _stub_build_store(_settings: object) -> InMemoryMaterialisationStore:
        return store

    monkeypatch.setattr(server_module, "_build_store", _stub_build_store)

    async with Client(server_module.mcp) as client:
        result = await client.call_tool(
            "rank",
            {"frame_ref": handles.plain_single_unit, "rank_by": ["reporter"], "aggregation": "sum"},
        )

    assert result.structured_content["role_keys"] == ["reporter"]


async def test_stat_var_count_is_two_for_a_join(monkeypatch: pytest.MonkeyPatch) -> None:
    store = InMemoryMaterialisationStore()
    handles = await seed_fixture_store(store)

    async def _stub_build_store(_settings: object) -> InMemoryMaterialisationStore:
        return store

    monkeypatch.setattr(server_module, "_build_store", _stub_build_store)

    async with Client(server_module.mcp) as client:
        result = await client.call_tool(
            "join",
            {"frame_ref": handles.join_pair[0], "right_frame_ref": handles.join_pair[1]},
        )

    assert result.structured_content["stat_var_count"] == 2


async def test_a_refusal_still_returns_its_own_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A refusal never goes through `VerbResult`/`VERB_OUTPUT_SCHEMA` --
    the MCP client validates `structuredContent` against `outputSchema` on
    success only, so a refusal's own payload shape is unconstrained by it.
    """
    store = InMemoryMaterialisationStore()
    handles = await seed_fixture_store(store)

    async def _stub_build_store(_settings: object) -> InMemoryMaterialisationStore:
        return store

    monkeypatch.setattr(server_module, "_build_store", _stub_build_store)

    async with Client(server_module.mcp) as client:
        result = await client.call_tool(
            "rank",
            {"frame_ref": handles.plain_single_unit, "rank_by": ["reporter"]},
            raise_on_error=False,
        )

    assert result.is_error is True
    assert result.structured_content["code"] == "role_ambiguous"


async def test_entity_parameter_schema_declares_an_optional_object() -> None:
    """`compare`'s `left_entity`/`right_entity` and `share_of`'s
    `numerator_entity`/`denominator_entity` are `dict[str, str] | None = None`
    at the tool boundary, not `dict[str, str] = {}`, and declared as an
    optional object with no null branch (`common.declaration`). The test pins
    the schema shape a real MCP client receives off the registered
    `server.mcp` tools, rather than a reconstruction or a stand-in tool
    defined only in this test.
    """
    async with Client(mcp) as client:
        tools = {tool.name: tool for tool in await client.list_tools()}

    for tool_name, params in (
        ("compare", ("left_entity", "right_entity")),
        ("share_of", ("numerator_entity", "denominator_entity")),
    ):
        schema = tools[tool_name].input_schema
        for param in params:
            prop = schema["properties"][param]
            assert (prop["type"], prop["additionalProperties"]) == ("object", {"type": "string"})
            assert "anyOf" not in prop
            assert "default" not in prop
            assert param not in schema.get("required", [])


async def test_build_store_does_not_seed_by_default() -> None:
    store = await _build_store(Settings())

    assert store._entries == {}


async def test_build_store_seeds_fixtures_when_the_flag_is_enabled() -> None:
    store = await _build_store(Settings(seed_fixture_frames=True))

    assert len(store._entries) == 8


async def test_build_store_forwards_settings_to_the_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = object()
    calls: list[dict[str, object]] = []

    def _fake_build_store(**kwargs: object) -> object:
        calls.append(kwargs)
        return sentinel

    monkeypatch.setattr(server_module, "build_store", _fake_build_store)

    store = await _build_store(
        Settings(frame_store_bucket="my-bucket", frame_store_verify_content_version=True)
    )

    assert store is sentinel
    assert calls[0]["bucket_name"] == "my-bucket"
    assert calls[0]["verify_content_version_on_read"] is True
    assert "derive_content_version" not in calls[0]


async def test_build_store_never_seeds_a_store_the_factory_did_not_build_in_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fixture seeding only ever targets `InMemoryMaterialisationStore`: a
    durable store the factory built for a configured bucket must never be
    seeded, even when `seed_fixture_frames` is on."""
    sentinel = object()
    seed_calls: list[object] = []

    async def _fake_seed(store: object) -> dict[str, object]:
        seed_calls.append(store)
        return {}

    monkeypatch.setattr(server_module, "build_store", lambda **_kwargs: sentinel)
    monkeypatch.setattr(server_module, "seed_fixture_store", _fake_seed)

    store = await _build_store(Settings(frame_store_bucket="my-bucket", seed_fixture_frames=True))

    assert store is sentinel
    assert seed_calls == []


async def test_lifespan_installs_a_store_when_none_is_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = InMemoryMaterialisationStore()

    async def _stub_build_store(_settings: object) -> InMemoryMaterialisationStore:
        return sentinel

    monkeypatch.setattr(server_module, "_build_store", _stub_build_store)

    async with _lifespan(mcp):
        assert get_store() is sentinel


async def test_lifespan_is_a_no_op_when_a_store_is_already_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mirrors the composed-process case where data-mcp's own lifespan, entered
    earlier, already installed a store: this lifespan must neither replace it nor
    build one of its own (which would also run fixture seeding needlessly)."""
    already_installed = InMemoryMaterialisationStore()
    set_store(already_installed)
    build_calls: list[object] = []

    async def _stub_build_store(settings: object) -> InMemoryMaterialisationStore:
        build_calls.append(settings)
        return InMemoryMaterialisationStore()

    monkeypatch.setattr(server_module, "_build_store", _stub_build_store)

    async with _lifespan(mcp):
        assert get_store() is already_installed

    assert build_calls == []


_IN_PROCESS_TWINS = {
    "aggregate": aggregate_tool.aggregate,
    "compare": compare_tool.compare,
    "growth": growth_tool.growth,
    "rank": rank_tool.rank,
    "share_of": share_of_tool.share_of,
    "join": join_tool.join,
    "filter": filter_tool.filter_verb,
    "sum_of": sum_of_tool.sum_of,
    "quotient_of": quotient_of_tool.quotient_of,
    "resolve_anchor": rolling_tool.resolve_anchor,
}


async def test_every_registered_tool_keeps_its_in_process_twins_schema_and_description() -> None:
    """The registered tools compute on the runner under the arguments and
    docstring of the in-process functions, so a caller sees the same tools."""
    twins = FastMCP(name="in-process")
    for name, fn in _IN_PROCESS_TWINS.items():
        twins.add_tool(declared_tool(Tool.from_function(fn, name=name)))
    async with Client(twins) as twin_client, Client(mcp) as client:
        expected = {tool.name: tool for tool in await twin_client.list_tools()}
        registered = {tool.name: tool for tool in await client.list_tools()}

    assert set(registered) == set(expected)
    leveled = set()
    rebuilding = set()
    for name, tool in registered.items():
        schema = copy.deepcopy(tool.input_schema)
        # Each verb adds the one argument its runner request reads (`verbs._on_runner`).
        if name != "resolve_anchor":
            assert schema["properties"].pop(FOLLOWS_TREE)["type"] == "boolean", name
        if schema["properties"].pop("place_level", None) is not None:
            leveled.add(name)
        if schema["properties"].pop("rebuild_lost_frames", None) is not None:
            rebuilding.add(name)
        assert schema == expected[name].input_schema, name
        assert tool.description == expected[name].description, name
    # `place_level` and `rebuild_lost_frames` are the registered tools' own: one
    # builds a runner filter step, the other rebuilds a frame the runner lost.
    assert leveled == {"aggregate", "rank", "share_of"}
    assert rebuilding == set(_IN_PROCESS_TWINS) - {"resolve_anchor"}


@pytest.mark.parametrize(("verb_fn", "verb_name"), [(aggregate, "aggregate"), (compare, "compare")])
async def test_backstop_wrapper_leaves_the_registered_schema_and_description_unchanged(
    verb_fn, verb_name: str
) -> None:
    """Confirms `_with_internal_error_backstop`'s docstring claim that the
    wrapped and bare registration produce byte-identical `inputSchema` and
    `description`, for at least two verbs. The docstring asserts this in
    prose; this test checks it directly. Guards the broader class of
    regression: any future change to the wrapper that alters what
    `fastmcp.tools.function_parsing` sees when building a tool's schema.

    Not a guard on `functools.wraps` specifically -- removing it would not
    surface as a failure of this assertion. `fastmcp` rejects a wrapper's
    bare `*args, **kwargs` signature outright (`ValueError: Functions with
    *args are not supported as tools`) at `mcp.tool()` registration time in
    `server.py`, which fails the whole suite at collection, well before
    this test's own body ever runs.
    """
    bare_server = FastMCP(name="bare")
    bare_server.tool(verb_fn)
    wrapped_server = FastMCP(name="wrapped")
    wrapped_server.tool(_with_internal_error_backstop(verb_fn, verb_name))

    async with Client(bare_server) as bare_client, Client(wrapped_server) as wrapped_client:
        (bare_tool,) = await bare_client.list_tools()
        (wrapped_tool,) = await wrapped_client.list_tools()

    assert bare_tool.name == wrapped_tool.name == verb_name
    assert wrapped_tool.description == bare_tool.description
    assert wrapped_tool.input_schema == bare_tool.input_schema


# The registration-site backstop -----------------------------------------------


def _aggregate_args(handles):
    return {"frame_ref": handles.plain_single_unit, "aggregation": "sum"}


def _compare_args(handles):
    return {
        "frame_ref": handles.plain_single_unit,
        "left_entity": {"reporter": "dcid:USA"},
        "right_entity": {"reporter": "dcid:GBR"},
    }


def _growth_args(handles):
    return {"frame_ref": handles.time_series_with_gap, "window_start": "2017", "window_end": "2018"}


def _rank_args(handles):
    return {"frame_ref": handles.plain_single_unit, "rank_by": ["reporter"]}


def _share_of_args(handles):
    return {
        "frame_ref": handles.join_pair[0],
        "denominator_frame_ref": handles.join_pair[1],
        "numerator_entity": {"reporter": "dcid:USA"},
        "denominator_entity": {"reporter": "dcid:USA"},
    }


def _quotient_of_args(handles):
    return {
        "frame_ref": handles.plain_single_unit,
        "numerator_entity": {"reporter": "dcid:USA"},
        "denominator_entity": {"reporter": "dcid:GBR"},
    }


def _join_args(handles):
    return {"frame_ref": handles.join_pair[0], "right_frame_ref": handles.join_pair[1]}


def _filter_args(handles):
    return {"frame_ref": handles.plain_single_unit}


_KNOWN_TRACE_ID = "0af7651916cd43dd8448eb211c80319c"
_SECRET_FAILURE = (
    "403 on gs://dfo-prod-frames/private/frame.parquet for "
    "serviceAccount:runtime@dfo-prod.iam.gserviceaccount.com"
)

_BACKSTOP_CASES = [
    ("aggregate", _aggregate_args),
    ("compare", _compare_args),
    ("growth", _growth_args),
    ("rank", _rank_args),
    ("share_of", _share_of_args),
    ("quotient_of", _quotient_of_args),
    ("join", _join_args),
    ("filter", _filter_args),
]


@pytest.mark.parametrize(
    ("tool_name", "args_builder"), _BACKSTOP_CASES, ids=[case[0] for case in _BACKSTOP_CASES]
)
async def test_registration_site_backstop_converts_an_unanticipated_exception(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tool_name: str,
    args_builder,
) -> None:
    """An exception no verb's own typed handling anticipates -- injected here
    by making the verb's runner call raise a plain `RuntimeError` --
    never escapes the registered tool as a raw traceback. `server.py`'s
    registration-site wrapper converts it to an `internal_error` refusal,
    per verb, that carries no trace id and none of the exception's text; the
    log line carries the trace id and the exception.
    """
    store = InMemoryMaterialisationStore()
    handles = await seed_fixture_store(store)

    async def _stub_build_store(_settings: object) -> InMemoryMaterialisationStore:
        return store

    monkeypatch.setattr(server_module, "_build_store", _stub_build_store)

    def _raise(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError(_SECRET_FAILURE)

    monkeypatch.setattr(verbs, "run_on_runner", _raise)
    monkeypatch.setattr(refusal_module, "current_trace_id", lambda: _KNOWN_TRACE_ID)

    with caplog.at_level(logging.ERROR, logger=server_module.__name__):
        async with Client(server_module.mcp) as client:
            result = await client.call_tool(tool_name, args_builder(handles), raise_on_error=False)

    assert result.is_error is True
    assert result.structured_content["code"] == "internal_error"
    assert result.structured_content["retry_mode"] == "do_not_retry"
    assert result.structured_content["fix"] == {"tool": tool_name}
    replied = result.content[0].text + json.dumps(result.structured_content)
    assert _KNOWN_TRACE_ID not in replied
    assert "gs://" not in replied
    assert "iam.gserviceaccount.com" not in replied
    [record] = [r for r in caplog.records if r.name == server_module.__name__]
    assert _KNOWN_TRACE_ID in record.getMessage()
    assert record.exc_info is not None
    assert str(record.exc_info[1]) == _SECRET_FAILURE


@pytest.mark.parametrize(
    ("tool_name", "args"),
    [("nonexistent_verb", {}), ("aggregate", {"not_a_parameter": 1})],
    ids=["unknown tool name", "unknown argument"],
)
async def test_backstop_leaves_protocol_failures_to_report_themselves(
    tool_name: str, args: dict[str, object]
) -> None:
    """The backstop wraps each verb function, which places it inside the tool
    boundary: `fastmcp` resolves the tool and validates its arguments before
    the wrapper runs, so neither failure reaches it. Both surface as their
    own errors. The registration-site backstop's `internal_error` mapping,
    covered above, does not apply to either kind of protocol failure.

    Guards the placement. A backstop moved outward to server middleware would
    intercept both of these and answer an unknown tool name with a refusal
    saying the server broke, so this fails if the wrapper is ever relocated.
    """
    async with Client(server_module.mcp) as client:
        result = await client.call_tool(tool_name, args, raise_on_error=False)

    assert result.is_error is True
    assert result.structured_content is None or (
        result.structured_content.get("code") != "internal_error"
    )


_VERBS = {
    "aggregate",
    "compare",
    "growth",
    "rank",
    "share_of",
    "join",
    "filter",
    "sum_of",
    "quotient_of",
}


@pytest.mark.parametrize(
    ("tool_name", "args"),
    [("frobnicate", {}), ("get_data", {})],
    ids=["made-up name", "another server's tool name"],
)
async def test_unknown_tool_name_names_the_registered_verbs(
    tool_name: str, args: dict[str, object]
) -> None:
    """An unknown tool name -- whether made up or belonging to another
    server, such as data-mcp's `get_data` -- names all nine registered
    verbs and points the caller at `tools/list`. FastMCP's own default
    response is the bare `Unknown tool: {name!r}`.
    """
    async with Client(server_module.mcp) as client:
        result = await client.call_tool(tool_name, args, raise_on_error=False)

    assert result.is_error is True
    (content,) = result.content
    assert tool_name in content.text
    for verb in _VERBS:
        assert verb in content.text
    assert "tools/list" in content.text


async def test_a_real_verb_still_dispatches_through_the_middleware(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The middleware only intervenes on a `NotFoundError` from an unknown
    tool name -- a real verb dispatches and returns its own result
    untouched.
    """
    store = InMemoryMaterialisationStore()
    handles = await seed_fixture_store(store)

    async def _stub_build_store(_settings: object) -> InMemoryMaterialisationStore:
        return store

    monkeypatch.setattr(server_module, "_build_store", _stub_build_store)

    async with Client(server_module.mcp) as client:
        result = await client.call_tool("aggregate", _aggregate_args(handles), raise_on_error=False)

    assert result.is_error is False


async def test_healthz_custom_route() -> None:
    app = mcp.http_app(transport="http")
    transport = httpx2.ASGITransport(app=app)

    async with httpx2.AsyncClient(transport=transport, base_url="http://testserver") as http_client:
        response = await http_client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

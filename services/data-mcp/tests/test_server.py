"""Server-scaffold tests for the FastMCP surface: registration, healthz, and the shutdown lifespan."""

from __future__ import annotations

import httpx2
import pytest
from analysis_mcp import server as analysis_server
from analysis_mcp.server import mcp as analysis_mcp
from common.materialisation import ContentVersionDeriverUnregistered, InMemoryMaterialisationStore
from common.materialisation import content_version as content_version_module
from common.materialisation import factory as materialisation_factory
from common.materialisation import get_store as get_frame_store
from common.materialisation import set_store as set_frame_store
from common.materialisation import store_is_set as frame_store_is_set
from data_mcp import server
from data_mcp.definition_store import CachingDefinitionStore, InMemoryDefinitionStore
from data_mcp.manifest import ARTIFACT_URI_TEMPLATE
from data_mcp.server import mcp
from data_mcp.settings import Settings
from data_mcp.skills_resources import SKILL_FILE_TEMPLATE
from fastmcp import Client
from pydantic import ValidationError

from .net_guard import no_network  # noqa: F401 -- imported for its autouse fixture


async def test_server_exposes_its_eight_tools() -> None:
    async with Client(mcp) as client:
        tools = await client.list_tools()

    assert [tool.name for tool in tools] == [
        "search",
        "get_data",
        "describe",
        "get_skill",
        "get_skill_file",
        "sources",
        "resolve_definition",
        "catalogue_counts",
    ]


async def test_every_tool_declares_a_version() -> None:
    """Read off the registered `FunctionTool` objects directly
    (`run_middleware=False`), matching `analysis_mcp.server`'s own
    version-declaration test. No `output_schema` is declared here -- see
    `server.py`'s own comment at the registration site.
    """
    tools = await mcp.list_tools(run_middleware=False)

    versions = {tool.name: tool.version for tool in tools}
    assert versions == {
        "search": "1",
        "get_data": "1",
        "describe": "1",
        "get_skill": "1",
        "get_skill_file": "1",
        "sources": "1",
        "resolve_definition": "1",
        "catalogue_counts": "1",
    }
    assert all(tool.output_schema is None for tool in tools)


async def test_server_registers_the_artifact_and_skill_file_templates() -> None:
    async with Client(mcp) as client:
        templates = await client.list_resource_templates()

    assert [t.uri_template for t in templates] == [ARTIFACT_URI_TEMPLATE, SKILL_FILE_TEMPLATE]


async def test_healthz_custom_route() -> None:
    app = mcp.http_app(transport="http")
    transport = httpx2.ASGITransport(app=app)

    async with httpx2.AsyncClient(transport=transport, base_url="http://testserver") as http_client:
        response = await http_client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_lifespan_still_closes_client_and_store_when_the_body_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed: list[str] = []

    async def fake_aclose_client() -> None:
        closed.append("client")

    async def fake_aclose_store() -> None:
        closed.append("store")

    async def fake_aclose_frame_store() -> None:
        closed.append("frame_store")

    monkeypatch.setattr(server, "aclose_client", fake_aclose_client)
    monkeypatch.setattr(server, "aclose_store", fake_aclose_store)
    monkeypatch.setattr(server, "aclose_frame_store", fake_aclose_frame_store)

    with pytest.raises(RuntimeError, match="boom"):
        async with server._lifespan(mcp):
            raise RuntimeError("boom")

    assert closed == ["client", "store", "frame_store"]


class _FakeGCSDefinitionStore:
    """Stands in for `GCSDefinitionStore`, monkeypatched over the name `_build_store`
    looks up, so the GCS branch can be exercised without real credentials or network.
    """

    def __init__(self, *, bucket_name: str) -> None:
        self.bucket_name = bucket_name


def test_build_store_defaults_to_in_memory_when_no_bucket_is_configured() -> None:
    store = server._build_store(Settings(definition_store_bucket=None))

    assert isinstance(store, CachingDefinitionStore)
    assert isinstance(store._inner, InMemoryDefinitionStore)


def test_build_store_uses_gcs_when_a_bucket_is_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server, "GCSDefinitionStore", _FakeGCSDefinitionStore)

    store = server._build_store(Settings(definition_store_bucket="my-bucket"))

    assert isinstance(store, CachingDefinitionStore)
    assert isinstance(store._inner, _FakeGCSDefinitionStore)
    assert store._inner.bucket_name == "my-bucket"


class _FakeGcsMaterialisationStore:
    """Stands in for `GcsMaterialisationStore`, monkeypatched over the name
    `common.materialisation.factory.build_store` looks up, so the frame store's own
    GCS branch can be exercised without real credentials or network."""

    def __init__(
        self,
        *,
        bucket_name: str,
        verify_content_version_on_read: bool = False,
        derive_content_version: object | None = None,
        **_kwargs: object,
    ) -> None:
        self.bucket_name = bucket_name
        self.verify_content_version_on_read = verify_content_version_on_read
        self.derive_content_version = derive_content_version

    async def aclose(self) -> None:
        pass


async def test_lifespan_does_not_touch_the_frame_store_when_no_bucket_is_configured() -> None:
    """The common case: standalone with no `FRAME_STORE_BUCKET`, or mounted alongside
    analysis-mcp, which configures its own. Either way this service's own lifespan
    leaves the shared holder alone rather than installing an unseeded default that
    could beat a later, better-configured install to the holder."""
    async with server._lifespan(mcp):
        assert frame_store_is_set() is False


async def test_lifespan_installs_a_durable_frame_store_when_a_bucket_is_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        materialisation_factory, "GcsMaterialisationStore", _FakeGcsMaterialisationStore
    )
    monkeypatch.setenv("FRAME_STORE_BUCKET", "my-bucket")

    async with server._lifespan(mcp):
        installed = get_frame_store()

    assert isinstance(installed, _FakeGcsMaterialisationStore)
    assert installed.bucket_name == "my-bucket"


async def test_lifespan_does_not_replace_a_frame_store_already_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Whatever installed a store first -- from any source -- this service's own
    install call must be a no-op, never a replacement: see `install_store_if_unset`'s
    own docstring for the ownership rule this covers from data-mcp's side.
    `test_composed_order_installs_one_store_with_analysis_mcps_own_verify_setting`
    below covers the actual production order, where this service's own lifespan is
    always the one that runs first."""
    already_installed = InMemoryMaterialisationStore()
    set_frame_store(already_installed)
    monkeypatch.setattr(
        materialisation_factory, "GcsMaterialisationStore", _FakeGcsMaterialisationStore
    )
    monkeypatch.setenv("FRAME_STORE_BUCKET", "my-bucket")

    async with server._lifespan(mcp):
        assert get_frame_store() is already_installed


async def test_composed_order_installs_one_store_with_analysis_mcps_own_verify_setting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real production order (`harness/server.py`'s `combine_lifespans` call):
    this service's own lifespan always enters before analysis-mcp's, so this
    service's own `install_store_if_unset` call always wins the race when both are
    configured with the same bucket. `verify_content_version_on_read` must still come
    out `True` and carry a real deriver, from analysis-mcp's own
    `FRAME_STORE_VERIFY_CONTENT_VERSION` setting reaching this service's identical
    field of the same name -- not from whichever service happened to install the
    store.
    """
    monkeypatch.setattr(
        materialisation_factory, "GcsMaterialisationStore", _FakeGcsMaterialisationStore
    )
    monkeypatch.setenv("FRAME_STORE_BUCKET", "shared-bucket")
    monkeypatch.setenv("FRAME_STORE_VERIFY_CONTENT_VERSION", "true")

    # This service's own install call already wins by the time analysis-mcp's own
    # lifespan runs -- `analysis_mcp.server._lifespan` finds `store_is_set()` true and
    # skips its own `_build_store` entirely, exactly as it does in the real composed
    # app.
    async with server._lifespan(mcp), analysis_server._lifespan(analysis_mcp):
        installed = get_frame_store()

    assert isinstance(installed, _FakeGcsMaterialisationStore)
    assert installed.bucket_name == "shared-bucket"
    assert installed.verify_content_version_on_read is True
    assert installed.derive_content_version is not None


async def test_standalone_lifespan_refuses_to_start_when_verify_is_on_with_no_deriver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A deployment that sets `FRAME_STORE_VERIFY_CONTENT_VERSION` without also
    mounting analysis-mcp in the same process (the only thing that ever registers a
    content_version deriver) has nothing for the check to run against. Forces that
    state directly, regardless of whether some other test in this one-process suite
    has already imported `analysis_mcp` and registered the real deriver."""
    monkeypatch.setattr(
        materialisation_factory, "GcsMaterialisationStore", _FakeGcsMaterialisationStore
    )
    monkeypatch.setattr(content_version_module, "_deriver", None)
    monkeypatch.setenv("FRAME_STORE_BUCKET", "my-bucket")
    monkeypatch.setenv("FRAME_STORE_VERIFY_CONTENT_VERSION", "true")

    with pytest.raises(ContentVersionDeriverUnregistered):
        async with server._lifespan(mcp):
            pass


async def test_lifespan_refuses_to_start_on_cloud_run_without_a_site_origin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Artifact links handed to models need the public site origin there, since
    dfo-apis refuses callers without an ID token."""
    monkeypatch.setenv("K_SERVICE", "dfo-dev-ai")
    monkeypatch.delenv("DFO_SITE_ORIGIN", raising=False)

    with pytest.raises(ValidationError, match="DFO_SITE_ORIGIN"):
        async with server._lifespan(mcp):
            pass

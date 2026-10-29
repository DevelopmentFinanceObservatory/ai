"""FastMCP server scaffold for data-mcp: the composition root.

Holds infrastructure: the FastMCP instance, the `/healthz` liveness route,
the shutdown lifespan, and the console-script entry point (settings,
logging, telemetry, transport). Tool modules define plain functions with no
knowledge of `mcp`; this module imports each one and registers it
explicitly against the module-level `mcp` instance.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from common.auth import BearerTokenVerifier, apply_gate
from common.declaration import declared_tool
from common.logging import configure_logging
from common.materialisation import aclose_store as aclose_frame_store
from common.materialisation import install_store_if_unset
from common.mcp_telemetry import LangfuseSessionMiddleware, LangfuseToolIoMiddleware
from common.telemetry import init_telemetry
from common.tool_annotations import READ_ONLY_CATALOGUE
from fastmcp import FastMCP
from fastmcp.tools import Tool
from starlette.requests import Request
from starlette.responses import JSONResponse

from data_mcp import search_tool
from data_mcp.artifact_resource import serve_artifact
from data_mcp.catalogue_counts import catalogue_counts
from data_mcp.client_holder import aclose_client, build_client, set_client
from data_mcp.definition_store import (
    CachingDefinitionStore,
    DefinitionStore,
    GCSDefinitionStore,
    InMemoryDefinitionStore,
    aclose_store,
    set_store,
)
from data_mcp.describe_tool import describe
from data_mcp.get_data_tool import aclose_background_mints, get_data
from data_mcp.get_skill_file_tool import get_skill_file
from data_mcp.get_skill_tool import get_skill
from data_mcp.manifest import ARTIFACT_URI_TEMPLATE
from data_mcp.resolve_definition_tool import resolve_definition
from data_mcp.result_bound import ResultBoundMiddleware
from data_mcp.settings import Settings
from data_mcp.skills_resources import SkillsProvider
from data_mcp.sources_tool import sources

logger = logging.getLogger(__name__)


def _build_store(settings: Settings) -> DefinitionStore:
    """Build the definition store this process serves, per `settings.definition_store_bucket`.

    No bucket means an in-memory store. On a Cloud Run instance (`K_SERVICE`
    set) that is a deliberate local choice logged as a warning: a second
    instance can't see it, so a `get_data` against this instance's own
    unregistered id refuses `definition_not_held`, whose message tells the
    model the store is up and to search again -- true of the store, false of
    why the id is missing.
    """
    if settings.definition_store_bucket is None:
        if os.environ.get("K_SERVICE"):
            logger.warning(
                "DEFINITION_STORE_BUCKET is unset on a Cloud Run instance "
                "(K_SERVICE=%s); falling back to an in-memory definition store "
                "that other instances can't see.",
                os.environ["K_SERVICE"],
            )
        inner: DefinitionStore = InMemoryDefinitionStore()
    else:
        inner = GCSDefinitionStore(bucket_name=settings.definition_store_bucket)
    return CachingDefinitionStore(inner, max_entries=settings.definition_store_cache_max_entries)


@asynccontextmanager
async def _lifespan(_server: FastMCP) -> AsyncIterator[None]:
    """Builds the definition store on startup; closes the store, the upstream client, and
    the shared frame materialisation store on shutdown.

    This lifespan, not `main()`, is what actually runs in production: the
    harness mounts this app's ASGI sub-app via `combine_lifespans` and never
    calls `main()`. This service never seeds the frame store itself -- `get_data`
    resolves it at request time via `common.materialisation.get_store()`.

    `settings.frame_store_bucket` unset leaves the frame-store holder untouched: a
    standalone deploy falls through to the holder's own lazy in-memory default, and
    a deploy mounted alongside analysis-mcp relies on that service's own lifespan
    instead. Set, this lifespan installs a durable store itself via
    `install_store_if_unset` -- see that function's own docstring for the ownership
    rule. `settings.frame_store_verify_content_version` is threaded through
    unconditionally on that call, so the installed store is configured identically
    regardless of which service's lifespan wins the install race.
    """
    settings = Settings()
    logger.info("search_sends_conversation_context=%s", settings.search_sends_conversation_context)
    # `_build_store` can construct a real `storage.Client()`, which touches
    # ADC and possibly the metadata server -- `to_thread` crosses that onto a
    # worker thread instead of blocking the event loop at startup, matching
    # every other GCS call in `gcs.py`.
    set_store(await asyncio.to_thread(_build_store, settings))
    if settings.frame_store_bucket is not None:
        install_store_if_unset(
            bucket_name=settings.frame_store_bucket,
            verify_content_version_on_read=settings.frame_store_verify_content_version,
        )
    try:
        yield
    finally:
        await aclose_background_mints()
        await aclose_client()
        await aclose_store()
        await aclose_frame_store()


mcp = FastMCP(name="data-mcp", lifespan=_lifespan, auth=BearerTokenVerifier())
mcp.add_middleware(LangfuseSessionMiddleware())
mcp.add_middleware(LangfuseToolIoMiddleware())
mcp.add_middleware(ResultBoundMiddleware())


@mcp.custom_route("/healthz", methods=["GET"])
async def healthz(_request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok"})


def _add_declared(fn: Callable[..., object], *, title: str) -> None:
    """Register `fn` as a catalogue tool whose listed schema is `declared_tool`'s."""
    tool = Tool.from_function(fn, version="1", title=title, annotations=READ_ONLY_CATALOGUE)
    mcp.add_tool(declared_tool(tool))


# Each tool's declared implementation version, read by
# `compiler.surface._declared_version` and pinned per row in
# `compiler.capabilities`, which lists `get_skill` and `get_skill_file`
# among the tools no workflow step calls. No `output_schema=`: each tool's
# structured result is the full `Manifest`, and deriving and validating that
# schema on every call is a risk left untaken here. `resolve_definition` and
# `catalogue_counts` are the harness's: the combined server outside clients
# reach leaves them out (`harness.combined_mcp`).
_add_declared(search_tool.search, title="Search DFO data")
_add_declared(get_data, title="Get data")
_add_declared(describe, title="Explain a data identifier")
_add_declared(get_skill, title="Load a skill")
_add_declared(get_skill_file, title="Read a skill file")
_add_declared(sources, title="List data sources")
mcp.tool(resolve_definition, version="1")
mcp.tool(catalogue_counts, version="1")

# A templated resource, not a tool -- it surfaces only via
# `resources/templates/list` discovery, not `resources/list`, so ADK cannot
# reach it (see `artifact_resource.py`'s module docstring). A raw MCP client
# that already has the URI, from a `get_data` result's resource link, can
# still read it.
mcp.resource(
    ARTIFACT_URI_TEMPLATE,
    name="artifact",
    mime_type="application/json",
    description=(
        "Read this to obtain the download URL for one artifact. The URL does not "
        "expire: a GET on it answers 302 to a download link that lasts a few minutes, "
        "so follow redirects and fetch the bytes when you need them."
    ),
)(serve_artifact)

# Every skill file as a `skill://` resource (`skills_resources.py`). The
# Skills over MCP extension that reads through the same client is registered
# by the combined server, since a mounted server's extensions do not reach it.
mcp.add_provider(SkillsProvider())


def main() -> None:
    """Console-script entry point: run data-mcp over streamable HTTP."""
    settings = Settings()
    configure_logging()
    init_telemetry(service_name="data-mcp")
    set_client(build_client(settings))
    apply_gate(mcp, token=settings.mcp_api_token, disabled=settings.mcp_auth_disabled)
    logger.info("Starting data-mcp on port %d", settings.port)
    mcp.run(transport="streamable-http", host="0.0.0.0", port=settings.port)


if __name__ == "__main__":
    main()

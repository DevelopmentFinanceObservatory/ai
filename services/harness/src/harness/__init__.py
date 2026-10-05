"""harness — agentic loop (Google ADK + FastAPI) powering the site's chat+canvas."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("harness")
except PackageNotFoundError:  # pragma: no cover - always installed in the workspace
    __version__ = "0.0.0"

__all__ = ["__version__"]

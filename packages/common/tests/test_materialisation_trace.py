"""`StoredFrameTrace`: a structural Protocol, satisfied by `MaterialisationTrace`
and by any other type that carries a compatible `upstream_trace`.

The cross-satisfaction proof against a real `analysis_mcp.trace.Trace` lives
in analysis-mcp's own `test_trace.py`, since `common` cannot import
`analysis_mcp`.
"""

from __future__ import annotations

from typing import Any

from common.materialisation import MaterialisationTrace
from common.materialisation.trace import StoredFrameTrace


def test_materialisation_trace_satisfies_the_protocol() -> None:
    trace: StoredFrameTrace = MaterialisationTrace(
        upstream_trace={"graph_revision": "gr_1"},
        spec_id="spec_" + "a" * 32,
        content_version="cv_" + "b" * 32,
    )

    assert trace.upstream_trace == {"graph_revision": "gr_1"}


def test_materialisation_trace_frame_content_version_is_its_own_content_version() -> None:
    trace = MaterialisationTrace(
        upstream_trace={"graph_revision": "gr_1"},
        spec_id="spec_" + "a" * 32,
        content_version="cv_" + "b" * 32,
    )

    assert trace.frame_content_version == "cv_" + "b" * 32


def test_a_hand_built_fake_also_satisfies_the_protocol() -> None:
    class _FakeTrace:
        """No relation to `common`'s own types -- proves the Protocol is
        satisfied structurally, not nominally."""

        @property
        def upstream_trace(self) -> dict[str, Any]:
            return {"fake": True}

        @property
        def frame_content_version(self) -> str:
            return "cv_" + "f" * 32

    trace: StoredFrameTrace = _FakeTrace()

    assert trace.upstream_trace == {"fake": True}
    assert trace.frame_content_version == "cv_" + "f" * 32

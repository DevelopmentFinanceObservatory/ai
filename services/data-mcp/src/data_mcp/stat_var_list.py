"""The StatVar list a `get_data` result carries, bounded, and the read that restores it.

A manifest's `trace.resolved_stat_vars` holds one entry for every StatVar in
the definition's family, about 220 bytes each, so a large family would grow
the result past what one server-sent event can carry
(`data_mcp.result_bound`). `without_long_stat_var_list` drops a list longer
than `MAX_LISTED_STAT_VARS` from the result and records how many entries it
held under `STAT_VARS_OMITTED_KEY`, beside the manifest. Code that reads the
list (the harness's evidence checks and provenance) calls
`with_stat_var_list` on the result first, which reads the list back off
dfo-apis' snapshot of the same `{spec_id, content_version}` pair. No model
ever reads the list.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from fastmcp.exceptions import ResourceError

from data_mcp.client_holder import get_client
from data_mcp.errors import UpstreamError

# The most `trace.resolved_stat_vars` entries a `get_data` result carries:
# about 46 KB of JSON.
MAX_LISTED_STAT_VARS = 200

# The key beside the manifest that counts the `trace.resolved_stat_vars`
# entries the result left out.
STAT_VARS_OMITTED_KEY = "resolved_stat_vars_omitted"

_SNAPSHOT_ROUTE = "/api/datasets/{id}/{content_version}"


def without_long_stat_var_list(result: Mapping[str, Any]) -> dict[str, Any]:
    """`result` with a `trace.resolved_stat_vars` longer than `MAX_LISTED_STAT_VARS`
    removed and its length under `STAT_VARS_OMITTED_KEY`; otherwise `result` as is."""
    trace = result.get("trace")
    listed = trace.get("resolved_stat_vars") if isinstance(trace, Mapping) else None
    if not isinstance(listed, list) or len(listed) <= MAX_LISTED_STAT_VARS:
        return dict(result)
    assert isinstance(trace, Mapping)
    kept = {key: value for key, value in trace.items() if key != "resolved_stat_vars"}
    return {**result, "trace": kept, STAT_VARS_OMITTED_KEY: len(listed)}


async def with_stat_var_list(result: Mapping[str, Any]) -> dict[str, Any]:
    """`result` with the `trace.resolved_stat_vars` `without_long_stat_var_list` left
    out read back from dfo-apis' snapshot route, or `result` as is when none was.

    Raises:
        fastmcp.exceptions.ResourceError: The snapshot could not be read, or its
            list does not hold the number of entries the result recorded.
    """
    omitted = result.get(STAT_VARS_OMITTED_KEY)
    if omitted is None:
        return dict(result)
    definition = result.get("definition") or {}
    spec_id = definition.get("spec_id")
    content_version = result.get("content_version")
    try:
        response = await get_client().request(
            "GET", f"/api/datasets/{spec_id}/{content_version}", route=_SNAPSHOT_ROUTE
        )
        payload = response.json()
    except (UpstreamError, ValueError) as exc:
        raise ResourceError(
            f"could not read the StatVar list of {spec_id}/{content_version}: {exc}"
        ) from exc
    trace = payload.get("trace") if isinstance(payload, dict) else None
    listed = trace.get("resolved_stat_vars") if isinstance(trace, dict) else None
    if not isinstance(listed, list) or len(listed) != omitted:
        raise ResourceError(
            f"the snapshot of {spec_id}/{content_version} does not list the {omitted} "
            "StatVars its get_data result left out"
        )
    restored = {key: value for key, value in result.items() if key != STAT_VARS_OMITTED_KEY}
    restored["trace"] = {**(result.get("trace") or {}), "resolved_stat_vars": listed}
    return restored

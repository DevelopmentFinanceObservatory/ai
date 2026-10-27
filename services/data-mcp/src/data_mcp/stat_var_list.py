"""The StatVar list a `get_data` result carries, bounded, and the read that restores it.

A manifest's `trace.resolved_stat_vars` holds one entry for every StatVar in
the definition's family, about 220 bytes each, so a large family would grow
the result past what one server-sent event can carry
(`data_mcp.result_bound`). `without_long_stat_var_list` drops a list longer
than `MAX_LISTED_STAT_VARS` from the result and records how many entries it
held under `STAT_VARS_OMITTED_KEY`, beside the manifest. Code that reads the
list (the harness's evidence checks and provenance) calls
`with_stat_var_list` on the result first, which reads the list back off
apis' snapshot of the same `{spec_id, content_version}` pair. No model
ever reads the list.

A composite frame's `trace.components` lists each component's members too
(`common.composite.ManifestComponent.stat_vars`), so the same bound applies to
the members of every component together: past it, each component's list is
emptied, their count goes under `COMPONENT_STAT_VARS_OMITTED_KEY`, and
`with_stat_var_list` reads them back with the rest.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from common.composite import COMPONENTS_KEY
from fastmcp.exceptions import ResourceError

from data_mcp.client_holder import get_client
from data_mcp.errors import UpstreamError

# The most `trace.resolved_stat_vars` entries a `get_data` result carries:
# about 46 KB of JSON.
MAX_LISTED_STAT_VARS = 200

# The key beside the manifest that counts the `trace.resolved_stat_vars`
# entries the result left out.
STAT_VARS_OMITTED_KEY = "resolved_stat_vars_omitted"

# The key beside the manifest that counts the members `trace.components` left out.
COMPONENT_STAT_VARS_OMITTED_KEY = "component_stat_vars_omitted"

_SNAPSHOT_ROUTE = "/api/datasets/{id}/{content_version}"


def without_long_stat_var_list(result: Mapping[str, Any]) -> dict[str, Any]:
    """`result` with a `trace.resolved_stat_vars` longer than `MAX_LISTED_STAT_VARS`
    removed and its length under `STAT_VARS_OMITTED_KEY`, and the members of
    `trace.components`, when they number more than `MAX_LISTED_STAT_VARS` together,
    emptied and counted under `COMPONENT_STAT_VARS_OMITTED_KEY`; otherwise `result` as is."""
    trace = result.get("trace")
    if not isinstance(trace, Mapping):
        return dict(result)
    bounded = dict(result)
    kept = dict(trace)
    listed = trace.get("resolved_stat_vars")
    if isinstance(listed, list) and len(listed) > MAX_LISTED_STAT_VARS:
        del kept["resolved_stat_vars"]
        bounded[STAT_VARS_OMITTED_KEY] = len(listed)
    components = _components(trace)
    members = sum(len(component.get("stat_vars") or ()) for component in components)
    if members > MAX_LISTED_STAT_VARS:
        kept[COMPONENTS_KEY] = [{**component, "stat_vars": []} for component in components]
        bounded[COMPONENT_STAT_VARS_OMITTED_KEY] = members
    bounded["trace"] = kept
    return bounded


def _components(trace: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """`trace`'s components as apis wrote them, every field kept, since the bound
    rewrites them and hands the trace on whole; `common.composite.manifest_components`
    would drop the fields it does not read."""
    listed = trace.get(COMPONENTS_KEY)
    if not isinstance(listed, list):
        return []
    return [component for component in listed if isinstance(component, Mapping)]


async def with_stat_var_list(result: Mapping[str, Any]) -> dict[str, Any]:
    """`result` with the `trace.resolved_stat_vars` and component members
    `without_long_stat_var_list` left out read back from apis' snapshot route, or
    `result` as is when it left none out.

    Raises:
        fastmcp.exceptions.ResourceError: The snapshot could not be read, or it does
            not hold the number of entries the result recorded.
    """
    omitted = result.get(STAT_VARS_OMITTED_KEY)
    members_omitted = result.get(COMPONENT_STAT_VARS_OMITTED_KEY)
    if omitted is None and members_omitted is None:
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
    snapshot = payload.get("trace") if isinstance(payload, dict) else None
    snapshot = snapshot if isinstance(snapshot, dict) else {}
    restored = {
        key: value
        for key, value in result.items()
        if key not in (STAT_VARS_OMITTED_KEY, COMPONENT_STAT_VARS_OMITTED_KEY)
    }
    trace = dict(result.get("trace") or {})
    if omitted is not None:
        listed = snapshot.get("resolved_stat_vars")
        if not isinstance(listed, list) or len(listed) != omitted:
            raise ResourceError(
                f"the snapshot of {spec_id}/{content_version} does not list the {omitted} "
                "StatVars its get_data result left out"
            )
        trace["resolved_stat_vars"] = listed
    if members_omitted is not None:
        components = _components(snapshot)
        members = sum(len(component.get("stat_vars") or ()) for component in components)
        if members != members_omitted or len(components) != len(_components(trace)):
            raise ResourceError(
                f"the snapshot of {spec_id}/{content_version} does not list the "
                f"{members_omitted} component StatVars its get_data result left out"
            )
        trace[COMPONENTS_KEY] = [
            {**component, "stat_vars": list(whole.get("stat_vars") or ())}
            for component, whole in zip(_components(trace), components, strict=True)
        ]
    restored["trace"] = trace
    return restored

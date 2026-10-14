"""What every verb tool returns as `structuredContent`: `VerbResult`, its
declared JSON Schema, and each verb's declared implementation version.

`server._register` passes `output_schema=VERB_OUTPUT_SCHEMA` and
`version=IMPLEMENTATION_VERSIONS[wire_name]` to `mcp.tool(...)` at each
verb's registration site. `-> ToolResult` stays the return annotation --
`fastmcp` suppresses schema derivation for it on purpose -- so the schema
is declared here, explicitly, from `VerbResult` itself, and
`store_result_and_render` builds the returned `structured_content` from a
real `VerbResult(...)` via `model_dump(mode="json")`, so the two can never
drift apart. A refusal builds its own payload (`tool_errors.to_tool_result`)
and never goes through `VerbResult` -- the MCP client validates
`structuredContent` against `outputSchema` on success only, so an output
schema is declared safely for refusals too.

`role_keys` and `stat_var_count` are read off the frame `store_result_and_render`
already holds, via `observed_shape` below -- no verb signature changes.
`dropped_key_count` is the one shape fact a frame cannot state, the keys a
verb left out of it, so the verb passes it to `store_result_and_render`.

Every verb declares version "1" until the toolkit's first release. Until
then a deploy ships the harness's capability pins and the servers together,
so no plan compiled against an earlier version outlives the deploy that
changes a verb.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dfo_frame_grammar.grammar import TIME_PERIOD_COLUMN
from pydantic import BaseModel, ConfigDict, Field

from analysis_mcp.artifact import Exclusion, Observation, Time

if TYPE_CHECKING:
    from collections.abc import Mapping

    from dfo_frame_grammar import Frame


class VerbResult(BaseModel):
    """What every verb tool returns as `structuredContent`.

    - `spec_id`/`content_version`: the `{spec_id, content_version}` pair the
      next chained call's own `frame_ref` binds to.
    - `frame_content_version`: the stored output frame's own frame-grammar `cv_`.
    - `rows`: the stored frame's own row count, including 0. Named `rows` to
      match `Exclusion.rows` and data-mcp's `Manifest.rows` -- one name for
      the same concept across both servers.
    - `role_keys`: the frame's own role columns, sorted as `Frame.build` sorts
      them -- `()` for a keyless verb output.
    - `stat_var_count`: distinct `stat_var`s the frame carries; 0 when
      `rows` is 0.
    - `dropped_key_count`: how many of the numerator's keys a keyed
      `quotient_of` holds no row for: those with no denominator group, those
      sharing no period with it, and those whose every matched cell has a
      zero denominator. 0 for an un-keyed call and for every other verb.
    - `ranked_before_limit`: how many entries a `rank` call with a `limit`
      ranked before the limit cut them to `rows`: entities, or entity
      periods with `per_period`. `None` for a call with no `limit` and for
      every other verb.
    - `exclusions`/`time`: as `Trace` carries them.
    """

    model_config = ConfigDict(extra="forbid")

    spec_id: str
    content_version: str
    frame_content_version: str
    rows: int = Field(ge=0)
    role_keys: tuple[str, ...] = ()
    stat_var_count: int = Field(ge=0)
    dropped_key_count: int = Field(default=0, ge=0)
    ranked_before_limit: int | None = Field(default=None, ge=0)
    exclusions: tuple[Exclusion, ...] = ()
    time: Time = Field(default_factory=Observation)


VERB_OUTPUT_SCHEMA: dict[str, Any] = VerbResult.model_json_schema(mode="serialization")

IMPLEMENTATION_VERSIONS: Mapping[str, str] = {
    "aggregate": "1",
    "compare": "1",
    "growth": "1",
    "rank": "1",
    "share_of": "1",
    "join": "1",
    "filter": "1",
    "sum_of": "1",
    "quotient_of": "1",
}


def observed_shape(frame: Frame) -> tuple[tuple[str, ...], int]:
    """`frame`'s own role columns and its distinct `stat_var` count, read off
    the built frame. `grammar.derive_columns` pins the column order as
    `stat_var`, the role columns, `time_period`, `value`, `unit`, the facet
    columns, `status`, so the role columns are every column between the
    first and `time_period`.
    """
    roles = frame.columns[1 : frame.columns.index(TIME_PERIOD_COLUMN)]
    return roles, len(frame.table.column(0).unique())

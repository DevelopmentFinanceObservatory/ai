# analysis-mcp

`analysis-mcp` provides analytical operations over tabular data frames stored in the shared materialisation store. The service exposes eight analytical tools over Model Context Protocol (MCP), executes deterministic operations defined by structured operation specifications, generates model-facing text digests, and records execution traces with lineage metadata.

## Analytical tools

The service registers eight analytical tools.

| Tool        | Operation                                                                                                                         |
| ----------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `aggregate` | Groups rows by specified dimensions and computes a sum, mean, max, or min reduction, reporting the winning period for max and min |
| `compare`   | Computes period-aligned values and absolute differences across entities or frames                                                 |
| `growth`    | Computes each entity's change between the window's two endpoint years, excluding and reporting any entity it cannot compute       |
| `rank`      | Orders entities by observed values or windowed aggregations                                                                       |
| `share_of`  | Calculates proportional shares between numerator and denominator definitions across time periods                                  |
| `join`      | Merges two distinct frames on matching time periods and shared role dimensions                                                    |
| `filter`    | Selects rows by an ordered threshold, including a bound on `time_period`, by a category set, or by a level of place               |
| `sum_of`    | Adds two single-series slices for each period both report, leaving out a period only one side reports                             |

## Operation specifications and execution pipeline

Every analytical request executes through a standard pipeline.

1. The tool parses incoming arguments into a typed `OpSpec` model with strict field validation.
1. The pipeline normalises strings, sorts column lists, and orders entity bindings for deterministic evaluation.
1. The system computes a canonical RFC 8785 JSON Canonicalization Scheme (JCS) hash of the operation specification (`op_spec_hash`).
1. The analytical engine evaluates the operation, applies coverage floor thresholds, and produces an output data frame.
1. The service mints an output content version (`ocv_` format) from the operation hash, input content versions, and execution identity.
1. The resulting frame and trace are saved to the materialisation store under the generated pair identifier, and the service returns a formatted digest and structured reference metadata.

## Frame reference resolution

Analytical tools receive inputs as `FrameRef` pairs consisting of a `spec_id` and a `content_version`.

The service resolves these references against the `InMemoryMaterialisationStore`. During resolution, the pipeline performs three safety checks.

- Input frame row counts are validated against `max_frame_rows`.
- Upstream lineage and frame byte hashes are verified against recorded trace metadata.
- A frame's `time_period` column must hold points at one grain, or one span such as `2018/2022` on every row. Anything else refuses `mixed_time_grain`.

## Refusal codes

A tool call that fails returns a typed refusal, carrying a `code`, a `message`, and a machine-readable `fix`. Each code maps to exactly one retry mode. `correct_and_retry` means a different argument value can succeed. `do_not_retry` means the identical call reaches the identical outcome, so the caller needs a different approach. `retry_after_delay` means the identical call can succeed after the wait `retry_after_s` names.

`op_spec_invalid`, `role_ambiguous`, `frame_not_held`, and `frame_expired` come from the pipeline stages every verb shares: op-spec validation and frame reference resolution. `frame_too_large` and `internal_error` are also shared, the former from the row-count ceiling every input frame is checked against, the latter from the registration-site backstop that catches an unanticipated failure at the wire. `service_unavailable` is shared too: it fires when this process's frame row budget (`common.row_budget`, sized by `FRAME_ROWS_IN_FLIGHT_MAX`) has no room for a resolved input's rows, and the same call fits once other calls release theirs.

| Code                                 | Retry               | Returned by                                                            |
| ------------------------------------ | ------------------- | ---------------------------------------------------------------------- |
| `op_spec_invalid`                    | `correct_and_retry` | every verb                                                             |
| `stat_var_unresolved`                | `correct_and_retry` | `aggregate`, `growth`, `join`, `rank`, `share_of`                      |
| `role_ambiguous`                     | `correct_and_retry` | every verb                                                             |
| `operation_uncovered`                | `correct_and_retry` | `aggregate`, `rank`                                                    |
| `frame_too_large`                    | `correct_and_retry` | every verb                                                             |
| `unit_heterogeneous`                 | `correct_and_retry` | `aggregate`, `compare`, `growth`, `join`, `rank`, `share_of`, `sum_of` |
| `facet_heterogeneous`                | `correct_and_retry` | `aggregate`, `compare`, `growth`, `join`, `rank`, `share_of`, `sum_of` |
| `coverage_insufficient`              | `correct_and_retry` | `aggregate`, `rank`                                                    |
| `non_finite_result`                  | `do_not_retry`      | `aggregate`, `compare`, `growth`, `rank`, `share_of`, `sum_of`         |
| `frame_not_held`                     | `correct_and_retry` | every verb                                                             |
| `frame_expired`                      | `correct_and_retry` | every verb                                                             |
| `mixed_time_grain`                   | `do_not_retry`      | every verb                                                             |
| `comparison_periods_disjoint`        | `correct_and_retry` | `compare`, `sum_of`                                                    |
| `growth_gap_year`                    | `correct_and_retry` | `growth`                                                               |
| `cross_frame_unit_mismatch`          | `correct_and_retry` | `compare`, `share_of`, `sum_of`                                        |
| `cross_frame_unit_unbound`           | `correct_and_retry` | `compare`, `share_of`, `sum_of`                                        |
| `cross_frame_facet_mismatch`         | `correct_and_retry` | `compare`, `share_of`                                                  |
| `share_of_grain_mismatch`            | `correct_and_retry` | `share_of`                                                             |
| `filter_predicate_type_mismatch`     | `correct_and_retry` | `filter`                                                               |
| `filter_unit_conversion_unsupported` | `correct_and_retry` | `filter`                                                               |
| `share_of_zero_denominator`          | `correct_and_retry` | `share_of`                                                             |
| `join_output_collision`              | `correct_and_retry` | `join`                                                                 |
| `join_no_common_key_columns`         | `correct_and_retry` | `join`                                                                 |
| `derived_input_unusable`             | `correct_and_retry` | `growth`                                                               |
| `derived_input_ordinal`              | `correct_and_retry` | `growth`, `rank`, `share_of`                                           |
| `derived_input_unrecoverable`        | `do_not_retry`      | every verb                                                             |
| `place_levels_mixed`                 | `correct_and_retry` | `aggregate`, `rank`, `share_of`                                        |
| `service_unavailable`                | `retry_after_delay` | every verb                                                             |
| `internal_error`                     | `do_not_retry`      | every verb                                                             |

`unit_heterogeneous` covers both a frame that carries more than one declared unit and a pinned unit absent from the frame's declared units. `derived_input_unusable` fires when `growth` reads a frame whose rows carry one span, such as the `2018/2022` a sum over years or a collapsed `rank` writes, and its `fix` names that span. Growth reads one row per period, so the retry is growth over the frame that result was computed from. The other verbs accept a span frame: `rank` and `aggregate` keep the span, `compare`, `sum_of`, `share_of` and `join` match on the span string, and `filter` refuses only an `ordered_threshold` on `time_period` (`filter_predicate_type_mismatch`). Over any frame, an `ordered_threshold` on `time_period` takes a point such as `2018`, and an interval operand refuses with the same code. `derived_input_ordinal` fires when an input frame's `value` is a rank position, which a per-period `rank` with no `limit` writes, or a result derived from one: `share_of` refuses it on either side, `growth` refuses it with `change="relative"` and computes a change in places with `change="absolute"`, and `rank` refuses it in either mode, since highest first over positions puts the last-placed entity first. A top or bottom N per period is one `rank` call with `per_period` and `limit` set. `derived_input_unrecoverable` fires on any result an engine before 0.3.0 computed, whose collapsed and grown rows carry one borrowed year in place of the years they cover. Its `fix` names the engine version, and the analysis has to run again from the source data.

`place_levels_mixed` fires over a frame carrying a place column's type enrichment, `observationAbout_type` from `get_data(place_types=true)`, whose several places share no level a `place_level` filter can keep (`common.place_type.PLACE_LEVELS`): a frame that names no place holds the world, regions, World Bank groups, countries and their subdivisions side by side. `rank` refuses it over the places it ranks, `aggregate` when `group_by` leaves out both the place and its type, and `share_of` on either side, whether the side sums across places or keys by them. A `filter` with a `place_level` predicate on the type column, such as `Country`, keeps one level, and the same calls then run. A frame whose places share such a level, such as the countries dfo-apis serves when it cannot list every place, runs with no filter, and so does a frame holding one place, whatever its type. A type outside those levels names no level: Earth (`Place`) beside countries typed `Country,Place` is refused, and so are several World Bank groups sharing `ProvisionalNode` or UN regions sharing `UNGeoRegion`, which overlap and which no `place_level` filter keeps: Data Commons types both Sub-Saharan Africa and the Eastern Africa it contains `UNGeoRegion`.

A place sits at each level its Data Commons types name (`common.place_type.levels_of`), with one exception: a place carrying a Eurostat NUTS type is not at `Country`. NUTS divides the territory of an EU or EFTA state, so the place's figures sit inside that state's. Data Commons types Åland (`nuts/FI2`) both `Country` and `EurostatNUTS1`, and Finland's figures include it, so a `Country` filter leaves it out and a total across countries counts it once. A territory typed `Country` and `AdministrativeArea1` with no NUTS type, such as Hong Kong, Guam or Greenland, stays a country, since its sovereign state's figures leave it out. Across GDP's 1,996 places and female life expectancy's 1,885 on the development mixer, 14 places carry `Country` beside another level type: 13 such territories, and Åland, which only GDP holds.

## Digests and trace metadata

Every successful tool execution produces two outputs.

### Model digest

The primary tool response is a text digest designed for language model consumption. Digests contain structured summaries, formatted numbers, group breakdowns, realised coverage statistics, and active warnings.

### Trace record

The service stores an execution trace with lineage and replay metadata.

- `upstream_trace`: Original materialisation lineage, including dataset revisions and applied facets.
- `op_spec_hash`: Cryptographic digest of the canonical operation specification.
- `execution_record`: Canonical JCS serialisation of the operation specification for replay.
- `applied_policy`: Record of coverage fractions, skipped cells, and derived metadata flags.
- `content_version`: Minted toolkit content version for the output frame.
- `input_content_versions`: Content versions of all consumed input frames.
- `execution_identity`: Python runtime version, package version, and engine build identifier.

## Configuration and environment variables

The service reads configuration through `Settings`.

| Variable                   | Type    | Default  | Description                                                                        |
| -------------------------- | ------- | -------- | ---------------------------------------------------------------------------------- |
| `SEED_FIXTURE_FRAMES`      | Boolean | `false`  | Seeds the materialisation store with development finance fixture frames on startup |
| `MAX_FRAME_ROWS`           | Integer | `250000` | Maximum row count permitted for an input frame                                     |
| `AGGREGATE_COVERAGE_FLOOR` | Float   | `0.5`    | Minimum realised cell coverage fraction required for reductions                    |

## Testing and baseline updates

Run the test suite using `uv`.

```bash
uv run pytest services/analysis-mcp/tests
```

A recipe in `justfile` runs the package tests.

```bash
just test-analysis-mcp
```

Digest outputs are verified against committed test baselines. When changes to digest formatting are intended, update the baseline file.

```bash
just analysis-digest-baseline
```

# analysis-mcp

`analysis-mcp` provides analytical operations over tabular data frames stored in the shared materialisation store. The service exposes nine analytical tools over Model Context Protocol (MCP), executes deterministic operations defined by structured operation specifications, generates model-facing text digests, and records execution traces with lineage metadata.

## Analytical tools

The service registers nine analytical tools.

| Tool          | Operation                                                                                                                                                                                                  |
| ------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `aggregate`   | Groups rows by specified dimensions and computes a sum, mean, max, or min reduction, reporting the winning period for max and min                                                                          |
| `compare`     | Computes period-aligned values and absolute differences across entities or frames                                                                                                                          |
| `growth`      | Computes each entity's change between the window's two endpoint years, excluding and reporting any entity it cannot compute                                                                                |
| `rank`        | Orders entities by observed values or windowed aggregations                                                                                                                                                |
| `share_of`    | Calculates proportional shares between numerator and denominator definitions across time periods                                                                                                           |
| `join`        | Merges two distinct frames on matching time periods and shared role dimensions                                                                                                                             |
| `filter`      | Selects rows by an ordered threshold, including a bound on `time_period`, by a category set, or by a level of place                                                                                        |
| `sum_of`      | Adds two single-series slices for each period both report, leaving out a period only one side reports                                                                                                      |
| `quotient_of` | Divides one measure by another at each period both report and states a composed unit, leaving out and recording unmatched periods, unmatched keys and zero denominators, and refusing when nothing is left |

It also registers `resolve_anchor`, which computes no frame. A `growth`, `rank` or `filter` call can state a period as rolling: relative to the latest period of the data, the newest calendar year at least half of the series hold. `resolve_anchor` takes the populations of a question (each root dataset with the bindings that narrow its series) and the root frames, finds that period, stores it as an anchor record in the materialisation store, and returns an `anchor_key`, the period, and the series that hold no value at it. A rolling call carries the key and the literal periods written from the period, and runs only when the record covers it: the same populations, every root frame the input was computed from, and each literal equal to the period moved by its offset. Anything else refuses `rolling_anchor_unresolved`, with the failed check in `fix.detail`.

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

| Code                                 | Retry               | Returned by                                                                           |
| ------------------------------------ | ------------------- | ------------------------------------------------------------------------------------- |
| `op_spec_invalid`                    | `correct_and_retry` | every verb                                                                            |
| `stat_var_unresolved`                | `correct_and_retry` | `aggregate`, `growth`, `join`, `quotient_of`, `rank`, `share_of`                      |
| `role_ambiguous`                     | `correct_and_retry` | every verb                                                                            |
| `operation_uncovered`                | `correct_and_retry` | `aggregate`, `rank`                                                                   |
| `frame_too_large`                    | `correct_and_retry` | every verb                                                                            |
| `unit_heterogeneous`                 | `correct_and_retry` | `aggregate`, `compare`, `growth`, `join`, `quotient_of`, `rank`, `share_of`, `sum_of` |
| `facet_heterogeneous`                | `correct_and_retry` | `aggregate`, `compare`, `growth`, `join`, `quotient_of`, `rank`, `share_of`, `sum_of` |
| `coverage_insufficient`              | `correct_and_retry` | `aggregate`, `rank`                                                                   |
| `non_finite_result`                  | `do_not_retry`      | `aggregate`, `compare`, `growth`, `quotient_of`, `rank`, `share_of`, `sum_of`         |
| `frame_not_held`                     | `correct_and_retry` | every verb                                                                            |
| `frame_expired`                      | `correct_and_retry` | every verb                                                                            |
| `mixed_time_grain`                   | `do_not_retry`      | every verb                                                                            |
| `comparison_periods_disjoint`        | `correct_and_retry` | `compare`, `quotient_of`, `sum_of`                                                    |
| `growth_gap_year`                    | `correct_and_retry` | `growth`                                                                              |
| `cross_frame_unit_mismatch`          | `correct_and_retry` | `compare`, `share_of`, `sum_of`                                                       |
| `cross_frame_unit_unbound`           | `correct_and_retry` | `compare`, `share_of`, `sum_of`                                                       |
| `cross_frame_facet_mismatch`         | `correct_and_retry` | `compare`, `share_of`                                                                 |
| `share_of_grain_mismatch`            | `correct_and_retry` | `share_of`                                                                            |
| `filter_predicate_type_mismatch`     | `correct_and_retry` | `filter`                                                                              |
| `filter_unit_conversion_unsupported` | `correct_and_retry` | `filter`                                                                              |
| `share_of_zero_denominator`          | `correct_and_retry` | `share_of`                                                                            |
| `quotient_of_zero_denominator`       | `correct_and_retry` | `quotient_of`                                                                         |
| `quotient_of_no_key_matched`         | `correct_and_retry` | `quotient_of`                                                                         |
| `quotient_of_side_unit_unsupported`  | `correct_and_retry` | `quotient_of`                                                                         |
| `composed_unit_sum_unsupported`      | `correct_and_retry` | `aggregate`, `rank`, `share_of`, `sum_of`                                             |
| `join_output_collision`              | `correct_and_retry` | `join`                                                                                |
| `join_no_common_key_columns`         | `correct_and_retry` | `join`                                                                                |
| `derived_input_unusable`             | `correct_and_retry` | `growth`                                                                              |
| `derived_input_ordinal`              | `correct_and_retry` | `growth`, `quotient_of`, `rank`, `share_of`                                           |
| `derived_input_unrecoverable`        | `do_not_retry`      | every verb                                                                            |
| `place_levels_mixed`                 | `correct_and_retry` | `aggregate`, `quotient_of`, `rank`, `share_of`                                        |
| `rolling_anchor_unresolved`          | `correct_and_retry` | `resolve_anchor`, `growth`, `rank`, `filter`                                          |
| `rolling_grain_unsupported`          | `do_not_retry`      | `resolve_anchor`                                                                      |
| `service_unavailable`                | `retry_after_delay` | every verb                                                                            |
| `internal_error`                     | `do_not_retry`      | every verb                                                                            |

`unit_heterogeneous` covers both a frame that carries more than one declared unit and a pinned unit absent from the frame's declared units. `derived_input_unusable` fires when `growth` reads a frame whose rows carry one span, such as the `2018/2022` a sum over years or a collapsed `rank` writes, and its `fix` names that span. Growth reads one row per period, so the retry is growth over the frame that result was computed from. The other verbs accept a span frame: `rank` and `aggregate` keep the span, `compare`, `sum_of`, `share_of`, `quotient_of` and `join` match on the span string, and `filter` refuses only an `ordered_threshold` on `time_period` (`filter_predicate_type_mismatch`). Over any frame, an `ordered_threshold` on `time_period` takes a point such as `2018`, and an interval operand refuses with the same code. `derived_input_ordinal` fires when an input frame's `value` is a rank position, which a per-period `rank` with no `limit` writes, or a result derived from one: `share_of` and `quotient_of` refuse it on either side, `growth` refuses it with `change="relative"` and computes a change in places with `change="absolute"`, and `rank` refuses it in either mode, since highest first over positions puts the last-placed entity first. A top or bottom N per period is one `rank` call with `per_period` and `limit` set. `derived_input_unrecoverable` fires on any result an engine before 0.3.0 computed, whose collapsed and grown rows carry one borrowed year in place of the years they cover. Its `fix` names the engine version, and the analysis has to run again from the source data.

`share_of` and `quotient_of` divide each numerator key over the periods it shares with its denominator group, and record each period one side holds alone as a `period_unmatched` exclusion on that side. Debt service reported for 1970-2024 over GDP reported for 1960-2025 gives 55 shares, and the 11 GDP years outside them are recorded. Both digests name the periods left out and each side's coverage. A numerator key that shares no period with its group, such as France reporting to 2026 against a denominator holding only 2027, is left out of the result, named in the digest's `numerator keys left out` line, counted in the result's `dropped_key_count` and in `applied_policy.detail["keys_without_shared_period_count"]`, and counted as entities under `period_unmatched` in the step, while every other key keeps its shares. A numerator key with no denominator group under `denominator_by` is left out the same way, in both verbs: named in the same line, counted in `dropped_key_count` and in `applied_policy.detail["unmatched_key_count"]`, recorded as an `unmatched_key` exclusion, and counted as entities under `unmatched_key` in the step. `share_of_grain_mismatch` fires only when no key shares a period, and its `fix` names both sides' periods and, for a call with one key, that key. `share_of_key_unmatched` fires only when no numerator key has a group, and its `fix` samples each side's keys and counts them, as `quotient_of_no_key_matched`'s does.

A `share_of` result holds the share in per cent: debt service of 5.33 billion dollars over GDP of 100 billion is stored as `5.33`, with unit `Percent`, the ref apis writes on a published share such as World Development Indicators' "% of GDP" series. A computed share therefore compares and plots with a published one. Its rows sit under the operation's own StatVar, `dcid:<op_spec_hash>`, which the trace describes as a `share` of the numerator's StatVar by the denominator's. The digest names the unit in its header and states the first and latest share on the next line, before the per-period list. An absolute `growth` over a `Percent` series reads in percentage points, and `quotient_of` refuses a share as a side with reason `dimensionless`.

`cross_frame_facet_mismatch` fires when the two sides of `share_of` or `compare` hold the same StatVar and declare different values on a facet axis (`provenance` or `measurementMethod`). That is one measure from two sources or methods, and its values do not combine. Its message asks for that measure read under one of the two refs on both sides, for example by binding `scope.facets` in `get_data`. Two sides holding different StatVars, such as IDS debt service over WDI GDP, keep their own selections and run with a warning naming both values, and a `compare` result leaves that axis unbound. `share_of` records the numerator's selections as `resolved_facets` and the denominator's as `denominator_resolved_facets`, and `compare` records the left side's as `resolved_facets` and the right side's as `right_resolved_facets`. One measure published under two StatVar dcids, such as an IDS and a WDI copy of one series, also runs with that warning, since the engine sees only the `stat_var` column.

`quotient_of` divides any pair of units, so it never raises `cross_frame_unit_mismatch`, `cross_frame_unit_unbound` or `cross_frame_facet_mismatch`. A unit the two sides disagree on becomes the composed unit, and a facet axis they disagree on becomes a warning. A matched period whose denominator sums to zero is left out and recorded, and `quotient_of_zero_denominator` fires when every matched period is. `quotient_of_no_key_matched` fires when no numerator key has a denominator group, and its `fix` samples each side's keys. `quotient_of_side_unit_unsupported` fires when a side can be no part of a composed unit. Its `fix["reason"]` is `dimensionless` for a `share_of` or relative `growth` result, `composed` for a `quotient_of` result, and `operation_stat_var` for unit-less rows under a `sum_of` result's own stat_var.

`composed_unit_sum_unsupported` fires when a verb is asked to add values in a composed unit, the unit a `quotient_of` result carries: `aggregate` and `rank` with `aggregation="sum"`, `sum_of` when either side resolves to one, and `share_of` when a side resolving to one holds more than one row in a (key, period) cell. Two figures per head added together are not a figure per head. Its `fix` names the verb, the unit and, for `sum_of` and `share_of`, the side. A `share_of` side of one row per cell computes, as does a mean, max or min over the same frame, and one figure pooled across entities comes from totalling each measure across them and dividing the totals with `quotient_of`. A frame that holds a composed unit beside another unit refuses `unit_heterogeneous` first.

`place_levels_mixed` fires over a frame carrying a place column's type enrichment, `observationAbout_type` from `get_data(place_types=true)`, whose several places share no level a `place_level` filter can keep (`common.place_type.PLACE_LEVELS`): a frame that names no place holds the world, regions, World Bank groups, countries and their subdivisions side by side. `rank` refuses it over the places it ranks, `aggregate` when `group_by` leaves out both the place and its type, and `share_of` and `quotient_of` on either side, whether the side sums across places or keys by them. A `filter` with a `place_level` predicate on the type column, such as `Country`, keeps one level, and the same calls then run. A frame whose places share such a level, such as the countries apis serves when it cannot list every place, runs with no filter, and so does a frame holding one place, whatever its type. A type outside those levels names no level: Earth (`Place`) beside countries typed `Country,Place` is refused, and so are several World Bank groups sharing `ProvisionalNode` or UN regions sharing `UNGeoRegion`, which overlap and which no `place_level` filter keeps: Data Commons types both Sub-Saharan Africa and the Eastern Africa it contains `UNGeoRegion`.

A flow party role (`financingProvider`, `financingRecipient`) with its kind column (`financingRecipient_party_kind`, which `get_data(place_types=true)` adds to a flow frame beside `financingRecipient_party_class` and `financingRecipient_party_whole`) follows a different rule, since DAC tables publish group totals as parties of their own. `aggregate`, `rank`, `share_of` and `quotient_of` leave out every party of kind `total` (a DAC regional total, an income group, the G7) and record those rows as one `party_total` exclusion naming them. Parties of kind `party` and `residual`, such as "Africa, regional", add up. A role of totals alone whose `{role}_party_whole` cell is `true` on every row, such as "Official Donors, Total" beside "Private Donors, Total", is summed, since whole totals on one axis hold disjoint populations. A whole total beside members stays when no member lies in its class: it is summed with them on a pooled role and keeps its own row on a role a verb groups by. A whole total that may hold a member (it has no class, or a member is of its class or of none) refuses on a pooled role and is left out of a grouped one. A role `rank` ranks keeps only parties of kind `party` where it holds any, and records its residuals as one `party_residual` exclusion. A sum's digest counts the private parties it pooled, the rows whose `{role}_party_class` cell is `private`, members and whole totals alike, with their value, and a rank tags each private party. A frame with no `{role}_party_class` column says nothing about private parties. `place_levels_mixed` fires there when several parties include one with no kind, when every party on a pooled role is a total and not every one is whole, or when a role holding several parties, pooled or ranked, carries no kind column. Over a kind column the refusal names a search naming the population ("all donors") for a total across the side, and a filter keeping `party` and `residual` for a sum of members; its `fix.member_kinds` lists the kinds a filter keeps to compare or rank members (`party`). A `share_of` whose share exceeds 100 per cent carries a `share_exceeds_whole` warning.

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
- `steps`: The step record of the verb's own operation: its inputs, policy, the caller's parameters, what it kept and left out, and the unit it produces. presentation-mcp joins the steps along a frame's ancestry and sends them when it publishes the frame.
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

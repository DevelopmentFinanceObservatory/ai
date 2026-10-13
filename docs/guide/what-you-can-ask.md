---
description: How a question to the DFO MCP server becomes a cited answer, and the limits and refusals that bound it.
---

# How the MCP server answers questions

Once the DFO is connected to your AI assistant (see [Connect](connect.md)), you can ask it questions about development finance in plain language, such as "What is health spending per person in Kenya?" or "How much health aid did Nigeria receive between 2020 and 2024?" The DFO covers countries, subregions, and continents. It can look up a figure or trend, compare places, rank them, or calculate per-person amounts, and every answer names the dataset it came from. You'll get the best results by naming the place, years, and unit you want. The DFO can't handle very large requests in one go. If you ask for something it can't do, or for years it doesn't have data for, the assistant will tell you and suggest an alternative.

## For developers

The DFO MCP server answers questions about any series it holds for a country, subregion, or continent. You can compare, rank, aggregate, or divide the results. Every answer carries the dataset's source and a frame reference that names the exact data used.

## One question, start to finish

The example below follows one question from start to finish in DFO Studio, which calls the same tools. The session is trimmed to the tool calls and the answer.

```text
Looking into health spending in Kenya.

search    Resolve the question against the data graph
          resolved: definite
get_data  Fetch the data
          24 observations · 2000–2023 · WorldDevelopmentIndicators
…

"provenance": {
  "inputs": [{"spec_id": "spec_0002907ee44bf25fc98f343372614a84",
              "content_version": "cv_bec21df8ebf432401c160f10858e61fb"}],
  "sources": [{"id": "dc/base/WorldDevelopmentIndicators",
               "name": "WorldDevelopmentIndicators",
               "licence": "not stated by the platform"}],
  "freshness": {"data_through": "2023", "last_refreshed": "2026-08-29", "partial_period": false},
  …
  "citation": "WorldDevelopmentIndicators"
}

Based on data from the World Development Indicators, healthcare expenditure per person
in Kenya spanned from 2000 to 2023 across 24 observations:

* Healthcare Expenditure per Person: Started at 75.5 International Dollars in 2000 and
  increased to 274.0 International Dollars by 2023.
```

The omitted calls are Studio's chart-drawing tools.

`search` turned the question into one dataset definition with a `spec_id`. The id identifies what is measured, where, and when. It stays the same when the data is refreshed.

`get_data` returns the data for that id and a frame reference containing `spec_id` and `content_version`. The `content_version` identifies this retrieval. When the World Bank revises the series, the same `spec_id` returns a new `content_version`. An answer that quotes both ids can be checked against the data it used.

A direct MCP call to `get_data` with the same id returned the pair and a plain-text citation:

```text
Retrieved the requested data for observationAbout country/KEN.
Coverage: 24 rows · 1 observationAbout, 24 years · 2000-2023
Frame: spec_id=spec_0002907ee44bf25fc98f343372614a84, content_version=cv_bec21df8ebf432401c160f10858e61fb -- pass both to downstream tools.
Source: WorldDevelopmentIndicators (WorldDevelopmentIndicators).
…
```

Kenya's series is available in international dollars and US dollars. The server chose international dollars because the question specified no unit. To request US dollars, the model passes `scope` with a `facets` choice on its next `get_data` call.

<!-- TODO: verify this get_data output against the public endpoint once it exists. -->

## Comparing two series

`compare` accepts two frame references, or one frame reference and two entities. It compares values over the periods both sides cover. To answer "How does Kenya's debt service compare to its climate finance receipts?", run `search` with two parts, call `get_data` once per part, then call `compare` with both frame references.

<!-- TODO: replace this paragraph with a real session that runs search, get_data and compare. No recorded session calls compare yet. -->

The analysis tools operate on frame references. `aggregate` groups and reduces, `growth` measures change across a window, `rank` orders entities, `share_of` gives one entity's share of another in one unit, `quotient_of` divides one measure by another in any two units, `sum_of` adds series, `filter` keeps matching rows, and `join` combines two frames.

## Dividing one measure by another

`quotient_of` divides one series by another at each period both report, whatever their units, and states the result in the unit the two compose. To answer "What was Kenya's debt service per head of population?", run `search` with two parts, one for debt service and one for population, call `get_data` once per part, then call `quotient_of` with both frame references. The population frame declares no unit, so the result's unit names it by its statistical variable (StatVar) identifier: `CurrencyUSD` per `Count_Person`.

To get one figure per entity, such as aid per head for each of several recipients, set `numerator_by` to the numerator's role column and `denominator_by` to the column the population frame holds the same entities on. A period only one side reports is left out and recorded, and so is a key with no counterpart or a period where the denominator is zero. The call refuses when no period is shared, when no key pairs, or when every matched period has a zero denominator. `share_of` divides a part by its whole in one unit and refuses two sides in different units.

## Skills

A `search` result can name skills that apply to it, each with an id and a one-line description, beneath the part blocks: "Skills for these results; get_skill loads one by id:". `get_skill(skill_id)` loads one as a SKILL.md document, front matter followed by its body, cut past 32,000 characters with a line giving the full length. An id no skill carries refuses `skill_not_found`.

## Limits and refusals

A frame holds at most 250,000 rows. `get_data` returns `over_ceiling` for a larger retrieval, and the analysis tools return `frame_too_large` for a larger input frame. Both refusals ask the model to narrow the request and retry.

A refusal arrives as a tool result marked as an error. Its text explains the problem and what to do. Structured content carries a `code`, a `retry_mode` and a `fix`. A Studio session asked for French inflation from 2015 to 2024, although the series ended in 2022. The `growth` call returned:

```text
growth_gap_year: {'window_start': '2015', 'window_end': '2024', 'entity_count': 1, 'missing_start': 0, 'missing_end': 1, 'suggested_window': ['2015', '2022'], 'suggested_window_entities': 1}
```

Growth reads each series' value in the first and last year of the window, and this series has no 2024 value. The fix suggests 2015 to 2022, a window in which the series has values in both the first and last year.

The `retry_mode` tells the model what to do next:

- `correct_and_retry`: change an argument and call again.
- `refresh_then_retry`: fetch fresh state, such as a new `get_data` call, then retry.
- `retry_after_delay`: retry the same call later, after a service outage or while a large dataset is still being prepared.
- `do_not_retry`: this call cannot succeed on this server.

The refusals a reader meets most often:

- The DFO resolves countries, subregions, continents, continental unions, places below a country, organisations, DAC aggregates and membership groups such as the G7, whose members it scopes in place of the group. "World" resolves to the world as one place. An ambiguous name such as "Congo" returns `entity_ambiguous` with up to five candidates to choose from. A name with no match returns `entity_not_found`.
- An unknown `spec_id` returns `definition_not_held`, and the model runs `search` again.
- A large retrieval, such as aid from every donor to one country, can take dfo-apis longer than one call allows. `get_data` then returns `operation_still_processing`. The same call made again picks up the same background operation while the data's coverage is unchanged.
- A mid-size retrieval that dfo-apis runs in the foreground can still take longer than the roughly 30 seconds this server gives a synchronous retrieval inside its 45-second answer window. `get_data` then returns `service_unavailable`, naming the retrieval as possibly still running upstream and suggesting a narrower scope, such as fewer entities or a shorter window, which finishes faster.
- `compare` requires a common year and the same unit on both sides. It returns `comparison_periods_disjoint` when periods do not overlap and `cross_frame_unit_mismatch` when units differ.
- `growth` returns `growth_gap_year` when no series has values in both the first and last year of the window, as above. A series missing only one of those years is left out of the result and listed as excluded, and missing years between them never refuse.
- The server returns `capability_unavailable` with `do_not_retry` for a feature this deployment has disabled.

The ordinary `search` results `no_data` and `unsupported` can resemble refusals. `no_data` means the DFO holds nothing for that part of the question and may suggest nearby series. `unsupported` means the question is outside the DFO's scope and gives the reason.

A single `search` call takes up to six parts, one per measured variable, and `describe` takes up to eight ids.

## Next

- [Connect](connect.md) shows how to add the server to Claude, ChatGPT, Gemini, Claude Code, Codex, or Gemini CLI.
- The Tools reference lists every argument, error code and retry mode. <!-- TODO: link when the generated Tools reference is published. -->

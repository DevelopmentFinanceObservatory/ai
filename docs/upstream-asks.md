# Upstream asks (apis / platform) from the studio v1.1 build

Each item names the studio screen that needs it and what the harness does
in the meantime. Live traces reference the hosted dev deployment.

## 1. Intent fields for card presentation (the Edit Card panel)

The Edit Card panel edits subtitle, description, axis labels, and style.
The platform intent schema is `type / frame_ref / select / title / window`
(verified live via `GET /api/presentation/release`), so none of those
fields can live on the component.

**Ask (optional):** subtitle/description/axis-label/style fields on the
intent, if ONE wants them platform-durable.
**Meanwhile:** the harness owns them on its composition manifest
(`CompositionBlock.meta`, written by the `set_card_meta` op) — no platform
involvement needed, additive migration if the intent grows them later.

## 2. Revisable `window` / windowed fetch (the Edit Card panel and the plan card)

Verified against the HOSTED dev platform (2026-08-28): a revise patch was
accepted only with `title` and/or `select` — a patch carrying `window`
(or `type`, even unchanged) refused with "'type' is create-only and
cannot be revised" / "legal values: title, select". Note: this repo's
presentation-mcp test assembler is more permissive (it allows window
merge-patches), so either the hosted deployment or the in-repo contract
stub is behind — worth confirming which. And `get_data` has no window parameters, while the
analysis `filter` verb is value-only by design ("a window is out of this
grammar"). So "only the last 10 years" is inexpressible end-to-end: plan
amendments can't window a fetch, and card edits can't window a chart
without recreating it.

**Ask:** either window parameters on the Datasets mint, or `window` made
revisable on components.
**Meanwhile:** window edits recreate the component (new id, block replaced
in place); window amendments are declined honestly in the plan's rationale.

## 3. Identity-reproduction failures on some specs (blocks compare/rank turns)

Some search envelopes fail data-mcp's identity check: "identity does not
reproduce spec_id". The failure is per-spec and deterministic:
"population of Ghana" (single-part) and Kenya+Uganda+Tanzania (3-part,
trace `tr_a2f3729c66804396`) fail; life expectancy / health spending /
inflation / Kenya+Tanzania compare resolve fine.

**Ask:** fix the non-reproducing spec ids (examples: expected
`spec_0e8100e4…` got `spec_f85f560d…`; expected `spec_f5dde1ad…` got
`spec_c473f804…`).
**Meanwhile:** the search step fails with the refusal and its trace id,
and the agent answers without that data.

## 4. `fact_summary.changed` is always empty (the Edit Card panel)

Component revisions from the hosted dev platform were observed returning
`changed: []` even when the revision genuinely changed title or selection
(observed: title patch 1→2, select narrowing 2 series → 1; the in-repo
contract models support non-empty `changed`). The wire `component.revised` frame requires a non-empty
`changed`, so the harness computes it from its own patch keys.

**Ask:** populate `changed` on revisions (nice-to-have; the harness
workaround is complete).

## 5. The datasets snapshot route is now production-critical

`GET /api/datasets/{id}/{cv}` (manifest + fresh signed artifact URL) is
the frontend's plot-data path: every chart card fetches its points through
it (`docs/frontend-interface.md`, section 5, verified on hosted dev
2026-08-30).

**Ask:** treat this route as production-critical — uptime, monitoring, and
no breaking changes to the manifest/artifact shape without a version bump.

## 6. Ambiguous questions need a `candidates` envelope (the More Details Question panel)

Search states a tie between readings only in warning prose. Observed live
(2026-08-31, "spending in Kenya"): search scores two readings within tolerance —
"'Government expenditure on military activities' and 'Amount of Debt:
Government' scored within 0.0001 of each other … surfacing both as
candidates" — but returns status `unsupported` with that tie described in
**warning prose**. The turn reads it as no data unless the agent picks
the tie out of that text.

**Ask:** when interpretations tie within tolerance, return status
`candidates` with the candidate specs, which the agent can offer the user
through `ask_user`.
**Meanwhile:** the tie reaches the agent only as warning text.

## 7. Skills for planner grounding

**Resolved.** apis serves skills through its Skills API, and data-mcp reads
them with `get_skill` and `get_skill_file`. The planner carries both as its own
tools (`services/harness/src/planner/skills.py`), called through the harness's
tool surface, and loads a skill that bears on the question. Where the Skills
API is not mounted, both tools refuse `capability_unavailable` and the planner drafts without the skill.

## 8. Frames produced by analysis-mcp are invisible to the presentation surface

**Resolved.** The rejection was not apis' id validator: `apis/src/presentation_api/path_params.py` passes an `ops_` spec_id and an `ocv_` content_version through its own format check and leaves the spec\_/ops\_ classification to `FrameStore.load`. The regexes that gate a `frame_ref` before it reaches the platform live in this repo, in `services/presentation-mcp/src/presentation_mcp/client.py` and `tool_schema.py`, and match `^(?:spec|ops)_[a-z0-9]{1,64}$` and `^(?:cv|ocv)_[a-z0-9]{1,64}$`.

apis exposes `POST /api/datasets/frames/publish` (`src/datasets_api/doors/publish.py`), which takes a derived frame's role keys and rows plus its derivation trace, renders frame1 through the same `render_and_store_frame1` every definition mint uses, and records a version row and artifact location under the frame's own `ops_`/`ocv_` pair. A repeat publish of the same pair with byte-identical content succeeds again; one carrying different bytes refuses `identity_collision` (409, not retryable) instead of silently reporting success.

`PlatformAssembler` calls the publish route lazily, ahead of the platform call itself, on both `get_frame_view` (`describe_frame`) and `create_component`/`revise_component` (`write_component`), so nothing upstream of these two tools has to publish a frame by hand first. `FrameStore.load`'s `ops_unsupported` reason keeps a narrower meaning: minted by analysis-mcp but not yet published to the platform, and retryable, since publishing the pair is the retry that resolves it.

The remaining gap was in spec-kit, not in apis or this repo: its component envelope schema accepted a `spec_`/`ops_` spec_id but only a `cv_` content_version, so stamping a resolved component with a real `ocv_` content_version failed envelope validation even with the platform and client-side checks clear. spec-kit widens that pattern to `^(cv|ocv)_[0-9a-f]{6,64}$` in commit `89ec9988aa2100e8e99684edc249a44a0720c448`. apis pins commit `91400b0470827a0f156c3a8e6f2cfb39ec280143` on branch `pin/envelope-ocv-on-44c8013d`, which cherry-picks that same envelope fix onto the previous pin. spec-kit main also carries the `facet_probe_ceiling` and `qualifier_not_bound` registry entries, which apis' `tests/platform/test_registry_binding.py` requires it to serve before it can pin past them.

## 9. Analysis verbs take raw column names (planner grounding)

`growth.entity_by`, `rank.rank_by`, `aggregate.group_by`, `join.join_keys`,
and filter predicates all name frame columns literally. The entity column is
a per-frame role (`observationAbout` on country frames, `reporter` on
others), so a planner writing a step before the frame exists has to guess
the name, and a wrong guess refuses `role_ambiguous` at run time. Observed
live on the pinned model: `country` for `observationAbout`, `year` for
`time_period`.

**Ask:** let these arguments take roles (`entity`, `time`) that the verb
resolves against the frame's declared role columns, keeping literal column
names as an override.
**Meanwhile:** the planner grammar names the fixed columns and the usual
entity column; a wrong name costs one replan and one further approval, after
which the turn closes naming the step that could not run.

## 10. A part refused for its denominator names neither side (per-head questions)

Search answers "debt service per head" for Kenya `no_data` with reason
`denominator_not_available`. The `no_data` block carries `reason`,
`category`, `message`, `elected_count`, `entity_terms`,
`ambiguous_entities` and `unbound_qualifiers`. It omits the
measure search read once the per-capita phrasing is set aside and the
denominator it looked for.

**Ask:** on `denominator_not_available`, return the measure's variable
text without the normalising phrase, the denominator as a variable text
or StatVar, and which of the part's entities the denominator is about.
**Meanwhile:** the agent reads the `no_data` reason and has to work out
the measure and the denominator itself before it can search for them as
two parts.

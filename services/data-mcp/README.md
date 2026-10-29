# data-mcp

Data MCP provides data discovery, dataset retrieval, graph description, and skill tools over the Model Context Protocol.

## Overview

The service connects LLM agents to the upstream data platform. It translates user queries into dataset definitions, mints concrete datasets into parquet artifacts, materialises frames into shared memory, and explains graph identifiers.

## Exposed tools

### search

The `search` tool finds datasets and indicators matching a structured query. It accepts split query parts, queries the upstream search API, stores resolved definitions in the definition store, and returns a text digest, a small structured answer, and the upstream envelope in the result's `_meta`.

#### Parameters

- `parts`: List of 1 to 6 `SearchPart` objects. Each part contains `variable` (string, 1 to 200 characters) and optional fields `entities` (list of `SearchEntity` objects with `text`, optional `role`, optional `members`, and an optional `name` or `dcid`), `time` (string, 1 to 100 characters), `unit` (string, 1 to 100 characters), and `sources` (list of 1 to 8 `SearchSource` objects with `text`, 1 to 120 characters, and optional `exclude` and `ref` (1 to 200 characters)).
- `variable` and each entity's `text` take the caller's own words. `role` and `time` are matched against a fixed grammar. `role` names which side of a flow the entity sits on: a word for where something originates resolves to "from", a word for where it is directed resolves to "to". `time` takes a bare year, a range ("2015 to 2020"), an open start ("since 2015"), or "latest". A `role_text_not_recognised` or `time_text_not_parsed` warning means that field needs rephrasing in the grammar.
- An entity whose `text` names a set of places (a continent, a subregion, or a group such as the European Union) takes `members: true` to ask for each member as its own series: "African countries" is `{"text": "Africa", "members": true}`. Without it the set reads as one place where the data holds a figure for it, and as its members where it holds none. The flag has no effect on a single place, and the wire request carries it only when true.
- `sources` holds each publisher or dataset the user names for a part, in the user's words, with `exclude: true` for one the user asks to leave out: "aid to Kenya, not from CRS" is `{"text": "CRS", "exclude": true}`. Each reaches apis as a part's `sources` entry, with `exclude` as `polarity` (`include` or `exclude`). apis resolves the words against every source it holds, so the tool description lists none. A lender or recipient the flow runs to or from is an entity with its role, not a source. Words apis cannot resolve come back as `no_data` with reason `source_not_resolved`, and words that name several sources as `source_ambiguous`. Ask which candidate the user means, then retry the same source entry with its chosen `ref`, retaining `text` and `exclude`.
- An entity's `name` or `dcid` says which entity its `text` means, so two entities written alike stay two rows: "aid from Congo to Congo" sends `{"text": "Congo", "role": "from", "name": "Democratic Republic of the Congo"}` and `{"text": "Congo", "role": "to", "name": "Republic of the Congo"}`. A row carrying both is refused before the call, as apis refuses it with `request_invalid`. A `no_data` part with reason `entity_ambiguous` lists each ambiguous entity by number, with its role and each candidate's label, dcid, and kind; the agent asks the user which one is meant and resends that row with the candidate's `dcid`, retaining `text` and `role`.
- `raw_query`: Optional string up to 2000 characters holding the original query text.

A compound ask splits into one part per measured variable.

```json
{
  "parts": [
    { "variable": "aid", "entities": [{ "text": "Kenya", "role": "to" }] },
    { "variable": "malaria deaths", "entities": [{ "text": "Kenya" }] }
  ]
}
```

#### Return format

Returns a `ToolResult` with the following items.

- Text content with a digest carrying one block per part, each categorised as `definite`, `candidates`, `no_data`, or `unsupported`; the envelope's own status is `mixed` when parts disagree.
- Every request asks the upstream door for option coverage. A `definite` block lists each facet axis's other observed values as `<dcid> (<label>)`, keyed by the axis name `get_data`'s `facets` scope takes.
- A `no_data` block counts the stat variables the check confirmed and found empty (`elected_count`), naming none.
- Every spec_id the digest offers is followed by what it measures: Search's measure label (`resolved.shape.label`), how many StatVars it holds (`resolved.stat_vars.member_count`, "at least" that many when `member_count_exact` is false), the axes they vary along (the open slots of `resolved.slots`), and the publisher codes of the family (`resolved.stat_vars.source_codes`), with the bound source's label once before the codes that source issues. No member StatVar is named.
- An `unsupported` block names the field the caller cannot bind, in plain words, and states the next step. Its warnings are limited to ones about the caller's own input.
- Beneath the blocks, up to five skills the envelope attaches to its specs, each by id and description, for `get_skill` to load. `common.skills.ordered_skill_ids` sets the order. Ids appear in order of first appearance across the specs the digest renders, then come ids no rendered spec names, in id order.
- Structured content holding the small answer a host may hand to a model. It states what the digest states, as data: the envelope's `status`, then per part a `label` (the part's `variable`), the part's `status`, its `specs` (each a `spec_id`, a `label` carrying the rendered sentence, and the measure line's facts as `measure`, `member_count`, `member_count_exact`, `varies_by` and `source_codes` (a list of `code` with its `source` label when known) when Search states them; the spec of a `definite` part, the candidates shown on a `candidates` part, the nearby specs a `no_data` part suggests), and the part's `warnings` when it has any. The ids of the skills the digest lists sit beside `parts` as `skills`, and the envelope's diagnostics warnings as `warnings`. Each warning is a `code` and a `message`. It stays under a kilobyte for a single-part answer.
- When the resolve door answers with a `catalogue` or `declined` `kind`, the digest is one statement of search's reading and carries no part blocks: a `catalogue` answer means the question asked what DFO holds (the `sources` tool answers that), a `declined` answer means it is out of scope or asks for a judgement rather than for data. Each names the country the question named, when it named one. The structured content then carries `kind`, an empty `parts`, the `scope` and, for `declined`, the `reason`. A body with no `kind` is a resolution, as every body was before.
- `_meta["org.one.dfo/search_envelope"]` holding the full JSON response envelope from the upstream resolution door, after its definitions were stored. Calling code reads the envelope there. A host keeps `_meta` out of a model's context. The combined server at `/mcp/` drops the envelope and the structured content before a result reaches an outside client.

A refusal keeps its own structured content, the teaching payload `code`, `retry_mode`, `message` and `fix`, and carries no envelope. A `default_facet_undefined` or `facet_not_available` refusal carries apis' `detail` and its `issues` in `fix`, each with its axis, alternatives, and own `code` and `detail`; `default_facet_undefined` adds `retryable` and, from `get_data`, the `spec_id` and `entities` it was minting. Its `retry_mode` is `retry_after_delay` when apis marks it retryable, `correct_and_retry` when an issue lists alternatives, and `do_not_retry` otherwise.

### get_data

The `get_data` tool retrieves the dataset behind a specified definition identifier. It applies optional scope overrides, requests dataset minting from the upstream platform, fetches the resulting parquet artifact, materialises a frame into the shared store, and returns a frame reference for analysis tools.

#### Parameters

- `spec_id`: String matching the pattern `^spec_[0-9a-f]{32}$`.
- `scope`: Optional `GetDataScope` object containing optional `entities` (dictionary mapping role names to lists of entity names or `dcid:`-marked dcids), `window` (dictionary specifying time bounds), and `facets` (dictionary specifying facet references).
- `place_types`: Optional, default `false`. Adds `observationAbout_type`, each place's Data Commons types sorted and comma-joined (`Country`, `AdministrativeArea1,State`, `Place` for Earth, `ProvisionalNode` for a World Bank group), through the mint's `options.columns`, which names no table column and so adds the type to every default column. A decoded `stat_var_<code system>` column stays. On a flow spec the call adds each party's type, kind, class and whole-population mark instead (`financingProvider_type`, `financingRecipient_type` and the party cells `party_kind`, `party_class` and `party_whole`). A spec that names no place covers every place with data at every level, and a `filter` on this column keeps one level before a rank or a total. The frame carries the column as a role, and a place with no type reads `untyped`.
- `without_party_totals`: Optional, default `false`. On a spec with a flow side, sends `options.without_party_totals` with the party cells (`options.columns` `["party_kind", "party_class", "party_whole"]`). With `place_types` also set, `options.columns` holds each party's type and the party cells, as `place_types` asks on a flow spec. apis leaves out, on each party role the identity it serves binds no entity on, the rows whose party its code index declares a group total (the DAC total, the G7, DAC EU members), unless every row on that role is one. The cut runs before any `groupby`, so a sum adds members alone. A cut that leaves rows out is a version of its own, so the manifest's `{spec_id, content_version}`, its coverage, `series_count` and statistics describe the rows served, and its `trace.party_totals_left_out` lists `{role, provenance, entities}` for each role and dataset that lost totals (`provenance` is null for rows with no dataset, and an entry without the key comes from an older trace whose cut covered every dataset); the digest names them in one line per role and dataset, and gives the whole-side search advice once per role. A cut that leaves nothing out serves the base version. On a spec with no flow side the parameter sends nothing.
- `breakdown`: Optional `{code_system, level}`, for example `{"code_system": "crsPurpose", "level": 0}`. Sets `options.groupby` to `{"by": ["time_period", <code_system>], "op": "sum", "levels": {<code_system>: <level>}}`, so apis sums every StatVar into its code's ancestor at that depth of the code hierarchy, per year and entity. `stat_var_<code_system>` names that ancestor and `stat_var` is `dcid:<spec_id>` on every row. It goes with `stat_vars` `"separate"`; beside `"sum"` it refuses `request_invalid` at `/stat_vars`. Before anything else is sent, `get_data` reads `GET /api/datasets/capabilities`. A document without `groupby_level_axes` refuses `capability_unavailable` with `fix.option` `breakdown`, and a code system that list does not name refuses `request_invalid` at `/breakdown/code_system`.
- Countries, subregions, continents, continental unions, places below a country, organisations such as the World Bank, DAC aggregates and membership groups (G7, the European Union) resolve as scope entities; a region or a group binds its member countries. "World" resolves to the world as one place. `entity_ambiguous` fires only when a name matches two or more distinct entities' own canonical or alias names, for example "Congo". Each candidate's `entity` is its dcid behind a `dcid:` marker (`dcid:country/COD`), and a `scope.entities` value carrying the marker is sent as that dcid with no resolve call, so a bare dcid such as `GroupOfSeven` passes as well as `country/COD`. An unmarked `namespace/id` such as `country/KEN`, as other tools' output shows dcids, is also sent as a dcid; any other value is a name. A not-held entity refuses `entity_not_found` with `fix.not_held` and no `entity_type`; a group or region with no member country refuses the same way and names its kind in `entity_type`.

#### Return format

Returns a `ToolResult` with the following items.

- Text content with a digest of retrieved series, coverage details, and the frame reference pair `{spec_id, content_version}`. The coverage line states the requested window beside the served one when they differ, and a coverage-scope warning renders before the frame reference and the statistics. A recognised unit code (the SDG price-basis family, or this deployment's own currency and constant-price refs) reads in plain words beside its code, for example `SDG_CU_USD_M` as "current prices, US dollars, millions". A frame whose `stat_var` column holds the family ref `dcid:<spec_id>` summed its family: the acknowledgement line names the family, by the version description's title, else by the cited sources, and says its StatVars were summed into one total series per provider, recipient, or entity, read from the role columns in `coverage.dimensions`. Each statistics block for the `dcid:<spec_id>` series is headed as the family's summed total. The result's `summed_stat_var_count` counts the trace's StatVars with rows, and `summed_stat_var_label` is the description's title, or `null`. A stored version's `stat_vars_summed` warning renders outside the five-warning budget.
- Resource links pointing to artifact resources for download URLs.
- Structured content containing the stripped manifest model, with the frame reference beside it as `frame_ref`. The manifest's own `{spec_id, content_version}` names apis' materialisation, which its artifacts are read under. A frame read with `place_types` is held under a `content_version` of its own (see Frame materialisation), so callers take the frame reference from `frame_ref`.

A mint over the coverage ceiling refuses `over_ceiling`, whose `fix.ceiling_kind` says whether narrowing `scope` can help. A `rows`, `bytes` or `requests` ceiling narrows with `scope`. A `member_cardinality` or `candidate_window` ceiling is fixed by the spec_id and calls for a narrower `search` instead, which `fix.remedy` tells the end user. A `candidate_window` ceiling is a family too broad for the mint to enumerate unaided, and its message gives the narrower `search`. A refusal that names no kind data-mcp maps states no `fix.ceiling_kind`. A stored definition that cannot be rebuilt because it binds an entity role the source no longer declares refuses `request_invalid`, names those roles in `fix.roles`, and offers a retry with them left out of `scope.entities` or a fresh `search`. A `stat_vars` `"sum"` mint over StatVars apis cannot add into one total refuses `rollup_not_summable` (`do_not_retry`). Its `fix.axes` names each axis the sum failed on, and its `fix.remedy` is a sentence for the end user. A mint over a definition whose StatVars the graph defines inconsistently refuses `member_not_resolvable` (`do_not_retry`): `fix.members` names each unresolved StatVar, or the spec_id when the identity as a whole did not resolve, and `fix.remedy` tells the end user the dataset holds indicators the platform cannot confirm.

A mint apis ran as an operation fails on the poll with `operation_failed`, which names the failing refusal in `cause.code`. A code `get_data` maps refuses with the code and retry mode of the synchronous mint. When apis stored the failing refusal, the poll also carries its detail, issues and ceiling fields, and the refusal matches the synchronous one. A poll without them carries the code alone: `over_ceiling` then states no `fix.ceiling_kind` unless the cause is `parquet_size_bound`, and `facet_not_available` lists no alternatives, so its message states the axis and that none was listed. A cause of `upstream_version_changed` refuses `upstream_error` with `retry_after_delay` and the cause as `upstream_code`, because submitting the same mint again reads one consistent source revision. Any unmapped cause refuses `upstream_error` with the cause as `upstream_code`.

### describe

The `describe` tool explains graph identifiers found in search and retrieval digests.

#### Parameters

- `ids`: List of 1 to 8 dcid strings, the bound enforced in the tool's published schema. A leading `dcid:` prefix is accepted and stripped before validation.

#### Return format

Returns a `ToolResult` with the following items.

- Text content with formatted summaries for each resolved node type, including statistical variables, provenance sources, entities, and general graph concepts. Labels return in English, whatever language the query was in. A node whose types are all enumeration types renders a line stating it is a classification value that may differ from the entity the caller meant, and that a place resolves through `search`.
- Structured content containing the dictionary of resolved node models. A classification-value node's entry carries `is_classification_value`.

### get_skill

The `get_skill` tool loads one skill a `search` result named, by id, or with `path` one of its files. It reads apis' Skills API (`GET /api/skills/{skill_id}`) through `data_mcp.skills_client`. With `path` it answers as `get_skill_file` does, through the same `get_skill_file_tool.read_skill_file`, and every call its text names is a `get_skill` call.

#### Parameters

- `skill_id`: String of lower-case letters and digits in hyphen-separated words, at most 64 characters, as named in a search result's skills list.
- `path`: Optional. A file's bundle-relative path exactly as listed, such as `references/core-oda.md`, or `SKILL.md`. Left out, the tool loads the skill.
- `offset`: Where a page of the file at `path` starts, in characters. Defaults to 0. Past 0 without a `path` refuses `request_invalid` at `/path`.
- `digest`: The file digest a previous page's closing line names. Required when `offset` is past 0.

#### Return format

Returns a `ToolResult` with the following items.

- Without `path`, text content with the first page of the skill's `SKILL.md` verbatim, front matter included, then its other files with their sizes and the `get_skill` call that reads one. A `SKILL.md` past 32,000 characters stops at a line break, with a closing line naming the `get_skill` call, with `path` `SKILL.md`, that reads the rest.
- One `resource_link` block per file after `SKILL.md`, at `skill://{skill_id}/{path}`.
- Structured content holding the skill's SEP-2640 entry: `uri` (`skill://{skill_id}/SKILL.md`), `frontmatter`, and `resources` as the full manifest of `uri`, `digest` and `size`.
- With `path`, the `get_skill_file` result below.

### get_skill_file

The `get_skill_file` tool reads one file of a skill verbatim, from `GET /api/skills/{skill_id}/resources/{path}`. `get_skill` with `path` reads the same way.

#### Parameters

- `skill_id`: The skill's id, as for `get_skill`.
- `path`: The file's bundle-relative path exactly as `get_skill` lists it, such as `references/core-oda.md`. `SKILL.md` is accepted.
- `offset`: Where the page starts, in characters of the decoded text. Defaults to 0.
- `digest`: The file digest a previous page's closing line names. Required when `offset` is past 0.

#### Return format

Returns a `ToolResult` with the following items.

- Text content with one page of the file. A page holds at most 32,000 characters from `offset` and ends at the last line break in its second half, or at 32,000 characters when there is none. A page that stops short of the end closes with a line naming the exact next call, with its `offset` and `digest="sha256:…"`. The digest sits in the text because the combined server strips structured content.
- Structured content `{uri, path, size, digest, offset, next_offset}`, with `next_offset` null on the last page.

### Skill refusals

Both skill tools refuse with the same codes.

- `skill_not_found`: no skill has the id. The fix is an id a search result named.
- `skill_resource_not_found`: the skill has no file at `path`. The fix is `get_skill`, which lists the files.
- `upstream_version_changed`: a continuation's `digest` is not the file's current digest, because an apis replica rebuilt its index between pages. The refusal names the call that reads the file again from offset 0.
- `request_invalid`: a malformed id, path or digest, an `offset` past 0 with no `digest`, which refuses before any read and names the call that reads the file from offset 0, or an `offset` at or past the end of the file.
- `capability_unavailable`: the deployment does not mount the Skills API. The client's probe of `/api/skills/healthz` tells this from an unknown id.
- `service_unavailable`: the Skills API has not finished building its index.

### sources

The `sources` tool answers a question about the data itself, which `search` cannot because such a question names no indicator to resolve. With no `ref` it reads the Graph API's source catalogue (`GET /api/graph/sources`); with a `ref` it reads one dataset's documentation (`GET /api/graph/documentation/{dcid}`).

#### Parameters

- `ref`: Optional string, 1 to 512 characters, one or more `/`-separated segments of letters, digits, `_`, `.` or `-`: a dataset's ref as the catalogue names it. Unset for the catalogue.

#### Return format

Returns a `ToolResult` with the following items.

- With no `ref`, text content with the catalogue grouped by publisher: an opening line with the dataset and publisher counts per data commons, then one heading per publisher and one line per dataset carrying its label, its ref in brackets, its years covered, update cadence and licence type when stated, and the first sentence of its description, cut at 240 characters. A publisher whose datasets are served through DFO's own version instead gets one counted line. Either result stays under `RESULT_CAP_CHARS` (32,000): a catalogue over it drops the description sentences and says so, and anything still over is cut at a line break with a count of the lines left out. The structured content carries the dataset count, the publishers' labels and, when the descriptions were dropped, `descriptions_left_out`.
- With a `ref`, text content with that dataset's documentation: its name and ref, publisher and data commons, description, licence with cadence and update dates, coverage (years, granularity, currency), citation, methodology (cut at 6,000 characters with a line saying so), data notes (10, then a count), how DFO transforms it (10 steps, then a count), its latest three releases, its source and download links, and closing lines naming in plain words what the door holds but could not read (from its `document_errors`, with the door's reason) and what is not published for it (from its `gaps`, less the unreadable members). The structured content carries the ref, name, publisher, licence type, years covered, the raw `gaps` and the unreadable members.

A ref the deployment does not hold refuses `node_not_held`; a ref that names a node that is not a source refuses `request_invalid` with the door's own detail; a malformed ref is refused before any read. A deployment that does not mount the Graph API's source doors refuses `capability_unavailable`.

### resolve_definition

The `resolve_definition` tool resolves one identity again through the upstream Search API's `dataset_resubmit` door, so the caller reads Search's own answer for it. A caller passing the Spec's `entity_groups` has them posted beside the identity, so Search names a set a role's places were asked for as again. The harness calls it, on the data-mcp at `DATA_MCP_URL`, for each Search result a Studio turn starts from and for a seed's identity over the years a plan retrieves (`windowed_identity`), and takes every label, sentence and population fact the turn uses from the answer. The combined server outside clients reach leaves it out (`harness.combined_mcp.HARNESS_ONLY_TOOLS`).

#### Parameters

- `spec_id`: String, `spec_` and 32 lowercase hex characters.
- `identity`: The definition's identity, which must hash to `spec_id` and take at most 65,536 bytes as compact JSON.

The request posts one `identity` entry, and Search derives the population from the identity. A resubmit whose member walk reads a family's whole subject set runs up to the apis service's 90 s deadline, so the call waits up to 95 s for a response and 100 s in all (`RESUBMIT_READ_TIMEOUT_S`, `RESUBMIT_CALL_BUDGET_S`).

#### Return format

Returns a `ToolResult` whose structured content is `{"spec_id": ..., "status": ...}`, the envelope's status, and whose `_meta["org.one.dfo/search_envelope"]` holds the upstream envelope, after every definition it carries was stored as `search` stores them. A `definite` part whose spec carries the same spec_id is the identity resolving as it did; the caller judges any other answer. An identity that does not hash to its spec_id, is past the cap, or has no RFC 8785 form refuses `request_invalid` at `/identity` and posts nothing. An upstream refusal or an unusable envelope refuses as `search` does, and a store that cannot take the write refuses `service_unavailable` with `retry_after_delay`. An identity that holds open a role the StatVars Search elects do not declare refuses `open_role_unavailable` with `correct_and_retry`, `fix.pointer` at the identity's `open_roles` and `fix.next_step` `search`. The same identity refuses every time, and a fresh `search` asking for the role elects a source that breaks the measure down by it where one exists.

## Exposed resources

### dfo://artifact/{subject_id}/{content_version}/{artifact_digest}

The artifact resource provides the stable download URL for a generated dataset artifact. It is a URI template, listed by `resources/templates/list` and absent from `resources/list`.

The URL does not expire: a GET on it answers 302 to a URL signed for at most five minutes, so a client follows redirects. A `410` (`artifact_expired`, `snapshot_revoked`) means the artifact is gone.

#### Parameters

- `subject_id`: The specification identifier.
- `content_version`: The content version token.
- `artifact_digest`: The SHA-256 digest of the artifact payload.

#### Return format

Returns a JSON document with the following fields.

- `url`: The absolute download URL, the artifact's root-relative link resolved against `DFO_APIS_URL`.
- `media_type`: MIME type of the artifact.
- `bytes`: Integer byte size of the artifact.
- `rendering`: Dictionary describing artifact format and layout.
- `tier`: Storage tier of the artifact.

### skill://{skill_id}/...

`data_mcp.skills_resources.SkillsProvider` serves every skill file as a resource, in the layout FastMCP's skill client helpers read. It reads apis' Skills API through `data_mcp.skills_client` at read time.

#### URIs

- `skill://{skill_id}/SKILL.md`: the skill's `SKILL.md` verbatim. The front matter's `description` is the resource's description.
- `skill://{skill_id}/_manifest`: JSON `{"skill": id, "files": [{"path", "size", "hash"}]}`, where `hash` is the Skills API's file digest.
- `skill://{skill_id}/{path*}`: one file of the skill, a URI template listed by `resources/templates/list`.

`resources/list` names the `SKILL.md` and `_manifest` resources of each skill search can name, the ones the Skills API list marks `attached` and not `excluded`. A direct read serves any skill by id. Each read carries the media type the Skills API reports for the file.

A failed read raises a resource error written as `<code>: <prose>`, with the codes listed under "Skill refusals". A `resources/list` whose Skills API read fails lists no skills and logs why.

## Skills over MCP extension

`data_mcp.skills_extension.SkillsExtension` implements the Skills over MCP extension (SEP-2640, `io.modelcontextprotocol/skills`) with two methods.

- `skills/list`: returns the same skills `resources/list` names, as SEP-2640 entries, from one read of `GET /api/skills`.
- `skills/get`: takes `{"uri": "skill://{skill_id}/SKILL.md"}` and returns that skill's entry under `skill`. It serves any skill by id.

Both results carry `resultType: "complete"`, `ttlMs` (the Skills API's `Cache-Control: max-age`, in milliseconds), and `cacheScope: "public"`. The extension's settings are empty. A host reads a listed file with `resources/read` against the `skill://` resources, so the digests it checks are the Skills API's own.

A malformed URI or an unknown skill fails with `-32602`. Every other failure, an unmounted Skills API or an index still building among them, fails with `-32603`. The message reads `<code>: <prose>`, as a resource error does, so `skill_not_found: No skill has the id 'x'.` answers an unknown skill.

A mounted server's extensions do not reach the server that mounts it, so the harness registers the extension on the combined server at `/mcp/` (`harness.combined_mcp`). That server wraps each method in its request span, its in-flight cap, and a time limit.

## Frame materialisation

When `get_data` mints a dataset, `data_mcp.frame_materialisation` downloads the parquet artifact, verifies its SHA-256 digest, checks row counts from parquet metadata against `max_frame_rows`, and parses the bytes into a `Frame` object.

The frame and its lineage trace are stored in `common.materialisation` under a `FrameRef(spec_id, content_version)` key. Downstream analysis tools in `analysis-mcp` resolve this frame reference directly from the shared store.

apis keeps one `content_version` for every rendering of a materialisation, so a mint with `options.columns` and one without come back under one pair, and the store is insert-only. `frame_materialisation.frame_ref_for` gives each rendering its own key. A frame with the default columns keeps the manifest's pair, so a frame stored under it before renderings had their own key still resolves. A frame with extra columns is held under `cv_` and the first 32 hex digits of the SHA-256 of the RFC 8785 form of `{"columns": [...], "content_version": ...}`, with the columns sorted. The artifact is still fetched under the manifest's pair.

## Definition store

The definition store records canonical dataset definitions keyed by `spec_id`, and holds the identity alone. Canonical identity bytes are verified using RFC 8785 canonicalisation and SHA-256 hashing. The `spec_id` is the identity's hash, so every write of one key carries the same bytes and the first write wins.

`search` and `resolve_definition` register every definition their envelope carries (`definition_writes.stored_definition`, `put_definitions`). `resolve_definition` first checks the identity a caller hands in (`caller_definition`, which also holds it to the 65,536 bytes apis registers). `windowed_identity` replaces an identity's `requested_window` and returns the spec_id that hashes from the result, which is the spec_id `get_data`'s `scope.window` mints. Every write is verified first, so a write can only add the definition its spec_id names.

The service provides two storage backends.

- `GCSDefinitionStore`: Backs storage with a Google Cloud Storage bucket when `DEFINITION_STORE_BUCKET` is set.
- `InMemoryDefinitionStore`: Keeps definitions in local memory when `DEFINITION_STORE_BUCKET` is unset.

Both backends are wrapped with `CachingDefinitionStore`, which maintains an in-memory LRU read cache bounded by `DEFINITION_STORE_CACHE_MAX_ENTRIES`. It caches every hit, since a stored definition never changes, and never caches a miss.

## Call budgets

`get_data` shares its deadline across the definition read, entity resolution, the mint POST, the operation poll, the frame fetch and rendering under a `CallBudget` (`data_mcp.deadline`), which the call's entry point selects through a context variable.

- `DIRECT_CALL_BUDGET`, the default: a 165 s deadline (`Settings.get_data_deadline_s`) and a 125 s mint cap, which outlasts apis' 120 s synchronous mint timeout, so a synchronous mint always answers inside it.
- `EXTERNAL_CALL_BUDGET`: the harness's combined server (`/mcp/`) sets this for every tool call. Its 42 s deadline ends 3 s before the combined server stops any call at 45 s (`EXTERNAL_CALL_LIMIT_S`, `EXTERNAL_CALL_MARGIN_S`), so `get_data`'s own refusal arrives first. Its 32 s mint cap falls short of apis' 120 s synchronous mint timeout: a mint that runs longer refuses `service_unavailable` and keeps running upstream, and a large mint that apis moves to a background operation refuses `operation_still_processing`, which a repeat call with the same arguments picks up.

`data_mcp.mint_ledger.MintLedger` holds one entry per mint POST sent under a budget whose mint cap falls short of apis' synchronous mint timeout, from the POST until apis answers or the timeout passes, whichever comes first. It refuses a mint while it already holds `external_mint_concurrency` others, and refuses a repeat of a mint it still holds rather than starting the same work twice. `external_mint_concurrency` (default 1) is per process; an outside caller can hold that many mint slots per harness instance.

## Configuration

Settings load from environment variables using `data_mcp.settings.Settings`.

| Environment variable                 | Default                 | Description                                                                   |
| ------------------------------------ | ----------------------- | ----------------------------------------------------------------------------- |
| `DFO_APIS_URL`                       | `http://localhost:8083` | Base URL for upstream APIs                                                    |
| `UPSTREAM_CONNECT_TIMEOUT_S`         | `5.0`                   | Upstream connection timeout in seconds                                        |
| `UPSTREAM_READ_TIMEOUT_S`            | `50.0`                  | Upstream read timeout in seconds                                              |
| `UPSTREAM_RETRY_BUDGET_S`            | `120.0`                 | Wall-clock retry budget for upstream requests                                 |
| `UPSTREAM_MAX_ATTEMPTS`              | `3`                     | Maximum retry attempts for upstream calls                                     |
| `UPSTREAM_BACKOFF_BASE_S`            | `1.0`                   | Base backoff duration in seconds                                              |
| `DEFINITION_STORE_BUCKET`            | `None`                  | GCS bucket name for definition persistence                                    |
| `DEFINITION_STORE_CACHE_MAX_ENTRIES` | `4096`                  | Capacity of the definition cache                                              |
| `MAX_FRAME_ROWS`                     | `250000`                | Maximum row count accepted during frame materialisation                       |
| `ARTIFACT_FETCH_BUDGET_S`            | `60.0`                  | Wall-clock budget for artifact retrieval                                      |
| `GET_DATA_DEADLINE_S`                | `165.0`                 | Wall-clock deadline for a `get_data` call under `DIRECT_CALL_BUDGET`          |
| `OPERATION_POLL_INTERVAL_S`          | `2.0`                   | Fallback wait between operation poll attempts                                 |
| `EXTERNAL_MINT_CONCURRENCY`          | `1`                     | Mint slots one process holds for outside callers under `EXTERNAL_CALL_BUDGET` |
| `PORT`                               | `8080`                  | Port for the streamable HTTP transport                                        |

## Running tests

Run the test suite with pytest from the workspace root or the service directory.

```bash
pytest services/data-mcp/tests
```

# Stability of stored Pydantic AI conversations

Question: does a `Conversation` stored under one `pydantic-ai-slim` release load and resume under a later one, and can a rollback read a newer record?

## Result

- Forward pass line met. All 55 pairs with reader >= writer load and resume, including every record written by 2.55.0 to 2.63.0 read under 2.64.0 (9 records, all `ok`).
- Rollback also loads and resumes in all 45 pairs (reader older than writer). Nothing fails; the cost is one dropped value (below).
- The range has ten releases on PyPI: 2.55.0 to 2.64.0, one each. There are no patch releases and none are yanked.
- The thought signatures, `provider_name`, `provider_details`, usage, tool return and `deferred_tool_requests` are identical in every round trip. The resumed model received every stored signature, in order, plus the answer as a tool return, in all 100 matrix cells.
- Every round-trip difference is a field added or removed. No value changes.

### First version where anything changed

The serialised shape changes in five releases. The "first reader" is the oldest release whose dump of an older record carries the field.

| field                                                                                                                        | first reader |
| ---------------------------------------------------------------------------------------------------------------------------- | ------------ |
| `messages[].history_digest`, `usage.cache_write_1h_tokens`, `usage.output_reasoning_tokens` (also on each message's `usage`) | 2.58.0       |
| `messages[].parts[].retry`                                                                                                   | 2.60.0       |
| `messages[].parts[].citations`                                                                                               | 2.61.0       |
| `messages[].canonical_system`                                                                                                | 2.63.0       |
| `messages[].model_id`                                                                                                        | 2.64.0       |

Forward, an older record reloaded under a newer reader gains these fields with their defaults (`null` or `0`). The reader ignores unknown fields, so a newer record read by an older reader loses them silently (the `unknown-field` control confirms the older readers do not reject extras either).

Rollback loses one stored value: `model_id` (a string such as `function:function:planner_model:` in this run) is dropped by readers older than 2.64.0. The source (`messages.py`, near the `model_id` field) says Pydantic AI routes later requests by it, so a rollback can change routing for a resumed history. This instrument did not test that routing; it only shows the field is dropped. Every other dropped field was `null` in these records.

### Controls

Each control is a copy of the 2.64.0 record, read under all ten releases. All three fire under all ten readers.

- `controls/corrupt-part-kind.json`: `part_kind: thinking` renamed to `thinkng`. Reported as a load failure.
- `controls/unknown-field.json`: an extra key on the first message. Reported as a round-trip diff (`removed`), which proves the diff check sees silent loss.
- `controls/no-pending-request.json`: `deferred_tool_requests` set to `null`. Loads, and the resume check fails.

`matrix.py` exits 1 if a forward pair fails or any control passes cleanly.

## What is in the record

`make_record.py` runs a real agent (`output_type=[str, DeferredToolRequests]`, `instructions` set) on a `FunctionModel`:

1. Request with the user prompt and instructions.
1. Response: `ThinkingPart` and `ToolCallPart` (`search`), each with `provider_name='google-gla'` and `provider_details={'thought_signature': <base64>}`.
1. The `search` tool runs and returns a dict (tool return).
1. Response: `ThinkingPart` (signed), `TextPart`, `ToolCallPart` (`confirm_plan`, signed). `confirm_plan` raises `CallDeferred(metadata=...)`, so the run ends with `DeferredToolRequests` and the conversation holds `deferred_tool_requests`.

Both responses also carry `usage`, `provider_name`, `provider_url`, `provider_response_id`, `finish_reason` and `provider_details` with `finish_reason` and a `timestamp`, as `GoogleModel._process_response` sets them.

## Deviations and limits

- The response parts are built by hand inside the `FunctionModel`, because it cannot produce Gemini's thought-signature fields. The shapes follow `GoogleModel` in v2.64.0 (`models/google.py`, the streamed part handling and `_process_response`).
- `FunctionModel` overwrites `model_name` and `model_id` on the response (`function:planner_model:` and `function:function:planner_model:`). A Gemini response carries a Gemini name there. The `model_id` result above rests on this value.
- A real Gemini response also fills fields these records leave empty: grounding `citations`, `retry`, `history_digest`, `canonical_system`, safety ratings and logprobs in `provider_details`. Their round trip under each version is not exercised, so an added-field diff here shows only that the field appears, not that a populated value survives.
- The resume runs on a `FunctionModel`, not `GoogleModel`. It proves the history reaches the model intact (signatures, tool return); it does not prove Gemini accepts it.
- `DeferredToolResults(calls={...})` is used for the resume instead of `build_results`, to keep one call that exists in every version.
- Each env resolves its own dependencies; every run here used Python 3.13.14 and pydantic 2.13.4. A different pydantic could change the serialised shape independently of Pydantic AI.
- A record written without a version tag cannot be told apart by its content alone. The version-to-version differences above are the only signal.

## Private imports and monkeypatches

None. All imports are public (`pydantic_ai`, `pydantic_ai.conversation`, `pydantic_ai.messages`, `pydantic_ai.models.function`). No harness import, no monkeypatch.

## Run it

```
cd <worktree>/research/pai-prototype/stability
python3 matrix.py            # rewrites records/, controls/, matrix.json, matrix.md; about 30 seconds with a warm uv cache
```

Needs `uv` and network access. One pair by hand:

```
PYDANTIC_AI_NO_BANNER=1 uv run --no-project --with 'pydantic-ai-slim==2.55.0' python make_record.py records/2.55.0.json
PYDANTIC_AI_NO_BANNER=1 uv run --no-project --with 'pydantic-ai-slim==2.64.0' python load_record.py records/2.55.0.json
```

`load_record.py` prints one JSON object per reader: `loaded`, `load_error`, `resumed`, `resume_error`, `round_trip_equal` and the field-level `diff`.

## Files

- `make_record.py`: writes a paused planner conversation under the installed version.
- `load_record.py`: loads records, resumes each pause, diffs the dump against the file.
- `matrix.py`: drives both over every (writer, reader) pair and the controls.
- `matrix.md`, `matrix.json`: the result. `records/` holds the ten stored records, `controls/` the three broken ones.

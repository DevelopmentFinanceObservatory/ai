# Stability matrix

Rows are writer versions, columns are reader versions.

| writer \\ reader | 2.55.0        | 2.56.0        | 2.57.0        | 2.58.0        | 2.59.0        | 2.60.0        | 2.61.0        | 2.62.0        | 2.63.0        | 2.64.0       |
| ---------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------ |
| 2.55.0           | ok            | ok            | ok            | ok, 8 diffs   | ok, 8 diffs   | ok, 9 diffs   | ok, 10 diffs  | ok, 10 diffs  | ok, 12 diffs  | ok, 14 diffs |
| 2.56.0           | (ok)          | ok            | ok            | ok, 8 diffs   | ok, 8 diffs   | ok, 9 diffs   | ok, 10 diffs  | ok, 10 diffs  | ok, 12 diffs  | ok, 14 diffs |
| 2.57.0           | (ok)          | (ok)          | ok            | ok, 8 diffs   | ok, 8 diffs   | ok, 9 diffs   | ok, 10 diffs  | ok, 10 diffs  | ok, 12 diffs  | ok, 14 diffs |
| 2.58.0           | (ok, 2 diffs) | (ok, 2 diffs) | (ok, 2 diffs) | ok            | ok            | ok, 1 diffs   | ok, 2 diffs   | ok, 2 diffs   | ok, 4 diffs   | ok, 6 diffs  |
| 2.59.0           | (ok, 2 diffs) | (ok, 2 diffs) | (ok, 2 diffs) | (ok)          | ok            | ok, 1 diffs   | ok, 2 diffs   | ok, 2 diffs   | ok, 4 diffs   | ok, 6 diffs  |
| 2.60.0           | (ok, 3 diffs) | (ok, 3 diffs) | (ok, 3 diffs) | (ok, 1 diffs) | (ok, 1 diffs) | ok            | ok, 1 diffs   | ok, 1 diffs   | ok, 3 diffs   | ok, 5 diffs  |
| 2.61.0           | (ok, 4 diffs) | (ok, 4 diffs) | (ok, 4 diffs) | (ok, 2 diffs) | (ok, 2 diffs) | (ok, 1 diffs) | ok            | ok            | ok, 2 diffs   | ok, 4 diffs  |
| 2.62.0           | (ok, 4 diffs) | (ok, 4 diffs) | (ok, 4 diffs) | (ok, 2 diffs) | (ok, 2 diffs) | (ok, 1 diffs) | (ok)          | ok            | ok, 2 diffs   | ok, 4 diffs  |
| 2.63.0           | (ok, 6 diffs) | (ok, 6 diffs) | (ok, 6 diffs) | (ok, 4 diffs) | (ok, 4 diffs) | (ok, 3 diffs) | (ok, 2 diffs) | (ok, 2 diffs) | ok            | ok, 2 diffs  |
| 2.64.0           | (ok, 8 diffs) | (ok, 8 diffs) | (ok, 8 diffs) | (ok, 6 diffs) | (ok, 6 diffs) | (ok, 5 diffs) | (ok, 4 diffs) | (ok, 4 diffs) | (ok, 2 diffs) | ok           |

Cells in brackets are rollback pairs (reader older than writer), reported only.

- Forward pairs: 55, failing to load or resume: 0.
- Rollback pairs: 45, failing to load or resume: 0.
- Pairs with a round-trip difference: 80.

## Round-trip differences

- writer 2.55.0:
  - added `messages[].canonical_system`, first under reader 2.63.0
  - added `messages[].history_digest`, first under reader 2.58.0
  - added `messages[].model_id`, first under reader 2.64.0
  - added `messages[].parts[].citations`, first under reader 2.61.0
  - added `messages[].parts[].retry`, first under reader 2.60.0
  - added `messages[].usage.cache_write_1h_tokens`, first under reader 2.58.0
  - added `messages[].usage.output_reasoning_tokens`, first under reader 2.58.0
  - added `usage.cache_write_1h_tokens`, first under reader 2.58.0
  - added `usage.output_reasoning_tokens`, first under reader 2.58.0
- writer 2.56.0:
  - added `messages[].canonical_system`, first under reader 2.63.0
  - added `messages[].history_digest`, first under reader 2.58.0
  - added `messages[].model_id`, first under reader 2.64.0
  - added `messages[].parts[].citations`, first under reader 2.61.0
  - added `messages[].parts[].retry`, first under reader 2.60.0
  - added `messages[].usage.cache_write_1h_tokens`, first under reader 2.58.0
  - added `messages[].usage.output_reasoning_tokens`, first under reader 2.58.0
  - added `usage.cache_write_1h_tokens`, first under reader 2.58.0
  - added `usage.output_reasoning_tokens`, first under reader 2.58.0
- writer 2.57.0:
  - added `messages[].canonical_system`, first under reader 2.63.0
  - added `messages[].history_digest`, first under reader 2.58.0
  - added `messages[].model_id`, first under reader 2.64.0
  - added `messages[].parts[].citations`, first under reader 2.61.0
  - added `messages[].parts[].retry`, first under reader 2.60.0
  - added `messages[].usage.cache_write_1h_tokens`, first under reader 2.58.0
  - added `messages[].usage.output_reasoning_tokens`, first under reader 2.58.0
  - added `usage.cache_write_1h_tokens`, first under reader 2.58.0
  - added `usage.output_reasoning_tokens`, first under reader 2.58.0
- writer 2.58.0:
  - added `messages[].canonical_system`, first under reader 2.63.0
  - added `messages[].model_id`, first under reader 2.64.0
  - added `messages[].parts[].citations`, first under reader 2.61.0
  - added `messages[].parts[].retry`, first under reader 2.60.0
  - removed `messages[].history_digest`, first under reader 2.55.0
- writer 2.59.0:
  - added `messages[].canonical_system`, first under reader 2.63.0
  - added `messages[].model_id`, first under reader 2.64.0
  - added `messages[].parts[].citations`, first under reader 2.61.0
  - added `messages[].parts[].retry`, first under reader 2.60.0
  - removed `messages[].history_digest`, first under reader 2.55.0
- writer 2.60.0:
  - added `messages[].canonical_system`, first under reader 2.63.0
  - added `messages[].model_id`, first under reader 2.64.0
  - added `messages[].parts[].citations`, first under reader 2.61.0
  - removed `messages[].history_digest`, first under reader 2.55.0
  - removed `messages[].parts[].retry`, first under reader 2.55.0
- writer 2.61.0:
  - added `messages[].canonical_system`, first under reader 2.63.0
  - added `messages[].model_id`, first under reader 2.64.0
  - removed `messages[].history_digest`, first under reader 2.55.0
  - removed `messages[].parts[].citations`, first under reader 2.55.0
  - removed `messages[].parts[].retry`, first under reader 2.55.0
- writer 2.62.0:
  - added `messages[].canonical_system`, first under reader 2.63.0
  - added `messages[].model_id`, first under reader 2.64.0
  - removed `messages[].history_digest`, first under reader 2.55.0
  - removed `messages[].parts[].citations`, first under reader 2.55.0
  - removed `messages[].parts[].retry`, first under reader 2.55.0
- writer 2.63.0:
  - added `messages[].model_id`, first under reader 2.64.0
  - removed `messages[].canonical_system`, first under reader 2.55.0
  - removed `messages[].history_digest`, first under reader 2.55.0
  - removed `messages[].parts[].citations`, first under reader 2.55.0
  - removed `messages[].parts[].retry`, first under reader 2.55.0
- writer 2.64.0:
  - removed `messages[].canonical_system`, first under reader 2.55.0
  - removed `messages[].history_digest`, first under reader 2.55.0
  - removed `messages[].model_id`, first under reader 2.55.0, a stored value is dropped
  - removed `messages[].parts[].citations`, first under reader 2.55.0
  - removed `messages[].parts[].retry`, first under reader 2.55.0

## Controls (read under each reader)

- corrupt-part-kind.json: load fails; reported by 10 of 10 readers.
- unknown-field.json: round-trip diff reports the field removed; reported by 10 of 10 readers.
- no-pending-request.json: loads, resume fails; reported by 10 of 10 readers.

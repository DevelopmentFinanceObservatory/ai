-- One row per ADK ledger session copied into the conversation tables, by
-- the session's key. `source_digest` is a digest of the ADK state copied,
-- `source_updated_at` the session's `update_time` when it was last read
-- (ADK's own revision marker), and `imported_digest` a digest of the rows
-- the copy wrote. Both digests are NULL for a session never copied because
-- the tables already held its conversation. `diverged` is true once the
-- session and the conversation's rows have both changed since the last
-- copy, and stays true until a copy of the session replaces the rows.
-- `kept_tables` is true when the operator kept the tables' copy of a
-- refused or diverged session (`import --keep-tables`) at the
-- `source_updated_at` recorded with it; a later copy or conflict resets it.

CREATE TABLE adk_ledger_imports (
    principal text NOT NULL,
    conversation_id text NOT NULL,
    source_digest text,
    source_updated_at timestamp NOT NULL,
    imported_digest text,
    diverged boolean NOT NULL DEFAULT false,
    kept_tables boolean NOT NULL DEFAULT false,
    imported_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (principal, conversation_id)
);

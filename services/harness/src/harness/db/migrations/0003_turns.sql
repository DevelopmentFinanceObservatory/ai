-- The turn's own durable record: where each turn stands, the answers it was
-- given, and its agent transcript.
--
-- A turn's writes are compare-and-swapped on `turn_cursors.version` and
-- fenced on the conversation's lease (`conversations.lease_segment_id`) in
-- the same transaction (`harness.turn.store.TurnStore`). The lease, the Stop
-- request and the frame numbering stay on `conversations`.

-- The cutover gate, first. This release reads conversations from the
-- harness's tables only, so it refuses to start, with SQLSTATE DFA02 and
-- nothing applied, while a ledger session is not cleared. A ledger session
-- is a row of ADK's `sessions` of app 'dfo_analyst_graph.ledger'. It is
-- cleared when `adk_ledger_imports` (0002) holds its row (`principal` =
-- `user_id`, `conversation_id` = `id`) with `source_updated_at` equal to
-- the session's `update_time`, and that row is either imported (`diverged`
-- false, `imported_digest` set, and a `conversations` row of the same key)
-- or kept (`kept_tables`, the operator's choice of the tables' copy over
-- ADK's). Without the import's tables no session is cleared. A table named
-- `sessions` that resolves through `search_path` without the columns of
-- ADK's v0 and v1 `sessions` is another service's, and with none there is
-- nothing to check. The message gives the count, the first ten keys and the
-- commands of the previous release's `python -m harness.adk_import` job
-- that clear them. A live lease on a turn ADK holds does not refuse: the
-- previous release serves while this one migrates, and that job's
-- `close-open-turns` checks it.
DO $$
DECLARE
    ledger_app constant text := 'dfo_analyst_graph.ledger';
    sessions_table regclass := to_regclass('sessions');
    imports regclass := to_regclass('adk_ledger_imports');
    conversations_table regclass := to_regclass('conversations');
    uncleared bigint;
    uncleared_keys text[];
BEGIN
    IF sessions_table IS NULL OR EXISTS (
        SELECT 1 FROM unnest(ARRAY[
            'app_name', 'user_id', 'id', 'state', 'create_time', 'update_time'
        ]) AS c(col)
        WHERE NOT EXISTS (
            SELECT 1 FROM pg_attribute a
            WHERE a.attrelid = sessions_table AND a.attname = c.col
              AND a.attnum > 0 AND NOT a.attisdropped)
    ) THEN
        RETURN;
    END IF;
    EXECUTE format(
        'SELECT count(*), (array_agg(s.user_id || ''/'' || s.id ORDER BY s.user_id, s.id))[1:10] '
        'FROM %s s WHERE s.app_name = %L AND NOT %s',
        sessions_table, ledger_app,
        CASE WHEN imports IS NULL OR conversations_table IS NULL THEN 'false' ELSE format(
            'EXISTS (SELECT 1 FROM %s i WHERE i.principal = s.user_id '
            'AND i.conversation_id = s.id AND i.source_updated_at = s.update_time '
            'AND (i.kept_tables OR (NOT i.diverged AND i.imported_digest IS NOT NULL '
            'AND EXISTS (SELECT 1 FROM %s c WHERE c.principal = s.user_id '
            'AND c.conversation_id = s.id))))',
            imports, conversations_table) END)
    INTO uncleared, uncleared_keys;
    IF uncleared > 0 THEN
        RAISE EXCEPTION USING ERRCODE = 'DFA02', MESSAGE = format(
            '%s ledger session(s) in ADK''s `sessions` are neither imported nor kept as of their last ADK write: %s%s. 0003 refuses to start the release that stops reading ADK. Run the previous release''s `python -m harness.adk_import` job: `import --apply` copies them, `import --apply --prefer-adk PRINCIPAL/CONVERSATION_ID` takes ADK''s copy of a conflict, and `import --apply --keep-tables PRINCIPAL/CONVERSATION_ID` keeps the tables'' copy. Then run its `gate`, and restart once it exits 0.',
            uncleared, array_to_string(uncleared_keys, ', '),
            CASE WHEN uncleared > 10 THEN ', ...' ELSE '' END);
    END IF;
END
$$;

-- The latest Stop on the conversation (`stop_id`) and the turn index it
-- names (`stop_turn`), recorded from `lease_cancel` by the next write of the
-- lease columns, which stamps the row's turn index as stored before that
-- write (`harness.studio_store.StopState`). A Stop ends when the turn it
-- names closes, whatever the status. No write marks it read.
ALTER TABLE conversations ADD COLUMN stop_id text;
ALTER TABLE conversations ADD COLUMN stop_turn integer;

-- Every value column is `json`, which keeps the text as written. A node's
-- input, the state and the transcript hold values a model reads verbatim,
-- such as a hand-back's fields and the orchestrator's stored history, and
-- `jsonb` would reorder their objects' keys.

-- One row per turn. `node` is where the turn stands and `node_input` what
-- that node runs on. `inputs` is fixed when the turn opens. `state` is the
-- turn-scoped state the nodes share, the orchestrator's stored history
-- included; its spec blocks are `turn_specs` rows. `round` counts the
-- turn's pauses. A paused turn's open pause is round `round`, of kind
-- `pause_kind`, under the wire interrupt id `interrupt_id`.
-- `owed_interrupt_id` names the pause of round `round` whose accepted
-- answer the node at the cursor owes its `resume`. The write that
-- accepts the answer and moves the turn from paused to running sets it and
-- the next write clears it, so it is set only on a running turn whose
-- cursor has not moved since its answer.
-- `stop_taken_id` is the Stop the turn's `StopScope` answered, set by the
-- pause write that ended the scope's work. The turn ignores that Stop from
-- then on.
CREATE TABLE turn_cursors (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    conversation bigint NOT NULL REFERENCES conversations (id) ON DELETE CASCADE,
    turn_index integer NOT NULL CHECK (turn_index >= 0),
    node text NOT NULL,
    node_input json,
    status text NOT NULL CHECK (
        status IN ('running', 'paused', 'done', 'failed', 'cancelled', 'abandoned')
    ),
    round integer NOT NULL DEFAULT 0 CHECK (round >= 0),
    pause_kind text,
    interrupt_id text,
    pause_payload json,
    owed_interrupt_id text,
    stop_taken_id text,
    inputs json NOT NULL,
    state json NOT NULL,
    error json,
    version bigint NOT NULL DEFAULT 0,
    -- The lease segment whose write the cursor last took, set by every
    -- cursor write (`harness.turn.store.TurnStore`). A segment cut off
    -- mid-write reads it back to tell its own committed write from another
    -- segment's (`harness.turn.loop`). Nullable: a row no cursor write has
    -- set it on holds NULL.
    written_by text,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (conversation, turn_index),
    CHECK (
        (status = 'paused')
        = (pause_kind IS NOT NULL AND interrupt_id IS NOT NULL AND pause_payload IS NOT NULL)
    ),
    CHECK (pause_kind IS NOT NULL OR (interrupt_id IS NULL AND pause_payload IS NULL)),
    CHECK ((status IN ('failed', 'abandoned')) = (error IS NOT NULL)),
    CHECK (owed_interrupt_id IS NULL OR (status = 'running' AND round >= 1))
);

-- At most one open turn per conversation. A new turn opens only once the
-- previous one is closed, so a late answer to an earlier pause finds nothing
-- open.
CREATE UNIQUE INDEX turn_cursors_one_open ON turn_cursors (conversation)
    WHERE status IN ('running', 'paused');

-- The orchestrator's spec blocks for the turn, one row per spec id. A
-- checkpoint writes only the blocks that changed since the last one and
-- deletes those the turn dropped. `block` is the JSON text as the turn
-- built it and `digest` its SHA-256, which that comparison reads. `seq` is
-- the block's position in the turn's state as last written. A checkpoint
-- that finds the order changed rewrites the positions of the blocks that
-- moved.
CREATE TABLE turn_specs (
    turn bigint NOT NULL REFERENCES turn_cursors (id) ON DELETE CASCADE,
    spec_id text NOT NULL,
    seq integer NOT NULL CHECK (seq >= 0),
    block json NOT NULL,
    digest text NOT NULL,
    PRIMARY KEY (turn, spec_id)
);

-- An accepted answer to one pause, inserted in the transaction that moves
-- the turn from paused to running. The primary key holds one answer per
-- pause round, whatever its interrupt id. `answer` is what the paused
-- node's resume reads; `request` is the answer as the client sent it (its
-- decision and payload), which tells a retry of the same answer apart from
-- another answer once the conversation no longer holds the pause.
CREATE TABLE turn_answers (
    turn bigint NOT NULL REFERENCES turn_cursors (id) ON DELETE CASCADE,
    interrupt_id text NOT NULL,
    round integer NOT NULL CHECK (round >= 1),
    answer json NOT NULL,
    request json,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (turn, round)
);

-- The turn's transcript in order: the loop's entries (the user's message,
-- each pause and its answer) and the planner's run records, appended in the
-- transaction that moves the cursor, so entries written after the last
-- cursor write die with a crash.
CREATE TABLE agent_records (
    turn bigint NOT NULL REFERENCES turn_cursors (id) ON DELETE CASCADE,
    seq integer NOT NULL CHECK (seq >= 0),
    agent text NOT NULL CHECK (agent IN ('loop', 'planner')),
    kind text NOT NULL,
    body json NOT NULL,
    PRIMARY KEY (turn, seq)
);

-- The turn's own durable record: where each turn stands, the answers it was
-- given, and its agent transcript.
--
-- A turn's writes are compare-and-swapped on `turn_cursors.version` and
-- fenced on the conversation's lease (`conversations.lease_segment_id`) in
-- the same transaction (`harness.turn.store.TurnStore`). The lease, the Stop
-- request and the frame numbering stay on `conversations`.

-- The latest Stop on the conversation (`stop_id`) and the turn index it
-- names (`stop_turn`), recorded from `lease_cancel` by the next write of the
-- lease columns, which stamps the row's turn index as stored before that
-- write (`harness.studio_store.StopState`). A Stop ends when the turn it
-- names closes, whatever the status; no write marks it read.
ALTER TABLE conversations ADD COLUMN stop_id text;
ALTER TABLE conversations ADD COLUMN stop_turn integer;

-- Every value column is `json`, which keeps the text as written: a node's
-- input, the state and the transcript hold values a model reads verbatim,
-- such as a hand-back's fields and the orchestrator's stored history, and
-- `jsonb` would reorder their objects' keys.

-- One row per turn. `node` is where the turn stands and `node_input` what
-- that node runs on; `inputs` is fixed when the turn opens; `state` is the
-- turn-scoped state the nodes share, the orchestrator's stored history
-- included; its spec blocks are `turn_specs` rows. `round` counts the turn's pauses; a paused turn's
-- open pause is round `round`, of kind `pause_kind`, under the wire
-- interrupt id `interrupt_id`. `owed_interrupt_id` names the pause of round
-- `round` whose accepted answer the node at the cursor still owes its
-- `resume`: set by the write that accepts the answer and moves the turn from
-- paused to running, cleared by the next write, so it is set only on a
-- running turn whose cursor has not moved since its answer.
-- `stop_taken_id` is the Stop the turn's `StopScope` answered, set by the
-- pause write that ended the scope's work; that Stop no longer stops the
-- turn.
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

-- At most one open turn per conversation: a new turn opens only once the
-- previous one is closed, so a late answer to an earlier pause finds nothing
-- open.
CREATE UNIQUE INDEX turn_cursors_one_open ON turn_cursors (conversation)
    WHERE status IN ('running', 'paused');

-- The orchestrator's spec blocks for the turn, one row per spec id. A
-- checkpoint writes only the blocks that changed since the last one and
-- deletes those the turn dropped; `block` is the JSON text as the turn
-- built it and `digest` its SHA-256, which that comparison reads. `seq` is
-- the block's position in the turn's state as last written: a checkpoint
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
-- pause round, whatever its interrupt id.
CREATE TABLE turn_answers (
    turn bigint NOT NULL REFERENCES turn_cursors (id) ON DELETE CASCADE,
    interrupt_id text NOT NULL,
    round integer NOT NULL CHECK (round >= 1),
    answer json NOT NULL,
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

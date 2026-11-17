-- The harness's own tables: one row per conversation, its per-item records,
-- and the run ledger.
--
-- Each column group of `conversations` has its own writer, so writers that
-- touch different groups never conflict:
--   version, turn_state  the turn's bookkeeping. Written by the lease holder
--                        (fenced on lease_segment_id) or, with no live lease,
--                        by a compare-and-swap on version.
--   composition          the canvas. Written under the row lock by
--                        StudioStore.change_canvas (model-originated writes,
--                        fenced on the segment) and by
--                        StudioStore.mutate_composition (the /composition
--                        route).
--   lease_*              the lease on the conversation's active stream.
--                        Claimed by a turn-state write, renewed and
--                        cancelled by single conditional updates.

CREATE TABLE conversations (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    principal text NOT NULL,
    conversation_id text NOT NULL,
    version bigint NOT NULL DEFAULT 0,
    turn_state jsonb NOT NULL DEFAULT '{}'::jsonb,
    composition jsonb,
    lease_segment_id text,
    lease_owner text,
    -- Epoch seconds on the database's clock, so instances whose clocks
    -- disagree still agree on when a lease lapses.
    lease_expires_at double precision,
    lease_cancel boolean NOT NULL DEFAULT false,
    -- Frame numbering for the open turn, beside the lease. seq_next is the
    -- next frame number the open segment will send, as of the last save;
    -- seq_reserved is the end of the numbers the turn's segments have
    -- reserved. A segment's claim reserves a block of 256 numbers, from 0
    -- for a new turn and from seq_reserved for a resumed one, and a stream
    -- nearing the end of its block reserves the next, so a stream that takes
    -- the turn over after a crash starts past every number the dead one
    -- could have sent.
    seq_next bigint NOT NULL DEFAULT 0,
    seq_reserved bigint NOT NULL DEFAULT 256,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (principal, conversation_id),
    CHECK ((lease_segment_id IS NULL) = (lease_expires_at IS NULL))
);

-- One closed turn's record, by its position in the conversation.
CREATE TABLE conversation_turns (
    conversation bigint NOT NULL REFERENCES conversations (id) ON DELETE CASCADE,
    position integer NOT NULL CHECK (position >= 0),
    record jsonb NOT NULL,
    PRIMARY KEY (conversation, position)
);

-- The digest entry of each component the conversation streamed.
-- `seq` keeps first-insertion order, which the canvas digest's hidden-card
-- list reads as "oldest first".
CREATE TABLE conversation_components (
    conversation bigint NOT NULL REFERENCES conversations (id) ON DELETE CASCADE,
    component_id text NOT NULL,
    seq bigint GENERATED ALWAYS AS IDENTITY,
    entry jsonb NOT NULL,
    PRIMARY KEY (conversation, component_id)
);

-- The bearer write token each component this conversation created was
-- issued. Kept apart from the digest entries, which feed the models.
CREATE TABLE conversation_write_tokens (
    conversation bigint NOT NULL REFERENCES conversations (id) ON DELETE CASCADE,
    component_id text NOT NULL,
    token text NOT NULL,
    PRIMARY KEY (conversation, component_id)
);

CREATE TABLE conversation_approvals (
    conversation bigint NOT NULL REFERENCES conversations (id) ON DELETE CASCADE,
    position integer NOT NULL CHECK (position >= 0),
    record jsonb NOT NULL,
    PRIMARY KEY (conversation, position)
);

-- The run ledger (`harness.run_ledger`). `record` is a `RunRecord` dump,
-- read with `executor.ledger.stored_run`; `submission` is written with the
-- admission and never again; `retrievals` with each seal that changes the
-- inputs.
CREATE TABLE runs (
    conversation bigint NOT NULL REFERENCES conversations (id) ON DELETE CASCADE,
    run_id text NOT NULL,
    record jsonb NOT NULL,
    submission jsonb,
    retrievals jsonb,
    PRIMARY KEY (conversation, run_id)
);

-- The idempotency-key index. The primary key is the admission's uniqueness
-- guarantee: an admission inserts its key first, so a second admission of
-- the same key waits on the first one's commit and then reads its run. The
-- foreign key is checked at commit, after the run row is written.
CREATE TABLE run_keys (
    conversation bigint NOT NULL,
    idempotency_key text NOT NULL,
    run_id text NOT NULL,
    PRIMARY KEY (conversation, idempotency_key),
    FOREIGN KEY (conversation, run_id) REFERENCES runs (conversation, run_id)
        ON DELETE CASCADE DEFERRABLE INITIALLY DEFERRED
);

-- The plan that compiled a run's work, one per revision, written once.
CREATE TABLE run_plans (
    conversation bigint NOT NULL REFERENCES conversations (id) ON DELETE CASCADE,
    revision text NOT NULL,
    plan jsonb NOT NULL,
    PRIMARY KEY (conversation, revision)
);

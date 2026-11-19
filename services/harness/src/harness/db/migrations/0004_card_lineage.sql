-- The lineage of the frame each card was drawn from, written in the card's
-- write (`harness.studio_store.StudioStore.change_canvas`) and read before a
-- later write on that card or frame (`StudioStore.card_lineage`), so an edit
-- on another instance or after a restart finds the card's sources and
-- indicator count. `ON CONFLICT DO NOTHING` keeps the first row a card
-- writes; a revise cannot move a card to another frame, so a row is never
-- updated.
--
-- `spec_id` and `content_version` name the frame the platform's resolved
-- revision names, and `requested_spec_id` and `requested_content_version`
-- the frame the write named: a rolling card's frame before its replay
-- (`harness.rolling_cards`), or a retrieval's `frame_ref` beside the pair it
-- is served under. `members` is the frame's lineage, each retrieval's frame
-- key and whole manifest (`harness.provenance.CardLineage`), and
-- `stat_var_count` the StatVar count stored with it, null when none was.
-- `members` is `json` so a manifest keeps its text as written.
--
-- Rows are best effort, and a card can have none: one written before this
-- table existed, one that did not land on the canvas (a card the turn did
-- not land, a held card, a revise whose block is gone), and one whose
-- lineage JSON cannot hold (a NaN in a manifest). A revise writes the card's
-- row only when its own read found none, so a revise on the instance that
-- holds the card's frame, which reads nothing, writes no row for a card that
-- had none. An edit of a card with no row restores the lineage of another
-- card drawn from the same frame. When no card on that frame has a row and
-- the instance holds no lineage for it, or the read fails or runs past its
-- budget, the edit is written and its card withheld `provenance_unavailable`.
--
-- Creating the table takes SHARE ROW EXCLUSIVE on `conversations` for the
-- foreign key, which waits for open writes of `conversations` rows and holds
-- new ones until the migration transaction commits. Reads and
-- `FOR UPDATE`/`FOR SHARE` row locks do not conflict with it.
CREATE TABLE conversation_card_lineage (
    conversation bigint NOT NULL REFERENCES conversations (id) ON DELETE CASCADE,
    component_id text NOT NULL,
    spec_id text,
    content_version text,
    requested_spec_id text,
    requested_content_version text,
    members json NOT NULL,
    stat_var_count integer,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (conversation, component_id)
);

-- The lineage of each frame an open turn's state names, so a card drawn
-- after a pause, on another instance or after a restart, finds the frames the
-- turn fetched or built before it (`harness.turn.store.TurnStore`, with the
-- frames read by `harness.graph.state.frame_store_keys`). One row per store
-- key an entry names (its `ref` and its `store_ref`), upserted by a cursor
-- write when the writing instance holds a lineage for the key it has not
-- stored for the turn since, so a lineage replaced in process is replaced
-- here too. A write refused for a key's values deletes the key's row in the
-- same write, so no row holds a lineage older than one an instance could not
-- store. When the database itself refuses a lineage write (a lock, a
-- timeout, the disk, a privilege), the table keeps the key's previous row
-- until a later write lands, and an instance that takes over the turn in
-- that window restores it. A segment that lacks a named key reads that
-- key's row once (`TurnStore.restore_frame_lineage`). `members` and
-- `stat_var_count` are encoded as `conversation_card_lineage` encodes them.
--
-- Rows are best effort: a frame no instance held at a write, or whose
-- lineage JSON cannot hold, has none, and a card drawn from it is withheld
-- `provenance_unavailable`. The closing write deletes the turn's rows, best
-- effort (`harness.turn.store.finish_on`). Nothing reads a closed turn's
-- rows, and a row a failed delete leaves stays until its conversation is
-- deleted.
--
-- Creating the table takes SHARE ROW EXCLUSIVE on `turn_cursors` for the
-- foreign key, as the table above does on `conversations`.
CREATE TABLE turn_frame_lineage (
    turn bigint NOT NULL REFERENCES turn_cursors (id) ON DELETE CASCADE,
    spec_id text NOT NULL,
    content_version text NOT NULL,
    members json NOT NULL,
    stat_var_count integer,
    PRIMARY KEY (turn, spec_id, content_version)
);

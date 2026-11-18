-- The lineage of the frame each card was drawn from, written with the card
-- (`harness.studio_store.StudioStore.change_canvas`) and read only before a
-- later write on that card or frame (`StudioStore.card_lineage`), so an
-- edit on another instance or after a restart finds the card's sources and
-- indicator count. A revise cannot move a card to another frame, so a row is
-- written once and never updated.
--
-- `spec_id` and `content_version` name the frame the platform's resolved
-- revision names, and `requested_spec_id` and `requested_content_version`
-- the frame the write named: a rolling card's frame before its replay
-- (`harness.rolling_cards`), or a retrieval's `frame_ref` beside the pair it
-- is served under. `members` is the frame's lineage, each
-- retrieval's frame key and whole manifest (`harness.provenance.CardLineage`),
-- and `stat_var_count` the StatVar count stored with it, null when none was.
-- `members` is `json` so a manifest keeps its text as written.
--
-- Additive: a card written before this table existed has no row. An edit
-- of it restores the lineage of another card drawn from the same frame and
-- commits it as the card's own row, and is refused `provenance_unavailable`
-- before anything is written when no card on that frame has a row and the
-- instance holds no lineage for it. Creating the table takes
-- SHARE ROW EXCLUSIVE on `conversations` for the foreign key, which waits
-- for open writes of `conversations` rows and holds new ones until the
-- migration transaction commits. Reads and `FOR UPDATE`/`FOR SHARE` row
-- locks do not conflict with it.
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

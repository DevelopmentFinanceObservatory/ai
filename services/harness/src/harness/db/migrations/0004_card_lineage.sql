-- The lineage of the frame each card was drawn from, written in the card's
-- write (`harness.studio_store.StudioStore.change_canvas`) and read before a
-- later write on that card or frame (`StudioStore.card_lineage`), so an edit
-- on another instance or after a restart finds the card's sources and
-- indicator count. `ON CONFLICT DO NOTHING` keeps the first row a card
-- writes; a revise cannot move a card to another frame, so a row is never
-- updated.
--
-- `spec_id` and `content_version` name the frame the platform's resolved
-- revision names, the one frame a row restores under, so no card is
-- assembled from another frame's lineage. `members` is the frame's lineage,
-- each retrieval's frame key and whole manifest
-- (`harness.provenance.CardLineage`), and `stat_var_count` the StatVar count
-- stored with it, null when none was.
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
    members json NOT NULL,
    stat_var_count integer,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (conversation, component_id)
);

-- The live-lease half of the cutover gate. The release that added the
-- conversation tables runs `python -m harness.adk_import gate`, which fails
-- while a ledger session's `studio:segment` lease is live: a stream of the
-- revision before that release still runs on ADK and can write the session
-- again, after the release that stops reading ADK owns the conversation, and
-- that write would be lost without an error. `0003_turns.sql` runs the
-- gate's other half, the import predicate. This file sorts before it, so a
-- start that applies 0003 runs this check first, in the same transaction,
-- and refuses with SQLSTATE DFA02 and nothing applied while any such lease
-- is live on the database's clock.
--
-- A ledger session is a row of ADK's `sessions` of app
-- 'dfo_analyst_graph.ledger'. Its lease is live when
-- `state -> 'studio:segment' -> 'expires_at'` is a JSON number, epoch
-- seconds, above `now()`, as the gate reads it. A table named `sessions`
-- that resolves through `search_path` without the columns of ADK's v0 and v1
-- `sessions` is another service's, and with none, as once
-- `0005_rename_adk_tables.sql` has archived ADK's tables, there is nothing
-- to check.
--
-- Lock note for the start that applies 0003 and 0004: `apply_pending`
-- (`harness.db.migrate`) applies every pending file in one transaction, so
-- 0003's `ALTER TABLE conversations` takes ACCESS EXCLUSIVE on
-- `conversations` and holds it until the transaction commits, through
-- 0004. The SHARE ROW EXCLUSIVE lock 0004's comment describes is held only
-- where 0004 applies alone. `deploy/README.md` gives the effect on the
-- serving revision.
DO $$
DECLARE
    ledger_app constant text := 'dfo_analyst_graph.ledger';
    sessions_table regclass := to_regclass('sessions');
    live bigint;
    live_keys text[];
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
        'FROM %s s WHERE s.app_name = %L '
        'AND CASE WHEN jsonb_typeof(s.state::jsonb -> ''studio:segment'' -> ''expires_at'') = ''number'' '
        'THEN (s.state::jsonb -> ''studio:segment'' ->> ''expires_at'')::double precision END '
        '> extract(epoch FROM now())',
        sessions_table, ledger_app)
    INTO live, live_keys;
    IF live > 0 THEN
        RAISE EXCEPTION USING ERRCODE = 'DFA02', MESSAGE = format(
            '%s ledger session(s) in ADK''s `sessions` hold a live `studio:segment` lease: %s%s. A stream of the revision before the release that added the conversation tables still runs on each, and a write it makes after this release starts would be lost. Delete that revision, as the previous release''s deploy/README.md cutover step does, wait for the leases to lapse (45 s after the last renewal), run that release''s `python -m harness.adk_import` `gate` until it exits 0, and restart.',
            live, array_to_string(live_keys, ', '),
            CASE WHEN live > 10 THEN ', ...' ELSE '' END);
    END IF;
END
$$;

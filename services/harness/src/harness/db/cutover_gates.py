"""Checks that `harness.db.migrate.apply_pending` runs in a migration's own
transaction, just before the file, for a check added after the file's bytes
were recorded in an environment.

A recorded file cannot change: `apply_pending` refuses one whose SHA-256
moved. A check shipped as a new migration file changes which images a
database admits: once applied, an image without the file refuses to start
when the file's name sorts at or below its newest. A check held here runs
only when its file is pending, records nothing, and leaves every image's
start as it was.

A check reads the operator's overrides from transaction settings that
`apply_pending` sets before it (`LAPSED_LEDGER_LEASES`).
"""

from __future__ import annotations

# The transaction setting holding `LAPSED_LEDGER_LEASES` (`harness.settings`),
# a JSON array of `principal/conversation_id` keys: ledger sessions whose
# unreadable lease the operator has checked by hand and declared lapsed.
LAPSED_LEDGER_LEASES = "harness.lapsed_ledger_leases"

# The live-lease half of the cutover gate, run before `0003_turns.sql`. The
# release that added the conversation tables runs `python -m
# harness.adk_import gate`, which fails while a ledger session's
# `studio:segment` lease is live: a stream of the revision before that
# release still runs on ADK and can write the session again after the
# release that stops reading ADK owns the conversation, and that write would
# be lost without an error. 0003 runs the gate's other half, the import
# predicate.
#
# A ledger session is a row of ADK's `sessions` of app
# 'dfo_analyst_graph.ledger'. Its lease is live when
# `state -> 'studio:segment' -> 'expires_at'` is a JSON number, epoch
# seconds, above `clock_timestamp()`, the time of the check, which can be
# seconds past the transaction's start after the advisory lock's wait. A
# table named `sessions` that resolves through `search_path` without the
# columns of ADK's v0 and v1 `sessions` is another service's, and with none
# there is nothing to check. The refusal aborts the transaction, so 0003 and
# every other pending file apply nothing. Its SQLSTATE, DFR02, does not begin
# `DFA`, so startup retries it within its budget (`harness.server`): a lease
# lapses 45 s after its stream's last renewal, and a stream that still runs
# past the budget fails the start.
#
# A lease the check cannot read, a `studio:segment` that is not JSON null
# and lacks a numeric `expires_at`, refuses with DFA06, which startup does
# not retry, since no wait changes it: the check cannot tell whether a
# stream holds it. No stream does once every revision of an image before
# the release that added the conversation tables is deleted and that
# release's drain check has passed; the operator then names each such
# session in `LAPSED_LEDGER_LEASES`, and both reads pass over it, so a
# numeric `expires_at` far in the future is cleared the same way.
LIVE_LEDGER_LEASE_GATE = r"""
DO $$
DECLARE
    ledger_app constant text := 'dfo_analyst_graph.ledger';
    sessions_table regclass := to_regclass('sessions');
    live bigint;
    live_keys text[];
    unreadable bigint;
    unreadable_keys text[];
    lapsed constant jsonb := coalesce(
        nullif(current_setting('@LAPSED@', true), ''), '[]')::jsonb;
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
        '> extract(epoch FROM clock_timestamp()) '
        'AND NOT $1 ? (s.user_id || ''/'' || s.id)',
        sessions_table, ledger_app)
    INTO live, live_keys
    USING lapsed;
    EXECUTE format(
        'SELECT count(*), (array_agg(s.user_id || ''/'' || s.id ORDER BY s.user_id, s.id))[1:10] '
        'FROM %s s WHERE s.app_name = %L '
        'AND coalesce(jsonb_typeof(s.state::jsonb -> ''studio:segment''), ''null'') <> ''null'' '
        'AND jsonb_typeof(s.state::jsonb -> ''studio:segment'' -> ''expires_at'') IS DISTINCT FROM ''number'' '
        'AND NOT $1 ? (s.user_id || ''/'' || s.id)',
        sessions_table, ledger_app)
    INTO unreadable, unreadable_keys
    USING lapsed;
    IF unreadable > 0 THEN
        RAISE EXCEPTION USING ERRCODE = 'DFA06', MESSAGE = format(
            '%s ledger session(s) in ADK''s `sessions` hold a `studio:segment` lease without a numeric `expires_at`, so no check can tell whether a stream of the revision before the release that added the conversation tables still runs on it: %s%s. No stream holds it once every revision of the images before the release that added the conversation tables is deleted and that release''s deploy/README.md drain check has passed; then add its key to LAPSED_LEDGER_LEASES (comma-separated `principal/conversation_id`) on this service and restart.',
            unreadable, array_to_string(unreadable_keys, ', '),
            CASE WHEN unreadable > 10 THEN ', ...' ELSE '' END);
    END IF;
    IF live > 0 THEN
        RAISE EXCEPTION USING ERRCODE = 'DFR02', MESSAGE = format(
            '%s ledger session(s) in ADK''s `sessions` hold a live `studio:segment` lease: %s%s. A stream of the revision before the release that added the conversation tables still runs on each, and a write it makes after this release starts would be lost. Startup retries this check within its budget. If it still fails, delete that revision, as the deploy/README.md cutover step of the release that added the conversation tables does, run that release''s `python -m harness.adk_import` `import --apply` and then `gate` until `gate` exits 0, and restart.',
            live, array_to_string(live_keys, ', '),
            CASE WHEN live > 10 THEN ', ...' ELSE '' END);
    END IF;
END
$$;
""".replace("@LAPSED@", LAPSED_LEDGER_LEASES)

# The check run before `0005_rename_adk_tables.sql`. The release that added
# the conversation tables runs turns on ADK turn sessions, and its
# `python -m harness.adk_import close-open-turns` closes each turn it left
# open, with the source blocks the turn's ADK session holds; it cannot run
# once 0005 renames ADK's tables. So the check refuses, with DFA07, while a
# conversation holds the turns that job closes: an open turn (`turn_index`
# above 0, `turn_recorded` not true) whose ADK turn session
# `{conversation_id}#t{turn_index}` (app 'dfo_analyst_graph') exists and
# that `turn_cursors` holds no row for. That covers a start that applies
# 0003 and 0005 together and one after a stage whose `close-open-turns`
# never exited 0. A database with no ADK `sessions` table passes, and so
# does one whose `sessions` or `conversations` the current user cannot
# read, which 0005 then refuses with DFA03, since it renames only tables
# the current user owns.
# `python -m harness.db.adk_archive archive` runs it too.
OPEN_TURNS_ON_ADK_GATE = r"""
DO $$
DECLARE
    sessions_table regclass := to_regclass('sessions');
    ids bigint[];
    turns numeric[];
    keys text[];
    open_turns bigint;
    open_keys text[];
BEGIN
    IF sessions_table IS NULL OR to_regclass('conversations') IS NULL
        OR NOT has_table_privilege(sessions_table, 'SELECT')
        OR NOT has_table_privilege('conversations', 'SELECT') OR EXISTS (
        SELECT 1 FROM unnest(ARRAY['app_name', 'user_id', 'id']) AS c(col)
        WHERE NOT EXISTS (
            SELECT 1 FROM pg_attribute a
            WHERE a.attrelid = sessions_table AND a.attname = c.col
              AND a.attnum > 0 AND NOT a.attisdropped)
    ) THEN
        RETURN;
    END IF;
    EXECUTE format(
        'WITH open AS ('
        '  SELECT c.id, c.principal, c.conversation_id, '
        '    CASE WHEN jsonb_typeof(c.turn_state -> ''turn_index'') = ''number'' '
        '    THEN (c.turn_state ->> ''turn_index'')::numeric END AS turn_index '
        '  FROM conversations c '
        '  WHERE coalesce(c.turn_state -> ''turn_recorded'', ''false''::jsonb) <> ''true''::jsonb) '
        'SELECT array_agg(o.id ORDER BY o.principal, o.conversation_id), '
        '  array_agg(o.turn_index ORDER BY o.principal, o.conversation_id), '
        '  array_agg(o.principal || ''/'' || o.conversation_id ORDER BY o.principal, o.conversation_id) '
        'FROM open o WHERE o.turn_index > 0 '
        'AND EXISTS (SELECT 1 FROM %s s WHERE s.app_name = %L AND s.user_id = o.principal '
        '  AND s.id = o.conversation_id || ''#t'' || o.turn_index::text)',
        sessions_table, 'dfo_analyst_graph')
    INTO ids, turns, keys;
    IF ids IS NULL THEN
        RETURN;
    END IF;
    IF to_regclass('turn_cursors') IS NULL THEN
        open_turns := cardinality(ids);
        open_keys := keys[1:10];
    ELSE
        SELECT count(*), (array_agg(u.key ORDER BY u.key))[1:10]
        INTO open_turns, open_keys
        FROM unnest(ids, turns, keys) AS u(id, turn_index, key)
        WHERE NOT EXISTS (
            SELECT 1 FROM turn_cursors t
            WHERE t.conversation = u.id AND t.turn_index = u.turn_index);
    END IF;
    IF open_turns > 0 THEN
        RAISE EXCEPTION USING ERRCODE = 'DFA07', MESSAGE = format(
            '%s conversation(s) hold a turn left open on ADK that `python -m harness.adk_import close-open-turns` closes, and archiving ADK''s tables would lose its source blocks: %s%s. Run that job''s guarded `close-open-turns --apply`, as the deploy/README.md of the release that added the conversation tables gives it, while the release that applied 0003_turns.sql holds all traffic, until it exits 0, then restart.',
            open_turns, array_to_string(open_keys, ', '),
            CASE WHEN open_turns > 10 THEN ', ...' ELSE '' END);
    END IF;
END
$$;
"""

# Each migration file name and the check run in its transaction just before
# it, while it is pending.
BEFORE: dict[str, str] = {
    "0003_turns.sql": LIVE_LEDGER_LEASE_GATE,
    "0005_rename_adk_tables.sql": OPEN_TURNS_ON_ADK_GATE,
}

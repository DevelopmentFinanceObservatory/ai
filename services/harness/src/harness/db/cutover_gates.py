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
# release's drain check has passed. The operator then names each such
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

# Each migration file name and the check run in its transaction just before
# it, while it is pending.
BEFORE: dict[str, str] = {"0003_turns.sql": LIVE_LEDGER_LEASE_GATE}

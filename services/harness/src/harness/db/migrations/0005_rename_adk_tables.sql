-- Renames ADK's session tables (`sessions`, `events`, `app_states`,
-- `user_states`, `adk_internal_metadata`) and the import job's ledger
-- `adk_ledger_imports` (0002) to `adk_archive_<name>`, in the schema each
-- one is in, with each table's indexes and owned sequences renamed likewise
-- (`adk_archive_<name>`). Nothing is dropped and no row changes: a renamed
-- table keeps its rows, keys and foreign keys. A rename needs ownership of
-- the table and CREATE on its schema, which the runtime user holds. Dropping the renamed tables and
-- renaming them back are manual steps (`deploy/README.md`).
--
-- Table names resolve through `search_path`, as ADK and the import job of
-- the image that holds it resolve them, so a split layout renames every
-- table found. A table is ADK's when it has the columns of ADK's v0 and v1
-- tables in `google/adk/sessions/schemas` (`events` without the v1-only
-- `event_data`). The import ledger is this schema's own table and always
-- renames. Four checks run first, in the order DFA01, DFA03, DFA02, DFA04 (a
-- table the user does not own cannot be read for DFA02), and a failed one
-- raises with its own SQLSTATE and renames nothing. Each is a guard against a misdeploy, and
-- renaming the tables by hand with the statements in `deploy/README.md` is
-- the override:
--   DFA01  A table of an ADK name lacks ADK's columns while another carries
--          them, so the layout is mixed and no rename would be right. With
--          no ADK table at all, tables of those names are another service's
--          and stay.
--   DFA02  ADK's `sessions` holds rows of the ledger app
--          'dfo_analyst_graph.ledger' and `adk_ledger_imports` holds no row,
--          so the import never ran and the new image would list no
--          conversation.
--   DFA03  A table to rename is not owned by the current user, or the user
--          lacks CREATE on its schema.
--   DFA04  A relation or type named `adk_archive_<name>` already exists in
--          the schema, for a table, index or sequence to rename, or a
--          relation named `adk_archive_<table>` resolves through
--          `search_path` in any schema, for a table to rename. So after the
--          rename each `adk_archive_<table>` resolves to the table it
--          archived, which the statements in `deploy/README.md` rely on.

DO $$
DECLARE
    specs constant jsonb := jsonb_build_object(
        'sessions', jsonb_build_array(
            'app_name', 'user_id', 'id', 'state', 'create_time', 'update_time'),
        'events', jsonb_build_array(
            'id', 'app_name', 'user_id', 'session_id', 'invocation_id', 'timestamp'),
        'app_states', jsonb_build_array('app_name', 'state', 'update_time'),
        'user_states', jsonb_build_array('app_name', 'user_id', 'state', 'update_time'),
        'adk_internal_metadata', jsonb_build_array('key', 'value')
    );
    adk_tables constant text[] := ARRAY[
        'sessions', 'events', 'app_states', 'user_states', 'adk_internal_metadata'
    ];
    ledger_app constant text := 'dfo_analyst_graph.ledger';
    adk_present text[] := ARRAY[]::text[];
    foreign_present text[] := ARRAY[]::text[];
    renaming text[] := ARRAY[]::text[];
    renaming_oids oid[] := ARRAY[]::oid[];
    table_name text;
    source regclass;
    object_oid oid;
    object_name text;
    object_schema text;
    object_kind text;
    owner_name text;
    has_ledger_sessions boolean;
    has_imports boolean;
    not_owner boolean;
    no_create boolean;
BEGIN
    FOREACH table_name IN ARRAY adk_tables LOOP
        source := to_regclass(table_name);
        CONTINUE WHEN source IS NULL;
        IF EXISTS (
            SELECT 1 FROM jsonb_array_elements_text(specs -> table_name) AS c(col)
            WHERE NOT EXISTS (
                SELECT 1 FROM pg_attribute a
                WHERE a.attrelid = source AND a.attname = c.col
                  AND a.attnum > 0 AND NOT a.attisdropped)
        ) THEN
            foreign_present := foreign_present || table_name;
        ELSE
            adk_present := adk_present || table_name;
        END IF;
    END LOOP;

    IF cardinality(adk_present) > 0 AND cardinality(foreign_present) > 0 THEN
        RAISE EXCEPTION USING ERRCODE = 'DFA01', MESSAGE = format(
            'ADK''s tables (%s) sit beside %s, which lack the columns of ADK''s tables of those names. 0005 refuses to rename a mixed layout. Rename the foreign tables, or ADK''s tables as deploy/README.md lists, by hand, then restart.',
            array_to_string(adk_present, ', '), array_to_string(foreign_present, ', '));
    END IF;

    renaming := array_append(adk_present, 'adk_ledger_imports');
    FOREACH table_name IN ARRAY renaming LOOP
        source := to_regclass(table_name);
        CONTINUE WHEN source IS NULL;
        renaming_oids := renaming_oids || source::oid
            || ARRAY(SELECT indexrelid FROM pg_index WHERE indrelid = source ORDER BY indexrelid)
            || ARRAY(
                SELECT d.objid FROM pg_depend d
                JOIN pg_class s ON s.oid = d.objid AND s.relkind = 'S'
                WHERE d.classid = 'pg_class'::regclass AND d.refobjid = source::oid
                  AND d.deptype IN ('a', 'i')
                ORDER BY d.objid);
    END LOOP;

    FOREACH object_oid IN ARRAY renaming_oids LOOP
        SELECT c.relname, n.nspname, pg_get_userbyid(c.relowner),
               NOT pg_has_role(current_user, c.relowner, 'USAGE'),
               NOT has_schema_privilege(current_user, c.relnamespace, 'CREATE')
        INTO object_name, object_schema, owner_name, not_owner, no_create
        FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE c.oid = object_oid;
        IF not_owner THEN
            RAISE EXCEPTION USING ERRCODE = 'DFA03', MESSAGE = format(
                '%I.%I is owned by %I, not by the current user %I; 0005 cannot rename it. Run the migration as its owner, or change the owner, then restart.',
                object_schema, object_name, owner_name, current_user);
        END IF;
        IF no_create THEN
            RAISE EXCEPTION USING ERRCODE = 'DFA03', MESSAGE = format(
                'the current user %I lacks CREATE on schema %I, which renaming %I requires; 0005 cannot rename it. Grant CREATE on the schema, or run the migration as a user that has it, then restart.',
                current_user, object_schema, object_name);
        END IF;
    END LOOP;

    IF 'sessions' = ANY (adk_present) THEN
        EXECUTE format('SELECT EXISTS (SELECT 1 FROM %s WHERE app_name = %L)',
                       to_regclass('sessions'), ledger_app) INTO has_ledger_sessions;
        IF to_regclass('adk_ledger_imports') IS NULL THEN
            has_imports := false;
        ELSE
            EXECUTE 'SELECT EXISTS (SELECT 1 FROM adk_ledger_imports)' INTO has_imports;
        END IF;
        IF has_ledger_sessions AND NOT has_imports THEN
            RAISE EXCEPTION USING ERRCODE = 'DFA02', MESSAGE =
                'ADK''s `sessions` holds ledger sessions and `adk_ledger_imports` holds no row, so the import never ran. 0005 refuses to rename the tables out from under the conversations. Run the import job of the image that holds it, or rename ADK''s tables by hand as deploy/README.md lists, then restart.';
        END IF;
    END IF;

    FOREACH table_name IN ARRAY renaming LOOP
        CONTINUE WHEN to_regclass(table_name) IS NULL;
        source := to_regclass('adk_archive_' || table_name);
        IF source IS NOT NULL THEN
            SELECT n.nspname INTO object_schema
            FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE c.oid = source;
            RAISE EXCEPTION USING ERRCODE = 'DFA04', MESSAGE = format(
                '%I.%I exists on search_path, so `adk_archive_%s` would not resolve to the renamed `%s`; 0005 refuses to rename it. Rename or drop it by hand, then restart.',
                object_schema, 'adk_archive_' || table_name, table_name, table_name);
        END IF;
    END LOOP;

    FOREACH object_oid IN ARRAY renaming_oids LOOP
        SELECT c.relname, n.nspname INTO object_name, object_schema
        FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE c.oid = object_oid;
        IF to_regclass(format('%I.%I', object_schema, 'adk_archive_' || object_name)) IS NOT NULL
           OR to_regtype(format('%I.%I', object_schema, 'adk_archive_' || object_name)) IS NOT NULL
        THEN
            RAISE EXCEPTION USING ERRCODE = 'DFA04', MESSAGE = format(
                '%I.%I exists; 0005 refuses to rename `%s` onto it. Rename or drop it by hand, then restart.',
                object_schema, 'adk_archive_' || object_name, object_name);
        END IF;
    END LOOP;

    FOREACH object_oid IN ARRAY renaming_oids LOOP
        SELECT c.relname, n.nspname,
               CASE c.relkind WHEN 'i' THEN 'INDEX' WHEN 'S' THEN 'SEQUENCE' ELSE 'TABLE' END
        INTO object_name, object_schema, object_kind
        FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE c.oid = object_oid;
        EXECUTE format('ALTER %s %I.%I RENAME TO %I',
                       object_kind, object_schema, object_name, 'adk_archive_' || object_name);
    END LOOP;
END
$$;

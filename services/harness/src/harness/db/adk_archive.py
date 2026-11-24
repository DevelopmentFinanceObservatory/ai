"""Archives, restores and drops ADK's session tables.

`python -m harness.db.adk_archive VERB [--apply] [--allow-missing]
[--backup BACKUP_ID]` runs one verb in one transaction and prints each
statement it ran. Without `--apply` it rolls the transaction back, so a dry
run checks everything the verb would do, the database's own refusals
included, and changes nothing. With `--apply` it commits.

- `archive` runs `0005_rename_adk_tables.sql`, the migration that renames
  ADK's five tables and `adk_ledger_imports` to `adk_archive_<name>`, so its
  checks and its rename are the migration's own, and records the file in
  `harness_schema_migrations` as applied when it is not recorded yet. The
  file is one `DO` block and renames nothing a second time, so a rerun
  finds nothing to do.
- `restore` renames every `adk_archive_<name>` table back, with its indexes
  and owned sequences, for an image that reads ADK's tables. It leaves 0005
  recorded in `harness_schema_migrations`, so an image that holds 0005
  refuses to start beside the restored tables
  (`harness.server._refuse_beside_adk_tables`); deploy/README.md's `DFA05`
  entry gives the step for each case. It prints which of ADK's five tables
  it restores and refuses when anything is archived but not all five and
  the import ledger, `adk_archive_adk_ledger_imports`, unless
  `--allow-missing` is given: without the ledger, ADK's restored history
  has no import records, and the import job cannot clear it. With
  nothing archived it finds nothing to do, so a rerun after a restore
  changes nothing. It refuses before renaming anything when an
  `adk_archive_<name>` table lacks the columns of the table it archived,
  or when a name it would restore exists in the object's schema, or, for a
  table, resolves on `search_path`, where an ADK-reading image that started
  after the archive created fresh tables. After renaming it checks, in the
  same transaction, that each restored table's name resolves on
  `search_path` to that table, and rolls back otherwise.
- `drop-adk` drops each of ADK's five tables that resolves on `search_path`
  and has ADK's columns, which discards ADK's copy of the conversations. It
  prints the number of conversations, the ledger sessions of `sessions`
  (app `dfo_analyst_graph.ledger`), before the `DROP`, in a dry run too. The
  count is a snapshot. A
  table of an ADK name without ADK's columns is another service's and stays.
- `drop-archive` drops every `adk_archive_<name>` table. It refuses, and
  drops nothing, when a table of such a name lacks the columns of the table
  it would archive.

Both drops name every table in one `DROP TABLE` without `CASCADE`, so a
foreign key among the dropped tables resolves and PostgreSQL refuses the
statement when anything else, such as a view, depends on one of them. They
cannot be undone, so with `--apply` each refuses, before it connects,
unless `--backup BACKUP_ID` gives the numeric id of the on-demand Cloud
SQL backup taken for it, which it prints with the statements. The job cannot read Cloud SQL's
backup list, so the id is the operator's record, checked by the runbook
step that takes the backup.

Each verb's transaction takes the advisory lock and the statement and lock
timeouts of `harness.db.migrate`, so it never runs beside a starting
instance's migrations.

Tables resolve through `search_path`, as the migration resolves them. The
module connects as the harness does (`harness.settings.Settings`,
`harness.db.engine.build_database`): `DATABASE_URL`, or the
`SESSION_STORE_*` settings over the Cloud SQL Connector with IAM auth.

Exit status: 0 when the verb ran or found nothing to do, 1 when it or the
database refused, and 2 on a usage error, a drop with `--apply` and no
`--backup`, or with no or an invalid database configuration.
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import re
import sys
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any

import asyncpg
import pydantic
from sqlalchemy.ext.asyncio import AsyncEngine

from harness.db.engine import build_database
from harness.db.migrate import MIGRATIONS_DIR, bound_and_lock, record_applied, sha256
from harness.settings import Settings

ARCHIVE_MIGRATION = MIGRATIONS_DIR / "0005_rename_adk_tables.sql"
PREFIX = "adk_archive_"
ADK_TABLES = ("sessions", "events", "app_states", "user_states", "adk_internal_metadata")
ARCHIVED = (*ADK_TABLES, "adk_ledger_imports")

# The columns that make a table of an ADK name ADK's, as
# `0005_rename_adk_tables.sql` checks them: those of ADK's v0 and v1 tables
# in `google/adk/sessions/schemas`.
_ADK_COLUMNS = {
    "sessions": ("app_name", "user_id", "id", "state", "create_time", "update_time"),
    "events": ("id", "app_name", "user_id", "session_id", "invocation_id", "timestamp"),
    "app_states": ("app_name", "state", "update_time"),
    "user_states": ("app_name", "user_id", "state", "update_time"),
    "adk_internal_metadata": ("key", "value"),
}
# The columns of `adk_ledger_imports` as `0002_adk_ledger_imports.sql`
# creates it.
_LEDGER_COLUMNS = (
    "principal",
    "conversation_id",
    "source_digest",
    "source_updated_at",
    "imported_digest",
    "diverged",
    "kept_tables",
    "imported_at",
)

# How many of the columns `$2` the relation `$1` resolves to on
# `search_path` lacks.
_MISSING_COLUMNS = (
    "SELECT count(*) FROM unnest($2::text[]) AS c(col) WHERE NOT EXISTS ("
    "SELECT 1 FROM pg_attribute a WHERE a.attrelid = to_regclass($1) "
    "AND a.attname = c.col AND a.attnum > 0 AND NOT a.attisdropped)"
)

# Each table that resolves on `search_path` from `$1`, then each one's indexes
# and owned sequences.
_OBJECTS = """
WITH tables AS (
    SELECT c.oid, t.position
    FROM unnest($1::text[]) WITH ORDINALITY AS t(name, position)
    JOIN pg_class c ON c.oid = to_regclass(t.name)
)
SELECT oid FROM (
    SELECT oid, position, 0 AS kind, oid AS member FROM tables
    UNION ALL
    SELECT i.indexrelid, t.position, 1, i.indexrelid
    FROM pg_index i JOIN tables t ON i.indrelid = t.oid
    UNION ALL
    SELECT d.objid, t.position, 2, d.objid
    FROM pg_depend d
    JOIN tables t ON d.refobjid = t.oid
    JOIN pg_class s ON s.oid = d.objid AND s.relkind = 'S'
    WHERE d.classid = 'pg_class'::regclass AND d.deptype IN ('a', 'i')
) objects
ORDER BY position, kind, member
"""

# ADK's app name for a conversation's ledger session; each turn also has a
# session of its own, under another app name.
_LEDGER_APP = "dfo_analyst_graph.ledger"

# The schema-qualified, quoted name `$1` resolves to on `search_path`.
_RESOLVED = """
SELECT format('%I.%I', n.nspname, c.relname)
FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE c.oid = to_regclass($1)
"""

_DESCRIBE = """
SELECT c.oid, n.nspname, c.relname,
       CASE c.relkind WHEN 'i' THEN 'INDEX' WHEN 'S' THEN 'SEQUENCE' ELSE 'TABLE' END AS kind,
       format('%I.%I', n.nspname, c.relname) AS qualified
FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE c.oid = ANY ($1::oid[])
"""


def _columns(name: str) -> tuple[str, ...]:
    """The columns the table `name` (one of `ARCHIVED`) has."""
    return _LEDGER_COLUMNS if name == "adk_ledger_imports" else _ADK_COLUMNS[name]


class ArchiveRefusedError(RuntimeError):
    """The verb found a state it will not act on; nothing was changed."""


@dataclass(frozen=True)
class _Relation:
    oid: int
    schema: str
    name: str
    kind: str
    qualified: str


async def _relations(conn: Any, names: Sequence[str]) -> list[_Relation]:
    """The tables `names` resolve to on `search_path`, each followed by its
    indexes and owned sequences, in the order of `names`."""
    oids = [row["oid"] for row in await conn.fetch(_OBJECTS, list(names))]
    described = {row["oid"]: row for row in await conn.fetch(_DESCRIBE, oids)}
    return [
        _Relation(
            oid=oid,
            schema=described[oid]["nspname"],
            name=described[oid]["relname"],
            kind=described[oid]["kind"],
            qualified=described[oid]["qualified"],
        )
        for oid in oids
    ]


async def _adk_tables(conn: Any) -> list[str]:
    """The schema-qualified, quoted name of each of ADK's five tables that
    resolves on `search_path` and has ADK's columns, as 0005 tells ADK's
    tables from another service's."""
    tables = []
    for name in ADK_TABLES:
        qualified = await conn.fetchval(_RESOLVED, name)
        if qualified is not None and not await conn.fetchval(
            _MISSING_COLUMNS, name, list(_ADK_COLUMNS[name])
        ):
            tables.append(qualified)
    return tables


async def live_adk_tables(engine: AsyncEngine) -> list[str]:
    """The schema-qualified name of each of ADK's tables that resolves on
    `search_path` under its own name with ADK's columns, the tables an image
    that reads ADK's tables serves from."""
    async with engine.connect() as sa_conn:
        raw = await sa_conn.get_raw_connection()
        conn = raw.driver_connection
        assert conn is not None
        return await _adk_tables(conn)


async def archive_state(engine: AsyncEngine) -> tuple[list[str], list[str]]:
    """The `adk_archive_<name>` tables that resolve on `search_path`, as
    `(archived, foreign)`, each schema-qualified. `archived` holds those with
    the columns of the table they archived, the archive 0005 or `archive`
    left. `foreign` holds those without, tables of another service that
    `restore` and `drop-archive` refuse."""
    async with engine.connect() as sa_conn:
        raw = await sa_conn.get_raw_connection()
        conn = raw.driver_connection
        assert conn is not None
        archived: list[str] = []
        foreign: list[str] = []
        for name in ARCHIVED:
            qualified = await conn.fetchval(_RESOLVED, PREFIX + name)
            if qualified is None:
                continue
            missing = await conn.fetchval(_MISSING_COLUMNS, PREFIX + name, list(_columns(name)))
            (foreign if missing else archived).append(qualified)
        return archived, foreign


async def ledger_resolves(engine: AsyncEngine) -> bool:
    """Whether `adk_ledger_imports` resolves on `search_path` under its own
    name with the ledger's columns, as the import job needs it."""
    async with engine.connect() as sa_conn:
        raw = await sa_conn.get_raw_connection()
        conn = raw.driver_connection
        assert conn is not None
        return await conn.fetchval(_RESOLVED, "adk_ledger_imports") is not None and not (
            await conn.fetchval(_MISSING_COLUMNS, "adk_ledger_imports", list(_LEDGER_COLUMNS))
        )


async def _quoted(conn: Any, name: str) -> str:
    return await conn.fetchval("SELECT quote_ident($1)", name)


async def _archive(conn: Any) -> list[str]:
    if await conn.fetchval("SELECT to_regclass('harness_schema_migrations')") is None:
        raise ArchiveRefusedError(
            "archive renames nothing, because `harness_schema_migrations` does not resolve on "
            "search_path, so this is not a database the harness has migrated. Point the job at "
            "the harness's database."
        )
    before = await _relations(conn, ARCHIVED)
    await conn.execute(ARCHIVE_MIGRATION.read_text())
    after = {
        row["oid"]: row["relname"] for row in await conn.fetch(_DESCRIBE, [r.oid for r in before])
    }
    statements = [
        f"ALTER {relation.kind} {relation.qualified} RENAME TO {await _quoted(conn, after[relation.oid])}"
        for relation in before
        if after[relation.oid] != relation.name
    ]
    recorded = await conn.fetchval(
        "SELECT EXISTS (SELECT 1 FROM harness_schema_migrations WHERE version = $1)",
        ARCHIVE_MIGRATION.name,
    )
    if not recorded:
        await record_applied(conn, ARCHIVE_MIGRATION)
        statements.append(
            "INSERT INTO harness_schema_migrations (version, content_sha256) "
            f"VALUES ('{ARCHIVE_MIGRATION.name}', '{sha256(ARCHIVE_MIGRATION)}')"
        )
    return statements


async def _restore(conn: Any, *, allow_missing: bool = False) -> list[str]:
    archived = [
        relation
        for relation in await _relations(conn, [PREFIX + name for name in ARCHIVED])
        if relation.name.startswith(PREFIX)
    ]
    tables = [relation for relation in archived if relation.kind == "TABLE"]
    found = {table.name.removeprefix(PREFIX) for table in tables}
    restoring = [name for name in ADK_TABLES if name in found]
    lacking = [name for name in ARCHIVED if name not in found]
    if not archived:
        return []
    print(
        f"adk_archive restore: ADK's tables in the archive: {', '.join(restoring) or 'none'}; "
        f"missing from the archive: {', '.join(lacking) or 'none'}"
    )
    for table in tables:
        name = table.name.removeprefix(PREFIX)
        if await conn.fetchval(_MISSING_COLUMNS, table.name, list(_columns(name))):
            raise ArchiveRefusedError(
                f"restore renames nothing, because `{table.name}` resolves to "
                f"{table.qualified}, which lacks the columns of `{name}`, so it is not the "
                "archive. Rename it or drop it by hand, then run restore again."
            )
    if lacking and not allow_missing:
        raise ArchiveRefusedError(
            f"restore renames nothing, because the archive lacks {', '.join(lacking)}."
            + (
                " An image that reads ADK's tables would create a missing table of ADK's empty."
                if set(lacking) & set(ADK_TABLES)
                else ""
            )
            + (
                " Without adk_ledger_imports ADK's history has no import records, so the "
                "import job cannot clear it."
                if "adk_ledger_imports" in lacking
                else ""
            )
            + " Pass --allow-missing to restore the rest anyway."
        )
    taken = await _restore_conflicts(conn, archived)
    if taken:
        raise ArchiveRefusedError(
            "restore renames nothing, because names it would restore are taken:\n"
            + "\n".join(f"  {line}" for line in taken)
            + "\nAn image that reads ADK's tables created them after the archive. Drop them "
            "with `drop-adk --apply --backup BACKUP_ID` after a backup, which discards what that "
            "image wrote to them, or rename "
            "them by hand, then run restore again."
        )
    statements = []
    for relation in archived:
        statement = (
            f"ALTER {relation.kind} {relation.qualified} "
            f"RENAME TO {await _quoted(conn, relation.name.removeprefix(PREFIX))}"
        )
        await conn.execute(statement)
        statements.append(statement)
    for table in tables:
        restored = table.name.removeprefix(PREFIX)
        resolved = await conn.fetchval(_RESOLVED, restored)
        if await conn.fetchval("SELECT to_regclass($1)::oid", restored) != table.oid:
            raise ArchiveRefusedError(
                f"restore rolled back: `{restored}` resolves to {resolved} on search_path, not "
                f"to the restored {table.schema}.{restored}, so an image reading ADK's tables "
                "would not read the restored rows."
            )
    return statements


async def _restore_conflicts(conn: Any, archived: Sequence[_Relation]) -> list[str]:
    """One line for each name `restore` would give that is already taken: in
    the object's own schema, or for a table on `search_path`."""
    taken = []
    for relation in archived:
        restored = relation.name.removeprefix(PREFIX)
        in_schema = await conn.fetchval(
            "SELECT to_regclass(format('%I.%I', $1::text, $2::text)) IS NOT NULL "
            "OR to_regtype(format('%I.%I', $1::text, $2::text)) IS NOT NULL",
            relation.schema,
            restored,
        )
        if in_schema:
            taken.append(f"{relation.schema}.{restored} exists, beside {relation.qualified}")
            continue
        if relation.kind != "TABLE":
            continue
        resolved = await conn.fetchval(_RESOLVED, restored)
        if resolved is not None:
            taken.append(
                f"`{restored}` resolves to {resolved} on search_path, ahead of {relation.qualified}"
            )
    return taken


async def _drop(conn: Any, tables: Sequence[str]) -> list[str]:
    """Drop the schema-qualified, quoted `tables` in one statement, without
    `CASCADE`, and return it."""
    if not tables:
        return []
    statement = f"DROP TABLE {', '.join(tables)}"
    await conn.execute(statement)
    return [statement]


async def _drop_adk(conn: Any) -> list[str]:
    """Print the number of conversations in ADK's `sessions` table, its ledger
    sessions, then drop ADK's tables. The count is a snapshot taken before the
    `DROP`."""
    tables = await _adk_tables(conn)
    sessions = await conn.fetchval(_RESOLVED, "sessions")
    if sessions in tables:
        conversations = await conn.fetchval(
            f"SELECT count(*) FROM {sessions} WHERE app_name = $1", _LEDGER_APP
        )
        print(f"adk_archive drop-adk: {sessions} holds {conversations} conversations")
    return await _drop(conn, tables)


async def _drop_archive(conn: Any) -> list[str]:
    tables = []
    for name in ARCHIVED:
        qualified = await conn.fetchval(_RESOLVED, PREFIX + name)
        if qualified is None:
            continue
        if await conn.fetchval(_MISSING_COLUMNS, PREFIX + name, list(_columns(name))):
            raise ArchiveRefusedError(
                f"drop-archive drops nothing, because `{PREFIX}{name}` resolves to {qualified}, "
                f"which lacks the columns of `{name}`, so it is not the archive. Rename it or "
                "drop it by hand, then run drop-archive again."
            )
        tables.append(qualified)
    return await _drop(conn, tables)


# The verbs that cannot be undone, which `--apply` runs only with `--backup`.
DROPS = ("drop-adk", "drop-archive")

_VERBS: dict[str, Callable[[Any], Awaitable[list[str]]]] = {
    "archive": _archive,
    "restore": _restore,
    "drop-adk": _drop_adk,
    "drop-archive": _drop_archive,
}


async def run(
    engine: AsyncEngine,
    *,
    verb: str,
    apply: bool,
    allow_missing: bool = False,
    backup: str | None = None,
) -> int:
    """Run `verb` in one transaction on `engine`, commit it when `apply` and
    roll it back otherwise, print what it ran, and return the exit status.
    `allow_missing` lets `restore` run when the archive lacks some of ADK's
    tables. A drop (`DROPS`) with `apply` returns 2 without connecting unless
    `backup` is the numeric id of the backup taken before it."""
    if apply and verb in DROPS and not re.fullmatch(r"[0-9]+", backup or ""):
        print(
            f"adk_archive {verb}: --apply drops tables for good, so it needs --backup "
            "BACKUP_ID, the numeric id of the on-demand Cloud SQL backup taken for it "
            "(deploy/README.md, The archive job)",
            file=sys.stderr,
        )
        return 2
    act = (
        functools.partial(_restore, allow_missing=allow_missing)
        if verb == "restore"
        else _VERBS[verb]
    )
    async with engine.connect() as sa_conn:
        raw = await sa_conn.get_raw_connection()
        conn = raw.driver_connection
        assert conn is not None
        transaction = conn.transaction()
        await transaction.start()
        try:
            await bound_and_lock(conn)
            statements = await act(conn)
        except ArchiveRefusedError as exc:
            await transaction.rollback()
            print(f"adk_archive {verb}: {exc}", file=sys.stderr)
            return 1
        except asyncpg.PostgresError as exc:
            await transaction.rollback()
            print(
                f"adk_archive {verb}: refused, SQLSTATE {getattr(exc, 'sqlstate', None)}: {exc}",
                file=sys.stderr,
            )
            return 1
        except BaseException:
            await transaction.rollback()
            raise
        if apply:
            await transaction.commit()
        else:
            await transaction.rollback()
    if not statements:
        print(f"adk_archive {verb}: nothing to do")
        return 0
    outcome = "ran and committed" if apply else "dry run, rolled back; --apply commits"
    if apply and verb in DROPS:
        outcome += f", after backup {backup}"
    print(f"adk_archive {verb}: {outcome}:")
    for statement in statements:
        print(f"  {statement}")
    return 0


def _arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m harness.db.adk_archive",
        description="Archive, restore or drop ADK's session tables. A dry run, rolled back, "
        "unless --apply is given.",
    )
    verbs = parser.add_subparsers(dest="verb", required=True)
    helps = {
        "archive": "rename ADK's tables and adk_ledger_imports to adk_archive_<name>, as "
        "migration 0005 does",
        "restore": "rename every adk_archive_<name> table back, for an image that reads ADK",
        "drop-adk": "drop ADK's five tables, discarding ADK's copy of the conversations",
        "drop-archive": "drop every adk_archive_<name> table",
    }
    for verb, help_text in helps.items():
        verb_parser = verbs.add_parser(verb, help=help_text)
        verb_parser.add_argument("--apply", action="store_true", help="commit the change")
        if verb == "restore":
            verb_parser.add_argument(
                "--allow-missing",
                action="store_true",
                help="restore when the archive lacks some of ADK's five tables or the ledger",
            )
        if verb in DROPS:
            verb_parser.add_argument(
                "--backup",
                metavar="BACKUP_ID",
                help="the on-demand Cloud SQL backup taken before this drop; --apply needs it",
            )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the verb and print what it ran. Returns the exit status."""
    arguments = _arguments(argv)
    try:
        settings = Settings()
    except pydantic.ValidationError as exc:
        # Each error's location and message only: an error's input can hold a
        # secret, such as the password in `DATABASE_URL`.
        errors = "\n".join(
            f"  {'.'.join(map(str, error['loc'])) or 'settings'}: {error['msg']}"
            for error in exc.errors(include_input=False)
        )
        print(f"adk_archive: the database settings are invalid:\n{errors}", file=sys.stderr)
        return 2
    database = build_database(settings=settings)
    if database is None:
        print("adk_archive: set DATABASE_URL or the SESSION_STORE_* variables", file=sys.stderr)
        return 2

    async def go() -> int:
        try:
            return await run(
                database.engine,
                verb=arguments.verb,
                apply=arguments.apply,
                allow_missing=getattr(arguments, "allow_missing", False),
                backup=getattr(arguments, "backup", None),
            )
        finally:
            await database.aclose()

    return asyncio.run(go())


if __name__ == "__main__":
    sys.exit(main())

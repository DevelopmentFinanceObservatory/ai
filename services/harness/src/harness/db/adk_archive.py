"""Archives, restores and drops ADK's session tables.

`python -m harness.db.adk_archive VERB [--apply]` runs one verb in one
transaction and prints each statement it ran. Without `--apply` it rolls the
transaction back, so a dry run checks everything the verb would do, the
database's own refusals included, and changes nothing. With `--apply` it
commits.

- `archive` runs `0005_rename_adk_tables.sql`, the migration that renames
  ADK's five tables and `adk_ledger_imports` to `adk_archive_<name>`, so its
  checks and its rename are the migration's own. The file is one `DO` block
  and renames nothing a second time, so a rerun finds nothing to do.
- `restore` renames every `adk_archive_<name>` table back, with its indexes
  and owned sequences, for an image that reads ADK's tables. It refuses
  before renaming anything when a name it would restore exists in the
  object's schema, or, for a table, resolves on `search_path`, where an
  ADK-reading image that started after the archive created fresh tables.
  After renaming it checks, in the same transaction, that each restored
  table's name resolves on `search_path` to that table, and rolls back
  otherwise.
- `drop-adk` drops each of ADK's five tables that resolves on `search_path`
  and has ADK's columns, which discards ADK's copy of the conversations. A
  table of an ADK name without ADK's columns is another service's and stays.
- `drop-archive` drops every `adk_archive_<name>` table.

Tables resolve through `search_path`, as the migration resolves them. The
module connects as the harness does (`harness.settings.Settings`,
`harness.db.engine.build_database`): `DATABASE_URL`, or the
`SESSION_STORE_*` settings over the Cloud SQL Connector with IAM auth.

Exit status: 0 when the verb ran or found nothing to do, 1 when it or the
database refused, and 2 on a usage error or with no database configured.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any

import asyncpg
from sqlalchemy.ext.asyncio import AsyncEngine

from harness.db.engine import build_database
from harness.db.migrate import MIGRATION_LOCK_TIMEOUT_S, MIGRATIONS_DIR
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


async def _quoted(conn: Any, name: str) -> str:
    return await conn.fetchval("SELECT quote_ident($1)", name)


async def _archive(conn: Any) -> list[str]:
    before = await _relations(conn, ARCHIVED)
    await conn.execute(ARCHIVE_MIGRATION.read_text())
    after = {
        row["oid"]: row["relname"] for row in await conn.fetch(_DESCRIBE, [r.oid for r in before])
    }
    return [
        f"ALTER {relation.kind} {relation.qualified} RENAME TO {await _quoted(conn, after[relation.oid])}"
        for relation in before
        if after[relation.oid] != relation.name
    ]


async def _restore(conn: Any) -> list[str]:
    archived = [
        relation
        for relation in await _relations(conn, [PREFIX + name for name in ARCHIVED])
        if relation.name.startswith(PREFIX)
    ]
    tables = [relation for relation in archived if relation.kind == "TABLE"]
    taken = await _restore_conflicts(conn, archived)
    if taken:
        raise ArchiveRefusedError(
            "restore renames nothing, because names it would restore are taken:\n"
            + "\n".join(f"  {line}" for line in taken)
            + "\nAn image that reads ADK's tables created them after the archive. Drop them "
            "with `drop-adk --apply`, which discards what that image wrote to them, or rename "
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


async def _drop_adk(conn: Any) -> list[str]:
    statements = []
    for name in ("events", *(table for table in ADK_TABLES if table != "events")):
        qualified = await conn.fetchval(_RESOLVED, name)
        if qualified is None:
            continue
        missing = await conn.fetchval(
            "SELECT count(*) FROM unnest($2::text[]) AS c(col) WHERE NOT EXISTS ("
            "SELECT 1 FROM pg_attribute a WHERE a.attrelid = to_regclass($1) "
            "AND a.attname = c.col AND a.attnum > 0 AND NOT a.attisdropped)",
            name,
            list(_ADK_COLUMNS[name]),
        )
        if missing:
            continue
        statement = f"DROP TABLE {qualified} CASCADE"
        await conn.execute(statement)
        statements.append(statement)
    return statements


async def _drop_archive(conn: Any) -> list[str]:
    statements = []
    for name in ARCHIVED:
        qualified = await conn.fetchval(_RESOLVED, PREFIX + name)
        if qualified is None:
            continue
        statement = f"DROP TABLE {qualified} CASCADE"
        await conn.execute(statement)
        statements.append(statement)
    return statements


_VERBS: dict[str, Callable[[Any], Awaitable[list[str]]]] = {
    "archive": _archive,
    "restore": _restore,
    "drop-adk": _drop_adk,
    "drop-archive": _drop_archive,
}


async def run(engine: AsyncEngine, *, verb: str, apply: bool) -> int:
    """Run `verb` in one transaction on `engine`, commit it when `apply` and
    roll it back otherwise, print what it ran, and return the exit status."""
    async with engine.connect() as sa_conn:
        raw = await sa_conn.get_raw_connection()
        conn = raw.driver_connection
        assert conn is not None
        transaction = conn.transaction()
        await transaction.start()
        try:
            await conn.execute(f"SET LOCAL lock_timeout = {round(MIGRATION_LOCK_TIMEOUT_S * 1000)}")
            statements = await _VERBS[verb](conn)
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
        verbs.add_parser(verb, help=help_text).add_argument(
            "--apply", action="store_true", help="commit the change"
        )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the verb and print what it ran. Returns the exit status."""
    arguments = _arguments(argv)
    database = build_database(settings=Settings())
    if database is None:
        print("adk_archive: set DATABASE_URL or the SESSION_STORE_* variables", file=sys.stderr)
        return 2

    async def go() -> int:
        try:
            return await run(database.engine, verb=arguments.verb, apply=arguments.apply)
        finally:
            await database.aclose()

    return asyncio.run(go())


if __name__ == "__main__":
    sys.exit(main())

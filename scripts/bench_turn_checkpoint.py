"""Measure the wall time of a turn checkpoint (`TurnStore.advance`) by the
size of the orchestrator's spec blocks the turn already holds.

For each size the turn first stores spec blocks totalling about that many
bytes of JSON. Then, `--writes` times each, it measures:

- carried: a checkpoint whose spec blocks are unchanged, so it writes the
  cursor alone after comparing every block's digest;
- one added: a checkpoint that adds one typical spec block;
- digests: the in-process cost of digesting every held block, which each
  checkpoint pays before it writes (included in both rows above).

A checkpoint is the store's own transaction: the fence on the conversation
lease, the compare-and-swap on the cursor, the spec rows, and the commit.
It prints p50 and p95 per size and measure.

The spec blocks copy the shape `orchestrator:specs` stores (spec id, the
resolved scope with its entities, the definition) with random ids and
labels, so Postgres's TOAST compression sees text about as compressible as a
real turn's.

The run connects as the harness does (`harness.db.engine.build_database`:
`DATABASE_URL`, or the Cloud SQL Connector settings on a deployed job),
works in a schema of its own, applies the harness migrations there, and
drops the schema at the end, so it leaves the harness's own tables alone:

    DATABASE_URL=postgresql+asyncpg://postgres:postgres@127.0.0.1:55432/postgres \
        uv run --package harness python scripts/bench_turn_checkpoint.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import secrets
import statistics
import time
import uuid
from typing import Any

from harness.db.engine import build_database
from harness.db.migrate import apply_pending
from harness.settings import Settings
from harness.turn.store import TurnStore, _spec_digest
from sqlalchemy import event, text

SIZES = {"100 KB": 100_000, "1 MB": 1_000_000, "5 MB": 5_000_000}


def _spec(index: int) -> dict[str, Any]:
    spec_id = f"spec_{secrets.token_hex(16)}"
    entities = [
        {
            "ref": f"country/{secrets.token_hex(2).upper()}",
            "role": "observationAbout",
            "label": f"Recipient {secrets.token_hex(4)}",
        }
        for _ in range(8)
    ]
    return {
        "spec_id": spec_id,
        "resolved": {
            "facets": {"measure": secrets.token_hex(6)},
            "coverage": {
                "kind": "exact",
                "window": {"end_year": 2023, "start_year": 1960 + index % 40},
                "has_data": True,
            },
            "entities": entities,
            "stat_vars": {
                "member_count": index % 17 + 1,
                "served_count": index % 17 + 1,
                "data_date_range": {"end": "2023", "start": "1960"},
                "member_count_exact": True,
            },
            "confidence": "high",
            "rendered_sentence": f"Indicator {secrets.token_hex(8)} for eight recipients",
        },
        "definition": {
            "spec_id": spec_id,
            "identity": {"stat_var": f"StatVar_{secrets.token_hex(8)}"},
            "schema_version": 1,
        },
    }


def _specs(size: int) -> dict[str, Any]:
    specs: dict[str, Any] = {}
    written = 2
    while written < size:
        spec = _spec(len(specs))
        specs[spec["spec_id"]] = spec
        written += len(json.dumps(spec)) + len(spec["spec_id"]) + 6
    return specs


def _percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    return ordered[min(len(ordered) - 1, round(fraction * (len(ordered) - 1)))]


async def _measure(writes: int) -> None:
    database = build_database(settings=Settings())
    if database is None:
        raise SystemExit("no database configured; set DATABASE_URL")
    engine = database.engine
    schema = f"bench_{uuid.uuid4().hex[:12]}"

    @event.listens_for(engine.sync_engine, "connect")
    def _in_schema(connection: Any, _record: Any) -> None:
        cursor = connection.cursor()
        cursor.execute(f"SET search_path TO {schema}")
        cursor.close()

    async with engine.begin() as conn:
        await conn.execute(text(f"CREATE SCHEMA {schema}"))
    try:
        await apply_pending(engine)
        store = TurnStore(engine)
        print(f"{'specs':>7} {'measure':>10} {'writes':>6} {'p50 ms':>8} {'p95 ms':>8}")
        for label, size in SIZES.items():
            conversation_id = f"bench-{label}"
            async with engine.begin() as conn:
                await conn.execute(
                    text(
                        "INSERT INTO conversations (principal, conversation_id, "
                        "lease_segment_id, lease_owner, lease_expires_at) VALUES "
                        "('bench', :conversation_id, 'segment', 'bench', "
                        "extract(epoch FROM now()) + 3600)"
                    ),
                    {"conversation_id": conversation_id},
                )
            specs = _specs(size)
            cursor = await store.open_turn(
                "bench",
                conversation_id,
                segment_id="segment",
                turn_index=0,
                node="seed",
                inputs={"message": "bench"},
                state={"plan": None},
            )
            cursor = await store.advance(
                cursor,
                segment_id="segment",
                node="orchestrate",
                node_input=None,
                state={"plan": None},
                specs=specs,
            )
            carried: list[float] = []
            added: list[float] = []
            digests: list[float] = []
            for write in range(writes):
                started = time.perf_counter()
                for block in specs.values():
                    _spec_digest(block)
                digests.append((time.perf_counter() - started) * 1000)

                started = time.perf_counter()
                cursor = await store.advance(
                    cursor,
                    segment_id="segment",
                    node="orchestrate",
                    node_input={"write": write},
                    state={"plan": None, "orchestrator:frames": write},
                    specs=specs,
                )
                carried.append((time.perf_counter() - started) * 1000)

                spec = _spec(len(specs))
                specs = {**specs, spec["spec_id"]: spec}
                started = time.perf_counter()
                cursor = await store.advance(
                    cursor,
                    segment_id="segment",
                    node="planner",
                    node_input={"write": write},
                    state={"plan": None, "orchestrator:frames": write},
                    specs=specs,
                )
                added.append((time.perf_counter() - started) * 1000)
            for measure, samples in (
                ("carried", carried),
                ("one added", added),
                ("digests", digests),
            ):
                print(
                    f"{label:>7} {measure:>10} {writes:>6} "
                    f"{statistics.median(samples):>8.1f} {_percentile(samples, 0.95):>8.1f}"
                )
    finally:
        async with engine.begin() as conn:
            await conn.execute(text(f"DROP SCHEMA {schema} CASCADE"))
        await database.aclose()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--writes", type=int, default=50, help="checkpoints per size")
    args = parser.parse_args()
    asyncio.run(_measure(args.writes))


if __name__ == "__main__":
    main()

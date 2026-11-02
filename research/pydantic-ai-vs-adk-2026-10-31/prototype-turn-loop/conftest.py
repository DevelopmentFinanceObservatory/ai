"""A throwaway Postgres in Docker for the whole session, bound to 127.0.0.1 and
removed at the end. The own loop's store and ADK's `DatabaseSessionService` each
get a database in it."""

import asyncio
import subprocess
import time
import uuid
from collections.abc import Iterator

import asyncpg
import pytest

NAME = f"turnloop-pg-{uuid.uuid4().hex[:8]}"


async def _ready(dsn: str) -> None:
    deadline = time.monotonic() + 60
    while True:
        try:
            conn = await asyncpg.connect(dsn)
        except (OSError, asyncpg.PostgresError):
            if time.monotonic() > deadline:
                raise
            await asyncio.sleep(0.3)
            continue
        await conn.execute("create database adk")
        await conn.close()
        return


@pytest.fixture(scope="session")
def pg() -> Iterator[dict[str, str]]:
    subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--rm",
            "--name",
            NAME,
            "-e",
            "POSTGRES_PASSWORD=pw",
            "-p",
            "127.0.0.1::5432",
            "postgres:18-alpine",
        ],
        check=True,
        capture_output=True,
    )
    try:
        port = (
            subprocess.run(
                ["docker", "port", NAME, "5432/tcp"], check=True, capture_output=True, text=True
            )
            .stdout.split(":")[-1]
            .strip()
        )
        base = f"postgres:pw@127.0.0.1:{port}"
        asyncio.run(_ready(f"postgresql://{base}/postgres"))
        yield {
            "own": f"postgresql://{base}/postgres",
            "adk": f"postgresql+asyncpg://{base}/adk",
            "adk_raw": f"postgresql://{base}/adk",
        }
    finally:
        subprocess.run(["docker", "rm", "-f", NAME], capture_output=True, check=False)


@pytest.fixture
def turn_id() -> str:
    return f"c-{uuid.uuid4().hex[:8]}#t1"

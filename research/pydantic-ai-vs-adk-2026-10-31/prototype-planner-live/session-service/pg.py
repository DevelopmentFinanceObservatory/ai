"""A throwaway Postgres in Docker, bound to 127.0.0.1 on a free port.

The password is generated per server and handed to Docker and to the child processes
through the environment; it is never written to a file or put in an argument list.
"""

from __future__ import annotations

import asyncio
import os
import secrets
import socket
import subprocess
import time
from dataclasses import dataclass

import asyncpg

IMAGE = "postgres:18-alpine"


@dataclass(frozen=True)
class Server:
    name: str
    port: int
    password: str

    def url(self, database: str) -> str:
        """The SQLAlchemy URL of `database`, as `DatabaseSessionService` takes it."""
        return f"postgresql+asyncpg://postgres:{self.password}@127.0.0.1:{self.port}/{database}"

    async def create_database(self) -> str:
        """A new empty database on this server; returns its name."""
        name = f"db_{secrets.token_hex(6)}"
        conn = await asyncpg.connect(
            host="127.0.0.1", port=self.port, user="postgres", password=self.password
        )
        try:
            await conn.execute(f'CREATE DATABASE "{name}"')
        finally:
            await conn.close()
        return name


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def docker_available() -> bool:
    try:
        done = subprocess.run(["docker", "info"], capture_output=True, timeout=30, check=False)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return done.returncode == 0


async def _wait_ready(server: Server, timeout_s: float = 90) -> None:
    """Connect over TCP until the real server answers (the image's first-start server listens on
    its socket only)."""
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            conn = await asyncpg.connect(
                host="127.0.0.1", port=server.port, user="postgres", password=server.password
            )
        except (OSError, asyncpg.PostgresError):
            if time.monotonic() > deadline:
                raise
            await asyncio.sleep(0.5)
            continue
        await conn.close()
        return


def start() -> Server:
    """Start the container. Raises when Docker is not there rather than using anything else."""
    if not docker_available():
        raise RuntimeError("Docker is not available; this directory needs a real Postgres")
    server = Server(
        name=f"pai-sessions-{secrets.token_hex(4)}",
        port=_free_port(),
        password=secrets.token_urlsafe(24),
    )
    subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--rm",
            "--name",
            server.name,
            "-p",
            f"127.0.0.1:{server.port}:5432",
            "-e",
            "POSTGRES_PASSWORD",
            IMAGE,
        ],
        env={**os.environ, "POSTGRES_PASSWORD": server.password},
        check=True,
        capture_output=True,
        timeout=300,
    )
    try:
        asyncio.run(_wait_ready(server))
    except BaseException:
        stop(server)
        raise
    return server


def stop(server: Server) -> None:
    subprocess.run(
        ["docker", "rm", "-f", server.name], capture_output=True, check=False, timeout=60
    )

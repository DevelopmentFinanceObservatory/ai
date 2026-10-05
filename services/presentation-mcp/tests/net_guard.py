"""A network-blocking pytest fixture for offline test modules.

`no_network` is an autouse-ready pytest fixture: import it into a test module
(`from .net_guard import no_network`) to make every test in that module refuse
outbound network sockets, so "no network call" is a fact the suite enforces rather
than a claim about how a test happens to be written.
"""

from __future__ import annotations

import socket
from collections.abc import Iterator
from typing import Any

import pytest

_NETWORK_BLOCKED_MESSAGE = "network access is blocked in this test suite"


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Refuse outbound network sockets for the duration of a test.

    Blocks `socket.socket` for `AF_INET`/`AF_INET6`, plus `socket.create_connection`
    and `socket.getaddrinfo`. `AF_UNIX` is left alone: asyncio's event loop opens an
    `AF_UNIX` socketpair for its own self-pipe, and blocking `socket.socket` outright
    breaks it.
    """
    real_socket = socket.socket

    def _guarded_socket(family: int = socket.AF_INET, *args: Any, **kwargs: Any) -> socket.socket:
        if family in (socket.AF_INET, socket.AF_INET6):
            raise RuntimeError(_NETWORK_BLOCKED_MESSAGE)
        return real_socket(family, *args, **kwargs)

    def _blocked(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(_NETWORK_BLOCKED_MESSAGE)

    monkeypatch.setattr(socket, "socket", _guarded_socket)
    monkeypatch.setattr(socket, "create_connection", _blocked)
    monkeypatch.setattr(socket, "getaddrinfo", _blocked)
    yield

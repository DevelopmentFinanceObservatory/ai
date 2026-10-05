"""Regression coverage for `scripts/ask.py`'s env-var defaults.

`scripts/ask.py` sits outside the uv workspace (`pyproject.toml`'s
`[tool.uv.workspace]` lists only `packages/*` and `services/*`), so it can't
be imported as a package. It's loaded by file path instead, the same way
`test_studio_conformance.py` loads `evals/studio/checks.py`.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
ASK_PATH = REPO_ROOT / "scripts" / "ask.py"


def _load_ask() -> ModuleType:
    spec = importlib.util.spec_from_file_location("ask", ASK_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("ask", module)
    spec.loader.exec_module(module)
    return module


ask = _load_ask()


def test_blank_studio_api_token_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """`just env` carries `STUDIO_API_TOKEN=` from `.env.example` into `.env`
    verbatim, even blank, so the key is always present -- never absent. A
    present-but-empty value must fall back to `_DEFAULT_TOKEN` the same way
    an absent one does; otherwise `--token` becomes `""` and the CLI builds
    the literal header `Bearer `, which httpx2 refuses to send.
    """
    monkeypatch.setenv("STUDIO_API_TOKEN", "")
    args = ask.parse_args(["question"])
    assert args.token == ask._DEFAULT_TOKEN
    assert args.token != ""


def test_unset_studio_api_token_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("STUDIO_API_TOKEN", raising=False)
    args = ask.parse_args(["question"])
    assert args.token == ask._DEFAULT_TOKEN


def test_real_studio_api_token_is_used_verbatim(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STUDIO_API_TOKEN", "a-real-token")
    args = ask.parse_args(["question"])
    assert args.token == "a-real-token"


def test_blank_studio_url_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STUDIO_URL", "")
    args = ask.parse_args(["question"])
    assert args.url == ask._DEFAULT_URL


def test_unset_studio_url_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("STUDIO_URL", raising=False)
    args = ask.parse_args(["question"])
    assert args.url == ask._DEFAULT_URL

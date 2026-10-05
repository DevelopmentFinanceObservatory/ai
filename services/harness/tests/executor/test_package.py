from __future__ import annotations

import executor


def test_package_importable() -> None:
    assert executor.__version__

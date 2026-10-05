from __future__ import annotations

import translator


def test_package_importable() -> None:
    assert translator.__version__

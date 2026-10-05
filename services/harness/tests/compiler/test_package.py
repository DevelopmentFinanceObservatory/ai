from __future__ import annotations

import compiler


def test_package_importable() -> None:
    assert compiler.__version__

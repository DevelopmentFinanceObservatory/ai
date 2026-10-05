"""`build_store`: picks `GcsMaterialisationStore` when a bucket is named,
`InMemoryMaterialisationStore` otherwise, sourcing its `derive_content_version`
callable from `common.materialisation.content_version`'s own registry rather
than from its own caller -- so which caller's `verify_content_version_on_read`
wins a race decides only whether the check runs, never what it checks
against. `install_store_if_unset`: the ownership rule for a process where
more than one service's own lifespan may call it -- first call wins, every
later call is a no-op."""

from __future__ import annotations

import pytest
from common.materialisation import (
    ContentVersionDeriverUnregistered,
    InMemoryMaterialisationStore,
    build_store,
    factory,
    get_store,
    install_store_if_unset,
    register_content_version_deriver,
    set_store,
)
from common.materialisation import content_version as content_version_module


class _FakeGcsMaterialisationStore:
    """Stands in for `GcsMaterialisationStore`, monkeypatched over the name
    `factory.build_store` looks up, so the GCS branch can be exercised
    without a real client or credentials.
    """

    def __init__(
        self,
        *,
        bucket_name: str,
        verify_content_version_on_read: bool = False,
        derive_content_version: object | None = None,
    ) -> None:
        self.bucket_name = bucket_name
        self.verify_content_version_on_read = verify_content_version_on_read
        self.derive_content_version = derive_content_version


@pytest.fixture(autouse=True)
def _reset_content_version_deriver(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test in this module sees a clean "nothing registered" state, regardless of
    whatever the rest of this one-process suite (`analysis_mcp.frame_ref_resolution`
    included) has registered before or after this module runs -- `monkeypatch` restores
    the module's real global once the test ends."""
    monkeypatch.setattr(content_version_module, "_deriver", None)


def test_build_store_returns_in_memory_store_when_no_bucket_is_named() -> None:
    store = build_store(bucket_name=None)

    assert isinstance(store, InMemoryMaterialisationStore)


def test_build_store_warns_when_no_bucket_is_named_on_a_cloud_run_instance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("K_SERVICE", "analysis-mcp")
    warnings: list[str] = []
    monkeypatch.setattr(factory.logger, "warning", lambda msg, *args: warnings.append(msg % args))

    build_store(bucket_name=None)

    assert any("FRAME_STORE_BUCKET" in w and "analysis-mcp" in w for w in warnings)


def test_build_store_does_not_warn_when_a_bucket_is_named_on_a_cloud_run_instance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("K_SERVICE", "analysis-mcp")
    monkeypatch.setattr(factory, "GcsMaterialisationStore", _FakeGcsMaterialisationStore)
    warnings: list[str] = []
    monkeypatch.setattr(factory.logger, "warning", lambda msg, *args: warnings.append(msg % args))

    build_store(bucket_name="my-bucket")

    assert warnings == []


def test_build_store_does_not_warn_when_no_bucket_is_named_off_cloud_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("K_SERVICE", raising=False)
    warnings: list[str] = []
    monkeypatch.setattr(factory.logger, "warning", lambda msg, *args: warnings.append(msg % args))

    build_store(bucket_name=None)

    assert warnings == []


def test_build_store_uses_gcs_when_a_bucket_is_named(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(factory, "GcsMaterialisationStore", _FakeGcsMaterialisationStore)

    store = build_store(bucket_name="my-bucket")

    assert isinstance(store, _FakeGcsMaterialisationStore)
    assert store.bucket_name == "my-bucket"
    assert store.verify_content_version_on_read is False


def test_build_store_resolves_the_deriver_from_the_shared_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(factory, "GcsMaterialisationStore", _FakeGcsMaterialisationStore)
    fake_deriver = object()
    register_content_version_deriver(fake_deriver)  # type: ignore[arg-type]

    store = build_store(bucket_name="my-bucket", verify_content_version_on_read=True)

    assert isinstance(store, _FakeGcsMaterialisationStore)
    assert store.verify_content_version_on_read is True
    assert store.derive_content_version is fake_deriver


def test_build_store_refuses_verify_content_version_on_read_with_no_deriver_registered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The misconfigured case: a caller asks for the recompute check but nothing in
    this process has registered a deriver to check against (the topology a standalone
    data-mcp deploy that sets `FRAME_STORE_VERIFY_CONTENT_VERSION` without also
    mounting analysis-mcp would hit). Refuses outright rather than installing a store
    with the check silently disabled."""
    monkeypatch.setattr(factory, "GcsMaterialisationStore", _FakeGcsMaterialisationStore)

    with pytest.raises(ContentVersionDeriverUnregistered):
        build_store(bucket_name="my-bucket", verify_content_version_on_read=True)


def test_install_store_if_unset_installs_and_returns_the_store_when_nothing_is_set() -> None:
    installed = install_store_if_unset(bucket_name=None)

    assert isinstance(installed, InMemoryMaterialisationStore)
    assert get_store() is installed


def test_install_store_if_unset_is_a_no_op_when_a_store_is_already_set() -> None:
    first = InMemoryMaterialisationStore()
    set_store(first)

    second = install_store_if_unset(bucket_name=None)

    assert second is None
    assert get_store() is first


def test_install_store_if_unset_is_a_no_op_against_the_holders_own_lazy_default() -> None:
    """The holder's lazy default (built by a bare `get_store()` with nothing set) counts
    as "already set" for this check too -- `install_store_if_unset` must not replace it,
    the same as it wouldn't replace an explicitly `set_store`d one."""
    lazy_default = get_store()

    installed = install_store_if_unset(bucket_name=None)

    assert installed is None
    assert get_store() is lazy_default


def test_either_order_of_two_install_calls_ends_with_exactly_one_store_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mirrors the composed-process case: two services' own lifespans each call this,
    and `combine_lifespans` may run either one first. Whichever call goes first wins;
    the second is always a no-op, regardless of which service that is.
    """
    monkeypatch.setattr(factory, "GcsMaterialisationStore", _FakeGcsMaterialisationStore)

    first = install_store_if_unset(bucket_name="shared-bucket")
    second = install_store_if_unset(bucket_name="shared-bucket")

    assert first is not None
    assert second is None
    assert get_store() is first

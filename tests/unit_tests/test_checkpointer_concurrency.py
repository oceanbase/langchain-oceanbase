"""Unit tests for OceanBaseCheckpointSaver concurrency behavior.

Covers the conditional global lock: embedded SeekDB (selected by ``path`` /
``pyseekdb_client``, and running over a non-thread-safe process singleton) must
serialize access, while pooled remote backends must not.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy.pool import NullPool, QueuePool

from langchain_oceanbase.checkpointer import OceanBaseCheckpointSaver


def _make_saver(
    monkeypatch: pytest.MonkeyPatch,
    pool: Any,
    connection_args: dict[str, Any] | None = None,
) -> OceanBaseCheckpointSaver:
    """Build a saver whose obvector.engine.pool is the given object."""

    def fake_create_client(self: OceanBaseCheckpointSaver, **_: Any) -> None:
        self.obvector = SimpleNamespace(engine=SimpleNamespace(pool=pool))

    monkeypatch.setattr(OceanBaseCheckpointSaver, "_create_client", fake_create_client)
    return OceanBaseCheckpointSaver(connection_args=connection_args or {})


def test_embedded_path_serializes_regardless_of_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``path`` is the authoritative embedded signal, even with a pooled engine.

    This guards against pyobvector changing the embedded engine's pool class:
    detection must not rely on ``NullPool`` alone.
    """
    saver = _make_saver(
        monkeypatch,
        QueuePool.__new__(QueuePool),  # not a NullPool — yet still embedded
        connection_args={"path": "/tmp/seekdb", "db_name": "test"},
    )
    assert saver._serialize_access is True


def test_embedded_pyseekdb_client_serializes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An externally supplied ``pyseekdb_client`` is also embedded → serialize."""
    saver = _make_saver(
        monkeypatch,
        QueuePool.__new__(QueuePool),
        connection_args={"pyseekdb_client": object()},
    )
    assert saver._serialize_access is True


def test_remote_pooled_does_not_serialize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A remote config (no path) with a real pool must not serialize."""
    saver = _make_saver(
        monkeypatch,
        QueuePool.__new__(QueuePool),
        connection_args={"host": "127.0.0.1", "port": "2881"},
    )
    assert saver._serialize_access is False


def test_nullpool_backstop_serializes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A NullPool engine serializes even without a path (defensive backstop)."""
    saver = _make_saver(monkeypatch, NullPool.__new__(NullPool))
    assert saver._serialize_access is True


def test_unknown_pool_defaults_to_serialized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the pool cannot be determined, default to the safe (serialized) path."""

    def fake_create_client(self: OceanBaseCheckpointSaver, **_: Any) -> None:
        # No obvector assigned at all.
        return None

    monkeypatch.setattr(OceanBaseCheckpointSaver, "_create_client", fake_create_client)
    saver = OceanBaseCheckpointSaver(connection_args={})
    assert saver._serialize_access is True


def test_cursor_acquires_lock_only_when_serializing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_cursor takes the global lock for embedded but not for pooled backends."""

    class TrackingLock:
        def __init__(self) -> None:
            self.entered = 0

        def __enter__(self) -> "TrackingLock":
            self.entered += 1
            return self

        def __exit__(self, *exc: Any) -> None:
            return None

    class FakeConn:
        def __enter__(self) -> "FakeConn":
            return self

        def __exit__(self, *exc: Any) -> None:
            return None

    def install_engine(saver: OceanBaseCheckpointSaver) -> None:
        saver.obvector = SimpleNamespace(
            engine=SimpleNamespace(connect=lambda: FakeConn())
        )

    # Embedded (path): lock is acquired.
    embedded = _make_saver(
        monkeypatch,
        QueuePool.__new__(QueuePool),
        connection_args={"path": "/tmp/seekdb"},
    )
    install_engine(embedded)
    lock = TrackingLock()
    embedded.lock = lock  # type: ignore[assignment]
    with embedded._cursor():
        pass
    assert lock.entered == 1

    # Remote: lock is never acquired.
    remote = _make_saver(
        monkeypatch,
        QueuePool.__new__(QueuePool),
        connection_args={"host": "127.0.0.1"},
    )
    install_engine(remote)
    lock2 = TrackingLock()
    remote.lock = lock2  # type: ignore[assignment]
    with remote._cursor():
        pass
    assert lock2.entered == 0

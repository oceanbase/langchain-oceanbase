"""Unit tests for the async checkpoint saver surface."""

from __future__ import annotations

import threading
from typing import cast
from unittest.mock import MagicMock

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import CheckpointMetadata
from langgraph.checkpoint.conformance.capabilities import (
    BASE_CAPABILITIES,
    DetectedCapabilities,
)

from langchain_oceanbase.checkpointer import OceanBaseCheckpointSaver


@pytest.fixture
def saver(monkeypatch: pytest.MonkeyPatch) -> OceanBaseCheckpointSaver:
    """Create a saver without opening a real database connection."""
    monkeypatch.setattr(
        OceanBaseCheckpointSaver, "_create_client", lambda self, **_: None
    )
    return OceanBaseCheckpointSaver(connection_args={})


def test_detected_capabilities_include_base_async_methods(
    saver: OceanBaseCheckpointSaver,
) -> None:
    """Conformance detection should see the base async checkpoint capabilities."""
    detected = DetectedCapabilities.from_instance(saver)
    assert BASE_CAPABILITIES.issubset(detected.detected)


@pytest.mark.asyncio
async def test_aget_tuple_delegates_to_sync_method(
    saver: OceanBaseCheckpointSaver,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """aget_tuple should delegate to get_tuple."""
    expected = MagicMock(name="checkpoint_tuple")
    get_tuple = MagicMock(return_value=expected)
    monkeypatch.setattr(saver, "get_tuple", get_tuple)

    config: RunnableConfig = {
        "configurable": {"thread_id": "thread-1", "checkpoint_ns": ""}
    }
    result = await saver.aget_tuple(config)

    get_tuple.assert_called_once_with(config)
    assert result is expected


@pytest.mark.asyncio
async def test_alist_materializes_sync_results_before_async_yield(
    saver: OceanBaseCheckpointSaver,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """alist should exhaust the sync iterator before the async generator yields."""
    events: list[str] = []
    expected = [
        MagicMock(name="checkpoint_tuple_1"),
        MagicMock(name="checkpoint_tuple_2"),
    ]

    def sync_items() -> object:
        events.append("start")
        yield expected[0]
        events.append("after-first-yield")
        yield expected[1]
        events.append("after-second-yield")

    monkeypatch.setattr(
        saver,
        "list",
        lambda *args, **kwargs: cast(object, sync_items()),
    )

    results = []
    async for item in saver.alist(None):
        results.append(item)
        if len(results) == 1:
            break

    assert results == [expected[0]]
    assert events == ["start", "after-first-yield", "after-second-yield"]


def test_prepare_metadata_matches_langgraph_serialization_rules(
    saver: OceanBaseCheckpointSaver,
) -> None:
    """Metadata should merge supported config metadata and drop internal keys."""
    config: RunnableConfig = {
        "configurable": {
            "thread_id": "thread-1",
            "checkpoint_ns": "",
            "checkpoint_id": "cp-1",
            "user_scope": "team-a",
        },
        "metadata": {
            "run_id": "run-123",
            "request_id": "req-456",
        },
    }
    metadata = cast(
        CheckpointMetadata,
        {
            "source": "loop",
            "step": 3,
            "writes": {"ignored": True},
            "custom": "has\000null",
        },
    )

    prepared = saver._prepare_metadata(config, metadata)

    assert prepared["run_id"] == "run-123"
    assert prepared["request_id"] == "req-456"
    assert prepared["user_scope"] == "team-a"
    assert prepared["custom"] == "hasnull"
    assert "writes" not in prepared
    assert "thread_id" not in prepared
    assert "checkpoint_id" not in prepared
    assert "checkpoint_ns" not in prepared


@pytest.mark.asyncio
async def test_asetup_delegates_to_sync_method(
    saver: OceanBaseCheckpointSaver,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """asetup should delegate to setup."""
    setup = MagicMock()
    monkeypatch.setattr(saver, "setup", setup)

    await saver.asetup()

    setup.assert_called_once_with()


@pytest.mark.asyncio
async def test_aput_delegates_to_sync_method(
    saver: OceanBaseCheckpointSaver,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """aput should delegate to put and propagate its return value."""
    expected = MagicMock(name="next_config")
    put = MagicMock(return_value=expected)
    monkeypatch.setattr(saver, "put", put)

    config: RunnableConfig = {"configurable": {"thread_id": "t", "checkpoint_ns": ""}}
    checkpoint = MagicMock(name="checkpoint")
    metadata = cast(CheckpointMetadata, {"source": "loop"})
    new_versions = MagicMock(name="new_versions")

    result = await saver.aput(config, checkpoint, metadata, new_versions)

    put.assert_called_once_with(config, checkpoint, metadata, new_versions)
    assert result is expected


@pytest.mark.asyncio
async def test_aput_writes_delegates_to_sync_method(
    saver: OceanBaseCheckpointSaver,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """aput_writes should delegate to put_writes."""
    put_writes = MagicMock()
    monkeypatch.setattr(saver, "put_writes", put_writes)

    config: RunnableConfig = {
        "configurable": {"thread_id": "t", "checkpoint_id": "cp-1"}
    }
    writes = [("channel-1", "value-1")]

    await saver.aput_writes(config, writes, "task-1", "task-path")

    put_writes.assert_called_once_with(config, writes, "task-1", "task-path")


@pytest.mark.asyncio
async def test_adelete_thread_delegates_to_sync_method(
    saver: OceanBaseCheckpointSaver,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """adelete_thread should delegate to delete_thread."""
    delete_thread = MagicMock()
    monkeypatch.setattr(saver, "delete_thread", delete_thread)

    await saver.adelete_thread("thread-1")

    delete_thread.assert_called_once_with("thread-1")


@pytest.mark.asyncio
async def test_aprune_delegates_to_sync_method(
    saver: OceanBaseCheckpointSaver,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """aprune should delegate to prune and forward the strategy kwarg."""
    prune = MagicMock()
    monkeypatch.setattr(saver, "prune", prune)

    await saver.aprune(["thread-1", "thread-2"], strategy="delete")

    prune.assert_called_once_with(["thread-1", "thread-2"], strategy="delete")


@pytest.mark.asyncio
async def test_async_methods_run_off_the_event_loop_thread(
    saver: OceanBaseCheckpointSaver,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The sync work must run on an executor thread, not the loop thread.

    This is the only behavior the thread-pool wrappers add over a plain inline
    call, so it is asserted explicitly.
    """
    loop_thread_id = threading.get_ident()
    observed: dict[str, int] = {}

    def record_thread(_config: RunnableConfig) -> None:
        observed["thread_id"] = threading.get_ident()
        return None

    monkeypatch.setattr(saver, "get_tuple", record_thread)

    config: RunnableConfig = {"configurable": {"thread_id": "t", "checkpoint_ns": ""}}
    await saver.aget_tuple(config)

    assert observed["thread_id"] != loop_thread_id


def test_close_shuts_down_executor_and_is_idempotent(
    saver: OceanBaseCheckpointSaver,
) -> None:
    """close() should shut the executor down and tolerate repeated calls."""
    saver.close()
    assert saver._executor._shutdown is True
    # Second call must not raise.
    saver.close()


def test_context_manager_closes_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Using the saver as a context manager should shut the executor down."""
    monkeypatch.setattr(
        OceanBaseCheckpointSaver, "_create_client", lambda self, **_: None
    )
    with OceanBaseCheckpointSaver(connection_args={}) as saver:
        executor = saver._executor
        assert executor._shutdown is False
    assert executor._shutdown is True


@pytest.mark.asyncio
async def test_async_context_manager_closes_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Using the saver as an async context manager should shut the executor down."""
    monkeypatch.setattr(
        OceanBaseCheckpointSaver, "_create_client", lambda self, **_: None
    )
    async with OceanBaseCheckpointSaver(connection_args={}) as saver:
        executor = saver._executor
        assert executor._shutdown is False
    assert executor._shutdown is True

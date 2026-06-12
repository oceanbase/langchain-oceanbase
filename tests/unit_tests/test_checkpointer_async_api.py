"""Unit tests for the sync checkpoint saver async surface (should raise NotImplementedError)."""

from __future__ import annotations

from typing import cast

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import CheckpointMetadata

from langchain_oceanbase.checkpointer import OceanBaseCheckpointSaver


@pytest.fixture
def saver(monkeypatch: pytest.MonkeyPatch) -> OceanBaseCheckpointSaver:
    """Create a saver without opening a real database connection."""
    monkeypatch.setattr(
        OceanBaseCheckpointSaver, "_create_client", lambda self, **_: None
    )
    return OceanBaseCheckpointSaver(connection_args={})


@pytest.mark.asyncio
async def test_aget_tuple_raises_not_implemented(
    saver: OceanBaseCheckpointSaver,
) -> None:
    """aget_tuple should raise NotImplementedError."""
    config: RunnableConfig = {
        "configurable": {"thread_id": "thread-1", "checkpoint_ns": ""}
    }
    with pytest.raises(NotImplementedError, match="AsyncOceanBaseCheckpointSaver"):
        await saver.aget_tuple(config)


@pytest.mark.asyncio
async def test_alist_raises_not_implemented(
    saver: OceanBaseCheckpointSaver,
) -> None:
    """alist should raise NotImplementedError."""
    with pytest.raises(NotImplementedError, match="AsyncOceanBaseCheckpointSaver"):
        async for _ in saver.alist(None):
            pass


@pytest.mark.asyncio
async def test_aput_raises_not_implemented(
    saver: OceanBaseCheckpointSaver,
) -> None:
    """aput should raise NotImplementedError."""
    config: RunnableConfig = {
        "configurable": {"thread_id": "thread-1", "checkpoint_ns": ""}
    }
    with pytest.raises(NotImplementedError, match="AsyncOceanBaseCheckpointSaver"):
        await saver.aput(config, {}, {}, {})  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_aput_writes_raises_not_implemented(
    saver: OceanBaseCheckpointSaver,
) -> None:
    """aput_writes should raise NotImplementedError."""
    config: RunnableConfig = {
        "configurable": {
            "thread_id": "thread-1",
            "checkpoint_ns": "",
            "checkpoint_id": "cp-1",
        }
    }
    with pytest.raises(NotImplementedError, match="AsyncOceanBaseCheckpointSaver"):
        await saver.aput_writes(config, [], "task-1")


@pytest.mark.asyncio
async def test_adelete_thread_raises_not_implemented(
    saver: OceanBaseCheckpointSaver,
) -> None:
    """adelete_thread should raise NotImplementedError."""
    with pytest.raises(NotImplementedError, match="AsyncOceanBaseCheckpointSaver"):
        await saver.adelete_thread("thread-1")


@pytest.mark.asyncio
async def test_aprune_raises_not_implemented(
    saver: OceanBaseCheckpointSaver,
) -> None:
    """aprune should raise NotImplementedError."""
    with pytest.raises(NotImplementedError, match="AsyncOceanBaseCheckpointSaver"):
        await saver.aprune(["thread-1"])


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

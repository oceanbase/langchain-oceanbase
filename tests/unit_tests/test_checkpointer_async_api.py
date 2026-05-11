"""Unit tests for the async checkpoint saver surface."""

from __future__ import annotations

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

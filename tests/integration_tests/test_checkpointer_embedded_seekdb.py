# mypy: disable-error-code="import-untyped,typeddict-unknown-key,arg-type"
"""Embedded SeekDB regression tests for the checkpoint saver."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Generator

import pytest
from langgraph.checkpoint.base import Checkpoint, CheckpointMetadata

from langchain_oceanbase.checkpointer import OceanBaseCheckpointSaver


def _embedded_seekdb_runtime_available() -> bool:
    try:
        import pylibseekdb  # noqa: F401
    except ImportError:
        return False
    try:
        import pyseekdb  # noqa: F401
    except ImportError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _embedded_seekdb_runtime_available(),
    reason=(
        "embedded SeekDB requires pylibseekdb (e.g. pip install 'pyseekdb>=1.2' "
        "or pip install 'pyobvector[pyseekdb]')"
    ),
)


@pytest.fixture
def seekdb_path(tmp_path: Path) -> Generator[str, None, None]:
    root = tmp_path / f"checkpoint_seekdb_{uuid.uuid4().hex}"
    root.mkdir(parents=True)
    try:
        yield str(root / "seekdb_data")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_setup_is_idempotent_with_embedded_seekdb(seekdb_path: str) -> None:
    """setup() should succeed twice against an empty embedded SeekDB path."""
    saver = OceanBaseCheckpointSaver(connection_args={"db_name": "test"}, path=seekdb_path)
    saver.setup()
    saver.setup()


def test_put_get_and_pending_writes_round_trip_with_embedded_seekdb(
    seekdb_path: str,
) -> None:
    """Checkpoint values and pending writes should round-trip in embedded mode."""
    saver = OceanBaseCheckpointSaver(connection_args={"db_name": "test"}, path=seekdb_path)
    saver.setup()

    config = {"configurable": {"thread_id": "thread-1", "checkpoint_ns": ""}}
    checkpoint: Checkpoint = {
        "v": 1,
        "id": "cp-1",
        "ts": "2024-01-01T00:00:00+00:00",
        "channel_values": {"msg": "hello"},
        "channel_versions": {"msg": 1},
        "versions_seen": {},
        "pending_sends": [],
        "updated_channels": None,
    }
    metadata: CheckpointMetadata = {"source": "input", "step": -1}

    stored = saver.put(config, checkpoint, metadata, {"msg": 1})
    saver.put_writes(stored, [("reply", "world")], "task-1")

    result = saver.get_tuple(stored)

    assert result is not None
    assert result.checkpoint["channel_values"]["msg"] == "hello"
    assert result.pending_writes == [("task-1", "reply", "world")]


def test_prune_keep_latest_preserves_latest_checkpoint_and_writes(
    seekdb_path: str,
) -> None:
    """Prune keep_latest should keep the newest checkpoint per namespace and its writes."""
    saver = OceanBaseCheckpointSaver(connection_args={"db_name": "test"}, path=seekdb_path)
    saver.setup()

    thread_id = "thread-prune"
    root_config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
    root_checkpoint_1: Checkpoint = {
        "v": 1,
        "id": "root-cp-1",
        "ts": "2024-01-01T00:00:00+00:00",
        "channel_values": {"msg": "root-1"},
        "channel_versions": {"msg": 1},
        "versions_seen": {},
        "pending_sends": [],
        "updated_channels": None,
    }
    root_checkpoint_2: Checkpoint = {
        "v": 1,
        "id": "root-cp-2",
        "ts": "2024-01-01T00:01:00+00:00",
        "channel_values": {"msg": "root-2"},
        "channel_versions": {"msg": 2},
        "versions_seen": {},
        "pending_sends": [],
        "updated_channels": None,
    }
    child_config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": "child:1"}}
    child_checkpoint: Checkpoint = {
        "v": 1,
        "id": "child-cp-1",
        "ts": "2024-01-01T00:02:00+00:00",
        "channel_values": {"msg": "child-1"},
        "channel_versions": {"msg": 1},
        "versions_seen": {},
        "pending_sends": [],
        "updated_channels": None,
    }
    metadata: CheckpointMetadata = {"source": "loop", "step": 0}

    stored_root_1 = saver.put(root_config, root_checkpoint_1, metadata, {"msg": 1})
    stored_root_2 = saver.put(
        {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "",
                "checkpoint_id": stored_root_1["configurable"]["checkpoint_id"],
            }
        },
        root_checkpoint_2,
        {"source": "loop", "step": 1},
        {"msg": 2},
    )
    saver.put_writes(stored_root_2, [("reply", "latest-root")], "task-root")
    stored_child = saver.put(child_config, child_checkpoint, metadata, {"msg": 1})
    saver.put_writes(stored_child, [("reply", "latest-child")], "task-child")

    saver.prune([thread_id], strategy="keep_latest")

    root_results = list(saver.list({"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}))
    child_results = list(
        saver.list({"configurable": {"thread_id": thread_id, "checkpoint_ns": "child:1"}})
    )

    assert len(root_results) == 1
    assert root_results[0].checkpoint["id"] == "root-cp-2"
    assert root_results[0].pending_writes == [("task-root", "reply", "latest-root")]
    assert root_results[0].parent_config is None
    assert len(child_results) == 1
    assert child_results[0].checkpoint["id"] == "child-cp-1"
    assert child_results[0].pending_writes == [("task-child", "reply", "latest-child")]

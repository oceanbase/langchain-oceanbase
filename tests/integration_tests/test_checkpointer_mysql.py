# mypy: disable-error-code="import-untyped,typeddict-unknown-key,arg-type,no-untyped-def,misc"
"""Integration tests for OceanBaseCheckpointSaver against a real MySQL server."""

from __future__ import annotations

import os
import uuid

import pytest
from langgraph.checkpoint.base import Checkpoint, CheckpointMetadata
from sqlalchemy import text

from langchain_oceanbase import OceanBaseCheckpointSaver


def _ci_db_type() -> str:
    """Return the live database type provisioned by the CI matrix."""
    return os.getenv("OB_CI_DB_TYPE", "").strip().lower()


def _ci_mysql_server_available() -> bool:
    """Return True when CI provisioned a live MySQL server for tests."""
    return _ci_db_type() == "mysql"


def _mysql_connection_args_from_env() -> dict[str, str]:
    """Build MySQL server connection arguments from the shared CI contract."""
    return {
        "host": os.getenv("OB_HOST", "127.0.0.1"),
        "port": os.getenv("OB_PORT", "3306"),
        "user": os.getenv("OB_USER", "root"),
        "password": os.getenv("OB_PASSWORD", ""),
        "db_name": os.getenv("OB_DB", "test"),
    }


pytestmark = pytest.mark.skipif(
    not _ci_mysql_server_available(),
    reason="MySQL checkpoint tests run only in the mysql CI matrix.",
)


@pytest.fixture
def checkpointer() -> OceanBaseCheckpointSaver:
    """Create a MySQL-backed checkpointer for testing."""
    saver = OceanBaseCheckpointSaver(
        connection_args=_mysql_connection_args_from_env(),
    )
    saver.setup()
    return saver


def _checkpoint(checkpoint_id: str) -> Checkpoint:
    return {
        "v": 1,
        "id": checkpoint_id,
        "ts": "2024-01-01T00:00:00+00:00",
        "channel_values": {"messages": ["hello"]},
        "channel_versions": {"messages": "1"},
        "versions_seen": {},
        "pending_sends": [],
        "updated_channels": None,
    }


def test_setup_creates_required_indexes_against_mysql() -> None:
    """setup() should create the expected checkpoint indexes on MySQL."""
    saver = OceanBaseCheckpointSaver(
        connection_args=_mysql_connection_args_from_env(),
    )
    saver.setup()
    saver.setup()

    expected_indexes = {
        "checkpoints": "idx_checkpoints_thread_id",
        "checkpoint_blobs": "idx_checkpoint_blobs_thread_id",
        "checkpoint_writes": "idx_checkpoint_writes_thread_id",
    }

    with saver._cursor() as conn:
        for table_name, index_name in expected_indexes.items():
            result = conn.execute(text(f"SHOW INDEX FROM `{table_name}`"))
            actual_indexes = {row[2] for row in result.fetchall()}
            assert index_name in actual_indexes


def test_list_filters_string_metadata_against_mysql(
    checkpointer: OceanBaseCheckpointSaver,
) -> None:
    """String metadata filters should match rows on MySQL JSON columns."""
    thread_id = f"mysql-thread-{uuid.uuid4()}"
    config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
    checkpoint_id = str(uuid.uuid4())
    metadata: CheckpointMetadata = {"source": "loop", "step": 1}

    try:
        stored = checkpointer.put(
            config,
            _checkpoint(checkpoint_id),
            metadata,
            {"messages": "1"},
        )

        results = list(
            checkpointer.list(
                {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}},
                filter={"source": "loop"},
            )
        )

        assert [item.checkpoint["id"] for item in results] == [
            stored["configurable"]["checkpoint_id"]
        ]
        assert results[0].metadata["source"] == "loop"
    finally:
        checkpointer.delete_thread(thread_id)

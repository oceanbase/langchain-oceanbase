# mypy: disable-error-code="import-untyped,typeddict-unknown-key,arg-type,no-untyped-def,misc"
"""Integration tests for AsyncOceanBaseCheckpointSaver.

These tests verify that AsyncOceanBaseCheckpointSaver correctly implements
the BaseCheckpointSaver async interface for LangGraph persistence.

They run against embedded SeekDB when the native runtime is available,
or against a live MySQL/OceanBase server in CI.
"""

import os
import uuid
from typing import Annotated, TypedDict

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.base import Checkpoint, CheckpointMetadata
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from langchain_oceanbase import AsyncOceanBaseCheckpointSaver

# ==============================================================================
# Test Fixtures
# ==============================================================================


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


def _connection_args() -> dict[str, str]:
    """Return connection args based on environment."""
    if _ci_mysql_server_available():
        return _mysql_connection_args_from_env()
    pytest.skip(
        "Async checkpointer integration tests require a live MySQL/OceanBase server. "
        "Set OB_CI_DB_TYPE=mysql and OB_HOST/OB_PORT/OB_USER/OB_PASSWORD/OB_DB."
    )


@pytest.fixture
def connection_args() -> dict[str, str]:
    """Get connection args, skip if no server available."""
    return _connection_args()


@pytest.fixture
async def checkpointer(connection_args) -> AsyncOceanBaseCheckpointSaver:
    """Create and set up an async checkpointer for testing."""
    saver = AsyncOceanBaseCheckpointSaver(connection_args=connection_args)
    await saver.setup()
    return saver


@pytest.fixture
def unique_thread_id() -> str:
    """Generate a unique thread ID for each test."""
    return f"test-async-thread-{uuid.uuid4()}"


# ==============================================================================
# Basic Functionality Tests
# ==============================================================================


class TestAsyncCheckpointSaverSetup:
    """Tests for async checkpointer setup and initialization."""

    @pytest.mark.asyncio
    async def test_setup_creates_tables(self, connection_args):
        """Test that setup() creates required tables without errors."""
        saver = AsyncOceanBaseCheckpointSaver(connection_args=connection_args)
        await saver.setup()
        # Idempotent - should not raise on second call
        await saver.setup()

    @pytest.mark.asyncio
    async def test_sync_methods_raise_not_implemented(self, connection_args):
        """Test that sync methods are disabled on the async saver."""
        saver = AsyncOceanBaseCheckpointSaver(connection_args=connection_args)
        config = {"configurable": {"thread_id": "t1", "checkpoint_ns": ""}}

        with pytest.raises(NotImplementedError):
            saver.get_tuple(config)

        with pytest.raises(NotImplementedError):
            saver.list(config)

        with pytest.raises(NotImplementedError):
            saver.put(config, {}, {}, {})

        with pytest.raises(NotImplementedError):
            saver.put_writes(
                {**config, "configurable": {**config["configurable"], "checkpoint_id": "x"}},
                [],
                "task",
            )


class TestAsyncCheckpointSaverPutGet:
    """Tests for async put and get operations."""

    @pytest.mark.asyncio
    async def test_aput_and_aget_tuple(self, checkpointer, unique_thread_id):
        """Test basic async put and get operations."""
        config = {
            "configurable": {
                "thread_id": unique_thread_id,
                "checkpoint_ns": "",
            }
        }

        checkpoint: Checkpoint = {
            "v": 1,
            "id": str(uuid.uuid4()),
            "ts": "2024-01-01T00:00:00+00:00",
            "channel_values": {"messages": ["Hello"]},
            "channel_versions": {"messages": "1"},
            "versions_seen": {},
            "pending_sends": [],
            "updated_channels": None,
        }

        metadata: CheckpointMetadata = {"source": "input", "step": 0}

        new_config = await checkpointer.aput(config, checkpoint, metadata, {"messages": "1"})

        assert "checkpoint_id" in new_config["configurable"]
        assert new_config["configurable"]["thread_id"] == unique_thread_id

        result = await checkpointer.aget_tuple(new_config)

        assert result is not None
        assert result.checkpoint["id"] == checkpoint["id"]
        assert result.metadata["source"] == "input"

        await checkpointer.adelete_thread(unique_thread_id)

    @pytest.mark.asyncio
    async def test_aget_tuple_returns_none_for_nonexistent(self, checkpointer):
        """Test getting a checkpoint that doesn't exist returns None."""
        config = {
            "configurable": {
                "thread_id": f"nonexistent-{uuid.uuid4()}",
                "checkpoint_ns": "",
            }
        }

        result = await checkpointer.aget_tuple(config)
        assert result is None

    @pytest.mark.asyncio
    async def test_aput_with_complex_channel_values(self, checkpointer, unique_thread_id):
        """Test putting checkpoints with complex nested values."""
        config = {
            "configurable": {
                "thread_id": unique_thread_id,
                "checkpoint_ns": "",
            }
        }

        complex_data = {
            "list": [1, 2, {"nested": True}],
            "dict": {"key": "value", "number": 42},
        }

        checkpoint: Checkpoint = {
            "v": 1,
            "id": str(uuid.uuid4()),
            "ts": "2024-01-01T00:00:00+00:00",
            "channel_values": {"data": complex_data},
            "channel_versions": {"data": "1"},
            "versions_seen": {},
            "pending_sends": [],
            "updated_channels": None,
        }

        metadata: CheckpointMetadata = {"source": "loop", "step": 1}

        new_config = await checkpointer.aput(config, checkpoint, metadata, {"data": "1"})
        result = await checkpointer.aget_tuple(new_config)

        assert result is not None
        assert result.checkpoint["channel_values"]["data"] == complex_data

        await checkpointer.adelete_thread(unique_thread_id)

    @pytest.mark.asyncio
    async def test_aget_tuple_returns_latest_when_no_checkpoint_id(
        self, checkpointer, unique_thread_id
    ):
        """Without checkpoint_id, aget_tuple should return the latest checkpoint."""
        config = {
            "configurable": {
                "thread_id": unique_thread_id,
                "checkpoint_ns": "",
            }
        }

        # Put two checkpoints
        cp1_id = "00000000-0000-0000-0000-000000000001"
        cp2_id = "00000000-0000-0000-0000-000000000002"

        for cp_id, step in [(cp1_id, 0), (cp2_id, 1)]:
            checkpoint: Checkpoint = {
                "v": 1,
                "id": cp_id,
                "ts": "2024-01-01T00:00:00+00:00",
                "channel_values": {"step": step},
                "channel_versions": {"step": str(step)},
                "versions_seen": {},
                "pending_sends": [],
                "updated_channels": None,
            }
            metadata: CheckpointMetadata = {"source": "loop", "step": step}
            config = await checkpointer.aput(config, checkpoint, metadata, {"step": str(step)})

        # Get without checkpoint_id - should return latest
        latest_config = {
            "configurable": {
                "thread_id": unique_thread_id,
                "checkpoint_ns": "",
            }
        }
        result = await checkpointer.aget_tuple(latest_config)
        assert result is not None
        assert result.checkpoint["id"] == cp2_id

        await checkpointer.adelete_thread(unique_thread_id)

    @pytest.mark.asyncio
    async def test_aput_with_parent_checkpoint_id(self, checkpointer, unique_thread_id):
        """Test that parent_checkpoint_id is stored correctly."""
        config = {
            "configurable": {
                "thread_id": unique_thread_id,
                "checkpoint_ns": "",
            }
        }

        cp1: Checkpoint = {
            "v": 1,
            "id": "parent-cp",
            "ts": "2024-01-01T00:00:00+00:00",
            "channel_values": {},
            "channel_versions": {},
            "versions_seen": {},
            "pending_sends": [],
            "updated_channels": None,
        }
        config = await checkpointer.aput(
            config, cp1, {"source": "input", "step": 0}, {}
        )

        cp2: Checkpoint = {
            "v": 1,
            "id": "child-cp",
            "ts": "2024-01-01T00:00:01+00:00",
            "channel_values": {},
            "channel_versions": {},
            "versions_seen": {},
            "pending_sends": [],
            "updated_channels": None,
        }
        child_config = await checkpointer.aput(
            config, cp2, {"source": "loop", "step": 1}, {}
        )

        result = await checkpointer.aget_tuple(child_config)
        assert result is not None
        assert result.parent_config is not None
        assert result.parent_config["configurable"]["checkpoint_id"] == "parent-cp"

        await checkpointer.adelete_thread(unique_thread_id)

    @pytest.mark.asyncio
    async def test_aput_with_multiple_channels(self, checkpointer, unique_thread_id):
        """Test that multiple channel values are stored and retrieved correctly."""
        config = {
            "configurable": {
                "thread_id": unique_thread_id,
                "checkpoint_ns": "",
            }
        }

        checkpoint: Checkpoint = {
            "v": 1,
            "id": str(uuid.uuid4()),
            "ts": "2024-01-01T00:00:00+00:00",
            "channel_values": {
                "messages": [{"role": "user", "content": "hi"}],
                "context": {"source": "web", "urls": ["http://example.com"]},
                "counter": 42,
            },
            "channel_versions": {"messages": "1", "context": "1", "counter": "1"},
            "versions_seen": {},
            "pending_sends": [],
            "updated_channels": None,
        }

        metadata: CheckpointMetadata = {"source": "loop", "step": 1}
        new_versions = {"messages": "1", "context": "1"}

        new_config = await checkpointer.aput(config, checkpoint, metadata, new_versions)
        result = await checkpointer.aget_tuple(new_config)

        assert result is not None
        cv = result.checkpoint["channel_values"]
        assert cv["messages"] == [{"role": "user", "content": "hi"}]
        assert cv["context"] == {"source": "web", "urls": ["http://example.com"]}

        await checkpointer.adelete_thread(unique_thread_id)


class TestAsyncCheckpointSaverList:
    """Tests for listing checkpoints."""

    @pytest.mark.asyncio
    async def test_alist_checkpoints(self, checkpointer, unique_thread_id):
        """Test listing multiple checkpoints asynchronously."""
        base_config = {
            "configurable": {
                "thread_id": unique_thread_id,
                "checkpoint_ns": "",
            }
        }

        config = base_config.copy()
        config["configurable"] = base_config["configurable"].copy()

        for i in range(3):
            checkpoint: Checkpoint = {
                "v": 1,
                "id": f"checkpoint-{i:04d}",
                "ts": f"2024-01-0{i + 1}T00:00:00+00:00",
                "channel_values": {"count": i},
                "channel_versions": {"count": str(i)},
                "versions_seen": {},
                "pending_sends": [],
                "updated_channels": None,
            }
            metadata: CheckpointMetadata = {"source": "loop", "step": i}
            config = await checkpointer.aput(config, checkpoint, metadata, {"count": str(i)})

        checkpoints = []
        async for cp in checkpointer.alist(base_config, limit=10):
            checkpoints.append(cp)

        assert len(checkpoints) >= 3

        # Verify ordering (newest first by checkpoint_id)
        ids = [cp.checkpoint["id"] for cp in checkpoints[:3]]
        assert ids == sorted(ids, reverse=True)

        await checkpointer.adelete_thread(unique_thread_id)

    @pytest.mark.asyncio
    async def test_alist_with_limit(self, checkpointer, unique_thread_id):
        """Test listing checkpoints with a limit."""
        base_config = {
            "configurable": {
                "thread_id": unique_thread_id,
                "checkpoint_ns": "",
            }
        }

        config = base_config.copy()
        config["configurable"] = base_config["configurable"].copy()

        for i in range(5):
            checkpoint: Checkpoint = {
                "v": 1,
                "id": f"cp-limit-{i:04d}",
                "ts": f"2024-01-0{i + 1}T00:00:00+00:00",
                "channel_values": {},
                "channel_versions": {},
                "versions_seen": {},
                "pending_sends": [],
                "updated_channels": None,
            }
            metadata: CheckpointMetadata = {"source": "loop", "step": i}
            config = await checkpointer.aput(config, checkpoint, metadata, {})

        checkpoints = []
        async for cp in checkpointer.alist(base_config, limit=2):
            checkpoints.append(cp)

        assert len(checkpoints) == 2

        await checkpointer.adelete_thread(unique_thread_id)

    @pytest.mark.asyncio
    async def test_alist_with_before_filter(self, checkpointer, unique_thread_id):
        """Test listing checkpoints before a specific checkpoint."""
        base_config = {
            "configurable": {
                "thread_id": unique_thread_id,
                "checkpoint_ns": "",
            }
        }

        config = base_config.copy()
        config["configurable"] = base_config["configurable"].copy()

        checkpoint_ids = []
        for i in range(4):
            cp_id = f"cp-before-{i:04d}"
            checkpoint_ids.append(cp_id)
            checkpoint: Checkpoint = {
                "v": 1,
                "id": cp_id,
                "ts": f"2024-01-0{i + 1}T00:00:00+00:00",
                "channel_values": {},
                "channel_versions": {},
                "versions_seen": {},
                "pending_sends": [],
                "updated_channels": None,
            }
            metadata: CheckpointMetadata = {"source": "loop", "step": i}
            config = await checkpointer.aput(config, checkpoint, metadata, {})

        # List before the last checkpoint
        before_config = {
            "configurable": {
                "thread_id": unique_thread_id,
                "checkpoint_ns": "",
                "checkpoint_id": checkpoint_ids[-1],
            }
        }

        checkpoints = []
        async for cp in checkpointer.alist(base_config, before=before_config):
            checkpoints.append(cp)

        # All returned checkpoints should have id < the before checkpoint
        for cp in checkpoints:
            assert cp.checkpoint["id"] < checkpoint_ids[-1]

        await checkpointer.adelete_thread(unique_thread_id)

    @pytest.mark.asyncio
    async def test_alist_with_metadata_filter(self, checkpointer, unique_thread_id):
        """Test listing checkpoints with metadata filter."""
        config = {
            "configurable": {
                "thread_id": unique_thread_id,
                "checkpoint_ns": "",
            }
        }

        # Put checkpoints with different sources
        for i, source in enumerate(["input", "loop", "loop"]):
            checkpoint: Checkpoint = {
                "v": 1,
                "id": f"cp-filter-{i:04d}",
                "ts": f"2024-01-0{i + 1}T00:00:00+00:00",
                "channel_values": {},
                "channel_versions": {},
                "versions_seen": {},
                "pending_sends": [],
                "updated_channels": None,
            }
            metadata: CheckpointMetadata = {"source": source, "step": i}
            config = await checkpointer.aput(config, checkpoint, metadata, {})

        base_config = {
            "configurable": {
                "thread_id": unique_thread_id,
                "checkpoint_ns": "",
            }
        }

        # Filter by source=input
        checkpoints = []
        async for cp in checkpointer.alist(base_config, filter={"source": "input"}):
            checkpoints.append(cp)

        assert len(checkpoints) == 1
        assert checkpoints[0].metadata["source"] == "input"

        await checkpointer.adelete_thread(unique_thread_id)

    @pytest.mark.asyncio
    async def test_alist_empty_thread(self, checkpointer):
        """Test listing checkpoints for a thread with no data."""
        config = {
            "configurable": {
                "thread_id": f"empty-{uuid.uuid4()}",
                "checkpoint_ns": "",
            }
        }

        checkpoints = []
        async for cp in checkpointer.alist(config):
            checkpoints.append(cp)

        assert checkpoints == []


class TestAsyncCheckpointSaverPutWrites:
    """Tests for async put_writes functionality."""

    @pytest.mark.asyncio
    async def test_aput_writes(self, checkpointer, unique_thread_id):
        """Test storing intermediate writes asynchronously."""
        config = {
            "configurable": {
                "thread_id": unique_thread_id,
                "checkpoint_ns": "",
            }
        }

        checkpoint: Checkpoint = {
            "v": 1,
            "id": "writes-cp",
            "ts": "2024-01-01T00:00:00+00:00",
            "channel_values": {},
            "channel_versions": {},
            "versions_seen": {},
            "pending_sends": [],
            "updated_channels": None,
        }
        metadata: CheckpointMetadata = {"source": "input", "step": 0}
        new_config = await checkpointer.aput(config, checkpoint, metadata, {})

        writes = [
            ("messages", {"role": "user", "content": "Hello"}),
            ("messages", {"role": "assistant", "content": "Hi!"}),
        ]

        await checkpointer.aput_writes(new_config, writes, task_id="task-1")

        result = await checkpointer.aget_tuple(new_config)
        assert result is not None
        assert len(result.pending_writes) >= 2

        await checkpointer.adelete_thread(unique_thread_id)

    @pytest.mark.asyncio
    async def test_aput_writes_with_task_path(self, checkpointer, unique_thread_id):
        """Test storing writes with a task path."""
        config = {
            "configurable": {
                "thread_id": unique_thread_id,
                "checkpoint_ns": "",
            }
        }

        checkpoint: Checkpoint = {
            "v": 1,
            "id": "writes-path-cp",
            "ts": "2024-01-01T00:00:00+00:00",
            "channel_values": {},
            "channel_versions": {},
            "versions_seen": {},
            "pending_sends": [],
            "updated_channels": None,
        }
        metadata: CheckpointMetadata = {"source": "input", "step": 0}
        new_config = await checkpointer.aput(config, checkpoint, metadata, {})

        writes = [("output", {"result": "done"})]
        await checkpointer.aput_writes(
            new_config, writes, task_id="task-2", task_path="node:echo"
        )

        result = await checkpointer.aget_tuple(new_config)
        assert result is not None
        assert len(result.pending_writes) >= 1

        await checkpointer.adelete_thread(unique_thread_id)

    @pytest.mark.asyncio
    async def test_aput_writes_empty_list(self, checkpointer, unique_thread_id):
        """Test putting empty writes doesn't error."""
        config = {
            "configurable": {
                "thread_id": unique_thread_id,
                "checkpoint_ns": "",
            }
        }

        checkpoint: Checkpoint = {
            "v": 1,
            "id": "empty-writes-cp",
            "ts": "2024-01-01T00:00:00+00:00",
            "channel_values": {},
            "channel_versions": {},
            "versions_seen": {},
            "pending_sends": [],
            "updated_channels": None,
        }
        metadata: CheckpointMetadata = {"source": "input", "step": 0}
        new_config = await checkpointer.aput(config, checkpoint, metadata, {})

        # Should not raise
        await checkpointer.aput_writes(new_config, [], task_id="task-empty")

        await checkpointer.adelete_thread(unique_thread_id)


class TestAsyncCheckpointSaverDelete:
    """Tests for async delete functionality."""

    @pytest.mark.asyncio
    async def test_adelete_thread(self, checkpointer, unique_thread_id):
        """Test deleting all data for a thread asynchronously."""
        config = {
            "configurable": {
                "thread_id": unique_thread_id,
                "checkpoint_ns": "",
            }
        }

        checkpoint: Checkpoint = {
            "v": 1,
            "id": "to-delete",
            "ts": "2024-01-01T00:00:00+00:00",
            "channel_values": {"key": "value"},
            "channel_versions": {"key": "1"},
            "versions_seen": {},
            "pending_sends": [],
            "updated_channels": None,
        }
        metadata: CheckpointMetadata = {"source": "input", "step": 0}
        new_config = await checkpointer.aput(config, checkpoint, metadata, {"key": "1"})

        result = await checkpointer.aget_tuple(new_config)
        assert result is not None

        await checkpointer.adelete_thread(unique_thread_id)

        result = await checkpointer.aget_tuple(new_config)
        assert result is None

    @pytest.mark.asyncio
    async def test_adelete_thread_nonexistent(self, checkpointer):
        """Deleting a nonexistent thread should not raise."""
        await checkpointer.adelete_thread(f"nonexistent-{uuid.uuid4()}")


class TestAsyncCheckpointSaverPrune:
    """Tests for async prune functionality."""

    @pytest.mark.asyncio
    async def test_aprune_keep_latest(self, checkpointer, unique_thread_id):
        """Test pruning keeps only the latest checkpoint."""
        config = {
            "configurable": {
                "thread_id": unique_thread_id,
                "checkpoint_ns": "",
            }
        }

        # Create multiple checkpoints
        for i in range(4):
            checkpoint: Checkpoint = {
                "v": 1,
                "id": f"prune-cp-{i:04d}",
                "ts": f"2024-01-0{i + 1}T00:00:00+00:00",
                "channel_values": {"step": i},
                "channel_versions": {"step": str(i)},
                "versions_seen": {},
                "pending_sends": [],
                "updated_channels": None,
            }
            metadata: CheckpointMetadata = {"source": "loop", "step": i}
            config = await checkpointer.aput(config, checkpoint, metadata, {"step": str(i)})

        # Prune
        await checkpointer.aprune([unique_thread_id], strategy="keep_latest")

        # Should only have 1 checkpoint remaining
        base_config = {
            "configurable": {
                "thread_id": unique_thread_id,
                "checkpoint_ns": "",
            }
        }
        checkpoints = []
        async for cp in checkpointer.alist(base_config):
            checkpoints.append(cp)

        assert len(checkpoints) == 1
        assert checkpoints[0].checkpoint["id"] == "prune-cp-0003"

        await checkpointer.adelete_thread(unique_thread_id)

    @pytest.mark.asyncio
    async def test_aprune_delete_strategy(self, checkpointer, unique_thread_id):
        """Test pruning with delete strategy removes everything."""
        config = {
            "configurable": {
                "thread_id": unique_thread_id,
                "checkpoint_ns": "",
            }
        }

        checkpoint: Checkpoint = {
            "v": 1,
            "id": "prune-del-cp",
            "ts": "2024-01-01T00:00:00+00:00",
            "channel_values": {},
            "channel_versions": {},
            "versions_seen": {},
            "pending_sends": [],
            "updated_channels": None,
        }
        metadata: CheckpointMetadata = {"source": "input", "step": 0}
        await checkpointer.aput(config, checkpoint, metadata, {})

        await checkpointer.aprune([unique_thread_id], strategy="delete")

        result = await checkpointer.aget_tuple(config)
        assert result is None

    @pytest.mark.asyncio
    async def test_aprune_invalid_strategy(self, checkpointer, unique_thread_id):
        """Test that an invalid prune strategy raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported prune strategy"):
            await checkpointer.aprune([unique_thread_id], strategy="invalid")

    @pytest.mark.asyncio
    async def test_aprune_empty_list(self, checkpointer):
        """Pruning an empty list should be a no-op."""
        await checkpointer.aprune([], strategy="keep_latest")


class TestAsyncCheckpointSaverNamespaces:
    """Tests for checkpoint namespace support."""

    @pytest.mark.asyncio
    async def test_different_namespaces_are_isolated(self, checkpointer, unique_thread_id):
        """Checkpoints in different namespaces should not interfere."""
        ns1_config = {
            "configurable": {
                "thread_id": unique_thread_id,
                "checkpoint_ns": "ns1",
            }
        }
        ns2_config = {
            "configurable": {
                "thread_id": unique_thread_id,
                "checkpoint_ns": "ns2",
            }
        }

        cp1: Checkpoint = {
            "v": 1,
            "id": "ns1-cp",
            "ts": "2024-01-01T00:00:00+00:00",
            "channel_values": {"ns": "one"},
            "channel_versions": {"ns": "1"},
            "versions_seen": {},
            "pending_sends": [],
            "updated_channels": None,
        }
        cp2: Checkpoint = {
            "v": 1,
            "id": "ns2-cp",
            "ts": "2024-01-01T00:00:00+00:00",
            "channel_values": {"ns": "two"},
            "channel_versions": {"ns": "1"},
            "versions_seen": {},
            "pending_sends": [],
            "updated_channels": None,
        }

        await checkpointer.aput(ns1_config, cp1, {"source": "input", "step": 0}, {"ns": "1"})
        await checkpointer.aput(ns2_config, cp2, {"source": "input", "step": 0}, {"ns": "1"})

        r1 = await checkpointer.aget_tuple(ns1_config)
        r2 = await checkpointer.aget_tuple(ns2_config)

        assert r1 is not None
        assert r2 is not None
        assert r1.checkpoint["id"] == "ns1-cp"
        assert r2.checkpoint["id"] == "ns2-cp"

        await checkpointer.adelete_thread(unique_thread_id)


class TestAsyncCheckpointSaverVersioning:
    """Tests for version generation."""

    @pytest.mark.asyncio
    async def test_get_next_version(self, checkpointer):
        """Test version ID generation."""
        v1 = checkpointer.get_next_version(None, None)
        assert int(v1.split(".")[0]) == 1

        v2 = checkpointer.get_next_version(v1, None)
        assert int(v2.split(".")[0]) == 2

        # Monotonically increasing
        assert v2 > v1


# ==============================================================================
# LangGraph Integration Tests
# ==============================================================================


class ConversationState(TypedDict):
    """State for the test graph."""

    messages: Annotated[list[BaseMessage], add_messages]


def echo_node(state: ConversationState) -> dict:
    """Simple echo node for testing."""
    last_msg = state["messages"][-1].content
    return {"messages": [AIMessage(content=f"Echo: {last_msg}")]}


class TestAsyncLangGraphIntegration:
    """Tests for async LangGraph integration."""

    @pytest.mark.asyncio
    async def test_langgraph_async_with_checkpointer(self, checkpointer, unique_thread_id):
        """Test using async checkpointer with LangGraph graph.ainvoke()."""
        builder = StateGraph(ConversationState)
        builder.add_node("echo", echo_node)
        builder.add_edge(START, "echo")
        builder.add_edge("echo", END)

        graph = builder.compile(checkpointer=checkpointer)

        config = {"configurable": {"thread_id": unique_thread_id}}

        result1 = await graph.ainvoke(
            {"messages": [HumanMessage(content="Hello")]},
            config,
        )
        assert "Echo: Hello" in result1["messages"][-1].content

        result2 = await graph.ainvoke(
            {"messages": [HumanMessage(content="World")]},
            config,
        )
        assert "Echo: World" in result2["messages"][-1].content

        # Check state persistence
        state = await graph.aget_state(config)
        assert state.values is not None
        assert len(state.values.get("messages", [])) >= 4  # 2 human + 2 AI

        await checkpointer.adelete_thread(unique_thread_id)

    @pytest.mark.asyncio
    async def test_async_state_recovery(self, unique_thread_id, connection_args):
        """Test that state can be recovered after creating a new checkpointer."""
        # First session
        saver1 = AsyncOceanBaseCheckpointSaver(connection_args=connection_args)
        await saver1.setup()

        builder = StateGraph(ConversationState)
        builder.add_node("echo", echo_node)
        builder.add_edge(START, "echo")
        builder.add_edge("echo", END)

        graph1 = builder.compile(checkpointer=saver1)
        config = {"configurable": {"thread_id": unique_thread_id}}

        await graph1.ainvoke(
            {"messages": [HumanMessage(content="Remember this")]},
            config,
        )

        # Second session
        saver2 = AsyncOceanBaseCheckpointSaver(connection_args=connection_args)
        await saver2.setup()

        graph2 = builder.compile(checkpointer=saver2)
        state = await graph2.aget_state(config)

        assert state.values is not None
        messages = state.values.get("messages", [])
        assert len(messages) >= 2
        assert any("Remember this" in str(m.content) for m in messages)

        await saver2.adelete_thread(unique_thread_id)

    @pytest.mark.asyncio
    async def test_concurrent_async_invocations(self, checkpointer):
        """Test that concurrent ainvoke calls on different threads don't interfere."""
        import asyncio

        builder = StateGraph(ConversationState)
        builder.add_node("echo", echo_node)
        builder.add_edge(START, "echo")
        builder.add_edge("echo", END)

        graph = builder.compile(checkpointer=checkpointer)

        thread_ids = [f"concurrent-{uuid.uuid4()}" for _ in range(5)]

        async def run_thread(tid: str, msg: str):
            config = {"configurable": {"thread_id": tid}}
            result = await graph.ainvoke(
                {"messages": [HumanMessage(content=msg)]}, config
            )
            return tid, result

        tasks = [
            run_thread(tid, f"Message for {tid}")
            for tid in thread_ids
        ]
        results = await asyncio.gather(*tasks)

        for tid, result in results:
            assert f"Echo: Message for {tid}" in result["messages"][-1].content

        # Cleanup
        for tid in thread_ids:
            await checkpointer.adelete_thread(tid)

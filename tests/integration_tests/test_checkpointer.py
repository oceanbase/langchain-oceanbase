"""Integration tests for OceanBaseCheckpointSaver.

These tests verify that OceanBaseCheckpointSaver correctly implements
the BaseCheckpointSaver interface for LangGraph persistence.

Prerequisites:
    - Running OceanBase instance
    - Set environment variables:
        OCEANBASE_HOST, OCEANBASE_PORT, OCEANBASE_USER,
        OCEANBASE_PASSWORD, OCEANBASE_DB
"""

import os
import uuid
from typing import Annotated, TypedDict

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.base import Checkpoint, CheckpointMetadata
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from langchain_oceanbase import OceanBaseCheckpointSaver

# ==============================================================================
# Test Fixtures
# ==============================================================================


def get_connection_args():
    """Get OceanBase connection arguments from environment."""
    return {
        "host": os.getenv("OCEANBASE_HOST", "127.0.0.1"),
        "port": os.getenv("OCEANBASE_PORT", "2881"),
        "user": os.getenv("OCEANBASE_USER", "root@test"),
        "password": os.getenv("OCEANBASE_PASSWORD", ""),
        "db_name": os.getenv("OCEANBASE_DB", "test"),
    }


@pytest.fixture
def checkpointer():
    """Create a checkpointer for testing."""
    saver = OceanBaseCheckpointSaver(connection_args=get_connection_args())
    saver.setup()
    return saver


@pytest.fixture
def unique_thread_id():
    """Generate a unique thread ID for each test."""
    return f"test-thread-{uuid.uuid4()}"


# ==============================================================================
# Basic Functionality Tests
# ==============================================================================


class TestOceanBaseCheckpointSaverSetup:
    """Tests for checkpointer setup and initialization."""

    def test_setup_creates_tables(self):
        """Test that setup() creates required tables."""
        saver = OceanBaseCheckpointSaver(connection_args=get_connection_args())

        # Setup should not raise
        saver.setup()

        # Verify tables exist by running setup again (should be idempotent)
        saver.setup()

    def test_default_connection_args(self):
        """Test that default connection args are used when none provided."""
        saver = OceanBaseCheckpointSaver()
        assert saver.connection_args is not None
        assert "host" in saver.connection_args


class TestOceanBaseCheckpointSaverPutGet:
    """Tests for put and get operations."""

    def test_put_and_get_checkpoint(self, checkpointer, unique_thread_id):
        """Test basic put and get operations."""
        config = {
            "configurable": {
                "thread_id": unique_thread_id,
                "checkpoint_ns": "",
            }
        }

        # Create a checkpoint
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

        metadata: CheckpointMetadata = {
            "source": "input",
            "step": 0,
        }

        # Put the checkpoint
        new_config = checkpointer.put(config, checkpoint, metadata, {"messages": "1"})

        # Verify returned config
        assert "checkpoint_id" in new_config["configurable"]
        assert new_config["configurable"]["thread_id"] == unique_thread_id

        # Get the checkpoint
        result = checkpointer.get_tuple(new_config)

        assert result is not None
        assert result.checkpoint["id"] == checkpoint["id"]
        assert result.metadata["source"] == "input"

        # Cleanup
        checkpointer.delete_thread(unique_thread_id)

    def test_get_nonexistent_checkpoint(self, checkpointer):
        """Test getting a checkpoint that doesn't exist."""
        config = {
            "configurable": {
                "thread_id": f"nonexistent-{uuid.uuid4()}",
                "checkpoint_ns": "",
            }
        }

        result = checkpointer.get_tuple(config)
        assert result is None

    def test_put_with_complex_values(self, checkpointer, unique_thread_id):
        """Test putting checkpoints with complex nested values."""
        config = {
            "configurable": {
                "thread_id": unique_thread_id,
                "checkpoint_ns": "",
            }
        }

        # Complex nested data
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

        metadata: CheckpointMetadata = {
            "source": "loop",
            "step": 1,
        }

        new_config = checkpointer.put(config, checkpoint, metadata, {"data": "1"})
        result = checkpointer.get_tuple(new_config)

        assert result is not None
        assert result.checkpoint["channel_values"]["data"] == complex_data

        # Cleanup
        checkpointer.delete_thread(unique_thread_id)


class TestOceanBaseCheckpointSaverList:
    """Tests for listing checkpoints."""

    def test_list_checkpoints(self, checkpointer, unique_thread_id):
        """Test listing multiple checkpoints."""
        config = {
            "configurable": {
                "thread_id": unique_thread_id,
                "checkpoint_ns": "",
            }
        }

        # Create multiple checkpoints
        for i in range(3):
            checkpoint: Checkpoint = {
                "v": 1,
                "id": f"checkpoint-{i}",
                "ts": f"2024-01-0{i+1}T00:00:00+00:00",
                "channel_values": {"count": i},
                "channel_versions": {"count": str(i)},
                "versions_seen": {},
                "pending_sends": [],
                "updated_channels": None,
            }

            metadata: CheckpointMetadata = {
                "source": "loop",
                "step": i,
            }

            config = checkpointer.put(config, checkpoint, metadata, {"count": str(i)})

        # List checkpoints
        checkpoints = list(checkpointer.list(config, limit=10))

        assert len(checkpoints) >= 3

        # Verify ordering (newest first)
        steps = [cp.metadata.get("step", -1) for cp in checkpoints[:3]]
        assert steps == sorted(steps, reverse=True)

        # Cleanup
        checkpointer.delete_thread(unique_thread_id)

    def test_list_with_limit(self, checkpointer, unique_thread_id):
        """Test listing checkpoints with a limit."""
        config = {
            "configurable": {
                "thread_id": unique_thread_id,
                "checkpoint_ns": "",
            }
        }

        # Create 5 checkpoints
        for i in range(5):
            checkpoint: Checkpoint = {
                "v": 1,
                "id": f"cp-{i}",
                "ts": f"2024-01-0{i+1}T00:00:00+00:00",
                "channel_values": {},
                "channel_versions": {},
                "versions_seen": {},
                "pending_sends": [],
                "updated_channels": None,
            }

            metadata: CheckpointMetadata = {"source": "loop", "step": i}
            config = checkpointer.put(config, checkpoint, metadata, {})

        # List with limit
        checkpoints = list(checkpointer.list(config, limit=2))
        assert len(checkpoints) == 2

        # Cleanup
        checkpointer.delete_thread(unique_thread_id)


class TestOceanBaseCheckpointSaverPutWrites:
    """Tests for put_writes functionality."""

    def test_put_writes(self, checkpointer, unique_thread_id):
        """Test storing intermediate writes."""
        # First create a checkpoint
        config = {
            "configurable": {
                "thread_id": unique_thread_id,
                "checkpoint_ns": "",
            }
        }

        checkpoint: Checkpoint = {
            "v": 1,
            "id": "main-checkpoint",
            "ts": "2024-01-01T00:00:00+00:00",
            "channel_values": {},
            "channel_versions": {},
            "versions_seen": {},
            "pending_sends": [],
            "updated_channels": None,
        }

        metadata: CheckpointMetadata = {"source": "input", "step": 0}
        new_config = checkpointer.put(config, checkpoint, metadata, {})

        # Add writes
        writes = [
            ("messages", {"role": "user", "content": "Hello"}),
            ("messages", {"role": "assistant", "content": "Hi there!"}),
        ]

        checkpointer.put_writes(new_config, writes, task_id="task-1")

        # Retrieve and verify writes are attached
        result = checkpointer.get_tuple(new_config)
        assert result is not None
        assert len(result.pending_writes) >= 2

        # Cleanup
        checkpointer.delete_thread(unique_thread_id)


class TestOceanBaseCheckpointSaverDelete:
    """Tests for delete functionality."""

    def test_delete_thread(self, checkpointer, unique_thread_id):
        """Test deleting all data for a thread."""
        config = {
            "configurable": {
                "thread_id": unique_thread_id,
                "checkpoint_ns": "",
            }
        }

        # Create a checkpoint
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
        new_config = checkpointer.put(config, checkpoint, metadata, {"key": "1"})

        # Verify it exists
        result = checkpointer.get_tuple(new_config)
        assert result is not None

        # Delete
        checkpointer.delete_thread(unique_thread_id)

        # Verify it's gone
        result = checkpointer.get_tuple(new_config)
        assert result is None


class TestOceanBaseCheckpointSaverVersioning:
    """Tests for version generation."""

    def test_get_next_version(self, checkpointer):
        """Test version ID generation."""
        # First version
        v1 = checkpointer.get_next_version(None, None)
        assert v1.startswith("1")

        # Next version
        v2 = checkpointer.get_next_version(v1, None)
        assert v2.startswith("2")

        # Versions should be monotonically increasing
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


class TestLangGraphIntegration:
    """Tests for LangGraph integration."""

    def test_langgraph_with_checkpointer(self, checkpointer, unique_thread_id):
        """Test using checkpointer with a LangGraph graph."""
        # Build graph
        builder = StateGraph(ConversationState)
        builder.add_node("echo", echo_node)
        builder.add_edge(START, "echo")
        builder.add_edge("echo", END)

        graph = builder.compile(checkpointer=checkpointer)

        # Run conversation
        config = {"configurable": {"thread_id": unique_thread_id}}

        result1 = graph.invoke(
            {"messages": [HumanMessage(content="Hello")]},
            config,
        )
        assert "Echo: Hello" in result1["messages"][-1].content

        result2 = graph.invoke(
            {"messages": [HumanMessage(content="World")]},
            config,
        )
        assert "Echo: World" in result2["messages"][-1].content

        # Check state persistence
        state = graph.get_state(config)
        assert state.values is not None
        assert len(state.values.get("messages", [])) >= 4  # 2 human + 2 AI

        # Cleanup
        checkpointer.delete_thread(unique_thread_id)

    def test_state_recovery_after_restart(self, unique_thread_id):
        """Test that state can be recovered after creating a new checkpointer."""
        # First session
        checkpointer1 = OceanBaseCheckpointSaver(connection_args=get_connection_args())
        checkpointer1.setup()

        builder = StateGraph(ConversationState)
        builder.add_node("echo", echo_node)
        builder.add_edge(START, "echo")
        builder.add_edge("echo", END)

        graph1 = builder.compile(checkpointer=checkpointer1)
        config = {"configurable": {"thread_id": unique_thread_id}}

        graph1.invoke(
            {"messages": [HumanMessage(content="Remember this")]},
            config,
        )

        # Second session (simulating restart)
        checkpointer2 = OceanBaseCheckpointSaver(connection_args=get_connection_args())
        checkpointer2.setup()

        graph2 = builder.compile(checkpointer=checkpointer2)

        # Recover state
        state = graph2.get_state(config)
        assert state.values is not None

        messages = state.values.get("messages", [])
        assert len(messages) >= 2
        assert any("Remember this" in str(m.content) for m in messages)

        # Cleanup
        checkpointer2.delete_thread(unique_thread_id)

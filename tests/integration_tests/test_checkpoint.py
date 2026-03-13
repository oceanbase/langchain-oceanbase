import time
import uuid
import pickle
import pytest
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import Checkpoint, CheckpointMetadata, ChannelVersions

from langchain_oceanbase.checkpoint import OceanBaseSaver
from langchain_oceanbase.vectorstores import DEFAULT_OCEANBASE_CONNECTION

# Define connection args (should match your environment)
CONNECTION_ARGS = DEFAULT_OCEANBASE_CONNECTION

@pytest.fixture
def saver():
    """Create a saver instance and clean up tables after test."""
    table_name = f"test_checkpoint_{uuid.uuid4().hex}"
    writes_table_name = f"test_writes_{uuid.uuid4().hex}"

    saver = OceanBaseSaver(
        connection_args=CONNECTION_ARGS,
        table_name=table_name,
        writes_table_name=writes_table_name
    )

    yield saver

    # Cleanup
    try:
        saver.client.drop_table(table_name)
        saver.client.drop_table(writes_table_name)
    except Exception:
        pass

def test_put_and_get_tuple(saver):
    """Test saving and retrieving a checkpoint."""
    thread_id = "thread-1"
    checkpoint_ns = ""

    config = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "checkpoint_id": "cp-1",
        }
    }

    checkpoint = {
        "v": 1,
        "ts": "2024-01-01T00:00:00.000000+00:00",
        "id": "cp-1",
        "channel_values": {
            "messages": [HumanMessage(content="Hello")]
        },
        "channel_versions": {"messages": 1},
        "versions_seen": {"messages": {"node-1": 1}},
        "pending_sends": [],
    }

    metadata = {
        "source": "input",
        "step": 1,
        "writes": {},
        "parents": {},
    }

    # 1. Put checkpoint
    saver.put(config, checkpoint, metadata, {})

    # 2. Get tuple
    cp_tuple = saver.get_tuple(config)

    assert cp_tuple is not None
    assert cp_tuple.checkpoint["v"] == 1
    assert cp_tuple.checkpoint["id"] == "cp-1"
    assert len(cp_tuple.checkpoint["channel_values"]["messages"]) == 1
    assert cp_tuple.checkpoint["channel_values"]["messages"][0].content == "Hello"
    assert cp_tuple.metadata["source"] == "input"

def test_list_checkpoints(saver):
    """Test listing checkpoints."""
    thread_id = "thread-list"

    # Create 3 checkpoints
    for i in range(3):
        cp_id = f"cp-{i}"
        config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_id": cp_id,
            }
        }
        checkpoint = {
            "v": 1,
            "ts": "...",
            "id": cp_id,
            "channel_values": {"msg": i},
            "channel_versions": {},
            "versions_seen": {},
            "pending_sends": [],
        }
        saver.put(config, checkpoint, {}, {})
        time.sleep(0.1)  # Ensure timestamps differ

    # List all
    config = {"configurable": {"thread_id": thread_id}}
    checkpoints = list(saver.list(config))

    assert len(checkpoints) == 3
    # Should be ordered by created_at DESC (latest first)
    assert checkpoints[0].checkpoint["id"] == "cp-2"
    assert checkpoints[2].checkpoint["id"] == "cp-0"

def test_put_writes(saver):
    """Test saving and retrieving pending writes."""
    thread_id = "thread-writes"
    checkpoint_id = "cp-writes"

    config = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_id": checkpoint_id,
        }
    }

    # Put checkpoint first (writes are usually associated with a checkpoint)
    checkpoint = {
        "v": 1,
        "ts": "...",
        "id": checkpoint_id,
        "channel_values": {},
        "channel_versions": {},
        "versions_seen": {},
        "pending_sends": []
    }
    saver.put(config, checkpoint, {}, {})

    # Put writes
    writes = [
        ("channel-1", "value-1"),
        ("channel-2", {"complex": "value"}),
    ]
    saver.put_writes(config, writes, "task-1")

    # Verify via get_tuple
    cp_tuple = saver.get_tuple(config)
    assert len(cp_tuple.pending_writes) == 2

    # Writes format: (task_id, channel, value)
    w1 = next(w for w in cp_tuple.pending_writes if w[1] == "channel-1")
    assert w1[0] == "task-1"
    assert w1[2] == "value-1"

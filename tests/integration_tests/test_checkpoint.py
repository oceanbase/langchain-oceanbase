import os
import time
import uuid

import pytest
from langchain_core.messages import HumanMessage

from langchain_oceanbase.checkpoint import OceanBaseSaver

# Define connection args from environment (falls back to defaults)
CONNECTION_ARGS = {
    "host": os.getenv("OB_HOST", "127.0.0.1"),
    "port": os.getenv("OB_PORT", "2881"),
    "user": os.getenv("OB_USER", "root@test"),
    "password": os.getenv("OB_PASSWORD", ""),
    "db_name": os.getenv("OB_DB", "test"),
}


@pytest.fixture
def saver():
    """Create a saver instance and clean up tables after test."""
    table_name = f"test_checkpoint_{uuid.uuid4().hex}"
    writes_table_name = f"test_writes_{uuid.uuid4().hex}"

    saver = OceanBaseSaver(
        connection_args=CONNECTION_ARGS,
        table_name=table_name,
        writes_table_name=writes_table_name,
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
        "channel_values": {"messages": [HumanMessage(content="Hello")]},
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
        "pending_sends": [],
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


@pytest.mark.parametrize("suffix", ["", "'\\; -- :bind 雪"])
def test_legacy_read_isolation_and_bound_values(saver, suffix):
    """Exercise bound keys, pending writes and pagination on the real backend."""
    from sqlalchemy import text

    keys = {"thread_id": "thread" + suffix, "checkpoint_ns": "namespace" + suffix}
    configs = []
    for thread_keys, checkpoint_id, created_at in [
        ({"thread_id": "other", "checkpoint_ns": ""}, "cp-2" + suffix, "9000"),
        (keys, "cp-1" + suffix, "1000"),
        (keys, "cp-2" + suffix, "2000"),
    ]:
        checkpoint = {
            "v": 1,
            "id": checkpoint_id,
            "ts": "2026-01-01T00:00:00+00:00",
            "channel_values": {"messages": ["private state"]},
            "channel_versions": {},
            "versions_seen": {},
        }
        config = saver.put({"configurable": thread_keys}, checkpoint, {}, {})
        configs.append(config)
        with saver.client.engine.begin() as conn:
            conn.execute(
                text(
                    f"UPDATE {saver.table_name} SET created_at = :created_at "
                    "WHERE thread_id = :thread_id AND checkpoint_ns = :checkpoint_ns "
                    "AND checkpoint_id = :checkpoint_id"
                ),
                {**config["configurable"], "created_at": created_at},
            )

    first, second = configs[1:]
    saver.put_writes(second, [("messages", "safe value")], "task-1")
    for config in [second, {"configurable": keys}]:
        result = saver.get_tuple(config)
        assert result is not None
        assert result.checkpoint["id"] == "cp-2" + suffix
        assert result.pending_writes == [("task-1", "messages", "safe value")]
    assert [item.config for item in saver.list(second, limit=1)] == [second]
    assert [item.config for item in saver.list(second, before=second)] == [first]

    for field in ["thread_id", "checkpoint_ns", "checkpoint_id"]:
        config = {"configurable": {**second["configurable"], field: "' OR '1'='1"}}
        assert saver.get_tuple(config) is None
        if field != "checkpoint_id":
            config["configurable"].pop("checkpoint_id")
            assert saver.get_tuple(config) is None
            assert list(saver.list(config)) == []
    before = {"configurable": {"checkpoint_id": "missing' OR '1'='1"}}
    assert list(saver.list(second, before=before)) == []
    with pytest.raises(ValueError, match="limit"):
        list(saver.list(second, limit="1 OFFSET 1"))

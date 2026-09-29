"""Tests for the legacy checkpoint saver surface."""

from __future__ import annotations

import pickle
from collections.abc import Iterator
from typing import Any

import pytest
from pyobvector import ObVecClient
from sqlalchemy import MetaData, Table, create_engine, text
from sqlalchemy.dialects.sqlite import insert

from langchain_oceanbase.checkpoint import OceanBaseSaver


def test_legacy_oceanbase_saver_emits_deprecation_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OceanBaseSaver should warn users to migrate to OceanBaseCheckpointSaver."""
    monkeypatch.setattr(OceanBaseSaver, "_create_client", lambda self, **_: None)
    monkeypatch.setattr(
        OceanBaseSaver, "_create_tables_if_not_exists", lambda self: None
    )

    with pytest.warns(DeprecationWarning, match="OceanBaseCheckpointSaver"):
        OceanBaseSaver(connection_args={})


@pytest.fixture
def saver(monkeypatch: pytest.MonkeyPatch) -> Iterator[OceanBaseSaver]:
    """Use real SQL execution and ObVecClient reads without an OceanBase server."""
    client = ObVecClient.__new__(ObVecClient)
    client.engine = create_engine("sqlite:///:memory:")
    client.metadata_obj = MetaData()
    with client.engine.begin() as conn:
        conn.execute(
            text("""
            CREATE TABLE checkpoints (
                thread_id TEXT, checkpoint_ns TEXT, checkpoint_id TEXT,
                parent_checkpoint_id TEXT, type TEXT, checkpoint BLOB,
                metadata BLOB, created_at TEXT,
                PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
            )
        """)
        )
        conn.execute(
            text("""
            CREATE TABLE writes (
                thread_id TEXT, checkpoint_ns TEXT, checkpoint_id TEXT,
                task_id TEXT, idx TEXT, channel TEXT, type TEXT, value BLOB,
                PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
            )
        """)
        )

    def upsert(table_name: str, data: list[dict[str, Any]]) -> None:
        # Only the backend-specific write syntax needs adapting for SQLite.
        table = Table(table_name, client.metadata_obj, autoload_with=client.engine)
        with client.engine.begin() as conn:
            conn.execute(insert(table).prefix_with("OR REPLACE"), data)

    monkeypatch.setattr(client, "upsert", upsert)
    monkeypatch.setattr(
        OceanBaseSaver,
        "_create_client",
        lambda self, **_: setattr(self, "client", client),
    )
    monkeypatch.setattr(
        OceanBaseSaver, "_create_tables_if_not_exists", lambda self: None
    )
    with pytest.warns(DeprecationWarning):
        instance = OceanBaseSaver({}, "checkpoints", "writes")
    try:
        yield instance
    finally:
        client.engine.dispose()


def save_checkpoint(
    saver: OceanBaseSaver,
    *,
    thread_id: str = "victim",
    checkpoint_ns: str = "",
    checkpoint_id: str = "cp-1",
    created_at: str = "1000",
    parent_id: str | None = None,
) -> dict[str, Any]:
    config = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "checkpoint_id": parent_id,
        }
    }
    checkpoint = {
        "v": 1,
        "id": checkpoint_id,
        "ts": "2026-01-01T00:00:00+00:00",
        "channel_values": {"messages": ["private state"]},
        "channel_versions": {},
        "versions_seen": {},
    }
    result = saver.put(config, checkpoint, {"source": "input", "step": 1}, {})
    with saver.client.engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE checkpoints SET created_at = :created_at "
                "WHERE thread_id = :thread_id AND checkpoint_ns = :checkpoint_ns "
                "AND checkpoint_id = :checkpoint_id"
            ),
            {**result["configurable"], "created_at": created_at},
        )
    return result


@pytest.mark.parametrize("method", ["get_tuple", "list"])
@pytest.mark.parametrize("field", ["thread_id", "checkpoint_ns"])
def test_sql_predicates_cannot_read_another_checkpoint(
    saver: OceanBaseSaver, method: str, field: str
) -> None:
    save_checkpoint(saver)
    config = {"configurable": {"thread_id": "victim", "checkpoint_ns": ""}}
    config["configurable"][field] = "' OR '1'='1"
    if method == "list":
        assert list(saver.list(config)) == []
    else:
        assert saver.get_tuple(config) is None


@pytest.mark.parametrize("field", ["thread_id", "checkpoint_ns", "checkpoint_id"])
def test_specific_checkpoint_requires_all_three_keys(
    saver: OceanBaseSaver, field: str
) -> None:
    config = save_checkpoint(saver)
    config["configurable"][field] = "not-the-requested-key"
    assert saver.get_tuple(config) is None


class PickleCanary:
    def __reduce__(self) -> tuple[Any, tuple[str]]:
        # Harmless observable side effect if an injected row is deserialized.
        return print, ("UNTRUSTED_PICKLE_EXECUTED",)


@pytest.mark.parametrize("field", ["thread_id", "checkpoint_ns", "checkpoint_id"])
@pytest.mark.parametrize("latest", [False, True])
def test_stored_sql_payload_cannot_inject_pending_write_pickle(
    saver: OceanBaseSaver, capsys: pytest.CaptureFixture[str], field: str, latest: bool
) -> None:
    payload = (
        "attacker' UNION SELECT 'task','channel','pickle',"
        f"x'{pickle.dumps(PickleCanary()).hex()}' FROM writes WHERE '1'='1' OR '1'='1"
    )
    keys = {"thread_id": "attacker", "checkpoint_ns": "", "checkpoint_id": "cp-1"}
    keys[field] = payload
    config = save_checkpoint(saver, **keys)
    saver.put_writes(config, [("messages", "safe value")], "safe-task")
    if latest:
        config["configurable"].pop("checkpoint_id")

    result = saver.get_tuple(config)

    assert "UNTRUSTED_PICKLE_EXECUTED" not in capsys.readouterr().out
    assert result is not None
    assert result.pending_writes == [("safe-task", "messages", "safe value")]


def test_before_id_cannot_inject_a_pagination_boundary(saver: OceanBaseSaver) -> None:
    save_checkpoint(saver, thread_id="other", created_at="9000")
    config = save_checkpoint(saver, created_at="1000")
    before = {"configurable": {"checkpoint_id": "missing' OR '1'='1"}}
    assert list(saver.list(config, before=before)) == []


def test_before_checkpoint_is_scoped_to_thread_and_namespace(
    saver: OceanBaseSaver,
) -> None:
    save_checkpoint(
        saver, thread_id="other", checkpoint_id="boundary", created_at="9000"
    )
    save_checkpoint(
        saver, checkpoint_ns="other", checkpoint_id="boundary", created_at="8000"
    )
    config = save_checkpoint(saver, checkpoint_id="older", created_at="1000")
    before = save_checkpoint(saver, checkpoint_id="boundary", created_at="2000")
    assert [item.config for item in saver.list(config, before=before)] == [config]


def test_simple_checkpoint_and_writes_control(saver: OceanBaseSaver) -> None:
    config = save_checkpoint(saver)
    saver.put_writes(config, [("messages", "hello")], "task-1")
    result = saver.get_tuple(config)
    assert result is not None
    assert result.checkpoint["id"] == "cp-1"
    assert result.pending_writes == [("task-1", "messages", "hello")]
    assert [item.config for item in saver.list(config)] == [config]


@pytest.mark.parametrize("limit", ["1 OFFSET 1", "1; DROP TABLE checkpoints", -1, 1.5])
def test_invalid_limit_is_rejected(saver: OceanBaseSaver, limit: Any) -> None:
    config = save_checkpoint(saver)
    with pytest.raises((TypeError, ValueError), match="limit"):
        list(saver.list(config, limit=limit))


@pytest.mark.parametrize("suffix", ["", "'\\; -- :bind 雪"])
def test_checkpoint_round_trip_and_pagination(
    saver: OceanBaseSaver, suffix: str
) -> None:
    save_checkpoint(saver, thread_id="other", created_at="9000")
    keys = {"thread_id": "thread" + suffix, "checkpoint_ns": "namespace" + suffix}
    first = save_checkpoint(saver, **keys, checkpoint_id="cp-1" + suffix)
    second = save_checkpoint(
        saver,
        **keys,
        checkpoint_id="cp-2" + suffix,
        created_at="2000",
        parent_id="cp-1" + suffix,
    )
    saver.put_writes(second, [("messages", {"text": "hello"})], "task-1")

    specific = saver.get_tuple(second)
    latest = saver.get_tuple({"configurable": keys})
    assert specific is not None and latest is not None
    assert specific.checkpoint == latest.checkpoint
    assert specific.checkpoint["id"] == "cp-2" + suffix
    assert specific.metadata == {"source": "input", "step": 1}
    assert specific.parent_config == first
    assert (
        specific.pending_writes
        == latest.pending_writes
        == [("task-1", "messages", {"text": "hello"})]
    )
    assert [item.config for item in saver.list(second)] == [second, first]
    assert [item.config for item in saver.list(second, before=second)] == [first]
    assert [item.config for item in saver.list(second, limit=1)] == [second]
    # Preserve the legacy zero-limit behavior (no limit).
    assert [item.config for item in saver.list(second, limit=0)] == [second, first]

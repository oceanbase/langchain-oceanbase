"""Unit tests for AsyncOceanBaseCheckpointSaver."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.runnables import RunnableConfig

from langchain_oceanbase.async_checkpointer import AsyncOceanBaseCheckpointSaver


@pytest.fixture
def saver(monkeypatch: pytest.MonkeyPatch) -> AsyncOceanBaseCheckpointSaver:
    """Create a saver without opening a real database connection."""
    monkeypatch.setattr(
        AsyncOceanBaseCheckpointSaver, "_create_engine", lambda self: None
    )
    return AsyncOceanBaseCheckpointSaver(connection_args={})


# ==============================================================================
# Sync methods must raise NotImplementedError
# ==============================================================================


class TestSyncMethodsRaiseNotImplemented:
    """All sync methods must raise NotImplementedError."""

    def test_get_tuple_raises(self, saver: AsyncOceanBaseCheckpointSaver) -> None:
        config: RunnableConfig = {
            "configurable": {"thread_id": "t1", "checkpoint_ns": ""}
        }
        with pytest.raises(NotImplementedError, match="does not support synchronous"):
            saver.get_tuple(config)

    def test_list_raises(self, saver: AsyncOceanBaseCheckpointSaver) -> None:
        config: RunnableConfig = {
            "configurable": {"thread_id": "t1", "checkpoint_ns": ""}
        }
        with pytest.raises(NotImplementedError, match="does not support synchronous"):
            saver.list(config)

    def test_put_raises(self, saver: AsyncOceanBaseCheckpointSaver) -> None:
        config: RunnableConfig = {
            "configurable": {"thread_id": "t1", "checkpoint_ns": ""}
        }
        with pytest.raises(NotImplementedError, match="does not support synchronous"):
            saver.put(config, {}, {}, {})  # type: ignore[arg-type]

    def test_put_writes_raises(self, saver: AsyncOceanBaseCheckpointSaver) -> None:
        config: RunnableConfig = {
            "configurable": {
                "thread_id": "t1",
                "checkpoint_ns": "",
                "checkpoint_id": "cp1",
            }
        }
        with pytest.raises(NotImplementedError, match="does not support synchronous"):
            saver.put_writes(config, [], "task1")

    def test_error_message_suggests_async_alternative(
        self, saver: AsyncOceanBaseCheckpointSaver
    ) -> None:
        config: RunnableConfig = {
            "configurable": {"thread_id": "t1", "checkpoint_ns": ""}
        }
        with pytest.raises(NotImplementedError, match="aget_tuple"):
            saver.get_tuple(config)

    def test_list_error_message_suggests_alist(
        self, saver: AsyncOceanBaseCheckpointSaver
    ) -> None:
        config: RunnableConfig = {
            "configurable": {"thread_id": "t1", "checkpoint_ns": ""}
        }
        with pytest.raises(NotImplementedError, match="alist"):
            saver.list(config)


# ==============================================================================
# Async method signatures
# ==============================================================================


class TestAsyncMethodSignatures:
    """Async methods have correct signatures and are coroutines."""

    def test_aget_tuple_is_coroutine(
        self, saver: AsyncOceanBaseCheckpointSaver
    ) -> None:
        assert asyncio.iscoroutinefunction(saver.aget_tuple)

    def test_aput_is_coroutine(self, saver: AsyncOceanBaseCheckpointSaver) -> None:
        assert asyncio.iscoroutinefunction(saver.aput)

    def test_aput_writes_is_coroutine(
        self, saver: AsyncOceanBaseCheckpointSaver
    ) -> None:
        assert asyncio.iscoroutinefunction(saver.aput_writes)

    def test_alist_is_async_generator_function(
        self, saver: AsyncOceanBaseCheckpointSaver
    ) -> None:
        import inspect

        assert inspect.isasyncgenfunction(saver.alist)

    def test_adelete_thread_is_coroutine(
        self, saver: AsyncOceanBaseCheckpointSaver
    ) -> None:
        assert asyncio.iscoroutinefunction(saver.adelete_thread)

    def test_aprune_is_coroutine(self, saver: AsyncOceanBaseCheckpointSaver) -> None:
        assert asyncio.iscoroutinefunction(saver.aprune)

    def test_setup_is_coroutine(self, saver: AsyncOceanBaseCheckpointSaver) -> None:
        assert asyncio.iscoroutinefunction(saver.setup)


# ==============================================================================
# Batch channel values (fixes N+1 query)
# ==============================================================================


class TestBatchChannelValues:
    """Test the batch query construction for channel values."""

    @pytest.mark.asyncio
    async def test_empty_channel_versions_returns_empty(
        self, saver: AsyncOceanBaseCheckpointSaver
    ) -> None:
        mock_conn = AsyncMock()
        result = await saver._load_channel_values(
            mock_conn, "t1", "", '{"channel_versions": {}}'
        )
        assert result == {}
        mock_conn.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_missing_channel_versions_key_returns_empty(
        self, saver: AsyncOceanBaseCheckpointSaver
    ) -> None:
        mock_conn = AsyncMock()
        result = await saver._load_channel_values(mock_conn, "t1", "", "{}")
        assert result == {}
        mock_conn.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_single_channel_uses_batch_query(
        self, saver: AsyncOceanBaseCheckpointSaver
    ) -> None:
        mock_conn = AsyncMock()
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [
            ("messages", "json", b"base64:dGVzdA=="),
        ]
        mock_conn.execute.return_value = mock_result

        checkpoint_data = '{"channel_versions": {"messages": "1.0"}}'

        with patch.object(saver.serde, "loads_typed", return_value="decoded"):
            result = await saver._load_channel_values(mock_conn, "t1", "", checkpoint_data)

        # Only 1 query for any number of channels
        assert mock_conn.execute.call_count == 1
        assert "messages" in result
        assert result["messages"] == "decoded"

    @pytest.mark.asyncio
    async def test_batch_query_with_multiple_channels(
        self, saver: AsyncOceanBaseCheckpointSaver
    ) -> None:
        mock_conn = AsyncMock()
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [
            ("messages", "json", b"base64:dGVzdA=="),
            ("state", "json", b"base64:c3RhdGU="),
            ("counter", "json", b"base64:Mg=="),
        ]
        mock_conn.execute.return_value = mock_result

        checkpoint_data = json.dumps({
            "channel_versions": {
                "messages": "1.0",
                "state": "2.0",
                "counter": "3.0",
            }
        })

        with patch.object(saver.serde, "loads_typed", return_value="decoded"):
            result = await saver._load_channel_values(mock_conn, "t1", "ns", checkpoint_data)

        # Still only 1 query
        assert mock_conn.execute.call_count == 1
        assert len(result) == 3
        # Verify params include all channels
        call_args = mock_conn.execute.call_args
        params = call_args[0][1]
        assert params["thread_id"] == "t1"
        assert params["checkpoint_ns"] == "ns"
        assert params["channel_0"] == "messages"
        assert params["channel_1"] == "state"
        assert params["channel_2"] == "counter"

    @pytest.mark.asyncio
    async def test_empty_type_channels_are_skipped(
        self, saver: AsyncOceanBaseCheckpointSaver
    ) -> None:
        mock_conn = AsyncMock()
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [
            ("messages", "empty", None),
            ("state", "json", b"base64:dGVzdA=="),
        ]
        mock_conn.execute.return_value = mock_result

        checkpoint_data = '{"channel_versions": {"messages": "1", "state": "1"}}'

        with patch.object(saver.serde, "loads_typed", return_value="decoded"):
            result = await saver._load_channel_values(mock_conn, "t1", "", checkpoint_data)

        assert "messages" not in result
        assert "state" in result

    @pytest.mark.asyncio
    async def test_none_blob_channels_are_skipped(
        self, saver: AsyncOceanBaseCheckpointSaver
    ) -> None:
        mock_conn = AsyncMock()
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [
            ("messages", "json", None),
        ]
        mock_conn.execute.return_value = mock_result

        checkpoint_data = '{"channel_versions": {"messages": "1"}}'

        result = await saver._load_channel_values(mock_conn, "t1", "", checkpoint_data)
        assert "messages" not in result

    @pytest.mark.asyncio
    async def test_accepts_dict_checkpoint_data(
        self, saver: AsyncOceanBaseCheckpointSaver
    ) -> None:
        """checkpoint_data can be a dict (already parsed) rather than a string."""
        mock_conn = AsyncMock()
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [
            ("messages", "json", b"base64:dGVzdA=="),
        ]
        mock_conn.execute.return_value = mock_result

        checkpoint_data = {"channel_versions": {"messages": "1.0"}}

        with patch.object(saver.serde, "loads_typed", return_value="decoded"):
            result = await saver._load_channel_values(mock_conn, "t1", "", checkpoint_data)

        assert result == {"messages": "decoded"}


# ==============================================================================
# Pending writes loading
# ==============================================================================


class TestLoadPendingWrites:
    """Test async pending writes loading."""

    @pytest.mark.asyncio
    async def test_load_empty_writes(
        self, saver: AsyncOceanBaseCheckpointSaver
    ) -> None:
        mock_conn = AsyncMock()
        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_conn.execute.return_value = mock_result

        result = await saver._load_pending_writes(mock_conn, "t1", "", "cp1")
        assert result == []

    @pytest.mark.asyncio
    async def test_load_multiple_writes(
        self, saver: AsyncOceanBaseCheckpointSaver
    ) -> None:
        mock_conn = AsyncMock()
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [
            ("task-1", "messages", "json", b"base64:dGVzdA=="),
            ("task-1", "output", "json", b"base64:b3V0"),
        ]
        mock_conn.execute.return_value = mock_result

        with patch.object(saver.serde, "loads_typed", return_value="value"):
            result = await saver._load_pending_writes(mock_conn, "t1", "", "cp1")

        assert len(result) == 2
        assert result[0] == ("task-1", "messages", "value")
        assert result[1] == ("task-1", "output", "value")

    @pytest.mark.asyncio
    async def test_writes_with_none_blob_are_skipped(
        self, saver: AsyncOceanBaseCheckpointSaver
    ) -> None:
        mock_conn = AsyncMock()
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [
            ("task-1", "messages", "json", None),
        ]
        mock_conn.execute.return_value = mock_result

        result = await saver._load_pending_writes(mock_conn, "t1", "", "cp1")
        assert result == []


# ==============================================================================
# Engine creation
# ==============================================================================


class TestEngineCreation:
    """Test engine URL construction."""

    def test_default_connection_url(self) -> None:
        with patch(
            "langchain_oceanbase.async_checkpointer.create_async_engine"
        ) as mock_create:
            mock_create.return_value = MagicMock()
            AsyncOceanBaseCheckpointSaver(
                connection_args={
                    "host": "127.0.0.1",
                    "port": "2881",
                    "user": "root@test",
                    "password": "pass",
                    "db_name": "mydb",
                }
            )
            mock_create.assert_called_once()
            url = mock_create.call_args[0][0]
            assert "mysql+aiomysql://" in url
            assert "127.0.0.1:2881" in url
            assert "mydb" in url

    def test_engine_kwargs_passed_through(self) -> None:
        with patch(
            "langchain_oceanbase.async_checkpointer.create_async_engine"
        ) as mock_create:
            mock_create.return_value = MagicMock()
            AsyncOceanBaseCheckpointSaver(
                connection_args={
                    "host": "localhost",
                    "port": "2881",
                    "user": "root@test",
                    "password": "",
                    "db_name": "test",
                },
                engine_kwargs={"pool_size": 10, "max_overflow": 20},
            )
            kwargs = mock_create.call_args[1]
            assert kwargs["pool_size"] == 10
            assert kwargs["max_overflow"] == 20

    def test_uses_default_connection_when_none_provided(self) -> None:
        with patch(
            "langchain_oceanbase.async_checkpointer.create_async_engine"
        ) as mock_create:
            mock_create.return_value = MagicMock()
            AsyncOceanBaseCheckpointSaver(connection_args=None)
            url = mock_create.call_args[0][0]
            assert "mysql+aiomysql://" in url
            assert "localhost" in url

    def test_connection_error_wraps_exception(self) -> None:
        with patch(
            "langchain_oceanbase.async_checkpointer.create_async_engine"
        ) as mock_create:
            mock_create.side_effect = Exception("connection refused")
            from langchain_oceanbase.exceptions import OceanBaseConnectionError

            with pytest.raises(OceanBaseConnectionError, match="connection refused"):
                AsyncOceanBaseCheckpointSaver(
                    connection_args={
                        "host": "bad-host",
                        "port": "9999",
                        "user": "root",
                        "password": "",
                        "db_name": "test",
                    }
                )

    def test_password_with_special_chars_in_url(self) -> None:
        with patch(
            "langchain_oceanbase.async_checkpointer.create_async_engine"
        ) as mock_create:
            mock_create.return_value = MagicMock()
            AsyncOceanBaseCheckpointSaver(
                connection_args={
                    "host": "127.0.0.1",
                    "port": "2881",
                    "user": "root@test",
                    "password": "p@ss:w0rd",
                    "db_name": "test",
                }
            )
            url = mock_create.call_args[0][0]
            assert "p@ss:w0rd" in url


# ==============================================================================
# Version generation
# ==============================================================================


class TestGetNextVersion:
    """Test version generation."""

    def test_none_current(self, saver: AsyncOceanBaseCheckpointSaver) -> None:
        version = saver.get_next_version(None, None)
        assert version.startswith("00000000000000000000000000000001.")

    def test_increments_version(self, saver: AsyncOceanBaseCheckpointSaver) -> None:
        version = saver.get_next_version("00000000000000000000000000000005.123", None)
        assert version.startswith("00000000000000000000000000000006.")

    def test_int_current(self, saver: AsyncOceanBaseCheckpointSaver) -> None:
        version = saver.get_next_version(3, None)  # type: ignore[arg-type]
        assert version.startswith("00000000000000000000000000000004.")

    def test_versions_are_monotonically_increasing(
        self, saver: AsyncOceanBaseCheckpointSaver
    ) -> None:
        v1 = saver.get_next_version(None, None)
        v2 = saver.get_next_version(v1, None)
        v3 = saver.get_next_version(v2, None)
        assert v1 < v2 < v3

    def test_version_format(self, saver: AsyncOceanBaseCheckpointSaver) -> None:
        version = saver.get_next_version(None, None)
        # Format: "{next_v:032}.{next_h:016}" where next_h is a float
        first_dot = version.index(".")
        assert first_dot == 32


# ==============================================================================
# WHERE clause building
# ==============================================================================


class TestBuildWhereClause:
    """Test the WHERE clause builder."""

    def test_no_filters_returns_empty(
        self, saver: AsyncOceanBaseCheckpointSaver
    ) -> None:
        where, params = saver._build_where_clause(None, None, None)
        assert where == ""
        assert params == {}

    def test_thread_id_filter(self, saver: AsyncOceanBaseCheckpointSaver) -> None:
        config: RunnableConfig = {
            "configurable": {"thread_id": "t1", "checkpoint_ns": "ns1"}
        }
        where, params = saver._build_where_clause(config, None, None)
        assert "c.thread_id = :thread_id" in where
        assert "c.checkpoint_ns = :checkpoint_ns" in where
        assert params["thread_id"] == "t1"
        assert params["checkpoint_ns"] == "ns1"

    def test_metadata_string_filter_uses_json_unquote(
        self, saver: AsyncOceanBaseCheckpointSaver
    ) -> None:
        where, params = saver._build_where_clause(None, {"source": "loop"}, None)
        assert "JSON_UNQUOTE" in where
        assert params["filter_source"] == "loop"

    def test_metadata_non_string_filter_uses_json_extract(
        self, saver: AsyncOceanBaseCheckpointSaver
    ) -> None:
        where, params = saver._build_where_clause(None, {"step": 5}, None)
        assert "JSON_EXTRACT" in where
        assert "JSON_UNQUOTE" not in where
        assert params["filter_step"] == 5

    def test_before_filter(self, saver: AsyncOceanBaseCheckpointSaver) -> None:
        before: RunnableConfig = {
            "configurable": {
                "thread_id": "t1",
                "checkpoint_ns": "",
                "checkpoint_id": "cp-100",
            }
        }
        where, params = saver._build_where_clause(None, None, before)
        assert "c.checkpoint_id < :before_checkpoint_id" in where
        assert params["before_checkpoint_id"] == "cp-100"

    def test_combined_filters(self, saver: AsyncOceanBaseCheckpointSaver) -> None:
        config: RunnableConfig = {
            "configurable": {"thread_id": "t1", "checkpoint_ns": ""}
        }
        before: RunnableConfig = {
            "configurable": {
                "thread_id": "t1",
                "checkpoint_ns": "",
                "checkpoint_id": "cp-50",
            }
        }
        where, params = saver._build_where_clause(
            config, {"source": "loop"}, before
        )
        assert "c.thread_id = :thread_id" in where
        assert "JSON_UNQUOTE" in where
        assert "c.checkpoint_id < :before_checkpoint_id" in where

    def test_checkpoint_id_in_config(
        self, saver: AsyncOceanBaseCheckpointSaver
    ) -> None:
        config: RunnableConfig = {
            "configurable": {
                "thread_id": "t1",
                "checkpoint_ns": "",
                "checkpoint_id": "exact-cp",
            }
        }
        where, params = saver._build_where_clause(config, None, None)
        assert "c.checkpoint_id = :checkpoint_id" in where
        assert params["checkpoint_id"] == "exact-cp"


# ==============================================================================
# Row to checkpoint tuple conversion
# ==============================================================================


class TestRowToCheckpointTuple:
    """Test conversion of DB rows to CheckpointTuple."""

    def test_basic_conversion(self, saver: AsyncOceanBaseCheckpointSaver) -> None:
        row = (
            "thread-1",
            "ns",
            "cp-1",
            "parent-cp",
            "checkpoint",
            json.dumps({
                "v": 1,
                "id": "cp-1",
                "ts": "2024-01-01T00:00:00+00:00",
                "channel_values": {},
                "channel_versions": {},
                "versions_seen": {},
                "pending_sends": [],
            }),
            json.dumps({"source": "input", "step": 0}),
        )

        result = saver._row_to_checkpoint_tuple(row, {}, [])

        assert result.config["configurable"]["thread_id"] == "thread-1"
        assert result.config["configurable"]["checkpoint_ns"] == "ns"
        assert result.config["configurable"]["checkpoint_id"] == "cp-1"
        assert result.checkpoint["id"] == "cp-1"
        assert result.metadata["source"] == "input"
        assert result.parent_config is not None
        assert result.parent_config["configurable"]["checkpoint_id"] == "parent-cp"

    def test_no_parent_checkpoint(self, saver: AsyncOceanBaseCheckpointSaver) -> None:
        row = (
            "thread-1",
            "",
            "cp-1",
            None,  # no parent
            "checkpoint",
            json.dumps({
                "v": 1,
                "id": "cp-1",
                "ts": "",
                "channel_values": {},
                "channel_versions": {},
                "versions_seen": {},
                "pending_sends": [],
            }),
            None,  # no metadata
        )

        result = saver._row_to_checkpoint_tuple(row, {}, [])

        assert result.parent_config is None
        assert result.metadata == {}

    def test_channel_values_merged(self, saver: AsyncOceanBaseCheckpointSaver) -> None:
        row = (
            "t1",
            "",
            "cp-1",
            None,
            "checkpoint",
            json.dumps({
                "v": 1,
                "id": "cp-1",
                "ts": "",
                "channel_values": {"inline": "value"},
                "channel_versions": {},
                "versions_seen": {},
                "pending_sends": [],
            }),
            "{}",
        )

        # blob channel_values override/merge with inline
        blob_values = {"blob_channel": [1, 2, 3]}
        result = saver._row_to_checkpoint_tuple(row, blob_values, [])

        assert result.checkpoint["channel_values"]["inline"] == "value"
        assert result.checkpoint["channel_values"]["blob_channel"] == [1, 2, 3]

    def test_pending_writes_attached(
        self, saver: AsyncOceanBaseCheckpointSaver
    ) -> None:
        row = (
            "t1",
            "",
            "cp-1",
            None,
            "checkpoint",
            json.dumps({
                "v": 1,
                "id": "cp-1",
                "ts": "",
                "channel_values": {},
                "channel_versions": {},
                "versions_seen": {},
                "pending_sends": [],
            }),
            "{}",
        )

        writes = [("task-1", "messages", {"content": "hi"})]
        result = saver._row_to_checkpoint_tuple(row, {}, writes)

        assert result.pending_writes == writes

    def test_dict_checkpoint_data_not_reparsed(
        self, saver: AsyncOceanBaseCheckpointSaver
    ) -> None:
        """If checkpoint_data is already a dict (not a string), don't error."""
        checkpoint_dict = {
            "v": 1,
            "id": "cp-1",
            "ts": "",
            "channel_values": {"x": 1},
            "channel_versions": {},
            "versions_seen": {},
            "pending_sends": [],
        }
        row = ("t1", "", "cp-1", None, "checkpoint", checkpoint_dict, {})

        result = saver._row_to_checkpoint_tuple(row, {}, [])
        assert result.checkpoint["channel_values"]["x"] == 1


# ==============================================================================
# Lock behavior
# ==============================================================================


class TestAsyncLock:
    """Test that the saver uses asyncio.Lock, not threading.Lock."""

    def test_lock_is_asyncio_lock(self, saver: AsyncOceanBaseCheckpointSaver) -> None:
        assert isinstance(saver.lock, asyncio.Lock)

    def test_no_threading_lock(self, saver: AsyncOceanBaseCheckpointSaver) -> None:
        import threading

        assert not isinstance(saver.lock, type(threading.Lock()))

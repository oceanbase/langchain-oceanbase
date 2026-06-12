"""Async OceanBase checkpoint saver for LangGraph.

This module provides a truly asynchronous checkpointer implementation using
SQLAlchemy AsyncEngine with aiomysql driver. It does not block the event loop.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional, cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
    get_serializable_checkpoint_metadata,
)
from langgraph.checkpoint.serde.base import SerializerProtocol
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, create_async_engine

from langchain_oceanbase.checkpointer import (
    MIGRATIONS,
    REQUIRED_INDEX_MIGRATIONS,
    SELECT_SQL,
    UPSERT_CHECKPOINT_BLOBS_SQL,
    OceanBaseCheckpointSaver,
)
from langchain_oceanbase.exceptions import OceanBaseConnectionError

logger = logging.getLogger(__name__)


class AsyncOceanBaseCheckpointSaver(BaseCheckpointSaver[str]):
    """Async checkpointer that stores checkpoints in an OceanBase database.

    This checkpointer uses SQLAlchemy AsyncEngine with aiomysql to provide
    truly non-blocking async database operations. It does not block the event
    loop during database I/O.

    Setup:
        Install ``langchain-oceanbase`` with async support and deploy OceanBase.

        .. code-block:: bash

            pip install -U langchain-oceanbase aiomysql

    Example:
        .. code-block:: python

            import asyncio
            from langchain_oceanbase import AsyncOceanBaseCheckpointSaver
            from langgraph.graph import StateGraph

            async def main():
                connection_args = {
                    "host": "127.0.0.1",
                    "port": "2881",
                    "user": "root@test",
                    "password": "",
                    "db_name": "test",
                }
                checkpointer = AsyncOceanBaseCheckpointSaver(
                    connection_args=connection_args
                )
                await checkpointer.setup()

                graph = StateGraph(...)
                app = graph.compile(checkpointer=checkpointer)

                config = {"configurable": {"thread_id": "my-thread"}}
                result = await app.ainvoke({"messages": [...]}, config)

            asyncio.run(main())
    """

    engine: AsyncEngine
    lock: asyncio.Lock

    def __init__(
        self,
        connection_args: Optional[Dict[str, Any]] = None,
        *,
        serde: Optional[SerializerProtocol] = None,
        engine_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the async OceanBase checkpoint saver.

        Args:
            connection_args: Connection parameters for OceanBase. Should include:
                - host: OceanBase server host (default: "localhost")
                - port: OceanBase server port (default: "2881")
                - user: Database username (default: "root@test")
                - password: Database password (default: "")
                - db_name: Database name (default: "test")
            serde: Optional serializer for encoding/decoding checkpoints.
            engine_kwargs: Additional keyword arguments passed to
                create_async_engine (e.g. pool_size, max_overflow).
        """
        super().__init__(serde=serde)
        from langchain_oceanbase.vectorstores import DEFAULT_OCEANBASE_CONNECTION

        self.connection_args: Dict[str, Any] = (
            connection_args
            if connection_args is not None
            else DEFAULT_OCEANBASE_CONNECTION
        )
        self.lock = asyncio.Lock()
        self._engine_kwargs = engine_kwargs or {}
        self._create_engine()

    def _create_engine(self) -> None:
        """Create the SQLAlchemy async engine."""
        host = self.connection_args.get("host", "localhost")
        port = self.connection_args.get("port", "2881")
        user = self.connection_args.get("user", "root@test")
        password = self.connection_args.get("password", "")
        db_name = self.connection_args.get("db_name", "test")

        url = f"mysql+aiomysql://{user}:{password}@{host}:{port}/{db_name}"

        try:
            self.engine = create_async_engine(url, **self._engine_kwargs)
        except Exception as e:
            raise OceanBaseConnectionError(
                f"Failed to create async engine for OceanBase: {e}",
                host=host,
                port=str(port),
            ) from e

    @asynccontextmanager
    async def _conn(self) -> AsyncIterator[AsyncConnection]:
        """Acquire an async connection from the engine pool."""
        async with self.engine.connect() as conn:
            yield conn

    async def setup(self) -> None:
        """Set up the checkpoint database asynchronously.

        Creates necessary tables and runs migrations. Must be called before
        using the checkpointer.
        """
        async with self._conn() as conn:
            await conn.execute(text(MIGRATIONS[0]))
            await conn.commit()

            result = await conn.execute(
                text("SELECT COALESCE(MAX(v), -1) FROM checkpoint_migrations")
            )
            row = result.fetchone()
            version = -1 if row is None else row[0]

            for v, migration in enumerate(MIGRATIONS[version + 1 :], start=version + 1):
                try:
                    await conn.execute(text(migration))
                except Exception as exc:
                    await conn.rollback()
                    if v < 4 or not OceanBaseCheckpointSaver._is_duplicate_index_error(
                        exc
                    ):
                        raise
                await conn.execute(
                    text("INSERT INTO checkpoint_migrations (v) VALUES (:v)"),
                    {"v": v},
                )
                await conn.commit()

            await self._ensure_required_indexes(conn)
            await conn.commit()

    async def _ensure_required_indexes(self, conn: AsyncConnection) -> None:
        """Create any missing checkpoint indexes idempotently."""
        for table_name, index_name, migration_sql in REQUIRED_INDEX_MIGRATIONS:
            existing = await self._get_index_names(conn, table_name)
            if index_name in existing:
                continue
            try:
                await conn.execute(text(migration_sql))
            except Exception as exc:
                if not OceanBaseCheckpointSaver._is_duplicate_index_error(exc):
                    raise

    async def _get_index_names(
        self, conn: AsyncConnection, table_name: str
    ) -> set[str]:
        """Return existing index names for a table."""
        result = await conn.execute(text(f"SHOW INDEX FROM `{table_name}`"))
        rows = result.fetchall()
        return {row[2] for row in rows}

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Get a checkpoint tuple from the database asynchronously.

        Args:
            config: Configuration specifying which checkpoint to retrieve.

        Returns:
            The retrieved checkpoint tuple, or None if not found.
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = get_checkpoint_id(config)
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")

        async with self._conn() as conn:
            if checkpoint_id:
                query = text(
                    SELECT_SQL
                    + "WHERE c.thread_id = :thread_id "
                    + "AND c.checkpoint_ns = :checkpoint_ns "
                    + "AND c.checkpoint_id = :checkpoint_id"
                )
                result = await conn.execute(
                    query,
                    {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": checkpoint_id,
                    },
                )
            else:
                query = text(
                    SELECT_SQL
                    + "WHERE c.thread_id = :thread_id "
                    + "AND c.checkpoint_ns = :checkpoint_ns "
                    + "ORDER BY c.checkpoint_id DESC LIMIT 1"
                )
                result = await conn.execute(
                    query,
                    {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns},
                )

            row = result.fetchone()
            if row is None:
                return None

            channel_values = await self._load_channel_values(
                conn, thread_id, checkpoint_ns, row[5]
            )
            pending_writes = await self._load_pending_writes(
                conn, thread_id, checkpoint_ns, row[2]
            )

            return self._row_to_checkpoint_tuple(row, channel_values, pending_writes)

    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """List checkpoints from the database asynchronously.

        Results are materialized before yielding to avoid holding connections.

        Args:
            config: Configuration for filtering checkpoints.
            filter: Additional metadata filters.
            before: Only return checkpoints before this one.
            limit: Maximum number of checkpoints to return.

        Yields:
            Checkpoint tuples ordered by checkpoint_id descending.
        """
        where_clause, params = self._build_where_clause(config, filter, before)
        query = SELECT_SQL + where_clause + " ORDER BY c.checkpoint_id DESC"
        if limit is not None:
            query += f" LIMIT {int(limit)}"

        tuples: list[CheckpointTuple] = []

        async with self._conn() as conn:
            result = await conn.execute(text(query), params)
            rows = result.fetchall()

            for row in rows:
                thread_id = row[0]
                checkpoint_ns = row[1]
                checkpoint_id = row[2]
                checkpoint_data = row[5]

                channel_values = await self._load_channel_values(
                    conn, thread_id, checkpoint_ns, checkpoint_data
                )
                pending_writes = await self._load_pending_writes(
                    conn, thread_id, checkpoint_ns, checkpoint_id
                )
                tuples.append(
                    self._row_to_checkpoint_tuple(row, channel_values, pending_writes)
                )

        for t in tuples:
            yield t

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save a checkpoint to the database asynchronously.

        Args:
            config: The config to associate with the checkpoint.
            checkpoint: The checkpoint to save.
            metadata: Additional metadata to save.
            new_versions: New channel versions as of this write.

        Returns:
            Updated configuration after storing the checkpoint.
        """
        configurable = config["configurable"].copy()
        thread_id = configurable.pop("thread_id")
        checkpoint_ns = configurable.pop("checkpoint_ns", "")
        parent_checkpoint_id = configurable.pop("checkpoint_id", None)

        checkpoint_copy = checkpoint.copy()
        checkpoint_copy["channel_values"] = checkpoint_copy["channel_values"].copy()

        blob_values: Dict[str, Any] = {}
        for k, v in checkpoint["channel_values"].items():
            if v is not None and not isinstance(v, (str, int, float, bool)):
                blob_values[k] = checkpoint_copy["channel_values"].pop(k)

        next_config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            }
        }

        async with self._conn() as conn:
            blob_versions = {k: v for k, v in new_versions.items() if k in blob_values}
            for channel, version in blob_versions.items():
                value = blob_values[channel]
                type_str, blob = self.serde.dumps_typed(value)
                await conn.execute(
                    text(UPSERT_CHECKPOINT_BLOBS_SQL),
                    {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "channel": channel,
                        "version": str(version),
                        "type": type_str,
                        "blob": OceanBaseCheckpointSaver._encode_storage_blob(blob),
                    },
                )

            storage_metadata = dict(
                get_serializable_checkpoint_metadata(config, metadata)
            )

            await conn.execute(
                text(
                    """
                    INSERT INTO checkpoints
                    (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id,
                     `type`, checkpoint, metadata)
                    VALUES (:thread_id, :checkpoint_ns, :checkpoint_id,
                            :parent_checkpoint_id, :type, :checkpoint, :metadata)
                    ON DUPLICATE KEY UPDATE
                        checkpoint = VALUES(checkpoint),
                        metadata = VALUES(metadata)
                    """
                ),
                {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint["id"],
                    "parent_checkpoint_id": parent_checkpoint_id,
                    "type": "checkpoint",
                    "checkpoint": json.dumps(checkpoint_copy),
                    "metadata": json.dumps(storage_metadata),
                },
            )
            await conn.commit()

        return next_config

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Store intermediate writes linked to a checkpoint asynchronously.

        Args:
            config: Configuration of the related checkpoint.
            writes: List of writes to store, each as (channel, value) tuple.
            task_id: Identifier for the task creating the writes.
            task_path: Path of the task creating the writes.
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"]["checkpoint_id"]

        async with self._conn() as conn:
            for idx, (channel, value) in enumerate(writes):
                type_str, blob = self.serde.dumps_typed(value)
                write_idx = WRITES_IDX_MAP.get(channel, idx)

                if channel in WRITES_IDX_MAP:
                    sql = """
                        INSERT INTO checkpoint_writes
                        (thread_id, checkpoint_ns, checkpoint_id, task_id,
                         task_path, idx, channel, `type`, `blob`)
                        VALUES (:thread_id, :checkpoint_ns, :checkpoint_id,
                                :task_id, :task_path, :idx, :channel, :type, :blob)
                        ON DUPLICATE KEY UPDATE
                            channel = VALUES(channel),
                            `type` = VALUES(`type`),
                            `blob` = VALUES(`blob`)
                    """
                else:
                    sql = """
                        INSERT IGNORE INTO checkpoint_writes
                        (thread_id, checkpoint_ns, checkpoint_id, task_id,
                         task_path, idx, channel, `type`, `blob`)
                        VALUES (:thread_id, :checkpoint_ns, :checkpoint_id,
                                :task_id, :task_path, :idx, :channel, :type, :blob)
                    """

                await conn.execute(
                    text(sql),
                    {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": checkpoint_id,
                        "task_id": task_id,
                        "task_path": task_path,
                        "idx": write_idx,
                        "channel": channel,
                        "type": type_str,
                        "blob": OceanBaseCheckpointSaver._encode_storage_blob(blob),
                    },
                )
            await conn.commit()

    async def adelete_thread(self, thread_id: str) -> None:
        """Delete all checkpoints and writes for a thread asynchronously.

        Args:
            thread_id: The thread ID to delete.
        """
        async with self._conn() as conn:
            await conn.execute(
                text("DELETE FROM checkpoints WHERE thread_id = :thread_id"),
                {"thread_id": thread_id},
            )
            await conn.execute(
                text("DELETE FROM checkpoint_blobs WHERE thread_id = :thread_id"),
                {"thread_id": thread_id},
            )
            await conn.execute(
                text("DELETE FROM checkpoint_writes WHERE thread_id = :thread_id"),
                {"thread_id": thread_id},
            )
            await conn.commit()

    async def aprune(
        self,
        thread_ids: Sequence[str],
        *,
        strategy: str = "keep_latest",
    ) -> None:
        """Prune checkpoints for the given threads asynchronously."""
        if not thread_ids:
            return

        if strategy == "delete":
            for thread_id in thread_ids:
                await self.adelete_thread(thread_id)
            return

        if strategy != "keep_latest":
            raise ValueError(f"Unsupported prune strategy: {strategy}")

        async with self._conn() as conn:
            for thread_id in thread_ids:
                result = await conn.execute(
                    text(
                        """
                        SELECT checkpoint_ns, MAX(checkpoint_id) AS checkpoint_id
                        FROM checkpoints
                        WHERE thread_id = :thread_id
                        GROUP BY checkpoint_ns
                        """
                    ),
                    {"thread_id": thread_id},
                )
                keepers = result.fetchall()
                if not keepers:
                    continue

                refs_by_ns: dict[str, set[tuple[str, str]]] = {}
                for checkpoint_ns, checkpoint_id in keepers:
                    cp_result = await conn.execute(
                        text(
                            """
                            SELECT checkpoint
                            FROM checkpoints
                            WHERE thread_id = :thread_id
                              AND checkpoint_ns = :checkpoint_ns
                              AND checkpoint_id = :checkpoint_id
                            """
                        ),
                        {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": checkpoint_id,
                        },
                    )
                    checkpoint_row = cp_result.fetchone()
                    if checkpoint_row is None:
                        refs_by_ns[checkpoint_ns] = set()
                        continue

                    checkpoint_data = checkpoint_row[0]
                    if isinstance(checkpoint_data, str):
                        checkpoint_data = json.loads(checkpoint_data)
                    channel_versions = checkpoint_data.get("channel_versions", {})
                    refs_by_ns[checkpoint_ns] = {
                        (channel, str(version))
                        for channel, version in channel_versions.items()
                    }

                    await conn.execute(
                        text(
                            """
                            UPDATE checkpoints
                            SET parent_checkpoint_id = NULL
                            WHERE thread_id = :thread_id
                              AND checkpoint_ns = :checkpoint_ns
                              AND checkpoint_id = :checkpoint_id
                            """
                        ),
                        {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": checkpoint_id,
                        },
                    )
                    await conn.execute(
                        text(
                            """
                            DELETE FROM checkpoints
                            WHERE thread_id = :thread_id
                              AND checkpoint_ns = :checkpoint_ns
                              AND checkpoint_id <> :checkpoint_id
                            """
                        ),
                        {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": checkpoint_id,
                        },
                    )
                    await conn.execute(
                        text(
                            """
                            DELETE FROM checkpoint_writes
                            WHERE thread_id = :thread_id
                              AND checkpoint_ns = :checkpoint_ns
                              AND checkpoint_id <> :checkpoint_id
                            """
                        ),
                        {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": checkpoint_id,
                        },
                    )

                for checkpoint_ns, refs in refs_by_ns.items():
                    if not refs:
                        await conn.execute(
                            text(
                                """
                                DELETE FROM checkpoint_blobs
                                WHERE thread_id = :thread_id
                                  AND checkpoint_ns = :checkpoint_ns
                                """
                            ),
                            {
                                "thread_id": thread_id,
                                "checkpoint_ns": checkpoint_ns,
                            },
                        )
                        continue

                    params: dict[str, Any] = {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                    }
                    keep_conditions = []
                    for idx, (channel, version) in enumerate(sorted(refs)):
                        params[f"channel_{idx}"] = channel
                        params[f"version_{idx}"] = version
                        keep_conditions.append(
                            f"(channel = :channel_{idx} AND version = :version_{idx})"
                        )

                    await conn.execute(
                        text(
                            f"""
                            DELETE FROM checkpoint_blobs
                            WHERE thread_id = :thread_id
                              AND checkpoint_ns = :checkpoint_ns
                              AND NOT ({" OR ".join(keep_conditions)})
                            """
                        ),
                        params,
                    )

            await conn.commit()

    # --- Sync methods: not supported ---

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Not supported. Use aget_tuple() instead."""
        raise NotImplementedError(
            "AsyncOceanBaseCheckpointSaver does not support synchronous operations. "
            "Use aget_tuple() or switch to OceanBaseCheckpointSaver for sync usage."
        )

    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """Not supported. Use alist() instead."""
        raise NotImplementedError(
            "AsyncOceanBaseCheckpointSaver does not support synchronous operations. "
            "Use alist() or switch to OceanBaseCheckpointSaver for sync usage."
        )

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Not supported. Use aput() instead."""
        raise NotImplementedError(
            "AsyncOceanBaseCheckpointSaver does not support synchronous operations. "
            "Use aput() or switch to OceanBaseCheckpointSaver for sync usage."
        )

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Not supported. Use aput_writes() instead."""
        raise NotImplementedError(
            "AsyncOceanBaseCheckpointSaver does not support synchronous operations. "
            "Use aput_writes() or switch to OceanBaseCheckpointSaver for sync usage."
        )

    # --- Helper methods ---

    async def _load_channel_values(
        self,
        conn: AsyncConnection,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_data: Any,
    ) -> Dict[str, Any]:
        """Load channel values using a single batch query (fixes N+1)."""
        if isinstance(checkpoint_data, str):
            checkpoint_data = json.loads(checkpoint_data)

        channel_versions = checkpoint_data.get("channel_versions", {})
        if not channel_versions:
            return {}

        params: Dict[str, Any] = {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
        }
        conditions = []
        for idx, (channel, version) in enumerate(channel_versions.items()):
            params[f"channel_{idx}"] = channel
            params[f"version_{idx}"] = str(version)
            conditions.append(
                f"(channel = :channel_{idx} AND version = :version_{idx})"
            )

        query = text(
            "SELECT channel, `type`, `blob` FROM checkpoint_blobs "
            "WHERE thread_id = :thread_id "
            "AND checkpoint_ns = :checkpoint_ns "
            f"AND ({' OR '.join(conditions)})"
        )
        result = await conn.execute(query, params)
        rows = result.fetchall()

        channel_values: Dict[str, Any] = {}
        for row in rows:
            channel, type_str, blob = row[0], row[1], row[2]
            if type_str != "empty":
                decoded_blob = OceanBaseCheckpointSaver._decode_storage_blob(blob)
                if decoded_blob is not None:
                    channel_values[channel] = self.serde.loads_typed(
                        (type_str, decoded_blob)
                    )

        return channel_values

    async def _load_pending_writes(
        self,
        conn: AsyncConnection,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
    ) -> list[tuple[str, str, Any]]:
        """Load pending writes from the checkpoint_writes table."""
        query = text(
            "SELECT task_id, channel, `type`, `blob` FROM checkpoint_writes "
            "WHERE thread_id = :thread_id "
            "AND checkpoint_ns = :checkpoint_ns "
            "AND checkpoint_id = :checkpoint_id "
            "ORDER BY task_id, idx"
        )
        result = await conn.execute(
            query,
            {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            },
        )
        rows = result.fetchall()

        writes = []
        for row in rows:
            task_id, channel, type_str, blob = row
            decoded_blob = OceanBaseCheckpointSaver._decode_storage_blob(blob)
            if decoded_blob is not None:
                value = self.serde.loads_typed((type_str, decoded_blob))
                writes.append((task_id, channel, value))

        return writes

    def _row_to_checkpoint_tuple(
        self,
        row: Any,
        channel_values: Dict[str, Any],
        pending_writes: list[tuple[str, str, Any]],
    ) -> CheckpointTuple:
        """Convert a database row to a CheckpointTuple."""
        (
            thread_id,
            checkpoint_ns,
            checkpoint_id,
            parent_checkpoint_id,
            type_str,
            checkpoint_data,
            metadata,
        ) = row

        if isinstance(checkpoint_data, str):
            checkpoint_data = json.loads(checkpoint_data)

        if isinstance(metadata, str):
            metadata = json.loads(metadata) if metadata else {}
        elif metadata is None:
            metadata = {}

        checkpoint_channel_values = checkpoint_data.get("channel_values", {})
        checkpoint_channel_values.update(channel_values)

        checkpoint: Checkpoint = {
            "v": checkpoint_data.get("v", 1),
            "id": checkpoint_data.get("id", checkpoint_id),
            "ts": checkpoint_data.get("ts", ""),
            "channel_values": checkpoint_channel_values,
            "channel_versions": checkpoint_data.get("channel_versions", {}),
            "versions_seen": checkpoint_data.get("versions_seen", {}),
            "pending_sends": checkpoint_data.get("pending_sends", []),  # type: ignore[typeddict-unknown-key]
            "updated_channels": checkpoint_data.get("updated_channels"),
        }

        return CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint_id,
                }
            },
            checkpoint=checkpoint,
            metadata=cast(CheckpointMetadata, metadata),
            parent_config=(
                {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": parent_checkpoint_id,
                    }
                }
                if parent_checkpoint_id
                else None
            ),
            pending_writes=pending_writes,
        )

    @staticmethod
    def _build_where_clause(
        config: Optional[RunnableConfig],
        filter: Optional[Dict[str, Any]],
        before: Optional[RunnableConfig],
    ) -> tuple[str, Dict[str, Any]]:
        """Build WHERE clause for list queries."""
        conditions = []
        params: Dict[str, Any] = {}

        if config:
            thread_id = config["configurable"]["thread_id"]
            conditions.append("c.thread_id = :thread_id")
            params["thread_id"] = thread_id

            checkpoint_ns = config["configurable"].get("checkpoint_ns")
            if checkpoint_ns is not None:
                conditions.append("c.checkpoint_ns = :checkpoint_ns")
                params["checkpoint_ns"] = checkpoint_ns

            if checkpoint_id := get_checkpoint_id(config):
                conditions.append("c.checkpoint_id = :checkpoint_id")
                params["checkpoint_id"] = checkpoint_id

        if filter:
            for key, value in filter.items():
                param_name = f"filter_{key}"
                if isinstance(value, str):
                    conditions.append(
                        f"JSON_UNQUOTE(JSON_EXTRACT(c.metadata, '$.{key}')) = :{param_name}"
                    )
                else:
                    conditions.append(
                        f"JSON_EXTRACT(c.metadata, '$.{key}') = :{param_name}"
                    )
                params[param_name] = (
                    json.dumps(value)
                    if not isinstance(value, (str, int, float, bool))
                    else value
                )

        if before:
            before_checkpoint_id = get_checkpoint_id(before)
            if before_checkpoint_id:
                conditions.append("c.checkpoint_id < :before_checkpoint_id")
                params["before_checkpoint_id"] = before_checkpoint_id

        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
        return where_clause, params

    def get_next_version(self, current: str | None, channel: None) -> str:
        """Generate the next version ID for a channel."""
        if current is None:
            current_v = 0
        elif isinstance(current, int):
            current_v = current
        else:
            current_v = int(current.split(".")[0])

        next_v = current_v + 1
        next_h = random.random()
        return f"{next_v:032}.{next_h:016}"


__all__ = ["AsyncOceanBaseCheckpointSaver"]

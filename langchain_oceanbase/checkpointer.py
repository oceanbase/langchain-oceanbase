"""OceanBase checkpoint saver for LangGraph.

This module provides a checkpointer implementation for LangGraph that uses
OceanBase as the persistence backend. It allows LangGraph agents to persist
their state within and across multiple interactions.
"""

from __future__ import annotations

import base64
import json
import logging
import random
import threading
from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import contextmanager
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
from pyobvector import ObVecClient  # type: ignore[import-untyped]
from sqlalchemy import text
from sqlalchemy.engine import Connection
from sqlalchemy.exc import ResourceClosedError

from langchain_oceanbase.exceptions import OceanBaseConnectionError
from langchain_oceanbase.vectorstores import DEFAULT_OCEANBASE_CONNECTION

logger = logging.getLogger(__name__)

MetadataInput = Dict[str, Any] | None
_BLOB_PREFIX = b"base64:"

# MySQL-compatible migrations for OceanBase
MIGRATIONS = [
    # Migration 0: Create migrations tracking table
    """CREATE TABLE IF NOT EXISTS checkpoint_migrations (
        v INT PRIMARY KEY
    );""",
    # Migration 1: Create checkpoints table
    """CREATE TABLE IF NOT EXISTS checkpoints (
        thread_id VARCHAR(255) NOT NULL,
        checkpoint_ns VARCHAR(255) NOT NULL DEFAULT '',
        checkpoint_id VARCHAR(255) NOT NULL,
        parent_checkpoint_id VARCHAR(255),
        `type` VARCHAR(255),
        checkpoint JSON NOT NULL,
        metadata JSON,
        PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
    );""",
    # Migration 2: Create checkpoint_blobs table
    """CREATE TABLE IF NOT EXISTS checkpoint_blobs (
        thread_id VARCHAR(255) NOT NULL,
        checkpoint_ns VARCHAR(255) NOT NULL DEFAULT '',
        channel VARCHAR(255) NOT NULL,
        version VARCHAR(255) NOT NULL,
        `type` VARCHAR(255) NOT NULL,
        `blob` LONGBLOB,
        PRIMARY KEY (thread_id, checkpoint_ns, channel, version)
    );""",
    # Migration 3: Create checkpoint_writes table
    """CREATE TABLE IF NOT EXISTS checkpoint_writes (
        thread_id VARCHAR(255) NOT NULL,
        checkpoint_ns VARCHAR(255) NOT NULL DEFAULT '',
        checkpoint_id VARCHAR(255) NOT NULL,
        task_id VARCHAR(255) NOT NULL,
        idx INT NOT NULL,
        channel VARCHAR(255) NOT NULL,
        `type` VARCHAR(255),
        `blob` LONGBLOB NOT NULL,
        task_path VARCHAR(255) NOT NULL DEFAULT '',
        PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
    );""",
    # Migration 4: Add indexes for better query performance
    """CREATE INDEX IF NOT EXISTS idx_checkpoints_thread_id
       ON checkpoints(thread_id);""",
    # Migration 5: Add index on checkpoint_blobs
    """CREATE INDEX IF NOT EXISTS idx_checkpoint_blobs_thread_id
       ON checkpoint_blobs(thread_id);""",
    # Migration 6: Add index on checkpoint_writes
    """CREATE INDEX IF NOT EXISTS idx_checkpoint_writes_thread_id
       ON checkpoint_writes(thread_id);""",
]

# SQL for selecting checkpoints with channel values and pending writes
SELECT_SQL = """
SELECT
    c.thread_id,
    c.checkpoint_ns,
    c.checkpoint_id,
    c.parent_checkpoint_id,
    c.type,
    c.checkpoint,
    c.metadata
FROM checkpoints c
"""

# SQL for upserting checkpoint blobs (MySQL ON DUPLICATE KEY UPDATE syntax)
UPSERT_CHECKPOINT_BLOBS_SQL = """
INSERT INTO checkpoint_blobs (thread_id, checkpoint_ns, channel, version, `type`, `blob`)
VALUES (:thread_id, :checkpoint_ns, :channel, :version, :type, :blob)
ON DUPLICATE KEY UPDATE
    `type` = VALUES(`type`),
    `blob` = VALUES(`blob`)
"""

# SQL for upserting checkpoints
UPSERT_CHECKPOINTS_SQL = """
INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, `type`, checkpoint, metadata)
VALUES (%s, %s, %s, %s, %s, %s, %s)
ON DUPLICATE KEY UPDATE
    checkpoint = VALUES(checkpoint),
    metadata = VALUES(metadata)
"""

# SQL for upserting checkpoint writes
UPSERT_CHECKPOINT_WRITES_SQL = """
INSERT INTO checkpoint_writes (thread_id, checkpoint_ns, checkpoint_id, task_id, task_path, idx, channel, `type`, `blob`)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
ON DUPLICATE KEY UPDATE
    channel = VALUES(channel),
    `type` = VALUES(`type`),
    `blob` = VALUES(`blob`)
"""

# SQL for inserting checkpoint writes (ignore on conflict)
INSERT_CHECKPOINT_WRITES_SQL = """
INSERT IGNORE INTO checkpoint_writes (thread_id, checkpoint_ns, checkpoint_id, task_id, task_path, idx, channel, `type`, `blob`)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
"""


class OceanBaseCheckpointSaver(BaseCheckpointSaver[str]):
    """Checkpointer that stores checkpoints in an OceanBase database.

    This checkpointer allows LangGraph agents to persist their state in OceanBase,
    enabling features like conversation memory, time travel, and fault tolerance.

    Setup:
        Install ``langchain-oceanbase`` and deploy a standalone OceanBase server.

        .. code-block:: bash

            pip install -U langchain-oceanbase
            docker run --name=oceanbase -e MODE=mini -e OB_SERVER_IP=127.0.0.1 \\
                -p 2881:2881 -d oceanbase/oceanbase-ce:latest

    Example:
        .. code-block:: python

            from langchain_oceanbase import OceanBaseCheckpointSaver
            from langgraph.graph import StateGraph

            # Create checkpointer
            connection_args = {
                "host": "127.0.0.1",
                "port": "2881",
                "user": "root@test",
                "password": "",
                "db_name": "test",
            }
            checkpointer = OceanBaseCheckpointSaver(connection_args=connection_args)
            checkpointer.setup()

            # Use with LangGraph
            graph = StateGraph(...)
            app = graph.compile(checkpointer=checkpointer)

            # Run with thread_id for persistence
            config = {"configurable": {"thread_id": "my-thread"}}
            result = app.invoke({"messages": [...]}, config)

    Attributes:
        connection_args: Dictionary containing OceanBase connection parameters.
        obvector: The OceanBase vector client used for database operations.
    """

    lock: threading.Lock

    def __init__(
        self,
        connection_args: Optional[Dict[str, Any]] = None,
        *,
        serde: Optional[SerializerProtocol] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the OceanBase checkpoint saver.

        Args:
            connection_args: Connection parameters for OceanBase. Should include:
                - host: OceanBase server host (default: "localhost")
                - port: OceanBase server port (default: "2881")
                - user: Database username (default: "root@test")
                - password: Database password (default: "")
                - db_name: Database name (default: "test")
            serde: Optional serializer for encoding/decoding checkpoints.
            **kwargs: Additional arguments passed to ObVecClient.
        """
        super().__init__(serde=serde)
        self.connection_args: Dict[str, Any] = (
            connection_args
            if connection_args is not None
            else DEFAULT_OCEANBASE_CONNECTION
        )
        self.lock = threading.Lock()
        self._kwargs = kwargs
        self._create_client(**kwargs)

    def _create_client(self, **kwargs: Any) -> None:
        """Create and initialize the OceanBase vector client.

        Args:
            **kwargs: Additional arguments passed to ObVecClient constructor.

        Raises:
            OceanBaseConnectionError: If connection to OceanBase fails.
        """
        host = self.connection_args.get("host", "localhost")
        port = self.connection_args.get("port", "2881")
        user = self.connection_args.get("user", "root@test")
        password = self.connection_args.get("password", "")
        db_name = self.connection_args.get("db_name", "test")

        try:
            self.obvector: ObVecClient = ObVecClient(
                uri=f"{host}:{port}",
                user=user,
                password=password,
                db_name=db_name,
                **kwargs,
            )
        except Exception as e:
            error_msg = str(e).lower()
            if (
                "connect" in error_msg
                or "refused" in error_msg
                or "timeout" in error_msg
            ):
                raise OceanBaseConnectionError(
                    f"Failed to connect to OceanBase: {e}",
                    host=host,
                    port=port,
                ) from e
            # Re-raise other exceptions as-is
            logger.error(f"Failed to create OceanBase client: {e}")
            raise

    @contextmanager
    def _cursor(self) -> Iterator[Connection]:
        """Create a database connection as a context manager.

        Yields:
            A SQLAlchemy connection object for executing SQL statements.
        """
        with self.lock:
            with self.obvector.engine.connect() as conn:
                yield conn

    @staticmethod
    def _result_fetchone_or_none(result: Any) -> Any:
        """Fetch one row, treating closed empty SELECT results as no rows."""
        try:
            return result.fetchone()
        except ResourceClosedError:
            return None

    @staticmethod
    def _result_fetchall(result: Any) -> list[Any]:
        """Fetch all rows, treating closed empty SELECT results as empty."""
        try:
            return list(result.fetchall())
        except ResourceClosedError:
            return []

    @staticmethod
    def _encode_storage_blob(blob: bytes | None) -> bytes | None:
        """Encode binary blobs in a backend-safe representation."""
        if blob is None:
            return None
        return _BLOB_PREFIX + base64.b64encode(blob)

    @staticmethod
    def _decode_storage_blob(blob: Any) -> bytes | None:
        """Decode storage blobs while remaining compatible with existing raw bytes."""
        if blob is None:
            return None
        if isinstance(blob, memoryview):
            blob = blob.tobytes()
        elif isinstance(blob, bytearray):
            blob = bytes(blob)
        elif isinstance(blob, str):
            blob = blob.encode("utf-8")
        if isinstance(blob, bytes) and blob.startswith(_BLOB_PREFIX):
            return base64.b64decode(blob[len(_BLOB_PREFIX) :])
        return blob

    def setup(self) -> None:
        """Set up the checkpoint database.

        This method creates the necessary tables in the OceanBase database if they
        don't already exist and runs database migrations. It MUST be called directly
        by the user the first time the checkpointer is used.

        Example:
            .. code-block:: python

                checkpointer = OceanBaseCheckpointSaver(connection_args=connection_args)
                checkpointer.setup()  # Must be called before using the checkpointer
        """
        with self._cursor() as conn:
            # Create migrations table first
            conn.execute(text(MIGRATIONS[0]))
            conn.commit()

            # Get current migration version
            result = conn.execute(
                text("SELECT COALESCE(MAX(v), -1) FROM checkpoint_migrations")
            )
            row = self._result_fetchone_or_none(result)
            version = -1 if row is None else row[0]

            # Run pending migrations
            for v, migration in enumerate(MIGRATIONS[version + 1 :], start=version + 1):
                try:
                    conn.execute(text(migration))
                    conn.execute(
                        text("INSERT INTO checkpoint_migrations (v) VALUES (:v)"),
                        {"v": v},
                    )
                    conn.commit()
                except Exception:
                    # Index might already exist, continue
                    conn.rollback()
                    try:
                        conn.execute(
                            text("INSERT INTO checkpoint_migrations (v) VALUES (:v)"),
                            {"v": v},
                        )
                        conn.commit()
                    except Exception:
                        conn.rollback()

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Async wrapper for get_tuple."""
        return self.get_tuple(config)

    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """Async wrapper for list."""
        for item in self.list(config, filter=filter, before=before, limit=limit):
            yield item

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Async wrapper for put."""
        return self.put(config, checkpoint, metadata, new_versions)

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Async wrapper for put_writes."""
        self.put_writes(config, writes, task_id, task_path)

    async def adelete_thread(self, thread_id: str) -> None:
        """Async wrapper for delete_thread."""
        self.delete_thread(thread_id)

    async def aprune(
        self,
        thread_ids: Sequence[str],
        *,
        strategy: str = "keep_latest",
    ) -> None:
        """Async wrapper for prune."""
        self.prune(thread_ids, strategy=strategy)

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Get a checkpoint tuple from the database.

        This method retrieves a checkpoint tuple from the OceanBase database based
        on the provided config. If the config contains a `checkpoint_id` key, the
        checkpoint with the matching thread ID and checkpoint ID is retrieved.
        Otherwise, the latest checkpoint for the given thread ID is retrieved.

        Args:
            config: Configuration specifying which checkpoint to retrieve.
                Must contain `thread_id` in the `configurable` key.

        Returns:
            The retrieved checkpoint tuple, or None if no matching checkpoint was found.

        Example:
            .. code-block:: python

                config = {"configurable": {"thread_id": "1"}}
                checkpoint_tuple = checkpointer.get_tuple(config)
                if checkpoint_tuple:
                    print(checkpoint_tuple.checkpoint)
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = get_checkpoint_id(config)
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")

        with self._cursor() as conn:
            if checkpoint_id:
                query = text(
                    SELECT_SQL
                    + "WHERE c.thread_id = :thread_id "
                    + "AND c.checkpoint_ns = :checkpoint_ns "
                    + "AND c.checkpoint_id = :checkpoint_id"
                )
                result = conn.execute(
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
                result = conn.execute(
                    query,
                    {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns},
                )

            row = self._result_fetchone_or_none(result)
            if row is None:
                return None

            # Load channel values from blobs
            channel_values = self._load_channel_values(
                conn,
                thread_id,
                checkpoint_ns,
                row[5],  # row[5] is checkpoint JSON
            )

            # Load pending writes
            pending_writes = self._load_pending_writes(
                conn,
                thread_id,
                checkpoint_ns,
                row[2],  # row[2] is checkpoint_id
            )

            return self._row_to_checkpoint_tuple(row, channel_values, pending_writes)

    def _load_channel_values(
        self,
        conn: Connection,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_data: Any,
    ) -> Dict[str, Any]:
        """Load channel values from the checkpoint_blobs table.

        Args:
            conn: Database connection.
            thread_id: The thread ID.
            checkpoint_ns: The checkpoint namespace.
            checkpoint_data: The checkpoint JSON data.

        Returns:
            Dictionary of channel names to deserialized values.
        """
        if isinstance(checkpoint_data, str):
            checkpoint_data = json.loads(checkpoint_data)

        channel_versions = checkpoint_data.get("channel_versions", {})
        if not channel_versions:
            return {}

        channel_values = {}
        for channel, version in channel_versions.items():
            query = text(
                "SELECT `type`, `blob` FROM checkpoint_blobs "
                "WHERE thread_id = :thread_id "
                "AND checkpoint_ns = :checkpoint_ns "
                "AND channel = :channel "
                "AND version = :version"
            )
            result = conn.execute(
                query,
                {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "channel": channel,
                    "version": str(version),
                },
            )
            row = self._result_fetchone_or_none(result)
            if row and row[0] != "empty":
                type_str, blob = row[0], row[1]
                decoded_blob = self._decode_storage_blob(blob)
                if decoded_blob is not None:
                    channel_values[channel] = self.serde.loads_typed(
                        (type_str, decoded_blob)
                    )

        return channel_values

    def _load_pending_writes(
        self,
        conn: Connection,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
    ) -> list[tuple[str, str, Any]]:
        """Load pending writes from the checkpoint_writes table.

        Args:
            conn: Database connection.
            thread_id: The thread ID.
            checkpoint_ns: The checkpoint namespace.
            checkpoint_id: The checkpoint ID.

        Returns:
            List of pending writes as (task_id, channel, value) tuples.
        """
        query = text(
            "SELECT task_id, channel, `type`, `blob` FROM checkpoint_writes "
            "WHERE thread_id = :thread_id "
            "AND checkpoint_ns = :checkpoint_ns "
            "AND checkpoint_id = :checkpoint_id "
            "ORDER BY task_id, idx"
        )
        result = conn.execute(
            query,
            {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            },
        )

        writes = []
        for row in self._result_fetchall(result):
            task_id, channel, type_str, blob = row
            decoded_blob = self._decode_storage_blob(blob)
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
        """Convert a database row to a CheckpointTuple.

        Args:
            row: Database row containing checkpoint data.
            channel_values: Loaded channel values.
            pending_writes: Loaded pending writes.

        Returns:
            A CheckpointTuple instance.
        """
        (
            thread_id,
            checkpoint_ns,
            checkpoint_id,
            parent_checkpoint_id,
            type_str,
            checkpoint_data,
            metadata,
        ) = row

        # Parse checkpoint JSON
        if isinstance(checkpoint_data, str):
            checkpoint_data = json.loads(checkpoint_data)

        # Parse metadata JSON
        if isinstance(metadata, str):
            metadata = json.loads(metadata) if metadata else {}
        elif metadata is None:
            metadata = {}

        # Merge loaded channel values with inline values
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

    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints from the database.

        This method retrieves a list of checkpoint tuples from the OceanBase database
        based on the provided config. The checkpoints are ordered by checkpoint ID
        in descending order (newest first).

        Args:
            config: The config to use for listing the checkpoints.
            filter: Additional filtering criteria for metadata.
            before: If provided, only checkpoints before the specified checkpoint ID
                are returned.
            limit: The maximum number of checkpoints to return.

        Yields:
            An iterator of checkpoint tuples.

        Example:
            .. code-block:: python

                config = {"configurable": {"thread_id": "1"}}
                for checkpoint in checkpointer.list(config, limit=5):
                    print(checkpoint.checkpoint["id"])
        """
        where_clause, params = self._build_where_clause(config, filter, before)

        query = SELECT_SQL + where_clause + " ORDER BY c.checkpoint_id DESC"
        if limit is not None:
            query += f" LIMIT {int(limit)}"

        with self._cursor() as conn:
            result = conn.execute(text(query), params)
            rows = self._result_fetchall(result)

            for row in rows:
                thread_id = row[0]
                checkpoint_ns = row[1]
                checkpoint_id = row[2]
                checkpoint_data = row[5]

                channel_values = self._load_channel_values(
                    conn, thread_id, checkpoint_ns, checkpoint_data
                )
                pending_writes = self._load_pending_writes(
                    conn, thread_id, checkpoint_ns, checkpoint_id
                )

                yield self._row_to_checkpoint_tuple(row, channel_values, pending_writes)

    def _build_where_clause(
        self,
        config: Optional[RunnableConfig],
        filter: Optional[Dict[str, Any]],
        before: Optional[RunnableConfig],
    ) -> tuple[str, Dict[str, Any]]:
        """Build WHERE clause for list queries.

        Args:
            config: Configuration for filtering.
            filter: Additional metadata filters.
            before: Config specifying a checkpoint to list before.

        Returns:
            Tuple of (WHERE clause string, parameters dict).
        """
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
            # OceanBase JSON query syntax
            for key, value in filter.items():
                param_name = f"filter_{key}"
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

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save a checkpoint to the database.

        This method saves a checkpoint to the OceanBase database. The checkpoint
        is associated with the provided config and its parent config (if any).

        Args:
            config: The config to associate with the checkpoint.
            checkpoint: The checkpoint to save.
            metadata: Additional metadata to save with the checkpoint.
            new_versions: New channel versions as of this write.

        Returns:
            Updated configuration after storing the checkpoint.

        Example:
            .. code-block:: python

                config = {"configurable": {"thread_id": "1", "checkpoint_ns": ""}}
                checkpoint = {...}  # Checkpoint data
                metadata = {"source": "input", "step": 1}
                new_config = checkpointer.put(config, checkpoint, metadata, {})
        """
        configurable = config["configurable"].copy()
        thread_id = configurable.pop("thread_id")
        checkpoint_ns = configurable.pop("checkpoint_ns", "")
        parent_checkpoint_id = configurable.pop("checkpoint_id", None)

        # Create a copy of checkpoint for storage
        checkpoint_copy = checkpoint.copy()
        checkpoint_copy["channel_values"] = checkpoint_copy["channel_values"].copy()

        # Separate blob values from inline values
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

        with self._cursor() as conn:
            # Save blob values
            blob_versions = {k: v for k, v in new_versions.items() if k in blob_values}
            for channel, version in blob_versions.items():
                value = blob_values[channel]
                type_str, blob = self.serde.dumps_typed(value)
                conn.execute(
                    text(UPSERT_CHECKPOINT_BLOBS_SQL),
                    {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "channel": channel,
                        "version": str(version),
                        "type": type_str,
                        "blob": self._encode_storage_blob(blob),
                    },
                )

            # Prepare metadata for storage
            storage_metadata = self._prepare_metadata(config, metadata)

            # Save checkpoint
            conn.execute(
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
            conn.commit()

        return next_config

    def _prepare_metadata(
        self, config: RunnableConfig, metadata: CheckpointMetadata
    ) -> Dict[str, Any]:
        """Prepare metadata for storage.

        Args:
            config: The configuration.
            metadata: The checkpoint metadata.

        Returns:
            Prepared metadata dictionary.
        """
        return dict(get_serializable_checkpoint_metadata(config, metadata))

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Store intermediate writes linked to a checkpoint.

        This method saves intermediate writes associated with a checkpoint to
        the OceanBase database.

        Args:
            config: Configuration of the related checkpoint.
            writes: List of writes to store, each as a (channel, value) tuple.
            task_id: Identifier for the task creating the writes.
            task_path: Path of the task creating the writes.

        Example:
            .. code-block:: python

                config = {"configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": "abc123"
                }}
                writes = [("messages", {"role": "user", "content": "Hello"})]
                checkpointer.put_writes(config, writes, task_id="task1")
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"]["checkpoint_id"]

        with self._cursor() as conn:
            for idx, (channel, value) in enumerate(writes):
                type_str, blob = self.serde.dumps_typed(value)
                write_idx = WRITES_IDX_MAP.get(channel, idx)

                # Use INSERT IGNORE for special writes, UPSERT for others
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

                conn.execute(
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
                        "blob": self._encode_storage_blob(blob),
                    },
                )
            conn.commit()

    def delete_thread(self, thread_id: str) -> None:
        """Delete all checkpoints and writes associated with a thread ID.

        Args:
            thread_id: The thread ID to delete.

        Example:
            .. code-block:: python

                checkpointer.delete_thread("my-thread")
        """
        with self._cursor() as conn:
            conn.execute(
                text("DELETE FROM checkpoints WHERE thread_id = :thread_id"),
                {"thread_id": thread_id},
            )
            conn.execute(
                text("DELETE FROM checkpoint_blobs WHERE thread_id = :thread_id"),
                {"thread_id": thread_id},
            )
            conn.execute(
                text("DELETE FROM checkpoint_writes WHERE thread_id = :thread_id"),
                {"thread_id": thread_id},
            )
            conn.commit()

    def prune(
        self,
        thread_ids: Sequence[str],
        *,
        strategy: str = "keep_latest",
    ) -> None:
        """Prune checkpoints for the given threads."""
        if not thread_ids:
            return

        if strategy == "delete":
            for thread_id in thread_ids:
                self.delete_thread(thread_id)
            return

        if strategy != "keep_latest":
            raise ValueError(f"Unsupported prune strategy: {strategy}")

        with self._cursor() as conn:
            for thread_id in thread_ids:
                keepers = self._result_fetchall(
                    conn.execute(
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
                )
                if not keepers:
                    continue

                refs_by_ns: dict[str, set[tuple[str, str]]] = {}
                for checkpoint_ns, checkpoint_id in keepers:
                    checkpoint_row = self._result_fetchone_or_none(
                        conn.execute(
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
                    )
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

                    conn.execute(
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
                    conn.execute(
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
                    conn.execute(
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
                        conn.execute(
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

                    conn.execute(
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

            conn.commit()

    def get_next_version(self, current: str | None, channel: None) -> str:
        """Generate the next version ID for a channel.

        This uses a string format that is both unique and monotonically increasing.

        Args:
            current: The current version identifier.
            channel: Unused, kept for interface compatibility.

        Returns:
            The next version identifier.
        """
        if current is None:
            current_v = 0
        elif isinstance(current, int):
            current_v = current
        else:
            current_v = int(current.split(".")[0])

        next_v = current_v + 1
        next_h = random.random()
        return f"{next_v:032}.{next_h:016}"


__all__ = ["OceanBaseCheckpointSaver"]

# mypy: ignore-errors
"""OceanBase LangGraph Checkpoint Saver."""

import pickle
import warnings
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple

from langchain_core.runnables import RunnableConfig

try:
    from langgraph.checkpoint.base import (
        BaseCheckpointSaver,
        ChannelVersions,
        Checkpoint,
        CheckpointMetadata,
        CheckpointTuple,
    )
except ImportError:
    BaseCheckpointSaver = None
    ChannelVersions = None
    Checkpoint = None
    CheckpointMetadata = None
    CheckpointTuple = None
from pyobvector import ObVecClient
from sqlalchemy import BLOB, Column, String, text

from langchain_oceanbase.vectorstores import DEFAULT_OCEANBASE_CONNECTION


class OceanBaseSaver(BaseCheckpointSaver):
    """Legacy checkpoint saver retained for backwards compatibility.

    New code should use ``langchain_oceanbase.OceanBaseCheckpointSaver`` instead.

    Requires langgraph to be installed. Use: pip install langchain-oceanbase[langgraph]

    Args:
        connection_args (Dict[str, Any]): Connection arguments for OceanBase.
        table_name (str): Name of the table to store checkpoints. Defaults to "langchain_checkpoints".
        writes_table_name (str): Name of the table to store pending writes. Defaults to "langchain_checkpoint_writes".
    """

    def __init__(
        self,
        connection_args: Optional[Dict[str, Any]] = None,
        table_name: str = "langchain_checkpoints",
        writes_table_name: str = "langchain_checkpoint_writes",
        **kwargs: Any,
    ) -> None:
        if BaseCheckpointSaver is None:
            raise ImportError(
                "langgraph is required for OceanBaseSaver. "
                "Install it with: pip install langchain-oceanbase[langgraph]"
            )
        warnings.warn(
            "OceanBaseSaver is deprecated and will be removed in a future release. "
            "Use OceanBaseCheckpointSaver instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__()
        self.connection_args = (
            connection_args
            if connection_args is not None
            else DEFAULT_OCEANBASE_CONNECTION
        )
        self.table_name = table_name
        self.writes_table_name = writes_table_name

        # Initialize OceanBase client
        self._create_client(**kwargs)

        # Create tables if they don't exist
        self._create_tables_if_not_exists()

    def _create_client(self, **kwargs: Any) -> None:
        """Create and initialize the OceanBase vector client."""
        host = self.connection_args.get("host", "localhost")
        port = self.connection_args.get("port", "2881")
        user = self.connection_args.get("user", "root@test")
        password = self.connection_args.get("password", "")
        db_name = self.connection_args.get("db_name", "test")

        self.client = ObVecClient(
            uri=f"{host}:{port}",
            user=user,
            password=password,
            db_name=db_name,
            **kwargs,
        )

    def _create_tables_if_not_exists(self) -> None:
        """Create the checkpoint and writes tables if they don't exist."""
        # Table 1: Checkpoints
        if not self.client.check_table_exists(self.table_name):
            cols = [
                Column("thread_id", String(255), primary_key=True),
                Column("checkpoint_ns", String(255), primary_key=True),
                Column("checkpoint_id", String(255), primary_key=True),
                Column("parent_checkpoint_id", String(255), nullable=True),
                Column("type", String(50), nullable=True),
                # Using BLOB for serialized binary data
                Column("checkpoint", BLOB, nullable=False),
                Column("metadata", BLOB, nullable=False),
                Column("created_at", String(50), nullable=False),
            ]

            # Note: pyobvector might not support composite primary keys perfectly in create_table_with_index_params
            # but we define them here. If pyobvector has limitations, we rely on its sqlalchemy underlying.
            self.client.create_table_with_index_params(
                table_name=self.table_name,
                columns=cols,
                indexes=None,
                vidxs=None,
            )

        # Table 2: Writes (Pending writes)
        if not self.client.check_table_exists(self.writes_table_name):
            cols = [
                Column("thread_id", String(255), primary_key=True),
                Column("checkpoint_ns", String(255), primary_key=True),
                Column("checkpoint_id", String(255), primary_key=True),
                Column("task_id", String(255), primary_key=True),
                Column(
                    "idx", String(50), primary_key=True
                ),  # Using String for index to be safe or Integer
                Column("channel", String(255), nullable=False),
                Column("type", String(50), nullable=True),
                Column("value", BLOB, nullable=False),
            ]

            self.client.create_table_with_index_params(
                table_name=self.writes_table_name,
                columns=cols,
                indexes=None,
                vidxs=None,
            )

    def _select(self, sql: str, params: Dict[str, Any]) -> list[Any]:
        """Execute a bound query and materialize rows before releasing the connection."""
        with self.client.engine.connect() as conn:
            return list(conn.execute(text(sql), params).fetchall())

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Get a checkpoint tuple from the database.

        This method retrieves a checkpoint tuple for the given configuration.

        Args:
            config (RunnableConfig): The configuration to use for retrieving the checkpoint.

        Returns:
            Optional[CheckpointTuple]: The retrieved checkpoint tuple, or None if no matching checkpoint is found.
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"].get("checkpoint_id")
        table_name = self.client.engine.dialect.identifier_preparer.quote(
            self.table_name
        )
        params = {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns}
        sql = f"""
            SELECT checkpoint, metadata, parent_checkpoint_id, checkpoint_id
            FROM {table_name}
            WHERE thread_id = :thread_id AND checkpoint_ns = :checkpoint_ns
        """

        if checkpoint_id:
            # ObVecClient.get(ids=...) only filters single-column primary keys.
            sql += " AND checkpoint_id = :checkpoint_id"
            params["checkpoint_id"] = checkpoint_id
            rows = self._select(sql, params)
        else:
            sql += " ORDER BY created_at DESC LIMIT 1"
            try:
                rows = self._select(sql, params)
            except Exception:
                return None

        if not rows:
            return None

        checkpoint_blob, metadata_blob, parent_checkpoint_id, checkpoint_id = rows[0]

        checkpoint = pickle.loads(checkpoint_blob)
        metadata = pickle.loads(metadata_blob)

        # Get pending writes
        writes = []
        # Need to find writes for this checkpoint
        # Writes PK: thread_id, checkpoint_ns, checkpoint_id, task_id, idx
        # We need all writes where thread_id, checkpoint_ns, checkpoint_id match
        writes_table_name = self.client.engine.dialect.identifier_preparer.quote(
            self.writes_table_name
        )
        sql_writes = f"""
            SELECT task_id, channel, type, value
            FROM {writes_table_name}
            WHERE thread_id = :thread_id
              AND checkpoint_ns = :checkpoint_ns
              AND checkpoint_id = :checkpoint_id
        """
        try:
            rows_writes = self._select(
                sql_writes,
                {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint_id,
                },
            )
            for r in rows_writes:
                t_id, channel, _, val_blob = r
                val = pickle.loads(val_blob)
                writes.append((t_id, channel, val))
        except Exception:
            pass

        return CheckpointTuple(
            config,
            checkpoint,
            metadata,
            (
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
            writes,
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

        This method lists checkpoints for the given configuration.

        Args:
            config (Optional[RunnableConfig]): The configuration to use for listing checkpoints.
            filter (Optional[Dict[str, Any]]): Additional filtering criteria.
            before (Optional[RunnableConfig]): List checkpoints before this configuration.
            limit (Optional[int]): A non-negative integer. Zero or None means no limit.

        Raises:
            ValueError: If limit is not a non-negative integer.

        Yields:
            Iterator[CheckpointTuple]: An iterator over the retrieved checkpoint tuples.
        """
        if limit is not None and (type(limit) is not int or limit < 0):
            raise ValueError("limit must be a non-negative integer")

        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        table_name = self.client.engine.dialect.identifier_preparer.quote(
            self.table_name
        )
        params: Dict[str, Any] = {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
        }

        sql = f"""
            SELECT checkpoint, metadata, parent_checkpoint_id, checkpoint_id
            FROM {table_name}
            WHERE thread_id = :thread_id AND checkpoint_ns = :checkpoint_ns
        """

        if before:
            before_id = before["configurable"].get("checkpoint_id")
            if before_id:
                subquery = f"""
                    SELECT created_at FROM {table_name}
                    WHERE thread_id = :thread_id AND checkpoint_ns = :checkpoint_ns
                      AND checkpoint_id = :before_id
                """
                sql += f" AND created_at < ({subquery})"
                params["before_id"] = before_id

        sql += " ORDER BY created_at DESC"

        if limit:
            sql += " LIMIT :limit"
            params["limit"] = limit

        try:
            rows = self._select(sql, params)
        except Exception:
            return

        for row in rows:
            checkpoint_blob, metadata_blob, parent_checkpoint_id, checkpoint_id = row
            checkpoint = pickle.loads(checkpoint_blob)
            metadata = pickle.loads(metadata_blob)

            yield CheckpointTuple(
                {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": checkpoint_id,
                    }
                },
                checkpoint,
                metadata,
                (
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
                [],  # list() typically doesn't return pending writes
            )

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save a checkpoint to the database.

        This method saves a checkpoint to the database.

        Args:
            config (RunnableConfig): The configuration to use for saving the checkpoint.
            checkpoint (Checkpoint): The checkpoint to save.
            metadata (CheckpointMetadata): The metadata associated with the checkpoint.
            new_versions (ChannelVersions): The new channel versions.

        Returns:
            RunnableConfig: The updated configuration.
        """
        import time

        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = checkpoint["id"]
        parent_checkpoint_id = config["configurable"].get("checkpoint_id")

        # Serialize using pickle
        type_ = "pickle"
        checkpoint_blob = pickle.dumps(checkpoint)
        metadata_blob = pickle.dumps(metadata)

        created_at = str(int(time.time() * 1000))

        data = [
            {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
                "parent_checkpoint_id": parent_checkpoint_id,
                "type": type_,
                "checkpoint": checkpoint_blob,
                "metadata": metadata_blob,
                "created_at": created_at,
            }
        ]

        self.client.upsert(
            table_name=self.table_name,
            data=data,
        )

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
    ) -> None:
        """Save pending writes to the database.

        This method saves pending writes to the database.

        Args:
            config (RunnableConfig): The configuration to use for saving the writes.
            writes (Sequence[Tuple[str, Any]]): The writes to save.
            task_id (str): The ID of the task associated with the writes.
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"]["checkpoint_id"]

        data = []
        for idx, (channel, value) in enumerate(writes):
            type_ = "pickle"
            value_blob = pickle.dumps(value)

            data.append(
                {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint_id,
                    "task_id": task_id,
                    "idx": str(idx),
                    "channel": channel,
                    "type": type_,
                    "value": value_blob,
                }
            )

        if data:
            self.client.upsert(
                table_name=self.writes_table_name,
                data=data,
            )

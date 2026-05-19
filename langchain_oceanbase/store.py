"""LangGraph store implementation backed by OceanBase-compatible SQL engines."""

from __future__ import annotations

import asyncio
import hashlib
import math
import threading
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
from typing import Any, Literal, cast

from langchain_core.embeddings import Embeddings
from langgraph.store.base import (
    BaseStore,
    GetOp,
    IndexConfig,
    InvalidNamespaceError,
    Item,
    ListNamespacesOp,
    MatchCondition,
    Op,
    PutOp,
    Result,
    SearchItem,
    SearchOp,
    TTLConfig,
    ensure_embeddings,
    get_text_at_path,
    tokenize_path,
)
from pyobvector import ObVecClient  # type: ignore[import-untyped]
from sqlalchemy import JSON, DateTime, Float, Index, MetaData, String, Table, Text
from sqlalchemy import Column, and_, delete, or_, select, update
from sqlalchemy.engine import Connection, Row

from langchain_oceanbase.exceptions import OceanBaseConnectionError
from langchain_oceanbase.vectorstores import DEFAULT_OCEANBASE_CONNECTION

_TOKENIZED_FIELDS_KEY = "__tokenized_fields"


class OceanBaseStore(BaseStore):
    """Persistent LangGraph store for OceanBase, SeekDB, and compatible backends."""

    supports_ttl: bool = True

    def __init__(
        self,
        connection_args: dict[str, Any] | None = None,
        *,
        index: IndexConfig | None = None,
        ttl_config: TTLConfig | None = None,
        table_name: str = "langgraph_store_items",
        **kwargs: Any,
    ) -> None:
        self.connection_args = (
            connection_args
            if connection_args is not None
            else DEFAULT_OCEANBASE_CONNECTION.copy()
        )
        self.table_name = table_name
        self.ttl_config = ttl_config
        self._kwargs = kwargs
        self._setup_lock = threading.Lock()
        self._setup_complete = False
        self._metadata = MetaData()
        self._items_table = Table(
            self.table_name,
            self._metadata,
            Column("item_id", String(64), primary_key=True),
            Column("namespace_path", String(512), nullable=False),
            Column("namespace_json", JSON, nullable=False),
            Column("item_key", Text, nullable=False),
            Column("value_json", JSON, nullable=False),
            Column("index_data", JSON, nullable=True),
            Column("created_at", DateTime(timezone=True), nullable=False),
            Column("updated_at", DateTime(timezone=True), nullable=False),
            Column("expires_at", DateTime(timezone=True), nullable=True),
            Column("ttl_minutes", Float, nullable=True),
            Index(f"idx_{self.table_name}_namespace_path", "namespace_path"),
            Index(f"idx_{self.table_name}_expires_at", "expires_at"),
        )
        self.index_config = self._prepare_index_config(index)
        self.embeddings: Embeddings | None = None
        if self.index_config is not None:
            self.embeddings = ensure_embeddings(self.index_config.get("embed"))
        self._create_client(**kwargs)

    def setup(self) -> None:
        """Create the backing table and indexes if they do not already exist."""
        if self._setup_complete:
            return
        with self._setup_lock:
            if self._setup_complete:
                return
            self._metadata.create_all(self.obvector.engine, checkfirst=True)
            self._setup_complete = True

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        self.setup()
        operations = list(ops)
        with self.obvector.engine.begin() as conn:
            self._delete_expired_items(conn)
            return [self._execute_op(conn, op) for op in operations]

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        operations = list(ops)
        return await asyncio.to_thread(self.batch, operations)

    def _create_client(self, **kwargs: Any) -> None:
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
        except Exception as exc:
            error_msg = str(exc).lower()
            if (
                "connect" in error_msg
                or "refused" in error_msg
                or "timeout" in error_msg
            ):
                raise OceanBaseConnectionError(
                    f"Failed to connect to OceanBase: {exc}",
                    host=host,
                    port=port,
                ) from exc
            raise

    def _prepare_index_config(self, index: IndexConfig | None) -> dict[str, Any] | None:
        if index is None:
            return None

        prepared = dict(index)
        prepared[_TOKENIZED_FIELDS_KEY] = [
            (path, tokenize_path(path)) if path != "$" else (path, path)
            for path in cast(list[str], prepared.get("fields") or ["$"])
        ]
        return prepared

    def _execute_op(self, conn: Connection, op: Op) -> Result:
        if isinstance(op, GetOp):
            return self._get_item(conn, op)
        if isinstance(op, SearchOp):
            return self._search_items(conn, op)
        if isinstance(op, ListNamespacesOp):
            return self._list_namespaces(conn, op)
        if isinstance(op, PutOp):
            self._put_item(conn, op)
            return None
        raise ValueError(f"Unknown operation type: {type(op)}")

    async def _aexecute_op(self, conn: Connection, op: Op) -> Result:
        if isinstance(op, GetOp):
            return self._get_item(conn, op)
        if isinstance(op, SearchOp):
            return await self._asearch_items(conn, op)
        if isinstance(op, ListNamespacesOp):
            return self._list_namespaces(conn, op)
        if isinstance(op, PutOp):
            await self._aput_item(conn, op)
            return None
        raise ValueError(f"Unknown operation type: {type(op)}")

    def _get_item(self, conn: Connection, op: GetOp) -> Item | None:
        row = self._fetch_item_row(conn, op.namespace, op.key)
        if row is None:
            return None

        item = self._row_to_item(row)
        if op.refresh_ttl and row._mapping["ttl_minutes"] is not None:
            self._refresh_ttl(conn, [row])
        return item

    def _search_items(self, conn: Connection, op: SearchOp) -> list[SearchItem]:
        rows = self._fetch_candidate_rows(conn, op.namespace_prefix)
        filtered = [
            row for row in rows if self._matches_filter(row._mapping["value_json"], op.filter)
        ]

        if op.query and self.index_config is not None and self.embeddings is not None:
            query_embedding = self._normalize_vector(
                self.embeddings.embed_query(op.query)
            )
            results, touched = self._rank_rows(filtered, query_embedding, op)
        else:
            touched = filtered[op.offset : op.offset + op.limit]
            results = [
                SearchItem(
                    namespace=tuple(row._mapping["namespace_json"]),
                    key=str(row._mapping["item_key"]),
                    value=cast(dict[str, Any], row._mapping["value_json"]),
                    created_at=row._mapping["created_at"],
                    updated_at=row._mapping["updated_at"],
                )
                for row in touched
            ]

        if op.refresh_ttl and touched:
            self._refresh_ttl(conn, touched)
        return results

    async def _asearch_items(
        self, conn: Connection, op: SearchOp
    ) -> list[SearchItem]:
        rows = self._fetch_candidate_rows(conn, op.namespace_prefix)
        filtered = [
            row for row in rows if self._matches_filter(row._mapping["value_json"], op.filter)
        ]

        if op.query and self.index_config is not None and self.embeddings is not None:
            query_embedding = self._normalize_vector(
                await self.embeddings.aembed_query(op.query)
            )
            results, touched = self._rank_rows(filtered, query_embedding, op)
        else:
            touched = filtered[op.offset : op.offset + op.limit]
            results = [
                SearchItem(
                    namespace=tuple(row._mapping["namespace_json"]),
                    key=str(row._mapping["item_key"]),
                    value=cast(dict[str, Any], row._mapping["value_json"]),
                    created_at=row._mapping["created_at"],
                    updated_at=row._mapping["updated_at"],
                )
                for row in touched
            ]

        if op.refresh_ttl and touched:
            self._refresh_ttl(conn, touched)
        return results

    def _rank_rows(
        self,
        rows: list[Row[Any]],
        query_embedding: list[float],
        op: SearchOp,
    ) -> tuple[list[SearchItem], list[Row[Any]]]:
        scored: list[tuple[float, Row[Any]]] = []
        scoreless: list[Row[Any]] = []

        for row in rows:
            index_data = row._mapping["index_data"] or []
            if index_data:
                vectors = [
                    self._normalize_vector(entry["embedding"])
                    for entry in index_data
                    if entry.get("embedding") is not None
                ]
                if vectors:
                    best_score = max(self._cosine_similarity(query_embedding, vectors))
                    scored.append((best_score, row))
                    continue
            scoreless.append(row)

        scored.sort(key=lambda item: item[0], reverse=True)
        kept = scored[op.offset : op.offset + op.limit]
        if len(kept) < op.limit:
            kept.extend((None, row) for row in scoreless[: op.limit - len(kept)])

        results: list[SearchItem] = []
        touched: list[Row[Any]] = []
        for score, row in kept:
            mapping = row._mapping
            results.append(
                SearchItem(
                    namespace=tuple(mapping["namespace_json"]),
                    key=str(mapping["item_key"]),
                    value=cast(dict[str, Any], mapping["value_json"]),
                    created_at=mapping["created_at"],
                    updated_at=mapping["updated_at"],
                    score=cast(float | None, score),
                )
            )
            touched.append(row)
        return results, touched

    def _list_namespaces(self, conn: Connection, op: ListNamespacesOp) -> list[tuple[str, ...]]:
        rows = self._fetch_candidate_rows(conn, None)
        namespaces = {
            tuple(cast(list[str], row._mapping["namespace_json"]))
            for row in rows
        }
        ordered = sorted(
            ns
            for ns in namespaces
            if not op.match_conditions
            or all(self._does_match(condition, ns) for condition in op.match_conditions)
        )
        if op.max_depth is not None:
            ordered = sorted({ns[: op.max_depth] for ns in ordered})
        return ordered[op.offset : op.offset + op.limit]

    def _put_item(self, conn: Connection, op: PutOp) -> None:
        self._validate_namespace(op.namespace)
        item_id = self._item_id(op.namespace, op.key)
        if op.value is None:
            conn.execute(
                delete(self._items_table).where(self._items_table.c.item_id == item_id)
            )
            return

        now = self._now()
        existing = self._fetch_row_by_item_id(conn, item_id)
        index_data = self._build_index_entries(op.value, op.index)
        ttl_minutes = op.ttl
        expires_at = self._compute_expires_at(now, ttl_minutes)

        values = {
            "item_id": item_id,
            "namespace_path": self._encode_namespace(op.namespace),
            "namespace_json": list(op.namespace),
            "item_key": str(op.key),
            "value_json": op.value,
            "index_data": index_data,
            "created_at": (
                existing._mapping["created_at"] if existing is not None else now
            ),
            "updated_at": now,
            "expires_at": expires_at,
            "ttl_minutes": ttl_minutes,
        }

        if existing is None:
            conn.execute(self._items_table.insert().values(**values))
        else:
            conn.execute(
                update(self._items_table)
                .where(self._items_table.c.item_id == item_id)
                .values(**values)
            )

    async def _aput_item(self, conn: Connection, op: PutOp) -> None:
        self._validate_namespace(op.namespace)
        item_id = self._item_id(op.namespace, op.key)
        if op.value is None:
            conn.execute(
                delete(self._items_table).where(self._items_table.c.item_id == item_id)
            )
            return

        now = self._now()
        existing = self._fetch_row_by_item_id(conn, item_id)
        index_data = await self._abuild_index_entries(op.value, op.index)
        ttl_minutes = op.ttl
        expires_at = self._compute_expires_at(now, ttl_minutes)

        values = {
            "item_id": item_id,
            "namespace_path": self._encode_namespace(op.namespace),
            "namespace_json": list(op.namespace),
            "item_key": str(op.key),
            "value_json": op.value,
            "index_data": index_data,
            "created_at": (
                existing._mapping["created_at"] if existing is not None else now
            ),
            "updated_at": now,
            "expires_at": expires_at,
            "ttl_minutes": ttl_minutes,
        }

        if existing is None:
            conn.execute(self._items_table.insert().values(**values))
        else:
            conn.execute(
                update(self._items_table)
                .where(self._items_table.c.item_id == item_id)
                .values(**values)
            )

    def _fetch_item_row(
        self, conn: Connection, namespace: tuple[str, ...], key: str
    ) -> Row[Any] | None:
        item_id = self._item_id(namespace, key)
        return self._fetch_row_by_item_id(conn, item_id)

    def _fetch_row_by_item_id(self, conn: Connection, item_id: str) -> Row[Any] | None:
        stmt = (
            select(self._items_table)
            .where(self._items_table.c.item_id == item_id)
            .where(self._live_condition())
            .limit(1)
        )
        return conn.execute(stmt).fetchone()

    def _fetch_candidate_rows(
        self, conn: Connection, namespace_prefix: tuple[str, ...] | None
    ) -> list[Row[Any]]:
        stmt = select(self._items_table).where(self._live_condition())
        if namespace_prefix is not None:
            prefix = self._escape_like_pattern(self._encode_namespace(namespace_prefix))
            stmt = stmt.where(
                self._items_table.c.namespace_path.like(f"{prefix}%", escape="\\")
            )
        stmt = stmt.order_by(
            self._items_table.c.namespace_path,
            self._items_table.c.created_at,
            self._items_table.c.item_id,
        )
        return list(conn.execute(stmt).fetchall())

    def _row_to_item(self, row: Row[Any]) -> Item:
        mapping = row._mapping
        return Item(
            namespace=tuple(cast(list[str], mapping["namespace_json"])),
            key=str(mapping["item_key"]),
            value=cast(dict[str, Any], mapping["value_json"]),
            created_at=mapping["created_at"],
            updated_at=mapping["updated_at"],
        )

    def _delete_expired_items(self, conn: Connection) -> None:
        conn.execute(
            delete(self._items_table).where(
                and_(
                    self._items_table.c.expires_at.is_not(None),
                    self._items_table.c.expires_at <= self._now(),
                )
            )
        )

    def _refresh_ttl(self, conn: Connection, rows: list[Row[Any]]) -> None:
        now = self._now()
        for row in rows:
            ttl_minutes = row._mapping["ttl_minutes"]
            if ttl_minutes is None:
                continue
            conn.execute(
                update(self._items_table)
                .where(self._items_table.c.item_id == row._mapping["item_id"])
                .values(expires_at=self._compute_expires_at(now, ttl_minutes))
            )

    def _build_index_entries(
        self,
        value: dict[str, Any],
        index: Literal[False] | list[str] | None,
    ) -> list[dict[str, Any]] | None:
        if (
            index is False
            or self.index_config is None
            or self.embeddings is None
        ):
            return None

        extracted = self._extract_texts(value, index)
        if not extracted:
            return None

        texts = [text for _, text in extracted]
        embeddings = self.embeddings.embed_documents(texts)
        return [
            {
                "path": path,
                "text": text,
                "embedding": self._normalize_vector(embedding),
            }
            for (path, text), embedding in zip(extracted, embeddings)
        ]

    async def _abuild_index_entries(
        self,
        value: dict[str, Any],
        index: Literal[False] | list[str] | None,
    ) -> list[dict[str, Any]] | None:
        if (
            index is False
            or self.index_config is None
            or self.embeddings is None
        ):
            return None

        extracted = self._extract_texts(value, index)
        if not extracted:
            return None

        texts = [text for _, text in extracted]
        embeddings = await self.embeddings.aembed_documents(texts)
        return [
            {
                "path": path,
                "text": text,
                "embedding": self._normalize_vector(embedding),
            }
            for (path, text), embedding in zip(extracted, embeddings)
        ]

    def _extract_texts(
        self, value: dict[str, Any], index: Literal[False] | list[str] | None
    ) -> list[tuple[str, str]]:
        if self.index_config is None:
            return []

        if index is None:
            paths = cast(list[tuple[str, str | list[str]]], self.index_config[_TOKENIZED_FIELDS_KEY])
        else:
            paths = [
                (path, tokenize_path(path)) if path != "$" else (path, path)
                for path in index
            ]

        extracted: list[tuple[str, str]] = []
        for path, tokenized in paths:
            texts = get_text_at_path(value, tokenized)
            if not texts:
                continue
            if len(texts) == 1:
                extracted.append((path, texts[0]))
            else:
                extracted.extend((f"{path}.{idx}", text) for idx, text in enumerate(texts))
        return extracted

    def _matches_filter(
        self, value: dict[str, Any], filter_value: dict[str, Any] | None
    ) -> bool:
        if not filter_value:
            return True
        return all(
            self._compare_values(value.get(key), expected)
            for key, expected in filter_value.items()
        )

    def _live_condition(self) -> Any:
        now = self._now()
        return or_(
            self._items_table.c.expires_at.is_(None),
            self._items_table.c.expires_at > now,
        )

    def _compute_expires_at(
        self, now: datetime, ttl_minutes: float | None
    ) -> datetime | None:
        if ttl_minutes is None:
            return None
        return now + timedelta(minutes=float(ttl_minutes))

    def _item_id(self, namespace: tuple[str, ...], key: str) -> str:
        digest = hashlib.sha256()
        digest.update(self._encode_namespace(namespace).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(key).encode("utf-8"))
        return digest.hexdigest()

    def _encode_namespace(self, namespace: tuple[str, ...]) -> str:
        return "".join(f"{len(part)}:{part}|" for part in namespace)

    def _escape_like_pattern(self, value: str) -> str:
        return (
            value.replace("\\", "\\\\")
            .replace("%", "\\%")
            .replace("_", "\\_")
        )

    def _validate_namespace(self, namespace: tuple[str, ...]) -> None:
        if not namespace:
            raise InvalidNamespaceError("Namespace cannot be empty.")
        for label in namespace:
            if not isinstance(label, str):
                raise InvalidNamespaceError(
                    f"Invalid namespace label '{label}' found in {namespace}. Namespace labels must be strings."
                )
            if "." in label:
                raise InvalidNamespaceError(
                    f"Invalid namespace label '{label}' found in {namespace}. Namespace labels cannot contain periods ('.')."
                )
            if not label:
                raise InvalidNamespaceError(
                    f"Namespace labels cannot be empty strings. Got {label} in {namespace}"
                )
        if namespace[0] == "langgraph":
            raise InvalidNamespaceError(
                f'Root label for namespace cannot be "langgraph". Got: {namespace}'
            )

    def _does_match(
        self, match_condition: MatchCondition, key: tuple[str, ...]
    ) -> bool:
        path = match_condition.path
        if len(key) < len(path):
            return False

        if match_condition.match_type == "prefix":
            pairs = zip(key, path)
        elif match_condition.match_type == "suffix":
            pairs = zip(reversed(key), reversed(path))
        else:
            raise ValueError(f"Unsupported match type: {match_condition.match_type}")

        for key_part, path_part in pairs:
            if path_part == "*":
                continue
            if key_part != path_part:
                return False
        return True

    def _compare_values(self, item_value: Any, filter_value: Any) -> bool:
        if isinstance(filter_value, dict):
            if any(str(key).startswith("$") for key in filter_value):
                return all(
                    self._apply_operator(item_value, operator, operand)
                    for operator, operand in filter_value.items()
                )
            if not isinstance(item_value, dict):
                return False
            return all(
                self._compare_values(item_value.get(key), nested)
                for key, nested in filter_value.items()
            )
        if isinstance(filter_value, (list, tuple)):
            return (
                isinstance(item_value, (list, tuple))
                and len(item_value) == len(filter_value)
                and all(
                    self._compare_values(actual, expected)
                    for actual, expected in zip(item_value, filter_value)
                )
            )
        return item_value == filter_value

    def _apply_operator(self, value: Any, operator: str, operand: Any) -> bool:
        if operator == "$eq":
            return value == operand
        if operator == "$ne":
            return value != operand
        if operator == "$gt":
            return float(value) > float(operand)
        if operator == "$gte":
            return float(value) >= float(operand)
        if operator == "$lt":
            return float(value) < float(operand)
        if operator == "$lte":
            return float(value) <= float(operand)
        raise ValueError(f"Unsupported operator: {operator}")

    def _normalize_vector(self, vector: list[float] | tuple[float, ...] | Any) -> list[float]:
        return [float(value) for value in vector]

    def _cosine_similarity(
        self, query_vector: list[float], item_vectors: list[list[float]]
    ) -> list[float]:
        query_norm = math.sqrt(sum(value * value for value in query_vector))
        if query_norm == 0:
            return [0.0 for _ in item_vectors]

        similarities: list[float] = []
        for vector in item_vectors:
            item_norm = math.sqrt(sum(value * value for value in vector))
            if item_norm == 0:
                similarities.append(0.0)
                continue
            dot = sum(left * right for left, right in zip(query_vector, vector))
            similarities.append(dot / (query_norm * item_norm))
        return similarities

    def _now(self) -> datetime:
        return datetime.now(timezone.utc)

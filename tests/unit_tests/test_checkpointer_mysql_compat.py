"""Unit tests for MySQL-compatible checkpoint saver behavior."""

from __future__ import annotations

from contextlib import contextmanager

import pytest

from langchain_oceanbase.checkpointer import MIGRATIONS, OceanBaseCheckpointSaver


@pytest.fixture
def saver(monkeypatch: pytest.MonkeyPatch) -> OceanBaseCheckpointSaver:
    """Create a saver without opening a real database connection."""
    monkeypatch.setattr(
        OceanBaseCheckpointSaver, "_create_client", lambda self, **_: None
    )
    return OceanBaseCheckpointSaver(connection_args={})


def test_build_where_clause_unquotes_string_metadata_filters(
    saver: OceanBaseCheckpointSaver,
) -> None:
    """String metadata filters should compare against unquoted JSON values."""
    where_clause, params = saver._build_where_clause(
        None,
        {"source": "loop"},
        None,
    )

    assert (
        "JSON_UNQUOTE(JSON_EXTRACT(c.metadata, '$.source')) = :filter_source"
        in where_clause
    )
    assert params == {"filter_source": "loop"}


def test_index_migrations_avoid_mysql_unsupported_if_not_exists() -> None:
    """MySQL should receive CREATE INDEX statements without IF NOT EXISTS."""
    index_migrations = [
        migration.strip()
        for migration in MIGRATIONS
        if migration.strip().startswith("CREATE INDEX")
    ]

    assert index_migrations
    assert all("IF NOT EXISTS" not in migration for migration in index_migrations)


def test_mysql_primary_keys_fit_innodb_utf8mb4_index_limit() -> None:
    """Composite primary key column widths should stay within MySQL's key limit."""
    checkpoint_blobs = MIGRATIONS[2]
    checkpoint_writes = MIGRATIONS[3]

    assert "channel VARCHAR(128)" in checkpoint_blobs
    assert "version VARCHAR(128)" in checkpoint_blobs
    assert "checkpoint_id VARCHAR(128)" in checkpoint_writes
    assert "task_id VARCHAR(128)" in checkpoint_writes


class _FakeResult:
    def __init__(self, row: tuple[int] | None = None) -> None:
        self._row = row

    def fetchone(self) -> tuple[int] | None:
        return self._row


class _FakeConnection:
    def __init__(self, *, version: int, fail_sql: str | None = None) -> None:
        self.version = version
        self.fail_sql = fail_sql
        self.executed_sql: list[str] = []
        self.commits = 0
        self.rollbacks = 0

    def execute(self, statement, params=None):  # type: ignore[no-untyped-def]
        sql = str(statement)
        self.executed_sql.append(sql)
        if "SELECT COALESCE(MAX(v), -1) FROM checkpoint_migrations" in sql:
            return _FakeResult((self.version,))
        if self.fail_sql is not None and self.fail_sql in sql:
            raise RuntimeError("syntax error")
        return _FakeResult()

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1


def test_setup_repairs_missing_indexes_when_versions_are_already_recorded(
    saver: OceanBaseCheckpointSaver,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """setup() should repair missing indexes even when index migrations are marked applied."""
    conn = _FakeConnection(version=6)

    @contextmanager
    def fake_cursor():  # type: ignore[no-untyped-def]
        yield conn

    monkeypatch.setattr(saver, "_cursor", fake_cursor)
    monkeypatch.setattr(saver, "_existing_index_names", lambda *_: set())

    saver.setup()

    create_index_sql = [sql for sql in conn.executed_sql if "CREATE INDEX" in sql]
    assert len(create_index_sql) == 3


def test_setup_reraises_unexpected_missing_index_repair_error(
    saver: OceanBaseCheckpointSaver,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unexpected index repair failures should abort setup instead of being recorded as success."""
    conn = _FakeConnection(version=6, fail_sql="CREATE INDEX idx_checkpoints_thread_id")

    @contextmanager
    def fake_cursor():  # type: ignore[no-untyped-def]
        yield conn

    monkeypatch.setattr(saver, "_cursor", fake_cursor)
    monkeypatch.setattr(saver, "_existing_index_names", lambda *_: set())

    with pytest.raises(RuntimeError, match="syntax error"):
        saver.setup()

# mypy: disable-error-code="import-untyped,no-untyped-def"
"""Upstream LangGraph checkpoint conformance tests for OceanBase."""

from __future__ import annotations

import os
import shutil
import tempfile
import uuid
from pathlib import Path

import pytest
from langgraph.checkpoint.conformance import checkpointer_test, validate
from langgraph.checkpoint.conformance.report import ProgressCallbacks

from langchain_oceanbase import OceanBaseCheckpointSaver


def _embedded_seekdb_runtime_available() -> bool:
    try:
        import pylibseekdb  # noqa: F401
    except ImportError:
        return False
    try:
        import pyseekdb  # noqa: F401
    except ImportError:
        return False
    return True


def _ci_db_type() -> str:
    """Return the live database type provisioned by the CI matrix."""
    return os.getenv("OB_CI_DB_TYPE", "").strip().lower()


def _ci_oceanbase_server_available() -> bool:
    """Return True when CI provisioned a live OceanBase server for tests."""
    return _ci_db_type() == "oceanbase"


def _ci_seekdb_server_available() -> bool:
    """Return True when CI provisioned a standalone SeekDB server for tests."""
    return _ci_db_type() == "seekdb"


def _ci_mysql_server_available() -> bool:
    """Return True when CI provisioned a live MySQL server for tests."""
    return _ci_db_type() == "mysql"


def _server_connection_args_from_env() -> dict[str, str]:
    """Build server connection arguments from the shared CI contract."""
    return {
        "host": os.getenv("OB_HOST", "127.0.0.1"),
        "port": os.getenv("OB_PORT", "2881"),
        "user": os.getenv("OB_USER", "root@test"),
        "password": os.getenv("OB_PASSWORD", ""),
        "db_name": os.getenv("OB_DB", "test"),
    }


@checkpointer_test(name="OceanBaseCheckpointSaver")
async def oceanbase_checkpointer():
    root = Path(
        tempfile.mkdtemp(prefix=f"ob_checkpoint_conformance_{uuid.uuid4().hex}_")
    )
    db_path = str(root / "seekdb_data")
    saver = OceanBaseCheckpointSaver(connection_args={"db_name": "test"}, path=db_path)
    saver.setup()
    try:
        yield saver
    finally:
        shutil.rmtree(root, ignore_errors=True)


@checkpointer_test(name="OceanBaseCheckpointSaver-Server")
async def oceanbase_server_checkpointer():
    if not _ci_oceanbase_server_available():
        pytest.skip(
            "OceanBase server conformance runs only in the oceanbase CI matrix."
        )

    saver = OceanBaseCheckpointSaver(
        connection_args=_server_connection_args_from_env(),
    )
    saver.setup()
    yield saver


@checkpointer_test(name="OceanBaseCheckpointSaver-SeekDB-Server")
async def seekdb_server_checkpointer():
    if not _ci_seekdb_server_available():
        pytest.skip("SeekDB server conformance runs only in the seekdb CI matrix.")

    saver = OceanBaseCheckpointSaver(
        connection_args=_server_connection_args_from_env(),
    )
    saver.setup()
    yield saver


@checkpointer_test(name="OceanBaseCheckpointSaver-MySQL-Server")
async def mysql_server_checkpointer():
    if not _ci_mysql_server_available():
        pytest.skip("MySQL server conformance runs only in the mysql CI matrix.")

    saver = OceanBaseCheckpointSaver(
        connection_args=_server_connection_args_from_env(),
    )
    saver.setup()
    yield saver


@pytest.mark.asyncio
@pytest.mark.skipif(
    not _embedded_seekdb_runtime_available(),
    reason=(
        "embedded SeekDB requires pylibseekdb (e.g. pip install 'pyseekdb>=1.2' "
        "or pip install 'pyobvector[pyseekdb]')"
    ),
)
async def test_checkpoint_conformance_base() -> None:
    """OceanBaseCheckpointSaver should satisfy the base conformance suite."""
    report = await validate(
        oceanbase_checkpointer,
        capabilities={
            "put",
            "put_writes",
            "get_tuple",
            "list",
            "delete_thread",
            "prune",
        },
        progress=ProgressCallbacks.verbose(),
    )
    report.print_report()
    assert report.passed_all_base()
    assert report.results["prune"].passed is True


@pytest.mark.asyncio
async def test_checkpoint_conformance_oceanbase_server() -> None:
    """OceanBaseCheckpointSaver should satisfy the base suite against a real server."""
    if not _ci_oceanbase_server_available():
        pytest.skip(
            "OceanBase server conformance runs only in the oceanbase CI matrix."
        )

    report = await validate(
        oceanbase_server_checkpointer,
        capabilities={
            "put",
            "put_writes",
            "get_tuple",
            "list",
            "delete_thread",
            "prune",
        },
        progress=ProgressCallbacks.verbose(),
    )
    report.print_report()
    assert report.passed_all_base()
    assert report.results["prune"].passed is True


@pytest.mark.asyncio
async def test_checkpoint_conformance_seekdb_server() -> None:
    """OceanBaseCheckpointSaver should satisfy the base suite against standalone SeekDB."""
    if not _ci_seekdb_server_available():
        pytest.skip("SeekDB server conformance runs only in the seekdb CI matrix.")

    report = await validate(
        seekdb_server_checkpointer,
        capabilities={
            "put",
            "put_writes",
            "get_tuple",
            "list",
            "delete_thread",
            "prune",
        },
        progress=ProgressCallbacks.verbose(),
    )
    report.print_report()
    assert report.passed_all_base()
    assert report.results["prune"].passed is True


@pytest.mark.asyncio
async def test_checkpoint_conformance_mysql_server() -> None:
    """OceanBaseCheckpointSaver should satisfy the base suite against MySQL."""
    if not _ci_mysql_server_available():
        pytest.skip("MySQL server conformance runs only in the mysql CI matrix.")

    report = await validate(
        mysql_server_checkpointer,
        capabilities={
            "put",
            "put_writes",
            "get_tuple",
            "list",
            "delete_thread",
            "prune",
        },
        progress=ProgressCallbacks.verbose(),
    )
    report.print_report()
    assert report.passed_all_base()
    assert report.results["prune"].passed is True

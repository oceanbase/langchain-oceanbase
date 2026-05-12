# mypy: disable-error-code="import-untyped,no-untyped-def"
"""Upstream LangGraph checkpoint conformance tests for OceanBase."""

from __future__ import annotations

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


pytestmark = pytest.mark.skipif(
    not _embedded_seekdb_runtime_available(),
    reason=(
        "embedded SeekDB requires pylibseekdb (e.g. pip install 'pyseekdb>=1.2' "
        "or pip install 'pyobvector[pyseekdb]')"
    ),
)


@checkpointer_test(name="OceanBaseCheckpointSaver")
async def oceanbase_checkpointer():
    root = Path(tempfile.mkdtemp(prefix=f"ob_checkpoint_conformance_{uuid.uuid4().hex}_"))
    db_path = str(root / "seekdb_data")
    saver = OceanBaseCheckpointSaver(connection_args={"db_name": "test"}, path=db_path)
    saver.setup()
    try:
        yield saver
    finally:
        shutil.rmtree(root, ignore_errors=True)


@pytest.mark.asyncio
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

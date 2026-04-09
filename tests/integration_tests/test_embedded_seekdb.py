"""Integration tests for OceanbaseVectorStore with embedded SeekDB (pyobvector ``path=`` / ``pyseekdb_client=``).

Requires the embedded runtime (``pylibseekdb``), installed with recent ``pyseekdb`` releases
(e.g. ``pip install 'pyseekdb>=1.2'``) or ``pip install 'pyobvector[pyseekdb]``.
Tests are skipped when the native wheel is unavailable so CI without embedded support still passes.
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Generator

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings

from langchain_oceanbase.vectorstores import OceanbaseVectorStore

EMBED_DIM = 384


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


_EMBEDDED_SEEKDB = _embedded_seekdb_runtime_available()

pytestmark = pytest.mark.skipif(
    not _EMBEDDED_SEEKDB,
    reason=(
        "embedded SeekDB requires pylibseekdb (e.g. pip install 'pyseekdb>=1.2' "
        "or pip install 'pyobvector[pyseekdb]')"
    ),
)


@pytest.fixture
def seekdb_parent_dir(tmp_path: Path) -> Generator[Path, None, None]:
    """Isolated directory tree per test; removed in teardown."""
    root = tmp_path / f"seekdb_root_{uuid.uuid4().hex}"
    root.mkdir(parents=True)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


@pytest.fixture
def embeddings() -> FakeEmbeddings:
    return FakeEmbeddings(size=EMBED_DIM)


@pytest.mark.embedded_seekdb
class TestEmbeddedSeekDBConnection:
    """Smoke test: ObVecClient via embedded path; add documents and similarity search."""

    def test_connection_with_path_add_and_search(
        self, seekdb_parent_dir: Path, embeddings: FakeEmbeddings
    ) -> None:
        db_path = str(seekdb_parent_dir / "seekdb_data")
        table = f"lc_embed_path_{uuid.uuid4().hex[:8]}"
        store = OceanbaseVectorStore(
            embedding_function=embeddings,
            table_name=table,
            path=db_path,
            embedding_dim=EMBED_DIM,
            drop_old=True,
            index_type="HNSW",
            vidx_metric_type="l2",
        )
        store.add_documents(
            [
                Document(
                    page_content="embedded seekdb path connection", metadata={"t": "p"}
                )
            ]
        )
        out = store.similarity_search("connection", k=1)
        assert len(out) == 1
        assert "embedded" in out[0].page_content

    def test_connection_with_pyseekdb_client_add_and_search(
        self, seekdb_parent_dir: Path, embeddings: FakeEmbeddings
    ) -> None:
        import pyseekdb

        db_path = str(seekdb_parent_dir / "seekdb_data_client")
        client = pyseekdb.Client(path=db_path, database="test")
        table = f"lc_embed_client_{uuid.uuid4().hex[:8]}"
        store = OceanbaseVectorStore(
            embedding_function=embeddings,
            table_name=table,
            pyseekdb_client=client,
            embedding_dim=EMBED_DIM,
            drop_old=True,
            index_type="HNSW",
            vidx_metric_type="l2",
        )
        store.add_documents(
            [Document(page_content="pyseekdb client embedded", metadata={"t": "c"})]
        )
        out = store.similarity_search("client", k=1)
        assert len(out) == 1
        assert "pyseekdb" in out[0].page_content

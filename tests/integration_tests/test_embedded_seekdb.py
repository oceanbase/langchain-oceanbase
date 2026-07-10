"""Integration tests for OceanbaseVectorStore with embedded SeekDB (pyobvector ``path=`` / ``pyseekdb_client=``).

Requires the embedded runtime (``pylibseekdb``), installed with recent ``pyseekdb`` releases
(e.g. ``pip install 'pyseekdb>=1.2'``) or ``pip install 'pyobvector[pyseekdb]``.
Tests are skipped when the native wheel is unavailable so CI without embedded support still passes.
"""

from __future__ import annotations

import multiprocessing as mp
import traceback
import uuid
from multiprocessing.queues import Queue
from pathlib import Path
from queue import Empty

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings

from langchain_oceanbase.vectorstores import OceanbaseVectorStore

EMBED_DIM = 384
NATIVE_PYSEEKDB_TIMEOUT_SECONDS = 60


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


@pytest.fixture(scope="session")
def seekdb_parent_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Session-scoped directory for embedded SeekDB data.

    Not removed during the process — the embedded engine is a process-wide
    singleton and deleting its data directory while it is still open causes
    segfaults (see oceanbase/seekdb#870).
    """
    return tmp_path_factory.mktemp("seekdb_root")


@pytest.fixture
def embeddings() -> FakeEmbeddings:
    return FakeEmbeddings(size=EMBED_DIM)


def _native_pyseekdb_collection_smoke(
    db_path: str, collection_name: str, queue: Queue
) -> None:
    try:
        import pyseekdb

        client = pyseekdb.Client(path=db_path, database="test")
        collection = client.create_collection(
            collection_name,
            configuration=pyseekdb.HNSWConfiguration(dimension=3, distance="l2"),
            embedding_function=None,
        )

        collection.add(
            ids=["native-1", "native-2"],
            embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            documents=["native pyseekdb embedded", "second native document"],
            metadatas=[{"source": "native"}, {"source": "native"}],
        )

        by_id = collection.get(ids="native-1", include=["documents", "metadatas"])
        # SeekDB 1.3.0 builds vector indexes asynchronously; flush before ANN query.
        collection.refresh_index()
        nearest = collection.query(
            query_embeddings=[[1.0, 0.0, 0.0]],
            n_results=2,
            include=["documents", "metadatas"],
        )

        assert by_id["ids"] == ["native-1"]
        assert by_id["documents"] == ["native pyseekdb embedded"]
        assert by_id["metadatas"] == [{"source": "native"}]
        assert len(nearest["ids"]) == 1
        assert set(nearest["ids"][0]) == {"native-1", "native-2"}, nearest
        assert set(nearest["documents"][0]) == {
            "native pyseekdb embedded",
            "second native document",
        }, nearest
    except BaseException:
        queue.put(("error", traceback.format_exc()))
    else:
        queue.put(("ok", "native pyseekdb collection smoke passed"))


@pytest.mark.embedded_seekdb
class TestEmbeddedSeekDBConnection:
    """Smoke tests for native pyseekdb and ObVecClient embedded paths."""

    def test_native_pyseekdb_collection_add_get_and_query(
        self, seekdb_parent_dir: Path
    ) -> None:
        """Exercise pyseekdb directly, without pyobvector's SQLAlchemy adapter."""
        ctx = mp.get_context("spawn")
        queue = ctx.Queue()
        process = ctx.Process(
            target=_native_pyseekdb_collection_smoke,
            args=(
                str(seekdb_parent_dir / f"seekdb_data_native_{uuid.uuid4().hex[:8]}"),
                f"lc_native_{uuid.uuid4().hex[:8]}",
                queue,
            ),
        )

        process.start()
        process.join(NATIVE_PYSEEKDB_TIMEOUT_SECONDS)
        if process.is_alive():
            process.terminate()
            process.join(5)
            pytest.fail(
                "native pyseekdb collection smoke did not complete within "
                f"{NATIVE_PYSEEKDB_TIMEOUT_SECONDS} seconds",
                pytrace=False,
            )

        try:
            status, payload = queue.get(timeout=5)
        except Empty:
            pytest.fail(
                "native pyseekdb collection smoke exited without reporting a "
                f"result; exitcode={process.exitcode}",
                pytrace=False,
            )

        if status != "ok":
            pytest.fail(payload, pytrace=False)
        assert process.exitcode == 0

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
            index_type="FLAT",
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
            index_type="FLAT",
            vidx_metric_type="l2",
        )
        store.add_documents(
            [Document(page_content="pyseekdb client embedded", metadata={"t": "c"})]
        )
        out = store.similarity_search("client", k=1)
        assert len(out) == 1
        assert "pyseekdb" in out[0].page_content

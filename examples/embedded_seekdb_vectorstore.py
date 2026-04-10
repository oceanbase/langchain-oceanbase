#!/usr/bin/env python3
"""
embedded_seekdb_vectorstore.py

Standalone example: ``langchain-oceanbase`` with latest ``pyobvector`` and **embedded SeekDB**
(local files, no OceanBase server). Demonstrates writes and similarity search on ``OceanbaseVectorStore``.

Install (virtualenv recommended)::

    pip install -U langchain-oceanbase "pyobvector[pyseekdb]"

Embedded mode needs a recent ``pyseekdb`` that installs the ``pylibseekdb`` wheel. If imports fail::

    pip install -U "pyseekdb>=1.2"

Environment:
    SEEKDB_DATA_DIR  Optional directory for SeekDB data; if unset, a temp directory is used and removed at exit.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path


def _require_embedded_runtime() -> None:
    try:
        import pylibseekdb  # noqa: F401
    except ImportError:
        print(
            "pylibseekdb (embedded SeekDB runtime) not found. Install with:\n"
            '  pip install -U langchain-oceanbase "pyobvector[pyseekdb]"\n'
            "or:\n"
            '  pip install -U "pyseekdb>=1.2"',
            file=sys.stderr,
        )
        sys.exit(1)


def main() -> None:
    _require_embedded_runtime()

    from langchain_core.documents import Document
    from langchain_core.embeddings import FakeEmbeddings

    from langchain_oceanbase.vectorstores import OceanbaseVectorStore

    if base := os.environ.get("SEEKDB_DATA_DIR"):
        db_path = str(Path(base).resolve())
        cleanup = False
    else:
        tmp = tempfile.mkdtemp(prefix="lc_ob_embedded_seekdb_")
        db_path = str(Path(tmp) / "seekdb_data")
        cleanup = True

    # FakeEmbeddings: no API keys; fixed dimension 384
    embeddings = FakeEmbeddings(size=384)

    # Embedded mode uses ``path=``; ``connection_args`` is optional (defaults include db_name="test").
    # Pass connection_args={"db_name": "mydb", ...} to use another logical database name.
    store = OceanbaseVectorStore(
        embedding_function=embeddings,
        table_name="demo_embedded_vectors",
        path=db_path,
        embedding_dim=384,
        drop_old=True,
        index_type="HNSW",
        vidx_metric_type="l2",
    )

    docs = [
        Document(
            page_content="Embedded SeekDB runs without a separate database server.",
            metadata={"topic": "deploy"},
        ),
        Document(
            page_content="OceanbaseVectorStore connects to local SeekDB via pyobvector path=.",
            metadata={"topic": "integration"},
        ),
        Document(
            page_content="LangChain retrievers can wrap the vector store for RAG.",
            metadata={"topic": "langchain"},
        ),
    ]
    store.add_documents(docs)

    query = "how to connect vector database locally"
    results = store.similarity_search(query, k=2)
    print(f"Query: {query!r}\n")
    for i, doc in enumerate(results, 1):
        print(f"[{i}] {doc.page_content}")
        print(f"    metadata: {doc.metadata}\n")

    if cleanup:
        shutil.rmtree(Path(db_path).parent, ignore_errors=True)


if __name__ == "__main__":
    main()

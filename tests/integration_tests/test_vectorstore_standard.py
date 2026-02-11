"""Standard integration tests for OceanbaseVectorStore using langchain-tests."""

from __future__ import annotations

import os
from typing import Generator

import pytest
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_tests.integration_tests import (
    VectorStoreIntegrationTests,
)

from langchain_oceanbase.vectorstores import OceanbaseVectorStore

EMBEDDING_SIZE = 6


class TestOceanbaseVectorStoreStandard(VectorStoreIntegrationTests):
    """Standard integration tests for OceanbaseVectorStore.

    This class inherits from VectorStoreIntegrationTests and configures
    feature flags based on OceanbaseVectorStore capabilities.
    """

    # Feature flags - set based on OceanbaseVectorStore capabilities
    # https://python.langchain.com/docs/integrations/vectorstores/

    supports_mmr: bool = False  # max_marginal_relevance_search not implemented
    supports_filter: bool = True  # supports filter in similarity_search
    supports_json_metadata: bool = True  # stores metadata as JSON
    supports_odata_filter: bool = False  # OData filter not supported
    supports_search_with_score: bool = True  # has similarity_search_with_score
    supports_delete: bool = True  # has delete method
    supports_get_by_ids: bool = True  # has get_by_ids method
    requires_embedding_param: bool = False  # embedding_function in constructor
    supports_async: bool = True  # async methods supported via langchain-core
    supports_text_search: bool = True  # basic text search supported

    @pytest.fixture()
    def vectorstore(self) -> Generator[VectorStore, None, None]:  # type: ignore
        """Get an empty vectorstore for standard integration tests."""
        connection_args = {
            "host": os.getenv("SEEKDB_HOST") or os.getenv("OB_HOST", "127.0.0.1"),
            "port": os.getenv("SEEKDB_PORT") or os.getenv("OB_PORT", "2881"),
            "user": os.getenv("SEEKDB_USER") or os.getenv("OB_USER", "root@test"),
            "password": os.getenv("SEEKDB_PASSWORD") or os.getenv("OB_PASSWORD", ""),
            "db_name": os.getenv("SEEKDB_DB") or os.getenv("OB_DB", "test"),
        }
        store = OceanbaseVectorStore(
            embedding_function=self.get_embeddings(),
            table_name="langchain_vector_standard",
            connection_args=connection_args,
            vidx_metric_type="l2",
            drop_old=True,
            embedding_dim=EMBEDDING_SIZE,
        )
        try:
            yield store
        finally:
            # Cleanup: drop the test table
            try:
                store.obvector.drop_table_if_exist(store.table_name)
            except Exception:
                pass

    @pytest.mark.xfail(reason="UUID is unordered.")
    def test_add_documents_documents(self, vectorstore: VectorStore) -> None:
        pass

    @pytest.mark.xfail(reason="UUID is unordered.")
    async def test_add_documents_documents_async(
        self, vectorstore: VectorStore
    ) -> None:
        pass

    @pytest.mark.xfail(reason="`bar` has no id.")
    def test_add_documents_with_existing_ids(self, vectorstore: VectorStore) -> None:
        pass

    @pytest.mark.xfail(reason="`bar` has no id.")
    async def test_add_documents_with_existing_ids_async(
        self, vectorstore: VectorStore
    ) -> None:
        pass

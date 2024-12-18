from typing import Generator

import pytest
from langchain_core.vectorstores import VectorStore
from langchain_tests.integration_tests import (
    VectorStoreIntegrationTests,
)

from langchain_oceanbase.vectorstores import OceanbaseVectorStore

EMBEDDING_SIZE = 6


class TestOceanbaseVectorStoreSync(VectorStoreIntegrationTests):
    @pytest.fixture()
    def vectorstore(self) -> Generator[VectorStore, None, None]:  # type: ignore
        """Get an empty vectorstore for unit tests."""
        connection_args = {
            "host": "127.0.0.1",
            "port": "2881",
            "user": "root@test",
            "password": "",
            "db_name": "test",
        }
        store = OceanbaseVectorStore(
            embedding_function=self.get_embeddings(),
            table_name="langchain_vector",
            connection_args=connection_args,
            vidx_metric_type="l2",
            drop_old=True,
            embedding_dim=EMBEDDING_SIZE,
        )
        # note: store should be EMPTY at this point
        # if you need to delete data, you may do so here
        try:
            yield store
        finally:
            # cleanup operations, or deleting data
            pass

    @pytest.mark.xfail(reason="UUID is unordered.")
    def test_add_documents_documents(self, vectorstore: VectorStore) -> None:
        pass

    @pytest.mark.xfail(reason=("UUID is unordered."))
    async def test_add_documents_documents_async(
        self, vectorstore: VectorStore
    ) -> None:
        pass

    @pytest.mark.xfail(reason=("`bar` has no id."))
    def test_add_documents_with_existing_ids(self, vectorstore: VectorStore) -> None:
        pass

    @pytest.mark.xfail(reason=("`bar` has no id."))
    async def test_add_documents_with_existing_ids_async(
        self, vectorstore: VectorStore
    ) -> None:
        pass

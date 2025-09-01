from typing import Generator
import pytest
from langchain_core.vectorstores import VectorStore
from langchain_tests.integration_tests import (
    VectorStoreIntegrationTests,
)
from langchain_oceanbase.vectorstores import OceanbaseVectorStore
from langchain_community.embeddings import FakeEmbeddings
from langchain_core.documents import Document

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


class TestOceanbaseVectorStoreIntegration:
    """Integration tests for OceanbaseVectorStore"""
    
    @pytest.fixture
    def vectorstore(self):
        """Create a vectorstore for integration tests"""
        connection_args = {
            "host": "127.0.0.1",
            "port": "2881",
            "user": "root@test",
            "password": "",
            "db_name": "test",
        }
        embeddings = FakeEmbeddings(size=6)
        
        store = OceanbaseVectorStore(
            embedding_function=embeddings,
            table_name="integration_test",
            connection_args=connection_args,
            vidx_metric_type="l2",
            drop_old=True,
            embedding_dim=6,
        )
        return store
    
    def test_basic_add_and_search(self, vectorstore):
        """Test basic document addition and search functionality"""
        documents = [
            Document(page_content="Hello world", metadata={"source": "test1"}),
            Document(page_content="Python programming", metadata={"source": "test2"}),
            Document(page_content="Machine learning", metadata={"source": "test3"}),
        ]
        
        # Add documents
        ids = vectorstore.add_documents(documents)
        assert len(ids) == 3
        
        # Test similarity search
        results = vectorstore.similarity_search("Hello", k=2)
        assert len(results) == 2
        assert any("Hello world" in doc.page_content for doc in results)
        
        # Test similarity search with score
        results_with_score = vectorstore.similarity_search_with_score("Python", k=2)
        assert len(results_with_score) == 2
        assert all(isinstance(result, tuple) and len(result) == 2 for result in results_with_score)
    
    def test_get_by_ids(self, vectorstore):
        """Test retrieving documents by IDs"""
        documents = [
            Document(page_content="Document 1", metadata={"id": "1"}),
            Document(page_content="Document 2", metadata={"id": "2"}),
            Document(page_content="Document 3", metadata={"id": "3"}),
        ]
        
        ids = vectorstore.add_documents(documents)
        
        # Retrieve by IDs
        retrieved_docs = vectorstore.get_by_ids(ids[:2])
        assert len(retrieved_docs) == 2
        
        # Check content
        content_list = [doc.page_content for doc in retrieved_docs]
        assert "Document 1" in content_list
        assert "Document 2" in content_list
    
    def test_from_texts_integration(self, vectorstore):
        """Test from_texts method integration"""
        texts = ["Integration test 1", "Integration test 2", "Integration test 3"]
        metadatas = [{"source": "int1"}, {"source": "int2"}, {"source": "int3"}]
        
        new_vectorstore = OceanbaseVectorStore.from_texts(
            texts=texts,
            embedding=vectorstore.embedding_function,
            metadatas=metadatas,
            table_name="from_texts_integration",
            connection_args=vectorstore.connection_args,
            vidx_metric_type="l2",
            drop_old=True,
            embedding_dim=6,
        )
        
        # Test search
        results = new_vectorstore.similarity_search("Integration", k=2)
        assert len(results) == 2
        assert any("Integration test" in doc.page_content for doc in results)
    
    def test_different_metric_types(self, vectorstore):
        """Test different metric types"""
        metric_types = ["l2", "inner_product", "cosine"]
        
        for metric_type in metric_types:
            test_vectorstore = OceanbaseVectorStore(
                embedding_function=vectorstore.embedding_function,
                table_name=f"metric_test_{metric_type}",
                connection_args=vectorstore.connection_args,
                vidx_metric_type=metric_type,
                drop_old=True,
                embedding_dim=6,
            )
            
            documents = [
                Document(page_content=f"Test for {metric_type}", metadata={"metric": metric_type}),
            ]
            
            ids = test_vectorstore.add_documents(documents)
            assert len(ids) == 1
            
            results = test_vectorstore.similarity_search("Test", k=1)
            assert len(results) == 1
            assert metric_type in results[0].page_content
    
    def test_different_index_types(self, vectorstore):
        """Test different index types"""
        index_types = ["HNSW", "IVF", "FLAT"]
        
        for index_type in index_types:
            test_vectorstore = OceanbaseVectorStore(
                embedding_function=vectorstore.embedding_function,
                table_name=f"index_test_{index_type.lower()}",
                connection_args=vectorstore.connection_args,
                vidx_metric_type="l2",
                drop_old=True,
                embedding_dim=6,
                index_type=index_type,
            )
            
            documents = [
                Document(page_content=f"Test for {index_type} index", metadata={"index": index_type}),
            ]
            
            ids = test_vectorstore.add_documents(documents)
            assert len(ids) == 1
            
            results = test_vectorstore.similarity_search("Test", k=1)
            assert len(results) == 1
            assert index_type in results[0].page_content
    
    def test_empty_search(self, vectorstore):
        """Test search behavior with empty vectorstore"""
        # Search in empty vectorstore
        results = vectorstore.similarity_search("test", k=5)
        assert len(results) == 0
        
        results_with_score = vectorstore.similarity_search_with_score("test", k=5)
        assert len(results_with_score) == 0
    
    def test_metadata_preservation(self, vectorstore):
        """Test that metadata is preserved correctly"""
        documents = [
            Document(page_content="Content 1", metadata={"key1": "value1", "key2": "value2"}),
            Document(page_content="Content 2", metadata={"key3": "value3"}),
        ]
        
        ids = vectorstore.add_documents(documents)
        
        # Retrieve and check metadata
        retrieved_docs = vectorstore.get_by_ids(ids)
        assert len(retrieved_docs) == 2
        
        # Check first document metadata
        doc1 = retrieved_docs[0]
        assert doc1.metadata["key1"] == "value1"
        assert doc1.metadata["key2"] == "value2"
        
        # Check second document metadata
        doc2 = retrieved_docs[1]
        assert doc2.metadata["key3"] == "value3"
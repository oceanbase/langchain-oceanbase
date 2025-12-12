"""
Tests for OceanBase Vector Store hybrid search capabilities.

This module contains comprehensive tests for the hybrid search features including:
- Sparse vector search with various configurations
- Full-text search with different query types
- Advanced hybrid search combining multiple modalities
- Result fusion algorithm validation
- Performance and edge case testing
- Error handling for unsupported configurations
- Integration tests with real-world scenarios
"""

import os
import time
from typing import Any, Dict, List

import pytest
from langchain_community.embeddings import FakeEmbeddings
from langchain_core.documents import Document

from langchain_oceanbase.vectorstores import OceanbaseVectorStore


class TestHybridSearch:
    """Test class for hybrid search functionality."""

    @pytest.fixture
    def connection_args(self):
        """Standard connection arguments for OceanBase/SeekDB."""
        return {
            "host": os.getenv("SEEKDB_HOST") or os.getenv("OB_HOST", "127.0.0.1"),
            "port": os.getenv("SEEKDB_PORT") or os.getenv("OB_PORT", "2881"),
            "user": os.getenv("SEEKDB_USER") or os.getenv("OB_USER", "root@test"),
            "password": os.getenv("SEEKDB_PASSWORD") or os.getenv("OB_PASSWORD", ""),
            "db_name": os.getenv("SEEKDB_DB") or os.getenv("OB_DB", "test"),
        }

    @pytest.fixture
    def embeddings(self):
        """Standard embeddings for testing."""
        return FakeEmbeddings(size=6)

    @pytest.fixture
    def hybrid_vectorstore(self, connection_args, embeddings):
        """Create a vector store with hybrid search capabilities."""
        return OceanbaseVectorStore(
            embedding_function=embeddings,
            table_name="hybrid_search_test",
            connection_args=connection_args,
            vidx_metric_type="l2",
            index_type="FLAT",
            include_sparse=True,
            include_fulltext=True,
            drop_old=True,
            embedding_dim=6,
        )

    @pytest.fixture
    def sample_documents(self):
        """Sample documents for testing."""
        return [
            Document(
                page_content="Machine learning is a subset of artificial intelligence",
                metadata={"topic": "AI", "author": "John Doe", "year": 2024},
            ),
            Document(
                page_content="Deep learning uses neural networks for pattern recognition",
                metadata={"topic": "AI", "author": "Jane Smith", "year": 2023},
            ),
            Document(
                page_content="Natural language processing enables computers to understand human language",
                metadata={"topic": "NLP", "author": "Bob Wilson", "year": 2024},
            ),
            Document(
                page_content="Computer vision allows machines to interpret visual information",
                metadata={"topic": "CV", "author": "Alice Brown", "year": 2023},
            ),
            Document(
                page_content="Python programming language is popular for data science",
                metadata={
                    "topic": "Programming",
                    "author": "Charlie Davis",
                    "year": 2024,
                },
            ),
        ]

    @pytest.fixture
    def sample_sparse_embeddings(self):
        """Sample sparse embeddings for testing."""
        return [
            {1: 0.8, 5: 0.6, 10: 0.4, 15: 0.2},
            {2: 0.7, 6: 0.5, 11: 0.3, 16: 0.1},
            {3: 0.9, 7: 0.4, 12: 0.6, 17: 0.3},
            {4: 0.6, 8: 0.8, 13: 0.2, 18: 0.5},
            {5: 0.5, 9: 0.3, 14: 0.7, 19: 0.4},
        ]

    @pytest.fixture
    def sample_fulltext_content(self):
        """Sample full-text content for testing."""
        return [
            "Machine learning algorithms enable computers to learn from data without explicit programming",
            "Deep learning neural networks can process complex patterns in images and text",
            "Natural language processing combines computational linguistics with machine learning",
            "Computer vision systems can identify objects and scenes in digital images",
            "Python programming language provides powerful libraries for data analysis and visualization",
        ]

    def test_hybrid_search_initialization(self, hybrid_vectorstore):
        """Test initialization with hybrid search features."""
        assert hybrid_vectorstore.include_sparse is True
        assert hybrid_vectorstore.include_fulltext is True
        assert hybrid_vectorstore.sparse_vector_field == "sparse_embedding"
        assert hybrid_vectorstore.fulltext_field == "fulltext_content"

    def test_add_sparse_documents(self, hybrid_vectorstore):
        """Test adding documents with sparse vector embeddings."""
        documents = [
            Document(
                page_content="Machine learning algorithms", metadata={"topic": "AI"}
            ),
            Document(
                page_content="Deep learning neural networks", metadata={"topic": "AI"}
            ),
        ]

        sparse_embeddings = [
            {1: 0.8, 5: 0.6, 10: 0.4},
            {2: 0.7, 6: 0.5, 11: 0.3},
        ]

        ids = hybrid_vectorstore.add_sparse_documents(
            documents=documents,
            sparse_embeddings=sparse_embeddings,
        )

        assert len(ids) == 2
        assert all(isinstance(id, str) for id in ids)

    def test_similarity_search_with_sparse_vector(self, hybrid_vectorstore):
        """Test similarity search using sparse vectors."""
        documents = [
            Document(
                page_content="Machine learning is a subset of AI",
                metadata={"topic": "AI"},
            ),
            Document(
                page_content="Deep learning uses neural networks",
                metadata={"topic": "AI"},
            ),
        ]

        sparse_embeddings = [
            {1: 0.8, 5: 0.6, 10: 0.4},
            {2: 0.7, 6: 0.5, 11: 0.3},
        ]

        hybrid_vectorstore.add_sparse_documents(
            documents=documents,
            sparse_embeddings=sparse_embeddings,
        )

        sparse_query = {1: 0.9, 5: 0.7, 10: 0.5}
        results = hybrid_vectorstore.similarity_search_with_sparse_vector(
            sparse_query=sparse_query,
            k=2,
        )

        assert len(results) >= 1
        assert all(isinstance(doc, Document) for doc in results)

    def test_add_documents_with_fulltext(self, hybrid_vectorstore):
        """Test adding documents with full-text content."""
        documents = [
            Document(
                page_content="Python programming", metadata={"language": "Python"}
            ),
            Document(
                page_content="JavaScript development",
                metadata={"language": "JavaScript"},
            ),
        ]

        fulltext_content = [
            "Python programming language features syntax simplicity and extensive libraries for data science",
            "JavaScript enables interactive web pages and server-side development with Node.js framework",
        ]

        ids = hybrid_vectorstore.add_documents_with_fulltext(
            documents=documents,
            fulltext_content=fulltext_content,
        )

        assert len(ids) == 2
        assert all(isinstance(id, str) for id in ids)

    def test_similarity_search_with_fulltext(self, hybrid_vectorstore):
        """Test hybrid search combining vector similarity and full-text search."""
        documents = [
            Document(
                page_content="Python programming", metadata={"language": "Python"}
            ),
            Document(
                page_content="JavaScript development",
                metadata={"language": "JavaScript"},
            ),
        ]

        fulltext_content = [
            "Python programming language features syntax simplicity and extensive libraries",
            "JavaScript enables interactive web pages and server-side development",
        ]

        hybrid_vectorstore.add_documents_with_fulltext(
            documents=documents,
            fulltext_content=fulltext_content,
        )

        results = hybrid_vectorstore.similarity_search_with_fulltext(
            query="programming language",
            fulltext_query="web development",
            k=2,
        )

        assert len(results) >= 1
        assert all(isinstance(doc, Document) for doc in results)

    def test_advanced_hybrid_search_vector_only(self, hybrid_vectorstore):
        """Test advanced hybrid search with vector similarity only."""
        documents = [
            Document(
                page_content="AI and machine learning research",
                metadata={"category": "AI"},
            ),
            Document(
                page_content="Deep learning neural networks",
                metadata={"category": "AI"},
            ),
        ]

        sparse_embeddings = [
            {1: 0.8, 5: 0.6, 10: 0.4},
            {2: 0.7, 6: 0.5, 11: 0.3},
        ]

        hybrid_vectorstore.add_sparse_documents(
            documents=documents,
            sparse_embeddings=sparse_embeddings,
        )

        results = hybrid_vectorstore.advanced_hybrid_search(
            vector_query="artificial intelligence",
            sparse_query={1: 0.8, 5: 0.6},
            k=2,
        )

        assert len(results) >= 1
        assert all(isinstance(doc, Document) for doc in results)

    def test_similarity_search_with_advanced_filters(self, hybrid_vectorstore):
        """Test similarity search with advanced filtering capabilities."""
        documents = [
            Document(
                page_content="Advanced AI research",
                metadata={"author": "John Doe", "year": 2024, "tags": ["AI", "ML"]},
            ),
            Document(
                page_content="Machine learning algorithms",
                metadata={"author": "Jane Smith", "year": 2023, "tags": ["ML", "NLP"]},
            ),
        ]

        fulltext_content = [
            "Advanced artificial intelligence research in machine learning and deep learning",
            "Machine learning algorithms for natural language processing and computer vision",
        ]

        hybrid_vectorstore.add_documents_with_fulltext(
            documents=documents,
            fulltext_content=fulltext_content,
        )

        filters = {
            "fulltext": "machine learning",
        }

        results = hybrid_vectorstore.similarity_search_with_advanced_filters(
            query="artificial intelligence",
            filters=filters,
            k=2,
        )

        assert len(results) >= 1
        assert all(isinstance(doc, Document) for doc in results)

    def test_error_handling_sparse_not_enabled(self, connection_args, embeddings):
        """Test error handling when sparse vector support is not enabled."""
        vector_store = OceanbaseVectorStore(
            embedding_function=embeddings,
            table_name="test_vector_no_sparse",
            connection_args=connection_args,
            include_sparse=False,
            include_fulltext=False,
            drop_old=True,
            embedding_dim=6,
            index_type="FLAT",
        )

        with pytest.raises(ValueError, match="Sparse vector support not enabled"):
            vector_store.add_sparse_documents(
                documents=[Document(page_content="test")],
                sparse_embeddings=[{1: 0.5}],
            )

    def test_error_handling_fulltext_not_enabled(self, connection_args, embeddings):
        """Test error handling when full-text search support is not enabled."""
        vector_store = OceanbaseVectorStore(
            embedding_function=embeddings,
            table_name="test_vector_no_fulltext",
            connection_args=connection_args,
            include_sparse=False,
            include_fulltext=False,
            drop_old=True,
            embedding_dim=6,
            index_type="FLAT",
        )

        with pytest.raises(ValueError, match="Full-text search support not enabled"):
            vector_store.similarity_search_with_fulltext(
                query="test query",
                fulltext_query="fulltext search",
            )

    def test_error_handling_no_search_modality(self, hybrid_vectorstore):
        """Test error handling when no search modality is provided."""
        with pytest.raises(
            ValueError, match="At least one search modality must be provided"
        ):
            hybrid_vectorstore.advanced_hybrid_search(k=10)

    def test_basic_vector_search_still_works(self, hybrid_vectorstore):
        """Test that basic vector search functionality still works with hybrid features enabled."""
        documents = [
            Document(page_content="Test document 1", metadata={"source": "test1"}),
            Document(page_content="Test document 2", metadata={"source": "test2"}),
        ]

        ids = hybrid_vectorstore.add_documents(documents)
        assert len(ids) == 2

        results = hybrid_vectorstore.similarity_search("test document", k=2)
        assert len(results) >= 1
        assert all(isinstance(doc, Document) for doc in results)


    def test_comprehensive_hybrid_search_workflow(
        self,
        hybrid_vectorstore,
        sample_documents,
        sample_sparse_embeddings,
        sample_fulltext_content,
    ):
        """Test comprehensive workflow with all hybrid search features."""
        ids = hybrid_vectorstore.add_documents_with_fulltext(
            documents=sample_documents,
            fulltext_content=sample_fulltext_content,
        )
        assert len(ids) == 5

        hybrid_vectorstore.add_sparse_documents(
            documents=sample_documents,
            sparse_embeddings=sample_sparse_embeddings,
        )

        vector_results = hybrid_vectorstore.similarity_search("machine learning", k=3)
        sparse_results = hybrid_vectorstore.similarity_search_with_sparse_vector(
            sparse_query={1: 0.8, 5: 0.6}, k=3
        )
        fulltext_results = hybrid_vectorstore.similarity_search_with_fulltext(
            query="artificial intelligence", fulltext_query="neural networks", k=3
        )

        assert len(vector_results) >= 1
        assert len(sparse_results) >= 1
        assert len(fulltext_results) >= 1

        hybrid_results = hybrid_vectorstore.advanced_hybrid_search(
            vector_query="AI technology",
            sparse_query={1: 0.8, 5: 0.6},
            fulltext_query="machine learning",
            k=5,
        )

        assert len(hybrid_results) >= 1
        assert all(isinstance(doc, Document) for doc in hybrid_results)

    def test_result_fusion_algorithm_validation(
        self,
        hybrid_vectorstore,
        sample_documents,
        sample_sparse_embeddings,
        sample_fulltext_content,
    ):
        """Test that result fusion algorithm works correctly."""
        hybrid_vectorstore.add_documents_with_fulltext(
            documents=sample_documents,
            fulltext_content=sample_fulltext_content,
        )
        hybrid_vectorstore.add_sparse_documents(
            documents=sample_documents,
            sparse_embeddings=sample_sparse_embeddings,
        )

        dual_results = hybrid_vectorstore.similarity_search_with_fulltext(
            query="machine learning", fulltext_query="artificial intelligence", k=3
        )

        multi_results = hybrid_vectorstore.advanced_hybrid_search(
            vector_query="AI research",
            sparse_query={1: 0.8, 5: 0.6},
            fulltext_query="neural networks",
            k=3,
        )

        assert len(dual_results) <= 3
        assert len(multi_results) <= 3

        for doc in dual_results + multi_results:
            assert isinstance(doc.metadata, dict)
            assert "topic" in doc.metadata or "author" in doc.metadata

    def test_performance_with_large_dataset(self, hybrid_vectorstore):
        """Test performance with a larger dataset."""
        large_documents = []
        large_sparse_embeddings = []
        large_fulltext_content = []

        for i in range(20):  # 20 documents
            large_documents.append(
                Document(
                    page_content=f"Document {i} about machine learning and AI",
                    metadata={"id": i, "category": "AI" if i % 2 == 0 else "ML"},
                )
            )
            large_sparse_embeddings.append(
                {i % 10: 0.8, (i + 1) % 10: 0.6, (i + 2) % 10: 0.4}
            )
            large_fulltext_content.append(
                f"Comprehensive analysis of machine learning algorithms and artificial intelligence applications in document {i}"
            )

        start_time = time.time()
        hybrid_vectorstore.add_documents_with_fulltext(
            documents=large_documents,
            fulltext_content=large_fulltext_content,
        )
        insertion_time = time.time() - start_time

        start_time = time.time()
        results = hybrid_vectorstore.advanced_hybrid_search(
            vector_query="machine learning",
            sparse_query={0: 0.8, 1: 0.6},
            fulltext_query="artificial intelligence",
            k=10,
        )
        search_time = time.time() - start_time

        assert insertion_time < 10.0
        assert search_time < 2.0
        assert len(results) >= 1

    def test_edge_cases_and_boundary_conditions(self, hybrid_vectorstore):
        """Test edge cases and boundary conditions."""
        empty_docs = [Document(page_content="", metadata={})]
        ids = hybrid_vectorstore.add_documents(empty_docs)
        assert len(ids) == 1

        long_content = "A" * 10000
        long_doc = Document(page_content=long_content, metadata={"type": "long"})
        ids = hybrid_vectorstore.add_documents([long_doc])
        assert len(ids) == 1

        special_doc = Document(
            page_content="Test with special characters",
            metadata={
                "special": "测试中文",
                "unicode": "🚀",
                "json": '{"key": "value"}',
            },
        )
        ids = hybrid_vectorstore.add_documents([special_doc])
        assert len(ids) == 1

        results = hybrid_vectorstore.similarity_search("", k=1)
        assert isinstance(results, list)  # Should return a list (may be empty)

    def test_metadata_filtering_integration(
        self, hybrid_vectorstore, sample_documents, sample_fulltext_content
    ):
        """Test metadata filtering integration with hybrid search."""
        hybrid_vectorstore.add_documents_with_fulltext(
            documents=sample_documents,
            fulltext_content=sample_fulltext_content,
        )

        filters = {"fulltext": "machine learning"}

        results = hybrid_vectorstore.similarity_search_with_advanced_filters(
            query="artificial intelligence", filters=filters, k=5
        )

        assert len(results) >= 1
        assert all(isinstance(doc, Document) for doc in results)

    def test_concurrent_search_operations(
        self,
        hybrid_vectorstore,
        sample_documents,
        sample_sparse_embeddings,
        sample_fulltext_content,
    ):
        """Test concurrent search operations."""
        hybrid_vectorstore.add_documents_with_fulltext(
            documents=sample_documents,
            fulltext_content=sample_fulltext_content,
        )
        hybrid_vectorstore.add_sparse_documents(
            documents=sample_documents,
            sparse_embeddings=sample_sparse_embeddings,
        )

        queries = [
            ("machine learning", {1: 0.8, 5: 0.6}, "artificial intelligence"),
            ("deep learning", {2: 0.7, 6: 0.5}, "neural networks"),
            ("natural language", {3: 0.9, 7: 0.4}, "language processing"),
        ]

        all_results = []
        for vector_q, sparse_q, fulltext_q in queries:
            results = hybrid_vectorstore.advanced_hybrid_search(
                vector_query=vector_q,
                sparse_query=sparse_q,
                fulltext_query=fulltext_q,
                k=2,
            )
            all_results.extend(results)

        assert len(all_results) >= 3
        assert all(isinstance(doc, Document) for doc in all_results)

    def test_error_recovery_and_robustness(self, hybrid_vectorstore):
        """Test error recovery and system robustness."""
        with pytest.raises((ValueError, TypeError, Exception)):
            hybrid_vectorstore.add_sparse_documents(
                documents=[Document(page_content="test")],
                sparse_embeddings=[{"invalid": "sparse_vector"}],
            )

        with pytest.raises(ValueError):
            hybrid_vectorstore.add_sparse_documents(
                documents=[
                    Document(page_content="doc1"),
                    Document(page_content="doc2"),
                ],
                sparse_embeddings=[{1: 0.5}],
            )

        results_neg = hybrid_vectorstore.similarity_search("test", k=-1)
        assert isinstance(results_neg, list)
        assert len(results_neg) == 0

        with pytest.raises((TypeError, ValueError)):
            hybrid_vectorstore.similarity_search("test", k="invalid")

    def test_advanced_hybrid_search_with_custom_weights(
        self,
        hybrid_vectorstore,
        sample_documents,
        sample_sparse_embeddings,
        sample_fulltext_content,
    ):
        """Test advanced hybrid search with custom modality weights."""
        hybrid_vectorstore.add_documents_with_fulltext(
            documents=sample_documents,
            fulltext_content=sample_fulltext_content,
        )
        hybrid_vectorstore.add_sparse_documents(
            documents=sample_documents,
            sparse_embeddings=sample_sparse_embeddings,
        )

        custom_weights = {
            "vector": 0.8,
            "sparse": 0.1,
            "fulltext": 0.1,
        }

        results = hybrid_vectorstore.advanced_hybrid_search(
            vector_query="AI technology",
            sparse_query={1: 0.8, 5: 0.6},
            fulltext_query="machine learning",
            k=3,
            modality_weights=custom_weights,
        )

        assert len(results) >= 1
        assert all(isinstance(doc, Document) for doc in results)

        sparse_weights = {
            "vector": 0.2,
            "sparse": 0.7,
            "fulltext": 0.1,
        }

        sparse_results = hybrid_vectorstore.advanced_hybrid_search(
            vector_query="AI technology",
            sparse_query={1: 0.8, 5: 0.6},
            fulltext_query="machine learning",
            k=3,
            modality_weights=sparse_weights,
        )

        assert len(sparse_results) >= 1
        assert all(isinstance(doc, Document) for doc in sparse_results)

    def test_advanced_hybrid_search_weight_validation(self, hybrid_vectorstore):
        """Test weight validation in advanced hybrid search."""
        with pytest.raises(ValueError, match="Modality weights must sum to 1.0"):
            hybrid_vectorstore.advanced_hybrid_search(
                vector_query="test",
                modality_weights={
                    "vector": 0.5,
                    "sparse": 0.3,
                    "fulltext": 0.1,
                },
            )

        valid_weights = {"vector": 0.6, "sparse": 0.3, "fulltext": 0.1}

        results = hybrid_vectorstore.advanced_hybrid_search(
            vector_query="test", modality_weights=valid_weights
        )
        assert isinstance(results, list)

    def test_advanced_hybrid_search_partial_weights(
        self, hybrid_vectorstore, sample_documents
    ):
        """Test advanced hybrid search with partial weight specification."""
        hybrid_vectorstore.add_documents(sample_documents)

        partial_weights = {
            "vector": 1.0,
        }

        results = hybrid_vectorstore.advanced_hybrid_search(
            vector_query="machine learning", k=3, modality_weights=partial_weights
        )

        assert len(results) >= 1
        assert all(isinstance(doc, Document) for doc in results)

    def test_integration_with_real_world_scenarios(self, hybrid_vectorstore):
        """Test integration with real-world usage scenarios."""
        academic_docs = [
            Document(
                page_content="Attention mechanisms in transformer architectures",
                metadata={"venue": "NeurIPS", "year": 2023, "citations": 150},
            ),
            Document(
                page_content="BERT: Pre-training of Deep Bidirectional Transformers",
                metadata={"venue": "NAACL", "year": 2019, "citations": 5000},
            ),
        ]

        academic_fulltext = [
            "This paper introduces novel attention mechanisms that improve transformer model performance",
            "BERT demonstrates state-of-the-art results on multiple natural language understanding tasks",
        ]

        hybrid_vectorstore.add_documents_with_fulltext(
            documents=academic_docs,
            fulltext_content=academic_fulltext,
        )

        results = hybrid_vectorstore.similarity_search_with_fulltext(
            query="transformer architecture", fulltext_query="attention mechanism", k=2
        )

        assert len(results) >= 1

        product_docs = [
            Document(
                page_content="Wireless Bluetooth headphones with noise cancellation",
                metadata={
                    "category": "Electronics",
                    "price": 199.99,
                    "brand": "TechCorp",
                },
            ),
            Document(
                page_content="Smart fitness tracker with heart rate monitoring",
                metadata={"category": "Wearables", "price": 149.99, "brand": "FitTech"},
            ),
        ]

        product_fulltext = [
            "Premium wireless headphones featuring active noise cancellation and 30-hour battery life",
            "Advanced fitness tracker with continuous heart rate monitoring and sleep tracking",
        ]

        hybrid_vectorstore.add_documents_with_fulltext(
            documents=product_docs,
            fulltext_content=product_fulltext,
        )

        results = hybrid_vectorstore.similarity_search_with_fulltext(
            query="wireless technology", fulltext_query="bluetooth headphones", k=2
        )

        assert len(results) >= 1

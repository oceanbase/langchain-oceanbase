"""Integration tests for embedding vectorization capabilities.

This module tests the basic vectorization capabilities of embedding utilities,
including DefaultEmbeddingFunction and DefaultEmbeddingFunctionAdapter.
These tests verify that embeddings can be generated correctly without requiring
database connections.

Note: pyseekdb is a required dependency (defined in pyproject.toml), so these
tests should always run in CI environments. If pyseekdb is not available,
the tests will fail, which helps identify dependency issues early.
"""

import pytest

from langchain_oceanbase.embedding_utils import (
    DefaultEmbeddingFunction,
    DefaultEmbeddingFunctionAdapter,
)

# Check if pyseekdb is available
try:
    from pyseekdb import DefaultEmbeddingFunction as PySeekDBDefaultEmbeddingFunction
except ImportError:
    PySeekDBDefaultEmbeddingFunction = None


@pytest.mark.skipif(
    PySeekDBDefaultEmbeddingFunction is None or DefaultEmbeddingFunction is None,
    reason="pyseekdb is not installed or not available on this platform",
)
class TestEmbeddingVectorization:
    """Test basic embedding vectorization capabilities."""

    def test_embedding_imports(self):
        """Test that embedding utilities can be imported."""
        # If this test runs, imports are successful
        assert DefaultEmbeddingFunction is not None
        assert DefaultEmbeddingFunctionAdapter is not None

    def test_default_embedding_function_creation(self):
        """Test creating DefaultEmbeddingFunction instance."""
        ef = DefaultEmbeddingFunction()
        assert ef is not None
        assert hasattr(ef, "dimension")
        assert isinstance(ef.dimension, int)
        assert ef.dimension > 0

    def test_single_text_embedding(self):
        """Test single text embedding generation."""
        ef = DefaultEmbeddingFunction()
        text = "Hello world"
        result = ef(text)
        assert isinstance(result, list), "Result should be a list"
        assert len(result) == 1, "Should return one embedding"
        assert isinstance(result[0], list), "Embedding should be a list"
        assert (
            len(result[0]) == ef.dimension
        ), f"Embedding dimension should be {ef.dimension}"
        assert all(
            isinstance(x, float) for x in result[0]
        ), "All values should be floats"

    def test_multiple_texts_embedding(self):
        """Test multiple texts embedding generation."""
        ef = DefaultEmbeddingFunction()
        texts = ["Hello world", "How are you?", "Python programming"]
        results = ef(texts)
        assert isinstance(results, list), "Results should be a list"
        assert len(results) == len(texts), f"Should return {len(texts)} embeddings"
        for i, emb in enumerate(results):
            assert isinstance(emb, list), f"Embedding {i} should be a list"
            assert (
                len(emb) == ef.dimension
            ), f"Embedding {i} dimension should be {ef.dimension}"

    def test_dimension_consistency(self):
        """Test that all embeddings have consistent dimensions."""
        ef = DefaultEmbeddingFunction()
        test_texts = ["Text 1", "Text 2", "Text 3"]
        test_results = ef(test_texts)
        for i, emb in enumerate(test_results):
            assert len(emb) == ef.dimension, f"Embedding {i} has inconsistent dimension"

    def test_adapter_creation(self):
        """Test creating DefaultEmbeddingFunctionAdapter instance."""
        adapter = DefaultEmbeddingFunctionAdapter()
        assert adapter is not None
        assert hasattr(adapter, "dimension")
        assert isinstance(adapter.dimension, int)
        assert adapter.dimension > 0

    def test_adapter_embed_documents(self):
        """Test adapter embed_documents method."""
        adapter = DefaultEmbeddingFunctionAdapter()
        documents = ["Hello world", "How are you?", "Python programming"]
        doc_embeddings = adapter.embed_documents(documents)
        assert isinstance(doc_embeddings, list), "Document embeddings should be a list"
        assert len(doc_embeddings) == len(
            documents
        ), f"Should return {len(documents)} embeddings"
        for i, emb in enumerate(doc_embeddings):
            assert isinstance(emb, list), f"Embedding {i} should be a list"
            assert (
                len(emb) == adapter.dimension
            ), f"Embedding {i} dimension should be {adapter.dimension}"
            assert all(
                isinstance(x, float) for x in emb
            ), f"All values in embedding {i} should be floats"

    def test_adapter_embed_query(self):
        """Test adapter embed_query method."""
        adapter = DefaultEmbeddingFunctionAdapter()
        query = "Hello world"
        query_embedding = adapter.embed_query(query)
        assert isinstance(query_embedding, list), "Query embedding should be a list"
        assert (
            len(query_embedding) == adapter.dimension
        ), f"Query embedding dimension should be {adapter.dimension}"
        assert all(
            isinstance(x, float) for x in query_embedding
        ), "All values should be floats"

    def test_adapter_consistency(self):
        """Test consistency between embed_query and embed_documents."""
        adapter = DefaultEmbeddingFunctionAdapter()
        test_text = "Test consistency"
        query_emb = adapter.embed_query(test_text)
        doc_emb = adapter.embed_documents([test_text])[0]
        assert len(query_emb) == len(
            doc_emb
        ), "Query and document embeddings should have same dimension"
        assert (
            query_emb == doc_emb
        ), "Query and document embeddings should be identical for same text"

    def test_similarity_computation(self):
        """Test that embeddings can be used for similarity computation."""
        adapter = DefaultEmbeddingFunctionAdapter()
        text1 = "Machine learning is a subset of artificial intelligence"
        text2 = "AI and machine learning are related technologies"
        text3 = "The weather today is sunny and warm"
        emb1 = adapter.embed_query(text1)
        emb2 = adapter.embed_query(text2)
        emb3 = adapter.embed_query(text3)

        # Compute cosine similarity
        def cosine_similarity(vec1, vec2):
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            norm1 = sum(a * a for a in vec1) ** 0.5
            norm2 = sum(b * b for b in vec2) ** 0.5
            return dot_product / (norm1 * norm2) if norm1 * norm2 > 0 else 0.0

        sim_12 = cosine_similarity(emb1, emb2)
        sim_13 = cosine_similarity(emb1, emb3)
        # Similar texts should have higher similarity
        assert sim_12 > sim_13, "Similar texts should have higher similarity"

    def test_empty_list_handling(self):
        """Test handling of empty input."""
        ef = DefaultEmbeddingFunction()
        result = ef([])
        assert isinstance(result, list), "Result should be a list"
        assert len(result) == 0, "Should return empty list for empty input"

        adapter = DefaultEmbeddingFunctionAdapter()
        doc_embeddings = adapter.embed_documents([])
        assert isinstance(doc_embeddings, list), "Document embeddings should be a list"
        assert len(doc_embeddings) == 0, "Should return empty list for empty input"

    def test_special_characters(self):
        """Test embedding generation with special characters."""
        adapter = DefaultEmbeddingFunctionAdapter()
        texts = [
            "Hello 世界",
            "Test with émojis 🚀",
            "Special chars: !@#$%^&*()",
        ]
        embeddings = adapter.embed_documents(texts)
        assert len(embeddings) == len(texts)
        for emb in embeddings:
            assert len(emb) == adapter.dimension

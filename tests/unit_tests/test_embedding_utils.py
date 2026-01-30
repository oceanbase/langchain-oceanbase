"""Unit tests for embedding_utils module.

This module contains comprehensive unit tests for:
- DefaultEmbeddingFunction re-export
- DefaultEmbeddingFunctionAdapter class
- Error handling and edge cases
- Integration with LangChain Embeddings interface
"""

import pytest

# Try to import pyseekdb first to check availability
try:
    from pyseekdb import DefaultEmbeddingFunction as PySeekDBDefaultEmbeddingFunction

    PYSEEKDB_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    PySeekDBDefaultEmbeddingFunction = None
    PYSEEKDB_AVAILABLE = False

# Import embedding_utils module directly (not through __init__.py to avoid circular imports)
# Always import the module, even if pyseekdb is not available, for error handling tests
try:
    import langchain_oceanbase.embedding_utils as embedding_utils_module
except (ImportError, ModuleNotFoundError):
    embedding_utils_module = None

if PYSEEKDB_AVAILABLE and embedding_utils_module is not None:
    DefaultEmbeddingFunction = embedding_utils_module.DefaultEmbeddingFunction
    DefaultEmbeddingFunctionAdapter = (
        embedding_utils_module.DefaultEmbeddingFunctionAdapter
    )
else:
    DefaultEmbeddingFunction = None
    DefaultEmbeddingFunctionAdapter = None


@pytest.mark.skipif(
    PySeekDBDefaultEmbeddingFunction is None or DefaultEmbeddingFunction is None,
    reason="pyseekdb is not installed",
)
class TestDefaultEmbeddingFunction:
    """Test class for DefaultEmbeddingFunction re-export."""

    def test_default_embedding_function_import(self):
        """Test that DefaultEmbeddingFunction can be imported."""
        assert DefaultEmbeddingFunction is not None
        assert DefaultEmbeddingFunction == PySeekDBDefaultEmbeddingFunction

    def test_default_embedding_function_creation(self):
        """Test creating DefaultEmbeddingFunction instance."""
        ef = DefaultEmbeddingFunction()
        assert ef is not None
        assert hasattr(ef, "dimension")
        assert isinstance(ef.dimension, int)
        assert ef.dimension > 0

    def test_default_embedding_function_single_text(self):
        """Test DefaultEmbeddingFunction with single text."""
        ef = DefaultEmbeddingFunction()
        text = "Hello world"
        result = ef(text)
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], list)
        assert len(result[0]) == ef.dimension

    def test_default_embedding_function_multiple_texts(self):
        """Test DefaultEmbeddingFunction with multiple texts."""
        ef = DefaultEmbeddingFunction()
        texts = ["Hello world", "How are you?", "Python programming"]
        result = ef(texts)
        assert isinstance(result, list)
        assert len(result) == len(texts)
        for emb in result:
            assert isinstance(emb, list)
            assert len(emb) == ef.dimension

    def test_default_embedding_function_dimension_consistency(self):
        """Test that all embeddings have consistent dimensions."""
        ef = DefaultEmbeddingFunction()
        texts = ["Text 1", "Text 2", "Text 3"]
        result = ef(texts)
        for emb in result:
            assert len(emb) == ef.dimension

    def test_default_embedding_function_empty_list(self):
        """Test DefaultEmbeddingFunction with empty list."""
        ef = DefaultEmbeddingFunction()
        result = ef([])
        assert isinstance(result, list)
        assert len(result) == 0


@pytest.mark.skipif(
    PySeekDBDefaultEmbeddingFunction is None or DefaultEmbeddingFunctionAdapter is None,
    reason="pyseekdb is not installed",
)
class TestDefaultEmbeddingFunctionAdapter:
    """Test class for DefaultEmbeddingFunctionAdapter."""

    def test_adapter_creation(self):
        """Test creating DefaultEmbeddingFunctionAdapter instance."""
        adapter = DefaultEmbeddingFunctionAdapter()
        assert adapter is not None
        assert hasattr(adapter, "dimension")
        assert isinstance(adapter.dimension, int)
        assert adapter.dimension > 0

    def test_adapter_creation_with_model_name(self):
        """Test creating adapter with custom model name."""
        adapter = DefaultEmbeddingFunctionAdapter(model_name="all-MiniLM-L6-v2")
        assert adapter is not None
        assert adapter.dimension > 0

    def test_adapter_creation_with_preferred_providers(self):
        """Test creating adapter with preferred providers."""
        adapter = DefaultEmbeddingFunctionAdapter(
            preferred_providers=["CPUExecutionProvider"]
        )
        assert adapter is not None
        assert adapter.dimension > 0

    def test_adapter_dimension_property(self):
        """Test dimension property."""
        adapter = DefaultEmbeddingFunctionAdapter()
        dimension = adapter.dimension
        assert isinstance(dimension, int)
        assert dimension > 0

    def test_adapter_embed_documents(self):
        """Test embed_documents method."""
        adapter = DefaultEmbeddingFunctionAdapter()
        texts = ["Hello world", "How are you?", "Python programming"]
        embeddings = adapter.embed_documents(texts)
        assert isinstance(embeddings, list)
        assert len(embeddings) == len(texts)
        for emb in embeddings:
            assert isinstance(emb, list)
            assert len(emb) == adapter.dimension
            assert all(isinstance(x, float) for x in emb)

    def test_adapter_embed_documents_empty_list(self):
        """Test embed_documents with empty list."""
        adapter = DefaultEmbeddingFunctionAdapter()
        embeddings = adapter.embed_documents([])
        assert isinstance(embeddings, list)
        assert len(embeddings) == 0

    def test_adapter_embed_documents_single_text(self):
        """Test embed_documents with single text."""
        adapter = DefaultEmbeddingFunctionAdapter()
        texts = ["Single text"]
        embeddings = adapter.embed_documents(texts)
        assert isinstance(embeddings, list)
        assert len(embeddings) == 1
        assert len(embeddings[0]) == adapter.dimension

    def test_adapter_embed_query(self):
        """Test embed_query method."""
        adapter = DefaultEmbeddingFunctionAdapter()
        query = "Hello world"
        embedding = adapter.embed_query(query)
        assert isinstance(embedding, list)
        assert len(embedding) == adapter.dimension
        assert all(isinstance(x, float) for x in embedding)

    def test_adapter_embed_query_empty_string(self):
        """Test embed_query with empty string."""
        adapter = DefaultEmbeddingFunctionAdapter()
        embedding = adapter.embed_query("")
        assert isinstance(embedding, list)
        assert len(embedding) == adapter.dimension

    def test_adapter_embed_query_long_text(self):
        """Test embed_query with long text."""
        adapter = DefaultEmbeddingFunctionAdapter()
        long_text = " ".join(["word"] * 1000)
        embedding = adapter.embed_query(long_text)
        assert isinstance(embedding, list)
        assert len(embedding) == adapter.dimension

    def test_adapter_consistency_between_embed_query_and_embed_documents(self):
        """Test that embed_query and embed_documents produce consistent results."""
        adapter = DefaultEmbeddingFunctionAdapter()
        text = "Test text"
        query_emb = adapter.embed_query(text)
        doc_emb = adapter.embed_documents([text])[0]
        assert len(query_emb) == len(doc_emb)
        assert len(query_emb) == adapter.dimension
        assert query_emb == doc_emb

    def test_adapter_dimension_consistency(self):
        """Test that all embeddings have consistent dimensions."""
        adapter = DefaultEmbeddingFunctionAdapter()
        texts = ["Text 1", "Text 2", "Text 3"]
        doc_embeddings = adapter.embed_documents(texts)
        query_embedding = adapter.embed_query("Query text")
        assert all(len(emb) == adapter.dimension for emb in doc_embeddings)
        assert len(query_embedding) == adapter.dimension

    def test_adapter_similarity_computation(self):
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
        assert sim_12 > sim_13

    def test_adapter_special_characters(self):
        """Test adapter with special characters."""
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

    def test_adapter_unicode_text(self):
        """Test adapter with unicode text."""
        adapter = DefaultEmbeddingFunctionAdapter()
        unicode_text = "测试中文文本"
        embedding = adapter.embed_query(unicode_text)
        assert isinstance(embedding, list)
        assert len(embedding) == adapter.dimension


class TestDefaultEmbeddingFunctionAdapterErrorHandling:
    """Test error handling for DefaultEmbeddingFunctionAdapter."""

    @pytest.mark.skipif(
        not PYSEEKDB_AVAILABLE or embedding_utils_module is None,
        reason="pyseekdb is not installed or module cannot be imported, cannot test error handling",
    )
    def test_adapter_import_error_when_default_is_none(self):
        """Test ImportError when DefaultEmbeddingFunction is None."""
        import unittest.mock

        # Use mock instead of reload because reload will re-import from pyseekdb
        with unittest.mock.patch.object(
            embedding_utils_module, "DefaultEmbeddingFunction", None
        ):
            with pytest.raises(ImportError, match="pyseekdb is not installed"):
                embedding_utils_module.DefaultEmbeddingFunctionAdapter()


@pytest.mark.skipif(
    PySeekDBDefaultEmbeddingFunction is None or DefaultEmbeddingFunctionAdapter is None,
    reason="pyseekdb is not installed",
)
class TestDefaultEmbeddingFunctionAdapterIntegration:
    """Integration tests for DefaultEmbeddingFunctionAdapter."""

    def test_adapter_implements_embeddings_interface(self):
        """Test that adapter implements LangChain Embeddings interface."""
        from langchain_core.embeddings import Embeddings

        adapter = DefaultEmbeddingFunctionAdapter()
        assert isinstance(adapter, Embeddings)
        assert hasattr(adapter, "embed_documents")
        assert hasattr(adapter, "embed_query")
        assert callable(adapter.embed_documents)
        assert callable(adapter.embed_query)

    def test_adapter_batch_processing(self):
        """Test batch processing of multiple documents."""
        adapter = DefaultEmbeddingFunctionAdapter()
        texts = [f"Document {i}" for i in range(10)]
        embeddings = adapter.embed_documents(texts)
        assert len(embeddings) == 10
        assert all(len(emb) == adapter.dimension for emb in embeddings)

    def test_adapter_reproducibility(self):
        """Test that same text produces same embedding."""
        adapter = DefaultEmbeddingFunctionAdapter()
        text = "Test reproducibility"
        emb1 = adapter.embed_query(text)
        emb2 = adapter.embed_query(text)
        assert emb1 == emb2

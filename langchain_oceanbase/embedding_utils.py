"""Embedding utility module.

Re-exports pyseekdb's DefaultEmbeddingFunction for easy access.
Also provides an adapter to use DefaultEmbeddingFunction with LangChain's Embeddings interface.

Usage Examples:

1. Direct use of DefaultEmbeddingFunction:
    >>> from langchain_oceanbase import DefaultEmbeddingFunction
    >>> ef = DefaultEmbeddingFunction()
    >>> print(f"Dimension: {ef.dimension}")  # 384
    >>> embeddings = ef(["Hello world", "How are you?"])
    >>> print(f"Generated {len(embeddings)} embeddings")

2. Use in OceanbaseVectorStore (automatic default embedding):
    >>> from langchain_oceanbase import OceanbaseVectorStore
    >>> vector_store = OceanbaseVectorStore(
    ...     embedding_function=None,  # 自动使用默认 embedding
    ...     table_name="test",
    ...     connection_args={...},
    ...     embedding_dim=384,  # all-MiniLM-L6-v2 的维度
    ... )

3. Use adapter for LangChain compatibility:
    >>> from langchain_oceanbase.embedding_utils import DefaultEmbeddingFunctionAdapter
    >>> embeddings = DefaultEmbeddingFunctionAdapter()
    >>> doc_embeddings = embeddings.embed_documents(["Hello", "World"])
    >>> query_embedding = embeddings.embed_query("Hello")
"""

from __future__ import annotations

import logging
from typing import List, Optional

from langchain_core.embeddings import Embeddings

try:
    from pyseekdb import DefaultEmbeddingFunction
except ImportError:
    DefaultEmbeddingFunction = None  # type: ignore

logger = logging.getLogger(__name__)

__all__ = ["DefaultEmbeddingFunction", "DefaultEmbeddingFunctionAdapter"]


class DefaultEmbeddingFunctionAdapter(Embeddings):
    """
    Adapter class that wraps pyseekdb's DefaultEmbeddingFunction to implement LangChain's Embeddings interface.

    Example:
        >>> from langchain_oceanbase.embedding_utils import DefaultEmbeddingFunctionAdapter
        >>> ef = DefaultEmbeddingFunctionAdapter()
        >>> embeddings = ef.embed_documents(["Hello world", "How are you?"])
        >>> query_embedding = ef.embed_query("Hello")
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        preferred_providers: Optional[List[str]] = None,
    ):
        """
        Initialize the adapter.

        Args:
            model_name: Name of the model (currently only 'all-MiniLM-L6-v2' is supported).
            preferred_providers: The preferred ONNX runtime providers.
        """
        super().__init__()
        if DefaultEmbeddingFunction is None:
            import platform

            system = platform.system()
            error_msg = (
                "pyseekdb is not installed or cannot be imported. "
                "Please install it with: pip install pyseekdb"
            )
            if system != "Linux":
                error_msg += (
                    f"\n\nNote: pyseekdb currently only supports Linux platforms. "
                    f"Your system is {system}. "
                    "Consider using Linux (via Docker, WSL2, or a Linux VM) "
                    "or connecting to a remote OceanBase/SeekDB instance."
                )
            raise ImportError(error_msg)

        self._pyseekdb_embedding_function = DefaultEmbeddingFunction(
            model_name=model_name,
            preferred_providers=preferred_providers,
        )

    @property
    def dimension(self) -> int:
        """Get the dimension of embeddings produced by this function."""
        return self._pyseekdb_embedding_function.dimension

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        return self._pyseekdb_embedding_function(texts)

    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        result = self._pyseekdb_embedding_function(text)
        if not result:
            raise ValueError(
                "Failed to generate embedding for text. "
                "Expected a list with at least one embedding, got empty list."
            )
        return result[0]

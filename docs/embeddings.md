---
sidebar_label: Embeddings
---
# Built-in Embedding Function Guide

This guide demonstrates how to use the built-in embedding functionality provided by `langchain-oceanbase`. The built-in embedding uses the `all-MiniLM-L6-v2` model (384 dimensions) and requires no external API keys, making it perfect for quick prototyping and testing.

## Installation

The built-in embedding functionality requires the `pyseekdb` package, which uses ONNX runtime for local inference:

```bash
pip install -qU "langchain-oceanbase" "pyseekdb"
```

## Method 1: Direct Use of DefaultEmbeddingFunction

`DefaultEmbeddingFunction` is the default embedding function provided by `pyseekdb` and can be used directly:

```python
from langchain_oceanbase import DefaultEmbeddingFunction

# Create embedding function
ef = DefaultEmbeddingFunction()
print(f"Embedding dimension: {ef.dimension}")  # 384

# Generate embedding for a single text
single_text = "Hello world"
single_embedding = ef(single_text)
print(f"Single text embedding count: {len(single_embedding)}")
print(f"Single text embedding dimension: {len(single_embedding[0]) if single_embedding else 0}")

# Generate embeddings for multiple texts (batch processing)
texts = ["Hello world", "How are you?", "Python programming", "Machine learning"]
embeddings = ef(texts)
print(f"\nBatch text count: {len(texts)}")
print(f"Generated embedding count: {len(embeddings)}")
print(f"Each embedding dimension: {len(embeddings[0]) if embeddings else 0}")
```

## Method 2: Using DefaultEmbeddingFunctionAdapter (LangChain Compatible Interface)

`DefaultEmbeddingFunctionAdapter` implements LangChain's `Embeddings` interface, allowing seamless integration into the LangChain ecosystem:

```python
from langchain_oceanbase.embedding_utils import DefaultEmbeddingFunctionAdapter

# Create adapter
adapter = DefaultEmbeddingFunctionAdapter()
print(f"Embedding dimension: {adapter.dimension}")

# Generate embeddings for multiple documents (batch)
documents = ["Hello world", "How are you?", "Python programming"]
doc_embeddings = adapter.embed_documents(documents)
print(f"\nDocument count: {len(documents)}")
print(f"Generated document embedding count: {len(doc_embeddings)}")
print(f"Each document embedding dimension: {len(doc_embeddings[0]) if doc_embeddings else 0}")

# Generate embedding for a query (single text)
query = "Hello"
query_embedding = adapter.embed_query(query)
print(f"\nQuery text: '{query}'")
print(f"Query embedding dimension: {len(query_embedding)}")
print(f"Query embedding type: {type(query_embedding)}")  # Should be a 1D list
```

## Method 3: Using Default Embedding in OceanbaseVectorStore

The simplest way is to set `embedding_function=None` when creating `OceanbaseVectorStore`, and the system will automatically use the default embedding function:

```python
from langchain_oceanbase.vectorstores import OceanbaseVectorStore
from langchain_core.documents import Document

# Connection configuration
connection_args = {
    "host": "127.0.0.1",
    "port": "2881",
    "user": "root@test",
    "password": "",
    "db_name": "test",
}

# Use default embedding (set embedding_function=None)
vector_store = OceanbaseVectorStore(
    embedding_function=None,  # Automatically uses DefaultEmbeddingFunction
    table_name="embedding_demo",
    connection_args=connection_args,
    vidx_metric_type="l2",
    drop_old=True,
    embedding_dim=384,  # all-MiniLM-L6-v2 dimension
)

print("✓ Vector store created successfully, using default embedding")
```

### Adding Documents and Performing Search

Add documents and perform similarity search using the default embedding:

```python
# Add documents
documents = [
    Document(page_content="Machine learning is a subset of artificial intelligence"),
    Document(page_content="Python is a popular programming language"),
    Document(page_content="OceanBase is a distributed relational database"),
    Document(page_content="Deep learning uses neural networks for pattern recognition"),
]

ids = vector_store.add_documents(documents)
print(f"✓ Successfully added {len(ids)} documents")

# Perform similarity search
results = vector_store.similarity_search("artificial intelligence", k=2)
print(f"\nSimilarity search for 'artificial intelligence' returned {len(results)} results:")
for i, doc in enumerate(results, 1):
    print(f"{i}. {doc.page_content}")
```

## Computing Text Similarity

Use embeddings to compute similarity between texts:

```python
from langchain_oceanbase.embedding_utils import DefaultEmbeddingFunctionAdapter
import math

# Create adapter
adapter = DefaultEmbeddingFunctionAdapter()

# Define similar and dissimilar texts
text1 = "Machine learning is a subset of artificial intelligence"
text2 = "AI and machine learning are related technologies"
text3 = "The weather today is sunny and warm"

# Generate embeddings
emb1 = adapter.embed_query(text1)
emb2 = adapter.embed_query(text2)
emb3 = adapter.embed_query(text3)

# Compute cosine similarity
def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors"""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    return dot_product / (norm1 * norm2) if norm1 * norm2 > 0 else 0.0

sim_12 = cosine_similarity(emb1, emb2)
sim_13 = cosine_similarity(emb1, emb3)

print(f"Text 1: '{text1[:50]}...'")
print(f"Text 2: '{text2[:50]}...'")
print(f"Text 3: '{text3[:50]}...'")
print(f"\nSimilarity between Text 1 and Text 2: {sim_12:.4f}")
print(f"Similarity between Text 1 and Text 3: {sim_13:.4f}")
print(f"\n✓ Related text similarity ({sim_12:.4f}) > Unrelated text similarity ({sim_13:.4f})")
```

## Advanced Usage: Custom Model and Providers

`DefaultEmbeddingFunctionAdapter` supports custom model names and ONNX runtime providers:

```python
# Use default configuration
adapter_default = DefaultEmbeddingFunctionAdapter()
print(f"Default configuration - Dimension: {adapter_default.dimension}")

# Custom model name (currently only 'all-MiniLM-L6-v2' is supported)
adapter_custom = DefaultEmbeddingFunctionAdapter(
    model_name="all-MiniLM-L6-v2",
    preferred_providers=["CPUExecutionProvider"]  # Specify CPU usage
)
print(f"Custom configuration - Dimension: {adapter_custom.dimension}")

# Test embedding generation
texts = ["Test embedding generation"]
embeddings_default = adapter_default.embed_documents(texts)
embeddings_custom = adapter_custom.embed_documents(texts)

print(f"\nDefault configuration embedding dimension: {len(embeddings_default[0])}")
print(f"Custom configuration embedding dimension: {len(embeddings_custom[0])}")
```

## Performance Comparison: Batch Processing vs Single Processing

Demonstrate the performance advantage of batch processing over single processing:

```python
import time
from langchain_oceanbase.embedding_utils import DefaultEmbeddingFunctionAdapter

adapter = DefaultEmbeddingFunctionAdapter()
texts = ["Text " + str(i) for i in range(100)]  # 100 texts

# Method 1: Batch processing
start_time = time.time()
batch_embeddings = adapter.embed_documents(texts)
batch_time = time.time() - start_time
print(f"Batch processing {len(texts)} texts took: {batch_time:.4f} seconds")
print(f"Average per text: {batch_time/len(texts)*1000:.2f} milliseconds")

# Method 2: Single processing (not recommended)
start_time = time.time()
single_embeddings = [adapter.embed_query(text) for text in texts]
single_time = time.time() - start_time
print(f"\nSingle processing {len(texts)} texts took: {single_time:.4f} seconds")
print(f"Average per text: {single_time/len(texts)*1000:.2f} milliseconds")

print(f"\n✓ Batch processing is {single_time/batch_time:.2f}x faster than single processing")
```

## Complete Example: Building a RAG Application

Build a complete RAG (Retrieval Augmented Generation) application using the default embedding:

```python
from langchain_oceanbase.vectorstores import OceanbaseVectorStore
from langchain_core.documents import Document

# Create vector store using default embedding
rag_vector_store = OceanbaseVectorStore(
    embedding_function=None,  # Use default embedding
    table_name="rag_knowledge_base",
    connection_args=connection_args,
    vidx_metric_type="l2",
    drop_old=True,
    embedding_dim=384,
)

# Add knowledge base documents
knowledge_docs = [
    Document(
        page_content="LangChain is a framework for developing applications powered by language models",
        metadata={"source": "langchain_docs", "topic": "framework"}
    ),
    Document(
        page_content="OceanBase is a distributed relational database developed by Ant Group",
        metadata={"source": "oceanbase_docs", "topic": "database"}
    ),
    Document(
        page_content="Vector databases are specialized databases for storing and searching vector embeddings",
        metadata={"source": "vector_db_docs", "topic": "database"}
    ),
    Document(
        page_content="Hybrid search combines multiple search modalities for better results",
        metadata={"source": "search_docs", "topic": "search"}
    ),
]

ids = rag_vector_store.add_documents(knowledge_docs)
print(f"✓ Knowledge base created with {len(ids)} documents")

# Create retriever
retriever = rag_vector_store.as_retriever(search_kwargs={"k": 2})

# Perform retrieval
query = "What is LangChain?"
results = retriever.invoke(query)
print(f"\nQuery: '{query}'")
print(f"Retrieved {len(results)} relevant documents:")
for i, doc in enumerate(results, 1):
    print(f"\n{i}. {doc.page_content}")
    print(f"   Source: {doc.metadata.get('source', 'N/A')}")
    print(f"   Topic: {doc.metadata.get('topic', 'N/A')}")
```

## Key Features Summary

### Advantages

1. **No API Keys Required**: Uses local ONNX models, no external API calls needed
2. **Quick Start**: Perfect for rapid prototyping and testing
3. **LangChain Compatible**: Fully compatible with LangChain's `Embeddings` interface
4. **Batch Processing**: Supports efficient batch embedding generation
5. **Automatic Integration**: Can be automatically used in `OceanbaseVectorStore`

### Technical Specifications

- **Model**: all-MiniLM-L6-v2
- **Dimension**: 384
- **Inference Engine**: ONNX Runtime
- **Supported Platforms**: CPU (default), optional GPU

### Use Cases

- Rapid prototyping
- Local development and testing
- Scenarios that don't require high-precision embeddings
- Applications that need to run offline
- Cost-sensitive applications

### Notes

- Requires `pyseekdb` package to be installed
- Model files will be downloaded on first use (approximately 80MB)
- For production environments, consider using more powerful embedding models (e.g., OpenAI, DashScope, etc.)

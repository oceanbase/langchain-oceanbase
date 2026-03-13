---
sidebar_label: Hybrid Search
---


# Hybrid Search with OceanBase Vector Store

This notebook demonstrates the hybrid search capabilities of the OceanBase vector store, including vector search, sparse vector search, and full-text search with intelligent result fusion.


## Setup

To use hybrid search features, you'll need to deploy a standalone OceanBase server and install the required packages:



```python
docker run --name=oceanbase -e MODE=mini -e OB_SERVER_IP=127.0.0.1 -p 2881:2881 -d oceanbase/oceanbase-ce:latest

```

And install the `langchain-oceanbase` integration package:



```python
pip install -qU "langchain-oceanbase"

```

Check the connection to OceanBase and set the memory usage ratio for vector data:



```python
from pyobvector import ObVecClient

tmp_client = ObVecClient()
tmp_client.perform_raw_text_sql("ALTER SYSTEM ob_vector_memory_limit_percentage = 30")

```

## Initialization

Configure the embedding model and initialize the OceanBase vector store with hybrid search capabilities. Here we use `DefaultEmbeddingFunctionAdapter` for demonstration purposes:



```python
import os
from langchain_oceanbase.embedding_utils import DefaultEmbeddingFunctionAdapter
from langchain_core.documents import Document
from langchain_oceanbase.vectorstores import OceanbaseVectorStore

# Connection configuration
connection_args = {
    "host": "127.0.0.1",
    "port": "2881", 
    "user": "root@test",
    "password": "",
    "db_name": "test",
}

# Initialize embeddings (using DefaultEmbeddingFunctionAdapter for demo)
embeddings = DefaultEmbeddingFunctionAdapter()

# Create vector store with hybrid search capabilities
vector_store = OceanbaseVectorStore(
    embedding_function=embeddings,
    table_name="hybrid_search_demo",
    connection_args=connection_args,
    vidx_metric_type="l2",
    include_sparse=True,      # Enable sparse vector search
    include_fulltext=True,    # Enable full-text search
    drop_old=True,
    embedding_dim=384,
)

print("Hybrid search vector store initialized successfully!")

```

## Hybrid Search Features

### Vector Search

Vector search provides semantic similarity matching using embeddings:



```python
# Add some sample documents
documents = [
    Document(
        page_content="Machine learning is a subset of artificial intelligence",
        metadata={"topic": "AI", "author": "John Doe", "year": 2024}
    ),
    Document(
        page_content="Deep learning uses neural networks for pattern recognition",
        metadata={"topic": "AI", "author": "Jane Smith", "year": 2023}
    ),
    Document(
        page_content="Natural language processing enables computers to understand human language",
        metadata={"topic": "NLP", "author": "Bob Wilson", "year": 2024}
    ),
    Document(
        page_content="Computer vision allows machines to interpret visual information",
        metadata={"topic": "CV", "author": "Alice Brown", "year": 2023}
    ),
]

# Add documents to vector store
ids = vector_store.add_documents(documents)
print(f"Added {len(ids)} documents to the vector store")

# Perform basic vector search
results = vector_store.similarity_search("artificial intelligence", k=3)
print(f"\nVector search results for 'artificial intelligence':")
for i, doc in enumerate(results, 1):
    print(f"{i}. {doc.page_content[:50]}...")
    print(f"   Metadata: {doc.metadata}")
    print()

```

### Sparse Vector Search

Sparse vector search is useful for keyword-based exact matching:



```python
# Create sparse embeddings for the documents
# Each sparse vector represents keyword weights
sparse_embeddings = [
    {1: 0.8, 5: 0.6, 10: 0.4, 15: 0.2},  # machine learning AI
    {2: 0.7, 6: 0.5, 11: 0.3, 16: 0.1},  # deep learning neural
    {3: 0.9, 7: 0.4, 12: 0.6, 17: 0.3},  # natural language processing
    {4: 0.6, 8: 0.8, 13: 0.2, 18: 0.5},  # computer vision
]

# Add documents with sparse embeddings
sparse_ids = vector_store.add_sparse_documents(
    documents=documents,
    sparse_embeddings=sparse_embeddings,
)
print(f"Added sparse embeddings for {len(sparse_ids)} documents")

# Perform sparse vector search
sparse_query = {1: 0.8, 5: 0.6, 10: 0.4}  # Query for "machine learning AI"
sparse_results = vector_store.similarity_search_with_sparse_vector(
    sparse_query=sparse_query,
    k=3
)

print(f"\nSparse vector search results:")
for i, doc in enumerate(sparse_results, 1):
    print(f"{i}. {doc.page_content[:50]}...")
    print(f"   Metadata: {doc.metadata}")
    print()

```

### Full-text Search

Full-text search enables searching within document content:



```python
# Create full-text content for the documents
fulltext_content = [
    "Machine learning algorithms enable computers to learn from data without explicit programming",
    "Deep learning neural networks can process complex patterns in images and text",
    "Natural language processing combines computational linguistics with machine learning",
    "Computer vision systems can identify objects and scenes in digital images",
]

# Add documents with full-text content
fulltext_ids = vector_store.add_documents_with_fulltext(
    documents=documents,
    fulltext_content=fulltext_content,
)
print(f"Added full-text content for {len(fulltext_ids)} documents")

# Perform full-text search
fulltext_results = vector_store.similarity_search_with_fulltext(
    query="artificial intelligence",
    fulltext_query="neural networks",
    k=3
)

print(f"\nFull-text search results:")
for i, doc in enumerate(fulltext_results, 1):
    print(f"{i}. {doc.page_content[:50]}...")
    print(f"   Metadata: {doc.metadata}")
    print()

```

### Advanced Hybrid Search

The most powerful feature is combining all three search modalities:



```python
# Advanced hybrid search combining all three modalities
hybrid_results = vector_store.advanced_hybrid_search(
    vector_query="AI technology",           # Semantic search
    sparse_query={1: 0.8, 5: 0.6},        # Keyword search
    fulltext_query="machine learning",     # Text search
    k=4
)

print(f"Advanced hybrid search results (default weights):")
for i, doc in enumerate(hybrid_results, 1):
    print(f"{i}. {doc.page_content[:50]}...")
    print(f"   Metadata: {doc.metadata}")
    print()

# Advanced hybrid search with custom weights
custom_weights = {
    'vector': 0.7,      # Emphasize semantic similarity
    'sparse': 0.2,      # Reduce keyword matching
    'fulltext': 0.1     # Reduce text search
}

weighted_results = vector_store.advanced_hybrid_search(
    vector_query="AI technology",
    sparse_query={1: 0.8, 5: 0.6},
    fulltext_query="machine learning",
    k=4,
    modality_weights=custom_weights
)

print(f"Advanced hybrid search results (custom weights):")
for i, doc in enumerate(weighted_results, 1):
    print(f"{i}. {doc.page_content[:50]}...")
    print(f"   Metadata: {doc.metadata}")
    print()

```

### Advanced Filtering

You can also combine hybrid search with advanced filtering:



```python
# Advanced filtering with hybrid search
filters = {
    'fulltext': 'machine learning'
}

filtered_results = vector_store.similarity_search_with_advanced_filters(
    query="artificial intelligence",
    filters=filters,
    k=3
)

print(f"Advanced filtering results:")
for i, doc in enumerate(filtered_results, 1):
    print(f"{i}. {doc.page_content[:50]}...")
    print(f"   Metadata: {doc.metadata}")
    print()

```

## Key Features

### Multi-modal Search
- **Vector Search**: Semantic similarity matching using embeddings
- **Sparse Vector Search**: Keyword-based exact matching
- **Full-text Search**: Content-based text search

### Intelligent Result Fusion
- Combines results from multiple search modalities
- Uses weighted scoring system for optimal ranking
- Normalizes scores across different modalities

### Flexible Configuration
- Enable/disable specific search modalities
- **Configurable weights for different search types** (NEW!)
- Support for various distance metrics
- Custom weight validation and error handling

### Performance Optimized
- Leverages OceanBase's vector indexes
- Parallel execution of multiple searches
- Efficient result fusion algorithms


## Summary

The OceanBase Vector Store hybrid search provides:

- **True multi-modal search** combining vector, sparse vector, and full-text search
- **Intelligent result fusion** with **configurable weights** (NEW!)
- **High performance** leveraging OceanBase's native capabilities
- **Flexible configuration** for different use cases
- **Backward compatibility** with existing vector search functionality
- **Weight validation** ensuring proper configuration

### Weight Configuration Examples

```python
# Default weights (automatic)
results = vector_store.advanced_hybrid_search(
    vector_query="AI technology",
    sparse_query={1: 0.8, 5: 0.6},
    fulltext_query="machine learning"
)

# Custom weights emphasizing semantic similarity
semantic_weights = {'vector': 0.8, 'sparse': 0.1, 'fulltext': 0.1}
results = vector_store.advanced_hybrid_search(
    vector_query="AI technology",
    sparse_query={1: 0.8, 5: 0.6},
    fulltext_query="machine learning",
    modality_weights=semantic_weights
)

# Custom weights emphasizing keyword matching
keyword_weights = {'vector': 0.2, 'sparse': 0.7, 'fulltext': 0.1}
results = vector_store.advanced_hybrid_search(
    vector_query="AI technology",
    sparse_query={1: 0.8, 5: 0.6},
    fulltext_query="machine learning",
    modality_weights=keyword_weights
)
```

This makes it ideal for applications requiring both semantic understanding and precise keyword matching, such as academic paper search, product catalogs, and knowledge bases.


---
sidebar_label: Oceanbase
---
# OceanbaseVectorStore

This notebook covers how to get started with the Oceanbase vector store.

## Table of Contents

- [Setup](#setup) - Deploy OceanBase and install dependencies
- [Initialization](#initialization) - Configure and create vector store
- [Manage vector store](#manage-vector-store) - Add, update, and delete vectors
- [Query vector store](#query-vector-store) - Search and retrieve vectors
- [Build RAG (Retrieval Augmented Generation)](#build-rag-retrieval-augmented-generation) - Build powerful RAG applications
- [Full-text Search](#full-text-search) - Implement full-text search capabilities
- [Hybrid Search](#hybrid-search) - Combine vector and text search for better results
- [Advanced Filtering](#advanced-filtering) - Metadata filtering and complex query conditions
- [Maximal Marginal Relevance](#maximal-marginal-relevance) - Filter for diversity in search results
- [Multiple Index Types](#multiple-index-types) - Different vector index types (HNSW, IVF, FLAT)

## Setup

To access Oceanbase vector stores you'll need to deploy a standalone OceanBase server:


```python
%docker run --name=oceanbase -e MODE=mini -e OB_SERVER_IP=127.0.0.1 -p 2881:2881 -d oceanbase/oceanbase-ce:latest
```

And install the `langchain-oceanbase` integration package.


```python
%pip install -qU "langchain-oceanbase"
```

Check the connection to OceanBase and set the memory usage ratio for vector data:


```python
from pyobvector import ObVecClient

tmp_client = ObVecClient()
tmp_client.perform_raw_text_sql("ALTER SYSTEM ob_vector_memory_limit_percentage = 30")
```




    <sqlalchemy.engine.cursor.CursorResult at 0x12696f2a0>



## Initialization

Configure the embedded model. Here we use `DefaultEmbeddingFunctionAdapter` as an example. When deploying `Oceanbase` with a Docker image as described above, simply follow the script below to set the `host`, `port`, `user`, `password`, and `database name`. For other deployment methods, set these parameters according to the actual situation.

```python
import os

from langchain_oceanbase.embedding_utils import DefaultEmbeddingFunctionAdapter

from langchain_oceanbase.vectorstores import OceanbaseVectorStore


connection_args = {
    "host": "127.0.0.1",
    "port": "2881",
    "user": "root@test",
    "password": "",
    "db_name": "test",
}

embeddings = DefaultEmbeddingFunctionAdapter()

vector_store = OceanbaseVectorStore(
    embedding_function=embeddings,
    table_name="langchain_vector",
    connection_args=connection_args,
    vidx_metric_type="l2",
    drop_old=True,
)

```

## Manage vector store

### Add items to vector store

- TODO: Edit and then run code cell to generate output


```python
from langchain_core.documents import Document

document_1 = Document(
    page_content="foo",
    metadata={"source": "https://foo.com"}
)

document_2 = Document(
    page_content="bar",
    metadata={"source": "https://bar.com"}
)

document_3 = Document(
    page_content="baz",
    metadata={"source": "https://baz.com"}
)

documents = [document_1, document_2, document_3]

vector_store.add_documents(documents=documents,ids=["1","2","3"])
```




    ['1', '2', '3']



### Update items in vector store


```python
updated_document = Document(
    page_content="qux",
    metadata={"source": "https://another-example.com"}
)

vector_store.add_documents(documents=[updated_document],ids=["1"])
```




    ['1']



### Delete items from vector store


```python
vector_store.delete(ids=["3"])
```

## Query vector store

Once your vector store has been created and the relevant documents have been added you will most likely wish to query it during the running of your chain or agent.

### Query directly

Performing a simple similarity search can be done as follows:


```python
results = vector_store.similarity_search(query="thud",k=1,filter={"source":"https://another-example.com"})
```

    * bar [{'source': 'https://bar.com'}]


If you want to execute a similarity search and receive the corresponding scores you can run:


```python
results = vector_store.similarity_search_with_score(query="thud",k=1,filter={"source":"https://example.com"})
```

    * [SIM=133.452299] bar [{'source': 'https://bar.com'}]


### Query by turning into retriever

You can also transform the vector store into a retriever for easier usage in your chains.


```python
retriever = vector_store.as_retriever(
    search_kwargs={"k": 1}
)
retriever.invoke("thud")
```




    [Document(metadata={'source': 'https://bar.com'}, page_content='bar')]



## Usage for retrieval-augmented generation

For guides on how to use this vector store for retrieval-augmented generation (RAG), see the following sections:

- [Tutorials](/docs/tutorials/)
- [How-to: Question and answer with RAG](https://python.langchain.com/docs/how_to/#qa-with-rag)
- [Retrieval conceptual docs](https://python.langchain.com/docs/concepts/#retrieval)

## Features

This section demonstrates the advanced features of OceanbaseVectorStore:

*   **Vector Storage**: Store embeddings from any LangChain embedding model in OceanBase with automatic table creation and index management.
*   **Similarity Search**: Perform efficient similarity searches on vector data with multiple distance metrics (L2, cosine, inner product).
*   **Hybrid Search**: Combine vector search with sparse vector search and full-text search for improved results with configurable weights.
*   **Maximal Marginal Relevance**: Filter for diversity in search results to avoid redundant information.
*   **Multiple Index Types**: Support for HNSW, IVF, FLAT and other vector index types with automatic parameter optimization.
*   **Sparse Embeddings**: Native support for sparse vector embeddings with BM25-like functionality.
*   **Advanced Filtering**: Built-in support for metadata filtering and complex query conditions.
*   **Async Support**: Full support for async operations and high-concurrency scenarios.


### Build RAG (Retrieval Augmented Generation)

Discover how to build powerful RAG applications by combining LangChain with OceanBase.



```python
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# Create vector store for knowledge base
rag_vectorstore = OceanbaseVectorStore(
    embedding_function=embeddings,
    table_name="knowledge_base",
    connection_args=connection_args,
    vidx_metric_type="l2",
    drop_old=True,
)

# Add knowledge documents
documents = [
    "LangChain is a framework for developing applications powered by language models.",
    "OceanBase is a distributed relational database developed by Ant Group.",
    "Vector databases are specialized databases for storing and searching vector embeddings.",
    "Hybrid search combines multiple search modalities for better results."
]
rag_vectorstore.add_texts(documents)

# Build RAG chain
retriever = rag_vectorstore.as_retriever(search_kwargs={"k": 3})
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    chain_type="stuff",
    retriever=retriever
)

# Ask questions
question = "What is LangChain?"
answer = qa_chain.run(question)
print(f"Q: {question}")
print(f"A: {answer}")

```

### Full-text Search

Explore how to implement full-text search capabilities using LangChain and OceanBase.



```python
# Create vector store with full-text search enabled
fulltext_vectorstore = OceanbaseVectorStore(
    embedding_function=embeddings,
    table_name="fulltext_docs",
    connection_args=connection_args,
    include_fulltext=True,
    drop_old=True,
)

# Add documents with full-text content
fulltext_documents = [
    Document(
        page_content="Python is a high-level programming language with dynamic semantics",
        metadata={"category": "programming", "language": "python"}
    ),
    Document(
        page_content="Machine learning algorithms can learn patterns from data automatically",
        metadata={"category": "AI", "language": "general"}
    ),
    Document(
        page_content="OceanBase provides high availability and linear scalability",
        metadata={"category": "database", "language": "general"}
    )
]
fulltext_vectorstore.add_documents_with_fulltext(fulltext_documents)

# Perform full-text search
fulltext_results = fulltext_vectorstore.similarity_search_with_fulltext(
    query="programming language",
    k=2
)
print("Full-text search results:")
for doc in fulltext_results:
    print(f"- {doc.page_content}")
    print(f"  Metadata: {doc.metadata}")
    print()

```

### Hybrid Search

Learn how to combine vector and keyword search for more accurate results.



```python
# Create vector store with hybrid search capabilities
hybrid_vectorstore = OceanbaseVectorStore(
    embedding_function=embeddings,
    table_name="hybrid_search_docs",
    connection_args=connection_args,
    include_sparse=True,
    include_fulltext=True,
    drop_old=True,
)

# Add documents with sparse vectors
hybrid_documents = [
    Document(
        page_content="Artificial intelligence and machine learning are transforming industries worldwide",
        metadata={"sparse_vector": {1: 0.8, 3: 0.6, 5: 0.4}, "category": "AI"}
    ),
    Document(
        page_content="Deep learning neural networks require large amounts of training data",
        metadata={"sparse_vector": {2: 0.7, 4: 0.5, 6: 0.3}, "category": "AI"}
    ),
    Document(
        page_content="OceanBase database provides excellent performance for vector operations",
        metadata={"sparse_vector": {1: 0.3, 7: 0.9, 8: 0.6}, "category": "database"}
    )
]
hybrid_vectorstore.add_documents(hybrid_documents)

# Advanced hybrid search with custom weights
hybrid_results = hybrid_vectorstore.advanced_hybrid_search(
    vector_query="AI and machine learning",
    sparse_query={1: 0.5, 3: 0.8, 5: 0.2},
    fulltext_query="artificial intelligence",
    modality_weights={
        "vector": 0.4,
        "sparse": 0.3,
        "fulltext": 0.3
    },
    k=3
)
print("Hybrid search results:")
for doc in hybrid_results:
    print(f"- {doc.page_content}")
    print(f"  Metadata: {doc.metadata}")
    print()

```

### Advanced Filtering

Use metadata filtering for precise document retrieval.



```python
# Search with metadata filters
filtered_results = hybrid_vectorstore.similarity_search_with_advanced_filters(
    query="machine learning",
    k=3,
    filter_conditions={
        "category": "AI"
    }
)
print("Filtered results (AI category only):")
for doc in filtered_results:
    print(f"- {doc.page_content}")
    print(f"  Metadata: {doc.metadata}")
    print()

```

### Maximal Marginal Relevance

Filter for diversity in search results to avoid redundant information.



```python
# Use MMR to get diverse results
diverse_results = hybrid_vectorstore.max_marginal_relevance_search(
    query="artificial intelligence",
    k=3,
    fetch_k=10,
    lambda_mult=0.5
)
print("Diverse results using MMR:")
for doc in diverse_results:
    print(f"- {doc.page_content}")
    print(f"  Metadata: {doc.metadata}")
    print()

```

### Multiple Index Types

OceanbaseVectorStore supports various vector index types for different use cases.



```python
# Example with different index types
index_types = ["HNSW", "IVF", "FLAT"]

for index_type in index_types:
    print(f"\n=== Testing {index_type} Index ===")

    # Create vector store with specific index type
    test_vectorstore = OceanbaseVectorStore(
        embedding_function=embeddings,
        table_name=f"test_{index_type.lower()}",
        connection_args=connection_args,
        vidx_metric_type="l2",
        index_type=index_type,
        drop_old=True,
    )

    # Add test documents
    test_docs = [
        "This is a test document for index performance comparison",
        "Vector search performance varies with different index types",
        "HNSW is good for high-dimensional vectors, IVF for large datasets"
    ]
    test_vectorstore.add_texts(test_docs)

    # Test search performance
    results = test_vectorstore.similarity_search("vector search", k=2)
    print(f"Search results with {index_type}:")
    for doc in results:
        print(f"- {doc.page_content}")

    # Clean up
    test_vectorstore.delete_table()

```

## API reference

For detailed documentation of all OceanbaseVectorStore features and configurations head to the API reference: https://github.com/langchain-ai/langchain/blob/v0.3/docs/docs/integrations/vectorstores/oceanbase.ipynb

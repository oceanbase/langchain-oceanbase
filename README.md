# langchain-oceanbase

This package contains the LangChain integration with OceanBase.

[OceanBase Database](https://github.com/oceanbase/oceanbase) is a distributed relational database.
It is developed entirely by Ant Group. The OceanBase Database is built on a common server cluster.
Based on the Paxos protocol and its distributed structure, the OceanBase Database provides high availability and linear scalability.

OceanBase currently has the ability to store vectors. Users can easily perform the following operations with SQL:

- Create a table containing vector type fields;
- Create a vector index table based on the HNSW algorithm;
- Perform vector approximate nearest neighbor queries;
- ...

## Features

* **Vector Storage**: Store embeddings from any LangChain embedding model in OceanBase with automatic table creation and index management.
* **Similarity Search**: Perform efficient similarity searches on vector data with multiple distance metrics (L2, cosine, inner product).
* **Hybrid Search**: Combine vector search with sparse vector search and full-text search for improved results with configurable weights.
* **Maximal Marginal Relevance**: Filter for diversity in search results to avoid redundant information.
* **Multiple Index Types**: Support for HNSW, IVF, FLAT and other vector index types with automatic parameter optimization.
* **Sparse Embeddings**: Native support for sparse vector embeddings with BM25-like functionality.
* **Advanced Filtering**: Built-in support for metadata filtering and complex query conditions.
* **Async Support**: Full support for async operations and high-concurrency scenarios.

## Installation

```bash
pip install -U langchain-oceanbase
```

### Requirements

- Python >=3.10
- langchain-core >=1.0.0
- pyobvector >=0.2.17

> **Tip**: The current version supports `langchain-core >=1.0.0`

We recommend using Docker to deploy OceanBase:

```shell
docker run --name=oceanbase -e MODE=mini -e OB_SERVER_IP=127.0.0.1 -p 2881:2881 -d oceanbase/oceanbase-ce:latest
```

[More methods to deploy OceanBase cluster](https://github.com/oceanbase/oceanbase-doc/blob/V4.3.1/en-US/400.deploy/500.deploy-oceanbase-database-community-edition/100.deployment-overview.md)

## Usage

### Documentation Formats

Choose your preferred format:

- **[Jupyter Notebook](./docs/vectorstores.ipynb)** - Interactive notebook with executable code cells
- **[Markdown](./docs/vectorstores.md)** - Static documentation for easy reading

### Additional Resources

- **[Hybrid Search Guide](./docs/hybrid_search.ipynb)** - Interactive notebook for hybrid search features
- **[Hybrid Search Guide (Markdown)](./docs/hybrid_search.md)** - Static documentation for hybrid search

#### Hybrid Search Sections:
- [**Setup**](./docs/hybrid_search.md#setup) - Deploy OceanBase and install packages
- [**Vector Search**](./docs/hybrid_search.md#vector-search) - Semantic similarity matching
- [**Sparse Vector Search**](./docs/hybrid_search.md#sparse-vector-search) - Keyword-based exact matching
- [**Full-text Search**](./docs/hybrid_search.md#full-text-search) - Content-based text search
- [**Multi-modal Search**](./docs/hybrid_search.md#multi-modal-search) - Combined search strategies

### Quick Start

Get started quickly with the following sections:

- [**Setup**](./docs/vectorstores.md#setup) - Deploy OceanBase and install dependencies
- [**Initialization**](./docs/vectorstores.md#initialization) - Configure and create vector store  
- [**Manage vector store**](./docs/vectorstores.md#manage-vector-store) - Add, update, and delete vectors
- [**Query vector store**](./docs/vectorstores.md#query-vector-store) - Search and retrieve vectors
- [**Build RAG(Retrieval Augmented Generation)**](./docs/vectorstores.md#build-rag-retrieval-augmented-generation) - Build powerful RAG applications
- [**Full-text Search**](./docs/vectorstores.md#full-text-search) - Implement full-text search capabilities
- [**Hybrid Search**](./docs/vectorstores.md#hybrid-search) - Combine vector and text search for better results
- [**Advanced Filtering**](./docs/vectorstores.md#advanced-filtering) - Metadata filtering and complex query conditions
- [**Maximal Marginal Relevance**](./docs/vectorstores.md#maximal-marginal-relevance) - Filter for diversity in search results
- [**Multiple Index Types**](./docs/vectorstores.md#multiple-index-types) - Different vector index types (HNSW, IVF, FLAT)


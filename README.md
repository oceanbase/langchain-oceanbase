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

* **Built-in Embedding**: Built-in embedding function using `all-MiniLM-L6-v2` model (384 dimensions) with no API keys required. Perfect for quick prototyping and local development.
  * **No API Keys Required**: Uses local ONNX models, no external API calls needed
  * **Quick Start**: Perfect for rapid prototyping and testing
  * **LangChain Compatible**: Fully compatible with LangChain's `Embeddings` interface
  * **Batch Processing**: Supports efficient batch embedding generation
  * **Automatic Integration**: Can be automatically used in `OceanbaseVectorStore` by setting `embedding_function=None`
  * **Technical Specs**: Model `all-MiniLM-L6-v2`, 384 dimensions, ONNX Runtime inference
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

- Python >=3.11
- langchain-core >=1.0.0
- pyobvector >=0.2.0 (required for database client)
- pyseekdb >=0.1.0 (optional, for built-in embedding functionality)

> **Tip**: The current version supports `langchain-core >=1.0.0`

### Platform Support

- ✅ **Linux**: Full support (x86_64 and ARM64)
- ✅ **macOS/Windows**: Supported - `pyobvector` works on all platforms

### Built-in Embedding Dependencies

For built-in embedding functionality (no API keys required), `pyseekdb` is automatically installed as an optional dependency. It provides:
- Local ONNX-based embedding inference
- Default embedding model: `all-MiniLM-L6-v2` (384 dimensions)
- No external API calls needed

We recommend using Docker to deploy OceanBase:

```shell
docker run --name=oceanbase -e MODE=mini -e OB_SERVER_IP=127.0.0.1 -p 2881:2881 -d oceanbase/oceanbase-ce:latest
```

For AI Functions support, use OceanBase 4.4.1 or later:

```shell
docker run --name=oceanbase -e MODE=mini -e OB_SERVER_IP=127.0.0.1 -p 2881:2881 -d oceanbase/oceanbase-ce:4.4.1.0-100000032025101610
```

[More methods to deploy OceanBase cluster](https://github.com/oceanbase/oceanbase-doc/blob/V4.3.1/en-US/400.deploy/500.deploy-oceanbase-database-community-edition/100.deployment-overview.md)

## Usage

### Documentation Formats

Choose your preferred format:

- **[Jupyter Notebook](./docs/vectorstores.ipynb)** - Interactive notebook with executable code cells
- **[Markdown](./docs/vectorstores.md)** - Static documentation for easy reading

### Additional Resources

- **[Built-in Embedding Guide](./docs/embeddings.ipynb)** - Interactive notebook for built-in embedding functionality
- **[Built-in Embedding Guide (Markdown)](./docs/embeddings.md)** - Static documentation for built-in embeddings
- **[Hybrid Search Guide](./docs/hybrid_search.ipynb)** - Interactive notebook for hybrid search features
- **[Hybrid Search Guide (Markdown)](./docs/hybrid_search.md)** - Static documentation for hybrid search
- **[AI Functions Guide](./docs/ai_functions.md)** - Documentation for AI Functions (AI_EMBED, AI_COMPLETE, AI_RERANK)
- **[AI Functions Guide (Notebook)](./docs/ai_functions.ipynb)** - Interactive notebook for AI Functions

#### Built-in Embedding Sections:
- [**Installation**](./docs/embeddings.md#installation) - Install required packages
- [**Direct Use**](./docs/embeddings.md#method-1-direct-use-of-defaultembeddingfunction) - Use DefaultEmbeddingFunction directly
- [**LangChain Compatible**](./docs/embeddings.md#method-2-using-defaultembeddingfunctionadapter-langchain-compatible-interface) - Use DefaultEmbeddingFunctionAdapter
- [**Vector Store Integration**](./docs/embeddings.md#method-3-using-default-embedding-in-oceanbasevectorstore) - Use in OceanbaseVectorStore
- [**Text Similarity**](./docs/embeddings.md#computing-text-similarity) - Compute similarity between texts
- [**Performance**](./docs/embeddings.md#performance-comparison-batch-processing-vs-single-processing) - Batch vs single processing comparison

#### Hybrid Search Sections:
- [**Setup**](./docs/hybrid_search.md#setup) - Deploy OceanBase and install packages
- [**Vector Search**](./docs/hybrid_search.md#vector-search) - Semantic similarity matching
- [**Sparse Vector Search**](./docs/hybrid_search.md#sparse-vector-search) - Keyword-based exact matching
- [**Full-text Search**](./docs/hybrid_search.md#full-text-search) - Content-based text search
- [**Multi-modal Search**](./docs/hybrid_search.md#multi-modal-search) - Combined search strategies

#### AI Functions Sections:
- [**Setup**](./docs/ai_functions.md#setup) - Deploy OceanBase and configure AI models
- [**Initialization**](./docs/ai_functions.md#initialization) - Configure and create AI functions client
- [**AI_EMBED**](./docs/ai_functions.md#ai_embed) - Convert text to vector embeddings
- [**AI_COMPLETE**](./docs/ai_functions.md#ai_complete) - Generate text completions
- [**AI_RERANK**](./docs/ai_functions.md#ai_rerank) - Rerank search results
- [**Model Configuration API**](./docs/ai_functions.md#model-configuration-api) - Setup AI models and endpoints

### Quick Start

#### Using Built-in Embedding (No API Keys Required)

The simplest way to get started is using the built-in embedding function, which requires no API keys:

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
    table_name="langchain_vector",
    connection_args=connection_args,
    vidx_metric_type="l2",
    drop_old=True,
    embedding_dim=384,  # all-MiniLM-L6-v2 dimension
)

# Add documents
documents = [
    Document(page_content="Machine learning is a subset of artificial intelligence"),
    Document(page_content="Python is a popular programming language"),
    Document(page_content="OceanBase is a distributed relational database"),
]
ids = vector_store.add_documents(documents)

# Perform similarity search
results = vector_store.similarity_search("artificial intelligence", k=2)
for doc in results:
    print(f"* {doc.page_content}")
```

**Key Benefits of Built-in Embedding:**
- ✅ No API keys or external services required
- ✅ Works offline with local ONNX models
- ✅ Fast batch processing
- ✅ Perfect for prototyping and testing
- ✅ Model files (~80MB) downloaded automatically on first use

#### Additional Quick Start Guides

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


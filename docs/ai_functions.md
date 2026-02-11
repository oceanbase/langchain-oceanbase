---
sidebar_label: AI Functions
---
# OceanBase AI Functions

This notebook covers how to use OceanBase AI functions, including AI_EMBED, AI_COMPLETE, and AI_RERANK functions available in OceanBase 4.4.1+ and SeekDB.

## Table of Contents

- [Setup](#setup) - Deploy OceanBase and install dependencies
- [Initialization](#initialization) - Configure and create AI functions client
- [Test AI Functions](#test-ai-functions) - Test AI_COMPLETE, AI_EMBED, and AI_RERANK
- [AI_EMBED](#ai_embed) - Convert text to vector embeddings
- [AI_COMPLETE](#ai_complete) - Generate text using LLM
- [AI_RERANK](#ai_rerank) - Rerank search results for better accuracy
- [Batch Operations](#batch-operations) - Process multiple texts efficiently
- [Use Cases](#use-cases) - Real-world application examples
- [Key Features](#key-features) - Version support, capabilities, and performance
- [Model Configuration API](#model-configuration-api) - Complete API reference for model management
- [API Reference](#api-reference) - Quick reference for all methods

## Setup

### Step 1: Deploy OceanBase Database

AI Functions require OceanBase 4.4.1+ or SeekDB. Deploy OceanBase using Docker:

```bash
docker run --name=oceanbase -e MODE=mini -e OB_SERVER_IP=127.0.0.1 -p 2881:2881 -d oceanbase/oceanbase-ce:4.4.1.0-100000032025101610
```

### Step 2: Install Dependencies

Install the `langchain-oceanbase` integration package:

```bash
pip install -qU "langchain-oceanbase"
```

### Step 3: Configure Database Connection

Check OceanBase connection and set the memory usage ratio for vector data:

```python
from pyobvector import ObVecClient

tmp_client = ObVecClient(
    uri="127.0.0.1:2881",
    user="root@test",
    password="",
    db_name="test"
)

# Set vector memory usage ratio (optional, recommended: 30%)
tmp_client.perform_raw_text_sql("ALTER SYSTEM ob_vector_memory_limit_percentage = 30")
```

### Step 4: Configure AI Models (Important)

Before using AI Functions, you need to configure the corresponding AI models in the OceanBase database. Follow these steps:

#### Step 4.1: Create Models

**Model Types and Usage**:

- **Embedding Model** (`model_type="dense_embedding"`): Used for both `AI_EMBED` and `AI_RERANK` functions. You can use the same embedding model for both functions, or create separate models if needed.
- **Completion Model** (`model_type="completion"`): Used for `AI_COMPLETE` function only.

**Note**: The same embedding model can be shared between `AI_EMBED` and `AI_RERANK` functions. This is the recommended approach as it simplifies configuration and reduces resource usage.

First, create an Embedding model (for AI_EMBED and AI_RERANK) and a Completion model (for AI_COMPLETE):

```python
from langchain_oceanbase.ai_functions import OceanBaseAIFunctions

# Initialize client
connection_args = {
    "host": "127.0.0.1",
    "port": "2881",
    "user": "root@test",
    "password": "",
    "db_name": "test",
}

ai_functions = OceanBaseAIFunctions(connection_args=connection_args)

# Create Embedding model
ai_functions.create_ai_model(
    model_name="your-embedding-model",
    model_type="dense_embedding"
)

# Create Completion model
ai_functions.create_ai_model(
    model_name="your-completion-model",
    model_type="completion"
)
```

#### Step 4.2: Query Models

After creating models, you can query all configured models:

```python
# Query all AI models
models = ai_functions.list_ai_models()

print(f"Found {len(models)} AI model(s):")
for model in models:
    print(f"  Model name: {model.get('model_name')}")
    print(f"  Type: {model.get('type')} (1=embedding, 3=completion)")
    print(f"  Created at: {model.get('gmt_create')}")
    print()
```

#### Step 4.3: Configure Model Endpoints

Configure endpoints (API access address and key) for each model.

**Important Relationship**:
- **One-to-One Relationship**: Each `ai_model` can only have **one** `ai_model_endpoint`. This is a one-to-one relationship.
- **Binding Requirement**: When creating an `ai_model_endpoint`, you **must** bind it to an existing `ai_model` by specifying the `ai_model_name`.
- **Independent Deletion**: `ai_model` and `ai_model_endpoint` can be deleted independently. However, if you want to delete an `ai_model`, you must delete its associated `ai_model_endpoint` first.

Configure endpoints:

```python
# Configure Embedding model endpoint
ai_functions.create_ai_model_endpoint(
    endpoint_name="embedding_endpoint",
    ai_model_name="your-embedding-model",
    url="https://api.example.com/v1",
    access_key="YOUR_API_KEY",
    provider="openai"
)

# Configure Completion model endpoint
ai_functions.create_ai_model_endpoint(
    endpoint_name="complete_endpoint",
    ai_model_name="your-completion-model",
    url="https://api.example.com/v1",
    access_key="YOUR_API_KEY",
    provider="openai"
)
```

#### Step 4.4: Query Model Endpoints

After configuring endpoints, you can query all configured endpoints:

```python
# Query all AI model endpoints
endpoints = ai_functions.list_ai_model_endpoints()

print(f"Found {len(endpoints)} AI model endpoint(s):")
for endpoint in endpoints:
    print(f"  Endpoint name: {endpoint.get('ENDPOINT_NAME')}")
    print(f"  Model name: {endpoint.get('AI_MODEL_NAME')}")
    print(f"  URL: {endpoint.get('URL')}")
    print(f"  Provider: {endpoint.get('PROVIDER')}")
    print(f"  Scope: {endpoint.get('SCOPE')}")
    print()
```

#### Step 4.5: Alter Model Endpoints (Optional)

If you need to update endpoint configuration (e.g., change URL or access key), you can use the `alter_ai_model_endpoint()` method:

```python
# Alter Embedding model endpoint
ai_functions.alter_ai_model_endpoint(
    endpoint_name="embedding_endpoint",
    ai_model_name="your-embedding-model",
    url="https://new-api.example.com/v1",
    access_key="NEW_API_KEY",
    provider="openai",
    scope="all"
)

# Alter Completion model endpoint
ai_functions.alter_ai_model_endpoint(
    endpoint_name="complete_endpoint",
    ai_model_name="your-completion-model",
    url="https://new-api.example.com/v1",
    access_key="NEW_API_KEY",
    provider="openai",
    scope="all"
)
```

**Note**: The `alter_ai_model_endpoint()` method updates an existing endpoint configuration. All parameters (ai_model_name, url, access_key, provider, scope) must be provided.

#### Step 4.6: Delete Models (Optional)

If you need to delete models, you can use the following methods:

**Important Relationship**:
- **Independent Deletion**: `ai_model` and `ai_model_endpoint` can be deleted independently.
- **Deletion Order**: However, if you want to delete an `ai_model`, you **must** delete its associated `ai_model_endpoint` first (due to the one-to-one binding relationship).
- **One-to-One Relationship**: Since each model can only have one endpoint, you only need to delete one endpoint per model.

```python
# Step 1: Delete model endpoints first (required before deleting models)
ai_functions.drop_ai_model_endpoint("embedding_endpoint")
ai_functions.drop_ai_model_endpoint("complete_endpoint")

# Step 2: Delete models (now safe to delete)
ai_functions.drop_ai_model("your-embedding-model")
ai_functions.drop_ai_model("your-completion-model")
```

**Note**: If you delete an embedding model that is shared between `AI_EMBED` and `AI_RERANK`, both functions will stop working until you create a new model and configure endpoints.

#### Step 4.7: Delete Model Endpoints

Delete model endpoints independently (endpoints can be deleted without deleting the associated model):

```python
# Delete model endpoints independently
# Note: Endpoints can be deleted without deleting the associated model
# The model will remain but won't be usable until a new endpoint is created
ai_functions.drop_ai_model_endpoint("embedding_endpoint")
ai_functions.drop_ai_model_endpoint("complete_endpoint")
```

## Initialization

### Step 1: Import Module

```python
from langchain_oceanbase.ai_functions import OceanBaseAIFunctions
```

### Step 2: Configure Database Connection Parameters

```python
connection_args = {
    "host": "127.0.0.1",        # OceanBase server address
    "port": "2881",             # OceanBase port
    "user": "root@test",        # Database username (format: username@tenant)
    "password": "",              # Database password
    "db_name": "test",          # Database name
}
```

### Step 3: Create AI Functions Client

```python
ai_functions = OceanBaseAIFunctions(connection_args=connection_args)
print("AI Functions client initialized successfully!")
```

### Step 4: Configure AI Models (Required Step)

Before using AI Functions, you must configure models first. You can use Python API or SQL to configure.

**Model Usage Overview**:

- **Embedding Model**: Can be used for both `AI_EMBED` (text-to-vector conversion) and `AI_RERANK` (document reranking). **You can use the same embedding model for both functions**, which is the recommended approach.
- **Completion Model**: Used exclusively for `AI_COMPLETE` (text generation).

**Model-Endpoint Relationship**:

- **One-to-One Relationship**: Each `ai_model` can only have **one** `ai_model_endpoint`. This is a one-to-one binding relationship.
- **Binding Requirement**: When creating an `ai_model_endpoint`, you **must** bind it to an existing `ai_model` by specifying `ai_model_name`. The model must be created first.
- **Independent Deletion**: `ai_model` and `ai_model_endpoint` can be deleted independently. However, to delete an `ai_model`, you must delete its associated `ai_model_endpoint` first.

#### Using Python API Configuration (Recommended)

```python
# Configure Embedding model (for AI_EMBED and AI_RERANK)
# Note: The same embedding model can be shared between AI_EMBED and AI_RERANK
# Step 1: Create model
ai_functions.create_ai_model(
    model_name="your-embedding-model",
    model_type="dense_embedding"
)

# Step 2: Create model endpoint (binds to the model created above)
# Note: Each model can only have ONE endpoint (one-to-one relationship)
ai_functions.create_ai_model_endpoint(
    endpoint_name="embedding_endpoint",
    ai_model_name="your-embedding-model",  # Must bind to existing model
    url="https://api.example.com/v1",
    access_key="YOUR_API_KEY",
    provider="openai"
)

# Configure Completion model (for AI_COMPLETE)
# Step 1: Create model
ai_functions.create_ai_model(
    model_name="your-completion-model",
    model_type="completion",
    provider_model_name="your-provider-model-name"  # Optional: actual model name in provider
)

# Step 2: Create model endpoint (binds to the model created above)
# Note: Each model can only have ONE endpoint (one-to-one relationship)
ai_functions.create_ai_model_endpoint(
    endpoint_name="complete_endpoint",
    ai_model_name="your-completion-model",  # Must bind to existing model
    url="https://api.example.com/v1",
    access_key="YOUR_API_KEY",
    provider="openai"
)
```

#### Using SQL Configuration

**Note**: Models must be created before endpoints. Each model can only have one endpoint (one-to-one relationship).

```sql
-- Step 1: Create Embedding model (must be done first)
CALL DBMS_AI_SERVICE.CREATE_AI_MODEL('your-embedding-model', '{
    "type": "embedding",
    "model_name": "your-embedding-model"
}');

-- Step 2: Create Completion model (must be done first)
CALL DBMS_AI_SERVICE.CREATE_AI_MODEL('your-completion-model', '{
    "type": "completion",
    "model_name": "your-provider-model-name"
}');

-- Step 3: Configure Embedding model endpoint (binds to model created above)
-- Note: Each model can only have ONE endpoint
CALL DBMS_AI_SERVICE.CREATE_AI_MODEL_ENDPOINT('embedding_endpoint', '{
    "ai_model_name": "your-embedding-model",
    "scope": "all",
    "url": "https://api.example.com/v1",
    "access_key": "YOUR_API_KEY",
    "provider": "openai"
}');

-- Step 4: Configure Completion model endpoint (binds to model created above)
-- Note: Each model can only have ONE endpoint
CALL DBMS_AI_SERVICE.CREATE_AI_MODEL_ENDPOINT('complete_endpoint', '{
    "ai_model_name": "your-completion-model",
    "scope": "all",
    "url": "https://api.example.com/v1",
    "access_key": "YOUR_API_KEY",
    "provider": "openai"
}');

-- Step 5: Alter model endpoint (optional)
-- Note: This updates the existing endpoint for the model
CALL DBMS_AI_SERVICE.ALTER_AI_MODEL_ENDPOINT('complete_endpoint', '{
    "ai_model_name": "your-completion-model",
    "scope": "all",
    "url": "https://new-api.example.com/v1",
    "access_key": "NEW_API_KEY",
    "provider": "openai"
}');
```

### Step 5: Verify Configuration

```python
# Verify Embedding model
try:
    vector = ai_functions.ai_embed(
        text="test",
        model_name="your-embedding-model"
    )
    print(f"✅ Embedding model configured successfully: {len(vector)} dimensions")
except Exception as e:
    print(f"❌ Embedding model not configured: {e}")

# Verify Completion model
try:
    completion = ai_functions.ai_complete(
        prompt="Hello",
        model_name="your-completion-model"
    )
    print(f"✅ Completion model configured successfully")
except Exception as e:
    print(f"❌ Completion model not configured: {e}")
```

**Important Notes**:

- AI Functions are only supported in OceanBase 4.4.1+ or SeekDB
- If using an older version, initialization will raise `ValueError`
- **You must configure models and endpoints before using AI Functions**, otherwise errors will occur

## Test AI Functions

After configuration is complete, you can test whether each AI Function works correctly.

### Test AI_COMPLETE

Test text generation functionality:

```python
# Test AI_COMPLETE
completion = ai_functions.ai_complete(
    prompt="Explain what machine learning is in one sentence",
    model_name="your-completion-model"
)
print(f"Completion: {completion}")
```

### Test AI_EMBED

Test text embedding functionality:

```python
# Test AI_EMBED
vector = ai_functions.ai_embed(
    text="Test text: Machine learning is a subset of artificial intelligence",
    model_name="your-embedding-model"
)
print(f"✅ Embedding successful: {len(vector)} dimensions")
print(f"First 5 values: {vector[:5]}")
```

### Test AI_RERANK

Test document reranking functionality:

```python
# Test AI_RERANK
query = "machine learning algorithms"
documents = [
    "Deep learning is a branch of machine learning that uses multi-layer neural networks",
    "Python is a popular programming language widely used in data science",
    "Supervised learning requires labeled data to train models"
]

# Note: AI_RERANK uses the same embedding model as AI_EMBED
# You can use the same model_name for both functions
reranked = ai_functions.ai_rerank(
    query=query,
    documents=documents,
    model_name="your-embedding-model",  # Same model used for AI_EMBED
    top_k=2
)

print("Reranked results:")
for result in reranked:
    print(f"Rank {result['rank']}: Score {result['score']:.4f}")
    print(f"  Document: {result['document']}")
    print()
```

## AI_EMBED

The `AI_EMBED` function converts text to vector embeddings, which can be used for semantic search and similarity matching.

### Basic Usage

Embed text without specifying a model (uses default model):

```python
# Embed text to vector
text = "Machine learning is a subset of artificial intelligence"
vector = ai_functions.ai_embed(text=text)
print(f"Embedding dimension: {len(vector)}")
print(f"First 5 values: {vector[:5]}")
```

### Specify Model

```python
# Specify embedding model explicitly
vector = ai_functions.ai_embed(
    text="Hello, world!",
    model_name="your-embedding-model"
)
print(f"Embedding generated with model: {len(vector)} dimensions")
```

### Specify Dimension

```python
# Specify embedding dimension explicitly
vector = ai_functions.ai_embed(
    text="Natural language processing",
    model_name="your-embedding-model",
    dimension=384
)
print(f"Embedding with specified dimension: {len(vector)}")
```

## AI_COMPLETE

The `AI_COMPLETE` function generates text completions using Large Language Models (LLMs).

### Basic Usage

Generate text completion without specifying a model (uses default model):

```python
# Generate text completion
prompt = "What is machine learning?"
completion = ai_functions.ai_complete(prompt=prompt)
print(f"Completion: {completion}")
```

### Specify Model

```python
# Specify LLM model explicitly
completion = ai_functions.ai_complete(
    prompt="Explain quantum computing in simple terms",
    model_name="your-completion-model"
)
print(f"Completion: {completion}")
```

### Use Content Template

```python
# Use template with {{TEXT}} placeholder for dynamic content
prompt = "Translate to English: {{TEXT}}"
completion = ai_functions.ai_complete(
    prompt=prompt,
    model_name="your-completion-model",
    content="Hello world"
)
print(f"Translation: {completion}")
```

### Customize Generation Parameters

```python
# Customize generation parameters with options dictionary
options = {
    "temperature": 0.7,
    "top_p": 0.9,
    "presence_penalty": 0.1
}

completion = ai_functions.ai_complete(
    prompt="Write a short story about AI",
    model_name="your-completion-model",
    options=options
)
print(f"Completion: {completion}")
```

## AI_RERANK

The `AI_RERANK` function reranks search results to improve relevance by using semantic understanding.

**Model Usage**: `AI_RERANK` uses an embedding model (same type as `AI_EMBED`). **You can use the same embedding model for both `AI_EMBED` and `AI_RERANK` functions**, or configure separate models if needed. Using the same model is recommended as it simplifies configuration and reduces resource usage.

**Model Usage**: `AI_RERANK` uses an embedding model (same type as `AI_EMBED`). You can use the same embedding model for both `AI_EMBED` and `AI_RERANK` functions, or configure separate models if needed.

### Basic Usage

Rerank documents without specifying a model (uses default model):

```python
# Rerank documents
# Note: AI_RERANK uses an embedding model (same type as AI_EMBED)
query = "machine learning algorithms"
documents = [
    "Deep learning uses neural networks for pattern recognition",
    "Supervised learning requires labeled training data",
    "Python is a popular programming language",
    "Reinforcement learning learns through trial and error",
    "Databases store structured information"
]

reranked = ai_functions.ai_rerank(
    query=query,
    documents=documents,
    top_k=3
)

print("Reranked results:")
for result in reranked:
    print(f"Rank {result['rank']}: Score {result['score']:.4f}")
    print(f"  Document: {result['document'][:50]}...")
    print()
```

### Specify Model

```python
# Specify embedding model explicitly
# Note: You can use the same embedding model for both AI_EMBED and AI_RERANK
reranked = ai_functions.ai_rerank(
    query="artificial intelligence",
    documents=[
        "Machine learning enables computers to learn from data",
        "Natural language processing understands human language",
        "Computer vision interprets visual information"
    ],
    model_name="your-embedding-model",  # Same model used for AI_EMBED
    top_k=2
)

print("Top 2 reranked results:")
for result in reranked:
    print(f"Rank {result['rank']}: {result['document']}")
    print(f"  Score: {result['score']:.4f}\n")
```

### Rerank All Documents

```python
# Return all reranked documents (no top_k limit)
reranked = ai_functions.ai_rerank(
    query="neural networks",
    documents=[
        "Convolutional neural networks excel at image recognition",
        "Recurrent neural networks process sequential data",
        "Transformers revolutionized NLP tasks"
    ]
)

print("All reranked results:")
for result in reranked:
    print(f"Rank {result['rank']}: Score {result['score']:.4f}")
    print(f"  {result['document']}\n")
```

## Batch Operations

Process multiple texts efficiently using batch operations.

### Batch Embedding

```python
# Embed multiple texts at once
texts = [
    "Machine learning algorithms",
    "Deep learning neural networks",
    "Natural language processing",
    "Computer vision systems"
]

vectors = ai_functions.batch_ai_embed(
    texts=texts,
    model_name="your-embedding-model"
)

print(f"Generated {len(vectors)} embeddings")
print(f"Each embedding has {len(vectors[0])} dimensions")
```

## Use Cases

### Use Case 1: Building a Semantic Search System

Combine AI_EMBED with vector search for semantic search:

```python
# Step 1: Embed query
query = "How does neural network training work?"
query_vector = ai_functions.ai_embed(
    text=query,
    model_name="your-embedding-model"
)

# Step 2: Use vector for similarity search
# (This would typically be done with OceanbaseVectorStore)
# vector_store.similarity_search_by_vector(query_vector, k=5)
```

### Use Case 2: RAG with Reranking

Improve RAG results by reranking retrieved documents:

```python
# Step 1: Retrieve documents (example)
retrieved_docs = [
    "Neural networks consist of layers of interconnected nodes",
    "Training involves forward and backward propagation",
    "Gradient descent optimizes network parameters",
    "Python libraries like TensorFlow simplify implementation"
]

# Step 2: Rerank for better relevance
# Note: AI_RERANK uses the same embedding model as AI_EMBED
query = "How to train a neural network?"
reranked = ai_functions.ai_rerank(
    query=query,
    documents=retrieved_docs,
    model_name="your-embedding-model",  # Same embedding model used for AI_EMBED
    top_k=2
)

print("Most relevant documents:")
for result in reranked:
    print(f"{result['document']}\n")
```

### Use Case 3: Text Generation Pipeline

Use AI_COMPLETE for content generation:

```python
# Generate summaries
documents = [
    "Machine learning is transforming industries...",
    "Deep learning enables breakthrough applications...",
    "AI is reshaping the future of technology..."
]

for doc in documents:
    prompt = f"Summarize the following text in one sentence: {{TEXT}}"
    summary = ai_functions.ai_complete(
        prompt=prompt,
        model_name="your-completion-model",
        content=doc
    )
    print(f"Summary: {summary}\n")
```

### Use Case 4: Multi-language Support

Use AI_COMPLETE for translation:

```python
# Translate text
texts = [
    "Hello, how are you?",
    "Machine learning is fascinating",
    "Thank you for your help"
]

for text in texts:
    prompt = "Translate to Chinese: {{TEXT}}"
    translation = ai_functions.ai_complete(
        prompt=prompt,
        model_name="your-completion-model",
        content=text
    )
    print(f"{text} -> {translation}")
```

## Key Features

### Version Support

- **OceanBase 4.4.1+**: Full support for all AI functions
- **SeekDB**: Full support for all AI functions
- **Automatic version checking**: Validates database version on initialization

### Function Capabilities

- **AI_EMBED**: Convert text to high-dimensional vector embeddings
- **AI_COMPLETE**: Generate text using state-of-the-art LLMs
- **AI_RERANK**: Improve search result relevance with semantic reranking

### Error Handling

- Graceful handling of missing model configurations
- Clear error messages for unsupported database versions
- Fallback mechanisms for batch operations

### Performance

- Efficient batch processing for multiple texts
- Optimized SQL execution for AI function calls
- Support for concurrent operations

## Model Configuration API

### Model Types and Usage

**Embedding Model** (`model_type="dense_embedding"`):
- Used for `AI_EMBED` function (text-to-vector conversion)
- Used for `AI_RERANK` function (document reranking)
- **The same embedding model can be shared between `AI_EMBED` and `AI_RERANK`** - this is the recommended approach as it simplifies configuration and reduces resource usage

**Completion Model** (`model_type="completion"`):
- Used exclusively for `AI_COMPLETE` function (text generation)

### Create Model

Use the `create_ai_model()` method to create an AI model:

```python
# Create Embedding model (can be used for both AI_EMBED and AI_RERANK)
ai_functions.create_ai_model(
    model_name="your-embedding-model",
    model_type="dense_embedding"
)

# Create Completion model (used only for AI_COMPLETE)
ai_functions.create_ai_model(
    model_name="your-completion-model",
    model_type="completion",
    provider_model_name="your-provider-model-name"  # Optional
)
```

### Delete Model

Use the `drop_ai_model()` method to delete a model:

**Important Relationship**:
- **One-to-One Relationship**: Each `ai_model` has only **one** `ai_model_endpoint` (one-to-one relationship).
- **Independent Deletion**: `ai_model` and `ai_model_endpoint` can be deleted independently.
- **Deletion Order**: However, to delete an `ai_model`, you **must** delete its associated `ai_model_endpoint` first.

```python
# Step 1: Delete the endpoint first (required)
ai_functions.drop_ai_model_endpoint("embedding_endpoint")

# Step 2: Delete the model (now safe to delete)
ai_functions.drop_ai_model("your-embedding-model")

# Repeat for Completion model
ai_functions.drop_ai_model_endpoint("complete_endpoint")
ai_functions.drop_ai_model("your-completion-model")
```

**Important Notes**:

- **You must delete the associated endpoint before deleting a model**. Since each model has only one endpoint, delete that endpoint first using `drop_ai_model_endpoint()`.
- Deleting a model will remove it from OceanBase, but will not affect any data that was previously processed using that model.
- If you delete an embedding model that is used by both `AI_EMBED` and `AI_RERANK`, both functions will stop working until you create a new model and configure endpoints.
- **Endpoints can be deleted independently** without deleting the model, but the model won't be usable until a new endpoint is created.

### Create Model Endpoint

Use the `create_ai_model_endpoint()` method to configure a model endpoint:

**Important Relationship**:
- **One-to-One Binding**: Each `ai_model` can only have **one** `ai_model_endpoint`. This is a one-to-one relationship.
- **Model Must Exist**: The `ai_model_name` specified must already exist (created using `create_ai_model()`).
- **Binding Requirement**: When creating an endpoint, you **must** bind it to an existing model by specifying `ai_model_name`.

```python
# Create endpoint for an existing model
# Note: The model "your-embedding-model" must already exist
ai_functions.create_ai_model_endpoint(
    endpoint_name="my_endpoint",
    ai_model_name="your-embedding-model",  # Must be an existing model
    url="https://api.example.com/v1",
    access_key="your-api-key",
    provider="openai",  # Optional, default "openai"
    scope="all"  # Optional, default "all"
)
```

**Note**: If you try to create a second endpoint for the same model, it will fail or replace the existing endpoint (depending on OceanBase behavior). Each model should have only one endpoint.

### Alter Model Endpoint

Use the `alter_ai_model_endpoint()` method to update endpoint configuration:

```python
ai_functions.alter_ai_model_endpoint(
    endpoint_name="my_endpoint",
    ai_model_name="your-embedding-model",
    url="https://new-api.example.com/v1",
    access_key="new-api-key",
    provider="openai"
)
```

### Delete Model Endpoint

Use the `drop_ai_model_endpoint()` method to delete an endpoint:

**Important Relationship**:
- **Independent Deletion**: `ai_model_endpoint` can be deleted independently without deleting the associated `ai_model`.
- **One-to-One Relationship**: Each `ai_model` has only one endpoint, so deleting the endpoint means the model will have no endpoint.
- **Model Remains**: After deleting an endpoint, the `ai_model` still exists but won't be usable until a new endpoint is created.

```python
# Delete endpoint independently (model remains but won't be usable)
ai_functions.drop_ai_model_endpoint("my_endpoint")
```

**Note**: After deleting an endpoint, the associated model still exists. You can create a new endpoint for the same model later if needed.

### Query AI Models

Use the `list_ai_models()` method to query all configured AI models:

```python
# Query all AI models
models = ai_functions.list_ai_models()

print(f"Found {len(models)} AI model(s):")
for model in models:
    print(f"  Model name: {model.get('model_name')}")
    print(f"  Type: {model.get('type')}")
    print(f"  Created at: {model.get('gmt_create')}")
    print()
```

**Returned information includes**:

- `model_name`: Model name
- `type`: Model type (1=embedding, 3=completion)
- `model_id`: Model ID
- `gmt_create`: Creation time
- `gmt_modified`: Modification time
- `tenant_id`: Tenant ID

### Query AI Model Endpoints

Use the `list_ai_model_endpoints()` method to query all configured AI model endpoints:

```python
# Query all AI model endpoints
endpoints = ai_functions.list_ai_model_endpoints()

print(f"Found {len(endpoints)} AI model endpoint(s):")
for endpoint in endpoints:
    print(f"  Endpoint name: {endpoint.get('ENDPOINT_NAME')}")
    print(f"  Model name: {endpoint.get('AI_MODEL_NAME')}")
    print(f"  URL: {endpoint.get('URL')}")
    print(f"  Provider: {endpoint.get('PROVIDER')}")
    print(f"  Scope: {endpoint.get('SCOPE')}")
    print()
```

**Returned information includes**:

- `ENDPOINT_NAME`: Endpoint name
- `AI_MODEL_NAME`: Associated AI model name
- `URL`: API endpoint URL
- `ACCESS_KEY`: Access key (encrypted)
- `PROVIDER`: Provider type (e.g., openai)
- `SCOPE`: Scope (all means all tenants)
- `ENDPOINT_ID`: Endpoint ID

### Complete Configuration Example

```python
from langchain_oceanbase.ai_functions import OceanBaseAIFunctions

# Initialize
ai_functions = OceanBaseAIFunctions(connection_args={
    "host": "127.0.0.1",
    "port": "2881",
    "user": "root@test",
    "password": "",
    "db_name": "test",
})

# Configure Embedding model
print("Configuring Embedding model...")
ai_functions.create_ai_model(
    model_name="your-embedding-model",
    model_type="dense_embedding"
)
ai_functions.create_ai_model_endpoint(
    endpoint_name="embedding_endpoint",
    ai_model_name="your-embedding-model",
    url="https://api.example.com/v1",
    access_key="YOUR_API_KEY",
    provider="openai"
)
print("✅ Embedding model configured")

# Configure Completion model
print("Configuring Completion model...")
ai_functions.create_ai_model(
    model_name="your-completion-model",
    model_type="completion"
)
ai_functions.create_ai_model_endpoint(
    endpoint_name="complete_endpoint",
    ai_model_name="your-completion-model",
    url="https://api.example.com/v1",
    access_key="YOUR_API_KEY",
    provider="openai"
)
print("✅ Completion model configured")

# Verify configuration
print("\nVerifying model configuration...")
vector = ai_functions.ai_embed(
    text="test",
    model_name="your-embedding-model"
)
print(f"✅ Embedding model available: {len(vector)} dimensions")

completion = ai_functions.ai_complete(
    prompt="Hello",
    model_name="your-completion-model"
)
print(f"✅ Completion model available")
```

## API Reference

For detailed documentation of all OceanBaseAIFunctions methods and parameters, see the API reference:

### AI Functions

- `ai_embed()`: Convert text to vector embeddings
- `ai_complete()`: Generate text completions
- `ai_rerank()`: Rerank documents by relevance
- `batch_ai_embed()`: Batch process multiple texts

### Model Configuration

- `create_ai_model()`: Create an AI model
- `drop_ai_model()`: Drop an AI model
- `create_ai_model_endpoint()`: Create a model endpoint
- `alter_ai_model_endpoint()`: Alter a model endpoint
- `drop_ai_model_endpoint()`: Drop a model endpoint
- `list_ai_models()`: List all configured AI models
- `list_ai_model_endpoints()`: List all configured AI model endpoints

## References

- [OceanBase AI Functions Documentation](https://www.oceanbase.com/docs/common-oceanbase-database-cn-1000000004018305)
- [OceanBase AI Functions Guide](https://www.oceanbase.com/docs/common-oceanbase-database-cn-1000000004018306)

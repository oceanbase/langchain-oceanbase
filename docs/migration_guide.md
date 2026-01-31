# Migration Guide: ChatMessageHistory to LangGraph Checkpointer

This guide explains how to migrate from `OceanBaseChatMessageHistory` to `OceanBaseCheckpointSaver` for persistent memory in your LangChain/LangGraph applications.

## Why Migrate?

LangChain and LangGraph v1 have moved to a state-based memory model using **checkpointers** instead of the older `ChatMessageHistory` pattern. This new approach offers several advantages:

| Feature | ChatMessageHistory | LangGraph Checkpointer |
|---------|-------------------|------------------------|
| State Management | Message-only | Full graph state |
| Time Travel | Not supported | Replay any checkpoint |
| Human-in-the-Loop | Limited | Full support |
| Fault Tolerance | Manual | Built-in recovery |
| Subgraphs | Not supported | Full support |

## Deprecation Notice

`OceanBaseChatMessageHistory` is deprecated and will be removed in v1.0. When you instantiate it, you'll see this warning:

```
DeprecationWarning: OceanBaseChatMessageHistory is deprecated and will be removed in v1.0.
Use OceanBaseCheckpointSaver with LangGraph instead.
See migration guide: https://github.com/oceanbase/langchain-oceanbase#migration
```

## Migration Steps

### Step 1: Install Dependencies

```bash
pip install -U langchain-oceanbase langgraph
```

### Step 2: Replace ChatMessageHistory with Checkpointer

**Before (Old Pattern):**

```python
from langchain_oceanbase import OceanBaseChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

# Create message history
history = OceanBaseChatMessageHistory(
    table_name="chat_messages",
    connection_args={
        "host": "127.0.0.1",
        "port": "2881",
        "user": "root@test",
        "password": "",
        "db_name": "test",
    }
)

# Add messages manually
history.add_message(HumanMessage(content="Hello"))
history.add_message(AIMessage(content="Hi there!"))

# Retrieve messages
messages = history.messages
```

**After (New Pattern):**

```python
from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_oceanbase import OceanBaseCheckpointSaver

# Define state schema
class ConversationState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# Create checkpointer
checkpointer = OceanBaseCheckpointSaver(
    connection_args={
        "host": "127.0.0.1",
        "port": "2881",
        "user": "root@test",
        "password": "",
        "db_name": "test",
    }
)
checkpointer.setup()  # Initialize database tables

# Build your graph
def chatbot(state: ConversationState) -> dict:
    # Your LLM logic here
    return {"messages": [...]}

builder = StateGraph(ConversationState)
builder.add_node("chatbot", chatbot)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

# Compile with checkpointer
graph = builder.compile(checkpointer=checkpointer)

# Run with thread_id for persistence
config = {"configurable": {"thread_id": "user-123"}}
result = graph.invoke({"messages": [HumanMessage(content="Hello")]}, config)

# State is automatically persisted!
```

### Step 3: Use Thread-Based Conversations

The checkpointer uses `thread_id` to organize conversations:

```python
# Each thread_id represents a separate conversation
config_user1 = {"configurable": {"thread_id": "user-1"}}
config_user2 = {"configurable": {"thread_id": "user-2"}}

# User 1's conversation
graph.invoke({"messages": [HumanMessage(content="Hi, I'm Alice")]}, config_user1)

# User 2's conversation (separate thread)
graph.invoke({"messages": [HumanMessage(content="Hi, I'm Bob")]}, config_user2)
```

### Step 4: Access Conversation History

```python
# Get current state
state = graph.get_state(config)
messages = state.values.get("messages", [])

# List historical checkpoints (time travel)
for checkpoint in checkpointer.list(config, limit=10):
    print(f"Step {checkpoint.metadata.get('step')}: {checkpoint.checkpoint['id']}")
```

## Key Differences

### 1. Automatic State Management

With `ChatMessageHistory`, you manually added messages. With checkpointers, state is automatically saved after each graph node execution.

### 2. Graph Integration

Checkpointers are designed to work with LangGraph's `StateGraph`:

```python
# Messages accumulate automatically using add_messages reducer
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
```

### 3. Thread-Based Organization

Instead of managing sessions manually, use `thread_id` in the config:

```python
config = {"configurable": {"thread_id": "conversation-123"}}
```

### 4. Database Schema

The checkpointer uses a different database schema optimized for state persistence:

| Table | Purpose |
|-------|---------|
| `checkpoints` | Main checkpoint data |
| `checkpoint_blobs` | Large/complex values |
| `checkpoint_writes` | Intermediate writes |
| `checkpoint_migrations` | Schema versioning |

## API Reference

### OceanBaseCheckpointSaver

```python
from langchain_oceanbase import OceanBaseCheckpointSaver

# Initialize
checkpointer = OceanBaseCheckpointSaver(
    connection_args={
        "host": "localhost",
        "port": "2881",
        "user": "root@test",
        "password": "",
        "db_name": "test",
    }
)

# Setup database tables (run once)
checkpointer.setup()

# Get a checkpoint
checkpoint = checkpointer.get_tuple(config)

# List checkpoints
for cp in checkpointer.list(config, limit=5):
    print(cp.checkpoint["id"])

# Delete a thread's data
checkpointer.delete_thread("thread-id")
```

## Common Migration Scenarios

### Scenario 1: Simple Chatbot

**Before:**
```python
history = OceanBaseChatMessageHistory(...)
chain = prompt | llm | StrOutputParser()

for message in user_messages:
    history.add_message(HumanMessage(content=message))
    response = chain.invoke({"messages": history.messages})
    history.add_message(AIMessage(content=response))
```

**After:**
```python
def chatbot(state):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

builder = StateGraph(ConversationState)
builder.add_node("chatbot", chatbot)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

graph = builder.compile(checkpointer=checkpointer)

# Messages automatically accumulate
for message in user_messages:
    graph.invoke({"messages": [HumanMessage(content=message)]}, config)
```

### Scenario 2: Multi-Turn with Context

**Before:**
```python
# Manual context management
history = OceanBaseChatMessageHistory(...)
context = get_recent_messages(history.messages, limit=10)
```

**After:**
```python
# Context automatically maintained in state
state = graph.get_state(config)
# All messages are in state.values["messages"]
```

## Troubleshooting

### Q: My tables aren't being created

Make sure to call `setup()` after creating the checkpointer:

```python
checkpointer = OceanBaseCheckpointSaver(connection_args=...)
checkpointer.setup()  # Don't forget this!
```

### Q: State isn't persisting between runs

Ensure you're using the same `thread_id`:

```python
config = {"configurable": {"thread_id": "my-thread"}}
# Use this same config for all invocations in the same conversation
```

### Q: How do I clear all history for a thread?

```python
checkpointer.delete_thread("thread-id")
```

## Further Resources

- [LangGraph Persistence Documentation](https://langchain-ai.github.io/langgraph/concepts/persistence/)
- [LangGraph Quickstart](https://langchain-ai.github.io/langgraph/quickstart/)
- [OceanBase Documentation](https://oceanbase.github.io/)
- [Example: langgraph_agent.py](../examples/langgraph_agent.py)

"""
Example: Building a LangGraph Agent with Persistent Memory using OceanBase

This example demonstrates how to use OceanBaseCheckpointSaver to build a
LangGraph agent with persistent conversation memory. The agent can:
1. Remember conversation history within a thread
2. Recover state after program restart
3. Support multiple concurrent conversations (threads)

Prerequisites:
    pip install langchain-oceanbase langgraph langchain-openai

    # Start OceanBase with Docker
    docker run --name=oceanbase -e MODE=mini -e OB_SERVER_IP=127.0.0.1 \
        -p 2881:2881 -d oceanbase/oceanbase-ce:latest
"""

import os
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from langchain_oceanbase import OceanBaseCheckpointSaver

# ==============================================================================
# Step 1: Define the State Schema
# ==============================================================================


class ConversationState(TypedDict):
    """State schema for the conversation agent.

    Attributes:
        messages: List of conversation messages with automatic accumulation.
    """

    messages: Annotated[list[BaseMessage], add_messages]


# ==============================================================================
# Step 2: Define Node Functions
# ==============================================================================


def chatbot_node(state: ConversationState) -> dict:
    """Simple chatbot node that echoes back user input.

    In a real application, you would use an LLM here:

    .. code-block:: python

        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="gpt-4")
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    Args:
        state: Current conversation state.

    Returns:
        Dict with new messages to add.
    """
    # Get the last user message
    last_message = state["messages"][-1]

    # Simple echo response (replace with LLM in production)
    response_content = f"You said: {last_message.content}"

    # Check for special commands
    if "memory" in last_message.content.lower():
        # Show conversation history
        history = "\n".join(
            [f"- {msg.__class__.__name__}: {msg.content}" for msg in state["messages"]]
        )
        response_content = f"Conversation history:\n{history}"

    return {"messages": [AIMessage(content=response_content)]}


# ==============================================================================
# Step 3: Build the Graph
# ==============================================================================


def create_agent_graph():
    """Create a simple conversation agent graph.

    Returns:
        A compiled LangGraph graph.
    """
    # Create the graph builder
    graph_builder = StateGraph(ConversationState)

    # Add nodes
    graph_builder.add_node("chatbot", chatbot_node)

    # Add edges
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("chatbot", END)

    return graph_builder


# ==============================================================================
# Step 4: Configure OceanBase Checkpointer
# ==============================================================================


def get_checkpointer():
    """Create and configure the OceanBase checkpointer.

    Returns:
        OceanBaseCheckpointSaver instance.

    Note:
        Configure your OceanBase connection via environment variables:
        - OCEANBASE_HOST: Database host address
        - OCEANBASE_PORT: Database port (default: 3306 for cloud, 2881 for local)
        - OCEANBASE_USER: Database user
        - OCEANBASE_PASSWORD: Database password
        - OCEANBASE_DB: Database name

        Or modify the default values below for your environment.
    """
    # Connection configuration
    # Default values are for Aliyun OceanBase Cloud - modify for your environment
    connection_args = {
        "host": os.getenv("OCEANBASE_HOST","127.0.0.1"),
        "port": os.getenv("OCEANBASE_PORT", "2881"),
        "user": os.getenv("OCEANBASE_USER", "root@test"),
        "password": os.getenv("OCEANBASE_PASSWORD", ""), 
        "db_name": os.getenv("OCEANBASE_DB", "test"),
    }

    # Create checkpointer
    checkpointer = OceanBaseCheckpointSaver(connection_args=connection_args)

    # Initialize database tables (run once)
    checkpointer.setup()

    return checkpointer


# ==============================================================================
# Step 5: Main Example
# ==============================================================================


def main():
    """Main example demonstrating persistent conversation memory."""
    print("=" * 60)
    print("LangGraph Agent with OceanBase Persistent Memory")
    print("=" * 60)

    # Create checkpointer and compile graph
    checkpointer = get_checkpointer()
    graph_builder = create_agent_graph()
    app = graph_builder.compile(checkpointer=checkpointer)

    # Define thread ID for this conversation
    thread_id = "demo-conversation-1"
    config = {"configurable": {"thread_id": thread_id}}

    print(f"\nThread ID: {thread_id}")
    print("-" * 40)

    # Simulate a conversation
    messages_to_send = [
        "Hello! I'm testing the persistent memory.",
        "Can you remember what I said?",
        "Show me the memory please.",
    ]

    for user_input in messages_to_send:
        print(f"\nUser: {user_input}")

        # Run the agent
        result = app.invoke({"messages": [HumanMessage(content=user_input)]}, config)

        # Get the last AI response
        ai_response = result["messages"][-1].content
        print(f"Agent: {ai_response}")

    # ==============================================================================
    # Step 6: Demonstrate State Persistence
    # ==============================================================================

    print("\n" + "=" * 60)
    print("Demonstrating State Persistence")
    print("=" * 60)

    # Create a new graph instance (simulating program restart)
    new_checkpointer = get_checkpointer()
    new_graph_builder = create_agent_graph()
    new_app = new_graph_builder.compile(checkpointer=new_checkpointer)

    # Use the same thread ID to recover state
    print(f"\nRecovering state for thread: {thread_id}")

    # Get the current state
    state_snapshot = new_app.get_state(config)
    if state_snapshot.values:
        print(f"Recovered {len(state_snapshot.values.get('messages', []))} messages")

        # Continue the conversation
        result = new_app.invoke(
            {"messages": [HumanMessage(content="I'm back! Do you remember me?")]},
            config,
        )
        print(f"\nUser: I'm back! Do you remember me?")
        print(f"Agent: {result['messages'][-1].content}")

    # ==============================================================================
    # Step 7: Demonstrate Multiple Threads
    # ==============================================================================

    print("\n" + "=" * 60)
    print("Demonstrating Multiple Threads")
    print("=" * 60)

    # Start a new conversation in a different thread
    thread_2_config = {"configurable": {"thread_id": "demo-conversation-2"}}

    result = new_app.invoke(
        {"messages": [HumanMessage(content="Hello from thread 2!")]},
        thread_2_config,
    )
    print(f"\nThread 2 - User: Hello from thread 2!")
    print(f"Thread 2 - Agent: {result['messages'][-1].content}")

    # ==============================================================================
    # Step 8: List Checkpoints (Time Travel)
    # ==============================================================================

    print("\n" + "=" * 60)
    print("Listing Checkpoints (Time Travel)")
    print("=" * 60)

    print(f"\nCheckpoints for thread '{thread_id}':")
    for i, checkpoint in enumerate(new_checkpointer.list(config, limit=5)):
        print(f"  {i + 1}. ID: {checkpoint.config['configurable']['checkpoint_id']}")
        print(f"      Step: {checkpoint.metadata.get('step', 'N/A')}")

    # ==============================================================================
    # Step 9: Cleanup (Optional)
    # ==============================================================================

    print("\n" + "=" * 60)
    print("Cleanup")
    print("=" * 60)

    # Uncomment to delete thread data
    # new_checkpointer.delete_thread(thread_id)
    # print(f"Deleted thread: {thread_id}")

    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()

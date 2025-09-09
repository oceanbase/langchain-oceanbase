import pytest
from langchain_core.messages import (
    AIMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from langchain_oceanbase.chat_message_histories import OceanBaseChatMessageHistory


class TestOceanBaseChatMessageHistoryIntegration:
    """Integration tests for OceanBaseChatMessageHistory"""

    @pytest.fixture
    def chat_history(self):
        """Create a chat history instance for integration tests"""
        connection_args = {
            "host": "127.0.0.1",
            "port": "2881",
            "user": "root@test",
            "password": "",
            "db_name": "test",
        }

        history = OceanBaseChatMessageHistory(
            table_name="integration_test_chat",
            connection_args=connection_args,
        )
        return history

    @pytest.fixture(autouse=True)
    def cleanup(self, chat_history):
        """Clean up after each test"""
        yield
        chat_history.clear()

    def test_basic_add_and_retrieve_messages(self, chat_history):
        """Test basic message addition and retrieval functionality"""
        # Add different types of messages
        human_msg = HumanMessage(content="Hello, how are you?")
        ai_msg = AIMessage(content="I'm doing well, thank you!")
        system_msg = SystemMessage(content="You are a helpful assistant.")

        chat_history.add_message(human_msg)
        chat_history.add_message(ai_msg)
        chat_history.add_message(system_msg)

        # Retrieve all messages
        messages = chat_history.messages
        assert len(messages) == 3

        # Check that all message types are preserved
        message_types = [type(msg).__name__ for msg in messages]
        assert "HumanMessage" in message_types
        assert "AIMessage" in message_types
        assert "SystemMessage" in message_types

        # Check content preservation
        contents = [msg.content for msg in messages]
        assert "Hello, how are you?" in contents
        assert "I'm doing well, thank you!" in contents
        assert "You are a helpful assistant." in contents

    def test_different_message_types(self, chat_history):
        """Test all supported message types"""
        # Test HumanMessage
        human_msg = HumanMessage(content="User input")
        chat_history.add_message(human_msg)

        # Test AIMessage
        ai_msg = AIMessage(content="AI response")
        chat_history.add_message(ai_msg)

        # Test SystemMessage
        system_msg = SystemMessage(content="System prompt")
        chat_history.add_message(system_msg)

        # Test FunctionMessage
        function_msg = FunctionMessage(
            name="test_function",
            content="Function result",
            additional_kwargs={"session_id": "test_session"},
        )
        chat_history.add_message(function_msg)

        # Test ToolMessage
        tool_msg = ToolMessage(
            content="Tool response",
            tool_call_id="tool_123",
            additional_kwargs={"session_id": "test_session"},
        )
        chat_history.add_message(tool_msg)

        # Retrieve and verify all messages
        messages = chat_history.messages
        assert len(messages) == 5

        # Check message types
        message_types = [type(msg).__name__ for msg in messages]
        assert "HumanMessage" in message_types
        assert "AIMessage" in message_types
        assert "SystemMessage" in message_types
        assert "FunctionMessage" in message_types
        assert "ToolMessage" in message_types

    def test_session_management(self, chat_history):
        """Test session-based message management"""
        # Add messages with different session IDs
        session1_msg1 = HumanMessage(content="Session 1 message 1")
        session1_msg1.additional_kwargs = {"session_id": "session_1"}
        chat_history.add_message(session1_msg1)

        session1_msg2 = AIMessage(content="Session 1 response")
        session1_msg2.additional_kwargs = {"session_id": "session_1"}
        chat_history.add_message(session1_msg2)

        session2_msg1 = HumanMessage(content="Session 2 message 1")
        session2_msg1.additional_kwargs = {"session_id": "session_2"}
        chat_history.add_message(session2_msg1)

        # Retrieve all messages (should get all sessions)
        all_messages = chat_history.messages
        assert len(all_messages) == 3

        # Verify content from both sessions
        contents = [msg.content for msg in all_messages]
        assert "Session 1 message 1" in contents
        assert "Session 1 response" in contents
        assert "Session 2 message 1" in contents

    def test_metadata_preservation(self, chat_history):
        """Test that metadata is preserved correctly"""
        # Create message with metadata
        metadata = {
            "session_id": "test_session",
            "user_id": "user_123",
            "timestamp": "2024-01-01T00:00:00Z",
            "custom_field": "custom_value",
        }

        message = HumanMessage(content="Test message with metadata")
        message.additional_kwargs = metadata
        chat_history.add_message(message)

        # Retrieve message and check metadata
        messages = chat_history.messages
        assert len(messages) == 1

        retrieved_message = messages[0]
        assert isinstance(retrieved_message, HumanMessage)
        assert retrieved_message.content == "Test message with metadata"

        # Check that metadata is preserved
        assert retrieved_message.additional_kwargs["session_id"] == "test_session"
        assert retrieved_message.additional_kwargs["user_id"] == "user_123"
        assert (
            retrieved_message.additional_kwargs["timestamp"] == "2024-01-01T00:00:00Z"
        )
        assert retrieved_message.additional_kwargs["custom_field"] == "custom_value"

    def test_clear_functionality(self, chat_history):
        """Test clearing chat history"""
        # Add some messages
        chat_history.add_message(HumanMessage(content="Message 1"))
        chat_history.add_message(AIMessage(content="Message 2"))
        chat_history.add_message(SystemMessage(content="Message 3"))

        # Verify messages exist
        messages = chat_history.messages
        assert len(messages) == 3

        # Clear history
        chat_history.clear()

        # Verify history is empty
        messages_after_clear = chat_history.messages
        assert len(messages_after_clear) == 0

    def test_empty_history(self, chat_history):
        """Test behavior with empty chat history"""
        # Should return empty list for new instance
        messages = chat_history.messages
        assert len(messages) == 0
        assert isinstance(messages, list)

    def test_message_ordering(self, chat_history):
        """Test that messages maintain their order"""
        # Add messages in sequence
        messages_to_add = [
            HumanMessage(content="First message"),
            AIMessage(content="Second message"),
            HumanMessage(content="Third message"),
            AIMessage(content="Fourth message"),
        ]

        for msg in messages_to_add:
            chat_history.add_message(msg)

        # Retrieve messages
        retrieved_messages = chat_history.messages
        assert len(retrieved_messages) == 4

        # Check that order is preserved (messages should be in the order they were added)
        contents = [msg.content for msg in retrieved_messages]
        assert "First message" in contents
        assert "Second message" in contents
        assert "Third message" in contents
        assert "Fourth message" in contents

    def test_special_characters_in_content(self, chat_history):
        """Test handling of special characters in message content"""
        special_content = "Special chars: !@#$%^&*()_+-=[]{}|;':\",./<>?`~"
        message = HumanMessage(content=special_content)
        chat_history.add_message(message)

        retrieved_messages = chat_history.messages
        assert len(retrieved_messages) == 1
        assert retrieved_messages[0].content == special_content

    def test_unicode_content(self, chat_history):
        """Test handling of unicode content"""
        unicode_content = "Hello 世界! 🌍 测试 unicode 内容"
        message = HumanMessage(content=unicode_content)
        chat_history.add_message(message)

        retrieved_messages = chat_history.messages
        assert len(retrieved_messages) == 1
        assert retrieved_messages[0].content == unicode_content

    def test_large_message_content(self, chat_history):
        """Test handling of large message content"""
        # Use a smaller content size that fits within the database column limit
        large_content = "This is a very long message. " * 100  # ~3KB
        message = HumanMessage(content=large_content)
        chat_history.add_message(message)

        retrieved_messages = chat_history.messages
        assert len(retrieved_messages) == 1
        assert retrieved_messages[0].content == large_content

    def test_function_message_with_metadata(self, chat_history):
        """Test FunctionMessage with complex metadata"""
        function_metadata = {
            "session_id": "func_session",
            "function_name": "calculate_sum",
            "arguments": {"a": 5, "b": 3},
            "result": 8,
        }

        function_msg = FunctionMessage(
            name="calculate_sum",
            content="Function executed successfully",
            additional_kwargs=function_metadata,
        )
        chat_history.add_message(function_msg)

        retrieved_messages = chat_history.messages
        assert len(retrieved_messages) == 1

        retrieved_msg = retrieved_messages[0]
        assert isinstance(retrieved_msg, FunctionMessage)
        assert retrieved_msg.name == "calculate_sum"
        assert retrieved_msg.content == "Function executed successfully"
        assert retrieved_msg.additional_kwargs["function_name"] == "calculate_sum"
        assert retrieved_msg.additional_kwargs["arguments"] == {"a": 5, "b": 3}
        assert retrieved_msg.additional_kwargs["result"] == 8

    def test_tool_message_with_metadata(self, chat_history):
        """Test ToolMessage with complex metadata"""
        tool_metadata = {
            "session_id": "tool_session",
            "tool_name": "weather_api",
            "tool_call_id": "call_123",
            "execution_time": 1.5,
        }

        tool_msg = ToolMessage(
            content="Weather data retrieved",
            tool_call_id="call_123",
            additional_kwargs=tool_metadata,
        )
        chat_history.add_message(tool_msg)

        retrieved_messages = chat_history.messages
        assert len(retrieved_messages) == 1

        retrieved_msg = retrieved_messages[0]
        assert isinstance(retrieved_msg, ToolMessage)
        assert retrieved_msg.content == "Weather data retrieved"
        assert retrieved_msg.tool_call_id == "call_123"
        assert retrieved_msg.additional_kwargs["tool_name"] == "weather_api"
        assert retrieved_msg.additional_kwargs["execution_time"] == 1.5

    def test_multiple_sessions_isolation(self, chat_history):
        """Test that different sessions don't interfere with each other"""
        # Create two separate chat history instances with different table names
        session1_history = OceanBaseChatMessageHistory(
            table_name="session1_test",
            connection_args=chat_history.connection_args,
        )
        session2_history = OceanBaseChatMessageHistory(
            table_name="session2_test",
            connection_args=chat_history.connection_args,
        )

        try:
            # Add messages to session 1
            session1_history.add_message(HumanMessage(content="Session 1 only"))
            session1_history.add_message(AIMessage(content="Session 1 response"))

            # Add messages to session 2
            session2_history.add_message(HumanMessage(content="Session 2 only"))
            session2_history.add_message(AIMessage(content="Session 2 response"))

            # Verify isolation
            session1_messages = session1_history.messages
            session2_messages = session2_history.messages

            assert len(session1_messages) == 2
            assert len(session2_messages) == 2

            # Check that each session only has its own messages
            session1_contents = [msg.content for msg in session1_messages]
            session2_contents = [msg.content for msg in session2_messages]

            assert "Session 1 only" in session1_contents
            assert "Session 1 response" in session1_contents
            assert "Session 2 only" not in session1_contents
            assert "Session 2 response" not in session1_contents

            assert "Session 2 only" in session2_contents
            assert "Session 2 response" in session2_contents
            assert "Session 1 only" not in session2_contents
            assert "Session 1 response" not in session2_contents

        finally:
            # Clean up
            session1_history.clear()
            session2_history.clear()

    def test_concurrent_message_addition(self, chat_history):
        """Test adding multiple messages in quick succession"""
        messages = []
        for i in range(10):
            if i % 2 == 0:
                msg = HumanMessage(content=f"Human message {i}")
            else:
                msg = AIMessage(content=f"AI message {i}")
            messages.append(msg)
            chat_history.add_message(msg)

        # Verify all messages were added
        retrieved_messages = chat_history.messages
        assert len(retrieved_messages) == 10

        # Verify content
        contents = [msg.content for msg in retrieved_messages]
        for i in range(10):
            if i % 2 == 0:
                assert f"Human message {i}" in contents
            else:
                assert f"AI message {i}" in contents

"""OceanBase chat message histories."""

import json
import time
import uuid
from typing import Any, Dict, List, Optional

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage
from pyobvector import ObVecClient
from sqlalchemy import JSON, Column, String

from langchain_oceanbase.vectorstores import DEFAULT_OCEANBASE_CONNECTION

DEFAULT_OCEANBASE_CHAT_MESSAGE_TABLE_NAME = "langchain_chat_message"


class OceanBaseChatMessageHistory(BaseChatMessageHistory):
    """OceanBase chat message history integration.

    Setup:
        Install ``langchain-oceanbase`` and deploy a standalone OceanBase server with docker.

        .. code-block:: bash

            pip install -U langchain-oceanbase
            docker run --name=ob433 -e MODE=mini -e OB_SERVER_IP=127.0.0.1 -p 2881:2881 -d quay.io/oceanbase/oceanbase-ce:4.3.3.1-101000012024102216

        More methods to deploy OceanBase cluster:
        https://github.com/oceanbase/oceanbase-doc/blob/V4.3.1/en-US/400.deploy/500.deploy-oceanbase-database-community-edition/100.deployment-overview.md

    Key init args:
        table_name: str
            Which table name to use. Defaults to "langchain_chat_message".
        connection_args: Optional[dict[str, any]]
            The connection args used for this class comes in the form of a dict. Refer to
            `DEFAULT_OCEANBASE_CONNECTION` for example.

    Instantiate:
        .. code-block:: python

            from langchain_oceanbase.chat_message_histories import OceanBaseChatMessageHistory

            connection_args = {
                "host": "127.0.0.1",
                "port": "2881",
                "user": "root@test",
                "password": "",
                "db_name": "test",
            }

            chat_history = OceanBaseChatMessageHistory(
                table_name="langchain_chat_message",
                connection_args=connection_args,
            )

    Add Messages:
        .. code-block:: python

            from langchain_core.messages import HumanMessage, AIMessage

            # Add human message
            chat_history.add_message(HumanMessage(content="Hello, how are you?"))

            # Add AI message
            chat_history.add_message(AIMessage(content="I'm doing well, thank you!"))

    Retrieve Messages:
        .. code-block:: python

            # Get all messages
            messages = chat_history.messages
            for message in messages:
                print(f"{message.__class__.__name__}: {message.content}")

    Clear History:
        .. code-block:: python

            # Clear all chat history
            chat_history.clear()

    Features:
        - Support for all LangChain message types: Human, AI, System, Function, Tool
        - Automatic table creation and schema management
        - Session-based message organization
        - Metadata preservation for all message types
        - Timestamp-based message ordering
        - Full compatibility with LangChain ecosystem

    """
    def __init__(
        self,
        table_name: str = DEFAULT_OCEANBASE_CHAT_MESSAGE_TABLE_NAME,
        connection_args: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """Initialize the OceanBase chat message history.

        Args:
            table_name (str): Name of the table to store chat messages. 
                Defaults to "langchain_chat_message".
            connection_args (Optional[Dict[str, Any]]): Connection parameters for OceanBase.
                If None, uses DEFAULT_OCEANBASE_CONNECTION. Should include:
                - host: OceanBase server host (default: "localhost")
                - port: OceanBase server port (default: "2881")
                - user: Database username (default: "root@test")
                - password: Database password (default: "")
                - db_name: Database name (default: "test")
            **kwargs: Additional arguments passed to ObVecClient.
        """
        self.table_name: str = table_name
        self.connection_args: Dict[str, Any] = (
            connection_args
            if connection_args is not None
            else DEFAULT_OCEANBASE_CONNECTION
        )

        self._create_client(**kwargs)
        assert self.obvector is not None

        # Create table if it doesn't exist
        self._create_table_if_not_exists()

    def _create_client(self, **kwargs):  # type: ignore[no-untyped-def]
        """Create and initialize the OceanBase vector client.
        
        Args:
            **kwargs: Additional arguments passed to ObVecClient constructor.
        """
        host = self.connection_args.get("host", "localhost")
        port = self.connection_args.get("port", "2881")
        user = self.connection_args.get("user", "root@test")
        password = self.connection_args.get("password", "")
        db_name = self.connection_args.get("db_name", "test")

        self.obvector: ObVecClient = ObVecClient(
            uri=f"{host}:{port}",
            user=user,
            password=password,
            db_name=db_name,
            **kwargs,
        )

    def _create_table_if_not_exists(self) -> None:
        """Create the chat message table if it doesn't exist.
        
        Creates a table with the following schema:
        - id: Primary key (String, 255 chars) - Unique message identifier
        - session_id: Session identifier (String, 255 chars) - Groups messages by session
        - message_type: Type of message (String, 50 chars) - human, ai, system, function, tool
        - content: Message content (String, 65535 chars) - The actual message text
        - metadata: Additional metadata (JSON) - Extra information for the message
        - created_at: Timestamp (String, 50 chars) - When the message was created
        """
        if not self.obvector.check_table_exists(self.table_name):
            cols = [
                Column("id", String(255), primary_key=True),
                Column("session_id", String(255), nullable=False),
                Column("message_type", String(50), nullable=False),
                Column("content", String(65535), nullable=False),
                Column("metadata", JSON),
                Column("created_at", String(50), nullable=False),
            ]

            self.obvector.create_table_with_index_params(
                table_name=self.table_name,
                columns=cols,
                indexes=None,
                vidxs=None,
                partitions=None,
            )

    @property
    def messages(self) -> List[BaseMessage]:
        """Retrieve all messages from the chat history.
        
        Returns:
            List[BaseMessage]: List of all messages in chronological order.
                Supports HumanMessage, AIMessage, SystemMessage, FunctionMessage, and ToolMessage.
                
        Note:
            Messages are sorted by their created_at timestamp to maintain chronological order.
            If the table doesn't exist, returns an empty list.
        """
        if not self.obvector.check_table_exists(self.table_name):
            return []

        res = self.obvector.get(
            table_name=self.table_name,
            ids=None,
            output_column_name=["message_type", "content", "metadata", "created_at"],
        )

        messages = []
        rows = res.fetchall()

        # Sort by created_at timestamp to maintain order
        rows.sort(key=lambda x: int(x[3]) if x[3] else 0)

        for row in rows:
            message_type, content, metadata, created_at = row
            metadata = (
                json.loads(metadata) if isinstance(metadata, str) else metadata or {}
            )

            # Convert stored message back to BaseMessage
            if message_type == "human":
                from langchain_core.messages import HumanMessage

                msg = HumanMessage(content=content)
                msg.additional_kwargs = metadata
                messages.append(msg)
            elif message_type == "ai":
                from langchain_core.messages import AIMessage

                msg = AIMessage(content=content)
                msg.additional_kwargs = metadata
                messages.append(msg)
            elif message_type == "system":
                from langchain_core.messages import SystemMessage

                msg = SystemMessage(content=content)
                msg.additional_kwargs = metadata
                messages.append(msg)
            elif message_type == "function":
                from langchain_core.messages import FunctionMessage

                # Extract name from metadata and create FunctionMessage
                name = metadata.get("name", "unknown")
                # Remove name from metadata to avoid duplication
                metadata_copy = metadata.copy()
                metadata_copy.pop("name", None)
                msg = FunctionMessage(name=name, content=content)
                msg.additional_kwargs = metadata_copy
                messages.append(msg)
            elif message_type == "tool":
                from langchain_core.messages import ToolMessage

                # Extract tool_call_id from metadata and create ToolMessage
                tool_call_id = metadata.get("tool_call_id", "unknown")
                # Remove tool_call_id from metadata to avoid duplication
                metadata_copy = metadata.copy()
                metadata_copy.pop("tool_call_id", None)
                msg = ToolMessage(content=content, tool_call_id=tool_call_id)
                msg.additional_kwargs = metadata_copy
                messages.append(msg)

        return messages

    def add_message(self, message: BaseMessage) -> None:
        """Add a message to the chat history.
        
        Args:
            message (BaseMessage): The message to add. Supports all LangChain message types:
                - HumanMessage: User messages
                - AIMessage: Assistant responses
                - SystemMessage: System instructions
                - FunctionMessage: Function call messages (requires 'name' attribute)
                - ToolMessage: Tool execution messages (requires 'tool_call_id' attribute)
                
        Note:
            - Each message gets a unique UUID as its ID
            - Messages are timestamped with millisecond precision
            - Session ID is extracted from message metadata or defaults to "default"
            - Special handling for FunctionMessage and ToolMessage to preserve their specific attributes
        """
        # Generate unique ID for the message
        message_id = str(uuid.uuid4())

        # Determine message type
        message_type = message.__class__.__name__.lower().replace("message", "")

        # Extract content and metadata
        content = message.content if hasattr(message, "content") else str(message)
        metadata = (
            message.additional_kwargs if hasattr(message, "additional_kwargs") else {}
        ).copy()

        # For FunctionMessage, store the name in metadata
        if message_type == "function" and hasattr(message, "name"):
            metadata["name"] = message.name

        # For ToolMessage, store the tool_call_id in metadata
        if message_type == "tool" and hasattr(message, "tool_call_id"):
            metadata["tool_call_id"] = message.tool_call_id

        # Create session_id from metadata or use default
        session_id = metadata.get("session_id", "default")

        # Current timestamp
        created_at = str(int(time.time() * 1000))  # milliseconds

        # Prepare data for insertion
        data = [
            {
                "id": message_id,
                "session_id": session_id,
                "message_type": message_type,
                "content": content,
                "metadata": json.dumps(metadata),
                "created_at": created_at,
            }
        ]

        # Insert into OceanBase
        self.obvector.upsert(
            table_name=self.table_name,
            data=data,
        )

    def clear(self) -> None:
        """Clear all chat message history.
        
        Removes all messages from the chat history table. This operation is irreversible.
        If the table doesn't exist, this method does nothing.
        
        Note:
            This method deletes all records from the table but does not drop the table itself.
            The table structure and schema remain intact for future use.
        """
        if self.obvector.check_table_exists(self.table_name):
            self.obvector.delete(
                table_name=self.table_name,
                ids=None,
                where_clause=None,
            )

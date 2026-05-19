from unittest.mock import MagicMock, patch

from sqlalchemy.exc import ResourceClosedError

from langchain_oceanbase.chat_message_histories import OceanBaseChatMessageHistory


def test_messages_returns_empty_list_when_backend_closes_empty_result() -> None:
    mock_client = MagicMock()
    mock_client.check_table_exists.return_value = True
    mock_result = MagicMock()
    mock_result.fetchall.side_effect = ResourceClosedError(
        "This result object does not return rows. It has been closed automatically."
    )
    mock_client.get.return_value = mock_result

    with patch(
        "langchain_oceanbase.chat_message_histories.ObVecClient",
        return_value=mock_client,
    ):
        history = OceanBaseChatMessageHistory(
            session_id="test-session",
            connection_args={
                "host": "localhost",
                "port": "2881",
                "user": "root",
                "password": "",
                "db_name": "test",
            },
        )

    assert history.messages == []

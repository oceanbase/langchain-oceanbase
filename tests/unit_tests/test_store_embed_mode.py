"""Unit tests for OceanBaseStore embed-mode (path-based) constructor routing."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from langchain_oceanbase.exceptions import OceanBaseConnectionError
from langchain_oceanbase.store import OceanBaseStore


@patch("langchain_oceanbase.store.ObVecClient")
def test_path_in_connection_args_uses_embed_mode(mock_client_cls: MagicMock) -> None:
    """When connection_args contains 'path', ObVecClient is created with path= kwarg."""
    OceanBaseStore(connection_args={"path": "/tmp/seekdb", "db_name": "mydb"})

    mock_client_cls.assert_called_once_with(path="/tmp/seekdb", db_name="mydb")


@patch("langchain_oceanbase.store.ObVecClient")
def test_embed_mode_omits_uri_user_password(mock_client_cls: MagicMock) -> None:
    """Embed-mode constructor must not pass uri, user, or password to ObVecClient."""
    OceanBaseStore(
        connection_args={"path": "/data/seekdb", "host": "ignored", "user": "ignored"}
    )

    assert mock_client_cls.call_args.args == ()
    call_kwargs = mock_client_cls.call_args.kwargs
    assert "uri" not in call_kwargs
    assert "user" not in call_kwargs
    assert "password" not in call_kwargs
    assert call_kwargs["path"] == "/data/seekdb"


@patch("langchain_oceanbase.store.ObVecClient")
def test_no_path_uses_remote_mode(mock_client_cls: MagicMock) -> None:
    """Without 'path', ObVecClient is created with uri/user/password."""
    OceanBaseStore(
        connection_args={
            "host": "10.0.0.1",
            "port": "3306",
            "user": "admin",
            "password": "secret",
            "db_name": "prod",
        }
    )

    mock_client_cls.assert_called_once_with(
        uri="10.0.0.1:3306",
        user="admin",
        password="secret",
        db_name="prod",
    )


@patch("langchain_oceanbase.store.ObVecClient")
def test_remote_mode_defaults(mock_client_cls: MagicMock) -> None:
    """Remote mode should apply defaults for missing host/port/user/password."""
    OceanBaseStore(connection_args={})

    mock_client_cls.assert_called_once_with(
        uri="localhost:2881",
        user="root@test",
        password="",
        db_name="test",
    )


@patch("langchain_oceanbase.store.ObVecClient")
def test_embed_mode_forwards_extra_kwargs(mock_client_cls: MagicMock) -> None:
    """Extra kwargs from the constructor should forward to ObVecClient in embed mode."""
    OceanBaseStore(
        connection_args={"path": "/tmp/seekdb"},
        timeout=30,
    )

    call_kwargs = mock_client_cls.call_args.kwargs
    assert call_kwargs["timeout"] == 30
    assert call_kwargs["path"] == "/tmp/seekdb"


@patch("langchain_oceanbase.store.ObVecClient")
def test_remote_mode_forwards_extra_kwargs(mock_client_cls: MagicMock) -> None:
    """Extra kwargs from the constructor should forward to ObVecClient in remote mode."""
    OceanBaseStore(
        connection_args={"host": "db.local"},
        pool_size=5,
    )

    call_kwargs = mock_client_cls.call_args.kwargs
    assert call_kwargs["pool_size"] == 5


@patch("langchain_oceanbase.store.ObVecClient")
def test_embed_mode_connection_error_reports_defaults(
    mock_client_cls: MagicMock,
) -> None:
    """A connection-related error in embed mode should not crash on missing host/port."""
    mock_client_cls.side_effect = RuntimeError("connection refused by seekdb")

    with pytest.raises(OceanBaseConnectionError) as exc_info:
        OceanBaseStore(connection_args={"path": "/bad/path"})

    assert exc_info.value.host == "localhost"
    assert exc_info.value.port == "2881"


@patch("langchain_oceanbase.store.ObVecClient")
def test_embed_mode_non_connection_error_reraises(
    mock_client_cls: MagicMock,
) -> None:
    """Non-connection errors in embed mode should re-raise as-is."""
    mock_client_cls.side_effect = ValueError("invalid db_name")

    with pytest.raises(ValueError, match="invalid db_name"):
        OceanBaseStore(connection_args={"path": "/tmp/seekdb"})


@patch("langchain_oceanbase.store.ObVecClient")
def test_path_none_in_connection_args_uses_remote_mode(
    mock_client_cls: MagicMock,
) -> None:
    """Explicit path=None in connection_args should use remote mode."""
    OceanBaseStore(
        connection_args={"path": None, "host": "myhost", "port": "1234"}
    )

    call_kwargs = mock_client_cls.call_args.kwargs
    assert "path" not in call_kwargs
    assert call_kwargs["uri"] == "myhost:1234"

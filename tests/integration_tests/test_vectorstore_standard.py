"""LangChain standard integration tests for OceanbaseVectorStore."""

from __future__ import annotations

import os
import shutil
import uuid
from typing import Generator

import pytest
from langchain_core.vectorstores import VectorStore
from langchain_tests.integration_tests import VectorStoreIntegrationTests

from langchain_oceanbase.vectorstores import OceanbaseVectorStore

EMBEDDING_SIZE = 6


def _embedded_seekdb_runtime_available() -> bool:
    try:
        import pylibseekdb  # noqa: F401
        import pyseekdb  # noqa: F401
    except ImportError:
        return False
    return True


def _ci_db_type() -> str:
    return os.getenv("OB_CI_DB_TYPE", "").strip().lower()


def _external_db_env_configured() -> bool:
    return any(
        os.getenv(env_var)
        for env_var in (
            "SEEKDB_HOST",
            "SEEKDB_PORT",
            "SEEKDB_USER",
            "SEEKDB_PASSWORD",
            "SEEKDB_DB",
            "OB_HOST",
            "OB_PORT",
            "OB_USER",
            "OB_PASSWORD",
            "OB_DB",
        )
    )


def _use_embedded_seekdb() -> bool:
    db_type = _ci_db_type()
    if db_type:
        return db_type == "embedded-seekdb"
    if _external_db_env_configured():
        return False
    return _embedded_seekdb_runtime_available()


def _connection_args_from_env() -> dict[str, str]:
    return {
        "host": os.getenv("SEEKDB_HOST") or os.getenv("OB_HOST", "127.0.0.1"),
        "port": os.getenv("SEEKDB_PORT") or os.getenv("OB_PORT", "2881"),
        "user": os.getenv("SEEKDB_USER") or os.getenv("OB_USER", "root@test"),
        "password": os.getenv("SEEKDB_PASSWORD") or os.getenv("OB_PASSWORD", ""),
        "db_name": os.getenv("SEEKDB_DB") or os.getenv("OB_DB", "test"),
    }


@pytest.fixture(scope="session")
def _embedded_seekdb_path(
    tmp_path_factory: pytest.TempPathFactory,
) -> Generator[str, None, None]:
    root = tmp_path_factory.mktemp("vectorstore_standard_embedded")
    try:
        yield str(root / "seekdb_data")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_use_embedded_seekdb_prefers_explicit_ci_selector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OB_CI_DB_TYPE", "embedded-seekdb")
    monkeypatch.setenv("OB_HOST", "external-db")
    monkeypatch.setattr(
        "tests.integration_tests.test_vectorstore_standard._embedded_seekdb_runtime_available",
        lambda: False,
    )

    assert _use_embedded_seekdb() is True


def test_use_embedded_seekdb_prefers_external_env_when_unspecified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OB_CI_DB_TYPE", raising=False)
    monkeypatch.setenv("OB_HOST", "external-db")
    monkeypatch.setattr(
        "tests.integration_tests.test_vectorstore_standard._embedded_seekdb_runtime_available",
        lambda: True,
    )

    assert _use_embedded_seekdb() is False


class TestOceanbaseVectorStoreStandard(VectorStoreIntegrationTests):
    """Exercise OceanbaseVectorStore against LangChain's shared vectorstore suite."""

    # Legacy capability flags retained so the pinned 0.3.x standard test suite still
    # interprets this class correctly when running the default dependency profile.
    supports_mmr: bool = False
    supports_filter: bool = True
    supports_json_metadata: bool = True
    supports_odata_filter: bool = False
    supports_search_with_score: bool = True
    supports_delete: bool = True
    supports_get_by_ids: bool = True
    supports_async: bool = True
    supports_text_search: bool = True

    @property
    def has_async(self) -> bool:
        return True

    @property
    def has_get_by_ids(self) -> bool:
        return True

    @pytest.fixture()
    def vectorstore(
        self, _embedded_seekdb_path: str
    ) -> Generator[VectorStore, None, None]:  # type: ignore
        """Get an empty vectorstore for standard integration tests."""
        table_name = f"langchain_vector_standard_{uuid.uuid4().hex[:8]}"

        if _use_embedded_seekdb():
            store = OceanbaseVectorStore(
                embedding_function=self.get_embeddings(),
                table_name=table_name,
                path=_embedded_seekdb_path,
                vidx_metric_type="l2",
                index_type="FLAT",
                drop_old=True,
                embedding_dim=EMBEDDING_SIZE,
            )
        else:
            store = OceanbaseVectorStore(
                embedding_function=self.get_embeddings(),
                table_name=table_name,
                connection_args=_connection_args_from_env(),
                vidx_metric_type="l2",
                drop_old=True,
                embedding_dim=EMBEDDING_SIZE,
            )

        try:
            yield store
        finally:
            try:
                store.obvector.drop_table_if_exist(store.table_name)
            except Exception:
                pass

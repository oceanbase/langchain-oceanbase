"""LangChain standard integration tests for OceanbaseVectorStore."""

from __future__ import annotations

from typing import Generator

import pytest
from langchain_core.vectorstores import VectorStore
from langchain_tests.integration_tests import VectorStoreIntegrationTests

from langchain_oceanbase.vectorstores import OceanbaseVectorStore
from tests.integration_tests._backend_utils import (
    use_embedded_seekdb,
    unique_table_name,
)

EMBEDDING_SIZE = 6
def test_use_embedded_seekdb_prefers_explicit_ci_selector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OB_CI_DB_TYPE", "embedded-seekdb")
    monkeypatch.setenv("OB_HOST", "external-db")
    monkeypatch.setattr(
        "tests.integration_tests._backend_utils.embedded_seekdb_runtime_available",
        lambda: False,
    )

    assert use_embedded_seekdb() is True


def test_use_embedded_seekdb_prefers_external_env_when_unspecified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OB_CI_DB_TYPE", raising=False)
    monkeypatch.setenv("OB_HOST", "external-db")
    monkeypatch.setattr(
        "tests.integration_tests._backend_utils.embedded_seekdb_runtime_available",
        lambda: True,
    )

    assert use_embedded_seekdb() is False


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
        self,
        integration_connection_args: dict[str, str],
        integration_client_kwargs: dict[str, str],
        default_vector_index_type: str,
    ) -> Generator[VectorStore, None, None]:  # type: ignore
        """Get an empty vectorstore for standard integration tests."""
        store = OceanbaseVectorStore(
            embedding_function=self.get_embeddings(),
            table_name=unique_table_name("langchain_vector_standard"),
            connection_args=integration_connection_args,
            vidx_metric_type="l2",
            index_type=default_vector_index_type,
            drop_old=True,
            embedding_dim=EMBEDDING_SIZE,
            **integration_client_kwargs,
        )

        try:
            yield store
        finally:
            try:
                store.obvector.drop_table_if_exist(store.table_name)
            except Exception:
                pass

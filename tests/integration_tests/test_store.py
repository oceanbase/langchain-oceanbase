# mypy: disable-error-code="import-untyped,no-untyped-def"
from __future__ import annotations

import time
from typing import Any

import pytest
from langchain_core.embeddings import Embeddings

from langchain_oceanbase import OceanBaseStore
from tests.integration_tests._backend_utils import (
    is_embedded_seekdb_capacity_error,
    unique_table_name,
)


class DummyEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)

    def _embed(self, text: str) -> list[float]:
        lowered = text.lower()
        return [
            1.0 if "python" in lowered else 0.0,
            1.0 if "java" in lowered else 0.0,
            1.0 if "python" not in lowered and "java" not in lowered else 0.0,
        ]


@pytest.fixture
def store_factory(
    integration_backend_name: str,
    integration_connection_args: dict[str, str],
    integration_client_kwargs: dict[str, Any],
):
    stores: list[OceanBaseStore] = []

    def factory(**kwargs: Any) -> OceanBaseStore:
        store = OceanBaseStore(
            connection_args=integration_connection_args,
            index={"dims": 3, "embed": DummyEmbeddings(), "fields": ["text"]},
            ttl_config={"refresh_on_read": True},
            table_name=unique_table_name("langgraph_store"),
            **integration_client_kwargs,
            **kwargs,
        )
        try:
            store.setup()
        except Exception as exc:
            if (
                integration_backend_name == "embedded-seekdb"
                and is_embedded_seekdb_capacity_error(exc)
            ):
                pytest.skip(
                    f"embedded SeekDB capacity exceeded while creating store indexes: {exc}"
                )
            raise
        stores.append(store)
        return store

    yield factory

    for store in stores:
        try:
            store._metadata.drop_all(store.obvector.engine, tables=[store._items_table])
        except Exception:
            pass


def test_store_round_trip_and_namespace_listing(store_factory) -> None:
    store = store_factory()

    store.put(("users", "1", "prefs"), "theme", {"text": "python preferences"})
    store.put(("users", "1", "docs"), "guide", {"text": "java guide"})
    store.put(("users", "2", "prefs"), "theme", {"text": "python notes"})

    item = store.get(("users", "1", "prefs"), "theme")
    namespaced = store.list_namespaces(prefix=("users", "*"), max_depth=2)
    filtered = store.search(("users",), filter={"text": "python notes"})

    assert item is not None
    assert item.value["text"] == "python preferences"
    assert namespaced == [("users", "1"), ("users", "2")]
    assert [result.key for result in filtered] == ["theme"]


def test_store_empty_backend_results_behave_like_empty_store(store_factory) -> None:
    store = store_factory()

    assert store.get(("missing",), "item") is None
    assert store.search(("missing",), limit=10) == []
    assert store.list_namespaces(prefix=("missing",), max_depth=2) == []


def test_store_namespace_prefix_metacharacters_are_literal(store_factory) -> None:
    store = store_factory()

    store.put(("team_%", "alpha"), "literal-percent", {"text": "python literal"})
    store.put(("teamX", "alpha"), "wildcard-candidate", {"text": "python wildcard"})
    store.put(("team__", "beta"), "literal-underscore", {"text": "python underscore"})
    store.put(("teamZZ", "beta"), "underscore-candidate", {"text": "python wildcard"})

    percent_results = store.search(("team_%",), limit=10)
    underscore_results = store.search(("team__",), limit=10)

    assert [result.key for result in percent_results] == ["literal-percent"]
    assert [result.key for result in underscore_results] == ["literal-underscore"]


def test_store_semantic_search_and_ttl(store_factory) -> None:
    store = store_factory()

    store.put(("memories",), "python", {"text": "python memory"})
    store.put(("memories",), "java", {"text": "java memory"})
    store.put(("memories",), "ttl", {"text": "ttl marker"}, ttl=0.05)
    store.put(("memories",), "ttl-control", {"text": "ttl control"}, ttl=0.05)

    ranked = store.search(("memories",), query="python", limit=2)
    assert [result.key for result in ranked] == ["python", "java"]

    time.sleep(1.0)
    assert store.get(("memories",), "ttl", refresh_ttl=True) is not None

    control_expiration_deadline = time.monotonic() + 4.5
    while time.monotonic() < control_expiration_deadline:
        if store.get(("memories",), "ttl-control", refresh_ttl=False) is None:
            break
        time.sleep(0.2)
    else:
        pytest.fail("Control TTL item did not expire in time")

    assert store.get(("memories",), "ttl", refresh_ttl=False) is not None

    expiration_deadline = time.monotonic() + 3.5
    while time.monotonic() < expiration_deadline:
        if store.get(("memories",), "ttl", refresh_ttl=False) is None:
            break
        time.sleep(0.2)
    else:
        pytest.fail("TTL item did not expire after the refreshed TTL window")


@pytest.mark.asyncio
async def test_store_async_round_trip(store_factory) -> None:
    store = store_factory()

    await store.aput(("async",), "note", {"text": "python async"})
    item = await store.aget(("async",), "note")
    results = await store.asearch(("async",), query="python", limit=1)

    assert item is not None
    assert item.key == "note"
    assert [result.key for result in results] == ["note"]

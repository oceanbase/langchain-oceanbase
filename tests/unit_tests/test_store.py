from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from langchain_core.embeddings import Embeddings
from sqlalchemy import MetaData, create_engine

from langchain_oceanbase.store import OceanBaseStore


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
            float((len(lowered) % 7) + 1),
        ]


@pytest.fixture
def store() -> OceanBaseStore:
    fake_client = SimpleNamespace(
        engine=create_engine("sqlite:///:memory:"),
        metadata_obj=MetaData(),
    )
    with patch("langchain_oceanbase.store.ObVecClient", return_value=fake_client):
        instance = OceanBaseStore(
            connection_args={
                "host": "localhost",
                "port": "2881",
                "user": "root",
                "password": "",
                "db_name": "test",
            },
            index={"dims": 3, "embed": DummyEmbeddings(), "fields": ["text"]},
            ttl_config={"refresh_on_read": True},
        )
        instance.setup()
        return instance


def test_put_and_get_round_trip(store: OceanBaseStore) -> None:
    store.put(("users", "123"), "prefs", {"theme": "dark", "age": 12})

    item = store.get(("users", "123"), "prefs")

    assert item is not None
    assert item.namespace == ("users", "123")
    assert item.key == "prefs"
    assert item.value == {"theme": "dark", "age": 12}


def test_search_honors_filters_and_comparison_operators(store: OceanBaseStore) -> None:
    store.put(("docs", "team-a"), "one", {"text": "python guide", "age": 12})
    store.put(("docs", "team-a"), "two", {"text": "java guide", "age": 8})
    store.put(("docs", "team-b"), "three", {"text": "python tips", "age": 20})

    results = store.search(
        ("docs",),
        filter={"age": {"$gte": 10}},
        limit=10,
    )

    assert [result.key for result in results] == ["one", "three"]


def test_search_treats_namespace_prefix_metacharacters_as_literals(
    store: OceanBaseStore,
) -> None:
    store.put(("team_%", "alpha"), "literal-percent", {"text": "python literal"})
    store.put(("teamX", "alpha"), "wildcard-candidate", {"text": "python wildcard"})
    store.put(("team__", "beta"), "literal-underscore", {"text": "python underscore"})
    store.put(("teamZZ", "beta"), "underscore-candidate", {"text": "python wildcard"})

    percent_results = store.search(("team_%",), limit=10)
    underscore_results = store.search(("team__",), limit=10)

    assert [result.key for result in percent_results] == ["literal-percent"]
    assert [result.key for result in underscore_results] == ["literal-underscore"]


def test_semantic_search_ranks_indexed_items_and_fills_with_scoreless_items(
    store: OceanBaseStore,
) -> None:
    store.put(("memories",), "python", {"text": "python patterns"})
    store.put(("memories",), "scoreless", {"text": "python hidden"}, index=False)
    store.put(("memories",), "java", {"text": "java patterns"})

    results = store.search(("memories",), query="python", limit=2)

    assert [result.key for result in results] == ["python", "java"]
    assert results[0].score is not None

    mixed = store.search(("memories",), query="python", limit=3)
    assert [result.key for result in mixed] == ["python", "java", "scoreless"]
    assert mixed[2].score is None


def test_list_namespaces_supports_prefix_suffix_and_depth(store: OceanBaseStore) -> None:
    store.put(("users", "1", "prefs"), "a", {"text": "python"})
    store.put(("users", "1", "docs"), "b", {"text": "java"})
    store.put(("users", "2", "prefs"), "c", {"text": "python"})
    store.put(("teams", "alpha"), "d", {"text": "python"})

    by_prefix = store.list_namespaces(prefix=("users", "*"), max_depth=2)
    by_suffix = store.list_namespaces(suffix=("prefs",))

    assert by_prefix == [("users", "1"), ("users", "2")]
    assert by_suffix == [("users", "1", "prefs"), ("users", "2", "prefs")]


def test_ttl_expires_items_and_refreshes_on_read(store: OceanBaseStore) -> None:
    store.put(("ttl",), "ephemeral", {"text": "python ttl"}, ttl=0.002)

    time.sleep(0.03)
    assert store.get(("ttl",), "ephemeral", refresh_ttl=True) is not None

    time.sleep(0.03)
    assert store.get(("ttl",), "ephemeral", refresh_ttl=False) is not None

    time.sleep(0.12)
    assert store.get(("ttl",), "ephemeral", refresh_ttl=False) is None


@pytest.mark.asyncio
async def test_async_store_methods_round_trip(store: OceanBaseStore) -> None:
    await store.aput(("async",), "doc", {"text": "python async"})

    item = await store.aget(("async",), "doc")
    results = await store.asearch(("async",), query="python", limit=1)

    assert item is not None
    assert item.key == "doc"
    assert [result.key for result in results] == ["doc"]


@pytest.mark.asyncio
async def test_abatch_offloads_sync_batch_work(store: OceanBaseStore) -> None:
    calls: list[tuple[object, tuple[object, ...]]] = []

    async def fake_to_thread(func: object, /, *args: object, **kwargs: object) -> object:
        calls.append((func, args))
        assert kwargs == {}
        return "sentinel"

    with patch("langchain_oceanbase.store.asyncio.to_thread", side_effect=fake_to_thread):
        result = await store.abatch([])

    assert result == "sentinel"
    assert len(calls) == 1
    func, args = calls[0]
    assert getattr(func, "__self__", None) is store
    assert getattr(func, "__func__", None) is OceanBaseStore.batch
    assert args == ([],)

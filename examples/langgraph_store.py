"""Minimal LangGraph store example for OceanBaseStore."""

from __future__ import annotations

from langchain_core.embeddings import Embeddings

from langchain_oceanbase import OceanBaseStore


class DemoEmbeddings(Embeddings):
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
            1.0 if "database" in lowered else 0.0,
            float((len(lowered) % 13) + 1),
        ]


def main() -> None:
    store = OceanBaseStore(
        connection_args={
            "host": "127.0.0.1",
            "port": "2881",
            "user": "root@test",
            "password": "",
            "db_name": "test",
        },
        index={"dims": 3, "embed": DemoEmbeddings(), "fields": ["memory"]},
        ttl_config={"refresh_on_read": True, "default_ttl": 60},
    )
    store.setup()

    namespace = ("memories", "user-123")
    store.put(
        namespace,
        "favorite-language",
        {"memory": "The user prefers Python for data tooling."},
    )
    store.put(
        namespace,
        "database-note",
        {"memory": "OceanBase is the preferred database backend."},
        ttl=30,
    )

    exact = store.get(namespace, "favorite-language")
    semantic = store.search(namespace, query="python tooling", limit=2)
    namespaces = store.list_namespaces(prefix=("memories",))

    print("Exact item:", exact)
    print("Semantic results:", semantic)
    print("Namespaces:", namespaces)


if __name__ == "__main__":
    main()

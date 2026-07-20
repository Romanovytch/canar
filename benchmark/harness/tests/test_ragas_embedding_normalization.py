from __future__ import annotations

import asyncio

from embedding_normalization import NormalizingEmbeddings


# APOSTROPHE NORMALIZATION WORKAROUND test delegate. This file can be removed
# together with the RAGAS wrapper if the upstream Ollama/bge-m3 issue is fixed.
class RecordingEmbeddings:
    def __init__(self):
        self.calls: list[tuple[str, object]] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(("documents", texts))
        return [[1.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        self.calls.append(("query", text))
        return [1.0]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(("adocuments", texts))
        return [[1.0] for _ in texts]

    async def aembed_query(self, text: str) -> list[float]:
        self.calls.append(("aquery", text))
        return [1.0]


def test_normalizing_embeddings_covers_sync_and_async_calls():
    delegate = RecordingEmbeddings()
    embeddings = NormalizingEmbeddings(delegate)  # type: ignore[arg-type]

    assert embeddings.embed_query("l’enquêté") == [1.0]
    assert embeddings.embed_documents(["l‘un", "déjà normalisé"]) == [[1.0], [1.0]]
    assert asyncio.run(embeddings.aembed_query("lʼenquêté")) == [1.0]
    assert asyncio.run(embeddings.aembed_documents(["l＇un"])) == [[1.0]]

    assert delegate.calls == [
        ("query", "l'enquêté"),
        ("documents", ["l'un", "déjà normalisé"]),
        ("aquery", "l'enquêté"),
        ("adocuments", ["l'un"]),
    ]

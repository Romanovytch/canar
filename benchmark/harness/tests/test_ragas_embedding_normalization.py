from __future__ import annotations

import asyncio
import json

import embedding_normalization
import pytest
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


# INTERMITTENT EMBEDDING FAILURE tests. The server answers 500 now and then and
# the text that triggers it is a RAGAS-generated question that survives nowhere,
# so the wrapper both retries and records what it sent.
class FlakyEmbeddings:
    """Fails the first `failures` calls, then works."""

    def __init__(self, failures: int):
        self.failures = failures
        self.attempts = 0

    def _maybe_fail(self):
        self.attempts += 1
        if self.attempts <= self.failures:
            raise RuntimeError("500 unsupported value: NaN")

    def embed_documents(self, texts):
        self._maybe_fail()
        return [[1.0] for _ in texts]

    def embed_query(self, text):
        self._maybe_fail()
        return [1.0]

    async def aembed_documents(self, texts):
        self._maybe_fail()
        return [[1.0] for _ in texts]

    async def aembed_query(self, text):
        self._maybe_fail()
        return [1.0]


def test_a_transient_failure_is_retried_and_the_score_survives(monkeypatch):
    monkeypatch.setattr(embedding_normalization, "RETRY_PAUSE_S", 0)
    flaky = FlakyEmbeddings(failures=2)

    result = NormalizingEmbeddings(flaky).embed_query("Quelle fonction utiliser ?")

    assert result == [1.0]
    assert flaky.attempts == 3          # two failures, then the answer


def test_a_failure_that_never_clears_is_raised(monkeypatch):
    """Retrying forever would hide the problem; the run must still see it."""
    monkeypatch.setattr(embedding_normalization, "RETRY_PAUSE_S", 0)

    with pytest.raises(RuntimeError):
        NormalizingEmbeddings(FlakyEmbeddings(failures=99)).embed_query("q")


def test_the_failing_text_is_written_down(monkeypatch, tmp_path):
    """The point of the log: the input is a question RAGAS generated and threw
    away, so without this there is nothing left to diagnose."""
    monkeypatch.setattr(embedding_normalization, "RETRY_PAUSE_S", 0)
    log = tmp_path / "embedding_failures.jsonl"
    wrapper = NormalizingEmbeddings(FlakyEmbeddings(failures=99), failure_log=log)

    with pytest.raises(RuntimeError):
        wrapper.embed_documents(["première question", "deuxième question"])

    entry = json.loads(log.read_text(encoding="utf-8").splitlines()[0])
    assert entry["count"] == 2
    assert "première question" in entry["texts"][0]
    assert "NaN" in entry["error"]


def test_nothing_is_written_when_the_embedding_works(monkeypatch, tmp_path):
    monkeypatch.setattr(embedding_normalization, "RETRY_PAUSE_S", 0)
    log = tmp_path / "embedding_failures.jsonl"

    NormalizingEmbeddings(FlakyEmbeddings(failures=0), failure_log=log).embed_query("q")

    assert not log.exists()


def test_the_async_path_retries_too(monkeypatch):
    """RAGAS picks sync or async depending on the metric, so both need it."""
    monkeypatch.setattr(embedding_normalization, "RETRY_PAUSE_S", 0)
    flaky = FlakyEmbeddings(failures=1)

    result = asyncio.run(NormalizingEmbeddings(flaky).aembed_query("q"))

    assert result == [1.0]
    assert flaky.attempts == 2

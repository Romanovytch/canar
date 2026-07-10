from __future__ import annotations

import pytest

from canar.app.retrieval.models import RetrievalHit
from canar.app.retrieval.rerank.rerankers import BGEReranker, QwenReranker, build_reranker
from canar.app.retrieval.rerank.rerankers._candidate_utils import resolve_reranker_device


class FakeCuda:
    def __init__(self, available: bool):
        self.available = available

    def is_available(self) -> bool:
        return self.available


class FakeTorch:
    def __init__(self, cuda_available: bool):
        self.cuda = FakeCuda(cuda_available)


class FakeBGEReranker(BGEReranker):
    def __init__(self, scores: list[float]):
        super().__init__(device="cuda")
        self.scores = scores
        self.pairs = []

    def _score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        self.pairs = pairs
        return self.scores


class FakeQwenReranker(QwenReranker):
    def __init__(self, scores: list[float]):
        super().__init__(device="cuda")
        self.scores = scores
        self.pairs = []

    def _score_pairs(self, pairs: list[str]) -> list[float]:
        self.pairs = pairs
        return self.scores


def test_bge_reranker_orders_hits_and_adds_rerank_score_without_model_load():
    hits = [
        RetrievalHit(text="first", collection="docs", score=1.0, score_norm=1.0),
        RetrievalHit(text="second", collection="docs", score=0.8, score_norm=0.8),
        RetrievalHit(text="third", collection="docs", score=0.6, score_norm=0.6),
    ]
    reranker = FakeBGEReranker([0.1, 0.9, 0.4])

    reranked = reranker.rerank("query", hits, top_k=2)

    assert [hit.text for hit in reranked] == ["second", "third"]
    assert [hit.rerank_score for hit in reranked] == [0.9, 0.4]
    assert all(hit.rerank_score is None for hit in hits)
    assert reranker.pairs == [
        ("query", "Document: first"),
        ("query", "Document: second"),
        ("query", "Document: third"),
    ]


def test_qwen_reranker_formats_instruction_and_keeps_tie_order():
    hits = [
        RetrievalHit(text="alpha", collection="docs", score=1.0, score_norm=1.0),
        RetrievalHit(text="beta", collection="docs", score=0.9, score_norm=0.9),
    ]
    reranker = FakeQwenReranker([0.5, 0.5])

    reranked = reranker.rerank("query", hits)

    assert [hit.text for hit in reranked] == ["alpha", "beta"]
    expected_prefix = f"<Instruct>: {QwenReranker.DEFAULT_INSTRUCTION}\n<Query>: query"
    assert reranker.pairs == [
        f"{expected_prefix}\n<Document>: Document: alpha",
        f"{expected_prefix}\n<Document>: Document: beta",
    ]


def test_reranker_rejects_blank_query_negative_top_k_and_missing_text():
    reranker = FakeBGEReranker([1.0])
    hit = RetrievalHit(text="answer", collection="docs", score=1.0, score_norm=1.0)

    with pytest.raises(ValueError, match="query must be a non-empty string"):
        reranker.rerank("  ", [hit])
    with pytest.raises(ValueError, match="top_k must be greater than or equal to 0"):
        reranker.rerank("query", [hit], top_k=-1)
    empty_hit = RetrievalHit(text="", collection="docs", score=1.0, score_norm=1.0)
    with pytest.raises(ValueError, match="candidate text must be non-empty"):
        reranker.rerank("query", [empty_hit])


@pytest.mark.parametrize(
    ("name", "expected_type", "expected_model"),
    [
        ("bge-v2-m3", BGEReranker, "BAAI/bge-reranker-v2-m3"),
        ("qwen-0.6b", QwenReranker, "Qwen/Qwen3-Reranker-0.6B"),
        ("qwen-4b", QwenReranker, "Qwen/Qwen3-Reranker-4B"),
        ("qwen-8b", QwenReranker, "Qwen/Qwen3-Reranker-8B"),
    ],
)
def test_build_reranker_supports_only_registered_models(
    name, expected_type, expected_model
):
    reranker = build_reranker(name, device="cuda", max_length=256)

    assert isinstance(reranker, expected_type)
    assert reranker.model_name == expected_model
    assert reranker.device == "cuda"
    assert reranker.max_length == 256


@pytest.mark.parametrize("name", ["bge", "qwen", "cross-encoder"])
def test_build_reranker_rejects_unregistered_names(name):
    with pytest.raises(ValueError, match="Unsupported reranker"):
        build_reranker(name)


def test_build_reranker_rejects_arbitrary_model_override():
    with pytest.raises(TypeError, match="model_name"):
        build_reranker("bge-v2-m3", model_name="custom/bge")


def test_resolve_reranker_device_supports_auto_force_and_default_modes():
    assert resolve_reranker_device(FakeTorch(cuda_available=True), "auto") == "cuda"
    assert resolve_reranker_device(FakeTorch(cuda_available=False), "auto") == "cpu"
    assert resolve_reranker_device(FakeTorch(cuda_available=True), "cuda") == "cuda"
    assert resolve_reranker_device(FakeTorch(cuda_available=True), "cpu") == "cpu"
    assert resolve_reranker_device(FakeTorch(cuda_available=True), None) is None

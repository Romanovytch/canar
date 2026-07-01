from __future__ import annotations

import argparse
from dataclasses import replace
from textwrap import shorten

from canar.app.api.embed_client import EmbedClient
from canar.app.config import AppConfig
from canar.app.retrieval.models import RetrievalHit
from canar.app.retrieval.service import RetrievalService


DEFAULT_COLLECTION = "utilitr_bgem3_ds"
DEFAULT_QDRANT_URL = "http://localhost:6360"


def build_service(cfg: AppConfig) -> RetrievalService:
    embed_client = EmbedClient(cfg.embed_base, cfg.embed_model, cfg.embed_key)
    return RetrievalService.from_config(cfg, embed_client=embed_client)


def print_hits(title: str, hits: list[RetrievalHit], max_chars: int) -> None:
    print(f"\n{'=' * 100}")
    print(title)
    print("=" * 100)
    if not hits:
        print("No hits")
        return

    for index, hit in enumerate(hits, start=1):
        rerank_score = getattr(hit, "rerank_score", None)
        rerank_display = "None" if rerank_score is None else f"{rerank_score:.6f}"
        source = hit.source_url or hit.source or hit.metadata.get("source") or ""
        section = hit.section or hit.metadata.get("section") or ""
        point_id = (
            hit.metadata.get("id")
            or hit.metadata.get("_id")
            or hit.metadata.get("point_id")
            or hit.metadata.get("chunk_id")
            or ""
        )
        text = shorten(" ".join(hit.text.split()), width=max_chars, placeholder=" ...")

        print(f"[{index}] collection={hit.collection}")
        print(f"    score={hit.score:.6f} score_norm={hit.score_norm:.6f}")
        print(f"    rerank_score={rerank_display}")
        if point_id:
            print(f"    id={point_id}")
        if source:
            print(f"    source={source}")
        if section:
            print(f"    section={section}")
        print(f"    text={text}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare hybrid retrieval with BGE and optional Qwen reranking."
    )
    parser.add_argument("query", help="Question/query to retrieve against.")
    parser.add_argument("--agent", default="r_helpdesk", help="Agent profile to use.")
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION,
        help="Qdrant collection to query.",
    )
    parser.add_argument(
        "--qdrant-url",
        default=DEFAULT_QDRANT_URL,
        help="Qdrant URL.",
    )
    parser.add_argument("--top-n", type=int, default=None, help="Override RERANK_TOP_N.")
    parser.add_argument("--device", default=None, help="Override RERANK_DEVICE.")
    parser.add_argument("--max-chars", type=int, default=280, help="Max text chars per hit.")
    parser.add_argument(
        "--include-qwen",
        action="store_true",
        help="Also run Qwen. This can download/load Qwen/Qwen3-Reranker-8B.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = AppConfig()
    cfg = replace(
        cfg,
        qdrant_url=args.qdrant_url,
        qdrant_collections=(args.collection,),
    )
    if args.top_n is not None:
        cfg = replace(cfg, rerank_top_n=args.top_n)
    if args.device is not None:
        cfg = replace(cfg, rerank_device=args.device)
    cfg.validate()

    print(f"query={args.query!r}")
    print(f"agent={args.agent}")
    print(f"qdrant_url={cfg.qdrant_url}")
    print(f"collection={args.collection}")
    print(f"sparse_model={cfg.fastembed_sparse_model}")
    print(f"rerank_device={cfg.rerank_device}")
    print(f"rerank_top_n={cfg.rerank_top_n}")

    hybrid_service = build_service(replace(cfg, reranker_name="bge", rerank_enabled=False))
    hybrid_hits = hybrid_service.search(args.agent, args.query, rerank=False)
    print_hits("HYBRID", hybrid_hits, args.max_chars)

    bge_service = build_service(replace(cfg, reranker_name="bge", rerank_enabled=True))
    bge_hits = bge_service.search(args.agent, args.query, rerank=True)
    print_hits("HYBRID + BGE RERANK", bge_hits, args.max_chars)

    if args.include_qwen:
        qwen_service = build_service(replace(cfg, reranker_name="qwen", rerank_enabled=True))
        qwen_hits = qwen_service.search(args.agent, args.query, rerank=True)
        print_hits("HYBRID + QWEN RERANK", qwen_hits, args.max_chars)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from dataclasses import replace
from textwrap import shorten

from canar.app.api.embed_client import EmbedClient
from canar.app.config import AppConfig
from canar.app.retrieval.models import RetrievalHit, RetrievalProfile
from canar.app.retrieval.profiles import build_retrieval_profiles
from canar.app.retrieval.rerank.rerankers import RERANKER_SPECS
from canar.app.retrieval.service import RetrievalService

DEFAULT_COLLECTION = "utilitr_bgem3_ds"

DEFAULT_QDRANT_URL = "http://localhost:6360"
def build_profiles(
    cfg: AppConfig,
    *,
    reranker_name: str = "bge-v2-m3",
    top_k: int | None = None,
    device: str | None = None,
) -> dict[str, RetrievalProfile]:
    profiles = build_retrieval_profiles(
        tuple(cfg.qdrant_collections),
        dense_vector_name=cfg.qdrant_dense_vector_name or None,
        sparse_vector_name=cfg.qdrant_sparse_vector_name or None,
    )
    template = profiles["hybrid_rerank_bge"]
    params = template.rerank_params()
    assert params is not None
    profile_name = f"compare_hybrid_rerank_{reranker_name}"
    profiles[profile_name] = replace(
        template,
        name=profile_name,
        rerank=replace(
            params,
            reranker_name=reranker_name,
            output_top_k=top_k if top_k is not None else params.output_top_k,
            device=device if device is not None else params.device,
        ),
    )
    return profiles


def build_service(
    cfg: AppConfig,
    *,
    agent: str,
    use_rerank: bool,
    reranker_name: str = "bge-v2-m3",
    top_k: int | None = None,
    device: str | None = None,
) -> RetrievalService:
    embed_client = EmbedClient(cfg.embed_base, cfg.embed_model, cfg.embed_key)
    profiles = build_profiles(
        cfg,
        reranker_name=reranker_name,
        top_k=top_k,
        device=device,
    )
    profile_name = (
        f"compare_hybrid_rerank_{reranker_name}" if use_rerank else "hybrid"
    )
    return RetrievalService.from_config(
        cfg,
        embed_client=embed_client,
        profiles=profiles,
        agent_profiles={agent: profile_name},
    )


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
        description="Compare hybrid retrieval with every registered reranker."
    )
    parser.add_argument("query", help="Question/query to retrieve against.")
    parser.add_argument("--agent", default="generic_agent", help="Agent profile to use.")
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
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Override the reranker output count; fusion.output_top_k controls candidates.",
    )
    parser.add_argument("--device", default=None, help="Override the profile rerank device.")
    parser.add_argument("--max-chars", type=int, default=280, help="Max text chars per hit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = AppConfig()
    cfg = replace(
        cfg,
        qdrant_url=args.qdrant_url,
        qdrant_collections=(args.collection,),
    )
    cfg.validate()

    print(f"query={args.query!r}")
    print(f"agent={args.agent}")
    print(f"qdrant_url={cfg.qdrant_url}")
    print(f"collection={args.collection}")
    print(f"sparse_model={cfg.fastembed_sparse_model}")
    print(f"rerank_device={args.device or 'profile default'}")
    print(f"output_top_k={args.top_k or 'profile default'}")

    hybrid_service = build_service(cfg, agent=args.agent, use_rerank=False)
    hybrid_hits = hybrid_service.search(args.agent, args.query)
    print_hits("HYBRID", hybrid_hits, args.max_chars)

    for reranker_name in RERANKER_SPECS:
        rerank_service = build_service(
            cfg,
            agent=args.agent,
            use_rerank=True,
            reranker_name=reranker_name,
            top_k=args.top_k,
            device=args.device,
        )
        reranked_hits = rerank_service.search(args.agent, args.query)
        print_hits(
            f"HYBRID + {reranker_name.upper()} RERANK",
            reranked_hits,
            args.max_chars,
        )


if __name__ == "__main__":
    main()

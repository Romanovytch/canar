"""
End-to-end RAGAS evaluation of the REAL CanaR + AgoRa pipeline.

Unlike ../replica/eval_demo.py and ../replica/eval_utilitr.py (which replicate
the RAG flow with their own code), this script imports and calls CanaR's
actual modules, against the collection that AgoRa actually built:

    AgoRa ingest  ->  Qdrant `utilitr_v1`  ->  CanaR retrieval + generation  ->  RAGAS

Mapping to the CanaR chat UI (canar/app/main.py, r_helpdesk branch),
using Julien's modular retrieval (canar/app/retrieval/):

    UI step (main.py)                              | here
    -----------------------------------------------|------------------------------
    retrieval = RetrievalService.from_config(...)  | same service, same profile
    citations = retrieval.search(agent, q)         | same call (embeds + retrieves)
    messages = r_helpdesk.build_messages(q, cits)  | same call, same CoachR prompt
    chat.stream_chat(messages)                     | same call, stream joined

So a score here measures the product, not a replica. The retrieval profiles to
run (top_k, score_threshold, ...) come from benchmark/config.yaml; the dataset
is run once per profile so strategies can be compared side by side.

Prerequisite — the collection must have been built by AgoRa (its payload
carries the `source` field CanaR filters on; a hand-rolled ingest won't match):

    cd /home/cereq/opt/pedro/agora && source .venv/bin/activate
    agora-ingest --sources-config-path sources.yaml --source utilitr \
                 --collection utilitr_v1 --dotenv-path .env --drop-collection

Run:
    cd /home/cereq/opt/pedro/canar/benchmark
    source .venv/bin/activate
    python e2e/eval_e2e.py
"""

import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from qdrant_client import QdrantClient
from ragas.metrics import Faithfulness, ResponseRelevancy
from ragas.run_config import RunConfig

HERE = Path(__file__).parent                  # benchmark/e2e/
BENCH_DIR = HERE.parent                      # benchmark/
REPO_ROOT = HERE.parent.parent                # the canar repo root

# Make the canar package importable without installing it into this venv.
# (When this becomes a feature in dev, `pip install -e .` does the same job.)
sys.path.insert(0, str(REPO_ROOT))
# Make the shared benchmark harness importable.
sys.path.insert(0, str(BENCH_DIR / "harness"))

# CanaR's AppConfig finds its .env by walking up from the *current working
# directory*. Running from benchmark/ it would pick up benchmark/.env
# (no QDRANT_COLLECTIONS) instead of canar/.env. Pre-loading the app's .env
# explicitly makes the benchmark independent of where it's launched from.
from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env", override=True)

# CanaR's real modules — the exact code the Streamlit app runs.
# Importing AppConfig also loads canar/.env (LLM, embeddings, Qdrant, collections).
from bench_config import load_config  # noqa: E402
from ragas_bench import DatasetSpec, PipelineOutput, run_benchmark  # noqa: E402
from resource_probe import hardware_profile, probe  # noqa: E402

from canar.app.agents import generic_agent  # noqa: E402
from canar.app.api.embed_client import EmbedClient, FastEmbedClient  # noqa: E402
from canar.app.api.llm_client import ChatClient  # noqa: E402
from canar.app.config import AppConfig  # noqa: E402
from canar.app.retrieval.models import RetrievalQuery  # noqa: E402
from canar.app.retrieval.service import RetrievalService  # noqa: E402

# ---------------------------------------------------------------------------
# Config — product settings from canar/.env; benchmark settings from config.yaml.
# ---------------------------------------------------------------------------

cfg = AppConfig()
cfg.validate()

BENCH = load_config(BENCH_DIR / "config.yaml")

DATASET = DatasetSpec(
    path=BENCH_DIR / BENCH.dataset,   # CSV or YAML — format detected from extension
    limit=BENCH.limit,                # null in YAML = all questions
)

# Generation budget. qwen3.5 is a *reasoning* model: it spends a hidden token
# budget "thinking" before answering, so the app's default max_tokens=2048 runs
# out mid-thought and returns EMPTY answers. Keep high. Env overrides YAML.
GEN_MAX_TOKENS = int(os.environ.get("GEN_MAX_TOKENS", BENCH.gen_max_tokens))

# RAGAS judge — may differ from the product LLM. A non-reasoning model
# (qwen2.5:7b) avoids the timeouts of the reasoning 9B. Priority: env > YAML > product.
JUDGE_MODEL = os.environ.get("JUDGE_MODEL") or BENCH.judge_model or cfg.llm_model

# Optional resource benchmark (CPU/memory/GPU per phase). Off by default; when on
# it adds a tiny sampler thread per phase. Priority: env > YAML. Turn on with
# MEASURE_RESOURCES=1 or run.measure_resources in config.yaml.
_measure_env = os.environ.get("MEASURE_RESOURCES")
MEASURE_RESOURCES = (
    _measure_env.strip().lower() not in ("", "0", "false", "no", "off")
    if _measure_env is not None
    else BENCH.measure_resources
)


def _git_commit() -> str:
    """Short commit of the canar repo, so a result can be traced to the code."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


# Provenance — stamped on every result row so a CSV describes what produced it
# (which embedding model, which collection, which judge, which code). This is
# what makes results comparable across machines despite per-developer configs.
PROVENANCE = {
    "embed_model": cfg.embed_model,
    "collection": ",".join(cfg.qdrant_collections),
    "judge_model": JUDGE_MODEL,
    "git_commit": _git_commit(),
}
# When measuring resources, also stamp the machine (CPU / RAM / GPU) so the
# numbers can be compared fairly across computers.
if MEASURE_RESOURCES:
    PROVENANCE.update(hardware_profile())

# CanaR clients (product code).
embed = EmbedClient(cfg.embed_base, cfg.embed_model, cfg.embed_key)
chat = ChatClient(cfg.llm_base, cfg.llm_key, cfg.llm_model,{"reasoning_effort": cfg.llm_thinking})

# Sparse query encoder (FastEmbed/BM25), built only when canar/.env sets
# FASTEMBED_SPARSE_MODEL. None means no sparse/hybrid profile can run.
sparse_embed = FastEmbedClient(cfg.fastembed_sparse_model) if cfg.fastembed_sparse_model else None

# The product's retrieval service builds the real strategies exactly as the app
# does — simple_vector (dense), simple_sparse (BM25), and hybrid (dense + sparse
# fused). The benchmark drives these instead of reconstructing them, so it
# measures the colleagues' actual retrieval code, hybrid composition included.

service = RetrievalService.from_config(cfg, embed, sparse_embed)

# RAGAS judge — reuses the same local LLM/embeddings endpoints.
# (LangChain wrappers because RAGAS expects them; this is eval-side only,
#  the product's answers are generated by CanaR's ChatClient above.)
judge_llm = ChatOpenAI(
    model=JUDGE_MODEL,
    openai_api_base=cfg.llm_base,
    openai_api_key=cfg.llm_key or "EMPTY",
    temperature=0.0,
    reasoning_effort=cfg.llm_thinking,   #
    extra_body={"keep_alive": "10m"},   # avoid Ollama unloading the model between judge calls
)
judge_embeddings = OpenAIEmbeddings(
    model=cfg.embed_model,
    openai_api_base=cfg.embed_base,
    openai_api_key=cfg.embed_key or "EMPTY",
    check_embedding_ctx_length=False,   # send raw strings; Ollama rejects token arrays
)


def check_environment() -> None:
    """
    Abort if canar/.env does not match the environment the config.yaml declares.
    Keeps results comparable: a CSV is only ever produced with the intended
    embedding model and collection, never silently with a developer's local one.
    """
    expected = BENCH.environment or {}
    exp_model = expected.get("embed_model")
    if exp_model and exp_model != cfg.embed_model:
        sys.exit(
            f"Environment mismatch: config.yaml expects EMBED_MODEL '{exp_model}' but "
            f"canar/.env has '{cfg.embed_model}'.\n"
            "Align canar/.env (or update config.yaml's environment block) before running."
        )
    exp_collection = expected.get("collection")
    if exp_collection and exp_collection not in cfg.qdrant_collections:
        sys.exit(
            f"Environment mismatch: config.yaml expects collection '{exp_collection}' but "
            f"canar/.env has QDRANT_COLLECTIONS={list(cfg.qdrant_collections)}.\n"
            "Align canar/.env (or update config.yaml's environment block) before running."
        )


def preflight() -> None:
    """Fail fast with a clear message if the AgoRa collection isn't ready."""
    check_environment()
    # Validate only the vectors the active profiles actually query.
    strategies = {p.strategy for p in BENCH.profiles}
    needs_dense = bool(strategies & {"simple_vector", "hybrid"})
    needs_sparse = bool(strategies & {"simple_sparse", "hybrid"})
    client = QdrantClient(url=cfg.qdrant_url, api_key=cfg.qdrant_api_key or None,
                          check_compatibility=False)
    for col in cfg.qdrant_collections:
        if not client.collection_exists(col):
            sys.exit(
                f"Collection '{col}' not found at {cfg.qdrant_url}.\n"
                "Build it with AgoRa first (see the command in this file's docstring)."
            )
        # one sample point is enough to verify the AgoRa payload contract
        pts, _ = client.scroll(collection_name=col, limit=1, with_payload=True)
        if not pts:
            sys.exit(f"Collection '{col}' is empty — run the AgoRa ingest first.")
        if "source" not in pts[0].payload:
            sys.exit(
                f"Collection '{col}' has no 'source' payload field — it was not built "
                "by AgoRa, and CanaR's source filter would return nothing. "
                "Re-ingest with agora-ingest."
            )
        params = client.get_collection(col).config.params
        vectors_cfg = params.vectors                 # dict (named) or single config
        sparse_cfg = params.sparse_vectors or {}     # dict of named sparse vectors

        if needs_dense:
            # The dense vector must exist (by name on multi-vector collections) and
            # its size must match the embedding model CanaR queries with
            # (e.g. nomic=768 vs bge-m3=1024), or every dense search crashes.
            if isinstance(vectors_cfg, dict):
                dense_name = cfg.qdrant_dense_vector_name or next(iter(vectors_cfg), None)
                if dense_name not in vectors_cfg:
                    sys.exit(
                        f"Collection '{col}' has no dense vector named {dense_name!r} "
                        f"(named vectors: {list(vectors_cfg)}). Fix QDRANT_DENSE_VECTOR_NAME "
                        "or re-ingest. See SETUP.md."
                    )
                col_dim = vectors_cfg[dense_name].size
            else:
                col_dim = vectors_cfg.size
            query_dim = len(embed.embed_query("probe"))
            if col_dim != query_dim:
                sys.exit(
                    f"Dimension mismatch: collection '{col}' dense vector is {col_dim}-dim "
                    f"but EMBED_MODEL '{cfg.embed_model}' produces {query_dim}-dim queries.\n"
                    "Re-ingest with the current embedding model (agora-ingest --drop-collection)."
                )

        if needs_sparse:
            # sparse/hybrid profiles query a named sparse vector; it must exist.
            sparse_name = cfg.qdrant_sparse_vector_name
            if not sparse_cfg or (sparse_name and sparse_name not in sparse_cfg):
                sys.exit(
                    f"Collection '{col}' has no sparse vector {sparse_name or '(any)'!r} "
                    f"(sparse vectors: {list(sparse_cfg)}). The sparse and hybrid profiles "
                    "need it — re-ingest with a sparse vector. See SETUP.md."
                )
    print(f"Preflight OK — collections {list(cfg.qdrant_collections)} ready at {cfg.qdrant_url}\n")


def build_searcher(spec):
    """
    Resolve the product strategy named by a config profile and return a callable
    that runs it. The strategy — and, for `hybrid`, its dense + sparse
    sub-strategies and fusion — is the one `RetrievalService` built from
    canar/.env, so the benchmark measures the shipped retrieval code, not a
    reimplementation. The query vectors are embedded the same way
    `RetrievalService.search` does, per strategy.
    """
    strategy = service.strategies.get(spec.strategy)
    if strategy is None:
        sys.exit(f"Strategy {spec.strategy!r} not available in RetrievalService "
                 f"(have: {list(service.strategies)}).")

    needs_dense = spec.strategy in {"simple_vector", "hybrid", "parent_child_hybrid"}
    needs_sparse = spec.strategy in {"simple_sparse", "hybrid", "parent_child_hybrid"}
    if needs_sparse and sparse_embed is None:
        sys.exit(f"Profile uses {spec.strategy!r} but FASTEMBED_SPARSE_MODEL is unset "
                 "in canar/.env — no sparse encoder available.")

    def search(question: str):
        return strategy.search(RetrievalQuery(
            text=question,
            profile_name=spec.name,
            dense_vector=embed.embed_query(question) if needs_dense else None,
            sparse_vector=sparse_embed.embed_query(question) if needs_sparse else None,
        ))

    return search


def make_pipeline(search):
    """Wrap one profile's retrieval into a benchmark pipeline = one product turn."""
    def ask_canar(question: str) -> PipelineOutput:
        # retrieve (timed for the Latency metric; resource-probed when enabled)
        with probe(MEASURE_RESOURCES) as r_usage:
            t0 = time.perf_counter()
            citations = search(question)              # list[RetrievalHit]
            retrieval_latency_s = time.perf_counter() - t0

        # build the exact prompt the app sends (CoachR system prompt + [S1].. context)
        messages, _src_list = generic_agent.build_messages(question, citations)

        # generate with CanaR's ChatClient (the app streams; we join the stream).
        # max_tokens generous so the reasoning model finishes thinking AND answers.
        # Timed separately from retrieval, useful e.g. to compare LLM models.
        with probe(MEASURE_RESOURCES) as g_usage:
            t1 = time.perf_counter()
            answer = "".join(
                chat.stream_chat(messages, temperature=0.0, max_tokens=GEN_MAX_TOKENS)
            )
            generation_latency_s = time.perf_counter() - t1
        # The reasoning model can spend its whole budget "thinking" and return an
        # empty answer. Flag it so a 0 score is read as "no answer", not "bad answer".
        if not answer.strip():
            answer = "[EMPTY_ANSWER]"

        # GPU is device-level; report the peak seen across the two phases as the
        # turn's figure (None when not measured / no GPU). The delta is the
        # turn's own memory growth, so it stays comparable across runs (#53).
        gpu_util = max(
            [v for v in (r_usage.gpu_util_pct, g_usage.gpu_util_pct) if v is not None],
            default=None,
        )
        gpu_mem_delta = max(
            [v for v in (r_usage.gpu_mem_delta_mb, g_usage.gpu_mem_delta_mb) if v is not None],
            default=None,
        )
        peak_rss = max(
            [v for v in (r_usage.peak_rss_mb, g_usage.peak_rss_mb) if v is not None],
            default=None,
        )
        # Per-process attribution: keep the phase with the larger footprint
        # (generation, in practice — that's where the LLM server loads memory).
        if (g_usage.gpu_mem_procs_mb or 0) >= (r_usage.gpu_mem_procs_mb or 0):
            proc_usage = g_usage
        else:
            proc_usage = r_usage

        # AgoRa stores the fiche path under "file_path" in the chunk payload
        # (RetrievalHit.metadata); that's what the retrieval metrics match on.
        return PipelineOutput(
            answer=answer,
            contexts=[hit.text for hit in citations],
            paths=[(hit.metadata or {}).get("file_path", "") for hit in citations],
            retrieval_latency_s=retrieval_latency_s,
            generation_latency_s=generation_latency_s,
            retrieval_cpu_s=r_usage.cpu_s,
            peak_rss_mb=peak_rss,
            gpu_util_pct=gpu_util,
            gpu_mem_delta_mb=gpu_mem_delta,
            gpu_mem_procs_mb=proc_usage.gpu_mem_procs_mb,
            gpu_procs=proc_usage.gpu_procs,
        )

    return ask_canar


def main() -> None:
    preflight()
    print(f"Profiles to benchmark: {[p.name for p in BENCH.profiles]} | judge: {JUDGE_MODEL}")
    print(f"Provenance: {PROVENANCE}\n")

    # One folder per benchmark run; each strategy gets a subfolder inside it.
    run_dir = HERE / "results" / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    summaries = []
    for spec in BENCH.profiles:
        df = run_benchmark(
            name=f"E2E [{spec.name}] (real CanaR + AgoRa pipeline)",
            dataset=DATASET,
            pipeline=make_pipeline(build_searcher(spec)),
            metrics=[Faithfulness(), ResponseRelevancy()],
            judge_llm=judge_llm,
            judge_embeddings=judge_embeddings,
            results_dir=HERE / "results",
            # local judge is slow: give each job room, and run a few in parallel
            # (Ollama serves them sequentially but overlaps prompt processing).
            run_config=RunConfig(timeout=900, max_workers=3),
            # profile name + provenance: tagged onto every row and kept out of the
            # score means, so each CSV records what produced it.
            tag={"profile": spec.name, **PROVENANCE},
            file_label=spec.name,
            group_dir=run_dir,            # all strategies of this run share run_dir
        )
        summaries.append((spec.name, df))

    # Side-by-side comparison — the point of comparing strategies. Printed and
    # also saved as comparison.csv at the run folder's root, so the run is a
    # self-contained artifact.
    if summaries:
        cols = ["hit_rate", "mrr", "recall", "precision", "ndcg",
                "retrieval_latency_s", "generation_latency_s",
                "retrieval_cpu_s", "peak_rss_mb",
                "gpu_util_pct", "gpu_mem_delta_mb", "gpu_mem_procs_mb",
                "faithfulness", "answer_relevancy"]
        rows = [
            {"profile": name, **{c: round(df[c].mean(), 3) for c in cols if c in df.columns}}
            for name, df in summaries
        ]
        comparison = pd.DataFrame(rows)
        run_dir.mkdir(parents=True, exist_ok=True)
        comparison.to_csv(run_dir / "comparison.csv", index=False)
        if len(summaries) > 1:
            print("\n=== Profile comparison (means) ===")
            print(comparison.to_string(index=False))
        print(f"\nRun saved to {run_dir}")


if __name__ == "__main__":
    main()

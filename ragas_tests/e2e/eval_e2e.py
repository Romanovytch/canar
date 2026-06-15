"""
End-to-end RAGAS evaluation of the REAL CanaR + AgoRa pipeline.

Unlike ../eval_rag.py and ../utilitr_bench/eval_utilitr.py (which replicate
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

So a score here measures the product, not a replica.

Prerequisite — the collection must have been built by AgoRa (its payload
carries the `source` field CanaR filters on; a hand-rolled ingest won't match):

    cd /home/cereq/opt/pedro/agora && source .venv/bin/activate
    agora-ingest --sources-config-path sources.yaml --source utilitr \
                 --collection utilitr_v1 --dotenv-path .env --drop-collection

Run:
    cd /home/cereq/opt/pedro/canar/ragas_tests
    source .venv/bin/activate
    python e2e/eval_e2e.py
"""

import os
import sys
from pathlib import Path

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from qdrant_client import QdrantClient
from ragas.metrics import Faithfulness, ResponseRelevancy
from ragas.run_config import RunConfig

HERE = Path(__file__).parent                  # ragas_tests/e2e/
RAGAS_TESTS = HERE.parent                      # ragas_tests/
REPO_ROOT = HERE.parent.parent                # the canar repo root

# Make the canar package importable without installing it into this venv.
# (When this becomes a feature in dev, `pip install -e .` does the same job.)
sys.path.insert(0, str(REPO_ROOT))
# Make the shared benchmark harness importable.
sys.path.insert(0, str(RAGAS_TESTS))

# CanaR's AppConfig finds its .env by walking up from the *current working
# directory*. Running from ragas_tests/ it would pick up ragas_tests/.env
# (no QDRANT_COLLECTIONS) instead of canar/.env. Pre-loading the app's .env
# explicitly makes the benchmark independent of where it's launched from.
from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env", override=True)

# CanaR's real modules — the exact code the Streamlit app runs.
# Importing AppConfig also loads canar/.env (LLM, embeddings, Qdrant, collections).
from ragas_bench import DatasetSpec, PipelineOutput, run_benchmark  # noqa: E402

from canar.app.agents import r_helpdesk  # noqa: E402
from canar.app.api.embed_client import EmbedClient  # noqa: E402
from canar.app.api.llm_client import ChatClient  # noqa: E402
from canar.app.config import AppConfig  # noqa: E402
from canar.app.retrieval.service import RetrievalService  # noqa: E402

# ---------------------------------------------------------------------------
# Config — single source of truth is canar/.env, exactly like the app.
# ---------------------------------------------------------------------------

cfg = AppConfig()
cfg.validate()

DATASET = DatasetSpec(
    path=HERE.parent / "utilitr_bench" / "datasets" / "utilitr_questions.csv",
    # columns: query, grading_notes, source_fiche (defaults already match)
    limit=None,   # None = all 12; set an int for a quicker pass
)
AGENT = "r_helpdesk"  # which agent profile RetrievalService exercises (main.py's RAG agent)
# top_k / source_filter / score-threshold pruning now live in the retrieval
# profile (canar/app/retrieval/profiles.py), exactly as the app uses them.

# qwen3.5 is a *reasoning* model: it spends a large hidden token budget
# "thinking" before emitting visible content. With the app's default
# max_tokens=2048 the budget runs out mid-thought and the answer comes back
# EMPTY (finish_reason="length", 0 chars) — which silently tanks every score.
# A generous cap lets the model think *and* answer. (Override via env.)
GEN_MAX_TOKENS = int(os.environ.get("GEN_MAX_TOKENS", "8192"))

# The RAGAS judge can be a different (faster / stronger) model than the product.
# Defaults to the product LLM; set JUDGE_MODEL to e.g. a non-reasoning model
# (qwen2.5:7b) to cut the judge timeouts seen with the reasoning 9B.
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", cfg.llm_model)

# CanaR's clients + retrieval service, built from the same config the app uses
embed = EmbedClient(cfg.embed_base, cfg.embed_model, cfg.embed_key)
chat = ChatClient(cfg.llm_base, cfg.llm_key, cfg.llm_model)
retrieval = RetrievalService.from_config(cfg, embed)  # Julien's modular retrieval, same as main.py

# RAGAS judge — reuses the same local LLM/embeddings endpoints.
# (LangChain wrappers because RAGAS expects them; this is eval-side only,
#  the product's answers are generated by CanaR's ChatClient above.)
judge_llm = ChatOpenAI(
    model=JUDGE_MODEL,
    openai_api_base=cfg.llm_base,
    openai_api_key=cfg.llm_key or "EMPTY",
    temperature=0.0,
    extra_body={"keep_alive": "10m"},   # avoid Ollama unloading the model between judge calls
)
judge_embeddings = OpenAIEmbeddings(
    model=cfg.embed_model,
    openai_api_base=cfg.embed_base,
    openai_api_key=cfg.embed_key or "EMPTY",
    check_embedding_ctx_length=False,   # send raw strings; Ollama rejects token arrays
)


def preflight() -> None:
    """Fail fast with a clear message if the AgoRa collection isn't ready."""
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
        # the collection's vector size must match the embedding model CanaR
        # queries with, or every search crashes (e.g. nomic=768 vs bge-m3=1024)
        col_dim = client.get_collection(col).config.params.vectors.size
        query_dim = len(embed.embed_query("probe"))
        if col_dim != query_dim:
            sys.exit(
                f"Dimension mismatch: collection '{col}' stores {col_dim}-dim vectors but "
                f"EMBED_MODEL '{cfg.embed_model}' produces {query_dim}-dim queries.\n"
                "Re-ingest the collection with the current embedding model "
                "(agora-ingest ... --drop-collection)."
            )
    print(f"Preflight OK — collections {list(cfg.qdrant_collections)} ready at {cfg.qdrant_url}\n")


def ask_canar(question: str) -> PipelineOutput:
    """
    The pipeline under test: one full CanaR turn, exactly as main.py does it
    for the r_helpdesk agent (Julien's modular retrieval: RetrievalService ->
    r_helpdesk -> ChatClient). This is the only product-specific glue; the
    benchmark loop/scoring/saving lives in ragas_bench.run_benchmark.
    """
    # 1+2. embed + retrieve through the same RetrievalService the app builds.
    #      search() embeds internally and applies the agent's profile
    #      (collections, top_k, source filter, score-threshold pruning) —
    #      the exact path main.py runs: retrieval.search(agent, user_input).
    citations = retrieval.search(AGENT, question)   # list[RetrievalHit]

    # 3. build the exact prompt the app sends (CoachR system prompt + [S1].. context)
    messages, _src_list = r_helpdesk.build_messages(question, citations)

    # 4. generate with CanaR's ChatClient. The app streams to the UI; here we
    #    join the stream. temperature=0.0 for reproducible benchmark runs
    #    (the app's default is 0.2). max_tokens generous so the reasoning model
    #    finishes thinking AND answers (see GEN_MAX_TOKENS note above).
    answer = "".join(
        chat.stream_chat(messages, temperature=0.0, max_tokens=GEN_MAX_TOKENS)
    )

    # RetrievalHit is a dataclass; the full Qdrant payload lives in .metadata.
    # AgoRa stores the fiche path under "file_path" (the standalone ingest used
    # "path") — this is what retrieval_hit matches against source_fiche.
    return PipelineOutput(
        answer=answer,
        contexts=[hit.text for hit in citations],
        paths=[(hit.metadata or {}).get("file_path", "") for hit in citations],
    )


def main() -> None:
    preflight()
    run_benchmark(
        name="E2E benchmark (real CanaR + AgoRa pipeline)",
        dataset=DATASET,
        pipeline=ask_canar,
        metrics=[Faithfulness(), ResponseRelevancy()],
        judge_llm=judge_llm,
        judge_embeddings=judge_embeddings,
        results_dir=HERE / "results",
        # local judge is slow: give each job room, and run a few in parallel
        # (Ollama serves them sequentially but overlaps prompt processing).
        run_config=RunConfig(timeout=900, max_workers=3),
    )


if __name__ == "__main__":
    main()

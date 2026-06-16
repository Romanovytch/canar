"""
utilitR-grounded RAG benchmark.

Same architecture as ./eval_demo.py (Qdrant retrieve -> Ollama generate ->
Ragas judge), but the dataset is derived from utilitR itself: each question
targets the official "Tache concernee et recommandation" block of one fiche,
and the reference reproduces that recommendation.

Extra: a pure-retrieval metric (retrieval_hit) checking whether the expected
source fiche appears in the top-k retrieved chunks — no LLM judging involved.

Run:
    cd benchmark
    source .venv/bin/activate
    python replica/eval_utilitr.py
"""

import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from qdrant_client import QdrantClient
from ragas import EvaluationDataset, evaluate
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import Faithfulness, ResponseRelevancy
from ragas.run_config import RunConfig

HERE = Path(__file__).parent          # benchmark/replica/
load_dotenv(HERE.parent / ".env")     # reuse benchmark/.env

# ---------------------------------------------------------------------------
# Config (knobs)
# ---------------------------------------------------------------------------

QDRANT_URL = os.environ["QDRANT_URL"]
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY") or None
COLLECTION = os.environ.get("QDRANT_COLLECTION", "utilitr")

EMBED_API_BASE = os.environ["EMBED_API_BASE"].rstrip("/")
EMBED_API_KEY = os.environ.get("EMBED_API_KEY", "EMPTY")
EMBED_MODEL = os.environ["EMBED_MODEL"]

LLM_API_BASE = os.environ["LLM_API_BASE"]
LLM_API_KEY = os.environ["LLM_API_KEY"]
LLM_MODEL = os.environ["LLM_MODEL"]

TOP_K = 3            # chunks retrieved per question
N_QUESTIONS = None   # None = all; set an int to run a subset

# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------

llm = ChatOpenAI(
    model=LLM_MODEL,
    openai_api_base=LLM_API_BASE,
    openai_api_key=LLM_API_KEY,
    temperature=0.0,
    extra_body={"keep_alive": "10m"},
)

embeddings = OpenAIEmbeddings(
    model=EMBED_MODEL,
    openai_api_base=EMBED_API_BASE,
    openai_api_key=EMBED_API_KEY,
    check_embedding_ctx_length=False,
)

qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, check_compatibility=False)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def embed_query(text: str) -> list[float]:
    prefix = "search_query: " if "nomic" in EMBED_MODEL else ""
    resp = requests.post(
        f"{EMBED_API_BASE}/embeddings",
        headers={"Authorization": f"Bearer {EMBED_API_KEY}"},
        json={"model": EMBED_MODEL, "input": [prefix + text]},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["data"][0]["embedding"]


def retrieve(query: str, top_k: int = TOP_K) -> tuple[list[str], list[str]]:
    """Returns (chunk texts, source paths) of the top-k hits."""
    hits = qdrant.query_points(
        collection_name=COLLECTION,
        query=embed_query(query),
        limit=top_k,
    ).points
    return [h.payload["text"] for h in hits], [h.payload["path"] for h in hits]


def generate(query: str, contexts: list[str]) -> str:
    context_block = "\n\n---\n\n".join(contexts)
    prompt = (
        f"Use the following context to answer the question.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {query}\n\n"
        f"Answer concisely based only on the context above."
    )
    resp = requests.post(
        f"{LLM_API_BASE.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {LLM_API_KEY}"},
        json={
            "model": LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "stream": False,
            "keep_alive": "10m",
        },
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def main():
    df = pd.read_csv(HERE.parent / "datasets" / "utilitr_questions.csv")
    if N_QUESTIONS:
        df = df.head(N_QUESTIONS)
    print(f"Benchmark: {len(df)} utilitR-grounded questions | "
          f"collection '{COLLECTION}' | top-{TOP_K}\n")

    samples, hits, paths_per_q = [], [], []
    for _, row in df.iterrows():
        query, reference, expected = row["query"], row["grading_notes"], row["source_fiche"]
        print(f"Q: {query}")
        contexts, paths = retrieve(query)
        hit = expected in paths
        hits.append(hit)
        paths_per_q.append("; ".join(paths))
        print(f"   retrieval_hit={'YES' if hit else 'NO'} (expected {Path(expected).name})")
        try:
            response = generate(query, contexts)
        except Exception as e:
            print(f"   [generation failed: {e}]")
            response = ""
        print(f"A: {response[:110]}...\n")
        samples.append(SingleTurnSample(
            user_input=query,
            retrieved_contexts=contexts,
            response=response,
            reference=reference,
        ))

    hit_rate = sum(hits) / len(hits)
    print(f"Retrieval hit rate (expected fiche in top-{TOP_K}): {hit_rate:.0%}\n")

    print("Running Ragas evaluation ...\n")
    results = evaluate(
        dataset=EvaluationDataset(samples=samples),
        metrics=[Faithfulness(), ResponseRelevancy()],
        llm=llm,
        embeddings=embeddings,
        run_config=RunConfig(timeout=600, max_workers=1),
    )

    out_df = results.to_pandas()
    out_df["retrieval_hit"] = hits
    out_df["retrieved_paths"] = paths_per_q
    out_df["source_fiche"] = df["source_fiche"].values

    score_cols = [c for c in out_df.columns
                  if c not in ("user_input", "retrieved_contexts", "response",
                               "reference", "retrieved_paths", "source_fiche")]
    print(out_df[["user_input"] + score_cols].to_string(index=False))
    print(f"\nMean scores:\n{out_df[score_cols].mean().to_string()}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    out = HERE / "results" / f"results_{stamp}.csv"
    out.parent.mkdir(exist_ok=True)
    out_df.to_csv(out, index=False)
    print(f"\nFull results saved to {out}")


if __name__ == "__main__":
    main()

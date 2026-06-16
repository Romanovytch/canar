"""
RAG evaluation using real Qdrant retrieval + LLM generation.

Run:
    cd benchmark
    source .venv/bin/activate
    python replica/eval_demo.py
"""

import os
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

load_dotenv()

HERE = Path(__file__).parent

# ---------------------------------------------------------------------------
# Config
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

TOP_K = 3          # chunks to retrieve per question
N_QUESTIONS = 5    # how many questions to evaluate (max 5; lower = faster)

# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------

llm = ChatOpenAI(
    model=LLM_MODEL,
    openai_api_base=LLM_API_BASE,
    openai_api_key=LLM_API_KEY,
    temperature=0.0,
    extra_body={"keep_alive": "10m"},  # keep model warm during Ragas's internal LLM calls
)

embeddings = OpenAIEmbeddings(
    model=EMBED_MODEL,
    openai_api_base=EMBED_API_BASE,
    openai_api_key=EMBED_API_KEY,
    # send raw strings, not token-ID arrays (Ollama rejects tokens)
    check_embedding_ctx_length=False,
)

qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def embed_query(text: str) -> list[float]:
    resp = requests.post(
        f"{EMBED_API_BASE}/embeddings",
        headers={"Authorization": f"Bearer {EMBED_API_KEY}"},
        json={"model": EMBED_MODEL, "input": [text]},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["data"][0]["embedding"]


def retrieve(query: str, top_k: int = TOP_K) -> list[str]:
    # nomic models need the "search_query: " prefix (pairs with ingest); others don't
    prefix = "search_query: " if "nomic" in EMBED_MODEL else ""
    vector = embed_query(prefix + query)
    hits = qdrant.query_points(
        collection_name=COLLECTION,
        query=vector,
        limit=top_k,
    ).points
    return [h.payload["text"] for h in hits]


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
            "keep_alive": "10m",  # keep model loaded so it doesn't reload (20GB) between calls
        },
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    dataset_path = HERE.parent / "datasets" / "demo.csv"
    df = pd.read_csv(dataset_path)
    df = df.head(N_QUESTIONS)
    print(f"Loaded {len(df)} questions from {dataset_path.name}\n")

    samples = []
    for _, row in df.iterrows():
        query = row["query"]
        reference = row["grading_notes"]

        print(f"Q: {query}")
        contexts = retrieve(query)
        try:
            response = generate(query, contexts)
        except Exception as e:
            print(f"  [generation failed: {e}]")
            response = ""
        print(f"A: {response[:120]}...\n")

        samples.append(SingleTurnSample(
            user_input=query,
            retrieved_contexts=contexts,
            response=response,
            reference=reference,
        ))

    print("Running Ragas evaluation ...\n")
    results = evaluate(
        dataset=EvaluationDataset(samples=samples),
        metrics=[Faithfulness(), ResponseRelevancy()],
        llm=llm,
        embeddings=embeddings,
        run_config=RunConfig(timeout=600, max_workers=1),
    )

    result_df = results.to_pandas()
    score_cols = [c for c in result_df.columns
                  if c not in ("user_input", "retrieved_contexts", "response", "reference")]

    print(result_df[["user_input"] + score_cols].to_string(index=False))
    print(f"\nMean scores:\n{result_df[score_cols].mean().to_string()}")

    out = HERE / "results" / "results_demo.csv"
    out.parent.mkdir(exist_ok=True)
    result_df.to_csv(out, index=False)
    print(f"\nFull results saved to {out}")


if __name__ == "__main__":
    main()

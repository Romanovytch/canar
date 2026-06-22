# Reproducibility — what matches across machines, and what doesn't

Two teammates running the same benchmark will **not** get identical numbers for
every metric. Some parts are deterministic; some depend on local LLMs that vary
run to run. Read a result with that in mind.

## Reproducible: retrieval

The collection and the retrieval metrics rebuild essentially identically, given
the same recipe:

- **Same docs** — pin utilitR to the same commit. AgoRa stamps that commit in
  every chunk's payload (`git_commit`), so a collection is auditable: check it
  was built from the expected commit.
- **Same AgoRa version** (`dev`) and the same `sources.yaml` (`vector_index`) —
  determines chunking and the named vectors.
- **Same models** — dense `bge-m3`, sparse `Qdrant/bm25`.

With those fixed, chunks, payloads and sparse (BM25) vectors are identical, dense
vectors are identical up to negligible float noise, and the retrieval metrics
(`hit_rate`, `mrr`, `recall`, `precision`, `ndcg`) match.

Minor caveats: Qdrant's HNSW index is approximate (graph built with randomness),
and dense embeddings can differ in the last decimals across hardware — both can
flip a rare tie, neither matters in practice at this dataset size.

## Not reproducible: answer quality

`faithfulness` and `answer_relevancy` come from local LLMs and **vary between
runs, even on the same machine**:

- The answer generator (`qwen3.5:9b`) is not deterministic on local Ollama, even
  at `temperature=0` (float/GPU/batching).
- The RAGAS judge (`qwen2.5:7b`) is itself an LLM, and returns 1 generation
  instead of the requested 3 on Ollama, adding noise.

Treat these as **indicative**, not exact. Report them as trends, ideally
averaged over a few runs.

## Provenance

Every `metrics.csv` row carries `embed_model`, `collection`, `judge_model` and
`git_commit` (of the canar code). Two results are only comparable when these
match — so always check provenance before comparing numbers across machines.

## Summary

| Metric | Reproduces? |
|---|---|
| hit_rate, mrr, recall, precision, ndcg | Yes (same recipe) |
| retrieval_latency_s | Approximately (hardware-dependent) |
| faithfulness, answer_relevancy | No — local LLMs, indicative only |

## To tighten reproducibility further

- Pin the utilitR commit (done in `SETUP.md`) and agree on the embedding model.
- Use a **shared Qdrant** so everyone queries the same collection (removes
  ingestion/HNSW variation) — a team infra decision.
- For answer scores, use a hosted/deterministic judge, or average several runs.

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

## Reproducible: input tokens

`input_tokens` is the prompt (system prompt + retrieved chunks + the question).
Given the same retrieval, the prompt is the same string, so the count is the
same — it reproduces exactly, and it's a fair way to compare how much context
each strategy sends.

`output_tokens` (and therefore `total_tokens`) does **not** reproduce: it measures
the generated answer, which varies run to run (see below).

Note the tokenizer used is recorded as `token_tokenizer` in `run_context.txt`.
Counts from different tokenizers aren't comparable — check it matches before
comparing token numbers across machines.

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

Each run folder also has a `run_context.txt` with what is shared by every
strategy: the machine (CPU/RAM/GPU), the tokenizer used for token counts, and —
when run with `MEASURE_RESOURCES=1` — the GPU memory in use and which processes
hold it. That's the fastest way to tell whether two runs are comparable.

## Summary

| Metric | Reproduces? |
|---|---|
| hit_rate, mrr, recall, precision, ndcg | Yes (same recipe) |
| input_tokens | Yes (same retrieval → same prompt → same count) |
| retrieval_latency_s, generation_latency_s | Approximately (hardware-dependent) |
| retrieval_cpu_s, peak_rss_mb | Approximately (hardware/machine-dependent) |
| output_tokens, total_tokens | No — depends on the generated answer |
| faithfulness, answer_relevancy | No — local LLMs, indicative only |

## To tighten reproducibility further

- Pin the utilitR commit (done in `SETUP.md`) and agree on the embedding model.
- Use a **shared Qdrant** so everyone queries the same collection (removes
  ingestion/HNSW variation) — a team infra decision.
- For answer scores, use a hosted/deterministic judge, or average several runs.

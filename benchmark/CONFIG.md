# Benchmark configuration (`config.yaml`)

The benchmark reads its settings from `config.yaml` so you can adjust a run —
and **compare retrieval strategies** — without touching code.

It has two sections:

```yaml
run:                         # how the benchmark runs
  dataset: datasets/utilitr_questions.csv
  agent: r_helpdesk
  limit: null                # null = all questions; int = quick subset
  judge_model: qwen2.5:7b    # RAGAS judge; null = use the product LLM
  gen_max_tokens: 8192       # generation budget (see note below)

environment:                 # expected env; preflight aborts if canar/.env differs
  embed_model: bge-m3
  collection: utilitr_v1

profiles:                    # retrieval strategies to compare
  - name: product_default
    strategy: simple_vector
    top_k: 5
    score_threshold: 0.35
    source_filter: utilitr
    fallback_top_k: 3
  - name: wider_topk
    strategy: simple_vector
    top_k: 10
    score_threshold: 0.20
```

## How it works

- The dataset is run **once per profile**. Every result row is tagged with the
  `profile` name and provenance (embed model, collection, judge, git commit).
- Each profile writes its own folder under `e2e/results/`, containing
  `metrics.csv` (the scores) and `answers.md` (question, answer, reference,
  retrieved context).
- When there's more than one profile, a **comparison table** of mean scores is
  printed at the end.
- Each profile's fields mirror CanaR's `RetrievalProfile`
  (`canar/app/retrieval/`). The benchmark builds the product's real strategy
  with those values — so `product_default` measures the product as it ships,
  and the others are experiments to compare against it.

## The `environment` block

`environment` declares the embedding model and collection a run is meant to use.
The preflight reads `canar/.env` and **aborts on a mismatch**, so a result is
never produced silently with the wrong model — and runs stay comparable across
machines, even though each developer has their own `.env` and Qdrant.

## Profiles vs. the product

This is the benchmark's own config — it is **not** the product loading profiles
from YAML (that's a separate CanaR ticket). Here we sweep the same knobs to
answer the project's central question: *which retrieval strategy works best for
which kind of question?* If the product later adopts a YAML profile format, the
schema is intentionally the same, so a profile can be shared.

## Resource benchmark (optional)

Set `run.measure_resources: true` (or env `MEASURE_RESOURCES=1`) to also measure
the **resource cost** of each retrieval strategy, not just answer quality. It's
**off by default** and adds a tiny sampler thread per phase only when on.

Extra columns appear in `metrics.csv` and in the per-strategy `comparison.csv`:

| Column | Meaning |
|---|---|
| `retrieval_cpu_s` / `generation_cpu_s` | CPU seconds (user+sys) of that phase |
| `retrieval_peak_rss_mb` / `generation_peak_rss_mb` | peak process memory in that phase |
| `gpu_util_pct` / `gpu_mem_mb` | GPU utilization / memory — **device-level** |

When on, the machine (CPU / cores / RAM / GPU) is also stamped into every row so
runs stay comparable across computers.

Needs `psutil` (in the `benchmark` extra). GPU columns need `nvidia-ml-py`
(`pip install -e ".[benchmark-gpu]"`); without it they stay empty. **Caveat:**
generation/embedding run in a separate server process, so GPU numbers are the
whole device's usage during the phase, not this process's — the clean
per-strategy signals are time, CPU and memory.

## Notes

- `judge_model`: the local reasoning model (`qwen3.5`) is slow and times out as
  a judge; a non-reasoning one (`qwen2.5:7b`) scores in seconds.
- `gen_max_tokens`: `qwen3.5` spends a hidden budget "thinking" before it
  answers, so the app's default of 2048 returns empty answers on many
  questions. Keep this high.
- Env vars override the matching `run` keys: `JUDGE_MODEL`, `GEN_MAX_TOKENS`.
- `simple_vector` (dense) and `simple_sparse` (BM25) are wired in. A commented
  `sparse_bm25` profile is ready in `config.yaml` — enable it once the collection
  is ingested with sparse vectors (see `SPARSE_TODO.md`). Add further strategies
  to the `STRATEGIES` map in `e2e/eval_e2e.py` as the product gains them.

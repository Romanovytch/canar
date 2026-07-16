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

**Per-strategy columns** — the signals that actually differ between retrieval
strategies. These are the ones in `comparison.csv`:

| Column | What it measures | What it does NOT measure |
|---|---|---|
| `retrieval_latency_s` / `generation_latency_s` | wall-clock time of each phase | — |
| `retrieval_cpu_s` | CPU seconds of the **benchmark process** during retrieval (real method cost, e.g. BM25) | the LLM server's CPU |
| `peak_rss_mb` | peak memory of the **benchmark process** in the turn | the LLM server's memory |
| `input_tokens` / `output_tokens` / `total_tokens` | tokens in the prompt / answer / both — differ per strategy because the retrieved context size differs | — |

Per profile, `comparison.csv` also carries the avg / min / max / total for each
token field (e.g. `input_tokens_avg`, `input_tokens_max`, `total_tokens_total`).
Tokens are counted with the configured model's tokenizer when available, else
`tiktoken`, else a char heuristic; which one was used is recorded as
`token_tokenizer` in `run_context.txt`. To get exact counts, install
`transformers` and set `BENCH_TOKENIZER=<hf-model-name>`.

**Run-level context** — written once per run to `run_context.txt` (not compared
per strategy). GPU work all happens in the LLM server, which is the same model
for every strategy, so GPU usage describes the setup, not the method:

| Field | Meaning |
|---|---|
| `gpu` / `cpu_cores` / `ram_gb` | the machine (also stamped on every row) |
| `gpu_mem_used_mb` | device GPU memory in use once the model is loaded |
| `gpu_procs` | who holds it: `name(pid)=MB` (e.g. the ollama runner) |
| `token_tokenizer` | which tokenizer produced the token counts (`tiktoken` / `transformers:<model>` / `heuristic`) |

Why GPU is not per-strategy: retrieval never touches the GPU in this setup
(embeddings go out over HTTP); the GPU is busy only during generation, and that
is the same LLM for dense, sparse and hybrid. So a per-strategy GPU number would
only reflect the shared model (and the one-time model load), not the method —
misleading. It is reported once as context instead.

Deliberately not reported per turn: `generation_cpu_s` (mostly HTTP-wait), and
absolute device GPU memory (only ever grows on a shared server — the source of
the original #53 confusion).

Needs `psutil` (in the `benchmark` extra, which also carries `nvidia-ml-py`;
without a GPU the run-context GPU fields are simply absent).

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

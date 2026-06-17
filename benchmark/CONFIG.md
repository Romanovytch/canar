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
  `profile` name, and each profile writes its own CSV in `e2e/results/`.
- When there's more than one profile, a **comparison table** of mean scores is
  printed at the end.
- Each profile's fields mirror CanaR's `RetrievalProfile`
  (`canar/app/retrieval/`). The benchmark builds the product's real strategy
  with those values — so `product_default` measures the product as it ships,
  and the others are experiments to compare against it.

## Profiles vs. the product

This is the benchmark's own config — it is **not** the product loading profiles
from YAML (that's a separate CanaR ticket). Here we sweep the same knobs to
answer the project's central question: *which retrieval strategy works best for
which kind of question?* If the product later adopts a YAML profile format, the
schema is intentionally the same, so a profile can be shared.

## Notes

- `judge_model`: the local reasoning model (`qwen3.5`) is slow and times out as
  a judge; a non-reasoning one (`qwen2.5:7b`) scores in seconds.
- `gen_max_tokens`: `qwen3.5` spends a hidden budget "thinking" before it
  answers, so the app's default of 2048 returns empty answers on many
  questions. Keep this high.
- Env vars override the matching `run` keys: `JUDGE_MODEL`, `GEN_MAX_TOKENS`.
- Only `simple_vector` exists today; add new strategies to the `STRATEGIES` map
  in `e2e/eval_e2e.py` as the product gains them (hybrid, rerank, ...).

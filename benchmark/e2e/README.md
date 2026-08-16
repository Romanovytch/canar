# End-to-end evaluation — CanaR + AgoRa + RAGAS

Evaluates the **real product**: AgoRa builds the collection, CanaR's own code
retrieves and generates, RAGAS judges. No replicated pipeline code.

```
AgoRa ingest ──> Qdrant `utilitr_v1` ──> CanaR (RetrievalService.search
                                          → r_helpdesk prompt → ChatClient)
                                                      │
                                                      ▼
                                       RAGAS + retrieval metrics
```

The script imports `canar.app.retrieval.service.RetrievalService` (Julien's
modular retrieval), `canar.app.api.{embed_client,llm_client}`,
`canar.app.agents.r_helpdesk` and `canar.app.config.AppConfig` — the same
call chain as the r_helpdesk branch of `canar/app/main.py`
(`retrieval.search(agent, q)` → `build_messages` → `stream_chat`). Config
comes from `canar/.env`, the same file the app reads. `top_k`, the source
filter and score-threshold pruning come from the retrieval profile
(`canar/app/retrieval/profiles.py`).

**Dataset:** reuses `../datasets/utilitr_questions.csv` —
12 French questions whose references reproduce the official
"Tâche concernée et recommandation" blocks of utilitR fiches.

**Metrics** — two independent layers:

*Retrieval (deterministic, instant — no LLM; in `../harness/retrieval_metrics.py`).*
Score the ranked list of retrieved sources against the question's expected
source. These are the roadmap's required retrieval metrics:
- `hit_rate` — Hit Rate@k: is the expected source in the top-k?
- `mrr` — Mean Reciprocal Rank: 1 / rank of the first relevant source.
- `recall` — Recall@k: fraction of relevant sources retrieved.
- `precision` — Precision@k, `ndcg` — nDCG@k (ranking quality).
- `retrieval_latency_s` — Latency: embed + Qdrant round-trip per question.
  (First question is slower — the embedding model loads on the first call.)

*Answer quality (RAGAS, judged by the local LLM — slow, indicative).*
- `faithfulness` — is the answer grounded in the retrieved context?
- `answer_relevancy` — does the answer address the question?

With the current dataset (one expected fiche per question) `recall` collapses
to `hit_rate` and `precision` is bounded by `1/k`; they get richer with the
end-June dataset (multiple relevant sources per question). `source_fiche` may
list several sources separated by `;`.

## Run

```bash
# 1. once: build the collection with AgoRa (the eval refuses to run without it)
cd /home/cereq/opt/pedro/agora && source .venv/bin/activate
agora-ingest --sources-config-path sources.yaml --source utilitr \
             --collection utilitr_v1 --dotenv-path .env --drop-collection

# 2. the benchmark — the judge comes from config.yaml; JUDGE_MODEL overrides it
cd /home/cereq/opt/pedro/canar/benchmark && source .venv/bin/activate
python e2e/eval_e2e.py
```

Select a benchmark YAML and retrieval-metric cutoff from the command line. Relative config paths are resolved from the current working directory:

```bash
QDRANT_COLLECTIONS=dicovar_sum MEASURE_RESOURCES=1 \
python e2e/eval_e2e.py --config config.dico.yaml --retrieval-k 5
```

Shell environment variables take precedence over values in `canar/.env`. The `--retrieval-k` option changes only metric computation, not product retrieval or reranker result counts.

Each run lands in a timestamped folder under `e2e/results/` (gitignored):
`metrics.csv` (the scores) and `answers.md` (question, answer, reference, context).

**Knobs** (env vars / top of `eval_e2e.py`):
- `--config PATH` (default `config.yaml`) — benchmark run/dataset/profile YAML.
- `--retrieval-k N` — positive cutoff for retrieval metrics; default: all returned results.
- `GEN_MAX_TOKENS` (default 8192) — generation cap. The product model
  (`qwen3.5`) is a *reasoning* model: it burns a hidden token budget thinking
  before it answers, so the app's default 2048 returns **empty** answers
  (`finish_reason=length`) on many questions. Keep this high. *(This is also a
  latent product bug: `main.py`'s 2048 default hits the same wall.)*
- `JUDGE_MODEL` (default: `config.yaml`, then the product LLM) — the RAGAS
  judge, and it must be a **non-reasoning** model: the reasoning 9B times out
  (≈hours, mostly `NaN`). `gemma3:12b` is the current choice. `qwen2.5:7b` was
  used before and is fast, but on some answers it returns the NLI schema
  (`statement`/`reason`/`verdict`) where the statement step expects plain
  strings — valid JSON, wrong shape, so RAGAS's repair pass cannot fix it and
  the question loses its faithfulness score. It is deterministic, so the same
  answers fail every run.
- `limit` in the `DatasetSpec` — subset for a quick pass.

Retrieval knobs (`top_k`, source filter, pruning) live in the retrieval
profile, so the benchmark always matches the app's real settings.

## Comparing the three benchmarks

| | pipeline code | collection | what a bad score means |
|---|---|---|---|
| `replica/eval_demo.py` | standalone replica | `utilitr` | methodology smoke test |
| `replica/eval_utilitr.py` | standalone replica | `utilitr` | corpus/embedding issues |
| `e2e/` (this) | **CanaR's real code** | **AgoRa's `utilitr_v1`** | a product problem worth a ticket |

Differences between `replica/eval_utilitr.py` and `e2e` scores localize the
cause: same dataset, same corpus — what changes is chunking (AgoRa's vs ours)
and retrieval logic (CanaR's fusion/filter/pruning vs plain top-k).

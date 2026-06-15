# End-to-end evaluation — CanaR + AgoRa + RAGAS

Evaluates the **real product**: AgoRa builds the collection, CanaR's own code
retrieves and generates, RAGAS judges. No replicated pipeline code.

```
AgoRa ingest ──> Qdrant `utilitr_v1` ──> CanaR (RetrievalService.search
                                          → r_helpdesk prompt → ChatClient)
                                                      │
                                                      ▼
                                            RAGAS + retrieval_hit
```

The script imports `canar.app.retrieval.service.RetrievalService` (Julien's
modular retrieval), `canar.app.api.{embed_client,llm_client}`,
`canar.app.agents.r_helpdesk` and `canar.app.config.AppConfig` — the same
call chain as the r_helpdesk branch of `canar/app/main.py`
(`retrieval.search(agent, q)` → `build_messages` → `stream_chat`). Config
comes from `canar/.env`, the same file the app reads. `top_k`, the source
filter and score-threshold pruning come from the retrieval profile
(`canar/app/retrieval/profiles.py`).

**Dataset:** reuses `../utilitr_bench/datasets/utilitr_questions.csv` —
12 French questions whose references reproduce the official
"Tâche concernée et recommandation" blocks of utilitR fiches.

**Metrics:**
- `retrieval_hit` — did CanaR retrieve the fiche the question targets? (deterministic, instant)
- `faithfulness` / `answer_relevancy` — RAGAS, judged by the local LLM (slow, indicative)

## Run

```bash
# 1. once: build the collection with AgoRa (the eval refuses to run without it)
cd /home/cereq/opt/pedro/agora && source .venv/bin/activate
agora-ingest --sources-config-path sources.yaml --source utilitr \
             --collection utilitr_v1 --dotenv-path .env --drop-collection

# 2. the benchmark — use a fast, non-reasoning judge to avoid timeouts
cd /home/cereq/opt/pedro/canar/ragas_tests && source .venv/bin/activate
JUDGE_MODEL=qwen2.5:7b python e2e/eval_e2e.py
```

Timestamped results land in `e2e/results/` (gitignored).

**Knobs** (env vars / top of `eval_e2e.py`):
- `GEN_MAX_TOKENS` (default 8192) — generation cap. The product model
  (`qwen3.5`) is a *reasoning* model: it burns a hidden token budget thinking
  before it answers, so the app's default 2048 returns **empty** answers
  (`finish_reason=length`) on many questions. Keep this high. *(This is also a
  latent product bug: `main.py`'s 2048 default hits the same wall.)*
- `JUDGE_MODEL` (default = product LLM) — the RAGAS judge. Point it at a fast
  **non-reasoning** model (`qwen2.5:7b`): the reasoning 9B judge times out
  (≈hours, mostly `NaN`); `qwen2.5:7b` scores all 12 in minutes.
- `limit` in the `DatasetSpec` — subset for a quick pass.

Retrieval knobs (`top_k`, source filter, pruning) live in the retrieval
profile, so the benchmark always matches the app's real settings.

## Comparing the three benchmarks

| | pipeline code | collection | what a bad score means |
|---|---|---|---|
| `eval_rag.py` | standalone replica | `utilitr` | methodology smoke test |
| `utilitr_bench/` | standalone replica | `utilitr` | corpus/embedding issues |
| `e2e/` (this) | **CanaR's real code** | **AgoRa's `utilitr_v1`** | a product problem worth a ticket |

Differences between `utilitr_bench` and `e2e` scores localize the cause:
same dataset, same corpus — what changes is chunking (AgoRa's vs ours) and
retrieval logic (CanaR's fusion/filter/pruning vs plain top-k).

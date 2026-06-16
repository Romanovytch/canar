# Benchmark results log

One entry per meaningful run. Raw per-question CSVs live in `experiments/`
(gitignored); this file keeps the configs and scores that matter.

---

## 2026-06-15 — End-to-end benchmark (real CanaR + AgoRa pipeline)

First run through the **actual product code**: AgoRa-built collection +
CanaR's modular `RetrievalService` + `r_helpdesk` prompt + `ChatClient`
(not a replica). See `e2e/`.

**Config:** 12 utilitR-grounded questions · collection `utilitr_v1`
(AgoRa ingest, 337 chunks, bge-m3, 1024-dim) · generator `qwen3.5:9b`
(the product LLM) · judge `qwen3.5:9b` · TOP_K 5 · metrics:
retrieval_hit, Faithfulness, Answer Relevancy.

| Metric | Score | Coverage |
|---|:-:|---|
| Retrieval hit rate | **100%** (12/12) | deterministic |
| Answer relevancy | **0.77** mean | 9/12 (3 judge NaN) |
| Faithfulness | 0.89 | **1/12** (11 judge timeouts) |

**Notes**
- Retrieval through CanaR's real fusion/filter/pruning still hits the target
  fiche every time — the product retrieves as well as the standalone replica.
- **Generation fix:** `qwen3.5` is a reasoning model; with the app's default
  `max_tokens=2048` the thinking budget runs out and answers come back EMPTY
  (`finish_reason=length`). Raising `GEN_MAX_TOKENS` to 8192 fixed 11/12 empty
  answers and lifted relevancy from ~0.10 to 0.77. (Latent product bug: the
  same 2048 default in `main.py` hits this wall.)
- The `qwen3.5:9b` reasoning **judge** is the remaining weak link: 11/12
  faithfulness timeouts and 3 relevancy NaN. A 2-question check with a
  non-reasoning judge (`JUDGE_MODEL=qwen2.5:7b`) scored every job in seconds
  (faithfulness 0.5–0.82, relevancy 0.91–0.95) — re-run the full 12 that way
  for clean faithfulness coverage.
- Faithfulness <1.0 is partly legitimate: CoachR adds SAS↔R analogies and
  general R advice not present in the retrieved chunks, which grounding
  penalizes.
- Raw CSV: `e2e/results/results_20260615_1622.csv` (local).

---

## 2026-06-11 — utilitR-grounded benchmark (utilitr_bench)

**Config:** 12 questions derived from utilitR fiches (`utilitr_bench/datasets/utilitr_questions.csv`) ·
collection `utilitr` (1,209 chunks) · embeddings `bge-m3` · generator/judge `qwen3.5:9b` ·
TOP_K 3 · metrics: retrieval_hit, Faithfulness, Answer Relevancy

| Metric | Score | Coverage |
|---|:-:|---|
| Retrieval hit rate | **100%** (12/12) | deterministic |
| Answer relevancy | **0.83** mean (~0.90 excluding one empty answer) | 12/12 |
| Faithfulness | **1.0** on every valid score | 5/12 (7 judge timeouts) |

**Notes**
- Every question retrieved its target fiche in the top-3 — confirms the bge-m3 fix.
- Relevancy up from 0.66 (demo set) to ~0.83–0.90 with corpus-grounded questions.
- The local 9B judge remains the weak link: 7 faithfulness timeouts, 1 failed generation.
- Raw CSV: `utilitr_bench/results/results_20260611_1105.csv` (local).

---

## 2026-06-11 — Baseline (bge-m3)

**Config:** collection `utilitr` (1,209 chunks, ~800 chars) · embeddings `bge-m3` ·
generator/judge `qwen3.5:9b` (Ollama, temp 0) · TOP_K 3 · 5 questions ·
metrics: Faithfulness, Answer Relevancy (Ragas 0.4.3)

| Question | Faithfulness | Answer Relevancy |
|---|:-:|:-:|
| Comment lire un fichier CSV en R ? | 1.0 | 0.92 |
| Différence data.frame / tibble ? | NaN* | 0.68 |
| Jointure entre deux tables ? | NaN* | 0.42 |
| Créer un graphique avec ggplot2 ? | 1.0 | 0.62 |
| Filtrer des lignes d'un data.frame ? | 1.0 | 0.65 |
| **Mean** | **1.00** (3/5 valid) | **0.66** |

\* judge-side timeouts (600s), not answer failures — the answers were generated fine.

**Notes**
- All 3 valid faithfulness scores are 1.0: answers fully grounded in retrieved chunks.
- Relevancy is depressed by the weak local judge (returns 1 reverse-question
  instead of 3); treat as indicative.
- Generated answers visibly match utilitR doctrine (e.g. recommends
  `read_delim()`/`readr`, not `read.csv()`).

---

## 2026-06-10 — Pre-fix run (nomic-embed-text, for reference)

**Config:** same, but embeddings `nomic-embed-text:v1.5` without task prefixes · 2 questions

| Question | Faithfulness | Answer Relevancy |
|---|:-:|:-:|
| Comment lire un fichier CSV en R ? | 0.33 | 0.0 |
| Différence data.frame / tibble ? | 1.0 | NaN |

Retrieval failure: for the CSV question the correct fiche
(`Fiche_import_fichiers_plats.qmd`) ranked ~40th; the answer said the context
contained no information. Fixed by switching to `bge-m3` (correct doc now ranks 1st).

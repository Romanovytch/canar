# Benchmark results log

One entry per meaningful run. Raw per-question CSVs live in `experiments/`
(gitignored); this file keeps the configs and scores that matter.

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

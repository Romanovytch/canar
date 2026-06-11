# Ragas tests

RAG quality evaluation using [Ragas](https://docs.ragas.io/) over the utilitR documentation.

## What is tested

A **standalone** RAG pipeline (it does *not* call canar or agora code — it replicates the same flow against the same infrastructure):

| Piece | This eval | The real stack |
|---|---|---|
| Ingest | `ingest.py` | agora (`agora-ingest`) |
| Vector DB | Qdrant — collection `utilitr` | Qdrant — collection `utilitr_v1` |
| Retrieval | `retrieve()` in `eval_rag.py` | canar `retrieval.py` |
| Generation | direct Ollama call | canar `llm_client.py` |

**Database under test:** Qdrant collection `utilitr` — ~1,200 chunks from the
[utilitR](https://book.utilitr.org/) French R documentation (75 `.md`/`.qmd`/`.Rmd`
files from `/home/cereq/opt/utilitR`, chunked at ~800 chars, embedded with `bge-m3`).

**Test set:** 5 French questions in `datasets/test_dataset.csv` (CSV import,
data.frame vs tibble, joins, ggplot2, row filtering), each with grading notes
used as reference.

## Scripts

| Script | Purpose |
|---|---|
| `ingest.py` | Read utilitR markdown → chunk → embed (Ollama) → upsert into Qdrant |
| `eval_rag.py` | For each question: retrieve top-3 chunks → generate answer (`qwen3.5:9b`) → score with Ragas |
| `eval.py` | Older demo with 3 hardcoded samples (no retrieval) |

## Metrics

| Metric | What it checks |
|---|---|
| `faithfulness` | Is the answer grounded in the retrieved context? Catches hallucinations. |
| `answer_relevancy` | Does the answer actually address the question? Catches evasive answers. |

Each metric internally calls the LLM as a judge (structured prompts parsed into 0–1 scores).

## Run

```bash
source .venv/bin/activate
python ingest.py     # only needed once, or after changing chunking/embeddings
python eval_rag.py   # generation is fast; the Ragas phase takes ~15-30 min locally
```

Results are saved to `experiments/results_rag.csv`. Knobs at the top of
`eval_rag.py`: `N_QUESTIONS`, `TOP_K`, and the metrics list in `main()`.

## Lessons learned / gotchas

- **Use a multilingual embedding model.** `nomic-embed-text` (English-focused)
  ranked the correct doc for "Comment lire un fichier CSV ?" at position 40;
  `bge-m3` ranks it 1st. The corpus and questions are French.
- If using a nomic model anyway, it **requires task prefixes**
  (`search_document:` at ingest, `search_query:` at query) — handled
  automatically in the scripts.
- `localhost` must be written as `127.0.0.1` in `.env` — Python resolves
  `localhost` to IPv6 but Ollama listens on IPv4 only.
- `keep_alive` is set on LLM calls so Ollama doesn't unload the 20 GB model
  between Ragas jobs (reloads caused timeouts).
- Ragas 0.4.3 + langchain-community 0.4 have a broken VertexAI import — the
  venv contains a small stub patch (`langchain_community/chat_models/vertexai.py`).
- Embeddings via LangChain need `check_embedding_ctx_length=False` or the
  OpenAI client sends token arrays, which Ollama rejects (`invalid input type`).

## Next steps

- Wire the eval to canar's real `search_qdrant()` / LLM client and the
  agora-built `utilitr_v1` collection, so scores measure the actual product.
- Add retrieval-level metrics (Context Precision/Recall) to separate
  agora-side issues from canar-side issues.
- Grow the test set beyond 5 questions; use a stronger judge LLM.

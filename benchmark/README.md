# Benchmark suite — CanaR + AgoRa retrieval & answer quality

RAG evaluation over the [utilitR](https://book.utilitr.org/) French R
documentation, using [RAGAS](https://docs.ragas.io/) for answer quality and
deterministic metrics for retrieval. The goal is to measure the **real
product** (CanaR + AgoRa) and to compare retrieval strategies as they land.

## Layout

```
benchmark/
├── harness/            shared, product-agnostic engine
│   ├── ragas_bench.py        run_benchmark(): loop → RAGAS → save metrics.csv + answers.md
│   ├── retrieval_metrics.py  hit_rate, MRR, recall, precision, nDCG
│   └── bench_config.py       loads config.yaml
├── datasets/           question sets + ground truth
│   └── utilitr_questions.csv  12 utilitR-grounded questions
├── config.yaml         run settings + the strategies to compare
├── e2e/                the product benchmark (CanaR's real code + AgoRa)
│   ├── eval_e2e.py
│   └── README.md
├── SETUP.md            enabling sparse + hybrid (named-vector ingestion)
├── REPRODUCIBILITY.md  what reproduces exactly vs what is indicative
└── requirements.txt   .env(.example)
```

## What it measures

`e2e/eval_e2e.py` imports `canar.app.*` and calls the exact chain the Streamlit
app runs, so a bad score is a real product problem. It drives the product's
retrieval strategies — `simple_vector` (dense), `simple_sparse` (BM25) and
`hybrid` (the two fused) — over the dataset and compares them side by side. See
[e2e/README.md](e2e/README.md).

## Metrics — two independent layers

- **Retrieval** (deterministic, no LLM — `harness/retrieval_metrics.py`):
  Hit Rate@k, MRR, Recall@k, Precision@k, nDCG@k, and latency. These are the
  roadmap's required retrieval metrics.
- **Answer quality** (RAGAS, judged by a local LLM): `faithfulness` (is the
  answer grounded in the retrieved context?) and `answer_relevancy` (does it
  address the question?).

## Run

```bash
source .venv/bin/activate
python e2e/eval_e2e.py
```

Needs the AgoRa-built collection (see `SETUP.md` for the dense + sparse
ingestion the sparse/hybrid strategies require).

Each `e2e` run saves a timestamped folder in `e2e/results/` (gitignored) with
`metrics.csv` (the scores) and `answers.md` (question, answer, reference,
retrieved context). Knobs live at the top of each script; key ones for `e2e`:

- `GEN_MAX_TOKENS` (default 8192) — `qwen3.5` is a *reasoning* model and burns
  a hidden token budget before answering; the app's default 2048 returns empty
  answers on many questions, so keep this high.
- `JUDGE_MODEL` (default: `config.yaml`, then the product LLM) — a non-reasoning
  model, so the RAGAS judge doesn't time out. Currently `gemma3:12b`; see
  `e2e/README.md` for why `qwen2.5:7b` was dropped.

## Gotchas (learned the hard way)

- **Use a multilingual embedding model.** `nomic-embed-text` (English-focused)
  ranked the correct doc for "Comment lire un fichier CSV ?" at position 40;
  `bge-m3` ranks it 1st. The corpus and questions are French.
- `localhost` must be `127.0.0.1` in `.env` — Python resolves `localhost` to
  IPv6 but Ollama listens on IPv4 only.
- `keep_alive` is set on LLM calls so Ollama doesn't unload the model between
  RAGAS jobs (reloads caused timeouts).
- RAGAS 0.4.3 + langchain-community 0.4 have a broken VertexAI import — the
  venv carries a small stub patch (`langchain_community/chat_models/vertexai.py`).
- Embeddings via LangChain need `check_embedding_ctx_length=False`, or the
  OpenAI client sends token arrays, which Ollama rejects (`invalid input type`).

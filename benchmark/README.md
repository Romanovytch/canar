# Benchmark suite — CanaR + AgoRa retrieval & answer quality

RAG evaluation over the [utilitR](https://book.utilitr.org/) French R
documentation, using [RAGAS](https://docs.ragas.io/) for answer quality and
deterministic metrics for retrieval. The goal is to measure the **real
product** (CanaR + AgoRa) and to compare retrieval strategies as they land.

## Layout

```
benchmark/
├── harness/            shared, product-agnostic engine
│   ├── ragas_bench.py        run_benchmark(): loop → RAGAS → save CSV
│   └── retrieval_metrics.py  hit_rate, MRR, recall, precision, nDCG
├── datasets/           question sets + ground truth
│   ├── utilitr_questions.csv  12 utilitR-grounded questions
│   └── demo.csv               5-question smoke set
├── e2e/            ⭐ the product benchmark (CanaR's real code + AgoRa)
│   ├── eval_e2e.py
│   └── README.md
├── replica/            standalone baselines (no product import)
│   ├── ingest.py              builds the `utilitr` collection
│   ├── eval_utilitr.py        utilitR-grounded, replica pipeline
│   └── eval_demo.py           5-question methodology smoke test
├── RESULTS.md          one entry per meaningful run
└── requirements.txt   .env(.example)
```

## The three levels

They form a progression. The difference between levels is **whose code runs**
and **which collection** is queried — so a bad score localizes the cause.

| Level | Script | Pipeline code | Collection | A bad score means |
|---|---|---|---|---|
| **e2e** ⭐ | `e2e/eval_e2e.py` | **CanaR's real modules** | AgoRa's `utilitr_v1` | a product problem worth a ticket |
| replica (utilitR) | `replica/eval_utilitr.py` | standalone replica | `utilitr` | corpus / embedding issue |
| replica (demo) | `replica/eval_demo.py` | standalone replica | `utilitr` | methodology smoke test |

`e2e` is the priority: it imports `canar.app.*` and calls the exact chain the
Streamlit app runs. The replicas don't import the product — they reproduce the
flow with their own code, as a reference baseline. See [e2e/README.md](e2e/README.md).

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

# 1. e2e (the product) — needs the AgoRa collection built first:
#    cd ../../agora && agora-ingest ... --collection utilitr_v1 --drop-collection
JUDGE_MODEL=qwen2.5:7b python e2e/eval_e2e.py

# 2. replicas (reference) — need the standalone `utilitr` collection:
python replica/ingest.py        # once, or after changing chunking/embeddings
python replica/eval_utilitr.py
python replica/eval_demo.py
```

Each run saves a timestamped CSV in the script's `results/` (gitignored).
Knobs live at the top of each script; key ones for `e2e`:

- `GEN_MAX_TOKENS` (default 8192) — `qwen3.5` is a *reasoning* model and burns
  a hidden token budget before answering; the app's default 2048 returns empty
  answers on many questions, so keep this high.
- `JUDGE_MODEL` (default = product LLM) — point at a fast non-reasoning model
  (`qwen2.5:7b`) so the RAGAS judge doesn't time out.

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

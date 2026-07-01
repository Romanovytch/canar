# Compare Reranking

Manual script for comparing retrieval order without launching the app.

It runs the same query against:

1. Hybrid retrieval
2. Hybrid retrieval plus BGE reranking
3. Hybrid retrieval plus Qwen reranking, only when `--include-qwen` is passed

The default Qdrant collection is `utilitr_bgem3_ds` and the default Qdrant URL
is `http://localhost:6360`.

```bash
source .venv_canar/bin/activate
python canar/app/retrieval/rerank/compare/compare.py "comment filtrer un dataframe en R ?"
```

Run Qwen only after BGE works:

```bash
python canar/app/retrieval/rerank/compare/compare.py "comment filtrer un dataframe en R ?" --include-qwen
```

Useful overrides:

```bash
python canar/app/retrieval/rerank/compare/compare.py "ma question" --top-n 10 --device cuda
python canar/app/retrieval/rerank/compare/compare.py "ma question" --collection utilitr_bgem3_ds
python canar/app/retrieval/rerank/compare/compare.py "ma question" --qdrant-url http://localhost:6360
```

## How to Read the Output

The script prints one ranked list per retrieval mode:

- `HYBRID`: dense + sparse hybrid retrieval only.
- `HYBRID + BGE RERANK`: the same hybrid candidates reordered by BGE.
- `HYBRID + QWEN RERANK`: the same hybrid candidates reordered by Qwen, only
  when `--include-qwen` is passed.

Inside each section, `[1]`, `[2]`, `[3]`, etc. are the final order for that
mode. `[1]` is the first chunk the downstream RAG pipeline would see.

Each hit prints:

- `score`: the original retrieval or fusion score from hybrid search.
- `score_norm`: the normalized hybrid score.
- `rerank_score`: the reranker score. It is `None` in the plain `HYBRID`
  section because no reranker was used there.
- `source`, `section`, and `text`: the content you should read to judge whether
  the hit is useful for the query.

Do not compare `score` and `rerank_score` as if they were the same scale. They
come from different systems:

- `score` / `score_norm` explain how hybrid retrieval ranked the chunk.
- `rerank_score` explains how the reranker ranked the chunk after retrieval.

After reranking, the order is determined by `rerank_score`. A chunk can have a
modest hybrid score but become `[1]` after reranking if the reranker thinks it
answers the query better.

To evaluate a run:

1. Read the query.
2. Read the top 3 to 5 hits in `HYBRID`.
3. Read the top 3 to 5 hits in `HYBRID + BGE RERANK`.
4. If Qwen was run, read the top 3 to 5 hits in `HYBRID + QWEN RERANK`.
5. Ask which list gives the LLM the best context to answer the query.

A good rerank result usually moves directly relevant chunks upward and pushes
lexical-but-off-topic chunks downward. A suspicious result puts irrelevant
chunks first or gives nearly identical `rerank_score` values to every hit.


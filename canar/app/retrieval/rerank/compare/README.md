# Compare Reranking

Manual script for comparing retrieval order without launching the app.

It runs the same query against hybrid retrieval and every model registered in
`RERANKER_SPECS`: BGE v2 M3 and Qwen 0.6B, 4B, and 8B.
The default Qdrant collection is `utilitr_bgem3_ds` and the default Qdrant URL
is `http://localhost:6360`.

```bash
source .venv_canar/bin/activate
python canar/app/retrieval/rerank/compare/compare.py "comment filtrer un dataframe en R ?"
```

All registered model weights are loaded lazily during this run. The first run can
download several large models and requires enough disk, RAM, and GPU memory.

Useful overrides:

```bash
python canar/app/retrieval/rerank/compare/compare.py "ma question" --top-k 10 --device cuda
python canar/app/retrieval/rerank/compare/compare.py "ma question" --collection utilitr_bgem3_ds
python canar/app/retrieval/rerank/compare/compare.py "ma question" --qdrant-url http://localhost:6360
```

`--top-k` only changes how many reranked results are printed/returned. It does
not increase the hybrid candidate pool sent to the reranker; that pool is set by
the selected rerank profile's `fusion.output_top_k` value.

## How to Read the Output

The script prints one ranked list for plain `HYBRID`, followed by one section
for each registered model:

- `HYBRID + BGE-V2-M3 RERANK`
- `HYBRID + QWEN-0.6B RERANK`
- `HYBRID + QWEN-4B RERANK`
- `HYBRID + QWEN-8B RERANK`

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

1. Read the query and the top hybrid hits.
2. Compare the top 3 to 5 hits in every `HYBRID + <MODEL> RERANK` section.
3. Ask which registered model gives the LLM the best context for the query.

A good rerank result usually moves directly relevant chunks upward and pushes
lexical-but-off-topic chunks downward. A suspicious result puts irrelevant
chunks first or gives nearly identical `rerank_score` values to every hit.


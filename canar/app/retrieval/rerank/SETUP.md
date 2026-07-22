# Setup: From Zero to Compare Reranking

This guide explains how to go from a fresh shell to running:

```bash
python canar/app/retrieval/rerank/compare/compare.py "comment filtrer un dataframe en R ?"
```

The goal is to compare:

1. Hybrid retrieval
2. Hybrid retrieval plus BGE reranking
3. Hybrid retrieval plus Qwen reranking, only if explicitly requested

You do not need to run the Streamlit app for this comparison.

## 1. Start in the Correct Repository

Use the repository on the large backup disk:

```bash
cd /mnt/backup/cereq/opt/xxx/canar
pwd
```

The output must be:

```text
/mnt/backup/cereq/opt/xxx/canar
```


## 2. Create the Virtual Environment

The project virtual environment is named `.venv_canar`.

```bash
make venv
```

Activate it:

```bash
source .venv_canar/bin/activate
```

Confirm that Python is using the correct virtual environment:

```bash
python -c "import sys; print(sys.prefix)"
```

The output must start with:

```text
/mnt/backup/cereq/opt/xxx/canar/.venv_canar
```

If it starts with `/home/cereq/...`, deactivate and return to the correct
repository path under `/mnt/backup/cereq`.

## 3. Install the Normal Project Dependencies

Install the normal app and dev dependencies:

```bash
make install
```

This installs the main CanaR dependencies plus test tools.

It should not install CUDA PyTorch. PyTorch is installed separately because it is
large and needs the CUDA wheel index.

## 4. Install CUDA PyTorch

Your server reports CUDA 13.0 through `nvidia-smi`, but PyTorch CUDA wheels are
published for specific bundled CUDA runtimes. Use the CUDA 12.8 PyTorch wheel;
the NVIDIA driver is backward-compatible with this runtime.

```bash
make install-torch
```

This runs the equivalent of:

```bash
.venv_canar/bin/pip install --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cu128 \
  "torch>=2.6,<3.0"
```

The Makefile also forces pip temporary files to use:

```text
/mnt/backup/cereq/pip-tmp
```

and pip cache to use:

```text
/mnt/backup/cereq/pip-cache
```

This avoids filling `/`.



## 5. Install Reranking Dependencies

Install the optional reranking dependency group:

```bash
make install-rerank
```

This installs:

```text
transformers
accelerate
safetensors
```

Torch is not in the `rerank` optional dependency group because Torch must be
installed with the CUDA-specific PyTorch index.

## 6. Configure `.env` 

Open:

```text
/mnt/backup/cereq/opt/xxx/canar/.env
```

Make sure the embedding API settings are valid:

```env
EMBED_API_BASE=http://localhost:11434/v1
EMBED_API_KEY=
EMBED_MODEL=bge-m3
FASTEMBED_SPARSE_MODEL=Qdrant/bm25
```

Make sure Qdrant points to the instance that contains the ready hybrid
collection:

```env
QDRANT_URL=
QDRANT_COLLECTIONS=name_of_your_collection_for_hybrid
QDRANT_SPARSE_VECTOR_NAME=sparse
QDRANT_DENSE_VECTOR_NAME=dense
QDRANT_MULTI_VECTOR_NAME=multi
```


Use BGE first. Rerank is configured in the retrieval profile, not in `.env`:

```python
rerank=RerankRetrievalParams(
    output_top_k=5,
    model="bge-v2-m3",
    device="auto",
    max_length=8192,
)
```

Device modes are `auto` for CUDA when available otherwise CPU, `cuda` to force
GPU, `cpu` to force CPU, and `None` to leave PyTorch's default placement alone.

Do not switch to Qwen until BGE works end to end.

## 7. Confirm Qdrant Has the Ready Collection


Confirm that the collection ,,name_of_your_collection_for_hybrid,, exists.



## 8. Run Hybrid vs BGE Reranking

Run one query:

```bash
python canar/app/retrieval/rerank/compare/compare.py "comment filtrer un dataframe en R ?"
```

The first run may download or initialize:

```text
Qdrant/bm25
BAAI/bge-reranker-v2-m3
Qwen/Qwen3-Reranker-0.6B
Qwen/Qwen3-Reranker-4B
Qwen/Qwen3-Reranker-8B
```

That first run can be slow. Later runs should be faster because models are
cached.

## 9. Read the Output

The script prints:

```text
HYBRID
HYBRID + BGE-V2-M3 RERANK
HYBRID + QWEN-0.6B RERANK
HYBRID + QWEN-4B RERANK
HYBRID + QWEN-8B RERANK
```

Inside each section:

```text
[1]
[2]
[3]
```

is the final order for that mode. `[1]` is the first chunk the RAG pipeline
would see.

Fields:

- `score`: original hybrid retrieval score.
- `score_norm`: normalized hybrid retrieval score.
- `rerank_score`: score from the reranker. This is `None` in the plain
  `HYBRID` section.
- `source`, `section`, `text`: the content to inspect manually.

Do not compare `score` directly with `rerank_score`; they are different scales.

Use `score` and `score_norm` to understand hybrid retrieval.
Use `rerank_score` to understand the reranker ordering.

The question to ask is:

```text
Do the top 3 to 5 chunks after reranking give better context than the top 3 to 5 hybrid chunks?
```

## 10. Model Resource Requirements

The comparison now runs every registered reranker automatically. The first run
may download and load all of these models:

```text
BAAI/bge-reranker-v2-m3
Qwen/Qwen3-Reranker-0.6B
Qwen/Qwen3-Reranker-4B
Qwen/Qwen3-Reranker-8B
```

Make sure the machine has enough disk, RAM, and GPU memory before running the
full comparison.


## 11. Useful Overrides

Change the number of reranked results returned:

```bash
python canar/app/retrieval/rerank/compare/compare.py \
  "comment filtrer un dataframe en R ?" \
  --top-k 10
```

This only changes the returned/printed reranker output count. It does not
increase the hybrid candidate pool; that is controlled by the resolved fusion `output_top_k`
on the selected rerank profile.

Override the collection:

```bash
python canar/app/retrieval/rerank/compare/compare.py \
  "comment filtrer un dataframe en R ?" \
  --collection utilitr_bgem3_ds
```

# Ranking

This folder contains reranker implementations for already-retrieved candidates.
Rerankers do not retrieve documents, build context, call the application LLM, or
generate answers. They only take a query plus a list of retrieval hits, run a
local Hugging Face model forward pass for each `(query, document)` pair, and
return the candidates sorted by descending reranker score.

## Available Rerankers

- `qwen-0.6b`: uses `Qwen/Qwen3-Reranker-0.6B`.
- `qwen-4b`: uses `Qwen/Qwen3-Reranker-4B`.
- `qwen-8b`: uses `Qwen/Qwen3-Reranker-8B`.
- `bge-v2-m3`: uses `BAAI/bge-reranker-v2-m3`.

There is no size-implicit `qwen` alias; profiles must select an explicit size.

Both classes expose the same public API:

```python
from canar.app.retrieval.rerank.rerankers import BGEReranker

reranker = BGEReranker()
results = reranker.rerank(query=query, candidates=hits, top_k=10)
```

The returned `RetrievalHit` objects preserve the original retrieval fields and
score. Each result has `rerank_score` set to the score used for ordering; input
hits remain unchanged with their existing `rerank_score` value.


## App Integration

Reranking is wired through `RetrievalService.search(...)` after hybrid fusion.
It is controlled by the selected retrieval profile, not by `.env`. Configure it
in `profiles.py` with `RerankRetrievalParams`:

```python
rerank=RerankRetrievalParams(
    output_top_k=5,
    reranker_name="bge-v2-m3",
    device="auto",
    max_length=8192,
)
```

Device modes are `auto` for CUDA when available otherwise CPU, `cuda` to force
GPU, `cpu` to force CPU, and `None` to leave PyTorch's default placement alone.

The reranker wrapper is built only when a profile has enabled rerank params. Model
weights are still lazy-loaded only on the first reranked query, so startup does
not download or load the Hugging Face model.

`rerank.output_top_k` controls how many reranked hits are returned. The number of
hybrid candidates sent into the reranker is the selected rerank profile's
`fusion.output_top_k`.

## Deployment Notes

These rerankers lazy-load their Hugging Face models on the first `rerank()` call.
When the app is ready to enable reranking, add the ML dependencies to the main
project dependency file, such as the root `requirements.txt` or `pyproject.toml`.

```text
torch>=2.6,<3.0
transformers>=4.51,<5.0
accelerate>=0.34,<2.0
safetensors>=0.4.5,<1.0
```

Install `torch` with the build that matches the target runtime:

- CPU deployment: install the CPU PyTorch build.
- GPU deployment: install the CUDA PyTorch build that matches the container or
  host CUDA runtime.
  #pip install torch --index-url https://download.pytorch.org/whl/cu126

The `transformers` package must be recent enough to load the selected Hugging
Face model classes. The transitive Hugging Face packages installed with
`transformers`, such as `tokenizers`, `safetensors`, and `huggingface-hub`, are
also required at runtime.

The runtime also needs access to download or load the model weights for:

- `Qwen/Qwen3-Reranker-0.6B`
- `Qwen/Qwen3-Reranker-4B`
- `Qwen/Qwen3-Reranker-8B`
- `BAAI/bge-reranker-v2-m3`

After dependencies and model access are available, the app can instantiate either
reranker wherever ranking is configured, without changing retrieval or context
assembly code.

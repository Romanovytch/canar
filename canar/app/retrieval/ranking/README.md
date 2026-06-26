# Ranking

This folder contains reranker implementations for already-retrieved candidates.
Rerankers do not retrieve documents, build context, call the application LLM, or
generate answers. They only take a query plus a list of retrieval hits, run a
local Hugging Face model forward pass for each `(query, document)` pair, and
return the candidates sorted by descending reranker score.

## Available Rerankers

- `QwenReranker`: uses `Qwen/Qwen3-Reranker-8B`.
- `BGEReranker`: uses `BAAI/bge-reranker-v2-m3`.

Both classes expose the same public API:

```python
from canar.app.retrieval.ranking import BGEReranker

reranker = BGEReranker()
results = reranker.rerank(query=query, candidates=hits, top_n=10)
```

The returned candidates preserve the original retrieval fields and score. Each
result also receives a new `rerank_score` field used for ordering.

## Deployment Notes

These rerankers lazy-load their Hugging Face models on the first `rerank()` call.
When the app is ready to enable reranking, add the ML dependencies to the main
project dependency file, such as the root `requirements.txt` or `pyproject.toml`.

```text
torch>=2.6,<3.0
transformers>=4.51,<5.0
sentence-transformers>=4.1,<6.0
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

- `Qwen/Qwen3-Reranker-8B`
- `BAAI/bge-reranker-v2-m3`

After dependencies and model access are available, the app can instantiate either
reranker wherever ranking is configured, without changing retrieval or context
assembly code.

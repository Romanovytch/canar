# Retrieval

This package contains the modular retrieval implementation used to build agent
context from the retrieval.

## Current Flow

```text
User query
  -> RetrievalService
  -> Agent-to-profile mapping
  -> RetrievalProfile
  -> RetrievalStrategy
  -> QdrantRetrievalAdapter
  -> RetrievalHit list
  -> Agent context assembly
```

## Main Components

- `RetrievalProfile`: Python-defined retrieval settings for a profile.
- `RetrievalQuery`: request object passed to a strategy.
- `RetrievalHit`: internal result object returned by adapters and strategies.
- `RetrievalService`: selects the profile and strategy for an agent.
- `RetrievalStrategy`: protocol for retrieval behavior.
- `SimpleVectorStrategy`: dense vector strategy for the current `simple_vector` profile.
- `QdrantRetrievalAdapter`: Qdrant-specific dense vector search adapter.

## Profiles

Profiles are currently built in Python by `build_retrieval_profiles(...)`; there is no
external YAML configuration for retrieval profiles.

Common retrieval behavior is configured at the profile root:

- `collections`
- `fetch_top_k`
- `min_score`
- `source_filter`
- `fallback_top_k`
- `output_top_k`
- `dense.vector_name` / `sparse.vector_name`

Hybrid profiles can tune dense retrieval, sparse retrieval, and fusion separately:

```python
RetrievalProfile(
    name="hybrid",
    strategy="hybrid",
    collections=collections,
    fetch_top_k=30,
    min_score=0.72,
    output_top_k=10,
    dense=DenseRetrievalParams(vector_name="text-dense"),
    sparse=SparseRetrievalParams(vector_name="text-sparse"),
    fusion=FusionRetrievalParams(
        method="weighted_rrf",
        rrf_k=60,
        weights={"dense": 1.0, "sparse": 1.2},
        output_top_k=8,
    ),
)
```

Dense and sparse blocks only contain retriever-specific options. Fusion and reranking
both use `output_top_k`; when omitted, they inherit the profile root value.

Parent-child profiles use the same dense/sparse/fusion blocks for child retrieval and a
small parent-child block for parent lookup behavior:

```python
RetrievalProfile(
    name="parent_child_vector",
    strategy="parent_child_vector",
    collections=collections,
    dense=DenseRetrievalParams(vector_name="text-dense"),
    parent_child=ParentChildRetrievalParams(parent_collection_suffix="_parents"),
)
```

Agent-to-profile mapping is defined in `AGENT_RETRIEVAL_PROFILES`.

## Layers

Strategies decide how retrieval should run for a `RetrievalQuery`. Adapters contain
backend-specific calls and convert backend results into `RetrievalHit`.

See `strategies/README.md` and `adapters/README.md` for layer-specific details.

## Current Limitations

The current implementation supports the `simple_vector` strategy with the Qdrant
adapter. Future retrieval modes may require changes to when embeddings are created
and how strategies compose adapter calls.

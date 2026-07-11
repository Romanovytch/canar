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

The first profile is `simple_vector`. Its configurable fields include:

- `collections`
- `dense.fetch_top_k` / `sparse.fetch_top_k`
- `score_threshold`
- `source_filter`
- `fallback_top_k`
- `vector_name`

Hybrid profiles can tune dense retrieval, sparse retrieval, and fusion separately:

```python
RetrievalProfile(
    name="hybrid",
    strategy="hybrid",
    collections=collections,
    dense=DenseRetrievalParams(fetch_top_k=30, min_score=0.72, max_kept=10),
    sparse=SparseRetrievalParams(
        fetch_top_k=30,
        min_score_ratio=0.10,
        gap_ratio=0.20,
        max_kept=8,
    ),
    fusion=FusionRetrievalParams(
        method="weighted_rrf",
        rrf_k=60,
        weights={"dense": 1.0, "sparse": 1.2},
        output_top_k=8,
    ),
)
```

Use the structured dense, sparse, fusion, and rerank parameter blocks directly.

Parent-child profiles use the same dense/sparse/fusion blocks for child retrieval and a
small parent-child block for parent lookup behavior:

```python
RetrievalProfile(
    name="parent_child_vector",
    strategy="parent_child_vector",
    collections=collections,
    dense=DenseRetrievalParams(fetch_top_k=5, min_score=0.35),
    parent_child=ParentChildRetrievalParams(parent_collection_suffix="_parent"),
)
```

The legacy flat `parent_collection_suffix` field is still resolved for compatibility.

Agent-to-profile mapping is defined in `AGENT_RETRIEVAL_PROFILES`.

## Layers

Strategies decide how retrieval should run for a `RetrievalQuery`. Adapters contain
backend-specific calls and convert backend results into `RetrievalHit`.

See `strategies/README.md` and `adapters/README.md` for layer-specific details.

## Current Limitations

The current implementation supports the `simple_vector` strategy with the Qdrant
adapter. Future retrieval modes may require changes to when embeddings are created
and how strategies compose adapter calls.

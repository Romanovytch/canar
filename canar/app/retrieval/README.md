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
- `top_k`
- `score_threshold`
- `source_filter`
- `fallback_top_k`
- `vector_name`

Agent-to-profile mapping is defined in `AGENT_RETRIEVAL_PROFILES`.

## Layers

Strategies decide how retrieval should run for a `RetrievalQuery`. Adapters contain
backend-specific calls and convert backend results into `RetrievalHit`.

See `strategies/README.md` and `adapters/README.md` for layer-specific details.

## Current Limitations

The current implementation supports the `simple_vector` strategy with the Qdrant
adapter. Future retrieval modes may require changes to when embeddings are created
and how strategies compose adapter calls.

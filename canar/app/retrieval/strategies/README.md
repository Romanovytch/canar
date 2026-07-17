# Retrieval Strategies

## Purpose

Strategies define how retrieval is executed after a `RetrievalQuery` has been built.
This layer lets the application support different retrieval methods without changing
the main application flow.

## Current Structure

```text
strategies/
  __init__.py
  base.py
  simple_vector.py
```

## `RetrievalStrategy`

`base.py` defines the strategy protocol:

```python
class RetrievalStrategy(Protocol):
    def search(self, query: RetrievalQuery) -> list[RetrievalHit]: ...
```

Future strategies should implement this same interface.

## `SimpleVectorStrategy`

`SimpleVectorStrategy` is the first implemented retrieval strategy. It performs
dense vector retrieval using `query.dense_vector` and delegates Qdrant search calls
to `QdrantRetrievalAdapter`.

Behavior:

- Requires `query.dense_vector`.
- Searches every collection defined by the `RetrievalProfile`.
- Normalizes scores within each collection.
- Sorts hits by normalized score, then raw score.
- Filters hits using the profile root `min_score`.
- Keeps `fallback_top_k` results if every hit is below the threshold.

## Current Limitations

Dense, sparse, and hybrid strategies are implemented.

`RetrievalService` currently embeds the query before calling the strategy. This works
The service creates the dense and/or sparse embeddings required by the selected profile.

## Adding a Strategy

1. Create a new strategy class inside `strategies/`.
2. Implement `search(query: RetrievalQuery) -> list[RetrievalHit]`.
3. Register the strategy in `RetrievalService.from_config(...)`.
4. Create or update a `RetrievalProfile` that uses the new strategy name.
5. Add tests for the new strategy behavior.

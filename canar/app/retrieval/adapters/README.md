# Retrieval Adapters

## Purpose

Adapters isolate external retrieval backends from the strategy layer. Backend-specific
API calls belong in adapters, while strategies stay focused on retrieval behavior.

## Current Structure

```text
adapters/
  __init__.py
  qdrant.py
```

## `QdrantRetrievalAdapter`

`QdrantRetrievalAdapter` wraps Qdrant dense vector search.

Responsibilities:

- Creates a `QdrantClient`.
- Performs dense vector search with `query_points`.
- Applies an optional payload filter through `source_filter`.
- Supports an optional `vector_name`.
- Requests payloads but not vectors.
- Converts Qdrant points into internal `RetrievalHit` objects.

## Method Behavior

```python
search_dense(
    collection: str,
    query_vector: list[float],
    top_k: int,
    source_filter: str | None = "utilitr",
    vector_name: str | None = None,
) -> list[RetrievalHit]
```

Parameters:

- `collection`: Qdrant collection to search.
- `query_vector`: dense embedding used as the query vector.
- `top_k`: maximum number of points to request from Qdrant.
- `source_filter`: optional payload `source` filter. Use `None` to disable it.
- `vector_name`: optional named vector passed to Qdrant as `using`.

## Output

The adapter returns `RetrievalHit` objects, not raw Qdrant points. Payload fields such
as `text`, `source`, `url`, `source_url`, and `section` are mapped into `RetrievalHit`.
The full payload is also copied into `metadata`.

## Current Limitations

Qdrant is the only implemented adapter.

## Adding an Adapter

1. Create a new adapter file inside `adapters/`.
2. Keep backend-specific API calls inside the adapter.
3. Convert backend results into `RetrievalHit`.
4. Keep strategy classes independent from backend response formats.
5. Add unit tests for the adapter behavior.

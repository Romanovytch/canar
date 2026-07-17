# Architecture — CanaR chat layer & retrieval

> Scope: chat application and retrieval strategy only.
> Ingestion pipeline and embedding generation are documented separately.

## Overview

CanaR is a Streamlit-based chat UI that authenticates users, persists conversations and messages in a database, and routes each user turn to an LLM-backed agent. The `r_helpdesk` agent augments responses with a modular retrieval service, while `sas_to_r` is prompt-only. Both agents stream LLM output token-by-token back into the UI.

Retrieval is strategy-based. Implemented strategies cover dense, sparse, and hybrid retrieval, with optional parent expansion and reranking. Qdrant-specific code is isolated behind a project-owned adapter, and high-level code consumes typed project-owned retrieval objects.

## Repository layout

```text
canar/
  app/
    main.py                  Streamlit entrypoint and chat flow
    state.py                 SQLModel DB for users, conversations, messages
    config.py                Runtime config from environment variables
    api/
      embed_client.py        OpenAI-compatible embedding client
      llm_client.py          OpenAI-compatible chat client
      retrieval.py           Backward-compatible wrapper for old search_qdrant callers
    agents/
      r_helpdesk.py          Retrieval-grounded French prompt assembly
      sas_to_r.py            Prompt-only SAS-to-R translation prompt assembly
    retrieval/
      models.py              RetrievalProfile, RetrievalQuery, RetrievalHit
      profiles.py            Python profile registry and agent-to-profile mapping
      service.py             RetrievalService orchestration
      adapters/
        qdrant.py            Qdrant-only adapter/client code
      strategies/
        base.py              RetrievalStrategy protocol
        simple_vector.py     Dense vector search strategy
        simple_sparse.py     Sparse vector search strategy
        hybrid.py            Dense/sparse fusion strategy
      fusion/                Project-owned fusion implementations
      expanders/             Optional result expansion, including parent-child
      rerank/                Optional reranker adapters and comparison tooling
    ui/
      chat.py                Chat rendering and assistant stream persistence
      sidebar.py             Conversation and agent sidebar
```

## Request lifecycle

1. User submits a turn via `st.chat_input`, and the UI immediately renders it.
2. The message is persisted to the DB with `DB.add_message`.
3. Agent routing happens based on `st.session_state["agent"]`.
4. For `sas_to_r`, `sas_to_r.build_messages` assembles the prompt, then `ChatClient.stream_chat` streams the answer.
5. For `r_helpdesk`, `RetrievalService.search(agent, query)` selects the agent retrieval profile, embeds the query, executes the configured retrieval strategy, and returns `list[RetrievalHit]`.
6. `r_helpdesk.build_messages` assembles context from `RetrievalHit` objects and returns both LLM messages and source metadata for the UI.
7. The response stream is rendered token-by-token and saved to the DB.

## Conversation state management

| Aspect | Implementation |
|---|---|
| Storage backend | SQLModel ORM with Postgres if `DB_POSTGRES_URL` is set, otherwise SQLite file at `APP_DB` / `data/app.db` |
| History format | `Message` rows with `role` and `content`, ordered by `Message.id` |
| Truncation / summarisation strategy | Not implemented |
| Session / user identity | `st.session_state["user_id"]` plus `Conversation.user_id` to scope access |
| Agent selection | `st.session_state["agent"]` and the conversation `agent` field |

## Retrieval architecture

Retrieval is split into five layers:

1. Application orchestration: `canar/app/main.py` calls `RetrievalService.search(...)`.
2. Retrieval service: `canar/app/retrieval/service.py` maps the active agent to a profile, embeds the query, and dispatches to a strategy.
3. Profile config: `canar/app/retrieval/profiles.py` defines simple Python profile objects and agent-to-profile mapping.
4. Strategy implementation: modules under `canar/app/retrieval/strategies/` own dense, sparse, and hybrid behavior such as per-collection normalization, sorting, pruning, and fusion.
5. Backend adapter: `canar/app/retrieval/adapters/qdrant.py` owns Qdrant client calls, filters, named vector selection, and payload-to-`RetrievalHit` conversion.

The old `canar/app/api/retrieval.py::search_qdrant` function remains as a compatibility wrapper. New application code should use `RetrievalService`.

## Retrieval profiles

Profiles are configured in Python, not YAML/TOML/env files. The active profile is selected by agent.

Current mapping:

```python
AGENT_RETRIEVAL_PROFILES = {
    "generic_agent": "hybrid_rerank_bge",
    "r_helpdesk": "simple_vector",
    "sas_to_r": None,
}
```

Root retrieval defaults are shared by dense and sparse strategies:

- `fetch_top_k=10`: candidates requested per collection and retriever.
- `min_score=0.75`: minimum normalized score after per-collection normalization.
- `fallback_top_k=3`: results returned when pruning removes every hit.
- `max_results=None`: optional cap applied after pruning or fallback.
- `source_filter=None`: no source payload filter by default.

Strategy-specific `fetch_top_k` and `min_score` values override the root values when
present. Named-vector selection belongs to the corresponding dense or sparse block.

Canonical dense and sparse profiles therefore remain concise:

```python
RetrievalProfile(
    name="simple_vector",
    strategy="simple_vector",
    collections=cfg.qdrant_collections,
    dense=DenseRetrievalParams(
        vector_name=cfg.qdrant_dense_vector_name or None,
    ),
)
RetrievalProfile(
    name="simple_sparse",
    strategy="simple_sparse",
    collections=cfg.qdrant_collections,
    sparse=SparseRetrievalParams(
        vector_name=cfg.qdrant_sparse_vector_name or None,
    ),
)
```

Hybrid rerank implemented profile:

```python
RetrievalProfile(
    name="hybrid_rerank_bge",
    strategy="hybrid",
    collections=cfg.qdrant_collections,
    fetch_top_k=20,
    fallback_top_k=5,
    dense=DenseRetrievalParams(
        vector_name=cfg.qdrant_dense_vector_name or None,
    ),
    sparse=SparseRetrievalParams(
        vector_name=cfg.qdrant_sparse_vector_name or None,
    ),
    fusion=FusionRetrievalParams(
        method="rrf",
        rrf_k=60,
        weights={"dense": 1.0, "sparse": 1.0},
        candidate_top_k=20,
    ),
    rerank=RerankRetrievalParams(
        final_top_k=5,
        model="bge-v2-m3",
    ),
)
```

Parent-child implemented profile:

```python
RetrievalProfile(
    name="simple_vector_parent_child",
    strategy="simple_vector",
    collections=cfg.qdrant_collections,
    dense=DenseRetrievalParams(
        vector_name=cfg.qdrant_dense_vector_name or None,
    ),
    parent_child=ParentChildRetrievalParams(parent_collection_suffix="_parent"),
)
```

Parent-child hybrid uses the same `dense`, `sparse`, and `fusion` blocks as `hybrid`
for child retrieval, plus `parent_child` for parent collection lookup. Parent-child and
reranker variants are derived from canonical profiles with `dataclasses.replace`; the
reranker model variants come from a small model-name mapping.

Sparse and hybrid retrieval assume compatible sparse vectors already exist in Qdrant. The sparse vector model and named-vector configuration must match the ingestion pipeline.

## Retrieval strategy

### `simple_vector`

The `simple_vector` strategy applies the common profile policy:

- Generate a dense query embedding using `EmbedClient`.
- Query each configured Qdrant collection with the dense vector.
- Apply `source_filter` when configured; the default is no filter.
- Retrieve `dense.fetch_top_k` hits per collection when set, otherwise use profile `fetch_top_k=10`.
- Min-max normalize scores within each collection.
- Fuse all collection hits by sorting on `(score_norm, score)` descending.
- Keep hits with `score_norm >= dense.min_score` when set, otherwise use profile `min_score=0.75`.
- If no hits survive the threshold, return the top 3 fused hits.
- Apply `max_results` last when configured.

### `simple_sparse`

The `simple_sparse` strategy mirrors `simple_vector` with a sparse query vector:

- Generate a sparse query vector using `FastEmbedClient` only when the selected profile requires it.
- Query each configured Qdrant collection with the sparse vector.
- Apply `source_filter` when configured; the default is no filter.
- Retrieve `sparse.fetch_top_k` hits per collection when set, otherwise use profile `fetch_top_k=10`.
- Min-max normalize scores within each collection.
- Fuse all collection hits by sorting on `(score_norm, score)` descending.
- Keep hits with `score_norm >= sparse.min_score` when set, otherwise use profile `min_score=0.75`.
- When `sparse.gap_ratio` is configured, keep results until the first consecutive normalized-score gap greater than that ratio.
- If no hits survive the threshold, return the top 3 fused hits.
- Apply `max_results` last when configured.

This strategy assumes sparse vectors already exist in Qdrant. The sparse vector model and named-vector configuration must match the ingestion pipeline.

### Hybrid retrieval

The `hybrid` strategy runs dense and sparse retrieval separately, then fuses both ranked lists with weighted Reciprocal Rank Fusion (RRF). The local RRF score is:

```text
score += retriever_weight / (rrf_k + rank)
```

Ranks are one-based to preserve the original local fusion behavior. With the default `fusion.weights={"dense": 1.0, "sparse": 1.0}` and `fusion.rrf_k=60`, hybrid retrieval behaves like equal-weight RRF. Raising the dense weight favors semantic matches; raising the sparse weight favors exact or lexical matches.

The result-list order is `[dense_hits, sparse_hits]`, so the positional RRF weights are `[fusion.weights["dense"], fusion.weights["sparse"]]`. This mirrors Qdrant weighted RRF semantics, where weights must follow the prefetch order exactly.

Hybrid retrieval runs both branches with the same root policy and their respective
overrides. The precedence is `dense/sparse override` then `RetrievalProfile` default.

Reranking is enabled by selecting a rerank profile, such as `hybrid_rerank_bge`, with
`rerank=RerankRetrievalParams(...)`. On rerank profiles, `fusion.candidate_top_k` is
the RRF candidate pool sent to rerank, and `rerank.final_top_k` is the final
reranked result count.

## Data model

Retrieval code exchanges typed project-owned objects:

```python
@dataclass(frozen=True)
class SparseVector:
    indices: list[int]
    values: list[float]


@dataclass(frozen=True)
class DenseRetrievalParams:
    vector_name: str | None = None
    fetch_top_k: int | None = None
    min_score: float | None = None


@dataclass(frozen=True)
class SparseRetrievalParams:
    vector_name: str | None = None
    fetch_top_k: int | None = None
    min_score: float | None = None
    gap_ratio: float | None = None


@dataclass(frozen=True)
class FusionRetrievalParams:
    method: FusionMethod = "rrf"
    rrf_k: int = 60
    weights: dict[str, float] = field(default_factory=dict)
    candidate_top_k: int = 5


@dataclass(frozen=True)
class ParentChildRetrievalParams:
    parent_collection_suffix: str = "_parent"


@dataclass(frozen=True)
class RerankRetrievalParams:
    final_top_k: int = 5
    model: str = "bge-v2-m3"
    device: str | None = "auto"
    max_length: int = 8192


@dataclass(frozen=True)
class RetrievalProfile:
    name: str
    strategy: RetrievalStrategyName
    collections: tuple[str, ...]
    fetch_top_k: int = 10
    min_score: float = 0.75
    fallback_top_k: int = 3
    max_results: int | None = None
    source_filter: str | None = None
    dense: DenseRetrievalParams | None = None
    sparse: SparseRetrievalParams | None = None
    fusion: FusionRetrievalParams | None = None
    rerank: RerankRetrievalParams | None = None
    parent_child: ParentChildRetrievalParams | None = None


@dataclass(frozen=True)
class RetrievalQuery:
    text: str
    profile_name: str
    dense_vector: list[float] | None = None
    sparse_vector: SparseVector | None = None


@dataclass(frozen=True)
class RetrievalHit:
    text: str
    collection: str
    score: float
    score_norm: float
    generation_text: str | None = None
    source: str | None = None
    source_url: str | None = None
    section: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    rerank_score: float | None = None
```

Parameter blocks validate their own numeric options. `RetrievalProfile` validates root
limits and required blocks during construction: `simple_vector` requires `dense`,
`simple_sparse` requires `sparse`, and `hybrid` requires `dense`, `sparse`, and
`fusion`. Reranking is supported only for hybrid profiles. Legacy flat fields,
string-based fusion configuration, and trivial `*_params()` accessors are not supported.

Agents and UI code must not depend on Qdrant result objects, Qdrant filters, payload response types, or SDK-specific classes.

## Prompt assembly

Logical template for `r_helpdesk`:

```text
[system prompt: SYSTEM_PROMPT_FR]
[user message: "Question: {query}\n\nContexte (extraits documentaires):\n{context_text}\n\nConsigne: ..."]
```

`r_helpdesk.assemble_context` accepts `list[RetrievalHit]`, enumerates citations as `[S1]`, `[S2]`, etc., and returns `src_list` for the UI sources panel. Conversation history is not injected into the prompt in the current code path.

Logical template for `sas_to_r`:

```text
[system prompt: SYSTEM_PROMPT_FR]
[user message: either "Voici le code SAS..." or "Demande utilisateur ..."]
```

No retrieval context or history is added for this agent.

## LLM and embedding integration

| Aspect | Detail |
|---|---|
| Chat provider & model | OpenAI-compatible client with configurable model name |
| Chat configurability | `LLM_API_BASE`, `LLM_API_KEY`, `LLM_MODEL` |
| Streaming method | `chat.completions.create(..., stream=True)` yielding token deltas |
| Embedding provider | OpenAI-compatible `/embeddings` endpoint |
| Embedding configurability | `EMBED_API_BASE`, `EMBED_API_KEY`, `EMBED_MODEL` |
| Query embedding normalization | `EmbedClient.embed_query` L2-normalizes returned vectors |
| Tool / function calling | Not implemented |
| Error & timeout handling | Minimal; no explicit retry strategy |

## External dependencies

| Dependency | Role |
|---|---|
| streamlit | Chat UI and session state |
| openai | OpenAI-compatible chat completions |
| qdrant-client | Vector search inside the Qdrant adapter only |
| fastembed | Optional sparse query vector generation behind `FastEmbedClient` |
| requests + numpy | Embedding HTTP call and vector normalization |
| sqlmodel + psycopg | User/conversation/message persistence |

## Environment variables

| Variable | Purpose |
|---|---|
| LLM_API_BASE | Base URL for OpenAI-compatible LLM API |
| LLM_API_KEY | LLM API key |
| LLM_MODEL | LLM model name |
| EMBED_API_BASE | Base URL for embeddings API |
| EMBED_API_KEY | Embeddings API key |
| EMBED_MODEL | Embeddings model name |
| FASTEMBED_SPARSE_MODEL | Optional local FastEmbed sparse model used by sparse retrieval profiles |
| QDRANT_URL | Qdrant endpoint |
| QDRANT_API_KEY | Qdrant API key |
| QDRANT_COLLECTIONS | Qdrant collections used by retrieval profiles |
| QDRANT_DENSE_VECTOR_NAME | Optional Qdrant named dense vector |
| QDRANT_SPARSE_VECTOR_NAME | Optional Qdrant named sparse vector |
| DB_POSTGRES_URL | Postgres connection string |
| APP_DB | SQLite path when Postgres is not configured |
| CANAR_HEADLESS | Streamlit headless mode toggle |
| CANAR_PORT | Streamlit port override |

## Key design decisions

- Use Streamlit as the only application entrypoint, with chat events driven by `st.chat_input` rather than HTTP routes.
- Store conversations and messages in a relational DB keyed by user ID, with a per-conversation agent field.
- Select retrieval profile by agent using Python config in `canar/app/retrieval/profiles.py`.
- Keep `simple_vector` as the default retrieval strategy for now.
- Keep shared retrieval policy at the profile root and allow dense/sparse blocks to override only retriever-specific values.
- Use stage-specific limits: `fetch_top_k`, `fusion.candidate_top_k`, and `rerank.final_top_k`.
- Generate parent-child and reranker variants from canonical profiles with `dataclasses.replace`.
- Keep Qdrant-specific imports and SDK calls in `canar/app/retrieval/adapters/qdrant.py`.
- Return `RetrievalHit` objects from retrieval code; do not pass Qdrant objects or Qdrant-shaped payload dicts into agents.
- Keep prompts per agent and do not inject conversation history into LLM messages in the current flow.

## Dependency isolation

When implementing features that rely on an external library, SDK, API client, or service-specific package, keep that dependency isolated behind a small project-owned interface or adapter.

Rules:

- Do not call external SDKs directly from Streamlit UI code, agents, or prompt assembly code.
- Wrap external libraries in project-owned modules such as `canar/app/retrieval/adapters/qdrant.py`, `canar/app/api/llm_client.py`, or `canar/app/api/embed_client.py`.
- Keep service-specific types, payloads, filters, and response formats inside the adapter layer.
- Return typed project-owned data structures such as `RetrievalHit` from adapters and strategies.
- If replacing an external dependency, such as Qdrant, OpenAI-compatible clients, or SQLModel, the change should affect the smallest possible number of files.
- Prefer dependency injection or configuration-based selection when multiple backends or strategies may be supported.
- Avoid leaking vendor-specific concepts into high-level application logic unless they are part of an explicit architectural decision.

## Open questions & gaps

- History truncation or summarization policy is not defined in code.
- No explicit LLM error/timeout handling or retry strategy.
- Tool/function calling support is not implemented.
- Retrieval result caching is not present.
- Sparse and hybrid retrieval/index availability belongs to the separate ingestion project and must match `FASTEMBED_SPARSE_MODEL` / `QDRANT_SPARSE_VECTOR_NAME`.
- `.env.example` references `MISTRAL_API_BASE`, while the runtime config expects `LLM_API_BASE`.

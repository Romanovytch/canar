# AGENTS.md

## Project context

Before making changes, read these files:

1. `ARCHITECTURE.md` — source of truth for system architecture, modules, data flow, and design constraints.
2. `.github/copilot-instructions.md` — existing project coding conventions and assistant instructions.

If those files conflict with this file, prefer this order:

1. `AGENTS.md`
2. `ARCHITECTURE.md`
3. `.github/copilot-instructions.md`
4. local code style already present in the touched files

## Working rules

- Do not make large architectural changes without explaining the tradeoff first.
- Prefer minimal, high-confidence changes.
- Preserve the existing architecture described in `ARCHITECTURE.md`.
- Follow the conventions from `.github/copilot-instructions.md`.
- Before editing, inspect the relevant files and summarize the intended change.
- After editing, run the relevant tests, linters, or type checks when available.
- If tests cannot be run, explain why and suggest the exact command the developer should run.

## Current retrieval architecture

- Retrieval is modular and strategy-based under `canar/app/retrieval/`.
- `canar/app/main.py` should call `RetrievalService.search(agent, query)` for retrieval; it should not embed queries or call Qdrant directly.
- Retrieval profile selection is by agent in `canar/app/retrieval/profiles.py`.
- The only implemented profile/strategy is `simple_vector`, mapped to `r_helpdesk`.
- `sas_to_r` has no retrieval profile.
- `simple_vector` must preserve the current dense-vector behavior: configured Qdrant collections, `top_k=5`, `source_filter="utilitr"`, per-collection min-max normalization, `score_threshold=0.35`, and top-3 fallback.
- Qdrant-specific imports, filters, query arguments, named-vector selection, and payload conversion belong in `canar/app/retrieval/adapters/qdrant.py`.
- Strategies should return project-owned `RetrievalHit` objects from `canar/app/retrieval/models.py`; agents and UI code must not depend on Qdrant result objects or Qdrant-shaped payload dictionaries.
- `canar/app/api/retrieval.py::search_qdrant` is a backward-compatible wrapper only. New code should use the retrieval service.
- Hybrid retrieval is intentionally not implemented in this repository yet. Ingestion and sparse retrieval support are out of scope unless explicitly requested.

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

## Project workflow

When asked to implement a feature or fix a bug:

1. Read the relevant part of `ARCHITECTURE.md`.
2. Read `.github/copilot-instructions.md`.
3. Inspect the affected code.
4. Propose a short plan for non-trivial changes.
5. Make the smallest safe change.
6. Run relevant validation commands.
7. Summarize files changed and remaining risks.

# Copilot instructions for CanaR

Use `ARCHITECTURE.md` as the deeper design reference for the chat layer and retrieval flow; keep this file aligned with it.

## Build, test, and lint

- Create the venv and install dev deps: `make venv && make install`
- Start local services: `make up` (`make down`, `make reset`, `make logs`)
- Run the app: `make run` or `canar`
- Run the full test suite: `make test` or `pytest -q`
- Run a single test: `pytest canar/app/tests/test_smoke.py::test_import_canar -q`
- Lint: `make lint` or `ruff check .`
- Format check: `make format-check` or `ruff format --check .`
- Auto-format: `make format` or `ruff format . && ruff check . --fix`
- CI parity: `make ci`

## High-level architecture

- CanaR is a Streamlit chat app whose chatbot identities, prompts, file types, and export behavior are loaded from `canar/app/chatbots/chatbotconfig.yaml`.
- `canar/launch.py` is the CLI entrypoint; it starts Streamlit with `CANAR_HEADLESS` and `CANAR_PORT`.
- `canar/app/main.py` orchestrates the whole request flow: auth, session state, sidebar, prompt assembly, retrieval, streaming, and persistence.
- `canar/app/state.py` stores users, conversations, and messages with SQLModel; it uses Postgres when `DB_POSTGRES_URL` is set, otherwise SQLite at `data/app.db`.
- `canar/app/retrieval/service.py` is the retrieval orchestration layer. It selects the agent retrieval profile, embeds the query, dispatches to the configured strategy, and returns typed `RetrievalHit` objects.
- `canar/app/profiles.py` contains chat profiles that combine generation settings with an optional retrieval profile. YAML attaches one by `profile_name`; `r_helpdesk` uses `simple_vector` and `sas_to_r` uses `prompt_only`.
- `canar/app/retrieval/profiles.py` contains the Python retrieval profile registry and a legacy agent mapping for non-YAML callers.
- Retrieval strategies support dense, sparse, and hybrid search. Dense and sparse branches use per-collection min-max normalization, fused sorting, thresholding, and fallback behavior.
- `canar/app/retrieval/adapters/qdrant.py` is the only place new retrieval code should call `qdrant-client` directly.
- `canar/app/api/retrieval.py` is a backward-compatible wrapper for old `search_qdrant` callers; new code should use `RetrievalService`.
- `canar/app/api/llm_client.py` and `canar/app/api/embed_client.py` isolate the OpenAI-compatible chat and embedding calls.
- `canar/app/ui/chat.py` renders history and streams assistant output back into the UI; `canar/app/ui/sidebar.py` manages conversation creation, rename, delete, and agent selection.
- `canar/app/yaml_loader.py` validates and atomically upserts chatbot configuration at startup.
- `canar/app/utils/llm_utils.py` converts typed `RetrievalHit` objects into cited context and is the universal message-building path for every chatbot.

## Key conventions

- Keep user-facing text consistent with the current French UI and prompts.
- Preserve Streamlit session keys and rerun flow: `user_id`, `conv_id`, and `agent` are the core state anchors.
- Keep ownership checks in `DB` methods; conversation access is always scoped to the logged-in user.
- `AppConfig` loads `.env` at import time and requires a non-empty `QDRANT_COLLECTIONS`.
- Use the runtime config names from `canar/app/config.py` and `canar/app/state.py` (`LLM_*`, `EMBED_*`, `QDRANT_*`, `DB_POSTGRES_URL`, `APP_DB`) when editing code.
- Host and Docker setups differ: host runs use `localhost`, containerized runs use service names like `qdrant` and `postgres`.
- Retrieval profiles default to `source_filter=None`, `fetch_top_k=10`, `min_score=0.75`, `output_top_k=5`, and top-3 fallback. Dense and sparse blocks may override `fetch_top_k` and `min_score`; fusion and rerank blocks inherit root `output_top_k` unless they override it. Preserve per-collection min-max normalization unless an explicit task changes it.
- Do not call Qdrant directly from Streamlit UI code, agents, prompt assembly, or retrieval strategies outside the adapter boundary.
- Do not pass Qdrant objects or Qdrant-shaped payload dictionaries into message assembly. Use project-owned `RetrievalHit` objects.
- Hybrid retrieval combines dense and sparse branches with configurable RRF fusion and optional reranking.
- Keep prompt text in chatbot YAML rather than in `main.py`.
- Keep retrieval collections and ranking parameters authoritative in `RetrievalProfile`, and generation parameters authoritative in `ChatProfile`. Similarly named YAML fields are compatibility metadata and must not bypass these profiles.
- Resolve YAML `profile_name` values before constructing `RetrievalService`; pass its retrieval mapping explicitly so rerankers are initialized for the configured chatbots.
- Prefer the existing `make` targets when documenting or verifying workflows, since they mirror CI.

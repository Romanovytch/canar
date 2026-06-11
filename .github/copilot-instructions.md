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

- CanaR is a Streamlit chat app for two assistant modes: `r_helpdesk` and `sas_to_r`.
- `canar/launch.py` is the CLI entrypoint; it starts Streamlit with `CANAR_HEADLESS` and `CANAR_PORT`.
- `canar/app/main.py` orchestrates the whole request flow: auth, session state, sidebar, prompt assembly, retrieval, streaming, and persistence.
- `canar/app/state.py` stores users, conversations, and messages with SQLModel; it uses Postgres when `DB_POSTGRES_URL` is set, otherwise SQLite at `data/app.db`.
- `canar/app/retrieval/service.py` is the retrieval orchestration layer. It selects the agent retrieval profile, embeds the query, dispatches to the configured strategy, and returns typed `RetrievalHit` objects.
- `canar/app/retrieval/profiles.py` contains the Python retrieval profile registry and agent-to-profile mapping. `r_helpdesk` currently uses `simple_vector`; `sas_to_r` has no retrieval profile.
- `canar/app/retrieval/strategies/simple_vector.py` implements the only current retrieval strategy. It preserves dense-vector search, per-collection min-max normalization, fused sorting, thresholding, and top-3 fallback.
- `canar/app/retrieval/adapters/qdrant.py` is the only place new retrieval code should call `qdrant-client` directly.
- `canar/app/api/retrieval.py` is a backward-compatible wrapper for old `search_qdrant` callers; new code should use `RetrievalService`.
- `canar/app/api/llm_client.py` and `canar/app/api/embed_client.py` isolate the OpenAI-compatible chat and embedding calls.
- `canar/app/ui/chat.py` renders history and streams assistant output back into the UI; `canar/app/ui/sidebar.py` manages conversation creation, rename, delete, and agent selection.
- `canar/app/agents/r_helpdesk.py` builds a retrieval-grounded French prompt from `RetrievalHit` objects; `canar/app/agents/sas_to_r.py` builds a prompt-only SAS→R translation prompt.

## Key conventions

- Keep user-facing text consistent with the current French UI and prompts.
- Preserve Streamlit session keys and rerun flow: `user_id`, `conv_id`, and `agent` are the core state anchors.
- Keep ownership checks in `DB` methods; conversation access is always scoped to the logged-in user.
- `AppConfig` loads `.env` at import time and requires a non-empty `QDRANT_COLLECTIONS`.
- Use the runtime config names from `canar/app/config.py` and `canar/app/state.py` (`LLM_*`, `EMBED_*`, `QDRANT_*`, `DB_POSTGRES_URL`, `APP_DB`) when editing code.
- Host and Docker setups differ: host runs use `localhost`, containerized runs use service names like `qdrant` and `postgres`.
- Retrieval edits should preserve the existing `source_filter="utilitr"` behavior, `top_k=5`, per-collection min-max normalization, `score_threshold=0.35`, and top-3 fallback unless an explicit task changes them.
- Do not call Qdrant directly from Streamlit UI code, agents, prompt assembly, or retrieval strategies outside the adapter boundary.
- Do not pass Qdrant objects or Qdrant-shaped payload dictionaries into agents. Use project-owned `RetrievalHit` objects.
- Hybrid retrieval is not implemented yet; do not add sparse or hybrid behavior unless explicitly requested.
- Keep prompt text in the agent modules rather than in `main.py`.
- Prefer the existing `make` targets when documenting or verifying workflows, since they mirror CI.

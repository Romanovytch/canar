[🇫🇷 Français](README.fr.md) | [🇬🇧 English](README.md)

# CanaR

CanaR is a chat application that lets you create and expose **RAG chatbots** (Retrieval-Augmented Generation) powered by **LLMs**. It provides a **web interface** to chat with assistants that can rely on a **document knowledge base** (e.g., internal documentation, HR or legal documents, publications, code, …) previously ingested into **Qdrant** — typically via the **AgoRa** ingestion tool.

CanaR lets you:
- :left_speech_bubble: **Access a chat interface** with user account management
- :robot: **Define multiple chatbots/assistants**, each with a custom prompt and behavior tailored to specific tasks
- :mag_right: **Augment answers with document retrieval** (RAG) and **cite the sources** used
- :page_facing_up: **Keep conversation history** (sessions and messages)

For more information about installation, features, and contributing, see the **documentation**: [CanaR Documentation](#canar) (under construction :construction: ).

---

## Quickstart (local)

A minimal `Makefile` is available (optional):

```shell
Targets:
  make up            - Start Qdrant + Postgres (docker compose)
  make down          - Stop containers
  make reset         - Stop + remove volumes
  make logs          - Follow docker logs
  make venv          - Create venv (.venv)
  make install       - Install CanaR (editable) + dev tools
  make run           - Run CanaR (entrypoint or Streamlit fallback)
  make test          - Run tests (pytest)
  make lint          - Run ruff lint (check)
  make format        - Auto-format with ruff
  make format-check  - Check formatting with ruff
  make ci            - Run lint + format-check + tests
```

### 1) Start Qdrant + Postgres (Docker Compose)

```shell
cd infra
docker compose up -d
docker compose ps
```
or `make up`

The Qdrant UI is available at `http://localhost:6333/dashboard`.

### 2) Configure environment variables

Create a `.env` file at the root of the project from `.env.example`.

> :warning: Important: the value of `QDRANT_URL` and `DB_POSTGRES_URL` depends on **where CanaR is running**:
> - CanaR started on the host (venv / canar / streamlit run) → use `localhost`
> - CanaR started in Docker (canar service in a compose) → use Docker service names: `qdrant`, `postgres`

| Variable             | Description                           | Example                                                                                             |
| -------------------- | ------------------------------------- | --------------------------------------------------------------------------------------------------- |
| `LLM_API_BASE`       | LLM API base URL                      | `https://api.mistral.ai/v1`                                                                         |
| `LLM_API_KEY`        | LLM API key                           | `sk-...`                                                                                            |
| `LLM_MODEL`          | Model name                            | `mistral-medium`                                                                                    |
| `LLM_PROVIDER_NAME`  | Optional provider name                | `mistral`                                                                                           |
| `EMBED_API_BASE`     | Embeddings API base URL               | `https://api.mistral.ai/v1`                                                                         |
| `EMBED_API_KEY`      | Embeddings API key                    | `sk-...`                                                                                            |
| `EMBED_MODEL`        | Embeddings model name                 | `mistral-embed`                                                                                     |
| `FASTEMBED_SPARSE_MODEL` | Optional sparse retrieval model   | `Qdrant/bm25`                                                                                       |
| `QDRANT_URL`         | Qdrant URL                            | `http://localhost:6333` (host) or `http://qdrant:6333` (docker)                                     |
| `QDRANT_API_KEY`     | Qdrant API key (if enabled)           | `...`                                                                                               |
| `QDRANT_COLLECTIONS` | Allowed collections (comma-separated) | `col1,col2`                                                                                         |
| `QDRANT_SPARSE_VECTOR_NAME` | Optional Qdrant named sparse vector | leave empty for default sparse vector                                                        |
| `DB_POSTGRES_URL`    | Postgres URL (SQLAlchemy/psycopg)     | `postgresql+psycopg://canar:canar@localhost:5432/canar` (host) or `...@postgres:5432/...` (docker) |

### 3) Install and run CanaR

```shell
python -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install -e .
```
or `make venv` & `make install`

Run the application:
```shell
canar
```
or `make run`

If the `canar` command is not available, run Streamlit directly:
```shell
streamlit run canar/app/main.py --server.headless true --server.port 8501
```

## Requirements

- **Docker + Docker Compose** (Linux) or **Docker Desktop** (Windows)
- **Python ≥ 3.10**

CanaR requires:
- a vector database (RAG): **Qdrant**
- a relational database (user accounts + history): **Postgres**

The docker-compose file is provided in `infra/docker-compose.yml`.

## Configuration

Example `.env`:

```dotenv
# LLM (example)
LLM_API_BASE=https://url_llm/v1
LLM_API_KEY=
LLM_MODEL=model_name

# Embeddings (example)
EMBED_API_BASE=https://url_embed/v1
EMBED_API_KEY=
EMBED_MODEL=model_name
FASTEMBED_SPARSE_MODEL=Qdrant/bm25

# Qdrant
# QDRANT_URL=http://qdrant:6333        # if CanaR runs in Docker
QDRANT_URL=http://localhost:6333       # if CanaR runs on the host
QDRANT_API_KEY=
QDRANT_COLLECTIONS=collection1_v1,collection2_v1
QDRANT_SPARSE_VECTOR_NAME=

# Postgres
# DB_POSTGRES_URL=postgresql+psycopg://canar:canar@postgres:5432/canar   # docker
DB_POSTGRES_URL=postgresql+psycopg://canar:canar@localhost:5432/canar    # host
```

## Contributing

See: [CONTRIBUTING.md](CONTRIBUTING.md)

## License

TBD (to be confirmed by maintainers).

## Common issues

### Ports

The following ports must be available:

| Service           | Default port |
| ----------------- | ------------ |
| Postgres          | `5432`       |
| Qdrant            | `6333`       |
| CanaR (Streamlit) | `8501`       |

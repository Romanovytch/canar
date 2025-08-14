[🇫🇷 Français](README.fr.md) | [🇬🇧 English](README.md)

# How to use CanaR

## Virtual env setup

(Optional) First you should use a virtual environment it's best practice:

```shell
cd CanaR
python -m venv .venv
source .venv/bin/activate
```

## Download dependencies

CanaR has a `requirements.txt` listing all the dependencies and their minimal versions and a `pyproject.toml` as well:

```shell
pip install -U pip
pip install -e .
```

Depending on your machine and network, it can take some time.

## Environement variables

You can setup environment variables in a `.env` file at the root of the project.
The `.env.example` file gives you an example of what it could look like :

```
# LLM (Mistral 24B) — OpenAI-compatible
MISTRAL_API_BASE="https://url_mistral_model/v1"
MISTRAL_API_KEY="apikeyhere"
MISTRAL_MODEL="mistralai/Mistral-Small-24B-Instruct-2501" # for instance

# Embeddings (Gemma2 multilingual) — OpenAI-compatible
EMBED_API_BASE="https://url_embedding_model/v1"
EMBED_API_KEY="apikeyhere"
EMBED_MODEL="BAAI/bge-multilingual-gemma2" # for instance

# Qdrant
QDRANT_URL="http://qdrant:6333"
QDRANT_API_KEY="apikeyhere"
# comma-separated list of collections (first is default current)
QDRANT_COLLECTIONS=collection1_v1, collection2_v1, collection3_v2

# DB
DB_POSTGRES_URL=postgresql+psycopg://user-name:password@base-url:5432/canar
```


## Usage

To launch the app:

```
canar
```


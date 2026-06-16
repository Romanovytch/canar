[🇫🇷 Français](README.fr.md) | [🇬🇧 English](README.md)

# CanaR

CanaR est une application de chat qui permet de créer et d’exposer des **chatbots RAG** (Retrieval-Augmented Generation) basés sur des **LLM**. Elle fournit une **interface web** pour converser avec des assistants capables de s’appuyer sur une **base documentaire** (ex. documentation interne, documents RH ou juridiques, publications, code, …) préalablement ingérée dans **Qdrant** — typiquement via l’outil d’ingestion **AgoRa**.

CanaR permet de :
- :left_speech_bubble: **Accéder à une interface de chat** avec gestion de comptes utilisateurs
- :robot: **Définir plusieurs chatbots/assistants**, chacun avec une prompt et un comportement adaptés à des tâches spécifiques
- :mag_right: **Enrichir les réponses par la recherche documentaire** (RAG) et **citer les sources** utilisées
- :page_facing_up: **Conserver l’historique** des conversations (sessions et messages)

Pour plus d’informations sur l’installation, les fonctionnalités et la contribution, consultez la **documentation** : [Documentation CanaR](#canar) (en construction :construction: ).

---

## Démarrage rapide (local)

Un `Makefile` minimal est disponible (optionnel) :

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

### 1) Démarrer Qdrant + Postgres (Docker Compose)

```shell
cd infra
docker compose up -d
docker compose ps
```
ou `make up`

L’UI Qdrant est accessible sur `http://localhost:6333/dashboard`.

### 2) Configurer les variables d’environnement

Crée un fichier `.env` à la racine du projet à partir de `.env.example`.
> :warning: Important : la valeur de `QDRANT_URL` et `DB_POSTGRES_URL` dépend de **l’endroit où tourne CanaR** :
> - CanaR lancé sur l’hôte (venv / canar / streamlit run) → utiliser `localhost`
> - CanaR lancé dans Docker (service canar dans un compose) → utiliser les noms de service Docker : `qdrant`, `postgres`

| Variable             | Description                                    | Exemple                                                                                            |
| -------------------- | ---------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| `LLM_API_BASE`       | URL de l’API du LLM                            | `https://api.mistral.ai/v1`                                                                        |
| `LLM_API_KEY`        | Clé API du LLM                                 | `sk-...`                                                                                           |
| `LLM_MODEL`          | Nom du modèle                                  | `mistral-medium`                                                                                   |
| `EMBED_API_BASE`     | URL de l’API embeddings                        | `https://api.mistral.ai/v1`                                                                        |
| `EMBED_API_KEY`      | Clé API embeddings                             | `sk-...`                                                                                           |
| `EMBED_MODEL`        | Nom du modèle d’embeddings                     | `mistral-embed`                                                                                    |
| `FASTEMBED_SPARSE_MODEL` | Modèle sparse optionnel                   | `Qdrant/bm25`                                                                                      |
| `QDRANT_URL`         | URL de Qdrant                                  | `http://localhost:6333` (host) ou `http://qdrant:6333` (docker)                                    |
| `QDRANT_API_KEY`     | Clé API Qdrant (si activée)                    | `...`                                                                                              |
| `QDRANT_COLLECTIONS` | Collections autorisées (séparées par virgules) | `col1,col2`                                                                                        |
| `QDRANT_SPARSE_VECTOR_NAME` | Nom optionnel du vecteur sparse Qdrant | laisser vide pour le vecteur sparse par défaut                                                     |
| `DB_POSTGRES_URL`    | URL Postgres (SQLAlchemy/psycopg)              | `postgresql+psycopg://canar:canar@localhost:5432/canar` (host) ou `...@postgres:5432/...` (docker) |

### 3) Installer et lancer CanaR

```shell
python -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install -e .
```
ou `make venv` pour créer l'environnement (python -m venv .venv) et `make install` pour charger l'environnement et installer CanaR.

Lancer l'application :
```shell
canar
```
ou `make run`

Si la commande `canar` n'est pas disponible, lancer Streamlit directement :
```shell
streamlit run canar/app/main.py --server.headless true --server.port 8501
```

## Pré-requis

- **Docker + Docker Compose** (Linux) ou **Docker Desktop** (Windows)
- **Python ≥ 3.10**

CanaR a besoin :
- d'une base vectorielle (RAG) : **Qdrant**
- d'une base relationnelle (comptes utilisateurs + historique) : **Postgres**

Le fichier docker-compose est fourni dans `infra/docker-compose.yml`

## Configuration

Exemple de `.env`:

```dotenv
# LLM (exemple)
LLM_API_BASE=https://url_llm/v1
LLM_API_KEY=
LLM_MODEL=nom_model

# Embeddings (exemple)
EMBED_API_BASE=https://url_embed/v1
EMBED_API_KEY=
EMBED_MODEL=nom_model
FASTEMBED_SPARSE_MODEL=Qdrant/bm25

# Qdrant
# QDRANT_URL=http://qdrant:6333        # si CanaR tourne dans Docker
QDRANT_URL=http://localhost:6333       # si CanaR tourne sur l'hôte
QDRANT_API_KEY=
QDRANT_COLLECTIONS=collection1_v1,collection2_v1
QDRANT_SPARSE_VECTOR_NAME=

# Postgres
# DB_POSTGRES_URL=postgresql+psycopg://canar:canar@postgres:5432/canar   # docker
DB_POSTGRES_URL=postgresql+psycopg://canar:canar@localhost:5432/canar    # host
```

## Contribuer

Consultez : [CONTRIBUTING.fr.md](CONTRIBUTING.fr.md)

## Licence

TBD (à confirmer par les mainteneurs).

## Problèmes courants

### Ports

Les ports suivants doivent être disponibles :

| Service           | Port par défaut |
| ----------------- | --------------- |
| Postgres          | `5432`          |
| Qdrant            | `6333`          |
| CanaR (Streamlit) | `8501`          |

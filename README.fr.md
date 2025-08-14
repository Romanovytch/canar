[🇫🇷 Français](README.fr.md) | [🇬🇧 English](README.md)

# Comment utiliser CanaR ?

## Initialisation de l'environnement virtuel

(Optionnel) C'est une bonne pratique, en python, d'utiliser un environnement virtuel :

```shell
cd CanaR
python -m venv .venv
source .venv/bin/activate
```

## Installer les dépendances

RAGnaR a un `requirements.txt` listant toutes les dépendances et leur version minimale ainsi qu'un `pyproject.toml` :

```shell
pip install -U pip
pip install -e .
```

Selon votre machine et votre réseau, cette opération peut prendre quelques minutes.

## Variables d'environnement

Vous pouvez spécifier les variables d'environnement dans un fichier `.env` à la racine du projet.
Le fichier `.env.example` vous donne un exemple de ce à quoi il devrait ressembler :

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
# liste des collections, séparées par des virgules
QDRANT_COLLECTIONS=collection1_v1, collection2_v1, collection3_v2

# DB
DB_POSTGRES_URL=postgresql+psycopg://user-name:password@base-url:5432/canar
```

## Utilisation

Pour lancer l'application CanaR :

```
canar
```


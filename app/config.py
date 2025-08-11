import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv(override=True)


@dataclass
class AppConfig:
    mistral_base: str = os.getenv("MISTRAL_API_BASE", "")
    mistral_key: str = os.getenv("MISTRAL_API_KEY", "")
    mistral_model: str = os.getenv("MISTRAL_MODEL", "mistralai/Mistral-Small-24B-Instruct-2501")

    embed_base: str = os.getenv("EMBED_API_BASE", "")
    embed_key: str = os.getenv("EMBED_API_KEY", "")
    embed_model: str = os.getenv("EMBED_MODEL", "BAAI/bge-multilingual-gemma2")

    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "")
    qdrant_collections: list[str] = tuple(
        c.strip() for c in os.getenv("QDRANT_COLLECTIONS", "utilitr_v1").split(",") if c.strip()
    )

    db_path: str = os.getenv("APP_DB", "data/app.db")

    def validate(self):
        assert len(self.qdrant_collections) >= 1, "QDRANT_COLLECTIONS cannot be empty"

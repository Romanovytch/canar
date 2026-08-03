import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv(override=True)


@dataclass
class AppConfig:
    llm_base: str = os.getenv("LLM_API_BASE", "")
    llm_key: str = os.getenv("LLM_API_KEY", "")
    llm_model: str = os.getenv("LLM_MODEL", "")

    embed_base: str = os.getenv("EMBED_API_BASE", "")
    embed_key: str = os.getenv("EMBED_API_KEY", "")
    embed_model: str = os.getenv("EMBED_MODEL", "")

    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "")
    qdrant_collections: list[str] = tuple(
        c.strip() for c in os.getenv("QDRANT_COLLECTIONS", "").split(",") if c.strip()
    )

    db_path: str = os.getenv("APP_DB", "data/app.db")

    chatbot_config_path: str = os.getenv(
        "CANAR_CHATBOTS_CONFIG", "canar/app/chatbots/chatbotconfig.yaml"
    )

    def validate(self):
        assert len(self.qdrant_collections) >= 1, "QDRANT_COLLECTIONS cannot be empty"

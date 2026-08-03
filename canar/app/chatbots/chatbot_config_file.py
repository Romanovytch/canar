import logging

from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger("ChatbotConfigFile")


class ChatbotYAMLConfig(BaseModel):
    # id avec des minuscules, chiffres et underscores uniquement
    id: str = Field(pattern=r"^[a-z0-9_]+$")
    # name et system_prompt ne peuvent pas être vides
    name: str = Field(min_length=1)
    description: str
    system_prompt: str = Field(min_length=1)
    collections: list[str] = Field(default_factory=list)
    accepted_file_types: list[str] = Field(default_factory=list)
    export_extension: str | None = None
    model: str = Field(default="gpt-4o")
    # top_k strictement positif
    top_k: int = Field(default=5, ge=1)
    # score_threshold entre 0.0 et 1.0
    score_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    # max_context_tokens cohérent avec le slider
    max_context_tokens: int = Field(default=2048, ge=256)
    allowed_tools: list[str] = Field(default_factory=list)

    @classmethod
    def createChatbot(cls, yaml_data: dict) -> "ChatbotYAMLConfig":
        """
        Crée et retourne une instance d'objet ChatbotYAMLConfig à partir des données YAML.
        La sauvegarde en base de données est géré séparément.
        """
        try:
            return cls.model_validate(yaml_data)
        except Exception as e:
            raise ValueError(f"Erreur de synthaxe YAML pour le chatbot {e}") from e

    @classmethod
    def checkChatbot(cls, yaml_data: dict) -> bool:
        """
        Utilise Pydantic pour valider la structure YAML avant toute création.
        Retourne True si les données respectent les règles, False sinon.
        """
        try:
            cls.model_validate(yaml_data)
            return True
        except ValidationError as e:
            logger.error(
                f"[Validation Échouée] Erreur pour le chatbot "
                f"'{yaml_data.get('id', 'inconnu')}' :\n{e}"
            )
            return False


class ChatbotConfigFile(BaseModel):
    chatbots: list[ChatbotYAMLConfig]

import logging
from pydantic import BaseModel, Field, model_validator

from canar.app.bot_tools.tool_config import ToolProvidersConfig
from canar.app.chatbots.chatbot_config_file import ChatbotYAMLConfig

logger = logging.getLogger("CanarConfigFile")


class CanarConfigFile(BaseModel):
    """
    Modèle racine représentant l'intégralité d'un fichier de configuration CanaR.
    Englobe les `tool_providers` et la liste des `chatbots`, et garantit la cohérence globale.
    """

    tool_providers: ToolProvidersConfig = Field(default_factory=ToolProvidersConfig)
    chatbots: list[ChatbotYAMLConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_cross_references(self) -> "CanarConfigFile":
        """
        Vérifie sans aucun accès réseau que tous les providers référencés dans
        `allowed_tools` de chaque chatbot existent bien dans `tool_providers`.
        """
        for bot in self.chatbots:
            allowed_providers = bot.get_allowed_provider_ids()
            for provider_id in allowed_providers:
                if not self.tool_providers.has_provider(provider_id):
                    mcp_keys = list(self.tool_providers.mcp.keys())
                    local_keys = list(self.tool_providers.local.keys())
                    available = mcp_keys + local_keys
                    raise ValueError(
                        f"[Erreur de Configuration] Le chatbot '{bot.id}' fait référence au provider "
                        f"inconnu '{provider_id}' dans ses 'allowed_tools'. "
                        f"Providers déclarés disponibles : {available}"
                    )
        return self

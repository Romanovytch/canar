import logging

from canar.app.bot_tools.base import BaseToolProvider
from canar.app.bot_tools.errors import ProviderUnavailable
from canar.app.bot_tools.local_provider import LocalPythonProvider
from canar.app.bot_tools.mcp_provider import MCPProvider
from canar.app.bot_tools.tool_config import (
    LocalProviderConfig,
    MCPProviderConfig,
)

logger = logging.getLogger("ToolProviderFactory")


class ToolProviderFactory:
    """
    Fabrique centralisant la création et l'instanciation des adaptateurs de providers d'outils
    (MCP, Python Local, etc.) à partir de leur modèle de configuration Pydantic.
    """

    @staticmethod
    def create_provider(
        provider_id: str,
        config: MCPProviderConfig | LocalProviderConfig,
    ) -> BaseToolProvider:
        """
        Instancie et retourne l'adaptateur de provider correspondant à la configuration fournie.
        """
        try:
            if not provider_id or not isinstance(provider_id, str):
                raise ValueError("L'identifiant du provider (provider_id) est invalide.")

            if isinstance(config, MCPProviderConfig):
                logger.info(f"Création du provider MCP '{provider_id}' ({config.url})...")
                return MCPProvider(
                    provider_id=provider_id,
                    server_url=config.url,
                    timeout_seconds=config.timeout_seconds,
                    read_timeout_seconds=config.read_timeout_seconds,
                    headers=config.headers,
                )

            elif isinstance(config, LocalProviderConfig):
                logger.info(f"Création du provider local '{provider_id}'...")
                return LocalPythonProvider(provider_id=provider_id)

            else:
                raise ValueError(
                    f"Type de configuration non pris en charge pour le provider "
                    f"'{provider_id}': {type(config).__name__}"
                )

        except Exception as e:
            logger.error(f"[Mode Dégradé] Échec de la création du provider '{provider_id}' : {e}")
            raise ProviderUnavailable(
                provider_id=provider_id
                if isinstance(provider_id, str) and provider_id
                else "inconnu",
                reason=f"Impossible d'instancier le provider : {e}",
            ) from e

import logging
from typing import Any

from canar.app.bot_tools.base import BaseToolProvider
from canar.app.bot_tools.errors import (
    ProviderUnavailable,
    ToolExecutionError,
    ToolNotFound,
)
from canar.app.bot_tools.models import ToolDefinition, ToolResult

logger = logging.getLogger("ToolRegistry")


class ToolRegistry:
    """
    Registre central gérant l'enregistrement des providers, la découverte des outils
    et le routage des appels vers le bon provider.
    """

    def __init__(self):
        self._providers: dict[str, BaseToolProvider] = {}

    def register_provider(self, provider_id: str, provider: BaseToolProvider) -> None:
        """Enregistre un fournisseur d'outils au registre."""
        if "__" in provider_id:
            raise ValueError(f"L'identifiant de provider '{provider_id}' ne doit pas contenir '__'.")

        self._providers[provider_id] = provider
        logger.info(f"Provider '{provider_id}' enregistré dans le registre.")

    async def initialize_all(self) -> None:
        """Initialise tous les providers enregistrés."""
        for pid, provider in self._providers.items():
            try:
                await provider.initialize()
                logger.info(f"Provider '{pid}' initialisé avec succès.")
            except Exception as e:
                logger.error(f"Échec de l'initialisation du provider '{pid}' : {e}")

    async def list_all_tools(self, allowed_providers: list[str] | None = None) -> list[ToolDefinition]:
        """
        Récupère la liste de tous les ToolDefinition enregistrés (filtrés optionnellement par provider).
        Applique la convention de nommage public : provider_id__tool_name.
        """
        all_definitions: list[ToolDefinition] = []

        for pid, provider in self._providers.items():
            if allowed_providers is not None and pid not in allowed_providers:
                continue

            try:
                raw_tools = await provider.list_tools()
                for tool in raw_tools:
                    # Garantir que le nom exposé est bien préfixé par provider_id__
                    full_name = tool.name if tool.name.startswith(f"{pid}__") else f"{pid}__{tool.name}"
                    prefixed_tool = ToolDefinition(
                        provider_id=pid,
                        name=full_name,
                        description=tool.description,
                        input_schema=tool.input_schema,
                    )
                    all_definitions.append(prefixed_tool)
            except Exception as e:
                logger.error(f"Erreur lors de la récupération des outils du provider '{pid}' : {e}")

        return all_definitions

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any], call_id: str | None = None) -> ToolResult:
        """
        Route un appel d'outil (format provider_id__tool_name) vers le bon provider.
        """
        if "__" not in tool_name:
            raise ToolNotFound(tool_name=tool_name)

        provider_id, original_tool_name = tool_name.split("__", 1)

        provider = self._providers.get(provider_id)
        if not provider:
            raise ProviderUnavailable(provider_id=provider_id, reason="Provider non trouvé dans le registre.")

        try:
            logger.info(f"Routage de '{tool_name}' vers le provider '{provider_id}'.")
            result = await provider.call_tool(original_tool_name, arguments)
            result.call_id = call_id
            return result
        except (ToolNotFound, ProviderUnavailable, Exception) as e:
            if not isinstance(e, (ToolNotFound, ProviderUnavailable)):
                raise ToolExecutionError(tool_name=tool_name, original_error=e) from e
            raise

    async def cleanup_all(self) -> None:
        """Ferme tous les providers enregistrés."""
        for pid, provider in self._providers.items():
            try:
                await provider.close()
            except Exception as e:
                logger.error(f"Erreur lors de la fermeture du provider '{pid}' : {e}")

import logging
from typing import Any

from canar.app.bot_tools.base import BaseToolProvider
from canar.app.bot_tools.errors import (
    ProviderUnavailable,
    ToolExecutionError,
    ToolNotFound,
)
from canar.app.bot_tools.models import ToolDefinition, ToolResult
from canar.app.chatbots.chatbot_config_file import AllowedToolConfig

logger = logging.getLogger("ToolRegistry")


class ToolRegistry:
    """
    Registre central gérant l'enregistrement des providers, la découverte des outils,
    le cache de découverte, le routage des appels et le filtrage des droits par chatbot.
    """

    def __init__(self):
        self._providers: dict[str, BaseToolProvider] = {}
        # Cache en mémoire des schémas découverts : provider_id -> list[ToolDefinition]
        self._tools_cache: dict[str, list[ToolDefinition]] = {}

    def register_provider(self, provider_id: str, provider: BaseToolProvider) -> None:
        """Enregistre un fournisseur d'outils au registre."""
        if "__" in provider_id:
            raise ValueError(
                f"L'identifiant de provider '{provider_id}' ne doit pas contenir '__'."
            )

        self._providers[provider_id] = provider
        logger.info(f"Provider '{provider_id}' enregistré dans le registre.")

    def invalidate_cache(self, provider_id: str | None = None) -> None:
        """
        Invalide le cache de découverte pour un provider spécifique ou pour tous les providers.
        """
        if provider_id:
            self._tools_cache.pop(provider_id, None)
            logger.info(f"Cache des outils invalidé pour le provider '{provider_id}'.")
        else:
            self._tools_cache.clear()
            logger.info("Cache de tous les outils invalidé.")

    async def initialize_all(self) -> None:
        """Initialise tous les providers enregistrés."""
        for pid, provider in self._providers.items():
            try:
                await provider.initialize()
                logger.info(f"Provider '{pid}' initialisé avec succès.")
            except Exception as e:
                logger.error(f"Échec de l'initialisation du provider '{pid}' : {e}")

    async def list_all_tools(
        self, allowed_providers: list[str] | None = None
    ) -> list[ToolDefinition]:
        """
        Récupère la liste de tous les ToolDefinition enregistrés
        (filtrés optionnellement par provider).
        Applique la mise en cache et la convention de nommage public : provider_id__tool_name.
        """
        all_definitions: list[ToolDefinition] = []

        for pid, provider in self._providers.items():
            if allowed_providers is not None and pid not in allowed_providers:
                continue

            # Vérifie si les outils de ce provider sont déjà en cache
            if pid in self._tools_cache:
                all_definitions.extend(self._tools_cache[pid])
                continue

            # Sinon, interroge le provider et met en cache
            try:
                raw_tools = await provider.list_tools()
                cached_tools: list[ToolDefinition] = []
                for tool in raw_tools:
                    # Garantir que le nom exposé est bien préfixé par provider_id__
                    full_name = (
                        tool.name if tool.name.startswith(f"{pid}__") else f"{pid}__{tool.name}"
                    )
                    prefixed_tool = ToolDefinition(
                        provider_id=pid,
                        name=full_name,
                        description=tool.description,
                        input_schema=tool.input_schema,
                    )
                    cached_tools.append(prefixed_tool)

                self._tools_cache[pid] = cached_tools
                all_definitions.extend(cached_tools)
            except Exception as e:
                logger.error(f"Erreur lors de la récupération des outils du provider '{pid}' : {e}")

        return all_definitions

    # --- MÉTHODES SPÉCIFIQUES DE SÉCURITÉ ET FILTRAGE PAR CHATBOT (Ticket III-3) ---

    async def list_tools_for_chatbot(
        self, allowed_tools: list[AllowedToolConfig | str]
    ) -> list[ToolDefinition]:
        """
        Retourne uniquement la liste des ToolDefinition autorisés pour un chatbot spécifique.
        """
        allowed_provider_ids = []
        allowlist_by_provider: dict[str, list[str]] = {}

        for item in allowed_tools:
            if isinstance(item, str):
                allowed_provider_ids.append(item)
                allowlist_by_provider[item] = []  # Liste vide = tous les outils du provider autorisés
            elif isinstance(item, AllowedToolConfig):
                allowed_provider_ids.append(item.provider)
                allowlist_by_provider[item.provider] = item.tools

        # Récupère la liste globale des outils pour les providers autorisés (avec gestion du cache)
        tools_from_providers = await self.list_all_tools(allowed_providers=allowed_provider_ids)

        filtered_tools = []
        for tool in tools_from_providers:
            pid = tool.provider_id
            specific_tools = allowlist_by_provider.get(pid, [])
            actual_tool_name = tool.name.replace(f"{pid}__", "")

            # Si aucune restriction spécifique d'outil, ou si l'outil est dans l'allowlist
            if not specific_tools or actual_tool_name in specific_tools or tool.name in specific_tools:
                filtered_tools.append(tool)

        return filtered_tools

    async def execute_tool_for_chatbot(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        allowed_tools: list[AllowedToolConfig | str],
        call_id: str | None = None,
    ) -> ToolResult:
        """
        Vérifie sans faille que l'outil est autorisé pour le chatbot courant avant de l'exécuter.
        Refuse et bloque l'exécution si l'outil n'est pas dans l'allowlist du chatbot.
        """
        available_tools = await self.list_tools_for_chatbot(allowed_tools)
        allowed_names = [t.name for t in available_tools]

        if tool_name not in allowed_names:
            logger.warning(
                f"[Sécurité Accès Refusé] Tentative d'exécution de l'outil non autorisé '{tool_name}'."
            )
            pid = tool_name.split("__")[0] if "__" in tool_name else None
            raise ToolNotFound(tool_name=tool_name, provider_id=pid)

        return await self.execute_tool(tool_name=tool_name, arguments=arguments, call_id=call_id)

    # --- MÉTHODES GLOBALES D'EXÉCUTION ET NETTOYAGE ---

    async def execute_tool(
        self, tool_name: str, arguments: dict[str, Any], call_id: str | None = None
    ) -> ToolResult:
        """
        Route un appel d'outil (format provider_id__tool_name) vers le bon provider.
        """
        if "__" not in tool_name:
            raise ToolNotFound(tool_name=tool_name)

        provider_id, original_tool_name = tool_name.split("__", 1)

        provider = self._providers.get(provider_id)
        if not provider:
            raise ProviderUnavailable(
                provider_id=provider_id, reason="Provider non trouvé dans le registre."
            )

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
        """Ferme tous les providers enregistrés et réinitialise le cache."""
        self.invalidate_cache()
        for pid, provider in self._providers.items():
            try:
                await provider.close()
            except Exception as e:
                logger.error(f"Erreur lors de la fermeture du provider '{pid}' : {e}")

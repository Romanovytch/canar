from abc import ABC, abstractmethod
from typing import Any

from canar.app.bot_tools.models import ToolDefinition, ToolResult


class BaseToolProvider(ABC):
    """
    Interface abstraite pour tous les fournisseurs d'outils de CanaR.
    Définition générique et indépendante de MCP, OpenAI et Streamlit.
    """

    provider_id: str

    async def initialize(self) -> None:  # noqa: B027
        """Prépare le provider si nécessaire (connexion réseau, etc.)."""
        pass

    @abstractmethod
    async def list_tools(self) -> list[ToolDefinition]:
        """Retourne la liste des outils disponibles sous forme de ToolDefinition."""
        pass

    @abstractmethod
    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """Exécute l'outil demandé et retourne un résultat normalisé ToolResult."""
        pass

    async def close(self) -> None:  # noqa: B027
        """Ferme proprement les connexions réseau ou sessions (no-op par défaut)."""
        pass

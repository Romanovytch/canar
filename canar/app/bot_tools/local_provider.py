import logging
from typing import Any

from canar.app.bot_tools.base import BaseToolProvider
from canar.app.bot_tools.errors import InvalidToolArguments, ToolNotFound
from canar.app.bot_tools.models import ToolDefinition, ToolResult

logger = logging.getLogger("LocalProvider")


class LocalPythonProvider(BaseToolProvider):
    def __init__(self, provider_id: str = "local"):
        self.provider_id = provider_id

    async def initialize(self):
        logger.info("LocalPythonProvider initialisé")

    async def list_tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                provider_id=self.provider_id,
                name="web_search",
                description=(
                    "Effectue une recherche sur internet pour trouver des informations "
                    "récentes. À utiliser quand tu as besoin de vérifier une actualité "
                    "ou une documentation publique."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "Les mots clés de la recherche (ex: 'Taux chômage France 2025')"
                            ),
                        }
                    },
                    "required": ["query"],
                },
            )
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        actual_name = name.replace(f"{self.provider_id}__", "")

        if actual_name == "web_search":
            query = arguments.get("query", "")
            if not query or not isinstance(query, str):
                raise InvalidToolArguments(
                    tool_name=name, reason="L'argument 'query' est requis et doit être une chaîne."
                )
            content = await self._do_web_search(query=query)
            return ToolResult(content=content, is_error=False)

        raise ToolNotFound(tool_name=name, provider_id=self.provider_id)

    async def _do_web_search(self, query: str) -> str:
        """Logique interne de l'outil web search"""
        logger.info(f"Recherche web lancée pour : {query}")

        try:
            from ddgs import DDGS

            result = DDGS().text(query, max_results=3)
            if not result:
                return f"Aucun résultat sur le web pour {query}."

            formated_results = []
            for r in result:
                formated_results.append(
                    f"Titre : {r['title']}\nLien : {r['href']}\nExtrait : {r['body']}"
                )

            return "\n\n---\n\n".join(formated_results)
        except Exception as e:
            return f"Erreur lors de la recherche web : {str(e)}"

    async def close(self):
        pass

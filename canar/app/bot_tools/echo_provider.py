from typing import Any

from canar.app.bot_tools.base import BaseToolProvider
from canar.app.bot_tools.errors import InvalidToolArguments, ToolNotFound
from canar.app.bot_tools.models import ToolDefinition, ToolResult


class EchoToolProvider(BaseToolProvider):
    """
    Provider factice pour tester et valider la chaîne de traitement des outils
    sans aucune dépendance réseau, MCP ou LLM.
    """

    def __init__(self, provider_id: str = "echo"):
        if "__" in provider_id:
            raise ValueError("Le 'provider_id' ne doit pas contenir le séparateur '__'")
        self.provider_id = provider_id

    async def list_tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                provider_id=self.provider_id,
                name="echo",
                description=(
                    "Répète le texte fourni en entrée. Utilisé pour les tests et diagnostics."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Le texte à répéter",
                        }
                    },
                    "required": ["text"],
                },
            )
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        # Nettoyage du préfixe éventuel s'il est fourni (ex: "echo__echo" -> "echo")
        actual_name = name.replace(f"{self.provider_id}__", "")

        if actual_name != "echo":
            raise ToolNotFound(tool_name=name, provider_id=self.provider_id)

        # Validation stricte des arguments
        if "text" not in arguments:
            raise InvalidToolArguments(
                tool_name=name, reason="L'argument requis 'text' est absent."
            )

        text_val = arguments["text"]
        if not isinstance(text_val, str):
            raise InvalidToolArguments(
                tool_name=name,
                reason=f"L'argument 'text' doit être une chaîne (reçu: {type(text_val).__name__}).",
            )

        return ToolResult(
            content=f"Echo: {text_val}",
            structured_content={"echo": text_val},
            is_error=False,
        )

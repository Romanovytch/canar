from typing import Any

from canar.app.bot_tools.models import ToolDefinition


def tool_definition_to_openai(tool: ToolDefinition) -> dict[str, Any]:
    """
    Convertit un ToolDefinition du domaine CanaR vers le schéma d'outil exigé par OpenAI.

    Format produit :
    {
        "type": "function",
        "function": {
            "name": "provider_id__tool_name",
            "description": "...",
            "parameters": {... input_schema ...}
        }
    }
    """
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.input_schema or {"type": "object", "properties": {}},
        },
    }


def tools_to_openai_schemas(tools: list[ToolDefinition]) -> list[dict[str, Any]]:
    """
    Convertit une liste de ToolDefinition en schémas d'outils compatibles avec l'API OpenAI.
    """
    return [tool_definition_to_openai(tool) for tool in tools]

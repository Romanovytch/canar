from typing import Any
from pydantic import BaseModel, Field


class ToolDefinition(BaseModel):
    """
    Définition agnostique d'un outil disponible.
    """
    provider_id: str
    name: str
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)


class ToolCall(BaseModel):
    """
    Représentation d'une demande d'exécution d'outil par un modèle ou un utilisateur.
    """
    call_id: str
    full_name: str  # ex. "datagouv__search" ou "echo__echo"
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """
    Résultat normalisé d'un appel d'outil.
    """
    call_id: str | None = None
    content: str
    structured_content: dict[str, Any] | list[Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    is_error: bool = False

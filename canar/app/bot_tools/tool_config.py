from pydantic import BaseModel, Field


class MCPProviderConfig(BaseModel):
    """Configuration d'un serveur distant MCP (Model Context Protocol)."""

    url: str
    transport: str = Field(default="streamable_http")
    timeout_seconds: int = Field(default=15, ge=1)
    read_timeout_seconds: int = Field(default=60, ge=1)
    headers: dict[str, str] = Field(default_factory=dict)
    description: str | None = None


class LocalProviderConfig(BaseModel):
    """Configuration d'un provider Python local."""

    description: str | None = None


class ToolProvidersConfig(BaseModel):
    """
    Section `tool_providers` sous-divisée par catégorie (mcp, local).
    """

    mcp: dict[str, MCPProviderConfig] = Field(default_factory=dict)
    local: dict[str, LocalProviderConfig] = Field(default_factory=dict)

    def has_provider(self, provider_id: str) -> bool:
        """Vérifie si un provider existe dans l'une des sous-sections (mcp ou local)."""
        return provider_id in self.mcp or provider_id in self.local

    def get_provider(self, provider_id: str) -> MCPProviderConfig | LocalProviderConfig | None:
        """Retourne la configuration d'un provider par son identifiant."""
        if provider_id in self.mcp:
            return self.mcp[provider_id]
        if provider_id in self.local:
            return self.local[provider_id]
        return None

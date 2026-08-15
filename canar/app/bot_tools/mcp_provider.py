import logging
from typing import Any

from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client

from canar.app.bot_tools.base import BaseToolProvider
from canar.app.bot_tools.models import ToolDefinition, ToolResult

logger = logging.getLogger("MCPProvider")


class MCPProvider(BaseToolProvider):
    def __init__(self, provider_id: str, server_url: str):
        self.provider_id = provider_id
        self.server_url = server_url
        self.session = None
        self._client_context = None

    async def initialize(self):
        logger.info(f"Connexion au MCP {self.provider_id} ({self.server_url})...")
        self._client_context = streamable_http_client(self.server_url)
        read_stream, write_stream = await self._client_context.__aenter__()

        self.session = ClientSession(read_stream, write_stream)
        await self.session.__aenter__()
        await self.session.initialize()

    async def list_tools(self) -> list[ToolDefinition]:
        if not self.session:
            logger.warning(
                f"Le provider MCP '{self.provider_id}' n'est pas connecté. Outils ignorés."
            )
            return []

        response = await self.session.list_tools()
        definitions = []

        for tool in response.tools:
            definitions.append(
                ToolDefinition(
                    provider_id=self.provider_id,
                    name=tool.name,
                    description=tool.description or "",
                    input_schema=tool.input_schema or {},
                )
            )

        return definitions

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        actual_name = name.replace(f"{self.provider_id}__", "")

        result = await self.session.call_tool(actual_name, arguments=arguments)
        extracted_text = [c.text for c in result if c.type == "text"]
        content = "\n".join(extracted_text)

        return ToolResult(content=content, is_error=False)

    async def close(self):
        if self.session:
            await self.session.__aexit__(None, None, None)
        if self._client_context:
            await self._client_context.__aexit__(None, None, None)

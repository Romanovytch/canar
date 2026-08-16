import logging
from typing import Any

from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client

from canar.app.bot_tools.base import BaseToolProvider
from canar.app.bot_tools.errors import ProviderUnavailable, ToolExecutionError
from canar.app.bot_tools.models import ToolDefinition, ToolResult

logger = logging.getLogger("MCPProvider")


class MCPProvider(BaseToolProvider):
    """
    Adaptateur de provider pour les serveurs MCP distants (Model Context Protocol)
    utilisant le transport Streamable HTTP.
    """

    def __init__(
        self,
        provider_id: str,
        server_url: str,
        timeout_seconds: float = 15.0,
        read_timeout_seconds: float = 60.0,
        headers: dict[str, str] | None = None,
    ):
        self.provider_id = provider_id
        self.server_url = server_url
        self.timeout_seconds = timeout_seconds
        self.read_timeout_seconds = read_timeout_seconds
        self.headers = headers or {}
        self.session: ClientSession | None = None
        self._client_context = None

    async def initialize(self):
        logger.info(f"Connexion au serveur MCP '{self.provider_id}' ({self.server_url})...")
        try:
            import httpx

            custom_timeout = httpx.Timeout(
                timeout=self.timeout_seconds,
                read=self.read_timeout_seconds,
            )

            # Masquage des headers sensibles pour les logs
            safe_headers = {
                k: ("***" if any(s in k.lower() for s in ["auth", "key", "token", "secret"]) else v)
                for k, v in self.headers.items()
            }
            logger.debug(f"Headers initialisés pour '{self.provider_id}': {safe_headers}")

            httpx_client = httpx.AsyncClient(
                timeout=custom_timeout,
                headers=self.headers,
            )

            self._client_context = streamable_http_client(
                self.server_url,
                http_client=httpx_client,
            )
            read_stream, write_stream = await self._client_context.__aenter__()

            self.session = ClientSession(read_stream, write_stream)
            await self.session.__aenter__()
            await self.session.initialize()
            logger.info(f"Session MCP '{self.provider_id}' connectée et initialisée avec succès.")

        except Exception as e:
            clean_err = str(e)
            for secret in self.headers.values():
                if secret and len(secret) > 3:
                    clean_err = clean_err.replace(secret, "***")
            logger.error(f"Échec de connexion au MCP '{self.provider_id}' : {clean_err}")
            raise ProviderUnavailable(provider_id=self.provider_id, reason=clean_err) from e

    async def list_tools(self) -> list[ToolDefinition]:
        if not self.session:
            logger.warning(
                f"Le provider MCP '{self.provider_id}' n'est pas connecté. Outils ignorés."
            )
            return []

        try:
            definitions: list[ToolDefinition] = []
            cursor: str | None = None

            # Boucle de pagination pour parcourir tous les curseurs `nextCursor`
            while True:
                if cursor:
                    response = await self.session.list_tools(cursor=cursor)
                else:
                    response = await self.session.list_tools()

                for tool in response.tools:
                    definitions.append(
                        ToolDefinition(
                            provider_id=self.provider_id,
                            name=tool.name,
                            description=tool.description or "",
                            input_schema=tool.input_schema or {},
                        )
                    )

                cursor = getattr(response, "nextCursor", None)
                if not cursor:
                    break

            logger.info(
                f"[MCP '{self.provider_id}'] {len(definitions)} outil(s) découvert(s) au total."
            )
            return definitions

        except Exception as e:
            clean_err = str(e)
            for secret in self.headers.values():
                if secret and len(secret) > 3:
                    clean_err = clean_err.replace(secret, "***")
            logger.error(
                f"Erreur lors de la découverte des outils sur MCP '{self.provider_id}' : {clean_err}"
            )
            raise ProviderUnavailable(provider_id=self.provider_id, reason=clean_err) from e

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        if not self.session:
            raise ProviderUnavailable(
                provider_id=self.provider_id, reason="Session MCP non initialisée."
            )

        actual_name = name.replace(f"{self.provider_id}__", "")

        try:
            raw_result = await self.session.call_tool(actual_name, arguments=arguments)

            text_parts = []
            structured_data = getattr(raw_result, "structuredContent", None)

            if hasattr(raw_result, "content") and raw_result.content:
                for item in raw_result.content:
                    if hasattr(item, "text") and item.text:
                        text_parts.append(item.text)
                    elif hasattr(item, "type"):
                        text_parts.append(f"[{item.type} content]")

            content_text = (
                "\n".join(text_parts) if text_parts else "Aucun contenu textuel renvoyé par l'outil."
            )
            is_error = bool(getattr(raw_result, "isError", False))

            return ToolResult(
                content=content_text,
                structured_content=structured_data,
                is_error=is_error,
            )

        except Exception as e:
            clean_err = str(e)
            for secret in self.headers.values():
                if secret and len(secret) > 3:
                    clean_err = clean_err.replace(secret, "***")
            logger.error(
                f"Erreur lors de l'exécution de l'outil '{name}' sur MCP '{self.provider_id}' : {clean_err}"
            )
            raise ToolExecutionError(tool_name=name, original_error=clean_err) from e

    async def close(self):
        if self.session:
            try:
                await self.session.__aexit__(None, None, None)
            except Exception:
                pass
            self.session = None

        if self._client_context:
            try:
                await self._client_context.__aexit__(None, None, None)
            except Exception:
                pass
            self._client_context = None

import asyncio
from unittest.mock import AsyncMock, MagicMock
import pytest

from canar.app.bot_tools.errors import ProviderUnavailable, ToolExecutionError
from canar.app.bot_tools.mcp_provider import MCPProvider


def test_mcp_provider_init_and_clean_headers():
    provider = MCPProvider(
        provider_id="datagouv",
        server_url="https://mcp.data.gouv.fr/mcp",
        timeout_seconds=10,
        headers={"Authorization": "Bearer secret_token"},
    )
    assert provider.provider_id == "datagouv"
    assert provider.headers["Authorization"] == "Bearer secret_token"


def test_mcp_provider_list_tools_with_cursor_pagination():
    provider = MCPProvider(provider_id="mock_mcp", server_url="https://fake.url")

    # Mock tool objects
    tool1 = MagicMock()
    tool1.name = "search"
    tool1.description = "Recherche"
    tool1.input_schema = {"type": "object"}

    tool2 = MagicMock()
    tool2.name = "details"
    tool2.description = "Détails"
    tool2.input_schema = {"type": "object"}

    # Mock page 1 response (with nextCursor)
    page1 = MagicMock()
    page1.tools = [tool1]
    page1.nextCursor = "cursor_page_2"

    # Mock page 2 response (no nextCursor)
    page2 = MagicMock()
    page2.tools = [tool2]
    page2.nextCursor = None

    mock_session = AsyncMock()
    mock_session.list_tools.side_effect = [page1, page2]
    provider.session = mock_session

    tools = asyncio.run(provider.list_tools())
    assert len(tools) == 2
    assert tools[0].name == "search"
    assert tools[1].name == "details"


def test_mcp_provider_call_tool_nominal_and_is_error():
    provider = MCPProvider(provider_id="mock_mcp", server_url="https://fake.url")

    # Content item
    item = MagicMock()
    item.type = "text"
    item.text = "Résultat factice"

    # Mock CallToolResult
    raw_result = MagicMock()
    raw_result.content = [item]
    raw_result.isError = False
    raw_result.structuredContent = {"status": "ok"}

    mock_session = AsyncMock()
    mock_session.call_tool.return_value = raw_result
    provider.session = mock_session

    res = asyncio.run(provider.call_tool("mock_mcp__search", {"query": "test"}))
    assert res.content == "Résultat factice"
    assert res.is_error is False
    assert res.structured_content == {"status": "ok"}


def test_mcp_provider_call_tool_handles_error():
    provider = MCPProvider(
        provider_id="mock_mcp",
        server_url="https://fake.url",
        headers={"Authorization": "super_secret"},
    )
    mock_session = AsyncMock()
    mock_session.call_tool.side_effect = Exception("Erreur réseau super_secret")
    provider.session = mock_session

    with pytest.raises(ToolExecutionError) as exc_info:
        asyncio.run(provider.call_tool("mock_mcp__search", {}))

    # Masquage du secret dans le message d'erreur
    assert "super_secret" not in str(exc_info.value)
    assert "***" in str(exc_info.value)

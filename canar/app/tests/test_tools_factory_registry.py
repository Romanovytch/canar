import asyncio

import pytest

from canar.app.bot_tools.echo_provider import EchoToolProvider
from canar.app.bot_tools.errors import ProviderUnavailable, ToolNotFound
from canar.app.bot_tools.factory import ToolProviderFactory
from canar.app.bot_tools.local_provider import LocalPythonProvider
from canar.app.bot_tools.mcp_provider import MCPProvider
from canar.app.bot_tools.registry import ToolRegistry
from canar.app.bot_tools.tool_config import LocalProviderConfig, MCPProviderConfig
from canar.app.chatbots.chatbot_config_file import AllowedToolConfig


def test_factory_create_mcp_provider():
    config = MCPProviderConfig(url="https://mcp.data.gouv.fr/mcp")
    provider = ToolProviderFactory.create_provider("datagouv", config)
    assert isinstance(provider, MCPProvider)
    assert provider.provider_id == "datagouv"


def test_factory_create_local_provider():
    config = LocalProviderConfig(description="Provider local")
    provider = ToolProviderFactory.create_provider("local", config)
    assert isinstance(provider, LocalPythonProvider)
    assert provider.provider_id == "local"


def test_factory_invalid_provider_id():
    config = LocalProviderConfig()
    with pytest.raises(ProviderUnavailable):
        ToolProviderFactory.create_provider("", config)


def test_registry_cache_and_invalidation():
    registry = ToolRegistry()
    echo_provider = EchoToolProvider(provider_id="echo")
    registry.register_provider("echo", echo_provider)

    # 1er appel : remplit le cache
    tools1 = asyncio.run(registry.list_all_tools())
    assert len(tools1) == 1
    assert "echo" in registry._tools_cache

    # 2nd appel : lit depuis le cache
    tools2 = asyncio.run(registry.list_all_tools())
    assert len(tools2) == 1

    # Invalidation du cache
    registry.invalidate_cache("echo")
    assert "echo" not in registry._tools_cache


def test_registry_chatbot_isolation_allowed():
    registry = ToolRegistry()
    echo_provider = EchoToolProvider(provider_id="echo")
    registry.register_provider("echo", echo_provider)

    allowed_tools = ["echo"]

    tools = asyncio.run(registry.list_tools_for_chatbot(allowed_tools))
    assert len(tools) == 1
    assert tools[0].name == "echo__echo"

    result = asyncio.run(
        registry.execute_tool_for_chatbot(
            tool_name="echo__echo",
            arguments={"text": "Bonjour Bot A!"},
            allowed_tools=allowed_tools,
            call_id="call_bot_a",
        )
    )
    assert result.content == "Echo: Bonjour Bot A!"


def test_registry_chatbot_isolation_denied():
    registry = ToolRegistry()
    echo_provider = EchoToolProvider(provider_id="echo")
    registry.register_provider("echo", echo_provider)

    allowed_tools = []

    tools = asyncio.run(registry.list_tools_for_chatbot(allowed_tools))
    assert len(tools) == 0

    with pytest.raises(ToolNotFound):
        asyncio.run(
            registry.execute_tool_for_chatbot(
                tool_name="echo__echo",
                arguments={"text": "Tentative non autorisée"},
                allowed_tools=allowed_tools,
            )
        )


def test_registry_chatbot_specific_tool_filter():
    registry = ToolRegistry()
    echo_provider = EchoToolProvider(provider_id="echo")
    registry.register_provider("echo", echo_provider)

    allowed_tools = [AllowedToolConfig(provider="echo", tools=["autre_outil"])]

    tools = asyncio.run(registry.list_tools_for_chatbot(allowed_tools))
    assert len(tools) == 0

    with pytest.raises(ToolNotFound):
        asyncio.run(
            registry.execute_tool_for_chatbot(
                tool_name="echo__echo",
                arguments={"text": "Tentative filtrée"},
                allowed_tools=allowed_tools,
            )
        )

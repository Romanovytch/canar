import asyncio

import pytest

from canar.app.bot_tools.echo_provider import EchoToolProvider
from canar.app.bot_tools.errors import (
    InvalidToolArguments,
    ProviderUnavailable,
    ToolNotFound,
)
from canar.app.bot_tools.registry import ToolRegistry


def test_echo_provider_list_tools():
    provider = EchoToolProvider(provider_id="echo")
    tools = asyncio.run(provider.list_tools())
    assert len(tools) == 1
    assert tools[0].name == "echo"
    assert tools[0].provider_id == "echo"
    assert "text" in tools[0].input_schema["properties"]


def test_echo_provider_nominal_call():
    provider = EchoToolProvider(provider_id="echo")
    result = asyncio.run(provider.call_tool("echo", {"text": "Bonjour CanaR!"}))
    assert result.is_error is False
    assert result.content == "Echo: Bonjour CanaR!"
    assert result.structured_content == {"echo": "Bonjour CanaR!"}


def test_echo_provider_missing_text_argument():
    provider = EchoToolProvider(provider_id="echo")
    with pytest.raises(InvalidToolArguments) as exc_info:
        asyncio.run(provider.call_tool("echo", {}))
    assert "absent" in str(exc_info.value)


def test_echo_provider_invalid_text_type():
    provider = EchoToolProvider(provider_id="echo")
    with pytest.raises(InvalidToolArguments) as exc_info:
        asyncio.run(provider.call_tool("echo", {"text": 12345}))
    assert "chaîne" in str(exc_info.value)


def test_echo_provider_unknown_tool():
    provider = EchoToolProvider(provider_id="echo")
    with pytest.raises(ToolNotFound):
        asyncio.run(provider.call_tool("unknown_tool", {"text": "test"}))


def test_registry_registration_and_routing():
    registry = ToolRegistry()
    echo_provider = EchoToolProvider(provider_id="echo")
    registry.register_provider("echo", echo_provider)

    tools = asyncio.run(registry.list_all_tools())
    assert len(tools) == 1
    assert tools[0].name == "echo__echo"

    result = asyncio.run(
        registry.execute_tool(
            tool_name="echo__echo",
            arguments={"text": "Test via Registry"},
            call_id="call_test_123",
        )
    )
    assert result.content == "Echo: Test via Registry"
    assert result.call_id == "call_test_123"


def test_registry_provider_unavailable():
    registry = ToolRegistry()
    with pytest.raises(ProviderUnavailable):
        asyncio.run(registry.execute_tool("inconnu__outil", {"text": "test"}))


def test_registry_invalid_provider_id_with_separator():
    registry = ToolRegistry()
    echo_provider = EchoToolProvider(provider_id="echo")
    with pytest.raises(ValueError) as exc_info:
        registry.register_provider("invalid__id", echo_provider)
    assert "ne doit pas contenir '__'" in str(exc_info.value)


def test_registry_cleanup():
    registry = ToolRegistry()
    echo_provider = EchoToolProvider(provider_id="echo")
    registry.register_provider("echo", echo_provider)
    asyncio.run(registry.cleanup_all())
    asyncio.run(registry.cleanup_all())  # Vérifie que close() répétitif ne crash pas

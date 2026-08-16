import asyncio
from unittest.mock import MagicMock
import pytest

from canar.app.bot_tools.echo_provider import EchoToolProvider
from canar.app.bot_tools.errors import ToolNotFound
from canar.app.bot_tools.registry import ToolRegistry
from canar.app.chatbots.chatbot_config_file import ToolPolicyConfig


def test_agentic_loop_one_tool_call():
    registry = ToolRegistry()
    echo_provider = EchoToolProvider(provider_id="echo")
    registry.register_provider("echo", echo_provider)

    allowed_tools = ["echo"]

    mock_tool_call = MagicMock()
    mock_tool_call.id = "call_abc123"
    mock_tool_call.function.name = "echo__echo"
    mock_tool_call.function.arguments = '{"text": "Bonjour du test"}'

    result = asyncio.run(
        registry.execute_tool_for_chatbot(
            tool_name=mock_tool_call.function.name,
            arguments={"text": "Bonjour du test"},
            allowed_tools=allowed_tools,
            call_id=mock_tool_call.id,
        )
    )

    assert result.content == "Echo: Bonjour du test"
    assert result.call_id == "call_abc123"
    assert result.is_error is False


def test_agentic_loop_denied_tool_call():
    registry = ToolRegistry()
    echo_provider = EchoToolProvider(provider_id="echo")
    registry.register_provider("echo", echo_provider)

    allowed_tools = []

    with pytest.raises(ToolNotFound):
        asyncio.run(
            registry.execute_tool_for_chatbot(
                tool_name="echo__echo",
                arguments={"text": "Non autorisé"},
                allowed_tools=allowed_tools,
                call_id="call_denied",
            )
        )


def test_agentic_loop_tool_policy_config():
    policy = ToolPolicyConfig(max_iterations=6, max_calls_per_turn=2)
    assert policy.max_iterations == 6
    assert policy.max_calls_per_turn == 2

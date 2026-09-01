import asyncio
from pathlib import Path

import pytest
import yaml

from canar.app.bot_tools.cli import run_call, run_list


@pytest.fixture
def cli_config_files(tmp_path: Path):
    """Génère deux fichiers YAML temporaires pour tester la CLI."""
    tools_data = {"tool_providers": {"local": {"echo": {"description": "Echo local provider"}}}}
    tools_path = tmp_path / "tools_config.yaml"
    with open(tools_path, "w", encoding="utf-8") as f:
        yaml.dump(tools_data, f)

    chatbot_data = {
        "chatbots": [
            {
                "id": "bot_cli_test",
                "name": "Bot CLI Test",
                "description": "Bot pour tester la CLI",
                "system_prompt": "...",
                "allowed_tools": ["echo"],
            }
        ]
    }
    chatbot_path = tmp_path / "chatbotconfig.yaml"
    with open(chatbot_path, "w", encoding="utf-8") as f:
        yaml.dump(chatbot_data, f)

    return tools_path, chatbot_path


def test_cli_list_nominal(cli_config_files, capsys):
    tools_path, chatbot_path = cli_config_files

    asyncio.run(
        run_list(
            chatbot_id="bot_cli_test",
            tools_yaml=str(tools_path),
            chatbot_yaml=str(chatbot_path),
        )
    )

    captured = capsys.readouterr()
    assert "Bot CLI Test" in captured.out
    assert "echo__echo" in captured.out


def test_cli_call_nominal(cli_config_files, capsys):
    tools_path, chatbot_path = cli_config_files

    asyncio.run(
        run_call(
            chatbot_id="bot_cli_test",
            tool_name="echo__echo",
            args_json='{"text": "Test via CLI"}',
            tools_yaml=str(tools_path),
            chatbot_yaml=str(chatbot_path),
        )
    )

    captured = capsys.readouterr()
    assert "Echo: Test via CLI" in captured.out


def test_cli_call_invalid_json(cli_config_files):
    tools_path, chatbot_path = cli_config_files

    with pytest.raises(SystemExit) as exc_info:
        asyncio.run(
            run_call(
                chatbot_id="bot_cli_test",
                tool_name="echo__echo",
                args_json="{invalid json",
                tools_yaml=str(tools_path),
                chatbot_yaml=str(chatbot_path),
            )
        )
    assert exc_info.value.code == 1


def test_cli_call_unknown_chatbot(cli_config_files):
    tools_path, chatbot_path = cli_config_files

    with pytest.raises(SystemExit) as exc_info:
        asyncio.run(
            run_call(
                chatbot_id="bot_inconnu",
                tool_name="echo__echo",
                args_json='{"text": "test"}',
                tools_yaml=str(tools_path),
                chatbot_yaml=str(chatbot_path),
            )
        )
    assert exc_info.value.code == 1

from pathlib import Path

import pytest
import yaml
from sqlmodel import Session, SQLModel, create_engine, select

from canar.app.bot_tools.tool_config import MCPProviderConfig, ToolProvidersConfig
from canar.app.chatbots.chatbot_config import ChatbotConfig
from canar.app.yaml_loader import load_bot_tools_on_boot, load_chatbot_on_boot

# définition des fixtures


@pytest.fixture
def mock_db():
    """"""
    engine = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(engine)

    class DummyDB:
        def __init__(self):
            self.engine = engine

    return DummyDB()


@pytest.fixture
def valid_yaml(tmp_path: Path):
    """Génère un vrai fichier YAML temporaire contenant 2 bots valides"""
    data = {
        "chatbots": [
            {"id": "bot_a", "name": "Bot A", "description": "...", "system_prompt": "..."},
            {"id": "bot_b", "name": "Bot B", "description": "...", "system_prompt": "..."},
        ]
    }

    filepath = tmp_path / "valid_bots.yaml"
    with open(filepath, "w", encoding="utf-8") as f:
        yaml.dump(data, f)

    return filepath


@pytest.fixture
def invalid_yaml(tmp_path: Path):
    """Génère un YAML avec un bot invalide"""
    data = {
        "chatbots": [
            {"id": "bot_parfait", "name": "Valid", "description": "...", "system_prompt": "..."},
            {
                "id": "bot_casse",
                "name": "Invalid",
                "description": "...",
                "system_prompt": "...",
                "top_k": -5,
            },
        ]
    }

    filepath = tmp_path / "mixed_bots.yaml"
    with open(filepath, "w", encoding="utf-8") as f:
        yaml.dump(data, f)
    return filepath


def test_loader_success(mock_db, valid_yaml):
    """"""
    load_chatbot_on_boot(valid_yaml, mock_db)

    with Session(mock_db.engine) as session:
        bots_in_db = session.exec(select(ChatbotConfig)).all()
        assert len(bots_in_db) == 2
        assert bots_in_db[0].id == "bot_a"
        assert bots_in_db[1].id == "bot_b"


def test_loader_rollback(mock_db, invalid_yaml):
    """"""
    load_chatbot_on_boot(invalid_yaml, mock_db)
    with Session(mock_db.engine) as session:
        bots_in_db = session.exec(select(ChatbotConfig)).all()
        assert len(bots_in_db) == 0


def test_loader_missing_file(mock_db, tmp_path: Path):
    """"""
    fake_path = tmp_path / "fichier_fantome.yaml"

    load_chatbot_on_boot(fake_path, mock_db)

    with Session(mock_db.engine) as session:
        bots_in_db = session.exec(select(ChatbotConfig)).all()
        assert len(bots_in_db) == 0


def test_loader_invalid_yaml_syntax(mock_db, tmp_path: Path):
    """"""
    filepath = tmp_path / "corrupted.yaml"
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("chatbots:\n- id: bot1\n  name: [mauvaise syntaxe")

    load_chatbot_on_boot(filepath, mock_db)

    with Session(mock_db.engine) as session:
        bots_in_db = session.exec(select(ChatbotConfig)).all()
        assert len(bots_in_db) == 0


# --- Nouveaux tests pour le Ticket III-2 ---


def test_load_bot_tools_on_boot_success(tmp_path: Path):
    """Vérifie le chargement de tools_config.yaml avec la structure Option A"""
    data = {
        "tool_providers": {
            "mcp": {
                "datagouv": {
                    "url": "https://mcp.data.gouv.fr/mcp",
                    "timeout_seconds": 15,
                }
            },
            "local": {"echo": {"description": "Provider echo local"}},
        }
    }
    filepath = tmp_path / "tools_config.yaml"
    with open(filepath, "w", encoding="utf-8") as f:
        yaml.dump(data, f)

    registry, providers_config = load_bot_tools_on_boot(filepath)

    assert providers_config.has_provider("datagouv") is True
    assert providers_config.has_provider("echo") is True
    assert isinstance(providers_config.mcp["datagouv"], MCPProviderConfig)


def test_loader_cross_validation_unknown_provider(mock_db, tmp_path: Path):
    """Vérifie que la validation croisée rejette un chatbot demandant un provider inconnu"""
    data = {
        "chatbots": [
            {
                "id": "bot_bad_tools",
                "name": "Bot Bad Tools",
                "description": "...",
                "system_prompt": "...",
                "allowed_tools": ["provider_inexistant"],
            }
        ]
    }
    filepath = tmp_path / "bot_bad_tools.yaml"
    with open(filepath, "w", encoding="utf-8") as f:
        yaml.dump(data, f)

    providers_config = ToolProvidersConfig()  # Aucun provider déclaré
    load_chatbot_on_boot(filepath, mock_db, providers_config=providers_config)

    with Session(mock_db.engine) as session:
        bots_in_db = session.exec(select(ChatbotConfig)).all()
        assert len(bots_in_db) == 0  # Rejeté par la validation croisée


def test_loader_cross_validation_success(mock_db, tmp_path: Path):
    """Vérifie que la validation croisée valide un chatbot référençant un provider existant"""
    data = {
        "chatbots": [
            {
                "id": "bot_good_tools",
                "name": "Bot Good Tools",
                "description": "...",
                "system_prompt": "...",
                "allowed_tools": [{"provider": "datagouv", "tools": ["search_datasets"]}],
                "tool_policy": {
                    "max_iterations": 5,
                    "max_calls_per_turn": 2,
                },
            }
        ]
    }
    filepath = tmp_path / "bot_good_tools.yaml"
    with open(filepath, "w", encoding="utf-8") as f:
        yaml.dump(data, f)

    providers_config = ToolProvidersConfig(
        mcp={"datagouv": MCPProviderConfig(url="https://mcp.data.gouv.fr/mcp")}
    )
    load_chatbot_on_boot(filepath, mock_db, providers_config=providers_config)

    with Session(mock_db.engine) as session:
        bots_in_db = session.exec(select(ChatbotConfig)).all()
        assert len(bots_in_db) == 1
        assert bots_in_db[0].id == "bot_good_tools"

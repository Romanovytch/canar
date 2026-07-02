import pytest
import yaml
from pathlib import Path
from sqlmodel import Session, create_engine, SQLModel, select

from canar.app.yaml_loader import load_chatbot_on_boot
from canar.app.chatbots.chatbot_config import ChatbotConfig


#définition des fixtures

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
def valid_yaml(tmp_path:Path):
    """Génère un vrai fichier YAML temporaire contenant 2 bots valides"""
    data = {
        "chatbots": [
            {"id": "bot_a", "name": "Bot A", "description": "...", "system_prompt": "..."},
            {"id": "bot_b", "name": "Bot B", "description": "...", "system_prompt": "..."}
        ]
    }

    filepath = tmp_path / "valid_bots.yaml"
    with open(filepath, "w", encoding="utf-8") as f:
        yaml.dump(data, f)

    return filepath

@pytest.fixture
def invalid_yaml(tmp_path:Path):
    """Génère une YAML avec un bot invalide"""
    data = {
        "chatbots": [
            {"id": "bot_parfait", "name": "Valid", "description": "...", "system_prompt": "..."},
            {"id": "bot_casse", "name": "Invalid", "description": "...", "system_prompt": "...", "top_k": -5}
        ]
    }
    
    filepath = tmp_path / "mixed_bots.yaml"
    with open(filepath, "w", encoding="utf-8") as f:
        yaml.dump(data, f)
    return filepath

def test_loader_success(mock_db, valid_yaml):
    """"""
    load_chatbot_on_boot(valid_yaml,mock_db)

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

def test_loader_missing_file(mock_db, tmp_path:Path):
    """"""
    fake_path = tmp_path / "fichier_fantome.yaml"

    load_chatbot_on_boot(fake_path, mock_db)

    with Session(mock_db.engine) as session:
        bots_in_db = session.exec(select(ChatbotConfig)).all()
        assert len(bots_in_db) == 0

def test_loader_invalid_yaml_syntax(mock_db, tmp_path:Path):
    """"""
    filepath = tmp_path / "corrupted.yaml"
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("chatbots:\n- id: bot1\n  name: [mauvaise syntaxe")
    
    load_chatbot_on_boot(filepath, mock_db)

    with Session(mock_db.engine) as session:
        bots_in_db = session.exec(select(ChatbotConfig)).all()
        assert len(bots_in_db) == 0
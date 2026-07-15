import pytest
from sqlmodel import Session, SQLModel, create_engine

from canar.app.chatbots.chatbot_config import ChatbotConfig

VALID_YAML_DATA = {
    "id": "test_bot_1",
    "name": "Bot de Test",
    "description": "Un bot pour les tests unitaires",
    "system_prompt": "Tu es un bot de test. Réponds brièvement.",
}


def test_valid_chatbot_creation():
    """
    Vérifie qu'un dictionnaire valide crée bien l'objet avec ces valeurs par défaut.
    """

    assert ChatbotConfig.checkChatbot(VALID_YAML_DATA) is True

    bot = ChatbotConfig.createChatbot(VALID_YAML_DATA)

    assert bot.id == "test_bot_1"
    assert bot.name == "Bot de Test"

    assert bot.model == "gpt-4o"
    assert bot.top_k == 5
    assert bot.score_threshold == 0.35
    assert bot.collections == []


def test_missing_mandatory_fields():
    """
    Vérifie que l'absence de champ obligatoire bloque la validation!
    """
    invalid_data = {"id": "bad_bot", "description": "Je n'ai pas de nom, ni de prompt"}

    assert ChatbotConfig.checkChatbot(invalid_data) is False


def test_invalid_top_k_constraint():
    """Vérifie que le top k est strictement positif"""
    invalid_data = VALID_YAML_DATA.copy()
    invalid_data["top_k"] = 0

    assert ChatbotConfig.checkChatbot(invalid_data) is False


def test_invalid_score_threshold_constraint():
    """Vérifie que le scrore score_threshold ne dépasse pas 1.0"""
    invalid_data = VALID_YAML_DATA.copy()
    invalid_data["score_threshold"] = 1.5

    assert ChatbotConfig.checkChatbot(invalid_data) is False


def test_invalid_max_context_token():
    """Vérifie la limite basse du context (>=256)"""
    invalid_data = VALID_YAML_DATA.copy()
    invalid_data["max_context_tokens"] = 100

    assert ChatbotConfig.checkChatbot(invalid_data) is False


@pytest.fixture
def mock_db():
    """"""
    engine = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(engine)

    class DummyDB:
        def __init__(self):
            self.engine = engine

    return DummyDB()


def test_save_chatbot_upsert(mock_db):
    """"""
    bot = ChatbotConfig.createChatbot(VALID_YAML_DATA)

    with Session(mock_db.engine) as session:
        bot.saveChatbot(session, True)
        db_bot = session.get(ChatbotConfig, "test_bot_1")

        assert db_bot is not None
        assert db_bot.name == "Bot de Test"

        bot.name = "Nom modifié"
        bot.saveChatbot(session, True)

        db_bot_updated = session.get(ChatbotConfig, "test_bot_1")

        assert db_bot_updated.name == "Nom modifié"


def test_delete_chatbot(mock_db):
    """"""
    bot = ChatbotConfig.createChatbot(VALID_YAML_DATA)

    with Session(mock_db.engine) as session:
        bot.saveChatbot(session, True)

        assert session.get(ChatbotConfig, "test_bot_1") is not None

        deleted = bot.deleteChatbot(session)

        assert deleted is True
        assert session.get(ChatbotConfig, "test_bot_1") is None


def test_delete_non_existent_chatbot(mock_db):
    """"""
    bot = ChatbotConfig.createChatbot(VALID_YAML_DATA)

    with Session(mock_db.engine) as session:
        deleted = bot.deleteChatbot(session)

        assert deleted is False

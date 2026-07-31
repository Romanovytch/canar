import logging
from pathlib import Path

import yaml
from pydantic import ValidationError
from sqlmodel import Session

from canar.app.chatbots.chatbot_config import ChatbotConfig
from canar.app.chatbots.chatbot_config_file import ChatbotConfigFile, ChatbotYAMLConfig

logger = logging.getLogger("YAMLoader")


def load_chatbot_on_boot(yaml_path: str | Path, db):
    """
    Lire le fichier YAML et orchestre le chargement des chatbots dans la base de données.
    """
    path = Path(yaml_path)

    if not path.exists():
        logger.warning(f"[Avertissement] Fichier de configuration YAML introuvable : {path}")
        return

    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        logger.exception(f"[Erreur Fatale] Syntaxe YAML invalide : {exc}")
        return
    except Exception as e:
        logger.exception(f"[Erreur] Une erreur s'est produite : {e}")

    # list_bot = data.get("chatbots", [])
    try:
        config_file = ChatbotConfigFile.model_validate(data)
    except ValidationError as e:
        logger.error(
            f"[Erreur de Configuration] Le fichier YAML est mal formaté :\n{e}"
            "\nCanaR a chargé son dernier état fonctionnel."
        )
        return

    with Session(db.engine) as session:
        try:
            for bot_yaml in config_file.chatbots:
                if ChatbotYAMLConfig.checkChatbot(bot_yaml.model_dump()):
                    bot_db = ChatbotConfig(**bot_yaml.model_dump())
                    success = bot_db.saveChatbot(session, False)

                    if not success:
                        raise ValueError(f"Le chatbot '{bot_db.id}' a echoué aux règles métiers.")

            session.commit()
            logger.info(
                f"[Succès] {len(config_file.chatbots)} chatbots chargés "
                "et validé dans la base de données."
            )
        except Exception as e:
            session.rollback()
            logger.exception(
                f"[Erreur Fatale] Chargement annulé pour tout les chatbots. Raison : {e}"
                "\nCanaR a chargé son dernier état fonctionnel."
            )

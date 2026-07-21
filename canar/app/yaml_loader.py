from pathlib import Path

import yaml
from sqlmodel import Session

from canar.app.chatbots.chatbot_config import ChatbotConfig


def load_chatbot_on_boot(yaml_path: str | Path, db):
    """
    Lire le fichier YAML et orchestre le chargement des chatbots dans la base de données.
    """
    path = Path(yaml_path)

    if not path.exists():
        print(f"[Avertissement] Fichier de configuration YAML introuvable : {path}")
        return

    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        print(f"[Erreur Fatale] Syntaxe YAML invalide : {exc}")
        return
    except Exception as e:
        print(f"[Erreur] Une erreur s'est produite : {e}")

    list_bot = data.get("chatbots", [])

    with Session(db.engine) as session:
        try:
            for bot_data in list_bot:
                if ChatbotConfig.createChatbot(bot_data):
                    bot = ChatbotConfig.createChatbot(bot_data)
                    success = bot.saveChatbot(session, False)

                    if not success:
                        raise ValueError(f"Le chatbot '{bot.id}' a echoué aux règles métiers.")

            session.commit()
            print(f"[Succès] {len(list_bot)} chatbots chargés et validé dans la base de données.")
        except Exception as e:
            session.rollback()
            print(f"[Erreur Fatale] Chargement annulé pour tout les chatbots. Raison : {e}")

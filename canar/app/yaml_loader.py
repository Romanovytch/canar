import logging
from pathlib import Path

import yaml
from pydantic import ValidationError
from sqlmodel import Session

from canar.app.bot_tools.local_provider import LocalPythonProvider
from canar.app.bot_tools.mcp_provider import MCPProvider
from canar.app.bot_tools.registry import ToolRegistry
from canar.app.bot_tools.tool_config import ToolProvidersConfig
from canar.app.chatbots.chatbot_config import ChatbotConfig
from canar.app.chatbots.chatbot_config_file import ChatbotConfigFile, ChatbotYAMLConfig

logger = logging.getLogger("YAMLoader")


def load_bot_tools_on_boot(
    yaml_path: str | Path,
) -> tuple[ToolRegistry, ToolProvidersConfig]:
    """
    Charge le fichier `tools_config.yaml`, instancie les providers locaux et MCP,
    et les enregistre dans le ToolRegistry.
    """
    registry = ToolRegistry()
    providers_config = ToolProvidersConfig()
    path = Path(yaml_path)

    if not path.exists():
        logger.warning(
            f"[Avertissement] Fichier de configuration des outils introuvable : {path}"
        )
        return registry, providers_config

    try:
        with open(path, encoding="utf-8") as f:
            raw_data = yaml.safe_load(f) or {}

        # Si les providers sont dans une section `tool_providers` ou à la racine
        tools_data = raw_data.get("tool_providers", raw_data)
        providers_config = ToolProvidersConfig.model_validate(tools_data)
    except ValidationError as e:
        logger.error(
            f"[Erreur de Configuration] Le fichier d'outils '{path}' est mal formaté :\n{e}"
        )
        return registry, providers_config
    except Exception as e:
        logger.error(f"Erreur lors du chargement de {yaml_path} : {e}")
        return registry, providers_config

    # Instanciation des providers locaux
    for pid, local_config in providers_config.local.items():
        provider = LocalPythonProvider(provider_id=pid)
        registry.register_provider(pid, provider)

    # Instanciation des providers MCP
    for pid, mcp_config in providers_config.mcp.items():
        provider = MCPProvider(provider_id=pid, server_url=mcp_config.url)
        registry.register_provider(pid, provider)

    return registry, providers_config


def load_chatbot_on_boot(
    yaml_path: str | Path, db, providers_config: ToolProvidersConfig | None = None
):
    """
    Lit le fichier `chatbotconfig.yaml`, valide les chatbots et effectue la validation croisée
    avec les `providers_config` déjà chargés.
    """
    path = Path(yaml_path)

    if not path.exists():
        logger.warning(
            f"[Avertissement] Fichier de configuration des chatbots introuvable : {path}"
        )
        return

    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception as e:
        logger.exception(
            f"[Erreur] Une erreur s'est produite lors de la lecture de {path} : {e}"
        )
        return

    try:
        config_file = ChatbotConfigFile.model_validate(data)
    except ValidationError as e:
        logger.error(
            f"[Erreur de Configuration] Le fichier YAML '{path}' est invalide :\n{e}"
            "\nCanaR a conservé son dernier état fonctionnel."
        )
        return

    # Validation croisée si providers_config est fourni
    if providers_config is not None:
        for bot in config_file.chatbots:
            for pid in bot.get_allowed_provider_ids():
                if not providers_config.has_provider(pid):
                    available = (
                        list(providers_config.mcp.keys())
                        + list(providers_config.local.keys())
                    )
                    logger.error(
                        f"[Erreur de Configuration] Le chatbot '{bot.id}' fait référence au provider "
                        f"inconnu '{pid}' dans ses 'allowed_tools'. Providers disponibles : {available}"
                    )
                    return

    with Session(db.engine) as session:
        try:
            for bot_yaml in config_file.chatbots:
                if ChatbotYAMLConfig.checkChatbot(bot_yaml.model_dump()):
                    bot_db = ChatbotConfig(**bot_yaml.model_dump())
                    success = bot_db.saveChatbot(session, False)

                    if not success:
                        raise ValueError(
                            f"Le chatbot '{bot_db.id}' a échoué aux règles métiers."
                        )

            session.commit()
            logger.info(
                f"[Succès] {len(config_file.chatbots)} chatbot(s) chargés "
                "et validés dans la base de données."
            )
        except Exception as e:
            session.rollback()
            logger.exception(
                f"[Erreur Fatale] Chargement annulé pour tous les chatbots. Raison : {e}"
                "\nCanaR a chargé son dernier état fonctionnel."
            )

import argparse
import asyncio
import json
import logging
import sys

from canar.app.bot_tools.errors import ToolError
from canar.app.chatbots.chatbot_config_file import ChatbotConfigFile
from canar.app.yaml_loader import load_bot_tools_on_boot

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("canar-tools")


async def run_list(chatbot_id: str, tools_yaml: str, chatbot_yaml: str):
    """Lister les outils autorisés pour un chatbot spécifié."""
    registry, providers_config = load_bot_tools_on_boot(tools_yaml)
    await registry.initialize_all()

    try:
        with open(chatbot_yaml, encoding="utf-8") as f:
            import yaml

            data = yaml.safe_load(f) or {}
            config = ChatbotConfigFile.model_validate(data)

        bot = next((b for b in config.chatbots if b.id == chatbot_id), None)
        if not bot:
            print(f"❌ Erreur : Chatbot '{chatbot_id}' introuvable dans {chatbot_yaml}")
            sys.exit(1)

        allowed_tools = bot.allowed_tools
        tools = await registry.list_tools_for_chatbot(allowed_tools)

        print(f"\n==================================================")
        print(f"🦆 CanaR Tools — Outils autorisés pour '{bot.name}' ({bot.id})")
        print(f"==================================================")
        print(f"Nombre d'outils disponibles : {len(tools)}\n")

        for tool in tools:
            print(f"• Nom public  : {tool.name}")
            print(f"  Provider    : {tool.provider_id}")
            print(f"  Description : {tool.description}")
            print(f"  Arguments   : {json.dumps(tool.input_schema, ensure_ascii=False, indent=2)}")
            print("-" * 50)

    except Exception as e:
        print(f"❌ Erreur de diagnostic : {e}")
        sys.exit(1)
    finally:
        await registry.cleanup_all()


async def run_call(
    chatbot_id: str, tool_name: str, args_json: str, tools_yaml: str, chatbot_yaml: str
):
    """Exécuter manuellement un outil pour un chatbot spécifié."""
    registry, providers_config = load_bot_tools_on_boot(tools_yaml)
    await registry.initialize_all()

    try:
        try:
            arguments = json.loads(args_json) if args_json else {}
        except json.JSONDecodeError as exc:
            print(f"❌ Erreur : Chaîne JSON d'arguments invalide : {exc}")
            sys.exit(1)

        with open(chatbot_yaml, encoding="utf-8") as f:
            import yaml

            data = yaml.safe_load(f) or {}
            config = ChatbotConfigFile.model_validate(data)

        bot = next((b for b in config.chatbots if b.id == chatbot_id), None)
        if not bot:
            print(f"❌ Erreur : Chatbot '{chatbot_id}' introuvable dans {chatbot_yaml}")
            sys.exit(1)

        print(f"\n🚀 Exécution de l'outil '{tool_name}' pour le chatbot '{bot.id}'...")
        result = await registry.execute_tool_for_chatbot(
            tool_name=tool_name,
            arguments=arguments,
            allowed_tools=bot.allowed_tools,
            call_id="cli_diagnostic_call",
        )

        print(f"\n==================================================")
        print(f"Résultat de l'exécution (is_error={result.is_error})")
        print(f"==================================================")
        print(result.content)
        if result.structured_content:
            print("\nDonnées structurées :")
            print(json.dumps(result.structured_content, ensure_ascii=False, indent=2))

        if result.is_error:
            sys.exit(1)

    except ToolError as e:
        print(f"\n❌ Échec de l'outil [{type(e).__name__}] : {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erreur inattendue : {e}")
        sys.exit(1)
    finally:
        await registry.cleanup_all()


def main():
    parser = argparse.ArgumentParser(
        prog="canar-tools",
        description="Outil de diagnostic CLI sans LLM ni Streamlit pour CanaR",
    )
    parser.add_argument(
        "--tools-config",
        default="canar/app/bot_tools/tools_config.yaml",
        help="Chemin du YAML des outils",
    )
    parser.add_argument(
        "--chatbot-config",
        default="canar/app/chatbots/chatbotconfig.yaml",
        help="Chemin du YAML des chatbots",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Commande list
    list_parser = subparsers.add_parser(
        "list", help="Lister les outils autorisés d'un chatbot"
    )
    list_parser.add_argument(
        "--chatbot",
        required=True,
        help="Identifiant du chatbot (ex: assistant_datagouv)",
    )

    # Commande call
    call_parser = subparsers.add_parser("call", help="Exécuter manuellement un outil")
    call_parser.add_argument(
        "--chatbot",
        required=True,
        help="Identifiant du chatbot (ex: assistant_datagouv)",
    )
    call_parser.add_argument(
        "--tool", required=True, help="Nom complet de l'outil (ex: echo__echo)"
    )
    call_parser.add_argument(
        "--arguments", default="{}", help="Arguments au format JSON string"
    )

    args = parser.parse_args()

    if args.command == "list":
        asyncio.run(run_list(args.chatbot, args.tools_config, args.chatbot_config))
    elif args.command == "call":
        asyncio.run(
            run_call(
                args.chatbot,
                args.tool,
                args.arguments,
                args.tools_config,
                args.chatbot_config,
            )
        )


if __name__ == "__main__":
    main()

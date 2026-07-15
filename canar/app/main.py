from __future__ import annotations

import asyncio
import json
import mimetypes
import time

import streamlit as st
from sqlmodel import Session, select

# from canar.app.agents import r_helpdesk, sas_to_r
from canar.app.api.embed_client import EmbedClient
from canar.app.api.llm_client import ChatClient
from canar.app.api.retrieval import search_qdrant
from canar.app.bot_tools.registry import ToolRegistry
from canar.app.chatbots.chatbot_config import ChatbotConfig
from canar.app.config import AppConfig
from canar.app.state import DB
from canar.app.ui.chat import render_messages, stream_answer
from canar.app.ui.sidebar import sidebar
from canar.app.utils.llm_utils import assemble_context, build_universal_messages
from canar.app.yaml_loader import load_bot_tools_on_boot, load_chatbot_on_boot

st.set_page_config(page_title="CanaR", page_icon="🦆", layout="wide")

cfg = AppConfig()
cfg.validate()
db = DB(cfg.db_path)


@st.cache_resource()
def init_app_agent_data(_db: DB):

    mimetypes.add_type("text/x-r-source", ".r")
    mimetypes.add_type("application/x-sas", ".sas")

    yaml_path = "canar/app/chatbots/chatbotconfig.yaml"
    load_chatbot_on_boot(yaml_path, _db)


@st.cache_resource()
def init_tool_registry() -> ToolRegistry:
    yaml_path = "canar/app/bot_tools/tools_config.yaml"
    return load_bot_tools_on_boot(yaml_path)


tool_registry = init_tool_registry()

init_app_agent_data(db)


@st.cache_data(ttl=3600)
def fetch_chatbot_list(_engine) -> list[ChatbotConfig]:
    with Session(_engine) as session:
        return session.exec(select(ChatbotConfig)).all()


chatbot_list = fetch_chatbot_list(db.engine)

# ---------- Auth (local) ----------
if "user_id" not in st.session_state:
    st.session_state["user_id"] = None


def show_auth():
    st.title("Connexion — CanaR")
    tab_login, tab_signup = st.tabs(["Se connecter", "Créer un compte"])

    with tab_login:
        u = st.text_input("IDEP", key="login_u")
        p = st.text_input("Mot de passe", type="password", key="login_p")
        if st.button("Connexion"):
            uid = db.verify_user(u, p)
            if uid is None:
                st.error("Identifiants invalides")
            else:
                st.session_state["user_id"] = uid
                # Create a starter conversation if none
                if not db.list_conversations(uid):
                    default_bot_id = chatbot_list[0].id if len(chatbot_list) > 0 else "no_bot"
                    cid = db.create_conversation(uid, "Nouvelle conversation modif", default_bot_id)
                    st.session_state["conv_id"] = cid
                    st.session_state["agent"] = default_bot_id

                st.rerun()

    with tab_signup:
        u2 = st.text_input("Nom d’utilisateur", key="signup_u")
        p2 = st.text_input("Mot de passe", type="password", key="signup_p")
        if st.button("Créer le compte"):
            try:
                uid = db.create_user(u2, p2)
                st.success("Compte créé. Connectez-vous.")
            except Exception as e:
                st.error(str(e))


if not st.session_state["user_id"]:
    show_auth()
    st.stop()

USER_ID = st.session_state["user_id"]

# Session defaults
if "conv_id" not in st.session_state:
    convs = db.list_conversations(USER_ID)
    if convs:
        st.session_state["conv_id"] = convs[0].id
        st.session_state["agent"] = convs[0].agent
    else:
        default_bot_id = chatbot_list[0].id if len(chatbot_list) > 0 else "no_bot"
        cid = db.create_conversation(USER_ID, "Nouvelle conversation", default_bot_id)
        st.session_state["conv_id"] = cid
        st.session_state["agent"] = default_bot_id

conv_id: int = st.session_state["conv_id"]
agent: str = st.session_state.get(
    "agent", (chatbot_list[0].id if len(chatbot_list) > 0 else "no_bot")
)

# Sidebar (conversations + create/rename/delete)

sidebar(db, USER_ID, conv_id, [bot.id for bot in chatbot_list], agent, chatbot_list)

# ---------- Header with current conversation name + agent selector ----------
AGENT_LABELS = {bot.id: bot.name for bot in chatbot_list}
ordered_agents = [bot.id for bot in chatbot_list]

conv = db.get_conversation(conv_id)
conv_title = conv.title if conv else "Nouvelle conversation"

# Fetch current user
user = db.get_user(USER_ID)
username = user.username if user else "?"

header_left, header_right = st.columns([1.8, 1], vertical_alignment="center")
with header_left:
    st.markdown(
        f"""
        <div class="canar-header">
          <span class="app-title">🦆 CanaR — Insee</span>
          <span class="sep">|</span>
          <span class="conv-title" title="{conv_title}">{conv_title}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with header_right:
    # User chip + logout
    ucol, lcol = st.columns([0.65, 0.35])
    with ucol:
        st.markdown(f"<div class='user-chip'>👤 {username}</div>", unsafe_allow_html=True)
    with lcol:
        if st.button("Déconnexion", use_container_width=True):
            for k in ("user_id", "conv_id", "agent"):
                st.session_state.pop(k, None)
            st.rerun()

    # Agent selector
    labels = [AGENT_LABELS[a] for a in ordered_agents]
    idx = ordered_agents.index(agent) if agent in ordered_agents else 0
    picked_label = st.selectbox("Agent", labels, index=idx)
    picked_agent = ordered_agents[labels.index(picked_label)]
    if picked_agent != agent:
        st.session_state["agent"] = picked_agent
        st.rerun()


# ---------- Controls (right) ----------
ctrl_left, ctrl_right = st.columns([2, 1])
with ctrl_right:
    max_tokens = st.slider(
        "Max tokens réponse",
        min_value=256,
        max_value=8192,
        value=2048,
        step=256,
        help="Augmente si tu colles de longs extraits de code.",
    )
    temperature = st.slider(
        "Température",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.05,
        help="Plus élevé = plus créatif.",
    )

st.markdown(
    """
    <style>
      /* Header layout: keep on one line when possible, with safe ellipsis */
      .canar-header {
        display: flex;
        align-items: baseline;
        gap: .4rem;
        flex-wrap: nowrap;
        width: 100%;
      }
      .canar-header .app-title {
        font-size: 2rem;     /* smaller than st.title */
        font-weight: 700;
        line-height: 1.3;
      }
      .canar-header .sep {
        color: rgba(0,0,0,.55);
      }
      .canar-header .conv-title {
        font-size: 1.5rem;      /* smaller than app title */
        font-style: italic;     /* more discrete */
        color: rgba(0,0,0,.6);  /* greyish */
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        flex: 1;                /* take remaining space and ellipsize */
        min-width: 0;           /* required for flex ellipsis in some browsers */
      }
      /* tighten spacing under header */
      div[data-baseweb="select"] { margin-top: .25rem; }


      .canar-header { display:flex; align-items:baseline; gap:.4rem; flex-wrap:nowrap; }
      .canar-header .app-title { font-size:1.25rem; font-weight:700; line-height:1.3; }
      .canar-header .sep { color:rgba(0,0,0,.55); }
      .canar-header .conv-title {
        font-size:1.0rem; font-style:italic; color:rgba(0,0,0,.6);
        white-space:nowrap; overflow:hidden; text-overflow:ellipsis; flex:1; min-width:0;
      }
      .user-chip {
        display:inline-flex; align-items:center; gap:.4rem;
        padding:.2rem .6rem; border:1px solid rgba(0,0,0,.15);
        border-radius:999px; font-size:.85rem; white-space:nowrap;
        background:rgba(0,0,0,.04);
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# LLM and Embedding clients
chat = ChatClient(cfg.llm_base, cfg.llm_key, cfg.llm_model)
embed = EmbedClient(cfg.embed_base, cfg.embed_model, cfg.embed_key)

# Show messages
render_messages(db, USER_ID, conv_id)

# --- Input area + turn handling ---
sas_code_uploaded = None
current_bot = next((bot for bot in chatbot_list if bot.id == st.session_state["agent"]), None)
if current_bot and current_bot.accepted_file_types:
    uploaded = st.file_uploader(
        "Uploader un fichier (optionnel)", type=current_bot.accepted_file_types
    )
    if uploaded is not None:
        sas_code_uploaded = uploaded.read().decode("utf-8", errors="ignore")

user_input = st.chat_input("Pose ta question (ou colle ton code)…")
if user_input:
    # 1) show the user message immediately in the chat
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2) persist it
    db.add_message(USER_ID, conv_id, "user", user_input)

    # Universal agent logic
    uploaded_file_cont = None
    if (
        current_bot.accepted_file_types
        and len(current_bot.accepted_file_types) > 0
        and sas_code_uploaded
    ):
        uploaded_file_cont = sas_code_uploaded

    context_text = None
    src_list = []
    if current_bot.collections and len(current_bot.collections) > 0:
        qvec = embed.embed_query(user_input)
        citations = search_qdrant(
            cfg.qdrant_url,
            cfg.qdrant_api_key,
            list(cfg.qdrant_collections),
            qvec,
            top_k_per_collection=5,
            source_filter="utilitr",
        )
        context_text, src_list = assemble_context(citations)

    messages = build_universal_messages(
        system_prompt=current_bot.system_prompt,
        user_question=user_input,
        file_content=uploaded_file_cont,
        rag_context=context_text,
    )

    allowed_tools = current_bot.allowed_tools

    allowed_tools_schemas = asyncio.run(tool_registry.get_all_tools_schemas(allowed_tools))

    if not allowed_tools or len(allowed_tools) == 0:
        gen = chat.stream_chat(messages, temperature=temperature, max_tokens=max_tokens)
        answer = stream_answer(db, USER_ID, conv_id, gen)
    else:
        is_final_answer = False
        final_text = ""

        MAX_ITERATIONS = 5
        iteration_count = 0

        with st.status("L'agent analyse la demande...", expanded=True) as status:
            while not is_final_answer and iteration_count < MAX_ITERATIONS:
                iteration_count += 1

                response = chat.sync_chat(
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    allowed_tools_schemas=allowed_tools_schemas,
                )

                llm_msg = response.choices[0].message

                if llm_msg.tool_calls:
                    messages.append(llm_msg.model_dump(exclude_none=True))
                    for tool_call in llm_msg.tool_calls:
                        tool_name = tool_call.function.name
                        st.write(f"Appel à l'outil {tool_name}...")
                        try:
                            tool_args = json.loads(tool_call.function.arguments)
                            result = asyncio.run(tool_registry.execute_tool(tool_name, tool_args))
                            st.write("Données récupérées avec succès")
                        except Exception as e:
                            result = f"Erreur lors de l'exécution de l'outil {tool_name} : {str(e)}"
                            st.write(f"Erreur : {result}")
                        messages.append(
                            {"role": "tool", "tool_call_id": tool_call.id, "content": str(result)}
                        )

                else:
                    is_final_answer = True
                    final_text = llm_msg.content
                    status.update(label="Réponse générée", state="complete", expanded=False)

            if not is_final_answer:
                final_text = (
                    "Désolé, j'ai dû interrompre mes recherches car elles prenaient "
                    "trop de temps ou tournaient en boucle. "
                    "Voici un résumé de ce que j'ai trouvé jusque-là..."
                )
                status.update(
                    label="Recherche interrompue (Limite atteinte)", state="error", expanded=False
                )

        def fake_stream_generator(text):
            for chunk in text.split(" "):
                yield chunk + " "
                time.sleep(0.01)

        gen = fake_stream_generator(final_text)
        answer = stream_answer(db, USER_ID, conv_id, gen)

    # Citations panel
    if len(src_list):
        with st.expander("Sources"):
            for src in src_list:
                st.markdown(f"""
                - **[{src["label"]}]** {src["section"]}  \n  
                {src["url"]}  \n  
                _({src["collection"]})_
                """)


# Footer / export for SAS→R
if current_bot.export_extension:
    msgs = db.get_messages(USER_ID, conv_id)
    if msgs and msgs[-1].role == "assistant":
        if st.button(f"Exporter la dernière réponse en .{current_bot.export_extension}"):
            content = msgs[-1].content
            # crude extract code block
            code = content
            if "```r" in content:
                code = content.split("```r", 1)[1].split("```", 1)[0]
            st.download_button(
                f"Télécharger .{current_bot.export_extension}",
                data=code.encode("utf-8"),
                file_name=f"export.{current_bot.export_extension}",
                mime=current_bot.exporte_mime_type,
            )

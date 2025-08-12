from __future__ import annotations
import streamlit as st
from app.config import AppConfig
from app.state import DB
from app.api.llm_client import ChatClient
from app.api.embed_client import EmbedClient
from app.api.retrieval import search_qdrant
from app.agents import sas_to_r, r_helpdesk
from app.ui.sidebar import sidebar
from app.ui.chat import render_messages, stream_answer

st.set_page_config(page_title="CanaR", page_icon="🦆", layout="wide")

cfg = AppConfig()
cfg.validate()
db = DB(cfg.db_path)

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
                    cid = db.create_conversation(uid, "Nouvelle conversation", "r_helpdesk")
                    st.session_state["conv_id"] = cid
                    st.session_state["agent"] = "r_helpdesk"
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
        cid = db.create_conversation(USER_ID, "Nouvelle conversation", "r_helpdesk")
        st.session_state["conv_id"] = cid
        st.session_state["agent"] = "r_helpdesk"

conv_id: int = st.session_state["conv_id"]
agent: str = st.session_state.get("agent", "r_helpdesk")

# Sidebar (conversations + create/rename/delete)
sidebar(db, USER_ID, conv_id, ["r_helpdesk", "sas_to_r"], agent)

# ---------- Header with current conversation name + agent selector ----------
AGENT_LABELS = {"r_helpdesk": "Assistant R", "sas_to_r": "Traduction SAS → R"}
ordered_agents = ["r_helpdesk", "sas_to_r"]

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
    labels = [AGENT_LABELS[a] for a in ordered_agents]
    idx = ordered_agents.index(agent) if agent in ordered_agents else 0
    picked_label = st.selectbox("Agent", labels, index=idx)
    picked_agent = ordered_agents[labels.index(picked_label)]
    if picked_agent != agent:
        st.session_state["agent"] = picked_agent
        st.rerun()

    # User chip + logout
    ucol, lcol = st.columns([0.65, 0.35])
    with ucol:
        st.markdown(f"<div class='user-chip'>👤 {username}</div>", unsafe_allow_html=True)
    with lcol:
        if st.button("Déconnexion", use_container_width=True):
            for k in ("user_id", "conv_id", "agent"):
                st.session_state.pop(k, None)
            st.rerun()

# ---------- Controls (right) ----------
ctrl_left, ctrl_right = st.columns([2, 1])
with ctrl_right:
    max_tokens = st.slider(
        "Max tokens réponse",
        min_value=256, max_value=8192, value=2048, step=256,
        help="Augmente si tu colles de longs extraits de code."
    )
    temperature = st.slider(
        "Température",
        min_value=0.0, max_value=1.0, value=0.2, step=0.05,
        help="Plus élevé = plus créatif."
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
chat = ChatClient(cfg.mistral_base, cfg.mistral_key, cfg.mistral_model)
embed = EmbedClient(cfg.embed_base, cfg.embed_model, cfg.embed_key)

# Show messages
render_messages(db, USER_ID, conv_id)

# --- Input area + turn handling ---
sas_code_uploaded = None
if st.session_state["agent"] == "sas_to_r":
    uploaded = st.file_uploader("Uploader un fichier .sas (optionnel)", type=["sas"])
    if uploaded is not None:
        sas_code_uploaded = uploaded.read().decode("utf-8", errors="ignore")

user_input = st.chat_input("Pose ta question (ou colle ton code)…")
if user_input:
    # 1) show the user message immediately in the chat
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2) persist it
    db.add_message(USER_ID, conv_id, "user", user_input)

    # 3) agent-specific logic
    if st.session_state["agent"] == "sas_to_r":
        messages = sas_to_r.build_messages(user_input, sas_code_uploaded)
        gen = chat.stream_chat(messages, temperature=temperature, max_tokens=max_tokens)
        _ = stream_answer(db, USER_ID, conv_id, gen)

    else:  # r_helpdesk
        qvec = embed.embed_query(user_input)
        citations = search_qdrant(
            cfg.qdrant_url, cfg.qdrant_api_key, list(cfg.qdrant_collections),
            qvec, top_k_per_collection=5, source_filter="utilitr"
        )
        messages, src_list = r_helpdesk.build_messages(user_input, citations)
        gen = chat.stream_chat(messages, temperature=temperature, max_tokens=max_tokens)
        answer = stream_answer(db, USER_ID, conv_id, gen)

        # Citations panel
        with st.expander("Sources"):
            for src in src_list:
                st.markdown(f"- **[{src['label']}]** {src['section']}  \n  {src['url']}  \n  _({src['collection']})_")

# Footer / export for SAS→R
if st.session_state["agent"] == "sas_to_r":
    msgs = db.get_messages(USER_ID, conv_id)
    if msgs and msgs[-1].role == "assistant":
        if st.button("Exporter la dernière réponse en .R"):
            content = msgs[-1].content
            # crude extract code block
            code = content
            if "```r" in content:
                code = content.split("```r", 1)[1].split("```", 1)[0]
            st.download_button("Télécharger .R", data=code.encode("utf-8"), file_name="translation.R", mime="text/x-r-source")

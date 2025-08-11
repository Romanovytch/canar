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

# Session defaults
if "conv_id" not in st.session_state:
    # create a starter conversation
    cid = db.create_conversation("Nouvelle conversation", "r_helpdesk")
    st.session_state["conv_id"] = cid
    st.session_state["agent"] = "r_helpdesk"

conv_id: int = st.session_state["conv_id"]
agent: str = st.session_state.get("agent", "r_helpdesk")

# Sidebar (conversations + create/rename/delete)
sidebar(db, conv_id, ["r_helpdesk", "sas_to_r"], agent)

# Top controls
left, right = st.columns([2, 1])
with left:
    st.title("🦆 CanaR — Insee")
    picked_agent = st.selectbox("Agent", ["r_helpdesk", "sas_to_r"],
                                index=0 if agent == "r_helpdesk" else 1)
    if picked_agent != agent:
        st.session_state["agent"] = picked_agent
        # Optionally create a new conv when switching; for now keep same
        # db.rename_conversation(conv_id, f"{picked_agent} — {db.get_messages(conv_id)[0].content[:20] if db.get_messages(conv_id) else 'session'}")
        st.rerun()

with right:
    max_tokens = st.slider("Max tokens réponse", min_value=256, max_value=8192,
                           value=2048, step=256)
    temperature = st.slider("Température", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

# LLM and Embedding clients
chat = ChatClient(cfg.mistral_base, cfg.mistral_key, cfg.mistral_model)
embed = EmbedClient(cfg.embed_base, cfg.embed_model, cfg.embed_key)

# Show messages
render_messages(db, conv_id)

# Input area
sas_code_uploaded = None
if st.session_state["agent"] == "sas_to_r":
    uploaded = st.file_uploader("Uploader un fichier .sas (optionnel)", type=["sas"])
    if uploaded is not None:
        sas_code_uploaded = uploaded.read().decode("utf-8", errors="ignore")

user_input = st.chat_input("Pose ta question (ou colle ton code)…")
if user_input:
    db.add_message(conv_id, "user", user_input)

    if st.session_state["agent"] == "sas_to_r":
        messages = sas_to_r.build_messages(user_input, sas_code_uploaded)
        gen = chat.stream_chat(messages, temperature=temperature, max_tokens=max_tokens)
        _ = stream_answer(db, conv_id, gen)

    else:  # r_helpdesk
        # Retrieval
        qvec = embed.embed_query(user_input)
        citations = search_qdrant(cfg.qdrant_url, cfg.qdrant_api_key, list(cfg.qdrant_collections),
                                  qvec, top_k_per_collection=5, source_filter="utilitr")
        messages, src_list = r_helpdesk.build_messages(user_input, citations)
        gen = chat.stream_chat(messages, temperature=temperature, max_tokens=max_tokens)
        answer = stream_answer(db, conv_id, gen)

        # Citations panel
        with st.expander("Sources"):
            for src in src_list:
                st.markdown(f"- **[{src['label']}]** {src['section']}  \n  {src['url']}  \n  _({src['collection']})_")

# Footer / export for SAS→R
if st.session_state["agent"] == "sas_to_r":
    msgs = db.get_messages(conv_id)
    if msgs and msgs[-1].role == "assistant":
        if st.button("Exporter la dernière réponse en .R"):
            content = msgs[-1].content
            # crude extract code block
            code = content
            if "```r" in content:
                code = content.split("```r", 1)[1].split("```", 1)[0]
            st.download_button("Télécharger .R", data=code.encode("utf-8"), file_name="translation.R", mime="text/x-r-source")

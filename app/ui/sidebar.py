from __future__ import annotations
import streamlit as st
from typing import Optional
from ..state import DB


def sidebar(db: DB, current_conv_id: Optional[int], agent_options: list[str], current_agent: str):
    st.sidebar.header("Conversations")
    convs = db.list_conversations()

    # Create new
    with st.sidebar.expander("Nouvelle conversation"):
        default_title = st.text_input("Titre", value="Nouvelle conversation")
        agent = st.selectbox("Agent", agent_options, index=agent_options.index(current_agent) if current_agent in agent_options else 0)
        if st.button("Créer", use_container_width=True):
            cid = db.create_conversation(default_title, agent)
            st.session_state["conv_id"] = cid
            st.session_state["agent"] = agent
            st.rerun()

    # List
    for c in convs:
        selected = (c.id == current_conv_id)
        cols = st.sidebar.columns([1, 5])
        if cols[0].button("▶" if not selected else "●", key=f"sel{c.id}", help="Ouvrir"):
            st.session_state["conv_id"] = c.id
            st.session_state["agent"] = c.agent
            st.rerun()
        with cols[1]:
            st.write(f"**{c.title}**  \n_{c.agent}_")
            new_name = st.text_input("Renommer", value=c.title, key=f"rn{c.id}")
            subcols = st.columns(2)
            if subcols[0].button("Renommer", key=f"btnrn{c.id}"):
                db.rename_conversation(c.id, new_name)
                st.rerun()
            if subcols[1].button("Supprimer", key=f"btndel{c.id}"):
                db.delete_conversation(c.id)
                if current_conv_id == c.id:
                    st.session_state.pop("conv_id", None)
                st.rerun()

    st.sidebar.divider()

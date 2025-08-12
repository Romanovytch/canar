from __future__ import annotations
import streamlit as st
from typing import Optional
from ..state import DB

AGENT_LABELS = {
    "r_helpdesk": "Assistant R",
    "sas_to_r": "Traduction SAS → R",
}

def sidebar(db: DB, current_conv_id: Optional[int], agent_options: list[str], current_agent: str):
    st.sidebar.header("Conversations")

    # Create new (compact)
    with st.sidebar.expander("Nouvelle conversation", expanded=False):
        default_title = st.text_input("Titre", value="Nouvelle conversation", key="new_title")
        labels = [AGENT_LABELS[a] for a in agent_options]
        idx = agent_options.index(current_agent) if current_agent in agent_options else 0
        picked_label = st.selectbox("Agent", labels, index=idx, key="new_agent_label")
        agent = agent_options[labels.index(picked_label)]
        if st.button("Créer", use_container_width=True, key="create_conv"):
            cid = db.create_conversation(default_title, agent)
            st.session_state["conv_id"] = cid
            st.session_state["agent"] = agent
            st.rerun()

    st.sidebar.markdown("---")

    # Conversation list
    convs = db.list_conversations()
    for c in convs:
        block = st.sidebar.container()

        # TOP ROW: title + ellipsis (same height, aligned)
        top = block.container()
        col_name, col_menu = top.columns([10, 2], vertical_alignment="center")

        if col_name.button(c.title, key=f"open_{c.id}", use_container_width=True):
            st.session_state["conv_id"] = c.id
            st.session_state["agent"] = c.agent
            st.rerun()

        with col_menu.popover("⋯", use_container_width=False):
            st.caption("Options")
            new_name = st.text_input("Renommer", value=c.title, key=f"rn_{c.id}")
            a, b = st.columns(2)
            if a.button("📝 Renommer", key=f"do_rn_{c.id}"):
                db.rename_conversation(c.id, new_name); st.rerun()
            if b.button("🗑️ Supprimer", key=f"do_del_{c.id}"):
                db.delete_conversation(c.id)
                if current_conv_id == c.id:
                    st.session_state.pop("conv_id", None)
                st.rerun()

        # SECOND ROW: agent label inside the "name block" (left col), smaller + italic
        row2_left, row2_right = block.columns([10, 2], vertical_alignment="center")
        label = AGENT_LABELS.get(c.agent, c.agent)
        row2_left.markdown(f"<div class='agent-label'>{label}</div>", unsafe_allow_html=True)
        row2_right.empty()  # keep grid alignment so the ⋯ stays aligned above

    # CSS: compact title buttons; ensure ⋯ fits; glue label to its title
    st.sidebar.markdown(
        """
        <style>
        /* Title buttons (uniform height) */
        section[data-testid="stSidebar"] .stButton > button {
            padding: 0.35rem 0.55rem;
            height: 2.0rem;
            line-height: 1.1;
        }
        /* Popover trigger ("⋯"): enough width for dots + caret */
        section[data-testid="stSidebar"] div[data-testid="stPopover"] > div > div > button {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            height: 2.0rem !important;
            min-width: 2.9rem !important;     /* wide enough for ⋯ + caret */
            padding: 0.2rem 0.8rem !important;
            line-height: 1.0 !important;
        }
        /* Keep the trigger aligned right & vertically centered in its column */
        section[data-testid="stSidebar"] div[data-testid="column"]:has(> div > div[data-testid="stPopover"]) {
            display: flex; justify-content: flex-end; align-items: center;
        }

        /* Agent label sits INSIDE the conversation block, just under the title */
        section[data-testid="stSidebar"] .agent-label {
            font-size: 0.8rem;        /* smaller than title */
            font-style: italic;       /* more discrete */
            opacity: 0.85;            /* keep your current color, just softer */
            margin-top: -0.25rem;     /* tuck closer to the title button */
            margin-left: 0.15rem;     /* slight indent under the button text */
            margin-bottom: 1.65rem;   /* extra gap before the next conversation */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
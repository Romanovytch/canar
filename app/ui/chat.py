from __future__ import annotations
import streamlit as st
from ..state import DB


def render_messages(db: DB, conv_id: int):
    msgs = db.get_messages(conv_id)
    for m in msgs:
        with st.chat_message(m.role):
            st.markdown(m.content)


def stream_answer(db: DB, conv_id: int, generator):
    with st.chat_message("assistant"):
        ph = st.empty()
        acc = ""
        for token in generator:
            acc += token
            ph.markdown(acc)
        full_text = acc
    db.add_message(conv_id, "assistant", full_text)
    return full_text

from __future__ import annotations
import os
import datetime as dt
from typing import Optional
from sqlmodel import SQLModel, Field, Session, create_engine, select


class Conversation(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    title: str
    agent: str  # "sas_to_r" | "r_helpdesk"
    created_at: dt.datetime = Field(default_factory=lambda: dt.datetime.utcnow())
    updated_at: dt.datetime = Field(default_factory=lambda: dt.datetime.utcnow())


class Message(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    conversation_id: int = Field(index=True)
    role: str  # "user" | "assistant" | "system"
    content: str
    created_at: dt.datetime = Field(default_factory=lambda: dt.datetime.utcnow())


class DB:
    def __init__(self, db_path: str):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.engine = create_engine(f"sqlite:///{db_path}", echo=False)
        SQLModel.metadata.create_all(self.engine)

    def create_conversation(self, title: str, agent: str) -> int:
        with Session(self.engine) as s:
            conv = Conversation(title=title, agent=agent)
            s.add(conv)
            s.commit()
            s.refresh(conv)
            return conv.id

    def list_conversations(self) -> list[Conversation]:
        with Session(self.engine) as s:
            return list(s.exec(select(Conversation).order_by(Conversation.updated_at.desc())))

    def rename_conversation(self, conv_id: int, new_title: str):
        with Session(self.engine) as s:
            c = s.get(Conversation, conv_id)
            if not c:
                return
            c.title = new_title
            c.updated_at = dt.datetime.utcnow()
            s.add(c)
            s.commit()

    def delete_conversation(self, conv_id: int):
        with Session(self.engine) as s:
            s.exec(f"DELETE FROM message WHERE conversation_id={conv_id}")
            s.exec(f"DELETE FROM conversation WHERE id={conv_id}")
            s.commit()

    def add_message(self, conv_id: int, role: str, content: str) -> int:
        with Session(self.engine) as s:
            m = Message(conversation_id=conv_id, role=role, content=content)
            s.add(m)
            c = s.get(Conversation, conv_id)
            if c:
                c.updated_at = dt.datetime.utcnow()
                s.add(c)
            s.commit()
            s.refresh(m)
            return m.id

    def get_messages(self, conv_id: int) -> list[Message]:
        with Session(self.engine) as s:
            return list(s.exec(select(Message).where(
                Message.conversation_id == conv_id).order_by(Message.id)))

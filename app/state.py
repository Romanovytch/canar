from __future__ import annotations
import os
import datetime as dt
from typing import Optional, List
from sqlmodel import SQLModel, Field, Session, create_engine, select, delete
from passlib.hash import bcrypt


# ---------- Models ----------
class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    username: str = Field(index=True, unique=True)
    password_hash: str
    created_at: dt.datetime = Field(default_factory=lambda: dt.datetime.utcnow())


class Conversation(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(index=True, foreign_key="user.id")
    title: str
    agent: str  # "sas_to_r" | "r_helpdesk"
    created_at: dt.datetime = Field(default_factory=lambda: dt.datetime.utcnow())
    updated_at: dt.datetime = Field(default_factory=lambda: dt.datetime.utcnow())


class Message(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    conversation_id: int = Field(index=True, foreign_key="conversation.id")
    role: str  # "user" | "assistant" | "system"
    content: str
    created_at: dt.datetime = Field(default_factory=lambda: dt.datetime.utcnow())


# ---------- DB ----------
class DB:
    def __init__(self, sqlite_path: str):
        db_url = os.getenv("DB_POSTGRES_URL", "").strip()
        if db_url:
            # Postgres with a small pool for concurrent users
            self.engine = create_engine(db_url, echo=False, pool_size=5,
                                        max_overflow=10, pool_pre_ping=True)
        else:
            os.makedirs(os.path.dirname(sqlite_path), exist_ok=True)
            self.engine = create_engine(f"sqlite:///{sqlite_path}", echo=False)
        SQLModel.metadata.create_all(self.engine)

    # ----- Users -----
    def create_user(self, username: str, password: str) -> int:
        with Session(self.engine) as s:
            existing = s.exec(select(User).where(User.username == username)).first()
            if existing:
                raise ValueError("Nom d’utilisateur déjà pris.")
            u = User(username=username, password_hash=bcrypt.hash(password))
            s.add(u)
            s.commit()
            s.refresh(u)
            return u.id

    def verify_user(self, username: str, password: str) -> Optional[int]:
        with Session(self.engine) as s:
            u = s.exec(select(User).where(User.username == username)).first()
            if not u or not bcrypt.verify(password, u.password_hash):
                return None
            return u.id

    def get_user(self, user_id: int) -> Optional[User]:
        with Session(self.engine) as s:
            return s.get(User, user_id)

    # ----- Conversations -----
    def create_conversation(self, user_id: int, title: str, agent: str) -> int:
        with Session(self.engine) as s:
            conv = Conversation(user_id=user_id, title=title, agent=agent)
            s.add(conv)
            s.commit()
            s.refresh(conv)
            return conv.id

    def get_conversation(self, conv_id: int) -> Optional[Conversation]:
        with Session(self.engine) as s:
            return s.get(Conversation, conv_id)

    def list_conversations(self, user_id: int) -> List[Conversation]:
        with Session(self.engine) as s:
            stmt = (select(Conversation)
                    .where(Conversation.user_id == user_id)
                    .order_by(Conversation.updated_at.desc()))
            return list(s.exec(stmt))

    def rename_conversation(self, user_id: int, conv_id: int, new_title: str):
        with Session(self.engine) as s:
            c = s.get(Conversation, conv_id)
            if not c or c.user_id != user_id:
                return
            c.title = new_title
            c.updated_at = dt.datetime.utcnow()
            s.add(c)
            s.commit()

    def delete_conversation(self, user_id: int, conv_id: int):
        with Session(self.engine) as s:
            c = s.get(Conversation, conv_id)
            if not c or c.user_id != user_id:
                return
            s.exec(delete(Message).where(Message.conversation_id == conv_id))
            s.exec(delete(Conversation).where(Conversation.id == conv_id))
            s.commit()

    # ----- Messages -----
    def add_message(self, user_id: int, conv_id: int, role: str, content: str) -> int:
        with Session(self.engine) as s:
            c = s.get(Conversation, conv_id)
            if not c or c.user_id != user_id:
                raise PermissionError("Conversation introuvable ou non propriétaire.")
            m = Message(conversation_id=conv_id, role=role, content=content)
            s.add(m)
            c.updated_at = dt.datetime.utcnow()
            s.add(c)
            s.commit()
            s.refresh(m)
            return m.id

    def get_messages(self, user_id: int, conv_id: int) -> List[Message]:
        with Session(self.engine) as s:
            c = s.get(Conversation, conv_id)
            if not c or c.user_id != user_id:
                return []
            stmt = select(Message).where(Message.conversation_id == conv_id).order_by(Message.id)
            return list(s.exec(stmt))

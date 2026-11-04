from __future__ import annotations

from canar.app.state import DB


def test_records_are_saved_with_timezone_aware_timestamps(tmp_path, monkeypatch):
    monkeypatch.delenv("DB_POSTGRES_URL", raising=False)
    db = DB(str(tmp_path / "app.db"))

    user_id = db.create_user("alice", "secret")
    conv_id = db.create_conversation(user_id, "Nouvelle conversation", "r_helpdesk")
    db.add_message(user_id, conv_id, "user", "Bonjour")
    db.rename_conversation(user_id, conv_id, "Renommée")

    conv = db.get_conversation(conv_id)
    assert conv.title == "Renommée"
    assert [m.content for m in db.get_messages(user_id, conv_id)] == ["Bonjour"]

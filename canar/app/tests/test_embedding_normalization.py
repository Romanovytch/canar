from __future__ import annotations

from unittest.mock import Mock

from canar.app.api.embed_client import EmbedClient
from canar.app.api.text_normalization import normalize_for_embedding


# APOSTROPHE NORMALIZATION WORKAROUND regression coverage. These tests can be
# removed together with the tagged production workaround.
def test_normalize_for_embedding_replaces_apostrophe_variants():
    text = "l‘un l’autre lʼenquêté l＇utilisateur"

    assert normalize_for_embedding(text) == "l'un l'autre l'enquêté l'utilisateur"


def test_normalize_for_embedding_preserves_accents_and_ascii_text():
    text = "La question est considérée comme sensible."

    assert normalize_for_embedding(text) == text


def test_embed_client_normalizes_only_the_request_payload(monkeypatch):
    response = Mock()
    response.json.return_value = {"data": [{"embedding": [3.0, 4.0]}]}
    post = Mock(return_value=response)
    monkeypatch.setattr("canar.app.api.embed_client.requests.post", post)
    client = EmbedClient("http://embedding.test/v1", "bge-m3")
    question = "Pourquoi répondre à l’enquêté ?"

    client.embed_query(question)

    assert question == "Pourquoi répondre à l’enquêté ?"
    assert post.call_args.kwargs["json"] == {
        "model": "bge-m3",
        "input": ["Pourquoi répondre à l'enquêté ?"],
    }

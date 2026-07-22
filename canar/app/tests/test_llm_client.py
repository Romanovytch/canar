from __future__ import annotations

from types import SimpleNamespace

from canar.app.api import llm_client


class FakeCompletions:
    def __init__(self):
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs["stream"]:
            delta = SimpleNamespace(content="réponse")
            return [SimpleNamespace(choices=[SimpleNamespace(delta=delta)])]
        return SimpleNamespace(choices=[])


class FakeOpenAI:
    instance = None

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.chat = SimpleNamespace(completions=FakeCompletions())
        FakeOpenAI.instance = self


def test_stream_and_sync_requests_preserve_extra_body(monkeypatch):
    monkeypatch.setattr(llm_client, "OpenAI", FakeOpenAI)
    client = llm_client.ChatClient(
        "https://llm.example.test/v1/",
        "secret",
        "model-name",
        {"reasoning_effort": "medium"},
    )
    messages = [{"role": "user", "content": "Bonjour"}]

    assert list(client.stream_chat(messages)) == ["réponse"]
    client.sync_chat(messages, allowed_tools_schemas=[{"type": "function"}])

    fake_client = FakeOpenAI.instance
    assert fake_client.init_kwargs == {
        "base_url": "https://llm.example.test/v1",
        "api_key": "secret",
    }
    stream_call, sync_call = fake_client.chat.completions.calls
    assert stream_call["extra_body"] == {"reasoning_effort": "medium"}
    assert sync_call["extra_body"] == {"reasoning_effort": "medium"}
    assert sync_call["tools"] == [{"type": "function"}]
    assert sync_call["tool_choice"] == "auto"


def test_extra_body_is_omitted_when_not_configured(monkeypatch):
    monkeypatch.setattr(llm_client, "OpenAI", FakeOpenAI)
    client = llm_client.ChatClient("https://llm.example.test", "", "model-name")

    list(client.stream_chat([{"role": "user", "content": "Bonjour"}]))

    fake_client = FakeOpenAI.instance
    assert fake_client.init_kwargs["api_key"] == "EMPTY"
    assert "extra_body" not in fake_client.chat.completions.calls[0]

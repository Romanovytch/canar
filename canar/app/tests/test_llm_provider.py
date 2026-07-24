import pytest

from canar.app.api.llm_provider import build_chat_provider_kwargs, detect_llm_provider


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        ("https://api.openai.com/v1", "openai"),
        ("https://api.openai.com/v1/", "openai"),
        ("http://localhost:11434/v1", "ollama"),
        ("localhost:11435/v1", "ollama"),
        ("http://ollama:11434/v1", "ollama"),
        ("https://ollama.internal.example/v1", "ollama"),
        ("https://api.mistral.ai/v1", "other"),
        ("http://localhost:8000/v1", "other"),
    ],
)
def test_detect_llm_provider(base_url, expected):
    assert detect_llm_provider(base_url) == expected


def test_build_chat_provider_kwargs_for_ollama():
    assert build_chat_provider_kwargs(
        "http://localhost:11435/v1",
        "medium",
        keep_alive="10m",
    ) == {
        "extra_body": {
            "reasoning_effort": "medium",
            "keep_alive": "10m",
        }
    }


@pytest.mark.parametrize(
    "base_url",
    [
        "https://api.openai.com/v1",
        "https://api.mistral.ai/v1",
    ],
)
def test_build_chat_provider_kwargs_omits_ollama_parameters_for_other_providers(base_url):
    assert (
        build_chat_provider_kwargs(
            base_url,
            "none",
            keep_alive="10m",
        )
        == {}
    )


def test_build_chat_provider_kwargs_omits_empty_ollama_parameters():
    assert build_chat_provider_kwargs("http://ollama:11434/v1") == {}

import pytest

from canar.app.api.llm_provider import (
    DEFAULT_OLLAMA_BASE_URL,
    build_chat_provider_kwargs,
    detect_llm_provider,
    resolve_llm_base_url,
)


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        ("https://api.openai.com/v1", "openai"),
        ("https://api.openai.com/v1/", "openai"),
        ("http://localhost:11434/v1", "ollama"),
        ("localhost:11434/v1", "ollama"),
        ("http://ollama:11434/v1", "ollama"),
        ("https://ollama.internal.example/v1", "ollama"),
        ("https://api.mistral.ai/v1", "mistral"),
        ("http://localhost:8000/v1", "other"),
        ("", "ollama"),
    ],
)
def test_detect_llm_provider(base_url, expected):
    assert detect_llm_provider(base_url) == expected


def test_explicit_provider_name_identifies_custom_ollama_url():
    assert detect_llm_provider("https://llm.example/v1", "Ollama") == "ollama"
    assert detect_llm_provider("localhost:11435/v1", "ollama") == "ollama"


def test_official_url_takes_precedence_over_provider_name():
    assert detect_llm_provider("https://api.openai.com/v1", "ollama") == "openai"
    assert detect_llm_provider("https://api.mistral.ai/v1", "ollama") == "mistral"


@pytest.mark.parametrize("provider_name", ["", "ollama", "Ollama"])
def test_missing_ollama_base_url_uses_default_endpoint(provider_name):
    assert resolve_llm_base_url(provider_name=provider_name) == DEFAULT_OLLAMA_BASE_URL


def test_missing_non_ollama_base_url_is_not_defaulted():
    assert resolve_llm_base_url(provider_name="openai") == ""


def test_build_chat_provider_kwargs_for_ollama():
    assert build_chat_provider_kwargs(
        "http://localhost:11434/v1",
        "medium",
        keep_alive="10m",
    ) == {
        "extra_body": {
            "reasoning_effort": "medium",
            "keep_alive": "10m",
        }
    }


def test_build_chat_provider_kwargs_for_explicit_ollama_provider():
    assert build_chat_provider_kwargs(
        "https://llm.example/v1",
        "medium",
        provider_name="ollama",
    ) == {"extra_body": {"reasoning_effort": "medium"}}


@pytest.mark.parametrize(
    "base_url",
    [
        "https://api.openai.com/v1",
        "https://api.mistral.ai/v1",
        "http://localhost:8000/v1",
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

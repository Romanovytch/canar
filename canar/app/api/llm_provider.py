from __future__ import annotations

from urllib.parse import urlparse

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1"
_OLLAMA_DEFAULT_PORT = 11434


def detect_llm_provider(base_url: str = "", provider_name: str = "") -> str:
    """Resolve the provider from its official URL, explicit name, or Ollama defaults."""
    parsed = urlparse(base_url if "://" in base_url else f"//{base_url}")
    hostname = (parsed.hostname or "").lower()

    if hostname == "api.openai.com" or hostname.endswith(".openai.com"):
        return "openai"

    if hostname == "api.mistral.ai" or hostname.endswith(".mistral.ai"):
        return "mistral"

    normalized_provider = provider_name.strip().lower()
    if normalized_provider:
        return normalized_provider

    if (
        hostname == "ollama"
        or hostname.startswith("ollama.")
        or parsed.port == _OLLAMA_DEFAULT_PORT
        or not base_url.strip()
    ):
        return "ollama"

    return "other"


def resolve_llm_base_url(base_url: str = "", provider_name: str = "") -> str:
    """Use Ollama's local OpenAI-compatible endpoint when no provider is configured."""
    if not base_url.strip() and not provider_name.strip():
        return DEFAULT_OLLAMA_BASE_URL
    return base_url.rstrip("/")


def build_chat_provider_kwargs(
    base_url: str,
    reasoning_effort: str = "",
    *,
    provider_name: str = "",
    keep_alive: str | None = None,
) -> dict[str, object]:
    """Build request kwargs supported by the provider behind ``base_url``."""
    if detect_llm_provider(base_url, provider_name) != "ollama":
        return {}

    extra_body: dict[str, str] = {}
    if reasoning_effort:
        extra_body["reasoning_effort"] = reasoning_effort
    if keep_alive:
        extra_body["keep_alive"] = keep_alive

    return {"extra_body": extra_body} if extra_body else {}

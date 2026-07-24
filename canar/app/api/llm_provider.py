from __future__ import annotations

from typing import Literal
from urllib.parse import urlparse

LLMProvider = Literal["openai", "ollama", "other"]

_OLLAMA_PORTS = {11434, 11435}


def detect_llm_provider(base_url: str) -> LLMProvider:
    """Infer the LLM provider from an OpenAI-compatible API base URL."""
    parsed = urlparse(base_url if "://" in base_url else f"//{base_url}")
    hostname = (parsed.hostname or "").lower()

    if hostname == "api.openai.com" or hostname.endswith(".openai.com"):
        return "openai"
    
    return "other"


def build_chat_provider_kwargs(
    base_url: str,
    reasoning_effort: str = "",
    *,
    keep_alive: str | None = None,
) -> dict[str, object]:
    """Build request kwargs supported by the provider behind ``base_url``."""
    if detect_llm_provider(base_url) != "other":
        return {}

    extra_body: dict[str, str] = {}
    if reasoning_effort:
        extra_body["reasoning_effort"] = reasoning_effort
    if keep_alive:
        extra_body["keep_alive"] = keep_alive

    return {"extra_body": extra_body} if extra_body else {}

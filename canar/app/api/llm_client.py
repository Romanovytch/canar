from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from openai import OpenAI


class ChatClient:
    def __init__(self, base_url: str, api_key: str, model: str):
        self.client = OpenAI(base_url=base_url.rstrip("/"), api_key=api_key or "EMPTY")
        self.model = model

    def stream_chat(
        self,
        messages: list[dict],
        temperature: float = 0.2,
        top_p: float = 1.0,
        max_tokens: int = 2048,
        allowed_tools_schemas: list[dict] | None = None,
        tool_choice: str | dict | None = None,
    ) -> Iterable[Any]:

        api_args: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": True,
        }

        if allowed_tools_schemas and len(allowed_tools_schemas) > 0:
            api_args["tools"] = allowed_tools_schemas
            api_args["tool_choice"] = tool_choice or "auto"

        resp = self.client.chat.completions.create(**api_args)
        for chunk in resp:
            yield chunk

    def sync_chat(
        self,
        messages: list[dict],
        temperature: float = 0.2,
        top_p: float = 1.0,
        max_tokens: int = 2048,
        allowed_tools_schemas: list[dict] | None = None,
        tool_choice: str | dict | None = None,
    ):

        api_args: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": False,
        }

        if allowed_tools_schemas and len(allowed_tools_schemas) > 0:
            api_args["tools"] = allowed_tools_schemas
            api_args["tool_choice"] = tool_choice or "auto"

        return self.client.chat.completions.create(**api_args)

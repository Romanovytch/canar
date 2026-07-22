from __future__ import annotations

from collections.abc import Iterable

from openai import OpenAI

from canar.app.api.llm_provider import build_chat_provider_kwargs, resolve_llm_base_url


class ChatClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        reasoning_effort: str = "",
        provider_name: str = "",
    ):
        resolved_base_url = resolve_llm_base_url(base_url, provider_name)
        self.client = OpenAI(base_url=resolved_base_url, api_key=api_key or "EMPTY")
        self.model = model
        self.provider_kwargs = build_chat_provider_kwargs(
            resolved_base_url,
            reasoning_effort,
            provider_name=provider_name,
        )

    def stream_chat(
        self,
        messages: list[dict],
        temperature: float = 0.2,
        top_p: float = 1.0,
        max_tokens: int = 2048,
    ) -> Iterable[str]:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stream=True,
            **self.provider_kwargs,
        )
        for chunk in resp:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content

    def sync_chat(
        self,
        messages: list[dict],
        temperature: float = 0.2,
        top_p: float = 1.0,
        max_tokens: int = 2048,
        allowed_tools_schemas: list[dict] | None = None,
    ):
        api_args = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": False,
            **self.provider_kwargs,
        }
        if allowed_tools_schemas:
            api_args["tools"] = allowed_tools_schemas
            api_args["tool_choice"] = "auto"

        return  self.client.chat.completions.create(**api_args)

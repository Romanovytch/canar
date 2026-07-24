from __future__ import annotations

from collections.abc import Iterable

from openai import OpenAI

from canar.app.api.llm_provider import build_chat_provider_kwargs


class ChatClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        reasoning_effort: str = "",
    ):
        self.client = OpenAI(base_url=base_url.rstrip("/"), api_key=api_key or "EMPTY")
        self.model = model
        self.provider_kwargs = build_chat_provider_kwargs(base_url, reasoning_effort)

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

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Protocol

from canar.app.retrieval.models import RetrievalProfile


@dataclass(frozen=True)
class GenerationParams:
    model: str | None = None
    temperature: float = 0.2
    top_p: float = 1.0
    max_tokens: int = 2048

    def __post_init__(self) -> None:
        if self.model is not None and not self.model.strip():
            raise ValueError("generation.model must not be empty")
        if not 0.0 <= self.temperature <= 1.0:
            raise ValueError("generation.temperature must be between 0 and 1")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("generation.top_p must be greater than 0 and at most 1")
        if not 256 <= self.max_tokens <= 8192:
            raise ValueError("generation.max_tokens must be between 256 and 8192")


@dataclass(frozen=True)
class ChatProfile:
    name: str
    retrieval: RetrievalProfile | None
    generation: GenerationParams

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("profile.name must not be empty")


class ProfiledChatbot(Protocol):
    id: str
    profile_name: str | None


def build_chat_profiles(
    retrieval_profiles: Mapping[str, RetrievalProfile],
    generation_overrides: Mapping[str, GenerationParams] | None = None,
) -> dict[str, ChatProfile]:
    if "prompt_only" in retrieval_profiles:
        raise ValueError("'prompt_only' is reserved for chatbots without document retrieval")

    generation_overrides = generation_overrides or {}
    known_names = set(retrieval_profiles) | {"prompt_only"}
    unknown_overrides = sorted(set(generation_overrides) - known_names)
    if unknown_overrides:
        raise ValueError(
            "Generation settings reference unknown chat profile(s): " + ", ".join(unknown_overrides)
        )

    default_generation = GenerationParams()
    profiles = {
        name: ChatProfile(
            name=name,
            retrieval=retrieval_profile,
            generation=generation_overrides.get(name, default_generation),
        )
        for name, retrieval_profile in retrieval_profiles.items()
    }
    profiles["prompt_only"] = ChatProfile(
        name="prompt_only",
        retrieval=None,
        generation=generation_overrides.get("prompt_only", default_generation),
    )
    return profiles


def resolve_chatbot_profiles(
    chatbots: Iterable[ProfiledChatbot],
    profiles: Mapping[str, ChatProfile],
) -> dict[str, ChatProfile]:
    resolved: dict[str, ChatProfile] = {}
    for chatbot in chatbots:
        if chatbot.id in resolved:
            raise ValueError(f"Duplicate chatbot id: {chatbot.id!r}")

        profile_name = chatbot.profile_name
        if profile_name is None or not profile_name.strip():
            raise ValueError(f"Chatbot {chatbot.id!r} must define profile_name")

        profile = profiles.get(profile_name)
        if profile is None:
            raise ValueError(f"Unknown profile_name for chatbot {chatbot.id!r}: {profile_name!r}")
        resolved[chatbot.id] = profile

    return resolved


def build_agent_profile_mapping(
    chatbot_profiles: Mapping[str, ChatProfile],
) -> dict[str, str | None]:
    return {
        chatbot_id: profile.retrieval.name if profile.retrieval is not None else None
        for chatbot_id, profile in chatbot_profiles.items()
    }

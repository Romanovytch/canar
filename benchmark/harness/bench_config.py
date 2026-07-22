"""
Load benchmark run settings + retrieval profiles from config.yaml.

Product-agnostic on purpose: this only parses YAML into plain dataclasses.
Resolving a ProfileSpec name to the product's RetrievalProfile is the caller's
job (e2e/eval_e2e.py), so the harness stays decoupled from CanaR.

Profile rows select named product retrieval profiles. Extra row keys are retained
in `ProfileSpec.params` for config compatibility, but the end-to-end runner does
not mutate product profiles from those values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ProfileSpec:
    """One named retrieval profile; params are parsed but not applied by E2E."""

    name: str
    params: dict


@dataclass(frozen=True)
class ChatbotSpec:
    """Prompt and profile fields needed by the benchmark from chatbot YAML."""

    id: str
    system_prompt: str
    profile_name: str


@dataclass
class BenchConfig:
    dataset: str  # path relative to benchmark/
    agent: str = "r_helpdesk"
    limit: int | None = None  # None = all questions
    judge_model: str | None = None  # None = fall back to the product LLM
    gen_max_tokens: int | None = None  # None = use the chatbot profile
    measure_resources: bool = False  # measure CPU/memory/GPU cost per phase
    profiles: list[ProfileSpec] = field(default_factory=list)
    # Expected environment (embed_model, collection). The caller's preflight
    # checks canar/.env against these so results are comparable across machines.
    environment: dict = field(default_factory=dict)


def load_config(path: str | Path) -> BenchConfig:
    data = yaml.safe_load(Path(path).read_text()) or {}
    run = data.get("run", {})

    profiles = []
    for raw in data.get("profiles", []) or []:
        raw = dict(raw)
        name = raw.pop("name")
        if "strategy" in raw:
            raise ValueError(
                f"Benchmark profile {name!r} uses deprecated 'strategy'. "
                "Use the product RetrievalProfile name as 'name' instead."
            )
        profiles.append(ProfileSpec(name=name, params=raw))

    return BenchConfig(
        dataset=run.get("dataset"),
        agent=run.get("agent", "r_helpdesk"),
        limit=run.get("limit"),
        judge_model=run.get("judge_model"),
        gen_max_tokens=run.get("gen_max_tokens"),
        measure_resources=run.get("measure_resources", False),
        profiles=profiles,
        environment=data.get("environment", {}) or {},
    )


def _required_string(raw: dict[str, Any], field_name: str, chatbot_id: str) -> str:
    value = raw.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"Chatbot {chatbot_id!r} must define a non-empty {field_name!r} in chatbot YAML."
        )
    return value


def load_chatbot(path: str | Path, chatbot_id: str) -> ChatbotSpec:
    """Load the benchmark-facing fields for one YAML-defined chatbot."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    raw_chatbots = data.get("chatbots") if isinstance(data, dict) else None
    if not isinstance(raw_chatbots, list):
        raise ValueError("Chatbot YAML must contain a 'chatbots' list.")

    matches = [raw for raw in raw_chatbots if isinstance(raw, dict) and raw.get("id") == chatbot_id]
    if not matches:
        raise ValueError(f"Chatbot {chatbot_id!r} was not found in {path}.")
    if len(matches) > 1:
        raise ValueError(f"Chatbot id {chatbot_id!r} is duplicated in {path}.")

    raw = matches[0]
    return ChatbotSpec(
        id=chatbot_id,
        system_prompt=_required_string(raw, "system_prompt", chatbot_id),
        profile_name=_required_string(raw, "profile_name", chatbot_id),
    )


def select_profile_specs(
    configured: list[ProfileSpec],
    attached_retrieval_profile_name: str | None,
) -> list[ProfileSpec]:
    """Use an explicit retrieval matrix, or default to the chatbot's profile."""
    if configured:
        return configured
    if attached_retrieval_profile_name is None:
        raise ValueError(
            "The selected chatbot profile has no retrieval configuration; "
            "add at least one benchmark profile."
        )
    return [ProfileSpec(name=attached_retrieval_profile_name, params={})]

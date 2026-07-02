"""
Load benchmark run settings + retrieval profiles from config.yaml.

Product-agnostic on purpose: this only parses YAML into plain dataclasses.
Turning a ProfileSpec into the product's RetrievalProfile/strategy is the
caller's job (e2e/eval_e2e.py), so the harness stays decoupled from CanaR.

This mirrors the shape of the product's retrieval profiles
(canar/app/retrieval/profiles.py) so the same hyperparameters — top_k,
score_threshold, source_filter, fallback_top_k — can be swept here to compare
retrieval strategies, which is the central question of the project.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ProfileSpec:
    """One retrieval profile to benchmark. `params` holds the strategy's knobs."""
    name: str
    strategy: str
    params: dict


@dataclass
class BenchConfig:
    dataset: str                       # path relative to benchmark/
    agent: str = "r_helpdesk"
    limit: int | None = None           # None = all questions
    judge_model: str | None = None     # None = fall back to the product LLM
    gen_max_tokens: int = 8192
    measure_resources: bool = False    # measure CPU/memory/GPU cost per phase
    profiles: list[ProfileSpec] = field(default_factory=list)
    # Expected environment (embed_model, collection). The caller's preflight
    # checks canar/.env against these so results are comparable across machines.
    environment: dict = field(default_factory=dict)


def load_config(path: str | Path) -> BenchConfig:
    data = yaml.safe_load(Path(path).read_text()) or {}
    run = data.get("run", {})

    profiles = []
    for raw in data.get("profiles", []):
        raw = dict(raw)
        name = raw.pop("name")
        strategy = raw.pop("strategy", "simple_vector")
        profiles.append(ProfileSpec(name=name, strategy=strategy, params=raw))

    return BenchConfig(
        dataset=run.get("dataset"),
        agent=run.get("agent", "r_helpdesk"),
        limit=run.get("limit"),
        judge_model=run.get("judge_model"),
        gen_max_tokens=run.get("gen_max_tokens", 8192),
        measure_resources=run.get("measure_resources", False),
        profiles=profiles,
        environment=data.get("environment", {}) or {},
    )

from pathlib import Path

import pytest
from bench_config import ProfileSpec, load_chatbot, load_config, select_profile_specs


def _write_chatbots(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


def test_load_chatbot_projects_prompt_and_profile(tmp_path):
    path = _write_chatbots(
        tmp_path / "chatbots.yaml",
        """
chatbots:
  - id: r_helpdesk
    name: ignored-by-benchmark
    profile_name: simple_vector
    system_prompt: Explain R clearly.
  - id: sas_to_r
    profile_name: prompt_only
    system_prompt: Translate SAS.
""",
    )

    chatbot = load_chatbot(path, "r_helpdesk")

    assert chatbot.id == "r_helpdesk"
    assert chatbot.system_prompt == "Explain R clearly."
    assert chatbot.profile_name == "simple_vector"


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("chatbots: []\n", "was not found"),
        (
            "chatbots:\n  - id: bot\n    system_prompt: prompt\n",
            "profile_name",
        ),
        (
            "chatbots:\n"
            "  - id: bot\n    profile_name: simple_vector\n    system_prompt: one\n"
            "  - id: bot\n    profile_name: hybrid\n    system_prompt: two\n",
            "duplicated",
        ),
    ],
)
def test_load_chatbot_rejects_incompatible_yaml(tmp_path, body, message):
    path = _write_chatbots(tmp_path / "chatbots.yaml", body)

    with pytest.raises(ValueError, match=message):
        load_chatbot(path, "bot")


def test_explicit_profile_matrix_overrides_attached_retrieval():
    configured = [ProfileSpec(name="hybrid", params={})]

    assert select_profile_specs(configured, None) is configured


def test_empty_profile_matrix_uses_attached_retrieval():
    selected = select_profile_specs([], "simple_vector")

    assert selected == [ProfileSpec(name="simple_vector", params={})]


def test_empty_profile_matrix_requires_attached_retrieval():
    with pytest.raises(ValueError, match="no retrieval configuration"):
        select_profile_specs([], None)


def test_missing_generation_budget_defers_to_chatbot_profile(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "run:\n  dataset: datasets/questions.yaml\n  agent: r_helpdesk\n",
        encoding="utf-8",
    )

    assert load_config(path).gen_max_tokens is None


def test_null_profile_matrix_is_empty(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "run:\n  dataset: datasets/questions.yaml\nprofiles:\n",
        encoding="utf-8",
    )

    assert load_config(path).profiles == []

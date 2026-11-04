from __future__ import annotations

from pathlib import Path

import pytest

from canar.app.profiles import build_chat_profiles
from canar.app.retrieval.profile_config import build_configured_profiles
from canar.app.retrieval.profiles import build_retrieval_profiles
from canar.app.yaml_loader import load_retrieval_profile_specs

EXAMPLE_YAML = Path(__file__).resolve().parents[1] / "chatbots" / "chatbotconfig.exemple.yaml"


@pytest.fixture
def presets():
    return build_retrieval_profiles(("utilitr_v1",), dense_vector_name="dense")


def test_extends_preset_and_overrides_root_values(presets):
    profiles = build_configured_profiles(
        {
            "helpdesk": {
                "extends": "hybrid_rerank_bge",
                "collections": ["utilitr_v2", "doc_r"],
                "fetch_top_k": 30,
            }
        },
        presets,
    )

    profile = profiles["helpdesk"]
    base = presets["hybrid_rerank_bge"]
    assert profile.name == "helpdesk"
    assert profile.strategy == "hybrid"
    assert profile.collections == ("utilitr_v2", "doc_r")
    assert profile.fetch_top_k == 30
    assert profile.min_score == base.min_score
    assert profile.rerank == base.rerank
    assert set(profiles) == {"helpdesk"}


def test_blocks_are_merged_key_by_key(presets):
    profiles = build_configured_profiles(
        {
            "tuned": {
                "extends": "hybrid",
                "dense": {"min_score": 0.6},
                "fusion": {"weights": {"dense": 2.0, "sparse": 0.5}, "rrf_k": 30},
            }
        },
        presets,
    )

    profile = profiles["tuned"]
    assert profile.dense.min_score == 0.6
    assert profile.dense.vector_name == "dense"
    assert profile.fusion.weights == {"dense": 2.0, "sparse": 0.5}
    assert profile.fusion.rrf_k == 30
    assert profile.fusion.method == presets["hybrid"].fusion.method


def test_block_is_created_when_base_has_none_and_removed_with_null(presets):
    profiles = build_configured_profiles(
        {
            "with_parents": {"extends": "simple_vector", "parent_child": {}},
            "without_rerank": {"extends": "hybrid_rerank_bge", "rerank": None},
        },
        presets,
    )

    assert profiles["with_parents"].parent_child.parent_collection_suffix == "_parents"
    assert profiles["without_rerank"].rerank is None


def test_profiles_can_extend_other_configured_profiles_in_any_order(presets):
    profiles = build_configured_profiles(
        {
            "child": {"extends": "parent", "output_top_k": 8},
            "parent": {"extends": "simple_vector", "collections": ["custom"]},
        },
        presets,
    )

    assert profiles["child"].collections == ("custom",)
    assert profiles["child"].output_top_k == 8


def test_profile_can_be_defined_without_extends(presets):
    profiles = build_configured_profiles(
        {
            "sparse_only": {
                "strategy": "simple_sparse",
                "collections": ["utilitr_v1"],
                "sparse": {"vector_name": "bm25", "gap_ratio": 0.5},
            }
        },
        presets,
    )

    profile = profiles["sparse_only"]
    assert profile.strategy == "simple_sparse"
    assert profile.sparse.vector_name == "bm25"
    assert profile.fetch_top_k == 10


@pytest.mark.parametrize(
    ("specs", "message"),
    [
        ({"simple_vector": {"extends": "hybrid"}}, "reuse a built-in profile name"),
        ({"p": {"extends": "missing"}}, "Unknown retrieval profile: 'missing'"),
        ({"a": {"extends": "b"}, "b": {"extends": "a"}}, "Cyclic 'extends'"),
        ({"p": {"extends": "hybrid", "fetch_topk": 3}}, "unknown setting(s): fetch_topk"),
        ({"p": {"extends": "hybrid", "dense": {"minscore": 1}}}, "unknown 'dense' setting(s)"),
        ({"p": {"extends": "hybrid", "dense": 0.5}}, "'dense' must be a mapping or null"),
        ({"p": {"extends": "hybrid", "collections": "utilitr"}}, "'collections' must be a list"),
        ({"p": {"strategy": "simple_vector"}}, "must define 'collections'"),
        ({"p": {"extends": "hybrid", "min_score": 2}}, "min_score must be between 0 and 1"),
        ({"p": {"extends": "simple_vector", "rerank": {}}}, "rerank is only supported"),
        ({"p": {"extends": "hybrid", "strategy": "bm42"}}, "unsupported retrieval strategy"),
        ({"p": {"extends": "hybrid", "summary": {}}}, "Retrieval profile 'p'"),
    ],
)
def test_invalid_specs_raise_with_profile_name(presets, specs, message):
    with pytest.raises(ValueError) as exc_info:
        build_configured_profiles(specs, presets)
    assert message in str(exc_info.value)


def test_configured_profiles_cannot_use_reserved_prompt_only_name(presets):
    configured = build_configured_profiles({"prompt_only": {"extends": "hybrid"}}, presets)

    with pytest.raises(ValueError, match="'prompt_only' is reserved"):
        build_chat_profiles({**presets, **configured})


def test_load_specs_reads_optional_section(tmp_path):
    path = tmp_path / "chatbots.yaml"
    path.write_text(
        "retrieval_profiles:\n"
        "  helpdesk:\n"
        "    extends: hybrid\n"
        "    dense: {min_score: 0.6}\n"
        "chatbots: []\n",
        encoding="utf-8",
    )

    assert load_retrieval_profile_specs(path) == {
        "helpdesk": {"extends": "hybrid", "dense": {"min_score": 0.6}}
    }


def test_load_specs_without_section_or_file_returns_empty(tmp_path):
    path = tmp_path / "chatbots.yaml"
    path.write_text("chatbots: []\n", encoding="utf-8")

    assert load_retrieval_profile_specs(path) == {}
    assert load_retrieval_profile_specs(tmp_path / "missing.yaml") == {}


def test_load_specs_rejects_malformed_section(tmp_path):
    path = tmp_path / "chatbots.yaml"
    path.write_text("retrieval_profiles:\n  - hybrid\n", encoding="utf-8")

    with pytest.raises(ValueError, match="retrieval_profiles"):
        load_retrieval_profile_specs(path)


def test_example_file_profiles_are_valid(presets):
    specs = load_retrieval_profile_specs(EXAMPLE_YAML)

    profiles = build_configured_profiles(specs, presets)

    assert set(profiles) == set(specs)

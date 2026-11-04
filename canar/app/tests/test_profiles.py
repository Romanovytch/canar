from types import SimpleNamespace

import pytest

from canar.app.profiles import (
    GenerationParams,
    build_agent_profile_mapping,
    build_chat_profiles,
    resolve_chatbot_profiles,
)
from canar.app.retrieval.models import DenseRetrievalParams, RetrievalProfile


@pytest.fixture
def retrieval_profiles() -> dict[str, RetrievalProfile]:
    profile = RetrievalProfile(
        name="simple_vector",
        strategy="simple_vector",
        collections=("docs",),
        dense=DenseRetrievalParams(),
    )
    return {profile.name: profile}


def test_chat_profiles_combine_retrieval_and_generation(retrieval_profiles):
    generation = GenerationParams(
        model="test-model",
        temperature=0.1,
        top_p=0.9,
        max_tokens=1024,
    )

    profiles = build_chat_profiles(
        retrieval_profiles,
        generation_overrides={"simple_vector": generation},
    )

    assert profiles["simple_vector"].retrieval is retrieval_profiles["simple_vector"]
    assert profiles["simple_vector"].generation is generation
    assert profiles["prompt_only"].retrieval is None
    assert profiles["prompt_only"].generation == GenerationParams()


def test_resolve_chatbot_profiles_and_build_retrieval_mapping(retrieval_profiles):
    profiles = build_chat_profiles(retrieval_profiles)
    chatbots = [
        SimpleNamespace(id="r_helpdesk", profile_name="simple_vector"),
        SimpleNamespace(id="sas_to_r", profile_name="prompt_only"),
    ]

    resolved = resolve_chatbot_profiles(chatbots, profiles)

    assert resolved["r_helpdesk"] is profiles["simple_vector"]
    assert resolved["sas_to_r"] is profiles["prompt_only"]
    assert build_agent_profile_mapping(resolved) == {
        "r_helpdesk": "simple_vector",
        "sas_to_r": None,
    }


@pytest.mark.parametrize("profile_name", [None, ""])
def test_resolve_chatbot_profiles_rejects_missing_profile(retrieval_profiles, profile_name):
    profiles = build_chat_profiles(retrieval_profiles)

    with pytest.raises(ValueError, match="must define profile_name"):
        resolve_chatbot_profiles(
            [SimpleNamespace(id="r_helpdesk", profile_name=profile_name)],
            profiles,
        )


def test_resolve_chatbot_profiles_rejects_unknown_profile(retrieval_profiles):
    profiles = build_chat_profiles(retrieval_profiles)

    with pytest.raises(ValueError, match="Unknown profile_name"):
        resolve_chatbot_profiles(
            [SimpleNamespace(id="r_helpdesk", profile_name="missing")],
            profiles,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"model": " "}, "model"),
        ({"temperature": 1.1}, "temperature"),
        ({"top_p": 0.0}, "top_p"),
        ({"max_tokens": 255}, "max_tokens"),
        ({"max_tokens": 8193}, "max_tokens"),
    ],
)
def test_generation_params_validate_values(kwargs, message):
    with pytest.raises(ValueError, match=message):
        GenerationParams(**kwargs)

"""Run settings read from config.yaml, and what they fall back to."""

from __future__ import annotations

from bench_config import load_config


def _config(tmp_path, run: str):
    path = tmp_path / "config.yaml"
    path.write_text(f"run:\n  dataset: d.yaml\n{run}profiles:\n  - name: simple_vector\n")
    return load_config(path)


def test_a_config_without_the_key_judges_each_question_three_times(tmp_path):
    """The benchmark compares profiles, and one judgement per question cannot
    tell a real difference from the judge's own noise."""
    assert _config(tmp_path, "").judge_repeats == 3


def test_the_config_can_ask_for_a_single_judgement(tmp_path):
    assert _config(tmp_path, "  judge_repeats: 1\n").judge_repeats == 1


def test_the_judge_temperature_falls_back_to_the_ragas_value(tmp_path):
    """0.3 is what RAGAS uses when it asks for several generations."""
    assert _config(tmp_path, "").judge_temperature == 0.3
    assert _config(tmp_path, "  judge_temperature: 0.7\n").judge_temperature == 0.7

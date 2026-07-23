from pathlib import Path

import pytest
from bench_cli import parse_eval_args


def test_defaults_to_config_and_dynamic_retrieval_cutoff(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    default = tmp_path / "config.yaml"
    args = parse_eval_args([], default_config=default)
    assert args.config == default.resolve()
    assert args.configs == [default.resolve()]
    assert args.retrieval_k is None


def test_custom_config_and_retrieval_k(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = parse_eval_args(
        ["--config", "config.dico.yaml", "--retrieval-k", "5"],
        default_config=Path("config.yaml"),
    )
    assert args.config == (tmp_path / "config.dico.yaml").resolve()
    assert args.configs == [(tmp_path / "config.dico.yaml").resolve()]
    assert args.retrieval_k == 5


def test_multiple_configs_are_resolved_in_order(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = parse_eval_args(
        ["--config", "config.dico.yaml", "--config", "config.utilitr.yaml"],
        default_config=Path("config.yaml"),
    )
    assert args.config == (tmp_path / "config.dico.yaml").resolve()
    assert args.configs == [
        (tmp_path / "config.dico.yaml").resolve(),
        (tmp_path / "config.utilitr.yaml").resolve(),
    ]


def test_retrieval_k_underscore_alias(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = parse_eval_args(["--retrieval_k", "3"], default_config=Path("config.yaml"))
    assert args.retrieval_k == 3


@pytest.mark.parametrize("value", ["0", "-1"])
def test_retrieval_k_must_be_positive(value, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        parse_eval_args(
            ["--retrieval-k", value],
            default_config=Path("config.yaml"),
        )

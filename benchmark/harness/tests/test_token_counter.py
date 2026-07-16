"""Issue #64 — token counting + per-profile aggregation."""

import pandas as pd
from token_counter import TokenCounter, summarize_token_usage


def test_counts_a_string_and_is_monotonic():
    counter = TokenCounter()          # no model name -> tiktoken or heuristic
    assert counter.count_text("") == 0
    short = counter.count_text("hello world")
    long = counter.count_text("hello world " * 50)
    assert short > 0
    assert long > short               # more text -> more tokens


def test_prompt_counts_all_messages():
    counter = TokenCounter()
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "What is the capital of France?"},
    ]
    total = counter.count_prompt(messages)
    # at least as many tokens as the longer single message
    assert total >= counter.count_text(messages[1]["content"])


def test_reports_its_source():
    # tiktoken is installed in the benchmark venv, so that's what it should pick
    # when no model tokenizer is configured.
    assert TokenCounter().source in ("tiktoken", "heuristic")


def test_heuristic_fallback(monkeypatch):
    # Force both real tokenizers to be unavailable -> char/4 heuristic.
    import builtins

    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name in ("transformers", "tiktoken"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    counter = TokenCounter()
    assert counter.source == "heuristic"
    assert counter.count_text("a" * 40) == 10   # 40 chars / 4


def test_summarize_token_usage_computes_avg_min_max_total():
    df_dense = pd.DataFrame(
        {"input_tokens": [100, 200], "output_tokens": [10, 30], "total_tokens": [110, 230]}
    )
    df_sparse = pd.DataFrame(
        {"input_tokens": [50, 50], "output_tokens": [5, 15], "total_tokens": [55, 65]}
    )

    out = summarize_token_usage([("dense", df_dense), ("sparse", df_sparse)])
    dense = out[out["profile"] == "dense"].iloc[0]

    assert dense["input_tokens_avg"] == 150.0
    assert dense["input_tokens_min"] == 100
    assert dense["input_tokens_max"] == 200
    assert dense["input_tokens_total"] == 300
    assert dense["total_tokens_total"] == 340

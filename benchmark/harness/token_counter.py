"""
Token counting for the benchmark (issue #64).

Counts the tokens sent to the LLM (the prompt) and the tokens generated (the
answer), so strategies can be compared on context size and generation cost.
Counting happens inside the benchmark on strings we already have — no product
code change, no extra LLM call.

`TokenCounter` uses the best tokenizer available, in order:
  1. the model's tokenizer via `transformers` (when installed and a name is set)
  2. `tiktoken` (a real BPE tokenizer; Qwen's is tiktoken-based, so it's close)
  3. a char/4 heuristic (last resort, clearly labelled by `.source`)
"""

from __future__ import annotations

TOKEN_FIELDS = ("input_tokens", "output_tokens", "total_tokens")


class TokenCounter:
    def __init__(self, model_name: str | None = None):
        self._encode = None
        self._chat_template = None
        self.source = "heuristic"

        # 1) the configured model's own tokenizer (most accurate)
        if model_name:
            try:
                from transformers import AutoTokenizer

                tok = AutoTokenizer.from_pretrained(model_name)
                self._encode = lambda text: tok.encode(text)
                self._chat_template = tok
                self.source = f"transformers:{model_name}"
                return
            except Exception:
                pass

        # 2) tiktoken — a real BPE tokenizer, good enough for comparison
        try:
            import tiktoken

            enc = tiktoken.get_encoding("cl100k_base")
            self._encode = lambda text: enc.encode(text)
            self.source = "tiktoken"
            return
        except Exception:
            pass

        # 3) heuristic: ~4 characters per token
        self._encode = lambda text: [None] * max(1, round(len(text) / 4)) if text else []

    def count_text(self, text: str | None) -> int:
        """Token count of a single string (e.g. the generated answer)."""
        if not text:
            return 0
        return len(self._encode(text))

    def count_prompt(self, messages: list[dict]) -> int:
        """Token count of the full chat prompt sent to the model.

        Uses the model's chat template when a real tokenizer is available (it
        adds the role/special tokens the model actually sees); otherwise sums
        the message contents.
        """
        if self._chat_template is not None:
            try:
                ids = self._chat_template.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True
                )
                return len(ids)
            except Exception:
                pass
        return sum(self.count_text(m.get("content", "")) for m in messages)


def summarize_token_usage(summaries):
    """Per-profile avg / min / max / total for each token field (issue #64).

    `summaries`: list of (profile_name, DataFrame) where the DataFrame carries
    the TOKEN_FIELDS columns. Returns a DataFrame, one row per profile.
    """
    import pandas as pd

    rows = []
    for name, df in summaries:
        row: dict[str, object] = {"profile": name}
        for field in TOKEN_FIELDS:
            if field in df.columns:
                series = df[field].dropna()
                if series.empty:
                    continue
                row[f"{field}_avg"] = round(float(series.mean()), 1)
                row[f"{field}_min"] = int(series.min())
                row[f"{field}_max"] = int(series.max())
                row[f"{field}_total"] = int(series.sum())
        rows.append(row)
    return pd.DataFrame(rows)

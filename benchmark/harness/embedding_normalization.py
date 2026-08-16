from __future__ import annotations

import asyncio
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from langchain_core.embeddings import Embeddings

from canar.app.api.text_normalization import normalize_for_embedding

# MARKDOWN TABLE WORKAROUND
# The embedding server answers 500 "failed to encode response: json: unsupported
# value: NaN" for some markdown tables, which loses the answer relevancy score
# for that question. The failure is deterministic — the same text failed ten
# times out of ten — so RAGAS's own retries cannot recover it.
#
# It is not tables as such: `| a | b |\n|---|---|\n| c | d |` embeds fine, while
# the same shape carrying certain words does not, and those same words embed
# fine as running text. Both the table structure and the content take part, and
# only the structure is something we can act on. Removing it fixed every case
# reproduced here.
#
# This lives in the benchmark rather than in the product's
# `normalize_for_embedding`, so the application's own embedding path is
# unchanged. The product can hit the same server bug when a user pastes a table
# into the chat; that is a separate decision.
_SEPARATOR_ROW = re.compile(r"^\|?[\s:|-]*\|[\s:|-]*\|?$")


def flatten_markdown_tables(text: str) -> str:
    """Turn markdown table rows into plain text, leaving other lines untouched."""
    lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        if "|" in stripped and "-" in stripped and _SEPARATOR_ROW.match(stripped):
            continue  # `|---|---|` carries no content
        if stripped.count("|") >= 2:
            stripped = stripped.strip("|").replace("|", " ")
            stripped = re.sub(r"\s{2,}", " ", stripped).strip()
            lines.append(stripped)
            continue
        lines.append(line)
    return "\n".join(lines)


def prepare_for_embedding(text: str) -> str:
    """Both workarounds, in the order the server needs them."""
    return flatten_markdown_tables(normalize_for_embedding(text))


# INTERMITTENT EMBEDDING FAILURE
# Separately from the deterministic table case above, the server still answers
# 500 "unsupported value: NaN" now and then — five answer relevancy scores lost
# out of 24 questions in one run. It could not be reproduced: the same answers,
# raw and normalized, embed fine, as do empty strings, batches containing empty
# strings, inputs past the model's context length, and sixteen concurrent
# requests. The text that fails is a question RAGAS generates internally and
# discards, so it survives nowhere and cannot be inspected after the fact.
#
# Hence two behaviors rather than a fix, which together cover both cases without
# knowing which one it is: retry, so a transient failure stops costing a score;
# and record the input, so a failure that repeats stops being invisible.
RETRY_ATTEMPTS = 3
RETRY_PAUSE_S = 1.0


def _record_failure(log_path: Path | None, texts: list[str], error: Exception) -> None:
    """Append the exact inputs that failed, so the next run can be diagnosed."""
    entry = {
        "when": datetime.now().isoformat(timespec="seconds"),
        "error": f"{type(error).__name__}: {error}",
        "count": len(texts),
        "lengths": [len(t) for t in texts],
        "texts": texts,
    }
    print(
        f"[embedding] {len(texts)} text(s) failed after {RETRY_ATTEMPTS} attempts: "
        f"{type(error).__name__}",
        file=sys.stderr,
    )
    if log_path is None:
        return
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError as exc:  # logging must never break a run
        print(f"[embedding] could not write {log_path}: {exc}", file=sys.stderr)


# APOSTROPHE NORMALIZATION WORKAROUND
# RAGAS may use sync or async query/document embedding methods, so all four are
# forwarded through the shared converter. To remove this workaround, delete this
# adapter and construct `OpenAIEmbeddings` directly in `e2e/eval_e2e.py` again.
class NormalizingEmbeddings(Embeddings):
    """Apply the embedding workarounds before forwarding RAGAS embeddings."""

    def __init__(self, delegate: Embeddings, failure_log: Path | None = None):
        self.delegate = delegate
        self.failure_log = failure_log

    def _call(self, method, texts: list[str], arg):
        """Call the delegate, retrying, and record the input if it never works."""
        for attempt in range(1, RETRY_ATTEMPTS + 1):
            try:
                return method(arg)
            except Exception as error:  # noqa: BLE001 — any failure is worth a retry
                if attempt == RETRY_ATTEMPTS:
                    _record_failure(self.failure_log, texts, error)
                    raise
                time.sleep(RETRY_PAUSE_S)

    async def _acall(self, method, texts: list[str], arg):
        """Async twin of `_call`; RAGAS uses whichever the metric happens to need."""
        for attempt in range(1, RETRY_ATTEMPTS + 1):
            try:
                return await method(arg)
            except Exception as error:  # noqa: BLE001
                if attempt == RETRY_ATTEMPTS:
                    _record_failure(self.failure_log, texts, error)
                    raise
                await asyncio.sleep(RETRY_PAUSE_S)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Normalize RAGAS document batches before the delegate's HTTP request.
        prepared = [prepare_for_embedding(t) for t in texts]
        return self._call(self.delegate.embed_documents, prepared, prepared)

    def embed_query(self, text: str) -> list[float]:
        # Normalize synchronous RAGAS query embeddings.
        prepared = prepare_for_embedding(text)
        return self._call(self.delegate.embed_query, [prepared], prepared)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        # Normalize asynchronous RAGAS document batches.
        prepared = [prepare_for_embedding(t) for t in texts]
        return await self._acall(self.delegate.aembed_documents, prepared, prepared)

    async def aembed_query(self, text: str) -> list[float]:
        # Normalize asynchronous RAGAS query embeddings.
        prepared = prepare_for_embedding(text)
        return await self._acall(self.delegate.aembed_query, [prepared], prepared)

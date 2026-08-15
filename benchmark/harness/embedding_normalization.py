from __future__ import annotations

import re

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


# APOSTROPHE NORMALIZATION WORKAROUND
# RAGAS may use sync or async query/document embedding methods, so all four are
# forwarded through the shared converter. To remove this workaround, delete this
# adapter and construct `OpenAIEmbeddings` directly in `e2e/eval_e2e.py` again.
class NormalizingEmbeddings(Embeddings):
    """Apply the embedding workarounds before forwarding RAGAS embeddings."""

    def __init__(self, delegate: Embeddings):
        self.delegate = delegate

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Normalize RAGAS document batches before the delegate's HTTP request.
        return self.delegate.embed_documents([prepare_for_embedding(t) for t in texts])

    def embed_query(self, text: str) -> list[float]:
        # Normalize synchronous RAGAS query embeddings.
        return self.delegate.embed_query(prepare_for_embedding(text))

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        # Normalize asynchronous RAGAS document batches.
        return await self.delegate.aembed_documents(
            [prepare_for_embedding(t) for t in texts]
        )

    async def aembed_query(self, text: str) -> list[float]:
        # Normalize asynchronous RAGAS query embeddings.
        return await self.delegate.aembed_query(prepare_for_embedding(text))

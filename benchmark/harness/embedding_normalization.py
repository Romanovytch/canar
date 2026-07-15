from __future__ import annotations

from langchain_core.embeddings import Embeddings

from canar.app.api.text_normalization import normalize_for_embedding


# APOSTROPHE NORMALIZATION WORKAROUND
# RAGAS may use sync or async query/document embedding methods, so all four are
# forwarded through the shared converter. To remove this workaround, delete this
# adapter and construct `OpenAIEmbeddings` directly in `e2e/eval_e2e.py` again.
class NormalizingEmbeddings(Embeddings):
    """Apply the apostrophe workaround before forwarding RAGAS embeddings."""

    def __init__(self, delegate: Embeddings):
        self.delegate = delegate

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Normalize RAGAS document batches before the delegate's HTTP request.
        normalized = [normalize_for_embedding(text) for text in texts]
        return self.delegate.embed_documents(normalized)

    def embed_query(self, text: str) -> list[float]:
        # Normalize synchronous RAGAS query embeddings.
        return self.delegate.embed_query(normalize_for_embedding(text))

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        # Normalize asynchronous RAGAS document batches.
        normalized = [normalize_for_embedding(text) for text in texts]
        return await self.delegate.aembed_documents(normalized)

    async def aembed_query(self, text: str) -> list[float]:
        # Normalize asynchronous RAGAS query embeddings.
        return await self.delegate.aembed_query(normalize_for_embedding(text))

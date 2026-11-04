from __future__ import annotations

import numpy as np
import requests

# APOSTROPHE NORMALIZATION WORKAROUND: remove this import and the tagged line
# in `embed_query` to restore the original embedding request behavior.
from canar.app.api.text_normalization import normalize_for_embedding
from canar.app.retrieval.models import SparseVector


class EmbedClient:
    def __init__(self, base_url: str, model: str, api_key: str = ""):
        self.url = base_url.rstrip("/") + "/embeddings"
        self.model = model
        self.key = api_key

    def embed_query(self, text: str) -> list[float]:
        # APOSTROPHE NORMALIZATION WORKAROUND: only the HTTP payload is converted;
        # the original `text` remains unchanged for prompts, logs, and display.
        embedding_text = normalize_for_embedding(text)
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        r = requests.post(
            self.url,
            json={"model": self.model, "input": [embedding_text]},
            headers=headers,
            timeout=60,
        )
        r.raise_for_status()
        v = np.array(r.json()["data"][0]["embedding"], dtype="float32")
        v /= np.linalg.norm(v) + 1e-12
        return v.tolist()


class FastEmbedClient:
    def __init__(self, model_name: str):
        from fastembed import SparseTextEmbedding

        self.model = SparseTextEmbedding(model_name)

    def embed_query(self, text: str) -> SparseVector:
        v = next(self.model.query_embed(text))
        as_object = v.as_object()
        return SparseVector(
            indices=list(as_object["indices"]),
            values=list(as_object["values"]),
        )


OpenAiEmbedClient = EmbedClient

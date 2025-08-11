from __future__ import annotations
import requests
import numpy as np


class EmbedClient:
    def __init__(self, base_url: str, model: str, api_key: str = ""):
        self.url = base_url.rstrip("/") + "/embeddings"
        self.model = model
        self.key = api_key

    def embed_query(self, text: str) -> list[float]:
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        r = requests.post(self.url, json={"model": self.model, "input": [text]},
                          headers=headers, timeout=60)
        r.raise_for_status()
        v = np.array(r.json()["data"][0]["embedding"], dtype="float32")
        v /= (np.linalg.norm(v) + 1e-12)
        return v.tolist()

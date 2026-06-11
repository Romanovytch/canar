"""
Ingest utilitR markdown files into Qdrant.

Run:
    source .venv/bin/activate
    pip install -r requirements.txt
    python ingest.py
"""

import os
import re
import uuid
from pathlib import Path

import requests
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

load_dotenv()

QDRANT_URL = os.environ["QDRANT_URL"]
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")
COLLECTION = os.environ.get("QDRANT_COLLECTION", "utilitr")

EMBED_API_BASE = os.environ["EMBED_API_BASE"].rstrip("/")
EMBED_API_KEY = os.environ.get("EMBED_API_KEY", "EMPTY")
EMBED_MODEL = os.environ["EMBED_MODEL"]

UTILITR_PATH = Path(os.environ.get("UTILITR_PATH", "/home/cereq/opt/utilitR"))

CHUNK_SIZE = 800       # chars (approximate — no tokenizer needed)
CHUNK_OVERLAP = 150
BATCH_SIZE = 16

EXCLUDE_DIRS = {".git", "_book", "docs", ".quarto", "renv", ".github", "node_modules"}
INCLUDE_GLOBS = ["**/*.md", "**/*.qmd", "**/*.Rmd"]


def find_files(root: Path) -> list[Path]:
    files = []
    for pattern in INCLUDE_GLOBS:
        for p in root.glob(pattern):
            if not any(part in EXCLUDE_DIRS for part in p.parts):
                files.append(p)
    return sorted(set(files))


def clean_text(text: str) -> str:
    # Remove YAML frontmatter
    text = re.sub(r"^---\n.*?\n---\n", "", text, flags=re.DOTALL)
    # Remove Quarto/RMarkdown directives
    text = re.sub(r"^:::\s*\{.*?\}\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^:::.*$", "", text, flags=re.MULTILINE)
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Collapse excess blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, path: str) -> list[dict]:
    chunks = []
    paragraphs = re.split(r"\n\n+", text)
    current = ""
    for para in paragraphs:
        if len(current) + len(para) > CHUNK_SIZE and current:
            chunks.append(current.strip())
            # keep overlap from end of previous chunk
            current = current[-CHUNK_OVERLAP:] + "\n\n" + para
        else:
            current = current + "\n\n" + para if current else para
    if current.strip():
        chunks.append(current.strip())

    return [
        {"text": c, "path": path, "chunk_index": i}
        for i, c in enumerate(chunks)
        if len(c) > 50  # skip near-empty chunks
    ]


def embed_batch(texts: list[str]) -> list[list[float]]:
    resp = requests.post(
        f"{EMBED_API_BASE}/embeddings",
        headers={"Authorization": f"Bearer {EMBED_API_KEY}"},
        json={"model": EMBED_MODEL, "input": texts},
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()["data"]
    return [item["embedding"] for item in sorted(data, key=lambda x: x["index"])]


def get_vector_size() -> int:
    sample = embed_batch(["test"])
    return len(sample[0])


def main():
    print(f"Scanning {UTILITR_PATH} ...")
    files = find_files(UTILITR_PATH)
    print(f"Found {len(files)} files")
    if not files:
        raise SystemExit(f"No markdown files found under {UTILITR_PATH} — check UTILITR_PATH")

    all_chunks = []
    for f in files:
        text = clean_text(f.read_text(errors="ignore"))
        rel = str(f.relative_to(UTILITR_PATH))
        all_chunks.extend(chunk_text(text, rel))

    print(f"Total chunks: {len(all_chunks)}")

    print("Detecting embedding dimension ...")
    dim = get_vector_size()
    print(f"Embedding dim: {dim}")

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None)

    client.recreate_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    print(f"Collection '{COLLECTION}' (re)created")

    points = []
    for i in range(0, len(all_chunks), BATCH_SIZE):
        batch = all_chunks[i : i + BATCH_SIZE]
        # nomic-embed-text requires task prefixes; other models (e.g. bge-m3) don't
        prefix = "search_document: " if "nomic" in EMBED_MODEL else ""
        texts = [prefix + c["text"] for c in batch]
        vectors = embed_batch(texts)
        for chunk, vector in zip(batch, vectors):
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload=chunk,
                )
            )
        done = min(i + BATCH_SIZE, len(all_chunks))
        print(f"  Embedded {done}/{len(all_chunks)}", end="\r")

    print()
    client.upsert(collection_name=COLLECTION, points=points)
    print(f"Done — {len(points)} points in '{COLLECTION}'")


if __name__ == "__main__":
    main()

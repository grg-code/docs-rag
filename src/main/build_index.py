import json
import os
import pickle
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

DATA_DIR = Path("data")
CHUNKS_PATH = DATA_DIR / "chunks.jsonl"
INDEX_DIR = Path("vector_store")
INDEX_DIR.mkdir(parents=True, exist_ok=True)

EMBED_MODEL = "text-embedding-3-small"  # 1536 dims;
BATCH = 128


def embed_texts(client: OpenAI, texts: list[str]) -> np.ndarray:
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    vecs = [d.embedding for d in resp.data]
    return np.array(vecs, dtype="float32")


def main():
    load_dotenv()

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError("Run chunk_docs.py first: data/chunks.jsonl not found")

    ids, rows = [], []
    texts = []

    # 1) Read chunks.jsonl
    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            ids.append(rec["id"])

            # concatenate section + text for embedding
            prefix = (rec.get("section") or "").strip()
            body = rec["text"]
            embed_input = (prefix + "\n\n" + body) if prefix else body
            texts.append(embed_input)

    # 2) Embed in batches
    all_vecs = []
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i + BATCH]
        vecs = embed_texts(client, batch)
        all_vecs.append(vecs)
    X = np.vstack(all_vecs)  # shape: (N, D)

    # 3) Build FAISS index (L2 with normalized vectors ≈ cosine)
    # normalize for cosine similarity
    faiss.normalize_L2(X)
    d = X.shape[1]
    index = faiss.IndexFlatIP(d)  # inner product after normalize = cosine
    index.add(X)

    # 4) Save index + metadata
    faiss.write_index(index, str(INDEX_DIR / "index.faiss"))
    meta = {
        "ids": ids,
        "model": EMBED_MODEL,
    }
    with open(INDEX_DIR / "meta.pkl", "wb") as f:
        pickle.dump(meta, f)

    print(f"✅ Built FAISS index with {len(ids)} vectors @ {INDEX_DIR}")


if __name__ == "__main__":
    main()

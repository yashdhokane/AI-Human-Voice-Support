# ingest.py
import os
from typing import List, Dict, Any

from config import get_or_create_index
from embeddings import embed_text

DOCS_DIR = r"C:\Users\Dell\T-RAG\data"
CHUNK_SIZE = 1000       # characters
CHUNK_OVERLAP = 200     # characters

def load_text_docs(path: str = DOCS_DIR) -> List[Dict[str, Any]]:
    docs = []
    for fname in os.listdir(path):
        if not fname.lower().endswith(".txt"):
            continue
        full_path = os.path.join(path, fname)
        with open(full_path, "r", encoding="utf-8") as f:
            content = f.read()
        docs.append(
            {
                "id": fname,
                "content": content,
                "source": full_path,
            }
        )
    return docs

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = end - overlap  # slide with overlap
    return chunks

def build_vectors(docs: List[Dict[str, Any]]):
    vectors = []
    for d in docs:
        chunks = chunk_text(d["content"])
        for i, chunk in enumerate(chunks):
            emb = embed_text(chunk)
            vectors.append(
                {
                    "id": f"{d['id']}_chunk_{i}",
                    "values": emb,
                    "metadata": {
                        "text": chunk,
                        "source": d["source"],
                        "doc_id": d["id"],
                        "chunk_index": i,
                    },
                }
            )
    return vectors

def ingest():
    index = get_or_create_index(dimension=768)
    docs = load_text_docs()
    vectors = build_vectors(docs)
    index.upsert(vectors=vectors)
    print(f"Ingested {len(vectors)} chunks into Pinecone index.")

if __name__ == "__main__":
    ingest()

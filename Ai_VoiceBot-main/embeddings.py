import ollama
from config import EMBED_MODEL

def embed_text(text: str) -> list[float]:
    """
    Return an embedding vector for the given text using Ollama.
    """
    res = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    return res["embedding"]

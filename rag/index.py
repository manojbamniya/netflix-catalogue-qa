"""Embeddings: turn every title into numbers once, then find titles like a question.

Build the index (after python ingest.py):
    python -m rag.index
"""

import os
import sqlite3
from functools import lru_cache
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

DB_PATH = Path(os.getenv("NETFLIX_DB_PATH", "data/netflix.db"))
INDEX_PATH = Path(os.getenv("NETFLIX_INDEX_PATH", "data/embeddings.npz"))
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")  # small and free


@lru_cache(maxsize=1)
def load_model() -> SentenceTransformer:
    """Load the embedding model once (it's downloaded the first time)."""
    return SentenceTransformer(MODEL_NAME)


def embed(texts: list[str]) -> np.ndarray:
    """Turn each text into 384 numbers, scaled to length 1.

    With length-1 vectors, a dot product is the cosine similarity.
    """
    return load_model().encode(texts, normalize_embeddings=True)


def document(title: dict) -> str:
    """The text we embed for one title. Title and country go first, because the
    model only reads about the first 256 word pieces."""
    return (
        f"{title['title']} ({title['type']}, {title['year']}). "
        f"Country: {title['country'] or 'Unknown'}. "
        f"Rating: {title['rating'] or 'Unrated'}. {title['movie_description']}"
    )


def build_index(db_path: Path, index_path: Path) -> int:
    """Embed every title and save show_ids + vectors. Returns how many."""
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        titles = [dict(row) for row in conn.execute("SELECT * FROM titles")]
    vectors = embed([document(t) for t in titles])
    np.savez(index_path, ids=[t["show_id"] for t in titles], vectors=vectors)
    return len(titles)


def load_index(index_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read back (show_ids, vectors) saved by build_index."""
    data = np.load(index_path)
    return data["ids"], data["vectors"]


def search(question: str, ids: np.ndarray, vectors: np.ndarray, k: int) -> list[str]:
    """Return the show_ids of the k titles most similar to the question."""
    scores = vectors @ embed([question])[0]  # one similarity score per title
    best = np.argsort(-scores)[:k]  # highest scores first
    return [str(ids[i]) for i in best]


if __name__ == "__main__":
    count = build_index(DB_PATH, INDEX_PATH)
    print(f"Embedded {count} titles into {INDEX_PATH}")

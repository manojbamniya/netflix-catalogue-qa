"""Find the titles most similar to a question (numpy cosine similarity)."""

from pathlib import Path

import numpy as np

from rag.index import Embedder


class Retriever:
    """Holds every title's vector in memory and searches them by similarity.

    For ~6,000 titles x 384 numbers (~9 MB) a full scan takes a few
    milliseconds, so no vector database is needed.
    """

    def __init__(self, ids: np.ndarray, vectors: np.ndarray, embed: Embedder):
        self.ids = ids
        self.vectors = vectors
        self.embed = embed

    @classmethod
    def load(cls, index_path: Path, embed: Embedder) -> "Retriever":
        """Load the index written by `python -m rag.index`."""
        data = np.load(index_path)
        return cls(ids=data["ids"], vectors=data["vectors"], embed=embed)

    def search(self, question: str, k: int) -> list[str]:
        """Return the show_ids of the k most similar titles, best first."""
        query = self.embed([question])[0]
        scores = self.vectors @ query  # cosine similarity: vectors are normalised
        best = np.argsort(-scores)[:k]
        return [str(self.ids[i]) for i in best]

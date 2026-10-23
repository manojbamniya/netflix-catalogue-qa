"""Build the embedding index: one vector per title, saved to data/embeddings.npz.

Run after ingest.py (takes a minute or two the first time, while the model
downloads):
    python -m rag.index
"""

import logging
import sqlite3
from collections.abc import Callable
from pathlib import Path

import numpy as np

from rag import settings

logger = logging.getLogger(__name__)

# A function that turns texts into one normalised vector per text.
Embedder = Callable[[list[str]], np.ndarray]

BATCH_SIZE = 64


def load_embedder(model_name: str = settings.EMBEDDING_MODEL) -> Embedder:
    """Load the sentence-transformers model and return an embed(texts) function.

    Vectors are normalised to length 1, so a dot product is cosine similarity.
    """
    from sentence_transformers import SentenceTransformer  # slow import

    model = SentenceTransformer(model_name)

    def embed(texts: list[str]) -> np.ndarray:
        return model.encode(
            texts, batch_size=BATCH_SIZE, normalize_embeddings=True
        ).astype(np.float32)

    return embed


def build_document(title: dict) -> str:
    """The text we embed for one title.

    Title, type, year and country go first: the model only reads the first
    ~256 word pieces, and "a comedy from India" needs the country to match.
    """
    country = title["country"] or "Unknown"
    rating = title["rating"] or "Unrated"
    return (
        f"{title['title']} ({title['type']}, {title['year']}). "
        f"Country: {country}. Rating: {rating}. {title['movie_description']}"
    )


def load_titles(db_path: Path) -> list[dict]:
    """Read every title from the database, in show_id order."""
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM titles ORDER BY show_id").fetchall()
    return [dict(row) for row in rows]


def build_index(db_path: Path, index_path: Path, embed: Embedder) -> int:
    """Embed every title and save show_ids + vectors. Returns the title count."""
    titles = load_titles(db_path)
    documents = [build_document(t) for t in titles]
    vectors = embed(documents)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(index_path, ids=np.array([t["show_id"] for t in titles]), vectors=vectors)
    return len(titles)


def main() -> None:
    """Build the index from the database configured in settings."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if not settings.DB_PATH.exists():
        raise SystemExit(f"{settings.DB_PATH} not found - run `python ingest.py` first")

    logger.info("Loading embedding model %s", settings.EMBEDDING_MODEL)
    embed = load_embedder()
    count = build_index(settings.DB_PATH, settings.INDEX_PATH, embed)
    print(f"Embedded {count} titles into {settings.INDEX_PATH}")


if __name__ == "__main__":
    main()

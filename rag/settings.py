"""RAG configuration, read from environment variables (or a .env file)."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # reads .env in the project root, if there is one

DB_PATH = Path(os.getenv("NETFLIX_DB_PATH", "data/netflix.db"))
INDEX_PATH = Path(os.getenv("NETFLIX_INDEX_PATH", "data/embeddings.npz"))

# Small, fast sentence-transformers model; 384 numbers per text.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

TOP_K = 8  # titles retrieved per question and passed to the LLM

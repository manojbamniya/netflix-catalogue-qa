"""FastAPI app for the Netflix catalogue.

Run (after python ingest.py and python -m rag.index):
    uvicorn api.main:app --reload
Then open http://127.0.0.1:8000/docs to try it.
"""

import logging
import os
import sqlite3
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from google.genai import errors as genai_errors
from pydantic import ValidationError

from api import queries
from api.schemas import AskRequest, AskResponse, Stats, Title, TitlePage
from rag import answer, index

logger = logging.getLogger(__name__)

DB_PATH = Path(os.getenv("NETFLIX_DB_PATH", "data/netflix.db"))
MAX_PAGE_SIZE = 100
DEFAULT_PAGE_SIZE = 20
MIN_YEAR, MAX_YEAR = 1900, 2100  # sane range for ?release_year=
TOP_K = 15  # titles retrieved per question and given to Gemini

app = FastAPI(title="Netflix Catalogue API")


def connect(db_path: Path) -> sqlite3.Connection:
    """Open the database read-only; rows can be read by column name.

    check_same_thread=False: FastAPI may open it in one thread and use it in
    another. Each request still gets its own connection.
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def get_connection() -> Iterator[sqlite3.Connection]:
    """Give each request a connection, and close it afterwards."""
    if not DB_PATH.exists():
        raise HTTPException(503, f"{DB_PATH} not found - run `python ingest.py`")
    conn = connect(DB_PATH)
    try:
        yield conn
    finally:
        conn.close()


@lru_cache(maxsize=1)
def get_index() -> tuple[np.ndarray, np.ndarray]:
    """Load the embeddings once, on the first /ask."""
    if not index.INDEX_PATH.exists():
        raise HTTPException(503, "Index not found - run `python -m rag.index`")
    return index.load_index(index.INDEX_PATH)


def get_llm() -> answer.AskLLM:
    """The function /ask uses to call the LLM (tests swap in a fake one)."""
    if not answer.GEMINI_API_KEY:
        raise HTTPException(503, "GEMINI_API_KEY is not set - add it to .env")
    return answer.ask_gemini


Connection = Annotated[sqlite3.Connection, Depends(get_connection)]


@app.exception_handler(RequestValidationError)
async def bad_input_is_400(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Return 400 (FastAPI's default is 422) and name each bad parameter."""
    errors = [{"param": e["loc"][-1], "message": e["msg"]} for e in exc.errors()]
    return JSONResponse(status_code=400, content={"detail": errors})


@app.get("/titles")
def list_titles(
    conn: Connection,
    country: Annotated[str | None, Query(min_length=1)] = None,
    release_year: Annotated[int | None, Query(ge=MIN_YEAR, le=MAX_YEAR)] = None,
    type: Literal["Movie", "TV Show"] | None = None,
    rating: Annotated[str | None, Query(min_length=1)] = None,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = DEFAULT_PAGE_SIZE,
) -> TitlePage:
    """List titles matching all the given filters, one page at a time.

    country=Unknown returns titles with no country.
    """
    filters = {"country": country, "year": release_year, "type": type, "rating": rating}
    offset = (page - 1) * page_size
    total, rows = queries.list_titles(conn, filters, limit=page_size, offset=offset)
    return TitlePage(total=total, page=page, page_size=page_size, items=rows)


@app.get("/titles/{show_id}")
def get_title(show_id: str, conn: Connection) -> Title:
    """One title by show_id, or 404."""
    row = queries.get_title(conn, show_id)
    if row is None:
        raise HTTPException(404, f"No title with show_id {show_id}")
    return Title(**row)


@app.get("/stats")
def get_stats(conn: Connection) -> Stats:
    """Total titles, count by type, top 10 countries, titles with no country."""
    return Stats(**queries.get_stats(conn))


@app.post("/ask")
def ask(
    body: AskRequest,
    conn: Connection,
    saved_index: Annotated[tuple[np.ndarray, np.ndarray], Depends(get_index)],
    llm: Annotated[answer.AskLLM, Depends(get_llm)],
) -> AskResponse:
    """Answer a question from the catalogue, with the show_ids it used."""
    ids, vectors = saved_index
    retrieved = index.search(body.question, ids, vectors, k=TOP_K)
    try:
        result = answer.answer_question(body.question, retrieved, conn, llm)
    except (genai_errors.APIError, ValidationError) as exc:
        # APIError: Gemini failed. ValidationError: its reply wasn't our JSON.
        logger.error("LLM request failed: %s", exc)
        if getattr(exc, "code", None) == 429:  # free quota used up (resets daily)
            raise HTTPException(503, "Gemini quota used up - try later") from exc
        raise HTTPException(
            502, "The language model request failed - try again"
        ) from exc
    return AskResponse(**result)

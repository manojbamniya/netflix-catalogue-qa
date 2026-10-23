"""FastAPI app: list, fetch and summarise Netflix titles.

Run from the project root (after python ingest.py):
    uvicorn api.main:app --reload
Then open http://127.0.0.1:8000/docs to try the endpoints.
"""

import logging
import sqlite3
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api import queries
from api.database import DB_PATH, get_connection
from api.schemas import Stats, Title, TitlePage, TitleType

logger = logging.getLogger(__name__)

DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 100
TOP_COUNTRIES = 10
MIN_YEAR = 1900
MAX_YEAR = 2100


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Refuse to start if the database hasn't been built yet."""
    if not DB_PATH.exists():
        raise RuntimeError(f"{DB_PATH} not found - run `python ingest.py` first")
    logger.info("Using database %s", DB_PATH)
    yield


app = FastAPI(title="Netflix Catalogue API", lifespan=lifespan)

Connection = Annotated[sqlite3.Connection, Depends(get_connection)]


@app.exception_handler(RequestValidationError)
async def validation_error_as_400(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Return 400 (not FastAPI's default 422) for bad query parameters."""
    errors = [
        {"param": ".".join(str(p) for p in e["loc"][1:]), "message": e["msg"]}
        for e in exc.errors()
    ]
    return JSONResponse(status_code=400, content={"detail": errors})


@app.get("/titles", response_model=TitlePage)
def list_titles(
    conn: Connection,
    country: Annotated[
        str | None,
        Query(min_length=1, description='e.g. "India", or "Unknown" for no country'),
    ] = None,
    release_year: Annotated[int | None, Query(ge=MIN_YEAR, le=MAX_YEAR)] = None,
    type: Annotated[TitleType | None, Query()] = None,
    rating: Annotated[str | None, Query(min_length=1, description="e.g. TV-MA")] = None,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = DEFAULT_PAGE_SIZE,
) -> TitlePage:
    """List titles matching all given filters, one page at a time."""
    filters = queries.TitleFilters(
        country=country,
        release_year=release_year,
        type=type.value if type else None,
        rating=rating,
    )
    offset = (page - 1) * page_size
    total, rows = queries.list_titles(conn, filters, limit=page_size, offset=offset)
    return TitlePage(
        total=total,
        page=page,
        page_size=page_size,
        items=[Title(**row) for row in rows],
    )


@app.get("/titles/{show_id}", response_model=Title)
def get_title(show_id: str, conn: Connection) -> Title:
    """Fetch one title by show_id; 404 if it doesn't exist."""
    row = queries.get_title(conn, show_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"No title with show_id {show_id}")
    return Title(**row)


@app.get("/stats", response_model=Stats)
def get_stats(conn: Connection) -> Stats:
    """Total titles, count by type, top 10 countries, titles with no country."""
    return Stats(**queries.get_stats(conn, top_n=TOP_COUNTRIES))

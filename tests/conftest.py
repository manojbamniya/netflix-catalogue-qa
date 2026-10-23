"""Shared test setup: a tiny database and a TestClient that reads it."""

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api.main
import ingest
from api.database import get_connection, open_connection

TITLES = [
    # show_id, type, title, country, year, rating
    ("1", "Movie", "Andhadhun", "India", 2018, "TV-MA"),
    ("2", "Movie", "Brick Lane", "United Kingdom, India", 2007, "R"),
    ("3", "TV Show", "Chhota Bheem", "India", 2019, "TV-Y7"),
    ("4", "Movie", "Diego Garcia", "British Indian Ocean Territory", 2018, "TV-MA"),
    ("5", "Movie", "Empty Country", None, 2018, "TV-MA"),
    ("6", "TV Show", "Fargo", "United States", 2014, "TV-MA"),
]


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Build the tiny database with ingest.write_database and return its path."""
    df = pd.DataFrame(
        TITLES, columns=["show_id", "type", "title", "country", "year", "rating"]
    ).assign(movie_description="Some text.")
    path = tmp_path / "test.db"
    ingest.write_database(df, path)
    return path


@pytest.fixture
def client(db_path: Path) -> Iterator[TestClient]:
    """A test client whose endpoints read the tiny database."""

    def test_connection() -> Iterator[sqlite3.Connection]:
        conn = open_connection(db_path)
        try:
            yield conn
        finally:
            conn.close()

    api.main.app.dependency_overrides[get_connection] = test_connection
    yield TestClient(api.main.app)
    api.main.app.dependency_overrides.clear()

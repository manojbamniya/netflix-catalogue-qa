"""Shared test setup: a tiny database, and a test client that reads it."""

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import ingest
from api import main

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
    """Build the tiny database with ingest.write_database."""
    df = pd.DataFrame(
        TITLES, columns=["show_id", "type", "title", "country", "year", "rating"]
    ).assign(movie_description="Some text.")
    path = tmp_path / "test.db"
    ingest.write_database(df, ingest.split_countries(df), path)
    return path


@pytest.fixture
def conn(db_path: Path) -> Iterator[sqlite3.Connection]:
    """A connection to the tiny database."""
    connection = main.connect(db_path)
    yield connection
    connection.close()


@pytest.fixture
def client(db_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """A test client whose endpoints read the tiny database."""
    monkeypatch.setattr(main, "DB_PATH", db_path)
    yield TestClient(main.app)
    main.app.dependency_overrides.clear()

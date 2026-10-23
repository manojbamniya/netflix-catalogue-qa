"""Tests for the API, using a tiny database built with ingest.write_database."""

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from httpx2 import Response

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
def client(tmp_path: Path) -> Iterator[TestClient]:
    """A test client whose endpoints read a small database in tmp_path."""
    df = pd.DataFrame(
        TITLES, columns=["show_id", "type", "title", "country", "year", "rating"]
    ).assign(movie_description="Some text.")
    db_path = tmp_path / "test.db"
    ingest.write_database(df, db_path)

    def test_connection() -> Iterator[sqlite3.Connection]:
        conn = open_connection(db_path)
        try:
            yield conn
        finally:
            conn.close()

    api.main.app.dependency_overrides[get_connection] = test_connection
    yield TestClient(api.main.app)
    api.main.app.dependency_overrides.clear()


def ids(response: Response) -> list[str]:
    """The show_ids on one page of GET /titles."""
    return [item["show_id"] for item in response.json()["items"]]


def test_country_filter_matches_whole_country_names_only(client: TestClient) -> None:
    response = client.get("/titles", params={"country": "india"})

    assert response.status_code == 200
    # Includes the two-country title, excludes "British Indian Ocean Territory".
    assert ids(response) == ["1", "2", "3"]
    assert response.json()["total"] == 3


def test_country_unknown_returns_titles_without_country(client: TestClient) -> None:
    response = client.get("/titles", params={"country": "Unknown"})

    assert ids(response) == ["5"]


def test_filters_combine(client: TestClient) -> None:
    params = {
        "country": "India",
        "type": "Movie",
        "release_year": 2018,
        "rating": "tv-ma",
    }

    response = client.get("/titles", params=params)

    assert ids(response) == ["1"]


def test_pagination_splits_results_without_overlap(client: TestClient) -> None:
    pages = [
        client.get("/titles", params={"page": page, "page_size": 4})
        for page in (1, 2, 3)
    ]

    assert [ids(p) for p in pages] == [["1", "2", "3", "4"], ["5", "6"], []]
    assert all(p.json()["total"] == len(TITLES) for p in pages)


@pytest.mark.parametrize(
    "params, bad_param",
    [
        ({"type": "Film"}, "type"),
        ({"release_year": "abc"}, "release_year"),
        ({"page": 0}, "page"),
        ({"page_size": 1000}, "page_size"),
        ({"country": ""}, "country"),
    ],
)
def test_bad_query_params_return_400(
    client: TestClient, params: dict, bad_param: str
) -> None:
    response = client.get("/titles", params=params)

    assert response.status_code == 400
    assert [error["param"] for error in response.json()["detail"]] == [bad_param]


def test_get_title_by_id_and_404_for_unknown_id(client: TestClient) -> None:
    found = client.get("/titles/6")
    missing = client.get("/titles/999")

    assert found.status_code == 200
    assert found.json()["title"] == "Fargo"
    assert missing.status_code == 404


def test_stats_counts_and_leaves_missing_country_out_of_top(client: TestClient) -> None:
    stats = client.get("/stats").json()

    assert stats["total_titles"] == 6
    assert stats["count_by_type"] == {"Movie": 4, "TV Show": 2}
    assert stats["top_countries"][0] == {"country": "India", "count": 3}
    assert all(c["country"] for c in stats["top_countries"])
    assert stats["titles_without_country"] == 1


def test_app_refuses_to_start_without_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(api.main, "DB_PATH", tmp_path / "missing.db")

    with pytest.raises(RuntimeError, match="run `python ingest.py` first"):
        with TestClient(api.main.app):
            pass

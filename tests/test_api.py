"""Tests for the API. The tiny database and `client` fixture are in conftest.py."""

from pathlib import Path

import pytest
from conftest import TITLES
from fastapi.testclient import TestClient
from httpx2 import Response

from api import main


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
    assert list(stats["top_countries"].items())[0] == ("India", 3)
    assert None not in stats["top_countries"]
    assert stats["titles_without_country"] == 1


def test_missing_database_returns_503(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(main, "DB_PATH", tmp_path / "missing.db")

    response = client.get("/stats")

    assert response.status_code == 503
    assert "run `python ingest.py`" in response.json()["detail"]

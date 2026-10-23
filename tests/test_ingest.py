"""Tests for ingest.py."""

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

import ingest


def test_read_csv_trims_spaces_and_blank_cells_become_missing(tmp_path: Path) -> None:
    path = tmp_path / "titles.csv"
    path.write_text('show_id,country,date_added\n1,   ," June 13, 2019"\n')

    df = ingest.read_csv(path)

    assert pd.isna(df.loc[0, "country"])
    assert df.loc[0, "date_added"] == "June 13, 2019"


def test_split_countries_one_row_each_and_none_for_missing() -> None:
    movies = pd.DataFrame(
        {
            "show_id": ["1", "2", "3"],
            "country": ["United States, India", None, "Poland,"],
        }
    )

    countries = ingest.split_countries(movies)

    assert sorted(countries.itertuples(index=False, name=None)) == [
        ("1", "India"),
        ("1", "United States"),
        ("3", "Poland"),
    ]


def test_drop_reason() -> None:
    kept_titles = {"Sarkar"}
    no_rating = pd.Series({"title": "A", "rating": None, "cast": "X"})
    too_empty = pd.Series(
        {"title": "B", "rating": "TV-Y", "a": None, "b": None, "c": None}
    )
    duplicate = pd.Series({"title": "Sarkar", "rating": "TV-MA"})

    assert ingest.drop_reason(no_rating, kept_titles) == "no rating"
    assert ingest.drop_reason(too_empty, kept_titles) == "3 or more empty fields"
    assert ingest.drop_reason(duplicate, kept_titles) == "duplicate of a kept title"


def test_main_writes_database_and_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    original = tmp_path / "titles.csv"
    original.write_text(
        "show_id,title,rating,cast,director,country\n"
        '1,Norm,TV-PG,Alan Marriott,,"United States, India"\n'
        "2,No Country,TV-14,Jo,,\n"
        '3,Hotel Pacific,R,Ann,Bo,"Poland,"\n'
        "4,No Rating,,Jo,,India\n"
    )
    cleaned = tmp_path / "movies.csv"
    cleaned.write_text(
        "show_id,type,title,country,date_added,year,rating,movie_description\n"
        '1,Movie,Norm,"United States, India",,2019,TV-PG,Text1\n'
        "2,TV Show,No Country,,,2018,TV-14,Text2\n"
        '3,Movie,Hotel Pacific,"Poland,",,1975,R,Text3\n'
    )
    db = tmp_path / "netflix.db"
    monkeypatch.setattr(ingest, "ORIGINAL_CSV", original)
    monkeypatch.setattr(ingest, "CLEANED_CSV", cleaned)
    monkeypatch.setattr(ingest, "DB_PATH", db)

    ingest.main()

    with sqlite3.connect(db) as conn:
        titles = conn.execute(
            "SELECT show_id, country, year FROM titles ORDER BY show_id"
        ).fetchall()
        countries = conn.execute(
            "SELECT show_id, country FROM title_countries ORDER BY country"
        ).fetchall()
    assert titles == [
        ("1", "United States, India", 2019),
        ("2", None, 2018),
        ("3", "Poland", 1975),  # trailing comma removed
    ]
    assert countries == [("1", "India"), ("3", "Poland"), ("1", "United States")]

    summary = capsys.readouterr().out
    assert "Rows dropped:         1" in summary
    assert "1  no rating" in summary
    assert "Fixed: 1 countries with a trailing comma" in summary
    assert "1 titles have no country" in summary

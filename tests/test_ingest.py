"""Tests for ingest.py, using tiny CSV files written to a temp folder."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd
import pytest

import ingest

ORIGINAL_COLUMNS = [
    "show_id",
    "type",
    "title",
    "director",
    "cast",
    "country",
    "date_added",
    "release_year",
    "rating",
    "duration",
    "listed_in",
    "description",
]


def original_row(show_id: str, title: str, **overrides: str) -> dict[str, str]:
    """One row of netflix_titles.csv with sensible defaults."""
    row = {
        "show_id": show_id,
        "type": "Movie",
        "title": title,
        "director": "Richard Finn",
        "cast": "Alan Marriott, Andrew Toth",
        "country": "United States",
        "date_added": "September 9, 2019",
        "release_year": "2019",
        "rating": "TV-PG",
        "duration": "90 min",
        "listed_in": "Comedies",
        "description": "A polar bear king.",
    }
    return {**row, **overrides}


def cleaned_row(show_id: str, title: str, **overrides: str) -> dict[str, str]:
    """One row of netflix_movies.csv with sensible defaults."""
    row = {
        "show_id": show_id,
        "type": "Movie",
        "title": title,
        "country": "United States",
        "year": "2019",
        "rating": "TV-PG",
        "movie_description": "A polar bear king. Comedies 90min AlanMarriott",
    }
    return {**row, **overrides}


def write_csv(path: Path, rows: list[dict[str, str]]) -> Path:
    """Write rows to a CSV file and return its path."""
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def run_ingest(
    tmp_path: Path,
    original: list[dict[str, str]],
    cleaned: list[dict[str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    """Run ingest.main() on the given rows and return the database path."""
    original_csv = write_csv(tmp_path / "titles.csv", original)
    cleaned_csv = write_csv(tmp_path / "movies.csv", cleaned)
    db = tmp_path / "out" / "netflix.db"
    argv = ["ingest.py", "--csv", str(cleaned_csv), "--titles-csv", str(original_csv)]
    monkeypatch.setattr(sys, "argv", [*argv, "--db", str(db)])
    ingest.main()
    return db


def test_cells_are_stripped_and_blank_cells_become_missing(tmp_path: Path) -> None:
    path = tmp_path / "titles.csv"
    path.write_text('show_id,country,date_added\n1,   ," June 13, 2019"\n')

    df = ingest.read_text_csv(path)

    assert pd.isna(df.loc[0, "country"])
    assert df.loc[0, "date_added"] == "June 13, 2019"


def test_missing_required_column_fails_clearly(tmp_path: Path) -> None:
    rows = [cleaned_row("1", "A")]
    for row in rows:
        del row["movie_description"]
    path = write_csv(tmp_path / "movies.csv", rows)

    with pytest.raises(ValueError, match="movie_description"):
        ingest.load_cleaned_csv(path)


def test_non_numeric_year_fails_clearly() -> None:
    df = pd.DataFrame([cleaned_row("1", "A"), cleaned_row("2", "B", year="soon")])

    with pytest.raises(ValueError, match="show_id \\['2'\\]"):
        ingest.parse_year(df)


def test_country_table_has_one_row_per_country_and_none_for_missing() -> None:
    df = pd.DataFrame(
        {
            "show_id": ["1", "2", "3"],
            "country": ["United States, India", None, "Poland,"],
        }
    )

    countries = ingest.build_country_table(df)

    assert sorted(countries.itertuples(index=False, name=None)) == [
        ("1", "India"),
        ("1", "United States"),
        ("3", "Poland"),
    ]


def test_dropped_rows_are_grouped_by_reason() -> None:
    original = pd.DataFrame(
        [
            original_row("1", "Sarkar"),
            original_row("2", "Sarkar"),  # same as 1 except show_id
            original_row("3", "No Rating", rating=None),
            original_row("4", "Nobody", director=None, cast=None, country=None),
            original_row("5", "Kikoriki", director=None, country=None),
            original_row("6", "Kept"),
            # Missing both: must be counted once, under "no rating" only.
            original_row(
                "7", "Both", rating=None, director=None, cast=None, country=None
            ),
        ],
        columns=ORIGINAL_COLUMNS,
    )
    cleaned = pd.DataFrame([cleaned_row("1", "Sarkar"), cleaned_row("6", "Kept")])
    report = ingest.IngestReport()

    ingest.report_dropped_rows(original, cleaned, report)

    assert report.dropped == {
        "duplicate listing (same as a kept title except show_id)": 1,
        "no rating": 2,
        "no director, cast or country": 1,
        "other reason": 1,
    }
    assert any("show_id 5" in note for note in report.anomalies)


def test_changed_country_is_counted_as_fixed() -> None:
    original = pd.DataFrame(
        [
            original_row("1", "Hotel Pacific", country="Poland,"),
            original_row("2", "Unchanged"),
            original_row("3", "Also Unchanged"),
            original_row("4", "No Country", country=None),
        ]
    )
    cleaned = pd.DataFrame(
        [
            cleaned_row("1", "Hotel Pacific", country="Poland"),
            cleaned_row("2", "Unchanged"),
            cleaned_row("3", "Also Unchanged"),
            cleaned_row("4", "No Country", country=None),
        ]
    )
    report = ingest.IngestReport()

    ingest.report_fixed_values(original, cleaned, report)

    assert list(report.fixed.values()) == [1]


def test_readable_parts_are_appended_and_missing_ones_skipped() -> None:
    original = pd.DataFrame(
        [
            original_row("1", "Norm"),
            original_row("2", "Stand-up", director=None),
        ]
    )
    cleaned = pd.DataFrame(
        [
            cleaned_row("1", "Norm", movie_description="Text1"),
            cleaned_row("2", "Stand-up", movie_description="Text2"),
        ]
    )

    result = ingest.add_readable_parts(cleaned, original, ingest.IngestReport())

    assert result["movie_description"].tolist() == [
        "Text1 Genres: Comedies. Duration: 90 min. "
        "Cast: Alan Marriott, Andrew Toth. Director: Richard Finn.",
        "Text2 Genres: Comedies. Duration: 90 min. Cast: Alan Marriott, Andrew Toth.",
    ]


def test_end_to_end_writes_database_with_null_country(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    original = [
        original_row("1", "Norm", country="United States, India"),
        original_row("2", "No Country", country=""),
        original_row("3", "No Rating", rating=""),
    ]
    cleaned = [
        cleaned_row("1", "Norm", country="United States, India"),
        cleaned_row("2", "No Country", country=""),
    ]

    db = run_ingest(tmp_path, original, cleaned, monkeypatch)

    with sqlite3.connect(db) as conn:
        titles = conn.execute(
            "SELECT show_id, country, year FROM titles ORDER BY show_id"
        ).fetchall()
        countries = conn.execute(
            "SELECT show_id, country FROM title_countries ORDER BY country"
        ).fetchall()
    assert titles == [("1", "United States, India", 2019), ("2", None, 2019)]
    assert countries == [("1", "India"), ("1", "United States")]

    summary = capsys.readouterr().out
    assert "Rows dropped during cleaning: 1" in summary
    assert "1 titles have no country (stored as NULL)" in summary

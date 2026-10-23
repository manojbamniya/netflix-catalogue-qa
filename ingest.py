"""Write the cleaned Netflix CSV into a SQLite database.

Input is data/netflix_movies.csv, which was cleaned beforehand (see NOTES.md):
director, cast, genres and duration are merged into movie_description. In that
file the spaces were removed ("AlanMarriott", "90min"), so the readable values
are read from data/netflix_titles.csv (matched by show_id) and added as
"Genres: ...", "Duration: ...", "Cast: ..." and "Director: ...".

This script does no cleaning of its own. It:
  1. loads the cleaned CSV and checks it has the expected columns
  2. compares it with the original CSV to report what was dropped and fixed
  3. adds the readable genres, duration, cast and director
  4. writes everything to SQLite, with one row per country in title_countries
  5. prints a summary

Usage:
    python ingest.py
    python ingest.py --csv data/netflix_movies.csv --titles-csv \
        data/netflix_titles.csv --db data/netflix.db

The paths can also be set with the NETFLIX_CSV_PATH, NETFLIX_TITLES_CSV_PATH and
NETFLIX_DB_PATH environment variables. Command-line flags win over env vars.
"""

from __future__ import annotations

import argparse
import logging
import os
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Defaults; override with env vars or --csv / --titles-csv / --db.
CSV_PATH = Path(os.getenv("NETFLIX_CSV_PATH", "data/netflix_movies.csv"))
TITLES_CSV_PATH = Path(os.getenv("NETFLIX_TITLES_CSV_PATH", "data/netflix_titles.csv"))
DB_PATH = Path(os.getenv("NETFLIX_DB_PATH", "data/netflix.db"))

MULTI_VALUE_SEPARATOR = ","
DATE_FORMAT = "%B %d, %Y"  # date_added in the original CSV, e.g. "September 9, 2019"
MAX_EXAMPLES = 5  # how many show_ids to list per anomaly in the summary

# Columns read from the original CSV and appended to movie_description, in this
# order, with the label used for each, e.g. "Cast: Alan Marriott, Andrew Toth."
READABLE_PARTS = {
    "listed_in": "Genres",
    "duration": "Duration",
    "cast": "Cast",
    "director": "Director",
}

# Columns the cleaned CSV must have; these are also the columns we store.
REQUIRED_COLUMNS = [
    "show_id",
    "type",
    "title",
    "country",
    "year",
    "rating",
    "movie_description",
]

SCHEMA_SQL = """
CREATE TABLE titles (
    show_id           TEXT PRIMARY KEY,
    type              TEXT NOT NULL,
    title             TEXT NOT NULL,
    country           TEXT,
    year              INTEGER NOT NULL,
    rating            TEXT,
    movie_description TEXT NOT NULL
);
CREATE INDEX idx_titles_type ON titles(type);
CREATE INDEX idx_titles_year ON titles(year);
CREATE INDEX idx_titles_rating ON titles(rating);

-- One row per (title, country), so "United States, India" becomes two rows.
-- Used for the country filter and the top-10 stats. Titles with no country
-- have no rows here.
CREATE TABLE title_countries (
    show_id TEXT NOT NULL REFERENCES titles(show_id),
    country TEXT NOT NULL,
    PRIMARY KEY (show_id, country)
);
CREATE INDEX idx_title_countries_country ON title_countries(country);
"""


@dataclass
class IngestReport:
    """Counts and notes collected while loading, printed at the end."""

    rows_loaded: int = 0
    original_rows: int = 0
    dropped: dict[str, int] = field(default_factory=dict)
    fixed: dict[str, int] = field(default_factory=dict)
    anomalies: list[str] = field(default_factory=list)


def read_text_csv(path: Path) -> pd.DataFrame:
    """Read a CSV as text, strip spaces, and turn empty cells into missing values."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df = df.apply(lambda col: col.str.strip())
    return df.mask(df == "")


def load_cleaned_csv(path: Path) -> pd.DataFrame:
    """Read the cleaned CSV and fail clearly if a required column is missing."""
    df = read_text_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return df[REQUIRED_COLUMNS]


def parse_year(df: pd.DataFrame) -> pd.DataFrame:
    """Store year as a number; stop if any year is missing or not a number."""
    years = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    bad = df.loc[years.isna(), "show_id"].tolist()
    if bad:
        raise ValueError(f"year is missing or not a number for show_id {bad}")
    return df.assign(year=years)


def examples(ids: pd.Series) -> str:
    """Format the first few show_ids for the summary, e.g. "81072516, 80159925"."""
    shown = ", ".join(ids.head(MAX_EXAMPLES))
    return shown + (", ..." if len(ids) > MAX_EXAMPLES else "")


def report_dropped_rows(
    original: pd.DataFrame, df: pd.DataFrame, report: IngestReport
) -> None:
    """Count rows in the original but not in the cleaned file, grouped by reason."""
    is_kept = original["show_id"].isin(df["show_id"])
    kept, dropped = original[is_kept], original[~is_kept]

    # A dropped row is a duplicate if it matches a kept row in every column
    # except show_id. Kept rows go first, so only the dropped copy is flagged.
    content = [c for c in original.columns if c != "show_id"]
    flagged = pd.concat([kept, dropped]).duplicated(subset=content)
    duplicate = pd.Series(flagged.iloc[len(kept) :].to_numpy(), index=dropped.index)

    # Each dropped row gets one reason, checked in this order.
    no_rating = ~duplicate & dropped["rating"].isna()
    no_people = (
        ~duplicate
        & ~no_rating
        & dropped[["director", "cast", "country"]].isna().all(axis=1)
    )
    other = ~(duplicate | no_rating | no_people)

    report.dropped = {
        "duplicate listing (same as a kept title except show_id)": int(duplicate.sum()),
        "no rating": int(no_rating.sum()),
        "no director, cast or country": int(no_people.sum()),
        "other reason": int(other.sum()),
    }
    if other.any():
        other_ids = examples(dropped.loc[other, "show_id"])
        report.anomalies.append(f"dropped for another reason: show_id {other_ids}")


def report_fixed_values(
    original: pd.DataFrame, df: pd.DataFrame, report: IngestReport
) -> None:
    """Count kept titles whose country differs from the original, e.g. "Poland,"."""
    before = original.set_index("show_id")["country"]
    after = df.set_index("show_id")["country"]
    before = before.reindex(after.index)
    changed = before.notna() & after.notna() & (before != after)
    report.fixed["country values changed (e.g. trailing comma removed)"] = int(
        changed.sum()
    )


def report_anomalies(
    original: pd.DataFrame, df: pd.DataFrame, report: IngestReport
) -> None:
    """Note things left in the data on purpose, so they're visible in the summary."""
    report.anomalies.append(
        f"{int(df['country'].isna().sum())} titles have no country (stored as NULL)"
    )

    kept = original[original["show_id"].isin(df["show_id"])]
    added = pd.to_datetime(kept["date_added"], format=DATE_FORMAT, errors="coerce")
    released = pd.to_numeric(kept["release_year"], errors="coerce")
    after_added = kept.loc[(released > added.dt.year).fillna(False), "show_id"]
    if len(after_added):
        report.anomalies.append(
            f"{len(after_added)} titles have release_year after date_added "
            f"(left as is): show_id {examples(after_added)}"
        )

    shared = df["title"].duplicated(keep=False)
    report.anomalies.append(
        f"{int(df.loc[shared, 'title'].nunique())} titles share a name with another "
        "title (different shows, e.g. 'Tiger'; kept)"
    )


def add_readable_parts(
    df: pd.DataFrame, original: pd.DataFrame, report: IngestReport
) -> pd.DataFrame:
    """Append "Genres: ...", "Duration: ...", "Cast: ..." and "Director: ...".

    Matched by show_id. The existing description is kept as is; a missing part
    is skipped.
    """
    parts = original.set_index("show_id")[list(READABLE_PARTS)]
    description = df["movie_description"]
    for column, label in READABLE_PARTS.items():
        found = df["show_id"].map(parts[column])
        has_value = found.notna()
        with_value = description + f" {label}: " + found + "."
        description = with_value.where(has_value, description)
        report.fixed[f"descriptions given readable {label.lower()}"] = int(
            has_value.sum()
        )
    return df.assign(movie_description=description)


def build_country_table(df: pd.DataFrame) -> pd.DataFrame:
    """Turn the comma-separated country column into one (show_id, country) row each."""
    countries = (
        df.set_index("show_id")["country"]
        .dropna()
        .str.split(MULTI_VALUE_SEPARATOR)
        .explode()
        .str.strip()
    )
    countries = countries[countries != ""]
    return countries.reset_index().drop_duplicates()


def write_database(df: pd.DataFrame, db_path: Path) -> dict[str, int]:
    """Rebuild the SQLite database from scratch and return row counts per table."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.unlink(missing_ok=True)
    countries = build_country_table(df)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(SCHEMA_SQL)
        df.to_sql("titles", conn, if_exists="append", index=False)
        countries.to_sql("title_countries", conn, if_exists="append", index=False)
    return {"titles": len(df), "title_countries": len(countries)}


def print_summary(
    report: IngestReport, table_counts: dict[str, int], db_path: Path
) -> None:
    """Print the end-of-run report: rows loaded, dropped, fixed, anomalies, tables."""
    print("\n=== Ingest summary ===")
    print(f"Original rows (netflix_titles.csv): {report.original_rows}")
    print(f"Rows loaded (cleaned CSV):          {report.rows_loaded}")
    print(f"Rows written to database:           {table_counts['titles']}")
    print(f"\nRows dropped during cleaning: {sum(report.dropped.values())}")
    for reason, count in report.dropped.items():
        print(f"  {count:>5}  {reason}")
    print("\nValues fixed or added:")
    for reason, count in report.fixed.items():
        print(f"  {count:>5}  {reason}")
    print("\nAnomalies:")
    for note in report.anomalies:
        print(f"  - {note}")
    print(f"\nDatabase written to {db_path}:")
    for table, count in table_counts.items():
        print(f"  {count:>6}  rows in {table}")


def parse_args() -> argparse.Namespace:
    """Read the input and database paths from the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", type=Path, default=CSV_PATH)
    parser.add_argument("--titles-csv", type=Path, default=TITLES_CSV_PATH)
    parser.add_argument("--db", type=Path, default=DB_PATH)
    return parser.parse_args()


def main() -> None:
    """Load the cleaned catalogue, store it in SQLite, then print a summary."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    logger.info("Reading %s", args.csv)
    df = parse_year(load_cleaned_csv(args.csv))
    logger.info("Reading %s", args.titles_csv)
    original = read_text_csv(args.titles_csv)
    report = IngestReport(rows_loaded=len(df), original_rows=len(original))

    report_dropped_rows(original, df, report)
    report_fixed_values(original, df, report)
    report_anomalies(original, df, report)
    df = add_readable_parts(df, original, report)

    logger.info("Writing %d titles to %s", len(df), args.db)
    table_counts = write_database(df, args.db)
    print_summary(report, table_counts, args.db)


if __name__ == "__main__":
    main()

"""Load my cleaned Netflix CSV into SQLite and print a summary.

Run: python ingest.py
"""

import os
import sqlite3
from pathlib import Path

import pandas as pd

CLEANED_CSV = Path(os.getenv("NETFLIX_CSV_PATH", "data/netflix_movies.csv"))
ORIGINAL_CSV = Path(os.getenv("NETFLIX_TITLES_CSV_PATH", "data/netflix_titles.csv"))
DB_PATH = Path(os.getenv("NETFLIX_DB_PATH", "data/netflix.db"))

# The columns stored in the database (netflix_movies.csv also has date_added,
# which nothing uses).
COLUMNS = ["show_id", "type", "title", "country", "year", "rating", "movie_description"]

INDEXES_SQL = """
CREATE UNIQUE INDEX idx_titles_show_id ON titles(show_id);
CREATE INDEX idx_titles_type ON titles(type);
CREATE INDEX idx_titles_year ON titles(year);
CREATE INDEX idx_titles_rating ON titles(rating);
CREATE INDEX idx_title_countries_country ON title_countries(country);
"""


def read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV as text, trim spaces, and turn empty cells into missing values."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df = df.apply(lambda col: col.str.strip())
    return df.mask(df == "")


def split_countries(movies: pd.DataFrame) -> pd.DataFrame:
    """One row per title and country: "United States, India" becomes two rows."""
    countries = movies[["show_id", "country"]].dropna()
    countries["country"] = countries["country"].str.split(",")
    countries = countries.explode("country")
    countries["country"] = countries["country"].str.strip()
    return countries[countries["country"] != ""].drop_duplicates()


def write_database(
    movies: pd.DataFrame, countries: pd.DataFrame, db_path: Path
) -> None:
    """Replace the database with two tables: titles and title_countries."""
    with sqlite3.connect(db_path) as conn:
        movies.to_sql("titles", conn, if_exists="replace", index=False)
        countries.to_sql("title_countries", conn, if_exists="replace", index=False)
        conn.executescript(INDEXES_SQL)


def drop_reason(row: pd.Series, kept_titles: set[str]) -> str:
    """Why my cleaning dropped this row of the original CSV."""
    if pd.isna(row["rating"]):
        return "no rating"
    if row.isna().sum() >= 3:
        return "3 or more empty fields"
    if row["title"] in kept_titles:
        return "duplicate of a kept title"
    return "other"


def print_summary(
    original: pd.DataFrame, movies: pd.DataFrame, trailing_commas: int
) -> None:
    """Print rows loaded, rows dropped and why, values fixed, and anomalies."""
    dropped = original[~original["show_id"].isin(movies["show_id"])]
    kept_titles = set(movies["title"])
    reasons = dropped.apply(drop_reason, axis=1, kept_titles=kept_titles)

    print("=== Ingest summary ===")
    print(f"Rows in original CSV: {len(original)}")
    print(f"Rows loaded:          {len(movies)}")
    print(f"Rows dropped:         {len(dropped)}")
    for reason, count in reasons.value_counts().items():
        print(f"  {count:>4}  {reason}")
    print(f"Fixed: {trailing_commas} countries with a trailing comma")
    print(f"Anomaly: {movies['country'].isna().sum()} titles have no country (NULL)")
    print(f"Database written to {DB_PATH}")


def main() -> None:
    """Load the cleaned CSV, fix countries, write to SQLite, print the summary."""
    original = read_csv(ORIGINAL_CSV)  # only used to explain dropped rows
    movies = read_csv(CLEANED_CSV)[COLUMNS]
    movies["year"] = movies["year"].astype(int)

    # Fix countries like "Poland," -> "Poland".
    trailing_commas = movies["country"].str.endswith(",").sum()
    movies["country"] = movies["country"].str.strip(", ")

    write_database(movies, split_countries(movies), DB_PATH)
    print_summary(original, movies, trailing_commas)


if __name__ == "__main__":
    main()

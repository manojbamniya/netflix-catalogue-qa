"""SQL queries behind the endpoints. All values are passed as ? parameters."""

import sqlite3
from dataclasses import dataclass
from typing import Any

UNKNOWN_COUNTRY = "Unknown"  # ?country=Unknown means "titles with no country"
TITLE_COLUMNS = "show_id, type, title, country, year, rating, movie_description"


@dataclass
class TitleFilters:
    """Optional filters for GET /titles. None means "don't filter on this"."""

    country: str | None = None
    release_year: int | None = None
    type: str | None = None
    rating: str | None = None


def build_where(filters: TitleFilters) -> tuple[str, list[Any]]:
    """Turn the filters into a WHERE clause and its parameters.

    Country uses the title_countries table, so "India" matches
    "United States, India" but not "British Indian Ocean Territory".
    """
    clauses: list[str] = []
    params: list[Any] = []

    if filters.country is not None:
        if filters.country.lower() == UNKNOWN_COUNTRY.lower():
            clauses.append("show_id NOT IN (SELECT show_id FROM title_countries)")
        else:
            clauses.append(
                "show_id IN (SELECT show_id FROM title_countries "
                "WHERE country = ? COLLATE NOCASE)"
            )
            params.append(filters.country)
    if filters.release_year is not None:
        clauses.append("year = ?")
        params.append(filters.release_year)
    if filters.type is not None:
        clauses.append("type = ?")
        params.append(filters.type)
    if filters.rating is not None:
        clauses.append("rating = ? COLLATE NOCASE")
        params.append(filters.rating)

    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    return where, params


def list_titles(
    conn: sqlite3.Connection, filters: TitleFilters, limit: int, offset: int
) -> tuple[int, list[dict[str, Any]]]:
    """Return (total matching titles, one page of titles), sorted by title."""
    where, params = build_where(filters)
    total = conn.execute(f"SELECT COUNT(*) FROM titles{where}", params).fetchone()[0]
    rows = conn.execute(
        f"SELECT {TITLE_COLUMNS} FROM titles{where} "
        "ORDER BY title, show_id LIMIT ? OFFSET ?",
        [*params, limit, offset],
    ).fetchall()
    return total, [dict(row) for row in rows]


def get_title(conn: sqlite3.Connection, show_id: str) -> dict[str, Any] | None:
    """Return one title by show_id, or None if it doesn't exist."""
    row = conn.execute(
        f"SELECT {TITLE_COLUMNS} FROM titles WHERE show_id = ?", [show_id]
    ).fetchone()
    return dict(row) if row else None


def get_stats(conn: sqlite3.Connection, top_n: int) -> dict[str, Any]:
    """Total titles, count per type, top countries, and titles with no country."""
    total = conn.execute("SELECT COUNT(*) FROM titles").fetchone()[0]
    by_type = conn.execute(
        "SELECT type, COUNT(*) FROM titles GROUP BY type ORDER BY type"
    ).fetchall()
    top_countries = conn.execute(
        "SELECT country, COUNT(*) AS count FROM title_countries "
        "GROUP BY country ORDER BY count DESC, country LIMIT ?",
        [top_n],
    ).fetchall()
    without_country = conn.execute(
        "SELECT COUNT(*) FROM titles "
        "WHERE show_id NOT IN (SELECT show_id FROM title_countries)"
    ).fetchone()[0]
    return {
        "total_titles": total,
        "count_by_type": {row[0]: row[1] for row in by_type},
        "top_countries": [dict(row) for row in top_countries],
        "titles_without_country": without_country,
    }

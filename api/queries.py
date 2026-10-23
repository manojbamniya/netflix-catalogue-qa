"""SQL for the endpoints. Values are always passed as ? (never pasted into SQL)."""

import sqlite3
from typing import Any

COLUMNS = "show_id, type, title, country, year, rating, movie_description"
UNKNOWN_COUNTRY = "unknown"  # ?country=Unknown means "titles with no country"


def build_where(
    country: str | None, year: int | None, type: str | None, rating: str | None
) -> tuple[str, list[Any]]:
    """Turn the filters into a WHERE clause and its values. None = no filter."""
    clauses: list[str] = []
    values: list[Any] = []

    if country and country.lower() == UNKNOWN_COUNTRY:
        clauses.append("show_id NOT IN (SELECT show_id FROM title_countries)")
    elif country:
        clauses.append(
            "show_id IN (SELECT show_id FROM title_countries "
            "WHERE country = ? COLLATE NOCASE)"
        )
        values.append(country)
    if year:
        clauses.append("year = ?")
        values.append(year)
    if type:
        clauses.append("type = ?")
        values.append(type)
    if rating:
        clauses.append("rating = ? COLLATE NOCASE")
        values.append(rating)

    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    return where, values


def list_titles(
    conn: sqlite3.Connection,
    filters: dict[str, Any],
    limit: int,
    offset: int,
) -> tuple[int, list[dict[str, Any]]]:
    """Return (how many titles match, one page of them sorted by title)."""
    where, values = build_where(**filters)
    total = conn.execute(f"SELECT COUNT(*) FROM titles{where}", values).fetchone()[0]
    rows = conn.execute(
        f"SELECT {COLUMNS} FROM titles{where} ORDER BY title, show_id LIMIT ? OFFSET ?",
        [*values, limit, offset],
    ).fetchall()
    return total, [dict(row) for row in rows]


def get_title(conn: sqlite3.Connection, show_id: str) -> dict[str, Any] | None:
    """Return one title, or None if there's no such show_id."""
    row = conn.execute(
        f"SELECT {COLUMNS} FROM titles WHERE show_id = ?", [show_id]
    ).fetchone()
    return dict(row) if row else None


def get_stats(conn: sqlite3.Connection) -> dict[str, Any]:
    """Total titles, count per type, top 10 countries, titles with no country."""
    total = conn.execute("SELECT COUNT(*) FROM titles").fetchone()[0]
    by_type = conn.execute("SELECT type, COUNT(*) FROM titles GROUP BY type")
    top_countries = conn.execute(
        "SELECT country, COUNT(*) AS n FROM title_countries "
        "GROUP BY country ORDER BY n DESC, country LIMIT 10"
    )
    without_country = conn.execute(
        "SELECT COUNT(*) FROM titles "
        "WHERE show_id NOT IN (SELECT show_id FROM title_countries)"
    ).fetchone()[0]
    return {
        "total_titles": total,
        "count_by_type": dict(by_type.fetchall()),
        "top_countries": dict(top_countries.fetchall()),
        "titles_without_country": without_country,
    }

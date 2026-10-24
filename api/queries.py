"""SQL for the endpoints.

Safety rule: a value from the user is never pasted into the SQL text.
The SQL has a ? where the value goes, and the value is passed separately.
SQLite fills each ? with the next value from the list, in order.
"""

import sqlite3

COLUMNS = "show_id, type, title, country, year, rating, movie_description"
UNKNOWN_COUNTRY = "unknown"  # ?country=Unknown means "titles with no country"
TOP_COUNTRIES = 10  # how many countries /stats lists


def build_where(
    country: str | None, year: int | None, type: str | None, rating: str | None
) -> tuple[str, list]:
    """Turn the filters into a WHERE clause and its values. None = no filter.

    Example: year=2020, type="Movie" gives
        (" WHERE year = ? AND type = ?", [2020, "Movie"])
    """
    conditions = []  # pieces of SQL, each with one ? (or none)
    values = []  # the value for each ?, in the same order

    if country is not None:
        if country.lower() == UNKNOWN_COUNTRY:
            # titles that have no row in title_countries
            conditions.append("show_id NOT IN (SELECT show_id FROM title_countries)")
        else:
            # titles that have a row with this country (case doesn't matter)
            conditions.append(
                "show_id IN (SELECT show_id FROM title_countries "
                "WHERE country = ? COLLATE NOCASE)"
            )
            values.append(country)

    if year is not None:
        conditions.append("year = ?")
        values.append(year)

    if type is not None:
        conditions.append("type = ?")
        values.append(type)

    if rating is not None:
        conditions.append("rating = ? COLLATE NOCASE")
        values.append(rating)

    if len(conditions) == 0:
        return "", values  # no filters: match every title

    where = " WHERE " + " AND ".join(conditions)
    return where, values


def list_titles(
    conn: sqlite3.Connection, filters: dict, limit: int, offset: int
) -> tuple[int, list[dict]]:
    """Return (how many titles match, one page of them sorted by title)."""
    where, values = build_where(
        country=filters["country"],
        year=filters["year"],
        type=filters["type"],
        rating=filters["rating"],
    )

    # 1. Count every matching title.
    count_sql = "SELECT COUNT(*) FROM titles" + where
    total = conn.execute(count_sql, values).fetchone()[0]

    # 2. Get one page of them. LIMIT and OFFSET add two more ?s at the end.
    page_sql = "SELECT " + COLUMNS + " FROM titles" + where
    page_sql = page_sql + " ORDER BY title, show_id LIMIT ? OFFSET ?"
    page_values = values + [limit, offset]
    rows = conn.execute(page_sql, page_values).fetchall()

    titles = []
    for row in rows:
        titles.append(dict(row))
    return total, titles


def get_title(conn: sqlite3.Connection, show_id: str) -> dict | None:
    """Return one title, or None if there's no such show_id."""
    sql = "SELECT " + COLUMNS + " FROM titles WHERE show_id = ?"
    row = conn.execute(sql, [show_id]).fetchone()
    if row is None:
        return None
    return dict(row)


def get_stats(conn: sqlite3.Connection) -> dict:
    """Total titles, count per type, top 10 countries, titles with no country."""
    total = conn.execute("SELECT COUNT(*) FROM titles").fetchone()[0]

    count_by_type = {}
    for type_name, count in conn.execute(
        "SELECT type, COUNT(*) FROM titles GROUP BY type"
    ):
        count_by_type[type_name] = count

    top_countries = {}
    for country, count in conn.execute(
        "SELECT country, COUNT(*) AS n FROM title_countries "
        "GROUP BY country ORDER BY n DESC, country LIMIT ?",
        [TOP_COUNTRIES],
    ):
        top_countries[country] = count

    without_country = conn.execute(
        "SELECT COUNT(*) FROM titles "
        "WHERE show_id NOT IN (SELECT show_id FROM title_countries)"
    ).fetchone()[0]

    return {
        "total_titles": total,
        "count_by_type": count_by_type,
        "top_countries": top_countries,
        "titles_without_country": without_country,
    }

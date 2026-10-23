"""Request and response shapes for the API."""

from enum import Enum

from pydantic import BaseModel


class TitleType(str, Enum):
    """The two values of the type column."""

    MOVIE = "Movie"
    TV_SHOW = "TV Show"


class Title(BaseModel):
    """One row of the titles table."""

    show_id: str
    type: str
    title: str
    country: str | None
    year: int
    rating: str | None
    movie_description: str


class TitlePage(BaseModel):
    """One page of results from GET /titles."""

    total: int  # matching titles across all pages
    page: int
    page_size: int
    items: list[Title]


class CountryCount(BaseModel):
    """How many titles list a country."""

    country: str
    count: int


class Stats(BaseModel):
    """Summary numbers for GET /stats."""

    total_titles: int
    count_by_type: dict[str, int]
    top_countries: list[CountryCount]
    titles_without_country: int

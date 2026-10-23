"""Request and response shapes for the API."""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

MIN_QUESTION_LENGTH = 3
MAX_QUESTION_LENGTH = 500


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


class AskRequest(BaseModel):
    """Body of POST /ask."""

    # Strip spaces first, so "   " counts as empty and is rejected.
    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(
        min_length=MIN_QUESTION_LENGTH,
        max_length=MAX_QUESTION_LENGTH,
        examples=["Suggest me a comedy movie from India with a strong female lead"],
    )


class Source(BaseModel):
    """A title the answer was based on."""

    show_id: str
    title: str


class AskResponse(BaseModel):
    """Answer from POST /ask, with the titles it was based on."""

    answer: str
    show_ids: list[str]
    sources: list[Source]

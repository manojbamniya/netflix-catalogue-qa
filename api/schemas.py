"""Shapes of the API's requests and responses."""

from pydantic import BaseModel, ConfigDict, Field

MIN_QUESTION_LENGTH = 3
MAX_QUESTION_LENGTH = 500


class Title(BaseModel):
    """One title from the database."""

    show_id: str
    type: str
    title: str
    country: str | None
    year: int
    rating: str | None
    movie_description: str


class TitlePage(BaseModel):
    """One page of GET /titles. total = matching titles across all pages."""

    total: int
    page: int
    page_size: int
    items: list[Title]


class Stats(BaseModel):
    """Summary numbers for GET /stats."""

    total_titles: int
    count_by_type: dict[str, int]
    top_countries: dict[str, int]
    titles_without_country: int


class AskRequest(BaseModel):
    """Body of POST /ask. Spaces are trimmed first, so "     " is rejected."""

    model_config = ConfigDict(str_strip_whitespace=True)
    question: str = Field(
        min_length=MIN_QUESTION_LENGTH, max_length=MAX_QUESTION_LENGTH
    )


class AskResponse(BaseModel):
    """The answer and the titles it was based on."""

    answer: str
    show_ids: list[str]
    sources: list[dict[str, str]]  # [{"show_id": ..., "title": ...}]

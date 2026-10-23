"""Tests for RAG: retrieval, the source check, and POST /ask.

No model download or Gemini call: a fake embedder and fake LLMs stand in.
"""

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from google.genai import errors as genai_errors

import api.main
from api.database import open_connection
from rag import settings
from rag.answer import NO_MATCH_ANSWER, LLMAnswer, answer_question
from rag.index import build_document, build_index
from rag.retriever import Retriever

VOCABULARY = ["india", "comedy", "crime", "kids"]


def fake_embed(texts: list[str]) -> np.ndarray:
    """Vector = which VOCABULARY words appear, normalised. Enough to rank."""
    vectors = np.array(
        [[float(word in text.lower()) for word in VOCABULARY] for text in texts]
    )
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return (vectors / np.where(norms == 0, 1, norms)).astype(np.float32)


class FakeRetriever:
    """Always 'retrieves' the same show_ids."""

    def __init__(self, ids: list[str]):
        self.ids = ids

    def search(self, question: str, k: int) -> list[str]:
        return self.ids[:k]


class FakeLLM:
    """Returns a fixed reply and remembers the context it was given."""

    def __init__(self, reply: LLMAnswer):
        self.reply = reply
        self.context = ""

    def generate(self, question: str, context: str) -> LLMAnswer:
        self.context = context
        return self.reply


class FailingLLM:
    """Behaves like Gemini returning a server error."""

    def generate(self, question: str, context: str) -> LLMAnswer:
        raise genai_errors.ServerError(503, {"error": {"message": "overloaded"}})


@pytest.fixture
def ask_client(client: TestClient) -> Iterator[TestClient]:
    """The test client with /ask wired to a fake retriever (ids 1, 3, 6)."""
    api.main.app.dependency_overrides[api.main.get_retriever] = lambda: FakeRetriever(
        ["1", "3", "6"]
    )
    yield client


def use_llm(reply: LLMAnswer) -> FakeLLM:
    """Make /ask use a FakeLLM with this reply."""
    llm = FakeLLM(reply)
    api.main.app.dependency_overrides[api.main.get_llm] = lambda: llm
    return llm


def test_retriever_ranks_most_similar_titles_first() -> None:
    texts = ["Indian comedy", "Indian crime drama", "kids cartoon"]
    retriever = Retriever(np.array(["a", "b", "c"]), fake_embed(texts), fake_embed)

    assert retriever.search("a comedy from india", k=2) == ["a", "b"]


def test_index_round_trip(db_path: Path, tmp_path: Path) -> None:
    index_path = tmp_path / "embeddings.npz"

    count = build_index(db_path, index_path, fake_embed)
    retriever = Retriever.load(index_path, fake_embed)

    assert count == 6
    assert sorted(retriever.ids.tolist()) == ["1", "2", "3", "4", "5", "6"]
    assert retriever.vectors.shape == (6, len(VOCABULARY))


def test_document_puts_title_and_country_first() -> None:
    title = {
        "title": "Fargo",
        "type": "TV Show",
        "year": 2014,
        "country": None,
        "rating": "TV-MA",
        "movie_description": "Crime in Minnesota.",
    }

    assert build_document(title) == (
        "Fargo (TV Show, 2014). Country: Unknown. Rating: TV-MA. Crime in Minnesota."
    )


@pytest.fixture
def conn(db_path: Path) -> Iterator[sqlite3.Connection]:
    """A connection to the tiny database, closed after the test."""
    connection = open_connection(db_path)
    yield connection
    connection.close()


def test_sources_the_llm_made_up_are_dropped(conn: sqlite3.Connection) -> None:
    # The LLM cites 3 (retrieved) and 999 (never retrieved, doesn't exist).
    llm = FakeLLM(LLMAnswer(answer="Try Chhota Bheem.", show_ids=["3", "999", "3"]))

    result = answer_question("kids show", conn, FakeRetriever(["1", "3"]), llm, 8)

    assert result["show_ids"] == ["3"]
    assert result["sources"] == [{"show_id": "3", "title": "Chhota Bheem"}]
    assert "[show_id: 1] Andhadhun" in llm.context


def test_answer_without_valid_sources_becomes_no_match(
    conn: sqlite3.Connection,
) -> None:
    llm = FakeLLM(LLMAnswer(answer="Watch Inception!", show_ids=["999"]))

    result = answer_question("sci-fi", conn, FakeRetriever(["1"]), llm, 8)

    assert result == {"answer": NO_MATCH_ANSWER, "show_ids": [], "sources": []}


def test_ask_returns_answer_with_sources(ask_client: TestClient) -> None:
    use_llm(LLMAnswer(answer="Andhadhun is an Indian thriller.", show_ids=["1"]))

    response = ask_client.post("/ask", json={"question": "Indian thriller?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Andhadhun is an Indian thriller."
    assert body["show_ids"] == ["1"]
    assert body["sources"] == [{"show_id": "1", "title": "Andhadhun"}]


@pytest.mark.parametrize(
    "reply",
    [
        LLMAnswer(answer="Fargo and Chhota Bheem.", show_ids=["6", "3"]),
        LLMAnswer(answer="Something made up.", show_ids=["42"]),
        LLMAnswer(answer="Nothing fits.", show_ids=[]),
    ],
)
def test_ask_never_returns_an_answer_without_sources(
    ask_client: TestClient, reply: LLMAnswer
) -> None:
    use_llm(reply)

    body = ask_client.post("/ask", json={"question": "anything"}).json()

    # Either a grounded answer with sources, or the fixed no-match message.
    assert body["sources"] or body["answer"] == NO_MATCH_ANSWER
    assert set(body["show_ids"]) <= {"1", "3", "6"}


@pytest.mark.parametrize("question", ["", "     ", "hi", "x" * 501])
def test_ask_rejects_bad_questions_with_400(
    ask_client: TestClient, question: str
) -> None:
    use_llm(LLMAnswer(answer="unused", show_ids=[]))

    response = ask_client.post("/ask", json={"question": question})

    assert response.status_code == 400
    assert response.json()["detail"][0]["param"] == "question"


def test_ask_returns_502_when_gemini_fails(ask_client: TestClient) -> None:
    api.main.app.dependency_overrides[api.main.get_llm] = FailingLLM

    response = ask_client.post("/ask", json={"question": "Indian thriller?"})

    assert response.status_code == 502


def test_ask_returns_503_without_api_key(
    ask_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "GEMINI_API_KEY", None)
    api.main.get_llm.cache_clear()

    response = ask_client.post("/ask", json={"question": "Indian thriller?"})

    assert response.status_code == 503
    assert "GEMINI_API_KEY" in response.json()["detail"]

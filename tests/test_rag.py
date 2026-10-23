"""Tests for RAG and POST /ask. A fake embedder and fake LLMs are used, so no
model download, no API key and no network are needed."""

import sqlite3
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from google.genai import errors as genai_errors

from api import main
from rag import answer, index
from rag.answer import NO_MATCH_ANSWER, LLMAnswer

WORDS = ["india", "comedy", "crime", "kids"]


def fake_embed(texts: list[str]) -> np.ndarray:
    """Vector = which WORDS appear in the text, scaled to length 1."""
    vectors = np.array([[float(w in t.lower()) for w in WORDS] for t in texts])
    lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(lengths == 0, 1, lengths)


def fake_llm(reply: LLMAnswer) -> answer.AskLLM:
    """An LLM that always gives this reply."""
    return lambda question, titles_text: reply


@pytest.fixture
def ask_client(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """/ask with a fake embedder and an index where titles 1, 3, 6 rank first."""
    monkeypatch.setattr(index, "embed", fake_embed)
    ids = np.array(["1", "3", "6"])
    main.app.dependency_overrides[main.get_index] = lambda: (ids, fake_embed(["x"] * 3))
    return client


def use_llm(reply: LLMAnswer) -> None:
    """Make /ask use a fake LLM with this reply."""
    main.app.dependency_overrides[main.get_llm] = lambda: fake_llm(reply)


def test_search_ranks_most_similar_first(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(index, "embed", fake_embed)
    vectors = fake_embed(["Indian comedy", "Indian crime", "kids cartoon"])

    found = index.search("a comedy from india", np.array(["a", "b", "c"]), vectors, k=2)

    assert found == ["a", "b"]


def test_build_and_load_index(
    db_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(index, "embed", fake_embed)
    path = tmp_path / "embeddings.npz"

    count = index.build_index(db_path, path)
    ids, vectors = index.load_index(path)

    assert count == 6
    assert sorted(ids.tolist()) == ["1", "2", "3", "4", "5", "6"]
    assert vectors.shape == (6, len(WORDS))


def test_document_puts_title_and_country_first() -> None:
    title = {
        "title": "Fargo",
        "type": "TV Show",
        "year": 2014,
        "country": None,
        "rating": "TV-MA",
        "movie_description": "Crime in Minnesota.",
    }

    assert index.document(title) == (
        "Fargo (TV Show, 2014). Country: Unknown. Rating: TV-MA. Crime in Minnesota."
    )


def test_made_up_show_ids_are_dropped(conn: sqlite3.Connection) -> None:
    # 3 was retrieved; 999 was not (and doesn't exist). 3 is repeated.
    llm = fake_llm(LLMAnswer(answer="Try Chhota Bheem.", show_ids=["3", "999", "3"]))

    result = answer.answer_question("kids show", ["1", "3"], conn, llm)

    assert result["show_ids"] == ["3"]
    assert result["sources"] == [{"show_id": "3", "title": "Chhota Bheem"}]


def test_no_valid_show_ids_gives_no_match(conn: sqlite3.Connection) -> None:
    llm = fake_llm(LLMAnswer(answer="Watch Inception!", show_ids=["999"]))

    result = answer.answer_question("sci-fi", ["1"], conn, llm)

    assert result == {"answer": NO_MATCH_ANSWER, "show_ids": [], "sources": []}


def test_ask_returns_answer_with_sources(ask_client: TestClient) -> None:
    use_llm(LLMAnswer(answer="Andhadhun is an Indian thriller.", show_ids=["1"]))

    response = ask_client.post("/ask", json={"question": "Indian thriller?"})

    assert response.status_code == 200
    assert response.json() == {
        "answer": "Andhadhun is an Indian thriller.",
        "show_ids": ["1"],
        "sources": [{"show_id": "1", "title": "Andhadhun"}],
    }


@pytest.mark.parametrize(
    "reply",
    [
        LLMAnswer(answer="Fargo and Chhota Bheem.", show_ids=["6", "3"]),
        LLMAnswer(answer="Something made up.", show_ids=["42"]),
        LLMAnswer(answer="Nothing fits.", show_ids=[]),
    ],
)
def test_ask_never_answers_without_sources(
    ask_client: TestClient, reply: LLMAnswer
) -> None:
    use_llm(reply)

    body = ask_client.post("/ask", json={"question": "anything"}).json()

    assert body["sources"] or body["answer"] == NO_MATCH_ANSWER
    assert set(body["show_ids"]) <= {"1", "3", "6"}


@pytest.mark.parametrize("question", ["", "     ", "hi", "x" * 501])
def test_ask_rejects_bad_questions(ask_client: TestClient, question: str) -> None:
    use_llm(LLMAnswer(answer="unused", show_ids=[]))

    response = ask_client.post("/ask", json={"question": question})

    assert response.status_code == 400
    assert response.json()["detail"][0]["param"] == "question"


@pytest.mark.parametrize(
    "error, expected_status",
    [
        (genai_errors.ServerError(503, {"error": {"message": "busy"}}), 502),
        (genai_errors.ClientError(429, {"error": {"message": "quota"}}), 503),
        (genai_errors.ClientError(400, {"error": {"message": "bad"}}), 502),
    ],
)
def test_ask_handles_gemini_errors(
    ask_client: TestClient, error: Exception, expected_status: int
) -> None:
    def failing_llm(question: str, titles_text: str) -> LLMAnswer:
        raise error

    main.app.dependency_overrides[main.get_llm] = lambda: failing_llm

    response = ask_client.post("/ask", json={"question": "Indian thriller?"})

    assert response.status_code == expected_status


def test_ask_returns_503_without_api_key(
    ask_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(answer, "GEMINI_API_KEY", None)

    response = ask_client.post("/ask", json={"question": "Indian thriller?"})

    assert response.status_code == 503
    assert "GEMINI_API_KEY" in response.json()["detail"]
